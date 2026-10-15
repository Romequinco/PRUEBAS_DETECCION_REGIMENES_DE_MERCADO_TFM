"""Cripto del Tema 1 (bloque 6): Coinbase Exchange (principal) y Binance Vision (control).

Coinbase Exchange (``coinbase``):

- API pública de datos de mercado, sin clave: ``GET /products/<par>/candles`` con
  ``granularity=86400``. Cada vela es ``[time, low, high, open, close, volume]`` (¡ese orden, no
  OHLC!), en orden DESCENDENTE de ``time`` (segundos Unix del inicio de la vela, día UTC), con
  ``volume`` en la moneda base (BTC, ETH). Como mucho ``MAX_VELAS`` velas por petición: se pide por
  ventanas de 300 días. Ritmo ``POR_SEGUNDO_COINBASE`` (3 pet/s; el límite público es mayor). Un
  día sin operaciones no trae vela: no se inventa (hueco medido en la ficha).
- Es una API documentada, no un sitio que se rastrea: no se consulta robots.txt (en el host de la
  API cualquier ruta desconocida, ``/robots.txt`` incluida, responde ``401`` en JSON).
- Tabla de PRECIOS del contrato: ticker ``cb:<par>`` (``cb:BTC-USD``, ``cb:ETH-USD``), mercado
  ``CRYPTO`` (vela diaria UTC: cierra a las 24:00 UTC, visible a las 00:30 UTC de D+1 →
  ``fecha_publicacion`` = D+1), divisa la cotizada del par (USD), sin dividendos (0) ni splits (1).
  La vela del día en curso se descarta (``descartar_velas_incompletas``). Se guarda con
  ``almacen.escribir_precios(proveedor='coinbase')``: base que solo crece.

Binance Vision (``binance_vision``), solo CONTROL:

- ``data.binance.vision``: un ZIP mensual por símbolo con las klines diarias en CSV sin cabecera
  (``open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_base,
  taker_quote, ignore``). ``open_time`` en MILISEGUNDOS hasta 2024 y en MICROSEGUNDOS desde los
  ficheros de 2025: la unidad se decide por magnitud, fila a fila. Licencia CC BY-NC-SA 4.0.
- Divisa ``USDT``: el contrato (``contrato._PATRON_DIVISA``) solo admite códigos ISO de 3 letras y
  ``get_precios`` convertiría con el FX del BCE, que no tiene USDT. Por eso el control NO entra en
  ``precios/``: se valida con el contrato (todo salvo la divisa) y se guarda aparte en
  ``<corpus>/mercados_series/control/binance/<ticker>.parquet`` (ticker ``bn:BTCUSDT``).
- ``comparar`` mide la diferencia de cierre Binance (USDT) frente a Coinbase (USD) en puntos
  básicos sobre las sesiones comunes: mediana y p95 del valor absoluto, mediana con signo (prima
  USDT/USD) y las sesiones de una fuente que faltan en la otra.

Crudos en ``<corpus>/mercados_series/crudo/<fuente>/``: Coinbase como JSON por descarga
(``<AAAAMMDD>_<par>.json``, día UTC); Binance, los ZIP tal cual con su ``.meta.json`` (URL,
``Last-Modified``, bytes, sha256). Los ZIP ya archivados no se vuelven a pedir: la descarga es
reanudable. El retraso de Binance se MIDE con ``Last-Modified`` frente al fin del mes.

CLI: ``python -m ingesta.mercados_series.fuentes.cripto [--coinbase] [--binance] [--comparar]
[--fichas] [--desde-binance AAAA-MM] [--hasta-binance AAAA-MM]``.
Datos solo en local; al repositorio van código, fichas y la tabla de metadatos de la comparación.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import zipfile
from collections.abc import Callable, Iterable
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingesta.comun import http
from ingesta.comun.esquema import FichaFuente
from ingesta.comun.fichas import guardar_ficha
from ingesta.comun.rutas import corpus_dir, resultados_dir
from ingesta.mercados_series import almacen, visibilidad
from ingesta.mercados_series.contrato import (
    COLUMNAS_PRECIOS,
    ErrorContrato,
    descartar_velas_incompletas,
    primera_decision_vec,
    validar_precios,
)

TEMA = "mercados_series"
MERCADO = "CRYPTO"

# --- Coinbase -----------------------------------------------------------------------------------
FUENTE_COINBASE = "coinbase"
PROVEEDOR_COINBASE = "coinbase"
URL_COINBASE = "https://api.exchange.coinbase.com/products/{par}/candles"
PARES_COINBASE = ("BTC-USD", "ETH-USD")
PREFIJO_COINBASE = "cb:"
GRANULARIDAD = 86400
MAX_VELAS = 300
POR_SEGUNDO_COINBASE = 3.0
# Antes de la primera vela de Coinbase Exchange (BTC-USD empieza en julio de 2015): las ventanas
# vacías del principio cuestan unas pocas peticiones y la cobertura se MIDE, no se declara.
INICIO_BUSQUEDA = pd.Timestamp("2015-01-01")
COLUMNAS_COINBASE = ("time", "low", "high", "open", "close", "volume")

# --- Binance Vision -----------------------------------------------------------------------------
FUENTE_BINANCE = "binance_vision"
PROVEEDOR_BINANCE = "binance"
URL_BINANCE = ("https://data.binance.vision/data/spot/monthly/klines/{simbolo}/1d/"
               "{simbolo}-1d-{mes}.zip")
PREFIJO_BINANCE = "bn:"
DIVISA_BINANCE = "USDT"
INICIO_BINANCE = "2017-08"   # primer mes de BTCUSDT y ETHUSDT en Binance
POR_SEGUNDO_BINANCE = 1.0
COLUMNAS_BINANCE = ("open_time", "open", "high", "low", "close", "volume", "close_time",
                    "quote_volume", "trades", "taker_base", "taker_quote", "ignore")
# open_time por encima de este umbral está en microsegundos (1e14 ms sería el año 5138).
UMBRAL_MICROSEGUNDOS = 10**14

# La comparación se da también desde 2020: 2017-2018 (primas de USDT, arbitraje lento) domina el p95.
CORTE_RECIENTE = pd.Timestamp("2020-01-01")

# Par de Coinbase → símbolo de control en Binance.
CONTROL = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}

LICENCIA_BINANCE = "CC BY-NC-SA 4.0 (Binance Vision): uso no comercial con atribución; solo local"
LICENCIA_COINBASE = ("Términos de Coinbase (Exchange API): datos de mercado públicos sin clave; "
                     "sin redistribución: solo local")

GetJson = Callable[[str, dict], Any]
GetBytes = Callable[[str], tuple[bytes, dict]]


# --- Utilidades ---------------------------------------------------------------------------------
def _ahora() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _escribir_atomico(ruta: Path, datos: bytes) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_name(ruta.name + ".tmp")
    tmp.write_bytes(datos)
    os.replace(tmp, ruta)


def _crudo_dir(fuente: str) -> Path:
    return corpus_dir(TEMA) / "crudo" / fuente


def ruta_control(ticker: str, carpeta: Path | None = None) -> Path:
    """``<corpus>/mercados_series/control/binance/<nombre_fichero_ticker(ticker)>.parquet``."""
    base = Path(carpeta) if carpeta is not None else corpus_dir(TEMA) / "control" / PROVEEDOR_BINANCE
    return base / f"{almacen.nombre_fichero_ticker(ticker)}.parquet"


def _tabla_precios(velas: pd.DataFrame, ticker: str, divisa: str, fuente: str,
                   descargado: pd.Timestamp) -> pd.DataFrame:
    """Velas diarias UTC (``sesion`` + OHLCV) → columnas del contrato, sin la vela en curso."""
    sesiones = pd.Series(pd.to_datetime(velas["sesion"]).to_numpy(), dtype="datetime64[ns]")
    disponible = visibilidad.disponible_cierre_serie(sesiones, MERCADO)
    out = pd.DataFrame({
        "ticker": ticker,
        "sesion": sesiones,
        "mercado": MERCADO,
        "open": velas["open"].to_numpy(float),
        "high": velas["high"].to_numpy(float),
        "low": velas["low"].to_numpy(float),
        "close": velas["close"].to_numpy(float),
        "volume": velas["volume"].to_numpy(float),
        "dividendo": 0.0,
        "split": 1.0,
        "divisa": divisa,
        "fecha_publicacion": primera_decision_vec(disponible).reset_index(drop=True),
        "disponible_utc": disponible.reset_index(drop=True),
        "fuente": fuente,
        "descargado": pd.Timestamp(descargado).tz_convert("UTC"),
    })[list(COLUMNAS_PRECIOS)]
    return descartar_velas_incompletas(out).reset_index(drop=True)


def _comprobar_ohlc(df: pd.DataFrame, origen: str) -> None:
    """low <= open, close <= high: si falla, el orden de columnas de la fuente ha cambiado."""
    malo = (df["low"] > df[["open", "close"]].min(axis=1) + 1e-12) | \
           (df["high"] < df[["open", "close"]].max(axis=1) - 1e-12) | (df["low"] > df["high"])
    if malo.any():
        ej = df.loc[malo, "sesion"].head(3).dt.date.astype(str).tolist()
        raise ValueError(f"{origen}: {int(malo.sum())} velas con low/high incoherentes con open/close "
                         f"(¿ha cambiado el orden de columnas?; p. ej. {ej})")


# --- Coinbase -----------------------------------------------------------------------------------
def parsear_velas_coinbase(filas: Iterable[Iterable[float]]) -> pd.DataFrame:
    """Respuesta de ``/candles`` (``[time, low, high, open, close, volume]``, descendente) → velas.

    Devuelve ``sesion`` (día UTC naive) + ``open, high, low, close, volume`` en orden ascendente y
    sin sesiones repetidas (gana la última vista: ventanas solapadas). ``ValueError`` si una fila no
    tiene 6 campos, si ``time`` no es un inicio de día UTC o si low/high no envuelven open/close.
    """
    filas = [list(f) for f in filas]
    malas = [f for f in filas if len(f) != len(COLUMNAS_COINBASE)]
    if malas:
        raise ValueError(f"coinbase: {len(malas)} velas sin {len(COLUMNAS_COINBASE)} campos (p. ej. {malas[0]})")
    if not filas:
        return pd.DataFrame({"sesion": pd.Series(dtype="datetime64[ns]"),
                             **{c: pd.Series(dtype=float) for c in ("open", "high", "low", "close", "volume")}})
    crudo = pd.DataFrame(filas, columns=list(COLUMNAS_COINBASE))
    t = pd.to_numeric(crudo["time"], errors="raise").astype("int64")
    if (t % GRANULARIDAD != 0).any():
        raise ValueError("coinbase: velas cuyo 'time' no es la medianoche UTC (¿granularidad distinta de 86400?)")
    velas = pd.DataFrame({
        "sesion": pd.to_datetime(t, unit="s").astype("datetime64[ns]"),
        **{c: pd.to_numeric(crudo[c], errors="raise").astype(float)
           for c in ("open", "high", "low", "close", "volume")},
    })
    velas = velas.drop_duplicates("sesion", keep="last").sort_values("sesion", kind="mergesort")
    velas = velas.reset_index(drop=True)
    _comprobar_ohlc(velas, "coinbase")
    return velas


def tabla_coinbase(velas: pd.DataFrame, par: str, descargado: pd.Timestamp) -> pd.DataFrame:
    """Velas de Coinbase → tabla de precios validada (``cb:<par>``, CRYPTO, divisa cotizada)."""
    divisa = par.split("-", 1)[1]
    return validar_precios(_tabla_precios(velas, PREFIJO_COINBASE + par, divisa, FUENTE_COINBASE, descargado))


def _get_json_coinbase(url: str, params: dict) -> Any:
    return http.get(url, params=params, por_segundo=POR_SEGUNDO_COINBASE, timeout=30).json()


def _ventanas(desde: pd.Timestamp, hasta: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Ventanas [inicio, fin] de ``MAX_VELAS`` días como mucho (ambos extremos incluidos)."""
    out = []
    t = pd.Timestamp(desde).normalize()
    fin = pd.Timestamp(hasta).normalize()
    while t <= fin:
        f = min(t + pd.Timedelta(days=MAX_VELAS - 1), fin)
        out.append((t, f))
        t = f + pd.Timedelta(days=1)
    return out


def descargar_coinbase(par: str, desde: Any = None, hasta: Any = None, *, get: GetJson | None = None,
                       archivar: bool = True) -> tuple[pd.DataFrame, dict]:
    """Velas diarias de ``par`` entre ``desde`` y ``hasta`` (días UTC; por defecto todo hasta hoy).

    Pide ventanas de ``MAX_VELAS`` días. ``get(url, params) -> json`` sustituye a la red (tests).
    Devuelve ``(velas, meta)`` con ``meta`` = peticiones, filas, descargado. Con ``archivar``, el
    JSON crudo va a ``crudo/coinbase/<AAAAMMDD>_<par>.json``.
    """
    get = get or _get_json_coinbase
    descargado = _ahora()
    d = pd.Timestamp(desde) if desde is not None else INICIO_BUSQUEDA
    h = pd.Timestamp(hasta) if hasta is not None else descargado.tz_localize(None).normalize()
    filas: list[list] = []
    peticiones = 0
    url = URL_COINBASE.format(par=par)
    for ini, fin in _ventanas(d, h):
        params = {"granularity": GRANULARIDAD, "start": ini.strftime("%Y-%m-%dT00:00:00Z"),
                  "end": fin.strftime("%Y-%m-%dT00:00:00Z")}
        resp = get(url, params)
        peticiones += 1
        if not isinstance(resp, list):
            raise ValueError(f"coinbase {par}: respuesta inesperada ({str(resp)[:200]})")
        filas.extend(resp)
    velas = parsear_velas_coinbase(filas)
    meta = {"par": par, "peticiones": peticiones, "velas": len(velas), "descargado": descargado.isoformat(),
            "desde": d.date().isoformat(), "hasta": h.date().isoformat()}
    if archivar:
        nombre = f"{descargado:%Y%m%d}_{par}.json"
        _escribir_atomico(_crudo_dir(FUENTE_COINBASE) / nombre,
                          json.dumps({"meta": meta, "velas": filas}).encode("utf-8"))
    meta["descargado_ts"] = descargado
    return velas, meta


def actualizar_coinbase(pares: Iterable[str] = PARES_COINBASE, *, get: GetJson | None = None,
                        ruta: Path | None = None, archivar: bool = True) -> dict[str, dict]:
    """Descarga incremental (desde la última sesión guardada) y ``escribir_precios('coinbase')``."""
    resumen = {}
    for par in pares:
        ticker = PREFIJO_COINBASE + par
        guardado = almacen.leer_precios(ticker, PROVEEDOR_COINBASE, ruta=ruta)
        desde = guardado["sesion"].max() + pd.Timedelta(days=1) if len(guardado) else None
        velas, meta = descargar_coinbase(par, desde, get=get, archivar=archivar)
        tabla = tabla_coinbase(velas, par, meta.pop("descargado_ts"))
        if len(tabla):
            almacen.escribir_precios(tabla, PROVEEDOR_COINBASE, ruta=ruta)
        meta["filas_nuevas"] = len(tabla)
        resumen[par] = meta
    return resumen


# --- Binance Vision -----------------------------------------------------------------------------
def parsear_klines_binance(contenido: bytes | str) -> pd.DataFrame:
    """CSV de klines diarias de Binance Vision → velas (``sesion`` + OHLCV), ascendente.

    ``open_time`` en milisegundos (hasta 2024) o microsegundos (desde 2025), decidido por magnitud.
    Se ignora una posible fila de cabecera. ``ValueError`` si una vela no empieza a medianoche UTC
    o si low/high no envuelven open/close.
    """
    texto = contenido.decode("utf-8") if isinstance(contenido, bytes) else contenido
    crudo = pd.read_csv(io.StringIO(texto), header=None, names=list(COLUMNAS_BINANCE), dtype=str)
    crudo = crudo[pd.to_numeric(crudo["open_time"], errors="coerce").notna()]
    t = pd.to_numeric(crudo["open_time"]).astype("int64")
    microsegundos = (t >= UMBRAL_MICROSEGUNDOS).to_numpy()
    ns = np.where(microsegundos, t.to_numpy() * 1_000, t.to_numpy() * 1_000_000)
    sesion = pd.to_datetime(ns, unit="ns")
    if (sesion != sesion.normalize()).any():
        raise ValueError("binance: velas que no empiezan a medianoche UTC (¿unidad de tiempo o intervalo?)")
    velas = pd.DataFrame({
        "sesion": pd.Series(sesion, dtype="datetime64[ns]").to_numpy(),
        **{c: pd.to_numeric(crudo[c]).astype(float).to_numpy() for c in ("open", "high", "low", "close", "volume")},
    })
    velas = velas.drop_duplicates("sesion", keep="last").sort_values("sesion", kind="mergesort")
    velas = velas.reset_index(drop=True)
    _comprobar_ohlc(velas, "binance")
    return velas


def validar_control(df: pd.DataFrame) -> pd.DataFrame:
    """``validar_precios`` para el control de Binance: todo el contrato salvo la divisa (USDT).

    Valida una copia con la divisa en ``USD`` y la restituye. ``ErrorContrato`` si la divisa no
    es ``DIVISA_BINANCE``.
    """
    otras = sorted(set(df["divisa"].dropna()) - {DIVISA_BINANCE})
    if otras:
        raise ErrorContrato("precios", [f"el control de Binance solo admite divisa {DIVISA_BINANCE}: {otras}"])
    out = validar_precios(df.assign(divisa="USD"))
    out["divisa"] = DIVISA_BINANCE
    return out


def tabla_binance(velas: pd.DataFrame, simbolo: str, descargado: pd.Timestamp) -> pd.DataFrame:
    """Velas de Binance → tabla de control (``bn:<simbolo>``, CRYPTO, USDT) validada."""
    return validar_control(_tabla_precios(velas, PREFIJO_BINANCE + simbolo, DIVISA_BINANCE,
                                          FUENTE_BINANCE, descargado))


def meses(desde: str, hasta: str) -> list[str]:
    """Meses ``AAAA-MM`` de ``desde`` a ``hasta`` (incluidos)."""
    return [p.strftime("%Y-%m") for p in pd.period_range(desde, hasta, freq="M")]


def ultimo_mes_cerrado(ahora: pd.Timestamp | None = None) -> str:
    """Último mes UTC completo (el fichero mensual del mes en curso aún no existe)."""
    a = (ahora or _ahora()).tz_convert("UTC").tz_localize(None)
    return (a.to_period("M") - 1).strftime("%Y-%m")


def _get_bytes_binance(url: str) -> tuple[bytes, dict]:
    r = http.get(url, por_segundo=POR_SEGUNDO_BINANCE, timeout=60, reintentos=2)
    return r.content, dict(r.headers)


def descargar_mes_binance(simbolo: str, mes: str, *, get: GetBytes | None = None,
                          carpeta: Path | None = None) -> tuple[pd.DataFrame | None, dict]:
    """ZIP mensual de ``simbolo`` (archivado y reutilizado si ya existe) → velas y metadatos.

    Devuelve ``(None, meta)`` si el mes no existe (404) o robots.txt no lo permite; ``meta``
    lleva ``estado``, ``last_modified``, bytes y sha256.
    """
    url = URL_BINANCE.format(simbolo=simbolo, mes=mes)
    carpeta = Path(carpeta) if carpeta is not None else _crudo_dir(FUENTE_BINANCE) / simbolo
    zip_ruta = carpeta / f"{simbolo}-1d-{mes}.zip"
    meta_ruta = zip_ruta.with_name(zip_ruta.name + ".meta.json")
    if zip_ruta.exists() and meta_ruta.exists():
        datos = zip_ruta.read_bytes()
        meta = json.loads(meta_ruta.read_text(encoding="utf-8"))
        meta["archivado"] = True
    else:
        if get is None:
            if not http.robots_permite(url):
                return None, {"url": url, "estado": "robots"}
            get = _get_bytes_binance
        try:
            datos, cabeceras = get(url)
        except Exception as e:  # noqa: BLE001 - 404 (mes inexistente) o fallo de red: se documenta
            codigo = getattr(getattr(e, "response", None), "status_code", None)
            return None, {"url": url, "estado": f"http_{codigo}" if codigo else "error",
                          "error": http.sin_clave(str(e))[:200]}
        lm = cabeceras.get("Last-Modified") or cabeceras.get("last-modified")
        meta = {"url": url, "estado": "ok", "bytes": len(datos), "sha256": hashlib.sha256(datos).hexdigest(),
                "last_modified": parsedate_to_datetime(lm).isoformat() if lm else None,
                "descargado": _ahora().isoformat(), "archivado": False}
        _escribir_atomico(zip_ruta, datos)
        _escribir_atomico(meta_ruta, json.dumps(meta, indent=2).encode("utf-8"))
    with zipfile.ZipFile(io.BytesIO(datos)) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(csvs) != 1:
            raise ValueError(f"binance {simbolo} {mes}: se esperaba un CSV en el ZIP, hay {csvs}")
        velas = parsear_klines_binance(z.read(csvs[0]))
    meta["velas"] = len(velas)
    return velas, meta


def descargar_binance(simbolo: str, desde: str = INICIO_BINANCE, hasta: str | None = None, *,
                      get: GetBytes | None = None, carpeta: Path | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Todos los meses de ``simbolo`` (reanudable: los ZIP archivados no se piden de nuevo)."""
    hasta = hasta or ultimo_mes_cerrado()
    partes, metas = [], []
    for mes in meses(desde, hasta):
        velas, meta = descargar_mes_binance(simbolo, mes, get=get, carpeta=carpeta)
        meta["mes"] = mes
        metas.append(meta)
        if velas is not None and len(velas):
            partes.append(velas)
    if not partes:
        return parsear_velas_coinbase([]), metas
    velas = pd.concat(partes, ignore_index=True).drop_duplicates("sesion", keep="last")
    return velas.sort_values("sesion", kind="mergesort").reset_index(drop=True), metas


def guardar_control(tabla: pd.DataFrame, carpeta: Path | None = None) -> list[Path]:
    """Guarda el control (validado) por ticker, reemplazando el fichero (atómico)."""
    tabla = validar_control(tabla)
    rutas = []
    for tk, g in tabla.groupby("ticker", sort=True):
        ruta = ruta_control(tk, carpeta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        tmp = ruta.with_name(ruta.name + ".tmp")
        g.reset_index(drop=True).to_parquet(tmp, index=False)
        os.replace(tmp, ruta)
        rutas.append(ruta)
    return rutas


def leer_control(ticker: str, carpeta: Path | None = None) -> pd.DataFrame:
    ruta = ruta_control(ticker, carpeta)
    if not ruta.exists():
        return validar_control(pd.DataFrame({c: [] for c in COLUMNAS_PRECIOS}).assign(divisa=DIVISA_BINANCE))
    return validar_control(pd.read_parquet(ruta))


def retraso_binance(metas: list[dict]) -> dict:
    """Días entre el fin de cada mes y el ``Last-Modified`` de su ZIP (retraso de publicación MEDIDO)."""
    pares = []
    for m in metas:
        if m.get("estado") == "ok" and m.get("last_modified") and m.get("mes"):
            fin = pd.Period(m["mes"], "M").to_timestamp(how="end").normalize() + pd.Timedelta(days=1)
            lm = pd.Timestamp(m["last_modified"]).tz_convert("UTC")
            pares.append((m["mes"], (lm - fin.tz_localize("UTC")).total_seconds() / 86400))
    if not pares:
        return {"n": 0}
    s = pd.Series([d for _, d in pares])
    reciente = [d for mes, d in pares if mes >= "2023-01"]
    return {"n": len(s), "mediana_dias": round(float(s.median()), 2), "min_dias": round(float(s.min()), 2),
            "max_dias": round(float(s.max()), 2),
            "mediana_dias_desde_2023": round(float(np.median(reciente)), 2) if reciente else None}


# --- Medidas ------------------------------------------------------------------------------------
def huecos(tabla: pd.DataFrame) -> list[str]:
    """Días naturales sin vela entre la primera y la última sesión (cripto cotiza todos los días)."""
    if tabla.empty:
        return []
    s = pd.DatetimeIndex(tabla["sesion"])
    todos = pd.date_range(s.min(), s.max(), freq="D")
    return [d.date().isoformat() for d in todos.difference(s)]


def comparar(cb: pd.DataFrame, bn: pd.DataFrame, columnas: tuple[str, ...] = ("close", "open")) -> dict:
    """Diferencia Binance (USDT) frente a Coinbase (USD) en pb, sobre las sesiones comunes.

    Por columna: mediana y p95 de ``|bn/cb - 1|`` × 1e4, mediana con signo y el máximo absoluto
    con su sesión. Más las sesiones comunes, las que solo están en una fuente y el periodo común.
    """
    m = cb[["sesion", *columnas]].merge(bn[["sesion", *columnas]], on="sesion", suffixes=("_cb", "_bn"))
    out: dict[str, Any] = {
        "sesiones_comunes": int(len(m)),
        "solo_coinbase_en_periodo_comun": 0, "solo_binance_en_periodo_comun": 0,
        "desde": m["sesion"].min().date().isoformat() if len(m) else None,
        "hasta": m["sesion"].max().date().isoformat() if len(m) else None,
    }
    if len(m):
        a, b = m["sesion"].min(), m["sesion"].max()
        cb_p = set(cb.loc[cb["sesion"].between(a, b), "sesion"])
        bn_p = set(bn.loc[bn["sesion"].between(a, b), "sesion"])
        out["solo_coinbase_en_periodo_comun"] = len(cb_p - bn_p)
        out["solo_binance_en_periodo_comun"] = len(bn_p - cb_p)
    for c in columnas:
        if not len(m):
            out[c] = None
            continue
        d = (m[f"{c}_bn"] / m[f"{c}_cb"] - 1.0) * 1e4
        ab = d.abs()
        i = int(ab.to_numpy().argmax())
        out[c] = {"mediana_abs_pb": round(float(ab.median()), 2), "p95_abs_pb": round(float(ab.quantile(0.95)), 2),
                  "mediana_pb": round(float(d.median()), 2), "max_abs_pb": round(float(ab.iloc[i]), 2),
                  "sesion_max": m["sesion"].iloc[i].date().isoformat(),
                  "sesiones_mas_de_100_pb": int((ab > 100).sum())}
        reciente = ab[m["sesion"] >= CORTE_RECIENTE]
        if len(reciente):
            out[c]["desde_2020"] = {"sesiones": int(len(reciente)),
                                    "mediana_abs_pb": round(float(reciente.median()), 2),
                                    "p95_abs_pb": round(float(reciente.quantile(0.95)), 2)}
    return out


def _cobertura(tabla: pd.DataFrame) -> tuple[str | None, str | None]:
    if tabla.empty:
        return None, None
    return tabla["sesion"].min().date().isoformat(), tabla["sesion"].max().date().isoformat()


# --- Fichas -------------------------------------------------------------------------------------
def ficha_coinbase(mediciones: dict, notas: list[str] | None = None) -> FichaFuente:
    ini = min((v["cobertura"][0] for v in mediciones.get("pares", {}).values() if v.get("cobertura", [None])[0]),
              default=None)
    fin = max((v["cobertura"][1] for v in mediciones.get("pares", {}).values() if v.get("cobertura", [None, None])[1]),
              default=None)
    return FichaFuente(
        fuente=FUENTE_COINBASE, nombre="Coinbase Exchange: velas diarias BTC-USD y ETH-USD",
        tema=TEMA, agentes=["risk", "portfolio", "cio"], rag="serie",
        url="https://docs.cdp.coinbase.com/exchange/reference/exchangerestapi_getproductcandles",
        acceso="api", autenticacion="ninguna", coste="0 €", licencia_y_terminos=LICENCIA_COINBASE,
        cobertura_inicio=ini, cobertura_fin=fin, frecuencia="diaria (vela UTC, 7 días a la semana)",
        retraso_publicacion=("medido: la vela de D es definitiva a las 00:00 UTC de D+1 y ya está en la API "
                             "(la vela del día en curso aparece incompleta y se descarta); visible en la "
                             "decisión de D+1 (00:30 UTC con el margen de 30 min)"),
        formato="json ([time, low, high, open, close, volume], descendente)",
        volumen_estimado="< 1 MB por par (unas 4.000 velas)",
        limites=(f"≤{MAX_VELAS} velas por petición; ritmo propio {POR_SEGUNDO_COINBASE:g} pet/s; "
                 "robots.txt no aplica (API; /robots.txt responde 401 JSON)"),
        estado_sonda="ok" if ini else "fallo", mediciones=mediciones,
        notas=(notas or []) + [
            "Precios as-traded sin ajustes (cripto: sin dividendos ni splits; dividendo=0, split=1)",
            "Ticker canónico cb:<par>; mercado CRYPTO de config/mercados.yaml (24:00 UTC + 30 min → D+1)",
            "Días sin vela = días sin operaciones o incidencias de Coinbase: no se rellenan (lista en mediciones)",
        ])


def ficha_binance(mediciones: dict, notas: list[str] | None = None) -> FichaFuente:
    ini = min((v["cobertura"][0] for v in mediciones.get("simbolos", {}).values() if v.get("cobertura", [None])[0]),
              default=None)
    fin = max((v["cobertura"][1] for v in mediciones.get("simbolos", {}).values()
               if v.get("cobertura", [None, None])[1]), default=None)
    r = mediciones.get("retraso_last_modified", {})
    retraso = (f"medido con Last-Modified de {r.get('n', 0)} ZIP mensuales: mediana {r.get('mediana_dias')} días "
               f"tras el fin de mes (mín {r.get('min_dias')}, máx {r.get('max_dias')}; desde 2023, mediana "
               f"{r.get('mediana_dias_desde_2023')}); solo control, no entra en el TimeGate"
               if r.get("n") else "no medido")
    return FichaFuente(
        fuente=FUENTE_BINANCE, nombre="Binance Vision: klines diarias BTCUSDT y ETHUSDT (control)",
        tema=TEMA, agentes=["risk"], rag="ninguno", url="https://data.binance.vision/",
        acceso="descarga", autenticacion="ninguna", coste="0 €", licencia_y_terminos=LICENCIA_BINANCE,
        cobertura_inicio=ini, cobertura_fin=fin, frecuencia="diaria (vela UTC), un ZIP por mes",
        retraso_publicacion=retraso, formato="zip con CSV sin cabecera (open_time en ms hasta 2024, en µs desde 2025)",
        volumen_estimado="unos 3 KB por ZIP mensual; < 1 MB por símbolo",
        limites=f"{POR_SEGUNDO_BINANCE:g} pet/s por host; robots.txt permite la descarga",
        estado_sonda="ok" if ini else "fallo", mediciones=mediciones,
        notas=(notas or []) + [
            "Divisa USDT: el contrato solo admite ISO de 3 letras y get_precios no tiene FX del BCE para USDT; "
            "el control se guarda fuera de precios/ (control/binance/) y se valida con todo el contrato salvo la divisa",
            "Solo meses cerrados (data/spot/monthly); el mes en curso estaría en data/spot/daily (no se descarga)",
            "No se comprueban los .CHECKSUM de Binance; se guarda el sha256 propio de cada ZIP",
            "Retraso: los meses anteriores a la apertura de Binance Vision (2021) se subieron después "
            "(Last-Modified de hasta años): la mediana desde 2023 es la representativa",
        ])


def escribir_tabla_comparacion(comparaciones: dict[str, dict], destino: Path | None = None) -> Path:
    """CSV de METADATOS de la comparación (estadísticos en pb, sin precios)."""
    filas = []
    for par, c in comparaciones.items():
        for col in ("close", "open"):
            if c.get(col):
                est = {k: v for k, v in c[col].items() if k != "desde_2020"}
                rec = c[col].get("desde_2020") or {}
                filas.append({"par_coinbase": par, "simbolo_binance": CONTROL.get(par), "columna": col,
                              "sesiones_comunes": c["sesiones_comunes"], "desde": c["desde"], "hasta": c["hasta"],
                              "solo_coinbase": c["solo_coinbase_en_periodo_comun"],
                              "solo_binance": c["solo_binance_en_periodo_comun"], **est,
                              "mediana_abs_pb_desde_2020": rec.get("mediana_abs_pb"),
                              "p95_abs_pb_desde_2020": rec.get("p95_abs_pb")})
    ruta = destino or resultados_dir(TEMA) / "tablas" / "cripto_comparacion_binance_coinbase.csv"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(ruta, index=False, encoding="utf-8")
    return ruta


# --- CLI ----------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ingesta.mercados_series.fuentes.cripto",
                                 description="Cripto (bloque 6): Coinbase (principal) y Binance Vision (control)")
    ap.add_argument("--coinbase", action="store_true", help="descarga incremental y escribe precios/coinbase")
    ap.add_argument("--binance", action="store_true", help="descarga (reanudable) los ZIP mensuales de control")
    ap.add_argument("--comparar", action="store_true", help="mide Binance frente a Coinbase")
    ap.add_argument("--fichas", action="store_true", help="guarda las fichas coinbase y binance_vision")
    ap.add_argument("--desde-binance", default=INICIO_BINANCE)
    ap.add_argument("--hasta-binance", default=None)
    a = ap.parse_args(argv)
    med_cb: dict[str, Any] = {"pares": {}}
    med_bn: dict[str, Any] = {"simbolos": {}}
    if a.coinbase:
        res = actualizar_coinbase()
        for par, meta in res.items():
            print(f"coinbase {par}: {meta['peticiones']} peticiones, {meta['filas_nuevas']} velas nuevas")
        med_cb["ultima_descarga"] = res
    metas_bn: list[dict] = []
    if a.binance:
        for sim in CONTROL.values():
            velas, metas = descargar_binance(sim, a.desde_binance, a.hasta_binance)
            metas_bn += metas
            tabla = tabla_binance(velas, sim, _ahora())
            if len(tabla):
                guardar_control(tabla)
            ok = sum(m.get("estado") == "ok" for m in metas)
            print(f"binance {sim}: {ok}/{len(metas)} meses, {len(tabla)} velas")
        med_bn["retraso_last_modified"] = retraso_binance(metas_bn)
        med_bn["meses_sin_fichero"] = [m["mes"] for m in metas_bn if m.get("estado") != "ok"][:50]
    comparaciones = {}
    for par, sim in CONTROL.items():
        cb = almacen.leer_precios(PREFIJO_COINBASE + par, PROVEEDOR_COINBASE)
        bn = leer_control(PREFIJO_BINANCE + sim)
        h = huecos(cb)
        med_cb["pares"][par] = {"velas": len(cb), "cobertura": list(_cobertura(cb)),
                                "dias_sin_vela": len(h), "ejemplos_sin_vela": h[:20]}
        med_bn["simbolos"][sim] = {"velas": len(bn), "cobertura": list(_cobertura(bn)),
                                   "dias_sin_vela": len(huecos(bn)), "ejemplos_sin_vela": huecos(bn)[:20]}
        if a.comparar and len(cb) and len(bn):
            comparaciones[par] = comparar(cb, bn)
            c = comparaciones[par]["close"]
            print(f"{par} vs {sim}: {comparaciones[par]['sesiones_comunes']} sesiones, cierre mediana "
                  f"{c['mediana_abs_pb']} pb, p95 {c['p95_abs_pb']} pb; desde 2020 {c.get('desde_2020')}")
    if comparaciones:
        med_bn["comparacion_con_coinbase"] = comparaciones
        print("tabla:", escribir_tabla_comparacion(comparaciones))
    if a.fichas:
        if not a.binance:
            # Retraso medido con los .meta.json archivados (sin red).
            metas_bn = []
            for sim in CONTROL.values():
                for p in sorted((_crudo_dir(FUENTE_BINANCE) / sim).glob("*.meta.json")):
                    m = json.loads(p.read_text(encoding="utf-8"))
                    m["mes"] = p.name.split("-1d-")[1][:7]
                    metas_bn.append(m)
            med_bn["retraso_last_modified"] = retraso_binance(metas_bn)
        print(guardar_ficha(ficha_coinbase(med_cb)))
        print(guardar_ficha(ficha_binance(med_bn)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
