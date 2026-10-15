"""Proveedor de precios Yahoo Finance (yfinance 0.2.66) del Tema 1 (bloque 6).

Descarga (``descargar_vista``): ``Ticker.history(period='max', auto_adjust=False, actions=True,
timeout=90)`` con reintentos. Se VERIFICA el inicio: si ``period='max'`` devuelve vacío o empieza
más de ``HOLGURA_INICIO`` después de ``firstTradeDate`` (metadatos de Yahoo), se reintenta con
``start`` explícito; si sigue recortado, se avisa en ``info['avisos']`` (nunca se recorta en silencio).
yfinance es la excepción a ``ingesta.comun.http``: librería con su propio cliente; el ritmo se fija
aquí (``RITMO``: <= 0,5 tickers/s).

Conversión al contrato (``a_contrato``, sin red):

- AS-TRADED: Yahoo da ``Open/High/Low/Close`` y ``Dividends`` divididos por el producto de los splits
  posteriores y ``Volume`` multiplicado por él; ``timegate.des_ajustar_yahoo`` lo deshace (OHLC y
  dividendo × producto, volumen ÷ producto). ``Adj Close`` se descarta. ``split`` = ratio de la
  sesión (1.0 sin evento), ``dividendo`` = importe por acción de la sesión ex.
- Eventos en filas sin precio (dividendo fechado en un día sin sesión): pasan a la sesión siguiente.
- Mercado: ``exchangeName`` / huso de ``history_metadata`` → clave de ``config/mercados.yaml``
  (``mercado_de``); ETF de NYSE Arca → ``ARCX``, fondos → ``NAV_US``, EURO STOXX 50 → ``STOXX_IDX``,
  divisas → ``FX_YAHOO`` (todos en ``config/mercados.yaml``; el mercado declarado en
  ``universo.yaml`` manda sobre ``exchangeName``).
- Divisa: ``currency`` tal cual (``GBp``, ``ZAc``, ``ILA`` son subunidades). Dividendos de valores en
  subunidad que Yahoo da en la unidad (dividendo < 0,1 % del cierre previo) se pasan a subunidad (×100).
- Visibilidad: ``disponible_utc`` = cierre local + 30 min (``visibilidad.disponible_cierre_serie``),
  ``fecha_publicacion`` = primera decisión (16:30 NY) posterior. Se descarta la vela incompleta
  (``disponible_utc > descargado``).
- Notas (``info['notas']``): contra-splits/consolidaciones (ratio < 1, p. ej. VOD.L 2006 y 2014) y
  splits fraccionarios pequeños (1 < ratio < 1,5: dividendos en acciones, típicos de Taiwán).

Lote reanudable (``descargar_lote``): registro JSONL por ticker en
``<corpus>/mercados_series/logs/yahoo_<etiqueta>.jsonl``; con ``reanudar=True`` los tickers ya
descargados bien hoy no se piden otra vez. Metadatos por ticker (sin precios) en
``results/ingesta/mercados_series/tablas/yahoo_<etiqueta>.csv``. CLI::

    python -m ingesta.mercados_series.proveedores.yahoo nucleo
    python -m ingesta.mercados_series.proveedores.yahoo ampliado --fichero <csv con columna ticker>
    python -m ingesta.mercados_series.proveedores.yahoo ficha
    python -m ingesta.mercados_series.proveedores.yahoo posproceso

Posproceso del nivel ampliado (``posproceso``, solo lectura de lo descargado): cobertura por índice
de origen y por mercado (tickers con datos, inicio mediano, sesiones), fallidos clasificados desde el
registro (vacío, deslistado, otro) y sin cotización reciente, e incidencias de ``calidad``; escribe
``tablas/yahoo_ampliado_cobertura.csv`` y la ficha ``yahoo`` (núcleo + ampliado, con el estado
«descarga en curso: N de M» mientras no se haya intentado toda la lista).
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingesta.comun.rutas import corpus_dir, resultados_dir
from ingesta.mercados_series import almacen, calidad, visibilidad
from ingesta.mercados_series.contrato import (
    COLUMNAS_PRECIOS,
    descartar_velas_incompletas,
    primera_decision_vec,
    validar_precios,
)
from ingesta.mercados_series.divisas import SUBUNIDADES
from ingesta.mercados_series.timegate import COLUMNAS_YAHOO, des_ajustar_yahoo

log = logging.getLogger(__name__)

TEMA = "mercados_series"
PROVEEDOR = "yahoo"
FUENTE = "yahoo"
TIMEOUT = 90
REINTENTOS = 3
RITMO = 0.5                                  # tickers por segundo, como máximo
HOLGURA_INICIO = pd.Timedelta(days=30)       # inicio aceptable respecto a firstTradeDate
UMBRAL_DIVIDENDO_UNIDAD = 0.001              # dividendo/cierre por debajo: Yahoo lo da en la unidad

# --- Núcleo (lista fija) ------------------------------------------------------------------------
# 42 ETF de la propuesta (reconstruidos: los 23 ETF del catálogo de la Fase 1 sin RSP + 19 de
# índice, país, renta fija y materias primas habituales; la lista original no está en el repo).
ETF_PROPUESTA = (
    "SPY", "QQQ", "DIA", "MDY", "IJR", "IWM", "IWD", "IWF",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    "MTUM", "QUAL", "USMV", "SPLV", "SPHB",
    "EFA", "EEM", "VGK", "EWJ", "FXI",
    "SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "EMB",
    "GLD", "SLV", "USO", "DBC", "VNQ", "UUP",
)
ETF_EXTRA = ("AGG", "RSP", "VBMFX", "IBIT", "ETHA")
INDICES = ("^GSPC", "^NDX", "^RUT", "^STOXX50E", "^GDAXI", "^FCHI", "^FTSE", "^IBEX", "^N225", "^HSI",
           "000001.SS", "^BVSP", "^NSEI", "^KS11", "^TWII", "^VIX")
FX = ("EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCNY=X")
FUTUROS = ("ES=F", "CL=F", "GC=F", "ZN=F", "HG=F")
ACCIONES = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "XOM", "JNJ", "SAN.MC", "ITX.MC", "SAP.DE",
            "ASML.AS", "MC.PA", "NESN.SW", "SHEL.L", "VOD.L", "7203.T", "6758.T", "2330.TW", "0700.HK")
NUCLEO = ETF_PROPUESTA + ETF_EXTRA + INDICES + FX + FUTUROS + ACCIONES

# --- Mercado ------------------------------------------------------------------------------------
# exchangeName de Yahoo → clave de mercados.yaml.
MAPA_EXCHANGE = {
    "NMS": "XNAS", "NGM": "XNAS", "NCM": "XNAS", "NAS": "XNAS", "NIM": "XNAS",
    "NYQ": "XNYS", "ASE": "XNYS", "SNP": "XNYS", "WCB": "XNYS", "DJI": "XNYS",
    "PCX": "ARCX", "BTS": "ARCX",
    "CBO": "CBOE_IDX", "CXI": "CBOE_IDX",
    "CME": "CME", "CMX": "CME", "NYM": "CME", "CBT": "CME",
    "CCY": "FX_YAHOO",
    "LSE": "XLON", "FGI": "XLON", "GER": "XETR", "ETR": "XETR", "PAR": "XPAR", "AMS": "XAMS",
    "MCE": "XMAD", "EBS": "XSWX", "MIL": "XMIL",
    "JPX": "XTKS", "OSA": "XTKS", "TYO": "XTKS", "HKG": "XHKG", "SHH": "XSHG", "SHZ": "XSHG",
    "NSI": "XNSE", "BSE": "XBOM", "KSC": "XKRX", "KOE": "XKRX", "TAI": "XTAI", "SAO": "BVMF",
    "TOR": "XTSE", "ASX": "XASX", "JNB": "XJSE", "TLV": "XTAE",
}
# Huso de la bolsa → mercado (si el exchangeName no está en el mapa).
MAPA_HUSO = {
    "America/New_York": "XNYS", "Europe/London": "XLON", "Europe/Berlin": "XETR", "Europe/Paris": "XPAR",
    "Europe/Amsterdam": "XAMS", "Europe/Madrid": "XMAD", "Europe/Zurich": "XSWX", "Europe/Rome": "XMIL",
    "Asia/Tokyo": "XTKS", "Asia/Hong_Kong": "XHKG", "Asia/Shanghai": "XSHG", "Asia/Kolkata": "XNSE",
    "Asia/Seoul": "XKRX", "Asia/Taipei": "XTAI", "America/Sao_Paulo": "BVMF", "America/Toronto": "XTSE",
    "Australia/Sydney": "XASX", "Africa/Johannesburg": "XJSE", "Asia/Jerusalem": "XTAE", "UTC": "CRYPTO",
}
# Excepciones por ticker (el exchange de Yahoo no dice la hora real del cierre).
MERCADO_POR_TICKER = {"^STOXX50E": "STOXX_IDX"}


@functools.lru_cache(maxsize=1)
def mercados_declarados() -> dict[str, str]:
    """``ticker_yahoo -> mercado`` declarado en ``config/universo.yaml`` (vacío si no se puede leer)."""
    from ingesta.mercados_series import universo  # import diferido: universo no depende de este módulo

    try:
        df = universo.tabla_universo(universo.cargar_universo())
    except (OSError, ValueError, KeyError) as e:
        log.warning("universo.yaml no disponible para el mercado declarado: %s", e)
        return {}
    df = df.dropna(subset=["ticker_yahoo", "mercado"])
    return {str(t): str(m) for t, m in zip(df["ticker_yahoo"], df["mercado"]) if str(m).strip()}


def mercado_de(ticker: str, meta: dict[str, Any]) -> str:
    """Clave de mercado de ``ticker``. ``KeyError`` si no se sabe.

    Prioridad: ``MERCADO_POR_TICKER`` > mercado declarado en ``universo.yaml`` (la hora real del
    cierre: ICE_DXY a las 17:00, ICE_MOVE...) > ``exchangeName`` de Yahoo > huso de la bolsa. El
    respaldo por huso avisa: puede dar un cierre anterior al real (fuga de una sesión).
    """
    if ticker in MERCADO_POR_TICKER:
        return MERCADO_POR_TICKER[ticker]
    declarado = mercados_declarados().get(ticker)
    if declarado:
        return declarado
    if meta.get("instrumentType") == "MUTUALFUND":
        return "NAV_US"
    if meta.get("instrumentType") == "CRYPTOCURRENCY":
        return "CRYPTO"
    ex = meta.get("exchangeName")
    if ex in MAPA_EXCHANGE:
        return MAPA_EXCHANGE[ex]
    huso = meta.get("exchangeTimezoneName")
    if huso in MAPA_HUSO:
        log.warning("%s: exchangeName=%r sin mapa; mercado por huso %s -> %s (verificar la hora de cierre)",
                    ticker, ex, huso, MAPA_HUSO[huso])
        return MAPA_HUSO[huso]
    raise KeyError(f"{ticker}: sin mercado para exchangeName={ex!r}, huso={huso!r}")


# --- Descarga -----------------------------------------------------------------------------------
def _history(ticker: str, **kw) -> tuple[pd.DataFrame, dict[str, Any]]:
    import yfinance as yf

    tk = yf.Ticker(ticker)
    ultimo: Exception | None = None
    for k in range(REINTENTOS):
        try:
            h = tk.history(auto_adjust=False, actions=True, timeout=TIMEOUT, raise_errors=True, **kw)
            return h, dict(tk.history_metadata or {})
        except Exception as e:   # noqa: BLE001 - yfinance lanza tipos variados
            ultimo = e
            time.sleep(3.0 * (k + 1))
    raise RuntimeError(f"{ticker}: yfinance falló {REINTENTOS} veces: {ultimo}")


def _primera_cotizacion(meta: dict[str, Any]) -> pd.Timestamp | None:
    v = meta.get("firstTradeDate")
    if v is None:
        return None
    return pd.Timestamp(int(v), unit="s").normalize()


def descargar_vista(ticker: str) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Vista de Yahoo (``auto_adjust=False``) de ``ticker`` con el inicio verificado.

    Devuelve ``(vista, metadatos, avisos)``. ``RuntimeError`` si no hay datos ni con ``start``.
    """
    avisos: list[str] = []
    try:
        vista, meta = _history(ticker, period="max")
    except RuntimeError as e:
        vista, meta = pd.DataFrame(), {}
        avisos.append(f"period='max' falló: {e}")
    primera = _primera_cotizacion(meta)
    inicio = pd.Timestamp(vista.index.min()).tz_localize(None).normalize() if len(vista) else None
    if inicio is None or (primera is not None and inicio > primera + HOLGURA_INICIO):
        start = (primera or pd.Timestamp("1900-01-01")).strftime("%Y-%m-%d")
        avisos.append(f"period='max' devolvió {'vacío' if inicio is None else f'inicio {inicio.date()}'} "
                      f"con firstTradeDate {primera.date() if primera is not None else None}: reintento con start={start}")
        vista2, meta2 = _history(ticker, start=start)
        if len(vista2) and (inicio is None or vista2.index.min() < vista.index.min()):
            vista, meta = vista2, meta2 or meta
            inicio = pd.Timestamp(vista.index.min()).tz_localize(None).normalize()
        if inicio is None:
            raise RuntimeError(f"{ticker}: sin datos ni con period='max' ni con start={start}")
        if primera is not None and inicio > primera + HOLGURA_INICIO:
            avisos.append(f"inicio {inicio.date()} posterior a firstTradeDate {primera.date()}: historia "
                          f"recortada por Yahoo (se guarda lo que hay)")
    return vista, meta, avisos


def _mover_eventos_sin_precio(vista: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Quita filas sin ``Close``; sus dividendos y splits pasan a la siguiente fila con precio."""
    v = vista.copy()
    sin = v["Close"].isna().to_numpy()
    if not sin.any():
        return v, 0
    movidos = 0
    div = v["Dividends"].fillna(0.0).to_numpy(dtype=float).copy()
    spl = v["Stock Splits"].fillna(0.0).to_numpy(dtype=float).copy()
    idx_validos = np.flatnonzero(~sin)
    for i in np.flatnonzero(sin):
        if div[i] == 0 and spl[i] == 0:
            continue
        sig = idx_validos[idx_validos > i]
        if len(sig):
            j = sig[0]
            div[j] += div[i]
            if spl[i] != 0:
                spl[j] = spl[i] if spl[j] == 0 else spl[j] * spl[i]
            movidos += 1
    v["Dividends"], v["Stock Splits"] = div, spl
    return v.loc[~sin], movidos


def a_contrato(vista: pd.DataFrame, meta: dict[str, Any], ticker: str, descargado: pd.Timestamp,
               mercado: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Vista de Yahoo → tabla del contrato de precios as-traded (sin validar) e info del ticker.

    No usa red. ``descargado``: instante UTC de la descarga (descarta velas incompletas).
    ``mercado``: si se omite, ``mercado_de(ticker, meta)``.
    """
    mercado = mercado or mercado_de(ticker, meta)
    divisa = meta.get("currency") or "USD"
    notas: list[str] = []
    v, movidos = _mover_eventos_sin_precio(vista[list(COLUMNAS_YAHOO)])
    if movidos:
        notas.append(f"{movidos} eventos en días sin precio movidos a la sesión siguiente")
    v = v.fillna({"Dividends": 0.0, "Stock Splits": 0.0, "Volume": 0.0})
    p = des_ajustar_yahoo(v)
    if divisa in SUBUNIDADES:
        previo = p["close"].shift(1)
        en_unidad = (p["dividendo"] > 0) & (p["dividendo"] / previo < UMBRAL_DIVIDENDO_UNIDAD)
        if en_unidad.any():
            p.loc[en_unidad, "dividendo"] *= SUBUNIDADES[divisa][1]
            notas.append(f"{int(en_unidad.sum())} dividendos en {SUBUNIDADES[divisa][0]} pasados a {divisa} (×100)")
    contra = p.loc[p["split"] < 1.0, "sesion"]
    if len(contra):
        notas.append(f"contra-splits/consolidaciones (ratio < 1) en {[str(s.date()) for s in contra]}")
    fracc = p.loc[(p["split"] > 1.0) & (p["split"] < 1.5), "sesion"]
    if len(fracc):
        notas.append(f"splits fraccionarios (1 < ratio < 1,5; dividendos en acciones) en "
                     f"{len(fracc)} sesiones, p. ej. {[str(s.date()) for s in fracc[:3]]}")
    disponible = visibilidad.disponible_cierre_serie(p["sesion"], mercado)
    out = pd.DataFrame({
        "ticker": ticker, "sesion": p["sesion"], "mercado": mercado,
        "open": p["open"], "high": p["high"], "low": p["low"], "close": p["close"],
        "volume": p["volume"], "dividendo": p["dividendo"], "split": p["split"], "divisa": divisa,
        "fecha_publicacion": primera_decision_vec(disponible), "disponible_utc": disponible,
        "fuente": FUENTE, "descargado": pd.Timestamp(descargado).tz_convert("UTC")
        if pd.Timestamp(descargado).tzinfo else pd.Timestamp(descargado, tz="UTC"),
    })[list(COLUMNAS_PRECIOS)]
    n0 = len(out)
    out = descartar_velas_incompletas(out)
    info = {
        "ticker": ticker, "mercado": mercado, "divisa": divisa, "clase": meta.get("instrumentType"),
        "exchange": meta.get("exchangeName"), "huso": meta.get("exchangeTimezoneName"),
        "primera_cotizacion": str(_primera_cotizacion(meta).date()) if _primera_cotizacion(meta) is not None else None,
        "inicio": str(out["sesion"].min().date()) if len(out) else None,
        "fin": str(out["sesion"].max().date()) if len(out) else None,
        "sesiones": len(out), "velas_incompletas": n0 - len(out),
        "splits": int((out["split"] != 1.0).sum()), "dividendos": int((out["dividendo"] > 0).sum()),
        "notas": notas,
    }
    return out.reset_index(drop=True), info


def descargar(ticker: str, *, ahora: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Descarga ``ticker`` y lo devuelve as-traded y validado."""
    descargado = ahora or pd.Timestamp.now(tz="UTC")
    vista, meta, avisos = descargar_vista(ticker)
    tabla, info = a_contrato(vista, meta, ticker, descargado)
    tabla = validar_precios(tabla)
    info["avisos"] = avisos
    return tabla, info


class ProveedorYahoo:
    """``ProveedorPrecios`` de Yahoo Finance."""

    nombre = PROVEEDOR

    def descargar(self, ticker: str) -> pd.DataFrame:
        return descargar(ticker)[0]


# --- Lote reanudable ----------------------------------------------------------------------------
def _ruta_registro(etiqueta: str) -> Path:
    carpeta = corpus_dir(TEMA) / "logs"
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta / f"yahoo_{etiqueta}.jsonl"


def leer_registro(etiqueta: str) -> pd.DataFrame:
    """Registro del lote ``etiqueta`` (una fila por intento; la última de cada ticker manda)."""
    ruta = _ruta_registro(etiqueta)
    if not ruta.exists():
        return pd.DataFrame()
    filas = [json.loads(x) for x in ruta.read_text(encoding="utf-8").splitlines() if x.strip()]
    return pd.DataFrame(filas)


# Revisiones de Yahoo que no rechazan el ticker al refrescar (almacen.escribir_precios): el volumen de la
# última sesión (provisional cuando se descarga poco después del cierre: MSFT y NVDA del 2026-10-12) y
# los dividendos corregidos (ITX.MC 2002-2008, MC.PA 2008-11-05, VOD.L 2026-06-04). Se conserva la vela
# guardada y la discrepancia queda en <corpus>/mercados_series/logs/revisiones_precios_yahoo.csv.
REVISIONES_TOLERADAS = ("volume", "dividendo")


def descargar_lote(tickers: list[str] | tuple[str, ...], *, etiqueta: str = "nucleo", reanudar: bool = True,
                   escribir: bool = True, ritmo: float = RITMO) -> pd.DataFrame:
    """Descarga ``tickers`` uno a uno (<= ``ritmo`` por segundo) y los guarda en ``precios/yahoo``.

    Registro JSONL en ``<corpus>/mercados_series/logs/yahoo_<etiqueta>.jsonl`` (estado ``ok`` o
    ``error`` por ticker). Con ``reanudar`` se saltan los tickers con ``ok`` de hoy (UTC). Devuelve
    la última fila del registro de cada ticker pedido.
    """
    ruta = _ruta_registro(etiqueta)
    hechos: set[str] = set()
    hoy = datetime.now(timezone.utc).date().isoformat()
    previo = leer_registro(etiqueta)
    if reanudar and len(previo):
        ok = previo[(previo["estado"] == "ok") & previo["descargado"].astype(str).str.startswith(hoy)]
        hechos = set(ok["ticker"])
    espera = 1.0 / ritmo
    for tk in tickers:
        if tk in hechos:
            continue
        t0 = time.monotonic()
        fila: dict[str, Any] = {"ticker": tk, "descargado": datetime.now(timezone.utc).isoformat()}
        try:
            tabla, info = descargar(tk)
            if escribir:
                almacen.escribir_precios(tabla, PROVEEDOR, revisiones_toleradas=REVISIONES_TOLERADAS)
            fila.update(info, estado="ok", error=None)
        except Exception as e:   # noqa: BLE001 - se registra y se sigue con el lote
            fila.update(estado="error", error=f"{type(e).__name__}: {e}"[:500])
            log.warning("%s: %s", tk, fila["error"])
        fila["segundos"] = round(time.monotonic() - t0, 2)
        with ruta.open("a", encoding="utf-8") as f:
            f.write(json.dumps(fila, ensure_ascii=False, default=str) + "\n")
        resto = espera - (time.monotonic() - t0)
        if resto > 0:
            time.sleep(resto)
    reg = leer_registro(etiqueta)
    reg = reg[reg["ticker"].isin(list(tickers))].drop_duplicates("ticker", keep="last")
    return reg.reset_index(drop=True)


def guardar_metadatos(etiqueta: str = "nucleo") -> Path:
    """CSV de metadatos del lote (sin precios) en ``results/ingesta/mercados_series/tablas/``."""
    reg = leer_registro(etiqueta).drop_duplicates("ticker", keep="last")
    cols = ["ticker", "estado", "clase", "mercado", "exchange", "huso", "divisa", "primera_cotizacion",
            "inicio", "fin", "sesiones", "splits", "dividendos", "velas_incompletas", "avisos", "notas",
            "error", "descargado", "segundos"]
    reg = reg.reindex(columns=cols)
    for c in ("avisos", "notas"):
        reg[c] = reg[c].map(lambda x: " | ".join(x) if isinstance(x, list) else x)
    destino = resultados_dir(TEMA) / "tablas" / f"yahoo_{etiqueta}.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    reg.to_csv(destino, index=False, encoding="utf-8")
    return destino


# --- Posproceso del nivel ampliado (solo lectura de lo descargado) -------------------------------
SUFIJO_INDICE = {"ftse100": ".L", "hang_seng": ".HK", "kospi200": ".KS", "nifty50": ".NS", "smi": ".SW"}
FICHERO_LISTA_AMPLIADO = "tickers_ampliado.csv"           # en <corpus>/mercados_series/logs/
FICHERO_COBERTURA_AMPLIADO = "yahoo_ampliado_cobertura.csv"
DIAS_SIN_COTIZAR = 30                                      # fin anterior a la descarga en más de esto
COLUMNAS_COBERTURA = ("grupo", "clave", "n_lista", "n_intentados", "n_ok", "n_vacio", "n_deslistado",
                      "n_otro_error", "n_sin_cotizacion_reciente", "inicio_min", "inicio_mediana",
                      "inicio_antes_2000", "sesiones_mediana", "sesiones_total",
                      *(f"inc_{t}" for t in calidad.TIPOS), "tickers_con_salto")
_COLS_REGISTRO = ["estado", "error", "inicio", "fin", "sesiones", "mercado", "descargado"]


def ticker_yahoo_constituyente(indice: str, ticker: Any) -> str:
    """Ticker de Yahoo de un constituyente (la regla con la que se construyó la lista ampliada).

    S&P 500: ``BRK.B`` → ``BRK-B``. Hang Seng: código con ceros a la izquierda (``5`` → ``0005.HK``).
    FTSE 100, KOSPI 200, NIFTY 50 y SMI: sufijo de la bolsa. El resto (DAX, CAC 40, IBEX 35, FTSE MIB,
    AEX) ya viene con sufijo; un ticker con punto se deja tal cual.
    """
    t = str(ticker).strip()
    if indice == "sp500":
        return t.replace(".", "-")
    if indice == "hang_seng" and t.isdigit():
        return f"{int(t):04d}.HK"
    if indice in SUFIJO_INDICE and "." not in t:
        return t + SUFIJO_INDICE[indice]
    return t


def indices_de_tickers(raiz: Path | None = None) -> pd.DataFrame:
    """``indice, ticker`` (de Yahoo) de los constituyentes guardados (S&P 500 histórico y fotos actuales)."""
    carpeta = (raiz or corpus_dir(TEMA)) / "constituyentes"
    partes = []
    for f in sorted(carpeta.glob("*.parquet")) if carpeta.exists() else []:
        d = pd.read_parquet(f, columns=["indice", "ticker"])
        partes.append(pd.DataFrame({"indice": d["indice"].astype(str),
                                    "ticker": [ticker_yahoo_constituyente(i, t)
                                               for i, t in zip(d["indice"], d["ticker"])]}))
    if not partes:
        return pd.DataFrame(columns=["indice", "ticker"])
    return pd.concat(partes, ignore_index=True).drop_duplicates().reset_index(drop=True)


def clasificar_error(error: Any) -> str:
    """``deslistado``, ``vacio`` u ``otro`` según el texto del error del registro."""
    t = str(error or "").lower()
    if "delisted" in t or "deslist" in t:
        return "deslistado"
    if any(k in t for k in ("sin datos", "no data", "empty", "no price data", "not found")):
        return "vacio"
    return "otro"


def calidad_por_ticker(tickers: list[str], clases: dict[str, str] | None = None) -> pd.DataFrame:
    """Incidencias de ``calidad`` por ticker (índice ``ticker``) leyendo los parquet guardados (solo lectura)."""
    filas = []
    for tk in tickers:
        try:
            p = almacen.leer_precios(tk, PROVEEDOR)
        except Exception as e:   # noqa: BLE001 - fichero en escritura o ausente: se anota y se sigue
            log.warning("%s: no se pudo leer para calidad: %s", tk, e)
            continue
        if len(p):
            filas.append(calidad.resumen_calidad(p, clases))
    if not filas:
        return pd.DataFrame(columns=[*calidad.TIPOS, "sesiones"])
    return pd.concat(filas)


def cobertura_ampliado(registro: pd.DataFrame, lista: list[str], indices: pd.DataFrame,
                       calidad_tk: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cobertura por índice de origen, por mercado y total (sin red ni E/S).

    ``registro``: el JSONL del lote (la última fila de cada ticker manda). ``lista``: tickers pedidos
    (los no intentados cuentan en ``n_lista`` y no en ``n_intentados``). ``indices``: ``indice, ticker``
    (un ticker puede estar en varios índices: AIR.PA en DAX y CAC 40). ``calidad_tk``: salida de
    ``calidad_por_ticker`` (opcional).
    """
    if len(registro):
        reg = registro.drop_duplicates("ticker", keep="last").set_index("ticker").reindex(columns=_COLS_REGISTRO)
    else:
        reg = pd.DataFrame(columns=_COLS_REGISTRO)
    base = pd.DataFrame(index=pd.Index(list(dict.fromkeys(lista)), name="ticker")).join(reg)
    fallo = base["estado"].notna() & (base["estado"] != "ok")
    base["tipo_fallo"] = base["error"].map(clasificar_error).where(fallo)
    base["inicio"] = pd.to_datetime(base["inicio"], errors="coerce")
    fin = pd.to_datetime(base["fin"], errors="coerce")
    desc = pd.to_datetime(base["descargado"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    base["sin_cotizacion_reciente"] = (base["estado"] == "ok") & ((desc - fin).dt.days > DIAS_SIN_COTIZAR)
    cal = calidad_tk if calidad_tk is not None else pd.DataFrame()
    base = base.join(cal.reindex(columns=list(calidad.TIPOS)).add_prefix("inc_"))

    def _fila(grupo: str, clave: str, g: pd.DataFrame) -> dict[str, Any]:
        ok = g[g["estado"] == "ok"]
        ses = pd.to_numeric(ok["sesiones"], errors="coerce")
        hay = ok["inicio"].notna().any()
        f = {"grupo": grupo, "clave": clave, "n_lista": int(len(g)), "n_intentados": int(g["estado"].notna().sum()),
             "n_ok": int(len(ok)), "n_vacio": int((g["tipo_fallo"] == "vacio").sum()),
             "n_deslistado": int((g["tipo_fallo"] == "deslistado").sum()),
             "n_otro_error": int((g["tipo_fallo"] == "otro").sum()),
             "n_sin_cotizacion_reciente": int(g["sin_cotizacion_reciente"].sum()),
             "inicio_min": str(ok["inicio"].min().date()) if hay else None,
             "inicio_mediana": str(ok["inicio"].median().date()) if hay else None,
             "inicio_antes_2000": int((ok["inicio"] < "2000-01-01").sum()),
             "sesiones_mediana": float(ses.median()) if ses.notna().any() else None,
             "sesiones_total": int(ses.sum()) if ses.notna().any() else 0}
        for t in calidad.TIPOS:
            f[f"inc_{t}"] = int(pd.to_numeric(ok[f"inc_{t}"], errors="coerce").fillna(0).sum())
        f["tickers_con_salto"] = int((pd.to_numeric(ok["inc_salto"], errors="coerce").fillna(0) > 0).sum())
        return f

    filas = [_fila("total", "total", base)]
    idx = indices[indices["ticker"].isin(base.index)]
    for indice, g in idx.groupby("indice", sort=True):
        filas.append(_fila("indice", str(indice), base.loc[list(dict.fromkeys(g["ticker"]))]))
    sin = base.loc[~base.index.isin(idx["ticker"])]
    if len(sin):
        filas.append(_fila("indice", "sin_indice", sin))
    for mercado, g in base[base["mercado"].notna()].groupby("mercado", sort=True):
        filas.append(_fila("mercado", str(mercado), g))
    return pd.DataFrame(filas, columns=list(COLUMNAS_COBERTURA))


def estado_descarga(n_intentados: int, n_lista: int) -> str:
    """``'completa (M de M)'`` o ``'descarga en curso: N de M'``."""
    if n_intentados >= n_lista:
        return f"completa ({n_lista} de {n_lista})"
    return f"descarga en curso: {n_intentados} de {n_lista}"


def posproceso(etiqueta: str = "ampliado", fichero: Path | None = None, *, con_calidad: bool = True,
               escribir: bool = True) -> dict[str, Any]:
    """Cobertura, fallidos y calidad de lo YA descargado del lote ``etiqueta`` (solo lectura).

    Escribe ``tablas/yahoo_ampliado_cobertura.csv`` (metadatos, sin precios) y devuelve el resumen
    para la ficha. Se puede ejecutar con la descarga en curso: el estado lo dice.
    """
    lista = _tickers_de_fichero(fichero)
    reg = leer_registro(etiqueta)
    ultimo = reg.drop_duplicates("ticker", keep="last") if len(reg) else pd.DataFrame(columns=["ticker", "estado"])
    ok = ultimo[ultimo["estado"] == "ok"]
    clases = dict(zip(ok["ticker"], ok["clase"])) if len(ok) else {}
    cal = calidad_por_ticker(list(ok["ticker"]), clases) if con_calidad and len(ok) else None
    tabla = cobertura_ampliado(reg, lista, indices_de_tickers(), cal)
    tot = tabla.iloc[0]
    fallidos = ultimo[ultimo["estado"] != "ok"]
    resumen = {
        "estado": estado_descarga(int(tot["n_intentados"]), int(tot["n_lista"])),
        "n_lista": int(tot["n_lista"]), "n_intentados": int(tot["n_intentados"]), "n_ok": int(tot["n_ok"]),
        "fallidos": {str(r.ticker): clasificar_error(r.error) for r in fallidos.itertuples()},
        "sin_cotizacion_reciente": int(tot["n_sin_cotizacion_reciente"]),
        "por_indice": {r.clave: {"lista": int(r.n_lista), "intentados": int(r.n_intentados), "ok": int(r.n_ok),
                                 "inicio_mediana": r.inicio_mediana}
                       for r in tabla[tabla["grupo"] == "indice"].itertuples()},
        "incidencias_calidad": {t: int(tot[f"inc_{t}"]) for t in calidad.TIPOS} if cal is not None else None,
        "tabla": f"results/ingesta/mercados_series/tablas/{FICHERO_COBERTURA_AMPLIADO}",
        "medido_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
    }
    if escribir:
        destino = resultados_dir(TEMA) / "tablas" / FICHERO_COBERTURA_AMPLIADO
        destino.parent.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(destino, index=False, encoding="utf-8")
    return resumen


# --- Ficha --------------------------------------------------------------------------------------
def medir_cobertura(etiqueta: str = "nucleo") -> dict[str, Any]:
    """Cobertura medida de lo guardado: por mercado y por clase (n, inicio mínimo y mediano, fin)."""
    reg = leer_registro(etiqueta).drop_duplicates("ticker", keep="last")
    ok = reg[reg["estado"] == "ok"].copy()
    ok["inicio"] = pd.to_datetime(ok["inicio"])
    ok["fin"] = pd.to_datetime(ok["fin"])
    retrasos: dict[str, Any] = {}
    for mercado, g in ok.groupby("mercado"):
        p = almacen.leer_precios(g["ticker"].iloc[0], PROVEEDOR)
        dias = (p["fecha_publicacion"] - p["sesion"]).dt.days
        retrasos[mercado] = {str(k): int(v) for k, v in dias.value_counts().sort_index().items()}

    def _agr(col: str) -> dict[str, Any]:
        return {str(k): {"n": int(len(g)), "inicio_min": str(g["inicio"].min().date()),
                         "inicio_mediana": str(g["inicio"].median().date()),
                         "fin_max": str(g["fin"].max().date())} for k, g in ok.groupby(col)}

    return {"tickers_ok": int(len(ok)), "tickers_error": int((reg["estado"] != "ok").sum()),
            "por_mercado": _agr("mercado"), "por_clase": _agr("clase"),
            "retraso_fecha_publicacion_dias_por_mercado": retrasos,
            "inicio_antes_2000": int((ok["inicio"] < "2000-01-01").sum()),
            "errores": reg.loc[reg["estado"] != "ok", ["ticker", "error"]].to_dict("records")}


def guardar_ficha_yahoo(etiqueta: str = "nucleo", ampliado: dict[str, Any] | None = None) -> Path:
    """Ficha ``yahoo`` con robots y términos declarados, la cobertura medida del núcleo y, si se da,
    el resumen del nivel ampliado (``posproceso``)."""
    from ingesta.comun.esquema import FichaFuente
    from ingesta.comun.fichas import guardar_ficha
    from ingesta.comun.http import robots_permite

    robots: dict[str, Any] = {}
    for url in ("https://query2.finance.yahoo.com/v8/finance/chart/AAPL",   # host que usa yfinance
                "https://finance.yahoo.com/quote/AAPL/history/"):
        try:
            robots[url] = robots_permite(url)
        except Exception as e:   # noqa: BLE001
            robots[url] = f"no comprobado ({type(e).__name__})"
    med = medir_cobertura(etiqueta)
    reg = leer_registro(etiqueta).drop_duplicates("ticker", keep="last")
    ok = reg[reg["estado"] == "ok"]
    med["robots_permite"] = robots
    med["segundos_medios_por_ticker"] = round(float(reg["segundos"].mean()), 2)
    med["avisos_inicio"] = {r.ticker: r.avisos for r in ok.itertuples() if isinstance(r.avisos, list) and r.avisos}
    if ampliado is not None:
        med["ampliado"] = ampliado
    estado_ampliado = ampliado["estado"] if ampliado else "sin posproceso"
    ficha = FichaFuente(
        fuente="yahoo",
        nombre="Yahoo Finance vía yfinance 0.2.66 (OHLCV diario, dividendos y splits)",
        tema=TEMA, agentes=["equity", "risk", "portfolio", "macro"], rag="serie",
        url="https://finance.yahoo.com/",
        acceso="libreria", autenticacion="ninguna", coste="0 €",
        licencia_y_terminos=("Términos de Yahoo: uso personal, no comercial; prohibida la redistribución. "
                             "robots.txt de query1/query2.finance.yahoo.com (la API de gráficos que usa "
                             "yfinance, no pública ni documentada) con 'User-agent: * / Disallow: /' "
                             f"(medido: {robots}). Riesgo DECLARADO: solo local, sin redistribuir; "
                             "alternativa de pago para la descarga final (EODHD)."),
        cobertura_inicio=str(ok["inicio"].min()) if len(ok) else None,
        cobertura_fin=str(ok["fin"].max()) if len(ok) else None,
        frecuencia="diaria (sesiones del mercado de cotización)",
        retraso_publicacion=("cierre local + 30 min (visibilidad por mercado): Europa y Asia y EE. UU. "
                             "(16:00) el mismo día de decisión; VIX (16:15), futuros CME, FX y fondos (NAV) "
                             "en D+1; medido en mediciones.retraso_fecha_publicacion_dias_por_mercado"),
        formato="parquet (as-traded, un fichero por ticker)",
        volumen_estimado=(f"{len(ok)} tickers del núcleo; ampliado: "
                          + (f"{ampliado['n_ok']} con datos de {ampliado['n_lista']} ({estado_ampliado})"
                             if ampliado else "sin posproceso")),
        limites=f"sin límite publicado; <= {RITMO} tickers/s, timeout {TIMEOUT} s, {REINTENTOS} reintentos",
        estado_sonda="ok" if len(ok) and med["tickers_error"] == 0 else ("parcial" if len(ok) else "fallo"),
        mediciones=med,
        notas=[
            "Proveedor: ingesta.mercados_series.proveedores.yahoo (auto_adjust=False, actions=True, period='max' "
            "con verificación del inicio contra firstTradeDate)",
            "As-traded: se deshacen los splits de la vista de Yahoo (OHLC y dividendos × producto de splits "
            "posteriores, volumen ÷); Adj Close nunca se guarda; el ajuste a fecha lo hace timegate.ajustar_a_fecha",
            "Mercado de cada ticker: el declarado en universo.yaml manda sobre exchangeName; ARCX, NAV_US, "
            "STOXX_IDX y FX_YAHOO están en mercados.yaml",
            "Yahoo empieza hacia 1993-2000 en muchos índices y acciones no estadounidenses: ver inicio por mercado",
            "Supervivencia: Yahoo solo sirve tickers vivos (o reutilizados); deslistados fuera (WRDS pendiente)",
            "Dividendos de valores en GBp que Yahoo da en GBP se pasan a peniques; consolidaciones (VOD.L) y "
            "dividendos en acciones de Taiwán (splits fraccionarios) se anotan",
            f"Nivel AMPLIADO (S&P 500 histórico y constituyentes actuales de 10 índices): {estado_ampliado}; "
            "cobertura por índice y mercado en tablas/yahoo_ampliado_cobertura.csv (orden posproceso); "
            "un ticker del S&P 500 histórico reutilizado por otra empresa devuelve la empresa ACTUAL",
        ],
    )
    return guardar_ficha(ficha)


def resolver_fichero(ruta: Path | str | None) -> Path:
    """Ruta de la lista de tickers: por defecto ``<corpus>/mercados_series/logs/tickers_ampliado.csv``.

    Una ruta relativa que no existe desde el directorio actual se busca en ``<corpus>/mercados_series``
    (``--fichero logs/tickers_ampliado.csv`` funciona desde la raíz del repo: la lista vive en el corpus).
    """
    if ruta is None:
        return corpus_dir(TEMA) / "logs" / FICHERO_LISTA_AMPLIADO
    r = Path(ruta)
    if not r.is_absolute() and not r.exists():
        en_corpus = corpus_dir(TEMA) / r
        if en_corpus.exists():
            return en_corpus
    return r


def _tickers_de_fichero(ruta: Path | str | None) -> list[str]:
    df = pd.read_csv(resolver_fichero(ruta))
    col = "ticker" if "ticker" in df.columns else df.columns[0]
    return list(dict.fromkeys(df[col].dropna().astype(str).str.strip()))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ingesta.mercados_series.proveedores.yahoo")
    ap.add_argument("orden", choices=["nucleo", "ampliado", "ficha", "posproceso"])
    ap.add_argument("--fichero", type=Path, help="CSV con columna 'ticker' (nivel ampliado; por defecto "
                    "<corpus>/mercados_series/logs/tickers_ampliado.csv; una ruta relativa también se busca ahí)")
    ap.add_argument("--etiqueta", default=None)
    ap.add_argument("--sin-reanudar", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    etiqueta = a.etiqueta or ("nucleo" if a.orden in ("nucleo", "ficha") else "ampliado")
    if a.orden == "posproceso":
        res = posproceso(etiqueta, a.fichero)
        print(res["estado"], {k: res[k] for k in ("n_ok", "sin_cotizacion_reciente", "incidencias_calidad")})
        print(guardar_ficha_yahoo("nucleo", ampliado=res))
        return
    if a.orden == "ficha":
        amp = posproceso(con_calidad=False, escribir=False) if len(leer_registro("ampliado")) else None
        print(guardar_ficha_yahoo(etiqueta, ampliado=amp))
        return
    tickers = list(NUCLEO) if a.orden == "nucleo" else _tickers_de_fichero(a.fichero)
    reg = descargar_lote(tickers, etiqueta=etiqueta, reanudar=not a.sin_reanudar)
    print(reg["estado"].value_counts().to_string())
    print(guardar_metadatos(etiqueta))
    if a.orden == "nucleo":
        print(guardar_ficha_yahoo(etiqueta))


if __name__ == "__main__":
    main()


__all__ = ["NUCLEO", "ETF_PROPUESTA", "MAPA_EXCHANGE", "ProveedorYahoo", "mercado_de", "descargar_vista", "a_contrato", "descargar", "descargar_lote",
           "leer_registro", "guardar_metadatos", "medir_cobertura", "guardar_ficha_yahoo",
           "ticker_yahoo_constituyente", "indices_de_tickers", "clasificar_error", "calidad_por_ticker",
           "cobertura_ampliado", "estado_descarga", "posproceso", "resolver_fichero"]
