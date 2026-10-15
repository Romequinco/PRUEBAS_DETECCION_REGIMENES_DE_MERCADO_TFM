"""Sesgo de supervivencia del S&P 500 con precios gratuitos de Yahoo (bloque 7).

Pregunta: ¿qué parte de los miembros que SALIERON del S&P 500 tiene precio en Yahoo durante su
etapa en el índice, y cuánto se equivoca un backtest que usa la composición ACTUAL?

Entrada: ``<corpus>/mercados_series/constituyentes/sp500.parquet`` (tramos point-in-time de
``fuentes/constituyentes.py``; si no existe se construye con ``descargar_sp500``).

Dos vías de precios:

- **Almacén (principal, ``medir_almacen``)**: TODOS los tramos (salidos y actuales) cuyo ticker
  (``.`` → ``-``) tiene parquet as-traded en ``<corpus>/mercados_series/precios/yahoo/`` (descarga
  ampliada del bloque 6; solo lectura). La rentabilidad total diaria (``rentabilidad_total_precios``)
  es la de ``close_aj_total`` de ``timegate.ajustar_a_fecha`` a la última ``fecha_publicacion`` del
  ticker: split → ``close_t * split_t / close_{t-1}``; dividendo → ``close_t / (close_{t-1} - D_t)``
  (método multiplicativo, el de ``Adj Close``).
- **Muestra antigua (respaldo, ``descargar_muestra``)**: 2 tramos salidos por año de salida y 60
  actuales (122 tramos, semilla fija) pedidos aparte a yfinance (<= 0,5 tickers/s, timeout 90 s,
  reintentos solo ante excepciones); en ``<corpus>/mercados_series/supervivencia/`` solo la
  rentabilidad diaria derivada (``(Close + Dividends) / Close previo - 1``) y un registro JSONL.

Pasos comunes:

1. Cobertura (``cobertura_tramo``): sesiones con precio DENTRO de ``[alta, baja)`` frente a las
   sesiones de NYSE (``exchange_calendars`` XNYS; si falta, días hábiles). Clases: ``completa``
   (>= 95 %), ``parcial``, ``fuera_de_tramo`` (el ticker tiene precio, pero solo fuera del tramo)
   y ``sin_datos``. Existir el fichero no basta.
2. Sospechas de otra empresa con el mismo ticker (``sospechas_tramo``; columna ``sospecha``):
   ``empieza_tarde`` (el precio empieza más de ``HOLGURA_INICIO`` después del alta),
   ``impropio_sp500`` (mediana del cierre as-traded < ``CIERRE_MINIMO`` o del efectivo diario <
   ``EFECTIVO_MINIMO`` dentro del tramo: un miembro del índice no cotiza así), ``ticker_q`` (sufijo
   de quiebra posterior al tramo con precio dentro del tramo), ``termina_tarde`` (salido cuya nota
   dice ticker con varios tramos o sufijo Q y cuyo precio sigue más de ``HOLGURA_FIN`` después de
   la baja) y ``reutilizado_despues`` (fuera de tramo, Yahoo empieza tras la baja). Las tres
   primeras sacan el tramo de la estimación principal (``en_estimacion=False``); ``termina_tarde``
   solo de la variante ``estricta`` (suele ser reentrada de la misma empresa); la variante
   ``con_sospechas`` las incluye todas.
3. Sesgo (``estimar_sesgo``): rentabilidad equiponderada mensual de la cartera POINT-IN-TIME con
   precio (media de los salidos y media de los actuales con precio, ponderadas por cuántos miembros
   de cada grupo había ese mes) frente a la de los miembros ACTUALES con precio (supervivientes: la
   composición de hoy aplicada hacia atrás). Limitación DECLARADA: los salidos sin precio en Yahoo
   (sobre todo quiebras y compras) no entran, y son los peores; el sesgo medido es una COTA INFERIOR.

Opciones de corrección (``OPCIONES_CORRECCION``): WRDS/CRSP (pendiente de la licencia de MIAX),
Alpha Vantage LISTING_STATUS (solo lista de deslistados, sin precios) y Tiingo (requiere cuenta del
usuario). CLI::

    python -m ingesta.mercados_series.supervivencia medir                  # vía almacén (CSV + ficha)
    python -m ingesta.mercados_series.supervivencia descargar --via muestra  # respaldo (~5 min)
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ingesta.comun.rutas import corpus_dir, resultados_dir

log = logging.getLogger(__name__)

TEMA = "mercados_series"
SEMILLA = 20261013
POR_ANIO = 2
N_ACTUALES = 60
RITMO = 0.5                       # tickers por segundo, como máximo (igual que el proveedor Yahoo)
TIMEOUT = 90
REINTENTOS = 3
UMBRAL_COMPLETA = 0.95
HOLGURA_INICIO = pd.Timedelta(days=31)     # precio que empieza después: sospecha (empieza_tarde)
HOLGURA_FIN = pd.Timedelta(days=365)       # precio que sigue después de la baja (con nota): termina_tarde
CIERRE_MINIMO = 1.0                        # USD: mediana del cierre as-traded en el tramo por debajo → impropio
EFECTIVO_MINIMO = 1e6                      # USD/día: mediana de cierre × volumen en el tramo por debajo → impropio
CLASES = ("completa", "parcial", "fuera_de_tramo", "sin_datos")
VIAS = ("almacen", "muestra")
SOSPECHAS = {
    "empieza_tarde": (f"el precio empieza más de {HOLGURA_INICIO.days} días después del alta: otra empresa con el "
                      "mismo ticker o historia incompleta"),
    "termina_tarde": (f"ticker con varios tramos o sufijo Q (nota) y precio que sigue más de {HOLGURA_FIN.days} días "
                      "después de la baja: la serie puede ser de la empresa que reutiliza el ticker"),
    "ticker_q": "sufijo Q/QQ de quiebra posterior al tramo con precio dentro del tramo: identidad dudosa",
    "impropio_sp500": (f"mediana del cierre < {CIERRE_MINIMO:g} USD o del efectivo diario < {EFECTIVO_MINIMO:,.0f} USD "
                       "dentro del tramo: impropio de un miembro del S&P 500 (otra empresa, OTC o ADR, con el ticker)"),
    "reutilizado_despues": "Yahoo empieza después de la baja: ticker reutilizado por otra empresa (probable)",
}
SOSPECHAS_EXCLUYEN = ("empieza_tarde", "impropio_sp500", "ticker_q")   # termina_tarde: solo variante estricta
COLUMNAS_COBERTURA = ("grupo", "ticker", "ticker_yahoo", "alta", "baja", "anio_salida", "sesiones_esperadas",
                      "sesiones_con_precio", "fraccion", "clase", "primera_yahoo", "ultima_yahoo", "sospecha",
                      "en_estimacion", "nota")
DECADAS = ([1995, 1999, 2009, 2019, 2026], ["1996-1999", "2000-2009", "2010-2019", "2020-2026"])
OPCIONES_CORRECCION = {
    "wrds_crsp": ("CRSP vía WRDS: precios y rentabilidades de deslistados (PERMNO, DLRET); pendiente de la "
                  "licencia de MIAX. Es la corrección completa."),
    "alpha_vantage_listing_status": ("Alpha Vantage LISTING_STATUS (state=delisted): solo la LISTA de deslistados "
                                     "con fechas, sin precios. Sirve para identificar, no para corregir rentabilidades."),
    "tiingo": ("Tiingo EOD incluye muchos deslistados, pero exige una cuenta (y clave) del usuario; plan gratuito "
               "con límites diarios. No se ha probado aquí."),
    "imputacion_delisting": ("Sin precios: imputar la rentabilidad de salida (p. ej. -30 % en bajas por quiebra, "
                             "Shumway 1997) solo como análisis de sensibilidad, declarado."),
}
NOTA_LIMITACION = ("Cota inferior: los salidos sin precio en Yahoo (quiebras, compras, tickers reutilizados) no "
                   "entran en la cartera PIT; son los de peor rentabilidad, así que el sesgo real es mayor.")


# --- Rutas --------------------------------------------------------------------------------------
def dir_supervivencia(raiz: Path | None = None) -> Path:
    d = (Path(raiz) if raiz else corpus_dir(TEMA)) / "supervivencia"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ruta_registro(raiz: Path | None = None) -> Path:
    return dir_supervivencia(raiz) / "registro.jsonl"


def cargar_tramos(raiz: Path | None = None) -> pd.DataFrame:
    """Tramos del S&P 500; si no existe el parquet, se construye con ``constituyentes.descargar_sp500``."""
    base = Path(raiz) if raiz else corpus_dir(TEMA)
    ruta = base / "constituyentes" / "sp500.parquet"
    if not ruta.exists():
        from ingesta.mercados_series.fuentes import constituyentes
        constituyentes.descargar_sp500(raiz)
    return pd.read_parquet(ruta)


# --- Tramos y muestra ---------------------------------------------------------------------------
def ticker_yahoo(ticker: str) -> str:
    """Ticker del fichero del S&P 500 → símbolo de Yahoo (``BRK.B`` → ``BRK-B``)."""
    return str(ticker).strip().replace(".", "-")


def todos_los_tramos(tramos: pd.DataFrame) -> pd.DataFrame:
    """Todos los tramos con ``grupo`` (``salido``/``actual``) y ``anio_salida`` (vía almacén)."""
    out = tramos.copy()
    out["grupo"] = np.where(out["baja"].notna(), "salido", "actual")
    out["anio_salida"] = out["baja"].dt.year.astype("Int64")
    return out.sort_values(["grupo", "anio_salida", "ticker", "alta"], na_position="last").reset_index(drop=True)


def muestra_salidos(tramos: pd.DataFrame, por_anio: int = POR_ANIO, semilla: int = SEMILLA) -> pd.DataFrame:
    """``por_anio`` tramos salidos por año de salida (todos si hay menos), con semilla fija."""
    s = tramos[tramos["baja"].notna()].copy()
    s["anio_salida"] = s["baja"].dt.year
    partes = [g.sample(min(por_anio, len(g)), random_state=semilla + int(a)) for a, g in s.groupby("anio_salida")]
    out = pd.concat(partes) if partes else s.iloc[:0]
    out["grupo"] = "salido"
    return out.sort_values(["anio_salida", "ticker"]).reset_index(drop=True)


def muestra_actuales(tramos: pd.DataFrame, n: int = N_ACTUALES, semilla: int = SEMILLA) -> pd.DataFrame:
    """``n`` miembros actuales (tramo abierto) al azar."""
    a = tramos[tramos["baja"].isna()]
    out = a.sample(min(n, len(a)), random_state=semilla).copy()
    out["anio_salida"] = pd.NA
    out["grupo"] = "actual"
    return out.sort_values("ticker").reset_index(drop=True)


def muestra_antigua(tramos: pd.DataFrame) -> pd.DataFrame:
    """La muestra de 122 tramos de la vía de respaldo (62 salidos y 60 actuales)."""
    return pd.concat([muestra_salidos(tramos), muestra_actuales(tramos)], ignore_index=True)


# --- Rentabilidad: almacén de precios (vía principal) ------------------------------------------
def _vacia() -> pd.DataFrame:
    return pd.DataFrame({"sesion": pd.Series(dtype="datetime64[ns]"), "rentabilidad": pd.Series(dtype=float)})


def rentabilidad_total_precios(precios: pd.DataFrame) -> pd.DataFrame:
    """Precios as-traded de UN ticker (contrato) → ``sesion, rentabilidad`` total diaria.

    Es el cambio de ``close_aj_total`` de ``timegate.ajustar_a_fecha`` a la última
    ``fecha_publicacion`` del ticker (lo que se sabe hoy): neutra ante splits y con los dividendos
    reinvertidos (``close_t / (close_{t-1} - D_t)`` en la fecha ex). Rentabilidades no finitas
    (cierres nulos o <= 0) → NaN. La primera sesión es NaN. Lleva también ``cierre`` (as-traded) y
    ``efectivo`` (``close * volume``), solo para las sospechas de identidad (no salen del módulo).
    """
    if precios is None or precios.empty:
        return _vacia()
    from ingesta.mercados_series.timegate import ajustar_a_fecha

    fecha = pd.Timestamp(precios["fecha_publicacion"].max()).normalize()
    aj = ajustar_a_fecha(precios, fecha, avisar=False)
    c = aj["close_aj_total"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.r_[np.nan, c[1:] / c[:-1] - 1.0] if len(c) else np.array([], dtype=float)
    r[~np.isfinite(r)] = np.nan
    return pd.DataFrame({"sesion": pd.DatetimeIndex(aj["sesion"]).normalize(), "rentabilidad": r,
                         "cierre": aj["close"].to_numpy(dtype=float),
                         "efectivo": (aj["close"] * aj["volume"]).to_numpy(dtype=float)})


def lector_almacen(ruta: Path | None = None) -> Callable[[str], pd.DataFrame | None]:
    """Lector ``símbolo → rentabilidad total diaria`` sobre ``precios/yahoo`` (o ``ruta``), con caché.

    Solo lectura del almacén (``almacen.leer_precios``). ``None`` si el símbolo no tiene fichero.
    """
    from ingesta.mercados_series import almacen

    @functools.lru_cache(maxsize=None)
    def leer(sym: str) -> pd.DataFrame | None:
        p = almacen.leer_precios(sym, "yahoo", ruta)
        return None if p.empty else rentabilidad_total_precios(p)

    return leer


# --- Rentabilidad: muestra antigua (vía de respaldo) -------------------------------------------
def rentabilidad_diaria(vista: pd.DataFrame) -> pd.DataFrame:
    """Vista de Yahoo (``Close`` y ``Dividends`` con splits ajustados) → ``sesion, rentabilidad``.

    Rentabilidad total: ``(Close_t + Dividends_t) / Close_{t-1} - 1``. Filas sin ``Close`` fuera.
    """
    if vista is None or vista.empty or "Close" not in vista:
        return _vacia()
    v = vista[vista["Close"].notna()].copy()
    idx = pd.DatetimeIndex(v.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    div = v["Dividends"].fillna(0.0) if "Dividends" in v else 0.0
    r = (v["Close"] + div) / v["Close"].shift(1) - 1.0
    return pd.DataFrame({"sesion": idx.normalize(), "rentabilidad": r.to_numpy(dtype=float)}).reset_index(drop=True)


def _vista_yf(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    ultimo: Exception | None = None
    for k in range(REINTENTOS):
        try:
            return yf.Ticker(ticker).history(period="max", auto_adjust=False, actions=True, timeout=TIMEOUT,
                                             raise_errors=False)
        except Exception as e:   # noqa: BLE001 - yfinance lanza tipos variados
            ultimo = e
            time.sleep(3.0 * (k + 1))
    raise RuntimeError(f"{ticker}: yfinance falló {REINTENTOS} veces: {ultimo}")


def leer_registro(raiz: Path | None = None) -> pd.DataFrame:
    ruta = ruta_registro(raiz)
    if not ruta.exists():
        return pd.DataFrame(columns=["ticker_yahoo", "estado", "filas", "error", "descargado", "segundos"])
    return pd.DataFrame([json.loads(x) for x in ruta.read_text(encoding="utf-8").splitlines() if x.strip()])


def descargar_muestra(tickers: list[str], *, raiz: Path | None = None, reanudar: bool = True,
                      ritmo: float = RITMO, obtener: Callable[[str], pd.DataFrame] | None = None) -> pd.DataFrame:
    """Descarga cada símbolo de Yahoo y guarda su rentabilidad diaria en ``supervivencia/<sym>.parquet``.

    Estados del registro: ``ok`` (con filas), ``sin_datos`` (Yahoo no devuelve nada: deslistado o
    desconocido) y ``error`` (excepción tras los reintentos). Con ``reanudar`` se saltan los símbolos
    ya registrados como ``ok`` o ``sin_datos``. ``obtener`` permite inyectar la vista (tests).
    """
    obtener = obtener or _vista_yf
    previo = leer_registro(raiz)
    hechos = set(previo.loc[previo["estado"].isin(["ok", "sin_datos"]), "ticker_yahoo"]) if reanudar and len(previo) else set()
    espera = 1.0 / ritmo if ritmo else 0.0
    d = dir_supervivencia(raiz)
    for sym in dict.fromkeys(tickers):
        if sym in hechos:
            continue
        t0 = time.monotonic()
        fila: dict[str, Any] = {"ticker_yahoo": sym, "descargado": datetime.now(timezone.utc).isoformat()}
        try:
            r = rentabilidad_diaria(obtener(sym))
            if len(r):
                r.to_parquet(d / f"{sym}.parquet", index=False)
            fila.update(estado="ok" if len(r) else "sin_datos", filas=int(len(r)), error=None)
        except Exception as e:   # noqa: BLE001 - se registra y se sigue
            fila.update(estado="error", filas=0, error=f"{type(e).__name__}: {e}"[:300])
        fila["segundos"] = round(time.monotonic() - t0, 2)
        with ruta_registro(raiz).open("a", encoding="utf-8") as f:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
        resto = espera - (time.monotonic() - t0)
        if resto > 0:
            time.sleep(resto)
    return leer_registro(raiz).drop_duplicates("ticker_yahoo", keep="last").reset_index(drop=True)


def leer_rentabilidad(sym: str, raiz: Path | None = None) -> pd.DataFrame | None:
    ruta = dir_supervivencia(raiz) / f"{sym}.parquet"
    return pd.read_parquet(ruta) if ruta.exists() else None


# --- Cobertura ----------------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _calendario_xnys():
    import exchange_calendars as xc

    return xc.get_calendar("XNYS", start="1990-01-02")


def sesiones_nyse(inicio: Any, fin: Any) -> pd.DatetimeIndex:
    """Sesiones de NYSE en ``[inicio, fin]`` (XNYS; días hábiles si no hay ``exchange_calendars``)."""
    a, b = pd.Timestamp(inicio).normalize(), pd.Timestamp(fin).normalize()
    if b < a:
        return pd.DatetimeIndex([])
    try:
        cal = _calendario_xnys()
        a2, b2 = max(a, cal.first_session), min(b, cal.last_session)
        return pd.DatetimeIndex(cal.sessions_in_range(a2, b2)).tz_localize(None) if b2 >= a2 else pd.DatetimeIndex([])
    except ImportError:
        return pd.bdate_range(a, b)


def cobertura_tramo(sesiones_precio: pd.Series | pd.DatetimeIndex | None, alta: Any, baja: Any,
                    hoy: Any = None, esperadas: pd.DatetimeIndex | None = None) -> dict[str, Any]:
    """Cobertura de un tramo ``[alta, baja)`` (``baja`` nula = hasta ``hoy``) con las sesiones con precio."""
    hoy = pd.Timestamp(hoy) if hoy is not None else pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    fin = (pd.Timestamp(baja) - pd.Timedelta(days=1)) if pd.notna(baja) else hoy
    if esperadas is None:
        esperadas = sesiones_nyse(alta, fin)
    n_esp = len(esperadas)
    if sesiones_precio is None or len(sesiones_precio) == 0:
        return {"sesiones_esperadas": n_esp, "sesiones_con_precio": 0, "fraccion": 0.0, "clase": "sin_datos",
                "primera_yahoo": None, "ultima_yahoo": None}
    sp = pd.DatetimeIndex(pd.to_datetime(sesiones_precio)).normalize()
    con = int(esperadas.isin(sp).sum())
    frac = con / n_esp if n_esp else 0.0
    if con == 0:
        clase = "fuera_de_tramo"
    elif frac >= UMBRAL_COMPLETA:
        clase = "completa"
    else:
        clase = "parcial"
    return {"sesiones_esperadas": n_esp, "sesiones_con_precio": con, "fraccion": round(frac, 4), "clase": clase,
            "primera_yahoo": str(sp.min().date()), "ultima_yahoo": str(sp.max().date())}


def sospechas_tramo(cob: dict[str, Any], alta: Any, baja: Any, nota: Any = None, *,
                    mediana_cierre: float | None = None, mediana_efectivo: float | None = None) -> list[str]:
    """Sospechas de identidad de un tramo (claves de ``SOSPECHAS``) a partir de su cobertura y su nota.

    ``mediana_cierre`` y ``mediana_efectivo``: medianas del cierre as-traded y de ``close * volume``
    DENTRO del tramo (si se conocen). Sacan el tramo de la estimación las de ``SOSPECHAS_EXCLUYEN``;
    ``termina_tarde`` solo en la variante estricta (casi siempre es una reentrada de la misma
    empresa, cuya serie empieza antes del alta); ``reutilizado_despues`` es informativa (el tramo
    no tiene precio).
    """
    if not cob.get("primera_yahoo"):
        return []
    primera, ultima = pd.Timestamp(cob["primera_yahoo"]), pd.Timestamp(cob["ultima_yahoo"])
    texto = nota if isinstance(nota, str) else ""
    nota_reuso = "reutilizado" in texto or "sufijo Q" in texto
    out = []
    if cob["clase"] == "fuera_de_tramo":
        if pd.notna(baja) and primera >= pd.Timestamp(baja):
            out.append("reutilizado_despues")
        return out
    if primera > pd.Timestamp(alta) + HOLGURA_INICIO:
        out.append("empieza_tarde")
    if pd.notna(baja) and nota_reuso and ultima > pd.Timestamp(baja) + HOLGURA_FIN:
        out.append("termina_tarde")
    if "sufijo Q" in texto:
        out.append("ticker_q")
    if (mediana_cierre is not None and mediana_cierre < CIERRE_MINIMO) or             (mediana_efectivo is not None and mediana_efectivo < EFECTIVO_MINIMO):
        out.append("impropio_sp500")
    return out


def _medianas_tramo(rent: pd.DataFrame | None, alta: Any, baja: Any) -> dict[str, float | None]:
    """Medianas de ``cierre`` y ``efectivo`` dentro de ``[alta, baja)`` (``None`` si no hay columnas)."""
    if rent is None or rent.empty or "cierre" not in rent:
        return {"mediana_cierre": None, "mediana_efectivo": None}
    dentro = rent["sesion"] >= pd.Timestamp(alta)
    if pd.notna(baja):
        dentro &= rent["sesion"] < pd.Timestamp(baja)
    q = rent.loc[dentro]
    if q.empty:
        return {"mediana_cierre": None, "mediana_efectivo": None}
    ef = q["efectivo"].median() if "efectivo" in q else np.nan
    return {"mediana_cierre": float(q["cierre"].median()),
            "mediana_efectivo": float(ef) if np.isfinite(ef) else None}


def tabla_cobertura(muestra: pd.DataFrame, raiz: Path | None = None, hoy: Any = None,
                    lector: Callable[[str], pd.DataFrame | None] | None = None) -> pd.DataFrame:
    """Una fila de METADATOS por tramo (sin valores de precio), con sospechas y ``en_estimacion``."""
    lector = lector or (lambda s: leer_rentabilidad(s, raiz))
    filas = []
    for r in muestra.itertuples(index=False):
        sym = ticker_yahoo(r.ticker)
        rent = lector(sym)
        cob = cobertura_tramo(None if rent is None else rent["sesion"], r.alta, r.baja, hoy=hoy)
        nota = getattr(r, "nota", None)
        sosp = sospechas_tramo(cob, r.alta, r.baja, nota, **_medianas_tramo(rent, r.alta, r.baja))
        notas = [str(nota)] if isinstance(nota, str) and nota else []
        notas += [SOSPECHAS[k] for k in sosp]
        if cob["clase"] in ("completa", "parcial") and pd.notna(r.baja) and not sosp:
            notas.append("identidad no verificable sin PERMNO/CIK: puede ser otra empresa con el mismo ticker")
        usable = cob["clase"] in ("completa", "parcial") and not set(sosp) & set(SOSPECHAS_EXCLUYEN)
        filas.append({"grupo": r.grupo, "ticker": r.ticker, "ticker_yahoo": sym, "alta": r.alta, "baja": r.baja,
                      "anio_salida": r.anio_salida, **cob, "sospecha": "+".join(sosp) or None,
                      "en_estimacion": bool(usable), "nota": "; ".join(notas) or None})
    return pd.DataFrame(filas, columns=list(COLUMNAS_COBERTURA))


def resumen_por_anio(cob: pd.DataFrame) -> pd.DataFrame:
    """Por año de salida: n, fracción media de sesiones con precio, reparto por clase y tramos usados."""
    s = cob[cob["grupo"] == "salido"]
    if s.empty:
        return pd.DataFrame()
    t = s.groupby("anio_salida").agg(n=("ticker", "size"), fraccion_media=("fraccion", "mean"),
                                     en_estimacion=("en_estimacion", "sum"))
    clases = s.pivot_table(index="anio_salida", columns="clase", values="ticker", aggfunc="size", fill_value=0)
    t = t.join(clases.reindex(columns=list(CLASES), fill_value=0))
    t["con_precio_completo"] = t["completa"] / t["n"]
    return t.reset_index()


def _resumen_grupo(g: pd.DataFrame) -> dict[str, Any]:
    util = g["fraccion"].where(g["en_estimacion"], 0.0)
    return {"n": int(len(g)), "con_fichero": int(g["primera_yahoo"].notna().sum()),
            "con_precio_en_tramo": int(g["clase"].isin(["completa", "parcial"]).sum()),
            "completa": int((g["clase"] == "completa").sum()),
            "sospechosos_excluidos": int((g["clase"].isin(["completa", "parcial"]) & ~g["en_estimacion"]).sum()),
            "en_estimacion": int(g["en_estimacion"].sum()),
            "fraccion_media": round(float(g["fraccion"].mean()), 3),
            "fraccion_media_util": round(float(util.mean()), 3)}


def resumen_por_decada(cob: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Salidos por década de salida: cobertura dentro del tramo y tramos usados en la estimación."""
    s = cob[cob["grupo"] == "salido"]
    if s.empty:
        return {}
    dec = pd.cut(s["anio_salida"].astype(int), DECADAS[0], labels=DECADAS[1])
    return {str(k): _resumen_grupo(g) for k, g in s.groupby(dec, observed=True)}


# --- Sesgo --------------------------------------------------------------------------------------
def rentabilidad_mensual(rent: pd.DataFrame, alta: Any = None, baja: Any = None) -> pd.Series:
    """Rentabilidad mensual compuesta (índice: fin de mes) de las sesiones en ``[alta, baja)``."""
    r = rent.dropna(subset=["rentabilidad"])
    if alta is not None and pd.notna(alta):
        r = r[r["sesion"] >= pd.Timestamp(alta)]
    if baja is not None and pd.notna(baja):
        r = r[r["sesion"] < pd.Timestamp(baja)]
    if r.empty:
        return pd.Series(dtype=float)
    s = r.set_index("sesion")["rentabilidad"]
    return (1.0 + s).groupby(s.index.to_period("M")).prod() - 1.0


def estimar_sesgo(tramos: pd.DataFrame, muestra: pd.DataFrame,
                  lector: Callable[[str], pd.DataFrame | None]) -> pd.DataFrame:
    """Rentabilidad equiponderada mensual PIT (con precio) frente a la de los miembros actuales.

    Mes ``m``: miembros PIT = tramos vivos el primer día del mes, en dos grupos (``salido``: tramo ya
    cerrado; ``actual``: tramo abierto) con ``n_salidos`` y ``n_actuales`` miembros. La media de cada
    grupo se estima con los tramos de ``muestra`` (la muestra antigua o todos los usables del
    almacén) vivos ese mes con rentabilidad; la cartera PIT pondera ambas por ``n``.
    Supervivientes = media de los actuales de ``muestra`` con rentabilidad ese mes, SIN mirar cuándo
    entraron (la composición de hoy aplicada hacia atrás). Si el grupo salido no tiene ningún dato
    ese mes, la PIT usa solo los actuales (``pit_solo_actuales=True``). ``cobertura_pit``: fracción
    de los miembros PIT del mes con rentabilidad en la estimación.
    """
    mens: dict[int, pd.Series] = {}
    completas: dict[int, pd.Series] = {}
    grupos: dict[int, str] = {}
    for i, r in enumerate(muestra.itertuples(index=False)):
        rent = lector(ticker_yahoo(r.ticker))
        if rent is None or rent.empty:
            continue
        s = rentabilidad_mensual(rent, r.alta, r.baja)
        if len(s):
            mens[i], grupos[i] = s, r.grupo
        if r.grupo == "actual":
            c = rentabilidad_mensual(rent)
            if len(c):
                completas[i] = c
    if not mens:
        return pd.DataFrame()
    m = pd.DataFrame(mens).replace([np.inf, -np.inf], np.nan)
    inicio = max(m.index.min(), tramos["alta"].min().to_period("M"))
    periodos = pd.period_range(inicio, m.index.max(), freq="M")
    m = m.reindex(periodos)
    sup = (pd.DataFrame(completas).replace([np.inf, -np.inf], np.nan).reindex(periodos) if completas
           else pd.DataFrame(index=periodos))
    m_sal = m[[i for i in m.columns if grupos[i] == "salido"]]
    m_act = m[[i for i in m.columns if grupos[i] == "actual"]]

    inicios = periodos.to_timestamp().to_numpy()[:, None]
    alta = tramos["alta"].to_numpy()[None, :]
    baja = tramos["baja"].to_numpy()[None, :]
    abierto = tramos["baja"].isna().to_numpy()[None, :]
    viva = (alta <= inicios) & (abierto | (baja > inicios))
    n_sal = (viva & ~abierto).sum(axis=1)
    n_act = (viva & abierto).sum(axis=1)

    media_sal, media_act = m_sal.mean(axis=1).to_numpy(), m_act.mean(axis=1).to_numpy()
    con_sal, con_act = m_sal.notna().sum(axis=1).to_numpy(), m_act.notna().sum(axis=1).to_numpy()
    ambos = np.isfinite(media_sal) & np.isfinite(media_act)
    total = n_sal + n_act
    with np.errstate(invalid="ignore", divide="ignore"):
        mezcla = np.where(total > 0, (n_sal * np.nan_to_num(media_sal) + n_act * np.nan_to_num(media_act)) / total,
                          np.nan)
        cobertura = np.where(total > 0, (con_sal + con_act) / total, np.nan)
    pit = np.where(ambos, mezcla, np.where(np.isfinite(media_act), media_act, media_sal))
    out = pd.DataFrame({
        "mes": periodos.to_timestamp(how="end").normalize(), "n_salidos_pit": n_sal, "n_actuales_pit": n_act,
        "muestra_salidos_con_precio": con_sal, "muestra_actuales_pit_con_precio": con_act,
        "muestra_supervivientes_con_precio": sup.notna().sum(axis=1).to_numpy(),
        "pit": pit, "supervivientes": sup.mean(axis=1).to_numpy() if sup.shape[1] else np.nan,
        "pit_solo_actuales": ~np.isfinite(media_sal), "cobertura_pit": cobertura,
    })
    out["diferencia"] = out["supervivientes"] - out["pit"]
    return out


def resumen_sesgo(sesgo: pd.DataFrame, limitacion: bool = True) -> dict[str, Any]:
    """Medias anualizadas (×12) de PIT, supervivientes y diferencia en los meses con ambos datos.

    ``sesgo_ic95_anual_pct``: intervalo al 95 % de la media de la diferencia mensual (normal, sin
    corrección por autocorrelación), anualizado ×12.
    """
    if sesgo is None or sesgo.empty:
        return {"meses": 0}
    s = sesgo.dropna(subset=["pit", "supervivientes"])
    s = s[~s["pit_solo_actuales"].astype(bool)]
    if s.empty:
        return {"meses": 0}
    dif = s["diferencia"]
    sd = float(dif.std(ddof=1)) if len(s) > 1 else 0.0
    ee = sd / np.sqrt(len(s)) if sd > 0 else 0.0
    out = {"meses": int(len(s)), "desde": str(s["mes"].min().date()), "hasta": str(s["mes"].max().date()),
           "pit_anual_pct": round(float(s["pit"].mean() * 1200), 2),
           "supervivientes_anual_pct": round(float(s["supervivientes"].mean() * 1200), 2),
           "sesgo_anual_pct": round(float(dif.mean() * 1200), 2),
           "sesgo_ic95_anual_pct": [round(float((dif.mean() - 1.96 * ee) * 1200), 2),
                                    round(float((dif.mean() + 1.96 * ee) * 1200), 2)],
           "sesgo_t": round(float(dif.mean() / ee), 2) if ee > 0 else None}
    if "cobertura_pit" in s:
        out["cobertura_pit_media"] = round(float(s["cobertura_pit"].mean()), 3)
    if limitacion:
        out["limitacion"] = NOTA_LIMITACION
    return out


# --- Medición completa (vía almacén) ------------------------------------------------------------
def medir_almacen(tramos: pd.DataFrame | None = None, *, ruta: Path | None = None, hoy: Any = None,
                  con_muestra_antigua: bool = True) -> dict[str, Any]:
    """Cobertura de TODOS los tramos y sesgo con el almacén de precios (más variantes de contraste).

    Devuelve ``{"cobertura", "sesgo", "variantes", "mediciones"}``. Variantes: ``sin_sospechas``
    (principal: ``en_estimacion``), ``estricta`` (excluye además ``termina_tarde``),
    ``con_sospechas`` (todos los tramos con precio en el tramo) y, si hay registro de la vía de
    respaldo y ``con_muestra_antigua``, ``muestra_122`` (la estimación anterior, sobre la muestra
    descargada aparte).
    """
    tramos = cargar_tramos() if tramos is None else tramos
    lector = lector_almacen(ruta)
    cob = tabla_cobertura(todos_los_tramos(tramos), hoy=hoy, lector=lector)
    con_precio = cob[cob["clase"].isin(["completa", "parcial"])]
    estricta = cob["en_estimacion"] & ~cob["sospecha"].fillna("").str.contains("termina_tarde")
    variantes = {"sin_sospechas": estimar_sesgo(tramos, cob[cob["en_estimacion"]], lector),
                 "estricta": estimar_sesgo(tramos, cob[estricta], lector),
                 "con_sospechas": estimar_sesgo(tramos, con_precio, lector)}
    reg = leer_registro() if con_muestra_antigua else pd.DataFrame()
    if len(reg):
        variantes["muestra_122"] = estimar_sesgo(tramos, muestra_antigua(tramos), leer_rentabilidad)
    med = mediciones(cob, variantes["sin_sospechas"], pd.DataFrame(), via="almacen", variantes=variantes)
    return {"cobertura": cob, "sesgo": variantes["sin_sospechas"], "variantes": variantes, "mediciones": med}


# --- Ficha y CLI --------------------------------------------------------------------------------
def guardar_tabla(cob: pd.DataFrame) -> Path:
    destino = resultados_dir(TEMA) / "tablas" / "supervivencia_cobertura.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    out = cob.copy()
    for c in ("alta", "baja"):
        out[c] = pd.to_datetime(out[c]).dt.date
    out.to_csv(destino, index=False, encoding="utf-8")
    return destino


def mediciones(cob: pd.DataFrame, sesgo: pd.DataFrame, reg: pd.DataFrame, *, via: str = "muestra",
               variantes: dict[str, pd.DataFrame] | None = None) -> dict[str, Any]:
    if via not in VIAS:
        raise ValueError(f"vía desconocida {via!r}; se espera una de {VIAS}")
    sal = cob[cob["grupo"] == "salido"]
    act = cob[cob["grupo"] == "actual"]
    por_anio = resumen_por_anio(cob)
    periodos = {}
    if len(sal):
        tramo = pd.cut(sal["anio_salida"].astype(int), [1995, 2000, 2005, 2010, 2015, 2020, 2026],
                       labels=["1996-2000", "2001-2005", "2006-2010", "2011-2015", "2016-2020", "2021-2026"])
        for k, g in sal.groupby(tramo, observed=True):
            periodos[str(k)] = {"n": int(len(g)), "completa": int((g["clase"] == "completa").sum()),
                                "fraccion_media": round(float(g["fraccion"].mean()), 3)}
    sosp = cob["sospecha"].dropna().str.split("+").explode() if "sospecha" in cob else pd.Series(dtype=str)
    out = {
        "via": via,
        "salidos_muestra": int(len(sal)), "actuales_muestra": int(len(act)),
        "salidos_por_clase": {k: int((sal["clase"] == k).sum()) for k in CLASES},
        "actuales_por_clase": {k: int((act["clase"] == k).sum()) for k in CLASES},
        "salidos_fraccion_media_sesiones": round(float(sal["fraccion"].mean()), 3) if len(sal) else None,
        "tramos_en_estimacion": {"salidos": int(sal["en_estimacion"].sum()), "actuales": int(act["en_estimacion"].sum())}
        if "en_estimacion" in cob else {},
        "sospechas": {k: int((sosp == k).sum()) for k in SOSPECHAS},
        "salidos_por_decada_de_salida": resumen_por_decada(cob) if "en_estimacion" in cob else {},
        "salidos_por_periodo_de_salida": periodos,
        "salidos_por_anio": por_anio[["anio_salida", "n", "completa", "fraccion_media"]]
        .assign(fraccion_media=lambda x: x["fraccion_media"].round(3), anio_salida=lambda x: x["anio_salida"].astype(int))
        .to_dict("records") if len(por_anio) else [],
        "sesgo": resumen_sesgo(sesgo) if len(sesgo) else {"meses": 0},
        "sesgo_variantes": {k: resumen_sesgo(v, limitacion=False) for k, v in (variantes or {}).items()},
    }
    if via == "muestra":
        out["simbolos_sin_descargar"] = sorted(set(cob["ticker_yahoo"]) - set(reg["ticker_yahoo"])) if len(reg) else []
        out["descargas"] = {"simbolos": int(len(reg)), "por_estado": reg["estado"].value_counts().to_dict() if len(reg) else {},
                            "segundos_medios": round(float(reg["segundos"].mean()), 2) if len(reg) else None}
    else:
        out["almacen"] = {"tramos_con_fichero": int(cob["primera_yahoo"].notna().sum()),
                          "salidos_con_fichero": int(sal["primera_yahoo"].notna().sum()),
                          "actuales_con_fichero": int(act["primera_yahoo"].notna().sum())}
    return out


def guardar_ficha_supervivencia(med: dict[str, Any], cob: pd.DataFrame) -> Path:
    from ingesta.comun.esquema import FichaFuente
    from ingesta.comun.fichas import guardar_ficha

    sal = cob[cob["grupo"] == "salido"]
    completa = med["salidos_por_clase"].get("completa", 0)
    almacen = med.get("via", "muestra") == "almacen"
    if almacen:
        volumen = (f"{len(cob)} tramos (todos: {len(sal)} salidos y {len(cob) - len(sal)} actuales); "
                   f"{med['almacen']['tramos_con_fichero']} con parquet as-traded en el almacén")
        formato = "parquet as-traded del almacén de precios (solo lectura) + csv de metadatos"
        limites = "sin red (lee el almacén del bloque 6); Yahoo no sirve la mayoría de deslistados"
        metodo = ("Módulo: ingesta.mercados_series.supervivencia, vía almacén (todos los tramos con precio; rentabilidad "
                  "total de close_aj_total de timegate.ajustar_a_fecha a la última fecha)")
        usados = med.get("tramos_en_estimacion", {})
        extra = [f"Tramos en la estimación: {usados.get('salidos', 0)} salidos y {usados.get('actuales', 0)} actuales; "
                 "los tramos con sospecha de otra empresa con el mismo ticker (empieza_tarde, impropio_sp500, ticker_q) "
                 "se excluyen de la principal; la variante estricta excluye además termina_tarde y con_sospechas los "
                 "incluye todos",
                 f"Sospechas heurísticas: precio que empieza > {HOLGURA_INICIO.days} días tras el alta; mediana del "
                 f"cierre < {CIERRE_MINIMO:g} USD o del efectivo diario < {EFECTIVO_MINIMO:,.0f} USD en el tramo; nota "
                 f"de varios tramos o sufijo Q y precio > {HOLGURA_FIN.days} días tras la baja",
                 "Vía de respaldo: muestra de 122 tramos descargada aparte (variante muestra_122)"]
    else:
        volumen = f"{len(cob)} tramos muestreados; {len(sal)} salidos en la muestra"
        formato = "parquet (rentabilidad diaria derivada) + csv de metadatos"
        limites = f"<= {RITMO} tickers/s, timeout {TIMEOUT} s; Yahoo no sirve la mayoría de deslistados"
        metodo = "Módulo: ingesta.mercados_series.supervivencia (muestra estratificada por año de salida, semilla fija)"
        extra = ["La muestra de salidos está estratificada por año de salida, no por mes: la media de los salidos de "
                 "cada mes es una estimación con pocos datos (ruido alto)"]
    return guardar_ficha(FichaFuente(
        fuente="supervivencia",
        nombre="Sesgo de supervivencia del S&P 500: cobertura de Yahoo para los miembros salidos",
        tema=TEMA, agentes=["equity", "portfolio"], rag="serie", url="https://finance.yahoo.com/",
        acceso="libreria", autenticacion="ninguna", coste="0 €",
        licencia_y_terminos=("Yahoo (yfinance): uso personal, sin redistribuir; composición del S&P 500 de "
                             "fja05680/sp500 (MIT). Solo metadatos en el repositorio."),
        cobertura_inicio=str(pd.to_datetime(sal["alta"]).min().date()) if len(sal) else None,
        cobertura_fin=str(pd.to_datetime(sal["baja"]).max().date()) if len(sal) else None,
        frecuencia="diaria (rentabilidad) → mensual (sesgo)",
        retraso_publicacion="no aplica (medición); los precios de Yahoo, cierre local + 30 min (ficha yahoo)",
        formato=formato,
        volumen_estimado=volumen,
        limites=limites,
        estado_sonda="ok" if len(sal) else "fallo",
        mediciones=med,
        notas=[
            metodo,
            f"Salidos con precio completo en su tramo: {completa} de {len(sal)}",
            NOTA_LIMITACION,
            *extra,
            "Un ticker con datos dentro del tramo puede ser otra empresa (sin PERMNO/CIK no se verifica)",
            "Intervalo del sesgo: 95 % normal sobre la diferencia mensual, sin corregir autocorrelación",
            *[f"Corrección '{k}': {v}" for k, v in OPCIONES_CORRECCION.items()],
            "Decisión: WRDS pendiente de MIAX; mientras tanto el sesgo se DECLARA y se MIDE con esta cota",
        ],
    ))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("accion", choices=["descargar", "medir", "todo"])
    p.add_argument("--via", choices=VIAS, default="almacen",
                   help="almacen: parquet as-traded de todos los tramos; muestra: 122 tramos descargados aparte")
    p.add_argument("--por-anio", type=int, default=POR_ANIO)
    p.add_argument("--actuales", type=int, default=N_ACTUALES)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    tramos = cargar_tramos()
    muestra = pd.concat([muestra_salidos(tramos, a.por_anio), muestra_actuales(tramos, a.actuales)], ignore_index=True)
    if a.accion in ("descargar", "todo") and a.via == "muestra":
        reg = descargar_muestra([ticker_yahoo(t) for t in muestra["ticker"]])
        print(reg["estado"].value_counts().to_dict())
    elif a.accion == "descargar":
        print("vía almacén: nada que descargar (los precios vienen del bloque 6)")
    if a.accion in ("medir", "todo"):
        if a.via == "almacen":
            res = medir_almacen(tramos)
            cob, med = res["cobertura"], res["mediciones"]
        else:
            cob = tabla_cobertura(muestra)
            sesgo = estimar_sesgo(tramos, muestra, leer_rentabilidad)
            med = mediciones(cob, sesgo, leer_registro().drop_duplicates("ticker_yahoo", keep="last"))
        print(guardar_tabla(cob))
        print(json.dumps({k: med[k] for k in ("salidos_por_clase", "actuales_por_clase", "sesgo", "sesgo_variantes")},
                         indent=1, ensure_ascii=False))
        print(guardar_ficha_supervivencia(med, cob))


if __name__ == "__main__":
    main()
