"""Sesgo de supervivencia del S&P 500 con precios gratuitos de Yahoo (bloque 7).

Pregunta: ¿qué parte de los miembros que SALIERON del S&P 500 tiene precio en Yahoo durante su
etapa en el índice, y cuánto se equivoca un backtest que usa la composición ACTUAL?

Entrada: ``<corpus>/mercados_series/constituyentes/sp500.parquet`` (tramos point-in-time de
``fuentes/constituyentes.py``; si no existe se construye con ``descargar_sp500``).

1. Muestra estratificada (``muestra_salidos``): ``por_anio`` tramos salidos por año de salida
   (1996-2026, ~60 en total) y ``n_actuales`` miembros actuales al azar (semilla fija).
2. Descarga (``descargar_muestra``): yfinance con el ritmo del proveedor Yahoo (<= 0,5 tickers/s,
   timeout 90 s, reintentos SOLO ante excepciones: un ticker sin datos no se reintenta, porque en
   un deslistado es determinista). Ticker del fichero → Yahoo cambiando ``.`` por ``-``. Se guarda en
   ``<corpus>/mercados_series/supervivencia/`` SOLO la rentabilidad total diaria derivada
   (``(Close + Dividends) / Close previo - 1`` sobre la vista de Yahoo con splits ajustados; no es
   una tabla de precios ni lleva ``Adj Close``) y un registro JSONL reanudable.
3. Cobertura (``cobertura_tramo``): sesiones con precio dentro de ``[alta, baja)`` frente a las
   sesiones de NYSE (``exchange_calendars`` XNYS; si falta, días hábiles). Clases: ``completa``
   (>= 95 %), ``parcial``, ``fuera_de_tramo`` (Yahoo tiene el ticker pero solo fuera del tramo:
   casi siempre ticker reutilizado por otra empresa) y ``sin_datos``.
4. Sesgo (``estimar_sesgo``): rentabilidad equiponderada mensual de la cartera POINT-IN-TIME con
   precio (mezcla de la muestra de salidos y la de actuales, ponderada por cuántos miembros de cada
   grupo había ese mes) frente a la de los miembros ACTUALES con precio (cartera de supervivientes:
   la composición de hoy aplicada hacia atrás). Limitación DECLARADA: los salidos sin precio (sobre
   todo quiebras y compras) no entran, y son los peores; el sesgo medido es una COTA INFERIOR.

Opciones de corrección (``OPCIONES_CORRECCION``): WRDS/CRSP (pendiente de la licencia de MIAX),
Alpha Vantage LISTING_STATUS (solo lista de deslistados, sin precios) y Tiingo (requiere cuenta del
usuario). CLI::

    python -m ingesta.mercados_series.supervivencia descargar   # reanudable (~5 min)
    python -m ingesta.mercados_series.supervivencia medir       # CSV de metadatos + ficha
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
CLASES = ("completa", "parcial", "fuera_de_tramo", "sin_datos")
COLUMNAS_COBERTURA = ("grupo", "ticker", "ticker_yahoo", "alta", "baja", "anio_salida", "sesiones_esperadas",
                      "sesiones_con_precio", "fraccion", "clase", "primera_yahoo", "ultima_yahoo", "nota")
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


# --- Muestra ------------------------------------------------------------------------------------
def ticker_yahoo(ticker: str) -> str:
    """Ticker del fichero del S&P 500 → símbolo de Yahoo (``BRK.B`` → ``BRK-B``)."""
    return str(ticker).strip().replace(".", "-")


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


# --- Descarga -----------------------------------------------------------------------------------
def rentabilidad_diaria(vista: pd.DataFrame) -> pd.DataFrame:
    """Vista de Yahoo (``Close`` y ``Dividends`` con splits ajustados) → ``sesion, rentabilidad``.

    Rentabilidad total: ``(Close_t + Dividends_t) / Close_{t-1} - 1``. Filas sin ``Close`` fuera.
    """
    if vista is None or vista.empty or "Close" not in vista:
        return pd.DataFrame({"sesion": pd.Series(dtype="datetime64[ns]"), "rentabilidad": pd.Series(dtype=float)})
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


def tabla_cobertura(muestra: pd.DataFrame, raiz: Path | None = None, hoy: Any = None,
                    lector: Callable[[str], pd.DataFrame | None] | None = None) -> pd.DataFrame:
    """Una fila de METADATOS por tramo de la muestra (sin valores de precio)."""
    lector = lector or (lambda s: leer_rentabilidad(s, raiz))
    filas = []
    for r in muestra.itertuples(index=False):
        sym = ticker_yahoo(r.ticker)
        rent = lector(sym)
        cob = cobertura_tramo(None if rent is None else rent["sesion"], r.alta, r.baja, hoy=hoy)
        notas = [str(r.nota)] if isinstance(r.nota, str) and r.nota else []
        if cob["clase"] == "fuera_de_tramo" and cob["primera_yahoo"] and pd.notna(r.baja) \
                and pd.Timestamp(cob["primera_yahoo"]) >= pd.Timestamp(r.baja):
            notas.append("Yahoo empieza después de la baja: ticker reutilizado por otra empresa (probable)")
        if cob["clase"] in ("completa", "parcial") and pd.notna(r.baja):
            notas.append("identidad no verificable sin PERMNO/CIK: puede ser otra empresa con el mismo ticker")
        filas.append({"grupo": r.grupo, "ticker": r.ticker, "ticker_yahoo": sym, "alta": r.alta, "baja": r.baja,
                      "anio_salida": r.anio_salida, **cob, "nota": "; ".join(notas) or None})
    return pd.DataFrame(filas, columns=list(COLUMNAS_COBERTURA))


def resumen_por_anio(cob: pd.DataFrame) -> pd.DataFrame:
    """Por año de salida: n, fracción media de sesiones con precio y reparto por clase."""
    s = cob[cob["grupo"] == "salido"]
    if s.empty:
        return pd.DataFrame()
    t = s.groupby("anio_salida").agg(n=("ticker", "size"), fraccion_media=("fraccion", "mean"))
    clases = s.pivot_table(index="anio_salida", columns="clase", values="ticker", aggfunc="size", fill_value=0)
    t = t.join(clases.reindex(columns=list(CLASES), fill_value=0))
    t["con_precio_completo"] = t["completa"] / t["n"]
    return t.reset_index()


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


def _miembros_mes(tramos: pd.DataFrame, inicio_mes: pd.Timestamp) -> pd.Series:
    viva = (tramos["alta"] <= inicio_mes) & (tramos["baja"].isna() | (tramos["baja"] > inicio_mes))
    return viva


def estimar_sesgo(tramos: pd.DataFrame, muestra: pd.DataFrame,
                  lector: Callable[[str], pd.DataFrame | None]) -> pd.DataFrame:
    """Rentabilidad equiponderada mensual PIT (con precio) frente a la de los miembros actuales.

    Mes ``m``: miembros PIT = tramos vivos el primer día del mes, en dos grupos (``salido``: tramo ya
    cerrado; ``actual``: tramo abierto) con ``n_salidos`` y ``n_actuales`` miembros. La media de cada
    grupo se estima con su muestra (tramos de la muestra vivos ese mes con rentabilidad); la cartera
    PIT pondera ambas por ``n``. Supervivientes = media de la muestra de actuales con rentabilidad ese
    mes, SIN mirar cuándo entraron (la composición de hoy aplicada hacia atrás). Si el grupo salido no
    tiene ningún dato ese mes, la PIT usa solo los actuales (``pit_solo_actuales=True``).
    """
    mens: dict[int, pd.Series] = {}
    completas: dict[int, pd.Series] = {}
    for i, r in enumerate(muestra.itertuples(index=False)):
        rent = lector(ticker_yahoo(r.ticker))
        if rent is None or rent.empty:
            continue
        mens[i] = rentabilidad_mensual(rent, r.alta, r.baja)
        if r.grupo == "actual":
            completas[i] = rentabilidad_mensual(rent)
    if not mens:
        return pd.DataFrame()
    mens = {i: s for i, s in mens.items() if len(s)}
    if not mens:
        return pd.DataFrame()
    todos = pd.concat(list(mens.values()))
    inicio = max(todos.index.min(), tramos["alta"].min().to_period("M"))
    periodos = pd.period_range(inicio, todos.index.max(), freq="M")
    grupo = muestra["grupo"].to_numpy()
    filas = []
    for p in periodos:
        im = p.to_timestamp()
        viva = _miembros_mes(tramos, im)
        n_sal = int((viva & tramos["baja"].notna()).sum())
        n_act = int((viva & tramos["baja"].isna()).sum())
        med: dict[str, list[float]] = {"salido": [], "actual": []}
        for i, s in mens.items():
            if p in s.index and np.isfinite(s[p]):
                med[grupo[i]].append(float(s[p]))
        sup = [float(s[p]) for s in completas.values() if p in s.index and np.isfinite(s[p])]
        m_sal = float(np.mean(med["salido"])) if med["salido"] else np.nan
        m_act = float(np.mean(med["actual"])) if med["actual"] else np.nan
        if np.isfinite(m_sal) and np.isfinite(m_act):
            pit = (n_sal * m_sal + n_act * m_act) / (n_sal + n_act) if (n_sal + n_act) else np.nan
        else:
            pit = m_act if np.isfinite(m_act) else m_sal
        filas.append({"mes": p.to_timestamp(how="end").normalize(), "n_salidos_pit": n_sal, "n_actuales_pit": n_act,
                      "muestra_salidos_con_precio": len(med["salido"]),
                      "muestra_actuales_pit_con_precio": len(med["actual"]),
                      "muestra_supervivientes_con_precio": len(sup),
                      "pit": pit, "supervivientes": float(np.mean(sup)) if sup else np.nan,
                      "pit_solo_actuales": not np.isfinite(m_sal)})
    out = pd.DataFrame(filas)
    out["diferencia"] = out["supervivientes"] - out["pit"]
    return out


def resumen_sesgo(sesgo: pd.DataFrame) -> dict[str, Any]:
    """Medias anualizadas (×12) de PIT, supervivientes y diferencia en los meses con ambos datos."""
    s = sesgo.dropna(subset=["pit", "supervivientes"])
    s = s[~s["pit_solo_actuales"]]
    if s.empty:
        return {"meses": 0}
    t = np.sqrt(len(s))
    dif = s["diferencia"]
    return {"meses": int(len(s)), "desde": str(s["mes"].min().date()), "hasta": str(s["mes"].max().date()),
            "pit_anual_pct": round(float(s["pit"].mean() * 1200), 2),
            "supervivientes_anual_pct": round(float(s["supervivientes"].mean() * 1200), 2),
            "sesgo_anual_pct": round(float(dif.mean() * 1200), 2),
            "sesgo_t": round(float(dif.mean() / dif.std(ddof=1) * t), 2) if dif.std(ddof=1) > 0 else None,
            "limitacion": NOTA_LIMITACION}


# --- Ficha y CLI --------------------------------------------------------------------------------
def guardar_tabla(cob: pd.DataFrame) -> Path:
    destino = resultados_dir(TEMA) / "tablas" / "supervivencia_cobertura.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    out = cob.copy()
    for c in ("alta", "baja"):
        out[c] = pd.to_datetime(out[c]).dt.date
    out.to_csv(destino, index=False, encoding="utf-8")
    return destino


def mediciones(cob: pd.DataFrame, sesgo: pd.DataFrame, reg: pd.DataFrame) -> dict[str, Any]:
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
    return {
        "salidos_muestra": int(len(sal)), "actuales_muestra": int(len(act)),
        "salidos_por_clase": {k: int((sal["clase"] == k).sum()) for k in CLASES},
        "actuales_por_clase": {k: int((act["clase"] == k).sum()) for k in CLASES},
        "salidos_fraccion_media_sesiones": round(float(sal["fraccion"].mean()), 3) if len(sal) else None,
        "salidos_por_periodo_de_salida": periodos,
        "salidos_por_anio": por_anio[["anio_salida", "n", "completa", "fraccion_media"]]
        .assign(fraccion_media=lambda x: x["fraccion_media"].round(3)).to_dict("records") if len(por_anio) else [],
        "sesgo": resumen_sesgo(sesgo) if len(sesgo) else {"meses": 0},
        "simbolos_sin_descargar": sorted(set(cob["ticker_yahoo"]) - set(reg["ticker_yahoo"])) if len(reg) else [],
        "descargas": {"simbolos": int(len(reg)), "por_estado": reg["estado"].value_counts().to_dict() if len(reg) else {},
                      "segundos_medios": round(float(reg["segundos"].mean()), 2) if len(reg) else None},
    }


def guardar_ficha_supervivencia(med: dict[str, Any], cob: pd.DataFrame) -> Path:
    from ingesta.comun.esquema import FichaFuente
    from ingesta.comun.fichas import guardar_ficha

    sal = cob[cob["grupo"] == "salido"]
    completa = med["salidos_por_clase"].get("completa", 0)
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
        formato="parquet (rentabilidad diaria derivada) + csv de metadatos",
        volumen_estimado=f"{len(cob)} tramos muestreados; 759 tramos salidos en total (1996-2026)",
        limites=f"<= {RITMO} tickers/s, timeout {TIMEOUT} s; Yahoo no sirve la mayoría de deslistados",
        estado_sonda="ok" if len(sal) else "fallo",
        mediciones=med,
        notas=[
            "Módulo: ingesta.mercados_series.supervivencia (muestra estratificada por año de salida, semilla fija)",
            f"Salidos con precio completo en su tramo: {completa} de {len(sal)}",
            NOTA_LIMITACION,
            "La muestra de salidos está estratificada por año de salida, no por mes: la media de los salidos de "
            "cada mes es una estimación con pocos datos (ruido alto)",
            "Un ticker con datos dentro del tramo puede ser otra empresa (sin PERMNO/CIK no se verifica)",
            *[f"Corrección '{k}': {v}" for k, v in OPCIONES_CORRECCION.items()],
            "Decisión: WRDS pendiente de MIAX; mientras tanto el sesgo se DECLARA y se MIDE con esta cota",
        ],
    ))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("accion", choices=["descargar", "medir", "todo"])
    p.add_argument("--por-anio", type=int, default=POR_ANIO)
    p.add_argument("--actuales", type=int, default=N_ACTUALES)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    tramos = cargar_tramos()
    muestra = pd.concat([muestra_salidos(tramos, a.por_anio), muestra_actuales(tramos, a.actuales)], ignore_index=True)
    if a.accion in ("descargar", "todo"):
        reg = descargar_muestra([ticker_yahoo(t) for t in muestra["ticker"]])
        print(reg["estado"].value_counts().to_dict())
    if a.accion in ("medir", "todo"):
        cob = tabla_cobertura(muestra)
        print(guardar_tabla(cob))
        sesgo = estimar_sesgo(tramos, muestra, leer_rentabilidad)
        med = mediciones(cob, sesgo, leer_registro().drop_duplicates("ticker_yahoo", keep="last"))
        print(json.dumps({k: med[k] for k in ("salidos_por_clase", "actuales_por_clase", "sesgo")}, indent=1,
                         ensure_ascii=False))
        print(guardar_ficha_supervivencia(med, cob))


if __name__ == "__main__":
    main()
