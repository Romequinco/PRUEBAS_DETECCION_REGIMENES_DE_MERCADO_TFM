"""Validez de los datos alternativos del Tema 4 (B2): calidad, look-ahead, señal frente a crisis y semáforo.

Todo se mide desde lo que un agente sabría cada día (TimeGate): las series del corpus se ponen en una
rejilla diaria HÁBIL por ``fecha_publicacion`` (as-of: el último dato publicado hasta ese día, vigente
solo hasta ``VIGENCIA_FACTOR`` veces su espaciado típico), nunca por ``fecha_referencia``.

Niveles:

0. Carga (``cargar_series``, ``a_rejilla``) y referencias: VIX (serie ``cboe_vix``), S&P 500 (yfinance,
   cacheado en ``corpus('crudo', 'validez')``) y su drawdown; crisis = ventanas pico->suelo de la Fase 1.
1. Calidad por serie (``calidad_serie``): cobertura, huecos, atípicos (|z robusto MAD| > 6 sobre las
   diferencias), rupturas de media candidatas (diferencia de medias por ventanas), repeticiones, vintages.
2. Look-ahead (``lookahead_*``): una fila por comprobación y fuente con evidencia medida; ``comprobar_z_causal``
   rehace el pipeline entero en varios cortes solo con lo publicado hasta cada corte.
3. Señal frente a crisis: z causal EXPANDING (``z_expanding``, mínimo 252 obs), estudio de eventos alrededor del
   pico (intervalo t, solo con ``CRISIS_BUENA`` crisis), AUC Mann-Whitney con la etiqueta adelantada (como el EDA
   de la Fase 1) en los mismos días que el VIX y con IC por bloques de crisis, lead-lag frente al VIX y al
   drawdown en dos rejillas (por referencia: contenido; por publicación: utilizable), estabilidad por subperiodos.
4. Calibración de los mercados de predicción (Brier, Murphy, fiabilidad en 10 tramos; mercados indecisos y,
   solo descriptivo, por volumen final).
5. Rentabilidad anormal ilustrativa (CAR ajustado por mercado) de compras del Congreso y de insiders,
   DESDE LA PUBLICACIÓN (y desde la operación, para ver la diferencia), con IC por semanas y control de
   supervivencia (eventos sin precio).
6. Semáforo por fuente/serie: ``apta_backtest`` | ``solo_vivo`` | ``contexto`` | ``descartar``; la señal de una
   serie solo cuenta en el sentido fijado a priori (``SIGNO_ESPERADO``).

``ejecutar()`` lo calcula todo y guarda ``results/ingesta/noticias_alt/validez_*.csv``. Las funciones de
cálculo son puras (DataFrames/Series de entrada, sin red) y se prueban con datos sintéticos.
"""
from __future__ import annotations

import io
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ingesta.comun.rutas import resultados_dir
from ingesta.noticias_alt import base

MODULO = "validez"
MIN_OBS_Z = 252                     # mínimo de observaciones (días hábiles de rejilla) del z expanding
VIGENCIA_FACTOR = 3                 # un dato publicado vale hasta 3 espaciados típicos (mínimo 5 días hábiles)
VIGENCIA_MIN = 5
UMBRAL_ATIPICO = 6.0                # |z robusto (MAD)| de las diferencias
UMBRAL_RUPTURA = 4.0                # diferencia de medias / desviación conjunta entre ventanas
ADELANTOS = (0, 20, 60)
VENTANA_EVENTOS = (-60, 60)
LAGS = range(-60, 61)
N_BOOT = 2000
NIVEL_IC = 0.90
HORIZONTES_CAR = (5, 20, 60)
URL_VIX = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
INICIO_REJILLA = "1985-01-01"
SERIE_VIX = "vix_cierre"
CORTE_ESTABILIDAD = 2015
# Semáforo: umbrales de los criterios (0 = malo, 1 = regular, 2 = bueno).
CRISIS_BUENA, CRISIS_REGULAR = 5, 2
AUC_BUENA, AUC_REGULAR = 0.15, 0.07          # |AUC - 0,5|
HUECOS_BUENO, HUECOS_REGULAR = 0.10, 0.30
RACHA_MALA = 30                     # racha de >= 30 valores idénticos seguidos: hueco codificado o fuente parada -> calidad 0
SERIES_REFERENCIA = {SERIE_VIX, "vix_maximo"}
N_BOOT_AUC = 500                    # remuestras del IC del AUC por bloques de crisis
VENTANA_SIGNO = (0, 20)             # días hábiles tras el pico para contrastar el signo con el estudio de eventos
PROB_INDECISA = (0.05, 0.95)        # mercados de predicción: fuera de este rango el mercado ya está «decidido»
MIN_INDECISOS = 30                  # mínimo de mercados indecisos para puntuar la calibración
MAX_SIN_PRECIO = 0.20               # CAR: con más eventos sin precio, sesgo de supervivencia -> sin señal
# Signo esperado en las crisis (+1 sube, −1 baja) por familia de serie, fijado por la definición de cada índice y
# no por los resultados. El semáforo solo cuenta la separación en ese sentido; sin signo a priori -> contexto.
SIGNO_ESPERADO: tuple[tuple[str, int, str], ...] = (
    (r"^vix_", +1, "volatilidad implícita: sube con el estrés"),
    (r"^skew$", -1, "SKEW: la prima de cola relativa cae cuando la volatilidad ATM ya se ha disparado"),
    (r"^pc_.*_(ratio|puts)$", +1, "put/call y volumen de puts: cobertura en las caídas"),
    (r"^(epu|gepu)", +1, "incertidumbre de política económica"),
    (r"^gpr", +1, "riesgo geopolítico (índices y subíndices por país, actos y amenazas)"),
    (r"^wiki_pv:", +1, "atención a temas de crisis (visitas)"),
    (r"^gdelt_tono", -1, "tono de las noticias: más negativo con el estrés"),
    (r"^ews_", +1, "actividad de jets privados (botón de cataclismo)"),
    (r"^polymarket_recesion", +1, "probabilidad de recesión"),
    (r"^polymarket_fed_recorte", +1, "probabilidad de recortes de tipos"),
    (r"^polymarket_fed_sin_recortes", -1, "probabilidad de que no haya recortes"),
    (r"^(nanc|kruz)_", -1, "precio de una cartera de acciones"),
)


def signo_esperado(serie: str) -> int | None:
    """Signo a priori de ``serie`` en las crisis según ``SIGNO_ESPERADO`` (None: sin signo claro)."""
    import re

    for patron, signo, _ in SIGNO_ESPERADO:
        if re.search(patron, str(serie)):
            return signo
    return None


# =========================================================================== 0. carga y rejilla

def rejilla(inicio: str | pd.Timestamp = INICIO_REJILLA, fin: str | pd.Timestamp | None = None) -> pd.DatetimeIndex:
    """Días hábiles (lunes-viernes) de ``inicio`` a ``fin`` (hoy por defecto)."""
    return pd.bdate_range(pd.Timestamp(inicio), pd.Timestamp(fin) if fin is not None else pd.Timestamp.today().normalize())


def _ultimas_por_publicacion(larga: pd.DataFrame) -> pd.DataFrame:
    """Una fila por ``fecha_publicacion``: la de ``fecha_referencia`` más reciente conocida hasta entonces.

    Una publicación tardía de una referencia antigua (PTR fuera de plazo, revisión) no pisa un dato más nuevo.
    """
    d = larga.dropna(subset=["valor", "fecha_publicacion", "fecha_referencia"]).copy()
    d["fecha_publicacion"] = pd.to_datetime(d["fecha_publicacion"]).dt.normalize()
    d["fecha_referencia"] = pd.to_datetime(d["fecha_referencia"]).dt.normalize()
    orden = ["fecha_publicacion", "fecha_referencia"] + (["disponible_utc"] if "disponible_utc" in d else [])
    d = d.sort_values(orden, kind="mergesort")
    d = d[d["fecha_referencia"] >= d["fecha_referencia"].cummax()]
    return d.groupby("fecha_publicacion", as_index=False).last()


def espaciado_habil(fechas: pd.Series | pd.DatetimeIndex) -> float:
    """Mediana de días hábiles entre publicaciones consecutivas (1 si no se puede medir)."""
    f = pd.DatetimeIndex(pd.to_datetime(pd.Series(fechas)).dropna().unique()).sort_values()
    if len(f) < 2:
        return 1.0
    d = np.busday_count(f[:-1].values.astype("datetime64[D]"), f[1:].values.astype("datetime64[D]"))
    d = d[d > 0]
    return float(np.median(d)) if d.size else 1.0


def es_conteo_sin_ceros(larga: pd.DataFrame) -> bool:
    """Conteo de eventos que solo guarda los días con eventos: enteros >= 1 y más del 5 % de días sin fila.

    En esas series un día sin fila es un 0, no un hueco (p. ej. PTR presentados por día).
    """
    v = pd.to_numeric(larga["valor"], errors="coerce").dropna()
    if len(v) < 20 or (v < 1).any() or not np.allclose(v, np.round(v)):
        return False
    f = pd.DatetimeIndex(pd.to_datetime(larga["fecha_referencia"]).dt.normalize().unique())
    nombre, paso = frecuencia(f)
    if not nombre.startswith("diaria"):
        return False
    return len(f) < 0.95 * esperadas(f, nombre, paso)


def a_rejilla_conteo(larga: pd.DataFrame, dias: pd.DatetimeIndex) -> pd.Series:
    """Conteo diario sobre ``dias``: suma de lo publicado desde el día hábil anterior (0 si nada).

    Vacío antes de la primera publicación y después de la última (más ``VIGENCIA_MIN`` días hábiles).
    """
    d = larga.dropna(subset=["valor", "fecha_publicacion"])
    pub = pd.to_datetime(d["fecha_publicacion"]).dt.normalize()
    pos = dias.searchsorted(pub)
    ok = pos < len(dias)
    suma = pd.Series(d["valor"].to_numpy(dtype=float)[ok]).groupby(pos[ok]).sum()
    v = np.zeros(len(dias))
    v[suma.index.to_numpy()] = suma.to_numpy()
    s = pd.Series(v, index=dias, name=str(larga["serie"].iloc[0]) if len(larga) else None)
    if suma.empty:
        return s * np.nan
    s.iloc[: int(suma.index.min())] = np.nan
    s.iloc[min(len(dias), int(suma.index.max()) + VIGENCIA_MIN + 1):] = np.nan
    return s


def a_rejilla(larga: pd.DataFrame, dias: pd.DatetimeIndex, vigencia: int | None = None) -> pd.Series:
    """Serie as-of por ``fecha_publicacion`` sobre ``dias``: lo que se sabía cada día.

    ``vigencia``: máximo de días hábiles que se arrastra el último dato (por defecto 3 espaciados típicos,
    mínimo 5); pasado ese plazo la serie queda vacía (fuente congelada o interrumpida). Los conteos que
    solo guardan días con eventos (``es_conteo_sin_ceros``) se suman por día y los días sin fila valen 0.
    """
    if len(larga) and es_conteo_sin_ceros(larga):
        return a_rejilla_conteo(larga, dias)
    u = _ultimas_por_publicacion(larga)
    if u.empty:
        return pd.Series(np.nan, index=dias, dtype=float)
    if vigencia is None:
        vigencia = max(VIGENCIA_MIN, int(math.ceil(VIGENCIA_FACTOR * espaciado_habil(u["fecha_publicacion"]))))
    izq = pd.DataFrame({"dia": dias})
    der = u[["fecha_publicacion", "valor"]].rename(columns={"fecha_publicacion": "pub"})
    m = pd.merge_asof(izq, der.sort_values("pub"), left_on="dia", right_on="pub", direction="backward")
    edad = np.full(len(m), np.iinfo(np.int64).max)
    ok = m["pub"].notna().to_numpy()
    edad[ok] = np.busday_count(m.loc[ok, "pub"].values.astype("datetime64[D]"),
                               m.loc[ok, "dia"].values.astype("datetime64[D]"))
    v = m["valor"].to_numpy(dtype=float)
    v[edad > vigencia] = np.nan
    return pd.Series(v, index=dias, name=str(larga["serie"].iloc[0]) if len(larga) else None)


def a_rejilla_ref(larga: pd.DataFrame, dias: pd.DatetimeIndex, vigencia: int | None = None) -> pd.Series:
    """Como ``a_rejilla`` pero por ``fecha_referencia`` (sin retraso de publicación): mide el contenido.

    Solo para comparar series entre sí (lead-lag); nunca para el backtest, que va por publicación.
    """
    d = larga.copy()
    if len(d):
        d["fecha_publicacion"] = d["fecha_referencia"]
    return a_rejilla(d, dias, vigencia)


def transformacion(valores: pd.Series) -> str:
    """'log' si la serie es positiva y muy asimétrica (asimetría > 2, conteos y visitas); si no, 'nivel'."""
    v = pd.Series(valores, dtype=float).dropna()
    if len(v) < 10 or (v <= 0).any():
        return "nivel"
    return "log" if float(v.skew()) > 2 else "nivel"


def aplicar_transformacion(s: pd.Series, tipo: str) -> pd.Series:
    return np.log(s) if tipo == "log" else s


def leer_corpus_series(carpeta: Path | None = None) -> pd.DataFrame:
    """Todas las series largas del corpus (sin las del modo rápido), con la columna ``fichero``."""
    carpeta = carpeta or base.corpus("series")
    partes = []
    for f in sorted(carpeta.glob("*.parquet")):
        if f.stem.endswith(base.SUFIJO_RAPIDO):
            continue
        d = pd.read_parquet(f)
        if d.empty:
            continue
        partes.append(d.assign(fichero=f.stem))
    if not partes:
        return pd.DataFrame(columns=base.COLUMNAS_SERIE + ["fichero"])
    return pd.concat(partes, ignore_index=True)


def cargar_series(largas: pd.DataFrame | None = None, dias: pd.DatetimeIndex | None = None, *,
                  por: str = "publicacion") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rejilla as-of con una columna por serie, metadatos por serie).

    ``por="publicacion"`` (TimeGate, lo que se sabía cada día) o ``"referencia"`` (contenido, sin retraso).
    """
    largas = leer_corpus_series() if largas is None else largas
    if dias is None:
        ini = max(pd.Timestamp(INICIO_REJILLA), pd.to_datetime(largas["fecha_publicacion"]).min()) \
            if len(largas) else pd.Timestamp(INICIO_REJILLA)
        dias = rejilla(ini)
    a_dias = a_rejilla_ref if por == "referencia" else a_rejilla
    cols, meta = {}, []
    for nombre, g in largas.groupby("serie", sort=True):
        cols[nombre] = a_dias(g, dias)
        meta.append(dict(serie=nombre, fichero=g["fichero"].iloc[0] if "fichero" in g else nombre,
                         fuente=g["fuente"].iloc[0], unidad=g["unidad"].iloc[0],
                         metodo_publicacion=str(g["metodo_publicacion"].iloc[0]),
                         espaciado_habil=espaciado_habil(_ultimas_por_publicacion(g)["fecha_publicacion"])))
    return pd.DataFrame(cols, index=dias), pd.DataFrame(meta)


def vix_referencia(largas: pd.DataFrame | None = None) -> pd.DataFrame:
    """VIX de cierre en tabla larga: la serie ``cboe_vix`` del corpus o, si falta, el CSV de CBOE (D+1)."""
    if largas is not None and (largas["serie"] == SERIE_VIX).any():
        return largas[largas["serie"] == SERIE_VIX]
    v = base.leer_serie("cboe_vix")
    if not v.empty:
        return v[v["serie"] == SERIE_VIX]
    from ingesta.comun.http import get

    ruta = base.corpus("crudo", MODULO) / "VIX_History.csv"
    if not ruta.exists():
        r = get(URL_VIX, timeout=60)
        r.raise_for_status()
        base.escribir_atomico(ruta, r.content)
    d = pd.read_csv(ruta)
    f = pd.to_datetime(d["DATE"])
    return pd.DataFrame(dict(serie=SERIE_VIX, fecha_referencia=f, fecha_publicacion=f + pd.offsets.BDay(1),
                             valor=d["CLOSE"].astype(float), unidad="puntos", fuente="cboe_vix"))


def precios_yf(tickers: Iterable[str], inicio: str = "1985-01-01", *, refrescar_dias: float = 3.0) -> dict[str, pd.Series]:
    """Cierres ajustados de yfinance cacheados en ``corpus('crudo','validez','precios')`` (un CSV por ticker).

    Un ticker sin datos deja un CSV vacío (no se vuelve a pedir hasta ``refrescar_dias``) y no se devuelve.
    """
    carpeta = base.corpus("crudo", MODULO, "precios")
    out: dict[str, pd.Series] = {}
    pedir = []
    for t in sorted({str(t).strip().upper() for t in tickers if t and str(t).strip()}):
        ruta = carpeta / f"{t.replace('^', '_')}.csv"
        if ruta.exists() and (time.time() - ruta.stat().st_mtime) / 86400 <= refrescar_dias:
            if ruta.stat().st_size > 20:
                s = pd.read_csv(ruta, index_col=0, parse_dates=True).iloc[:, 0].dropna()
                if not s.empty:
                    out[t] = s
            continue
        pedir.append(t)
    if pedir:
        import yfinance as yf  # import aquí: solo si hay que bajar precios

        for i in range(0, len(pedir), 40):
            lote = pedir[i:i + 40]
            try:
                d = yf.download(lote, start=inicio, auto_adjust=True, progress=False, threads=True)
            except Exception:  # noqa: BLE001
                d = pd.DataFrame()
            cierre = d["Close"] if isinstance(d, pd.DataFrame) and "Close" in d else pd.DataFrame()
            if isinstance(cierre, pd.Series):
                cierre = cierre.to_frame(lote[0])
            for t in lote:
                s = cierre[t].dropna() if t in cierre else pd.Series(dtype=float)
                ruta = carpeta / f"{t.replace('^', '_')}.csv"
                s.rename("close").to_csv(ruta)
                if not s.empty:
                    s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) else pd.to_datetime(s.index)
                    out[t] = s
    return out


def sp500_y_drawdown(dias: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """S&P 500 (^GSPC) de cierre y drawdown (cierre / máximo previo − 1) en ``dias`` (as-of, hábiles)."""
    s = precios_yf(["^GSPC"]).get("^GSPC")
    if s is None or s.empty:
        return pd.DataFrame(columns=["sp500", "drawdown"])
    s = s.sort_index()
    dd = s / s.cummax() - 1
    d = pd.DataFrame(dict(sp500=s, drawdown=dd))
    if dias is not None:
        d = d.reindex(dias, method="ffill", limit=5)
    return d


# =========================================================================== crisis, z y AUC

def etiqueta_crisis(dias: pd.DatetimeIndex, crisis: dict[str, tuple[str, str]]) -> pd.Series:
    """1 en los días hábiles dentro de alguna ventana pico->suelo (incluidos), 0 fuera."""
    y = pd.Series(0, index=dias, dtype=int)
    for pico, suelo in crisis.values():
        y.loc[(dias >= pd.Timestamp(pico)) & (dias <= pd.Timestamp(suelo))] = 1
    return y


def z_expanding(s: pd.Series, minimo: int = MIN_OBS_Z) -> pd.Series:
    """z causal: (x_t − media_{≤t}) / desv_{≤t} con ventana expandible; NaN con menos de ``minimo`` obs.

    Solo usa el pasado y el presente: cambiar el futuro no cambia ningún valor anterior.
    """
    s = pd.Series(s, dtype=float)
    m = s.expanding(min_periods=minimo).mean()
    d = s.expanding(min_periods=minimo).std()
    z = (s - m) / d.where(d > 0)
    return z.where(s.notna())


def auc_mann_whitney(puntuacion: Iterable[float], etiqueta: Iterable[int]) -> float:
    """AUC = P(puntuación de un positivo > la de un negativo), empates a la mitad (rangos medios)."""
    x = np.asarray(list(puntuacion) if not isinstance(puntuacion, (np.ndarray, pd.Series)) else puntuacion, dtype=float)
    y = np.asarray(list(etiqueta) if not isinstance(etiqueta, (np.ndarray, pd.Series)) else etiqueta, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def adelantar(y: pd.Series, h: int) -> pd.Series:
    """Etiqueta adelantada: en t vale la etiqueta de t+h (¿la serie anticipa la crisis h días?)."""
    return y.shift(-h) if h else y


def crisis_cubiertas(z: pd.Series, crisis: dict[str, tuple[str, str]]) -> list[str]:
    """Crisis con al menos un día de z válido dentro de su ventana pico->suelo."""
    out = []
    for nombre, (pico, suelo) in crisis.items():
        tramo = z.loc[pd.Timestamp(pico):pd.Timestamp(suelo)]
        if tramo.notna().any():
            out.append(nombre)
    return out


def bloques_crisis(dias: pd.DatetimeIndex, crisis: dict[str, tuple[str, str]]) -> np.ndarray:
    """Bloque de cada día: tramo (suelo de la crisis anterior, suelo de esta]; la calma final va al último.

    Cada bloque tiene una crisis completa y la calma que la precede: remuestrear bloques es remuestrear crisis.
    """
    suelos = sorted(pd.Timestamp(s) for _, s in crisis.values())
    b = np.searchsorted(np.array(suelos, dtype="datetime64[ns]"), dias.values, side="left")
    return np.minimum(b, len(suelos) - 1) if suelos else np.zeros(len(dias), dtype=int)


def auc_ic_bloques(x: pd.Series, y: pd.Series, crisis: dict[str, tuple[str, str]], n_boot: int = N_BOOT_AUC,
                   nivel: float = NIVEL_IC, semilla: int = 0) -> tuple[float, float]:
    """IC percentil del AUC remuestreando BLOQUES de crisis completos (``bloques_crisis``), no días.

    El n efectivo del AUC es el número de crisis, no el de días: un IC por días sería mucho más estrecho.
    NaN si hay menos de 2 bloques con días de crisis.
    """
    ok = x.notna() & y.notna()
    if ok.sum() == 0:
        return float("nan"), float("nan")
    xv, yv = x[ok].to_numpy(dtype=float), y[ok].to_numpy(dtype=float)
    blq = bloques_crisis(x.index[ok.to_numpy()], crisis)
    ids = np.unique(blq)
    partes = [(xv[blq == i], yv[blq == i]) for i in ids]
    if sum(1 for _, yy in partes if yy.sum() > 0) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(semilla)
    aucs = []
    for _ in range(n_boot):
        el = rng.integers(0, len(partes), len(partes))
        xx = np.concatenate([partes[i][0] for i in el])
        yy = np.concatenate([partes[i][1] for i in el])
        a = auc_mann_whitney(xx, yy)
        if not np.isnan(a):
            aucs.append(a)
    if len(aucs) < n_boot // 2:
        return float("nan"), float("nan")
    a = (1 - nivel) / 2
    return float(np.quantile(aucs, a)), float(np.quantile(aucs, 1 - a))


def auc_crisis(z: pd.Series, crisis: dict[str, tuple[str, str]], adelantos: Iterable[int] = ADELANTOS,
               z_vix: pd.Series | None = None, n_boot: int = N_BOOT_AUC) -> pd.DataFrame:
    """AUC del nivel z causal en crisis frente a no crisis, con la etiqueta adelantada h días.

    ``auc`` usa todos los días de la serie (descriptivo). Si se da ``z_vix``, ``auc_mismo_periodo`` y
    ``auc_vix_mismo_periodo`` usan solo los días con las dos series (la comparación con el VIX va siempre en los
    mismos días). ``auc_lo``/``auc_hi``: IC por bloques de crisis del AUC principal (el de mismo periodo si hay
    VIX). AUC < 0,5 = inversa.
    """
    y = etiqueta_crisis(z.index, crisis)
    cub = crisis_cubiertas(z, crisis)
    filas = []
    for h in adelantos:
        yh = adelantar(y, h)
        ok = z.notna() & yh.notna()
        auc = auc_mann_whitney(z[ok].to_numpy(), yh[ok].to_numpy())
        fila = dict(adelanto=h, auc=auc, direccion=_direccion(auc), n_dias=int(ok.sum()),
                    n_dias_crisis=int(yh[ok].sum()), n_crisis=len(cub))
        principal = ok
        if z_vix is not None:
            okv = ok & z_vix.notna()
            fila["auc_mismo_periodo"] = auc_mann_whitney(z[okv].to_numpy(), yh[okv].to_numpy())
            fila["n_dias_mismo_periodo"] = int(okv.sum())
            fila["auc_vix_mismo_periodo"] = auc_mann_whitney(z_vix[okv].to_numpy(), yh[okv].to_numpy())
            fila["n_dias_vix"] = int(okv.sum())
            fila["n_crisis_mismo_periodo"] = len(crisis_cubiertas(z.where(okv), crisis))
            principal = okv
        if n_boot:
            fila["auc_lo"], fila["auc_hi"] = auc_ic_bloques(z.where(principal), yh.where(principal), crisis,
                                                            n_boot=n_boot, semilla=h)
        filas.append(fila)
    return pd.DataFrame(filas)


def _direccion(auc: float) -> str:
    if auc is None or np.isnan(auc):
        return "sin_datos"
    return "directa" if auc > 0.5 else ("inversa" if auc < 0.5 else "nula")


def ic_t_media(x: Iterable[float], nivel: float = NIVEL_IC, minimo: int = CRISIS_BUENA) -> tuple[float, float, float]:
    """(media, lo, hi) con intervalo t de Student (n − 1 g.l.); lo = hi = NaN con menos de ``minimo`` valores.

    Para medias entre pocas crisis: el bootstrap percentil con 2-5 valores apenas tiene medias distintas y
    subcubre; con menos de ``minimo`` crisis no se da intervalo.
    """
    from scipy.stats import t as t_student

    v = np.asarray([a for a in x if a is not None and not np.isnan(a)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(v.mean())
    if v.size < max(2, minimo):
        return m, float("nan"), float("nan")
    semi = float(t_student.ppf(1 - (1 - nivel) / 2, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size))
    return m, m - semi, m + semi


def bootstrap_media_cluster(x: Iterable[float], grupos: Iterable[Any], n_boot: int = N_BOOT, nivel: float = NIVEL_IC,
                            semilla: int = 0) -> tuple[float, float, float]:
    """(media, lo, hi) con bootstrap por conglomerados: se remuestrean GRUPOS completos (p. ej. semanas).

    Eventos de la misma semana comparten el movimiento del mercado: un bootstrap iid daría un IC demasiado
    estrecho. Con menos de 2 grupos, lo = hi = NaN.
    """
    d = pd.DataFrame(dict(v=list(x), g=list(grupos))).dropna(subset=["v"])
    if d.empty:
        return float("nan"), float("nan"), float("nan")
    m = float(d["v"].mean())
    por = d.groupby("g")["v"].agg(["sum", "count"])
    if len(por) < 2:
        return m, float("nan"), float("nan")
    s_, n_ = por["sum"].to_numpy(), por["count"].to_numpy()
    rng = np.random.default_rng(semilla)
    el = rng.integers(0, len(por), size=(n_boot, len(por)))
    medias = s_[el].sum(axis=1) / n_[el].sum(axis=1)
    a = (1 - nivel) / 2
    return m, float(np.quantile(medias, a)), float(np.quantile(medias, 1 - a))


def bootstrap_media(x: Iterable[float], n_boot: int = N_BOOT, nivel: float = NIVEL_IC, semilla: int = 0
                    ) -> tuple[float, float, float]:
    """(media, lo, hi) con IC percentil bootstrap; lo = hi = NaN con menos de 2 valores."""
    v = np.asarray([a for a in x if a is not None and not np.isnan(a)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size < 2:
        return float(v.mean()), float("nan"), float("nan")
    rng = np.random.default_rng(semilla)
    medias = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    a = (1 - nivel) / 2
    return float(v.mean()), float(np.quantile(medias, a)), float(np.quantile(medias, 1 - a))


def estudio_eventos(z: pd.Series, crisis: dict[str, tuple[str, str]], ventana: tuple[int, int] = VENTANA_EVENTOS,
                    n_boot: int = N_BOOT) -> pd.DataFrame:
    """Trayectoria media del z causal alrededor del PICO de cada crisis cubierta (días hábiles de rejilla).

    Una crisis entra si su pico cae en la rejilla y hay z válido en el día del pico. Intervalo t entre crisis
    (``ic_t_media``) solo con al menos ``CRISIS_BUENA`` crisis; con menos, media sin intervalo. ``n_boot`` se
    conserva por compatibilidad (no se usa).
    """
    idx = z.index
    tray = {}
    for nombre, (pico, _) in crisis.items():
        p = idx.searchsorted(pd.Timestamp(pico))
        if p >= len(idx) or abs((idx[p] - pd.Timestamp(pico)).days) > 4 or np.isnan(z.iloc[p]):
            continue
        fila = {}
        for k in range(ventana[0], ventana[1] + 1):
            j = p + k
            fila[k] = float(z.iloc[j]) if 0 <= j < len(idx) else np.nan
        tray[nombre] = fila
    filas = []
    for k in range(ventana[0], ventana[1] + 1):
        vals = [t[k] for t in tray.values() if not np.isnan(t[k])]
        m, lo, hi = ic_t_media(vals)
        filas.append(dict(dia_rel=k, media=m, lo=lo, hi=hi, n_crisis=len(vals)))
    return pd.DataFrame(filas)


def lead_lag(x: pd.Series, ref: pd.Series, lags: Iterable[int] = LAGS, min_n: int = 60) -> pd.DataFrame:
    """corr(x_t, ref_{t+lag}): lag > 0 = la serie va por delante de la referencia."""
    filas = []
    for lag in lags:
        r = ref.shift(-lag)
        ok = x.notna() & r.notna()
        n = int(ok.sum())
        c = float(np.corrcoef(x[ok], r[ok])[0, 1]) if n >= min_n and x[ok].std() > 0 and r[ok].std() > 0 else np.nan
        filas.append(dict(lag=int(lag), corr=c, n=n))
    return pd.DataFrame(filas)


def n_efectivo(x: pd.Series, r: pd.Series, metrica: str) -> float:
    """n efectivo de una correlación entre ``x`` y ``r`` (días con las dos).

    ``cambio_5d``: n/5 (ventanas de 5 días solapadas). ``nivel_z`` (niveles persistentes): aproximación de
    Bartlett para dos AR(1), n·(1 − ρx·ρr)/(1 + ρx·ρr), con ρ la autocorrelación de orden 1.
    """
    ok = x.notna() & r.notna()
    n = int(ok.sum())
    if n < 3:
        return float(n)
    if metrica == "cambio_5d":
        return n / 5
    rx, rr = x[ok].autocorr(1), r[ok].autocorr(1)
    if not np.isfinite(rx) or not np.isfinite(rr):
        return float(n)
    p = max(min(rx * rr, 0.999), -0.999)
    return n * (1 - p) / (1 + p)


def resumen_lead_lag(t: pd.DataFrame, n_eff: float | None = None) -> dict[str, Any]:
    """Lag de máxima |corr|, su valor y la corr contemporánea.

    ``lag_max`` es None («indeterminado») si |corr_max| − |corr_lag0| < 2/√n_eff: la correlación es plana entre
    lags y el máximo no se distingue del lag 0. ``lag_max_bruto`` guarda el máximo sin ese filtro. Sin
    ``n_eff`` se usa la n del lag 0.
    """
    v = t.dropna(subset=["corr"])
    if v.empty:
        return dict(lag_max=None, lag_max_bruto=None, corr_max=None, corr_lag0=None, n_eff=None, margen=None)
    i = v["corr"].abs().idxmax()
    c0 = v.loc[v["lag"] == 0, "corr"]
    corr_max, corr0 = float(v.loc[i, "corr"]), (float(c0.iloc[0]) if len(c0) else None)
    if n_eff is None:
        n0 = v.loc[v["lag"] == 0, "n"] if "n" in v else pd.Series(dtype=float)
        n_eff = float(n0.iloc[0]) if len(n0) else float("nan")
    margen = 2 / np.sqrt(n_eff) if n_eff and n_eff > 0 else float("inf")
    lag_b = int(v.loc[i, "lag"])
    indeterminado = corr0 is None or abs(corr_max) - abs(corr0) < margen
    lag = 0 if lag_b == 0 else (None if indeterminado else lag_b)
    return dict(lag_max=lag, lag_max_bruto=lag_b, corr_max=corr_max, corr_lag0=corr0,
                n_eff=round(float(n_eff), 1), margen=round(float(margen), 4))


def periodos_estabilidad(dias: pd.DatetimeIndex, corte: int = CORTE_ESTABILIDAD) -> dict[str, np.ndarray]:
    """Máscaras de subperiodo: antes/desde ``corte`` y por década."""
    anio = dias.year
    out = {f"antes_{corte}": np.asarray(anio < corte), f"desde_{corte}": np.asarray(anio >= corte)}
    for dec in range((anio.min() // 10) * 10, anio.max() + 1, 10):
        out[f"decada_{dec}s"] = np.asarray((anio >= dec) & (anio < dec + 10))
    return out


def estabilidad(z: pd.Series, crisis: dict[str, tuple[str, str]], z_vix: pd.Series | None = None) -> pd.DataFrame:
    """AUC (adelanto 0) por subperiodo; solo los subperiodos con días de crisis y de no crisis.

    ``auc`` usa todos los días de la serie en el subperiodo; con ``z_vix``, ``auc_mismo_periodo`` y ``auc_vix``
    usan solo los días con las dos series (comparables entre sí).
    """
    y = etiqueta_crisis(z.index, crisis)
    filas = []
    for nombre, m in periodos_estabilidad(z.index).items():
        ok = m & z.notna().to_numpy()
        if ok.sum() == 0 or y[ok].nunique() < 2:
            continue
        zz = z.where(ok)
        fila = dict(periodo=nombre, auc=auc_mann_whitney(z[ok].to_numpy(), y[ok].to_numpy()),
                    n_dias=int(ok.sum()), n_dias_crisis=int(y[ok].sum()), n_crisis=len(crisis_cubiertas(zz, crisis)))
        if z_vix is not None:
            okv = ok & z_vix.notna().to_numpy()
            fila["auc_mismo_periodo"] = auc_mann_whitney(z[okv].to_numpy(), y[okv].to_numpy()) if okv.any() else np.nan
            fila["auc_vix"] = auc_mann_whitney(z_vix[okv].to_numpy(), y[okv].to_numpy()) if okv.any() else np.nan
            fila["n_dias_mismo_periodo"] = int(okv.sum())
        filas.append(fila)
    return pd.DataFrame(filas)


# =========================================================================== 1. calidad

def frecuencia(fechas: pd.DatetimeIndex) -> tuple[str, float]:
    """Frecuencia nominal por la mediana de días naturales entre referencias: (nombre, días)."""
    f = pd.DatetimeIndex(fechas).unique().sort_values()
    if len(f) < 2:
        return "unica", float("nan")
    d = float(np.median(np.diff(f.values).astype("timedelta64[D]").astype(float)))
    if d <= 1.5:
        return ("diaria" if (f.dayofweek >= 5).mean() > 0.05 else "diaria_habil"), d
    if d <= 8:
        return "semanal", d
    if d <= 32:
        return "mensual", d
    if d <= 95:
        return "trimestral", d
    return "irregular", d


def esperadas(fechas: pd.DatetimeIndex, freq: str, dias: float) -> int:
    """Observaciones esperadas entre la primera y la última referencia según la frecuencia."""
    a, b = fechas.min(), fechas.max()
    if freq == "diaria":
        return (b - a).days + 1
    if freq == "diaria_habil":
        return len(pd.bdate_range(a, b))
    if freq == "semanal":
        return (b - a).days // 7 + 1
    if freq == "mensual":
        return (b.year - a.year) * 12 + b.month - a.month + 1
    if freq == "trimestral":
        return (b.year - a.year) * 4 + (b.quarter - a.quarter) + 1
    return int((b - a).days / dias) + 1 if dias and not np.isnan(dias) else len(fechas)


def atipicos_mad(v: pd.Series, umbral: float = UMBRAL_ATIPICO) -> pd.Series:
    """Máscara de atípicos: |diferencia − mediana| / (1,4826·MAD) > umbral (diagnóstico, no feature)."""
    d = pd.Series(v, dtype=float).diff()
    med = d.median()
    mad = (d - med).abs().median() * 1.4826
    if not mad or np.isnan(mad):
        mad = (d - med).abs().mean() * 1.2533
    if not mad or np.isnan(mad):
        return pd.Series(False, index=d.index)
    return ((d - med).abs() / mad > umbral).fillna(False)


def rupturas_media(v: pd.Series, ventana: int | None = None, umbral: float = UMBRAL_RUPTURA, max_n: int = 3
                   ) -> list[tuple[pd.Timestamp, float]]:
    """Saltos de nivel candidatos: |media(después) − media(antes)| / desv. conjunta > ``umbral``.

    Ventanas de ``ventana`` observaciones a cada lado (por defecto n/8, entre 20 y 250); devuelve hasta
    ``max_n`` máximos locales separados al menos una ventana: (fecha, estadístico).
    """
    s = pd.Series(v, dtype=float).dropna()
    n = len(s)
    w = ventana or int(min(250, max(20, n // 8)))
    if n < 2 * w + 1:
        return []
    antes_m = s.rolling(w).mean()
    antes_v = s.rolling(w).var()
    desp = s[::-1].rolling(w).mean()[::-1].shift(-1)
    desp_v = s[::-1].rolling(w).var()[::-1].shift(-1)
    den = np.sqrt((antes_v + desp_v) / 2)
    est = ((desp - antes_m).abs() / den.where(den > 0)).dropna()
    cand = est[est > umbral].sort_values(ascending=False)
    out: list[tuple[pd.Timestamp, float]] = []
    pos = {t: i for i, t in enumerate(s.index)}
    for t, e in cand.items():
        if all(abs(pos[t] - pos[o]) >= w for o, _ in out):
            out.append((t, float(e)))
        if len(out) >= max_n:
            break
    return out


def racha_maxima_constante(v: pd.Series) -> int:
    """Longitud de la racha más larga de valores idénticos consecutivos."""
    s = pd.Series(v).dropna().to_numpy()
    if s.size == 0:
        return 0
    cambio = np.r_[True, s[1:] != s[:-1]]
    grupos = np.cumsum(cambio)
    return int(np.bincount(grupos).max())


def valores_nativos(larga: pd.DataFrame) -> pd.Series:
    """Observaciones nativas: una por ``fecha_referencia`` (la última publicada), indexadas por esa fecha."""
    d = larga.dropna(subset=["fecha_referencia"]).copy()
    d["fecha_referencia"] = pd.to_datetime(d["fecha_referencia"]).dt.normalize()
    orden = ["fecha_referencia"] + (["disponible_utc"] if "disponible_utc" in d else [])
    d = d.sort_values(orden, kind="mergesort").groupby("fecha_referencia", as_index=False).last()
    return pd.Series(d["valor"].to_numpy(dtype=float), index=pd.DatetimeIndex(d["fecha_referencia"]))


def calidad_serie(larga: pd.DataFrame) -> dict[str, Any]:
    """Calidad de una serie en sus observaciones nativas (una por ``fecha_referencia``, la última publicada)."""
    v = valores_nativos(larga)
    nombre = str(larga["serie"].iloc[0]) if len(larga) else ""
    if v.dropna().empty:
        return dict(serie=nombre, n=0)
    vv = v.dropna()
    freq, paso = frecuencia(vv.index)
    esp = esperadas(vv.index, freq, paso)
    tr = transformacion(vv)
    vt = aplicar_transformacion(vv, tr)
    at = atipicos_mad(vt)
    rup = rupturas_media(vt)
    iguales = float((vv.diff() == 0).mean()) if len(vv) > 1 else 0.0
    metodo = larga["metodo_publicacion"].astype(str) if "metodo_publicacion" in larga else pd.Series(dtype=str)
    vint = larga["vintage"].dropna().nunique() if "vintage" in larga else 0
    lag = (pd.to_datetime(larga["fecha_publicacion"]) - pd.to_datetime(larga["fecha_referencia"])).dt.days
    return dict(
        serie=nombre, fuente=str(larga["fuente"].iloc[0]), inicio=vv.index.min().date().isoformat(),
        fin=vv.index.max().date().isoformat(), n=int(len(vv)), frecuencia=freq, n_esperadas=int(esp),
        pct_huecos=round(100 * max(0.0, 1 - len(vv) / esp), 2) if esp else None,
        conteo_sin_ceros=es_conteo_sin_ceros(larga),
        n_nulos=int(v.isna().sum()), transformacion=tr,
        n_atipicos=int(at.sum()), pct_atipicos=round(100 * float(at.mean()), 3),
        fechas_atipicos=";".join(t.date().isoformat() for t in at[at].index[:5]),
        n_rupturas=len(rup), fechas_rupturas=";".join(f"{t.date().isoformat()}({e:.1f})" for t, e in rup),
        pct_repetidos=round(100 * iguales, 2), racha_max_constante=racha_maxima_constante(vv),
        n_vintages=int(vint), revisable=bool((metodo == base.REVISABLE).any()),
        lag_pub_mediana_dias=float(lag.median()) if len(lag) else None,
        lag_pub_max_dias=float(lag.max()) if len(lag) else None,
        n_pub_antes_ref=int((lag < 0).sum()),
    )


# =========================================================================== 2. look-ahead

def _fila_la(comprobacion: str, fuente: str, resultado: str, evidencia: str, valor: Any = None) -> dict[str, Any]:
    return dict(comprobacion=comprobacion, fuente=fuente, resultado=resultado, evidencia=evidencia, valor=valor)


def lookahead_series(largas: pd.DataFrame) -> list[dict[str, Any]]:
    """Por serie: fecha_publicacion >= fecha_referencia, lag medido y vintage revisable."""
    filas = []
    for nombre, g in largas.groupby("serie"):
        lag = (pd.to_datetime(g["fecha_publicacion"]) - pd.to_datetime(g["fecha_referencia"])).dt.days
        neg = int((lag < 0).sum())
        filas.append(_fila_la("publicacion_no_anterior_a_referencia", nombre, "fallo" if neg else "ok",
                              f"{neg} de {len(g)} filas con publicación < referencia; lag mediano {lag.median():.0f} d, "
                              f"p90 {lag.quantile(.9):.0f} d; campo: {str(g['metodo_publicacion'].iloc[0])[:90]}",
                              neg))
        rev = (g["metodo_publicacion"].astype(str) == base.REVISABLE).mean()
        if rev > 0:
            filas.append(_fila_la("valor_revisado_vintage_actual", nombre, "riesgo",
                                  f"{100 * rev:.0f} % de filas con el vintage de hoy (valor revisado después de su "
                                  f"publicación; fecha correcta, valor no point-in-time)", round(float(rev), 3)))
    return filas


def lookahead_timegate(tg: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Tabla ``timegate`` de la sonda: por fuente, campo de fecha usado y lags negativos."""
    if tg is None or tg.empty:
        return [_fila_la("campo_fecha_timegate", "todas", "riesgo", "timegate.csv aún no disponible: campo de fecha "
                         "tomado de metodo_publicacion del corpus")]
    filas = []
    for fuente, g in tg.groupby("fuente"):
        lag = pd.to_numeric(g["lag_h"], errors="coerce")
        neg = int((lag < 0).sum())
        campos = "; ".join(sorted({str(c)[:60] for c in g["campo_fecha"].dropna()}))[:200]
        filas.append(_fila_la("campo_fecha_timegate", fuente, "fallo" if neg else "ok",
                              f"campo: {campos}; {len(g)} ejemplos, lag mediano {lag.median():.1f} h, {neg} negativos",
                              neg))
    return filas


def lookahead_documentos(carpeta: Path | None = None) -> list[dict[str, Any]]:
    """Documentos del corpus: fecha_publicacion >= fecha_referencia (sin leer texto)."""
    carpeta = carpeta or base.corpus("documentos")
    filas = []
    for f in sorted(carpeta.glob("*.jsonl")):
        if f.stem.endswith(base.SUFIJO_RAPIDO):
            continue
        n = neg = sin = sin_pub = 0
        lags = []
        with f.open(encoding="utf-8") as fh:
            for linea in fh:
                if not linea.strip():
                    continue
                d = json.loads(linea)
                n += 1
                fp, fr = d.get("fecha_publicacion"), d.get("fecha_referencia")
                sin_pub += not fp
                if not fp or not fr:
                    sin += 1
                    continue
                lg = (pd.Timestamp(fp).tz_localize(None) if pd.Timestamp(fp).tzinfo else pd.Timestamp(fp)) - \
                     (pd.Timestamp(fr).tz_localize(None) if pd.Timestamp(fr).tzinfo else pd.Timestamp(fr))
                lags.append(lg.total_seconds() / 86400)
                neg += lg.total_seconds() < 0
        med = float(np.median(lags)) if lags else float("nan")
        res = "fallo" if neg else ("riesgo" if sin_pub else "ok")
        filas.append(_fila_la("documentos_publicacion_no_anterior", f.stem, res,
                              f"{n} documentos: {sin_pub} sin fecha_publicacion, {sin - sin_pub} sin fecha_referencia "
                              f"(se fechan solo por publicación), {neg} con publicación < referencia; lag mediano "
                              f"{med:.2f} d", neg))
    return filas


def lookahead_ews(zpub: pd.DataFrame, zcau: pd.DataFrame, tabla_z: pd.DataFrame | None = None,
                  umbral: float = 2.0) -> list[dict[str, Any]]:
    """z del EWS publicado (revisado en cada instantánea) frente al causal recalculado."""
    if tabla_z is not None and {"z_publicado", "z_causal_piso"} <= set(tabla_z.columns):
        a = tabla_z["z_publicado"].astype(float)
        b = tabla_z["z_causal_piso"].astype(float)
    else:
        if zpub.empty or zcau.empty:
            return [_fila_la("ews_z_publicado_vs_causal", "ews", "riesgo", "sin series del EWS")]
        x = zpub.set_index("disponible_utc")["valor"].astype(float)
        y = zcau.set_index("disponible_utc")["valor"].astype(float)
        j = pd.concat([x.rename("p"), y.rename("c")], axis=1, join="inner")
        a, b = j["p"], j["c"]
    ok = a.notna() & b.notna()
    a, b = a[ok], b[ok]
    if ok.sum() < 10:
        return [_fila_la("ews_z_publicado_vs_causal", "ews", "riesgo", "menos de 10 pares publicado/causal")]
    c = float(np.corrcoef(a, b)[0, 1])
    al_p, al_c = int((a >= umbral).sum()), int((b >= umbral).sum())
    zpub_pub = zpub["fecha_publicacion"].nunique() if "fecha_publicacion" in zpub else None
    return [
        _fila_la("ews_z_publicado_vs_causal", "ews", "riesgo",
                 f"{len(a)} semihoras: corr {c:.2f}, |dif| media {float((a - b).abs().mean()):.2f}; alarmas z>={umbral}: "
                 f"publicado {al_p}, causal {al_c}. El z publicado se recalcula con el archivo entero (mira el futuro)",
                 round(c, 3)),
        _fila_la("ews_z_publicado_fecha_vintage", "ews", "ok" if zpub_pub == 1 else "riesgo",
                 f"z publicado guardado con fecha_publicacion = día de la instantánea ({zpub_pub} fechas distintas): "
                 "no entra en el pasado del backtest", zpub_pub),
    ]


def lookahead_gdelt(ev: pd.DataFrame | None, tg: pd.DataFrame | None) -> list[dict[str, Any]]:
    """GDELT Events: SQLDATE (hecho) frente a DATEADDED (captura)."""
    if ev is not None and not ev.empty and "pct_sqldate_anterior" in ev:
        pct = pd.to_numeric(ev["pct_sqldate_anterior"], errors="coerce")
        p90 = pd.to_numeric(ev.get("lag_dias_p90"), errors="coerce")
        mx = pd.to_numeric(ev.get("lag_dias_max"), errors="coerce")
        return [_fila_la("gdelt_sqldate_vs_dateadded", "gdelt_eventos", "ok",
                         f"{len(ev)} ficheros/días: {pct.mean():.1f} % de eventos con SQLDATE anterior a DATEADDED "
                         f"(p90 {p90.max():.0f} d, máx {mx.max():.0f} d). Indexar por SQLDATE adelantaría información; "
                         "el corpus usa DATEADDED", round(float(pct.mean()), 2))]
    if tg is not None and not tg.empty:
        g = tg[tg["fuente"].astype(str).str.contains("gdelt_ev|gdelt_eventos", regex=True)]
        if not g.empty:
            return [_fila_la("gdelt_sqldate_vs_dateadded", "gdelt_eventos", "ok",
                             f"{len(g)} ejemplos timegate, lag mediano {pd.to_numeric(g['lag_h']).median():.0f} h, "
                             "campo DATEADDED", None)]
    return [_fila_la("gdelt_sqldate_vs_dateadded", "gdelt_eventos", "riesgo",
                     "tabla gdelt_eventos_lag no disponible: regla documentada (DATEADDED, no SQLDATE)")]


def lookahead_wikipedia(ce: pd.DataFrame | None, docs: list[dict] | None = None) -> list[dict[str, Any]]:
    """Wikipedia Current Events: revisión al cierre (23:59 UTC) -> publicación D+1."""
    lags = []
    for d in docs or []:
        if d.get("fecha_publicacion") and d.get("fecha_referencia"):
            lags.append((pd.Timestamp(d["fecha_publicacion"]) - pd.Timestamp(d["fecha_referencia"])).days)
    if lags:
        s = pd.Series(lags)
        res = "fallo" if (s < 1).any() else "ok"
        return [_fila_la("wiki_revision_al_cierre", "wiki_current_events", res,
                         f"{len(s)} hechos: lag publicación-referencia mín {s.min()} d, mediana {s.median():.0f} d "
                         "(revisión de las 23:59 UTC posterior al cierre de NY -> D+1)", int((s < 1).sum()))]
    n = 0 if ce is None else len(ce)
    return [_fila_la("wiki_revision_al_cierre", "wiki_current_events", "riesgo",
                     f"sin documentos medidos ({n} filas de tabla): regla D+1 documentada")]


def lookahead_congreso(ops: pd.DataFrame | None) -> list[dict[str, Any]]:
    """PTR de la Cámara: publicación por FilingDate (+k hábiles) y lag operación->publicación."""
    if ops is None or ops.empty:
        return [_fila_la("ptr_filingdate", "congreso_house_ptr", "riesgo", "congreso_operaciones no disponible")]
    fo = pd.to_datetime(ops["fecha_operacion"], errors="coerce")
    fpres = pd.to_datetime(ops["fecha_presentacion"], errors="coerce")
    fpub = pd.to_datetime(ops["fecha_publicacion"], errors="coerce")
    lag = (fpub - fo).dt.days.dropna()
    neg = int(((fpub < fpres) | (fpub < fo)).sum())
    fuera = float((lag > 45).mean()) if len(lag) else float("nan")
    return [
        _fila_la("ptr_filingdate", "congreso_house_ptr", "fallo" if neg else "ok",
                 f"{len(ops)} operaciones: {neg} con publicación anterior a presentación u operación", neg),
        _fila_la("ptr_lag_operacion_publicacion", "congreso_house_ptr", "riesgo" if len(lag) else "ok",
                 f"lag operación->publicación: mediana {lag.median():.0f} d, p90 {lag.quantile(.9):.0f} d, "
                 f"{100 * fuera:.1f} % > 45 d (plazo STOCK Act). Indexar por fecha de operación sería look-ahead",
                 float(lag.median()) if len(lag) else None),
    ]


def lookahead_insiders(ops: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Form 4: publicación por aceptación EDGAR frente a la fecha de la transacción."""
    if ops is None or ops.empty:
        return [_fila_la("form4_aceptacion", "sec_form4_insiders", "riesgo", "insiders_operaciones no disponible")]
    ft = pd.to_datetime(ops["fecha_transaccion"], errors="coerce")
    fp = pd.to_datetime(ops["fecha_publicacion"], errors="coerce")
    lag = (fp - ft).dt.days.dropna()
    neg = int((lag < 0).sum())
    return [_fila_la("form4_aceptacion", "sec_form4_insiders", "fallo" if neg else "ok",
                     f"{len(lag)} transacciones: lag transacción->publicación mediana {lag.median():.0f} d, "
                     f"p90 {lag.quantile(.9):.0f} d, {100 * float((lag > 2).mean()):.1f} % > 2 d; {neg} negativos",
                     neg)]


def lookahead_fijas() -> list[dict[str, Any]]:
    """Riesgos documentados de fuentes no medibles aquí."""
    return [
        _fila_la("trends_renormalizacion", "google_trends", "riesgo",
                 "Google Trends reescala 0-100 por consulta y ventana: el valor de un día cambia según el periodo "
                 "pedido; solo es point-in-time guardando cada consulta el día que se hace (no probada)"),
        _fila_la("polymarket_corte_16_30", "polymarket", "ok",
                 "precio diario = último punto CLOB <= 16:30 NY del día (metodo_publicacion de las series)"),
    ]


def z_pipeline(largas: pd.DataFrame, dias: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict[str, str]]:
    """z causal de cada serie con el pipeline completo (rejilla as-of, transformación, z expanding).

    Todo se decide con ``largas`` tal como llega: la transformación (asimetría de la serie), la vigencia del
    dato (espaciado de las publicaciones) y si es un conteo sin ceros. Devuelve (z, transformación por serie).
    """
    rej, _ = cargar_series(largas, dias)
    trans = {n: transformacion(valores_nativos(g).dropna()) for n, g in largas.groupby("serie")}
    z = pd.DataFrame({c: z_expanding(aplicar_transformacion(rej[c], trans.get(c, "nivel"))) for c in rej.columns},
                     index=dias)
    return z, trans


def comprobar_z_causal(largas: pd.DataFrame, zs: pd.DataFrame, cortes: Iterable[Any] | None = None,
                       tol: float = 1e-9) -> list[dict[str, Any]]:
    """El pipeline entero rehecho en cada corte t solo con lo publicado hasta t coincide con ``zs`` hasta t.

    En cada corte se toma ``largas`` con ``fecha_publicacion <= t`` y se repite ``z_pipeline`` en la rejilla
    hasta t: si la transformación (log o nivel), la vigencia o el tratamiento de conteos cambian al ver solo el
    pasado, el z difiere y la serie queda en ``riesgo`` (una fila por serie afectada, con su evidencia) además de
    la fila resumen ``todas``. Cortes por defecto: 30 %, 60 % y 90 % de la rejilla.
    """
    dias = zs.index
    cortes = [dias[int(q * (len(dias) - 1))] for q in (0.3, 0.6, 0.9)] if cortes is None else list(cortes)
    pub = pd.to_datetime(largas["fecha_publicacion"]).dt.normalize()
    tr_full = {n: transformacion(valores_nativos(g).dropna()) for n, g in largas.groupby("serie")}
    peor: dict[str, dict[str, Any]] = {}
    n_comp = set()
    for t in [pd.Timestamp(c) for c in cortes]:
        l_t = largas[pub <= t]
        d_t = dias[dias <= t]
        if l_t.empty or len(d_t) == 0:
            continue
        z_t, tr_t = z_pipeline(l_t, d_t)
        for c in z_t.columns:
            if c not in zs or (z_t[c].notna().sum() == 0 and zs[c].loc[:t].notna().sum() == 0):
                continue
            n_comp.add(c)
            a, b = z_t[c], zs[c].reindex(d_t)
            ambos = a.notna() & b.notna()
            dif = float((a[ambos] - b[ambos]).abs().max()) if ambos.any() else 0.0
            dist_nan = int((a.notna() != b.notna()).sum())
            cambia_tr = tr_t.get(c) != tr_full.get(c)
            if dif > tol or dist_nan or cambia_tr:
                p = peor.setdefault(c, dict(dif=0.0, nan=0, cortes=[], tr=None))
                p["dif"] = max(p["dif"], dif)
                p["nan"] = max(p["nan"], dist_nan)
                p["cortes"].append(t.date().isoformat())
                if cambia_tr:
                    p["tr"] = f"{tr_t.get(c)} con datos hasta {t.date()} frente a {tr_full.get(c)} con la muestra entera"
    filas = []
    for c, p in sorted(peor.items()):
        filas.append(_fila_la("normalizacion_sin_futuro_serie", c, "riesgo",
                              f"máx |z(hasta t) − z(muestra entera)| = {p['dif']:.3g}; {p['nan']} días con dato en un "
                              f"pipeline y no en el otro; cortes {', '.join(p['cortes'])}"
                              + (f"; transformación {p['tr']}" if p["tr"] else ""), round(p["dif"], 4)))
    mx = max((p["dif"] for p in peor.values()), default=0.0)
    filas.insert(0, _fila_la(
        "normalizacion_expanding_sin_futuro", "todas", "riesgo" if peor else "ok",
        f"pipeline completo rehecho en {len(cortes)} cortes solo con lo publicado hasta cada corte "
        f"(rejilla, vigencia, conteos, transformación y z expanding de mínimo {MIN_OBS_Z} obs): {len(peor)} de "
        f"{len(n_comp)} series difieren (máx |dif. z| {mx:.3g})"
        + (f": {', '.join(sorted(peor))[:200]}" if peor else ""), round(mx, 4)))
    return filas


# =========================================================================== 4. calibración

def brier(p: Iterable[float], y: Iterable[float]) -> float:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    ok = ~(np.isnan(p) | np.isnan(y))
    return float(np.mean((p[ok] - y[ok]) ** 2)) if ok.any() else float("nan")


def murphy(p: Iterable[float], y: Iterable[float], n_bins: int = 10) -> dict[str, float]:
    """Descomposición de Murphy en ``n_bins`` tramos iguales: brier ≈ fiabilidad − resolución + incertidumbre."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    ok = ~(np.isnan(p) | np.isnan(y))
    p, y = p[ok], y[ok]
    n = p.size
    if n == 0:
        return dict(n=0, brier=np.nan, fiabilidad=np.nan, resolucion=np.nan, incertidumbre=np.nan)
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    o = y.mean()
    rel = res = 0.0
    for k in np.unique(b):
        m = b == k
        rel += m.sum() * (p[m].mean() - y[m].mean()) ** 2
        res += m.sum() * (y[m].mean() - o) ** 2
    return dict(n=int(n), brier=float(np.mean((p - y) ** 2)), fiabilidad=rel / n, resolucion=res / n,
                incertidumbre=float(o * (1 - o)))


def bins_fiabilidad(p: Iterable[float], y: Iterable[float], n_bins: int = 10) -> pd.DataFrame:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    ok = ~(np.isnan(p) | np.isnan(y))
    p, y = p[ok], y[ok]
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    filas = []
    for k in range(n_bins):
        m = b == k
        filas.append(dict(bin=k, lim_inf=k / n_bins, lim_sup=(k + 1) / n_bins, n=int(m.sum()),
                          p_media=float(p[m].mean()) if m.any() else np.nan,
                          frec_obs=float(y[m].mean()) if m.any() else np.nan))
    return pd.DataFrame(filas)


def calibracion(cal: pd.DataFrame, horizontes: Iterable[int] = (1, 7, 30), n_bins: int = 10
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Brier + Murphy por plataforma, horizonte y grupo.

    Grupos: ``todos``; ``volumen_alto``/``volumen_bajo`` (mediana del volumen FINAL de la plataforma: dato ex
    post, solo descriptivo); ``indeciso``: mercados con la probabilidad del horizonte dentro de
    ``PROB_INDECISA`` (fuera, el mercado ya está decidido y el Brier frente al climatológico es trivial).
    ``n_decididos``: mercados del grupo con probabilidad fuera de ese rango. ``bss`` = 1 − brier/brier_clima.
    """
    if cal is None or cal.empty:
        return pd.DataFrame(), pd.DataFrame()
    cal = cal.copy()
    cal["resultado"] = pd.to_numeric(cal["resultado"], errors="coerce")
    cal["volumen_usd"] = pd.to_numeric(cal.get("volumen_usd"), errors="coerce")
    lo_i, hi_i = PROB_INDECISA
    res, bins = [], []
    for plat, dp in cal.groupby("plataforma"):
        med = dp["volumen_usd"].median()
        for h in horizontes:
            col = f"p_{h}d"
            if col not in dp:
                continue
            ph = pd.to_numeric(dp[col], errors="coerce")
            grupos = {"todos": dp, "volumen_alto": dp[dp["volumen_usd"] >= med],
                      "volumen_bajo": dp[dp["volumen_usd"] < med], "indeciso": dp[ph.between(lo_i, hi_i)]}
            for liq, g in grupos.items():
                p = pd.to_numeric(g[col], errors="coerce")
                m = murphy(p, g["resultado"], n_bins)
                base_rate = float(g["resultado"].mean()) if len(g) else np.nan
                clima = float(base_rate * (1 - base_rate)) if not np.isnan(base_rate) else np.nan
                res.append(dict(plataforma=plat, horizonte_dias=h, liquidez=liq, umbral_volumen_usd=round(float(med), 2),
                                **m, brier_clima=clima, frec_yes=base_rate,
                                n_decididos=int((p.notna() & ~p.between(lo_i, hi_i)).sum()),
                                bss=1 - m["brier"] / clima if clima and not np.isnan(clima) and m["n"] else np.nan))
                if liq == "todos":
                    bins.append(bins_fiabilidad(p, g["resultado"], n_bins).assign(plataforma=plat, horizonte_dias=h))
    fb = pd.concat(bins, ignore_index=True) if bins else pd.DataFrame()
    if not fb.empty:
        fb = fb[["plataforma", "horizonte_dias", "bin", "lim_inf", "lim_sup", "n", "p_media", "frec_obs"]]
    return pd.DataFrame(res), fb


# =========================================================================== 5. rentabilidad anormal

def car_evento(precio: pd.Series, mercado: pd.Series, fecha: pd.Timestamp, h: int) -> float:
    """CAR ajustado por mercado: (P_{t+h}/P_t − 1) − (M_{t+h}/M_t − 1), t = primera sesión >= ``fecha``.

    Entrada al cierre de la primera sesión en o después de la fecha (lo publicado ese día se puede operar al
    cierre). NaN si no hay ``h`` sesiones después.
    """
    j = pd.concat([precio.rename("p"), mercado.rename("m")], axis=1, join="inner").dropna().sort_index()
    if j.empty:
        return float("nan")
    i = j.index.searchsorted(pd.Timestamp(fecha))
    if i >= len(j) or i + h >= len(j) or (j.index[i] - pd.Timestamp(fecha)).days > 7:
        return float("nan")
    a, b = j.iloc[i], j.iloc[i + h]
    return float((b["p"] / a["p"] - 1) - (b["m"] / a["m"] - 1))


def eventos_compra(congreso: pd.DataFrame | None, insiders: pd.DataFrame | None) -> pd.DataFrame:
    """Eventos (grupo, ticker, fecha_publicacion, fecha_operacion), uno por ticker, grupo y día de publicación."""
    partes = []
    if congreso is not None and not congreso.empty:
        c = congreso[(congreso["tipo"].astype(str) == "compra") & congreso["ticker"].notna()]
        partes.append(pd.DataFrame(dict(grupo="congreso", ticker=c["ticker"], fecha_publicacion=c["fecha_publicacion"],
                                        fecha_operacion=c["fecha_operacion"])))
    if insiders is not None and not insiders.empty:
        der = insiders["derivado"].astype(str).str.lower().isin(["true", "1"])
        i = insiders[(insiders["codigo"].astype(str) == "P") & ~der & insiders["ticker"].notna()]
        partes.append(pd.DataFrame(dict(grupo="insiders", ticker=i["ticker"], fecha_publicacion=i["fecha_publicacion"],
                                        fecha_operacion=i["fecha_transaccion"])))
    if not partes:
        return pd.DataFrame(columns=["grupo", "ticker", "fecha_publicacion", "fecha_operacion"])
    e = pd.concat(partes, ignore_index=True)
    e["ticker"] = e["ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    e = e[e["ticker"].str.fullmatch(r"[A-Z][A-Z\-]{0,6}")]
    for c in ("fecha_publicacion", "fecha_operacion"):
        e[c] = pd.to_datetime(e[c], errors="coerce").dt.normalize()
    e = e.dropna(subset=["fecha_publicacion"])
    e = e.sort_values("fecha_operacion").groupby(["grupo", "ticker", "fecha_publicacion"], as_index=False).first()
    return e.reset_index(drop=True)


def car_tabla(eventos: pd.DataFrame, precios: dict[str, pd.Series], mercado: pd.Series,
              horizontes: Iterable[int] = HORIZONTES_CAR, n_boot: int = N_BOOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(resumen por grupo/referencia/horizonte con IC bootstrap, CAR por evento). Tickers sin precio se cuentan.

    El IC remuestrea SEMANAS completas (``bootstrap_media_cluster``): los eventos de una misma semana comparten
    el mercado y las ventanas de ``h`` sesiones se solapan. ``pct_sin_precio`` mide el riesgo de supervivencia
    (los tickers deslistados no tienen precio en yfinance).
    """
    filas_ev = []
    for e in eventos.itertuples(index=False):
        p = precios.get(e.ticker)
        f = dict(grupo=e.grupo, ticker=e.ticker, fecha=e.fecha_publicacion.date().isoformat(),
                 fecha_operacion=e.fecha_operacion.date().isoformat() if pd.notna(e.fecha_operacion) else None,
                 con_precio=p is not None)
        for h in horizontes:
            f[f"car_{h}"] = car_evento(p, mercado, e.fecha_publicacion, h) if p is not None else np.nan
            f[f"car_{h}_desde_operacion"] = (car_evento(p, mercado, e.fecha_operacion, h)
                                             if p is not None and pd.notna(e.fecha_operacion) else np.nan)
        filas_ev.append(f)
    ev = pd.DataFrame(filas_ev)
    filas = []
    if ev.empty:
        return pd.DataFrame(columns=["grupo", "referencia_fecha", "horizonte", "n", "car_medio", "lo", "hi",
                                     "pct_positivos", "n_eventos", "n_sin_precio", "pct_sin_precio",
                                     "n_semanas"]), ev
    for g, d in ev.groupby("grupo"):
        for ref, suf, col_f in (("publicacion", "", "fecha"), ("operacion", "_desde_operacion", "fecha_operacion")):
            semana = pd.to_datetime(d[col_f]).dt.to_period("W").astype(str)
            for h in horizontes:
                v = d[f"car_{h}{suf}"]
                ok = v.notna()
                m, lo, hi = bootstrap_media_cluster(v[ok], semana[ok], n_boot=n_boot, semilla=h)
                n_sp = int((~d["con_precio"]).sum())
                filas.append(dict(grupo=g, referencia_fecha=ref, horizonte=h, n=int(ok.sum()), car_medio=m, lo=lo, hi=hi,
                                  pct_positivos=float((v[ok] > 0).mean() * 100) if ok.any() else np.nan,
                                  n_eventos=int(len(d)), n_sin_precio=n_sp,
                                  pct_sin_precio=round(n_sp / len(d), 4) if len(d) else np.nan,
                                  n_semanas=int(semana[ok].nunique())))
    return pd.DataFrame(filas), ev


# =========================================================================== 6. semáforo

def _punt(valor: float | None, bueno: float, regular: float, mayor_mejor: bool = True) -> int | None:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return None
    if mayor_mejor:
        return 2 if valor >= bueno else (1 if valor >= regular else 0)
    return 2 if valor <= bueno else (1 if valor <= regular else 0)


def veredicto(p: dict[str, Any]) -> tuple[str, str]:
    """Veredicto y motivo a partir de las puntuaciones (0-2 o None) de un candidato.

    Claves: historia, calidad, lookahead, senal, acceso (0-2/None), ``tipo`` ('serie', 'texto', 'evento') y
    ``n_crisis``. Series y eventos solo son ``apta_backtest`` con historia 2 (>= ``CRISIS_BUENA`` crisis
    cubiertas): con menos, ningún AUC ni CAR tiene potencia y la fuente queda para el vivo. ``sin_senal``
    (opcional) explica por qué no hay medida de señal (sin signo a priori, supervivencia...).
    """
    h, c, la, s, a = (p.get(k) for k in ("historia", "calidad", "lookahead", "senal", "acceso"))
    tipo, n = p.get("tipo"), p.get("n_crisis")
    if a == 0:
        return "descartar", "acceso fallido o no probado: sin datos medibles"
    if la == 0:
        return "descartar", "look-ahead: hay fechas de publicación anteriores al hecho"
    if c == 0:
        return "descartar", "calidad insuficiente (huecos o interrupciones graves)"
    corta = f"historia corta ({n} crisis cubiertas, < {CRISIS_BUENA})" if n is not None else "historia corta"
    if h is None or h == 0:
        return "solo_vivo", f"{corta}; útil en vivo, no para backtest"
    if tipo == "texto" and a == 1 and h == 2:
        return "contexto", ("texto con cobertura de crisis pero acceso no verificado (gated, sin vista previa o "
                            "fechas solo documentadas): pendiente de medirlo antes de usarlo en backtest")
    if tipo == "texto":
        return ("apta_backtest", "texto con cobertura de varias crisis y fecha de publicación fiable (cajón RAG)") \
            if h == 2 else ("contexto", "texto con cobertura parcial de las crisis: contexto para el RAG")
    if h == 1:
        extra = "; la señal medida es clara pero sin potencia" if s == 2 else "; útil en vivo, no para backtest"
        return "solo_vivo", f"{corta}{extra}"
    if s is None:
        return "contexto", "historia suficiente pero " + (p.get("sin_senal") or "sin medida de señal frente a crisis")
    riesgo = " (look-ahead con riesgo documentado: ver validez_lookahead)" if la == 1 else ""
    if a == 1:
        riesgo += " (acceso parcial o fuente congelada: solo backtest hasta su fin)"
    if s >= 1:
        if tipo == "evento":
            return "apta_backtest", f"cubre crisis y su medida propia (CAR o Brier) se separa del azar{riesgo}"
        return "apta_backtest", f"cubre crisis y separa crisis de no crisis en el sentido esperado{riesgo}"
    if tipo == "evento":
        return "contexto", "cubre crisis pero su medida propia no se separa del azar (CAR o Brier)"
    return "contexto", ("cubre crisis pero la señal no basta en el sentido esperado (|AUC − 0,5| < "
                        f"{AUC_REGULAR:g}, IC por bloques con el 0,5 o signo contrario al del estudio de eventos)")


def semaforo(candidatos: list[dict[str, Any]]) -> pd.DataFrame:
    filas = []
    for p in candidatos:
        v, m = veredicto(p)
        filas.append({**p, "veredicto": v, "motivo": m})
    cols = ["candidato", "tipo", "fuente", "inicio", "fin", "anios", "n_crisis", "historia", "calidad", "lookahead",
            "senal", "acceso", "metrica_senal", "veredicto", "motivo"]
    t = pd.DataFrame(filas)
    for c in cols:
        if c not in t:
            t[c] = None
    return t[cols + [c for c in t.columns if c not in cols]]


def crisis_en_rango(inicio: Any, fin: Any, crisis: dict[str, tuple[str, str]]) -> int:
    """Crisis cuya ventana pico->suelo solapa [inicio, fin]."""
    if inicio is None or (isinstance(inicio, float) and np.isnan(inicio)) or str(inicio) in ("", "None", "nan"):
        return 0
    a = pd.Timestamp(str(inicio)[:10])
    b = pd.Timestamp(str(fin)[:10]) if fin is not None and str(fin) not in ("", "None", "nan") else pd.Timestamp.today()
    return sum(1 for p, s in crisis.values() if pd.Timestamp(p) <= b and pd.Timestamp(s) >= a)


# =========================================================================== ejecutar

def _tabla(nombre: str) -> pd.DataFrame | None:
    ruta = resultados_dir(base.TEMA) / f"{nombre}.csv"
    if ruta.exists():
        try:
            return pd.read_csv(ruta)
        except Exception:  # noqa: BLE001
            return None
    return None


def _tablas_sonda(nombres: Iterable[str], esperar_s: float = 0.0) -> dict[str, pd.DataFrame | None]:
    """Lee tablas de la sonda en results/; si faltan, reintenta cada 30 s hasta ``esperar_s``."""
    t0 = time.monotonic()
    out = {n: _tabla(n) for n in nombres}
    while any(v is None for v in out.values()) and time.monotonic() - t0 < esperar_s:
        time.sleep(30)
        out.update({n: _tabla(n) for n, v in out.items() if v is None})
    return out


# Tabla de la sonda -> módulo que la produce (para recalcularla desde la caché si aún no está en results/).
MODULO_TABLA = {"prediccion_calibracion": "prediccion", "congreso_operaciones": "congreso",
                "insiders_operaciones": "insiders", "ews_z": "ews", "gdelt_eventos_lag": "gdelt",
                "wiki_current_events": "wikipedia"}


def _tablas_desde_modulos(faltan: Iterable[str]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Recalcula con la caché del corpus (``sondear``) las tablas que faltan; devuelve (tablas, timegate)."""
    import importlib

    mods = sorted({MODULO_TABLA[n] for n in faltan if n in MODULO_TABLA})
    tablas: dict[str, pd.DataFrame] = {}
    tg = []
    for m in mods:
        try:
            res = importlib.import_module(f"ingesta.noticias_alt.{m}").sondear(base.ConfigSonda(verbose=False))
        except Exception as e:  # noqa: BLE001
            print(f"[validez] {m}.sondear falló: {type(e).__name__}: {e}", flush=True)
            continue
        tablas.update(res.tablas)
        tg.extend(res.timegate)
    return tablas, pd.DataFrame(tg, columns=base.COLUMNAS_TIMEGATE)


def _fichas() -> list[dict]:
    try:
        from ingesta.comun.fichas import cargar_fichas

        return cargar_fichas(base.TEMA)
    except Exception:  # noqa: BLE001
        return []


def _acceso(estado: str | None) -> int | None:
    if estado is None:
        return None
    e = str(estado)
    return {"ok": 2, "parcial": 1, "fallo": 0, "no_probada": 0}.get(e.split()[0], None)


def ejecutar(*, esperar_s: float = 120.0, sondear_si_falta: bool = True, n_boot: int = N_BOOT, con_car: bool = True,
             verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Calcula los seis niveles con el corpus y la sonda; guarda ``validez_*.csv``. Devuelve las tablas.

    Las tablas de la sonda se leen de results/ (esperando hasta ``esperar_s``); las que sigan faltando se
    recalculan con ``sondear`` del módulo y su caché (``sondear_si_falta``).
    """
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    tabs = _tablas_sonda(["timegate", "cobertura", "prediccion_calibracion", "congreso_operaciones",
                          "insiders_operaciones", "ews_z", "gdelt_eventos_lag", "wiki_current_events"], esperar_s)
    faltan = [n for n, t in tabs.items() if t is None and n in MODULO_TABLA]
    if sondear_si_falta and faltan:
        log(f"[validez] recalculando desde la caché: {faltan}")
        extra, tg = _tablas_desde_modulos(faltan)
        for n in faltan:
            tabs[n] = extra.get(n)
        if tabs["timegate"] is None and not tg.empty:
            tabs["timegate"] = tg
    log("[validez] tablas de la sonda:", {k: (None if v is None else len(v)) for k, v in tabs.items()})
    crisis = _crisis()
    largas = leer_corpus_series()
    rej, meta = cargar_series(largas)
    dias = rej.index
    if SERIE_VIX not in rej:
        rej[SERIE_VIX] = a_rejilla(vix_referencia(largas), dias)
    spx = sp500_y_drawdown(dias)
    log(f"[validez] {rej.shape[1]} series en rejilla {dias.min().date()}..{dias.max().date()}; "
        f"S&P {spx['sp500'].notna().sum() if not spx.empty else 0} días")

    # 1. calidad
    calidad = pd.DataFrame([calidad_serie(g) for _, g in largas.groupby("serie")])
    calidad = calidad.merge(meta[["serie", "fichero", "espaciado_habil"]], on="serie", how="left")

    # 3. señal: z causal en rejilla (tras la transformación de calidad)
    trans = dict(zip(calidad["serie"], calidad["transformacion"]))
    zs = pd.DataFrame({c: z_expanding(aplicar_transformacion(rej[c], trans.get(c, "nivel"))) for c in rej.columns},
                      index=dias)
    z_vix = zs[SERIE_VIX]
    dd = spx["drawdown"] if "drawdown" in spx else pd.Series(np.nan, index=dias)
    # Lead-lag en dos rejillas: por fecha_referencia (contenido: ¿quién se mueve antes?) y por publicación
    # (utilizable: ¿quién se sabe antes?). El drawdown sigue la convención del VIX en cada una: día D por
    # referencia, D+1 por publicación (el cierre de D se usa desde el día hábil siguiente).
    rej_ref, _ = cargar_series(largas, dias, por="referencia")
    if SERIE_VIX not in rej_ref:
        rej_ref[SERIE_VIX] = a_rejilla_ref(vix_referencia(largas), dias)
    zs_ref = pd.DataFrame({c: z_expanding(aplicar_transformacion(rej_ref[c], trans.get(c, "nivel")))
                           for c in rej_ref.columns}, index=dias)
    rejillas = {"referencia": (rej_ref, zs_ref, dd), "publicacion": (rej, zs, dd.shift(1))}
    suf = {"referencia": "_ref", "publicacion": "_pub"}
    lag_pub = dict(zip(calidad["serie"], calidad["lag_pub_mediana_dias"]))
    ev, aucs, ll, est = [], [], [], []
    ll_res: dict[tuple[str, str, str], dict[str, Any]] = {}
    for c in zs.columns:
        z = zs[c]
        if z.notna().sum() < 20:
            continue
        ev.append(estudio_eventos(z, crisis, n_boot=n_boot).assign(serie=c))
        aucs.append(auc_crisis(z, crisis, z_vix=z_vix).assign(serie=c))
        est.append(estabilidad(z, crisis, z_vix=z_vix).assign(serie=c))
        for nombre_r, (rj, zz, ddr) in rejillas.items():
            if c not in rj:
                continue
            x_niv = aplicar_transformacion(rj[c], trans.get(c, "nivel"))
            for ref_nombre, ref_serie, ref_z in (("vix", rj[SERIE_VIX], zz[SERIE_VIX]), ("drawdown_sp500", ddr, ddr)):
                if c == SERIE_VIX and ref_nombre == "vix":
                    continue
                for metrica, xs, rs in (("cambio_5d", x_niv.diff(5), ref_serie.diff(5)), ("nivel_z", zz[c], ref_z)):
                    t = lead_lag(xs, rs).assign(serie=c, referencia=ref_nombre, metrica=metrica, rejilla=nombre_r)
                    ll.append(t)
                    r = resumen_lead_lag(t, n_efectivo(xs, rs, metrica))
                    fila = ll_res.setdefault((c, ref_nombre, metrica), dict(
                        serie=c, referencia=ref_nombre, metrica=metrica, lag_pub_serie_d=lag_pub.get(c),
                        lag_pub_referencia_d=lag_pub.get(SERIE_VIX) if ref_nombre == "vix" else 1.0))
                    fila.update({k + suf[nombre_r]: v for k, v in r.items()})
    t_ev = _orden(pd.concat(ev, ignore_index=True), ["serie", "dia_rel", "media", "lo", "hi", "n_crisis"])
    t_auc = _orden(pd.concat(aucs, ignore_index=True), ["serie"])
    t_est = _orden(pd.concat(est, ignore_index=True), ["serie"]) if est else pd.DataFrame()
    t_ll = _orden(pd.concat(ll, ignore_index=True), ["serie", "referencia", "metrica", "rejilla", "lag", "corr", "n"])
    t_llr = pd.DataFrame(list(ll_res.values()))
    a0 = t_auc[(t_auc["serie"] == SERIE_VIX) & (t_auc["adelanto"] == 0)]["auc"]
    log(f"[validez] AUC del VIX (adelanto 0): {float(a0.iloc[0]):.3f}" if len(a0) else "[validez] sin AUC del VIX")

    # 2. look-ahead
    la = (lookahead_series(largas) + lookahead_timegate(tabs["timegate"]) + lookahead_documentos()
          + lookahead_ews(largas[largas["serie"] == "ews_z_publicado"], largas[largas["serie"] == "ews_z_causal"],
                          tabs["ews_z"])
          + lookahead_gdelt(tabs["gdelt_eventos_lag"], tabs["timegate"])
          + lookahead_wikipedia(tabs["wiki_current_events"], base.leer_documentos("wiki_current_events"))
          + lookahead_congreso(tabs["congreso_operaciones"]) + lookahead_insiders(tabs["insiders_operaciones"])
          + lookahead_fijas() + comprobar_z_causal(largas, zs))
    t_la = pd.DataFrame(la)

    # 4. calibración
    t_cal, t_bins = calibracion(tabs["prediccion_calibracion"])

    # 5. CAR
    t_car, t_car_ev = pd.DataFrame(), pd.DataFrame()
    if con_car:
        eventos = eventos_compra(tabs["congreso_operaciones"], tabs["insiders_operaciones"])
        if not eventos.empty:
            precios = precios_yf(list(eventos["ticker"].unique()) + ["SPY"], inicio="2007-01-01")
            spy = precios.get("SPY", pd.Series(dtype=float))
            t_car, t_car_ev = car_tabla(eventos, precios, spy, n_boot=n_boot)
            log(f"[validez] CAR: {len(eventos)} eventos, {t_car_ev['con_precio'].sum()} con precio")

    # 6. semáforo
    t_sem = semaforo(_candidatos(calidad, t_auc, t_la, t_cal, t_car, tabs, crisis, t_ev))

    salidas = {"validez_calidad": calidad, "validez_lookahead": t_la, "validez_eventos": t_ev, "validez_auc": t_auc,
               "validez_leadlag": t_ll, "validez_leadlag_resumen": t_llr, "validez_estabilidad": t_est,
               "validez_calibracion": t_cal, "validez_fiabilidad_bins": t_bins, "validez_car": t_car,
               "validez_car_eventos": t_car_ev, "validez_semaforo": t_sem}
    for nombre, df in salidas.items():
        if df is not None and not df.empty:
            base.guardar_tabla(_redondear(df), nombre)
    log("[validez] guardadas:", ", ".join(f"{k} ({len(v)})" for k, v in salidas.items() if v is not None and not v.empty))
    return salidas


def _crisis() -> dict[str, tuple[str, str]]:
    from ingesta.noticias_alt.muestra import crisis_fase1

    return crisis_fase1("A")


def _orden(df: pd.DataFrame, primeras: list[str]) -> pd.DataFrame:
    return df[[c for c in primeras if c in df] + [c for c in df.columns if c not in primeras]]


def _redondear(df: pd.DataFrame, dec: int = 4) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes(include="float").columns:
        out[c] = out[c].round(dec)
    return out


def senal_serie(serie: str, a: pd.DataFrame, t_ev: pd.DataFrame | None) -> tuple[int | None, str, str | None]:
    """Puntuación de señal (0-2 o None), métrica y motivo si no hay señal, para una serie con su ``validez_auc``.

    Reglas fijadas antes de mirar los datos: (1) solo con ``SIGNO_ESPERADO`` (sin signo a priori -> None,
    contexto); (2) AUC con adelanto 0 en los mismos días que el VIX, contado SOLO en el sentido esperado
    (dev = signo·(AUC − 0,5); en contra -> 0); (3) señal >= 1 solo si el IC por bloques de crisis excluye 0,5 en
    ese sentido; (4) y si la media del estudio de eventos en ``VENTANA_SIGNO`` tiene el mismo signo.
    """
    signo = signo_esperado(serie)
    n_cri = int(a["n_crisis"].max()) if len(a) else 0
    a0 = a[a["adelanto"] == 0]
    col = "auc_mismo_periodo" if "auc_mismo_periodo" in a0 else "auc"
    if a0.empty or a0[col].isna().all():
        return None, "sin z suficiente (menos de 252 obs) o sin días de crisis", None
    r = a0.iloc[0]
    auc, vix = float(r[col]), r.get("auc_vix_mismo_periodo")
    lo, hi = r.get("auc_lo", np.nan), r.get("auc_hi", np.nan)
    met = (f"AUC h=0 {auc:.3f} [{lo:.3f}, {hi:.3f}] en los días del VIX (VIX {vix:.3f})"
           + "; otros adelantos: " + ", ".join(f"h={int(x.adelanto)} {getattr(x, col):.3f}"
                                               for x in a[a["adelanto"] != 0].itertuples()))
    if signo is None:
        return None, met + "; sin signo esperado a priori", "sin signo esperado a priori (contexto)"
    met += f"; signo esperado {'+' if signo > 0 else '−'}"
    if n_cri < CRISIS_REGULAR:
        return None, met, None
    dev = signo * (auc - 0.5)
    senal = _punt(dev, AUC_BUENA, AUC_REGULAR) if dev > 0 else 0
    if dev <= 0:
        met += "; AUC en el sentido contrario al esperado"
    ic_ok = (lo > 0.5) if signo > 0 else (hi < 0.5)
    if senal and not (np.isfinite(lo) and np.isfinite(hi) and ic_ok):
        senal, met = 0, met + "; IC por bloques de crisis con el 0,5"
    if t_ev is not None and not t_ev.empty:
        e = t_ev[(t_ev["serie"] == serie) & t_ev["dia_rel"].between(*VENTANA_SIGNO)
                 & (t_ev["n_crisis"] >= CRISIS_REGULAR)]
        if len(e):
            m = float(e["media"].mean())
            met += f"; eventos (días {VENTANA_SIGNO[0]}..+{VENTANA_SIGNO[1]}) z {m:+.2f}"
            if senal and np.sign(m) != signo:
                senal, met = 0, met + " (signo contrario al esperado)"
        else:
            met += "; sin estudio de eventos para contrastar el signo"
    return senal, met, None


def _candidatos(calidad: pd.DataFrame, t_auc: pd.DataFrame, t_la: pd.DataFrame, t_cal: pd.DataFrame,
                t_car: pd.DataFrame, tabs: dict[str, pd.DataFrame | None], crisis: dict[str, tuple[str, str]],
                t_ev: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    """Puntuaciones por serie, por fuente de texto y por alternativo de evento."""
    out: list[dict[str, Any]] = []
    peor_la = {}
    for f, g in t_la.groupby("fuente"):
        r = set(g["resultado"])
        peor_la[f] = 0 if "fallo" in r else (1 if "riesgo" in r else 2)
    # --- series
    for q in calidad.itertuples(index=False):
        a = t_auc[t_auc["serie"] == q.serie]
        n_cri = int(a["n_crisis"].max()) if len(a) else 0
        senal, met, sin_senal = senal_serie(q.serie, a, t_ev)
        huecos = q.pct_huecos / 100 if q.pct_huecos is not None and not np.isnan(q.pct_huecos) else None
        if q.conteo_sin_ceros:
            huecos = 0.0                                   # días sin fila = 0 eventos, no huecos
        cal = _punt(huecos, HUECOS_BUENO, HUECOS_REGULAR, mayor_mejor=False)
        racha = getattr(q, "racha_max_constante", 0)
        if racha is not None and not pd.isna(racha) and racha >= RACHA_MALA:
            cal = 0                                        # hueco codificado como valor constante o fuente parada
            met += f"; racha de {int(racha)} valores idénticos seguidos (>= {RACHA_MALA})"
        congelada = pd.Timestamp(q.fin) < pd.Timestamp.today() - pd.Timedelta(days=365)
        if congelada and cal is not None:
            cal = min(cal, 1)
        anios = (pd.Timestamp(q.fin) - pd.Timestamp(q.inicio)).days / 365.25
        out.append(dict(candidato=q.serie, tipo="serie", fuente=q.fuente, inicio=q.inicio, fin=q.fin,
                        anios=round(anios, 1), n_crisis=n_cri,
                        historia=_punt(n_cri, CRISIS_BUENA, CRISIS_REGULAR), calidad=cal,
                        lookahead=peor_la.get(q.serie, 2), senal=senal, acceso=2 if not congelada else 1,
                        metrica_senal=met + ("; serie congelada desde " + q.fin if congelada else ""),
                        signo_esperado=signo_esperado(q.serie), sin_senal=sin_senal))
    # --- fuentes de texto y de evento desde cobertura (o fichas)
    cob = tabs.get("cobertura")
    filas = []
    if cob is not None and not cob.empty:
        filas = cob.to_dict("records")
    else:
        for f in _fichas():
            filas.append(dict(fuente=f.get("fuente"), familia=None, tipo=None, inicio=f.get("cobertura_inicio"),
                              fin=f.get("cobertura_fin"), estado=f.get("estado_sonda"), rag=f.get("rag")))
    series_fuentes = set(calidad["fuente"])
    for f in filas:
        tipo_f = str(f.get("tipo") or "")
        if f.get("fuente") in series_fuentes and tipo_f in ("serie", ""):
            continue
        tipo = "texto" if tipo_f in ("texto", "titular", "") else "evento"
        n_cri = crisis_en_rango(f.get("inicio"), f.get("fin"), crisis)
        ini = None if f.get("inicio") is None or str(f.get("inicio")) == "nan" else str(f.get("inicio"))[:10]
        fin = None if f.get("fin") is None or str(f.get("fin")) == "nan" else str(f.get("fin"))[:10]
        out.append(dict(candidato=f.get("fuente"), tipo=tipo, fuente=f.get("fuente"), inicio=ini, fin=fin,
                        anios=round((pd.Timestamp(fin or pd.Timestamp.today()) - pd.Timestamp(ini)).days / 365.25, 1) if ini else None,
                        n_crisis=n_cri, historia=_punt(n_cri, CRISIS_BUENA, CRISIS_REGULAR), calidad=None,
                        lookahead=peor_la.get(f.get("fuente"), 2), senal=None, acceso=_acceso(f.get("estado")),
                        metrica_senal="cobertura por épocas (" + str(f.get("estado")) + ")"))
    # --- alternativos de evento con su medida propia (sustituyen a la fila genérica si la había)
    propios = _candidatos_evento(t_cal, t_car, t_la, tabs, crisis, peor_la)
    nombres = {p["candidato"] for p in propios}
    return [o for o in out if o["candidato"] not in nombres] + propios


def _candidatos_evento(t_cal: pd.DataFrame, t_car: pd.DataFrame, t_la: pd.DataFrame,
                       tabs: dict[str, pd.DataFrame | None], crisis: dict[str, tuple[str, str]],
                       peor_la: dict[str, int]) -> list[dict[str, Any]]:
    out = []
    # Congreso y Form 4: señal 2 solo si el IC por semanas del CAR desde la publicación excluye el 0 en al menos
    # 2 de los 3 horizontes con el mismo signo; sin señal (None) si falta precio en > MAX_SIN_PRECIO de los eventos.
    # Historia: crisis dentro de la muestra realmente medida (fechas de publicación de las operaciones).
    for grupo, cand, ops, col_f in (("congreso", "congreso_house_ptr", tabs.get("congreso_operaciones"), "fecha_publicacion"),
                                    ("insiders", "sec_form4_insiders", tabs.get("insiders_operaciones"), "fecha_publicacion")):
        if ops is None or ops.empty:
            continue
        f = pd.to_datetime(ops[col_f], errors="coerce").dropna()
        ini, fin = f.min().date().isoformat(), f.max().date().isoformat()
        senal, met, sin_senal = None, "sin CAR", None
        if t_car is not None and not t_car.empty:
            r = t_car[(t_car["grupo"] == grupo) & (t_car["referencia_fecha"] == "publicacion")].set_index("horizonte")
            if len(r) and r["n"].max() >= 10:
                signos = [int(np.sign(x.lo)) if x.lo > 0 or x.hi < 0 else 0 for x in r.itertuples()]
                mejor = max((signos.count(1), 1), (signos.count(-1), -1))
                senal = 2 if mejor[0] >= 2 else 0
                met = "CAR desde publicación (IC por semanas): " + "; ".join(
                    f"{int(h)} d {100 * x.car_medio:+.2f} % [{100 * x.lo:+.2f}, {100 * x.hi:+.2f}]" for h, x in r.iterrows())
                met += f"; n={int(r['n'].max())}; {mejor[0]} de {len(r)} horizontes excluyen el 0 con el mismo signo"
                psp = (float(r["pct_sin_precio"].iloc[0]) if "pct_sin_precio" in r
                       else float(r["n_sin_precio"].iloc[0]) / float(r["n_eventos"].iloc[0]))
                if psp > MAX_SIN_PRECIO:
                    senal, sin_senal = None, f"sin señal evaluable: {100 * psp:.0f} % de eventos sin precio (supervivencia)"
                    met += f"; {100 * psp:.0f} % sin precio (> {100 * MAX_SIN_PRECIO:.0f} %: supervivencia, no evaluable)"
        n_cri = crisis_en_rango(ini, fin, crisis)
        out.append(dict(candidato=cand, tipo="evento", fuente=cand, inicio=ini, fin=fin,
                        anios=round((pd.Timestamp(fin) - pd.Timestamp(ini)).days / 365.25, 1), n_crisis=n_cri,
                        historia=_punt(n_cri, CRISIS_BUENA, CRISIS_REGULAR), calidad=1,
                        lookahead=peor_la.get(cand, 2), senal=senal, acceso=2, sin_senal=sin_senal,
                        metrica_senal=met + f"; crisis en la muestra medida {ini}..{fin}"))
    # Mercados de predicción: señal 2 solo si, entre los mercados INDECISOS a 7 d (p en PROB_INDECISA, n >=
    # MIN_INDECISOS), el Brier queda por debajo del climatológico de ese mismo subconjunto.
    if t_cal is not None and not t_cal.empty:
        cal = tabs.get("prediccion_calibracion")
        for plat, g in t_cal[t_cal["horizonte_dias"] == 7].groupby("plataforma"):
            todos = g[g["liquidez"] == "todos"].iloc[0]
            ind = g[g["liquidez"] == "indeciso"]
            if len(ind) and ind["n"].iloc[0] >= MIN_INDECISOS:
                ri = ind.iloc[0]
                senal = 2 if ri["brier"] < ri["brier_clima"] else 0
                met = (f"indecisos a 7 d (p en [{PROB_INDECISA[0]:g}, {PROB_INDECISA[1]:g}]): Brier {ri['brier']:.3f} frente a "
                       f"climatológico {ri['brier_clima']:.3f}, n={int(ri['n'])}")
                sin_senal = None
            else:
                n_i = int(ind["n"].iloc[0]) if len(ind) else 0
                senal, met = None, f"solo {n_i} mercados indecisos a 7 d (< {MIN_INDECISOS})"
                sin_senal = "sin señal evaluable: pocos mercados indecisos"
            met += (f"; todos: Brier {todos['brier']:.3f} frente a {todos['brier_clima']:.3f}, n={int(todos['n'])}, "
                    f"{int(todos.get('n_decididos', 0))} ya decididos")
            ini = None
            if cal is not None and "inicio_historial" in cal:
                fi = pd.to_datetime(cal.loc[cal["plataforma"] == plat, "inicio_historial"], errors="coerce").dropna()
                ini = fi.min().date().isoformat() if len(fi) else None
            n_cri = crisis_en_rango(ini, None, crisis)
            out.append(dict(candidato=str(plat), tipo="evento", fuente=str(plat), inicio=ini, fin=None,
                            anios=round((pd.Timestamp.today() - pd.Timestamp(ini)).days / 365.25, 1) if ini else None,
                            n_crisis=n_cri, historia=_punt(n_cri, CRISIS_BUENA, CRISIS_REGULAR), calidad=1,
                            lookahead=peor_la.get(str(plat), 2), senal=senal, acceso=2, sin_senal=sin_senal,
                            metrica_senal=met))
    # EWS: archivo de un año móvil y z publicado revisado.
    e = t_la[(t_la["fuente"] == "ews") & (t_la["comprobacion"] == "ews_z_publicado_vs_causal")]
    if len(e):
        out.append(dict(candidato="ews", tipo="evento", fuente="ews", inicio=None, fin=None, anios=1.0, n_crisis=0,
                        historia=0, calidad=1, lookahead=1, senal=None, acceso=2,
                        metrica_senal="archivo de un año móvil; " + str(e["evidencia"].iloc[0])[:160]))
    return out


if __name__ == "__main__":  # pragma: no cover
    ejecutar()
