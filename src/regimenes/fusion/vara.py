"""Vara de medir común de las fusiones (ADR-011 §2 y §5).

La comparten las fusiones fijas (:mod:`regimenes.fusion.fijas`, notebook 15) y la online
(:mod:`regimenes.fusion.online`, notebook 16): mismos candidatos, mismas señales, mismas mitades,
mismo orden y mismo MCS que los individuales. Solo lee paneles OOS que ya existen; no reentrena
nada ni escribe en disco.

- :func:`candidatos`: los no supervisados con panel OOS de la pista (§2), en orden estable.
- :func:`paneles`, :func:`senales`, :func:`probabilidades`: panel, señal binaria
  (``state == n_states − 1``, §2.1) y ``p_crisis`` de cada candidato, sobre el índice OOS común.
- :func:`mitades`: H1 (selección) y H2 (evaluación), corte en ⌊n/2⌋ (§5.2).
- :func:`evaluar`: ``score_deteccion``, nivel y puesto en un tramo, con el orden del §5.2.
- :func:`mcs`: MCS al 10 % con los baselines de la vara y bloque fijo (§5.1), deduplicando
  señales idénticas.
- :func:`mejor_individual`: el primero de los candidatos en el orden de un tramo (I*).
- :func:`utilidad`: utilidad operativa de las fusiones antiguas D7+D8 y D2+D6 (lectura sin peso, §5.1).

Las cargas se cachean por pista (los paneles no cambian durante una sesión); las funciones
devuelven copias para que el llamador pueda modificarlas.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from regimenes import rutas

PISTAS = ("A", "B")
# §2: los 12 de v2 y las variantes no supervisadas de la Fase 2 (D16 y D17 quedan fuera).
V2_IDS: tuple[str, ...] = tuple(f"D{i:02d}" for i in range(1, 13))
POOL_IDS: dict[str, tuple[str, ...]] = {
    "A": ("D15-U", "D10-P", "D18-P"),
    "B": ("D15-U", "D10-P", "D18-P", "D10-R", "D14-R"),
}
PANELES_V2: Path = rutas.RESULTS_BENCHMARK / "panels"
PANELES_POOL: Path = rutas.RESULTS_DETECTORES / "pool"
# §5.1: bloque del bootstrap de todo el OOS (mcs.longitud_bloque_crisis); no se recalcula por tramo.
BLOQUE_ESPERADO = {"A": 66, "B": 50}
BASELINES = ("SIEMPRE_CRISIS", "SIEMPRE_CALMA", "AZAR_IID", "AZAR_PERSISTENTE", "VOL_P90")
COLUMNAS_EVALUAR = [
    "nivel", "score_deteccion", "det_precision", "det_event_recall", "switching_rate",
    "det_lift_precision", "det_base_rate", "det_marked_rate", "det_mean_crisis_run",
    "mean_regime_duration", "det_n_eventos", "det_n_detectados", "n_sesiones",
]


def _pista(pista: str) -> str:
    p = str(pista).upper()
    if p not in PISTAS:
        raise ValueError(f"pista desconocida: {pista!r} (A | B)")
    return p


# --------------------------------------------------------------------------- #
# Candidatos, paneles y señales (§2)
# --------------------------------------------------------------------------- #
def candidatos(pista: str) -> list[str]:
    """Ids de los candidatos de la pista (§2): D01–D12 y luego las variantes del pool.

    15 en A y 17 en B, en el orden de la tabla del §2 (es el orden de prioridad de los
    individuales al deduplicar en el MCS)."""
    p = _pista(pista)
    return [*V2_IDS, *POOL_IDS[p]]


def _carpeta(detector_id: str) -> Path:
    return PANELES_V2 if detector_id in V2_IDS else PANELES_POOL


@lru_cache(maxsize=None)
def _paneles(pista: str) -> dict[str, pd.DataFrame]:
    from regimenes.evaluacion import mcs as mcs_mod

    p = _pista(pista)
    out: dict[str, pd.DataFrame] = {}
    for d in candidatos(p):
        ruta = _carpeta(d) / f"{p}_{d}.parquet"
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el panel OOS {ruta}.")
        pan = pd.read_parquet(ruta, columns=["state", "p_crisis", "fold"])
        pan.index = pd.DatetimeIndex(pd.to_datetime(pan.index), name="date")
        # §2: un NaN es un error (como mcs.cargar_paneles y fusion.load_panel); no se rellena.
        if pan[["state", "p_crisis"]].isna().any().any():
            raise ValueError(f"{ruta}: 'state'/'p_crisis' contienen NaN.")
        if not pan.index.is_unique or not pan.index.is_monotonic_increasing:
            raise ValueError(f"{ruta}: el índice debe ser temporal, único y ordenado.")
        out[d] = pan
    ref = next(iter(out.values())).index
    distintos = [d for d, pan in out.items() if not pan.index.equals(ref)]
    if distintos:
        raise ValueError(f"Pista {p}: índice OOS distinto en {distintos} (el §2 exige el mismo).")
    # n_states del CSV de métricas (results/benchmark/metrics o results/detectores/pool).
    ns = {**mcs_mod.n_states_metricas(p, PANELES_V2, [d for d in out if d in V2_IDS]),
          **mcs_mod.n_states_metricas(p, PANELES_POOL, [d for d in out if d not in V2_IDS])}
    faltan = [d for d in out if d not in ns]
    if faltan:
        raise ValueError(f"Pista {p}: sin n_states en el CSV de métricas de {faltan}.")
    _N_STATES[p] = {d: int(ns[d]) for d in out}
    return out


_N_STATES: dict[str, dict[str, int]] = {}


def paneles(pista: str) -> dict[str, pd.DataFrame]:
    """``id -> panel OOS`` (``state``, ``p_crisis``, ``fold``) de cada candidato, índice común."""
    return {d: pan.copy() for d, pan in _paneles(_pista(pista)).items()}


def n_states(pista: str) -> dict[str, int]:
    """``n_states`` de cada candidato según su CSV de métricas (``mcs.n_states_metricas``)."""
    p = _pista(pista)
    _paneles(p)
    return dict(_N_STATES[p])


def estado_crisis(pista: str) -> dict[str, int]:
    """Estado de crisis de cada candidato: ``n_states − 1`` (§2.1)."""
    return {d: n - 1 for d, n in n_states(pista).items()}


@lru_cache(maxsize=None)
def _senales(pista: str) -> pd.DataFrame:
    from regimenes.evaluacion import mcs as mcs_mod

    p = _pista(pista)
    pan = _paneles(p)
    sen = mcs_mod.senales_crisis(pan, n_states=_N_STATES[p])
    return sen[candidatos(p)]


def senales(pista: str) -> pd.DataFrame:
    """Señal binaria de crisis (§2.1): bool, índice OOS, columnas = :func:`candidatos`."""
    return _senales(_pista(pista)).copy()


def probabilidades(pista: str) -> pd.DataFrame:
    """``p_crisis`` (float) de cada candidato, mismas filas y columnas que :func:`senales`."""
    p = _pista(pista)
    pan = _paneles(p)
    return pd.DataFrame({d: pan[d]["p_crisis"].astype(float) for d in candidatos(p)})


def indice(pista: str) -> pd.DatetimeIndex:
    """Índice OOS común de la pista."""
    return pd.DatetimeIndex(_senales(_pista(pista)).index)


def ventanas(pista: str) -> dict[str, tuple[str, str]]:
    """Ventanas ``[pico, suelo]`` del juez (``ranking.track_crisis_windows``)."""
    from regimenes.evaluacion import ranking as rk

    return rk.track_crisis_windows(_pista(pista))


# --------------------------------------------------------------------------- #
# Mitades (§5.2)
# --------------------------------------------------------------------------- #
def mitades(index: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """H1 (selección) y H2 (evaluación): corte en ⌊n/2⌋ sesiones, como el control por mitades
    de la antigua fusión D2+D6.

    A: 7.066 + 7.067 sesiones; B: 2.029 + 2.030."""
    index = pd.DatetimeIndex(index)
    corte = len(index) // 2
    return index[:corte], index[corte:]


def tramos(pista: str) -> dict[str, pd.DatetimeIndex]:
    """``{"oos": todo, "H1": ..., "H2": ...}`` de la pista."""
    idx = indice(pista)
    h1, h2 = mitades(idx)
    return {"oos": idx, "H1": h1, "H2": h2}


def _recortar(df: pd.DataFrame | pd.Series, tramo: pd.DatetimeIndex | None):
    if tramo is None:
        return df
    tramo = pd.DatetimeIndex(tramo)
    faltan = tramo.difference(df.index)
    if len(faltan):
        raise ValueError(f"El tramo tiene {len(faltan)} fechas fuera del índice de las señales.")
    return df.loc[tramo]


# --------------------------------------------------------------------------- #
# Evaluación y orden (§5.1 y §5.2)
# --------------------------------------------------------------------------- #
def nivel_deteccion(fila: dict | pd.Series) -> int:
    """Nivel de ``ranking.rank_detection`` sobre la señal binaria (§5.1).

    3 degenerado (``switching_rate`` 0, nunca marca o marca siempre); 2 parpadeo (duración media
    de régimen < 5 o de los episodios de crisis < 3); 1 precisión que no supera al azar
    (lift ≤ 1); 0 elegible. Mismos umbrales (``DETECTION_DEFAULTS``) y misma precedencia."""
    from regimenes.evaluacion import ranking as rk

    d = rk.DETECTION_DEFAULTS
    sw = fila.get("switching_rate", np.nan)
    sw = 0.0 if pd.isna(sw) else float(sw)
    prec, marked = fila.get("det_precision", np.nan), fila.get("det_marked_rate", np.nan)
    dur, run = fila.get("mean_regime_duration", np.nan), fila.get("det_mean_crisis_run", np.nan)
    lift = fila.get("det_lift_precision", np.nan)
    if sw == 0 or pd.isna(prec) or (pd.notna(marked) and (marked <= 0 or marked >= 1)):
        return 3
    if (pd.isna(dur) or dur < d["min_mean_duration"]
            or pd.isna(run) or run < d["min_crisis_run"]):
        return 2
    if not (pd.notna(lift) and lift > d["lift_min"]):
        return 1
    return 0


def metricas_senal(senal: pd.Series, win: dict[str, tuple[str, str]]) -> dict[str, float]:
    """Métricas de la vara de una señal binaria ya recortada al tramo."""
    from regimenes.evaluacion import metricas as mt
    from regimenes.evaluacion import ranking as rk

    s = pd.Series(senal).astype(bool)
    fila = mt.detection_summary(s, win, min_run=int(rk.DETECTION_DEFAULTS["min_run"]))
    fila = {k: v for k, v in fila.items() if not k.startswith("det_ev_")}
    estados = s.astype(int)
    fila["switching_rate"] = mt.switching_rate(estados)
    fila["mean_regime_duration"] = mt.mean_regime_duration(estados)
    fila["score_deteccion"] = rk.detection_score(fila["det_precision"], fila["det_event_recall"],
                                                 rk.DETECTION_DEFAULTS["beta"])
    fila["nivel"] = nivel_deteccion(fila)
    fila["n_sesiones"] = int(len(s))
    return fila


def ordenar(tabla: pd.DataFrame) -> pd.DataFrame:
    """Orden del §5.2: nivel → score a 3 decimales (desc.) → ``switching_rate`` (asc.) → id.

    ``tabla`` indexada por id con ``nivel``, ``score_deteccion`` y ``switching_rate``. Añade
    ``puesto`` (1..n, sin empates: el id desempata) y devuelve la tabla ordenada."""
    from regimenes.evaluacion import ranking as rk

    dec = int(rk.DETECTION_DEFAULTS["score_decimals"])
    claves = pd.DataFrame({
        "nivel": tabla["nivel"].astype(int).to_numpy(),
        "score": -tabla["score_deteccion"].astype(float).round(dec).to_numpy(),
        "switching": tabla["switching_rate"].astype(float).fillna(np.inf).to_numpy(),
        "clave_id": tabla.index.astype(str).to_numpy(),
    })
    pos = claves.sort_values(["nivel", "score", "switching", "clave_id"], kind="mergesort").index
    orden = tabla.index[pos.to_numpy()]
    out = tabla.loc[orden].copy()
    out["puesto"] = np.arange(1, len(out) + 1)
    return out


def evaluar(senales: pd.DataFrame, pista: str, tramo: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Una fila por columna de ``senales`` con la vara en el tramo (todo el OOS si ``None``).

    La señal se construye sobre todo el OOS y aquí solo se recorta (§5.2): ``detection_summary``
    usa las ventanas con días en el tramo, y el nivel, la tasa base y el ``switching_rate`` se
    calculan sobre el tramo. Columnas: ``score_deteccion``, ``det_precision``,
    ``det_event_recall``, ``switching_rate``, ``nivel``, ``puesto`` (orden del §5.2) y las
    auxiliares del nivel. Índice ``id``, ordenado por ``puesto``."""
    win = ventanas(pista)
    sen = _recortar(senales, tramo)
    filas = {str(c): metricas_senal(sen[c], win) for c in sen.columns}
    tabla = pd.DataFrame.from_dict(filas, orient="index")
    tabla.index.name = "id"
    tabla = ordenar(tabla[COLUMNAS_EVALUAR])
    return tabla[["puesto", *COLUMNAS_EVALUAR]]


def mejor_individual(pista: str, tramo: pd.DatetimeIndex | None = None) -> str:
    """I*: el primero de los candidatos en el orden del §5.2 del tramo.

    Con todo el OOS es el puesto 1 del ranking v3 ampliado (D10 en A, D05 en B)."""
    return str(evaluar(senales(pista), pista, tramo).index[0])


# --------------------------------------------------------------------------- #
# MCS (§5.1)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _baselines(pista: str) -> pd.DataFrame:
    from regimenes.benchmark import ablaciones_lectura as al

    b = al.senales_baselines_vara(_pista(pista))
    if not pd.DatetimeIndex(b.index).equals(indice(pista)):
        raise ValueError(f"{pista}: los baselines no tienen el índice OOS de los candidatos.")
    return b[list(BASELINES)]


def baselines(pista: str) -> pd.DataFrame:
    """Los 5 baselines de la vara (``ablaciones_lectura.senales_baselines_vara``)."""
    return _baselines(_pista(pista)).copy()


def bloque(pista: str) -> int:
    """Bloque del bootstrap de todo el OOS (``mcs.longitud_bloque_crisis``): 66 en A, 50 en B."""
    from regimenes.evaluacion import mcs as mcs_mod

    return int(mcs_mod.longitud_bloque_crisis(indice(pista), ventanas(pista)))


def deduplicar(senales: pd.DataFrame, prioridad: Sequence[str]) -> dict[str, str]:
    """``{columna eliminada: columna que queda}`` para señales idénticas día a día.

    Se recorre ``prioridad`` y se queda la primera de cada grupo de señales idénticas (§5.1:
    individuales, baselines, F*, V, M y O). Las columnas que no estén en ``prioridad`` van
    detrás, en el orden de ``senales``."""
    orden = [c for c in prioridad if c in senales.columns]
    orden += [c for c in senales.columns if c not in orden]
    vistas: dict[bytes, str] = {}
    fuera: dict[str, str] = {}
    for c in orden:
        firma = np.ascontiguousarray(senales[c].fillna(False).to_numpy(dtype=bool)).tobytes()
        if firma in vistas:
            fuera[str(c)] = vistas[firma]
        else:
            vistas[firma] = str(c)
    return fuera


def prioridad_por_defecto(columnas: Iterable[str], pista: str) -> list[str]:
    """Individuales (orden de :func:`candidatos`), baselines y el resto en su orden."""
    columnas = [str(c) for c in columnas]
    cand = [c for c in candidatos(pista) if c in columnas]
    base = [c for c in BASELINES if c in columnas]
    return [*cand, *base, *[c for c in columnas if c not in cand and c not in base]]


def mcs(senales: pd.DataFrame, pista: str, *, baselines: bool = True,
        tramo: pd.DatetimeIndex | None = None, prioridad: list[str] | None = None,
        bloque_len: int | None = None, **kwargs) -> pd.DataFrame:
    """MCS al 10 % de ``senales`` (más los 5 baselines si ``baselines``) en el tramo.

    Es la construcción de ``benchmark.pool.mcs_conjunto`` con otras columnas: pérdida 1 − J
    (λ = 1), ``T_max``, bootstrap estacionario, 5.000 réplicas de razón, semilla 0
    (``ablaciones_lectura.MCS_KW``; ``kwargs`` las sobrescribe, solo para tests). Bloque fijo
    de todo el OOS (66 A / 50 B) también en las mitades, salvo ``bloque_len``.

    Señales idénticas en el tramo: entra una sola (la primera según ``prioridad``; por defecto
    :func:`prioridad_por_defecto`) y la eliminada hereda su fila, con ``duplicado_de``.
    Devuelve la tabla del MCS (índice ``modelo``, columna ``incluido``) con ``pista`` y
    ``bloque``."""
    from regimenes.benchmark import ablaciones_lectura as al
    from regimenes.evaluacion import mcs as mcs_mod

    p = _pista(pista)
    sen = senales.copy()
    sen.columns = [str(c) for c in sen.columns]
    if baselines:
        b = _baselines(p)
        sen = pd.concat([sen, b[[c for c in b.columns if c not in sen.columns]]], axis=1)
    sen = _recortar(sen.astype(bool), tramo)
    orden = prioridad_por_defecto(sen.columns, p) if prioridad is None else [str(c) for c in prioridad]
    orden = [c for c in orden if c in sen.columns]
    orden += [c for c in sen.columns if c not in orden]
    fuera = deduplicar(sen, orden)
    quedan = [c for c in orden if c not in fuera]
    bl = int(bloque_len) if bloque_len is not None else bloque(p)
    _, res = mcs_mod.mcs_senales(sen[quedan], ventanas(p), block_len=bl, **{**al.MCS_KW, **kwargs})
    t = res.tabla.copy()
    t.index = t.index.astype(str)
    t["duplicado_de"] = None
    for c, origen in fuera.items():
        fila = t.loc[[origen]].copy()
        fila.index = [c]
        fila["duplicado_de"] = origen
        t = pd.concat([t, fila])
    t.index.name = "modelo"
    return t.assign(pista=p, bloque=bl, n_sesiones=len(sen))


# --------------------------------------------------------------------------- #
# Utilidad operativa (§5.1, sin peso)
# --------------------------------------------------------------------------- #
@contextmanager
def juez(pista: str) -> Iterator[None]:
    """Carga en el juez las labels de la pista (``configure_evaluation``) y las restaura al salir.

    ``maquina.signal_scores`` lee ``ev.CRISIS_WINDOWS`` y ``ev.FALSE_POSITIVE_WINDOWS`` globales."""
    from regimenes import evaluacion as ev
    from regimenes.benchmark import configure_evaluation

    guardado = (dict(ev.CRISIS_WINDOWS), dict(ev.FALSE_POSITIVE_WINDOWS), dict(ev.DRAWDOWN_TROUGHS))
    configure_evaluation(_pista(pista))
    try:
        yield
    finally:
        for destino, valor in zip((ev.CRISIS_WINDOWS, ev.FALSE_POSITIVE_WINDOWS, ev.DRAWDOWN_TROUGHS),
                                  guardado):
            destino.clear()
            destino.update(valor)


def utilidad(senales: pd.DataFrame, pista: str, tramo: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """``maquina.signal_scores`` (utilidad operativa de las fusiones antiguas D7+D8 y D2+D6) por columna.

    Como ``maquina.period_scorecard``: si en el tramo no cae ninguna ventana trampa (2013 está en
    H2 en A y en H1 en B), la utilidad se calcula sin ese término (``trampas_en_tramo=False``)."""
    from regimenes.fusion.maquina import operational_utility, signal_scores

    sen = _recortar(senales, tramo)
    filas = {}
    with juez(pista):
        for c in sen.columns:
            sc = signal_scores(sen[c].astype(bool))
            sc["trampas_en_tramo"] = bool(np.isfinite(sc["mean_trap_activation"]))
            if not sc["trampas_en_tramo"]:
                sc["operational_utility"] = operational_utility(
                    sc["mean_crisis_coverage"], sc["false_alarm_rate"], 0.0, sc["switching_rate"])
            filas[str(c)] = sc
    t = pd.DataFrame.from_dict(filas, orient="index")
    t.index.name = "id"
    return t


def limpiar_cache() -> None:
    """Vacía las cachés de paneles, señales y baselines (p. ej. si se rehacen los paneles)."""
    for f in (_paneles, _senales, _baselines):
        f.cache_clear()
    _N_STATES.clear()


__all__ = [
    "PISTAS", "BASELINES", "candidatos", "paneles", "n_states", "estado_crisis", "senales",
    "probabilidades", "indice", "ventanas", "mitades", "tramos", "nivel_deteccion",
    "metricas_senal", "ordenar", "evaluar", "mejor_individual", "baselines", "bloque",
    "deduplicar", "prioridad_por_defecto", "mcs", "juez", "utilidad", "limpiar_cache",
]
