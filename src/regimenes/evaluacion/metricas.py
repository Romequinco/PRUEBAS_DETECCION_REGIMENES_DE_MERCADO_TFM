"""
metricas.py — Métricas causales del juez común (antes parte de src/evaluation.py).

Contiene las ventanas de referencia de eventos (fuente de verdad única,
configurables por pista con :func:`configure_event_windows`), las métricas
individuales que consume ``evaluate`` y las métricas de detección por evento
(``det_*``) que ``regimenes.benchmark.ejecucion.run_one`` escribe en el CSV y que
usa el ranking ADR-003 (antes en src/benchmark.py).

Ventanas de crisis y falsos positivos conocidos (fuente de verdad única)
------------------------------------------------------------------------
Se usan para medir cobertura de crisis y tasa de falsas alarmas. Fechas
aproximadas de mercado (S&P 500), ajustables en FASE 1 con el EDA.

Los tres diccionarios (``CRISIS_WINDOWS``, ``FALSE_POSITIVE_WINDOWS``,
``DRAWDOWN_TROUGHS``) son objetos ÚNICOS: ``configure_event_windows`` los muta
in situ y el resto de módulos (walk_forward, fusion, viz, ranking) los importan
de aquí por referencia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Ventanas de referencia (event study). Se afinan en FASE 1 con el EDA.
# --------------------------------------------------------------------------- #
CRISIS_WINDOWS: dict[str, tuple[str, str]] = {
    "GFC_2008": ("2008-09-01", "2009-03-31"),       # Lehman -> suelo
    "EuroDebt_2011": ("2011-07-01", "2011-10-31"),  # crisis soberana europea
    "COVID_2020": ("2020-02-20", "2020-04-30"),     # crash COVID
    "Inflation_2022": ("2022-01-01", "2022-10-31"), # bear market tipos/inflación
}

# Episodios que NO son crisis sistémicas: el detector NO debería dispararse de
# forma sostenida aquí. Sirven para medir falsos positivos.
FALSE_POSITIVE_WINDOWS: dict[str, tuple[str, str]] = {
    "TaperTantrum_2013": ("2013-05-01", "2013-09-30"),
    "Selloff_Q4_2018": ("2018-10-01", "2018-12-31"),
}

# Picos (fondo) de drawdown del S&P 500 para medir lead/lag de la señal.
# CALCULADOS en FASE 1 desde la serie real del S&P 500 (no a mano):
# fecha del mínimo del drawdown (precio/máx_expanding - 1) dentro de cada
# episodio. Drawdown alcanzado entre paréntesis. Ver docs/memory/01_data_and_eda.md
# y el notebook 00_eda de la Capa 1 (celda de cálculo de troughs; tag capa1-final).
DRAWDOWN_TROUGHS: dict[str, str] = {
    "GFC_2008": "2009-03-09",       # -56.8%
    "EuroDebt_2011": "2011-10-03",  # -29.8%
    "COVID_2020": "2020-03-23",     # -33.9%
    "Inflation_2022": "2022-10-12", # -25.4%
}
# Referencia fuera de la ventana común del set ampliado (datos completos desde
# 2007-04-11, por inicio de HYG): el crash DotCom tocó suelo el 2002-10-09
# (-49.1%). Solo evaluable por detectores que usen un subconjunto de features con
# histórico más largo (p. ej. solo S&P 500 + VIX, disponibles desde 1990).


def configure_event_windows(
    crisis_windows: dict[str, tuple[str, str]],
    false_positive_windows: dict[str, tuple[str, str]],
    drawdown_troughs: dict[str, str],
) -> None:
    """Configura las etiquetas de evaluación para un benchmark concreto.

    La Capa 1 usaba cuatro episodios fijos. La Fase D v2 tiene dos pistas con
    catálogos diferentes, congelados en ``configs/benchmark_spec.yaml``. Esta función
    permite cambiar únicamente las *labels* del juez antes de ejecutar cada pista;
    no altera features, ventanas de entrenamiento ni parámetros del detector.
    """
    if not crisis_windows:
        raise ValueError("crisis_windows no puede estar vacío.")
    if not false_positive_windows:
        raise ValueError("false_positive_windows no puede estar vacío.")
    missing = set(crisis_windows) - set(drawdown_troughs)
    if missing:
        raise ValueError(f"Faltan troughs para estas crisis: {sorted(missing)}")

    def _windows(values: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
        out: dict[str, tuple[str, str]] = {}
        for name, bounds in values.items():
            if len(bounds) != 2:
                raise ValueError(f"Ventana inválida para {name!r}: {bounds!r}")
            start, end = map(str, bounds)
            if pd.Timestamp(start) > pd.Timestamp(end):
                raise ValueError(f"Ventana invertida para {name!r}: {bounds!r}")
            out[str(name)] = (start, end)
        return out

    new_crisis_windows = _windows(crisis_windows)
    new_false_positive_windows = _windows(false_positive_windows)
    new_drawdown_troughs = {
        str(name): str(drawdown_troughs[name]) for name in new_crisis_windows
    }

    # Se mutan los diccionarios en lugar de reasignarlos. ``regimenes.viz`` y otros
    # consumidores históricos importan estas constantes por referencia; una
    # reasignación dejaría esos módulos apuntando silenciosamente a las ventanas
    # antiguas.
    CRISIS_WINDOWS.clear()
    CRISIS_WINDOWS.update(new_crisis_windows)
    FALSE_POSITIVE_WINDOWS.clear()
    FALSE_POSITIVE_WINDOWS.update(new_false_positive_windows)
    DRAWDOWN_TROUGHS.clear()
    DRAWDOWN_TROUGHS.update(new_drawdown_troughs)


def _in_any_window(dates: pd.DatetimeIndex, windows: dict[str, tuple[str, str]]) -> np.ndarray:
    """Máscara booleana: True si la fecha cae dentro de alguna ventana [a, b]."""
    mask = np.zeros(len(dates), dtype=bool)
    for a, b in windows.values():
        mask |= (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))
    return mask


# --------------------------------------------------------------------------- #
# Métricas individuales (todas causales: operan sobre etiquetas walk-forward)
# --------------------------------------------------------------------------- #
def crisis_coverage(
    states: pd.Series, crisis_state: int, windows: dict[str, tuple[str, str]] = None
) -> dict[str, float]:
    """% de días etiquetados como 'crisis' dentro de cada ventana de crisis conocida.

    Mide sensibilidad: idealmente alto (cercano a 1) en 2008/2011/2020/2022. Si la
    ventana queda fuera del rango out-of-sample del detector, devuelve NaN para esa
    ventana (no se penaliza lo que no se pudo ver).
    """
    windows = windows or CRISIS_WINDOWS
    out: dict[str, float] = {}
    for name, (a, b) in windows.items():
        seg = states.loc[(states.index >= pd.Timestamp(a)) & (states.index <= pd.Timestamp(b))]
        out[name] = float((seg == crisis_state).mean()) if len(seg) else float("nan")
    return out


def false_alarm_in_windows(
    states: pd.Series, crisis_state: int, windows: dict[str, tuple[str, str]] = None
) -> dict[str, float]:
    """% de días 'crisis' dentro de ventanas que NO son crisis (2013, 2018).

    Mide especificidad en episodios trampa: idealmente bajo.
    """
    windows = windows or FALSE_POSITIVE_WINDOWS
    out: dict[str, float] = {}
    for name, (a, b) in windows.items():
        seg = states.loc[(states.index >= pd.Timestamp(a)) & (states.index <= pd.Timestamp(b))]
        out[name] = float((seg == crisis_state).mean()) if len(seg) else float("nan")
    return out


def false_alarm_rate(
    states: pd.Series, crisis_state: int, crisis_windows: dict[str, tuple[str, str]] = None
) -> float:
    """Tasa global de falsas alarmas: fracción de días marcados 'crisis' que caen
    FUERA de todas las ventanas de crisis conocidas.

    Aproxima 1 - precisión usando las ventanas conocidas como ground truth laxo.
    NaN si el detector nunca marca crisis (denominador 0).
    """
    crisis_windows = crisis_windows or CRISIS_WINDOWS
    is_crisis = (states == crisis_state).values
    n_crisis = int(is_crisis.sum())
    if n_crisis == 0:
        return float("nan")
    in_crisis_win = _in_any_window(states.index, crisis_windows)
    false_alarms = int((is_crisis & ~in_crisis_win).sum())
    return false_alarms / n_crisis


def _first_sustained(mask: np.ndarray, persist: int) -> int | None:
    """Índice del inicio de la PRIMERA racha de >= `persist` True consecutivos."""
    if persist <= 1:
        nz = np.flatnonzero(mask)
        return int(nz[0]) if nz.size else None
    count = 0
    for i, v in enumerate(mask):
        count = count + 1 if v else 0
        if count >= persist:
            return i - persist + 1
    return None


def lead_lag(
    p_crisis: pd.Series,
    troughs: dict[str, str] = None,
    threshold: float = 0.5,
    persist: int = 3,
    lookback: int = 252,
) -> dict[str, float]:
    """Días de adelanto/retraso entre la señal SOSTENIDA de crisis y el suelo del
    drawdown del S&P 500, por evento.

    Para cada trough busca, en los `lookback` días previos al suelo, el primer día
    en que `p_crisis` cruza `threshold` y SE MANTIENE >= `persist` días
    consecutivos (coherente con el gatillo de cambio de régimen del TFM: 3 días).
    Exigir persistencia evita premiar el flickering: un detector ruidoso que cruza
    el umbral un día suelto por azar ya NO cuenta como "anticipador".

    Devuelve (fecha_señal_sostenida - fecha_trough) en días de trading (posiciones
    del índice OOS): negativo = la señal ANTICIPA el suelo, positivo = va por
    detrás. NaN si el trough cae fuera del rango OOS o no hay señal sostenida en la
    ventana previa.
    """
    troughs = troughs or DRAWDOWN_TROUGHS
    out: dict[str, float] = {}
    idx = p_crisis.index
    for name, tdate in troughs.items():
        T = pd.Timestamp(tdate)
        if T < idx.min() or T > idx.max():
            out[name] = float("nan")
            continue
        # posición del trough (o el día OOS inmediatamente anterior)
        pos_T = idx.searchsorted(T, side="right") - 1
        if pos_T < 0:
            out[name] = float("nan")
            continue
        lo = max(0, pos_T - lookback)
        window = p_crisis.iloc[lo:pos_T + 1]
        start = _first_sustained((window.values >= threshold), persist)
        if start is None:
            out[name] = float("nan")
            continue
        pos_signal = idx.searchsorted(window.index[start], side="left")
        out[name] = float(pos_signal - pos_T)  # negativo = anticipa
    return out


def switching_rate(states: pd.Series) -> float:
    """Frecuencia de conmutación = nº de cambios de estado / nº de días.

    Penaliza el 'flickering' (regímenes que parpadean día a día). Más bajo =
    más persistente/estable.
    """
    if len(states) < 2:
        return float("nan")
    changes = int((states.values[1:] != states.values[:-1]).sum())
    return changes / len(states)


def mean_regime_duration(states: pd.Series) -> float:
    """Duración media (en días) de los episodios de régimen. Inverso del flicker."""
    if len(states) == 0:
        return float("nan")
    v = states.values
    n_runs = 1 + int((v[1:] != v[:-1]).sum())
    return len(v) / n_runs


def label_stability(
    panel: pd.DataFrame,
) -> float:
    """Estabilidad de etiquetas bajo walk-forward.

    Mide cuánto cambian las etiquetas asignadas a una misma fecha cuando el
    modelo se reentrena en folds sucesivos (idealmente la etiqueta de una fecha
    no debería bailar al añadir datos posteriores). Devuelve una métrica de
    concordancia en [0, 1] (1 = totalmente estable).

    Parameters
    ----------
    panel : pd.DataFrame
        Etiquetas por fecha (filas) y fold (columnas), de las reestimaciones
        sucesivas del walk-forward. Para cada fecha con >=2 reestimaciones se
        calcula la fracción de la moda (acuerdo); se promedia sobre esas fechas.
    """
    if panel is None or panel.empty:
        return float("nan")
    agreements = []
    for _, rowvals in panel.iterrows():
        vals = rowvals.dropna().values
        if len(vals) < 2:
            continue
        _, counts = np.unique(vals, return_counts=True)
        agreements.append(counts.max() / len(vals))
    return float(np.mean(agreements)) if agreements else float("nan")


def silhouette_states(X: pd.DataFrame, states: pd.Series, *, max_n: int = 4000,
                      random_state: int = 42) -> float:
    """Coeficiente de silueta medio de las etiquetas de régimen en el espacio de features.

    Interpreta los regímenes como un "clustering" sobre las features causales y mide
    cuán separados/compactos quedan (en [-1, 1]; >0 = regímenes bien separados). Es
    una métrica de calidad de partición ÚTIL sobre todo para los detectores de
    clustering (D3 GMM, D9 jump, D12 AE/PCA), pero se computa para cualquiera que
    exponga features alineadas. Submuestrea a `max_n` filas por coste (la silueta es
    O(n²)). Devuelve NaN si hay <2 estados observados o <3 muestras por estado.
    """
    from sklearn.metrics import silhouette_score

    idx = X.index.intersection(states.index)
    if len(idx) < 10:
        return float("nan")
    Xa = X.loc[idx]
    s = states.loc[idx].astype(int)
    # quitar filas con NaN en features (la silueta no las admite)
    ok = ~Xa.isna().any(axis=1)
    Xa, s = Xa[ok], s[ok]
    labels, counts = np.unique(s.values, return_counts=True)
    if len(labels) < 2 or counts.min() < 3:
        return float("nan")
    if len(Xa) > max_n:
        rng = np.random.default_rng(random_state)
        sel = rng.choice(len(Xa), size=max_n, replace=False)
        Xa, s = Xa.iloc[sel], s.iloc[sel]
        if len(np.unique(s.values)) < 2:
            return float("nan")
    try:
        return float(silhouette_score(Xa.values, s.values))
    except Exception:  # noqa: BLE001
        return float("nan")


def block_bootstrap_coverage_ci(
    states: pd.Series, crisis_state: int,
    windows: dict[str, tuple[str, str]] = None, *,
    n_boot: int = 500, block: int = 5, alpha: float = 0.05, random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    """Intervalo de confianza (bootstrap por bloques) de la cobertura de crisis por ventana.

    Para cada ventana de crisis remuestrea el indicador diario (estado==crisis) en
    BLOQUES de `block` días —preservando la autocorrelación de los episodios— y toma
    el percentil [alpha/2, 1-alpha/2] de la cobertura remuestreada. Responde de forma
    directa al trabajo futuro "B1: bootstrap por bloques" declarado en el informe.

    IMPORTANTE (honestidad): el IC cuantifica la incertidumbre DENTRO de la ventana
    (muestreo de días autocorrelados), NO entre eventos. Con n≈4 crisis no sustituye
    a un test de significancia entre eventos; es una banda de precisión intra-evento.
    Devuelve {ventana: (lo, hi)}; (NaN, NaN) si la ventana cae fuera del rango OOS.
    """
    windows = windows or CRISIS_WINDOWS
    rng = np.random.default_rng(random_state)
    out: dict[str, tuple[float, float]] = {}
    for name, (a, b) in windows.items():
        seg = states.loc[(states.index >= pd.Timestamp(a)) & (states.index <= pd.Timestamp(b))]
        ind = (seg == crisis_state).astype(int).values
        n = len(ind)
        if n == 0:
            out[name] = (float("nan"), float("nan"))
            continue
        n_blocks = int(np.ceil(n / block))
        max_start = max(1, n - block + 1)
        boots = np.empty(n_boot)
        for it in range(n_boot):
            starts = rng.integers(0, max_start, size=n_blocks)
            sample = np.concatenate([ind[s:s + block] for s in starts])[:n]
            boots[it] = sample.mean()
        lo = float(np.quantile(boots, alpha / 2))
        hi = float(np.quantile(boots, 1 - alpha / 2))
        out[name] = (lo, hi)
    return out


# --------------------------------------------------------------------------- #
# Detección por evento (ADR-003; antes en src/benchmark.py). Columnas ``det_*``
# que run_one escribe en el CSV y que usa ``regimenes.evaluacion.ranking``.
# --------------------------------------------------------------------------- #
def _longest_true_run(values: np.ndarray) -> int:
    """Longitud de la racha más larga de True consecutivos."""
    values = np.asarray(values, dtype=bool)
    if not values.any():
        return 0
    padded = np.concatenate(([0], values.astype(np.int8), [0]))
    steps = np.diff(padded)
    starts = np.flatnonzero(steps == 1)
    ends = np.flatnonzero(steps == -1)
    return int((ends - starts).max())


def event_detection_table(
    crisis_flags: pd.Series,
    windows: dict[str, tuple[str, str]],
    *,
    min_run: int = 3,
) -> pd.DataFrame:
    """Detección por evento a partir de la señal OOS de crisis (bool por fecha).

    Por crisis: sesiones OOS dentro de ``[pico, suelo]``, sesiones marcadas,
    racha máxima y ``detectada`` (1/0; NaN si la ventana cae fuera del OOS).
    Detectada = racha de ``min(min_run, dias_oos)`` sesiones consecutivas.
    """
    flags = pd.Series(crisis_flags).astype(bool)
    index = pd.DatetimeIndex(flags.index)
    values = flags.to_numpy()
    rows = []
    for name, (start, end) in windows.items():
        mask = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        segment = values[mask]
        n_days = int(mask.sum())
        longest = _longest_true_run(segment)
        rows.append({
            "crisis": name,
            "dias_oos": n_days,
            "dias_marcados": int(segment.sum()),
            "racha_max": longest,
            "detectada": float(longest >= min(int(min_run), n_days)) if n_days else np.nan,
        })
    return pd.DataFrame(rows).set_index("crisis")


def detection_summary(
    crisis_flags: pd.Series,
    windows: dict[str, tuple[str, str]],
    *,
    min_run: int = 3,
) -> dict[str, float]:
    """Métricas de detección (columnas ``det_*``) de una señal OOS de crisis.

    - ``det_event_recall``: eventos detectados / eventos con días OOS.
    - ``det_precision``: días marcados dentro de alguna ventana / días marcados
      (= ``1 − false_alarm_rate``; NaN si nunca marca).
    - ``det_base_rate``: fracción de días OOS dentro de ventanas (= precisión de
      marcar crisis siempre); ``det_lift_precision`` = precisión / tasa base.
    - ``det_marked_rate``: fracción de días OOS marcados; ``det_day_recall``:
      fracción de días de ventana marcados (cobertura ponderada por duración).
    - ``det_mean_crisis_run``: duración media (sesiones) de los episodios de crisis.
    - ``det_ev_<crisis>``: 1/0/NaN por evento.
    """
    flags = pd.Series(crisis_flags).astype(bool)
    index = pd.DatetimeIndex(flags.index)
    values = flags.to_numpy()
    table = event_detection_table(flags, windows, min_run=min_run)
    evaluated = table["dias_oos"] > 0
    inside = np.zeros(len(values), dtype=bool)
    for start, end in windows.values():
        inside |= (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
    n_days = len(values)
    n_marked = int(values.sum())
    true_positive = int((values & inside).sum())
    base_rate = float(inside.mean()) if n_days else np.nan
    precision = true_positive / n_marked if n_marked else np.nan
    n_events = int(evaluated.sum())
    n_detected = int(table.loc[evaluated, "detectada"].sum())
    n_episodes = int(np.sum(np.diff(np.concatenate(([0], values.astype(np.int8)))) == 1))
    out: dict[str, float] = {
        "det_min_run": int(min_run),
        "det_n_eventos": n_events,
        "det_n_detectados": n_detected,
        "det_event_recall": n_detected / n_events if n_events else np.nan,
        "det_precision": precision,
        "det_base_rate": base_rate,
        "det_lift_precision": (
            precision / base_rate if n_marked and base_rate and base_rate > 0 else np.nan
        ),
        "det_marked_rate": n_marked / n_days if n_days else np.nan,
        "det_day_recall": true_positive / int(inside.sum()) if inside.any() else np.nan,
        "det_mean_crisis_run": n_marked / n_episodes if n_episodes else np.nan,
    }
    for name, value in table["detectada"].items():
        out[f"det_ev_{name}"] = value
    return out
