"""
curvas.py — Detección frente a falsas alarmas y retraso a igual tasa de falsas
alarmas (A3; Hammond 2026, KLR 1998, Sarlin 2013).

Dada una puntuación diaria continua de un detector (mayor = más crisis; una
probabilidad sirve), se calibra el umbral para cada tasa objetivo de falsas
alarmas al año (``FA_OBJETIVO``: 0,5 / 1 / 2) y se reporta, en ese punto de
operación, la tasa de detección, el retraso mediano y el ratio ruido-señal.

Definiciones (coherentes con ``metricas.py``):

- **Señal**: ``score_t >= umbral_t`` (NaN → sin señal).
- **Episodio de alarma**: racha de >= ``min_run`` sesiones consecutivas con
  señal (el mismo gatillo de persistencia que ``lead_lag`` y la detección por
  evento). Las rachas más cortas no son alarmas.
- **Falsa alarma**: episodio que EMPIEZA fuera de todas las ventanas de crisis
  [pico, suelo] (y fuera de los ``margen_previo`` días previos a un pico, 0 por
  defecto). ``metricas.false_alarm_rate`` mide otra cosa (fracción de DÍAS
  marcados fuera de ventanas, 1 − precisión); para contar «alarmas al año» hace
  falta la unidad episodio.
- **FA/año** = falsas alarmas / años de calma, con años de calma = sesiones OOS
  fuera de las ventanas de crisis / ``dias_por_anio`` (solo en calma puede haber
  falsa alarma).
- **Detección**: ``event_detection_table`` (racha de ``min(min_run, n)``
  sesiones dentro de [pico, suelo]).
- **Retraso** por crisis: sesiones desde el pico hasta el inicio de la primera
  racha sostenida dentro de [pico, suelo], vía ``lead_lag`` con el suelo de la
  ventana como referencia y ``lookback`` = longitud de la ventana. También se da
  ``leadlag_suelo`` (= retraso − sesiones pico→suelo; negativo = antes del suelo).
- **Ratio ruido-señal** (KLR adaptado a días): FPR / TPR, con FPR = días de
  calma con señal / días de calma y TPR = días de ventana con señal / días de
  ventana. 1 = azar; < 1 = la señal se concentra en las crisis.

Modos de calibración:

- ``ex_post``: umbral único elegido sobre toda la OOS (el más bajo con FA/año
  ≤ objetivo sin que lo supere ningún umbral superior). Dispositivo descriptivo
  tipo ROC para comparar detectores a igual tasa de falsas alarmas; NO es
  operativo (usa el futuro).
- ``expansivo``: cada ``paso`` sesiones se recalibra con la misma regla sobre la
  historia ≤ t−1 y solo las crisis ya terminadas (suelo < t); el umbral es un
  cuantil del pasado y se aplica a las ``paso`` sesiones siguientes. Causal:
  truncar el futuro no cambia el umbral hasta t. Las primeras ``min_obs``
  sesiones son burn-in y no se evalúan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regimenes.evaluacion.metricas import event_detection_table, lead_lag

FA_OBJETIVO: tuple[float, ...] = (0.5, 1.0, 2.0)

CURVA_DEFAULTS: dict[str, float] = {
    "min_run": 3,          # = DETECTION_DEFAULTS["min_run"] (gatillo de 3 días)
    "dias_por_anio": 252,
    "n_grid": 200,         # cuantiles candidatos para el umbral
    "min_obs": 756,        # burn-in del modo expansivo (3 años)
    "paso": 252,           # recalibración anual en el modo expansivo
}


# --------------------------------------------------------------------------- #
# Episodios y falsas alarmas
# --------------------------------------------------------------------------- #
def _runs(flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inicios y finales (exclusivos) de las rachas de True."""
    steps = np.diff(np.concatenate(([0], np.asarray(flags, dtype=np.int8), [0])))
    return np.flatnonzero(steps == 1), np.flatnonzero(steps == -1)


def _window_mask(index: pd.DatetimeIndex, windows: dict[str, tuple[str, str]],
                 margen_previo: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(dentro de ventana, dentro de ventana o en los ``margen_previo`` días previos)."""
    inside = np.zeros(len(index), dtype=bool)
    for a, b in windows.values():
        inside |= (index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b))
    allowed = inside.copy()
    if margen_previo > 0:
        for a, _ in windows.values():
            pos = index.searchsorted(pd.Timestamp(a), side="left")
            allowed[max(0, pos - int(margen_previo)):pos] = True
    return inside, allowed


def _n_false_alarms(flags: np.ndarray, allowed: np.ndarray, min_run: int) -> int:
    starts, ends = _runs(flags)
    keep = (ends - starts) >= int(min_run)
    return int((keep & ~allowed[starts]).sum()) if starts.size else 0


def episodios_alarma(
    flags: pd.Series,
    windows: dict[str, tuple[str, str]],
    *,
    min_run: int = int(CURVA_DEFAULTS["min_run"]),
    margen_previo: int = 0,
) -> pd.DataFrame:
    """Episodios de alarma (rachas >= ``min_run``) con ``falsa`` = empieza en calma."""
    flags = pd.Series(flags).fillna(False).astype(bool)
    index = pd.DatetimeIndex(flags.index)
    _, allowed = _window_mask(index, windows, margen_previo)
    starts, ends = _runs(flags.to_numpy())
    keep = (ends - starts) >= int(min_run)
    starts, ends = starts[keep], ends[keep]
    return pd.DataFrame({
        "inicio": index[starts],
        "fin": index[ends - 1],
        "duracion": ends - starts,
        "falsa": ~allowed[starts],
    })


# --------------------------------------------------------------------------- #
# Métricas en un punto de operación
# --------------------------------------------------------------------------- #
def metricas_umbral(
    score: pd.Series,
    umbral: float | pd.Series,
    windows: dict[str, tuple[str, str]],
    *,
    min_run: int = int(CURVA_DEFAULTS["min_run"]),
    margen_previo: int = 0,
    fp_windows: dict[str, tuple[str, str]] | None = None,
    dias_por_anio: float = CURVA_DEFAULTS["dias_por_anio"],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Métricas de la señal ``score >= umbral`` (umbral fijo o serie por fecha).

    Devuelve (resumen, detalle por crisis con ``detectada``, ``retraso`` y
    ``leadlag_suelo``). Ver el docstring del módulo para las definiciones.
    """
    score = pd.Series(score, dtype=float)
    index = pd.DatetimeIndex(score.index)
    thr = umbral.reindex(index) if isinstance(umbral, pd.Series) else umbral
    flags = (score >= thr).fillna(False).astype(bool)
    values = flags.to_numpy()
    inside, allowed = _window_mask(index, windows, margen_previo)
    n_fa = _n_false_alarms(values, allowed, min_run)
    calm_years = (~inside).sum() / float(dias_por_anio)

    table = event_detection_table(flags, windows, min_run=min_run)
    retraso, leadlag = [], []
    for name, (a, b) in windows.items():
        pos = np.flatnonzero((index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b)))
        if not pos.size:
            retraso.append(np.nan)
            leadlag.append(np.nan)
            continue
        span = int(pos[-1] - pos[0])
        ll = lead_lag(flags.astype(float), {name: str(index[pos[-1]].date())},
                      threshold=0.5, persist=min(int(min_run), pos.size), lookback=span)[name]
        retraso.append(ll + span)
        leadlag.append(ll)
    table["retraso"] = retraso
    table["leadlag_suelo"] = leadlag

    evaluated = table["dias_oos"] > 0
    n_events = int(evaluated.sum())
    n_detected = int(table.loc[evaluated, "detectada"].sum())
    tpr = values[inside].mean() if inside.any() else np.nan
    fpr = values[~inside].mean() if (~inside).any() else np.nan
    out: dict[str, float] = {
        "fa_anio": n_fa / calm_years if calm_years > 0 else np.nan,
        "n_falsas_alarmas": n_fa,
        "anios_calma": calm_years,
        "n_eventos": n_events,
        "n_detectadas": n_detected,
        "tasa_deteccion": n_detected / n_events if n_events else np.nan,
        "retraso_mediano": float(table["retraso"].median()) if table["retraso"].notna().any() else np.nan,
        "leadlag_suelo_mediano": (
            float(table["leadlag_suelo"].median()) if table["leadlag_suelo"].notna().any() else np.nan
        ),
        "tpr_dias": tpr,
        "fpr_dias": fpr,
        "ratio_ruido_senal": fpr / tpr if tpr and tpr > 0 else np.nan,
    }
    if fp_windows:
        episodes = episodios_alarma(flags, windows, min_run=min_run, margen_previo=margen_previo)
        false_starts = pd.DatetimeIndex(episodes.loc[episodes["falsa"], "inicio"])
        out["n_fa_trampas"] = int(_window_mask(false_starts, fp_windows)[0].sum())
    return out, table


# --------------------------------------------------------------------------- #
# Calibración del umbral
# --------------------------------------------------------------------------- #
def _candidatos(values: np.ndarray, n_grid: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.array([np.inf])
    grid = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, int(n_grid))))
    return np.append(grid, np.inf)  # +inf = no marcar nunca (siempre factible)


def _fa_por_candidato(values: np.ndarray, cands: np.ndarray, inside: np.ndarray,
                      allowed: np.ndarray, min_run: int, dias_por_anio: float) -> np.ndarray:
    calm_years = (~inside).sum() / float(dias_por_anio)
    if calm_years <= 0:
        return np.full(len(cands), np.nan)
    with np.errstate(invalid="ignore"):
        return np.array([
            _n_false_alarms(values >= c, allowed, min_run) for c in cands
        ]) / calm_years


def _elegir(cands: np.ndarray, fa: np.ndarray, objetivo: float) -> float:
    """Bajando desde +inf, el último umbral antes de superar el objetivo: el más
    bajo tal que él y todos los superiores tienen FA/año ≤ objetivo. (El más bajo
    factible a secas no vale: un umbral bajísimo funde todo en un único episodio
    larguísimo y «cumple» con 1 falsa alarma.)"""
    bad = np.flatnonzero(~(fa <= objetivo + 1e-12))
    return float(cands[bad[-1] + 1]) if bad.size else float(cands[0])


def calibrar_umbral_ex_post(
    score: pd.Series,
    windows: dict[str, tuple[str, str]],
    fa_objetivo: float | tuple[float, ...] = FA_OBJETIVO,
    *,
    min_run: int = int(CURVA_DEFAULTS["min_run"]),
    margen_previo: int = 0,
    n_grid: int = int(CURVA_DEFAULTS["n_grid"]),
    dias_por_anio: float = CURVA_DEFAULTS["dias_por_anio"],
) -> dict[float, float]:
    """Umbral único sobre toda la OOS para cada objetivo (descriptivo, NO operativo).

    Candidatos: ``n_grid`` cuantiles de la puntuación más +inf. Como el nº de
    episodios no es monótono en el umbral (subirlo puede partir un episodio en
    dos, y bajarlo mucho los funde en uno), se baja desde +inf y se para justo
    antes del primer candidato que supera el objetivo.
    """
    score = pd.Series(score, dtype=float)
    values = score.to_numpy()
    targets = (fa_objetivo,) if np.isscalar(fa_objetivo) else tuple(fa_objetivo)
    inside, allowed = _window_mask(pd.DatetimeIndex(score.index), windows, margen_previo)
    cands = _candidatos(values, n_grid)
    fa = _fa_por_candidato(values, cands, inside, allowed, min_run, dias_por_anio)
    return {float(t): _elegir(cands, fa, float(t)) for t in targets}


def umbral_expansivo(
    score: pd.Series,
    windows: dict[str, tuple[str, str]],
    fa_objetivo: float | tuple[float, ...] = FA_OBJETIVO,
    *,
    min_run: int = int(CURVA_DEFAULTS["min_run"]),
    margen_previo: int = 0,
    n_grid: int = int(CURVA_DEFAULTS["n_grid"]),
    min_obs: int = int(CURVA_DEFAULTS["min_obs"]),
    paso: int = int(CURVA_DEFAULTS["paso"]),
    dias_por_anio: float = CURVA_DEFAULTS["dias_por_anio"],
) -> pd.DataFrame:
    """Umbral causal por fecha (una columna por objetivo; NaN en el burn-in).

    En cada posición de recalibración p = ``min_obs``, ``min_obs + paso``, … se
    aplica la regla ex post a las sesiones [0, p) con las ventanas de las crisis
    cuyo suelo es anterior a la fecha p (las que siguen abiertas cuentan como
    calma: un operador en tiempo real aún no las conoce). El umbral vale para
    [p, p + paso).
    """
    score = pd.Series(score, dtype=float)
    index = pd.DatetimeIndex(score.index)
    values = score.to_numpy()
    targets = (fa_objetivo,) if np.isscalar(fa_objetivo) else tuple(fa_objetivo)
    out = np.full((len(values), len(targets)), np.nan)
    for p in range(int(min_obs), len(values), max(1, int(paso))):
        known = {k: v for k, v in windows.items() if pd.Timestamp(v[1]) < index[p]}
        inside, allowed = _window_mask(index[:p], known, margen_previo)
        past = values[:p]
        cands = _candidatos(past, n_grid)
        fa = _fa_por_candidato(past, cands, inside, allowed, min_run, dias_por_anio)
        for j, t in enumerate(targets):
            out[p:p + int(paso), j] = _elegir(cands, fa, float(t))
    return pd.DataFrame(out, index=index, columns=[float(t) for t in targets])


# --------------------------------------------------------------------------- #
# Curva
# --------------------------------------------------------------------------- #
def curva_deteccion(
    score: pd.Series,
    windows: dict[str, tuple[str, str]],
    *,
    oos_index: pd.DatetimeIndex | None = None,
    fa_objetivo: tuple[float, ...] = FA_OBJETIVO,
    modo: str = "ex_post",
    min_run: int = int(CURVA_DEFAULTS["min_run"]),
    margen_previo: int = 0,
    fp_windows: dict[str, tuple[str, str]] | None = None,
    n_grid: int = int(CURVA_DEFAULTS["n_grid"]),
    min_obs: int = int(CURVA_DEFAULTS["min_obs"]),
    paso: int = int(CURVA_DEFAULTS["paso"]),
    dias_por_anio: float = CURVA_DEFAULTS["dias_por_anio"],
) -> pd.DataFrame:
    """Tabla (objetivo de FA/año → umbral, FA/año realizadas, detección, retraso
    mediano, ratio ruido-señal) para una puntuación diaria.

    ``score`` se reindexa a ``oos_index`` si se pasa. ``modo``: ``ex_post`` o
    ``expansivo`` (ver el docstring del módulo). En ``expansivo`` solo se evalúan
    las sesiones con umbral definido (tras el burn-in) y ``umbral`` es la mediana
    de los umbrales usados. El detalle por crisis queda en ``attrs["detalle"]``
    ({objetivo: DataFrame}).
    """
    score = pd.Series(score, dtype=float).sort_index()
    if oos_index is not None:
        score = score.reindex(pd.DatetimeIndex(oos_index))
    common = dict(min_run=min_run, margen_previo=margen_previo, n_grid=n_grid,
                  dias_por_anio=dias_por_anio)
    if modo == "ex_post":
        thresholds = calibrar_umbral_ex_post(score, windows, fa_objetivo, **common)
    elif modo == "expansivo":
        panel = umbral_expansivo(score, windows, fa_objetivo, min_obs=min_obs, paso=paso, **common)
    else:
        raise ValueError(f"modo desconocido: {modo!r} (ex_post | expansivo)")
    rows, detail = [], {}
    for t in (fa_objetivo,) if np.isscalar(fa_objetivo) else fa_objetivo:
        t = float(t)
        if modo == "ex_post":
            thr, sc, burn = thresholds[t], score, 0
        else:
            col = panel[t]
            sc, thr = score[col.notna()], col.dropna()
            burn = int(col.isna().sum())
        summary, table = metricas_umbral(
            sc, thr, windows, min_run=min_run, margen_previo=margen_previo,
            fp_windows=fp_windows, dias_por_anio=dias_por_anio,
        )
        umbral = float(thr) if np.isscalar(thr) else (float(thr.median()) if len(thr) else np.nan)
        rows.append({"modo": modo, "fa_objetivo": t, "umbral": umbral,
                     "dias_burn_in": burn, **summary})
        detail[t] = table
    out = pd.DataFrame(rows)
    out.attrs["detalle"] = detail
    return out
