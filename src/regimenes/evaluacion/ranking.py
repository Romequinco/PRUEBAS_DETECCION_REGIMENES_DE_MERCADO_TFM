"""Criterio de ranking de detección (ADR-003) y ranking descriptivo heredado.

:func:`rank_detection` es el ranking principal;
:func:`rank_within_track` (``rank_medio``) se conserva solo para comparar. Incluye
las líneas base triviales y el nulo de azar persistente.

Depende de ``regimenes.benchmark.cache`` (spec congelada, rutas de paneles) y de
``regimenes.evaluacion.metricas``; ``regimenes.evaluacion`` no lo importa en su
``__init__`` para no crear el ciclo ranking → benchmark → evaluacion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from regimenes.benchmark.cache import DEFAULT_OUTPUT, _panel_path, load_benchmark_spec
from regimenes.evaluacion import metricas as ev
from regimenes.evaluacion.metricas import detection_summary


def add_comparison_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Añade agregados comparables sin mezclar pistas.

    - ``mean_crisis_coverage``: media NO ponderada, por evento, de ``cov_<crisis>``
      (fracción de días OOS de la ventana [pico, suelo] marcados como crisis). Las
      crisis sin días OOS en esa pista son NaN y se omiten; una ventana solo
      parcialmente OOS (p. ej. ``bear_1969_70`` en A: 11 de 370 días) cuenta como
      un evento completo. Un volmageddon de 10 días pesa igual que la GFC.
    - ``mean_trap_activation``: media de ``fa_<trampa>`` (fracción de días de la
      trampa marcados como crisis; menor es mejor).
    """
    out = metrics.copy()
    coverage = [
        c for c in out
        if c.startswith("cov_") and not c.endswith(("_lo", "_hi"))
    ]
    traps = [c for c in out if c.startswith("fa_")]
    out["mean_crisis_coverage"] = out[coverage].mean(axis=1, skipna=True)
    out["mean_trap_activation"] = out[traps].mean(axis=1, skipna=True)
    return out


def rank_within_track(metrics: pd.DataFrame) -> pd.DataFrame:
    """[LEGACY] Ranking descriptivo sobre cinco ejes, calculado dentro de cada pista.

    Desde ADR-003 el ranking principal es :func:`rank_detection`; este se
    conserva (``rank_medio``/``puesto_pista`` de ``metrics_master_v2.csv``, y
    ``rank_medio_legacy``/``puesto_legacy`` en ``ranking_v2.csv``) solo para
    comparar con el criterio anterior.

    Cada eje se ordena por separado (1 = mejor); los empates reciben el rango
    medio (``method="average"``) y los NaN van al fondo (p. ej. un
    ``false_alarm_rate`` NaN porque el detector nunca marca crisis).
    ``rank_medio`` es la media simple de los cinco rangos y ``puesto_pista`` su
    rango con ``method="min"`` (los empates comparten puesto).

    Limitación conocida: tres de los cinco ejes (trampas, switching y
    estabilidad) premian la inactividad. Un detector trivial que nunca o siempre
    marca crisis quedaría en mitad de la tabla; el notebook 05 lo muestra con
    líneas base triviales. Es un mapa descriptivo, no una función de utilidad.
    """
    out = add_comparison_metrics(metrics)
    directions = {
        "mean_crisis_coverage": False,
        "false_alarm_rate": True,
        "mean_trap_activation": True,
        "switching_rate": True,
        "label_stability": False,
    }
    # ``mean_regime_duration`` se conserva en la tabla como diagnóstico, pero no
    # entra en el promedio: para una secuencia fija es una transformación monótona
    # del número de cambios y duplicaría exactamente el peso de ``switching_rate``.
    rank_cols = []
    for metric, ascending in directions.items():
        col = f"rank_{metric}"
        out[col] = out.groupby("pista")[metric].rank(
            ascending=ascending, method="average", na_option="bottom"
        )
        rank_cols.append(col)
    out["rank_medio"] = out[rank_cols].mean(axis=1)
    out["puesto_pista"] = out.groupby("pista")["rank_medio"].rank(method="min")
    return out.sort_values(["pista", "puesto_pista", "id"])


# --------------------------------------------------------------------------- #
# Criterio de ranking de detección (ADR-003)
# --------------------------------------------------------------------------- #
# Parámetros explícitos del criterio. Cualquier cambio debe registrarse en la ADR.
DETECTION_DEFAULTS: dict[str, float] = {
    # Un evento cuenta como detectado si hay >= ``min_run`` sesiones OOS
    # CONSECUTIVAS marcadas como crisis dentro de su ventana [pico, suelo]
    # (o todas, si la ventana tiene menos sesiones OOS). 3 = el gatillo de
    # persistencia de ``regimenes.evaluacion.metricas.lead_lag`` y de la regla de cambio del TFM.
    "min_run": 3,
    # F-beta entre recall por evento y precisión diaria (beta > 1 prima recall).
    "beta": 1.0,
    # Restricción de falsas alarmas: precisión / tasa base > ``lift_min``
    # (equivale a FAR < FAR de marcar crisis siempre).
    "lift_min": 1.0,
    # Filtros mínimos de persistencia ("switching absurdo"): duración media de
    # régimen >= 1 semana hábil y episodios de crisis de al menos ``min_run``
    # sesiones de media (más cortos que el propio gatillo de detección = parpadeo).
    "min_mean_duration": 5.0,
    "min_crisis_run": 3.0,
    # Empate si el score coincide a estos decimales → desempate por persistencia.
    "score_decimals": 3,
}

DETECTION_LEVELS: dict[int, str] = {
    0: "elegible",
    1: "precision_no_supera_azar",
    2: "parpadeo",
    3: "degenerado",
}


def track_crisis_windows(track: str, benchmark: dict | None = None) -> dict[str, tuple[str, str]]:
    """Ventanas ``[pico, suelo]`` congeladas de una pista (las del juez)."""
    benchmark = benchmark or load_benchmark_spec()
    return {
        name: tuple(map(str, bounds))
        for name, bounds in benchmark["crisis_windows"][f"pista_{track.upper()}"].items()
    }


def track_false_positive_windows(benchmark: dict | None = None) -> dict[str, tuple[str, str]]:
    benchmark = benchmark or load_benchmark_spec()
    return {
        name: tuple(map(str, info["ventana"]))
        for name, info in benchmark["false_positive_windows"].items()
        if isinstance(info, dict) and info.get("en_catalogo") is False and "ventana" in info
    }


def track_oos_index(
    track: str,
    metrics: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT,
) -> tuple[pd.DatetimeIndex, bool]:
    """Índice OOS común de una pista y si es exacto.

    Exacto: el de cualquier panel OOS de la pista cuyo largo coincide con
    ``n_oos`` (el gate de 05 exige el mismo índice a los 12 detectores).
    Aproximado: sesiones lunes–viernes entre ``oos_start`` y ``oos_end``.
    """
    track = track.upper()
    part = metrics[metrics["pista"].astype(str).str.upper() == track]
    n_oos = int(part["n_oos"].iloc[0]) if "n_oos" in part and len(part) else None
    for path in sorted((Path(output_dir) / "panels").glob(f"{track}_D*.parquet")):
        try:
            index = pd.DatetimeIndex(pd.read_parquet(path, columns=["state"]).index)
        except (OSError, ValueError, KeyError):
            continue
        if n_oos is None or len(index) == n_oos:
            return index, True
    if not len(part):
        raise KeyError(f"No hay métricas de la pista {track}.")
    return pd.bdate_range(part["oos_start"].iloc[0], part["oos_end"].iloc[0]), False


def _detection_from_coverage(
    row: pd.Series,
    oos_index: pd.DatetimeIndex,
    windows: dict[str, tuple[str, str]],
    min_run: int,
) -> dict[str, float]:
    """Aproximación de ``detection_summary`` desde las columnas del CSV.

    Sin panel OOS no se conoce la racha máxima: se exige ``round(cov·n) >=
    min_run`` sesiones marcadas en la ventana, no necesariamente consecutivas
    (cota superior del criterio exacto; coincide con él en 13/14 paneles
    disponibles en la revisión de 2026-09-29).
    """
    inside = np.zeros(len(oos_index), dtype=bool)
    n_days = len(oos_index)
    true_positive = 0.0
    out: dict[str, float] = {"det_min_run": int(min_run)}
    detected: list[float] = []
    for name, (start, end) in windows.items():
        mask = (oos_index >= pd.Timestamp(start)) & (oos_index <= pd.Timestamp(end))
        inside |= mask
        n_window = int(mask.sum())
        coverage = row.get(f"cov_{name}", np.nan)
        if n_window == 0 or pd.isna(coverage):
            out[f"det_ev_{name}"] = np.nan
            continue
        marked = round(float(coverage) * n_window)
        true_positive += float(coverage) * n_window
        flag = float(marked >= min(int(min_run), n_window))
        out[f"det_ev_{name}"] = flag
        detected.append(flag)
    far = row.get("false_alarm_rate", np.nan)
    precision = 1.0 - float(far) if pd.notna(far) else np.nan
    base_rate = float(inside.mean()) if n_days else np.nan
    if pd.isna(far):
        marked_rate = 0.0  # metricas.false_alarm_rate es NaN solo si nunca marca
    elif precision > 0:
        marked_rate = min(1.0, true_positive / precision / n_days)
    else:
        marked_rate = np.nan
    out.update({
        "det_n_eventos": len(detected),
        "det_n_detectados": int(np.sum(detected)),
        "det_event_recall": float(np.mean(detected)) if detected else np.nan,
        "det_precision": precision,
        "det_base_rate": base_rate,
        "det_lift_precision": precision / base_rate if pd.notna(precision) and base_rate else np.nan,
        "det_marked_rate": marked_rate,
        "det_day_recall": true_positive / inside.sum() if inside.any() else np.nan,
        # nº de episodios de crisis ≈ cambios/2 (exacto con 2 estados; con K > 2
        # los cambios entre estados no-crisis lo hacen conservador).
        "det_mean_crisis_run": _approx_crisis_run(marked_rate, row.get("switching_rate", np.nan), n_days),
    })
    return out


def _approx_crisis_run(marked_rate: float, switching_rate: float, n_days: int) -> float:
    if pd.isna(marked_rate) or marked_rate <= 0:
        return np.nan
    if pd.isna(switching_rate):
        return np.nan
    if switching_rate <= 0:
        return float(marked_rate * n_days)
    return float(max(1.0, 2.0 * marked_rate / switching_rate))


def add_detection_metrics(
    metrics: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    min_run: int = int(DETECTION_DEFAULTS["min_run"]),
    benchmark: dict | None = None,
) -> pd.DataFrame:
    """Completa las columnas ``det_*`` y ``fuente_deteccion`` fila a fila.

    Prioridad de la fuente: (1) ``csv`` si el CSV ya trae ``det_*`` con el mismo
    ``min_run`` (benchmark re-ejecutado con este módulo); (2) ``panel`` si
    existe ``panels/<P>_<D>.parquet`` (exacto; crisis = ``n_states − 1``);
    (3) ``cobertura_aprox`` desde ``cov_*`` y ``false_alarm_rate`` (sufijo
    ``_bdate`` si además el índice OOS es aproximado).
    """
    out = metrics.copy().reset_index(drop=True)
    benchmark = benchmark or load_benchmark_spec()
    output_dir = Path(output_dir)
    if "fuente_deteccion" not in out:
        out["fuente_deteccion"] = pd.Series([None] * len(out), dtype=object)
    updates: dict[int, dict[str, float]] = {}
    sources: dict[int, str] = {}
    for track in out["pista"].astype(str).str.upper().unique():
        windows = track_crisis_windows(track, benchmark)
        oos: tuple[pd.DatetimeIndex, bool] | None = None
        for i in out.index[out["pista"].astype(str).str.upper() == track]:
            row = out.loc[i]
            has_csv = (
                pd.notna(row.get("det_event_recall", np.nan))
                and pd.notna(row.get("det_min_run", np.nan))
                and int(row["det_min_run"]) == int(min_run)
            )
            if has_csv:
                sources[i] = row["fuente_deteccion"] if isinstance(row["fuente_deteccion"], str) else "csv"
                continue
            panel_path = _panel_path(output_dir, track, str(row["id"]))
            if panel_path.is_file() and pd.notna(row.get("n_states", np.nan)):
                panel = pd.read_parquet(panel_path, columns=["state"])
                flags = panel["state"].eq(int(row["n_states"]) - 1)
                updates[i] = detection_summary(flags, windows, min_run=min_run)
                sources[i] = "panel"
                continue
            if oos is None:
                oos = track_oos_index(track, out, output_dir)
            updates[i] = _detection_from_coverage(row, oos[0], windows, min_run)
            sources[i] = "cobertura_aprox" if oos[1] else "cobertura_aprox_bdate"
    if updates:
        update_frame = pd.DataFrame.from_dict(updates, orient="index")
        for col in update_frame:
            if col not in out:
                out[col] = np.nan
            out.loc[update_frame.index, col] = update_frame[col].astype(float)
    for i, source in sources.items():
        out.at[i, "fuente_deteccion"] = source
    return out


def detection_score(
    precision: float | pd.Series, recall: float | pd.Series, beta: float = 1.0
) -> float | pd.Series:
    """F-beta entre precisión diaria y recall por evento (0 si no detecta nada).

    Es el *composite F-score* de Garg et al. (2021): recall a nivel de evento y
    precisión a nivel de instante. NaN de precisión (nunca marca) → 0.
    """
    precision = pd.Series(precision, dtype=float) if not np.isscalar(precision) else precision
    recall = pd.Series(recall, dtype=float) if not np.isscalar(recall) else recall
    b2 = float(beta) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        score = (1 + b2) * precision * recall / (b2 * precision + recall)
    if np.isscalar(score):
        return 0.0 if not np.isfinite(score) else float(score)
    return score.where(np.isfinite(score), 0.0)


def rank_detection(
    metrics: pd.DataFrame,
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    benchmark: dict | None = None,
    min_run: int = int(DETECTION_DEFAULTS["min_run"]),
    beta: float = DETECTION_DEFAULTS["beta"],
    lift_min: float = DETECTION_DEFAULTS["lift_min"],
    min_mean_duration: float = DETECTION_DEFAULTS["min_mean_duration"],
    min_crisis_run: float = DETECTION_DEFAULTS["min_crisis_run"],
    score_decimals: int = int(DETECTION_DEFAULTS["score_decimals"]),
) -> pd.DataFrame:
    """Ranking de detección (ADR-003), dentro de cada pista.

    1. **Niveles** (se ordenan antes que el score):
       0 ``elegible``; 1 ``precision_no_supera_azar`` (lift ≤ ``lift_min``);
       2 ``parpadeo`` (duración media de régimen < ``min_mean_duration`` o
       duración media de los episodios de crisis < ``min_crisis_run``);
       3 ``degenerado`` (salida constante: ``switching_rate == 0`` o nunca marca).
    2. **Score** dentro de cada nivel: ``score_deteccion`` = F-beta entre
       ``det_event_recall`` y ``det_precision``.
    3. **Desempate** (score igual a ``score_decimals``): menor
       ``switching_rate`` y después mayor ``label_stability``.

    ``puesto_deteccion`` comparte puesto solo si nivel, score redondeado y
    desempates coinciden. Se añaden ``rank_medio_legacy`` y ``puesto_legacy``
    (:func:`rank_within_track`) solo como referencia descriptiva.
    """
    out = add_detection_metrics(metrics, output_dir, min_run=min_run, benchmark=benchmark)
    out = add_comparison_metrics(out)
    legacy = rank_within_track(out)[["pista", "id", "rank_medio", "puesto_pista"]].rename(
        columns={"rank_medio": "rank_medio_legacy", "puesto_pista": "puesto_legacy"}
    )
    out = out.merge(legacy, on=["pista", "id"], how="left")
    out["score_deteccion"] = detection_score(out["det_precision"], out["det_event_recall"], beta)
    switching = pd.to_numeric(out["switching_rate"], errors="coerce")
    duration = pd.to_numeric(out["mean_regime_duration"], errors="coerce")
    marked = pd.to_numeric(out["det_marked_rate"], errors="coerce")
    lift = pd.to_numeric(out["det_lift_precision"], errors="coerce")
    degenerate = (
        switching.fillna(0).eq(0)
        | out["det_precision"].isna()
        | marked.le(0)
        | marked.ge(1)
    )
    crisis_run = pd.to_numeric(out["det_mean_crisis_run"], errors="coerce")
    flicker = (
        duration.lt(min_mean_duration) | duration.isna()
        | crisis_run.lt(min_crisis_run) | crisis_run.isna()
    )
    weak = ~lift.gt(lift_min)
    out["nivel"] = np.select([degenerate, flicker, weak], [3, 2, 1], default=0)
    out["nivel_etiqueta"] = out["nivel"].map(DETECTION_LEVELS)
    out["elegible"] = out["nivel"].eq(0)

    keys = pd.DataFrame({
        "nivel": out["nivel"],
        "score": -out["score_deteccion"].round(score_decimals),
        "switching": switching.fillna(np.inf),
        "stability": -pd.to_numeric(out["label_stability"], errors="coerce").fillna(-np.inf),
    })
    order = pd.concat([out[["pista", "id"]], keys], axis=1).sort_values(
        ["pista", "nivel", "score", "switching", "stability", "id"]
    )
    places = pd.Series(np.nan, index=order.index)
    for _, part in order.groupby("pista", sort=False):
        key_cols = part[["nivel", "score", "switching", "stability"]]
        changed = key_cols.ne(key_cols.shift()).any(axis=1).to_numpy()
        position = np.arange(1, len(part) + 1, dtype=float)
        places.loc[part.index] = pd.Series(np.where(changed, position, np.nan)).ffill().to_numpy()
    out["puesto_deteccion"] = places.astype(int)
    out.attrs["detection_params"] = {
        "min_run": int(min_run), "beta": float(beta), "lift_min": float(lift_min),
        "min_mean_duration": float(min_mean_duration), "min_crisis_run": float(min_crisis_run),
        "score_decimals": int(score_decimals),
    }
    return out.sort_values(["pista", "puesto_deteccion", "id"]).reset_index(drop=True)


RANKING_COLUMNS = [
    "pista", "puesto_deteccion", "id", "detector", "familia", "nivel_etiqueta",
    "score_deteccion", "det_event_recall", "det_n_detectados", "det_n_eventos",
    "det_precision", "det_lift_precision", "det_base_rate", "det_marked_rate",
    "det_day_recall", "det_mean_crisis_run", "mean_crisis_coverage", "false_alarm_rate", "mean_trap_activation",
    "switching_rate", "mean_regime_duration", "label_stability", "elapsed_seconds",
    "fuente_deteccion", "rank_medio_legacy", "puesto_legacy",
]


def pareto_mask(frame: pd.DataFrame, maximize: Iterable[str]) -> np.ndarray:
    """True para las filas no dominadas (todas las columnas se maximizan)."""
    values = frame[list(maximize)].to_numpy(dtype=float)
    values = np.where(np.isnan(values), -np.inf, values)
    keep = np.ones(len(values), dtype=bool)
    for i in range(len(values)):
        dominated = (values >= values[i]).all(axis=1) & (values > values[i]).any(axis=1)
        keep[i] = not dominated.any()
    return keep


# --------------------------------------------------------------------------- #
# Líneas base triviales y nulo de azar persistente
# --------------------------------------------------------------------------- #
def _markov_flags(n: int, rate: float, mean_run: float, rng: np.random.Generator) -> np.ndarray:
    """Señal binaria de Markov estacionaria: fracción ``rate`` y rachas de crisis
    de duración media ``mean_run`` (rachas de calma: ``mean_run·(1−rate)/rate``)."""
    if rate <= 0:
        return np.zeros(n, dtype=bool)
    if rate >= 1:
        return np.ones(n, dtype=bool)
    crisis_run = max(float(mean_run), 1.0)
    calm_run = max(crisis_run * (1 - rate) / rate, 1.0)
    flags = np.zeros(n, dtype=bool)
    state = bool(rng.random() < rate)
    position = 0
    while position < n:
        length = int(rng.geometric(1.0 / (crisis_run if state else calm_run)))
        flags[position:position + length] = state
        position += length
        state = not state
    return flags


def _baseline_row(
    flags: np.ndarray,
    oos_index: pd.DatetimeIndex,
    track: str,
    name: str,
    windows: dict[str, tuple[str, str]],
    fp_windows: dict[str, tuple[str, str]],
    min_run: int,
) -> dict:
    states = pd.Series(np.asarray(flags, dtype=int), index=oos_index)
    constant = states.nunique() <= 1
    row: dict = {
        "pista": track, "id": name, "detector": name.lower(), "familia": "Línea base",
        "n_states": 2, "n_oos": len(states),
        "false_alarm_rate": ev.false_alarm_rate(states, 1, windows),
        "switching_rate": ev.switching_rate(states),
        "mean_regime_duration": ev.mean_regime_duration(states),
        # Un modelo constante no cambia al reentrenar; uno aleatorio no tiene
        # estabilidad definida (NaN → al fondo en el eje heredado).
        "label_stability": 1.0 if constant else np.nan,
        "fuente_deteccion": "linea_base",
    }
    row.update({f"cov_{k}": v for k, v in ev.crisis_coverage(states, 1, windows).items()})
    row.update({f"fa_{k}": v for k, v in ev.false_alarm_in_windows(states, 1, fp_windows).items()})
    row.update(detection_summary(states.eq(1), windows, min_run=min_run))
    return row


def trivial_baselines(
    track: str,
    oos_index: pd.DatetimeIndex,
    *,
    rate: float,
    mean_run: float,
    benchmark: dict | None = None,
    min_run: int = int(DETECTION_DEFAULTS["min_run"]),
    n_sims: int = 200,
    seed: int = 20260929,
) -> pd.DataFrame:
    """Cuatro detectores sin información, con el mismo esquema que las métricas.

    - ``SIEMPRE_CRISIS`` / ``SIEMPRE_CALMA``: salidas constantes.
    - ``AZAR_IID``: Bernoulli diario con tasa ``rate`` (una realización fija).
    - ``AZAR_PERSISTENTE``: Markov con tasa ``rate`` y rachas medias
      ``mean_run``; fila = **media** de ``n_sims`` realizaciones, con la
      precisión fijada en su valor esperado (la tasa base, lift = 1): una señal
      independiente de las crisis no puede, en esperanza, concentrarse en ellas.
    """
    track = track.upper()
    benchmark = benchmark or load_benchmark_spec()
    windows = track_crisis_windows(track, benchmark)
    fp_windows = track_false_positive_windows(benchmark)
    n = len(oos_index)
    rng = np.random.default_rng(seed)
    rows = [
        _baseline_row(np.ones(n, bool), oos_index, track, "SIEMPRE_CRISIS", windows, fp_windows, min_run),
        _baseline_row(np.zeros(n, bool), oos_index, track, "SIEMPRE_CALMA", windows, fp_windows, min_run),
        _baseline_row(rng.random(n) < rate, oos_index, track, "AZAR_IID", windows, fp_windows, min_run),
    ]
    sims = pd.DataFrame([
        _baseline_row(_markov_flags(n, rate, mean_run, rng), oos_index, track,
                      "AZAR_PERSISTENTE", windows, fp_windows, min_run)
        for _ in range(int(n_sims))
    ])
    expected = sims.iloc[0].to_dict()
    numeric = sims.select_dtypes("number").mean()
    expected.update(numeric.to_dict())
    base_rate = float(expected["det_base_rate"])
    expected.update({
        "det_precision": base_rate,
        "det_lift_precision": 1.0,
        "false_alarm_rate": 1.0 - base_rate,
        "label_stability": np.nan,
        "fuente_deteccion": f"linea_base_media_{int(n_sims)}_sim",
    })
    rows.append(expected)
    return pd.DataFrame(rows)


def persistent_random_null(
    oos_index: pd.DatetimeIndex,
    windows: dict[str, tuple[str, str]],
    *,
    rate: float,
    mean_run: float,
    n_sims: int = 300,
    seed: int = 0,
    min_run: int = int(DETECTION_DEFAULTS["min_run"]),
    beta: float = DETECTION_DEFAULTS["beta"],
) -> pd.DataFrame:
    """Distribución nula de (recall, precisión, lift, score) para una señal de
    Markov independiente de las crisis con la misma tasa y persistencia."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(int(n_sims)):
        flags = pd.Series(_markov_flags(len(oos_index), rate, mean_run, rng), index=oos_index)
        summary = detection_summary(flags, windows, min_run=min_run)
        rows.append({
            "det_event_recall": summary["det_event_recall"],
            "det_precision": summary["det_precision"],
            "det_lift_precision": summary["det_lift_precision"],
        })
    null = pd.DataFrame(rows)
    null["score_deteccion"] = detection_score(null["det_precision"], null["det_event_recall"], beta)
    return null
