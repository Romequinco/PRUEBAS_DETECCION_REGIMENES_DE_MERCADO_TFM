"""Fusión causal e interpretable de alerta temprana y confirmación de régimen.

Convenciones que usan los notebooks 06 y 07:

* La decisión en ``t`` solo depende de filas ``<= t`` de los paneles OOS del
  benchmark (``assert_prefix_causal`` lo demuestra por truncado).
* Las crisis conocidas (``ev.CRISIS_WINDOWS``, configuradas por pista con
  ``regimenes.benchmark.configure_evaluation``) solo se usan *después* para puntuar.
* "Precisión de avisos" (``warning_precision``) es **acuerdo con el
  confirmador**, no acierto frente a crisis reales; para eso está
  ``warning_ground_truth_table`` y su línea base ``warning_phase_base_rates``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Literal, Mapping

import numpy as np
import pandas as pd

from regimenes import evaluacion as ev


DECISION_LABELS = {0: "normal", 1: "vigilancia", 2: "confirmado"}
PHASES = ("anticipacion", "durante_crisis", "posterior", "falsa_alarma")
SWITCHING_PENALTY = 5.0


@dataclass(frozen=True)
class FusionConfig:
    """Parámetros fijados antes de comparar la fusión con las crisis conocidas.

    ``alert_crisis_state`` es obligatorio si ``alert_source='state'``: el índice
    del estado de crisis es una propiedad del detector (``n_states - 1`` en la
    convención canónica de ``RegimeDetector``), no algo que deba inferirse del
    panel. Inferirlo con ``state.max()`` usaría la muestra completa (si el
    estado más severo no aparece hasta el final, las decisiones pasadas
    cambiarían al añadir futuro).
    """

    alert_threshold: float = 0.5
    confirmation_threshold: float = 0.5
    alert_persistence: int = 1
    confirmation_persistence: int = 3
    warning_horizon: int = 63
    alert_source: Literal["state", "probability"] = "state"
    alert_crisis_state: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("alert_threshold", self.alert_threshold),
            ("confirmation_threshold", self.confirmation_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} debe estar en [0, 1].")
        for name, value in (
            ("alert_persistence", self.alert_persistence),
            ("confirmation_persistence", self.confirmation_persistence),
            ("warning_horizon", self.warning_horizon),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} debe ser >= 1.")
        if self.alert_source not in ("state", "probability"):
            raise ValueError("alert_source debe ser 'state' o 'probability'.")
        if self.alert_crisis_state is not None and int(self.alert_crisis_state) < 1:
            raise ValueError("alert_crisis_state debe ser >= 1 (0 = calma).")


def load_panel(output_dir: Path, track: str, detector_id: str) -> pd.DataFrame:
    """Carga un panel OOS guardado por el benchmark y valida su contrato mínimo."""
    track = track.upper()
    detector_id = detector_id.upper()
    path = Path(output_dir) / "panels" / f"{track}_{detector_id}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"No existe el panel walk-forward {path}.")
    panel = pd.read_parquet(path)
    missing = {"state", "p_crisis"} - set(panel.columns)
    if missing:
        raise ValueError(f"{path}: faltan columnas {sorted(missing)}.")
    panel.index = pd.to_datetime(panel.index)
    if not panel.index.is_unique or not panel.index.is_monotonic_increasing:
        raise ValueError(f"{path}: el índice debe ser temporal, único y ordenado.")
    if panel[["state", "p_crisis"]].isna().any().any():
        raise ValueError(f"{path}: 'state'/'p_crisis' contienen NaN.")
    return panel


def crisis_state_from_metrics(
    metrics: pd.DataFrame, track: str, detector_id: str
) -> int:
    """Estado de crisis canónico (``n_states - 1``) según las métricas del benchmark."""
    rows = metrics.loc[
        metrics["pista"].astype(str).str.upper().eq(track.upper())
        & metrics["id"].astype(str).str.upper().eq(detector_id.upper()),
        "n_states",
    ]
    if len(rows) != 1:
        raise KeyError(f"No hay una única fila de métricas para {track}/{detector_id}.")
    return int(rows.iloc[0]) - 1


def operational_utility(
    coverage: float | pd.Series,
    false_alarm_rate: float | pd.Series,
    trap_activation: float | pd.Series,
    switching_rate: float | pd.Series,
    switching_penalty: float = SWITCHING_PENALTY,
) -> float | pd.Series:
    """Índice descriptivo ``cobertura − falsas alarmas − trampas − 5×cambios``.

    No es rentabilidad ni una función de pérdida calibrada: los pesos son
    convencionales y fijos. Solo sirve para **ordenar** alternativas evaluadas
    con las mismas labels. Su valor absoluto no tiene lectura directa: una
    señal que nunca se activa tiene ``false_alarm_rate`` NaN (y utilidad NaN), y
    una señal siempre activa obtiene ``−(fracción de días fuera de crisis)``.
    """
    return (
        coverage
        - false_alarm_rate
        - trap_activation
        - switching_penalty * switching_rate
    )


def _sustained(signal: pd.Series, persistence: int) -> pd.Series:
    persistence = int(persistence)
    if persistence == 1:
        return signal.astype(bool)
    return (
        signal.astype(int)
        .rolling(persistence, min_periods=persistence)
        .sum()
        .eq(persistence)
    )


def _nanmean_or_nan(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.nanmean(array)) if np.isfinite(array).any() else np.nan


def _align_panels(
    alert_panel: pd.DataFrame, confirmation_panel: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = alert_panel.index.intersection(confirmation_panel.index).sort_values()
    if index.empty:
        raise ValueError("Los detectores no comparten fechas OOS.")
    return alert_panel.loc[index], confirmation_panel.loc[index]


def fuse_early_warning(
    alert_panel: pd.DataFrame,
    confirmation_panel: pd.DataFrame,
    config: FusionConfig = FusionConfig(),
) -> pd.DataFrame:
    """Construye una decisión causal: normal → vigilancia → confirmado.

    La vigilancia comienza únicamente en el flanco de subida del detector de alerta
    y caduca tras ``warning_horizon`` sesiones. La confirmación exige evidencia
    contemporánea persistente del segundo detector. No se usa ningún dato futuro.
    """
    alert, confirmation = _align_panels(alert_panel, confirmation_panel)
    if config.alert_source == "state":
        if config.alert_crisis_state is None:
            raise ValueError(
                "alert_source='state' exige FusionConfig.alert_crisis_state "
                "(p. ej. crisis_state_from_metrics); inferirlo del panel no es causal."
            )
        crisis_state = int(config.alert_crisis_state)
        if int(alert["state"].max()) > crisis_state:
            raise ValueError(
                f"El panel de alerta tiene estados > alert_crisis_state={crisis_state}."
            )
        raw_alert = alert["state"].eq(crisis_state)
    else:
        raw_alert = alert["p_crisis"].ge(config.alert_threshold)
    alert_active = _sustained(raw_alert, config.alert_persistence)
    confirmation_active = _sustained(
        confirmation["p_crisis"].ge(config.confirmation_threshold),
        config.confirmation_persistence,
    )
    alert_onset = alert_active & ~alert_active.shift(fill_value=False)
    confirmation_onset = (
        confirmation_active & ~confirmation_active.shift(fill_value=False)
    )
    warning_onset = alert_onset & ~confirmation_active

    warning_window = np.zeros(len(alert), dtype=bool)
    warning_age = np.full(len(alert), np.nan)
    last_alert: int | None = None
    for position, starts in enumerate(warning_onset.to_numpy()):
        if starts:
            last_alert = position
        if last_alert is not None:
            age = position - last_alert
            if age <= config.warning_horizon:
                warning_window[position] = True
                warning_age[position] = age

    decision_code = np.where(
        confirmation_active.to_numpy(),
        2,
        np.where(warning_window, 1, 0),
    ).astype(int)
    fused = pd.DataFrame(
        {
            "alert_state": alert["state"].astype(int),
            "alert_probability": alert["p_crisis"].astype(float),
            "confirmation_state": confirmation["state"].astype(int),
            "confirmation_probability": confirmation["p_crisis"].astype(float),
            "alert_active": alert_active.astype(bool),
            "alert_onset": alert_onset.astype(bool),
            "warning_onset": warning_onset.astype(bool),
            "confirmation_active": confirmation_active.astype(bool),
            "confirmation_onset": confirmation_onset.astype(bool),
            "warning_window": warning_window,
            "warning_age": warning_age,
            "decision_code": decision_code,
        },
        index=alert.index,
    )
    fused["decision"] = pd.Categorical(
        pd.Series(decision_code, index=fused.index).map(DECISION_LABELS),
        categories=list(DECISION_LABELS.values()),
        ordered=True,
    )
    fused["actionable"] = fused["decision_code"].ge(1)
    return fused


def warning_episode_table(
    fused: pd.DataFrame, warning_horizon: int
) -> pd.DataFrame:
    """Una fila por alerta con su primera confirmación posterior, si existe."""
    alert_positions = np.flatnonzero(fused["alert_onset"].to_numpy())
    matches = _one_to_one_episode_matches(fused, warning_horizon)
    rows: list[dict] = []
    for position in alert_positions:
        already_confirmed = bool(fused["confirmation_active"].iloc[position])
        confirmed_at = matches.get(int(position))
        rows.append(
            {
                "alert_date": fused.index[position],
                "already_confirmed": already_confirmed,
                "confirmation_date": (
                    fused.index[confirmed_at] if confirmed_at is not None else pd.NaT
                ),
                "lead_sessions": (
                    confirmed_at - position if confirmed_at is not None else np.nan
                ),
                "outcome": (
                    "already_confirmed" if already_confirmed
                    else "confirmed" if confirmed_at is not None
                    else "false_or_unconfirmed"
                ),
            }
        )
    return pd.DataFrame(rows)


def _one_to_one_episode_matches(
    fused: pd.DataFrame, warning_horizon: int
) -> dict[int, int]:
    """Empareja cada aviso y confirmación como máximo una vez.

    En cada entrada del confirmador se consume el aviso elegible no utilizado más
    reciente dentro del horizonte. Es exactamente la información disponible en
    ese instante y evita que un único aviso explique múltiples reentradas.
    """
    warnings = np.flatnonzero(fused["warning_onset"].to_numpy())
    confirmations = np.flatnonzero(fused["confirmation_onset"].to_numpy())
    unmatched = list(map(int, warnings))
    matches: dict[int, int] = {}
    for confirmation in map(int, confirmations):
        candidates = [
            warning for warning in unmatched
            if warning < confirmation <= warning + int(warning_horizon)
        ]
        if not candidates:
            continue
        warning = candidates[-1]
        matches[warning] = confirmation
        unmatched.remove(warning)
    return matches


def confirmation_episode_table(
    fused: pd.DataFrame, warning_horizon: int
) -> pd.DataFrame:
    """Una fila por confirmación y la alerta previa más reciente dentro del horizonte."""
    # ``warning_onset`` ya excluye las alertas aparecidas durante confirmación.
    matches = _one_to_one_episode_matches(fused, warning_horizon)
    confirmation_to_alert = {
        confirmation: warning for warning, confirmation in matches.items()
    }
    confirmation_positions = np.flatnonzero(
        fused["confirmation_onset"].to_numpy()
    )
    rows: list[dict] = []
    for position in confirmation_positions:
        alert_at = confirmation_to_alert.get(int(position))
        rows.append(
            {
                "confirmation_date": fused.index[position],
                "preceded_by_alert": alert_at is not None,
                "alert_date": (
                    fused.index[alert_at] if alert_at is not None else pd.NaT
                ),
                "lead_sessions": (
                    position - alert_at if alert_at is not None else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _window_positions(
    index: pd.DatetimeIndex, windows: Mapping[str, tuple[str, str]]
) -> list[tuple[str, pd.Timestamp, pd.Timestamp, int, int]]:
    out = []
    for name, (start_text, end_text) in windows.items():
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        start_pos = int(index.searchsorted(start, side="left"))
        end_pos = int(index.searchsorted(end, side="right")) - 1
        if end_pos < 0 or start_pos >= len(index):
            continue
        out.append((str(name), start, end, start_pos, end_pos))
    return out


def _classify_alert(
    alert_pos: int,
    alert_date: pd.Timestamp,
    bounds: list[tuple[str, pd.Timestamp, pd.Timestamp, int, int]],
    warning_horizon: int,
) -> tuple[str | None, pd.Timestamp, pd.Timestamp, str, float]:
    """Fase de un aviso frente a las ventanas de crisis (solo para evaluar)."""
    candidates: list[tuple[int, int, str, pd.Timestamp, pd.Timestamp, int]] = []
    for name, start, end, start_pos, end_pos in bounds:
        signed_lead = start_pos - alert_pos
        if start <= alert_date <= end:
            candidates.append((0, 0, name, start, end, signed_lead))
        elif alert_date < start and 0 < signed_lead <= warning_horizon:
            candidates.append((signed_lead, 1, name, start, end, signed_lead))
        elif alert_date > end:
            distance = alert_pos - end_pos
            if 0 < distance <= warning_horizon:
                candidates.append((distance, 2, name, start, end, signed_lead))
    if not candidates:
        return None, pd.NaT, pd.NaT, "falsa_alarma", np.nan
    _, phase_order, crisis, start, end, signed_lead = min(candidates)
    phase = {0: "durante_crisis", 1: "anticipacion", 2: "posterior"}[phase_order]
    return crisis, start, end, phase, float(signed_lead)


def warning_phase_base_rates(
    index: pd.DatetimeIndex,
    warning_horizon: int,
    crisis_windows: dict[str, tuple[str, str]] | None = None,
) -> pd.Series:
    """Línea base de azar para ``warning_ground_truth_table``.

    Clasifica *cada* sesión OOS como si en ella se hubiera emitido un aviso y
    devuelve la fracción de sesiones en cada fase. Es la distribución esperada
    de fases para avisos emitidos en fechas uniformemente aleatorias: un
    detector solo anticipa "de verdad" si su fracción de ``anticipacion`` supera
    claramente esta base (con n pequeño, la comparación es orientativa).
    """
    windows = crisis_windows or ev.CRISIS_WINDOWS
    index = pd.DatetimeIndex(index)
    bounds = _window_positions(index, windows)
    phases = [
        _classify_alert(position, date, bounds, warning_horizon)[3]
        for position, date in enumerate(index)
    ]
    return (
        pd.Series(phases, dtype=object)
        .value_counts(normalize=True)
        .reindex(list(PHASES), fill_value=0.0)
        .astype(float)
    )


def warning_ground_truth_table(
    fused: pd.DataFrame,
    warning_horizon: int,
    crisis_windows: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Atribuye cada aviso elegible a una crisis real o a una falsa alarma.

    ``anticipacion`` significa que el aviso ocurrió como máximo
    ``warning_horizon`` sesiones antes del inicio; ``durante_crisis`` y
    ``posterior`` distinguen señales reactivas. Si hay dos eventos cercanos se
    elige la frontera temporal más próxima.
    """
    windows = crisis_windows or ev.CRISIS_WINDOWS
    warnings = warning_episode_table(fused, warning_horizon)
    if warnings.empty:
        return pd.DataFrame(
            columns=[
                "alert_date", "confirmation_date", "lead_to_confirmation_sessions",
                "crisis", "crisis_start", "crisis_end", "phase",
                "lead_to_crisis_start_sessions",
            ]
        )
    eligible = warnings.loc[~warnings["already_confirmed"]].copy()
    index = fused.index
    bounds = _window_positions(index, windows)
    rows: list[dict] = []
    for warning in eligible.itertuples(index=False):
        alert_date = pd.Timestamp(warning.alert_date)
        alert_pos = int(index.searchsorted(alert_date))
        crisis, start, end, phase, signed_lead = _classify_alert(
            alert_pos, alert_date, bounds, warning_horizon
        )
        rows.append(
            {
                "alert_date": alert_date,
                "confirmation_date": warning.confirmation_date,
                "lead_to_confirmation_sessions": warning.lead_sessions,
                "crisis": crisis,
                "crisis_start": start,
                "crisis_end": end,
                "phase": phase,
                "lead_to_crisis_start_sessions": signed_lead,
            }
        )
    return pd.DataFrame(rows)


def per_crisis_scorecard(
    fused: pd.DataFrame,
    alert_label: str = "D7",
    confirmation_label: str = "D8",
) -> pd.DataFrame:
    """Cobertura de cada crisis para facilitar una comparación no promediada."""
    signals = {
        f"{alert_label}_alerta": fused["alert_active"],
        f"{confirmation_label}_confirmacion": fused["confirmation_active"],
        "fusion_paralela": fused["actionable"],
    }
    rows = []
    for crisis, (start, end) in ev.CRISIS_WINDOWS.items():
        row: dict[str, float | str] = {"crisis": crisis}
        for name, signal in signals.items():
            segment = signal.loc[
                (signal.index >= pd.Timestamp(start))
                & (signal.index <= pd.Timestamp(end))
            ]
            row[name] = float(segment.mean()) if len(segment) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _gated_confirmation(fused: pd.DataFrame, warning_horizon: int) -> pd.Series:
    """Acepta una confirmación completa solo si tuvo aviso previo elegible."""
    matched_confirmations = set(
        _one_to_one_episode_matches(fused, warning_horizon).values()
    )
    accepted = np.zeros(len(fused), dtype=bool)
    keep = False
    onsets = fused["confirmation_onset"].to_numpy()
    for position, active in enumerate(fused["confirmation_active"].to_numpy()):
        if not active:
            keep = False
        elif bool(onsets[position]):
            keep = position in matched_confirmations
        accepted[position] = bool(active and keep)
    return pd.Series(accepted, index=fused.index)


def signal_scores(signal: pd.Series) -> dict[str, float]:
    """Métricas del juez para una señal binaria (True = crisis) con las labels activas."""
    states = signal.astype(int)
    coverage = ev.crisis_coverage(states, crisis_state=1)
    traps = ev.false_alarm_in_windows(states, crisis_state=1)
    scores = {
        "mean_crisis_coverage": _nanmean_or_nan(coverage.values()),
        "false_alarm_rate": ev.false_alarm_rate(states, crisis_state=1),
        "mean_trap_activation": _nanmean_or_nan(traps.values()),
        "switching_rate": ev.switching_rate(states),
        "mean_duration": ev.mean_regime_duration(states),
    }
    scores["operational_utility"] = operational_utility(
        scores["mean_crisis_coverage"],
        scores["false_alarm_rate"],
        scores["mean_trap_activation"],
        scores["switching_rate"],
    )
    return scores


def ablation_scorecard(
    fused: pd.DataFrame,
    warning_horizon: int,
    alert_label: str = "D7",
    confirmation_label: str = "D8",
) -> pd.DataFrame:
    """Compara sensores aislados, fusión paralela y embudo secuencial."""
    gated = _gated_confirmation(fused, warning_horizon)
    a, c = alert_label, confirmation_label
    signals = {
        f"{a}_solo": fused["alert_active"],
        f"{c}_solo": fused["confirmation_active"],
        "fusion_paralela": fused["actionable"],
        f"confirmacion_{c}_tras_{a}": gated,
        "fusion_secuencial": fused["warning_window"] | gated,
    }
    return pd.DataFrame(
        [{"arquitectura": name, **signal_scores(signal)} for name, signal in signals.items()]
    )


def fusion_verdict(
    joint_utility: float, alert_utility: float, confirmation_utility: float
) -> str:
    """Etiqueta la utilidad de la fusión frente a cada sensor aislado (4 casos)."""
    beats_alert = joint_utility > alert_utility
    beats_confirmation = joint_utility > confirmation_utility
    if beats_alert and beats_confirmation:
        return "mejora_ambos"
    if beats_confirmation:
        return "mejora_confirmador_pero_no_alerta"
    if beats_alert:
        return "mejora_alerta_pero_no_confirmador"
    return "no_mejora"


def warning_summary(fused: pd.DataFrame, warning_horizon: int) -> dict[str, float]:
    """Resumen de acuerdo alerta→confirmación (no de acierto frente a crisis).

    ``warnings`` cuenta solo avisos elegibles (el confirmador no estaba activo).
    ``warning_precision`` = fracción de esos avisos que el confirmador ratifica
    dentro del horizonte; ``confirmation_recall`` = fracción de entradas del
    confirmador precedidas por un aviso emparejado. Ambas miden **acuerdo entre
    detectores**; la anticipación de crisis reales se mide con
    ``warning_ground_truth_table`` frente a ``warning_phase_base_rates``.
    """
    warnings = warning_episode_table(fused, warning_horizon)
    confirmations = confirmation_episode_table(fused, warning_horizon)
    eligible = (
        warnings.loc[~warnings["already_confirmed"]]
        if len(warnings)
        else pd.DataFrame(columns=["outcome"])
    )
    confirmed = eligible["outcome"].eq("confirmed") if len(eligible) else pd.Series(dtype=bool)
    preceded = confirmations["preceded_by_alert"] if len(confirmations) else pd.Series(dtype=bool)
    leads = confirmations.loc[preceded, "lead_sessions"] if len(confirmations) else pd.Series(dtype=float)
    decision = pd.Series(fused["decision_code"].to_numpy(), index=fused.index)
    return {
        "warnings": float(len(eligible)),
        "confirmed_warnings": float(confirmed.sum()),
        "false_or_unconfirmed_warnings": float((~confirmed).sum()),
        "warning_precision": float(confirmed.mean()) if len(confirmed) else np.nan,
        "confirmation_episodes": float(len(confirmations)),
        "preceded_confirmations": float(preceded.sum()),
        "confirmation_recall": float(preceded.mean()) if len(preceded) else np.nan,
        "median_lead_sessions": float(leads.median()) if len(leads) else np.nan,
        "share_warning_days": float(fused["decision_code"].eq(1).mean()),
        "share_confirmed_days": float(fused["decision_code"].eq(2).mean()),
        "decision_switching_rate": ev.switching_rate(decision),
        "mean_decision_duration": ev.mean_regime_duration(decision),
    }


def event_scorecard(fused: pd.DataFrame) -> pd.DataFrame:
    """Compara alerta, confirmación y decisión accionable con las mismas labels."""
    signals = {
        "alerta": fused["alert_active"],
        "confirmacion": fused["confirmation_active"],
        "accionable": fused["actionable"],
    }
    return pd.DataFrame(
        [{"signal": name, **signal_scores(signal)} for name, signal in signals.items()]
    )


def period_scorecard(
    fused: pd.DataFrame,
    periods: Mapping[str, tuple[str | pd.Timestamp, str | pd.Timestamp]],
    column: str = "actionable",
) -> pd.DataFrame:
    """Puntúa una señal de ``fused`` por sub-periodos (p. ej. mitades temporales).

    La máquina de estados se ejecuta una sola vez sobre toda la historia (es
    causal, así que la primera mitad no ve la segunda); aquí solo se recorta el
    tramo que se puntúa. Una crisis que cruza la frontera se evalúa con los días
    que caen en cada tramo. Usa las labels activas del juez
    (``configure_evaluation`` de la pista antes de llamar).

    Si en un tramo no cae ninguna ventana trampa (p. ej. 2013 en la primera
    mitad de la pista A), la activación en trampas es NaN y la utilidad del
    tramo se calcula sin ese término (``trampas_en_tramo=False``).
    """
    rows = []
    for name, (start, end) in periods.items():
        segment = fused.loc[
            (fused.index >= pd.Timestamp(start)) & (fused.index <= pd.Timestamp(end)),
            column,
        ]
        scores = signal_scores(segment)
        has_traps = bool(np.isfinite(scores["mean_trap_activation"]))
        if not has_traps:
            scores["operational_utility"] = operational_utility(
                scores["mean_crisis_coverage"], scores["false_alarm_rate"],
                0.0, scores["switching_rate"],
            )
        rows.append(
            {"periodo": name, "n_sesiones": int(len(segment)),
             "trampas_en_tramo": has_traps, **scores}
        )
    return pd.DataFrame(rows)


def select_then_evaluate(
    table: pd.DataFrame,
    *,
    candidate_col: str = "alerta",
    period_col: str = "periodo",
    score_col: str = "utilidad",
    selection_period: str = "seleccion",
    evaluation_period: str = "evaluacion",
    eligible_col: str | None = "elegible",
) -> pd.DataFrame:
    """Selecciona en un tramo y evalúa en otro (control del sesgo de selección).

    ``table`` tiene una fila por candidato×periodo (ya agregada entre pistas si
    procede). Devuelve, por candidato elegible, la puntuación y el puesto en cada
    tramo, y marca el ganador del tramo de selección. Si el ganador cae de puesto
    en el tramo de evaluación, parte de su ventaja en muestra completa se debía a
    quedarse con el máximo de muchas comparaciones.
    """
    wide = table.pivot_table(
        index=candidate_col, columns=period_col, values=score_col, dropna=False
    )
    for period in (selection_period, evaluation_period):
        if period not in wide.columns:
            raise KeyError(f"Falta el periodo {period!r} en la tabla.")
    out = pd.DataFrame(
        {
            f"{score_col}_{selection_period}": wide[selection_period],
            f"{score_col}_{evaluation_period}": wide[evaluation_period],
        }
    )
    if eligible_col is not None:
        eligible = table.groupby(candidate_col)[eligible_col].all()
        out["elegible"] = eligible.reindex(out.index).fillna(False).astype(bool)
    else:
        out["elegible"] = True
    pool = out.loc[out["elegible"]]
    for period in (selection_period, evaluation_period):
        column = f"{score_col}_{period}"
        out[f"puesto_{period}"] = pool[column].rank(ascending=False, method="min")
    winner = pool[f"{score_col}_{selection_period}"].idxmax() if len(pool) else None
    out["elegido_en_seleccion"] = out.index == winner
    return out.sort_values([f"puesto_{selection_period}", f"{score_col}_{selection_period}"],
                           ascending=[True, False])


def sensitivity_table(
    alert_panel: pd.DataFrame,
    confirmation_panel: pd.DataFrame,
    base_config: FusionConfig,
    horizons: Iterable[int] = (21, 63, 126),
) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        config = replace(base_config, warning_horizon=int(horizon))
        fused = fuse_early_warning(alert_panel, confirmation_panel, config)
        summary = warning_summary(fused, config.warning_horizon)
        actionable = event_scorecard(fused).set_index("signal").loc["accionable"]
        rows.append(
            {
                "warning_horizon": int(horizon),
                **summary,
                "actionable_crisis_coverage": actionable["mean_crisis_coverage"],
                "actionable_false_alarm_rate": actionable["false_alarm_rate"],
                "actionable_trap_activation": actionable["mean_trap_activation"],
            }
        )
    return pd.DataFrame(rows)


def assert_prefix_causal(
    alert_panel: pd.DataFrame,
    confirmation_panel: pd.DataFrame,
    config: FusionConfig,
    fractions: Iterable[float] = (0.4, 0.7, 0.9),
) -> bool:
    """Demuestra por truncado que añadir futuro no cambia decisiones pasadas."""
    alert, confirmation = _align_panels(alert_panel, confirmation_panel)
    full = fuse_early_warning(alert, confirmation, config)
    checked = [
        "alert_active",
        "alert_onset",
        "warning_onset",
        "confirmation_active",
        "confirmation_onset",
        "warning_window",
        "decision_code",
        "actionable",
    ]
    for fraction in fractions:
        stop = max(1, min(len(full), int(len(full) * float(fraction))))
        truncated = fuse_early_warning(alert.iloc[:stop], confirmation.iloc[:stop], config)
        pd.testing.assert_frame_equal(
            full.iloc[:stop][checked],
            truncated[checked],
            check_dtype=True,
            check_exact=True,
        )
    return True
