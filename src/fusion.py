"""Fusión causal e interpretable de alerta temprana y confirmación de régimen."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from src import evaluation as ev


DECISION_LABELS = {0: "normal", 1: "vigilancia", 2: "confirmado"}


@dataclass(frozen=True)
class FusionConfig:
    """Parámetros fijados antes de comparar la fusión con las crisis conocidas."""

    alert_threshold: float = 0.5
    confirmation_threshold: float = 0.5
    alert_persistence: int = 1
    confirmation_persistence: int = 3
    warning_horizon: int = 63
    alert_source: Literal["state", "probability"] = "state"

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


def load_panel(output_dir: Path, track: str, detector_id: str) -> pd.DataFrame:
    """Carga un panel OOS guardado por el benchmark y valida su contrato mínimo."""
    track = track.upper()
    detector_id = detector_id.upper()
    path = Path(output_dir) / "panels" / f"{track}_{detector_id}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"No existe el panel walk-forward {path}.")
    panel = pd.read_parquet(path).sort_index()
    missing = {"state", "p_crisis"} - set(panel.columns)
    if missing:
        raise ValueError(f"{path}: faltan columnas {sorted(missing)}.")
    if not panel.index.is_unique or not panel.index.is_monotonic_increasing:
        raise ValueError(f"{path}: el índice debe ser temporal, único y ordenado.")
    panel.index = pd.to_datetime(panel.index)
    return panel


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
        crisis_state = alert["state"].max()
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
    rows: list[dict] = []
    for warning in eligible.itertuples(index=False):
        alert_date = pd.Timestamp(warning.alert_date)
        alert_pos = int(index.searchsorted(alert_date))
        candidates: list[tuple[int, int, str, pd.Timestamp, pd.Timestamp, int]] = []
        for name, (start_text, end_text) in windows.items():
            start = pd.Timestamp(start_text)
            end = pd.Timestamp(end_text)
            start_pos = int(index.searchsorted(start, side="left"))
            end_pos = int(index.searchsorted(end, side="right")) - 1
            if end_pos < 0 or start_pos >= len(index):
                continue
            signed_lead = start_pos - alert_pos
            if start <= alert_date <= end:
                candidates.append((0, 0, str(name), start, end, signed_lead))
            elif alert_date < start and 0 < signed_lead <= warning_horizon:
                candidates.append((signed_lead, 1, str(name), start, end, signed_lead))
            elif alert_date > end:
                distance = alert_pos - end_pos
                if 0 < distance <= warning_horizon:
                    candidates.append((distance, 2, str(name), start, end, signed_lead))
        if candidates:
            _, phase_order, crisis, start, end, signed_lead = min(candidates)
            phase = {0: "durante_crisis", 1: "anticipacion", 2: "posterior"}[phase_order]
        else:
            crisis, start, end, phase, signed_lead = None, pd.NaT, pd.NaT, "falsa_alarma", np.nan
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


def per_crisis_scorecard(fused: pd.DataFrame) -> pd.DataFrame:
    """Cobertura de cada crisis para facilitar una comparación no promediada."""
    signals = {
        "D7_alerta": fused["alert_active"],
        "D8_confirmacion": fused["confirmation_active"],
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
    for position, active in enumerate(fused["confirmation_active"].to_numpy()):
        if not active:
            keep = False
        elif bool(fused["confirmation_onset"].iloc[position]):
            keep = position in matched_confirmations
        accepted[position] = bool(active and keep)
    return pd.Series(accepted, index=fused.index)


def ablation_scorecard(
    fused: pd.DataFrame, warning_horizon: int
) -> pd.DataFrame:
    """Compara sensores aislados, fusión paralela y embudo secuencial."""
    gated = _gated_confirmation(fused, warning_horizon)
    signals = {
        "D7_solo": fused["alert_active"],
        "D8_solo": fused["confirmation_active"],
        "fusion_paralela": fused["actionable"],
        "confirmacion_D8_tras_D7": gated,
        "fusion_secuencial": fused["warning_window"] | gated,
    }
    rows = []
    for name, signal in signals.items():
        states = signal.astype(int)
        coverage = ev.crisis_coverage(states, crisis_state=1)
        traps = ev.false_alarm_in_windows(states, crisis_state=1)
        rows.append(
            {
                "arquitectura": name,
                "mean_crisis_coverage": _nanmean_or_nan(coverage.values()),
                "false_alarm_rate": ev.false_alarm_rate(states, crisis_state=1),
                "mean_trap_activation": _nanmean_or_nan(traps.values()),
                "switching_rate": ev.switching_rate(states),
                "mean_duration": ev.mean_regime_duration(states),
            }
        )
    table = pd.DataFrame(rows)
    table["operational_utility"] = (
        table["mean_crisis_coverage"]
        - table["false_alarm_rate"]
        - table["mean_trap_activation"]
        - 5.0 * table["switching_rate"]
    )
    return table


def warning_summary(fused: pd.DataFrame, warning_horizon: int) -> dict[str, float]:
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
    rows = []
    for name, signal in signals.items():
        states = signal.astype(int)
        coverage = ev.crisis_coverage(states, crisis_state=1)
        traps = ev.false_alarm_in_windows(states, crisis_state=1)
        rows.append(
            {
                "signal": name,
                "mean_crisis_coverage": _nanmean_or_nan(coverage.values()),
                "false_alarm_rate": ev.false_alarm_rate(states, crisis_state=1),
                "mean_trap_activation": _nanmean_or_nan(traps.values()),
                "switching_rate": ev.switching_rate(states),
                "mean_duration": ev.mean_regime_duration(states),
            }
        )
    return pd.DataFrame(rows)


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
