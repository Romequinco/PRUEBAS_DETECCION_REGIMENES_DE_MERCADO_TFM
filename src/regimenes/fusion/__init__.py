"""Fusión causal normal/vigilancia/confirmado.

La implementación vive en ``regimenes.fusion.maquina``; aquí se re-exporta su API
pública para ``from regimenes.fusion import fuse_early_warning``.
"""

from regimenes.fusion.maquina import (
    DECISION_LABELS,
    PHASES,
    SWITCHING_PENALTY,
    FusionConfig,
    load_panel,
    crisis_state_from_metrics,
    operational_utility,
    fuse_early_warning,
    warning_episode_table,
    confirmation_episode_table,
    warning_phase_base_rates,
    warning_ground_truth_table,
    per_crisis_scorecard,
    signal_scores,
    ablation_scorecard,
    fusion_verdict,
    warning_summary,
    event_scorecard,
    period_scorecard,
    select_then_evaluate,
    sensitivity_table,
    assert_prefix_causal,
)

__all__ = [
    "DECISION_LABELS",
    "PHASES",
    "SWITCHING_PENALTY",
    "FusionConfig",
    "load_panel",
    "crisis_state_from_metrics",
    "operational_utility",
    "fuse_early_warning",
    "warning_episode_table",
    "confirmation_episode_table",
    "warning_phase_base_rates",
    "warning_ground_truth_table",
    "per_crisis_scorecard",
    "signal_scores",
    "ablation_scorecard",
    "fusion_verdict",
    "warning_summary",
    "event_scorecard",
    "period_scorecard",
    "select_then_evaluate",
    "sensitivity_table",
    "assert_prefix_causal",
]
