"""Fusión causal normal/vigilancia/confirmado y fusiones del ADR-011.

La máquina vive en ``regimenes.fusion.maquina``; aquí se re-exporta su API
pública para ``from regimenes.fusion import fuse_early_warning``. La vara común
de las fusiones (ADR-011 §2 y §5) está en ``regimenes.fusion.vara`` y las
fusiones fijas (§3) en ``regimenes.fusion.fijas`` y la online (§4) en
``regimenes.fusion.online``; se exponen como submódulos
(``from regimenes.fusion import vara, fijas, online``), cargados al pedirlos.
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


def __getattr__(nombre: str):
    """``regimenes.fusion.vara``, ``.fijas`` y ``.online`` se cargan al pedirlos (PEP 562).

    No se importan aquí para que ``python -m regimenes.fusion.fijas`` no encuentre el módulo ya
    cargado por el paquete (``runpy`` avisaría)."""
    if nombre in ("vara", "fijas", "online"):
        import importlib

        return importlib.import_module(f"regimenes.fusion.{nombre}")
    raise AttributeError(f"module 'regimenes.fusion' has no attribute {nombre!r}")


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
    "vara",
    "fijas",
    "online",
]
