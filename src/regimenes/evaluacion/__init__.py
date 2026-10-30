"""El juez: protocolo walk-forward, métricas y ranking de detección.

- ``walk_forward``  ``EvaluationResult``, ``walk_forward``, ``evaluate``,
                    ``results_table``.
- ``metricas``      ventanas de eventos, métricas individuales causales y
                    detección por evento ``det_*``.
- ``mcs``           Model Confidence Set (A1) y matriz de pérdidas preliminar
                    crisis × detector (``perdidas_por_crisis`` importa ranking
                    de forma diferida).
- ``curvas``        detección/retraso/ruido-señal a igual tasa de falsas
                    alarmas al año (A3).
- ``ranking``       criterio de ranking de detección ADR-003.
                    NO se importa aquí (evita el ciclo ranking → benchmark → evaluacion):
                    ``from regimenes.evaluacion import ranking``.

``from regimenes import evaluacion as ev`` expone la API del juez
(``ev.walk_forward``, ``ev.CRISIS_WINDOWS``…). Los tres dicts de
ventanas son los MISMOS objetos que ``regimenes.evaluacion.metricas``.
"""

from regimenes.evaluacion.metricas import (
    CRISIS_WINDOWS,
    DRAWDOWN_TROUGHS,
    FALSE_POSITIVE_WINDOWS,
    block_bootstrap_coverage_ci,
    configure_event_windows,
    crisis_coverage,
    detection_summary,
    event_detection_table,
    false_alarm_in_windows,
    false_alarm_rate,
    label_stability,
    lead_lag,
    mean_regime_duration,
    silhouette_states,
    switching_rate,
)
from regimenes.evaluacion.mcs import (
    MCSResult,
    bootstrap_indices,
    model_confidence_set,
    perdidas_por_crisis,
    tasa_dias_calma_marcados,
)
from regimenes.evaluacion.curvas import (
    FA_OBJETIVO,
    calibrar_umbral_ex_post,
    curva_deteccion,
    episodios_alarma,
    metricas_umbral,
    umbral_expansivo,
)
from regimenes.evaluacion.walk_forward import (
    CONTEXT_YEARS,
    EvaluationResult,
    evaluate,
    resolve_context,
    results_table,
    walk_forward,
)

__all__ = [
    "CRISIS_WINDOWS", "FALSE_POSITIVE_WINDOWS", "DRAWDOWN_TROUGHS", "CONTEXT_YEARS",
    "configure_event_windows", "EvaluationResult", "resolve_context", "walk_forward",
    "crisis_coverage", "false_alarm_in_windows", "false_alarm_rate", "lead_lag",
    "switching_rate", "mean_regime_duration", "label_stability", "silhouette_states",
    "block_bootstrap_coverage_ci", "evaluate", "results_table",
    "event_detection_table", "detection_summary",
    "MCSResult", "bootstrap_indices", "model_confidence_set", "perdidas_por_crisis",
    "tasa_dias_calma_marcados",
    "FA_OBJETIVO", "episodios_alarma", "metricas_umbral", "calibrar_umbral_ex_post",
    "umbral_expansivo", "curva_deteccion",
]
