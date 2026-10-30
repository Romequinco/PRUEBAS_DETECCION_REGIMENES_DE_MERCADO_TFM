"""Evaluación de variantes detector × menú sobre un panel dado (ADR-008).

Repite el protocolo de :func:`regimenes.benchmark.ejecucion.run_one` (mismo juez, mismo
walk-forward, mismo ajuste final y misma detección por evento) pero:

- recibe el panel y el índice OOS en vez de leerlos de ``data/processed``, para poder
  comparar menús sobre el MISMO panel (por ejemplo, uno construido en memoria);
- no escribe nada ni usa la caché por huella.

Vive aparte de ``ejecucion`` a propósito: ``run_one``, ``load_track_panel`` y
``common_track_index`` forman parte de la huella de la caché del benchmark, y tocarlos
invalidaría los 24 resultados oficiales.
"""

from __future__ import annotations

import time
import warnings

import pandas as pd

from regimenes import evaluacion as ev
from regimenes.benchmark.ejecucion import DEFAULT_TRAIN_DAYS, configure_evaluation
from regimenes.detectores.registry import DetectorSpec
from regimenes.evaluacion.metricas import detection_summary
from regimenes.evaluacion.ranking import detection_score


def evaluar_variante(track: str, spec: DetectorSpec, panel: pd.DataFrame, index: pd.DatetimeIndex,
                     *, train_days: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Métricas (una fila, esquema de ``run_one`` + ``score_deteccion``) y panel OOS.

    ``panel`` debe traer las columnas de ``spec.features`` y ``SP500_ret``; ``index`` es
    el índice común de la comparación (el mismo para todas las variantes que se comparan).
    """
    track = track.upper()
    min_train = int(train_days or DEFAULT_TRAIN_DAYS[track])
    configure_evaluation(track)
    faltan = sorted(set(spec.features) - set(panel.columns))
    if faltan:
        raise KeyError(f"{track}/{spec.detector_id}: faltan columnas {faltan}")
    X = panel.loc[index, list(spec.features)]
    if X.isna().any().any():
        raise ValueError(f"{track}/{spec.detector_id}: NaN dentro del índice común")
    market_returns = panel["SP500_ret"].reindex(X.index)

    started = time.perf_counter()
    wf_panel = ev.walk_forward(spec.factory, X, market_returns=market_returns, train_size=min_train,
                               min_train=min_train, step=spec.step, expanding=spec.expanding)
    final = spec.factory()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*label_states_economically.*")
        warnings.filterwarnings("ignore", message=r".*market_returns.*")
        final.fit(X)
    final.label_states_economically(X, market_returns=market_returns)
    result = ev.evaluate(final, wf_panel, market_returns=market_returns, X_full=X)
    metrics = ev.results_table([result])
    flags = wf_panel["state"].eq(final.crisis_state)
    for key, value in detection_summary(flags, dict(ev.CRISIS_WINDOWS)).items():
        metrics[key] = value
    metrics["score_deteccion"] = detection_score(metrics["det_precision"], metrics["det_event_recall"])
    for key, value in reversed(list({
        "id": spec.detector_id, "pista": track, "n_features_input": len(spec.features),
        "features_input": "|".join(spec.features), "train_days": min_train, "refit_step": spec.step,
        "elapsed_seconds": time.perf_counter() - started,
    }.items())):
        metrics.insert(0, key, value)
    wf_panel = wf_panel.copy()
    wf_panel.attrs = {}
    return metrics, wf_panel
