"""
walk_forward.py — Marco de evaluación COMÚN, comparable y CAUSAL
(antes src/evaluation.py).

Este es el corazón del proyecto: el conjunto de métricas y el protocolo
walk-forward con los que se juzga a TODOS los detectores de la misma manera. Un
detector "bueno" no es el que mejor encaja in-sample, sino el que mejor se porta
out-of-sample bajo este marco.

Dos bloques:
  1. Protocolo causal (`walk_forward`): reentrena/predice en ventanas móviles sin
     ver el futuro y devuelve una serie de etiquetas/probabilidades out-of-sample.
  2. Métricas (`evaluate`): a partir de esas etiquetas causales calcula la tabla
     de métricas estandarizada que va a results/.

Las ventanas de eventos y las métricas individuales viven en
``regimenes.evaluacion.metricas`` (mismos objetos, importados por referencia).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from regimenes.detectores.base import RegimeDetector

from regimenes.evaluacion.metricas import (
    CRISIS_WINDOWS,
    DRAWDOWN_TROUGHS,
    FALSE_POSITIVE_WINDOWS,
    block_bootstrap_coverage_ci,
    crisis_coverage,
    false_alarm_in_windows,
    false_alarm_rate,
    label_stability,
    lead_lag,
    mean_regime_duration,
    silhouette_states,
    switching_rate,
)


# --------------------------------------------------------------------------- #
# Resultado estandarizado
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationResult:
    """Contenedor del resultado de evaluación de UN detector.

    `to_row()` produce una fila plana para la tabla maestra de results/, de modo
    que todos los detectores sean comparables en un único DataFrame/CSV.
    """

    detector_name: str
    crisis_coverage: dict[str, float] = field(default_factory=dict)   # % días crisis por ventana
    crisis_coverage_ci: dict[str, tuple[float, float]] = field(default_factory=dict)  # IC bootstrap por ventana
    false_alarm_in_fp: dict[str, float] = field(default_factory=dict) # % días crisis en ventanas FP
    lead_lag_days: dict[str, float] = field(default_factory=dict)     # señal vs trough (días, - = anticipa)
    false_alarm_rate: float = float("nan")                            # global, fuera de crisis
    switching_rate: float = float("nan")                              # conmutaciones / nº días
    mean_regime_duration: float = float("nan")                        # persistencia (días)
    label_stability: float = float("nan")                             # estabilidad walk-forward [0,1]
    silhouette: float = float("nan")                                  # separación de regímenes en feature-space
    log_likelihood: float = float("nan")
    aic: float = float("nan")
    bic: float = float("nan")
    n_states: int = -1
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        """Aplana el resultado a un dict de una fila (para concatenar en results/).

        ESQUEMA FIJO e idéntico para todos los detectores (las claves de ventanas
        y troughs vienen de las constantes del módulo), de modo que las filas de
        distintos detectores concatenen sin desalinearse. La columna
        'ventana_eval' es obligatoria (decisión de proyecto): identifica en qué
        ventana out-of-sample se evaluó el detector.
        """
        row: dict = {
            "detector": self.detector_name,
            "n_states": self.n_states,
            "ventana_eval": self.extra.get("ventana_eval", "?"),
            "oos_start": self.extra.get("oos_start", None),
            "oos_end": self.extra.get("oos_end", None),
            "n_oos": self.extra.get("n_oos", None),
            "false_alarm_rate": self.false_alarm_rate,
            "switching_rate": self.switching_rate,
            "mean_regime_duration": self.mean_regime_duration,
            "label_stability": self.label_stability,
            "silhouette": self.silhouette,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
        }
        # Métricas por ventana (claves constantes -> esquema estable). Junto a cada
        # cobertura puntual se añade su IC bootstrap por bloques [lo, hi] (NaN si la
        # ventana cae fuera del rango OOS o si el detector nunca marca crisis ahí).
        for k in CRISIS_WINDOWS:
            row[f"cov_{k}"] = self.crisis_coverage.get(k, float("nan"))
            lo, hi = self.crisis_coverage_ci.get(k, (float("nan"), float("nan")))
            row[f"cov_{k}_lo"] = lo
            row[f"cov_{k}_hi"] = hi
        for k in FALSE_POSITIVE_WINDOWS:
            row[f"fa_{k}"] = self.false_alarm_in_fp.get(k, float("nan"))
        for k in DRAWDOWN_TROUGHS:
            row[f"leadlag_{k}"] = self.lead_lag_days.get(k, float("nan"))
        return row


# --------------------------------------------------------------------------- #
# Protocolo walk-forward (causal)
# --------------------------------------------------------------------------- #
# Contexto causal por defecto (ADR-003): UN AÑO de observaciones previas al bloque.
# Se expresa en años y se traduce a filas según la frecuencia del índice de X
# (252 sesiones si es diario, 52 semanal, 12 mensual, 4 trimestral).
CONTEXT_YEARS: float = 1.0
_OBS_PER_YEAR: tuple[tuple[float, int], ...] = (
    # (mediana máxima de días naturales entre observaciones, observaciones/año)
    (4.0, 252),   # diario hábil (mediana 1 día; fines de semana no la mueven)
    (10.0, 52),   # semanal
    (45.0, 12),   # mensual
    (120.0, 4),   # trimestral
)


def resolve_context(context: int | str | None, index: pd.Index) -> int | None:
    """Traduce el parámetro ``context`` de `walk_forward` a nº de filas.

    - ``"auto"``: ``CONTEXT_YEARS`` años en la frecuencia inferida del índice
      (252 filas en datos diarios, 12 en mensuales...).
    - ``None``: todo el tramo de train disponible (propagación exacta, más cara).
    - ``int >= 0``: nº de filas explícito; ``0`` reproduce el protocolo anterior
      (cada bloque arranca en frío, sin contexto).
    """
    if context is None:
        return None
    if isinstance(context, str):
        if context != "auto":
            raise ValueError(f"context debe ser 'auto', None o int >= 0; recibido {context!r}")
        obs_per_year = 252
        if isinstance(index, pd.DatetimeIndex) and len(index) > 2:
            gaps = np.diff(index.asi8) / 86_400e9  # días naturales
            median_gap = float(np.median(gaps))
            obs_per_year = next(
                (n for limit, n in _OBS_PER_YEAR if median_gap <= limit), 1
            )
        return int(round(CONTEXT_YEARS * obs_per_year))
    context = int(context)
    if context < 0:
        raise ValueError(f"context debe ser >= 0; recibido {context}")
    return context


def _ctx_start(start: int, lo: int, context: int | None) -> int:
    """Primera fila del contexto: ``context`` filas antes de ``start``, sin salir
    del train del fold (``lo``). ``None`` = desde el inicio del train."""
    first = lo if context is None else max(lo, start - context)
    return min(first, start)


def _predict_block(det: RegimeDetector, X: pd.DataFrame, start: int, stop: int, lo: int, context: int | None):
    """Predice las filas ``[start, stop)`` de X arrancando ``context`` filas antes.

    Las filas de contexto son la cola del train (``>= lo``, inicio del train del
    fold) y solo sirven para que los detectores con estado (autómatas de
    histéresis, filtros forward, CUSUM) lleguen al bloque con el estado que
    tendrían en tiempo real. Todas son anteriores a ``start``: causal. Se
    devuelven únicamente las predicciones del bloque.
    """
    ctx_start = _ctx_start(start, lo, context)
    window = X.iloc[ctx_start:stop]
    offset = start - ctx_start
    states = np.asarray(det.predict_online(window))[offset:]
    proba = np.asarray(det.predict_proba(window))[offset:]
    return states, proba


def walk_forward(
    detector_factory: Callable[[], RegimeDetector],
    X: pd.DataFrame,
    *,
    market_returns: pd.Series | None = None,
    train_size: int = 252 * 8,
    step: int = 21,
    expanding: bool = True,
    min_train: int = 252 * 5,
    context: int | str | None = "auto",
) -> pd.DataFrame:
    """Genera etiquetas y probabilidades OUT-OF-SAMPLE de forma causal.

    Reentrena el detector en ventanas crecientes (expanding) o móviles (rolling)
    y predice el siguiente bloque de `step` días usando SOLO datos <= t. Es el
    único punto donde se decide cómo se simula el "tiempo real".

    Parameters
    ----------
    detector_factory : Callable[[], RegimeDetector]
        Función SIN argumentos que devuelve una instancia NUEVA del detector
        (para reentrenar desde cero en cada ventana sin fugas de estado).
    X : pd.DataFrame
        Features causales completas, indexadas por fecha y ya alineadas.
    market_returns : pd.Series | None
        Retornos del S&P 500 (mismo índice que X o reindexable). Si se pasan, en
        CADA fold se RE-FIJA el orden económico de estados (0=calma..n-1=crisis)
        con estos retornos del tramo de train — robusto para detectores que NO
        operan sobre retornos crudos (varianza, sigma GARCH, change-point,
        Mahalanobis). Si es None, cada detector ordena por su propio criterio
        (fallback con warning); ver `RegimeDetector.label_states_economically`.
    train_size : int
        Tamaño de la ventana de entrenamiento inicial (días de trading).
    step : int
        Nº de días que se predicen out-of-sample antes de reentrenar (p. ej. 21
        ≈ 1 mes). Menor = más fiel a online, más costoso.
    expanding : bool
        True -> ventana de entrenamiento creciente (recomendado, más datos).
        False -> ventana móvil de tamaño fijo `train_size`.
    min_train : int
        Mínimo de observaciones antes de empezar a predecir.
    context : int | 'auto' | None
        Propagación de estado entre bloques (ADR-003). Tras ajustar con el train
        del fold, se predice sobre ``[cola del train de `context` filas] + bloque``
        y se conservan solo las predicciones del bloque. Así un detector con
        estado (histéresis de D01/D02/D06/D10, filtros forward, CUSUM) no
        reinicia en "calma" en cada refit. Es causal: las filas de contexto son
        anteriores al bloque y ya estaban en el train. ``'auto'`` (defecto) = 1
        año en la frecuencia del índice (252 filas diarias, 12 mensuales);
        ``None`` = todo el train; ``0`` = protocolo anterior (arranque en frío).
        Para detectores sin estado (GMM, AE+GMM) el resultado es idéntico; los
        que ya anteponen el train como burn-in (D04-D08, D10, D11) no duplican
        nada porque recortan su burn-in a las fechas anteriores a la entrada.

    Returns
    -------
    pd.DataFrame indexado por fecha con columnas:
        - 'state'       : etiqueta dura canónica out-of-sample
        - 'p_crisis'    : probabilidad de crisis out-of-sample
        - 'fold'        : id de la ventana que produjo la predicción
    Solo contiene fechas predichas out-of-sample (las del primer train no). Cada
    fecha aparece UNA vez: la predicción CAUSAL del fold que la cubrió por primera
    vez (modelo entrenado con datos < inicio de su bloque). ESTAS son las
    etiquetas que consumen TODAS las métricas (cobertura, falsas alarmas,
    lead/lag, switching, duración).

    Diagnóstico de estabilidad (AISLADO): en `.attrs['stability_panel']` se
    adjunta un DataFrame date×fold con re-predicciones del bloque ANTERIOR hechas
    por el modelo del fold actual (que vio más datos de los que le tocaban a esas
    fechas). Es información NO CAUSAL y se usa EXCLUSIVAMENTE por `label_stability`
    como diagnóstico de cuánto cambia la etiqueta de una fecha al reentrenar con
    más datos. NUNCA debe leerse para cobertura/falsas alarmas/lead-lag: esas
    métricas solo tocan las columnas del panel, jamás `.attrs`.

    Causalidad: cada fold se ajusta SOLO con datos < inicio del bloque de test.
    Los parámetros (y el orden económico de estados) se fijan con el train; la
    predicción del bloque usa esos parámetros congelados. Para causalidad estricta
    intra-bloque (sin suavizado que mire días futuros del propio bloque), un
    detector DEBE sobrescribir `predict_online` con filtrado causal (p. ej. los HMM
    usan filtrado forward, no Viterbi); aquí se invoca `predict_online`.
    """
    first_split = max(int(train_size), int(min_train))
    n = len(X)
    if first_split >= n:
        raise ValueError(
            f"train_size/min_train ({first_split}) >= nº de observaciones ({n})"
        )
    ctx_rows = resolve_context(context, X.index)
    records: list[tuple] = []       # CAUSAL OOS: 1 fila/fecha; lo leen TODAS las métricas
    stab_records: list[tuple] = []  # DIAGNÓSTICO (no causal): solo para label_stability
    prev_test: pd.DataFrame | None = None
    prev_start = 0
    fold_id = 0
    t = first_split
    while t < n:
        train_lo = 0 if expanding else max(0, t - train_size)
        train = X.iloc[train_lo:t]
        test = X.iloc[t:t + step]
        if len(test) == 0:
            break
        det = detector_factory()
        # Si vamos a re-fijar el orden con market_returns, silenciamos el warning
        # del etiquetado provisional que el detector hace dentro de fit (sería
        # ruido redundante por fold). Sin market_returns, el aviso SÍ pasa.
        with warnings.catch_warnings():
            if market_returns is not None:
                warnings.filterwarnings("ignore", message=r".*label_states_economically.*")
                warnings.filterwarnings("ignore", message=r".*market_returns.*")
            det.fit(train)
        # Re-fijar el orden económico CAUSALMENTE con los retornos del train
        # (sobrescribe el etiquetado provisional que el detector hiciera en fit).
        if market_returns is not None:
            mr_train = pd.Series(market_returns).reindex(train.index)
            det.label_states_economically(train, market_returns=mr_train)
        # Predicción del bloque con la cola del train como contexto (ADR-003).
        states, proba = _predict_block(det, X, t, t + len(test), train_lo, ctx_rows)
        p_crisis = proba[:, det.crisis_state]
        for d, s, p in zip(test.index, states, p_crisis):
            records.append((d, int(s), float(p), fold_id))
        # Estabilidad: re-predecir el bloque PREVIO con este modelo (más datos),
        # con el mismo contexto que el bloque propio para que sea comparable.
        if prev_test is not None and len(prev_test):
            s_prev = np.asarray(det.predict_online(
                X.iloc[_ctx_start(prev_start, train_lo, ctx_rows):prev_start + len(prev_test)]
            ))[-len(prev_test):]
            for d, s in zip(prev_test.index, s_prev):
                stab_records.append((d, fold_id, int(s)))
        # Y registrar el bloque actual bajo su propio fold (para comparar).
        for d, s in zip(test.index, states):
            stab_records.append((d, fold_id, int(s)))
        prev_test = test
        prev_start = t
        fold_id += 1
        t += step

    panel = pd.DataFrame(records, columns=["date", "state", "p_crisis", "fold"]).set_index("date")
    panel.index = pd.to_datetime(panel.index)
    if stab_records:
        sp = pd.DataFrame(stab_records, columns=["date", "fold", "state"])
        stab_panel = sp.pivot_table(index="date", columns="fold", values="state", aggfunc="first")
        panel.attrs["stability_panel"] = stab_panel
    return panel

# --------------------------------------------------------------------------- #
# Orquestador: una llamada -> EvaluationResult completo
# --------------------------------------------------------------------------- #
def evaluate(
    detector: RegimeDetector,
    wf_panel: pd.DataFrame,
    *,
    market_returns: pd.Series | None = None,
    X_full: pd.DataFrame | None = None,
) -> EvaluationResult:
    """Calcula TODAS las métricas estandarizadas para un detector ya pasado por
    walk-forward y devuelve un `EvaluationResult`.

    Combina las métricas causales (cobertura de crisis, falsas alarmas,
    lead/lag, switching, persistencia, estabilidad) con las de bondad de ajuste
    (logL/AIC/BIC) cuando el modelo las expone. No recalcula nada in-sample.

    Parameters
    ----------
    detector : RegimeDetector
        Instancia (para name, n_states, crisis_state, score/aic/bic).
    wf_panel : pd.DataFrame
        Salida de `walk_forward` (state, p_crisis, fold) indexada por fecha.
    market_returns : pd.Series | None
        Retornos S&P 500 para validación económica (retorno medio por estado).
    X_full : pd.DataFrame | None
        Features completas, solo para logL/AIC/BIC donde aplique.

    Returns
    -------
    EvaluationResult
    """
    states = wf_panel["state"]
    p_crisis = wf_panel["p_crisis"]
    cs = detector.crisis_state

    res = EvaluationResult(
        detector_name=detector.name,
        crisis_coverage=crisis_coverage(states, cs),
        crisis_coverage_ci=block_bootstrap_coverage_ci(states, cs),
        false_alarm_in_fp=false_alarm_in_windows(states, cs),
        lead_lag_days=lead_lag(p_crisis),
        false_alarm_rate=false_alarm_rate(states, cs),
        switching_rate=switching_rate(states),
        mean_regime_duration=mean_regime_duration(states),
        label_stability=label_stability(wf_panel.attrs.get("stability_panel")),
        n_states=detector.n_states,
    )

    # Silueta de los regímenes en el espacio de features (calidad de partición; útil
    # sobre todo para clustering). Se computa sobre las etiquetas causales OOS.
    if X_full is not None:
        try:
            res.silhouette = silhouette_states(X_full, states)
        except Exception:  # noqa: BLE001
            pass

    # Bondad de ajuste (donde el modelo la exponga; NaN si no).
    if X_full is not None:
        try:
            res.log_likelihood = float(detector.score(X_full))
            res.aic = float(detector.aic(X_full))
            res.bic = float(detector.bic(X_full))
        except Exception:  # noqa: BLE001  (modelos no generativos)
            pass

    # Metadatos de ventana OOS (obligatorio 'ventana_eval').
    oos_start, oos_end = states.index.min(), states.index.max()
    res.extra["oos_start"] = oos_start.date().isoformat()
    res.extra["oos_end"] = oos_end.date().isoformat()
    res.extra["n_oos"] = int(len(states))
    res.extra["ventana_eval"] = f"{oos_start.date()}→{oos_end.date()} (n={len(states)})"

    # Validación económica: retorno medio por estado canónico.
    if market_returns is not None:
        mr = market_returns.reindex(states.index)
        by_state = {int(s): float(mr[states == s].mean()) for s in np.unique(states.values)}
        res.extra["mean_return_by_state"] = by_state

    return res


def results_table(results: list[EvaluationResult]) -> pd.DataFrame:
    """Apila varios `EvaluationResult` en la tabla maestra comparativa de results/.

    Una fila por detector, columnas = métricas (esquema fijo de `to_row`). Es el
    artefacto central de la FASE 4 (síntesis comparativa) y el formato común que
    cada detector vuelca en results/.
    """
    return pd.DataFrame([r.to_row() for r in results])
