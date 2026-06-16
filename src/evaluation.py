"""
evaluation.py — Marco de evaluación COMÚN, comparable y CAUSAL.

Este es el corazón del proyecto: el conjunto de métricas y el protocolo
walk-forward con los que se juzga a TODOS los detectores de la misma manera. Un
detector "bueno" no es el que mejor encaja in-sample, sino el que mejor se porta
out-of-sample bajo este marco.

Dos bloques:
  1. Protocolo causal (`walk_forward`): reentrena/predice en ventanas móviles sin
     ver el futuro y devuelve una serie de etiquetas/probabilidades out-of-sample.
  2. Métricas (`evaluate`): a partir de esas etiquetas causales calcula la tabla
     de métricas estandarizada que va a results/.

Ventanas de crisis y falsos positivos conocidos (fuente de verdad única)
------------------------------------------------------------------------
Se usan para medir cobertura de crisis y tasa de falsas alarmas. Fechas
aproximadas de mercado (S&P 500), ajustables en FASE 1 con el EDA.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
# Fecha del mínimo de cada gran caída. Se recalculan en FASE 1 desde los datos.
DRAWDOWN_TROUGHS: dict[str, str] = {
    "GFC_2008": "2009-03-09",
    "COVID_2020": "2020-03-23",
    "Inflation_2022": "2022-10-12",
}


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
    false_alarm_in_fp: dict[str, float] = field(default_factory=dict) # % días crisis en ventanas FP
    lead_lag_days: dict[str, float] = field(default_factory=dict)     # señal vs trough (días, - = anticipa)
    false_alarm_rate: float = float("nan")                            # global, fuera de crisis
    switching_rate: float = float("nan")                              # conmutaciones / nº días
    mean_regime_duration: float = float("nan")                        # persistencia (días)
    label_stability: float = float("nan")                             # estabilidad walk-forward [0,1]
    log_likelihood: float = float("nan")
    aic: float = float("nan")
    bic: float = float("nan")
    n_states: int = -1
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        """Aplana el resultado a un dict de una fila (para concatenar en results/)."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Protocolo walk-forward (causal)
# --------------------------------------------------------------------------- #
def walk_forward(
    detector_factory,
    X: pd.DataFrame,
    *,
    train_size: int = 252 * 8,
    step: int = 21,
    expanding: bool = True,
    min_train: int = 252 * 5,
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

    Returns
    -------
    pd.DataFrame indexado por fecha con columnas:
        - 'state'       : etiqueta dura canónica out-of-sample
        - 'p_crisis'    : probabilidad de crisis out-of-sample
        - 'fold'        : id de la ventana que produjo la predicción
    Solo contiene fechas predichas out-of-sample (las del primer train no).
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Métricas individuales (todas causales: operan sobre etiquetas walk-forward)
# --------------------------------------------------------------------------- #
def crisis_coverage(
    states: pd.Series, crisis_state: int, windows: dict[str, tuple[str, str]] = None
) -> dict[str, float]:
    """% de días etiquetados como 'crisis' dentro de cada ventana de crisis conocida.

    Mide sensibilidad: idealmente alto (cercano a 1) en 2008/2011/2020/2022.
    """
    raise NotImplementedError


def false_alarm_in_windows(
    states: pd.Series, crisis_state: int, windows: dict[str, tuple[str, str]] = None
) -> dict[str, float]:
    """% de días 'crisis' dentro de ventanas que NO son crisis (2013, 2018).

    Mide especificidad en episodios trampa: idealmente bajo.
    """
    raise NotImplementedError


def false_alarm_rate(
    states: pd.Series, crisis_state: int, crisis_windows: dict[str, tuple[str, str]] = None
) -> float:
    """Tasa global de falsas alarmas: fracción de días marcados 'crisis' que caen
    FUERA de todas las ventanas de crisis conocidas.

    Aproxima 1 - precisión usando las ventanas conocidas como ground truth laxo.
    """
    raise NotImplementedError


def lead_lag(
    p_crisis: pd.Series,
    troughs: dict[str, str] = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Días de adelanto/retraso entre la primera señal de crisis y el suelo del
    drawdown del S&P 500, por evento.

    Para cada trough busca el primer día (dentro de una ventana previa) en que
    `p_crisis` cruza `threshold`. Devuelve (fecha_señal - fecha_trough) en días
    de trading: negativo = la señal ANTICIPA el suelo, positivo = va por detrás.
    """
    raise NotImplementedError


def switching_rate(states: pd.Series) -> float:
    """Frecuencia de conmutación = nº de cambios de estado / nº de días.

    Penaliza el 'flickering' (regímenes que parpadean día a día). Más bajo =
    más persistente/estable.
    """
    raise NotImplementedError


def mean_regime_duration(states: pd.Series) -> float:
    """Duración media (en días) de los episodios de régimen. Inverso del flicker."""
    raise NotImplementedError


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
        sucesivas del walk-forward.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Orquestador: una llamada -> EvaluationResult completo
# --------------------------------------------------------------------------- #
def evaluate(
    detector,
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
        Retornos S&P 500 para lead/lag y validación económica.
    X_full : pd.DataFrame | None
        Features completas, solo para logL/AIC/BIC donde aplique.

    Returns
    -------
    EvaluationResult
    """
    raise NotImplementedError


def results_table(results: list[EvaluationResult]) -> pd.DataFrame:
    """Apila varios `EvaluationResult` en la tabla maestra comparativa de results/.

    Una fila por detector, columnas = métricas. Es el artefacto central de la
    FASE 4 (síntesis comparativa).
    """
    raise NotImplementedError
