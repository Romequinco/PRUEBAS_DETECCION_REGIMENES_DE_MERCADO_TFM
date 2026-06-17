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

import warnings
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
# CALCULADOS en FASE 1 desde la serie real del S&P 500 (no a mano):
# fecha del mínimo del drawdown (precio/máx_expanding - 1) dentro de cada
# episodio. Drawdown alcanzado entre paréntesis. Ver docs/memory/01_data_and_eda.md
# y notebooks/00_eda.ipynb (celda de cálculo de troughs).
DRAWDOWN_TROUGHS: dict[str, str] = {
    "GFC_2008": "2009-03-09",       # -56.8%
    "EuroDebt_2011": "2011-10-03",  # -29.8%
    "COVID_2020": "2020-03-23",     # -33.9%
    "Inflation_2022": "2022-10-12", # -25.4%
}
# Referencia fuera de la ventana común del set ampliado (datos completos desde
# 2007-04-11, por inicio de HYG): el crash DotCom tocó suelo el 2002-10-09
# (-49.1%). Solo evaluable por detectores que usen un subconjunto de features con
# histórico más largo (p. ej. solo S&P 500 + VIX, disponibles desde 1990).


def _in_any_window(dates: pd.DatetimeIndex, windows: dict[str, tuple[str, str]]) -> np.ndarray:
    """Máscara booleana: True si la fecha cae dentro de alguna ventana [a, b]."""
    mask = np.zeros(len(dates), dtype=bool)
    for a, b in windows.values():
        mask |= (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))
    return mask


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
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
        }
        # Métricas por ventana (claves constantes -> esquema estable).
        for k in CRISIS_WINDOWS:
            row[f"cov_{k}"] = self.crisis_coverage.get(k, float("nan"))
        for k in FALSE_POSITIVE_WINDOWS:
            row[f"fa_{k}"] = self.false_alarm_in_fp.get(k, float("nan"))
        for k in DRAWDOWN_TROUGHS:
            row[f"leadlag_{k}"] = self.lead_lag_days.get(k, float("nan"))
        return row


# --------------------------------------------------------------------------- #
# Protocolo walk-forward (causal)
# --------------------------------------------------------------------------- #
def walk_forward(
    detector_factory,
    X: pd.DataFrame,
    *,
    market_returns: pd.Series | None = None,
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
    records: list[tuple] = []       # CAUSAL OOS: 1 fila/fecha; lo leen TODAS las métricas
    stab_records: list[tuple] = []  # DIAGNÓSTICO (no causal): solo para label_stability
    prev_test: pd.DataFrame | None = None
    fold_id = 0
    t = first_split
    while t < n:
        train = X.iloc[:t] if expanding else X.iloc[max(0, t - train_size):t]
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
        states = np.asarray(det.predict_online(test))
        proba = det.predict_proba(test)
        p_crisis = proba[:, det.crisis_state]
        for d, s, p in zip(test.index, states, p_crisis):
            records.append((d, int(s), float(p), fold_id))
        # Estabilidad: re-predecir el bloque PREVIO con este modelo (más datos).
        if prev_test is not None and len(prev_test):
            s_prev = np.asarray(det.predict_online(prev_test))
            for d, s in zip(prev_test.index, s_prev):
                stab_records.append((d, fold_id, int(s)))
        # Y registrar el bloque actual bajo su propio fold (para comparar).
        for d, s in zip(test.index, states):
            stab_records.append((d, fold_id, int(s)))
        prev_test = test
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
# Métricas individuales (todas causales: operan sobre etiquetas walk-forward)
# --------------------------------------------------------------------------- #
def crisis_coverage(
    states: pd.Series, crisis_state: int, windows: dict[str, tuple[str, str]] = None
) -> dict[str, float]:
    """% de días etiquetados como 'crisis' dentro de cada ventana de crisis conocida.

    Mide sensibilidad: idealmente alto (cercano a 1) en 2008/2011/2020/2022. Si la
    ventana queda fuera del rango out-of-sample del detector, devuelve NaN para esa
    ventana (no se penaliza lo que no se pudo ver).
    """
    windows = windows or CRISIS_WINDOWS
    out: dict[str, float] = {}
    for name, (a, b) in windows.items():
        seg = states.loc[(states.index >= pd.Timestamp(a)) & (states.index <= pd.Timestamp(b))]
        out[name] = float((seg == crisis_state).mean()) if len(seg) else float("nan")
    return out


def false_alarm_in_windows(
    states: pd.Series, crisis_state: int, windows: dict[str, tuple[str, str]] = None
) -> dict[str, float]:
    """% de días 'crisis' dentro de ventanas que NO son crisis (2013, 2018).

    Mide especificidad en episodios trampa: idealmente bajo.
    """
    windows = windows or FALSE_POSITIVE_WINDOWS
    out: dict[str, float] = {}
    for name, (a, b) in windows.items():
        seg = states.loc[(states.index >= pd.Timestamp(a)) & (states.index <= pd.Timestamp(b))]
        out[name] = float((seg == crisis_state).mean()) if len(seg) else float("nan")
    return out


def false_alarm_rate(
    states: pd.Series, crisis_state: int, crisis_windows: dict[str, tuple[str, str]] = None
) -> float:
    """Tasa global de falsas alarmas: fracción de días marcados 'crisis' que caen
    FUERA de todas las ventanas de crisis conocidas.

    Aproxima 1 - precisión usando las ventanas conocidas como ground truth laxo.
    NaN si el detector nunca marca crisis (denominador 0).
    """
    crisis_windows = crisis_windows or CRISIS_WINDOWS
    is_crisis = (states == crisis_state).values
    n_crisis = int(is_crisis.sum())
    if n_crisis == 0:
        return float("nan")
    in_crisis_win = _in_any_window(states.index, crisis_windows)
    false_alarms = int((is_crisis & ~in_crisis_win).sum())
    return false_alarms / n_crisis


def _first_sustained(mask: np.ndarray, persist: int) -> int | None:
    """Índice del inicio de la PRIMERA racha de >= `persist` True consecutivos."""
    if persist <= 1:
        nz = np.flatnonzero(mask)
        return int(nz[0]) if nz.size else None
    count = 0
    for i, v in enumerate(mask):
        count = count + 1 if v else 0
        if count >= persist:
            return i - persist + 1
    return None


def lead_lag(
    p_crisis: pd.Series,
    troughs: dict[str, str] = None,
    threshold: float = 0.5,
    persist: int = 3,
    lookback: int = 252,
) -> dict[str, float]:
    """Días de adelanto/retraso entre la señal SOSTENIDA de crisis y el suelo del
    drawdown del S&P 500, por evento.

    Para cada trough busca, en los `lookback` días previos al suelo, el primer día
    en que `p_crisis` cruza `threshold` y SE MANTIENE >= `persist` días
    consecutivos (coherente con el gatillo de cambio de régimen del TFM: 3 días).
    Exigir persistencia evita premiar el flickering: un detector ruidoso que cruza
    el umbral un día suelto por azar ya NO cuenta como "anticipador".

    Devuelve (fecha_señal_sostenida - fecha_trough) en días de trading (posiciones
    del índice OOS): negativo = la señal ANTICIPA el suelo, positivo = va por
    detrás. NaN si el trough cae fuera del rango OOS o no hay señal sostenida en la
    ventana previa.
    """
    troughs = troughs or DRAWDOWN_TROUGHS
    out: dict[str, float] = {}
    idx = p_crisis.index
    for name, tdate in troughs.items():
        T = pd.Timestamp(tdate)
        if T < idx.min() or T > idx.max():
            out[name] = float("nan")
            continue
        # posición del trough (o el día OOS inmediatamente anterior)
        pos_T = idx.searchsorted(T, side="right") - 1
        if pos_T < 0:
            out[name] = float("nan")
            continue
        lo = max(0, pos_T - lookback)
        window = p_crisis.iloc[lo:pos_T + 1]
        start = _first_sustained((window.values >= threshold), persist)
        if start is None:
            out[name] = float("nan")
            continue
        pos_signal = idx.searchsorted(window.index[start], side="left")
        out[name] = float(pos_signal - pos_T)  # negativo = anticipa
    return out


def switching_rate(states: pd.Series) -> float:
    """Frecuencia de conmutación = nº de cambios de estado / nº de días.

    Penaliza el 'flickering' (regímenes que parpadean día a día). Más bajo =
    más persistente/estable.
    """
    if len(states) < 2:
        return float("nan")
    changes = int((states.values[1:] != states.values[:-1]).sum())
    return changes / len(states)


def mean_regime_duration(states: pd.Series) -> float:
    """Duración media (en días) de los episodios de régimen. Inverso del flicker."""
    if len(states) == 0:
        return float("nan")
    v = states.values
    n_runs = 1 + int((v[1:] != v[:-1]).sum())
    return len(v) / n_runs


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
        sucesivas del walk-forward. Para cada fecha con >=2 reestimaciones se
        calcula la fracción de la moda (acuerdo); se promedia sobre esas fechas.
    """
    if panel is None or panel.empty:
        return float("nan")
    agreements = []
    for _, rowvals in panel.iterrows():
        vals = rowvals.dropna().values
        if len(vals) < 2:
            continue
        _, counts = np.unique(vals, return_counts=True)
        agreements.append(counts.max() / len(vals))
    return float(np.mean(agreements)) if agreements else float("nan")


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
        false_alarm_in_fp=false_alarm_in_windows(states, cs),
        lead_lag_days=lead_lag(p_crisis),
        false_alarm_rate=false_alarm_rate(states, cs),
        switching_rate=switching_rate(states),
        mean_regime_duration=mean_regime_duration(states),
        label_stability=label_stability(wf_panel.attrs.get("stability_panel")),
        n_states=detector.n_states,
    )

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
