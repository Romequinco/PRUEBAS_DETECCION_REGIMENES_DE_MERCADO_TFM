"""
rule_vix_threshold.py — D1 (FASE 3, Tanda 1). Familia F1: Reglas / Umbrales.

Detector causal de régimen basado en el NIVEL del VIX con HISTÉRESIS (banda muerta
τ_in/τ_out) y DWELL-TIME mínimo, para evitar el flickering típico de un umbral
simple. Dos estados: 0 = calma, 1 = crisis.

Idea
----
El VIX es el "índice del miedo": su nivel salta el mismo día que el mercado se
estresa, así que un umbral reactivo sobre el nivel captura crisis sin retardo
(reglas_bloom2009). El problema de un umbral simple es que parpadea cuando el VIX
oscila alrededor del corte. La histéresis lo arregla con dos umbrales:

    - Entra en crisis cuando el nivel cruza τ_in POR ARRIBA.
    - Sale de crisis solo cuando baja de τ_out (< τ_in) Y se han cumplido al menos
      `min_dwell` días en crisis (banda muerta + permanencia mínima).

Esto es un autómata de 2 estados recorrido secuencialmente, pero CAUSAL: el estado
en t depende solo del estado en t-1 y del valor del VIX en t. No hay look-ahead.

Causalidad de los umbrales
--------------------------
Los umbrales τ_in/τ_out se fijan en `fit` como percentiles (`q_in`, `q_out`) del
VIX z-scoreado SOLO en el tramo de train. El z-score de entrada (`VIX_level_z`) ya
es causal (expanding, ver features.causal_zscore). En walk-forward cada fold
recalcula sus percentiles con su propio train, así que nunca se miran datos
futuros para calibrar el corte.

Ventana
-------
A diferencia de la mayoría de detectores (atados a features.parquet, que empieza en
2007 por HYG), D1 PUEDE y DEBE usar histórico largo: el VIX existe desde 1990. La
feature se construye en el notebook directamente desde el panel crudo largo, de
modo que el walk-forward cubre 2008 y 2011 out-of-sample.

Bibliografía
------------
reglas_bloom2009         — shocks de incertidumbre / volatilidad como señal de régimen.
reglas_moreiramuir2017   — managed volatility: escalar exposición con la vol (motiva el umbral sobre vol).
kritzman2012             — turbulence / regímenes de riesgo sistémico.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.detector_base import RegimeDetector


class RuleVixThreshold(RegimeDetector):
    """Regla de umbral sobre el nivel de VIX con histéresis y dwell-time.

    Parameters
    ----------
    q_in : float
        Percentil (sobre el VIX z del train) que define τ_in, el umbral de ENTRADA
        a crisis. P. ej. 0.90 = entra cuando el VIX supera su percentil 90 del train.
    q_out : float
        Percentil que define τ_out, el umbral de SALIDA. Debe ser < q_in para crear
        la banda muerta (histéresis). P. ej. 0.70.
    min_dwell : int
        Nº mínimo de días que el estado debe permanecer en crisis antes de poder
        salir (permanencia mínima, anti-flickering).
    feature : str
        Columna de X usada como nivel de VIX z-scoreado causal.
    """

    def __init__(
        self,
        q_in: float = 0.90,
        q_out: float = 0.70,
        min_dwell: int = 5,
        feature: str = "VIX_level_z",
        random_state: int | None = 42,
    ) -> None:
        super().__init__(n_states=2, random_state=random_state)
        if not (0.0 < q_out < q_in < 1.0):
            raise ValueError(
                f"Se requiere 0 < q_out < q_in < 1 (histéresis); recibido "
                f"q_out={q_out}, q_in={q_in}."
            )
        self.q_in = float(q_in)
        self.q_out = float(q_out)
        self.min_dwell = int(min_dwell)
        self.feature = str(feature)
        self._tau_in: float | None = None
        self._tau_out: float | None = None

    # ------------------------------------------------------------------ #
    # Identidad
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return "rule_vix_threshold"

    @property
    def bibliography(self) -> list[str]:
        return ["reglas_bloom2009", "reglas_moreiramuir2017", "kritzman2012"]

    # ------------------------------------------------------------------ #
    # Ajuste
    # ------------------------------------------------------------------ #
    def fit(self, X_train: pd.DataFrame) -> "RuleVixThreshold":
        if self.feature not in X_train.columns:
            raise KeyError(
                f"{self.name}: falta la columna '{self.feature}' en X_train "
                f"(columnas: {list(X_train.columns)})."
            )
        v = X_train[self.feature].dropna().values
        # Umbrales CAUSALES: percentiles del VIX z SOLO en el train.
        self._tau_in = float(np.quantile(v, self.q_in))
        self._tau_out = float(np.quantile(v, self.q_out))
        self._is_fitted = True
        # Fija el orden económico canónico (0=calma..1=crisis) con el train.
        self.label_states_economically(X_train)
        return self

    # ------------------------------------------------------------------ #
    # Predicción interna (sin canonicalizar) — autómata de histéresis
    # ------------------------------------------------------------------ #
    def _predict_states(self, X: pd.DataFrame) -> np.ndarray:
        """Recorre la serie aplicando histéresis + dwell-time. CAUSAL.

        Estado interno: 0 = calma, 1 = crisis (coincide con el orden económico
        natural de la regla; la canonicalización lo confirma).
        """
        if self._tau_in is None or self._tau_out is None:
            raise RuntimeError(f"{self.name}: llama a fit() antes de predecir.")
        v = X[self.feature].values
        n = len(v)
        out = np.zeros(n, dtype=int)
        state = 0           # arranca en calma
        dwell = 0           # días consecutivos dentro del estado crisis
        for t in range(n):
            x = v[t]
            if np.isnan(x):
                # Sin dato: mantiene el estado previo (causal, no inventa señal).
                out[t] = state
                if state == 1:
                    dwell += 1
                continue
            if state == 0:
                # En calma: entra a crisis si cruza τ_in por arriba.
                if x > self._tau_in:
                    state = 1
                    dwell = 1
            else:
                # En crisis: solo sale si baja de τ_out Y ya cumplió el dwell mínimo.
                dwell += 1
                if x < self._tau_out and dwell >= self.min_dwell:
                    state = 0
                    dwell = 0
            out[t] = state
        return out
