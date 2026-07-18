"""
rule_composite_riskoff.py — D2 (FASE 3, Tanda 2). Familia F1: Reglas / Umbrales.

Regla COMPUESTA causal de 2 estados (0=calma, 1=crisis) que combina varias señales
de estrés ya disponibles en `data/processed/features.parquet` en un SCORE de
"risk-off" y lo umbraliza con HISTÉRESIS (banda muerta τ_in/τ_out) + DWELL-TIME
mínimo, igual que D1 pero sobre un voto multivariante en vez de un único nivel de
VIX.

Idea
----
Una crisis sistémica no se ve solo en el "miedo" (VIX): suele coincidir un repunte
de volatilidad equity, un deterioro del crédito (high-yield cae frente a treasuries)
y/o una señal de curva. D2 agrega 4 señales causales en un único score de estrés
orientado (ALTO = risk-off) y aplica el mismo autómata de histéresis de D1:

    - Entra en crisis cuando el score cruza τ_in POR ARRIBA.
    - Sale solo cuando baja de τ_out (< τ_in) Y se cumplen `min_dwell` días.

Señales y ORIENTACIÓN del estrés (signo con el que entran al score)
------------------------------------------------------------------
- `VIX_level_z`     (+): miedo equity; ALTO = estrés (reglas_bloom2009).
- `credit_spread_z` (−): proxy de crédito = ret(HYG) − ret(IEF). Cuando el crédito
  se deteriora, HYG cae frente a treasuries → este z se vuelve NEGATIVO. Por eso
  entra con signo − (Gilchrist-Zakrajšek 2012: el crédito anticipa el ciclo).
- `yield_slope_z`   (−): pendiente 10Y−3M. Curva BAJA/invertida = estrés
  (Estrella-Mishkin 1998); signo INVERTIDO respecto a las demás. OJO: en 2008/2011
  la curva se EMPINÓ (Fed recortando el corto), así que esta señal es de aviso
  ADELANTADO (predice recesión) más que contemporánea — fuente de fricción de pesos.
- `SP500_drawdown`  (−): drawdown corriente ∈[−1,0]; más NEGATIVO = estrés
  (kritzman2012, turbulencia / riesgo sistémico).

Causalidad
----------
Cada señal se orienta (sign·valor) y se RE-ESTANDARIZA con media/desv del TRAIN
(causal, nunca con toda la muestra) para que las 4 sean comparables y los pesos
tengan sentido (el drawdown vive en [−1,0], las otras son z≈unitarias). El score es
la media ponderada de esas 4 z orientadas. Los umbrales τ_in/τ_out son percentiles
(`q_in`,`q_out`) del score SOLO en el train. En walk-forward cada fold recalcula
sus medias/desv y sus cortes con su propio train: sin look-ahead. El autómata es
secuencial pero causal (estado en t depende solo de t−1 y del score en t).

Ventana
-------
Las señales de crédito (HYG) y curva viven en `features.parquet` desde 2007-07. Con
`train_size≈8 años`, el primer bloque OOS cae hacia 2015, de modo que **2008 y 2011
quedan dentro del train inicial → cobertura NaN out-of-sample** (correcto y
declarado): D2 solo puede evaluarse OOS sobre 2020 y 2022 (+ trampas 2013/2018 caen
también en el train inicial). market_returns = retorno log del S&P 500.

Bibliografía
------------
reglas_gilchristzakrajsek2012 — spreads de crédito como predictor del ciclo.
estrellamishkin1998           — pendiente de la curva como señal de recesión.
reglas_bloom2009              — shocks de incertidumbre / volatilidad.
kritzman2012                  — turbulencia / regímenes de riesgo sistémico.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.detector_base import RegimeDetector

# Señales del score y SIGNO de orientación (sign·valor de modo que ALTO = estrés).
_DEFAULT_SIGNS: dict[str, int] = {
    "VIX_level_z": +1,
    "credit_spread_z": -1,
    "yield_slope_z": -1,
    "SP500_drawdown": -1,
}


class RuleCompositeRiskoff(RegimeDetector):
    """Regla compuesta de risk-off: score multivariante con histéresis + dwell.

    Parameters
    ----------
    weights : dict[str, float] | None
        Peso de cada señal en el score (media ponderada). None = pesos iguales.
        Las claves deben ser un subconjunto de las señales orientadas.
    q_in : float
        Percentil (sobre el score del train) que define τ_in, umbral de ENTRADA.
    q_out : float
        Percentil que define τ_out, umbral de SALIDA. Debe ser < q_in (histéresis).
    min_dwell : int
        Nº mínimo de días en crisis antes de poder salir (anti-flickering).
    signs : dict[str, int] | None
        Orientación de cada señal (+1/−1). None = `_DEFAULT_SIGNS`.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        q_in: float = 0.90,
        q_out: float = 0.70,
        min_dwell: int = 5,
        signs: dict[str, int] | None = None,
        random_state: int | None = 42,
    ) -> None:
        super().__init__(n_states=2, random_state=random_state)
        if not (0.0 < q_out < q_in < 1.0):
            raise ValueError(
                f"Se requiere 0 < q_out < q_in < 1 (histéresis); recibido "
                f"q_out={q_out}, q_in={q_in}."
            )
        self.signs = dict(signs) if signs is not None else dict(_DEFAULT_SIGNS)
        self.features = list(self.signs.keys())
        if weights is None:
            self.weights = {f: 1.0 for f in self.features}
        else:
            self.weights = {f: float(weights.get(f, 0.0)) for f in self.features}
        wsum = sum(self.weights.values())
        if wsum <= 0:
            raise ValueError("La suma de pesos debe ser > 0.")
        self.q_in = float(q_in)
        self.q_out = float(q_out)
        self.min_dwell = int(min_dwell)
        # Estadísticos causales fijados en fit:
        self._mu: dict[str, float] = {}
        self._sigma: dict[str, float] = {}
        self._tau_in: float | None = None
        self._tau_out: float | None = None

    # ------------------------------------------------------------------ #
    # Identidad
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return "rule_composite_riskoff"

    @property
    def bibliography(self) -> list[str]:
        return [
            "reglas_gilchristzakrajsek2012",
            "estrellamishkin1998",
            "reglas_bloom2009",
            "kritzman2012",
        ]

    # ------------------------------------------------------------------ #
    # Score compuesto (orientado, re-estandarizado causalmente)
    # ------------------------------------------------------------------ #
    def _oriented(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """sign·valor de cada señal (ALTO = estrés), sin estandarizar."""
        out = {}
        for f in self.features:
            if f not in X.columns:
                raise KeyError(
                    f"{self.name}: falta la columna '{f}' en X "
                    f"(columnas: {list(X.columns)})."
                )
            out[f] = self.signs[f] * X[f].values.astype(float)
        return out

    def composite_score(self, X: pd.DataFrame) -> np.ndarray:
        """Score de risk-off = media ponderada de las z orientadas (causal).

        Cada señal se re-estandariza con μ/σ del TRAIN (fijados en fit) para que
        sean comparables; luego se promedia con `weights`. NaN-safe por señal.
        """
        if not self._mu:
            raise RuntimeError(f"{self.name}: llama a fit() antes de calcular el score.")
        oriented = self._oriented(X)
        n = len(X)
        num = np.zeros(n, dtype=float)
        den = np.zeros(n, dtype=float)
        for f in self.features:
            sig = self._sigma[f] if self._sigma[f] > 0 else 1.0
            z = (oriented[f] - self._mu[f]) / sig
            w = self.weights[f]
            valid = ~np.isnan(z)
            num[valid] += w * z[valid]
            den[valid] += w
        score = np.full(n, np.nan, dtype=float)
        nz = den > 0
        score[nz] = num[nz] / den[nz]
        return score

    # ------------------------------------------------------------------ #
    # Ajuste
    # ------------------------------------------------------------------ #
    def fit(self, X_train: pd.DataFrame) -> "RuleCompositeRiskoff":
        oriented = self._oriented(X_train)
        # μ/σ CAUSALES por señal (solo train).
        for f in self.features:
            v = oriented[f]
            v = v[~np.isnan(v)]
            self._mu[f] = float(np.mean(v)) if v.size else 0.0
            self._sigma[f] = float(np.std(v)) if v.size else 1.0
        self._is_fitted = True
        # Umbrales CAUSALES: percentiles del score SOLO en el train.
        score = self.composite_score(X_train)
        score = score[~np.isnan(score)]
        self._tau_in = float(np.quantile(score, self.q_in))
        self._tau_out = float(np.quantile(score, self.q_out))
        # Orden económico PROVISIONAL por construcción: el estado interno 1 (score
        # alto) ES el risk-off/crisis. walk_forward lo RE-FIJA causalmente con
        # market_returns; aquí evitamos el warning de fallback fijando identidad.
        self._canonical_order = np.array([0, 1], dtype=int)
        return self

    # ------------------------------------------------------------------ #
    # Predicción interna (sin canonicalizar) — autómata de histéresis
    # ------------------------------------------------------------------ #
    def _predict_states(self, X: pd.DataFrame) -> np.ndarray:
        """Recorre el score aplicando histéresis + dwell-time. CAUSAL.

        Estado interno: 0 = calma, 1 = crisis (score alto = varias señales
        risk-off simultáneas). Coincide con el orden económico natural.
        """
        if self._tau_in is None or self._tau_out is None:
            raise RuntimeError(f"{self.name}: llama a fit() antes de predecir.")
        s = self.composite_score(X)
        n = len(s)
        out = np.zeros(n, dtype=int)
        state = 0
        dwell = 0
        for t in range(n):
            x = s[t]
            if np.isnan(x):
                out[t] = state
                if state == 1:
                    dwell += 1
                continue
            if state == 0:
                if x > self._tau_in:
                    state = 1
                    dwell = 1
            else:
                dwell += 1
                if x < self._tau_out and dwell >= self.min_dwell:
                    state = 0
                    dwell = 0
            out[t] = state
        return out
