"""
clustering_gmm.py — D3 `clustering_gmm` (FASE 3, familia CLUSTERING).

Baseline NO temporal: una mixtura gaussiana estática (sin cadena de Markov sobre
los estados). Cada día se asigna de forma INDEPENDIENTE al componente más
probable → no hay término de persistencia, así que se ESPERA flickering alto.
Justamente por eso es el baseline contra el que se mide cuánto aporta la dinámica
temporal del HMM (D4): mismas 15 features causales, misma evaluación walk-forward.

Elección de diseño:
- `covariance_type='full'`: cada componente tiene su matriz de covarianza plena,
  lo que permite separar regímenes con ESTRUCTURA DE CORRELACIÓN distinta —en
  particular el cambio de signo de la correlación S&P500/bonos (Gulko, 2002;
  feature `corr_spx_bond`). Two Sigma (2021) usa un GMM análogo sobre factores.
- Selección de nº de estados por BIC (se prueban k=2 y k=3 en el notebook).

Causalidad: el detector solo se ajusta sobre `X_train`; el "wrapper causal" lo
provee `evaluation.walk_forward` (re-fit expanding por fold). El alineado de
etiquetas entre folds lo resuelve la canonicalización económica
(`label_states_economically` en cada `fit`).
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from src.detector_base import RegimeDetector


class ClusteringGMM(RegimeDetector):
    """Mixtura gaussiana estática (GMM) como detector de régimen NO temporal.

    Detector PROBABILÍSTICO: sobreescribe `predict_proba` para devolver los
    posteriores reales del GMM, reordenados al orden canónico económico
    (0=calma .. n-1=crisis). Expone `score`/`n_parameters` para que el núcleo
    calcule AIC/BIC.
    """

    def __init__(self, n_states: int = 2, random_state: int | None = 42) -> None:
        super().__init__(n_states=n_states, random_state=random_state)
        self._model: GaussianMixture | None = None

    @property
    def name(self) -> str:
        return f"clustering_gmm_k{self.n_states}"

    @property
    def bibliography(self) -> list[str]:
        # Claves reales verificadas en docs/references.bib.
        return [
            "clust_twosigma2021regime",  # GMM estático sobre factores -> ~4 regímenes
            "clust_munnix2012",          # estados de mercado por clustering (fundacional)
            "gulko2002",                 # decoupling: la correlación equity/bonos cambia de régimen
            "lopezdeprado2018",          # ML financiero / metodología clustering
        ]

    # ------------------------------------------------------------------ #
    def fit(self, X_train) -> "ClusteringGMM":
        self._model = GaussianMixture(
            n_components=self.n_states,
            covariance_type="full",
            random_state=self.random_state,
            n_init=5,
            reg_covar=1e-6,
            max_iter=300,
        ).fit(X_train.values)
        self._is_fitted = True
        # Fija self._canonical_order por criterio económico (retorno/vol por estado).
        self.label_states_economically(X_train)
        return self

    def _predict_states(self, X) -> np.ndarray:
        """Etiquetas INTERNAS del GMM (sin canonicalizar)."""
        return self._model.predict(X.values)

    def predict_proba(self, X) -> np.ndarray:
        """Posteriores reales del GMM, reordenados al orden CANÓNICO.

        `_model.predict_proba` devuelve columnas en orden interno; permutamos para
        que la columna canónica i tome la interna `self._canonical_order[i]`
        (0=calma .. n-1=crisis). Así `crisis_probability` = última columna.
        """
        self._check_fitted()
        raw = self._model.predict_proba(X.values)  # (n, k) en orden interno
        if self._canonical_order is None:
            return raw
        return raw[:, self._canonical_order]

    def score(self, X) -> float:
        """Log-likelihood TOTAL sobre X (sklearn da la media -> multiplico por n)."""
        self._check_fitted()
        return float(self._model.score(X.values) * len(X))

    def n_parameters(self) -> int:
        """Parámetros libres con covarianza full: pesos + medias + covarianzas.

        k-1 (pesos, restricción suma=1) + k*d (medias) + k*d*(d+1)/2 (covar full).
        """
        k = self.n_states
        d = self._model.means_.shape[1]
        return int((k - 1) + k * d + k * d * (d + 1) / 2)
