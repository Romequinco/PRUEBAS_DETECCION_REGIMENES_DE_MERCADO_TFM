"""
jump_model.py — D9 `jump_model` (FASE 3, Tanda 4 — EXPLORATORIA).

Statistical Jump Model (SJM) de Nystrup, Lindström & Madsen (2020), "Learning
Hidden Markov Models with Persistent States by Penalizing Jumps".

Idea
----
Un SJM es un clustering de estados (como k-means/GMM) PERO con una **penalización
de salto** λ que castiga cambiar de estado entre t y t+1. El ajuste minimiza, por
descenso por coordenadas:

    sum_t  dist(x_t, centro_{s_t})  +  λ · sum_t  1[s_t != s_{t-1}]

resuelto por programación dinámica sobre la secuencia de estados (dado los
centroides) alternando con el recálculo de centroides. λ introduce una
**histéresis aprendida** → persistencia, online, anti-flickering. Es el **rival
honesto de D3** (`clustering_gmm`, GMM sin término temporal) y de D12 (autoencoder):
mismas 15 features causales, misma evaluación walk-forward. La diferencia esperada
respecto a D3 es MENOS flickering (mayor duración media, menor switching_rate) sin
perder cobertura de crisis, gracias a λ.

Vía de implementación
---------------------
Se usa la librería **`jumpmodels`** (Nystrup et al., `JumpModel`), que implementa
el coordinate-descent + DP estándar y expone métodos CAUSALES:
  - `predict`            : asignación con DP sobre TODA la ventana (no causal intra-bloque).
  - `predict_online`     : asignación causal — la etiqueta de la fila i usa solo filas < i.
  - `predict_proba_online`: idem en probabilidades.
En walk-forward se usan los métodos *online* para no mirar el futuro del bloque.

Escalado de features (CAUSAL, train-only)
-----------------------------------------
El SJM mide distancias EUCLÍDEAS, sensibles a la escala. De las 15 features, 12 son
z-scores causales (std≈1) pero `corr_spx_bond`, `SP500_drawdown` y `SP500_momentum`
tienen escalas mucho menores (std≈0.1–0.3) y quedarían infra-ponderadas. Se aplica
un `StandardScaler` ajustado **solo con el train** dentro de `fit` (estadísticos del
pasado → causal; NO usa la muestra completa) y se reutiliza en predicción. Esto pone
las 15 features en escala comparable, como recomienda la propia librería.

Causalidad: `fit` solo ve `X_train`; el wrapper causal lo provee
`evaluation.walk_forward` (re-fit expanding por fold). El alineado de etiquetas entre
folds lo resuelve la canonicalización económica (`label_states_economically`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.detector_base import RegimeDetector


class JumpModel(RegimeDetector):
    """Statistical Jump Model (Nystrup et al., 2020) como detector de régimen.

    Clustering de estados con penalización de salto λ → persistencia aprendida.
    Detector de etiqueta dura (proba one-hot causal). No expone logL → AIC/BIC NaN
    (igual que las reglas y el clustering duro).

    Parameters
    ----------
    n_states : int
        Nº de estados/regímenes (2 = bull/bear clásico de Nystrup; 3 disponible).
    jump_penalty : float
        Penalización de salto λ. λ=0 ⇒ k-means puro (sin persistencia). Mayor λ ⇒
        más histéresis (episodios más largos, menos flickering). Sobre features
        estandarizadas, λ≈50 da persistencia mensual sin congelar la señal.
    random_state : int | None
        Semilla (n_init múltiples arranques del coordinate descent).
    """

    def __init__(
        self,
        n_states: int = 2,
        jump_penalty: float = 50.0,
        random_state: int | None = 42,
    ) -> None:
        super().__init__(n_states=n_states, random_state=random_state)
        self.jump_penalty = float(jump_penalty)
        self._model = None
        self._scaler: StandardScaler | None = None

    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return f"jump_model_k{self.n_states}_lam{int(self.jump_penalty)}"

    @property
    def bibliography(self) -> list[str]:
        # Claves verificadas en docs/references.bib.
        return [
            "hmm_nystrup2020",  # Statistical Jump Model: penalizar saltos => estados persistentes
            "hmm_nystrup2017",  # regularización temporal de regímenes (línea previa)
            "clust_munnix2012",  # estados de mercado por clustering (fundacional)
        ]

    # ------------------------------------------------------------------ #
    def fit(self, X_train) -> "JumpModel":
        from jumpmodels.jump import JumpModel as _JM

        # Escalado CAUSAL (estadísticos del train; el wrapper walk-forward re-fitea
        # por fold expanding, así que nunca ve el futuro del bloque a predecir).
        self._scaler = StandardScaler().fit(X_train.values)
        Xs = pd.DataFrame(
            self._scaler.transform(X_train.values),
            index=X_train.index,
            columns=X_train.columns,
        )
        self._model = _JM(
            n_components=self.n_states,
            jump_penalty=self.jump_penalty,
            cont=False,            # SJM discreto (estados duros)
            random_state=self.random_state,
            n_init=10,
            max_iter=1000,
        )
        # sort_by=None: el orden interno es arbitrario; la canonicalización económica
        # (vol-primario, Arreglo 4) la fija label_states_economically más abajo.
        self._model.fit(Xs, sort_by=None)
        self._is_fitted = True
        self.label_states_economically(X_train)
        return self

    # ------------------------------------------------------------------ #
    def _scale(self, X) -> pd.DataFrame:
        return pd.DataFrame(
            self._scaler.transform(X.values), index=X.index, columns=X.columns
        )

    def _predict_states(self, X) -> np.ndarray:
        """Etiquetas INTERNAS (DP sobre toda la ventana). Para uso in-sample /
        fijación del orden canónico sobre el train. NO usar en evaluación causal
        del bloque: para eso está `predict_online`."""
        return np.asarray(self._model.predict(self._scale(X)))

    def predict_online(self, X, refit: bool = False) -> np.ndarray:
        """Predicción CAUSAL: la etiqueta de la fila i usa solo filas < i del bloque.

        Sobrescribe la base (que delegaría en `predict`, DP no causal intra-bloque).
        """
        self._check_fitted()
        raw = np.asarray(self._model.predict_online(self._scale(X)))
        return self._apply_canonical(raw)

    def predict_proba(self, X) -> np.ndarray:
        """Probabilidades por estado, **online causal**, reordenadas al orden canónico.

        El SJM discreto devuelve proba one-hot (estado más cercano penalizado);
        usamos la versión online para que `p_crisis` del walk-forward sea causal.
        """
        self._check_fitted()
        raw = np.asarray(self._model.predict_proba_online(self._scale(X)))
        if self._canonical_order is None:
            return raw
        return raw[:, self._canonical_order]
