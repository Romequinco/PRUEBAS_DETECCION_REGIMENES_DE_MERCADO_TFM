"""sparse_jump.py — D18 · Sparse jump model de Nystrup, Kolm y Lindström (2021) (ADR-010 §3).

Jump model con selección de variables: además de los centroides y la secuencia de estados,
aprende un peso ``w_j >= 0`` por columna con una restricción L1 (``||w||_1 <= κ``), de modo que
las columnas que no separan los estados quedan con peso 0. Se usa
``jumpmodels.SparseJumpModel`` (v0.1.1) sobre el menú P:

- ``n_components = 2``, ``jump_penalty = 50`` (la de D09; la librería la divide internamente por
  ``√p``), ``max_feats = 9`` en A y 14 en B (el tamaño de N). ``max_feats`` es κ², el número
  **efectivo** de variables, no un número exacto: :meth:`SparseJumpPool.diagnostico` devuelve
  las columnas con peso > 0 de cada reajuste.
- Igual que D09 (``detectores/f2_clustering/jump_model.py``, que no se toca): escalado
  ``StandardScaler`` ajustado solo con el train, ``sort_by=None`` (orden interno arbitrario),
  canonicalización económica de ``base.py`` (la re-fija el walk-forward con ``SP500_ret``) y
  predicción **online** causal (``predict_online`` / ``predict_proba_online``).
- Semilla 42 y ``n_init_jm = 10`` arranques del jump model interno (el ``n_init`` de D09).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from regimenes.detectores.base import RegimeDetector


class SparseJumpPool(RegimeDetector):
    """D18 · ``jumpmodels.SparseJumpModel`` como detector de régimen (ver el docstring del módulo)."""

    def __init__(self, n_states: int = 2, jump_penalty: float = 50.0, max_feats: float = 9.0,
                 n_init_jm: int = 10, max_iter: int = 30, max_iter_jm: int = 1000,
                 random_state: int | None = 42) -> None:
        super().__init__(n_states=n_states, random_state=random_state)
        self.jump_penalty = float(jump_penalty)
        self.max_feats = float(max_feats)
        self.n_init_jm = int(n_init_jm)
        self.max_iter = int(max_iter)
        self.max_iter_jm = int(max_iter_jm)
        self._model = None
        self._scaler: StandardScaler | None = None
        self._columnas: list[str] = []

    @property
    def name(self) -> str:
        return f"sparse_jump_k{self.n_states}_lam{int(self.jump_penalty)}_kappa2_{self.max_feats:g}"

    @property
    def bibliography(self) -> list[str]:
        return ["nystrup2021", "hmm_nystrup2020"]

    # ------------------------------------------------------------------ #
    def _scale(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self._scaler.transform(X[self._columnas].to_numpy(dtype=float)),
                            index=X.index, columns=self._columnas)

    def fit(self, X_train: pd.DataFrame) -> "SparseJumpPool":
        from jumpmodels.sparse_jump import SparseJumpModel

        self._columnas = list(X_train.columns)
        self._scaler = StandardScaler().fit(X_train.to_numpy(dtype=float))
        Xs = self._scale(X_train)
        self._model = SparseJumpModel(
            n_components=self.n_states, max_feats=self.max_feats, jump_penalty=self.jump_penalty,
            cont=False, random_state=self.random_state, max_iter=self.max_iter,
            max_iter_jm=self.max_iter_jm, n_init_jm=self.n_init_jm,
        )
        self._model.fit(Xs, sort_by=None)
        self._is_fitted = True
        self.label_states_economically(X_train)
        return self

    def pesos(self) -> pd.Series:
        """Pesos de las columnas (``feat_weights`` = √w de la formulación original)."""
        self._check_fitted()
        return pd.Series(np.asarray(self._model.feat_weights, dtype=float), index=self._columnas)

    def diagnostico(self) -> dict[str, Any]:
        """Columnas con peso > 0 y sus pesos (para el informe del ADR-010 §5)."""
        w = self.pesos()
        pos = w[w > 0].sort_values(ascending=False)
        return {"d18_n_peso_pos": int(len(pos)), "d18_columnas_peso_pos": "|".join(pos.index),
                "d18_pesos": "|".join(f"{c}={v:.6g}" for c, v in pos.items())}

    # ------------------------------------------------------------------ #
    def _predict_states(self, X: pd.DataFrame) -> np.ndarray:
        """Etiquetas internas con DP sobre toda la ventana (solo para fijar el orden canónico)."""
        return np.asarray(self._model.predict(self._scale(X)))

    def predict_online(self, X: pd.DataFrame, refit: bool = False) -> np.ndarray:
        self._check_fitted()
        return self._apply_canonical(np.asarray(self._model.predict_online(self._scale(X))))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        raw = np.asarray(self._model.predict_proba_online(self._scale(X)))
        return raw if self._canonical_order is None else raw[:, self._canonical_order]
