"""
hmm_gaussian_2s.py — D4 (FASE 3, Tanda 1). Familia F3: HMM (emisiones latentes).

BASELINE PUENTE con la tarea previa (`Tarea_riesgos.ipynb`). Reproduce
honestamente el GaussianHMM de 2 estados con `covariance_type='full'` de la tarea
previa, pero dentro del marco causal/comparable del banco de pruebas. Su razón de
ser es MEDIR el efecto del look-ahead: el notebook compara dos versiones del mismo
modelo (in-sample no causal vs walk-forward causal).

Idea
----
Un Hidden Markov Model gaussiano modela el régimen como un estado LATENTE de una
cadena de Markov; cada estado emite las features según una gaussiana multivariante
(medias y covarianzas propias). La diagonal de la matriz de transición A da
persistencia explícita (menos flickering) y `predict_proba` da una probabilidad de
crisis blanda [hmm_rabiner1989; guidolintimmermann2007]. El supuesto gaussiano de
las emisiones subestima las colas gordas (kurt 25–40 en el EDA), por lo que se
espera que capte las crisis grandes y persistentes (2008, 2020) pero se pierda las
correcciones rápidas (taper 2013, Q4 2018) [hmm_bulla2011].

Causalidad (clave de D4) — FILTRADO FORWARD, no Viterbi intra-bloque
--------------------------------------------------------------------
hmmlearn decodifica con Viterbi y `predict_proba` con forward-backward: ambos
SUAVIZAN (la etiqueta del día t usa todo el bloque, incluidos días futuros). Eso
es look-ahead de hasta `step` días e infla artificialmente la suavidad — justo el
sesgo de la tarea previa. Por eso D4 tiene dos modos NÍTIDAMENTE separados:

  1. IN-SAMPLE (NO causal): `predict` (Viterbi) sobre TODA la muestra. Reproduce
     la tarea previa con su look-ahead. SOLO marcado como tal; no comparable.
  2. CAUSAL walk-forward (`evaluation.walk_forward`): `predict_online` y
     `predict_proba` usan FILTRADO FORWARD causal (`_hmm_utils.filtered_posterior`):
     la etiqueta/probabilidad de t usa SOLO observaciones <= t. Para que el primer
     día de cada bloque arranque con el pasado (y no desde startprob_), se antepone
     un contexto de burn-in con las filas de train anteriores al bloque. Sin
     Viterbi intra-bloque: cero look-ahead dentro del mes.

Patrón reutilizable por D8 (hmm_tstudent): misma separación predict=Viterbi
(in-sample) vs predict_online/predict_proba=filtrado forward (causal), vía
`detectors/_hmm_utils.py`.

Los parámetros (medias, covarianzas, A) y el orden económico de estados se fijan
con el train; la predicción del bloque usa esos parámetros congelados.

Ventana (versión puente)
------------------------
Usa el subconjunto de 7 features de la tarea previa (de las 15 de
features.parquet): SP500_ret_z, TLT_ret_z, IEF_ret_z, HYG_ret_z, SP500_vol_z,
credit_spread_z, VIX_level_z. Ventana 2007+. Con walk-forward (train inicial de
varios años) 2008/2011 caen DENTRO del train → no son OOS (coberturas NaN en la
versión causal); eso es correcto y es justo el punto a documentar.

Bibliografía
------------
hamilton1989            — base del régimen como estado latente markoviano.
hmm_rabiner1989         — tutorial fundacional de HMM (forward-backward, Viterbi).
hmm_bulla2011           — HMM con componentes t: motiva por qué el gaussiano falla en colas.
guidolintimmermann2007  — regímenes en retornos de activos vía HMM/Markov-switching.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from detectors._hmm_utils import filtered_posterior
from src.detector_base import RegimeDetector

# Subconjunto de features de la tarea previa (orden estable).
BRIDGE_FEATURES = [
    "SP500_ret_z",
    "TLT_ret_z",
    "IEF_ret_z",
    "HYG_ret_z",
    "SP500_vol_z",
    "credit_spread_z",
    "VIX_level_z",
]


class HMMGaussian2S(RegimeDetector):
    """Gaussian HMM de K estados (por defecto 2), covarianza full, puente previo.

    Parameters
    ----------
    n_states : int
        Número de estados latentes (2 = calma/crisis, baseline puente).
    n_init : int
        Nº de inicializaciones aleatorias (semillas) entre las que se elige la de
        mayor log-verosimilitud (mitiga óptimos locales del EM), como en la tarea
        previa.
    n_iter : int
        Iteraciones máximas de Baum-Welch (EM) por inicialización.
    features : list[str] | None
        Columnas a usar. Si None, usa BRIDGE_FEATURES si están todas presentes;
        en caso contrario usa todas las columnas de X.
    random_state : int
        Semilla base; se prueban random_state .. random_state+n_init-1.
    """

    def __init__(
        self,
        n_states: int = 2,
        n_init: int = 5,
        n_iter: int = 200,
        features: list[str] | None = None,
        random_state: int | None = 42,
    ) -> None:
        super().__init__(n_states=n_states, random_state=random_state)
        self.n_init = int(n_init)
        self.n_iter = int(n_iter)
        self.features = list(features) if features is not None else None
        self._model: GaussianHMM | None = None
        self._used_features: list[str] | None = None
        self._Xtrain_sel: pd.DataFrame | None = None  # burn-in para filtrado causal

    # ------------------------------------------------------------------ #
    # Identidad
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return f"hmm_gaussian_{self.n_states}s"

    @property
    def bibliography(self) -> list[str]:
        return [
            "hamilton1989",
            "hmm_rabiner1989",
            "hmm_bulla2011",
            "guidolintimmermann2007",
        ]

    # ------------------------------------------------------------------ #
    # Utilidad: seleccionar las features de trabajo
    # ------------------------------------------------------------------ #
    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._used_features is not None:
            return X[self._used_features]
        if self.features is not None:
            cols = self.features
        elif all(c in X.columns for c in BRIDGE_FEATURES):
            cols = BRIDGE_FEATURES
        else:
            cols = list(X.columns)
        self._used_features = cols
        return X[cols]

    # ------------------------------------------------------------------ #
    # Ajuste: varias inicializaciones, se queda con la de mayor logL
    # ------------------------------------------------------------------ #
    def fit(self, X_train: pd.DataFrame) -> "HMMGaussian2S":
        Xtr = self._select(X_train)
        vals = Xtr.values
        base = self.random_state if self.random_state is not None else 0
        best = None
        best_ll = -np.inf
        for seed in range(base, base + self.n_init):
            m = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=seed,
            )
            try:
                m.fit(vals)
                ll = m.score(vals)
            except Exception:  # noqa: BLE001  (EM puede no converger en algún seed)
                continue
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = m, ll
        if best is None:
            raise RuntimeError(f"{self.name}: ninguna inicialización del HMM convergió.")
        self._model = best
        self._is_fitted = True
        self._Xtrain_sel = Xtr.copy()  # contexto de burn-in para el filtrado causal
        # Fija el orden económico canónico (0=calma..n-1=crisis) con el train.
        self.label_states_economically(Xtr)
        return self

    # ------------------------------------------------------------------ #
    # Predicción interna (sin canonicalizar) — Viterbi (modo IN-SAMPLE, no causal)
    # ------------------------------------------------------------------ #
    def _predict_states(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"{self.name}: llama a fit() antes de predecir.")
        return self._model.predict(self._select(X).values)

    # ------------------------------------------------------------------ #
    # Filtrado forward CAUSAL con contexto de burn-in (estados internos)
    # ------------------------------------------------------------------ #
    def _filtered_internal(self, X: pd.DataFrame) -> np.ndarray:
        """Posteriores filtrados P(estado_t | obs<=t), (len(X), k), orden INTERNO.

        Antepone como burn-in las filas de train estrictamente anteriores al primer
        índice de X, para que el primer día del bloque arranque con el pasado.
        """
        Xsel = self._select(X)
        if self._Xtrain_sel is not None and len(Xsel):
            ctx = self._Xtrain_sel[self._Xtrain_sel.index < Xsel.index[0]]
        else:
            ctx = (self._Xtrain_sel if self._Xtrain_sel is not None else Xsel).iloc[:0]
        full = pd.concat([ctx, Xsel])
        filt = filtered_posterior(self._model, full.values)
        return filt[len(ctx):]  # solo las filas correspondientes a X

    # ------------------------------------------------------------------ #
    # Predicción CAUSAL online (filtrado forward) — la que usa walk_forward
    # ------------------------------------------------------------------ #
    def predict_online(self, X: pd.DataFrame, refit: bool = False) -> np.ndarray:
        """Etiqueta dura CAUSAL canónica: argmax del filtrado forward (obs<=t)."""
        self._check_fitted()
        raw_states = self._filtered_internal(X).argmax(axis=1)
        return self._apply_canonical(raw_states)

    # ------------------------------------------------------------------ #
    # Probabilidades posteriores CAUSALES (filtradas) en orden CANÓNICO
    # ------------------------------------------------------------------ #
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Posteriores FILTRADOS P(estado_t | obs<=t) en orden económico canónico.

        Filtrado forward (causal), NO el forward-backward suavizado de hmmlearn.
        Así `p_crisis = predict_proba[:, crisis_state]` no mira el futuro del bloque.
        """
        self._check_fitted()
        filt = self._filtered_internal(X)
        if self._canonical_order is None:
            return filt
        # filt[:, j] = P(estado interno j). El estado canónico i corresponde al
        # estado interno self._canonical_order[i], así que reordenamos columnas.
        return filt[:, self._canonical_order]

    # ------------------------------------------------------------------ #
    # Bondad de ajuste
    # ------------------------------------------------------------------ #
    def score(self, X: pd.DataFrame) -> float:
        self._check_fitted()
        return float(self._model.score(self._select(X).values))

    def n_parameters(self) -> int:
        """Parámetros libres: transición (k²-1) + medias (k·d) + covarianzas full
        (k·d(d+1)/2). Las startprob se omiten (degeneradas / no identificadas en
        series cortas), coherente con la convención de la tarea previa."""
        k = self.n_states
        d = self._model.means_.shape[1] if self._model is not None else len(
            self._used_features or BRIDGE_FEATURES
        )
        return int(k * k - 1 + k * d + k * d * (d + 1) / 2)
