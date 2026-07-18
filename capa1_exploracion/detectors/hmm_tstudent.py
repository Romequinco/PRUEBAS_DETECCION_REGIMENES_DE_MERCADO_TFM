"""
hmm_tstudent.py — D8 (FASE 3, Tanda 3). Familia F3: HMM avanzado (colas pesadas).

HMM con EMISIONES t-STUDENT multivariantes. Extiende el HMM gaussiano puente (D4)
sustituyendo la gaussiana de cada estado por una t multivariante (location m_i,
matriz de escala S_i, grados de libertad ν_i por estado). La t tiene COLAS PESADAS
(kurtosis 25-40 del EDA): cada estado puede acomodar saltos extremos sin inflar la
varianza ni "robar" observaciones a un estado de crisis, lo que —según el CP2— debería
atacar las fat tails y POTENCIALMENTE captar correcciones rápidas (taper 2013, Q4
2018) donde el gaussiano D4 se queda corto, con el riesgo de sobreajuste cuando hay
pocas observaciones por estado.

t-Student vs GMM-HMM — qué se usó y por qué
-------------------------------------------
El CP2 contempló "HMM t-Student o GMM-HMM" como alternativas para las colas. Aquí se
usa la **t-Student multivariante propia** (EM con la variable de escala latente, ver
`_hmm_t_utils.StudentTHMM`), NO el GMM-HMM, por dos razones:
  1. PARSIMONIA / BIC JUSTO: la t añade solo **k** parámetros (un ν por estado) sobre
     el HMM gaussiano equivalente, mientras que un GMM-HMM con mezcla por estado
     multiplica medias y covarianzas (k·n_mix·d(d+1)/2) y dispara el BIC. Como el
     objetivo es comparar el BIC con D4 de forma honesta, la t es la opción correcta.
  2. ROBUSTEZ: se inicializa desde un GaussianHMM (varias semillas, mayor logL) y se
     refina con EM-t; el GaussianHMM da un arranque estable y el refinamiento t es un
     número acotado de iteraciones. Sin mezcla, no hay componentes que colapsen.

Selección de K por BIC sobre {3, 4}
-----------------------------------
D5 ya mostró que el BIC prefiere k=3 sobre k=2 (un tercer régimen de varianza). D8
busca el mejor entre **{3, 4}**: con ≥3 estados el orden económico es calma →
corrección → crisis (severidad creciente en volatilidad). El notebook ajusta ambos
sobre la ventana puente y despliega el K de menor BIC.

Causalidad — FILTRADO FORWARD t, no Viterbi intra-bloque (igual patrón que D4)
------------------------------------------------------------------------------
  1. IN-SAMPLE (NO causal): `predict` = Viterbi t (mejor ruta global). Solo marcado.
  2. CAUSAL walk-forward: `predict_online`/`predict_proba` usan el FILTRADO FORWARD t
     (`_hmm_t_utils.filtered_posterior_t`) con burn-in de train: la etiqueta/prob de t
     usa SOLO observaciones <= t. Cero look-ahead intra-bloque.

Features / ventana (comparable con D4)
--------------------------------------
Mismas 7 features puente que D4 (`BRIDGE_FEATURES`), misma ventana 2007+ → con
train_size=252*5 + step, 2008/2011 caen en train (coberturas NaN OOS), igual que D4.
Así el BIC de D8 (t) es comparable al de D4 (gaussiano) sobre los mismos datos.

Bibliografía
------------
hamilton1989            — régimen como estado latente markoviano.
hmm_rabiner1989         — tutorial fundacional de HMM (forward-backward, Viterbi).
hmm_bulla2011           — HMM con componentes t: motiva la emisión de colas pesadas.
guidolintimmermann2007  — regímenes en retornos de activos vía HMM/Markov-switching.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from detectors._hmm_t_utils import (
    StudentTHMM,
    filtered_posterior_t,
    fit_student_t_hmm,
    t_log_emission,
)
from detectors.hmm_gaussian_2s import BRIDGE_FEATURES
from src.detector_base import RegimeDetector


class HMMTStudent(RegimeDetector):
    """HMM de K estados con emisiones t-Student multivariantes (colas pesadas).

    Parameters
    ----------
    n_states : int
        Nº de estados latentes (3 o 4; se elige por BIC en el notebook).
    n_init : int
        Inicializaciones GaussianHMM entre las que se elige la de mayor logL antes
        de refinar con EM-t (mitiga óptimos locales, como D4).
    gauss_n_iter, t_n_iter : int
        Iteraciones de Baum-Welch del GaussianHMM de arranque y del EM-t de refinado.
    features : list[str] | None
        Columnas a usar. None → BRIDGE_FEATURES si están todas; si no, todas.
    random_state : int
        Semilla base; se prueban random_state .. random_state+n_init-1.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_init: int = 4,
        gauss_n_iter: int = 100,
        t_n_iter: int = 30,
        features: list[str] | None = None,
        random_state: int | None = 42,
    ) -> None:
        super().__init__(n_states=n_states, random_state=random_state)
        self.n_init = int(n_init)
        self.gauss_n_iter = int(gauss_n_iter)
        self.t_n_iter = int(t_n_iter)
        self.features = list(features) if features is not None else None
        self._model: StudentTHMM | None = None
        self._used_features: list[str] | None = None
        self._Xtrain_sel: pd.DataFrame | None = None  # burn-in para filtrado causal

    # ------------------------------------------------------------------ #
    # Identidad
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return f"hmm_tstudent_{self.n_states}s"

    @property
    def bibliography(self) -> list[str]:
        return [
            "hamilton1989",
            "hmm_rabiner1989",
            "hmm_bulla2011",
            "guidolintimmermann2007",
        ]

    # ------------------------------------------------------------------ #
    # Selección de features
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
    # Ajuste: GaussianHMM (mejor de n_init semillas) -> refinado EM-t
    # ------------------------------------------------------------------ #
    def fit(self, X_train: pd.DataFrame) -> "HMMTStudent":
        Xtr = self._select(X_train)
        base = self.random_state if self.random_state is not None else 0
        self._model = fit_student_t_hmm(
            Xtr.values,
            n_components=self.n_states,
            n_init=self.n_init,
            gauss_n_iter=self.gauss_n_iter,
            t_n_iter=self.t_n_iter,
            random_state=base,
        )
        self._is_fitted = True
        self._Xtrain_sel = Xtr.copy()  # contexto de burn-in para el filtrado causal
        # Orden económico canónico (0=calma..n-1=crisis) con el train.
        self.label_states_economically(Xtr)
        return self

    # ------------------------------------------------------------------ #
    # Viterbi t (modo IN-SAMPLE, NO causal) — estados internos sin canonicalizar
    # ------------------------------------------------------------------ #
    def _viterbi(self, values: np.ndarray) -> np.ndarray:
        m = self._model
        logB, _ = t_log_emission(values, m.means_, m.scales_, m.dofs_)
        T, k = logB.shape
        log_pi = np.log(np.clip(m.startprob_, 1e-300, None))
        log_A = np.log(np.clip(m.transmat_, 1e-300, None))
        delta = np.empty((T, k))
        psi = np.zeros((T, k), dtype=int)
        delta[0] = log_pi + logB[0]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + log_A  # (k_prev, k_next)
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores[psi[t], np.arange(k)] + logB[t]
        states = np.empty(T, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def _predict_states(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"{self.name}: llama a fit() antes de predecir.")
        return self._viterbi(self._select(X).values)

    # ------------------------------------------------------------------ #
    # Filtrado forward CAUSAL t con burn-in (estados internos)
    # ------------------------------------------------------------------ #
    def _filtered_internal(self, X: pd.DataFrame) -> np.ndarray:
        """Posteriores filtrados t P(estado_t | obs<=t), (len(X), k), orden INTERNO.

        Antepone como burn-in las filas de train anteriores al primer índice de X.
        """
        Xsel = self._select(X)
        if self._Xtrain_sel is not None and len(Xsel):
            ctx = self._Xtrain_sel[self._Xtrain_sel.index < Xsel.index[0]]
        else:
            ctx = (self._Xtrain_sel if self._Xtrain_sel is not None else Xsel).iloc[:0]
        full = pd.concat([ctx, Xsel])
        filt = filtered_posterior_t(self._model, full.values)
        return filt[len(ctx):]

    # ------------------------------------------------------------------ #
    # Predicción CAUSAL online (filtrado forward t) — la que usa walk_forward
    # ------------------------------------------------------------------ #
    def predict_online(self, X: pd.DataFrame, refit: bool = False) -> np.ndarray:
        """Etiqueta dura CAUSAL canónica: argmax del filtrado forward t (obs<=t)."""
        self._check_fitted()
        raw_states = self._filtered_internal(X).argmax(axis=1)
        return self._apply_canonical(raw_states)

    # ------------------------------------------------------------------ #
    # Probabilidades posteriores CAUSALES (filtradas) en orden CANÓNICO
    # ------------------------------------------------------------------ #
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Posteriores FILTRADOS t P(estado_t | obs<=t) en orden económico canónico."""
        self._check_fitted()
        filt = self._filtered_internal(X)
        if self._canonical_order is None:
            return filt
        return filt[:, self._canonical_order]

    # ------------------------------------------------------------------ #
    # Parámetros por estado en orden CANÓNICO (para verificación de monotonía)
    # ------------------------------------------------------------------ #
    def dofs_canonical(self) -> np.ndarray:
        self._check_fitted()
        if self._canonical_order is None:
            return self._model.dofs_
        return self._model.dofs_[self._canonical_order]

    def transition_canonical(self) -> np.ndarray:
        self._check_fitted()
        A = self._model.transmat_
        if self._canonical_order is None:
            return A
        order = self._canonical_order
        return A[np.ix_(order, order)]

    # ------------------------------------------------------------------ #
    # Bondad de ajuste
    # ------------------------------------------------------------------ #
    def score(self, X: pd.DataFrame) -> float:
        self._check_fitted()
        return float(self._model.score(self._select(X).values))

    def n_parameters(self) -> int:
        """Parámetros libres: transición+startprob (k²−1) + medias (k·d) + escalas
        full (k·d(d+1)/2) + grados de libertad (k). MISMA convención que D4 más los k
        ν, de modo que el BIC sea comparable: la t añade SOLO k parámetros sobre el
        HMM gaussiano equivalente."""
        k = self.n_states
        d = self._model.means_.shape[1] if self._model is not None else len(
            self._used_features or BRIDGE_FEATURES
        )
        return int(k * k - 1 + k * d + k * d * (d + 1) / 2 + k)
