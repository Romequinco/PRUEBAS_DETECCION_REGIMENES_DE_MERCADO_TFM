"""
markov_switching_var.py — D5 (FASE 3, Tanda 2). Familia F4: Markov-Switching econométrico.

BASELINE ECONOMÉTRICO INTERPRETABLE en la tradición de Hamilton (1989): un
Markov-Switching model sobre el RETORNO del S&P 500 cuyo régimen latente (cadena de
Markov de 2 estados) gobierna a la vez la MEDIA y la VARIANZA del retorno. El estado
de alta varianza (y media baja/negativa) es el régimen de "crisis/estrés"; el de baja
varianza es la "calma". A diferencia del HMM gaussiano multivariante (D4), aquí el
modelo es UNIVARIANTE (solo ve el retorno del propio índice, no la correlación
cross-asset) pero se estima con la maquinaria econométrica estándar
(`statsmodels.tsa.regime_switching.markov_regression.MarkovRegression`), lo que da
parámetros interpretables (medias, varianzas y matriz de transición por régimen) y
test de bondad de ajuste (logL/AIC/BIC) para elegir k.

Idea
----
y_t = μ_{S_t} + ε_t,  ε_t ~ N(0, σ²_{S_t}),  S_t ∈ {0..k-1} cadena de Markov.
Con `switching_variance=True` y `trend='c'` (constante conmutante por defecto) los
regímenes difieren en media Y varianza. La diagonal alta de la matriz de transición da
persistencia (menos flickering). `crisis = régimen de ALTA varianza`.

Causalidad — PROBABILIDADES FILTRADAS, no smoothed (igual que D4)
----------------------------------------------------------------
`MarkovRegression` ofrece `filtered_marginal_probabilities` P(S_t | y_1..t) (CAUSALES,
solo pasado) y `smoothed_marginal_probabilities` P(S_t | y_1..T) (usan TODA la muestra
= look-ahead). Para evaluación online se usan FILTRADAS.

Reto del walk-forward: ajustar en train, congelar parámetros y obtener probabilidades
FILTRADAS de un bloque de test SIN reestimar. `statsmodels` no expone un `res.apply`
para reaplicar params a nuevo endog en estos modelos, así que (igual que D4 con su
filtrado forward multivariante en `_hmm_utils`) se EXTRAEN los parámetros del MS
ajustado (μ_k, σ²_k, matriz de transición) y se corre un FORWARD FILTER GAUSSIANO
UNIVARIANTE propio sobre `burn-in de train + bloque`, devolviendo solo el bloque. Cada
t usa SOLO observaciones <= t: cero look-ahead intra-bloque. El burn-in (filas de train
anteriores al bloque) hace que el primer día del bloque arranque con el pasado, no desde
la distribución estacionaria.

Modo IN-SAMPLE (NO causal): para comparar, el notebook lee directamente
`filtered_marginal_probabilities` vs `smoothed_marginal_probabilities` del ajuste sobre
toda la muestra (las smoothed son look-ahead y se marcan como tal).

Ventana (LARGA)
---------------
D5 modela SOLO el retorno del S&P 500, disponible desde 1985 en
`data/raw/raw_panel.parquet` (columna SP500). Con histórico largo y train inicial de
~8 años, 2008 y 2011 SÍ son evaluables OUT-OF-SAMPLE (como D1), a diferencia de D4
(atado a features con HYG desde 2007).

Bibliografía
------------
hamilton1989       — modelo Markov-Switching seminal (régimen como cadena de Markov latente).
ms_kim1994         — filtro de Kim para Markov-switching (filtrado causal + suavizado).
ms_guidolin2011    — Markov-switching en retornos de activos / asignación.
ms_kimnelson1999   — state-space con cambio de régimen (libro de referencia, filtro/smoother).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from src.detector_base import RegimeDetector

_TINY = 1e-300


def _gaussian_logpdf(y: np.ndarray, mu: float, var: float) -> np.ndarray:
    """log N(y; mu, var) elemento a elemento (var > 0)."""
    var = max(float(var), _TINY)
    return -0.5 * (np.log(2.0 * np.pi * var) + (y - mu) ** 2 / var)


def _stationary_distribution(A: np.ndarray) -> np.ndarray:
    """Distribución estacionaria π de una matriz de transición fila-estocástica A
    (A[i, j] = P(s_t=j | s_{t-1}=i)). Vía iteración de potencia (robusto)."""
    k = A.shape[0]
    pi = np.full(k, 1.0 / k)
    for _ in range(10_000):
        nxt = pi @ A
        nxt = nxt / nxt.sum()
        if np.max(np.abs(nxt - pi)) < 1e-14:
            pi = nxt
            break
        pi = nxt
    return pi


def _univariate_forward_filter(
    y: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    A: np.ndarray,
    pi: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Filtrado forward gaussiano UNIVARIANTE causal.

    Devuelve (post, loglik) donde post es (T, k) con P(S_t | y_1..t) — la etiqueta de
    t usa SOLO observaciones <= t (estrictamente causal) — y loglik es la log-
    verosimilitud total del bloque bajo los parámetros congelados.

    A : (k, k) fila-estocástica, A[i, j] = P(s_t=j | s_{t-1}=i).
    """
    y = np.asarray(y, dtype=float)
    T, k = len(y), len(means)
    logB = np.empty((T, k))
    for i in range(k):
        logB[:, i] = _gaussian_logpdf(y, means[i], variances[i])
    log_pi = np.log(np.clip(pi, _TINY, None))
    log_A = np.log(np.clip(A, _TINY, None))

    log_alpha = np.empty((T, k))
    log_alpha[0] = log_pi + logB[0]
    c0 = logsumexp(log_alpha[0])
    log_alpha[0] -= c0
    total_ll = c0
    for t in range(1, T):
        pred = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)  # paso de predicción
        log_alpha[t] = pred + logB[t]                                # corrección con y_t
        c = logsumexp(log_alpha[t])
        log_alpha[t] -= c                                            # normaliza (escalado)
        total_ll += c
    return np.exp(log_alpha), float(total_ll)


class MarkovSwitchingVar(RegimeDetector):
    """Markov-Switching de la VARIANZA (y media) del retorno del S&P 500.

    Parameters
    ----------
    n_states : int
        Nº de regímenes (k). 2 = calma/crisis (default); se prueba también 3 y se
        reporta por AIC/BIC en el notebook.
    feature : str
        Columna de X usada como endog (retorno log del S&P 500). Está en
        `_RETURN_COLS` del núcleo, así que el etiquetado económico la reconoce sin
        disparar el warning peligroso de fallback.
    scale : float
        Factor de escala interno aplicado al endog SOLO para estabilidad numérica del
        optimizador (retornos ~1e-2 dan σ² ~1e-4). No afecta a las probabilidades ni
        al orden de los estados; el etiquetado económico usa el retorno sin escalar.
    search_reps : int
        Nº de búsquedas aleatorias de parámetros iniciales en `fit` (mitiga óptimos
        locales del EM). 0 = solo el arranque por defecto de statsmodels.
    """

    def __init__(
        self,
        n_states: int = 2,
        feature: str = "SP500_ret",
        scale: float = 100.0,
        search_reps: int = 0,
        random_state: int | None = 42,
    ) -> None:
        super().__init__(n_states=n_states, random_state=random_state)
        self.feature = str(feature)
        self.scale = float(scale)
        self.search_reps = int(search_reps)
        self._res = None                       # MarkovRegressionResults ajustado
        self._means: np.ndarray | None = None  # μ_k (escala interna), orden INTERNO
        self._vars: np.ndarray | None = None   # σ²_k (escala interna), orden INTERNO
        self._A: np.ndarray | None = None      # transición fila-estocástica (interno)
        self._pi: np.ndarray | None = None     # estacionaria de A
        self._endog_train: pd.Series | None = None  # burn-in para filtrado causal

    # ------------------------------------------------------------------ #
    # Identidad
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return f"markov_switching_var_{self.n_states}s"

    @property
    def bibliography(self) -> list[str]:
        return ["hamilton1989", "ms_kim1994", "ms_guidolin2011", "ms_kimnelson1999"]

    # ------------------------------------------------------------------ #
    # Utilidad: extraer el endog (retorno) de X, escalado
    # ------------------------------------------------------------------ #
    def _endog(self, X: pd.DataFrame) -> pd.Series:
        if self.feature not in X.columns:
            raise KeyError(
                f"{self.name}: falta la columna '{self.feature}' en X "
                f"(columnas: {list(X.columns)})."
            )
        return (X[self.feature] * self.scale).astype(float)

    # ------------------------------------------------------------------ #
    # Ajuste: estima el MS y EXTRAE parámetros para el filtrado causal propio
    # ------------------------------------------------------------------ #
    def fit(self, X_train: pd.DataFrame) -> "MarkovSwitchingVar":
        y = self._endog(X_train).dropna()
        mod = MarkovRegression(
            y.values,
            k_regimes=self.n_states,
            trend="c",
            switching_variance=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergencia/Hessian: ruido por fold
            if self.search_reps > 0:
                res = mod.fit(search_reps=self.search_reps)
            else:
                res = mod.fit()
        self._res = res

        # Parámetros por régimen (orden INTERNO del modelo). `res.params` es un ndarray;
        # mapeo por nombre vía `param_names` (p. ej. 'const[0]', 'sigma2[1]', ...).
        k = self.n_states
        pmap = dict(zip(res.model.param_names, np.asarray(res.params, dtype=float)))
        self._means = np.array([pmap[f"const[{i}]"] for i in range(k)], dtype=float)
        self._vars = np.array([pmap[f"sigma2[{i}]"] for i in range(k)], dtype=float)
        # regime_transition[i, j] = P(s_t=i | s_{t-1}=j) (columnas suman 1, convención
        # statsmodels). El forward filter quiere A fila-estocástica P(j | i) => transpongo.
        rt = np.asarray(res.regime_transition).squeeze()
        self._A = np.ascontiguousarray(rt.T)
        self._pi = _stationary_distribution(self._A)

        self._endog_train = y.copy()  # burn-in para el filtrado causal del bloque
        self._is_fitted = True
        # Orden económico canónico (0=calma..k-1=crisis). Pasamos el retorno SIN escalar
        # como market_returns explícito -> evita el warning de fallback del núcleo.
        self.label_states_economically(
            X_train, market_returns=X_train[self.feature]
        )
        return self

    # ------------------------------------------------------------------ #
    # Filtrado forward CAUSAL con contexto de burn-in (orden INTERNO)
    # ------------------------------------------------------------------ #
    def _filtered_internal(self, X: pd.DataFrame) -> np.ndarray:
        """Posteriores FILTRADOS P(S_t | y<=t), (len(X), k), orden interno.

        Antepone como burn-in las filas de train estrictamente anteriores al primer
        índice de X para que el primer día del bloque arranque con el pasado.
        """
        y = self._endog(X)
        if self._endog_train is not None and len(y):
            ctx = self._endog_train[self._endog_train.index < y.index[0]]
        else:
            ctx = (self._endog_train if self._endog_train is not None else y).iloc[:0]
        full = pd.concat([ctx, y])
        post, _ = _univariate_forward_filter(
            full.values, self._means, self._vars, self._A, self._pi
        )
        return post[len(ctx):]

    # ------------------------------------------------------------------ #
    # Predicción interna (sin canonicalizar): argmax del filtrado causal
    # ------------------------------------------------------------------ #
    def _predict_states(self, X: pd.DataFrame) -> np.ndarray:
        if self._res is None:
            raise RuntimeError(f"{self.name}: llama a fit() antes de predecir.")
        return self._filtered_internal(X).argmax(axis=1)

    # ------------------------------------------------------------------ #
    # Predicción CAUSAL online (filtrado forward) — la que usa walk_forward
    # ------------------------------------------------------------------ #
    def predict_online(self, X: pd.DataFrame, refit: bool = False) -> np.ndarray:
        """Etiqueta dura CAUSAL canónica: argmax de la prob FILTRADA (y<=t)."""
        self._check_fitted()
        raw_states = self._filtered_internal(X).argmax(axis=1)
        return self._apply_canonical(raw_states)

    # ------------------------------------------------------------------ #
    # Probabilidades FILTRADAS (causales) en orden CANÓNICO
    # ------------------------------------------------------------------ #
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Posteriores FILTRADOS P(S_t | y<=t) en orden económico canónico.

        Filtrado forward propio (causal), NO las smoothed de statsmodels. Así
        `p_crisis = predict_proba[:, crisis_state]` no mira el futuro del bloque.
        """
        self._check_fitted()
        filt = self._filtered_internal(X)
        if self._canonical_order is None:
            return filt
        # filt[:, j] = P(estado interno j). El canónico i = estado interno
        # canonical_order[i] -> reordeno columnas.
        return filt[:, self._canonical_order]

    # ------------------------------------------------------------------ #
    # Probabilidades IN-SAMPLE del ajuste (para la comparación filtrada vs smoothed)
    # ------------------------------------------------------------------ #
    def insample_proba(self, kind: str = "filtered") -> np.ndarray:
        """Probabilidades marginales del ajuste sobre TODA la muestra de train, en
        orden canónico. `kind='filtered'` (causal) o `'smoothed'` (NO causal,
        look-ahead). Solo para el notebook (comparación), NO para evaluación online.
        """
        self._check_fitted()
        if kind == "filtered":
            p = np.asarray(self._res.filtered_marginal_probabilities)
        elif kind == "smoothed":
            p = np.asarray(self._res.smoothed_marginal_probabilities)
        else:
            raise ValueError("kind debe ser 'filtered' o 'smoothed'")
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        if self._canonical_order is None:
            return p
        return p[:, self._canonical_order]

    # Varianza por estado en orden CANÓNICO (para verificar crisis = alta varianza).
    def variances_canonical(self) -> np.ndarray:
        self._check_fitted()
        if self._canonical_order is None:
            return self._vars
        return self._vars[self._canonical_order]

    def means_canonical(self) -> np.ndarray:
        self._check_fitted()
        if self._canonical_order is None:
            return self._means
        return self._means[self._canonical_order]

    def transition_canonical(self) -> np.ndarray:
        """Matriz de transición fila-estocástica P(s_t=j | s_{t-1}=i) en orden canónico."""
        self._check_fitted()
        if self._canonical_order is None:
            return self._A
        order = self._canonical_order
        return self._A[np.ix_(order, order)]

    # ------------------------------------------------------------------ #
    # Bondad de ajuste (con parámetros congelados del train, vía forward filter)
    # ------------------------------------------------------------------ #
    def score(self, X: pd.DataFrame) -> float:
        """Log-verosimilitud total de X bajo los parámetros congelados del train."""
        self._check_fitted()
        y = self._endog(X).dropna()
        _, ll = _univariate_forward_filter(
            y.values, self._means, self._vars, self._A, self._pi
        )
        return float(ll)

    def n_parameters(self) -> int:
        """Parámetros libres del MS: transición k(k-1) + medias k + varianzas k = k²+k.
        (coincide con len(res.params): para k=2 -> 6, k=3 -> 12)."""
        if self._res is not None:
            return int(len(self._res.params))
        k = self.n_states
        return int(k * k + k)
