"""
_hmm_utils.py — Utilidades CAUSALES compartidas por los detectores HMM (D4, D8).

`filtered_posterior` calcula el FILTRADO FORWARD P(estado_t | obs_1..t): la
probabilidad de estado en t usando SOLO observaciones <= t. Es la alternativa
causal al Viterbi/`predict_proba` de hmmlearn, que suavizan con todo el bloque
(forward-backward) y por tanto miran días futuros del propio bloque — el sesgo de
look-ahead que el banco de pruebas debe evitar en evaluación online.

Patrón de uso en un detector HMM (ver hmm_gaussian_2s.py):
  - `predict` (duro) sigue usando Viterbi → modo IN-SAMPLE, NO causal, marcado.
  - `predict_proba` y `predict_online` usan `filtered_posterior` con un contexto
    de "burn-in" (las filas de train anteriores al bloque) para que el primer día
    del bloque ya arranque con la información del pasado, no desde startprob_.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

_TINY = 1e-300


def _log_emission(model, values: np.ndarray) -> np.ndarray:
    """log P(obs_t | estado=i) por frame y estado, para un GaussianHMM full/diag."""
    k = model.n_components
    covs = model.covars_  # (k, d, d) en covariance_type='full'
    logB = np.empty((len(values), k))
    for i in range(k):
        logB[:, i] = multivariate_normal.logpdf(
            values, mean=model.means_[i], cov=covs[i], allow_singular=True
        )
    return logB


def filtered_posterior(model, values: np.ndarray) -> np.ndarray:
    """Filtrado forward causal: devuelve (T, k) con P(estado_t | obs_1..t).

    Algoritmo forward escalado en espacio log (numéricamente estable). En cada t
    solo entran observaciones <= t, así que el resultado es ESTRICTAMENTE causal.

    Parameters
    ----------
    model : GaussianHMM ya ajustado (usa startprob_, transmat_, means_, covars_).
    values : np.ndarray (T, d) de observaciones en el orden temporal real.
    """
    values = np.asarray(values, dtype=float)
    T, k = len(values), model.n_components
    logB = _log_emission(model, values)
    log_pi = np.log(np.clip(model.startprob_, _TINY, None))
    log_A = np.log(np.clip(model.transmat_, _TINY, None))

    log_alpha = np.empty((T, k))
    log_alpha[0] = log_pi + logB[0]
    log_alpha[0] -= logsumexp(log_alpha[0])  # normaliza -> posterior filtrado en t=0
    for t in range(1, T):
        pred = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)  # paso de predicción
        log_alpha[t] = pred + logB[t]                                # corrección con emisión t
        log_alpha[t] -= logsumexp(log_alpha[t])                      # normaliza (escalado)
    return np.exp(log_alpha)
