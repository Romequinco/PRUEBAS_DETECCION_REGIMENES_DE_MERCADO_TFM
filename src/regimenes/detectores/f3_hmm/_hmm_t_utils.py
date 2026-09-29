"""
_hmm_t_utils.py — HMM con emisiones t-Student MULTIVARIANTES (para D8).

Por qué un módulo aparte (no se toca `_hmm_utils.py`)
----------------------------------------------------
`_hmm_utils.filtered_posterior` calcula el filtrado forward con emisiones
GAUSSIANAS (`multivariate_normal`). D8 necesita la MISMA maquinaria causal pero con
emisiones t-Student (colas pesadas, kurtosis 25-40 del EDA). Aquí se reimplementan:

  - `t_log_emission`      : log densidad t multivariante por frame y estado.
  - `StudentTHMM`         : HMM con emisión t (location m_i, escala S_i, dof ν_i por
                            estado), estimado por EM/ECM (Baum-Welch + actualización
                            de t como mezcla de escala gaussiana). API mínima estilo
                            `hmmlearn` (startprob_, transmat_, means_, scales_, dofs_,
                            score). Inicialización ROBUSTA desde un GaussianHMM.
  - `filtered_posterior_t`: filtrado forward CAUSAL P(estado_t | obs<=t) con emisión
                            t — el análogo causal de `_hmm_utils.filtered_posterior`.

La t como mezcla de escala (Gaussian scale mixture)
---------------------------------------------------
x_t | (S_t=i, u) ~ N(m_i, S_i / u),  u ~ Gamma(ν_i/2, ν_i/2).  Marginalizando u se
obtiene la t multivariante. El EM trata u como variable latente: en el E-step, además
de los posteriores de estado γ_t(i) (Baum-Welch), se calcula
    E[u | x_t, S_t=i]      = (ν_i + d) / (ν_i + δ_ti)
    E[log u | x_t, S_t=i]  = ψ((ν_i+d)/2) − log((ν_i+δ_ti)/2)
con δ_ti = (x_t−m_i)' S_i^{-1} (x_t−m_i) (Mahalanobis). En el M-step las medias y la
matriz de escala se actualizan PONDERADAS por γ·E[u] (los outliers reciben menos
peso → colas robustas) y ν_i se resuelve de la ecuación de punto fijo de
McLachlan-Peel. Ref.: Bulla (2011) "HMM with t components"; McLachlan & Peel (2000).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq
from scipy.special import digamma, gammaln, logsumexp

_TINY = 1e-300


# --------------------------------------------------------------------------- #
# Emisión t-Student multivariante
# --------------------------------------------------------------------------- #
def t_log_emission(
    values: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    dofs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """log t_d(x; m_i, S_i, ν_i) por frame y estado, y distancia de Mahalanobis.

    Devuelve (logB, maha) ambos (T, k). `maha[t, i]` = (x_t−m_i)' S_i^{-1} (x_t−m_i).
    Robusto a escalas casi singulares vía jitter creciente en la factorización
    Cholesky.
    """
    values = np.asarray(values, dtype=float)
    T, d = values.shape
    k = len(means)
    logB = np.empty((T, k))
    maha = np.empty((T, k))
    for i in range(k):
        S = np.asarray(scales[i], dtype=float)
        # Cholesky robusta (añade jitter si no es definida positiva).
        jit = 0.0
        for _ in range(8):
            try:
                cf = cho_factor(S + jit * np.eye(d), lower=True, check_finite=False)
                break
            except Exception:  # noqa: BLE001
                jit = 1e-8 if jit == 0.0 else jit * 10.0
        else:
            cf = cho_factor(S + 1e-3 * np.eye(d), lower=True, check_finite=False)
        L = cf[0]
        logdet = 2.0 * np.sum(np.log(np.abs(np.diag(L)) + _TINY))
        diff = values - means[i]
        sol = cho_solve(cf, diff.T, check_finite=False).T  # (T, d) = S^{-1} diff
        delta = np.einsum("ij,ij->i", diff, sol)           # Mahalanobis (T,)
        maha[:, i] = delta
        nu = float(dofs[i])
        c = (
            gammaln((nu + d) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * d * np.log(nu * np.pi)
            - 0.5 * logdet
        )
        logB[:, i] = c - 0.5 * (nu + d) * np.log1p(delta / nu)
    return logB, maha


def _solve_dof(c_i: float, d: int, lo: float = 2.05, hi: float = 200.0) -> float:
    """Resuelve ψ-ecuación de McLachlan-Peel para ν:
        φ(ν) = −ψ(ν/2) + log(ν/2) + 1 + c_i = 0,
    donde c_i = (1/N_i) Σ_t γ_ti (E[log u_ti] − E[u_ti]) (≤ 0). Si no hay raíz en el
    intervalo, se devuelve el extremo más cercano (ν grande ≈ gaussiano).
    """
    def phi(nu: float) -> float:
        return -digamma(nu / 2.0) + np.log(nu / 2.0) + 1.0 + c_i

    flo, fhi = phi(lo), phi(hi)
    if np.isnan(flo) or np.isnan(fhi):
        return hi
    if flo * fhi > 0:           # sin cambio de signo -> extremo más plausible
        return lo if abs(flo) < abs(fhi) else hi
    try:
        return float(brentq(phi, lo, hi, maxiter=100, xtol=1e-3))
    except Exception:  # noqa: BLE001
        return hi


# --------------------------------------------------------------------------- #
# HMM con emisiones t multivariantes (EM/ECM)
# --------------------------------------------------------------------------- #
class StudentTHMM:
    """HMM de `n_components` estados con emisión t-Student multivariante por estado.

    Inicialización desde un GaussianHMM ya ajustado (means_, covars_, transmat_,
    startprob_) y refinamiento por EM con la variable de escala latente. Expone una
    API mínima compatible con el patrón de los detectores HMM del banco.

    Atributos tras `fit`: startprob_ (k,), transmat_ (k,k), means_ (k,d),
    scales_ (k,d,d), dofs_ (k,), n_components, monitor_loglik (float).
    """

    def __init__(
        self,
        startprob: np.ndarray,
        transmat: np.ndarray,
        means: np.ndarray,
        scales: np.ndarray,
        dofs: np.ndarray,
        n_iter: int = 30,
        tol: float = 1e-3,
        min_covar: float = 1e-4,
    ) -> None:
        self.startprob_ = np.asarray(startprob, dtype=float)
        self.transmat_ = np.asarray(transmat, dtype=float)
        self.means_ = np.asarray(means, dtype=float)
        self.scales_ = np.asarray(scales, dtype=float)
        self.dofs_ = np.asarray(dofs, dtype=float)
        self.n_components = len(self.startprob_)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.min_covar = float(min_covar)
        self.monitor_loglik = -np.inf

    # ---- Baum-Welch en espacio log (no escalado, estable para T grandes) ---- #
    def _forward_backward(self, logB: np.ndarray):
        T, k = logB.shape
        log_pi = np.log(np.clip(self.startprob_, _TINY, None))
        log_A = np.log(np.clip(self.transmat_, _TINY, None))

        log_alpha = np.empty((T, k))
        log_alpha[0] = log_pi + logB[0]
        for t in range(1, T):
            tmp = log_alpha[t - 1][:, None] + log_A           # (k_prev, k_next)
            mx = tmp.max(axis=0)
            log_alpha[t] = mx + np.log(np.exp(tmp - mx).sum(axis=0)) + logB[t]
        log_beta = np.zeros((T, k))
        for t in range(T - 2, -1, -1):
            tmp = log_A + (logB[t + 1] + log_beta[t + 1])[None, :]  # (k_i, k_j)
            mx = tmp.max(axis=1)
            log_beta[t] = mx + np.log(np.exp(tmp - mx[:, None]).sum(axis=1))
        total_ll = logsumexp(log_alpha[-1])
        log_gamma = log_alpha + log_beta - total_ll
        gamma = np.exp(log_gamma)

        # ξ acumulado (k,k): Σ_t P(S_t=i, S_{t+1}=j | X). Vectorizado sobre t.
        a_part = log_alpha[:-1][:, :, None]                       # (T-1, k, 1)
        b_part = (logB[1:] + log_beta[1:])[:, None, :]            # (T-1, 1, k)
        log_xi = a_part + log_A[None, :, :] + b_part - total_ll   # (T-1, k, k)
        xi_sum = np.exp(log_xi).sum(axis=0)
        return gamma, xi_sum, float(total_ll)

    def fit(self, values: np.ndarray) -> "StudentTHMM":
        X = np.asarray(values, dtype=float)
        T, d = X.shape
        k = self.n_components
        eye = np.eye(d)
        prev_ll = -np.inf
        for _ in range(self.n_iter):
            logB, maha = t_log_emission(X, self.means_, self.scales_, self.dofs_)
            gamma, xi_sum, total_ll = self._forward_backward(logB)
            # E-step de la escala latente u (por estado).
            nu = self.dofs_[None, :]
            u = (nu + d) / (nu + maha)                       # E[u]   (T,k)
            elog_u = digamma((nu + d) / 2.0) - np.log((nu + maha) / 2.0)  # E[log u]

            # M-step.
            self.startprob_ = np.clip(gamma[0], _TINY, None)
            self.startprob_ /= self.startprob_.sum()
            row = xi_sum.sum(axis=1, keepdims=True)
            self.transmat_ = np.where(row > 0, xi_sum / np.clip(row, _TINY, None), 1.0 / k)

            for i in range(k):
                g = gamma[:, i]
                gu = g * u[:, i]
                sg = g.sum()
                sgu = gu.sum()
                if sgu <= _TINY or sg <= _TINY:
                    continue  # estado degenerado en este iter: conserva params
                m_i = (gu[:, None] * X).sum(axis=0) / sgu
                diff = X - m_i
                S_i = (gu[:, None, None] * np.einsum("ti,tj->tij", diff, diff)).sum(axis=0) / sg
                S_i = S_i + self.min_covar * eye
                self.means_[i] = m_i
                self.scales_[i] = S_i
                c_i = float((g * (elog_u[:, i] - u[:, i])).sum() / sg)
                self.dofs_[i] = _solve_dof(c_i, d)

            self.monitor_loglik = total_ll
            if total_ll - prev_ll < self.tol * (1 + abs(prev_ll)):
                break
            prev_ll = total_ll
        return self

    def score(self, values: np.ndarray) -> float:
        """Log-verosimilitud total (forward) bajo los parámetros actuales."""
        X = np.asarray(values, dtype=float)
        logB, _ = t_log_emission(X, self.means_, self.scales_, self.dofs_)
        log_pi = np.log(np.clip(self.startprob_, _TINY, None))
        log_A = np.log(np.clip(self.transmat_, _TINY, None))
        log_alpha = log_pi + logB[0]
        for t in range(1, len(X)):
            tmp = log_alpha[:, None] + log_A
            mx = tmp.max(axis=0)
            log_alpha = mx + np.log(np.exp(tmp - mx).sum(axis=0)) + logB[t]
        return float(logsumexp(log_alpha))


def filtered_posterior_t(model: StudentTHMM, values: np.ndarray) -> np.ndarray:
    """Filtrado forward CAUSAL: (T, k) con P(estado_t | obs_1..t), emisión t.

    Análogo a `_hmm_utils.filtered_posterior` pero con densidades t-Student. En cada
    t solo entran observaciones <= t (estrictamente causal). Forward escalado en log.
    """
    X = np.asarray(values, dtype=float)
    T, k = len(X), model.n_components
    logB, _ = t_log_emission(X, model.means_, model.scales_, model.dofs_)
    log_pi = np.log(np.clip(model.startprob_, _TINY, None))
    log_A = np.log(np.clip(model.transmat_, _TINY, None))

    log_alpha = np.empty((T, k))
    log_alpha[0] = log_pi + logB[0]
    log_alpha[0] -= logsumexp(log_alpha[0])
    for t in range(1, T):
        tmp = log_alpha[t - 1][:, None] + log_A
        mx = tmp.max(axis=0)
        pred = mx + np.log(np.exp(tmp - mx).sum(axis=0))
        log_alpha[t] = pred + logB[t]
        log_alpha[t] -= logsumexp(log_alpha[t])
    return np.exp(log_alpha)


def fit_student_t_hmm(
    values: np.ndarray,
    n_components: int,
    *,
    n_init: int = 4,
    gauss_n_iter: int = 100,
    t_n_iter: int = 30,
    random_state: int = 42,
    dof_init: float = 8.0,
    min_covar: float = 1e-4,
) -> StudentTHMM:
    """Ajusta un StudentTHMM robusto: varias inicializaciones GaussianHMM (se elige
    la de mayor logL gaussiana) → refinamiento EM t. Devuelve el modelo t ya ajustado.
    """
    from hmmlearn.hmm import GaussianHMM  # import local (dependencia pesada)

    X = np.asarray(values, dtype=float)
    d = X.shape[1]
    best_g, best_ll = None, -np.inf
    for seed in range(random_state, random_state + n_init):
        g = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=gauss_n_iter,
            random_state=seed,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g.fit(X)
                ll = g.score(X)
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(ll) and ll > best_ll:
            best_g, best_ll = g, ll
    if best_g is None:
        raise RuntimeError("StudentTHMM: ninguna inicialización GaussianHMM convergió.")

    nu0 = np.full(n_components, float(dof_init))
    # Escala inicial S = Cov · (ν−2)/ν (relación Cov = ν/(ν−2)·S de la t).
    scales0 = best_g.covars_ * ((dof_init - 2.0) / dof_init) + min_covar * np.eye(d)
    model = StudentTHMM(
        startprob=best_g.startprob_,
        transmat=best_g.transmat_,
        means=best_g.means_.copy(),
        scales=scales0,
        dofs=nu0,
        n_iter=t_n_iter,
        min_covar=min_covar,
    )
    model.fit(X)
    return model
