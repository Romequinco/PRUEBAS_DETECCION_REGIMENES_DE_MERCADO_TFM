"""
hsmm_tstudent.py — D13. HSMM con emisiones t-Student y duración explícita.

El detector conserva las emisiones t multivariantes de D8 y sustituye la
permanencia geométrica del HMM por una distribución de duración explícita para
cada estado. La estimación es deliberadamente transparente y reproducible:

1. se ajusta el HMM-t de D8 sobre el train;
2. se decodifica el train y se estiman duraciones Gamma desplazadas y
   regularizadas hacia la duración geométrica del HMM;
3. se construye la cadena semi-Markov embebida (sin auto-transiciones);
4. inferencia y evaluación usan filtrado forward sobre (estado, edad), por lo
   que la predicción del día t solo depende de observaciones <= t.

No es un EM conjunto de todos los parámetros HSMM: es una estimación en dos
etapas (emisiones t + duraciones). Esta decisión permite aislar el efecto de la
ley de permanencia frente a D8 sin introducir otra familia de emisiones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import gamma as gamma_dist

from detectors._hmm_t_utils import t_log_emission
from detectors.hmm_tstudent import HMMTStudent

_TINY = 1e-300


class HSMMTStudent(HMMTStudent):
    """HSMM-t de K estados con duraciones Gamma desplazadas explícitas.

    Parameters
    ----------
    max_duration : int
        Duración máxima representada. La última celda de la PMF acumula toda la
        cola restante, por lo que no se pierde masa de probabilidad.
    duration_prior_strength : float
        Número de episodios equivalentes usados para contraer los momentos de
        duración hacia la geometría del HMM de arranque. Reduce inestabilidad en
        estados con pocos episodios.
    transition_prior_strength : float
        Fuerza del prior sobre la matriz de saltos entre estados distintos.
    Resto de parámetros
        Iguales que :class:`detectors.hmm_tstudent.HMMTStudent`.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_init: int = 4,
        gauss_n_iter: int = 100,
        t_n_iter: int = 30,
        features: list[str] | None = None,
        random_state: int | None = 42,
        max_duration: int = 126,
        duration_prior_strength: float = 12.0,
        transition_prior_strength: float = 2.0,
    ) -> None:
        super().__init__(
            n_states=n_states,
            n_init=n_init,
            gauss_n_iter=gauss_n_iter,
            t_n_iter=t_n_iter,
            features=features,
            random_state=random_state,
        )
        if max_duration < 2:
            raise ValueError("max_duration debe ser >= 2.")
        self.max_duration = int(max_duration)
        self.duration_prior_strength = float(duration_prior_strength)
        self.transition_prior_strength = float(transition_prior_strength)
        self._duration_pmf: np.ndarray | None = None
        self._duration_hazard: np.ndarray | None = None
        self._duration_shape: np.ndarray | None = None
        self._duration_scale: np.ndarray | None = None
        self._embedded_transmat: np.ndarray | None = None
        self._hsmm_ready = False

    @property
    def name(self) -> str:
        return f"hsmm_tstudent_{self.n_states}s"

    @property
    def bibliography(self) -> list[str]:
        return [
            "hmm_bullabulla2006",
            "hmm_bulla2011",
            "hmm_rabiner1989",
            "guidolintimmermann2007",
        ]

    @staticmethod
    def _runs(states: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
        """Duraciones por estado y conteos de saltos entre episodios."""
        states = np.asarray(states, dtype=int)
        if states.size == 0:
            return [], np.empty((0, 0), dtype=float)
        k = int(states.max()) + 1
        durations: list[list[int]] = [[] for _ in range(k)]
        jumps = np.zeros((k, k), dtype=float)
        start = 0
        for stop in range(1, len(states) + 1):
            if stop == len(states) or states[stop] != states[start]:
                s = int(states[start])
                durations[s].append(stop - start)
                if stop < len(states):
                    jumps[s, int(states[stop])] += 1.0
                start = stop
        return durations, jumps

    def _fit_explicit_durations(self, raw_states: np.ndarray) -> None:
        """Estima PMF Gamma desplazada y matriz embebida regularizadas."""
        k, dmax = self.n_states, self.max_duration
        durations, jumps = self._runs(raw_states)
        if len(durations) < k:
            durations.extend([[] for _ in range(k - len(durations))])
        if jumps.shape != (k, k):
            jumps_full = np.zeros((k, k), dtype=float)
            jumps_full[: jumps.shape[0], : jumps.shape[1]] = jumps
            jumps = jumps_full

        base_A = np.asarray(self._model.transmat_, dtype=float)
        shapes = np.empty(k)
        scales = np.empty(k)
        pmf = np.empty((k, dmax))

        for s in range(k):
            obs = np.asarray(durations[s], dtype=float)
            # Y = D - 1 >= 0. Bajo un HMM, Y sigue una geométrica de fallos.
            p_exit = float(np.clip(1.0 - base_A[s, s], 1.0 / dmax, 0.95))
            prior_mean_y = (1.0 - p_exit) / p_exit
            prior_var_y = (1.0 - p_exit) / (p_exit * p_exit)
            prior_second_y = prior_var_y + prior_mean_y * prior_mean_y
            y = np.maximum(obs - 1.0, 0.0)
            w0 = max(self.duration_prior_strength, 0.0)
            denom = len(y) + w0
            mean_y = (
                float(y.sum()) + w0 * prior_mean_y
            ) / max(denom, _TINY)
            second_y = (
                float(np.square(y).sum()) + w0 * prior_second_y
            ) / max(denom, _TINY)
            var_y = max(second_y - mean_y * mean_y, 0.05)
            mean_y = max(mean_y, 0.05)
            shape = float(np.clip(mean_y * mean_y / var_y, 0.10, 500.0))
            scale = float(np.clip(var_y / mean_y, 0.01, 5000.0))
            shapes[s], scales[s] = shape, scale

            # D=1+Y: bins [0,1), [1,2), ...; el último acumula la cola.
            edges = np.arange(dmax, dtype=float)
            cdf = gamma_dist.cdf(edges, a=shape, scale=scale)
            q = np.empty(dmax)
            q[:-1] = np.diff(np.r_[0.0, cdf[1:]])
            q[-1] = 1.0 - cdf[-1]
            q = np.clip(q, 1e-12, None)
            pmf[s] = q / q.sum()

        # Cadena embebida: al terminar un episodio siempre se salta a otro estado.
        embedded = np.zeros((k, k), dtype=float)
        for s in range(k):
            prior = np.asarray(base_A[s], dtype=float).copy()
            prior[s] = 0.0
            if prior.sum() <= 0:
                prior[:] = 1.0
                prior[s] = 0.0
            prior /= prior.sum()
            row = jumps[s] + self.transition_prior_strength * prior
            row[s] = 0.0
            if row.sum() <= 0:
                row = prior
            embedded[s] = row / row.sum()

        survival = np.flip(np.cumsum(np.flip(pmf, axis=1), axis=1), axis=1)
        hazard = np.divide(pmf, survival, out=np.ones_like(pmf), where=survival > 0)
        hazard[:, -1] = 1.0

        self._duration_shape = shapes
        self._duration_scale = scales
        self._duration_pmf = pmf
        self._duration_hazard = np.clip(hazard, 0.0, 1.0)
        self._embedded_transmat = embedded

    def fit(self, X_train: pd.DataFrame) -> "HSMMTStudent":
        # super().fit ajusta exactamente las emisiones t de D8. Mientras aún no
        # hay duraciones, _predict_states cae al Viterbi HMM para etiquetar.
        self._hsmm_ready = False
        super().fit(X_train)
        Xtr = self._select(X_train)
        raw_hmm = super()._predict_states(Xtr)
        self._fit_explicit_durations(raw_hmm)
        self._hsmm_ready = True
        # Rehacer el orden económico con la decodificación ya semi-Markov.
        self.label_states_economically(Xtr)
        return self

    def _emission_likelihood(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        log_b, _ = t_log_emission(
            values, self._model.means_, self._model.scales_, self._model.dofs_
        )
        shift = log_b.max(axis=1)
        return np.exp(log_b - shift[:, None]), shift

    def _forward_expanded(self, values: np.ndarray) -> tuple[np.ndarray, float]:
        """Filtro causal sobre (estado, edad) y log-verosimilitud escalada."""
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return np.empty((0, self.n_states)), 0.0
        b, log_shift = self._emission_likelihood(values)
        k, dmax = self.n_states, self.max_duration
        hazard = self._duration_hazard
        embedded = self._embedded_transmat
        alpha = np.zeros((k, dmax), dtype=float)
        alpha[:, 0] = np.asarray(self._model.startprob_) * b[0]
        norm = max(float(alpha.sum()), _TINY)
        alpha /= norm
        loglik = float(np.log(norm) + log_shift[0])
        state_post = np.empty((len(values), k), dtype=float)
        state_post[0] = alpha.sum(axis=1)

        for t in range(1, len(values)):
            pred = np.zeros_like(alpha)
            pred[:, 1:] = alpha[:, :-1] * (1.0 - hazard[:, :-1])
            exits = np.sum(alpha * hazard, axis=1)
            pred[:, 0] = exits @ embedded
            pred *= b[t, :, None]
            norm = max(float(pred.sum()), _TINY)
            alpha = pred / norm
            loglik += float(np.log(norm) + log_shift[t])
            state_post[t] = alpha.sum(axis=1)
        return state_post, loglik

    def _viterbi_expanded(self, values: np.ndarray) -> np.ndarray:
        """Ruta global HSMM en el espacio ampliado; solo diagnóstico in-sample."""
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return np.empty(0, dtype=int)
        log_b, _ = t_log_emission(
            values, self._model.means_, self._model.scales_, self._model.dofs_
        )
        k, dmax = self.n_states, self.max_duration
        log_h = np.log(np.clip(self._duration_hazard, _TINY, 1.0))
        log_stay = np.log(np.clip(1.0 - self._duration_hazard, _TINY, 1.0))
        log_jump = np.log(np.clip(self._embedded_transmat, _TINY, 1.0))
        delta = np.full((k, dmax), -np.inf)
        delta[:, 0] = np.log(np.clip(self._model.startprob_, _TINY, 1.0)) + log_b[0]
        enter_state = np.zeros((len(values), k), dtype=np.int16)
        enter_age = np.zeros((len(values), k), dtype=np.int16)

        for t in range(1, len(values)):
            nxt = np.full_like(delta, -np.inf)
            nxt[:, 1:] = delta[:, :-1] + log_stay[:, :-1] + log_b[t, :, None]
            exit_score = delta + log_h
            best_age = np.argmax(exit_score, axis=1)
            best_by_state = exit_score[np.arange(k), best_age]
            incoming = best_by_state[:, None] + log_jump
            best_prev_state = np.argmax(incoming, axis=0)
            nxt[:, 0] = incoming[best_prev_state, np.arange(k)] + log_b[t]
            enter_state[t] = best_prev_state
            enter_age[t] = best_age[best_prev_state]
            delta = nxt

        state = int(np.unravel_index(np.argmax(delta), delta.shape)[0])
        age = int(np.unravel_index(np.argmax(delta), delta.shape)[1])
        path = np.empty(len(values), dtype=int)
        for t in range(len(values) - 1, -1, -1):
            path[t] = state
            if t == 0:
                break
            if age > 0:
                age -= 1
            else:
                prev_state = int(enter_state[t, state])
                age = int(enter_age[t, state])
                state = prev_state
        return path

    def _predict_states(self, X: pd.DataFrame) -> np.ndarray:
        if not self._hsmm_ready:
            return super()._predict_states(X)
        return self._viterbi_expanded(self._select(X).values)

    def _filtered_internal(self, X: pd.DataFrame) -> np.ndarray:
        """Posterior HSMM filtrado, con el train anterior como burn-in causal."""
        Xsel = self._select(X)
        if self._Xtrain_sel is not None and len(Xsel):
            ctx = self._Xtrain_sel[self._Xtrain_sel.index < Xsel.index[0]]
        else:
            ctx = (self._Xtrain_sel if self._Xtrain_sel is not None else Xsel).iloc[:0]
        full = pd.concat([ctx, Xsel])
        post, _ = self._forward_expanded(full.values)
        return post[len(ctx):]

    def predict_online(self, X: pd.DataFrame, refit: bool = False) -> np.ndarray:
        self._check_fitted()
        raw = self._filtered_internal(X).argmax(axis=1)
        return self._apply_canonical(raw)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        post = self._filtered_internal(X)
        if self._canonical_order is None:
            return post
        return post[:, self._canonical_order]

    def duration_pmf_canonical(self) -> np.ndarray:
        self._check_fitted()
        if self._canonical_order is None:
            return self._duration_pmf.copy()
        return self._duration_pmf[self._canonical_order].copy()

    def duration_hazard_canonical(self) -> np.ndarray:
        self._check_fitted()
        if self._canonical_order is None:
            return self._duration_hazard.copy()
        return self._duration_hazard[self._canonical_order].copy()

    def duration_parameters_canonical(self) -> pd.DataFrame:
        self._check_fitted()
        order = self._canonical_order if self._canonical_order is not None else np.arange(self.n_states)
        q = self._duration_pmf[order]
        days = np.arange(1, self.max_duration + 1, dtype=float)
        return pd.DataFrame(
            {
                "shape_gamma": self._duration_shape[order],
                "scale_gamma": self._duration_scale[order],
                "mean_days": q @ days,
                "median_days": [days[np.searchsorted(np.cumsum(row), 0.5)] for row in q],
                "mode_days": days[np.argmax(q, axis=1)],
            },
            index=pd.Index(range(self.n_states), name="state"),
        )

    def embedded_transition_canonical(self) -> np.ndarray:
        self._check_fitted()
        if self._canonical_order is None:
            return self._embedded_transmat.copy()
        o = self._canonical_order
        return self._embedded_transmat[np.ix_(o, o)].copy()

    def hmm_transition_canonical(self) -> np.ndarray:
        return super().transition_canonical()

    def score(self, X: pd.DataFrame) -> float:
        self._check_fitted()
        _, ll = self._forward_expanded(self._select(X).values)
        return float(ll)

    def n_parameters(self) -> int:
        """BIC aproximado: D8 + K parámetros netos de duración explícita.

        El HMM usa K²-1 parámetros dinámicos (inicio + matriz completa). El HSMM
        usa (K-1) de inicio, K(K-2) de saltos sin diagonal y 2K de duración Gamma:
        K²+K-1, es decir, K más que D8. Las emisiones son idénticas.
        """
        return int(super().n_parameters() + self.n_states)
