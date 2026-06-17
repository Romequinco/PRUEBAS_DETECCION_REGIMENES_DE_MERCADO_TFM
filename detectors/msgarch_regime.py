"""
msgarch_regime.py — D11 `msgarch_regime` (FASE 3, Tanda 4 — EXPLORATORIA).
Familia F5: VOLATILIDAD con cambio de régimen (Markov-Switching GARCH).

Qué es y por qué esta variante (decisión técnica honesta)
---------------------------------------------------------
El estado del arte (CP2) avisó: NO hay librería madura de MS-GARCH en Python; el
estándar bueno es el paquete `MSGARCH` de R, y en este entorno `rpy2` NO está
instalado (no se puentea a R) y `arch` 8.0 hace GARCH univariante pero NO
MS-GARCH. En vez de declararlo no implementable, aquí se IMPLEMENTA desde cero la
variante que SÍ es tratable en Python puro: el **MS-GARCH de Haas-Mittnik-Paolella
(2004)**, estimado por máxima verosimilitud con un **filtro de Hamilton** propio
(numpy/scipy, sin R).

Por qué HMP-2004 es implementable y la "naive" no
-------------------------------------------------
Un MS-GARCH "naive" (la varianza de t depende de la varianza de t-1 que a su vez
depende del régimen de t-1...) sufre PATH DEPENDENCE: la verosimilitud integra
sobre 2^t trayectorias de régimen → intratable. Gray (1996) lo evita COLAPSANDO la
varianza con la prob. de régimen en cada paso (una aproximación). Haas-Mittnik-
Paolella (2004) lo resuelve de raíz con K recursiones GARCH **EN PARALELO**, una
por régimen, cada una alimentada SOLO por su propia varianza pasada:

    h_{k,t} = ω_k + α_k · ε²_{t-1} + β_k · h_{k,t-1}      (k = 0..K-1, en paralelo)
    ε_{t-1} = y_{t-1} - μ        (residuo común; media única)
    y_t | (s_t=k, pasado) ~ μ + sqrt(h_{k,t}) · t_ν estandarizada (Var=1)
    s_t  cadena de Markov 2 estados, matriz de transición P.

Como cada h_{k,t} usa su propio h_{k,t-1} (no la trayectoria de regímenes), NO hay
path dependence y la verosimilitud se computa con el filtro de Hamilton en O(T·K²).
Esta es la razón EXACTA por la que HMP-2004 es el caballo de batalla del MS-GARCH.

Qué se pierde / honestidad del alcance
--------------------------------------
- Es la especificación HMP **paralela / path-INDEPENDIENTE**. NO reproduce el
  MS-GARCH path-dependiente "completo" (intratable) ni el colapso de Gray (1996);
  son modelos distintos, no estrictamente anidados. El paquete `MSGARCH` de R
  ofrece más familias (eGARCH/gjr por régimen, varias innovaciones, bayesiano);
  aquí hay UNA familia (GARCH(1,1) por régimen, innovación t con ν compartido).
- Media μ única y ν compartido entre regímenes: PARSIMONIA y estabilidad numérica
  del ML (la ML de MS-GARCH es no convexa y propensa a óptimos locales). Lo que
  separa a los regímenes es la dinámica de varianza (ω,α,β por régimen), que es el
  objeto de interés. El régimen de alta varianza incondicional = crisis.
- Coste de cómputo: el ML re-estima en cada ventana walk-forward un modelo de 10
  parámetros con un bucle de filtro en Python puro (sin numba). Por eso el build
  usa `step` trimestral (no 21 d): ver `notebooks/_build_11.py`.

Posición frente a D5 y D6 (no es redundante)
--------------------------------------------
- D5 `markov_switching_var`: régimen de Markov con varianza CONSTANTE dentro del
  régimen (sin dinámica ARCH).
- D6 `garch_t_vol`: dinámica GARCH(1,1)-t pero UN solo régimen, "régimen" = umbral
  determinista de la sigma.
- D11 = la síntesis: régimen de Markov latente Y dinámica GARCH-t dentro de cada
  régimen, con posterior de crisis FILTRADO (Hamilton), logL/AIC/BIC comparables.

Causalidad (mismo patrón que D5/D6/D8)
-------------------------------------
`fit(train)` estima por ML y CONGELA los parámetros. Para un bloque de test se
antepone un burn-in de retornos de train anteriores al bloque y se corre el filtro
de Hamilton hacia delante con parámetros fijos: la prob. filtrada P(s_t | y<=t)
usa SOLO retornos <= t (causal). El posterior FILTRADO (no suavizado) es la señal
online; nada mira el futuro del bloque.

Ventana (LARGA)
---------------
Solo retorno log del S&P 500 desde 1985 (`data/raw/raw_panel.parquet`) → el
walk-forward cubre 2008 y 2011 OUT-OF-SAMPLE (como D5/D6).

Bibliografía
------------
vol_haasmittnikpaolella2004 — MS-GARCH paralelo path-independiente (modelo base).
vol_gray1996                — MS-GARCH con colapso de varianza (alternativa citada).
vol_marcucci2005            — MS-GARCH para prever volatilidad (aplicación/ref).
vol_bollerslev1986          — GARCH(1,1) (recursión de varianza por régimen).
vol_bollerslev1987          — innovaciones t-Student (colas gordas).
hamilton1989                — régimen de Markov latente + filtro de Hamilton.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.detector_base import RegimeDetector

_TINY = 1e-300
# α_k + β_k debe ser < 1 (estacionariedad por régimen). Penalización si lo roza.
_PERSIST_CAP = 0.9995


def _stationary_2(p00: float, p11: float) -> tuple[float, float]:
    """Distribución estacionaria de la cadena 2-estados con P=[[p00,1-p00],[1-p11,p11]]."""
    a = 1.0 - p00  # 0 -> 1
    b = 1.0 - p11  # 1 -> 0
    denom = a + b
    if denom <= 0:
        return 0.5, 0.5
    pi1 = a / denom
    return 1.0 - pi1, pi1


def _std_t_logconst(nu: float) -> float:
    """Constante log de la t-Student ESTANDARIZADA a varianza 1 (depende solo de ν)."""
    return (
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * math.log((nu - 2.0) * math.pi)
    )


def msgarch_filter(
    y: np.ndarray,
    mu: float,
    om0: float, al0: float, be0: float,
    om1: float, al1: float, be1: float,
    p00: float, p11: float,
    nu: float,
    h0_0: float, h0_1: float,
    want_filtered: bool = False,
):
    """Filtro de Hamilton para el MS-GARCH(1,1)-t HMP-2004 de 2 regímenes (K=2 unrolled).

    Devuelve (loglik, filtered) donde `filtered` es (T,2) con P(s_t=k | y<=t) si
    `want_filtered`, o None en caso contrario (modo optimizador, más rápido).

    Recursiones de varianza EN PARALELO (path-independiente). Causal: cada paso usa
    solo y<=t. h0_0/h0_1 son las varianzas iniciales por régimen (incondicionales).
    """
    T = y.shape[0]
    logc = _std_t_logconst(nu)
    half = (nu + 1.0) / 2.0
    nm2 = nu - 2.0

    # estado inicial: predicción = estacionaria
    pi0, pi1 = _stationary_2(p00, p11)
    h0, h1 = float(h0_0), float(h0_1)
    xi_pred0, xi_pred1 = pi0, pi1

    filt = np.empty((T, 2), dtype=float) if want_filtered else None
    loglik = 0.0

    for t in range(T):
        yt = y[t]
        r = yt - mu
        # densidad t estandarizada por régimen (en log, luego exp)
        z0 = r * r / h0
        z1 = r * r / h1
        logf0 = logc - 0.5 * math.log(h0) - half * math.log1p(z0 / nm2)
        logf1 = logc - 0.5 * math.log(h1) - half * math.log1p(z1 / nm2)
        f0 = math.exp(logf0)
        f1 = math.exp(logf1)

        num0 = xi_pred0 * f0
        num1 = xi_pred1 * f1
        lik = num0 + num1
        if lik < _TINY:
            lik = _TINY
        loglik += math.log(lik)

        xi0 = num0 / lik
        xi1 = num1 / lik
        if want_filtered:
            filt[t, 0] = xi0
            filt[t, 1] = xi1

        # predicción del siguiente régimen: xi_filt @ P
        xi_pred0 = xi0 * p00 + xi1 * (1.0 - p11)
        xi_pred1 = xi0 * (1.0 - p00) + xi1 * p11

        # actualización de varianzas (paralelo, para t+1)
        eps2 = r * r
        h0 = om0 + al0 * eps2 + be0 * h0
        h1 = om1 + al1 * eps2 + be1 * h1
        if h0 < _TINY:
            h0 = _TINY
        if h1 < _TINY:
            h1 = _TINY

    return loglik, filt


def _uncond_var(om: float, al: float, be: float, fallback: float) -> float:
    """Varianza incondicional ω/(1-α-β); fallback (var muestral) si no estacionario."""
    persist = al + be
    if persist < 1.0 - 1e-6 and om > 0:
        v = om / (1.0 - persist)
        if np.isfinite(v) and v > 0:
            return float(v)
    return float(fallback)


class MSGarchRegime(RegimeDetector):
    """MS-GARCH(1,1)-t de Haas-Mittnik-Paolella (2004), 2 regímenes, ML por filtro de Hamilton.

    Parameters
    ----------
    feature : str
        Columna de X con el retorno log del S&P 500 (escala natural, sin ×100).
    scale : float
        Escala de los retornos para el ajuste (×100, como `arch`/D6: estabilidad ML).
    n_init : int
        Nº de arranques del optimizador (multistart, mitiga óptimos locales del ML
        no convexo). En walk-forward conviene 1-2 por coste; in-sample 2-3.
    maxiter : int
        Iteraciones máximas de L-BFGS-B por arranque.
    """

    def __init__(
        self,
        feature: str = "SP500_ret",
        scale: float = 100.0,
        n_init: int = 2,
        maxiter: int = 200,
        random_state: int | None = 42,
    ) -> None:
        super().__init__(n_states=2, random_state=random_state)
        self.feature = str(feature)
        self.scale = float(scale)
        self.n_init = int(n_init)
        self.maxiter = int(maxiter)
        self._params: dict | None = None              # parámetros congelados (escala interna)
        self._train_returns: pd.Series | None = None  # burn-in para filtrado causal
        self._sample_var: float = 1.0                 # var muestral (escala interna) para init h
        self._loglik: float = float("nan")
        self._nparams: int = 10  # mu, (om,al,be)x2, p00, p11, nu

    # ------------------------------------------------------------------ #
    # Identidad
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return "msgarch_regime"

    @property
    def bibliography(self) -> list[str]:
        # Claves verificadas en docs/references.bib.
        return [
            "vol_haasmittnikpaolella2004",  # MS-GARCH paralelo path-independiente
            "vol_gray1996",                 # MS-GARCH con colapso (alternativa)
            "vol_marcucci2005",             # MS-GARCH para volatilidad
            "vol_bollerslev1986",           # GARCH(1,1) por régimen
            "vol_bollerslev1987",           # innovaciones t
            "hamilton1989",                 # filtro de Hamilton / régimen de Markov
        ]

    # ------------------------------------------------------------------ #
    # Utilidad: retorno limpio (escala interna) de X
    # ------------------------------------------------------------------ #
    def _returns(self, X: pd.DataFrame) -> pd.Series:
        if self.feature not in X.columns:
            raise KeyError(
                f"{self.name}: falta la columna '{self.feature}' en X "
                f"(columnas: {list(X.columns)})."
            )
        return X[self.feature].dropna()

    # ------------------------------------------------------------------ #
    # Empaquetado de parámetros <-> vector del optimizador
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bounds() -> list[tuple[float, float]]:
        return [
            (-5.0, 5.0),      # mu
            (1e-6, 50.0),     # om0
            (1e-6, 0.5),      # al0
            (1e-6, 0.999),    # be0
            (1e-6, 50.0),     # om1
            (1e-6, 0.5),      # al1
            (1e-6, 0.999),    # be1
            (0.50, 0.9999),   # p00  (regímenes persistentes; identifica calma)
            (0.50, 0.9999),   # p11
            (2.1, 60.0),      # nu (>2 para varianza finita)
        ]

    def _negloglik(self, theta: np.ndarray, y: np.ndarray) -> float:
        mu, om0, al0, be0, om1, al1, be1, p00, p11, nu = theta
        # penalización suave de estacionariedad por régimen (α+β<1)
        pen = 0.0
        for al, be in ((al0, be0), (al1, be1)):
            s = al + be
            if s >= _PERSIST_CAP:
                pen += 1e4 * (s - _PERSIST_CAP + 1e-3) ** 2 + 1e3
        h0 = _uncond_var(om0, al0, be0, self._sample_var)
        h1 = _uncond_var(om1, al1, be1, self._sample_var)
        try:
            ll, _ = msgarch_filter(
                y, mu, om0, al0, be0, om1, al1, be1, p00, p11, nu, h0, h1, False
            )
        except (ValueError, OverflowError):
            return 1e12
        if not np.isfinite(ll):
            return 1e12
        return -ll + pen

    def _init_thetas(self, y: np.ndarray) -> list[np.ndarray]:
        """Arranques del multistart: un régimen de baja vol y otro de alta vol."""
        v = float(np.var(y))
        mu0 = float(np.mean(y))
        base = [
            # (regimen calmo: poco omega, beta alto), (crisis: mas omega, mas alpha)
            np.array([mu0, 0.02 * v, 0.05, 0.90, 0.20 * v, 0.15, 0.80, 0.98, 0.94, 7.0]),
            np.array([mu0, 0.05 * v, 0.08, 0.88, 0.40 * v, 0.20, 0.74, 0.97, 0.90, 10.0]),
            np.array([mu0, 0.10 * v, 0.10, 0.85, 0.30 * v, 0.10, 0.85, 0.99, 0.96, 5.0]),
        ]
        return base[: max(1, self.n_init)]

    # ------------------------------------------------------------------ #
    # Ajuste: ML por L-BFGS-B (multistart), congela parámetros
    # ------------------------------------------------------------------ #
    def fit(self, X_train: pd.DataFrame) -> "MSGarchRegime":
        r = self._returns(X_train)
        y = (r * self.scale).to_numpy(dtype=float)
        self._sample_var = float(np.var(y)) if len(y) > 1 else 1.0
        bounds = self._bounds()

        best_ll = -np.inf
        best_theta = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for theta0 in self._init_thetas(y):
                try:
                    res = minimize(
                        self._negloglik, theta0, args=(y,), method="L-BFGS-B",
                        bounds=bounds, options={"maxiter": self.maxiter},
                    )
                except Exception:  # noqa: BLE001
                    continue
                ll = -float(res.fun)
                if np.isfinite(ll) and ll > best_ll:
                    best_ll = ll
                    best_theta = np.asarray(res.x, dtype=float)

        if best_theta is None:  # degenerado: usa el primer arranque tal cual
            best_theta = self._init_thetas(y)[0]
            best_ll = -self._negloglik(best_theta, y)

        keys = ["mu", "om0", "al0", "be0", "om1", "al1", "be1", "p00", "p11", "nu"]
        self._params = dict(zip(keys, best_theta.tolist()))
        self._train_returns = r.copy()
        self._loglik = float(best_ll)
        self._is_fitted = True
        # Orden económico provisional; walk_forward lo re-fija con market_returns.
        self.label_states_economically(X_train)
        return self

    # ------------------------------------------------------------------ #
    # Posterior FILTRADO causal con burn-in (orden interno: 0,1)
    # ------------------------------------------------------------------ #
    def _filtered_on_returns(self, r: pd.Series, with_burnin: bool) -> pd.Series:
        """Devuelve (filt DataFrame indexado por r.index) — P(s_t=k | y<=t), causal.

        Si `with_burnin`, antepone retornos de train anteriores al primer índice de r
        y descarta esa porción tras filtrar (arranque con pasado real). Si no, filtra
        r sola desde la distribución estacionaria (modo in-sample)."""
        p = self._params
        if with_burnin and self._train_returns is not None and len(r):
            ctx = self._train_returns[self._train_returns.index < r.index[0]]
        else:
            ctx = r.iloc[:0]
        full = pd.concat([ctx, r])
        y = (full * self.scale).to_numpy(dtype=float)
        h0 = _uncond_var(p["om0"], p["al0"], p["be0"], self._sample_var)
        h1 = _uncond_var(p["om1"], p["al1"], p["be1"], self._sample_var)
        _, filt = msgarch_filter(
            y, p["mu"], p["om0"], p["al0"], p["be0"], p["om1"], p["al1"], p["be1"],
            p["p00"], p["p11"], p["nu"], h0, h1, True,
        )
        out = pd.DataFrame(filt[len(ctx):], index=r.index, columns=[0, 1])
        return out

    def _filtered_canonical(self, X: pd.DataFrame, with_burnin: bool) -> np.ndarray:
        """(len(X), 2) posterior filtrado en orden interno, reindexado a X.index."""
        r = self._returns(X)
        if len(r) == 0:
            return np.zeros((len(X), 2), dtype=float)
        filt = self._filtered_on_returns(r, with_burnin)
        full = filt.reindex(X.index).ffill()
        full.iloc[0] = full.iloc[0].fillna(0.5)  # primer día sin retorno -> neutro
        full = full.fillna(0.5)
        return full.to_numpy(dtype=float)

    # ------------------------------------------------------------------ #
    # Predicción interna (sin canonicalizar): argmax del filtrado in-sample
    # ------------------------------------------------------------------ #
    def _predict_states(self, X: pd.DataFrame) -> np.ndarray:
        if self._params is None:
            raise RuntimeError(f"{self.name}: llama a fit() antes de predecir.")
        filt = self._filtered_canonical(X, with_burnin=False)
        return filt.argmax(axis=1)

    # ------------------------------------------------------------------ #
    # Predicción CAUSAL online (filtrado con burn-in) — la que usa walk_forward
    # ------------------------------------------------------------------ #
    def predict_online(self, X: pd.DataFrame, refit: bool = False) -> np.ndarray:
        """Etiqueta dura CAUSAL canónica: argmax del posterior FILTRADO (y<=t)."""
        self._check_fitted()
        raw = self._filtered_canonical(X, with_burnin=True).argmax(axis=1)
        return self._apply_canonical(raw)

    # ------------------------------------------------------------------ #
    # Probabilidades FILTRADAS (causales) en orden CANÓNICO
    # ------------------------------------------------------------------ #
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Posterior FILTRADO P(s_t | y<=t) en orden económico canónico (con burn-in)."""
        self._check_fitted()
        filt = self._filtered_canonical(X, with_burnin=True)
        if self._canonical_order is None:
            return filt
        return filt[:, self._canonical_order]

    # ------------------------------------------------------------------ #
    # Verificación: varianza incondicional por estado canónico (crisis = alta)
    # ------------------------------------------------------------------ #
    def uncond_vol_canonical(self) -> np.ndarray:
        """Vol incondicional (sqrt var, escala interna ×scale) por estado canónico."""
        self._check_fitted()
        p = self._params
        v = np.array([
            _uncond_var(p["om0"], p["al0"], p["be0"], self._sample_var),
            _uncond_var(p["om1"], p["al1"], p["be1"], self._sample_var),
        ])
        vol = np.sqrt(v)
        if self._canonical_order is None:
            return vol
        return vol[self._canonical_order]

    def transition_canonical(self) -> np.ndarray:
        """Matriz de transición fila-estocástica P(s_t=j|s_{t-1}=i) en orden canónico."""
        self._check_fitted()
        p = self._params
        A = np.array([[p["p00"], 1 - p["p00"]], [1 - p["p11"], p["p11"]]])
        if self._canonical_order is None:
            return A
        order = self._canonical_order
        return A[np.ix_(order, order)]

    # ------------------------------------------------------------------ #
    # Bondad de ajuste (logL del MS-GARCH -> AIC/BIC del núcleo)
    # ------------------------------------------------------------------ #
    def score(self, X: pd.DataFrame) -> float:
        """Log-verosimilitud del MS-GARCH-t bajo parámetros congelados sobre X."""
        self._check_fitted()
        r = self._returns(X)
        if len(r) == 0:
            return float("nan")
        p = self._params
        y = (r * self.scale).to_numpy(dtype=float)
        h0 = _uncond_var(p["om0"], p["al0"], p["be0"], self._sample_var)
        h1 = _uncond_var(p["om1"], p["al1"], p["be1"], self._sample_var)
        ll, _ = msgarch_filter(
            y, p["mu"], p["om0"], p["al0"], p["be0"], p["om1"], p["al1"], p["be1"],
            p["p00"], p["p11"], p["nu"], h0, h1, False,
        )
        return float(ll)

    def n_parameters(self) -> int:
        """10 libres: μ, (ω,α,β)×2 regímenes = 6, p00, p11 (transición), ν compartido."""
        return int(self._nparams)
