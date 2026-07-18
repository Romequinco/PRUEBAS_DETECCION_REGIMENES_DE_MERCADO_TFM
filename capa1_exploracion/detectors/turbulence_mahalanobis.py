"""
turbulence_mahalanobis.py — D10 `turbulence_mahalanobis` (FASE 3, Tanda 3).
Familia F1: MULTIVARIANTE (Kritzman, Page & Turkington 2012; Gulko 2002).

Detector de régimen basado en el **índice de turbulencia financiera** de Kritzman
et al. (2012): la **distancia de Mahalanobis** multivariante del vector de mercado
del día respecto a su distribución histórica,

        d_t = (x_t − μ)ᵀ Σ⁻¹ (x_t − μ),

donde μ (vector de medias) y Σ (matriz de covarianzas) se estiman de forma
**CAUSAL EXPANDING**: para el día t solo se usan las observaciones ESTRICTAMENTE
anteriores (`x_0 … x_{t−1}`), nunca toda la muestra ni la propia x_t. d_t es grande
cuando el vector de mercado del día es "raro" respecto a su covarianza histórica:
o bien magnitudes extremas, o bien un **patrón de co-movimiento atípico** (colapso /
inversión de correlaciones). Justo esa segunda parte — la geometría de la matriz de
covarianzas — es lo que las reglas UNIVARIANTES (D1 VIX, D6 GARCH-equity) NO ven.

De distancia a estado (2 estados)
---------------------------------
La turbulencia es una señal continua; el régimen binario (0=calma, 1=crisis) se
obtiene **umbralizando** d_t con un percentil del train (q_in) + **histéresis**
(q_out < q_in, banda muerta) + **dwell-time** mínimo (anti-flickering). Mismo
autómata causal que D1 (rule_vix) y D6 (garch), pero sobre la distancia de
Mahalanobis en vez del nivel de VIX o la sigma GARCH.

Causalidad de d_t en walk-forward (lo importante)
-------------------------------------------------
La covarianza Mahalanobis debe ser expanding causal. La distancia del día t exige
TODAS las observaciones anteriores, incluidas las del train. Por eso `fit(train)`
GUARDA el train y, al predecir un bloque de test, se antepone como **burn-in/contexto**
(igual que el burn-in de la recursión de varianza de D6): se concatena
`[train anterior al bloque] + bloque` y se recorren los acumuladores expanding hacia
delante, de modo que μ_t y Σ_t usan solo datos < t. No se reestima nada por fold
(Mahalanobis no ajusta modelo): el coste por fold es trivial.

Features y ventana — 2013 DEBE ser OOS (contraste clave de CP2)
---------------------------------------------------------------
Vector multivariante de **cambios causales** desde **1990** (4 features, todas con
histórico largo en `data/raw/raw_panel.parquet`):

    SP500_ret        retorno log del S&P 500            (equity)
    VIX_change       Δ nivel de VIX                     (volatilidad implícita)
    DXY_change       Δlog del dólar (DXY)               (divisa / flight-to-quality)
    yield_slope_chg  Δ pendiente 10Y−3M                 (tipos / curva)

Se usan los CAMBIOS CRUDOS (no z-scores): la propia Σ⁻¹ de Mahalanobis estandariza
y decorrela las features, así que normalizarlas antes sería redundante y rompería la
métrica. NO se incluyen HYG/oro: restringirían el histórico a 2007 y mandarían 2013
al train. Con inicio en 1990 y train de ~8 años, el OOS empieza ~1998 → **2008,
2011, 2020 y 2022 son OOS** (las 4 crisis), y **2013 (taper) y 2018 (trampas) también
OOS**. Esto permite el contraste clave: ¿capta D10 el colapso de correlaciones de
2013 que D4 (HMM gaussiano) y D6 (GARCH equity) NO tapan?

Bibliografía
------------
kritzman2012 — financial turbulence = distancia de Mahalanobis multivariante.
gulko2002    — descorrelación / reequilibrado en crisis (colapso de correlaciones).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.detector_base import RegimeDetector


class TurbulenceMahalanobis(RegimeDetector):
    """Índice de turbulencia de Kritzman (Mahalanobis expanding causal) + umbral.

    Parameters
    ----------
    features : list[str] | None
        Columnas de X que forman el vector multivariante x_t. Si None, usa todas
        las columnas de X (se asume que X ya trae solo las features del vector).
    q_in : float
        Percentil (sobre la turbulencia d del train) que define τ_in, umbral de
        ENTRADA a crisis (p. ej. 0.90 = entra cuando d supera su p90 del train).
    q_out : float
        Percentil que define τ_out, umbral de SALIDA (< q_in → banda muerta).
    min_dwell : int
        Nº mínimo de días en crisis antes de poder salir (anti-flickering).
    cov_min_periods : int
        Mínimo de observaciones anteriores para estimar μ y Σ (antes → d = NaN,
        estado = calma). Debe superar holgadamente la dimensión para invertir Σ.
    ridge : float
        Regularización diagonal (× traza/dim) que se suma a Σ antes de invertir,
        para estabilidad numérica cuando Σ está mal condicionada.
    """

    def __init__(
        self,
        features: list[str] | None = None,
        q_in: float = 0.90,
        q_out: float = 0.70,
        min_dwell: int = 5,
        cov_min_periods: int = 252,
        ridge: float = 1e-6,
        random_state: int | None = 42,
    ) -> None:
        super().__init__(n_states=2, random_state=random_state)
        if not (0.0 < q_out < q_in < 1.0):
            raise ValueError(
                f"Se requiere 0 < q_out < q_in < 1 (histéresis); recibido "
                f"q_out={q_out}, q_in={q_in}."
            )
        self.features = list(features) if features is not None else None
        self.q_in = float(q_in)
        self.q_out = float(q_out)
        self.min_dwell = int(min_dwell)
        self.cov_min_periods = int(cov_min_periods)
        self.ridge = float(ridge)
        self._train_X: pd.DataFrame | None = None   # contexto/burn-in para d causal
        self._tau_in: float | None = None
        self._tau_out: float | None = None

    # ------------------------------------------------------------------ #
    # Identidad
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return "turbulence_mahalanobis"

    @property
    def bibliography(self) -> list[str]:
        # Claves verificadas en docs/references.bib.
        return ["kritzman2012", "gulko2002"]

    # ------------------------------------------------------------------ #
    # Selección del vector de features
    # ------------------------------------------------------------------ #
    def _matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = self.features if self.features is not None else list(X.columns)
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise KeyError(
                f"{self.name}: faltan columnas {missing} en X "
                f"(columnas: {list(X.columns)})."
            )
        return X[cols]

    # ------------------------------------------------------------------ #
    # Distancia de Mahalanobis CAUSAL EXPANDING (con burn-in del train)
    # ------------------------------------------------------------------ #
    def _turbulence_on(self, M: np.ndarray, eval_from: int) -> np.ndarray:
        """d_t para las filas de M con índice posicional >= eval_from.

        Recorre M acumulando media y covarianza EXPANDING. En el paso i, μ y Σ se
        estiman con las filas 0..i-1 (ESTRICTAMENTE anteriores a i); luego se mide
        d_i = (x_i − μ)ᵀ Σ⁻¹ (x_i − μ) y por último se incorpora x_i a los
        acumuladores. Así d_i nunca usa x_i ni el futuro (causal one-step).
        Devuelve un array de longitud (len(M) − eval_from).
        """
        n, dim = M.shape
        s1 = np.zeros(dim)               # suma de x
        s2 = np.zeros((dim, dim))        # suma de x xᵀ
        count = 0
        out = np.full(n - eval_from, np.nan)
        ridge_base = self.ridge
        for i in range(n):
            xi = M[i]
            if count >= self.cov_min_periods and not np.any(np.isnan(xi)):
                mu = s1 / count
                cov = s2 / count - np.outer(mu, mu)
                cov = cov * (count / (count - 1))  # insesgada (ddof=1)
                # ridge diagonal para condicionar la inversión
                reg = ridge_base * (np.trace(cov) / dim + 1e-12)
                cov_r = cov + reg * np.eye(dim)
                diff = xi - mu
                try:
                    sol = np.linalg.solve(cov_r, diff)
                    d = float(diff @ sol)
                except np.linalg.LinAlgError:
                    d = np.nan
                if i >= eval_from:
                    out[i - eval_from] = d if d >= 0 else np.nan
            # incorporar x_i a los acumuladores (si es válido)
            if not np.any(np.isnan(xi)):
                s1 += xi
                s2 += np.outer(xi, xi)
                count += 1
        return out

    def turbulence(self, X: pd.DataFrame) -> pd.Series:
        """Serie de turbulencia d_t (Mahalanobis expanding causal) para X.

        Antepone como burn-in las filas del train estrictamente anteriores al
        primer índice de X, para que la covarianza expanding de los primeros días
        de X cuente con toda la historia disponible (causal). Devuelve d_t alineada
        a X.index.
        """
        Mx = self._matrix(X)
        if self._train_X is not None:
            ctx = self._matrix(self._train_X)
            ctx = ctx[ctx.index < Mx.index[0]] if len(Mx) else ctx
            full = pd.concat([ctx, Mx])
            eval_from = len(ctx)
        else:
            full = Mx
            eval_from = 0
        d = self._turbulence_on(full.values.astype(float), eval_from)
        return pd.Series(d, index=Mx.index, name="turbulence")

    # ------------------------------------------------------------------ #
    # Ajuste: guarda el train (burn-in) y fija umbrales causales sobre d
    # ------------------------------------------------------------------ #
    def fit(self, X_train: pd.DataFrame) -> "TurbulenceMahalanobis":
        self._train_X = self._matrix(X_train).copy()
        # Turbulencia in-sample del train (sin contexto previo: expanding desde 0).
        d_train = self._turbulence_on(self._train_X.values.astype(float), 0)
        d_valid = d_train[~np.isnan(d_train)]
        if d_valid.size == 0:
            raise ValueError(
                f"{self.name}: train demasiado corto para estimar la covarianza "
                f"(cov_min_periods={self.cov_min_periods})."
            )
        self._tau_in = float(np.quantile(d_valid, self.q_in))
        self._tau_out = float(np.quantile(d_valid, self.q_out))
        self._is_fitted = True
        # Orden económico provisional (walk_forward lo re-fija con market_returns).
        self.label_states_economically(X_train)
        return self

    # ------------------------------------------------------------------ #
    # Autómata histéresis + dwell sobre d (estados internos)
    # ------------------------------------------------------------------ #
    def _states_from_d(self, d: pd.Series) -> np.ndarray:
        """0 = calma, 1 = crisis. Entra si d>τ_in; sale si d<τ_out y se cumplió el
        dwell mínimo. CAUSAL: el estado en t depende de t-1 y d_t."""
        v = d.values
        n = len(v)
        out = np.zeros(n, dtype=int)
        state = 0
        dwell = 0
        for t in range(n):
            x = v[t]
            if np.isnan(x):
                out[t] = state
                if state == 1:
                    dwell += 1
                continue
            if state == 0:
                if x > self._tau_in:
                    state = 1
                    dwell = 1
            else:
                dwell += 1
                if x < self._tau_out and dwell >= self.min_dwell:
                    state = 0
                    dwell = 0
            out[t] = state
        return out

    def _predict_states(self, X: pd.DataFrame) -> np.ndarray:
        """Etiquetas INTERNAS (0=calma,1=crisis) por umbral de la turbulencia causal.

        Reindexa al índice completo de X (días sin d válida heredan el estado
        previo) para que len(salida)==len(X)."""
        if self._tau_in is None or self._tau_out is None:
            raise RuntimeError(f"{self.name}: llama a fit() antes de predecir.")
        d = self.turbulence(X)
        states = pd.Series(self._states_from_d(d), index=d.index)
        full = states.reindex(X.index).ffill().fillna(0.0)
        return full.astype(int).values

    # ------------------------------------------------------------------ #
    # Probabilidad blanda de crisis: sigmoide monótona de (d − τ_in)
    # ------------------------------------------------------------------ #
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(crisis) blanda = sigmoide((d − τ_in)/ancho), en orden CANÓNICO.

        Función monótona creciente de la turbulencia causal (≈0.5 en τ_in). Se
        construye en orden interno [p_calma, p_crisis] y se reordena con
        `self._canonical_order` (mismo patrón que D6)."""
        self._check_fitted()
        d = self.turbulence(X)
        width = max(self._tau_in - self._tau_out, 1e-6)
        z = (d.values - self._tau_in) / (0.5 * width)
        p_crisis = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        p_ser = pd.Series(p_crisis, index=d.index).reindex(X.index).ffill().fillna(0.0)
        pc = p_ser.values
        raw = np.column_stack([1.0 - pc, pc])  # interno: col0=calma, col1=crisis
        if self._canonical_order is None:
            return raw
        return raw[:, self._canonical_order]
