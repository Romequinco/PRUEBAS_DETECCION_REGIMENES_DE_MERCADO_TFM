"""
garch_t_vol.py — D6 `garch_t_vol` (FASE 3, Tanda 2). Familia F5: VOLATILIDAD.

Detector de régimen basado en la VOLATILIDAD CONDICIONAL de un GJR-GARCH(1,1)-t
ajustado SOLO sobre el retorno log del S&P 500. Dos estados: 0 = calma, 1 = crisis.

Idea
----
Un modelo GARCH no produce "estados tipados": produce una serie de sigma
condicional (la volatilidad esperada del día t dado el pasado). Para convertir esa
señal continua en un régimen binario se UMBRALIZA sigma con un percentil del train
(p. ej. p80), con HISTÉRESIS (banda muerta τ_in/τ_out) y DWELL-TIME mínimo para no
parpadear. Esto materializa la HIPÓTESIS CP2 de D6:

  "El GARCH no da estados tipados; se umbraliza la sigma condicional. Es causal por
   construcción y REACCIONA EL MISMO DÍA que llega el shock (la sigma de t salta con
   el retorno de t), así que DEBERÍA captar también las correcciones rápidas de
   2013 (taper tantrum) y 2018 (sell-off Q4), donde los modelos de estado latente
   gaussiano (D4) se quedan cortos. Univariante sobre equity."

Por qué GJR-GARCH-t
-------------------
- `o=1` (término de Glosten-Jagannathan-Runkle) captura la ASIMETRÍA / efecto
  apalancamiento: las caídas elevan la vol futura más que las subidas del mismo
  tamaño (vol_glosten1993, vol_zakoian1994).
- `dist='t'` (Student-t) captura las COLAS GORDAS (kurtosis 25–40 del EDA) que un
  GARCH gaussiano subestima (vol_bollerslev1987).
- Base GARCH(1,1): vol_bollerslev1986, generalizando el ARCH de vol_engle1982.
- Se trabaja en el retorno ×100 (escala recomendada por `arch`, vol_sheppard_arch).

Causalidad de la sigma en walk-forward (clave de D6)
----------------------------------------------------
La sigma condicional de GARCH en t depende SOLO del pasado (retornos y sigma de
t-1, ...) → causal nativo. El riesgo del walk-forward es REESTIMAR con el test. Lo
evitamos así:

  1. `fit(train)`: se estima el GJR-GARCH-t por ML sobre el train y se CONGELAN los
     parámetros (mu, omega, alpha, gamma, beta, nu). El umbral τ se fija como
     percentil de la sigma condicional in-sample del train.
  2. Para un bloque de test, NO se reestima: se reconstruye un `arch_model` sobre la
     serie `[burn-in de train anterior al bloque] + bloque`, se FIJAN los parámetros
     congelados con `.fix(params)` y se lee `conditional_volatility`. La recursión de
     varianza se propaga hacia delante desde el burn-in, de modo que sigma_t del
     bloque usa solo retornos <= t. El burn-in hace que el efecto del "backcast"
     inicial (único uso de estadística de muestra, para arrancar la recursión) sea
     despreciable al llegar al bloque. Es el mismo patrón de contexto/burn-in que D4.

  Se verifica en el notebook (test truncado vs completo) que la sigma del bloque no
  cambia al ocultar el futuro → causal.

Ventana (LARGA)
---------------
D6 modela solo el S&P 500, disponible desde 1985 en `data/raw/raw_panel.parquet`.
Con histórico largo el walk-forward cubre 2008 y 2011 OUT-OF-SAMPLE (a diferencia de
D4, atado a HYG desde 2007). returns = log(SP500 / SP500.shift(1)).

Bibliografía
------------
vol_bollerslev1986 — GARCH(1,1).
vol_glosten1993    — GJR-GARCH (asimetría / leverage), término o=1.
vol_bollerslev1987 — GARCH con innovaciones t-Student (colas gordas).
vol_engle1982      — ARCH (origen de la familia).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model

from src.detector_base import RegimeDetector


class GarchTVol(RegimeDetector):
    """GJR-GARCH(1,1)-t sobre el retorno del S&P 500 + umbral causal de sigma.

    Parameters
    ----------
    feature : str
        Columna de X con el retorno log del S&P 500 (escala natural, sin ×100).
    q_in : float
        Percentil (sobre la sigma condicional del train) que define τ_in, el umbral
        de ENTRADA a crisis (p. ej. 0.80 = entra cuando la sigma supera su p80 del
        train).
    q_out : float
        Percentil que define τ_out, el umbral de SALIDA (< q_in → banda muerta /
        histéresis).
    min_dwell : int
        Nº mínimo de días en crisis antes de poder salir (anti-flickering).
    scale : float
        Factor de escala de los retornos para el ajuste (arch recomienda ×100).
    """

    def __init__(
        self,
        feature: str = "SP500_ret",
        q_in: float = 0.80,
        q_out: float = 0.60,
        min_dwell: int = 5,
        scale: float = 100.0,
        random_state: int | None = 42,
    ) -> None:
        super().__init__(n_states=2, random_state=random_state)
        if not (0.0 < q_out < q_in < 1.0):
            raise ValueError(
                f"Se requiere 0 < q_out < q_in < 1 (histéresis); recibido "
                f"q_out={q_out}, q_in={q_in}."
            )
        self.feature = str(feature)
        self.q_in = float(q_in)
        self.q_out = float(q_out)
        self.min_dwell = int(min_dwell)
        self.scale = float(scale)
        self._params: pd.Series | None = None          # parámetros GARCH congelados
        self._train_returns: pd.Series | None = None    # burn-in para sigma causal
        self._tau_in: float | None = None               # umbral entrada (sigma escalada)
        self._tau_out: float | None = None              # umbral salida (sigma escalada)
        self._loglik: float = float("nan")
        self._nparams: int = 6  # mu, omega, alpha, gamma, beta, nu

    # ------------------------------------------------------------------ #
    # Identidad
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return "garch_t_vol"

    @property
    def bibliography(self) -> list[str]:
        # Claves verificadas en docs/references.bib.
        return [
            "vol_bollerslev1986",  # GARCH(1,1)
            "vol_glosten1993",     # GJR-GARCH (asimetría, o=1)
            "vol_bollerslev1987",  # innovaciones t (colas)
            "vol_engle1982",       # ARCH (origen)
        ]

    # ------------------------------------------------------------------ #
    # Utilidad: retorno limpio (sin NaN) de X, en la columna de trabajo
    # ------------------------------------------------------------------ #
    def _returns(self, X: pd.DataFrame) -> pd.Series:
        if self.feature not in X.columns:
            raise KeyError(
                f"{self.name}: falta la columna '{self.feature}' en X "
                f"(columnas: {list(X.columns)})."
            )
        return X[self.feature].dropna()

    def _new_model(self, r_scaled: pd.Series):
        """Construye el spec GJR-GARCH(1,1)-t (idéntico en fit y en .fix)."""
        return arch_model(
            r_scaled, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t"
        )

    # ------------------------------------------------------------------ #
    # Ajuste: estima por ML, congela parámetros, fija umbral causal de sigma
    # ------------------------------------------------------------------ #
    def fit(self, X_train: pd.DataFrame) -> "GarchTVol":
        r = self._returns(X_train)
        am = self._new_model(r * self.scale)
        res = am.fit(disp="off", show_warning=False)
        self._params = res.params
        self._train_returns = r.copy()          # burn-in para sigma causal del test
        self._loglik = float(res.loglikelihood)
        self._nparams = int(len(res.params))     # 6: mu, omega, alpha[1], gamma[1], beta[1], nu
        # Umbrales CAUSALES: percentiles de la sigma condicional in-sample del train.
        sigma_train = np.asarray(res.conditional_volatility)
        self._tau_in = float(np.quantile(sigma_train, self.q_in))
        self._tau_out = float(np.quantile(sigma_train, self.q_out))
        self._is_fitted = True
        # Orden económico provisional (walk_forward lo re-fija con market_returns).
        self.label_states_economically(X_train)
        return self

    # ------------------------------------------------------------------ #
    # Sigma condicional CAUSAL con burn-in (parámetros congelados)
    # ------------------------------------------------------------------ #
    def _conditional_sigma(self, X: pd.DataFrame) -> pd.Series:
        """Sigma condicional (escala ×scale) de cada t de X, causal.

        Antepone como burn-in los retornos de train estrictamente anteriores al
        primer índice de X y propaga la recursión de varianza con los parámetros
        CONGELADOS (`.fix`). Devuelve solo la parte correspondiente a X. sigma_t usa
        únicamente retornos <= t (causal).
        """
        if self._params is None:
            raise RuntimeError(f"{self.name}: llama a fit() antes de predecir.")
        r = self._returns(X)
        if len(r) == 0:
            return pd.Series(dtype=float)
        if self._train_returns is not None:
            ctx = self._train_returns[self._train_returns.index < r.index[0]]
        else:
            ctx = r.iloc[:0]
        full = pd.concat([ctx, r])
        am = self._new_model(full * self.scale)
        fixed = am.fix(self._params)
        sigma_full = pd.Series(np.asarray(fixed.conditional_volatility), index=full.index)
        return sigma_full.loc[r.index]

    # ------------------------------------------------------------------ #
    # Etiquetado económico: lo hace el NÚCLEO (Arreglo 4, vol-primario).
    # ------------------------------------------------------------------ #
    # D6 ya NO sobrescribe label_states_economically. Tras el Arreglo 4, el núcleo
    # ordena por VOLATILIDAD primaria (banda de vol) con el retorno medio solo como
    # desempate entre vols próximas, así que el estado de alta sigma (= alta std de
    # retornos) queda correctamente como crisis SIN posibilidad de inversión. El
    # antiguo override local (ordenar por sigma media) era el parche que el Arreglo 4
    # generaliza en el núcleo; se elimina para que D6 confíe en el núcleo como el
    # resto de detectores.

    # ------------------------------------------------------------------ #
    # Autómata de histéresis + dwell sobre la sigma (estados internos)
    # ------------------------------------------------------------------ #
    def _states_from_sigma(self, sigma: pd.Series) -> np.ndarray:
        """0 = calma, 1 = crisis. Entra si sigma>τ_in; sale si sigma<τ_out y se
        cumplió el dwell mínimo. CAUSAL: el estado en t depende de t-1 y sigma_t."""
        v = sigma.values
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
        """Etiquetas INTERNAS (0=calma,1=crisis) por umbral de la sigma causal.

        Reindexa al índice completo de X (incluido el primer NaN de retorno, si lo
        hubiera) manteniendo el estado previo, para que len(salida)==len(X).
        """
        sigma = self._conditional_sigma(X)
        states_on_ret = pd.Series(self._states_from_sigma(sigma), index=sigma.index)
        # Alinear al índice de X (días sin retorno heredan el estado previo).
        full = states_on_ret.reindex(X.index).ffill().fillna(0.0)
        return full.astype(int).values

    # ------------------------------------------------------------------ #
    # Probabilidad blanda de crisis: logística monótona de (sigma - τ_in)
    # ------------------------------------------------------------------ #
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(crisis) blanda = sigmoide((sigma - τ_in)/ancho), en orden CANÓNICO.

        Función monótona creciente de la sigma condicional causal: vale ~0.5 en
        τ_in. Se construye en orden interno [p_calma, p_crisis] y se reordena con
        `self._canonical_order` (como el GMM)."""
        self._check_fitted()
        sigma = self._conditional_sigma(X)
        width = max(self._tau_in - self._tau_out, 1e-6)
        z = (sigma.values - self._tau_in) / (0.5 * width)
        p_crisis_ret = 1.0 / (1.0 + np.exp(-z))
        p_ser = pd.Series(p_crisis_ret, index=sigma.index).reindex(X.index).ffill().fillna(0.0)
        p_crisis = p_ser.values
        raw = np.column_stack([1.0 - p_crisis, p_crisis])  # interno: col0=calma, col1=crisis
        if self._canonical_order is None:
            return raw
        return raw[:, self._canonical_order]

    # ------------------------------------------------------------------ #
    # Bondad de ajuste (logL del GARCH -> AIC/BIC del núcleo)
    # ------------------------------------------------------------------ #
    def score(self, X: pd.DataFrame) -> float:
        """Log-verosimilitud del GJR-GARCH-t ajustado en fit (sobre su train).

        arch maximiza la verosimilitud del modelo; se devuelve la logL del ajuste
        congelado. Sirve para AIC/BIC comparables (k=6)."""
        self._check_fitted()
        return float(self._loglik)

    def n_parameters(self) -> int:
        """6 parámetros libres: mu (media), omega, alpha[1], gamma[1] (asimetría),
        beta[1], nu (grados de libertad de la t)."""
        return int(self._nparams)
