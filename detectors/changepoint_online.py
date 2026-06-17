"""
changepoint_online.py — D7 `changepoint_online` (FASE 3, Tanda 3). Familia F6:
CHANGE-POINT.

Detector de cambio estructural ONLINE y CAUSAL sobre la volatilidad del retorno log
del S&P 500. Núcleo: un **CUSUM de Page (1954)** secuencial sobre un estadístico de
volatilidad (|retorno| o retorno²), que detecta los cambios de nivel de varianza con
cierto retardo y SIN mirar el futuro. Dos estados: 0 = calma (baja vol), 1 = crisis
(tramo de alta vol tras un cambio al alza).

Por qué CUSUM ONLINE (y no PELT/BinSeg de ruptures)
---------------------------------------------------
El reto de la familia change-point es que la variante potente de `ruptures`
(PELT — cp_killick2012, BinSeg — cp_truong2020) es OFFLINE y ANTI-causal: segmenta
mirando TODA la serie, así que no puede usarse en evaluación walk-forward (vería el
futuro del propio bloque). El CUSUM de Page es la variante SECUENCIAL/online: acumula
evidencia de un cambio de nivel y dispara una alarma cuando el cumulativo supera un
umbral `h`, usando solo datos <= t. PELT se usa SOLO en el notebook como ORÁCULO
in-sample (marcado NO causal) para comparar dónde caen los cambios.

De change-point a 2 estados recurrentes (el reto de integración)
----------------------------------------------------------------
Un change-point SEGMENTA, no etiqueta estados recurrentes. El mapeo a {calma, crisis}:
- Autómata de 2 estados gobernado por dos CUSUM de una cara (Page):
  * En CALMA (0) vigilamos un cambio AL ALZA de la vol (C+). Si C+ > h → entramos en
    crisis (1) y reseteamos los acumuladores.
  * En CRISIS (1) vigilamos un cambio A LA BAJA (C−). Si C− > h → volvemos a calma (0).
  El umbral `h` da PERSISTENCIA por construcción (hace falta evidencia acumulada para
  conmutar) → poco flickering, sin necesidad de dwell explícito.
- QUÉ tramo es "crisis" lo decide el NÚCLEO vía `label_states_economically` (Arreglo 4,
  vol-primario): el tramo con mayor σ de los retornos del S&P 500 queda como crisis.
  D7 solo separa por nivel de vol; el núcleo pone la polaridad → crisis = alta vol, sin
  inversión (verificado en walk-forward).

Coste gaussiano vs robusto (HIPÓTESIS CP2)
------------------------------------------
CP2 para D7: "detección temprana (lead/lag) pero riesgo de FALSAS ALARMAS con
outliers; preferir kernel/robusto frente a CUSUM gaussiano". Con kurtosis 25–40 (EDA),
un CUSUM sobre el retorno² (coste gaussiano) da un peso enorme a un único día de cola
(p. ej. −7 %): `r²` explota y el acumulado dispara la alarma por un outlier aislado,
no por un cambio sostenido de régimen. Por eso D7 ofrece dos costes y el notebook los
compara:
- `cost='gaussian'`: estadístico = retorno², estandarizado con media/desv del train.
  Sensible a colas (más falsas alarmas).
- `cost='robust'` (DEFECTO): estadístico = |retorno|, estandarizado con MEDIANA y MAD
  del train (escala robusta) y WINSORIZADO a ±`clip`. Acota la influencia de cada día
  de cola → menos falsas alarmas por outlier, manteniendo la detección del cambio
  sostenido. Es el espíritu "robusto/kernel" que pide CP2.

Causalidad en walk-forward (patrón burn-in, como D6)
----------------------------------------------------
El CUSUM es causal nativo (C±_t depende solo de z_s con s <= t). El riesgo del
walk-forward (bloques de `step` días con detector nuevo por fold) es que el autómata
arranque en frío en CALMA cada bloque y pierda un tramo de crisis que ya venía de
antes. Se evita con el mismo patrón burn-in de D6: en cada bloque se RE-EJECUTA el
CUSUM sobre `[retornos de train anteriores al bloque] + bloque` con la base (μ0/escala
y umbrales) CONGELADA del train, y se devuelve solo la parte del bloque. Como el CUSUM
solo mira hacia atrás, el estado al inicio del bloque refleja la historia real → causal
y continuo entre folds.

Ventana (LARGA)
---------------
D7 modela solo el S&P 500 (desde 1985 en `data/raw/raw_panel.parquet`), así que el
walk-forward cubre 2008 y 2011 OUT-OF-SAMPLE. returns = log(SP500/SP500.shift(1)).

Bibliografía
------------
cp_page1954        — CUSUM (detección secuencial online), núcleo del detector.
cp_inclantiao1994  — ICSS / cumulative-sum-of-squares para cambios de varianza (motiva
                     monitorizar la varianza del retorno).
cp_killick2012     — PELT (oráculo OFFLINE de comparación in-sample en el notebook).
cp_truong2020      — survey CPD offline (base de `ruptures`, usado como oráculo).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.detector_base import RegimeDetector


class ChangepointOnline(RegimeDetector):
    """CUSUM de Page online + autómata de 2 estados sobre la vol del S&P 500.

    Parameters
    ----------
    feature : str
        Columna de X con el retorno log del S&P 500 (escala natural).
    cost : {'robust', 'gaussian'}
        Estadístico monitorizado y su estandarización (ver módulo):
        - 'robust'  : log|retorno| estandarizado con MEDIANA/MAD del train + winsorizado
                      a ±`clip`. Defecto (escala robusta + colas acotadas → simétrico y
                      menos falsas alarmas).
        - 'gaussian': retorno² estandarizado con MEDIA/desv del train (coste cuadrático
                      L2, muy sensible a las colas/kurtosis).
    k : float
        Holgura (slack/reference) del CUSUM en unidades del estadístico estandarizado.
        Cuanto mayor, más cambio hace falta para acumular evidencia.
    h : float
        Umbral de alarma del CUSUM. Mayor `h` → detección más tardía pero menos falsas
        alarmas (más persistencia).
    clip : float
        Winsorización (solo en 'robust'): el z estandarizado se acota a ±clip.
    """

    def __init__(
        self,
        feature: str = "SP500_ret",
        cost: str = "robust",
        k: float = 0.5,
        h: float = 5.0,
        clip: float = 3.0,
        random_state: int | None = 42,
    ) -> None:
        super().__init__(n_states=2, random_state=random_state)
        if cost not in ("robust", "gaussian"):
            raise ValueError(f"cost debe ser 'robust' o 'gaussian'; recibido {cost!r}.")
        self.feature = str(feature)
        self.cost = str(cost)
        self.k = float(k)
        self.h = float(h)
        self.clip = float(clip)
        # Base congelada del train (estandarización causal del estadístico de vol).
        self._center: float | None = None   # media (gaussian) o mediana (robust)
        self._scale: float | None = None     # desv (gaussian) o 1.4826*MAD (robust)
        self._train_returns: pd.Series | None = None   # burn-in para CUSUM causal
        # Para predict_proba: media/desv del z_ewma in-sample (centra la logística).
        self._z_ewma_mu: float = 0.0
        self._z_ewma_sd: float = 1.0
        self._ewma_halflife: float = 10.0

    # ------------------------------------------------------------------ #
    # Identidad
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return "changepoint_online"

    @property
    def bibliography(self) -> list[str]:
        # Claves verificadas en docs/references.bib.
        return [
            "cp_page1954",        # CUSUM secuencial online (núcleo)
            "cp_inclantiao1994",  # cambios de varianza (motivación del estadístico)
            "cp_killick2012",     # PELT (oráculo offline del notebook)
            "cp_truong2020",      # survey CPD offline (ruptures)
        ]

    # ------------------------------------------------------------------ #
    # Estadístico de volatilidad monitorizado (sin estandarizar)
    # ------------------------------------------------------------------ #
    def _returns(self, X: pd.DataFrame) -> pd.Series:
        if self.feature not in X.columns:
            raise KeyError(
                f"{self.name}: falta la columna '{self.feature}' en X "
                f"(columnas: {list(X.columns)})."
            )
        return X[self.feature].dropna()

    def _raw_stat(self, r: pd.Series) -> pd.Series:
        """Estadístico de vol bruto: retorno² (gaussian, L2) o log|retorno| (robust).

        log|retorno| es aproximadamente SIMÉTRICO (log-normalidad de la vol), lo que
        permite que el CUSUM acumule evidencia tanto al alza como a la baja y el
        autómata pueda CONMUTAR en ambos sentidos. El retorno² (gaussian) es
        fuertemente asimétrico y dominado por las colas."""
        if self.cost == "gaussian":
            return r ** 2
        return np.log(r.abs() + 1e-6)

    def _standardize(self, raw: pd.Series) -> np.ndarray:
        """Estandariza el estadístico con la base CONGELADA del train y, si robusto,
        winsoriza a ±clip. Causal: usa solo μ0/escala fijados en fit."""
        z = (raw.values - self._center) / self._scale
        if self.cost == "robust":
            z = np.clip(z, -self.clip, self.clip)
        return z

    # ------------------------------------------------------------------ #
    # Ajuste: fija la base de estandarización causal (μ0, escala) del train
    # ------------------------------------------------------------------ #
    def fit(self, X_train: pd.DataFrame) -> "ChangepointOnline":
        r = self._returns(X_train)
        raw = self._raw_stat(r)
        if self.cost == "gaussian":
            self._center = float(np.mean(raw.values))
            self._scale = float(np.std(raw.values)) or 1.0
        else:  # robust: mediana y MAD (escala robusta de Gauss)
            med = float(np.median(raw.values))
            mad = float(np.median(np.abs(raw.values - med)))
            self._center = med
            self._scale = (1.4826 * mad) or 1.0
        self._train_returns = r.copy()
        # Calibrar la logística de predict_proba con el z_ewma in-sample del train.
        z = self._standardize(raw)
        z_ewma = pd.Series(z, index=r.index).ewm(halflife=self._ewma_halflife).mean().values
        self._z_ewma_mu = float(np.mean(z_ewma))
        self._z_ewma_sd = float(np.std(z_ewma)) or 1.0
        self._is_fitted = True
        # Orden económico provisional (walk_forward lo re-fija con market_returns).
        self.label_states_economically(X_train)
        return self

    # ------------------------------------------------------------------ #
    # Autómata CUSUM de 2 estados (Page 1954), causal
    # ------------------------------------------------------------------ #
    def _cusum_states(self, z: np.ndarray) -> np.ndarray:
        """0 = calma, 1 = crisis. En calma vigila C+ (cambio al alza de vol); al
        cruzar h entra en crisis y resetea. En crisis vigila C− (cambio a la baja);
        al cruzar h vuelve a calma. CAUSAL: el estado en t depende solo de z_{<=t}."""
        n = len(z)
        out = np.zeros(n, dtype=int)
        state = 0
        c_up = 0.0
        c_dn = 0.0
        for t in range(n):
            zt = z[t]
            if not np.isfinite(zt):
                out[t] = state
                continue
            if state == 0:
                c_up = max(0.0, c_up + (zt - self.k))
                if c_up > self.h:
                    state = 1
                    c_up = 0.0
                    c_dn = 0.0
            else:
                c_dn = max(0.0, c_dn + (-zt - self.k))
                if c_dn > self.h:
                    state = 0
                    c_up = 0.0
                    c_dn = 0.0
            out[t] = state
        return out

    def _states_on_returns(self, X: pd.DataFrame) -> pd.Series:
        """Corre el CUSUM sobre `[burn-in de train] + retornos de X` con la base
        congelada y devuelve los estados solo para el índice de retornos de X."""
        if self._center is None:
            raise RuntimeError(f"{self.name}: llama a fit() antes de predecir.")
        r = self._returns(X)
        if len(r) == 0:
            return pd.Series(dtype=int)
        if self._train_returns is not None:
            ctx = self._train_returns[self._train_returns.index < r.index[0]]
        else:
            ctx = r.iloc[:0]
        full = pd.concat([ctx, r])
        z = self._standardize(self._raw_stat(full))
        states_full = pd.Series(self._cusum_states(z), index=full.index)
        return states_full.loc[r.index]

    def _predict_states(self, X: pd.DataFrame) -> np.ndarray:
        """Etiquetas INTERNAS (0=calma, 1=crisis). Reindexa al índice completo de X
        (días sin retorno heredan el estado previo) para que len(salida)==len(X)."""
        states_on_ret = self._states_on_returns(X)
        full = states_on_ret.reindex(X.index).ffill().fillna(0.0)
        return full.astype(int).values

    # ------------------------------------------------------------------ #
    # Probabilidad blanda de crisis: logística monótona del z_ewma de la vol
    # ------------------------------------------------------------------ #
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(crisis) blanda = sigmoide del EWMA del estadístico de vol estandarizado
        (centrado/escalado con el in-sample del train), en orden CANÓNICO.

        Función monótona creciente del nivel de vol reciente: ~0.5 cuando el z_ewma
        está en su media in-sample, ↑ cuando la vol se eleva. Causal (EWMA solo mira
        atrás, con el mismo burn-in que los estados). Se construye en orden interno
        [calma, crisis] y se reordena con `self._canonical_order`."""
        self._check_fitted()
        r = self._returns(X)
        if len(r) == 0:
            return np.zeros((len(X), self.n_states))
        if self._train_returns is not None:
            ctx = self._train_returns[self._train_returns.index < r.index[0]]
        else:
            ctx = r.iloc[:0]
        full = pd.concat([ctx, r])
        z = self._standardize(self._raw_stat(full))
        z_ewma = pd.Series(z, index=full.index).ewm(halflife=self._ewma_halflife).mean()
        z_ewma = z_ewma.loc[r.index].values
        zc = (z_ewma - self._z_ewma_mu) / self._z_ewma_sd
        p_crisis_ret = 1.0 / (1.0 + np.exp(-zc))
        p_ser = pd.Series(p_crisis_ret, index=r.index).reindex(X.index).ffill().fillna(0.0)
        p_crisis = p_ser.values
        raw = np.column_stack([1.0 - p_crisis, p_crisis])  # interno: col0=calma, col1=crisis
        if self._canonical_order is None:
            return raw
        return raw[:, self._canonical_order]
