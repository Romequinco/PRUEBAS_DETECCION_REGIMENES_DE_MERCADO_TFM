"""absorption_ratio.py — D14 · Absorption ratio de Kritzman, Li, Page y Rigobon (2011) (ADR-010 §3).

Especificación pre-registrada (menú R, solo pista B), con los parámetros de Kinlaw, Kritzman y
Turkington (2011, ec. 2):

- En cada día ``t`` con las 22 rentabilidades presentes («fila completa»), matriz de covarianza
  de las **500 filas completas** más recientes ``<= t``, con ponderación exponencial de
  **semivida 250** filas (peso ``0,5^(edad/250)``, edad 0 = la fila de ``t``).
- ``AR_t`` = fracción de la varianza total (traza) que explican los ``n = 4`` mayores
  autovalores (≈ 1/5 de 22).
- Desplazamiento estandarizado sobre la serie de AR (una observación por fila completa):
  ``ΔAR_t = (media de los 15 últimos AR − media de los 252 últimos) / desviación típica de los
  252 últimos`` (``ddof = 1``).
- **Crisis = ΔAR ≥ 1** (estado interno 1). Un día sin fila completa (festivo de FRED) hereda
  el ΔAR de la última fila completa. Sin 252 AR todavía (calentamiento): calma.

No se ajusta a nada: ``fit`` solo guarda la historia para que la predicción de un bloque use
todas las filas anteriores (como el burn-in de D10). Todo se calcula con datos ``<= t``. Sin
canonicalización económica (ADR-010 §8.1, errata del 03-11-2026): ΔAR ≥ 1 es por definición el
riesgo alto de Kritzman, así que el estado interno 1 es siempre la crisis. Con la canonicalización
de D10, los días de calentamiento sin ΔAR (toda la crisis de 2008) inflaban la volatilidad del
estado 0 e invertían el orden de 2011 a 2019.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from regimenes.detectores.base import RegimeDetector


def absorption_ratio(R: np.ndarray, *, ventana: int = 500, semivida: float = 250.0,
                     n_autovectores: int = 4) -> np.ndarray:
    """AR de cada fila de ``R`` (filas completas, en orden temporal) con las ``ventana`` filas
    hasta ella incluida; NaN en las ``ventana - 1`` primeras."""
    R = np.asarray(R, dtype=float)
    n, p = R.shape
    out = np.full(n, np.nan)
    if n < ventana:
        return out
    edad = np.arange(ventana - 1, -1, -1, dtype=float)          # la última fila tiene edad 0
    w = 0.5 ** (edad / float(semivida))
    w = w / w.sum()
    k = int(min(n_autovectores, p))
    for i in range(ventana - 1, n):
        V = R[i - ventana + 1:i + 1]
        mu = w @ V
        D = V - mu
        C = (D * w[:, None]).T @ D
        ev = np.linalg.eigvalsh(C)                               # ascendentes
        traza = ev.sum()
        out[i] = float(ev[-k:].sum() / traza) if traza > 0 else np.nan
    return out


def delta_ar(ar: pd.Series, *, corto: int = 15, largo: int = 252) -> pd.Series:
    """ΔAR = (media de ``corto`` − media de ``largo``) / desviación de ``largo`` (sobre la serie de AR)."""
    m_c = ar.rolling(corto, min_periods=corto).mean()
    roll = ar.rolling(largo, min_periods=largo)
    return (m_c - roll.mean()) / roll.std(ddof=1)


class AbsorptionRatio(RegimeDetector):
    """D14 · regla ΔAR ≥ umbral sobre el absorption ratio causal (ver el docstring del módulo)."""

    def __init__(self, ventana: int = 500, semivida: float = 250.0, n_autovectores: int = 4,
                 corto: int = 15, largo: int = 252, umbral: float = 1.0,
                 random_state: int | None = 42) -> None:
        super().__init__(n_states=2, random_state=random_state)
        self.ventana = int(ventana)
        self.semivida = float(semivida)
        self.n_autovectores = int(n_autovectores)
        self.corto = int(corto)
        self.largo = int(largo)
        self.umbral = float(umbral)
        self._historia: pd.DataFrame | None = None
        self._cache: dict[tuple, pd.Series] = {}

    @property
    def name(self) -> str:
        return f"absorption_ratio_n{self.n_autovectores}_w{self.ventana}"

    @property
    def bibliography(self) -> list[str]:
        # Claves de docs/references.bib (Kinlaw, Kritzman y Turkington 2011 no está: solo fija
        # los parámetros, citados en ADR-010 §8).
        return ["kritzman2011"]

    # ------------------------------------------------------------------ #
    def _con_historia(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._historia is None or not len(X):
            return X
        previa = self._historia[self._historia.index < X.index[0]]
        return pd.concat([previa, X]) if len(previa) else X

    def serie_delta_ar(self, X: pd.DataFrame) -> pd.Series:
        """ΔAR sobre ``X.index`` (con la historia del ``fit`` delante), sin rellenar."""
        full = self._con_historia(X)
        clave = (full.index[0], full.index[-1], len(full)) if len(full) else None
        if clave in self._cache:
            d_full = self._cache[clave]
        else:
            comp = full.dropna(how="any")
            ar = pd.Series(absorption_ratio(comp.to_numpy(), ventana=self.ventana, semivida=self.semivida,
                                            n_autovectores=self.n_autovectores), index=comp.index)
            d_full = delta_ar(ar.dropna(), corto=self.corto, largo=self.largo).reindex(full.index).ffill()
            self._cache = {clave: d_full}
        return d_full.reindex(X.index)

    def fit(self, X_train: pd.DataFrame) -> "AbsorptionRatio":
        self._historia = X_train.copy()
        self._cache = {}
        self._is_fitted = True
        self.label_states_economically(X_train)
        return self

    def label_states_economically(self, X: pd.DataFrame, market_returns: pd.Series | None = None
                                  ) -> "AbsorptionRatio":
        """Sin canonicalización económica: estado interno 1 (ΔAR ≥ 1) = crisis."""
        self._canonical_order = np.arange(self.n_states)
        return self

    def resumen_reajuste(self) -> dict[str, Any]:
        """Fin de la historia y orden canónico del último ``fit`` (para el informe del ejecutor)."""
        if self._historia is None:
            return {}
        out: dict[str, Any] = {"fin_train": self._historia.index[-1],
                               "n_filas_ventana": int(len(self._historia)),
                               "n_filas_ajuste": int(self._historia.notna().all(axis=1).sum())}
        if self._canonical_order is not None:
            out["orden_canonico"] = "|".join(str(int(o)) for o in self._canonical_order)
        return out

    def _predict_states(self, X: pd.DataFrame) -> np.ndarray:
        d = self.serie_delta_ar(X).to_numpy()
        return np.where(np.isfinite(d) & (d >= self.umbral), 1, 0).astype(int)

    def informe_oos(self, X: pd.DataFrame, oos: pd.DatetimeIndex) -> dict[str, Any]:
        """Días del OOS sin ΔAR (calentamiento: se predicen en calma)."""
        d = self.serie_delta_ar(X).reindex(pd.DatetimeIndex(oos))
        sin = d.isna()
        return {"d14_dias_oos_calentamiento": int(sin.sum()),
                "d14_primer_dia_con_delta_ar": str(d.first_valid_index().date()) if sin.size and not sin.all() else ""}
