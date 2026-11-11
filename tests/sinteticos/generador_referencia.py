"""Generador de referencia y paneles de juguete para los tests de ``regimenes.sinteticos``.

No es un test (no empieza por ``test_``): lo importan ``test_contrato_generadores``
y ``test_sinteticos_cimientos``. El generador de referencia vive aqui y no en el
paquete porque solo sirve para probar ``GeneradorBase``: es el modelo mas simple
que cumple el contrato (normal multivariante i.i.d. por regimen).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from regimenes.sinteticos.comun import GeneradorBase
from regimenes.sinteticos.espacio import derivar_sp500

COLUMNAS_JUGUETE = ["ret", "spread", "tipos", "macro"]


class GaussianoReferencia(GeneradorBase):
    """Normal multivariante i.i.d. por regimen (media y covarianza de train)."""

    nombre = "_referencia"
    familia = "tests"
    PARAMS = {"regularizacion": 1e-6, "largo_contexto": 3}

    def _fit(self, X, reg, fechas):
        d = X.shape[1]
        self.mu_ = np.zeros((self.n_regimenes_, d))
        self.chol_ = np.tile(np.eye(d), (self.n_regimenes_, 1, 1))
        for k in range(self.n_regimenes_):
            Xk = X[reg == k]
            if len(Xk) > d:
                self.mu_[k] = Xk.mean(axis=0)
                cov = np.cov(Xk, rowvar=False).reshape(d, d) + self.regularizacion * np.eye(d)
                self.chol_[k] = np.linalg.cholesky(cov)
            self.registrar(regimen=k, n_obs=len(Xk))

    def _sample(self, reg, rng, contexto):
        eps = rng.standard_normal(reg.shape + (self.d_,))
        return self.mu_[reg] + np.einsum("ptij,ptj->pti", self.chol_[reg], eps)


def regimen_juguete(n: int) -> np.ndarray:
    """Regimen por bloques: crisis (1) en tramos de 60 sesiones cada 210."""
    reg = np.zeros(n, dtype=int)
    for ini in range(150, n, 210):
        reg[ini:ini + 60] = 1
    return reg


def panel_juguete(n: int = 600, semilla: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Panel ``n`` x 4 con volatilidad x3 en crisis (sin columnas del S&P 500)."""
    rng = np.random.default_rng(semilla)
    idx = pd.bdate_range("2010-01-04", periods=n)
    reg = regimen_juguete(n)
    escala = np.where(reg == 1, 3.0, 1.0)[:, None]
    x = rng.standard_normal((n, len(COLUMNAS_JUGUETE))) * escala * np.array([0.01, 1.0, 0.5, 2.0])
    x[:, 3] += 5.0
    panel = pd.DataFrame(x, index=idx, columns=COLUMNAS_JUGUETE)
    return panel, pd.Series(reg, index=idx, name="regime")


def panel_con_sp500(n: int = 900, semilla: int = 1) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Panel de juguete con las 4 columnas derivadas del S&P 500, ``SP500_ret`` y 2 libres.

    Devuelve ``(panel sin NaN, regimen, precios)``; ``precios`` cubre todo el
    periodo (incluido el warm-up que el panel pierde).
    """
    rng = np.random.default_rng(semilla)
    idx = pd.bdate_range("2000-01-03", periods=n)
    reg = regimen_juguete(n)
    ret = rng.standard_normal(n) * np.where(reg == 1, 0.03, 0.01)
    precios = pd.Series(100.0 * np.exp(np.cumsum(ret)), index=idx, name="SP500")
    panel = derivar_sp500(precios)
    panel["FF_MKT_z"] = rng.standard_normal(n)
    panel["SP500_ret"] = np.log(precios / precios.shift(1))
    panel["macro"] = rng.standard_normal(n).cumsum() / 10
    panel = panel.dropna()
    return panel, pd.Series(reg, index=idx, name="regime").loc[panel.index], precios
