"""disponibilidad.py — Regla causal de disponibilidad de columnas de los menús P y R (ADR-010 §2.1).

En cada reajuste (cada ``fit``):

1. una columna entra solo si tiene **al menos** ``min_obs`` (756 = 3 años) observaciones no
   NaN en la ventana de entrenamiento;
2. el detector base se ajusta con las filas de la ventana en las que están **todas** las
   columnas que entran;
3. hasta el siguiente reajuste se predice con esas mismas columnas.

Es causal y no usa etiquetas: solo cuenta datos de la ventana de entrenamiento. Se implementa
como envoltorio (:class:`ConDisponibilidad`) y no dentro de cada detector.

Predicción con NaN en las columnas que entran (no lo fija el ADR; en P no pasa en el OOS, en R
sí, por los festivos de las divisas de FRED):

- ``predecir_con_nan=True``: se pasan tal cual al detector base, que los trata a su manera
  (D10 mantiene el estado del día anterior; así D10-P y D10-R son D10 con otras columnas).
- ``predecir_con_nan=False`` (por defecto): el base predice solo las filas completas y una fila
  incompleta hereda el estado y las probabilidades de la última fila completa (calma si no hay
  ninguna antes). Es causal: solo mira hacia atrás.

Diagnóstico: :meth:`ConDisponibilidad.resumen_reajuste` devuelve, del último ``fit``, la fecha
del último día de entrenamiento, las columnas que entran y las que no, las filas usadas, el
orden canónico tras ``label_states_economically`` y, si el base lo expone, su
``diagnostico()``. El ejecutor (``regimenes.benchmark.pool``) lo recoge en cada reajuste.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd

from regimenes.detectores.base import RegimeDetector

PAQUETE = "regimenes.detectores.pool"
MIN_OBS = 756


def resolver_clase(base: str) -> type:
    """``"modulo.Clase"`` dentro de :data:`PAQUETE` o ruta completa importable ``"a.b.c.Clase"``."""
    modulo, _, clase = str(base).rpartition(".")
    if not modulo:
        raise ValueError(f"base debe ser 'modulo.Clase' o una ruta completa; recibido {base!r}")
    try:
        mod = import_module(modulo if modulo.startswith("regimenes.") else f"{PAQUETE}.{modulo}")
    except ModuleNotFoundError:
        mod = import_module(modulo)
    return getattr(mod, clase)


def columnas_disponibles(X: pd.DataFrame, min_obs: int = MIN_OBS) -> list[str]:
    """Columnas de ``X`` con al menos ``min_obs`` valores no NaN, en el orden de ``X``."""
    cuenta = X.notna().sum()
    return [c for c in X.columns if int(cuenta[c]) >= int(min_obs)]


class ConDisponibilidad(RegimeDetector):
    """Envoltorio que aplica la regla de disponibilidad del ADR-010 §2.1 a un detector base.

    Parameters
    ----------
    base : str
        Clase del detector: ``"modulo.Clase"`` dentro de ``regimenes.detectores.pool`` o ruta
        completa importable (p. ej. la clase oficial de D10, sin modificarla).
    params : dict | None
        Argumentos del constructor del base (en cada ``fit`` se crea uno nuevo).
    min_obs : int
        Observaciones no NaN mínimas de una columna en la ventana para entrar (756).
    predecir_con_nan : bool
        Ver el docstring del módulo.
    """

    def __init__(self, base: str, params: dict | None = None, min_obs: int = MIN_OBS,
                 predecir_con_nan: bool = False) -> None:
        self.base_path = str(base)
        self.params = dict(params or {})
        self.min_obs = int(min_obs)
        self.predecir_con_nan = bool(predecir_con_nan)
        self._cls = resolver_clase(self.base_path)
        self.base: RegimeDetector = self._cls(**self.params)
        super().__init__(n_states=self.base.n_states,
                         random_state=getattr(self.base, "random_state", None))
        self.columnas_: list[str] | None = None
        self.columnas_fuera_: list[str] = []

    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return f"{self.base.name}+disp{self.min_obs}"

    @property
    def bibliography(self) -> list[str]:
        return list(self.base.bibliography)

    @property
    def n_states(self) -> int:
        return self.base.n_states

    @property
    def crisis_state(self) -> int:
        return self.base.crisis_state

    # ------------------------------------------------------------------ #
    def _seleccion(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columnas_ is None:
            raise RuntimeError(f"{self.name}: llama a fit() antes de predecir.")
        faltan = [c for c in self.columnas_ if c not in X.columns]
        if faltan:
            raise KeyError(f"{self.name}: faltan columnas {faltan}")
        return X[self.columnas_]

    def fit(self, X_train: pd.DataFrame) -> "ConDisponibilidad":
        cols = columnas_disponibles(X_train, self.min_obs)
        if not cols:
            raise ValueError(f"{self.name}: ninguna columna con >= {self.min_obs} observaciones "
                             f"en la ventana ({len(X_train)} filas)")
        Xc = X_train[cols].dropna(how="any")
        if Xc.empty:
            raise ValueError(f"{self.name}: ninguna fila completa con las columnas que entran")
        self.columnas_ = cols
        self.columnas_fuera_ = [c for c in X_train.columns if c not in cols]
        self.base = self._cls(**self.params)
        self.base.fit(Xc)
        self._is_fitted = True
        fila = {
            "fin_train": X_train.index[-1], "n_filas_ventana": int(len(X_train)),
            "n_filas_ajuste": int(len(Xc)), "n_columnas": len(cols),
            "columnas_entran": "|".join(cols), "columnas_fuera": "|".join(self.columnas_fuera_),
        }
        diag = getattr(self.base, "diagnostico", None)
        if callable(diag):
            fila.update(diag())
        self._fila = fila
        return self

    def resumen_reajuste(self) -> dict[str, Any]:
        """Ficha del último ``fit`` (ver el docstring del módulo)."""
        return dict(getattr(self, "_fila", None) or {})

    # ------------------------------------------------------------------ #
    def label_states_economically(self, X: pd.DataFrame, market_returns: pd.Series | None = None
                                  ) -> "ConDisponibilidad":
        Xs = self._seleccion(X)
        completas = Xs.notna().all(axis=1).to_numpy()
        mr = None
        if market_returns is not None:
            mr = pd.Series(np.asarray(market_returns, dtype=float), index=X.index)[completas]
        self.base.label_states_economically(Xs[completas], market_returns=mr)
        orden = getattr(self.base, "_canonical_order", None)
        if getattr(self, "_fila", None) is not None and orden is not None:
            self._fila["orden_canonico"] = "|".join(str(int(o)) for o in orden)
        return self

    def _con_relleno(self, X: pd.DataFrame, metodo: str):
        """Aplica ``metodo`` del base a las filas completas y rellena hacia delante el resto."""
        Xs = self._seleccion(X)
        if self.predecir_con_nan:
            return getattr(self.base, metodo)(Xs)
        completas = Xs.notna().all(axis=1).to_numpy()
        proba = metodo == "predict_proba"
        if completas.all():
            return np.asarray(getattr(self.base, metodo)(Xs))
        n_s = self.n_states
        if proba:
            out = np.full((len(Xs), n_s), np.nan)
            if completas.any():
                out[completas] = np.asarray(self.base.predict_proba(Xs[completas]), dtype=float)
            df = pd.DataFrame(out).ffill()
            calma = np.zeros(n_s)
            calma[0] = 1.0
            vals = df.to_numpy()
            vals[np.isnan(vals).any(axis=1)] = calma
            return vals
        out = pd.Series(np.nan, index=range(len(Xs)))
        if completas.any():
            out[completas] = np.asarray(getattr(self.base, metodo)(Xs[completas]), dtype=float)
        return out.ffill().fillna(0).astype(int).to_numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self._con_relleno(X, "predict"))

    def predict_online(self, X: pd.DataFrame, refit: bool = False) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self._con_relleno(X, "predict_online"))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self._con_relleno(X, "predict_proba"))

    def _predict_states(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.base._predict_states(self._seleccion(X).dropna(how="any")))

    # ------------------------------------------------------------------ #
    def score(self, X: pd.DataFrame) -> float:
        return float(self.base.score(self._seleccion(X).dropna(how="any")))

    def n_parameters(self) -> int:
        return self.base.n_parameters()

    def aic(self, X: pd.DataFrame) -> float:
        return float(self.base.aic(self._seleccion(X).dropna(how="any")))

    def bic(self, X: pd.DataFrame) -> float:
        return float(self.base.bic(self._seleccion(X).dropna(how="any")))

    def informe_oos(self, X: pd.DataFrame, oos: pd.DatetimeIndex) -> dict[str, Any]:
        """Lo que reporte el base (si expone ``informe_oos``) sobre las columnas elegidas."""
        f = getattr(self.base, "informe_oos", None)
        return dict(f(self._seleccion(X), oos)) if callable(f) else {}
