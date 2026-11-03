"""``DetectorAumentado``: ajusta un detector con [bloque sintetico] + [train real].

Contrato (pre-registro ``aumento``, «Contrato del envoltorio»):

- ``fit(train)`` pide al proveedor el bloque del pliegue, lo fecha con dias
  habiles ficticios que acaban la sesion anterior al primer dia real, se queda con
  las columnas del train y llama a ``interno.fit(concat(bloque, train))``. Sin
  bloque (proporcion 0 o pliegue sin ancla) llama a ``interno.fit(train)`` con el
  MISMO objeto: el resultado es bit a bit el del benchmark.
- Todo lo demas se delega en el interno: ``label_states_economically`` (que
  ``walk_forward`` llama con el train REAL y su ``SP500_ret``), ``predict``,
  ``predict_online``, ``predict_proba``, ``n_states``, ``crisis_state`` y ``name``.
- Diagnostico de ocupacion (opcional, cada ``cada_ocupacion`` pliegues): tras fijar
  el orden economico, fraccion de cada estado canonico en el bloque sintetico y en
  el train real. Un estado que solo ocupa el sintetico queda al final del orden,
  es decir, como crisis (``detectores/base.py``): se publica, no se excluye.

Los detectores con burn-in (D04-D08, D10, D11) recorren tambien el bloque
sintetico al re-etiquetar el train real, porque sus fechas son anteriores
(declarado en el pre-registro).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from regimenes.sinteticos.datos import COL_REGIMEN


def fechas_previas(primera: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """``n`` dias habiles consecutivos que acaban el dia habil anterior a ``primera``."""
    fin = pd.Timestamp(primera) - pd.offsets.BDay(1)
    return pd.bdate_range(end=fin, periods=int(n)).astype("datetime64[ns]")


class DetectorAumentado:
    """Envoltorio de un ``RegimeDetector`` (ver docstring del modulo).

    Parameters
    ----------
    interno:
        Instancia nueva del detector (``spec.factory()``).
    proveedor:
        ``proveedor(indice_train) -> DataFrame | None`` con el bloque del pliegue.
    ocupacion:
        Lista donde se anaden los diagnosticos de ocupacion (o ``None``).
    pliegue_ocupacion:
        ``True`` si en este pliegue se calcula la ocupacion.
    """

    def __init__(self, interno, proveedor: Callable[[pd.DatetimeIndex], pd.DataFrame | None] | None,
                 ocupacion: list | None = None, pliegue_ocupacion: bool = False) -> None:
        self.interno = interno
        self.proveedor = proveedor
        self.ocupacion = ocupacion
        self.pliegue_ocupacion = pliegue_ocupacion
        self.bloque_: pd.DataFrame | None = None

    # ------------------------------------------------------------- identidad
    @property
    def name(self) -> str:
        return f"{self.interno.name}+aumento"

    @property
    def n_states(self) -> int:
        return self.interno.n_states

    @property
    def crisis_state(self) -> int:
        return self.interno.crisis_state

    # ----------------------------------------------------------------- ajuste
    def fit(self, X_train: pd.DataFrame) -> "DetectorAumentado":
        bloque = None if self.proveedor is None else self.proveedor(X_train.index)
        if bloque is None or len(bloque) == 0:
            self.bloque_ = None
            self.interno.fit(X_train)
            return self
        faltan = [c for c in X_train.columns if c not in bloque.columns]
        if faltan:
            raise KeyError(f"El bloque sintetico no tiene las columnas {faltan} del detector.")
        sint = bloque[list(X_train.columns)].copy()
        sint.index = fechas_previas(X_train.index[0], len(sint))
        sint = sint.astype(X_train.dtypes.to_dict())
        self.bloque_ = sint
        self.regimen_bloque_ = (bloque[COL_REGIMEN].to_numpy() if COL_REGIMEN in bloque
                                else np.zeros(len(sint), dtype=int))
        self.interno.fit(pd.concat([sint, X_train]))
        return self

    def label_states_economically(self, X: pd.DataFrame, market_returns: pd.Series | None = None):
        self.interno.label_states_economically(X, market_returns=market_returns)
        if self.pliegue_ocupacion and self.ocupacion is not None and self.bloque_ is not None:
            self._registrar_ocupacion(X)
        return self

    def _registrar_ocupacion(self, X_real: pd.DataFrame) -> None:
        k = self.interno.n_states
        try:
            est_sint = np.asarray(self.interno.predict(self.bloque_))
            est_real = np.asarray(self.interno.predict(X_real))
        except Exception as exc:  # noqa: BLE001 - un diagnostico no puede tumbar el trabajo
            self.ocupacion.append({"fin_train": X_real.index[-1], "error": f"{type(exc).__name__}: {exc}"})
            return
        fila = {"fin_train": X_real.index[-1], "n_sint": len(est_sint), "n_real": len(est_real)}
        for s in range(k):
            fila[f"sint_{s}"] = float(np.mean(est_sint == s))
            fila[f"real_{s}"] = float(np.mean(est_real == s))
        crisis = self.regimen_bloque_ == 1
        fila["crisis_en_estado_crisis_sint"] = (float(np.mean(est_sint[crisis] == k - 1))
                                                if crisis.any() else np.nan)
        self.ocupacion.append(fila)

    # ------------------------------------------------------------- prediccion
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.interno.predict(X)

    def predict_online(self, X: pd.DataFrame, refit: bool = False) -> np.ndarray:
        return self.interno.predict_online(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.interno.predict_proba(X)


def fabrica(spec, proveedor, *, ocupacion: list | None = None, cada_ocupacion: int = 0):
    """Fabrica sin argumentos para ``walk_forward``: un envoltorio nuevo por pliegue."""
    contador = {"k": 0}

    def crear() -> DetectorAumentado:
        k = contador["k"]
        contador["k"] += 1
        marca = bool(cada_ocupacion) and k % int(cada_ocupacion) == 0
        return DetectorAumentado(spec.factory(), proveedor, ocupacion, marca)

    return crear
