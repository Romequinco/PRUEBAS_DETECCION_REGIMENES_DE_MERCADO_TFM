"""Interfaz abstracta de los generadores sinteticos (sin implementacion)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Generador(ABC):
    """Generador de trayectorias sinteticas condicionadas a regimen.

    Contrato:

    - ``fit(train)`` ajusta el generador SOLO con el tramo de entrenamiento
      (causalidad: nunca ve datos posteriores al corte del walk-forward).
    - ``sample(n_paths, length, regimes)`` devuelve trayectorias nuevas.
    - ``name`` identificador estable (clave del registro y de las rutas de salida).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificador corto y estable del generador (p. ej. ``"msvar"``)."""
        raise NotImplementedError

    @abstractmethod
    def fit(self, train: pd.DataFrame, regimes: pd.Series | None = None) -> "Generador":
        """Ajusta el generador.

        Parameters
        ----------
        train:
            Panel de features (indice DatetimeIndex, columnas = features) del
            tramo de entrenamiento.
        regimes:
            Etiquetas de regimen alineadas con ``train`` (opcional; necesarias
            para generadores condicionados).

        Returns
        -------
        Generador
            ``self`` ajustado.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        n_paths: int,
        length: int,
        regimes: np.ndarray | pd.Series | None = None,
        *,
        random_state: int | None = None,
    ) -> list[pd.DataFrame]:
        """Genera ``n_paths`` trayectorias de ``length`` pasos.

        Parameters
        ----------
        n_paths:
            Numero de trayectorias independientes.
        length:
            Longitud (sesiones) de cada trayectoria.
        regimes:
            Secuencia de regimenes impuesta (longitud ``length``); ``None`` deja
            que el generador simule tambien la cadena de regimenes.
        random_state:
            Semilla para reproducibilidad.

        Returns
        -------
        list[pandas.DataFrame]
            Una trayectoria por elemento, mismas columnas que ``train`` mas una
            columna ``regime`` con la verdad-terreno.
        """
        raise NotImplementedError
