"""Validacion de trayectorias sinteticas (firmas documentadas, sin logica)."""

from __future__ import annotations

import pandas as pd


def fidelidad_marginal(real: pd.DataFrame, sintetico: list[pd.DataFrame]) -> pd.DataFrame:
    """Compara distribuciones marginales por feature (p. ej. KS, momentos, colas).

    Devuelve una fila por feature con los estadisticos de discrepancia.
    """
    raise NotImplementedError


def fidelidad_dependencia(real: pd.DataFrame, sintetico: list[pd.DataFrame]) -> pd.DataFrame:
    """Compara dependencia temporal y cruzada (ACF, ACF de |r|, correlaciones)."""
    raise NotImplementedError


def fidelidad_regimenes(real_regimes: pd.Series, sintetico: list[pd.DataFrame]) -> pd.DataFrame:
    """Compara duraciones de regimen y matriz de transicion real vs sintetica."""
    raise NotImplementedError


def utilidad_tstr(real: pd.DataFrame, sintetico: list[pd.DataFrame], detector_id: str) -> pd.DataFrame:
    """Train-on-Synthetic / Test-on-Real: ajusta el detector en sintetico y evalua en real."""
    raise NotImplementedError


def memorizacion(real: pd.DataFrame, sintetico: list[pd.DataFrame]) -> pd.DataFrame:
    """Distancia al vecino real mas cercano (detecta copia del entrenamiento)."""
    raise NotImplementedError
