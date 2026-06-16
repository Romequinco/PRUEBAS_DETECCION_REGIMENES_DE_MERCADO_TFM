"""
features.py — Features CAUSALES para los detectores de régimen.

Hallazgo clave de la tarea previa: estandarizar con media/desviación de TODA la
muestra mete información del futuro (look-ahead sutil en el z-score). Aquí TODA
estandarización es causal: en t solo se usan estadísticos calculados con datos
<= t (expanding o rolling). Ningún estadístico de muestra completa.

Features previstas (set base, AMPLIABLE según estado del arte en FASE 2)
------------------------------------------------------------------------
A partir del set ampliado de data_loader:
  - Retorno log del S&P 500 (z-score causal)
  - Volatilidad realizada 21d del S&P 500 (z-score causal)
  - Nivel de VIX (z-score causal) y ΔVIX (velocidad del deterioro)
  - Nivel de MOVE (z-score causal), si disponible
  - Retornos z de TLT, IEF, HYG
  - Spread de crédito proxy = ret(HYG) - ret(IEF), y/o HY_OAS real (z-score)
  - Pendiente de curva 10Y-3M (nivel, z-score causal)
  - ΔDXY (z-score causal), retorno del oro
  - Correlación rolling RV/Bonos (cambia de signo entre regímenes, Gulko 2002)
  - Drawdown corriente del S&P 500, momentum 12M-1M
La lista final se cierra tras el EDA y el estado del arte.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def causal_zscore(
    s: pd.Series, method: str = "expanding", window: int = 252, min_periods: int = 60
) -> pd.Series:
    """Z-score CAUSAL de una serie: en t usa solo media/std de datos <= t.

    Parameters
    ----------
    method : {'expanding', 'rolling'}
        'expanding' usa toda la historia hasta t (más estable a largo plazo).
        'rolling' usa una ventana de `window` días (más adaptativo a cambios de
        nivel estructural).
    window : int
        Ventana para method='rolling'.
    min_periods : int
        Mínimo de observaciones antes de emitir un z-score (antes -> NaN, sin
        imputar). Evita z-scores degenerados al inicio de cada serie.

    Returns
    -------
    pd.Series alineada con `s`. Crucial: usa media/std DESPLAZADAS (shift) para
    no incluir el propio t en su normalización si se requiere estricta causalidad
    one-step; se documenta la convención elegida en FASE 1.
    """
    raise NotImplementedError


def log_returns(prices: pd.Series) -> pd.Series:
    """Retorno logarítmico diario."""
    raise NotImplementedError


def realized_vol(returns: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    """Volatilidad realizada rolling (anualizada ×√252 si annualize)."""
    raise NotImplementedError


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 60) -> pd.Series:
    """Correlación rolling causal entre dos series (p. ej. S&P 500 vs Treasuries)."""
    raise NotImplementedError


def drawdown(prices: pd.Series) -> pd.Series:
    """Drawdown corriente respecto al máximo histórico causal (expanding max)."""
    raise NotImplementedError


def build_features(raw: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Construye la matriz de features CAUSALES a partir del panel crudo.

    Aplica retornos, vol realizada, z-scores causales, spreads, correlaciones y
    drawdown según el set definido arriba. NO imputa; descarta filas iniciales
    sin suficientes datos vía dropna controlado y deja registro de la ventana
    efectiva resultante. Guarda en data/processed/features.parquet.

    Returns
    -------
    pd.DataFrame de features, indexado por fecha, listo para los detectores y el
    evaluador.
    """
    raise NotImplementedError
