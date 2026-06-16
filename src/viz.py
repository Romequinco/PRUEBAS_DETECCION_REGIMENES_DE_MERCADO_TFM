"""
viz.py — Visualizaciones estándar reutilizables por todos los detectores.

Para que la comparación sea honesta, todos los detectores se visualizan igual.
Estas funciones se llaman desde cada notebook de detector y desde la síntesis
comparativa (FASE 4).
"""

from __future__ import annotations

import pandas as pd


def plot_price_by_regime(prices: pd.Series, states: pd.Series, crisis_state: int, ax=None):
    """S&P 500 coloreado por el estado de régimen (sombreado por episodio).

    Visualización canónica: precio con bandas de color según el régimen
    canónico, resaltando los tramos de crisis.
    """
    raise NotImplementedError


def plot_regime_timeline(states: pd.Series, ax=None):
    """Timeline horizontal de regímenes a lo largo del tiempo (heatmap 1xN)."""
    raise NotImplementedError


def plot_crisis_probability(p_crisis: pd.Series, crisis_windows: dict = None, ax=None):
    """Probabilidad continua de crisis con las ventanas de crisis conocidas
    sombreadas, para inspección visual de cobertura y lead/lag.
    """
    raise NotImplementedError


def plot_duration_histogram(states: pd.Series, ax=None):
    """Histograma de duraciones de episodios por régimen (detecta flickering)."""
    raise NotImplementedError


def plot_transition_matrix(matrix, labels=None, ax=None):
    """Heatmap de la matriz de transición (para detectores con dinámica explícita)."""
    raise NotImplementedError


def plot_metric_comparison(results_df: pd.DataFrame, metric: str, ax=None):
    """Barras comparando una métrica entre todos los detectores (FASE 4)."""
    raise NotImplementedError
