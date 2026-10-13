"""Bloque 8 del Tema 1: tipo libre de riesgo, carteras de referencia (baselines) y costes.

- ``libre_riesgo``: DTB3 punto en el tiempo para los agentes, RF de Ken French solo ex post y
  EONIA→€STR en EUR.
- ``baselines``: motor de simulación que solo ve ``get_precios`` en cada fecha de control
  (sin mirar al futuro) y las carteras de referencia (buy&hold SPY, 60/40, 1/N, RSP, momentum
  sectorial 12-1 y 60/40 europeo).
- ``costes``: comisiones públicas (DEGIRO, IBKR), FX y spread base por clase por el
  multiplicador de crisis (Corwin-Schultz y Abdi-Ranaldo sobre OHLC as-traded).
"""
from ingesta.mercados_series.cartera import baselines, costes, libre_riesgo

__all__ = ["baselines", "costes", "libre_riesgo"]
