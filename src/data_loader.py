"""
data_loader.py — Descarga de datos crudos (yfinance + FRED), SIN imputar.

Política de datos (decisión de proyecto)
----------------------------------------
- Rango: LO MÁS LARGO POSIBLE POR SERIE. Cada serie arranca en su fecha real de
  inicio; NO se rellenan NaN artificialmente.
- Se documenta la fecha de inicio efectiva de cada serie y la ventana común
  resultante (ver `coverage_report`).
- Crudos se guardan en data/raw/ (gitignored). Las features causales se calculan
  aparte en features.py y se guardan en data/processed/.

Universo de datos — SET AMPLIADO
--------------------------------
Mercado / riesgo (yfinance):
    ^GSPC  S&P 500
    ^VIX   volatilidad implícita (nivel de miedo)
    TLT    Treasuries largos
    IEF    Treasuries medios
    HYG    high yield (crédito)
    ^MOVE  volatilidad implícita de tipos (si yfinance lo sirve; si no, ver nota)
    GLD    oro (refugio)            [alternativa: GC=F futuros]
Macro / tasas (FRED, CSV público sin API key):
    T10Y3M     pendiente curva 10Y-3M (predictor de recesión, Estrella-Mishkin)
    DTWEXBGS   índice dólar broad (DXY)  [alternativa yfinance: DX-Y.NYB]
    BAMLH0A0HYM2  OAS high yield (spread de crédito real, opcional/ampliable)

Notas de disponibilidad (se confirman en FASE 1):
- ^MOVE no siempre está en yfinance; si falla, alternativa documentada: usar el
  proxy de vol de tipos desde TLT/IEF o la serie FRED equivalente.
- El set es AMPLIABLE: si el estado del arte (FASE 2) lo pide, se añaden series
  de correlación, breadth, liquidez, etc. Aquí solo se centraliza la descarga.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Tickers de Yahoo Finance y series de FRED del set ampliado.
YF_TICKERS: dict[str, str] = {
    "SP500": "^GSPC",
    "VIX": "^VIX",
    "TLT": "TLT",
    "IEF": "IEF",
    "HYG": "HYG",
    "MOVE": "^MOVE",
    "GOLD": "GLD",
}

FRED_SERIES: dict[str, str] = {
    "YIELD_10Y_3M": "T10Y3M",
    "DXY": "DTWEXBGS",
    "HY_OAS": "BAMLH0A0HYM2",
}


def download_yfinance(
    tickers: dict[str, str] = None, start: str = "1990-01-01", save: bool = True
) -> pd.DataFrame:
    """Descarga precios/niveles diarios de yfinance, sin imputar.

    Devuelve un DataFrame de precios de cierre ajustado (o nivel, para índices),
    una columna por activo, indexado por fecha. Cada columna empieza en su fecha
    real; las fechas previas quedan como NaN (no se rellenan).
    """
    raise NotImplementedError


def download_fred(
    series: dict[str, str] = None, start: str = "1990-01-01", save: bool = True
) -> pd.DataFrame:
    """Descarga series de FRED vía el endpoint CSV público (sin API key), sin imputar.

    Usa https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>. Devuelve un
    DataFrame diario, una columna por serie. FRED ya marca festivos/cierres como
    huecos; no se rellenan.
    """
    raise NotImplementedError


def load_raw(save: bool = True) -> pd.DataFrame:
    """Descarga TODO el set ampliado (yfinance + FRED) y lo une por fecha.

    No alinea a ventana común ni imputa: une por unión de índices. La selección
    de la ventana común se hace explícita y documentada en `coverage_report`.
    Guarda el crudo en data/raw/raw_panel.parquet.
    """
    raise NotImplementedError


def coverage_report(raw: pd.DataFrame) -> pd.DataFrame:
    """Tabla de cobertura por serie: fecha de inicio efectiva, fin, nº de obs y
    % de NaN. Incluye la ventana común (intersección) resultante.

    Es el artefacto que documenta la decisión de 'no imputar' para el EDA y la
    memoria (docs/memory/01_data_and_eda.md).
    """
    raise NotImplementedError
