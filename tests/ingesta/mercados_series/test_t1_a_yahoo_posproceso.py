"""Tests del posproceso del nivel ampliado de Yahoo (``proveedores.yahoo.posproceso``), sin red.

Registro y constituyentes SINTÉTICOS: reglas de ticker por índice, clasificación de fallidos,
cobertura por índice/mercado con la descarga a medias y el estado «descarga en curso: N de M».
"""
from __future__ import annotations

import pandas as pd
import pytest

from ingesta.mercados_series import calidad
from ingesta.mercados_series.proveedores import yahoo as y


@pytest.mark.parametrize("indice,ticker,esperado", [
    ("sp500", "BRK.B", "BRK-B"), ("sp500", "AAPL", "AAPL"), ("hang_seng", "5", "0005.HK"),
    ("hang_seng", 388, "0388.HK"), ("kospi200", "002790", "002790.KS"), ("nifty50", "TCS", "TCS.NS"),
    ("smi", "NOVN", "NOVN.SW"), ("ftse100", "III", "III.L"), ("dax", "SAP.DE", "SAP.DE"),
])
def test_ticker_yahoo_constituyente(indice, ticker, esperado):
    assert y.ticker_yahoo_constituyente(indice, ticker) == esperado


@pytest.mark.parametrize("error,tipo", [
    ("RuntimeError: X: yfinance falló 3 veces: $X: possibly delisted; no price data found", "deslistado"),
    ("RuntimeError: X: sin datos ni con period='max' ni con start=1900-01-01", "vacio"),
    ("ReadTimeout: tiempo agotado", "otro"),
])
def test_clasificar_error(error, tipo):
    assert y.clasificar_error(error) == tipo


def _registro() -> pd.DataFrame:
    base = {"descargado": "2026-10-13T14:00:00+00:00", "clase": "EQUITY"}
    return pd.DataFrame([
        {**base, "ticker": "AAA", "estado": "ok", "mercado": "XNYS", "inicio": "1990-01-02",
         "fin": "2026-10-12", "sesiones": 9000},
        {**base, "ticker": "BBB", "estado": "ok", "mercado": "XNYS", "inicio": "2005-01-03",
         "fin": "2010-06-30", "sesiones": 1400},                     # ya no cotiza
        {**base, "ticker": "CCC", "estado": "error", "error": "possibly delisted; no price data found"},
        {**base, "ticker": "0005.HK", "estado": "error", "error": "timeout"},
        {**base, "ticker": "0005.HK", "estado": "ok", "mercado": "XHKG", "inicio": "2000-01-04",
         "fin": "2026-10-13", "sesiones": 6600},                     # reintento: la última fila manda
    ])


def test_cobertura_ampliado_parcial():
    indices = pd.DataFrame({"indice": ["sp500", "sp500", "sp500", "hang_seng", "sp500"],
                            "ticker": ["AAA", "BBB", "CCC", "0005.HK", "DDD"]})
    cal = pd.DataFrame({"salto": [2, 0], "volumen_cero": [1, 0], "cierre_repetido": [0, 3],
                        "dividendo_alto": [0, 0]}, index=pd.Index(["AAA", "0005.HK"], name="ticker"))
    tabla = y.cobertura_ampliado(_registro(), ["AAA", "BBB", "CCC", "0005.HK", "DDD"], indices, cal)
    assert list(tabla.columns) == list(y.COLUMNAS_COBERTURA)
    tot = tabla.set_index("clave").loc["total"]
    assert (tot["n_lista"], tot["n_intentados"], tot["n_ok"], tot["n_deslistado"]) == (5, 4, 3, 1)
    assert tot["n_sin_cotizacion_reciente"] == 1 and tot["inicio_antes_2000"] == 1
    assert tot["inc_salto"] == 2 and tot["inc_cierre_repetido"] == 3 and tot["tickers_con_salto"] == 1
    sp = tabla[(tabla["grupo"] == "indice") & (tabla["clave"] == "sp500")].iloc[0]
    assert (sp["n_lista"], sp["n_ok"], sp["inicio_mediana"]) == (4, 2, "1997-07-04")
    assert set(tabla.loc[tabla["grupo"] == "mercado", "clave"]) == {"XNYS", "XHKG"}
    assert y.estado_descarga(4, 5) == "descarga en curso: 4 de 5"
    assert y.estado_descarga(5, 5).startswith("completa")


def test_cobertura_ampliado_sin_registro():
    tabla = y.cobertura_ampliado(pd.DataFrame(), ["AAA"], pd.DataFrame(columns=["indice", "ticker"]))
    tot = tabla.iloc[0]
    assert tot["n_lista"] == 1 and tot["n_intentados"] == 0 and tot["inicio_mediana"] is None
    assert set(f"inc_{t}" for t in calidad.TIPOS) <= set(tabla.columns)
