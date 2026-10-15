"""Tests de ``ingesta.mercados_series.fuentes.constituyentes`` (bloque 5).

Sin red: CSV SINTÉTICO con el formato de ``fja05680/sp500`` (``date,tickers`` con la lista completa
entrecomillada) y tablas sintéticas con la forma de las de Wikipedia. ``@pytest.mark.live``: la
descarga real del CSV (``INGESTA_LIVE=1``).
"""
from __future__ import annotations

import pandas as pd
import pytest

from ingesta.mercados_series.fuentes import constituyentes as C

CSV = """date,tickers
1996-01-02,"AAA,BBB,CCC"
1997-05-01,"AAA,BBB,DDD"
2001-03-15,"AAA,DDD,EEE"
2005-07-01,"AAA,BBB,DDD,EEE"
2005-07-01,"AAA,BBB,DDD,EEE"
"""


@pytest.fixture()
def tramos():
    return C.tramos_sp500(C.leer_sp500_historico(CSV))


def test_t1_b5_cons_lectura(tramos):
    h = C.leer_sp500_historico(CSV.encode())
    assert len(h) == 4 and h["miembros"].iloc[0] == frozenset({"AAA", "BBB", "CCC"})
    assert list(tramos.columns) == list(C.COLUMNAS_SP500)


def test_t1_b5_cons_tramos(tramos):
    t = tramos.set_index(["ticker", "alta"])
    assert pd.isna(t.loc[("AAA", pd.Timestamp("1996-01-02")), "baja"])
    assert t.loc[("CCC", pd.Timestamp("1996-01-02")), "baja"] == pd.Timestamp("1997-05-01")
    bbb = tramos[tramos["ticker"] == "BBB"]
    assert len(bbb) == 2 and bbb["nota"].str.contains("reutilizado").all()
    assert (tramos["fecha_publicacion"] == tramos["alta"]).all()
    assert tramos.loc[tramos["alta"] == pd.Timestamp("1996-01-02"), "nota"].str.contains("truncada").all()
    assert pd.isna(tramos.loc[tramos["ticker"] == "EEE", "nota"]).all()


def test_t1_b5_cons_miembros_en(tramos):
    assert C.miembros_en(tramos, "1996-06-01") == ["AAA", "BBB", "CCC"]
    assert C.miembros_en(tramos, "1997-05-01") == ["AAA", "BBB", "DDD"]   # el día del cambio ya vale
    assert C.miembros_en(tramos, "2003-01-01") == ["AAA", "DDD", "EEE"]
    assert C.miembros_en(tramos, "1995-01-01") == []


def test_t1_b5_cons_cabecera_incorrecta():
    with pytest.raises(ValueError, match="cabecera"):
        C.leer_sp500_historico("fecha,lista\n2000-01-01,A\n")


def test_t1_b5_cons_elegir_tabla_y_normalizar():
    ruido = pd.DataFrame({"Year": range(30), "Closing level": range(30)})
    buena = pd.DataFrame({"Company": [f"Empresa {i}" for i in range(39)],
                          "Ticker": [f"T{i}[1]" for i in range(39)], "Sector": "x"})
    pequena = pd.DataFrame({"Symbol": ["A", "B"]})
    t = C.elegir_tabla([ruido, pequena, buena], 40)
    assert t is buena
    df = C.normalizar_foto(t, "dax", pd.Timestamp("2026-10-13"), "https://en.wikipedia.org/wiki/DAX")
    assert list(df.columns) == list(C.COLUMNAS_FOTO)
    assert df["ticker"].iloc[0] == "T0" and df["foto_actual"].all() and df["nombre"].iloc[0] == "Empresa 0"
    assert (df["fecha_publicacion"] == pd.Timestamp("2026-10-13")).all()
    assert df["nota"].str.contains("supervivencia").all()
    assert C.elegir_tabla([ruido], 40) is None


def test_t1_b5_cons_limpiar_ticker():
    s = pd.Series(["SEHK:\xa05", "NSE: RELIANCE", "ADS.DE[2]", " 005930 "])
    assert C.limpiar_ticker(s).tolist() == ["5", "RELIANCE", "ADS.DE", "005930"]


def test_t1_b5_cons_indices_configurados():
    assert len(C.INDICES) == 12 and C.INDICES["ftse100"][1] == 100


@pytest.mark.live
def test_t1_b5_cons_live_sp500(tmp_path):
    df, med = C.descargar_sp500(raiz=tmp_path)
    assert med["inicio"] == "1996-01-02" and 480 <= med["miembros_ultimo"] <= 510
    assert len(C.miembros_en(df, "2008-09-15")) >= 495
    assert (tmp_path / "constituyentes" / "sp500.parquet").exists()
