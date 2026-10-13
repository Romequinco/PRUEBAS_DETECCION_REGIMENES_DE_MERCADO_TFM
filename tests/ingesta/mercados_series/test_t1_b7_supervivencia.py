"""Tests del bloque 7 (supervivencia del S&P 500): sin red, con tramos y vistas de Yahoo sintéticos."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from ingesta.mercados_series import supervivencia as sv


def _tramos() -> pd.DataFrame:
    filas = [
        ("AAA", "1996-01-02", "2000-03-01"), ("BBB", "1996-01-02", "2000-07-03"), ("CCC", "1996-01-02", "2000-09-01"),
        ("DDD", "1998-01-02", "2002-01-02"), ("EEE", "1996-01-02", None), ("FFF", "2001-01-02", None),
        ("GG.B", "1996-01-02", None),
    ]
    df = pd.DataFrame(filas, columns=["ticker", "alta", "baja"])
    df["alta"] = pd.to_datetime(df["alta"])
    df["baja"] = pd.to_datetime(df["baja"])
    df["indice"] = "sp500"
    df["fecha_publicacion"] = df["alta"]
    df["fecha_publicacion_baja"] = df["baja"]
    df["nota"] = None
    return df


def _vista(inicio: str, fin: str, crec: float = 0.0005, div_cada: int = 0) -> pd.DataFrame:
    """Vista con el formato de ``Ticker.history(auto_adjust=False, actions=True)`` (índice con huso)."""
    idx = pd.bdate_range(inicio, fin, tz="America/New_York")
    close = 50.0 * (1 + crec) ** np.arange(len(idx))
    div = np.zeros(len(idx))
    if div_cada:
        div[div_cada::div_cada] = 0.25
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Adj Close": close,
                         "Volume": 1e6, "Dividends": div, "Stock Splits": 0.0}, index=idx)


def test_ticker_yahoo():
    assert sv.ticker_yahoo("BRK.B") == "BRK-B"
    assert sv.ticker_yahoo(" GG.B ") == "GG-B"


def test_muestra_estratificada_por_anio():
    t = _tramos()
    m = sv.muestra_salidos(t, por_anio=2, semilla=1)
    assert set(m["anio_salida"]) == {2000, 2002}
    assert (m.groupby("anio_salida").size() <= 2).all()
    assert len(m[m["anio_salida"] == 2002]) == 1          # solo hay uno ese año
    assert (m["grupo"] == "salido").all()
    pd.testing.assert_frame_equal(m, sv.muestra_salidos(t, por_anio=2, semilla=1))   # reproducible
    a = sv.muestra_actuales(t, n=10)
    assert set(a["ticker"]) == {"EEE", "FFF", "GG.B"} and (a["grupo"] == "actual").all()


def test_rentabilidad_diaria_con_dividendo():
    v = _vista("2020-01-01", "2020-01-10", crec=0.0)
    v.iloc[3, v.columns.get_loc("Dividends")] = 1.0
    r = sv.rentabilidad_diaria(v)
    assert r["sesion"].dt.tz is None
    assert np.isnan(r["rentabilidad"].iloc[0])
    assert r["rentabilidad"].iloc[3] == pytest.approx(1.0 / 50.0)
    assert r["rentabilidad"].iloc[2] == pytest.approx(0.0)
    assert sv.rentabilidad_diaria(pd.DataFrame()).empty


def test_cobertura_clases():
    esp = pd.bdate_range("2000-01-03", "2000-12-29")
    completa = sv.cobertura_tramo(esp, "2000-01-03", "2001-01-01", esperadas=esp)
    assert completa["clase"] == "completa" and completa["fraccion"] == 1.0
    parcial = sv.cobertura_tramo(esp[len(esp) // 2:], "2000-01-03", "2001-01-01", esperadas=esp)
    assert parcial["clase"] == "parcial" and 0.4 < parcial["fraccion"] < 0.6
    fuera = sv.cobertura_tramo(pd.bdate_range("2010-01-01", "2010-02-01"), "2000-01-03", "2001-01-01", esperadas=esp)
    assert fuera["clase"] == "fuera_de_tramo" and fuera["sesiones_con_precio"] == 0
    nada = sv.cobertura_tramo(None, "2000-01-03", "2001-01-01", esperadas=esp)
    assert nada["clase"] == "sin_datos" and nada["primera_yahoo"] is None


def test_sesiones_nyse_excluye_festivos():
    pytest.importorskip("exchange_calendars")
    s = sv.sesiones_nyse("2024-12-23", "2024-12-27")
    assert pd.Timestamp("2024-12-25") not in s and len(s) == 4


def test_descarga_reanudable_y_tabla(tmp_path):
    vistas = {"AAA": _vista("1990-01-01", "2026-01-01"), "BBB": _vista("1999-01-01", "2000-03-01"),
              "CCC": pd.DataFrame(), "DDD": _vista("2010-01-01", "2026-01-01"),
              "EEE": _vista("1990-01-01", "2026-10-09"), "FFF": _vista("2001-01-02", "2026-10-09"),
              "GG-B": _vista("1990-01-01", "2026-10-09")}
    pedidos: list[str] = []

    def obtener(sym):
        pedidos.append(sym)
        if sym == "XXX":
            raise ValueError("fallo de red simulado")
        return vistas[sym]

    syms = [*vistas, "XXX"]
    reg = sv.descargar_muestra(syms, raiz=tmp_path, ritmo=0, obtener=obtener)
    estados = dict(zip(reg["ticker_yahoo"], reg["estado"], strict=True))
    assert estados["CCC"] == "sin_datos" and estados["XXX"] == "error" and estados["AAA"] == "ok"
    pedidos.clear()
    sv.descargar_muestra(syms, raiz=tmp_path, ritmo=0, obtener=obtener)
    assert pedidos == ["XXX"]                             # reanuda: solo reintenta el error

    t = _tramos()
    muestra = pd.concat([sv.muestra_salidos(t, 5), sv.muestra_actuales(t, 5)], ignore_index=True)
    cob = sv.tabla_cobertura(muestra, raiz=tmp_path, hoy="2026-10-09")
    assert list(cob.columns) == list(sv.COLUMNAS_COBERTURA)
    clases = dict(zip(cob["ticker"], cob["clase"], strict=True))
    assert clases["AAA"] == "completa" and clases["CCC"] == "sin_datos" and clases["DDD"] == "fuera_de_tramo"
    assert clases["BBB"] == "parcial"
    assert "reutilizado" in cob.set_index("ticker").loc["DDD", "nota"]
    assert not {"close", "rentabilidad", "Close"} & set(cob.columns)   # solo metadatos
    res = sv.resumen_por_anio(cob)
    assert set(res["anio_salida"]) == {2000, 2002}
    assert res.loc[res["anio_salida"] == 2000, "n"].item() == 3


def test_sesgo_supervivientes_ganan_a_pit(tmp_path):
    t = _tramos()
    # salidos: rentabilidad negativa en su tramo; actuales: positiva
    vistas = {"AAA": _vista("1995-01-02", "2000-02-29", crec=-0.002), "BBB": _vista("1995-01-02", "2000-07-01", crec=-0.002),
              "CCC": _vista("1995-01-02", "2000-08-31", crec=-0.002), "DDD": _vista("1998-01-02", "2001-12-31", crec=-0.002),
              "EEE": _vista("1995-01-02", "2003-12-31", crec=0.001), "FFF": _vista("2001-01-02", "2003-12-31", crec=0.001),
              "GG-B": _vista("1995-01-02", "2003-12-31", crec=0.001)}
    muestra = pd.concat([sv.muestra_salidos(t, 5), sv.muestra_actuales(t, 5)], ignore_index=True)
    sesgo = sv.estimar_sesgo(t, muestra, lambda s: sv.rentabilidad_diaria(vistas[s]))
    assert {"pit", "supervivientes", "diferencia", "n_salidos_pit", "pit_solo_actuales"} <= set(sesgo.columns)
    m = sesgo.set_index("mes")
    feb98 = m.loc[pd.Timestamp("1998-02-28")]
    assert feb98["n_salidos_pit"] == 4 and feb98["n_actuales_pit"] == 2      # FFF entra en 2001
    assert feb98["pit"] < feb98["supervivientes"]
    # PIT pondera: 4/6 de salidos (negativos) y 2/6 de actuales
    n = len(pd.bdate_range("1998-02-01", "1998-02-28"))
    m_sal, m_act = 0.998 ** n - 1, 1.001 ** n - 1
    assert feb98["pit"] == pytest.approx((4 * m_sal + 2 * m_act) / 6)
    assert feb98["supervivientes"] == pytest.approx(m_act)
    res = sv.resumen_sesgo(sesgo)
    assert res["sesgo_anual_pct"] > 0 and res["meses"] > 0
    assert m.loc[pd.Timestamp("2003-06-30"), "pit_solo_actuales"]           # ya no queda ningún salido vivo


def test_mediciones_y_fragmento(tmp_path):
    t = _tramos()
    vistas = {s: _vista("1995-01-02", "2026-10-09") for s in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GG-B")}
    muestra = pd.concat([sv.muestra_salidos(t, 5), sv.muestra_actuales(t, 5)], ignore_index=True)
    cob = sv.tabla_cobertura(muestra, hoy="2026-10-09", lector=lambda s: sv.rentabilidad_diaria(vistas[s]))
    sesgo = sv.estimar_sesgo(t, muestra, lambda s: sv.rentabilidad_diaria(vistas[s]))
    reg = pd.DataFrame({"ticker_yahoo": ["AAA"], "estado": ["ok"], "segundos": [1.0]})
    med = sv.mediciones(cob, sesgo, reg)
    assert med["salidos_muestra"] == 4 and med["salidos_por_clase"]["completa"] == 4
    assert "limitacion" in med["sesgo"]


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("INGESTA_LIVE"), reason="INGESTA_LIVE=1 para pruebas con red")
def test_live_yahoo_salido_y_actual(tmp_path):
    pytest.importorskip("yfinance")
    reg = sv.descargar_muestra(["AAPL", "LEHMQ_NO_EXISTE"], raiz=tmp_path, ritmo=0.5)
    estados = dict(zip(reg["ticker_yahoo"], reg["estado"], strict=True))
    assert estados["AAPL"] == "ok"
    assert estados["LEHMQ_NO_EXISTE"] in ("sin_datos", "error")
