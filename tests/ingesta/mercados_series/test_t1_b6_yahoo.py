"""Tests del bloque 6 (Yahoo): ``proveedores.yahoo`` y ``calidad``.

Sin red: una vista SINTÉTICA con el formato real de yfinance (``auto_adjust=False, actions=True``:
índice con el huso de la bolsa, ``Open/High/Low/Close/Adj Close/Volume/Dividends/Stock Splits``)
recortada alrededor del split 4:1 de AAPL de 2020-08-31 y del dividendo de 2020-08-07, generada con
``sinteticos.vista_yahoo_hoy`` a partir de precios as-traded conocidos. Comprueban que se deshace el
split (cierre 499,23 y dividendo 0,82), que ``ajustar_a_fecha(hoy)`` reproduce ``Adj Close``, el
descarte de la vela incompleta, los dividendos en GBP de un valor en GBp, los eventos en días sin
precio, el mapa de mercados, la verificación del inicio y los controles de calidad. El
``@pytest.mark.live`` contrasta AAPL y NVDA reales (``INGESTA_LIVE=1``).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from ingesta.mercados_series import calidad, sinteticos, timegate, visibilidad
from ingesta.mercados_series.contrato import COLUMNAS_PRECIOS, validar_precios
from ingesta.mercados_series.proveedores import ProveedorPrecios
from ingesta.mercados_series.proveedores import yahoo as y

META_AAPL = {"currency": "USD", "exchangeName": "NMS", "instrumentType": "EQUITY",
             "exchangeTimezoneName": "America/New_York", "firstTradeDate": 345479400}
DESCARGA = pd.Timestamp("2026-10-13 12:00", tz="UTC")


def _as_traded() -> pd.DataFrame:
    """As-traded sintético tipo AAPL: 2020-08-03 a 2020-09-04 con split 4:1 el 31-08 y dividendo 0,82."""
    sesiones = pd.bdate_range("2020-08-03", "2020-09-04")
    n = len(sesiones)
    close = np.linspace(435.0, 499.23, n)
    i_split = list(sesiones).index(pd.Timestamp("2020-08-31"))
    close[i_split:] = close[i_split:] / 4.0 + 5.0
    close[list(sesiones).index(pd.Timestamp("2020-08-28"))] = 499.23
    p = pd.DataFrame({"ticker": "AAPL", "sesion": sesiones, "open": close * 0.99, "high": close * 1.01,
                      "low": close * 0.98, "close": close, "volume": 3.0e7, "dividendo": 0.0, "split": 1.0})
    p.loc[p["sesion"] == "2020-08-07", "dividendo"] = 0.82
    p.loc[i_split, "split"] = 4.0
    return p


def _vista(p: pd.DataFrame, huso: str = "America/New_York") -> pd.DataFrame:
    v = sinteticos.vista_yahoo_hoy(p)
    v.index = pd.DatetimeIndex(v.index).tz_localize(huso)
    return v[["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]]


def _contrato(v, meta=META_AAPL, ticker="AAPL", descargado=DESCARGA):
    tabla, info = y.a_contrato(v, meta, ticker, descargado)
    return validar_precios(tabla), info


def test_deshace_split_y_dividendo():
    p = _as_traded()
    tabla, info = _contrato(_vista(p))
    assert list(tabla.columns) == list(COLUMNAS_PRECIOS)
    t = tabla.set_index("sesion")
    assert t.loc["2020-08-28", "close"] == pytest.approx(499.23, rel=1e-9)
    assert t.loc["2020-08-07", "dividendo"] == pytest.approx(0.82, rel=1e-9)
    assert t.loc["2020-08-31", "split"] == 4.0
    np.testing.assert_allclose(tabla["volume"], 3.0e7)
    assert info["mercado"] == "XNAS" and info["splits"] == 1 and info["dividendos"] == 1
    assert not any("adj" in c.lower() for c in tabla.columns)


def test_ajustar_a_fecha_hoy_reproduce_adj_close():
    p = _as_traded()
    v = _vista(p)
    tabla, _ = _contrato(v)
    aj = timegate.ajustar_a_fecha(tabla, "2026-10-13")
    np.testing.assert_allclose(aj["close_aj_total"].to_numpy(), v["Adj Close"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(aj["close_aj_splits"].to_numpy(), v["Close"].to_numpy(), rtol=1e-12)


def test_visibilidad_y_vela_incompleta():
    p = _as_traded()
    # Descarga a las 15:00 NY del 2020-09-04: la vela de ese día está en curso y se descarta.
    tabla, info = _contrato(_vista(p), descargado=pd.Timestamp("2020-09-04 19:00", tz="UTC"))
    assert tabla["sesion"].max() == pd.Timestamp("2020-09-03")
    assert info["velas_incompletas"] == 1
    # XNAS 16:00 + 30 min = 16:30 NY: visible en la decisión del mismo día.
    assert (tabla["fecha_publicacion"] == tabla["sesion"]).all()


def test_dividendo_en_unidad_de_valor_en_peniques():
    p = _as_traded().assign(ticker="VOD.L")
    p.loc[p["sesion"] == "2020-08-07", "dividendo"] = 0.045   # GBP sobre un cierre en peniques
    meta = {"currency": "GBp", "exchangeName": "LSE", "instrumentType": "EQUITY",
            "exchangeTimezoneName": "Europe/London"}
    tabla, info = _contrato(_vista(p, "Europe/London"), meta=meta, ticker="VOD.L")
    assert tabla.set_index("sesion").loc["2020-08-07", "dividendo"] == pytest.approx(4.5)
    assert tabla["divisa"].iloc[0] == "GBp" and info["mercado"] == "XLON"
    assert any("×100" in n for n in info["notas"])


def test_evento_en_dia_sin_precio_pasa_a_la_sesion_siguiente():
    v = _vista(_as_traded())
    extra = pd.DataFrame({c: [np.nan] for c in v.columns}, index=pd.DatetimeIndex(
        [pd.Timestamp("2020-08-08", tz="America/New_York")]))
    extra["Dividends"], extra["Stock Splits"] = 0.1, 0.0
    v = pd.concat([v, extra]).sort_index()
    tabla, info = _contrato(v)
    assert tabla.set_index("sesion").loc["2020-08-10", "dividendo"] == pytest.approx(0.1 * 4)
    assert pd.Timestamp("2020-08-08") not in set(tabla["sesion"])
    assert any("movidos" in n for n in info["notas"])


def test_contra_split_y_split_fraccionario_anotados():
    p = _as_traded()
    p.loc[p["sesion"] == "2020-08-31", "split"] = 0.875
    p.loc[p["sesion"] == "2020-08-14", "split"] = 1.05
    _, info = _contrato(_vista(p))
    assert any("contra-splits" in n for n in info["notas"])
    assert any("fraccionarios" in n for n in info["notas"])


@pytest.mark.parametrize("ticker,meta,esperado", [
    ("SPY", {"exchangeName": "PCX"}, "ARCX"),
    ("^VIX", {"exchangeName": "CBO"}, "CBOE_IDX"),
    ("ES=F", {"exchangeName": "CME"}, "CME"),
    ("EURUSD=X", {"exchangeName": "CCY"}, "FX_YAHOO"),
    ("VBMFX", {"exchangeName": "NAS", "instrumentType": "MUTUALFUND"}, "NAV_US"),
    ("^STOXX50E", {"exchangeName": "ZRH"}, "STOXX_IDX"),
    ("XYZ.XX", {"exchangeName": "???", "exchangeTimezoneName": "Asia/Tokyo"}, "XTKS"),
])
def test_mercado_de(ticker, meta, esperado):
    assert y.mercado_de(ticker, meta) == esperado
    assert esperado in visibilidad.cargar_mercados()


def test_mercado_desconocido_lanza():
    with pytest.raises(KeyError):
        y.mercado_de("X", {"exchangeName": "???", "exchangeTimezoneName": "Mars/Base"})


def test_inicio_recortado_reintenta_con_start(monkeypatch):
    v = _vista(_as_traded())
    llamadas = []

    def falso(ticker, **kw):
        llamadas.append(kw)
        meta = dict(META_AAPL, firstTradeDate=int(pd.Timestamp("2020-08-03").timestamp()))
        return (v.iloc[10:] if "period" in kw else v), meta

    monkeypatch.setattr(y, "_history", falso)
    monkeypatch.setattr(y, "HOLGURA_INICIO", pd.Timedelta(days=5))
    vista, _, avisos = y.descargar_vista("AAPL")
    assert llamadas[1] == {"start": "2020-08-03"}
    assert vista.index.min().tz_localize(None) == pd.Timestamp("2020-08-03")
    assert avisos and "reintento" in avisos[0]


def test_inicio_que_sigue_recortado_se_avisa(monkeypatch):
    v = _vista(_as_traded())
    meta = dict(META_AAPL, firstTradeDate=int(pd.Timestamp("2019-01-02").timestamp()))
    monkeypatch.setattr(y, "_history", lambda ticker, **kw: (v, meta))
    _, _, avisos = y.descargar_vista("AAPL")
    assert any("recortada" in a for a in avisos)


def test_calidad():
    p = _as_traded()
    p.loc[p["sesion"] == "2020-08-12", "close"] *= 2.0          # salto sin split
    p.loc[p["sesion"] == "2020-08-13", "close"] = p.loc[p["sesion"] == "2020-08-12", "close"].iloc[0]
    p.loc[p["sesion"] == "2020-08-17", "volume"] = 0.0
    p.loc[p["sesion"] == "2020-08-19", "dividendo"] = 200.0
    inc = calidad.incidencias_precios(p)
    tipos = set(zip(inc["sesion"].dt.strftime("%Y-%m-%d"), inc["tipo"]))
    assert ("2020-08-12", "salto") in tipos and ("2020-08-13", "cierre_repetido") in tipos
    assert ("2020-08-17", "volumen_cero") in tipos and ("2020-08-19", "dividendo_alto") in tipos
    assert ("2020-08-31", "salto") not in tipos                 # el split no cuenta como salto
    assert calidad.incidencias_precios(p, {"AAPL": "INDEX"}).query("tipo == 'volumen_cero'").empty
    res = calidad.resumen_calidad(p)
    assert res.loc["AAPL", "sesiones"] == len(p)


def test_interfaz_y_nucleo():
    assert isinstance(y.ProveedorYahoo(), ProveedorPrecios)
    assert len(y.ETF_PROPUESTA) == 42 and len(set(y.NUCLEO)) == len(y.NUCLEO)


@pytest.mark.live
def test_live_aapl_nvda_y_adj_close():
    if os.environ.get("INGESTA_LIVE") != "1":
        pytest.skip("INGESTA_LIVE=1 para tocar la red")
    aapl, _ = y.descargar("AAPL")
    a = aapl.set_index("sesion")
    assert a.loc["2020-08-28", "close"] == pytest.approx(499.23, abs=0.01)
    assert a.loc["2020-08-07", "dividendo"] == pytest.approx(0.82, abs=1e-6)
    nvda, _ = y.descargar("NVDA")
    assert nvda.set_index("sesion").loc["2024-06-07", "close"] == pytest.approx(1208.88, abs=0.01)
    # Adj Close de Yahoo leído al vuelo (nunca se guarda) frente al ajuste causal con fecha = hoy.
    vista, _, _ = y.descargar_vista("AAPL")
    adj = vista["Adj Close"].set_axis(pd.DatetimeIndex(vista.index).tz_localize(None).normalize())
    aj = timegate.ajustar_a_fecha(aapl, pd.Timestamp.now(tz="America/New_York").date())
    comun = aj.set_index("sesion")["close_aj_total"].loc[lambda s: s.index >= "2000-01-01"]
    rel = (comun / adj.reindex(comun.index) - 1.0).abs().dropna()
    assert len(rel) > 6000
    assert rel.median() < 1e-4 and rel.max() < 5e-3
