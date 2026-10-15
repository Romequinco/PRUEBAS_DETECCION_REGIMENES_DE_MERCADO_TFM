"""Tests del bloque 8 (libre de riesgo, baselines y costes) sin red, con precios sintéticos.

Los precios sintéticos siguen el contrato de precios (as-traded, ``fecha_publicacion`` a cierre
+ 30 min: el mismo día en EE. UU.; un fondo publica el valor liquidativo en D+1). El test clave
es el de truncado: la curva de un baseline en ``t`` no cambia al añadir precios posteriores. El
``@pytest.mark.live`` descarga SPY con yfinance y compara el buy&hold con el cierre ajustado.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ingesta.mercados_series import timegate
from ingesta.mercados_series.cartera import baselines as b
from ingesta.mercados_series.cartera import costes as c
from ingesta.mercados_series.cartera import libre_riesgo as lr
from ingesta.mercados_series.contrato import COLUMNAS_PRECIOS, COLUMNAS_SERIES

RAIZ = Path(__file__).resolve().parents[3]
CONFIG = RAIZ / "src/ingesta/mercados_series/config"


# --- Fixtures sintéticas -------------------------------------------------------------------------
def precios_sinteticos(ticker: str, desde: str, hasta: str, *, semilla: int = 0, deriva: float = 0.0003,
                       vol: float = 0.01, retraso_pub: int = 0, split: tuple[str, float] | None = None,
                       dividendos: dict[str, float] | None = None, mercado: str = "XNYS",
                       divisa: str = "USD") -> pd.DataFrame:
    """Tabla del contrato de precios con OHLC aleatorio, split y dividendos as-traded."""
    rng = np.random.default_rng(semilla)
    sesiones = pd.bdate_range(desde, hasta)
    n = len(sesiones)
    cierre = 50.0 * np.exp(np.cumsum(rng.normal(deriva, vol, n)))
    apertura = cierre * np.exp(rng.normal(0, vol / 3, n))
    alto = np.maximum(apertura, cierre) * np.exp(np.abs(rng.normal(0, vol / 2, n)))
    bajo = np.minimum(apertura, cierre) * np.exp(-np.abs(rng.normal(0, vol / 2, n)))
    fsplit = np.ones(n)
    if split is not None:
        i = sesiones.searchsorted(pd.Timestamp(split[0]))
        fsplit[i] = split[1]
        escala = np.where(np.arange(n) >= i, 1.0 / split[1], 1.0)  # as-traded: precio cae tras el split
        apertura, alto, bajo, cierre = (x * escala for x in (apertura, alto, bajo, cierre))
    div = np.zeros(n)
    for f, d in (dividendos or {}).items():
        div[sesiones.searchsorted(pd.Timestamp(f))] = d
    pub = sesiones + pd.offsets.BDay(retraso_pub) if retraso_pub else sesiones
    disp = (pd.DatetimeIndex(pub).tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=30)).tz_convert("UTC")
    df = pd.DataFrame({
        "ticker": ticker, "sesion": sesiones, "mercado": mercado, "open": apertura, "high": alto, "low": bajo,
        "close": cierre, "volume": 1e6, "dividendo": div, "split": fsplit, "divisa": divisa,
        "fecha_publicacion": pd.DatetimeIndex(pub), "disponible_utc": disp, "fuente": "sintetico",
        "descargado": pd.Timestamp("2026-01-01", tz="UTC")})
    return df[list(COLUMNAS_PRECIOS)].astype(COLUMNAS_PRECIOS)


@pytest.fixture(scope="module")
def universo():
    p = {
        "SPY": precios_sinteticos("SPY", "2001-01-02", "2006-12-29", semilla=1, split=("2004-06-15", 2.0),
                                  dividendos={"2002-03-15": 0.4, "2005-03-18": 0.3}),
        "VBMFX": precios_sinteticos("VBMFX", "2001-01-02", "2006-12-29", semilla=2, vol=0.003, retraso_pub=1,
                                    mercado="NAV_US"),
        "AGG": precios_sinteticos("AGG", "2003-09-29", "2006-12-29", semilla=3, vol=0.003),
    }
    for k, tk in enumerate(("XLB", "XLE", "XLF", "XLK", "XLV")):
        p[tk] = precios_sinteticos(tk, "2001-01-02", "2006-12-29", semilla=10 + k, deriva=0.0001 * k)
    p["XLRE"] = precios_sinteticos("XLRE", "2004-11-01", "2006-12-29", semilla=30, deriva=0.003)
    return p


def _truncar(precios: dict[str, pd.DataFrame], fecha) -> dict[str, pd.DataFrame]:
    f = timegate.fecha_decision(fecha)
    return {tk: t[t["fecha_publicacion"] <= f] for tk, t in precios.items()}


ESTRATEGIAS = {
    "buy_hold": lambda: b.buy_hold("SPY"),
    "6040": b.sesenta_cuarenta,
    "uno_n": lambda: b.uno_n(["SPY", "AGG", "XLB", "XLE", "XLRE"]),
    "momentum": lambda: b.momentum_sectorial(("XLB", "XLE", "XLF", "XLK", "XLV", "XLRE")),
}


# --- Baselines -----------------------------------------------------------------------------------
@pytest.mark.parametrize("nombre", sorted(ESTRATEGIAS))
@pytest.mark.parametrize("corte", ["2004-06-15", "2005-03-17"])
def test_t1_b8_cartera_truncado_no_mira_al_futuro(universo, nombre, corte):
    """El valor en t no cambia al añadir precios posteriores (ni al alargar la simulación)."""
    modelo = c.ModeloCostes(broker=c.DEGIRO, eur_usd=1.2)
    completo = b.simular(ESTRATEGIAS[nombre](), universo, "2001-06-01", "2006-12-29", costes=modelo)
    truncado = b.simular(ESTRATEGIAS[nombre](), _truncar(universo, corte), "2001-06-01", corte, costes=modelo)
    con_futuro = b.simular(ESTRATEGIAS[nombre](), universo, "2001-06-01", corte, costes=modelo)
    assert len(truncado.curva) > 100
    pd.testing.assert_series_equal(truncado.curva, con_futuro.curva, rtol=0, atol=0)
    comun = completo.curva.loc[:pd.Timestamp(corte)]
    np.testing.assert_allclose(truncado.curva.loc[comun.index].to_numpy(), comun.to_numpy(), rtol=1e-12)
    previos = completo.rebalanceos[completo.rebalanceos["fecha"] < pd.Timestamp(corte)]
    assert [r["pesos"] for _, r in previos.iterrows()] == \
           [r["pesos"] for _, r in truncado.rebalanceos[truncado.rebalanceos["fecha"] < pd.Timestamp(corte)].iterrows()]


def test_t1_b8_cartera_buy_hold_es_retorno_total(universo):
    """Buy&hold = retorno total (split neutro, dividendos reinvertidos) visto al final."""
    r = b.simular(b.buy_hold("SPY"), universo, "2001-01-31", "2006-12-29")
    aj = timegate.ajustar_a_fecha(universo["SPY"], "2006-12-29", avisar=False).set_index("sesion")
    esperado = aj["close_aj_total"].loc["2006-12-29"] / aj["close_aj_total"].loc["2001-01-31"]
    assert r.curva.iloc[-1] / r.curva.iloc[0] == pytest.approx(esperado, rel=1e-12)
    dia = r.curva.loc["2004-06-14":"2004-06-16"].pct_change().dropna()
    assert (dia.abs() < 0.2).all()  # el split 2:1 no aparece como caída del 50 %


def test_t1_b8_cartera_6040_empalme_vbmfx_agg(universo):
    r = b.simular(b.sesenta_cuarenta(), universo, "2001-06-01", "2006-12-29")
    reb = r.rebalanceos.set_index("fecha")
    assert reb["pesos"].iloc[0] == {"SPY": 0.6, "VBMFX": 0.4}
    assert reb.loc["2003-09-30", "pesos"] == {"AGG": 0.4, "SPY": 0.6}  # primer control con AGG visible
    meses = sorted({f.month for f in reb.index[reb.index > "2003-09-30"]})
    assert meses == [12]  # después, solo el rebalanceo anual
    # VBMFX publica en D+1: en el control el valor liquidativo usado es el de la sesión anterior.
    v = timegate.ajustar_a_fecha(universo["VBMFX"], "2002-12-31", avisar=False)
    assert v["sesion"].iloc[-1] == pd.Timestamp("2002-12-30")


def test_t1_b8_cartera_momentum_12_1(universo):
    est = b.momentum_sectorial(("XLB", "XLE", "XLF", "XLK", "XLV", "XLRE"))
    r = b.simular(est, universo, "2001-06-01", "2006-12-29")
    reb = r.rebalanceos.set_index("fecha")
    assert reb.index[0] == pd.Timestamp("2002-01-31")  # 12 meses de historia
    assert all(len(p) == 3 for p in reb["pesos"])
    assert all("XLRE" not in p for p in reb.loc[:"2005-10-31", "pesos"])  # XLRE nace en 2004-11
    # Comprobación independiente de un control: ranking 12-1 con cierres ajustados a esa fecha.
    f = pd.Timestamp("2006-06-30")
    puntos = {}
    for tk in ("XLB", "XLE", "XLF", "XLK", "XLV", "XLRE"):
        aj = timegate.ajustar_a_fecha(universo[tk], f, avisar=False).set_index("fecha_publicacion")["close_aj_total"]
        puntos[tk] = aj.loc[:f - pd.DateOffset(months=1)].iloc[-1] / aj.loc[:f - pd.DateOffset(months=12)].iloc[-1]
    assert set(reb.loc[f, "pesos"]) == set(sorted(puntos, key=puntos.get, reverse=True)[:3])


def test_t1_b8_cartera_uno_n_y_costes(universo):
    est = b.uno_n(["SPY", "AGG", "XLB"])
    sin = b.simular(est, universo, "2002-01-01", "2004-12-31")
    con = b.simular(b.uno_n(["SPY", "AGG", "XLB"]), universo, "2002-01-01", "2004-12-31",
                    costes=c.ModeloCostes(broker=c.DEGIRO, eur_usd=1.2))
    assert sin.rebalanceos["pesos"].iloc[0] == pytest.approx({"SPY": 0.5, "XLB": 0.5})
    assert len(con.rebalanceos.loc[con.rebalanceos["fecha"] == "2003-09-30", "pesos"].iloc[0]) == 3
    assert (con.rebalanceos["coste"] > 0).all()
    assert con.curva.iloc[-1] < sin.curva.iloc[-1]


# --- Costes --------------------------------------------------------------------------------------
def test_t1_b8_cartera_comisiones_y_fx():
    f = pd.Timestamp("2020-01-02")
    d = c.ModeloCostes(broker=c.DEGIRO, info={"SPY": {"mercado": "ARCX", "divisa": "USD", "clase": "ETF"}},
                       eur_usd=1.10).desglose("SPY", 1000.0, f, 300.0)
    assert d["comision"] == pytest.approx(2.0 * 1.10)
    assert d["fx"] == pytest.approx(2.5)  # 0,25 % de 1.000 USD
    assert d["spread"] == pytest.approx(0.5 * 0.0005 * 1000.0)
    ib = c.IBKR.comision("US")
    assert ib.importe_usd(1000.0, 1.10, 300.0) == pytest.approx(0.90 * 1.10)       # mínimo
    assert ib.importe_usd(50.0, 1.10, 0.10) == pytest.approx(0.5)                   # tope 1 %
    sin_fx = c.ModeloCostes(info={"SAN.MC": {"mercado": "XMAD", "divisa": "EUR", "clase": "EQUITY"}})
    assert sin_fx.desglose("SAN.MC", 1000.0, f)["fx"] == 0.0


def test_t1_b8_cartera_multiplicador_punto_en_el_tiempo():
    p = precios_sinteticos("SPY", "2000-01-03", "2009-12-31", semilla=7, split=("2005-05-02", 2.0), vol=0.002)
    # Modelo de rebote bid-ask: spread 0,2 % en calma y 1 % en la crisis sintética (2008-09..2008-12).
    rng = np.random.default_rng(8)
    crisis = ((p["sesion"] >= "2008-09-01") & (p["sesion"] <= "2008-12-31")).to_numpy()
    spread = np.where(crisis, 0.01, 0.002)
    medio = p["close"].to_numpy()
    p["high"] = np.maximum(p["open"], medio) * (1 + spread / 2)
    p["low"] = np.minimum(p["open"], medio) * (1 - spread / 2)
    p["close"] = medio * (1 + rng.choice([-1, 1], len(p)) * spread / 2)
    m = c.multiplicador_crisis(p)
    assert m["m"].between(1.0, 10.0).all()
    assert (m["m"].loc[:"2000-12-29"] == 1.0).all()  # menos de 252 sesiones en calma
    assert m["m"].loc["2008-11-01":"2008-12-31"].median() > 2.0
    assert np.isnan(c.spread_abdi_ranaldo(p).loc["2005-05-02"])  # el par que cruza el split se descarta
    assert (c.spread_corwin_schultz(p).dropna() >= 0).all()
    corto = c.multiplicador_crisis(p[p["sesion"] <= "2008-10-15"])
    pd.testing.assert_frame_equal(corto, m.loc[:"2008-10-15"])


def test_t1_b8_cartera_tabla_parametros_sin_precios(tmp_path):
    t = c.tabla_parametros()
    assert {"tipo", "broker", "clave", "valor", "verificado", "fuente"} <= set(t.columns)
    assert not ({"open", "high", "low", "close", "precio"} & set(t.columns))
    assert {"degiro", "ibkr"} <= set(t["broker"])
    assert t.loc[(t["tipo"] == "fx") & (t["broker"] == "degiro"), "valor"].iloc[0] == 0.0025


# --- Libre de riesgo -----------------------------------------------------------------------------
def _serie(sid: str, refs, pubs, valores, vintages=None) -> pd.DataFrame:
    n = len(refs)
    pubs = pd.DatetimeIndex(pubs)
    df = pd.DataFrame({
        "serie": sid, "fecha_referencia": pd.DatetimeIndex(refs), "fecha_publicacion": pubs,
        "disponible_utc": pubs.tz_localize("UTC"), "vintage": vintages or ["unico"] * n,
        "valor": valores, "unidad": "%", "divisa": None, "fuente": "sintetico", "metodo_publicacion": "calendario",
        "nota": None})
    return df[list(COLUMNAS_SERIES)].astype(COLUMNAS_SERIES)


def test_t1_b8_cartera_ultimo_visible_igual_que_as_of():
    refs = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-02", "2020-01-06"]
    pubs = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-08", "2020-01-07"]
    t = _serie(lr.SERIE_DTB3, refs, pubs, [1.0, 2.0, 3.0, 2.5, 4.0],
               vintages=["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-08", "2020-01-07"])
    fechas = pd.bdate_range("2020-01-01", "2020-01-10")
    out = lr.ultimo_visible(t, fechas)
    for f in fechas:
        vis = timegate.as_of_series(t, lr.SERIE_DTB3, f)
        if vis.empty:
            assert np.isnan(out.loc[f, "tasa"])
        else:
            assert out.loc[f, "tasa"] == vis["valor"].iloc[-1]
            assert out.loc[f, "fecha_referencia"] == vis["fecha_referencia"].iloc[-1]


def test_t1_b8_cartera_dtb3_rendimiento_y_devengo():
    assert lr.rendimiento_equivalente(5.0) == pytest.approx(365 * 0.05 / (360 - 0.05 * 91))
    t = _serie(lr.SERIE_DTB3, ["2020-01-02", "2020-01-03"], ["2020-01-03", "2020-01-06"], [4.0, 8.0])
    fechas = pd.DatetimeIndex(["2020-01-03", "2020-01-06", "2020-01-07"])
    tasa = lr.tasa_dtb3(fechas, base=t)
    assert list(tasa["tasa"]) == [4.0, 8.0, 8.0]
    ind = lr.indice_devengo(tasa["rendimiento"])
    y4 = lr.rendimiento_equivalente(4.0)
    assert ind.iloc[1] == pytest.approx(1 + y4 * 3 / 365)  # el tramo usa el tipo conocido al inicio


def test_t1_b8_cartera_tasa_eur_empalme_y_ausencia():
    fechas = pd.DatetimeIndex(["2019-09-30", "2019-10-02", "2019-10-03"])
    vacia = lr.tasa_eur(fechas, base=_serie("fred:DTB3", ["2019-01-02"], ["2019-01-03"], [2.0]))
    assert vacia["tasa"].isna().all() and "no están en la base" in vacia.attrs["avisos"][0]
    base = pd.concat([
        _serie(lr.SERIE_EONIA, ["2019-09-27", "2019-10-01"], ["2019-09-30", "2019-10-02"], [-0.45, -0.46]),
        _serie(lr.SERIE_ESTR, ["2019-10-01", "2019-10-02"], ["2019-10-03", "2019-10-03"], [-0.55, -0.54])])
    out = lr.tasa_eur(fechas, base=base)
    assert list(out["serie"]) == [lr.SERIE_EONIA, lr.SERIE_EONIA, lr.SERIE_ESTR]
    assert list(out["tasa"]) == [-0.45, -0.46, -0.54]


def test_t1_b8_cartera_dependencias_integradas():
    """Las series de libre de riesgo están catalogadas y los mercados de los ETF y del fondo, declarados."""
    catalogo = yaml.safe_load((CONFIG / "catalogo_series.yaml").read_text(encoding="utf-8"))["series"]
    assert {lr.SERIE_DTB3, lr.SERIE_FRENCH_RF, lr.SERIE_EONIA, lr.SERIE_ESTR} <= set(catalogo)
    mercados = yaml.safe_load((CONFIG / "mercados.yaml").read_text(encoding="utf-8"))
    assert {"ARCX", "NAV_US"} <= set(mercados)


@pytest.mark.live
def test_t1_b8_cartera_live_buy_hold_spy_contra_yahoo():
    yf = pytest.importorskip("yfinance")
    from ingesta.mercados_series.proveedores import yahoo
    tabla, _ = yahoo.descargar("SPY")
    tabla = tabla[tabla["sesion"] >= "2023-01-01"]
    fin = tabla["fecha_publicacion"].max()
    r = b.simular(b.buy_hold("SPY"), {"SPY": tabla}, "2023-01-31", fin)
    ref = yf.Ticker("SPY").history(start="2023-01-31", end=(fin + pd.Timedelta(days=1)).date().isoformat(),
                                   auto_adjust=True, timeout=90)["Close"]
    assert r.curva.iloc[-1] / r.curva.iloc[0] == pytest.approx(ref.iloc[-1] / ref.iloc[0], rel=2e-3)
