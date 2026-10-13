"""Regresión de las fugas corregidas tras la revisión del Tema 1.

Sin red: fixtures SINTÉTICAS con el formato real (CSV del Data Portal del BCE con
``includeHistory``, SDMX-CSV de Eurostat, ``history_metadata`` de Yahoo e instantáneas del
S&P 500). Ningún valor real de las fuentes.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ingesta.mercados_series import catalogo
from ingesta.mercados_series.contrato import validar_series
from ingesta.mercados_series.fuentes import bce, constituyentes, eurostat
from ingesta.mercados_series.proveedores import yahoo

CAB_ESTAT = "DATAFLOW,LAST UPDATE,freq,unit,geo,TIME_PERIOD,OBS_VALUE,OBS_FLAG"
CAB_HIST = "KEY,FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE,ACTION,VALID_FROM,VALID_TO"


def _estat(conj: str, geo: str, periodos: list[str]) -> pd.DataFrame:
    filas = [f"ESTAT:{conj.upper()}(1.0),02/10/26 11:00:00,M,X,{geo},{p},1.0," for p in periodos]
    return eurostat.parsear_sdmx_csv("\n".join([CAB_ESTAT, *filas]))


def _tabla(conj: str, geo: str, periodos: list[str]) -> pd.DataFrame:
    sid = eurostat.id_eurostat(conj, eurostat.CONJUNTOS[conj]["clave"], geo)
    e = eurostat.entradas_eurostat({conj: [geo]})[sid]
    return validar_series(eurostat.tabla_conjunto(sid, _estat(conj, geo, periodos), e))


def _pub(t: pd.DataFrame, ref: str) -> pd.Timestamp:
    return t.loc[t["fecha_referencia"] == pd.Timestamp(ref), "fecha_publicacion"].iloc[0]


# --- Paro y PIB de Eurostat: claves de lag propias -------------------------------------------------
def test_t1_paro_eurostat_no_usa_la_regla_de_unrate():
    assert eurostat.CONJUNTOS["une_rt_m"]["lag_clave"] == "EUROSTAT_PARO"
    assert bce.SERIES_BCE["ecb:LFSI.M.U2.S.UNEHRT.TOTAL0.15_74.T"]["previa"] == "EUROSTAT_PARO"
    t = _tabla("une_rt_m", "DE", ["2009-11", "2026-01"])
    # Publicaciones reales: noviembre de 2009 el 2010-01-08; enero de 2026 el 2026-03-04.
    assert _pub(t, "2009-11-01") >= pd.Timestamp("2010-01-08")
    assert _pub(t, "2026-01-01") >= pd.Timestamp("2026-03-04")


def test_t1_paro_reino_unido_lag_propio():
    e = eurostat.entradas_eurostat({"une_rt_m": ["UK", "FR"]})
    lags = {sid.rsplit(".", 1)[1]: v["lag_clave"] for sid, v in e.items()}
    assert lags == {"UK": "EUROSTAT_PARO_UK", "FR": "EUROSTAT_PARO"}
    t = _tabla("une_rt_m", "UK", ["2020-09"])
    assert _pub(t, "2020-09-01") > pd.Timestamp("2020-11-10")  # ONS: julio-septiembre salió el 2020-11-10


def test_t1_pib_eurostat_y_bce_no_usan_la_regla_de_gdpc1():
    assert eurostat.CONJUNTOS["namq_10_gdp"]["lag_clave"] == "EUROSTAT_PIB"
    mna = "ecb:MNA.Q.Y.I10.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.GY"
    assert bce.SERIES_BCE[mna]["previa"] == "EUROSTAT_PIB"
    lag = catalogo.cargar_lags()["EUROSTAT_PIB"]
    ref = pd.Series(pd.to_datetime(["1999-01-01", "2010-01-01"]))
    pub = catalogo.fecha_publicacion_lag(ref, lag["lag"], lag["fin_periodo"])
    assert pub.iloc[0] >= pd.Timestamp("1999-06-08")  # primera estimación hacia t+70 antes de 2003
    assert pub.iloc[1] >= pd.Timestamp("2010-05-12")  # flash t+45 entre 2003 y 2015


# --- Series y agregados posteriores al periodo -------------------------------------------------------
def test_t1_agregados_ea20_ea21_y_hicp_v2_no_antes_de_existir():
    t = _tabla("une_rt_m", "EA21", ["1999-01", "2026-06"])
    viejo = t[t["fecha_referencia"] == pd.Timestamp("1999-01-01")].iloc[0]
    assert viejo["fecha_publicacion"] >= pd.Timestamp(eurostat.CREACION_GEO["EA21"])
    assert eurostat.NOTA_POSTERIOR in viejo["nota"]
    # Después de la creación rige la regla: EUROSTAT_PARO desde 2012 = 1M10D (+ hábil TARGET)
    assert _pub(t, "2026-06-01") == pd.Timestamp("2026-08-10")
    t = _tabla("prc_hicp_minr", "DE", ["2005-03"])
    assert _pub(t, "2005-03-01") >= pd.Timestamp("2026-01-28")
    t = _tabla("prc_hicp_manr", "DE", ["2005-03"])  # v1: existía en su época
    assert _pub(t, "2005-03-01") < pd.Timestamp("2006-01-01")
    assert eurostat.no_antes_de("une_rt_m/M.SA.TOTAL.PC_ACT.T.EA") is None


def test_t1_bce_v2_e_i10_solo_desde_su_primera_version():
    clave = "HICP.M.U2.N.000000.4D0.ANR"
    hist = bce.parsear_csv("\n".join([
        CAB_HIST,
        f"{clave},M,U2,2005-01,1.0,Replace,2026-01-28T11:00:00.000+01:00,",
        f"{clave},M,U2,2026-09,2.0,Replace,2026-10-01T11:00:00.000+02:00,",
    ]))
    t = validar_series(bce.tabla_versiones("ecb:" + clave, hist, bce.SERIES_BCE["ecb:" + clave]))
    vieja = t[t["fecha_referencia"] == pd.Timestamp("2005-01-01")].iloc[0]
    assert vieja["fecha_publicacion"] == pd.Timestamp("2026-01-28")
    assert vieja["nota"] == bce.NOTA_POSTERIOR_BCE
    # LFSI U2 (composición cambiante, existía): la regla previa se mantiene (2M desde fin de mes).
    lf = "LFSI.M.U2.S.UNEHRT.TOTAL0.15_74.T"
    hist = bce.parsear_csv("\n".join([CAB_HIST, f"{lf},M,U2,2009-11,9.9,Replace,2026-03-04T11:00:00.000+01:00,"]))
    t = validar_series(bce.tabla_versiones("ecb:" + lf, hist, bce.SERIES_BCE["ecb:" + lf]))
    assert t["fecha_publicacion"].iloc[0] == pd.Timestamp("2010-01-31") and t["nota"].iloc[0] == bce.NOTA_PREVIA


# --- Yahoo: el mercado declarado en el universo manda --------------------------------------------------
@pytest.mark.parametrize(("ticker", "esperado"), [("DX-Y.NYB", "ICE_DXY"), ("^MOVE", "ICE_MOVE")])
def test_t1_yahoo_mercado_declarado_en_universo(ticker, esperado):
    meta = {"exchangeName": "NYB", "exchangeTimezoneName": "America/New_York"}
    assert yahoo.mercado_de(ticker, meta) == esperado


# --- S&P 500: tickers con sufijo de quiebra -----------------------------------------------------------
def test_t1_sp500_ticker_posterior_anotado():
    fechas = pd.to_datetime(["1996-01-02", "2001-12-03", "2005-01-03"])
    hist = pd.DataFrame({"fecha": fechas, "miembros": [frozenset({"ENRNQ", "MERQ", "NDAQ", "CPQ"}),
                                                       frozenset({"MERQ", "NDAQ"}), frozenset({"NDAQ"})]})
    t = constituyentes.tramos_sp500(hist).set_index("ticker")
    assert constituyentes.NOTA_TICKER_POSTERIOR in t.loc["ENRNQ", "nota"]
    for limpio in ("MERQ", "NDAQ", "CPQ"):
        assert constituyentes.NOTA_TICKER_POSTERIOR not in (t.loc[limpio, "nota"] or "")
