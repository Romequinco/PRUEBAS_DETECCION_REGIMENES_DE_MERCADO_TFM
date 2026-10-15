"""Tests de los generadores sintéticos del Tema 1 (``ingesta.mercados_series.sinteticos``)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ingesta.mercados_series import sinteticos as s
from ingesta.mercados_series.contrato import ErrorContrato, validar_precios, validar_series

GENERADORES_SERIES = [s.serie_con_vintages, s.serie_mercado_diaria, s.fx_ecb_sintetico]
GENERADORES_PRECIOS = [s.accion_con_eventos, s.accion_eventos_dificiles, s.precios_multimercado]


@pytest.fixture(scope="module")
def vintages():
    return s.serie_con_vintages()


@pytest.fixture(scope="module")
def accion():
    return s.accion_con_eventos()


@pytest.fixture(scope="module")
def multi():
    return s.precios_multimercado()


# --- Contrato y determinismo --------------------------------------------------------------------
@pytest.mark.parametrize("generador", GENERADORES_SERIES, ids=lambda f: f.__name__)
def test_t1_series_sinteticas_cumplen_contrato_y_son_deterministas(generador):
    a = generador()
    pd.testing.assert_frame_equal(a, validar_series(a))
    pd.testing.assert_frame_equal(a, generador())
    assert not np.allclose(a["valor"].fillna(0).to_numpy(), generador(semilla=99)["valor"].fillna(0).to_numpy())


@pytest.mark.parametrize("generador", GENERADORES_PRECIOS, ids=lambda f: f.__name__)
def test_t1_precios_sinteticos_cumplen_contrato_y_son_deterministas(generador):
    a = generador()
    pd.testing.assert_frame_equal(a, validar_precios(a))
    pd.testing.assert_frame_equal(a, generador())
    assert not np.allclose(a["close"].to_numpy(), generador(semilla=99)["close"].to_numpy())
    assert not any("adj" in c.lower() for c in a.columns)


# --- serie_con_vintages -------------------------------------------------------------------------
def test_t1_vintages_solo_revision_anual_como_el_paro_real(vintages):
    """Como la encuesta de hogares: primera estimación y solo la revisión anual del informe de enero."""
    assert set(vintages["metodo_publicacion"]) == {"vintage_exacto"}
    assert (vintages["vintage"] == vintages["fecha_publicacion"].dt.strftime("%Y-%m-%d")).all()
    n = vintages.groupby("fecha_referencia").size()
    assert n.max() == 4                                                     # 2023: primera + 3 anuales
    assert (n[n.index.year == 2023] == 4).all() and (n[n.index.year == 2024] == 3).all()
    assert n[pd.Timestamp("2025-06-01")] == 2 and n[pd.Timestamp("2026-02-01")] == 1
    assert n[s.MES_HUECO] == 1                                              # el hueco no se rellena
    # toda vintage que no es la primera de su mes es un informe de enero (revisión anual)
    informes_enero = set(vintages.loc[vintages["fecha_referencia"].dt.month == 1]
                         .groupby("fecha_referencia")["fecha_publicacion"].min())
    revisiones = vintages[vintages.duplicated("fecha_referencia", keep="first")]
    assert set(revisiones["fecha_publicacion"]) <= informes_enero
    for _, g in vintages.groupby("fecha_referencia"):
        assert g["fecha_publicacion"].is_monotonic_increasing and g["fecha_publicacion"].is_unique
    # siempre se publica después del mes de referencia; nivel parecido al real (3,4-4,6 %)
    assert (vintages["fecha_publicacion"] > vintages["fecha_referencia"] + pd.offsets.MonthEnd(0)).all()
    nivel = vintages.loc[vintages["fecha_referencia"].dt.year.isin([2024, 2025]), "valor"]
    assert 3.6 <= nivel.min() and nivel.max() <= 4.9
    assert vintages.set_index("fecha_referencia").loc["2026-01-01", "fecha_publicacion"] == pd.Timestamp("2026-02-11")


def test_t1_vintages_hueco_permanente(vintages):
    hueco = vintages[vintages["valor"].isna()]
    assert len(hueco) == 1
    fila = hueco.iloc[0]
    assert fila["fecha_referencia"] == s.MES_HUECO and fila["fecha_publicacion"] == s.INFORME_HUECO
    assert "hueco permanente" in fila["nota"]


def test_t1_vintages_retraso_por_cierre_del_gobierno(vintages):
    sep = vintages[vintages["fecha_referencia"] == s.MES_RETRASADO]
    primera = sep.iloc[0]
    assert primera["fecha_publicacion"] == pd.Timestamp("2025-11-20")
    assert primera["fecha_publicacion"] > s.FECHA_PREVISTA_RETRASADO and "retrasada" in primera["nota"]
    assert sep["nota"].iloc[1:].isna().all()


def test_t1_vintages_publicadas_a_las_0830_nueva_york(vintages):
    local = vintages["disponible_utc"].dt.tz_convert("America/New_York")
    assert (local.dt.strftime("%H:%M") == "08:30").all()
    assert (local.dt.tz_localize(None).dt.normalize() == vintages["fecha_publicacion"]).all()


# --- serie_mercado_diaria -----------------------------------------------------------------------
def test_t1_serie_diaria_visible_el_mismo_dia():
    d = s.serie_mercado_diaria()
    assert set(d["metodo_publicacion"]) == {"cierre_mercado"} and set(d["vintage"]) == {"unico"}
    assert (d["fecha_publicacion"] == d["fecha_referencia"]).all()       # 16:30 NY: borde incluido
    local = d["disponible_utc"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
    assert (local == "16:30").all()
    assert d["fecha_referencia"].dt.dayofweek.max() <= 4 and d["valor"].gt(0).all()


# --- fx_ecb_sintetico ---------------------------------------------------------------------------
def test_t1_fx_ecb():
    fx = s.fx_ecb_sintetico()
    assert set(fx["serie"]) == set(s.SERIES_FX)
    assert set(fx["metodo_publicacion"]) == {"calendario"}
    assert (fx["fecha_publicacion"] == fx["fecha_referencia"]).all()     # 16:00 Berlín < 16:30 NY
    ref = fx["fecha_referencia"]
    assert ref.dt.dayofweek.max() <= 4
    assert not ((ref.dt.month == 12) & (ref.dt.day.isin([25, 26]))).any()
    assert pd.Timestamp("2024-03-29") not in set(ref)                     # Viernes Santo 2024
    medias = fx.groupby("serie")["valor"].median()
    assert 0.8 < medias["ecb:EXR.D.USD.EUR"] < 1.5 and 0.6 < medias["ecb:EXR.D.GBP.EUR"] < 1.1
    assert 80 < medias["ecb:EXR.D.JPY.EUR"] < 250
    assert fx.groupby("serie")["divisa"].first().to_dict() == {
        "ecb:EXR.D.GBP.EUR": "GBP", "ecb:EXR.D.JPY.EUR": "JPY", "ecb:EXR.D.USD.EUR": "USD"}


# --- accion_con_eventos -------------------------------------------------------------------------
def test_t1_accion_split_y_dividendos(accion):
    assert set(accion["ticker"]) == {s.TICKER_ACCION} and set(accion["mercado"]) == {"XNYS"}
    assert 700 <= len(accion) <= 800                                     # ~3 años de sesiones
    splits = accion[accion["split"] != 1.0]
    assert splits["sesion"].tolist() == [s.SESION_SPLIT] and splits["split"].iloc[0] == 2.0
    divs = accion[accion["dividendo"] > 0].set_index("sesion")["dividendo"]
    assert divs.index.tolist() == sorted(s.DIVIDENDOS_ECONOMICOS)
    antes = divs[divs.index < s.SESION_SPLIT]
    despues = divs[divs.index > s.SESION_SPLIT]
    assert len(antes) == 2 and len(despues) == 2
    assert np.allclose(antes, 0.80) and np.allclose(despues, 0.40)       # as-traded


def test_t1_accion_as_traded_se_parte_en_el_split(accion):
    i = accion.index[accion["sesion"] == s.SESION_SPLIT][0]
    ratio = accion.loc[i, "close"] / accion.loc[i - 1, "close"]
    assert 0.4 < ratio < 0.6                                             # sin ajustar: cae ~50 %
    vol = accion.set_index("sesion")["volume"]
    assert vol[vol.index >= s.SESION_SPLIT].median() > 1.5 * vol[vol.index < s.SESION_SPLIT].median()


def test_t1_accion_cae_lo_que_el_dividendo(accion):
    for ex in s.DIVIDENDOS_ECONOMICOS:
        i = accion.index[accion["sesion"] == ex][0]
        hueco = accion.loc[i - 1, "close"] - accion.loc[i, "open"]
        assert hueco > 0.5 * accion.loc[i, "dividendo"]


def test_t1_accion_ohlc_coherente(accion):
    assert (accion["high"] >= accion[["open", "close"]].max(axis=1)).all()
    assert (accion["low"] <= accion[["open", "close"]].min(axis=1)).all()


# --- vista_yahoo_hoy ----------------------------------------------------------------------------
def test_t1_vista_yahoo_ajusta_todo(accion):
    y = s.vista_yahoo_hoy(accion)
    a = accion.set_index("sesion")
    antes = y.index < s.SESION_SPLIT
    assert np.allclose(y.loc[antes, "Close"], a.loc[antes, "close"] / 2)
    assert np.allclose(y.loc[~antes, "Close"], a.loc[~antes, "close"])
    assert np.allclose(y["Dividends"][y["Dividends"] > 0], 0.40)
    assert y["Stock Splits"].tolist().count(2.0) == 1 and (y["Stock Splits"].isin([0.0, 2.0])).all()
    ultimo_ex = max(s.DIVIDENDOS_ECONOMICOS)
    assert np.allclose(y.loc[y.index >= ultimo_ex, "Adj Close"], y.loc[y.index >= ultimo_ex, "Close"])
    assert (y.loc[y.index < ultimo_ex, "Adj Close"] < y.loc[y.index < ultimo_ex, "Close"]).all()
    assert np.allclose(y.loc[antes, "Volume"], a.loc[antes, "volume"] * 2)


def test_t1_vista_yahoo_retorno_continuo_en_el_split(accion):
    y = s.vista_yahoo_hoy(accion)
    r = y["Close"].pct_change()
    assert abs(r.loc[s.SESION_SPLIT]) < 0.1                               # el split no es un retorno


def test_t1_vista_yahoo_nunca_cumple_el_contrato(accion):
    y = s.vista_yahoo_hoy(accion)
    assert "Adj Close" in y.columns
    with pytest.raises(ErrorContrato, match="columnas prohibidas"):
        validar_precios(y.reset_index())
    with pytest.raises(ValueError):
        s.vista_yahoo_hoy(s.precios_multimercado())


# --- precios_multimercado -----------------------------------------------------------------------
def test_t1_multimercado_mercados_y_divisas(multi):
    assert set(multi["mercado"]) == {"XNYS", "XLON", "XETR", "XTKS", "CME", "CRYPTO"}
    divisas = multi.groupby("mercado")["divisa"].first()
    assert divisas["XLON"] == "GBp" and divisas["XTKS"] == "JPY" and divisas["XETR"] == "EUR"
    assert (multi["split"] == 1.0).all() and (multi["dividendo"] == 0.0).all()
    cripto = multi[multi["mercado"] == "CRYPTO"]
    assert cripto["sesion"].dt.dayofweek.max() == 6
    assert multi.loc[multi["mercado"] != "CRYPTO", "sesion"].dt.dayofweek.max() <= 4


def test_t1_multimercado_fecha_publicacion_por_mercado(multi):
    dias = (multi["fecha_publicacion"] - multi["sesion"]).dt.days.groupby(multi["mercado"]).unique()
    assert {m: v.tolist() for m, v in dias.items()} == {
        "CME": [1], "CRYPTO": [1], "XETR": [0], "XLON": [0], "XNYS": [0], "XTKS": [0]}


def test_t1_multimercado_dst_de_cada_huso(multi):
    hora = multi.assign(h=multi["disponible_utc"].dt.strftime("%H:%M")).set_index(["mercado", "sesion"])["h"]
    # Londres sigue en GMT todo el periodo: 16:30 + 30 min = 17:00 UTC.
    assert set(hora["XLON"]) == {"17:00"}
    # Nueva York cambia el 2024-03-10: 16:30 EST = 21:30 UTC; 16:30 EDT = 20:30 UTC.
    assert hora[("XNYS", pd.Timestamp("2024-03-08"))] == "21:30"
    assert hora[("XNYS", pd.Timestamp("2024-03-11"))] == "20:30"
    assert set(hora["XTKS"]) == {"06:30"} and set(hora["CRYPTO"]) == {"00:30"}


def test_t1_serie_diaria_no_se_llama_vix():
    """El VIX real (Cboe, 16:15) se ve en t+1; la serie del borde de las 16:30 no puede llamarse VIX."""
    d = s.serie_mercado_diaria(hasta="2023-01-31")
    assert set(d["serie"]) == {s.SERIE_DIARIA} and "VIX" not in s.SERIE_DIARIA


def test_t1_accion_eventos_dificiles():
    p = s.accion_eventos_dificiles()
    assert set(p["ticker"]) == {s.TICKER_DIFICIL}
    eventos = p[p["split"] != 1.0].set_index("sesion")["split"].to_dict()
    assert eventos == s.SPLITS_DIFICILES                                      # contrasplit 0.1 y 3:2
    divs = p[p["dividendo"] > 0].set_index("sesion")["dividendo"]
    assert divs.index.tolist() == sorted(s.DIVIDENDOS_DIFICILES)
    assert divs.loc["2024-02-15"] == pytest.approx(0.30 * 0.1 * 1.5)         # as-traded antes de ambos
    # el contrasplit multiplica por 10 el precio as-traded
    i = p.index[p["sesion"] == pd.Timestamp("2024-03-15")][0]
    assert 6 < p.loc[i, "close"] / p.loc[i - 1, "close"] < 15
