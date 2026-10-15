"""Figuras del Tema 4: cada función se llama con tablas mínimas inventadas (backend Agg) y no debe fallar.

Las columnas imitan las de ``Resultado.tablas`` de cada extractor y las de las tablas de validez de t4_02.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from ingesta.noticias_alt import graficos as g  # noqa: E402

RNG = np.random.default_rng(0)
SERIES = ["gprd", "epu_usa_diario", "skew", "vix_cierre"]


@pytest.fixture(autouse=True)
def _cerrar():
    g.estilo()
    yield
    plt.close("all")


def _ok(salida):
    fig, ax = salida
    assert isinstance(fig, plt.Figure)
    assert ax is not None
    fig.canvas.draw()          # fuerza el dibujado completo (errores de ejes, fechas y escalas)
    return fig, ax


# --------------------------------------------------------------------------- tablas comunes

def _cobertura():
    return pd.DataFrame(dict(
        fuente=["ccnews", "gdelt_gkg", "rss_medios", "congreso_house_ptr", "polymarket", "gpr", "reddit"],
        familia=["texto_historico", "gdelt_wiki", "vivo_redes", "divulgaciones", "prediccion", "indices_ews",
                 "vivo_redes"],
        tipo=["texto", "titular", "titular", "evento", "serie", "serie", "ficha"],
        inicio=["2016-08-24", "2015-02-18", None, "2014-01-02", "2020-06-01", "1985-01-01", None],
        fin=["2026-10-13", "2026-10-13", None, "2026-10-12", None, "2026-10-13", None],
        hora=[True, True, True, False, True, False, False],
        idiomas=["multi", "multi", "en", "en", "en", "en", ""],
        lag_mediana_h=[0.7, 0.25, 0.3, 720.0, 0.0, 96.0, np.nan],
        destino=["base+cajon", "cajon", "vivo", "serie", "serie", "serie", "ninguno"],
        backtest=[True, True, False, True, True, True, False],
        estado=["ok", "ok", "parcial", "ok", "ok", "ok", "no_probada"],
        nota="",
    ))


def _registro():
    return pd.DataFrame(dict(fuente=["ccnews", "ccnews", "gdelt_gkg", "rss_medios", "rss_medios", "gpr"],
                             item="x", dia="", url="", ok=[True, True, True, True, False, False], bytes=1,
                             segundos=0.1, error=""))


def _serie_larga(nombre, n=400, desde="2015-01-01"):
    f = pd.date_range(desde, periods=n, freq="D")
    return pd.DataFrame(dict(serie=nombre, fecha_referencia=f, fecha_publicacion=f + pd.Timedelta(days=1),
                             valor=RNG.lognormal(5, 0.5, n), unidad="u", fuente="x"))


# --------------------------------------------------------------------------- estilo y base existente

def test_color_valor():
    assert g.color_valor("ok") == g.C_ESTADO["ok"]
    assert g.color_valor("fallo") == g.C_ESTADO["fallo"]
    assert g.color_valor("apta_backtest") == g.C_SEMAFORO["apta_backtest"]
    assert g.color_valor(True) == g.C_ESTADO["ok"]
    assert g.color_valor(None) == "#f2f2f2"
    assert g.color_valor("desconocido") == "#ffffff"
    assert g.color_valor(1.0).startswith("#")


def test_base_existente():
    cob = _cobertura()
    _ok(g.gantt_cobertura(cob))
    _ok(g.mapa_epocas(cob))
    _ok(g.barras_estado(_registro()))
    _ok(g.paneles_series({"a": pd.Series(RNG.normal(size=50), index=pd.date_range("2020-01-01", periods=50))}))


# --------------------------------------------------------------------------- t4_01

def test_esquema_y_estado():
    cob = _cobertura()
    _ok(g.esquema_destinos(cob))
    _ok(g.estado_fuentes(cob, _registro()))
    _ok(g.estado_fuentes(cob))


def test_ccnews():
    art = pd.DataFrame(dict(
        dia=["2016-11-09"] * 3 + ["2020-03-16"] * 2,
        idioma=["en", "es", "en", "de", "en"], n_tokens=[40, 450, 1200, 300, 5000],
        fecha_publicacion=["2016-11-09T14:20:40+00:00", "2016-11-09T10:00:00+00:00", None,
                           "2020-03-16T08:00:00+00:00", "2020-03-15T22:00:00+00:00"],
        warc_date=["2016-11-09T15:00:07+00:00", "2016-11-09T15:01:00+00:00", "2016-11-09T15:02:00+00:00",
                   "2020-03-16T09:00:00+00:00", "2020-03-16T09:30:00+00:00"],
        regla=["meta", "meta", "rastreo_sin_meta", "meta", "meta"],
    ))
    idi = pd.DataFrame(dict(dataset="stanford-oval/ccnews", config="2020", idioma=["en", "es", "fr"],
                            filas=[600, 200, 100], pct=[66.7, 22.2, 11.1]))
    _ok(g.ccnews_tokens_idiomas(art, idi))
    _ok(g.ccnews_tokens_idiomas(art))
    _ok(g.fecha_articulo_vs_rastreo(art))
    emb = pd.DataFrame(dict(etapa=["bruto", "tras_url", "tras_titular", "tras_simhash"], orden=[0, 1, 2, 3],
                            n=[100, 90, 80, 76], eliminados=[0, 10, 10, 4], pct_restante=[100, 90, 80, 76]))
    _ok(g.embudo_duplicados(emb))
    _ok(g.barras_idiomas(art))


def test_hf_y_volumen():
    hf = pd.DataFrame(dict(dataset=["a/x", "a/x", "b/y"], fuente="f", anio=[2016, 2017, 2017],
                           filas=[10, 2000, 50], base="config"))
    _ok(g.filas_anio_datasets(hf))
    vol = pd.DataFrame(dict(fuente=["ccnews", "gpr", "rss_medios"], n=[1000, 15000, 0],
                            familia=["texto_historico", "indices_ews", "vivo_redes"]))
    _ok(g.volumen_fuentes(vol))


def test_gdelt():
    tono = pd.DataFrame(dict(clave=["covid"] * 3 + ["svb"] * 3, dia="d", hora_utc=[0, 8, 16] * 2,
                             tono_medio=RNG.normal(-1, 0.3, 6), n=100))
    temas = pd.DataFrame(dict(clave="covid", dia="d", grupo=["economia"] * 3 + ["todos"],
                              tema=["EPU_POLICY", "ECON_STOCKMARKET", "EPU_ECONOMY", "TAX_FNCACT"],
                              n=[50, 30, 20, 90], pct_articulos=1.0, tono_medio=[-1.2, 0.4, None, -0.5]))
    _ok(g.gdelt_tono_temas(tono, temas))
    resumen = pd.DataFrame(dict(clave=["lehman", "covid"], dia="d", version=["1.0_historico", "2.0"],
                                ficheros=1, n=[100, 200], pct_sqldate_anterior=[100.0, 2.0],
                                lag_dias_mediana=[1602.0, 0.0], lag_dias_p90=[1602.0, 0.0],
                                lag_dias_max=[1604.0, 365.0], pct_con_url=[0.0, 100.0]))
    _ok(g.gdelt_desfase_eventos(resumen))
    _ok(g.gdelt_desfase_eventos(pd.DataFrame(dict(lag_dias=[0, 0, 0, 1, 3, 30, 365]))))


def test_wikipedia():
    ce = pd.DataFrame(dict(clave=["lehman", "covid"], dia=["2008-09-15", "2020-03-16"], n_hechos=[9, 31],
                           n_hechos_actual=[19, 45]))
    pv = pd.concat([_serie_larga("wiki_pv:Recession"), _serie_larga("wiki_pv:Bank_run")])
    picos = pd.DataFrame(dict(articulo=["Recession"], fecha_pico=["2015-06-01"], visitas_pico=[2000]))
    _ok(g.wikipedia_hechos_visitas(ce, pv, picos))


def test_rss():
    feeds = pd.DataFrame(dict(fuente=["rss_medios", "rss_medios", "bing_news_rss"], feed=["a", "b", "c"],
                              pais=["GB", "US", "US"], robots_ok=[True, False, True],
                              estado=["ok", "fallo", "ok"], profundidad_h=[96.4, np.nan, 12.0]))
    _ok(g.rss_feeds(feeds))


def test_congreso_e_insiders():
    anio = pd.DataFrame(dict(anio=[2014, 2015], n_electronicos=[285, 390], n_escaneados=[423, 338]))
    ops = pd.DataFrame(dict(lag_dias=[3, 32, 60, 12, 45]))
    _ok(g.congreso_ptr(anio, ops))
    f4 = pd.DataFrame(dict(lag_dias=[0, 1, 2, 3, 5, 9], codigo=["S", "P", "M", "A", "S", None]))
    _ok(g.form4_lag_codigos(f4))


def test_prediccion():
    pm = pd.DataFrame(dict(anio=[2020, 2021], volumen_usd=[7.6e7, 6.0e7], truncado=[False, True]))
    ka = pd.DataFrame(dict(anio=[2021, 2022], volumen_contratos=[14609.0, 5.7e6]))
    _ok(g.prediccion_liquidez(pm, ka))
    f = pd.date_range("2025-01-08", periods=60)
    serie = pd.DataFrame(dict(serie="polymarket_recesion_eeuu_2025", fecha_referencia=f, valor=np.linspace(.2, .6, 60)))
    saltos = pd.DataFrame(dict(serie="polymarket_recesion_eeuu_2025", fecha=[f[10], f[40]], de=[.3, .5],
                               a=[.5, .3], cambio_pp=[20.0, -20.0]))
    _ok(g.serie_con_saltos(serie, saltos))
    calib = pd.DataFrame(dict(plataforma=["kalshi"] * 2 + ["polymarket"] * 2, horizonte=["1d", "7d"] * 2,
                              tramo="t", n=[3, 5, 10, 2], p_media=[0.1, 0.8, 0.3, 0.9], frec_yes=[0.0, 1.0, 0.25, 1.0],
                              brier=[0.02, 0.05, 0.09, 0.1]))
    _ok(g.fiabilidad_calibracion(calib))


def test_indices_ews_lags():
    larga = pd.concat([_serie_larga("gprd", n=3000, desde="2000-01-01"), _serie_larga("vix_cierre")])
    _ok(g.indices_paneles(larga, log=["gprd"]))
    ts = pd.date_range("2025-10-14", periods=200, freq="30min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    tz = pd.DataFrame(dict(ts=ts, valor=1, z_publicado=RNG.normal(size=200),
                           z_causal=np.r_[np.full(50, np.nan), RNG.normal(size=150)]))
    _ok(g.ews_z(tz, umbrales=(2.35, 3.0)))
    tg = pd.DataFrame(dict(fuente=["ccnews", "ccnews", "wiki_current_events", "gdelt_eventos"],
                           lag_h=[0.7, 1.2, -20.5, 8760.0]))
    _ok(g.lags_publicacion(tg, _cobertura()))


# --------------------------------------------------------------------------- t4_02

def test_validez_formas_genericas():
    """Tablas largas/anchas genéricas y nombres alternativos de columnas (ic_inf, desde, subperiodo...)."""
    criterios = ["huecos", "revision", "look_ahead"]
    larga = pd.DataFrame([dict(serie=s, criterio=c, valor=RNG.uniform()) for s in SERIES for c in criterios])
    _ok(g.calor_calidad(larga, columna="criterio", valor="valor"))
    _ok(g.calor_calidad(larga.pivot(index="serie", columns="criterio", values="valor").reset_index(), criterios=None))
    texto = larga.assign(valor=RNG.choice(["ok", "parcial", "fallo"], len(larga)))
    _ok(g.calor_calidad(texto, columna="criterio", valor="valor"))
    la = pd.DataFrame(dict(fuente=SERIES, revisable=["sí", "no", "no", "no"],
                           veredicto=["parcial", "ok", "look-ahead", "ok"], lag_h=[1.0, 2.0, 3.0, 4.0]))
    _ok(g.tabla_semaforo(la, invertir=["revisable"]))
    ev = pd.DataFrame([dict(serie=s, grupo=gr, dia_rel=t, media=t / 10, ic_inf=t / 10 - 1, ic_sup=t / 10 + 1)
                       for s in SERIES for gr in ("crisis", "control") for t in range(-5, 6)])
    _ok(g.estudio_eventos(ev, grupo="grupo"))
    auc = pd.DataFrame([dict(serie=s, adelanto=a, auc=RNG.uniform(.3, .9), ic_inf=.3, ic_sup=.95)
                        for s in SERIES for a in (0, 20, 60)])
    _ok(g.auc_vs_vix(auc))
    ll = pd.DataFrame([dict(serie=s, lag=k, corr=RNG.uniform(-.5, .5)) for s in SERIES for k in range(-5, 6)])
    _ok(g.calor_lead_lag(ll))
    est = pd.DataFrame([dict(serie=s, subperiodo=p, auc=RNG.uniform(.4, .8)) for s in SERIES
                        for p in ("1990-2006", "2007-2019", "2020-hoy")])
    _ok(g.estabilidad_subperiodos(est, subperiodo="subperiodo"))
    _ok(g.estabilidad_subperiodos(est, subperiodo="subperiodo", como="calor"))
    car = pd.DataFrame([dict(grupo=gr, horizonte=h, desde=d, car=RNG.normal(), ic_inf=-2, ic_sup=2)
                        for gr in ("congreso", "insiders") for h in (5, 20) for d in ("publicacion", "operacion")])
    _ok(g.car_grupos(car, escala=1.0))
    sem = pd.DataFrame([dict(fuente=s, criterio=c, valor=RNG.choice(["ok", "parcial", "fallo"]), veredicto=v)
                        for s, v in zip(SERIES, ["apta_backtest", "solo_vivo", "contexto", "descartar"])
                        for c in criterios])
    _ok(g.semaforo_final(sem, fila="fuente", columna="criterio", valor="valor"))
    ancha = sem.pivot(index="fuente", columns="criterio", values="valor").reset_index()
    _ok(g.semaforo_final(ancha, fila="fuente", criterios=None, veredicto={"gprd": "apta_backtest"}))


def test_validez_columnas_reales():
    """Columnas de las tablas ``validez_*`` que escribe ``validez.py``."""
    calidad = pd.DataFrame(dict(serie=SERIES, fuente="x", pct_huecos=[0.0, 30.7, 3.5, 0.0],
                                pct_atipicos=[0.1, 0.2, 11.0, 0.0], n_rupturas=[0, 1, 2, 0],
                                pct_repetidos=[0.0, 24.6, 0.3, 0.0], racha_max_constante=[1, 11, 2, 1],
                                lag_pub_mediana_dias=[1.0, 31.0, 4.0, np.nan], n_pub_antes_ref=0,
                                revisable=[True, False, True, False]))
    _ok(g.calor_calidad(calidad))
    look = pd.DataFrame(dict(comprobacion=["publicacion_no_anterior_a_referencia"] * 4 + ["valor_revisado_vintage_actual"],
                             fuente=SERIES + ["gprd"], resultado=["ok", "ok", "ok", "ok", "riesgo"],
                             evidencia="texto", valor=0.0))
    _ok(g.lookahead_semaforo(look))
    _ok(g.lookahead_semaforo(look, solo_con_riesgo=True))
    ev = pd.DataFrame([dict(serie=s, dia_rel=t, media=t / 10, lo=t / 10 - 1, hi=t / 10 + 1, n_crisis=7)
                       for s in SERIES for t in range(-60, 61, 5)])
    fig, _ = _ok(g.estudio_eventos(ev, series=SERIES[:3], nivel_ic=0.9))
    assert "IC 90%" in fig._suptitle.get_text() and "pico" in fig._suptitle.get_text()
    auc = pd.DataFrame([dict(serie=s, adelanto=a, auc=RNG.uniform(.3, .9), direccion="directa", n_dias=100,
                             auc_vix_mismo_periodo=RNG.uniform(.5, .8)) for s in SERIES[:3] for a in (0, 20, 60)])
    auc.loc[0, "auc"] = np.nan                      # serie sin datos en un adelanto
    _ok(g.auc_vs_vix(auc))
    ll = pd.DataFrame([dict(serie=s, referencia=r, metrica=mt, lag=k, corr=RNG.uniform(-.5, .5), n=100)
                       for s in SERIES for r in ("vix", "drawdown_sp500") for mt in ("cambio_5d", "nivel_z")
                       for k in range(-60, 61, 2)])
    _ok(g.calor_lead_lag(ll))
    _ok(g.calor_lead_lag(ll, filtro={"referencia": "drawdown_sp500", "metrica": "nivel_z"}))
    est = pd.DataFrame([dict(serie=f"s{i}", periodo=p, auc=RNG.uniform(.4, .8), auc_vix=0.7) for i in range(12)
                        for p in ("desde_2015", "decada_2010s", "antes_2015", "decada_1990s")])
    _ok(g.estabilidad_subperiodos(est))
    _ok(g.estabilidad_subperiodos(est[est["serie"].isin(["s0", "s1"])], como="lineas"))
    assert g._orden_periodos(["desde_2015", "decada_2010s", "antes_2015", "decada_1990s"]) == \
        ["decada_1990s", "decada_2010s", "antes_2015", "desde_2015"]
    bins = pd.DataFrame(dict(plataforma=["kalshi"] * 3 + ["polymarket"] * 3, horizonte_dias=[1, 1, 7] * 2,
                             bin=0, n=[9, 2, 5, 4, 3, 8], p_media=[0.02, 0.8, 0.5] * 2, frec_obs=[0.0, 1.0, 0.4] * 2))
    resumen = pd.DataFrame(dict(plataforma=["kalshi", "kalshi", "polymarket"], horizonte_dias=[1, 7, 1],
                                liquidez="todos", brier=[0.13, 0.12, 0.08]))
    _ok(g.fiabilidad_calibracion(bins, resumen))
    _ok(g.fiabilidad_calibracion(bins))
    car = pd.DataFrame([dict(grupo=gr, referencia_fecha=d, horizonte=h, n=100, car_medio=RNG.normal(0, .01),
                             lo=-0.02, hi=0.02) for gr in ("congreso", "insiders") for d in ("publicacion", "operacion")
                        for h in (5, 20, 60)])
    fig, _ = _ok(g.car_grupos(car, nivel_ic=0.9))
    assert "IC 90%" in fig._suptitle.get_text()
    sem = pd.DataFrame(dict(candidato=SERIES, tipo="serie", fuente="x", historia=[2, 0, 2, 1],
                            calidad=[2, 2, np.nan, 1], lookahead=[2, 1, 2, 2], senal=[2, 0, np.nan, 1],
                            acceso=[2, 2, 0, 1], veredicto=["apta_backtest", "solo_vivo", "descartar", "contexto"],
                            motivo="m"))
    fig, ax = _ok(g.semaforo_final(sem))
    assert [t.get_text() for t in ax.get_yticklabels()] == ["gprd", "epu_usa_diario", "vix_cierre", "skew"]


def test_tablas_vacias_no_fallan():
    vacio = pd.DataFrame()
    _ok(g.embudo_duplicados(vacio))
    _ok(g.filas_anio_datasets(vacio))
    _ok(g.gdelt_desfase_eventos(vacio))
    _ok(g.rss_feeds(vacio))
    _ok(g.form4_lag_codigos(vacio))
    _ok(g.prediccion_liquidez(None, None))
    _ok(g.ews_z(vacio))
    _ok(g.serie_con_saltos({}))
    _ok(g.lags_publicacion(vacio))
    _ok(g.estabilidad_subperiodos(vacio))
    _ok(g.fiabilidad_calibracion(vacio))
    _ok(g.calor_calidad(vacio, criterios=None))
    _ok(g.semaforo_final(pd.DataFrame(columns=["candidato", "veredicto"])))
