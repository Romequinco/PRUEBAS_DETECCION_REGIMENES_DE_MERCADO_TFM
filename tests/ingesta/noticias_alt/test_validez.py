"""Tests de ``noticias_alt.validez`` con datos sintéticos (sin red)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ingesta.noticias_alt import base, validez as v

CRISIS = {"c1": ("2001-03-01", "2001-04-30"), "c2": ("2003-06-02", "2003-07-31")}


def _larga(fechas_ref, fechas_pub, valores, nombre="x"):
    return pd.DataFrame(dict(serie=nombre, fecha_referencia=pd.to_datetime(fechas_ref),
                             fecha_publicacion=pd.to_datetime(fechas_pub), valor=valores, unidad="u", fuente="f",
                             vintage=None, disponible_utc=None, metodo_publicacion="m", nota=None))


# --------------------------------------------------------------------------- AUC

def test_auc_perfecto_inverso_y_azar():
    y = np.r_[np.zeros(500), np.ones(500)]
    assert v.auc_mann_whitney(np.arange(1000), y) == 1.0
    assert v.auc_mann_whitney(-np.arange(1000), y) == 0.0
    rng = np.random.default_rng(1)
    assert abs(v.auc_mann_whitney(rng.normal(size=20000), rng.integers(0, 2, 20000)) - 0.5) < 0.02
    assert v.auc_mann_whitney([1.0, 1.0], [0, 1]) == 0.5           # empates a la mitad
    assert np.isnan(v.auc_mann_whitney([1.0, 2.0], [1, 1]))         # una sola clase


def test_auc_crisis_adelanto_detecta_anticipo():
    dias = pd.bdate_range("2000-01-03", "2004-12-31")
    y = v.etiqueta_crisis(dias, CRISIS)
    # La serie sube 20 días ANTES de cada crisis y vuelve a 0 al empezar: anticipa pero no coincide.
    z = pd.Series(y.shift(-20).fillna(0).to_numpy() * (1 - y.to_numpy()), index=dias, dtype=float)
    t = v.auc_crisis(z, CRISIS, adelantos=(0, 20), z_vix=z).set_index("adelanto")
    assert t.loc[20, "auc"] > t.loc[0, "auc"]
    assert t.loc[0, "direccion"] == "inversa"
    assert t.loc[20, "n_crisis"] == 2
    assert t.loc[20, "auc_vix_mismo_periodo"] == pytest.approx(t.loc[20, "auc"])
    assert t.loc[20, "auc_mismo_periodo"] == pytest.approx(t.loc[20, "auc"])


def test_auc_mismo_periodo_usa_solo_los_dias_del_vix():
    dias = pd.bdate_range("2000-01-03", "2004-12-31")
    y = v.etiqueta_crisis(dias, CRISIS).astype(float)
    z = y.copy()
    z.loc[:"2002-12-31"] = -y.loc[:"2002-12-31"]                     # antes de 2003, señal invertida
    z_vix = pd.Series(np.where(dias >= pd.Timestamp("2003-01-01"), y, np.nan), index=dias)
    t = v.auc_crisis(z, CRISIS, adelantos=(0,), z_vix=z_vix, n_boot=0).iloc[0]
    assert t["auc_mismo_periodo"] == pytest.approx(1.0) and t["auc"] < 0.9
    assert t["n_dias_mismo_periodo"] == t["n_dias_vix"] < t["n_dias"]
    assert t["n_crisis_mismo_periodo"] == 1


def test_auc_ic_bloques_ancho_con_pocas_crisis():
    rng = np.random.default_rng(5)
    dias = pd.bdate_range("2000-01-03", "2004-12-31")
    y = v.etiqueta_crisis(dias, CRISIS).astype(float)
    z = pd.Series(rng.normal(size=len(dias)), index=dias) + y      # separa, pero solo 2 crisis
    lo, hi = v.auc_ic_bloques(z, y, CRISIS, n_boot=200)
    assert lo <= v.auc_mann_whitney(z.to_numpy(), y.to_numpy()) <= hi
    blq = v.bloques_crisis(pd.DatetimeIndex(["2000-06-01", "2002-01-01", "2004-01-01"]), CRISIS)
    assert list(blq) == [0, 1, 1]


# --------------------------------------------------------------------------- z causal

def test_z_expanding_no_usa_el_futuro():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=800), index=pd.bdate_range("2000-01-03", periods=800))
    z1 = v.z_expanding(s)
    s2 = s.copy()
    s2.iloc[500:] = s2.iloc[500:] * 50 + 100                         # cambia el futuro
    z2 = v.z_expanding(s2)
    pd.testing.assert_series_equal(z1.iloc[:500], z2.iloc[:500])
    assert z1.iloc[: v.MIN_OBS_Z - 1].isna().all() and z1.iloc[v.MIN_OBS_Z - 1:].notna().all()


def test_comprobar_z_causal_ok():
    rng = np.random.default_rng(3)
    dias = pd.bdate_range("2000-01-03", periods=900)
    larga = _larga(dias, dias + pd.offsets.BDay(1), rng.normal(size=900), "a")
    rej_dias = pd.bdate_range("2000-01-03", periods=905)
    zs, _ = v.z_pipeline(larga, rej_dias)
    filas = v.comprobar_z_causal(larga, zs)
    assert filas[0]["resultado"] == "ok" and len(filas) == 1


def test_comprobar_z_causal_detecta_transformacion_con_toda_la_muestra():
    # Positiva y simétrica al principio; picos enormes al final: con toda la muestra se elige log, con el
    # pasado no. El z de los primeros años cambia -> riesgo (la comprobación anterior no lo veía).
    rng = np.random.default_rng(4)
    dias = pd.bdate_range("2000-01-03", periods=1500)
    val = 100 + rng.normal(size=1500)
    val[1400:] = rng.uniform(1e3, 1e5, 100)
    larga = _larga(dias, dias, val, "b")
    zs, tr = v.z_pipeline(larga, dias)
    assert tr["b"] == "log"
    filas = v.comprobar_z_causal(larga, zs)
    assert filas[0]["resultado"] == "riesgo"
    fila_b = [f for f in filas if f["fuente"] == "b"][0]
    assert fila_b["resultado"] == "riesgo" and "nivel con datos hasta" in fila_b["evidencia"]


# --------------------------------------------------------------------------- rejilla as-of

def test_a_rejilla_por_publicacion_y_vigencia():
    # Mensual publicada con 10 días de retraso; la última referencia vence tras 3 espaciados.
    ref = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
    pub = ref + pd.Timedelta(days=10)
    dias = pd.bdate_range("2020-01-01", "2020-08-31")
    s = v.a_rejilla(_larga(ref, pub, [1.0, 2.0, 3.0]), dias)
    assert s.loc[:"2020-01-10"].isna().all()                        # antes de publicarse no se sabe
    assert s.loc["2020-01-13"] == 1.0 and s.loc["2020-02-11"] == 2.0
    assert s.loc["2020-03-11"] == 3.0
    assert s.loc["2020-08-31":].isna().all()                        # fuente interrumpida: no se arrastra


def test_a_rejilla_publicacion_tardia_no_pisa_dato_nuevo():
    ref = pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-02"])
    pub = pd.to_datetime(["2020-01-07", "2020-01-08", "2020-01-09"])   # la del día 2 llega tarde
    s = v.a_rejilla(_larga(ref, pub, [10.0, 20.0, 99.0]), pd.bdate_range("2020-01-06", "2020-01-10"))
    assert s.loc["2020-01-09"] == 20.0


# --------------------------------------------------------------------------- estudio de eventos y lead-lag

def test_estudio_eventos_pico():
    dias = pd.bdate_range("2000-01-03", "2004-12-31")
    z = pd.Series(0.0, index=dias)
    for pico, _ in CRISIS.values():
        z.loc[pd.Timestamp(pico)] = 5.0
    t = v.estudio_eventos(z, CRISIS, ventana=(-2, 2), n_boot=200).set_index("dia_rel")
    assert t.loc[0, "media"] == 5.0 and t.loc[0, "n_crisis"] == 2 and t.loc[1, "media"] == 0.0


def test_lead_lag_encuentra_el_adelanto():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(size=1000), index=pd.bdate_range("2000-01-03", periods=1000))
    x = r.shift(-5)                                                    # x_t = r_{t+5}: x va 5 días por delante
    res = v.resumen_lead_lag(v.lead_lag(x, r, range(-10, 11)))
    assert res["lag_max"] == 5 and res["corr_max"] == pytest.approx(1.0)


def test_lead_lag_plano_es_indeterminado():
    # Serie persistente frente a sí misma desplazada: CCF casi plana -> el máximo no se distingue del lag 0.
    rng = np.random.default_rng(6)
    r = pd.Series(np.cumsum(rng.normal(size=1500)), index=pd.bdate_range("2000-01-03", periods=1500))
    x = r.shift(-2)
    t = v.lead_lag(x, r, range(-10, 11))
    n_eff = v.n_efectivo(x, r, "nivel_z")
    assert n_eff < 100
    res = v.resumen_lead_lag(t, n_eff)
    assert res["lag_max_bruto"] == 2 and res["lag_max"] is None
    ok = (x.diff(5).notna() & r.diff(5).notna()).sum()
    assert v.n_efectivo(x.diff(5), r.diff(5), "cambio_5d") == pytest.approx(ok / 5)


def test_a_rejilla_ref_sin_retraso_de_publicacion():
    ref = pd.bdate_range("2020-01-06", periods=10)
    larga = _larga(ref, ref + pd.Timedelta(days=7), np.arange(10.0))
    dias = pd.bdate_range("2020-01-06", periods=20)
    assert v.a_rejilla_ref(larga, dias).loc["2020-01-07"] == 1.0
    assert np.isnan(v.a_rejilla(larga, dias).loc["2020-01-07"])


def test_ic_t_media_y_bootstrap_por_conglomerados():
    m, lo, hi = v.ic_t_media([0.5, 1.3])                               # 2 crisis: sin intervalo
    assert m == pytest.approx(0.9) and np.isnan(lo) and np.isnan(hi)
    m, lo, hi = v.ic_t_media([1.0, 2.0, 3.0, 4.0, 5.0])
    semi = 2.1318 * np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5)      # t(0,95; 4 g.l.) = 2,1318
    assert m == 3.0 and hi - lo == pytest.approx(2 * semi, rel=1e-3)
    # 10 semanas con 20 eventos casi idénticos cada una: el IC por semanas es mucho más ancho que el iid
    rng = np.random.default_rng(7)
    sem = np.repeat(np.arange(10), 20)
    x = np.repeat(rng.normal(size=10), 20) + rng.normal(scale=0.01, size=200)
    _, lo_c, hi_c = v.bootstrap_media_cluster(x, sem, n_boot=500)
    _, lo_i, hi_i = v.bootstrap_media(x, n_boot=500)
    assert hi_c - lo_c > 3 * (hi_i - lo_i)


# --------------------------------------------------------------------------- calidad

def test_calidad_huecos_atipicos_y_ruptura():
    dias = pd.bdate_range("2010-01-04", periods=1000)
    rng = np.random.default_rng(0)
    val = rng.normal(size=1000)
    val[600:] += 20                                                    # salto de nivel
    val[300] += 50                                                     # atípico
    keep = np.ones(1000, bool)
    keep[100:200] = False                                              # 10 % de huecos
    d = _larga(dias[keep], dias[keep], val[keep])
    q = v.calidad_serie(d)
    assert q["frecuencia"] == "diaria_habil"
    assert q["pct_huecos"] == pytest.approx(10.0)
    assert q["n_atipicos"] >= 1 and "2011-02" in q["fechas_atipicos"]  # el día 300 (o su vuelta)
    assert q["n_rupturas"] >= 1 and q["fechas_rupturas"].startswith(dias[600].date().isoformat()[:7])
    assert q["n_pub_antes_ref"] == 0


# --------------------------------------------------------------------------- CAR

def test_car_desde_publicacion_con_precios_inventados():
    dias = pd.bdate_range("2020-01-01", periods=100)
    mercado = pd.Series(100.0, index=dias)
    precio = pd.Series(100.0, index=dias)
    precio.loc[dias[10]:] = 110.0                                     # salto tras la operación
    precio.loc[dias[25]:] = 121.0                                     # salto tras la publicación
    # Desde la operación (día 10) se captura el primer salto; desde la publicación (día 20) solo el segundo.
    assert v.car_evento(precio, mercado, dias[20], 20) == pytest.approx(0.10)
    assert v.car_evento(precio, mercado, dias[9], 20) == pytest.approx(0.21)
    assert np.isnan(v.car_evento(precio, mercado, dias[90], 20))      # sin 20 sesiones después
    ev = pd.DataFrame(dict(grupo=["insiders", "insiders"], ticker=["AAA", "ZZZ"],
                           fecha_publicacion=[dias[20], dias[20]], fecha_operacion=[dias[9], dias[9]]))
    res, por_ev = v.car_tabla(ev, {"AAA": precio}, mercado, horizontes=(20,), n_boot=100)
    r = res.set_index("referencia_fecha")
    assert r.loc["publicacion", "car_medio"] == pytest.approx(0.10)
    assert r.loc["operacion", "car_medio"] == pytest.approx(0.21)
    assert r.loc["publicacion", "n_sin_precio"] == 1 and r.loc["publicacion", "n"] == 1
    assert r.loc["publicacion", "pct_sin_precio"] == 0.5 and r.loc["publicacion", "n_semanas"] == 1
    assert set(por_ev.columns) >= {"ticker", "fecha", "grupo", "car_20"}


def test_eventos_compra_filtra_compras_y_quita_nombres():
    cong = pd.DataFrame(dict(representante=["X"] * 3, tipo=["compra", "venta", "compra"], ticker=["brk.b", "AAA", None],
                             fecha_operacion=["2020-01-02"] * 3, fecha_publicacion=["2020-02-03"] * 3))
    ins = pd.DataFrame(dict(insider=["Y", "Y"], codigo=["P", "S"], derivado=[False, False], ticker=["MSFT", "MSFT"],
                            fecha_transaccion=["2020-01-02"] * 2, fecha_publicacion=["2020-01-03"] * 2))
    e = v.eventos_compra(cong, ins)
    assert sorted(e["ticker"]) == ["BRK-B", "MSFT"]
    assert "representante" not in e and "insider" not in e


# --------------------------------------------------------------------------- calibración

def test_brier_y_murphy_conocidos():
    assert v.brier([0.0, 1.0], [0, 1]) == 0.0
    assert v.brier([0.5] * 4, [0, 1, 0, 1]) == 0.25
    assert v.brier([0.8, 0.2], [1, 1]) == pytest.approx((0.04 + 0.64) / 2)
    p = np.r_[np.full(50, 0.15), np.full(50, 0.85)]
    y = np.r_[np.zeros(42), np.ones(8), np.ones(43), np.zeros(7)]
    m = v.murphy(p, y)
    assert m["brier"] == pytest.approx(m["fiabilidad"] - m["resolucion"] + m["incertidumbre"])
    cal = pd.DataFrame(dict(plataforma="pm", p_1d=p, p_7d=p, p_30d=p, resultado=y, volumen_usd=np.arange(100)))
    t, bins = v.calibracion(cal)
    assert bins[(bins["horizonte_dias"] == 7) & (bins["n"] > 0)]["n"].sum() == 100
    assert set(t["liquidez"]) == {"todos", "volumen_alto", "volumen_bajo", "indeciso"}
    # mercados ya decididos (p = 0,01 / 0,99) fuera del grupo indeciso
    p2 = np.r_[np.full(50, 0.01), np.full(30, 0.5), np.full(20, 0.99)]
    y2 = np.r_[np.zeros(50), np.ones(15), np.zeros(15), np.ones(20)]
    t2, _ = v.calibracion(pd.DataFrame(dict(plataforma="pm", p_7d=p2, resultado=y2, volumen_usd=1.0)), horizontes=(7,))
    r = t2.set_index("liquidez")
    assert r.loc["indeciso", "n"] == 30 and r.loc["todos", "n_decididos"] == 70
    assert r.loc["indeciso", "bss"] == pytest.approx(0.0) and r.loc["todos", "bss"] > 0.5


# --------------------------------------------------------------------------- semáforo

def test_semaforo_caso_trivial():
    t = v.semaforo([
        dict(candidato="buena", tipo="serie", historia=2, calidad=2, lookahead=2, senal=2, acceso=2),
        dict(candidato="corta", tipo="serie", historia=0, calidad=2, lookahead=2, senal=None, acceso=2),
        dict(candidato="mira_futuro", tipo="serie", historia=2, calidad=2, lookahead=0, senal=2, acceso=2),
        dict(candidato="debil", tipo="serie", historia=2, calidad=2, lookahead=2, senal=0, acceso=2),
        dict(candidato="texto", tipo="texto", historia=1, calidad=None, lookahead=2, senal=None, acceso=2),
        dict(candidato="corta_clara", tipo="serie", historia=1, calidad=2, lookahead=2, senal=2, acceso=2, n_crisis=3),
    ]).set_index("candidato")
    assert t.loc["buena", "veredicto"] == "apta_backtest"
    assert t.loc["corta", "veredicto"] == "solo_vivo"
    assert t.loc["mira_futuro", "veredicto"] == "descartar"
    assert t.loc["debil", "veredicto"] == "contexto"
    assert t.loc["texto", "veredicto"] == "contexto"
    assert t.loc["corta_clara", "veredicto"] == "solo_vivo" and "3 crisis" in t.loc["corta_clara", "motivo"]
    sin = v.semaforo([dict(candidato="x", tipo="serie", historia=2, calidad=2, lookahead=2, senal=None, acceso=2,
                           sin_senal="sin signo esperado a priori (contexto)")]).iloc[0]
    assert sin["veredicto"] == "contexto" and "sin signo" in sin["motivo"]
    assert t["motivo"].str.len().gt(10).all()


def test_signo_esperado_y_senal_en_el_sentido_esperado():
    assert v.signo_esperado("vix_cierre") == 1 and v.signo_esperado("gdelt_tono:economy") == -1
    assert v.signo_esperado("pc_equity_calls") is None and v.signo_esperado("n10d") is None
    a = pd.DataFrame(dict(adelanto=[0, 20, 60], auc=0.30, auc_mismo_periodo=[0.30, 0.35, 0.37], auc_lo=0.2,
                          auc_hi=0.4, auc_vix_mismo_periodo=0.8, n_crisis=12))
    # AUC inversa en una serie que debería subir (EPU): no cuenta como señal
    s, met, _ = v.senal_serie("gepu_current", a, None)
    assert s == 0 and "contrario" in met
    # la misma AUC inversa en el tono de noticias (signo −) sí cuenta
    s, _, _ = v.senal_serie("gdelt_tono:economy", a, None)
    assert s == 2
    # pero no si el estudio de eventos va en contra del signo esperado
    ev = pd.DataFrame(dict(serie="gdelt_tono:economy", dia_rel=range(0, 21), media=0.8, n_crisis=6))
    s, met, _ = v.senal_serie("gdelt_tono:economy", a, ev)
    assert s == 0 and "signo contrario" in met
    # IC por bloques con el 0,5: sin señal
    s, _, _ = v.senal_serie("gdelt_tono:economy", a.assign(auc_hi=0.55), None)
    assert s == 0
    s, _, motivo = v.senal_serie("n10d", a, None)
    assert s is None and "sin signo" in motivo


def test_candidatos_evento_car_dos_horizontes_y_supervivencia():
    ops = pd.DataFrame(dict(fecha_publicacion=["2020-06-01", "2025-06-01"]))
    car = pd.DataFrame(dict(grupo="congreso", referencia_fecha="publicacion", horizonte=[5, 20, 60], n=141,
                            car_medio=[-0.004, -0.012, -0.001], lo=[-0.01, -0.02, -0.02], hi=[0.001, -0.001, 0.02],
                            n_eventos=152, n_sin_precio=10, pct_sin_precio=10 / 152))
    tabs = {"congreso_operaciones": ops, "insiders_operaciones": None}
    crisis = {"a": ("2022-01-03", "2022-10-12"), "b": ("2023-02-02", "2023-03-13"), "c": ("2008-01-01", "2009-01-01")}
    la = pd.DataFrame(columns=["fuente", "comprobacion"])
    c = v._candidatos_evento(pd.DataFrame(), car, la, tabs, crisis, {})[0]
    assert c["senal"] == 0 and c["n_crisis"] == 2 and c["inicio"] == "2020-06-01"   # un solo horizonte excluye el 0
    car2 = car.assign(lo=[-0.01, -0.02, -0.03], hi=[-0.001, -0.001, -0.002])
    assert v._candidatos_evento(pd.DataFrame(), car2, la, tabs, crisis, {})[0]["senal"] == 2
    car3 = car2.assign(n_sin_precio=60, pct_sin_precio=60 / 152)
    c3 = v._candidatos_evento(pd.DataFrame(), car3, la, tabs, crisis, {})[0]
    assert c3["senal"] is None and "supervivencia" in c3["sin_senal"]


def test_crisis_en_rango():
    assert v.crisis_en_rango("2002-01-01", "2002-12-31", CRISIS) == 0
    assert v.crisis_en_rango("2001-04-01", "2003-06-02", CRISIS) == 2
    assert v.crisis_en_rango(None, None, CRISIS) == 0


# --------------------------------------------------------------------------- look-ahead

def test_lookahead_series_detecta_publicacion_anterior(sin_red):
    ok = _larga(["2020-01-01"], ["2020-01-02"], [1.0], "a")
    mal = _larga(["2020-01-05"], ["2020-01-02"], [1.0], "b")
    mal["metodo_publicacion"] = base.REVISABLE
    t = pd.DataFrame(v.lookahead_series(pd.concat([ok, mal])))
    r = t.set_index(["fuente", "comprobacion"])["resultado"]
    assert r[("a", "publicacion_no_anterior_a_referencia")] == "ok"
    assert r[("b", "publicacion_no_anterior_a_referencia")] == "fallo"
    assert r[("b", "valor_revisado_vintage_actual")] == "riesgo"


def test_cargar_series_desde_corpus_sintetico(sin_red):
    dias = pd.bdate_range("2019-01-01", periods=400)
    base.guardar_serie(_larga(dias, dias + pd.offsets.BDay(1), np.arange(400.0), "s1"), "s1")
    base.guardar_serie(_larga(dias[:5], dias[:5], np.arange(5.0), "s1"), "s1", rapido=True)  # se ignora
    rej, meta = v.cargar_series(dias=pd.bdate_range("2019-01-01", periods=410))
    assert list(rej.columns) == ["s1"] and meta.loc[0, "fichero"] == "s1"
    assert np.isnan(rej["s1"].iloc[0]) and rej["s1"].iloc[1] == 0.0


def test_conteo_sin_ceros_rellena_con_cero():
    dias = pd.bdate_range("2020-01-01", periods=60)
    con = dias[np.arange(60) % 5 != 4]                                  # sin eventos uno de cada cinco días
    larga = _larga(con, con, np.ones(len(con)) * 2, "ptr")
    assert v.es_conteo_sin_ceros(larga)
    s = v.a_rejilla(larga, dias)
    assert s.loc[dias[3]] == 2.0 and s.loc[dias[4]] == 0.0             # sin fila = 0, no arrastre
    assert v.calidad_serie(larga)["conteo_sin_ceros"]
    assert not v.es_conteo_sin_ceros(_larga(dias, dias, np.arange(60.0) + 0.5, "nivel"))
