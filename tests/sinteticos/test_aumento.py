"""Tests del aumento real + sintetico (regimenes.sinteticos.aumento, notebook 20)."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from regimenes import rutas
from regimenes.sinteticos.aumento import analisis as an_
from regimenes.sinteticos.aumento import anclas as an
from regimenes.sinteticos.aumento import muestras as mu
from regimenes.sinteticos.aumento.config import config_aumento, detectores_pista, fuentes
from regimenes.sinteticos.aumento.envoltorio import DetectorAumentado, fabrica, fechas_previas

HAY_BENCH = (rutas.RESULTS_BENCHMARK / "panels" / "B_D01.parquet").exists()
HAY_PROCESSED_B = (rutas.DATA_PROCESSED / "pistaB_diaria.parquet").exists()


# --------------------------------------------------------------------------- configuracion
def test_config_pre_registro():
    cfg = config_aumento()
    assert fuentes(cfg) == ["gaussiano_regimen", "var_regimen", "garch_regimen", "rbig", "flow_matching",
                            "bootstrap_regimen"]
    assert len(detectores_pista("A", cfg)) == 12
    assert {"D02", "D10"}.isdisjoint(detectores_pista("B", cfg)) and len(detectores_pista("B", cfg)) == 10
    assert cfg["proporciones"] == [0.25, 0.5, 1.0]
    assert cfg["contrastes"]["medida"] == "exceso_sobre_nulo"


def test_numero_de_trabajos_y_lotes():
    from regimenes.sinteticos.aumento.ejecucion import Trabajo, lote, trabajos

    cfg = config_aumento()
    lista = trabajos(cfg=cfg)
    n_a = 12 * 1 * 6 * 2 * 3 + 2 * 2
    n_b = 10 * 2 * 6 * 2 * 3 + 5 * 2
    assert len(lista) == n_a + n_b
    assert len({t.clave + t.pista for t in lista}) == len(lista)
    assert lote(Trabajo("B", "D01", "rbig", "crisis", 0.5, 0), cfg) == 0
    assert lote(Trabajo("B", "D01", "rbig", "crisis", 0.5, 1), cfg) == 7
    assert lote(Trabajo("A", "D05", "rbig", "crisis", 0.5, 0), cfg) == 3
    assert lote(Trabajo("A", "D09", "rbig", "completas", 1.0, 0), cfg) == 4
    assert lote(Trabajo("A", "D09", "rbig", "crisis", 1.0, 0), cfg) == 5
    assert lote(Trabajo("A", "D03", "ninguna", "-", 0.0, 0, 7), cfg) == 1


# --------------------------------------------------------------------------- anclas
def test_candidatas_ultima_sesion_de_cada_ano():
    idx = pd.bdate_range("2000-03-01", "2002-06-30")
    c = an.candidatas(idx)
    assert list(c.strftime("%Y-%m-%d")) == ["2000-12-29", "2001-12-31", "2002-06-28"]


def test_corte_ajuste_recorta_ventana_abierta_y_reconoce_cerradas():
    fin, reconocidas, abierta = an.corte_ajuste("A", pd.Timestamp("1968-12-31"))
    assert fin == pd.Timestamp("1968-11-28") and abierta == "bear_1969_70"
    assert reconocidas == ["credit_crunch_1966"]
    fin, reconocidas, abierta = an.corte_ajuste("A", pd.Timestamp("2009-12-31"))
    assert fin == pd.Timestamp("2009-12-31") and abierta is None and "gfc_2007_09" in reconocidas


def test_limite_y_ancla_para():
    idx = pd.bdate_range("2000-01-03", periods=300)
    lim = an.limite_reconocimiento(idx, 126)
    assert lim == idx[-127]
    assert an.limite_reconocimiento(idx[:100], 126) is None
    validas = pd.DatetimeIndex(["2000-03-31", "2000-06-30", "2001-01-31"])
    assert an.ancla_para(validas, lim) == pd.Timestamp("2000-06-30")
    assert an.ancla_para(validas, pd.Timestamp("2000-01-10")) is None
    assert an.ancla_para(pd.DatetimeIndex([]), lim) is None


@pytest.mark.datos
@pytest.mark.parametrize("pista", ["A", "B"])
def test_ningun_pliegue_usa_crisis_no_reconocidas(pista):
    """Con TODAS las candidatas como validas: tau <= t - margen y el ajuste no toca ventanas abiertas."""
    from regimenes.benchmark.ejecucion import DEFAULT_TRAIN_DAYS, common_track_index
    from regimenes.sinteticos.datos import ventanas_crisis

    try:
        indice = common_track_index(pista)
    except FileNotFoundError:
        pytest.skip("sin data/processed")
    margen = int(config_aumento()["anclas"]["margen_reconocimiento"])
    validas = an.candidatas(indice)
    for _, pl in mu.pliegues(len(indice), DEFAULT_TRAIN_DAYS[pista], 21, True).iterrows():
        idx = indice[: int(pl["t"])]
        lim = an.limite_reconocimiento(idx, margen)
        tau = an.ancla_para(validas, lim)
        if tau is None:
            continue
        assert tau <= lim <= idx[-1]
        fin, reconocidas, _ = an.corte_ajuste(pista, tau)
        for nombre, (pico, suelo) in ventanas_crisis(pista).items():
            if nombre in reconocidas:
                assert suelo <= lim             # reconocida al menos `margen` sesiones antes de t
            else:
                assert fin < pico               # ningun dia de una crisis no reconocida entra en el ajuste


# --------------------------------------------------------------------------- muestras
def test_bases_y_mascara():
    idx = pd.bdate_range("2007-09-03", "2010-12-31")
    n_crisis = int(mu.mascara_reconocidas(idx, "B", "2009-12-31").sum())
    assert n_crisis == len(pd.bdate_range("2007-10-09", "2009-03-09"))
    assert mu.base_proporcion("crisis", idx, "B", "2008-12-31") == 0   # la GFC aun no esta reconocida
    assert mu.base_proporcion("completas", idx, "B", "2009-12-31") == len(idx)
    assert mu.n_sintetico(0.25, 10) == 2 and mu.n_sintetico(1.0, 10) == 10
    with pytest.raises(ValueError):
        mu.base_proporcion("otro", idx, "B", "2009-12-31")


def test_pliegues_misma_aritmetica_que_walk_forward():
    p = mu.pliegues(100, 40, 21, True)
    assert p["t"].tolist() == [40, 61, 82] and (p["lo"] == 0).all()
    q = mu.pliegues(100, 40, 21, False)
    assert q["lo"].tolist() == [0, 21, 42]


def _reservorio_falso(tmp_path, n=500):
    tau = pd.Timestamp("2009-12-31")
    ruta = mu.ruta_reservorio("B", "falso", tau, "completas", 0, tmp_path)
    ruta.parent.mkdir(parents=True)
    rng = np.random.default_rng(0)
    tabla = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                          "regime": rng.integers(0, 2, n), mu.COL_TRAY: np.repeat(np.arange(5), n // 5)})
    tabla.to_parquet(ruta)
    return tau, tabla


def test_proveedor_bloques_anidados_y_registro(tmp_path):
    tau, tabla = _reservorio_falso(tmp_path)
    idx = pd.bdate_range("2010-01-04", periods=400)
    bloques = {}
    for p in (0.25, 0.5, 1.0):
        prov = mu.Proveedor("B", "falso", "completas", p, 0, pd.DatetimeIndex([tau]), 126, tmp_path)
        bloques[p] = prov(idx)
        assert prov.registro_pliegues[-1]["n_sint"] == round(p * 400)
        assert prov.registro_pliegues[-1]["tau"] == tau
    assert bloques[0.25].equals(bloques[1.0].iloc[:100])
    assert bloques[0.5].equals(bloques[1.0].iloc[:200])
    assert mu.COL_TRAY not in bloques[1.0]
    # sin ancla valida todavia: sin bloque
    prov = mu.Proveedor("B", "falso", "completas", 1.0, 0, pd.DatetimeIndex([tau]), 126, tmp_path)
    assert prov(pd.bdate_range("2009-06-01", periods=200)) is None
    # reservorio corto: error explicito
    prov = mu.Proveedor("B", "falso", "completas", 1.0, 0, pd.DatetimeIndex([tau]), 126, tmp_path)
    with pytest.raises(RuntimeError):
        prov(pd.bdate_range("2010-01-04", periods=800))


# --------------------------------------------------------------------------- envoltorio
class _Juguete:
    """Detector minimo que registra lo que recibe."""

    name = "juguete"
    n_states = 2
    crisis_state = 1

    def __init__(self):
        self.vistos = []

    def fit(self, X):
        self.vistos.append(X)
        self.umbral_ = float(X["a"].mean())
        return self

    def label_states_economically(self, X, market_returns=None):
        self.etiquetado_con_ = X
        return self

    def predict(self, X):
        return (X["a"].to_numpy() > self.umbral_).astype(int)

    predict_online = predict

    def predict_proba(self, X):
        s = self.predict(X)
        return np.stack([1 - s, s], axis=1).astype(float)


def test_fechas_previas():
    f = fechas_previas(pd.Timestamp("2010-01-04"), 10)
    assert len(f) == 10 and f.is_monotonic_increasing and f[-1] == pd.Timestamp("2010-01-01")


def test_envoltorio_sin_bloque_pasa_el_mismo_objeto():
    X = pd.DataFrame({"a": np.arange(10.0)}, index=pd.bdate_range("2010-01-04", periods=10))
    interno = _Juguete()
    det = DetectorAumentado(interno, lambda idx: None)
    det.fit(X)
    assert interno.vistos[0] is X
    det.label_states_economically(X, X["a"])
    assert interno.etiquetado_con_ is X
    assert det.n_states == 2 and det.crisis_state == 1
    np.testing.assert_array_equal(det.predict_online(X), interno.predict(X))


def test_envoltorio_antepone_bloque_con_columnas_del_train():
    X = pd.DataFrame({"a": np.ones(5)}, index=pd.bdate_range("2010-01-04", periods=5))
    bloque = pd.DataFrame({"b": np.zeros(3), "a": np.full(3, 5.0), "regime": [1, 0, 1]})
    interno = _Juguete()
    ocup: list = []
    det = DetectorAumentado(interno, lambda idx: bloque, ocupacion=ocup, pliegue_ocupacion=True)
    det.fit(X)
    visto = interno.vistos[0]
    assert list(visto.columns) == ["a"] and len(visto) == 8
    assert (visto.index[:3] < X.index[0]).all() and visto.index.is_monotonic_increasing
    assert visto["a"].iloc[:3].eq(5.0).all() and visto.iloc[3:].equals(X)
    det.label_states_economically(X, X["a"])
    assert interno.etiquetado_con_ is X          # el orden economico solo ve el real
    assert ocup and ocup[0]["n_sint"] == 3 and ocup[0]["crisis_en_estado_crisis_sint"] == 1.0


def test_fabrica_crea_uno_nuevo_por_pliegue_y_marca_ocupacion():
    class Spec:
        def factory(self):
            return _Juguete()

    crear = fabrica(Spec(), lambda idx: None, ocupacion=[], cada_ocupacion=2)
    dets = [crear() for _ in range(4)]
    assert len({id(d.interno) for d in dets}) == 4
    assert [d.pliegue_ocupacion for d in dets] == [True, False, True, False]


# --------------------------------------------------------------------------- analisis
def _toy():
    idx = pd.bdate_range("2000-01-03", periods=600)
    ventanas = {"c1": ("2000-03-01", "2000-04-14"), "c2": ("2000-09-01", "2000-10-31"),
                "c3": ("2001-05-01", "2001-05-31"), "fuera": ("1999-01-01", "1999-02-01")}
    return idx, ventanas


def test_segmentos_parten_por_puntos_medios():
    idx, ventanas = _toy()
    seg = an_.segmentos(idx, ventanas)
    assert seg.min() == 0 and seg.max() == 2 and np.all(np.diff(seg) >= 0)
    for k, (a, b) in enumerate(list(an_.ventanas_evaluadas(idx, ventanas).values())):
        m = (idx >= a) & (idx <= b)
        assert (seg[m] == k).all()


@pytest.mark.parametrize("semilla_", [0, 1, 2])
def test_estadisticos_reproducen_detection_summary(semilla_):
    from regimenes.evaluacion.metricas import detection_summary
    from regimenes.evaluacion.ranking import detection_score

    idx, ventanas = _toy()
    rng = np.random.default_rng(semilla_)
    flags = an_._flags_markov(len(idx), 0.3, 8.0, rng)
    seg = an_.segmentos(idx, ventanas)
    st = an_.estadisticos_segmento(flags, idx, ventanas, seg).sum(0)
    ref = detection_summary(pd.Series(flags, index=idx), ventanas)
    assert st[1] / st[0] == pytest.approx(ref["det_marked_rate"])
    assert st[2] / st[1] == pytest.approx(ref["det_precision"])
    assert st[5] / st[4] == pytest.approx(ref["det_event_recall"])
    esperado = float(detection_score(np.float64(ref["det_precision"]), np.float64(ref["det_event_recall"])))
    assert an_.score_desde(st) == pytest.approx(esperado)


def test_score_rapido_igual_que_el_completo():
    idx, ventanas = _toy()
    seg = an_.segmentos(idx, ventanas)
    ev = an_.ventanas_evaluadas(idx, ventanas)
    dentro = np.zeros(len(idx), bool)
    cortes = []
    for a, b in ev.values():
        m = (idx >= a) & (idx <= b)
        dentro |= m
        i = np.flatnonzero(m)
        cortes.append((i[0], i[-1] + 1))
    rng = np.random.default_rng(3)
    for _ in range(20):
        f = an_._flags_markov(len(idx), rng.uniform(0.05, 0.6), rng.uniform(1, 40), rng)
        st = an_.estadisticos_segmento(f, idx, ventanas, seg).sum(0)
        assert an_._score_rapido(f, dentro, cortes, 3) == pytest.approx(float(an_.score_desde(st)))


def test_flags_markov_misma_ley_que_el_nulo_del_ranking():
    from regimenes.evaluacion.ranking import persistent_random_null

    idx, ventanas = _toy()
    ref = persistent_random_null(idx, {k: v for k, v in ventanas.items() if k != "fuera"},
                                 rate=0.25, mean_run=12.0, n_sims=400, seed=1)["score_deteccion"]
    tabla = an_.tabla_nulo(idx, ventanas, n_sims=400, tasas=[0.25], rachas=[12.0])
    assert tabla["mediana"].iloc[0] == pytest.approx(float(ref.median()), abs=0.03)
    rng = np.random.default_rng(0)
    f = np.concatenate([an_._flags_markov(20000, 0.25, 12.0, rng) for _ in range(5)])
    assert f.mean() == pytest.approx(0.25, abs=0.02)


def test_permutacion_exacta_casos_limite():
    idx, ventanas = _toy()
    tabla = an_.tabla_nulo(idx, ventanas, n_sims=20, tasas=np.linspace(0, 1, 6), rachas=[1.0, 10.0, 100.0])
    nulo = an_.Nulo(tabla)
    seg = an_.segmentos(idx, ventanas)
    rng = np.random.default_rng(5)
    a = an_.estadisticos_segmento(an_._flags_markov(len(idx), 0.3, 10, rng), idx, ventanas, seg)
    t, p = an_.permutacion_exacta(a, a, nulo)
    assert t == 0 and p == 1.0
    # senal perfecta frente a senal vacia: p minimo = 2 / 2^S
    perfecta = np.zeros(len(idx), bool)
    for x, y in an_.ventanas_evaluadas(idx, ventanas).values():
        perfecta |= (idx >= x) & (idx <= y)
    b = an_.estadisticos_segmento(perfecta, idx, ventanas, seg)
    c = an_.estadisticos_segmento(np.zeros(len(idx), bool), idx, ventanas, seg)
    t, p = an_.permutacion_exacta(b, c, nulo)
    assert t > 0 and p == pytest.approx(2 / 2 ** 3)
    t2, p2 = an_.permutacion_exacta(c, b, nulo)
    assert t2 == pytest.approx(-t) and p2 == p


def test_holm():
    assert an_.holm([0.01, 0.04, 0.03, 0.2], 0.05).tolist() == [True, False, False, False]
    assert an_.holm([0.001, 0.01, 0.02], 0.05).tolist() == [True, True, True]


def test_nivel_binario():
    f = np.zeros(200, bool)
    assert an_.nivel_binario(f, 0.2, np.nan) == 3
    f[::2] = True                                  # parpadeo
    assert an_.nivel_binario(f, 0.2, 0.5) == 2
    g = np.zeros(200, bool)
    g[50:80] = True
    assert an_.nivel_binario(g, 0.2, 0.1) == 1     # precision por debajo de la tasa base
    assert an_.nivel_binario(g, 0.2, 0.6) == 0


# --------------------------------------------------------------------------- integracion
@pytest.mark.datos
@pytest.mark.resultados
@pytest.mark.skipif(not HAY_BENCH, reason="sin paneles del benchmark locales")
def test_proporcion_cero_reproduce_benchmark_bit_a_bit():
    """Con un hilo BLAS (como el benchmark), el envoltorio sin bloque reproduce B/D01 y B/D07."""
    codigo = (
        "import numpy as np\n"
        "from regimenes.sinteticos.aumento import ejecucion as ej\n"
        "for det in ('D01', 'D07'):\n"
        "    spec = ej.spec_de('B', det); X, mr, ts = ej.datos_trabajo('B', spec)\n"
        "    wf = ej.correr_walk_forward(spec, X, mr, ts); ref = ej.panel_benchmark('B', det)\n"
        "    assert wf.index.equals(ref.index)\n"
        "    assert np.array_equal(wf['state'].to_numpy(), ref['state'].to_numpy())\n"
        "    assert np.array_equal(wf['p_crisis'].to_numpy(), ref['p_crisis'].to_numpy())\n"
        "print('ok')\n"
    )
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    res = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, env=env, timeout=600)
    assert res.returncode == 0 and "ok" in res.stdout, res.stderr[-2000:]


@pytest.mark.datos
@pytest.mark.resultados
@pytest.mark.skipif(not HAY_BENCH, reason="sin paneles del benchmark locales")
def test_ultimos_pliegues_reproducen_el_benchmark():
    from regimenes.sinteticos.aumento import ejecucion as ej

    for det in ("D11", "D05"):                      # ventana movil y step 63
        spec = ej.spec_de("B", det)
        X, mr, ts = ej.datos_trabajo("B", spec)
        Xp, mrp, t0 = ej.ultimos_pliegues(spec, X, mr, ts, 2)
        wf = ej.correr_walk_forward(spec, Xp, mrp, t0)
        ref = ej.panel_benchmark("B", det).reindex(wf.index)
        assert np.array_equal(wf["state"].to_numpy(), ref["state"].to_numpy())


def test_huella_cambia_con_el_trabajo(monkeypatch):
    from regimenes.sinteticos.aumento import ejecucion as ej

    monkeypatch.setattr(ej, "huella_benchmark", lambda p, d: "fija")
    a = ej.Trabajo("A", "D03", "ninguna", "-", 0.0, 0, 7)
    b = ej.Trabajo("A", "D03", "ninguna", "-", 0.0, 0, 1234)
    assert ej.huella(a) == ej.huella(a) and ej.huella(a) != ej.huella(b)
    assert ej.spec_de("A", "D03", 7).params["random_state"] == 7
    with pytest.raises(ValueError):
        ej.spec_de("A", "D01", 7)                   # D01 no es estocastico


# --------------------------------------------------------------------------- añadidos tras la revision
def test_agregar_da_la_racha_media_exacta():
    from regimenes.evaluacion.metricas import detection_summary

    idx, ventanas = _toy()
    seg = an_.segmentos(idx, ventanas)
    rng = np.random.default_rng(11)
    for _ in range(10):
        f = an_._flags_markov(len(idx), 0.4, 60.0, rng)          # rachas largas: cruzan fronteras
        st = an_.agregar(an_.estadisticos_segmento(f, idx, ventanas, seg))
        ref = detection_summary(pd.Series(f, index=idx), ventanas)
        assert st[1] / st[3] == pytest.approx(ref["det_mean_crisis_run"])


def test_reservorio_se_alarga_sin_cambiar_el_principio_y_se_regenera_si_cambia_la_huella(tmp_path):
    from regimenes.sinteticos.registry import crear

    idx = pd.bdate_range("2000-01-03", periods=600)
    rng = np.random.default_rng(0)
    reg = pd.Series((np.arange(600) % 200 > 150).astype(int), index=idx)
    panel = pd.DataFrame({"a": rng.normal(size=600) * (1 + reg), "b": rng.normal(size=600)}, index=idx)
    gen = crear("gaussiano_regimen").fit(panel, reg)
    tau = "2002-04-30"
    n1 = mu.asegurar_reservorio("B", "gaussiano_regimen", tau, "completas", 0, 3000, gen=gen, base=tmp_path)
    ruta = mu.ruta_reservorio("B", "gaussiano_regimen", tau, "completas", 0, tmp_path)
    corto = pd.read_parquet(ruta)
    n2 = mu.asegurar_reservorio("B", "gaussiano_regimen", tau, "completas", 0, 6000, gen=gen, base=tmp_path)
    largo = pd.read_parquet(ruta)
    assert n1 >= 3000 and n2 >= 6000 and largo.iloc[:len(corto)].equals(corto)
    ruta.with_suffix(".huella").write_text("otra", encoding="utf-8")
    n3 = mu.asegurar_reservorio("B", "gaussiano_regimen", tau, "completas", 0, 1000, gen=gen, base=tmp_path)
    assert n3 == 2520                                  # regenerado de cero: una sola trayectoria
    assert pd.read_parquet(ruta).equals(largo.iloc[:2520].reset_index(drop=True))


def test_h1_familia_fija_cuenta_los_ausentes():
    idx, ventanas = _toy()
    seg = an_.segmentos(idx, ventanas)
    nulo = an_.Nulo(an_.tabla_nulo(idx, ventanas, n_sims=10, tasas=np.linspace(0, 1, 6), rachas=[1.0, 50.0]))
    rng = np.random.default_rng(2)
    st = {"D01__base": an_.estadisticos_segmento(an_._flags_markov(len(idx), 0.2, 20, rng), idx, ventanas, seg),
          "D01__voto": an_.estadisticos_segmento(an_._flags_markov(len(idx), 0.2, 20, rng), idx, ventanas, seg)}
    senales = pd.DataFrame([
        {"senal": "D01__base", "detector": "D01", "tipo": "base", "fuente": None, "brazo": None,
         "proporcion": np.nan, "rep": np.nan, "nivel": 0},
        {"senal": "D01__voto", "detector": "D01", "tipo": "voto", "fuente": "voto", "brazo": "completas",
         "proporcion": 1.0, "rep": 0, "nivel": 0}])
    d = an_.DatosPista("B", senales, st, nulo, an_.replicas_bootstrap(3, 200, 0), ["c1", "c2", "c3"])
    t = an_.h1({"B": d})
    assert len(t) == 10 and (t["veredicto"] == "falta").sum() == 9
    # errata 2026-10-15: Holm por pista, con el global como sensibilidad al lado
    assert config_aumento()["contrastes"]["familia_holm"] == "pista"
    assert {"holm", "holm_global", "veredicto_global"} <= set(t.columns)
    assert t.set_index("detector").loc["D02" if "D02" in t["detector"].values else "D03", "p"] == 1.0
    assert t.set_index("detector").loc["D01", "veredicto"] in {"sin efecto detectable", "empeora", "mejora"}


@pytest.mark.datos
@pytest.mark.resultados
@pytest.mark.skipif(not HAY_BENCH, reason="sin paneles del benchmark locales")
def test_un_hilo_reproduce_d03_sin_variables_de_entorno():
    """``un_hilo`` limita BLAS en caliente: B/D03 (sensible a hilos) sale bit a bit sin OMP_NUM_THREADS."""
    codigo = (
        "import numpy as np\n"
        "from regimenes.sinteticos.aumento import ejecucion as ej\n"
        "from regimenes.sinteticos.aumento.config import un_hilo\n"
        "spec = ej.spec_de('B', 'D03'); X, mr, ts = ej.datos_trabajo('B', spec)\n"
        "with un_hilo():\n"
        "    wf = ej.correr_walk_forward(spec, X, mr, ts)\n"
        "ref = ej.panel_benchmark('B', 'D03')\n"
        "assert np.array_equal(wf['p_crisis'].to_numpy(), ref['p_crisis'].to_numpy())\n"
        "print('ok')\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                            "OPENBLAS_NUM_THREADS")}
    res = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, env=env, timeout=900)
    assert res.returncode == 0 and "ok" in res.stdout, res.stderr[-2000:]


# --------------------------------------------------------------------------- lecturas y figuras del notebook
def _datos_juguete(pista: str = "B", detectores=("D01", "D04"), reps: int = 2) -> an_.DatosPista:
    """DatosPista con base, cada fuente, voto y suelo para todas las proporciones y brazos."""
    cfg = config_aumento()
    idx, ventanas = _toy()
    seg = an_.segmentos(idx, ventanas)
    nulo = an_.Nulo(an_.tabla_nulo(idx, ventanas, n_sims=10, tasas=np.linspace(0, 1, 6), rachas=[1.0, 50.0]))
    rng = np.random.default_rng(7)
    st, filas = {}, []

    def nueva(clave, **meta):
        f = an_._flags_markov(len(idx), rng.uniform(0.1, 0.3), 20, rng)
        st[clave] = an_.estadisticos_segmento(f, idx, ventanas, seg)
        filas.append({"senal": clave, "detector": None, "tipo": None, "fuente": None, "brazo": None,
                      "proporcion": np.nan, "rep": np.nan, "nivel": 0, "tasa": f.mean(),
                      "precision": rng.uniform(), "recall": rng.uniform(), "score": rng.uniform(), **meta})

    for det in detectores:
        nueva(f"{det}__base", detector=det, tipo="base")
        for s in (7, 1234):
            nueva(f"{det}__suelo__s{s}", detector=det, tipo="suelo", fuente="ninguna", brazo="-", proporcion=0.0,
                  rep=0)
        for rep in range(reps):
            for brazo in ("completas", "crisis"):
                for p in cfg["proporciones"]:
                    for f in [*cfg["generadores"], cfg["control"]]:
                        tipo = "control" if f == cfg["control"] else "fuente"
                        nueva(f"{det}__{f}__{brazo}__p{p:.2f}__r{rep}", detector=det, tipo=tipo, fuente=f,
                              brazo=brazo, proporcion=float(p), rep=rep)
                    nueva(f"{det}__voto__{brazo}__p{p:.2f}__r{rep}", detector=det, tipo="voto", fuente="voto",
                          brazo=brazo, proporcion=float(p), rep=rep)
    return an_.DatosPista(pista, pd.DataFrame(filas), st, nulo, an_.replicas_bootstrap(3, 200, 0), ["c1", "c2", "c3"])


def test_cobertura_por_lote(tmp_path):
    from regimenes.sinteticos.aumento.ejecucion import trabajos

    lista = trabajos(["B"])
    pd.DataFrame([{"pista": "B", "clave": t.clave} for t in lista[:3]]).to_csv(tmp_path / "filas_B.csv",
                                                                                 index=False)
    cob = an_.cobertura(["B"], tmp_path, horas=False)
    assert cob["trabajos"].sum() == len(lista) and cob["hechos"].sum() == 3
    assert (cob["pendientes"] == cob["trabajos"] - cob["hechos"]).all()
    assert cob.loc[cob["lote"] == 0, "contenido"].iloc[0] == "pista B | repeticion 0"
    vacia = an_.cobertura(["B"], tmp_path / "no_existe", horas=False)
    assert vacia["hechos"].sum() == 0


def test_categoria_y_resumen_anclas():
    reg = pd.DataFrame({"fuente": ["g"] * 5, "tau": pd.to_datetime(["2001-12-31", "2002-12-31", "2003-12-31",
                                                                       "2004-12-31", "2005-12-31"]),
                        "valida": [False, False, False, True, False],
                        "motivo": ["pocas crisis en el ajuste", "no pasa memorizacion y condicionamiento",
                                   "no pasa condicionamiento", np.nan, "error: ValueError: x"]})
    assert an_.categoria_ancla(reg).tolist() == ["pocas crisis", "no pasa ambas", "no pasa condicionamiento",
                                                 "valida", "error"]
    res = an_.resumen_anclas({"B": reg}).loc[("B", "g")]
    assert res["candidatas"] == 5 and res["valida"] == 1 and str(res["primera_valida"]) == "2004-12-31"


def test_lecturas_del_notebook_con_datos_de_juguete():
    cfg = config_aumento()
    d = _datos_juguete()
    datos = {"B": d}
    mapa = an_.mapa_deltas(datos, cfg=cfg)
    assert len(mapa) == 2 * (len(cfg["generadores"]) + 2) and mapa["delta"].notna().all()
    fila = mapa[(mapa["detector"] == "D04") & (mapa["fuente"] == "voto")]["delta"].iloc[0]
    assert fila == pytest.approx(d.delta("D04__voto__completas__p1.00__r0", "D04__base"))
    dosis = an_.dosis_respuesta(datos, cfg)
    assert (dosis.loc[dosis["p"] == 0, "delta"] == 0).all()
    media = dosis[(dosis["detector"] == "(media)") & (dosis["brazo"] == "completas") & (dosis["grupo"] == "umbral")]
    assert len(media) == 4 and (media["ic_inf"] <= media["ic_sup"]).all()
    dm = an_.deltas_metricas(datos)
    assert len(dm) == len(d.senales) - 2 and np.isfinite(dm["d_exceso"]).all()
    rep = an_.replica(datos, cfg)
    assert len(rep) == 2 * 2 * len(cfg["proporciones"])
    suelo = an_.suelo_ruido(datos, cfg)
    assert set(suelo["detector"]) == set(cfg["suelo_ruido"]["detectores"]["B"])
    assert suelo.set_index("detector").loc["D04", "suelo"] > 0
    assert np.isnan(suelo.set_index("detector").loc["D03", "suelo"])
    t2 = an_.h2(datos, cfg)
    tabla = an_.con_ic(t2, "pendiente", columnas="brazo")
    assert tabla.shape == (2, 2) and tabla.iloc[0, 0].count("[") == 1
    assert an_.con_ic(pd.DataFrame(), "pendiente").empty


def test_ocupacion_y_regla_19_sin_h3():
    filas = pd.DataFrame({"pista": ["B"] * 3, "detector": ["D04"] * 3, "fuente": ["rbig", "rbig", "ninguna"],
                          "brazo": ["completas", "crisis", "-"], "proporcion": [1.0, 1.0, 0.0], "rep": [0, 0, 0],
                          "pliegues_estado_solo_sint": [2, 0, np.nan], "pliegues_ocupacion": [10, 10, np.nan],
                          "ocup_crisis_sint_media": [0.3, 0.9, np.nan], "ocup_crisis_real_media": [0.2, 0.2, np.nan]})
    o = an_.ocupacion(filas)
    assert o["frac_solo_sint"].tolist() == [0.2, 0.0]
    assert an_.ocupacion(pd.DataFrame()).empty
    t1 = pd.DataFrame({"pista": ["B"], "detector": ["D04"], "veredicto": ["mejora"]})
    assert not an_.regla_19(t1, pd.DataFrame())["adoptar"].any()


def test_figuras_con_tablas_vacias_y_de_juguete():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from regimenes.sinteticos.aumento import figuras as fg

    cfg = config_aumento()
    assert fg.etiqueta("D04") == "D4 hmm-g" and fg.etiqueta("D12") == "D12 ae"
    vacio = pd.DataFrame()
    for fig in (fg.fig_h1(vacio, cfg), fg.fig_mapa(an_.mapa_deltas({}), cfg), fg.fig_dosis(an_.dosis_respuesta({}), cfg),
                fg.fig_contrastes(vacio, vacio, vacio, cfg),
                fg.fig_diagnosticos(an_.deltas_metricas({}), an_.ocupacion(vacio), cfg),
                fg.fig_robustez(vacio, vacio, an_.suelo_ruido({}), an_.replica({}))):
        plt.close(fig)
    datos = {"B": _datos_juguete()}
    t1 = an_.h1(datos, cfg)
    t3, t4, t5 = an_.h3(datos, cfg), an_.h4(datos, cfg), an_.h5(datos, cfg)
    assert {"a_holm", "b_minima", "c_nivel", "d_generadores", "e_suelo"} <= set(t1.columns)
    for fig in (fg.fig_h1(t1, cfg), fg.fig_mapa(an_.mapa_deltas(datos, cfg=cfg), cfg),
                fg.fig_dosis(an_.dosis_respuesta(datos, cfg), cfg), fg.fig_contrastes(t3, t5, t4, cfg),
                fg.fig_diagnosticos(an_.deltas_metricas(datos), vacio, cfg),
                fg.fig_robustez(t1, vacio, an_.suelo_ruido(datos, cfg), an_.replica(datos, cfg))):
        assert fig.axes
        plt.close(fig)


@pytest.mark.datos
@pytest.mark.skipif(not HAY_PROCESSED_B, reason="sin data/processed de la pista B")
def test_fig_diseno_con_registro_de_juguete():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from regimenes.sinteticos.aumento import figuras as fg

    cfg = config_aumento()
    taus = pd.to_datetime([f"{a}-12-31" for a in range(2010, 2020)])
    reg = pd.DataFrame([{"pista": "B", "fuente": f, "tau": t, "valida": t.year % 3 != 0,
                         "motivo": "" if t.year % 3 else "no pasa condicionamiento"}
                        for f in fuentes(cfg) for t in taus])
    fig = fg.fig_diseno({"B": reg}, cfg)
    assert len(fig.axes) == 2
    plt.close(fig)
