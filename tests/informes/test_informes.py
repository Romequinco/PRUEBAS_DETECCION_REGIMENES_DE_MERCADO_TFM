"""Tests sintéticos de ``regimenes.informes`` (utilidades de los notebooks 05–11).

No leen ``results/benchmark`` ni ``data/``: la caché del benchmark se sustituye por
métricas, paneles y ranking sintéticos en un directorio temporal.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from regimenes import informes as inf  # noqa: E402
from regimenes.benchmark import ejecucion  # noqa: E402

CRISIS = {"crisis_a": ("2020-01-06", "2020-01-17"), "crisis_b": ("2020-03-02", "2020-03-13"),
          "en_train": ("2019-01-01", "2019-02-01")}
TRAMPAS = {"trampa_x": ("2020-02-03", "2020-02-07")}


def _fila_metricas(**extra) -> pd.Series:
    base = {
        "n_states": 2, "n_oos": 60,
        "cov_crisis_a": 0.8, "cov_crisis_a_lo": 0.6, "cov_crisis_a_hi": 0.95,
        "det_ev_crisis_a": 1, "leadlag_crisis_a": -3,
        "cov_crisis_b": 0.1, "cov_crisis_b_lo": 0.0, "cov_crisis_b_hi": 0.3,
        "det_ev_crisis_b": 0, "leadlag_crisis_b": np.nan,
        "cov_en_train": np.nan, "fa_trampa_x": 0.2,
        "det_n_detectados": 1, "det_n_eventos": 2, "det_event_recall": 0.5,
        "det_precision": 0.4, "det_marked_rate": 0.25,
    }
    base.update(extra)
    return pd.Series(base)


# --------------------------------------------------------------------------- #
# Tablas
# --------------------------------------------------------------------------- #
def test_tabla_cobertura_omite_crisis_fuera_del_oos_y_respeta_columnas():
    tab = inf.tabla_cobertura(_fila_metricas(), CRISIS)
    assert list(tab.columns) == inf.COLUMNAS_COBERTURA
    assert list(tab.index) == ["crisis_a", "crisis_b"]          # en_train es NaN -> fuera
    assert tab.loc["crisis_a", "cobertura"] == pytest.approx(0.8)
    assert tab.loc["crisis_a", "detectada"] == 1.0
    assert tab.loc["crisis_b", "detectada"] == 0.0
    assert tab.loc["crisis_a", "lead_lag_dias"] == -3
    assert np.isnan(tab.loc["crisis_b", "lead_lag_dias"])
    assert tab.loc["crisis_a", "pico"] == "2020-01-06"


def test_tabla_cobertura_con_crisis_de_train_y_tabla_vacia():
    completa = inf.tabla_cobertura(_fila_metricas(), CRISIS, solo_oos=False)
    assert "en_train" in completa.index and np.isnan(completa.loc["en_train", "cobertura"])
    vacia = inf.tabla_cobertura(pd.Series({"n_states": 2}), CRISIS)
    assert vacia.empty and list(vacia.columns) == inf.COLUMNAS_COBERTURA


def test_tabla_trampas_todas_o_seleccion():
    fila = _fila_metricas(fa_otra=0.5)
    todas = inf.tabla_trampas(fila)
    assert set(todas.index) == {"trampa_x", "otra"}
    sel = inf.tabla_trampas(fila, TRAMPAS)
    assert list(sel.index) == ["trampa_x"] and sel["trampa_x"] == pytest.approx(0.2)
    assert np.isnan(inf.tabla_trampas(fila, ["no_existe"])["no_existe"])


def test_matriz_jaccard():
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    flags = {
        "a": pd.Series([1, 1, 0, 0, 1, 0], index=idx, dtype=bool),
        "b": pd.Series([1, 0, 0, 0, 1, 1], index=idx, dtype=bool),
        "c": pd.Series([0] * 6, index=idx, dtype=bool),
    }
    J = inf.matriz_jaccard(flags)
    assert J.loc["a", "a"] == 1.0
    assert J.loc["a", "b"] == pytest.approx(2 / 4)
    assert J.loc["a", "b"] == J.loc["b", "a"]
    assert np.isnan(J.loc["c", "c"])             # nadie marca: Jaccard indefinido
    assert J.loc["a", "c"] == 0.0


def test_retardo_confirmacion():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    flags = pd.Series(False, index=idx)
    flags.loc["2020-01-09":"2020-01-15"] = True    # crisis_a: pico 01-06 -> 3 sesiones después
    r = inf.retardo_confirmacion(flags, CRISIS)
    assert r["crisis_a"] == 3
    assert np.isnan(r["crisis_b"])                 # ningún día marcado en la ventana
    assert "en_train" not in r.index               # sin sesiones OOS en la ventana
    flags.loc["2020-03-02"] = True
    assert inf.retardo_confirmacion(flags, CRISIS)["crisis_b"] == 0


def test_etiquetas_y_comando():
    assert inf.etiquetas_estados(2) == ["calma", "crisis"]
    assert inf.etiquetas_estados(5) == ["calma", "intermedio 1", "intermedio 2", "intermedio 3", "crisis"]
    assert inf.comando_benchmark(["a", "B"], ["d05"]) == "python -m regimenes.benchmark --track A B --detector D05"


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def _ranking_sintetico(metricas: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for (t, d), m in metricas.iterrows():
        filas.append({"pista": t.lower(), "id": d, "puesto_deteccion": 0, "nivel_etiqueta": "elegible",
                      "score_deteccion": 0.0, **{c: m[c] for c in inf.COLUMNAS_COHERENCIA if c in m.index}})
    rk = pd.DataFrame(filas)
    rk["score_deteccion"] = 2 * rk.det_precision * rk.det_event_recall / (rk.det_precision + rk.det_event_recall)
    rk["puesto_deteccion"] = rk.groupby("pista")["score_deteccion"].rank(ascending=False, method="min").astype(int)
    # Un detector de otra familia en cada pista.
    extra = rk.drop_duplicates("pista").assign(id="D99", puesto_deteccion=99, det_precision=0.01)
    return pd.concat([rk, extra], ignore_index=True)


def test_buscar_ranking_prefiere_ranking_csv_y_falla_sin_archivo(tmp_path):
    with pytest.raises(FileNotFoundError, match="12_comparativa"):
        inf.buscar_ranking(tmp_path)
    (tmp_path / "ranking_zzz.csv").write_text("pista,id\n", encoding="utf-8")
    assert inf.buscar_ranking(tmp_path).name == "ranking_zzz.csv"
    (tmp_path / "ranking_v2.csv").write_text("pista,id\n", encoding="utf-8")
    assert inf.buscar_ranking(tmp_path).name == "ranking_v2.csv"
    (tmp_path / "ranking.csv").write_text("pista,id\n", encoding="utf-8")
    assert inf.buscar_ranking(tmp_path).name == "ranking.csv"


def test_claves_incoherentes_y_resumen_ranking():
    metricas = pd.DataFrame(
        [_fila_metricas().rename(None).to_dict() | {"pista": "A", "id": "D01"},
         _fila_metricas(det_precision=0.6).to_dict() | {"pista": "A", "id": "D02"}]
    ).set_index(["pista", "id"])
    ranking = _ranking_sintetico(metricas)
    ranking["pista"] = ranking["pista"].str.upper()
    assert inf.claves_incoherentes(ranking, metricas, [("A", "D01"), ("A", "D02")]) == []
    roto = ranking.copy()
    roto.loc[roto["id"] == "D02", "det_precision"] = 0.9
    assert inf.claves_incoherentes(roto, metricas, [("A", "D01"), ("A", "D02")]) == ["A/D02 (difiere)"]
    assert inf.claves_incoherentes(ranking, metricas, [("B", "D01")]) == ["B/D01 (ausente del ranking)"]
    res = inf.tabla_resumen_ranking(ranking, [("A", "D02"), ("A", "D01")])
    assert list(res.index) == [("A", "D02"), ("A", "D01")]
    assert res.columns[1] == "de" and set(res["de"]) == {3}
    assert res.loc[("A", "D02"), "puesto_deteccion"] == 1


def test_clasificar_lead_lag_distingue_saturacion_alarma_previa_y_reaccion():
    idx = pd.bdate_range("2020-01-01", periods=400)
    pico, suelo = idx[300], idx[320]          # 20 sesiones entre pico y suelo
    tabla = pd.DataFrame(
        {"pico": [str(pico.date())] * 4, "suelo": [str(suelo.date())] * 4,
         "lead_lag_dias": [-252.0, -100.0, -5.0, np.nan]},
        index=pd.Index(["sat", "previa", "reaccion", "nada"], name="crisis"))
    out = inf.clasificar_lead_lag(tabla, idx, lookback=252)
    assert list(out["lectura_lead_lag"]) == ["saturado", "antes del pico", "entre pico y suelo", "sin señal"]
    assert set(out["sesiones_pico_suelo"]) == {20}
    assert inf.lookback_lead_lag() == 252


def test_claves_incoherentes_detecta_ranking_de_otra_ejecucion():
    metricas = pd.DataFrame(
        [_fila_metricas().rename(None).to_dict() | {"pista": "A", "id": "D01", "elapsed_seconds": 10.0}]
    ).set_index(["pista", "id"])
    ranking = metricas.reset_index()
    assert inf.claves_incoherentes(ranking, metricas, [("A", "D01")]) == []
    ranking.loc[0, "elapsed_seconds"] = 12.0
    assert inf.claves_incoherentes(ranking, metricas, [("A", "D01")]) == ["A/D01 (difiere)"]


# --------------------------------------------------------------------------- #
# Carga de resultados (run_benchmark sustituido por una caché sintética)
# --------------------------------------------------------------------------- #
def _escribir_cache(tmp_path, pistas=("A", "B"), detectores=("D01", "D02"), n=60):
    idx = pd.bdate_range("2020-01-01", periods=n)
    filas = []
    (tmp_path / "panels").mkdir()
    rng = np.random.default_rng(0)
    for t in pistas:
        for k, d in enumerate(detectores):
            estado = (rng.random(n) > 0.7).astype(int)
            pd.DataFrame({"state": estado, "p_crisis": estado * 0.9, "fold": 0},
                         index=idx.strftime("%Y-%m-%d")).to_parquet(tmp_path / "panels" / f"{t}_{d}.parquet")
            filas.append(_fila_metricas(n_oos=n, det_precision=0.3 + 0.1 * k).to_dict() | {"pista": t, "id": d})
    metricas = pd.DataFrame(filas)
    ranking = _ranking_sintetico(metricas.set_index(["pista", "id"]))
    ranking.to_csv(tmp_path / "ranking_v2.csv", index=False)
    return metricas


def _falso_run_benchmark(metricas, estados=None):
    def run_benchmark(*, tracks, detector_ids, output_dir, cache_only):
        assert cache_only is True
        sel = metricas[metricas["pista"].isin(list(tracks)) & metricas["id"].isin(list(detector_ids))]
        estado = pd.DataFrame({"pista": sel["pista"], "id": sel["id"], "estado": "cache", "detalle": "ok"})
        if estados is not None:
            estado["estado"] = estados
        return sel.reset_index(drop=True), estado.reset_index(drop=True)
    return run_benchmark


def test_cargar_resultados_familia_sintetico(tmp_path, monkeypatch):
    metricas = _escribir_cache(tmp_path)
    monkeypatch.setattr(ejecucion, "run_benchmark", _falso_run_benchmark(metricas))
    res = inf.cargar_resultados_familia(["d01", "D02"], ["a", "B"], output_dir=tmp_path)
    assert res.pistas == ["A", "B"] and res.detectores == ["D01", "D02"]
    assert res.claves == [("A", "D01"), ("A", "D02"), ("B", "D01"), ("B", "D02")]
    assert isinstance(res.paneles[("A", "D01")].index, pd.DatetimeIndex)
    assert res.ruta_ranking.name == "ranking_v2.csv"
    assert res.n_por_pista == {"A": 3, "B": 3}
    assert res.puesto("A", "D02") == "1 de 3"
    assert res.estado_crisis("B", "D01") == 1
    esperado = res.paneles[("A", "D01")]["state"].eq(1)
    pd.testing.assert_series_equal(res.es_crisis("A", "D01"), esperado)
    assert res.fila("A", "D01")["n_oos"] == 60
    assert res.fila("A", "D01")["mean_crisis_coverage"] == pytest.approx((0.8 + 0.1) / 2)
    assert res.fila("A", "D01")["mean_trap_activation"] == pytest.approx(0.2)
    assert res.resumen_ranking().shape[0] == 4
    assert res.comando == "python -m regimenes.benchmark --track A B --detector D01 D02"


def test_cargar_resultados_falla_con_comando_si_la_cache_no_esta_vigente(tmp_path, monkeypatch):
    metricas = _escribir_cache(tmp_path, pistas=("A",), detectores=("D05",))
    monkeypatch.setattr(ejecucion, "run_benchmark", _falso_run_benchmark(metricas, estados="sin_cache"))
    with pytest.raises(RuntimeError, match=r"python -m regimenes\.benchmark --track A --detector D05"):
        inf.cargar_resultados_familia(["D05"], ["A"], output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="no contiene ninguna"):
        inf.verificar_estado_cache(pd.DataFrame(), "cmd")


def test_cargar_resultados_detecta_ranking_incoherente_y_panel_corto(tmp_path, monkeypatch):
    metricas = _escribir_cache(tmp_path, pistas=("A",), detectores=("D01",))
    ranking = pd.read_csv(tmp_path / "ranking_v2.csv")
    ranking.loc[ranking["id"] == "D01", "det_n_detectados"] = 7
    ranking.to_csv(tmp_path / "ranking_v2.csv", index=False)
    monkeypatch.setattr(ejecucion, "run_benchmark", _falso_run_benchmark(metricas))
    with pytest.raises(RuntimeError, match="12_comparativa"):
        inf.cargar_resultados_familia(["D01"], ["A"], output_dir=tmp_path)
    res = inf.cargar_resultados_familia(["D01"], ["A"], output_dir=tmp_path, exigir_ranking_coherente=False)
    assert res.incoherencias == ["A/D01 (difiere)"]
    metricas.loc[:, "n_oos"] = 61
    monkeypatch.setattr(ejecucion, "run_benchmark", _falso_run_benchmark(metricas))
    with pytest.raises(RuntimeError, match="no cuadra con n_oos"):
        inf.cargar_resultados_familia(["D01"], ["A"], output_dir=tmp_path, exigir_ranking_coherente=False)


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def _panel_sintetico(n=80, k=2):
    idx = pd.bdate_range("2020-01-01", periods=n)
    estado = np.zeros(n, dtype=int)
    estado[20:35] = k - 1
    if k > 2:
        estado[50:60] = 1
    return pd.DataFrame({"state": estado, "p_crisis": (estado == k - 1) * 0.8 + 0.1}, index=idx)


def test_figura_estados_oos_por_defecto_y_con_diagnostico(tmp_path):
    panel = _panel_sintetico()
    precio = pd.Series(np.exp(np.cumsum(np.full(200, 0.001))), index=pd.bdate_range("2019-10-01", periods=200))
    fig, (ax1, ax2) = inf.figura_estados_oos(precio, panel, 2, crisis=CRISIS, trampas=TRAMPAS, titulo="t")
    assert ax1.get_yscale() == "log"
    assert ax1.get_title() == "t"
    assert len(ax1.patches) >= 2                   # tramo de crisis + franjas
    assert ax2.get_ylim() == pytest.approx((-0.02, 1.02))
    ruta = inf.guardar_figura(fig, tmp_path / "f", "D01_A_estados_oos", verbose=False)
    assert ruta.name == "D01_A_estados_oos.png" and ruta.stat().st_size > 0
    plt.close(fig)

    llamadas = []
    fig, (_, ax2) = inf.figura_estados_oos(precio, _panel_sintetico(k=3), 3, crisis=CRISIS,
                                           diagnostico=lambda ax: llamadas.append(ax),
                                           titulo_diagnostico="diag", todos_los_estados=True)
    assert llamadas == [ax2] and ax2.get_title() == "diag"
    plt.close(fig)


def test_figura_cobertura_con_trampas_y_extra(tmp_path):
    tab = inf.tabla_cobertura(_fila_metricas(), CRISIS)
    trampas = inf.tabla_trampas(_fila_metricas(), TRAMPAS)
    fig, ax = inf.figura_cobertura(tab, trampas, titulo="cob", min_run=3,
                                   extra=pd.Series({"crisis_a": 0.9}), etiqueta_extra="corrección + crisis")
    etiquetas = [t.get_text() for t in ax.get_yticklabels()]
    assert etiquetas == ["crisis_a", "crisis_b", "trampa: trampa_x"]
    assert ax.get_xlim() == pytest.approx((0, 1.02))
    assert ax.get_title() == "cob"
    assert inf.guardar_figura(fig, tmp_path, "x.png", verbose=False).name == "x.png"
    plt.close(fig)


def test_franjas_y_guardar_crea_carpeta(tmp_path):
    fig, ax = plt.subplots()
    inf.franjas_eventos(ax, CRISIS, TRAMPAS)
    etiquetas = [h.get_label() for h in ax.patches]
    assert etiquetas.count("crisis del catálogo [pico, suelo]") == 1
    assert etiquetas.count("ventana trampa (no-crisis)") == 1
    assert len(ax.patches) == len(CRISIS) + len(TRAMPAS)
    plt.close(fig)
    assert inf.ruta_relativa(tmp_path / "x").endswith("x")


def test_dibujar_heatmap_anota_y_omite_nan():
    fig, ax = plt.subplots()
    m = pd.DataFrame([[1.0, np.nan], [0.25, -0.5]], index=["x", "y"], columns=["a", "b"])
    inf.dibujar_heatmap(ax, m, cmap="RdBu_r", vmin=None, vmax=None, titulo="h")
    textos = sorted(t.get_text() for t in ax.texts)
    assert textos == ["-0.50", "0.25", "1.00"]
    assert ax.get_title() == "h"
    plt.close(fig)


def test_dibujar_plano_ranking_resalta_la_familia():
    rk_pista = pd.DataFrame({
        "id": ["D01", "D02", "D03"], "puesto_deteccion": [1, 2, 3],
        "det_event_recall": [0.9, 0.5, 0.2], "det_precision": [0.5, 0.4, 0.3], "det_base_rate": 0.2,
    })
    fig, ax = plt.subplots()
    inf.dibujar_plano_ranking(ax, rk_pista, ["D02"], etiqueta="familia F1", titulo="p")
    anotaciones = [t.get_text() for t in ax.texts if not t.get_text().startswith("score=")]
    assert "D02 (#2)" in anotaciones and "D01" in anotaciones and "D03" in anotaciones
    assert "familia F1" in [h.get_label() for h in ax.get_legend().legend_handles]
    plt.close(fig)


class _SpecFalsa:
    def __init__(self, did, n_states=2, random_state=None):
        self.detector_id, self.label, self.family = did, f"det {did}", "F0"
        self.module, self.class_name = "modulo", "Clase"
        self.features = ("x1", "x2")
        self.params = {"features": self.features, "n_states": n_states}
        if random_state is not None:
            self.params["random_state"] = random_state
        self.step, self.expanding, self.slow, self.variant = 21, True, False, None

    def factory(self):
        class _Det:
            def __init__(self, features=None, n_states=2, q_in=0.9, min_dwell=5, random_state=None):
                self.n_states, self.q_in = n_states, q_in

        return _Det(**self.params)


def test_tabla_configuracion_desde_specs_sinteticas():
    specs = {("B", "D02"): _SpecFalsa("D02", random_state=7), ("A", "D02"): _SpecFalsa("D02"),
             ("A", "D01"): _SpecFalsa("D01", n_states=3)}
    tab = inf.tabla_configuracion(specs, train_days={"A": 2016, "B": 756})
    assert list(tab.index) == [("D01", "A"), ("D02", "A"), ("D02", "B")]
    assert tab.loc[("D01", "A"), "params_efectivos"] == "n_states=3, q_in=0.9"
    assert tab.loc[("D01", "A"), "params_declarados"] == "n_states=3"
    assert tab.loc[("D01", "A"), "params_por_defecto"] == "q_in=0.9, min_dwell=5"
    assert tab.loc[("D02", "B"), "semilla"] == 7
    assert tab.loc[("D02", "A"), "semilla"] == "no consume azar"
    assert tab.loc[("D02", "B"), "train_inicial"] == "756 ses. (~3 años)"
    assert tab.loc[("D01", "A"), "ventana"] == "expanding"
    assert tab.loc[("D01", "A"), "variante_v2"] == "—"
