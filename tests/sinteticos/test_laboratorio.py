"""Tests del laboratorio de detectores (regimenes.sinteticos.laboratorio)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimenes.sinteticos.laboratorio import escenarios as esc
from regimenes.sinteticos.laboratorio.metricas import metricas_trayectoria

CFG = {
    "semilla": 1, "generadores": ["g1", "g2"], "pistas": ["A", "B"], "largo_puntuado": 1260,
    "crisis_calentamiento": {"A": 2, "B": 1}, "crisis_puntuadas": 2, "margen_min": 21,
    "duraciones": [21, 252], "intensidades": [1.0, 0.5], "excluir": {"B": ["D02", "D10"]}, "min_run": 3,
}


def test_calentamiento_igual_que_benchmark():
    from regimenes.benchmark.ejecucion import DEFAULT_TRAIN_DAYS

    assert esc.CALENTAMIENTO == DEFAULT_TRAIN_DAYS


def test_config_real_tiene_seccion_laboratorio():
    cfg = esc.config_laboratorio()
    assert set(cfg["generadores"]) == {"gaussiano_regimen", "var_regimen", "garch_regimen"}
    assert len(esc.celdas("A", cfg)) == len(cfg["generadores"]) * len(cfg["duraciones"]) * len(cfg["intensidades"])


def test_celdas_y_detectores():
    assert len(esc.celdas("A", CFG)) == 8
    assert {"D02", "D10"} & set(esc.detectores_pista("B", CFG)) == set()
    assert len(esc.detectores_pista("A", CFG)) == 12


@pytest.mark.parametrize("pista", ["A", "B"])
@pytest.mark.parametrize("dur", [21, 252])
def test_secuencias_numero_duracion_y_corte(pista, dur):
    reg = esc.secuencias(pista, dur, 5, np.random.default_rng(0), CFG)
    calent = esc.CALENTAMIENTO[pista]
    assert reg.shape == (5, calent + CFG["largo_puntuado"])
    for fila in reg:
        bordes = np.diff(np.concatenate(([0], fila, [0])))
        ini, fin = np.flatnonzero(bordes == 1), np.flatnonzero(bordes == -1)
        assert len(ini) == CFG["crisis_calentamiento"][pista] + CFG["crisis_puntuadas"]
        assert np.all(fin - ini == dur)
        assert not any(a < calent < b for a, b in zip(ini, fin))  # ninguna cruza el corte
        assert ini.min() >= CFG["margen_min"]
        assert fila[calent:].sum() == CFG["crisis_puntuadas"] * dur


def test_secuencias_no_caben():
    cfg = {**CFG, "largo_puntuado": 300}
    with pytest.raises(ValueError):
        esc.secuencias("A", 252, 1, np.random.default_rng(0), cfg)


def test_ventanas_verdad():
    idx = pd.bdate_range("2020-01-01", periods=10)
    reg = pd.Series([0, 1, 1, 0, 0, 1, 0, 0, 1, 1], index=idx)
    v = esc.ventanas_verdad(reg)
    assert list(v.values()) == [(str(idx[1].date()), str(idx[2].date())), (str(idx[5].date()), str(idx[5].date())),
                                (str(idx[8].date()), str(idx[9].date()))]


def _serie(valores, idx):
    return pd.Series(valores, index=idx)


def test_metricas_detector_perfecto_y_nulo():
    idx = pd.bdate_range("2020-01-01", periods=200)
    verdad = np.zeros(200, dtype=int)
    verdad[20:40] = 1
    verdad[120:150] = 1
    perfecto = metricas_trayectoria(_serie(verdad.astype(bool), idx), _serie(verdad, idx))
    assert perfecto["score_deteccion"] == pytest.approx(1.0)
    assert perfecto["exactitud_equilibrada"] == pytest.approx(1.0)
    assert perfecto["retraso_medio"] == 0.0
    assert perfecto["falsas_alarmas_ano"] == 0.0
    nulo = metricas_trayectoria(_serie(np.zeros(200, bool), idx), _serie(verdad, idx))
    assert nulo["score_deteccion"] == 0.0
    assert nulo["det_n_detectados"] == 0


def test_metricas_retraso_y_falsa_alarma():
    idx = pd.bdate_range("2020-01-01", periods=300)
    verdad = np.zeros(300, dtype=int)
    verdad[50:100] = 1
    flags = np.zeros(300, dtype=bool)
    flags[60:100] = True      # detecta con 10 sesiones de retraso
    flags[200:205] = True     # un episodio de falsa alarma
    m = metricas_trayectoria(_serie(flags, idx), _serie(verdad, idx))
    assert m["retraso_medio"] == 10.0
    assert m["det_n_detectados"] == 1
    assert m["falsas_alarmas_ano"] == pytest.approx(1 / (250 / 252))


class _EspacioIdentidad:
    columnas_trabajo_ = ["SP500_ret", "X"]

    def a_trabajo(self, panel):
        return panel[self.columnas_trabajo_].to_numpy(dtype=float).copy()

    def a_publico(self, Z, index):
        return pd.DataFrame(Z, index=index, columns=self.columnas_trabajo_)


class _GenFalso:
    espacio_ = _EspacioIdentidad()


def test_atenuar_interpola_media_y_dispersion():
    rng = np.random.default_rng(0)
    n = 4000
    reg = (np.arange(n) % 4 == 0).astype(int)
    z = np.where(reg[:, None] == 1, 2.0 + 3.0 * rng.standard_normal((n, 2)), rng.standard_normal((n, 2)))
    tray = pd.DataFrame(z, index=pd.bdate_range("2000-01-03", periods=n), columns=["SP500_ret", "X"])
    tray["regime"] = reg
    assert esc.atenuar(tray, _GenFalso(), 1.0) is tray
    cero = esc.atenuar(tray, _GenFalso(), 0.0)
    medio = esc.atenuar(tray, _GenFalso(), 0.5)
    c = reg == 1
    calma_m, calma_s = tray.loc[~c, "X"].mean(), tray.loc[~c, "X"].std(ddof=0)
    assert cero.loc[c, "X"].mean() == pytest.approx(calma_m, abs=1e-9)
    assert cero.loc[c, "X"].std(ddof=0) == pytest.approx(calma_s, rel=1e-9)
    nat_m, nat_s = tray.loc[c, "X"].mean(), tray.loc[c, "X"].std(ddof=0)
    assert medio.loc[c, "X"].mean() == pytest.approx((calma_m + nat_m) / 2, abs=1e-9)
    assert medio.loc[c, "X"].std(ddof=0) == pytest.approx((calma_s + nat_s) / 2, rel=1e-9)
    pd.testing.assert_series_equal(medio.loc[~c, "X"], tray.loc[~c, "X"])  # la calma no se toca
    assert (medio["regime"] == reg).all()


def test_semilla_pareada_por_intensidad_y_huella_distinta():
    from regimenes.detectores.registry import detector_specs
    from regimenes.sinteticos.laboratorio import ejecucion as ej

    a = esc.Celda("A", "g1", 21, 1.0)
    b = esc.Celda("A", "g1", 21, 0.5)
    assert ej.semilla_trayectoria(CFG, a, 3) == ej.semilla_trayectoria(CFG, b, 3)
    assert ej.semilla_trayectoria(CFG, a, 3) != ej.semilla_trayectoria(CFG, a, 4)
    spec = detector_specs("A")[0]
    assert ej.huella(CFG, a, 0, spec) != ej.huella(CFG, b, 0, spec)


def _filas_ficticias():
    from itertools import product

    rng = np.random.default_rng(0)
    filas = []
    base = {"D01": 0.6, "D02": 0.4, "D03": 0.2}
    for g, d, i, k, det in product(["g1", "g2"], [21, 252], [1.0, 0.5], range(8), base):
        s = base[det] + (0.1 if d == 252 else 0) + (0.1 if i == 1.0 else 0) + (0.2 if (g, det) == ("g1", "D03") else 0)
        filas.append({"pista": "A", "generador": g, "duracion": d, "intensidad": i, "path_id": k, "id": det,
                      "celda": f"{g}__d{d}__i{i:.2f}", "score_deteccion": s + 0.01 * rng.standard_normal(),
                      "det_lift_precision": 2.0, "det_event_recall": 0.6})
    return pd.DataFrame(filas)


def test_analisis_hipotesis():
    from regimenes.sinteticos.laboratorio import analisis as an

    f = _filas_ficticias()
    m, lo, hi = an.ic_media([1.0, 2.0, 3.0], n_boot=200)
    assert m == 2.0 and lo <= m <= hi
    rk = an.ranking_laboratorio(f, "A")
    assert list(rk.index) == ["D01", "D02", "D03"]
    assert an.h1_recuperacion(f)["cumple"].all()
    h2 = an.h2_degradacion(f, n_boot=200).set_index("id")
    assert (h2["efecto_acortar"] < 0).all() and (h2["efecto_atenuar"] < 0).all()
    real = pd.DataFrame({"id": ["D01", "D02", "D03"], "score_deteccion": [0.9, 0.5, 0.1]})
    h3 = an.h3_concordancia(rk, real, n_perm=500)
    assert h3["rho"] == pytest.approx(1.0) and h3["n_detectores"] == 3
    h4 = an.h4_circularidad(f, {"g1": ["D03"]}, n_boot=200).iloc[0]
    assert h4["ventaja_casa"] == pytest.approx(0.2, abs=0.02)
    assert h4["ventaja_relativa"] == pytest.approx(0.2, abs=0.02)
    assert h4["ic_lo"] > 0 and h4["rel_ic_lo"] > 0
    assert (h2["acortar_ic_hi"] < 0).all()


def test_metricas_precision_y_recall_cero():
    idx = pd.bdate_range("2020-01-01", periods=100)
    verdad = np.zeros(100, dtype=int)
    verdad[10:20] = 1
    flags = np.zeros(100, dtype=bool)
    flags[60:70] = True  # solo marca fuera de la crisis
    m = metricas_trayectoria(_serie(flags, idx), _serie(verdad, idx))
    assert m["score_deteccion"] == 0.0


@pytest.mark.parametrize("nombre", ["gaussiano_regimen", "var_regimen", "garch_regimen"])
def test_ida_y_vuelta_espacio_es_identidad(nombre):
    """lambda = 1 no pasa por el espacio de trabajo; comprobar que hacerlo no cambiaria nada
    (asi el efecto de atenuar no es un artefacto de re-derivar las features del S&P 500)."""
    from regimenes.sinteticos import persistencia

    if not (persistencia.dir_trayectorias(nombre, "A") / "generador.pkl").exists():
        pytest.skip("generador no ajustado (ejecuta el notebook 15)")
    from regimenes.sinteticos.laboratorio import ejecucion as ej

    gen = ej.cargar_generador(nombre, "A")
    t = gen.sample(1, 400, {"inicial": "estacionaria"}, random_state=0)[0]
    vuelta = gen.espacio_.a_publico(gen.espacio_.a_trabajo(t), pd.DatetimeIndex(t.index))
    cols = [c for c in t.columns if c != "regime"]
    np.testing.assert_allclose(vuelta[cols].to_numpy(), t[cols].to_numpy(), atol=1e-10)
