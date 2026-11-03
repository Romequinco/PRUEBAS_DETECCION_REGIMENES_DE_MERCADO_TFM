"""D16 y D17 (ADR-010 §3, §3.2): etiquetas sin fuga, prefijo, semilla y especificaciones."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from regimenes.detectores.pool.especificaciones import SpecPool
from regimenes.detectores.pool.shu_supervisados import (
    MIN_OBS_P, columnas_pool, especificaciones_shu_supervisados)
from regimenes.detectores.pool.supervisados import BosqueAleatorio, LogitL1, etiquetas_causales
from regimenes.detectores.registry import CORE_A, CORE_B

FECHAS = pd.bdate_range("2001-01-01", periods=600)
CRISIS = [(FECHAS[100], FECHAS[160]), (FECHAS[400], FECHAS[470])]
TRAMPAS = [(FECHAS[250], FECHAS[270])]


def _panel(semilla: int = 0) -> pd.DataFrame:
    """Dos columnas informativas en las crisis plantadas, una de ruido y la trampa."""
    rng = np.random.default_rng(semilla)
    X = pd.DataFrame(rng.normal(size=(len(FECHAS), 3)), index=FECHAS, columns=["a", "b", "ruido"])
    for i, s in CRISIS:
        X.loc[i:s, ["a", "b"]] += 2.5
    X.loc[TRAMPAS[0][0]:TRAMPAS[0][1], "a"] += 2.5
    return X


def _clases():
    return [
        lambda **kw: LogitL1(crisis=CRISIS, trampas=TRAMPAS, **kw),
        lambda **kw: BosqueAleatorio(crisis=CRISIS, trampas=TRAMPAS, n_estimators=30,
                                     min_samples_leaf=5, **kw),
    ]


def test_etiquetas_sin_fuga():
    # Fin del train dentro de la segunda crisis: aún no ha tocado suelo.
    idx = FECHAS[:430]
    y = etiquetas_causales(idx, CRISIS, TRAMPAS)
    assert y.loc[CRISIS[0][0]:CRISIS[0][1]].eq(1).all()
    assert y.loc[CRISIS[1][0]:].eq(0).all()                 # sin terminar = calma
    assert y.loc[TRAMPAS[0][0]:TRAMPAS[0][1]].eq(0).all()
    assert y.sum() == 61
    excl = etiquetas_causales(idx, CRISIS, TRAMPAS, excluir_sin_terminar=True)
    assert excl.loc[CRISIS[1][0]:].isna().all()
    assert excl.dropna().sum() == 61
    # Con el suelo ya dentro del train, la crisis existe.
    assert etiquetas_causales(FECHAS[:471], CRISIS)[CRISIS[1][0]:CRISIS[1][1]].eq(1).all()
    # Una trampa dentro de una crisis cuenta como calma.
    y2 = etiquetas_causales(FECHAS[:200], [(FECHAS[10], FECHAS[50])], [(FECHAS[20], FECHAS[30])])
    assert y2.loc[FECHAS[20]:FECHAS[30]].eq(0).all() and y2.sum() == 41 - 11


@pytest.mark.parametrize("hacer", _clases())
def test_sin_crisis_terminadas_predice_calma(hacer):
    X = _panel()
    det = hacer().fit(X.iloc[:150])                       # la primera crisis aún no tiene suelo
    assert det.siempre_calma and det.n_crisis_terminadas == 0
    assert det.predict(X).sum() == 0
    np.testing.assert_array_equal(det.predict_proba(X)[:, 1], 0.0)


@pytest.mark.parametrize("hacer", _clases())
def test_futuro_no_cambia_el_ajuste(hacer):
    X = _panel()
    fin = 430                                             # dentro de la segunda crisis
    a = hacer().fit(X.iloc[:fin])
    Xf = X.copy()
    Xf.iloc[fin:] = 99.0                                  # el futuro es basura
    b = hacer().fit(Xf.iloc[:fin])
    np.testing.assert_allclose(a.crisis_probability(X), b.crisis_probability(X))
    # La crisis sin terminar no está como crisis: con exclusión cambia el número de filas.
    assert a.n_dias_crisis == 61 and a.n_filas_ajuste == fin
    c = hacer(excluir_sin_terminar=True).fit(X.iloc[:fin])
    assert c.n_dias_crisis == 61 and c.n_filas_ajuste == fin - 30


@pytest.mark.parametrize("hacer", _clases())
def test_prefijo_semilla_y_clase_de_crisis(hacer):
    X = _panel()
    det = hacer().fit(X.iloc[:480])
    det.label_states_economically(X.iloc[:480], market_returns=X["ruido"])
    todo = det.predict_online(X)
    np.testing.assert_array_equal(todo[:500], det.predict_online(X.iloc[:500]))
    np.testing.assert_array_equal(todo, hacer().fit(X.iloc[:480]).predict_online(X))
    assert det.n_states == 2 and det.crisis_state == 1
    p = det.predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    np.testing.assert_array_equal(todo, (p[:, 1] >= 0.5).astype(int))
    # La clase 1 es la crisis: las crisis plantadas puntúan más que la calma.
    assert todo[100:160].mean() > 0.8 and todo[300:380].mean() < 0.2


def test_hiperparametros_del_adr():
    lr = LogitL1()._nuevo_clasificador()
    assert (lr.C, lr.class_weight, lr.solver) == (1.0, "balanced", "liblinear")
    # Penalización L1 de verdad: con C pequeño anula los coeficientes del ruido.
    rng = np.random.default_rng(0)
    A = rng.normal(size=(300, 20))
    y = (A[:, 0] + rng.normal(size=300) > 0).astype(int)
    lr_fuerte = LogitL1(C=0.05)._nuevo_clasificador().fit(A, y)
    assert (lr_fuerte.coef_ != 0).sum() <= 2
    rf = BosqueAleatorio()._nuevo_clasificador()
    assert (rf.n_estimators, rf.max_depth, rf.min_samples_leaf, rf.class_weight, rf.random_state) == \
           (500, 5, 50, "balanced_subsample", 42)


@pytest.mark.parametrize("pista, n_p, core", [("A", 35, CORE_A), ("B", 77, CORE_B)])
def test_especificaciones(pista, n_p, core):
    specs = especificaciones_shu_supervisados(pista)
    ids = [s.detector_id for s in specs]
    assert ids == ["D15-U", "D16-P", "D16-N", "D17-P", "D17-N", "D16-P-excl", "D17-P-excl"]
    assert all(isinstance(s, SpecPool) and s.step == 21 and s.expanding for s in specs)
    por_id = {s.detector_id: s for s in specs}
    assert por_id["D15-U"].features == ("SP500_ret",) and por_id["D15-U"].menu == "U"
    assert len(columnas_pool(pista)) == n_p
    for ident in ("D16-P", "D17-P", "D16-P-excl", "D17-P-excl"):
        s = por_id[ident]
        assert s.module == "disponibilidad" and s.class_name == "ConDisponibilidad" and s.menu == "P"
        assert s.features == columnas_pool(pista) and s.params["min_obs"] == MIN_OBS_P
        assert s.params["params"]["pista"] == pista
        assert s.params["params"]["excluir_sin_terminar"] is ident.endswith("-excl")
    assert por_id["D16-P"].params["base"] == "supervisados.LogitL1"
    assert por_id["D17-P"].params["base"] == "supervisados.BosqueAleatorio"
    for ident in ("D16-N", "D17-N"):
        s = por_id[ident]
        assert s.features == core and s.menu == "N" and s.module == "supervisados"
        det = s.factory()
        assert det.pista == pista and det.crisis_state == 1


def test_d15_se_construye_desde_la_spec():
    pytest.importorskip("jumpmodels")
    det = especificaciones_shu_supervisados("A")[0].factory()
    assert det.name == "shu_jump_model_k2" and det.random_state == 42


def test_p_con_envoltorio_si_existe():
    """Si el envoltorio del Agente 1 ya está, la spec de P se instancia con él."""
    try:
        importlib.import_module("regimenes.detectores.pool.disponibilidad")
    except ModuleNotFoundError:
        pytest.skip("pool/disponibilidad.py (ConDisponibilidad) aún no existe")
    det = next(s for s in especificaciones_shu_supervisados("B") if s.detector_id == "D16-P").factory()
    assert det.crisis_state == 1


def test_resumen_reajuste_de_las_etiquetas():
    """Menú N (sin envoltorio) y menú P (``ConDisponibilidad`` lo recoge con ``diagnostico``)."""
    from regimenes.detectores.pool.disponibilidad import ConDisponibilidad

    X = _panel()
    det = LogitL1(crisis=CRISIS, trampas=TRAMPAS).fit(X.iloc[:430])
    r = det.resumen_reajuste()
    assert r == {"sup_n_crisis_terminadas": 1, "sup_n_dias_crisis": 61, "sup_n_filas_ajuste": 430,
                 "sup_siempre_calma": False}
    env = ConDisponibilidad("supervisados.LogitL1", {"crisis": CRISIS, "trampas": TRAMPAS},
                            min_obs=100).fit(X.iloc[:150])
    assert env.resumen_reajuste()["sup_siempre_calma"] is True
