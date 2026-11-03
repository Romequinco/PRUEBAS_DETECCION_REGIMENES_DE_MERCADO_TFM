"""Tests de los detectores de la Fase 2 sobre el pool sin comprimir (ADR-010 §2–§4).

Disponibilidad (§2.1), prefijo, D14 con un factor plantado, semilla fija de D18, panel R y
especificaciones. Datos de juguete pequeños; nada lee ``data/`` salvo ``configs/bloques.yaml``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimenes import evaluacion as ev
from regimenes.detectores.pool import absorption_ratio as ar
from regimenes.detectores.pool.disponibilidad import ConDisponibilidad, columnas_disponibles, resolver_clase
from regimenes.detectores.pool.pool_crudo import columnas_p, especificaciones_pool_crudo
from regimenes.features import rentabilidades as rent

D10 = "regimenes.detectores.f1_reglas.turbulence_mahalanobis.TurbulenceMahalanobis"
D10_JUGUETE = {"q_in": 0.9, "q_out": 0.7, "min_dwell": 3, "cov_min_periods": 40}


def _panel(n: int = 700, p: int = 4, semilla: int = 0, tarde: int = 250) -> pd.DataFrame:
    """Rentabilidades con una racha de volatilidad alta y una columna que empieza tarde."""
    rng = np.random.default_rng(semilla)
    idx = pd.bdate_range("2001-01-01", periods=n)
    vol = np.where((np.arange(n) % 200) > 160, 3.0, 1.0)
    X = pd.DataFrame(rng.normal(size=(n, p)) * vol[:, None], index=idx,
                     columns=[f"x{i}" for i in range(p)])
    X["tarde"] = rng.normal(size=n) * vol
    X.iloc[:tarde, X.columns.get_loc("tarde")] = np.nan
    return X


def _envuelto(min_obs: int = 150, **kw) -> ConDisponibilidad:
    return ConDisponibilidad(D10, dict(D10_JUGUETE), min_obs=min_obs, **kw)


# --------------------------------------------------------------------------- #
# Disponibilidad (§2.1)
# --------------------------------------------------------------------------- #
def test_columna_entra_solo_con_min_obs_en_la_ventana() -> None:
    X = _panel()
    det = _envuelto(min_obs=150).fit(X.iloc[:350])           # "tarde" tiene 100 datos
    assert "tarde" not in det.columnas_ and det.columnas_fuera_ == ["tarde"]
    assert det.resumen_reajuste()["n_filas_ajuste"] == 350
    det = _envuelto(min_obs=150).fit(X.iloc[:450])           # ahora tiene 200
    assert "tarde" in det.columnas_
    fila = det.resumen_reajuste()
    assert fila["n_filas_ajuste"] == 200                     # solo filas con todas las que entran
    assert fila["n_columnas"] == 5 and fila["fin_train"] == X.index[449]
    assert columnas_disponibles(X.iloc[:450], 201) == ["x0", "x1", "x2", "x3"]


def test_predice_con_las_columnas_del_ajuste() -> None:
    X = _panel()
    det = _envuelto(min_obs=150).fit(X.iloc[:350])
    bloque = X.iloc[350:400]
    a = det.predict_online(bloque)
    b = det.predict_online(bloque.assign(tarde=99.0))        # la columna fuera no cuenta
    assert np.array_equal(a, b) and len(a) == 50


def test_sin_columnas_suficientes_falla() -> None:
    with pytest.raises(ValueError, match="ninguna columna"):
        _envuelto(min_obs=1000).fit(_panel().iloc[:300])


def test_relleno_hacia_delante_en_filas_incompletas() -> None:
    X = _panel()
    det = _envuelto(min_obs=150).fit(X.iloc[:500])
    bloque = X.iloc[500:560].copy()
    bloque.iloc[10:13, 0] = np.nan
    s = det.predict_online(bloque)
    p = det.predict_proba(bloque)
    assert np.all(s[10:13] == s[9]) and np.allclose(p[10:13], p[9])
    assert np.allclose(p.sum(axis=1), 1.0)


def test_resolver_clase_corta_y_completa() -> None:
    assert resolver_clase("absorption_ratio.AbsorptionRatio") is ar.AbsorptionRatio
    assert resolver_clase(D10).__name__ == "TurbulenceMahalanobis"


def test_crisis_state_y_n_states_del_base() -> None:
    det = _envuelto()
    assert det.n_states == 2 and det.crisis_state == 1


# --------------------------------------------------------------------------- #
# Prefijo: calcular hasta t = calcular con todo y recortar en t
# --------------------------------------------------------------------------- #
def _wf(factory, X: pd.DataFrame, train: int = 300, step: int = 21) -> pd.DataFrame:
    mr = X["x0"]
    return ev.walk_forward(factory, X, market_returns=mr, train_size=train, min_train=train, step=step)


def test_prefijo_disponibilidad_d10() -> None:
    X = _panel()
    corte = 300 + 21 * 10                                    # fin de un bloque
    todo = _wf(lambda: _envuelto(min_obs=150), X)
    recortado = _wf(lambda: _envuelto(min_obs=150), X.iloc[:corte])
    pd.testing.assert_frame_equal(todo.loc[recortado.index], recortado)


def test_prefijo_absorption_ratio() -> None:
    X = _panel(n=500, p=6, tarde=0).drop(columns="tarde")
    X.iloc[100:103, 2] = np.nan                              # festivos: hereda el ΔAR anterior
    kw = {"ventana": 60, "semivida": 30, "n_autovectores": 2, "corto": 5, "largo": 40}
    todo = ar.AbsorptionRatio(**kw).fit(X).serie_delta_ar(X)
    corte = X.iloc[:300]
    parcial = ar.AbsorptionRatio(**kw).fit(corte).serie_delta_ar(corte)
    pd.testing.assert_series_equal(todo.loc[parcial.index], parcial)
    assert todo.iloc[101] == todo.iloc[99]


def test_prefijo_rentabilidades() -> None:
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2010-01-01", periods=120)
    crudos = {s: pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))), index=idx) for s in rent.SERIES}
    todo = rent.panel_rentabilidades(crudos, idx)
    corte = idx[70]
    parcial = rent.panel_rentabilidades({s: c.loc[:corte] for s, c in crudos.items()}, idx[:71])
    pd.testing.assert_frame_equal(todo.loc[:corte], parcial)


# --------------------------------------------------------------------------- #
# D14: factor plantado y calentamiento
# --------------------------------------------------------------------------- #
def test_ar_sube_al_plantar_un_factor_comun() -> None:
    rng = np.random.default_rng(3)
    n, p = 600, 10
    R = rng.normal(size=(n, p))
    factor = rng.normal(size=n) * 3.0
    R[400:] += factor[400:, None]                            # factor común desde la fila 400
    a = ar.absorption_ratio(R, ventana=100, semivida=50, n_autovectores=2)
    assert np.isnan(a[:99]).all() and np.isfinite(a[99:]).all()
    assert np.nanmean(a[150:400]) < 0.4 < 0.8 < np.nanmean(a[520:])
    X = pd.DataFrame(R, index=pd.bdate_range("2005-01-03", periods=n))
    det = ar.AbsorptionRatio(ventana=100, semivida=50, n_autovectores=2, corto=5, largo=60).fit(X.iloc[:380])
    estados = det._predict_states(X)
    assert estados[:380].mean() < 0.1 and estados[400:440].mean() > 0.5   # ΔAR >= 1 al llegar el factor
    informe = det.informe_oos(X, X.index[380:])
    assert informe["d14_dias_oos_calentamiento"] == 0


def test_d14_calentamiento_predice_calma() -> None:
    X = _panel(n=200, p=5, tarde=0).drop(columns="tarde")
    det = ar.AbsorptionRatio(ventana=100, largo=150).fit(X)
    assert (det._predict_states(X) == 0).all()
    assert det.informe_oos(X, X.index[100:])["d14_dias_oos_calentamiento"] == 100


# --------------------------------------------------------------------------- #
# D18: semilla fija
# --------------------------------------------------------------------------- #
def test_d18_semilla_fija_y_columnas_con_peso() -> None:
    pytest.importorskip("jumpmodels")
    rng = np.random.default_rng(7)
    n = 400
    estado = (np.arange(n) // 50) % 2
    X = pd.DataFrame(rng.normal(size=(n, 6)), index=pd.bdate_range("2003-01-01", periods=n),
                     columns=[f"c{i}" for i in range(6)])
    X["c0"] += 3 * estado
    X["c1"] -= 3 * estado
    params = {"n_states": 2, "jump_penalty": 50.0, "max_feats": 2.0, "n_init_jm": 3, "random_state": 42}
    a = ConDisponibilidad("sparse_jump.SparseJumpPool", params, min_obs=100).fit(X)
    b = ConDisponibilidad("sparse_jump.SparseJumpPool", params, min_obs=100).fit(X)
    assert np.array_equal(a.predict_online(X), b.predict_online(X))
    pd.testing.assert_series_equal(a.base.pesos(), b.base.pesos())
    fila = a.resumen_reajuste()
    assert {"c0", "c1"} <= set(fila["d18_columnas_peso_pos"].split("|"))
    assert fila["d18_n_peso_pos"] <= 6


# --------------------------------------------------------------------------- #
# Especificaciones y panel R
# --------------------------------------------------------------------------- #
def test_especificaciones_pool_crudo() -> None:
    a = {s.detector_id: s for s in especificaciones_pool_crudo("A")}
    b = {s.detector_id: s for s in especificaciones_pool_crudo("B")}
    assert sorted(a) == ["D10-P", "D18-P"]
    assert sorted(b) == ["D10-P", "D10-R", "D14-R", "D18-P"]
    assert len(columnas_p("A")) == 35 and len(columnas_p("B")) == 77
    assert len(b["D10-R"].features) == 22 and b["D14-R"].features == rent.columnas_r()
    assert a["D18-P"].params["params"]["max_feats"] == 9 and b["D18-P"].params["params"]["max_feats"] == 14
    assert all(s.step == 21 and s.expanding for s in [*a.values(), *b.values()])
    assert {s.menu for s in b.values()} == {"P", "R"}
    assert b["D10-P"].params["base"] == D10 and b["D10-P"].params["min_obs"] == 756
    assert isinstance(a["D10-P"].factory(), ConDisponibilidad)


def test_rentabilidades_divisas_y_huecos() -> None:
    idx = pd.bdate_range("2010-01-04", periods=6)
    crudos = {s: pd.Series([100.0, 101, 102, 103, 104, 105], index=idx) for s in rent.SERIES}
    crudos["DEXUSAL"] = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], index=idx)
    crudos["DEXJPUS"] = crudos["DEXJPUS"].drop(idx[2])        # festivo de FRED
    out = rent.panel_rentabilidades(crudos, idx)
    assert out.shape == (6, 22) and np.isnan(out.iloc[0]).all()
    assert np.isclose(out.loc[idx[1], "DEXUSAL_logret"], -np.log(1.1))
    assert np.isnan(out.loc[idx[2], "DEXJPUS_logret"])        # sin rellenar
    assert np.isclose(out.loc[idx[3], "DEXJPUS_logret"], np.log(103 / 101))
    with pytest.raises(KeyError):
        rent.panel_rentabilidades({k: v for k, v in crudos.items() if k != "SP500"}, idx)


def test_d14_estado_de_crisis_fijo_sin_canonicalizar() -> None:
    """ΔAR ≥ 1 es siempre la crisis, aunque el calentamiento sea más volátil (errata §8.1)."""
    from regimenes.detectores.pool.absorption_ratio import AbsorptionRatio

    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2000-01-03", periods=900)
    X = pd.DataFrame(rng.normal(0, 0.01, (900, 6)), index=idx, columns=[f"r{i}" for i in range(6)])
    X.iloc[:300] *= 3.0                          # calentamiento muy volátil, sin ΔAR
    det = AbsorptionRatio(n_autovectores=1).fit(X)
    det.label_states_economically(X, market_returns=X["r0"])
    assert det.crisis_state == 1
    assert list(det._canonical_order) == [0, 1]
