"""Tests sintéticos (sin datos en disco) de las primitivas y recetas causales de src/features.py.

Idea central: una transformación es causal si recomputarla con la entrada truncada en `cut`
da exactamente los mismos valores <= cut que con la muestra completa (test de truncado).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features as ft


def _serie_diaria(n: int = 800, seed: int = 0, nombre: str = "X", precio: bool = True) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    r = rng.standard_t(df=4, size=n) * 0.01
    v = 100 * np.exp(np.cumsum(r)) if precio else np.cumsum(rng.normal(size=n))
    return pd.Series(v, index=idx, name=nombre)


def _serie_mensual(n: int = 240, seed: int = 1, nombre: str = "M") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-01-01", periods=n, freq="MS")
    return pd.Series(100 + np.cumsum(rng.normal(size=n)), index=idx, name=nombre)


CUTS = ["2000-06-30", "2001-07-16", "2002-12-31"]


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize(
    "builder",
    [
        lambda s: ft.causal_zscore(s),
        lambda s: ft.causal_zscore(s, method="rolling", window=100),
        lambda s: ft.causal_zscore(s, lag=1),
        lambda s: ft.realized_vol(ft.log_returns(s)),
        lambda s: ft.drawdown(s),
        lambda s: ft.momentum(s),
    ],
    ids=["zscore_expanding", "zscore_rolling", "zscore_lag1", "realized_vol", "drawdown", "momentum"],
)
def test_primitivas_causales_por_truncado(builder, cut):
    s = _serie_diaria()
    full, trunc = builder(s), builder(s.loc[:cut])
    assert ft.truncation_max_abs_diff(full, trunc, cut).max() == 0.0


def test_zscore_no_es_de_muestra_completa():
    """Control negativo: un z-score de muestra completa SÍ falla el truncado."""
    s = _serie_diaria()
    full_sample = lambda x: (x - x.mean()) / x.std()  # noqa: E731 - el error de la tarea previa
    cut = CUTS[1]
    assert ft.truncation_max_abs_diff(full_sample(s), full_sample(s.loc[:cut]), cut).max() > 1e-6


def test_truncation_detecta_nan_que_aparece():
    idx = pd.bdate_range("2020-01-01", periods=5)
    a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx, name="x")
    b = a.copy()
    b.iloc[1] = np.nan
    assert np.isinf(ft.truncation_max_abs_diff(a, b, idx[-1]).iloc[0])


def test_lag_publicacion_desplaza_indice_no_valores():
    m = _serie_mensual()
    lag = ft.lag_publicacion(m, 1)
    assert (lag.values == m.values).all()
    assert lag.index[0] == pd.Timestamp("1990-02-01")


def test_macro_z_aplica_lag_antes_del_zscore():
    """El z-score del dato del mes M no puede existir antes de M+lag."""
    m = _serie_mensual(nombre="INDPRO")
    z = ft.macro_z({"INDPRO": m}, "INDPRO", "INDPRO_yoy_z", "nivel", lag_months=1)
    esperado = ft.causal_zscore(ft.lag_publicacion(m, 1))
    pd.testing.assert_series_equal(z, esperado.rename("INDPRO_yoy_z"))
    # y el primer z válido aparece 60 obs + 1 mes después del primer dato crudo
    assert z.first_valid_index() == m.index[59] + pd.DateOffset(months=1)


def test_alinear_diaria_es_asof_backward():
    grid = pd.bdate_range("2021-01-04", periods=10)
    # serie con hueco (festivo propio) y fechas fuera de la rejilla
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2021-01-04", "2021-01-06", "2021-01-12"]), name="a")
    out = ft.alinear_diaria(s.to_frame(), grid)["a"]
    assert out.loc["2021-01-05"] == 1.0  # arrastra el último <= t, no interpola
    assert out.loc["2021-01-11"] == 2.0  # nunca toma el 3.0 del 12 (futuro)
    assert out.loc["2021-01-12"] == 3.0


def test_alinear_diaria_asof_por_columna_recupera_columnas_fuera_de_rejilla():
    """Regresión del bug conocido de la v1 (ICSA fechado en sábado quedaba todo NaN)."""
    grid = pd.bdate_range("2021-01-04", periods=10)
    diaria = pd.Series(np.arange(10.0), index=grid, name="diaria")
    semanal = pd.Series([7.0, 8.0], index=pd.to_datetime(["2021-01-02", "2021-01-09"]), name="semanal")  # sábados
    panel = pd.concat([diaria, semanal], axis=1).sort_index()
    v1 = ft.alinear_diaria(panel, grid)
    assert v1["semanal"].isna().all()  # comportamiento v1 documentado (bug)
    ok = ft.alinear_diaria(panel, grid, asof_por_columna=True)
    assert ok.loc["2021-01-08", "semanal"] == 7.0
    assert ok.loc["2021-01-11", "semanal"] == 8.0
    pd.testing.assert_series_equal(ok["diaria"], v1["diaria"])


def test_edad_dias_diente_de_sierra_y_causal():
    m = _serie_mensual(n=24).to_frame("f")
    grid = pd.bdate_range("1990-01-01", "1991-12-31")
    out = ft.alinear_mensual_con_edad(m, grid)
    assert out.loc["1990-03-01", "f_edad_dias"] == 0
    assert out.loc["1990-03-30", "f_edad_dias"] == 29
    cut = "1991-02-15"
    trunc = ft.alinear_mensual_con_edad(m.loc[:cut], grid[grid <= cut])
    assert ft.truncation_max_abs_diff(out, trunc, cut).max() == 0.0


def test_construir_paneles_sintetico_causal_de_extremo_a_extremo():
    """Pipeline completo (features + rejilla + edad) sobre una mini-pista sintética."""
    raw = {
        "SP500": _serie_diaria(n=1500, seed=1, nombre="SP500"),
        "DGS10": _serie_diaria(n=1500, seed=2, nombre="DGS10", precio=False),
        "FF_FACTORS_3_DAILY": _serie_diaria(n=1500, seed=3, nombre="FF_FACTORS_3_DAILY", precio=False).diff(),
        "MOODYS_BAA": _serie_mensual(n=120, seed=4, nombre="MOODYS_BAA"),
        "MOODYS_AAA": _serie_mensual(n=120, seed=5, nombre="MOODYS_AAA"),
        "INDPRO": _serie_mensual(n=120, seed=6, nombre="INDPRO"),
    }
    raw["MOODYS_BAA"].index = pd.date_range("1998-01-01", periods=120, freq="MS")
    raw["MOODYS_AAA"].index = raw["MOODYS_BAA"].index
    raw["INDPRO"].index = raw["MOODYS_BAA"].index
    full = ft.construir_paneles(raw, "2001-01-02", "2005-09-30")
    assert {"SP500_ret_z", "DGS10_change_z", "FF_MKT_z"} <= set(full["diaria"].columns)
    assert {"credit_BaaAaa_mensual_z", "INDPRO_yoy_z", "INDPRO_yoy_z_edad_dias"} <= set(full["mensual"].columns)
    for cut in ["2004-06-15", "2005-03-31"]:  # tras el calentamiento (60 obs + yoy + lag)
        trunc = ft.construir_paneles(ft.truncar(raw, cut), "2001-01-02", "2005-09-30")
        for k in ("diaria", "mensual"):
            assert ft.truncation_max_abs_diff(full[k], trunc[k], cut).max() == 0.0, (k, cut)


def test_assert_causal_devuelve_tabla():
    raw = {"SP500": _serie_diaria(nombre="SP500")}
    tabla = ft.assert_causal(lambda r: ft.ret_z(r, "SP500", "SP500_ret_z"), raw, cut=CUTS[1])
    assert list(tabla.columns) == ["feature", "max_abs_diff", "causal_ok"]
    assert tabla["causal_ok"].all()
