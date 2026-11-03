"""D15 · jump model de Shu (ADR-010 §3.1): historia previa causal, prefijo, λ y semilla."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jumpmodels")

from regimenes.detectores.pool import shu
from regimenes.detectores.pool.shu import ShuJumpModel, elegir_lambda, sharpe_estrategia

LAMBDAS = (5.0, 50.0, 220.0)


def _crudos(n: int = 900, semilla: int = 0) -> tuple[pd.Series, pd.Series]:
    """Precio con un tramo bajista volátil en medio y un T-bill del 2 %."""
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range("2000-01-03", periods=n)
    mu = np.full(n, 0.0006)
    sd = np.full(n, 0.008)
    mu[300:420], sd[300:420] = -0.003, 0.025
    mu[650:720], sd[650:720] = -0.002, 0.02
    precios = pd.Series(100 * np.exp(np.cumsum(rng.normal(mu, sd))), index=fechas)
    tipos = pd.Series(2.0, index=fechas[5:])            # empieza más tarde: corta las features
    return precios, tipos


def _detector(precios, tipos, **kw) -> ShuJumpModel:
    kw = {"lambdas": LAMBDAS, "ventana_ajuste": 300, "validacion_max": 200, "n_init": 3,
          "precios": precios, "tipos": tipos, **kw}
    return ShuJumpModel(**kw)


def _X(precios: pd.Series, desde: int = 0) -> pd.DataFrame:
    ret = np.log(precios / precios.shift(1))
    return ret.iloc[desde:].to_frame("SP500_ret")


def test_features_empiezan_con_dtb3_y_son_las_del_paper():
    precios, tipos = _crudos()
    base = shu.base_exceso(precios, tipos)
    assert base.index[0] > tipos.index[0]                 # rf de t usa el tipo de t − 1
    np.testing.assert_allclose(base["exceso"], base["ret"] - 0.02 / 252)
    f = shu.features_shu(base["exceso"])
    assert list(f.columns) == list(shu.FEATURES)
    neg = np.minimum(base["exceso"], 0) ** 2
    dd60 = np.sqrt(neg.ewm(halflife=60).mean())
    np.testing.assert_allclose(f["sortino_60"], (base["exceso"].ewm(halflife=60).mean() / dd60).loc[f.index])


def test_historia_previa_sin_dias_posteriores_a_t():
    precios, tipos = _crudos()
    X = _X(precios, desde=400)                           # la pista empieza después de la historia
    fin = X.index[200]
    a = _detector(precios, tipos).fit(X.loc[:fin])
    # Mismo ajuste si los crudos posteriores a t no existen o son basura.
    p2 = precios.copy()
    p2.loc[p2.index > fin] = p2.loc[p2.index > fin] * 3.0
    b = _detector(p2, tipos).fit(X.loc[:fin])
    c = _detector(precios.loc[:fin], tipos.loc[:fin]).fit(X.loc[:fin])
    for otro in (b, c):
        assert otro.lambda_elegida == a.lambda_elegida
        assert otro.sharpe_validacion == a.sharpe_validacion
        np.testing.assert_allclose(otro._model.centers_, a._model.centers_)
    # La historia previa a la pista sí se usa: n supera las filas de la pista.
    assert a.n_historia > len(X.loc[:fin])
    assert a.fin_ajuste == fin
    assert a.inicio_ajuste == a._datos_ajuste.index[0]
    assert len(a._datos_ajuste) == min(300, a.n_historia)


def test_validacion_y_ventana_de_ajuste():
    precios, tipos = _crudos()
    X = _X(precios)
    det = _detector(precios, tipos).fit(X.iloc[:700])
    assert det.n_validacion == min(200, det.n_historia // 2)
    assert set(det.sharpe_validacion) == set(LAMBDAS)
    assert det.lambda_elegida in LAMBDAS


def test_prefijo_y_futuro_no_cambian_la_prediccion():
    precios, tipos = _crudos()
    X = _X(precios)
    det = _detector(precios, tipos).fit(X.iloc[:600])
    det.label_states_economically(X.iloc[:600], market_returns=X["SP500_ret"].iloc[:600])
    todo = det.predict_online(X.iloc[350:])
    corto = det.predict_online(X.iloc[350:700])
    np.testing.assert_array_equal(todo[:350], corto)
    # El inicio de la ventana pedida no cambia la etiqueta (el filtrado arranca en el ajuste).
    np.testing.assert_array_equal(det.predict_online(X.iloc[600:700]), corto[250:])
    # Cambiar el futuro de los crudos no cambia el pasado.
    p2 = precios.copy()
    p2.iloc[750:] = p2.iloc[750:] * 0.5
    det._precios = p2
    np.testing.assert_array_equal(det.predict_online(X.iloc[350:700]), corto)
    proba = det.predict_proba(X.iloc[350:700])
    np.testing.assert_array_equal(proba[:, det.crisis_state], (corto == 1).astype(float))


def test_semilla_fija_y_estado_de_crisis():
    precios, tipos = _crudos()
    X = _X(precios)
    a = _detector(precios, tipos).fit(X.iloc[:600])
    b = _detector(precios, tipos).fit(X.iloc[:600])
    assert a.lambda_elegida == b.lambda_elegida and a.sharpe_validacion == b.sharpe_validacion
    np.testing.assert_array_equal(a.predict_online(X), b.predict_online(X))
    assert a.n_states == 2 and a.crisis_state == 1
    # El tramo bajista plantado (filas 300-420) es sobre todo crisis.
    estados = a.predict_online(X.iloc[:600])
    assert estados[330:420].mean() > estados[:280].mean()
    assert isinstance(a.discrepa_regla_paper, bool)


def test_sharpe_retraso_coste_y_empates():
    idx = pd.RangeIndex(6)
    exceso = pd.Series([0.01, -0.02, 0.01, 0.01, -0.03, 0.02], index=idx)
    nunca = sharpe_estrategia(pd.Series(0, index=idx), exceso, desde=0, coste=0.0)
    assert nunca == pytest.approx(exceso.mean() / exceso.std() * np.sqrt(252))
    # Señal de crisis en t = 2: fuera del mercado desde t = 4; un cambio cuesta 10 pb.
    senal = pd.Series([0, 0, 1, 1, 1, 1], index=idx)
    r = pd.Series([0.01, -0.02, 0.01, 0.01, 0.0 - 0.001, 0.0])
    assert sharpe_estrategia(senal, exceso, desde=0) == pytest.approx(r.mean() / r.std() * np.sqrt(252))
    assert sharpe_estrategia(pd.Series(1, index=idx), exceso, desde=3) == 0.0
    assert elegir_lambda({5.0: 1.0, 50.0: 1.0, 220.0: 0.5}) == 50.0
    assert elegir_lambda({5.0: 1.2, 50.0: 1.0}) == 5.0


@pytest.mark.datos
def test_historia_real_empieza_con_dtb3():
    if not (shu.RUTA_SP500.exists() and shu.RUTA_DTB3.exists()):
        pytest.skip("faltan data/raw/yfinance/SP500.parquet o data/raw/fred/DTB3.parquet")
    precios, tipos = shu._crudos(str(shu.RUTA_SP500), str(shu.RUTA_DTB3))
    base = shu.base_exceso(precios.loc[:"1960-12-31"], tipos.loc[:"1960-12-31"])
    assert base.index[0] >= pd.Timestamp("1954-01-04")
    assert base.index[-1] <= pd.Timestamp("1960-12-31")


def test_resumen_reajuste_guarda_lambda_y_discrepancia():
    """El ejecutor guarda por reajuste la λ elegida y si la regla del paper discrepa (§3, §3.1)."""
    precios, tipos = _crudos()
    X = _X(precios)
    assert _detector(precios, tipos).resumen_reajuste() == {}
    det = _detector(precios, tipos).fit(X.iloc[:600])
    r = det.resumen_reajuste()
    assert r["lambda_elegida"] == det.lambda_elegida and r["fin_train"] == X.index[599]
    assert r["discrepa_regla_paper"] is det.discrepa_regla_paper
    assert len(r["sharpe_validacion"].split("|")) == len(LAMBDAS)
    assert r["orden_canonico"] in {"0|1", "1|0"}
