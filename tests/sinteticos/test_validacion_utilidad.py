"""Utilidad TSTR frente a TRTR (``regimenes.sinteticos.validacion.utilidad``).

Panel de juguete con el detector barato D01 (pista A: umbral sobre
``SP500_vol_z``) y las ventanas de crisis reales de la pista A
(``configs/benchmark_spec.yaml``, versionado): la volatilidad sube dentro de las
ventanas que caen en el tramo de juguete (LTCM 1998, dotcom 2000-02). Se prueba
el contrato de salida, el determinismo, la reutilizacion de la referencia TRTR
y que nunca se lee el real posterior a ``fin_train``. El caso con datos reales
va marcado ``datos``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimenes import rutas
from regimenes.sinteticos.validacion import utilidad
from regimenes.sinteticos.validacion.utilidad import referencia_trtr, utilidad_tstr

FIN_TRAIN = pd.Timestamp("2006-12-31")
TRAIN_DAYS = 500


def _ventanas_a() -> dict:
    from regimenes.evaluacion.ranking import track_crisis_windows

    return track_crisis_windows("A")


def _crisis(index: pd.DatetimeIndex) -> np.ndarray:
    dentro = np.zeros(len(index), dtype=bool)
    for ini, fin in _ventanas_a().values():
        dentro |= (index >= pd.Timestamp(ini)) & (index <= pd.Timestamp(fin))
    return dentro


def _panel(index: pd.DatetimeIndex, crisis: np.ndarray, rng) -> pd.DataFrame:
    n = len(index)
    vol = rng.normal(0.0, 0.5, n) + 2.5 * crisis
    ret = rng.normal(0.0, 0.01, n) * np.where(crisis, 2.5, 1.0) - 0.002 * crisis
    return pd.DataFrame({"SP500_vol_z": vol, "SP500_ret": ret}, index=index)


@pytest.fixture(scope="module")
def real() -> pd.DataFrame:
    idx = pd.bdate_range("1996-01-01", FIN_TRAIN)
    panel = _panel(idx, _crisis(idx), np.random.default_rng(0))
    panel["regime"] = _crisis(idx).astype(int)
    return panel


@pytest.fixture(scope="module")
def sintetico() -> list[pd.DataFrame]:
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2009-01-01", periods=700)
    trays = []
    for _ in range(4):
        reg = np.zeros(len(idx), dtype=int)
        for ini in rng.choice(np.arange(50, 600), size=3, replace=False):
            reg[ini:ini + 60] = 1
        tray = _panel(idx, reg.astype(bool), rng)
        tray["regime"] = reg
        trays.append(tray)
    return trays


@pytest.fixture(scope="module")
def trtr(real) -> dict:
    return referencia_trtr(real, "D01", pista="A", train_days=TRAIN_DAYS)


@pytest.fixture(scope="module")
def salida(real, sintetico, trtr) -> pd.DataFrame:
    return utilidad_tstr(real, sintetico, "D01", pista="A", n_trayectorias=3, semilla=7,
                         trtr=trtr, n_sims_nulo=50)


def test_contrato_de_salida(salida: pd.DataFrame) -> None:
    assert list(salida.columns) == utilidad.COLUMNAS
    assert list(salida["metrica"]) == [*utilidad.METRICAS, "score_nulo_p95", "score_trtr_fijo"]
    assert (salida["detector"] == "D01").all() and (salida["regimen"] == "todos").all()
    assert salida["en_banda"].isna().all()
    cuerpo = salida.set_index("metrica").loc[list(utilidad.METRICAS)]
    assert (cuerpo["banda_inf"] <= cuerpo["sintetico"] + 1e-12).all()
    assert (cuerpo["sintetico"] <= cuerpo["banda_sup"] + 1e-12).all()
    score = cuerpo.loc["score_deteccion"]
    assert np.isclose(score["cociente"], score["sintetico"] / score["real"])
    por_tray = salida.attrs["por_trayectoria"]
    assert len(por_tray) == 3 and por_tray["trayectoria"].is_unique


def test_senal_de_juguete_supera_al_nulo(salida: pd.DataFrame) -> None:
    t = salida.set_index("metrica")
    assert t.loc["det_event_recall", "real"] == 1.0
    assert t.loc["score_deteccion", "sintetico"] > t.loc["score_nulo_p95", "sintetico"]
    assert t.loc["score_deteccion", "real"] > t.loc["score_nulo_p95", "real"]


def test_determinista_y_reutiliza_trtr(real, sintetico, trtr, salida) -> None:
    otra = utilidad_tstr(real, sintetico, "D01", pista="A", n_trayectorias=3, semilla=7,
                         train_days=TRAIN_DAYS, n_sims_nulo=50)
    pd.testing.assert_frame_equal(otra, salida)
    pd.testing.assert_frame_equal(otra.attrs["por_trayectoria"], salida.attrs["por_trayectoria"])
    with pytest.raises(ValueError, match="trtr"):
        utilidad_tstr(real, sintetico, "D06", pista="A", trtr=trtr)


def test_dias_evaluados_son_los_oos_del_trtr(salida, trtr) -> None:
    dias = salida.attrs["dias_evaluados"]
    assert dias.equals(trtr["oos_index"])
    assert dias.max() <= FIN_TRAIN and len(dias) == len(trtr["flags"])
    # solo ventanas con algun dia evaluado
    assert set(trtr["ventanas"]) == {"ltcm_russia_1998", "dotcom_2000_02"}


def test_nunca_lee_el_real_posterior_a_fin_train(real, sintetico, monkeypatch) -> None:
    """Con real contaminado despues de fin_train, el resultado no cambia y ningun
    fit/predict del detector recibe fechas reales > fin_train."""
    from regimenes.detectores.f1_reglas.rule_vix_threshold import RuleVixThreshold

    vistos: list[pd.DatetimeIndex] = []
    fit0, pred0 = RuleVixThreshold.fit, RuleVixThreshold.predict_online

    def fit(self, X, *a, **k):
        vistos.append(pd.DatetimeIndex(X.index))
        return fit0(self, X, *a, **k)

    def pred(self, X, *a, **k):
        vistos.append(pd.DatetimeIndex(X.index))
        return pred0(self, X, *a, **k)

    monkeypatch.setattr(RuleVixThreshold, "fit", fit)
    monkeypatch.setattr(RuleVixThreshold, "predict_online", pred)

    post = pd.bdate_range(FIN_TRAIN + pd.Timedelta(days=1), "2008-06-30")
    veneno = pd.DataFrame({"SP500_vol_z": 1e6, "SP500_ret": -0.5, "regime": 1}, index=post)
    contaminado = pd.concat([real, veneno])
    limpia = utilidad_tstr(real, sintetico, "D01", pista="A", n_trayectorias=2,
                           train_days=TRAIN_DAYS, n_sims_nulo=20)
    vistos.clear()
    sucia = utilidad_tstr(contaminado, sintetico, "D01", pista="A", n_trayectorias=2,
                          train_days=TRAIN_DAYS, n_sims_nulo=20, fin_train=FIN_TRAIN)
    pd.testing.assert_frame_equal(limpia, sucia)
    sint_ini = sintetico[0].index.min()
    reales = [ix for ix in vistos if ix.max() < sint_ini]
    assert reales and all(ix.max() <= FIN_TRAIN for ix in reales)
    assert not any(ix.min() < sint_ini <= ix.max() for ix in vistos)  # nunca mezcla real y sintetico


def test_errores_claros(real, sintetico, trtr) -> None:
    sin_feature = [t.drop(columns=["SP500_vol_z"]) for t in sintetico]
    with pytest.raises(ValueError, match="SP500_vol_z"):
        utilidad_tstr(real, sin_feature, "D01", pista="A", trtr=trtr)
    solapado = [t.set_axis(pd.bdate_range("2006-06-01", periods=len(t))) for t in sintetico]
    with pytest.raises(ValueError, match="fechas"):
        utilidad_tstr(real, solapado, "D01", pista="A", trtr=trtr)
    with pytest.raises(ValueError, match="trayectorias"):
        utilidad_tstr(real, [], "D01", pista="A", trtr=trtr)
    with pytest.raises(ValueError, match="SP500_ret"):
        utilidad_tstr(real.drop(columns=["SP500_ret"]), sintetico, "D01", pista="A", trtr=trtr)


# --------------------------------------------------------------------------- datos reales

@pytest.mark.datos
def test_pista_a_real_bootstrap_d01() -> None:
    from regimenes.benchmark.cache import processed_available
    from regimenes.sinteticos import datos, persistencia

    if not processed_available("A") or not any(rutas.DATA_RAW.glob("*/SP500.parquet")):
        pytest.skip("faltan data/processed (pista A) o data/raw/<fuente>/SP500.parquet")
    try:
        trays = persistencia.cargar_trayectorias("bootstrap_regimen", "A")
    except FileNotFoundError:
        pytest.skip("faltan las trayectorias de bootstrap_regimen (notebook 15)")
    panel, reg = datos.cargar_entrenamiento("A", "2006-12-31")
    out = utilidad_tstr(panel.assign(regime=reg), trays, "D01", pista="A", n_trayectorias=2,
                        n_sims_nulo=50)
    assert out.attrs["dias_evaluados"].max() <= pd.Timestamp("2006-12-31")
    t = out.set_index("metrica")
    assert np.isfinite(t.loc["score_deteccion", ["real", "sintetico"]].astype(float)).all()
    assert 0 < t.loc["score_deteccion", "real"] <= 1


def test_unir_utilidad_concatena_detectores(real, sintetico):
    from regimenes.sinteticos.validacion import unir_utilidad

    a = utilidad_tstr(real, sintetico, "D01", n_trayectorias=2, train_days=TRAIN_DAYS)
    b = a.assign(detector="DXX")
    b.attrs = dict(a.attrs)
    unida, por_tray = unir_utilidad([a, b])
    assert len(unida) == 2 * len(a) and not unida.attrs
    assert set(por_tray["detector"]) == {"D01", "DXX"}
