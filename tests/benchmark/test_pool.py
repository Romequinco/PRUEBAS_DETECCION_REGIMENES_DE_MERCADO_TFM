"""Tests del ejecutor de la Fase 2 (``regimenes.benchmark.pool``, ADR-010).

Índice igual al de v2, NaN en el OOS, evaluación de una variante de juguete con su tabla de
reajustes, extrapolación del piloto, huella y veredicto del §5. Datos de juguete; el único test
con datos locales lleva el marcador ``datos``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from regimenes import rutas
from regimenes.benchmark import pool as bp
from regimenes.detectores.pool.especificaciones import SpecPool

D10 = "regimenes.detectores.f1_reglas.turbulence_mahalanobis.TurbulenceMahalanobis"


def _spec(menu: str = "P", cols: tuple[str, ...] = ("x0", "x1", "x2", "tarde")) -> SpecPool:
    return SpecPool(detector_id=f"D10-{menu}", label="juguete", family="Reglas", module="disponibilidad",
                    class_name="ConDisponibilidad", features=cols,
                    params={"base": D10, "params": {"cov_min_periods": 40, "min_dwell": 3}, "min_obs": 100,
                            "predecir_con_nan": True}, menu=menu)


def _panel(n: int = 520, tarde: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2007-01-01", periods=n)
    vol = np.where((np.arange(n) % 150) > 120, 3.0, 1.0)
    p = pd.DataFrame(rng.normal(size=(n, 4)) * vol[:, None], index=idx, columns=["x0", "x1", "x2", "tarde"])
    p.iloc[:tarde, 3] = np.nan
    p["SP500_ret"] = p["x0"] * 0.01
    return p


# --------------------------------------------------------------------------- #
# Índice y NaN
# --------------------------------------------------------------------------- #
def test_indice_igual_a_v2_o_falla() -> None:
    idx = pd.bdate_range("2000-01-03", periods=900)
    train = bp.DEFAULT_TRAIN_DAYS["B"]
    info = bp.comprobar_indice("B", idx, idx[train:])
    assert info["n_oos"] == 900 - train and info["mismo_oos_que_v2"]
    with pytest.raises(ValueError, match="no es el de v2"):
        bp.comprobar_indice("B", idx, idx[train:].delete(5))


def test_nan_en_oos_falla_salvo_en_r() -> None:
    train = bp.DEFAULT_TRAIN_DAYS["B"]
    X = pd.DataFrame({"a": np.arange(900.0), "b": np.arange(900.0)}, index=pd.bdate_range("2000-01-03", periods=900))
    X.iloc[:100, 1] = np.nan                                    # calentamiento: se permite
    info = bp.informe_nan("B", _spec("P", ("a", "b")), X)
    assert info["nan_calentamiento"] == 100 and info["columnas_que_empiezan_tarde"] == {"b": str(X.index[100].date())}
    X.iloc[train + 3, 0] = np.nan                               # OOS
    with pytest.raises(ValueError, match="NaN dentro del OOS"):
        bp.informe_nan("B", _spec("P", ("a", "b")), X)
    assert bp.informe_nan("B", _spec("R", ("a", "b")), X)["nan_oos_por_columna"] == {"a": 1}


# --------------------------------------------------------------------------- #
# Evaluación de una variante de juguete
# --------------------------------------------------------------------------- #
def test_evaluar_pool_juguete() -> None:
    panel = _panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics, wf, reaj = bp.evaluar_pool("B", _spec(), panel, panel.index, train_days=200)
    assert pd.DatetimeIndex(wf.index).equals(panel.index[200:])
    assert len(reaj) == wf["fold"].nunique() == metrics.loc[0, "n_reajustes"]
    assert "tarde" not in reaj.loc[0, "columnas_entran"]           # 50 datos < 100
    assert "tarde" in reaj.loc[3, "columnas_entran"]               # 263 filas: 113 datos
    assert {"base", "menu", "id", "score_deteccion", "orden_canonico"} <= set(metrics) | set(reaj)
    assert metrics.loc[0, "base"] == "D10" and metrics.loc[0, "menu"] == "P"
    panel.iloc[300, 1] = np.nan
    with pytest.raises(ValueError, match="NaN dentro del OOS"):
        bp.evaluar_pool("B", _spec(), panel, panel.index, train_days=200)


def test_recolector_resume_cada_reajuste() -> None:
    contador = iter(range(100))

    class Falso:
        _canonical_order = np.array([1, 0])

        def __init__(self) -> None:
            self.k = next(contador)

        def resumen_reajuste(self) -> dict:
            return {"k": self.k}

    rec = bp.Recolector(Falso)
    for _ in range(3):
        rec()
    tabla = rec.cerrar()
    assert tabla["k"].tolist() == [0, 1, 2] and tabla["orden_canonico"].tolist() == ["1|0"] * 3


# --------------------------------------------------------------------------- #
# Piloto, huella y veredicto
# --------------------------------------------------------------------------- #
def test_coste_extrapolado_lineal() -> None:
    # coste por reajuste = 1 + 0,01 · tamaño del train
    tamanos = [100 + 10 * i for i in range(20)]
    est = bp.coste_extrapolado(3 * 2.1, 110, 3, 3 * 3.8, 280, 3, tamanos)
    esperado = sum(1 + 0.01 * m for m in tamanos) + 1 + 0.01 * 290
    assert est == pytest.approx(esperado, rel=1e-9)


def test_huella_cambiada_falla(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        bp.comprobar_huella({}, "B", tmp_path)
    prep = {"huella_codigo": {**bp.huella_codigo(), "src/regimenes/benchmark/pool.py": "otro"}}
    with pytest.raises(ValueError, match="cambió"):
        bp.comprobar_huella(prep, "B", tmp_path)
    bp.comprobar_huella({"huella_codigo": bp.huella_codigo()}, "B", tmp_path)


def _mcs(incluidos: dict[str, bool]) -> pd.DataFrame:
    return pd.DataFrame({"incluido": list(incluidos.values())}, index=pd.Index(list(incluidos), name="modelo"))


def _comp(filas: list[tuple[str, str, bool]]) -> pd.DataFrame:
    return pd.DataFrame([{"pista": p, "id": i, "a_mejora_par": m, "cuenta_para_veredicto": not i.startswith("D15")}
                         for p, i, m in filas])


def test_criterio_exige_a_y_b_en_las_dos_pistas() -> None:
    comp = _comp([("A", "D10-P", True), ("A", "D18-P", False), ("B", "D14-R", True), ("B", "D10-P", False),
                  ("B", "D15-U", True)])
    mcs = {"A": _mcs({"D10-P": True, "D05": True, "VOL_P90": True}),
           "B": _mcs({"D14-R": False, "D10-P": True, "D15-U": True, "VOL_P90": False})}
    out = bp.criterio(comp, mcs).set_index("pista")
    assert out.loc["A", "cumple_pista"] and not out.loc["B", "cumple_pista"]   # D15 no cuenta
    assert out.loc["A", "veredicto"] == "indicio (una sola pista)"
    assert out.loc["B", "vol_p90_fuera_del_mcs"] and not out.loc["A", "vol_p90_fuera_del_mcs"]
    mcs["B"] = _mcs({"D14-R": True})
    assert bp.criterio(comp, mcs).loc[0, "veredicto"] == "aporta"
    comp.loc[comp["id"] == "D10-P", "a_mejora_par"] = False
    assert bp.criterio(comp, mcs).loc[0, "veredicto"] == "indicio (una sola pista)"
    comp.loc[comp["id"] == "D14-R", "a_mejora_par"] = False
    assert bp.criterio(comp, mcs).loc[0, "veredicto"].startswith("no aporta")


def test_ids_base_y_menu() -> None:
    assert bp.base_de("D10-P") == "D10" and bp.menu_de("D14-R") == "R" and bp.menu_de("D09") == "N"
    assert bp.es_supervisado("D16-P") and not bp.es_supervisado("D18-P")
    assert bp.PARES == {"D10-P": "D10", "D10-R": "D10", "D14-R": "D10", "D18-P": "D09"}


# --------------------------------------------------------------------------- #
# Con datos locales
# --------------------------------------------------------------------------- #
@pytest.mark.datos
def test_indice_oficial_b_es_el_de_v2() -> None:
    """Con datos locales: el índice común del panel oficial (con R) da el OOS de v2 en B."""
    if not (rutas.DATA_PROCESSED / "pistaB_diaria.parquet").exists() or not (
            rutas.RESULTS_BENCHMARK / "panels" / "B_D02.parquet").exists():
        pytest.skip("faltan data/processed o los paneles de v2")
    panel, indice, _ = bp.construir_panel("B")
    assert bp.comprobar_indice("B", indice, bp.indice_v2("B"))["mismo_oos_que_v2"]
