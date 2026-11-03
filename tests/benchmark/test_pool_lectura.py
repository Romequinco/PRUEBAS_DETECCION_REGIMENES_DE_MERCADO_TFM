"""Tests de la lectura de la Fase 2 para los notebooks (``regimenes.benchmark.pool_lectura``).

Resultados a medias: tablas que faltan, tablas desfasadas, criterio (a) provisional y días en
crisis. Datos de juguete; los que leen ``results/`` locales llevan el marcador ``datos``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from regimenes import rutas
from regimenes.benchmark import pool_lectura as pl


def test_leer_tabla_falta_y_desfasada(tmp_path) -> None:
    assert pl.leer_tabla("criterio_adr010", tmp_path) is None
    assert pl.aviso_tabla("criterio_adr010.csv", None).startswith(pl.PENDIENTE)
    tabla = tmp_path / "criterio_adr010.csv"
    pd.DataFrame({"pista": ["A"], "cumple_pista": [False]}).to_csv(tabla, index=False)
    resultado = tmp_path / "A_D10-P.csv"
    pd.DataFrame({"pista": ["A"], "id": ["D10-P"]}).to_csv(resultado, index=False)
    os.utime(tabla, (1_000_000, 1_000_000))                  # la tabla es más antigua
    t = pl.leer_tabla("criterio_adr010", tmp_path)
    assert t is not None and t.attrs["desfasada"] and "AVISO" in pl.aviso_tabla("x", t)
    os.utime(resultado, (500_000, 500_000))
    t = pl.leer_tabla("criterio_adr010", tmp_path)
    assert not t.attrs["desfasada"] and pl.aviso_tabla("x", t) is None


def test_estado_y_comparacion_sin_resultados(tmp_path) -> None:
    est = pl.estado(tmp_path)
    assert list(est["pista"]) == ["A", "B"] and (est["hechos"] == 0).all()
    assert "D14-R" in est.loc[est["pista"] == "B", "pendientes"].iloc[0]
    assert "D14-R" not in est.loc[est["pista"] == "A", "pendientes"].iloc[0]
    comp = pl.comparacion(tmp_path)
    assert comp.empty and "a_mejora_par" in comp
    prov = pl.criterio_provisional(comp)
    assert list(prov["b_en_mcs"]) == [pl.PENDIENTE] * 2


def test_criterio_provisional_solo_cuenta_las_del_veredicto() -> None:
    comp = pd.DataFrame({"pista": ["B", "B", "B"], "id": ["D10-R", "D14-R", "D15-U"],
                         "a_mejora_par": [True, False, True], "cuenta_para_veredicto": [True, True, False]})
    prov = pl.criterio_provisional(comp).set_index("pista")
    assert prov.loc["B", "candidatas"] == "D10-R|D14-R" and prov.loc["B", "a_mejoran_par"] == "D10-R"
    assert prov.loc["A", "candidatas"] == ""


def test_dias_en_crisis() -> None:
    from regimenes.evaluacion import ranking as rk

    a, b = next(iter(rk.track_crisis_windows("B").values()))
    idx = pd.bdate_range(pd.Timestamp(a) - pd.Timedelta(days=60), pd.Timestamp(b) + pd.Timedelta(days=60))
    dentro = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
    sen = pd.DataFrame({"todo": np.ones(len(idx), bool), "dentro": dentro}, index=idx)
    d = pl.dias_en_crisis(sen, "B")
    assert d.loc["todo", "fraccion_oos"] == 1 and d.loc["dentro", "cobertura_dias_crisis"] == 1
    assert d.loc["dentro", "marcados_en_calma"] == 0 and d.loc["dentro", "dias_crisis"] == dentro.sum()


@pytest.mark.datos
def test_ranking_v3_con_v2_local() -> None:
    """Con las métricas versionadas de v2: 12 por pista en N ∪ U, puestos desde 1."""
    if not (rutas.RESULTS_BENCHMARK / "metrics" / "A_D01.csv").exists():
        pytest.skip("faltan las métricas de v2")
    r = pl.ranking_v3(salida=rutas.RESULTS_BENCHMARK / "_no_existe")
    assert set(r["menu"]) <= {"N", "U"}
    assert r.groupby("pista")["id"].nunique().to_dict() == {"A": 12, "B": 12}
    assert r.groupby("pista")["puesto_deteccion"].min().eq(1).all()
