"""Tests ligeros de ``regimenes.fusion.figuras_fijas`` (notebook 15): utilidades puras con datos de
juguete y, si están los resultados de ``results/fusion/fijas/``, que las tablas y figuras se
construyen y cuadran con el criterio."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from regimenes.fusion import fijas as fj
from regimenes.fusion import figuras_fijas as ff

HAY_RESULTADOS = (fj.SALIDA / "criterio.csv").exists() and (fj.SALIDA / "senales_A.parquet").exists()
con_resultados = pytest.mark.skipif(not HAY_RESULTADOS, reason="faltan results/fusion/fijas/")


def test_num_coma_decimal() -> None:
    assert ff.num(0.5966) == "0,597"
    assert ff.num(0.36, 2) == "0,36"
    assert ff.num(np.nan) == "—"


@pytest.mark.parametrize("i, fila", [("PAR_D07>D08", "a"), ("VOTO_3de5", "b"), ("OR_D01+D02", "c_or"),
                                     ("AND_D01+D10-P", "c_and"), ("D10-P", "individual")])
def test_fila_de(i: str, fila: str) -> None:
    assert ff.fila_de(i) == fila


def test_color_de_fusion_e_individual() -> None:
    from regimenes.benchmark import panorama as pn

    assert ff.color_de("AND_D01+D10") == ff.COLOR_FUSION["c_and"]
    assert ff.color_de("D10-P") == pn.COLOR_FAMILIA["F1"]


def test_rachas() -> None:
    idx = pd.bdate_range("2020-01-01", periods=8)
    s = pd.Series([0, 1, 1, 0, 0, 1, 0, 1], index=idx).astype(bool)
    r = ff._rachas(s)
    assert r == [(idx[1], idx[2]), (idx[5], idx[5]), (idx[7], idx[7])]
    assert ff._rachas(pd.Series(False, index=idx)) == []


def _ev_juguete() -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = ["D01", "D02", "D03"]
    filas, cat = [], [{"pista": "A", "grupo": "individual", "id": i} for i in ids]
    for k, i in enumerate(ids):
        filas.append({"pista": "A", "tramo": "oos", "grupo": "individual", "id": i, "score_deteccion": 0.5 + k / 10,
                      "nivel": 0})
    for x in ids:
        for y in ids:
            if x != y:
                filas.append({"pista": "A", "tramo": "oos", "grupo": "principal", "id": f"PAR_{x}>{y}",
                              "score_deteccion": 0.4, "nivel": 0})
    for a, b in (("D01", "D02"), ("D01", "D03"), ("D02", "D03")):
        filas.append({"pista": "A", "tramo": "oos", "grupo": "principal", "id": f"OR_{a}+{b}",
                      "score_deteccion": 0.3, "nivel": 0})
        filas.append({"pista": "A", "tramo": "oos", "grupo": "principal", "id": f"AND_{a}+{b}",
                      "score_deteccion": 0.2, "nivel": 1})
    return pd.DataFrame(filas), pd.DataFrame(cat)


def test_matriz_parejas_orientacion() -> None:
    ev, cat = _ev_juguete()
    sc, nv = ff.matriz_parejas(ev, cat, "A", "a")
    assert sc.loc["D03", "D03"] == pytest.approx(0.7)          # diagonal = individual
    assert sc.loc["D01", "D02"] == pytest.approx(0.4)          # fila alerta, columna confirmador
    sc, nv = ff.matriz_parejas(ev, cat, "A", "c")
    assert sc.loc["D01", "D03"] == pytest.approx(0.3)          # triángulo superior = OR
    assert sc.loc["D03", "D01"] == pytest.approx(0.2)          # triángulo inferior = AND
    assert nv.loc["D03", "D01"] == 1


@con_resultados
def test_tablas_cuadran_con_el_criterio() -> None:
    d = ff.cargar()
    crit = d["criterio"].set_index("pista")
    t = ff.tabla_criterio(d["criterio"])
    for p in ff.PISTAS:
        assert t.loc[(p, "**pista**"), "cumple"] == ("✓" if crit.loc[p, "cumple_pista"] else "✗")
    rec = ff.tabla_recuento(d["catalogo"])
    assert rec.loc["total principal (entra en la selección de F*)", "A"] == 423
    assert rec.loc["total principal (entra en la selección de F*)", "B"] == 547
    h = ff.hechos(d)
    assert h["veredicto"] == crit["veredicto"].iloc[0]
    assert len(ff.conclusiones(h)) >= 4
    sel = ff.tabla_seleccion(d)
    for p in ff.PISTAS:
        f = sel[(sel["pista"] == p) & (sel["papel"] == "F* (criterio)")].iloc[0]
        assert f["score_oos"] == pytest.approx(crit.loc[p, "score_f_star_oos"])
        tr = ff.tramos_pista(p)
        assert tr["sesiones"].sum() == len(ff.indice(p))


@con_resultados
def test_figuras_se_dibujan() -> None:
    import matplotlib.pyplot as plt

    d = ff.cargar()
    figs = [ff.figura_resumen(d["evaluacion"], d["criterio"], d["optimismo"]),
            ff.figura_mapas(d["evaluacion"], d["catalogo"], d["criterio"]),
            ff.figura_h1_h2(d["evaluacion"], d["criterio"]),
            ff.figura_seleccion(ff.tabla_seleccion(d)),
            ff.figura_mcs(d["mcs"], d["criterio"]),
            ff.figura_cronograma(d["criterio"], d["eleccion"], "B")]
    for f in figs:
        assert f.axes
        plt.close(f)
