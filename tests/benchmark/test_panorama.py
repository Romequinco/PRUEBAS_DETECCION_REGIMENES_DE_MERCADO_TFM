"""Tests del panorama del notebook 14 (``regimenes.benchmark.panorama``).

Ligeros: comprueban la forma de cada tabla y figura y que no aparecen ids que no existan en
``results/``. Leen los CSV versionados de ``results/`` (no los paneles OOS, que no se versionan);
si faltan, se saltan.
"""

from __future__ import annotations

import re
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from regimenes.benchmark import panorama as pn  # noqa: E402
from regimenes.benchmark import pool_lectura as pl  # noqa: E402

HAY_RESULTADOS = (pn.V2 / "metrics_master_v2.csv").exists() and (pn.POOL / "criterio_adr010.csv").exists()
necesita_resultados = pytest.mark.skipif(not HAY_RESULTADOS, reason="faltan results/ versionados")


# --------------------------------------------------------------------------- #
# Sin datos
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("det_id, menu, familia", [
    ("D10", "N", "F1"), ("D06", "U", "F5"), ("D10-R", "R", "F1"), ("D14-R", "R", "F1"),
    ("D15-U", "U", "F2"), ("D18-P", "P", "F2"), ("D16-P-excl", "P", "F8"), ("D17-N", "N", "F8"),
    ("D02-T", "T", "F1"), ("D08_T", "T", "F3"), ("VOL_P90", "baseline", "baseline"),
])
def test_menu_y_familia(det_id: str, menu: str, familia: str) -> None:
    assert pn.menu_de(det_id) == menu
    assert pn.familia_de(det_id) == familia
    assert pn.color_de(det_id).startswith("#")


def test_familias_y_paleta_completas() -> None:
    assert set(pn.FAMILIA_DE_BASE.values()) == set(pn.FAMILIAS) == set(pn.COLOR_FAMILIA)
    assert len(set(pn.COLOR_FAMILIA.values())) == len(pn.COLOR_FAMILIA)        # un color por familia
    assert set(pl.MENU_V2) <= set(pn.FAMILIA_DE_BASE)
    assert pn.grupo_de("D15-U") == pn.GRUPOS[0] and pn.grupo_de("D16-N") == pn.GRUPOS[3]


# --------------------------------------------------------------------------- #
# Con los resultados versionados
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def datos() -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v2, info = pn.cargar_v2()
        menus = pn.menus_principales()
        principal = pn.ranking_principal(menus["menus"], v2)
        todas = pn.metricas_todas(v2)
        base = pn.baselines()
    return {"v2": v2, "info": info, "menus": menus, "principal": principal, "todas": todas, "base": base}


def _ids_conocidos(d: dict) -> set[str]:
    ids = set(d["v2"]["id"]) | set(pl.BASELINES)
    for t in pn.PISTAS:
        ids |= set(pl.ids_esperados(t))
    return ids | set(pn.cargar_menu_t()["id"])


@necesita_resultados
def test_ranking_principal_reproduce_ranking_v3(datos) -> None:
    menus = datos["menus"]["menus"]
    r3 = pl.ranking_v3(menus=menus)
    r = datos["principal"]
    assert list(zip(r["pista"], r["id"], r["puesto_deteccion"])) == \
        list(zip(r3["pista"], r3["id"], r3["puesto_deteccion"]))
    assert set(pn.COLUMNAS_RANKING) <= set(r.columns)
    assert not r["id"].map(pn.es_supervisado).any()
    assert set(r["menu"]) <= set(menus)


@necesita_resultados
def test_inventario_sin_ids_inventados(datos) -> None:
    inv = pn.inventario(datos["todas"], datos["principal"])
    assert inv["id"].is_unique
    assert set(inv["id"]) <= _ids_conocidos(datos)
    assert set(inv["id"]) >= set(datos["principal"]["id"])
    assert inv.loc[inv["supervisado"], "en_ranking_principal"].eq(False).all()
    assert inv.loc[inv["menu"] == "T", "en_ranking_principal"].eq(False).all()
    resumen = pn.resumen_inventario(inv)
    assert int(resumen.drop(columns="ids").to_numpy().sum()) == len(inv)


@necesita_resultados
def test_gate_y_ventanas(datos) -> None:
    tabla, ok = pn.gate(datos["v2"], pn.cargar_pool(), pn.cargar_menu_t())
    assert list(tabla.index) == list(pn.PISTAS) and ok
    v = pn.ventanas_crisis(datos["v2"])
    assert set(v["estado"]) <= {"completa", "parcial", "fuera"}
    for t in pn.PISTAS:
        evaluadas = int((v[(v["pista"] == t)]["estado"] != "fuera").sum())
        assert evaluadas == int(datos["principal"].query("pista == @t")["det_n_eventos"].iloc[0])


@necesita_resultados
def test_matriz_crisis_forma(datos) -> None:
    for t in pn.PISTAS:
        det, cov = pn.matriz_crisis(datos["principal"], t)
        p = datos["principal"][datos["principal"]["pista"] == t]
        assert det.shape == cov.shape == (len(p), int(p["det_n_eventos"].iloc[0]))
        assert list(det.index) == list(p.sort_values("puesto_deteccion")["id"])
        assert set(det.stack().unique()) <= {0.0, 1.0}


@necesita_resultados
def test_tablas_pool_techo_y_menu_t(datos) -> None:
    tp = pn.tabla_pool(datos["principal"])
    assert set(tp["id"]) <= set(datos["principal"]["id"]) and not tp["id"].map(pn.es_supervisado).any()
    techo = pn.tabla_techo()
    assert set(techo["id"]) <= _ids_conocidos(datos) and techo["id"].map(pn.es_supervisado).all()
    tt = pn.tablas_menu_t()
    assert {"n_vs_t", "criterio_b", "azar", "media", "sin_bloque", "mcs_t"} <= set(tt)
    assert set(tt["n_vs_t"]["base"]) <= set(datos["v2"]["id"])


@necesita_resultados
def test_conclusiones_citan_ids_reales(datos) -> None:
    h = pn.hechos(datos["principal"], datos["base"], pn.tabla_pool(datos["principal"]),
                  datos["menus"]["criterio"], pn.tablas_menu_t(), pn.tabla_techo())
    textos = pn.conclusiones(h)
    assert len(textos) == 5
    citados = set(re.findall(r"\bD\d\d(?:-[A-Z](?:-excl)?)?\b", " ".join(textos)))
    conocidos = _ids_conocidos(datos)
    assert citados and citados <= conocidos | {pn.base_de(i) for i in conocidos}


@necesita_resultados
def test_figuras_dos_pistas(datos) -> None:
    r, todas, base = datos["principal"], datos["todas"], datos["base"]
    figuras = [
        pn.figura_resumen(pn.tabla_resumen(r, todas, base)),
        pn.figura_ranking(r, base),
        pn.figura_frontera(todas, r, base),
        pn.figura_crisis(r),
        pn.figura_estabilidad(r),
        pn.figura_pool(pn.tabla_pool(r), r, datos["menus"]["criterio"]),
        pn.figura_techo(pn.tabla_techo()),
    ]
    for fig in figuras:
        ejes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
        assert len(ejes) >= 2 and fig._suptitle is not None and fig._suptitle.get_text()
        plt.close(fig)
    fig = pn.figura_menu_t(pn.tablas_menu_t())
    assert len(fig.axes) == 4
    plt.close(fig)


@necesita_resultados
def test_loco_y_sensibilidad_forma(datos) -> None:
    m = pn.metricas_principal(datos["menus"]["menus"], datos["v2"])
    lo = pn.loco(m)
    assert set(lo.index) == set(zip(datos["principal"]["pista"], datos["principal"]["id"]))
    assert lo.min().min() >= 1
    s = pn.sensibilidad(m[m["menu"].isin(["N", "U"])])
    assert list(s.columns) == ["oficial", *pn.VARIANTES_CRITERIO]
    assert isinstance(s.index, pd.MultiIndex)
