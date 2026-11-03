"""Tests de la vara común de las fusiones (``regimenes.fusion.vara``, ADR-011 §2 y §5).

Orden del §5.2, niveles, mitades y deduplicación con datos de juguete; el de regresión (el orden
con todo el OOS reproduce el ``puesto_deteccion`` oficial) y el MCS leen los paneles locales y
llevan los marcadores ``datos`` y ``resultados``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimenes import rutas
from regimenes.fusion import vara


def _hay_paneles() -> bool:
    return all((vara.PANELES_V2 / f"{p}_D01.parquet").exists()
               and (vara.PANELES_POOL / f"{p}_D15-U.parquet").exists() for p in vara.PISTAS)


def test_rutas_de_resultados() -> None:
    assert rutas.RESULTS_FUSION_FIJAS == rutas.RESULTS / "fusion" / "fijas"
    assert rutas.RESULTS_FUSION_ONLINE == rutas.RESULTS / "fusion" / "online"
    assert {"RESULTS_FUSION_FIJAS", "RESULTS_FUSION_ONLINE"} <= set(rutas.__all__)


def test_candidatos() -> None:
    a, b = vara.candidatos("A"), vara.candidatos("b")
    assert len(a) == 15 and len(b) == 17 and len(set(b)) == 17
    assert a[:12] == [f"D{i:02d}" for i in range(1, 13)] and b[:15] == a
    assert not any(c.startswith(("D16", "D17")) for c in b)
    with pytest.raises(ValueError):
        vara.candidatos("C")


def test_mitades_corte_entero() -> None:
    idx = pd.bdate_range("2000-01-03", periods=11)
    h1, h2 = vara.mitades(idx)
    assert len(h1) == 5 and len(h2) == 6 and h1.append(h2).equals(idx)


def test_nivel_deteccion() -> None:
    base = {"switching_rate": 0.01, "det_precision": 0.5, "det_marked_rate": 0.1,
            "mean_regime_duration": 50.0, "det_mean_crisis_run": 20.0, "det_lift_precision": 2.0}
    assert vara.nivel_deteccion(base) == 0
    assert vara.nivel_deteccion({**base, "det_lift_precision": 1.0}) == 1
    assert vara.nivel_deteccion({**base, "det_mean_crisis_run": 2.5}) == 2
    assert vara.nivel_deteccion({**base, "mean_regime_duration": 4.0}) == 2
    assert vara.nivel_deteccion({**base, "switching_rate": 0.0}) == 3
    assert vara.nivel_deteccion({**base, "det_precision": np.nan}) == 3
    # la precedencia es la de rank_detection: degenerado antes que parpadeo y que azar
    assert vara.nivel_deteccion({**base, "switching_rate": 0.0, "det_lift_precision": 0.5}) == 3


def test_ordenar_redondeo_switching_e_id() -> None:
    t = pd.DataFrame({
        "nivel": [0, 0, 0, 0, 1],
        "score_deteccion": [0.6001, 0.6004, 0.6004, 0.59, 0.99],
        "switching_rate": [0.02, 0.01, 0.01, 0.0, 0.0],
    }, index=["E", "C", "B", "A", "Z"])
    o = vara.ordenar(t)
    # 0,6001 y 0,6004 empatan a 3 decimales → menor switching; B y C empatan en todo → id
    assert list(o.index) == ["B", "C", "E", "A", "Z"]
    assert list(o["puesto"]) == [1, 2, 3, 4, 5]


def test_evaluar_y_tramo_con_ventanas_de_juguete(monkeypatch) -> None:
    idx = pd.bdate_range("2001-01-01", periods=200)
    win = {"c1": ("2001-02-01", "2001-02-28"), "c2": ("2001-08-01", "2001-08-31")}
    monkeypatch.setattr(vara, "ventanas", lambda pista: win)
    dentro = pd.Series(False, index=idx)
    for a, b in win.values():
        dentro |= (idx >= a) & (idx <= b)
    sen = pd.DataFrame({"perfecta": dentro, "solo_c1": dentro & (idx < "2001-06-01"),
                        "nunca": False}, index=idx)
    e = vara.evaluar(sen, "A")
    assert e.loc["perfecta", "score_deteccion"] == 1.0 and e.index[0] == "perfecta"
    assert e.loc["solo_c1", "det_event_recall"] == 0.5 and e.loc["nunca", "nivel"] == 3
    h1, h2 = vara.mitades(idx)
    e2 = vara.evaluar(sen, "A", h2)
    # en H2 solo cuenta c2 (c1 sale como NaN): solo_c1 no marca nada en H2 → degenerada
    assert e2.loc["perfecta", "det_n_eventos"] == 1 and e2.loc["solo_c1", "nivel"] == 3
    assert (e2["n_sesiones"] == len(h2)).all()


def test_deduplicar_por_prioridad() -> None:
    idx = pd.bdate_range("2001-01-01", periods=6)
    s = pd.DataFrame({"F*": [1, 0, 1, 0, 0, 0], "D02": [1, 0, 1, 0, 0, 0], "BASE": [0, 0, 0, 0, 0, 0],
                      "V": [0, 0, 0, 0, 0, 0], "D01": [1, 1, 0, 0, 0, 0]}, index=idx).astype(bool)
    assert vara.deduplicar(s, ["D01", "D02", "BASE", "F*", "V"]) == {"F*": "D02", "V": "BASE"}
    assert vara.prioridad_por_defecto(["F*", "SIEMPRE_CALMA", "D02", "x"], "A") == \
        ["D02", "SIEMPRE_CALMA", "F*", "x"]


@pytest.mark.datos
@pytest.mark.resultados
@pytest.mark.skipif(not _hay_paneles(), reason="faltan los paneles OOS locales")
@pytest.mark.parametrize("pista,n,n_h1,primero,bloque", [("A", 14133, 7066, "D10", 66),
                                                           ("B", 4059, 2029, "D05", 50)])
def test_regresion_orden_oficial(pista, n, n_h1, primero, bloque) -> None:
    """§5.2/§8: el orden con todo el OOS reproduce el ``puesto_deteccion`` del v3 ampliado."""
    from regimenes.benchmark import pool_lectura as pl

    sen, p = vara.senales(pista), vara.probabilidades(pista)
    assert sen.shape == (n, len(vara.candidatos(pista))) and sen.dtypes.eq(bool).all()
    assert p.shape == sen.shape and p.index.equals(sen.index) and list(p.columns) == list(sen.columns)
    h1, h2 = vara.mitades(sen.index)
    assert len(h1) == n_h1 and len(h2) == n - n_h1
    assert vara.bloque(pista) == bloque

    r = pl.ranking_v3(menus=("N", "U", "P", "R"))
    oficial = r[r["pista"] == pista].set_index("id")
    e = vara.evaluar(sen, pista)
    assert set(e.index) == set(oficial.index)
    assert (e["puesto"] == oficial.loc[e.index, "puesto_deteccion"]).all()
    assert np.allclose(e["score_deteccion"], oficial.loc[e.index, "score_deteccion"], atol=1e-12)
    niveles = oficial["nivel_etiqueta"].map({"elegible": 0, "precision_no_supera_azar": 1,
                                             "parpadeo": 2, "degenerado": 3})
    assert (e["nivel"] == niveles.loc[e.index]).all()
    assert vara.mejor_individual(pista) == primero


@pytest.mark.datos
@pytest.mark.resultados
@pytest.mark.skipif(not _hay_paneles(), reason="faltan los paneles OOS locales")
def test_mcs_hereda_duplicados() -> None:
    """Una copia de un individual no entra en el MCS y hereda su resultado (§5.1)."""
    sen = vara.senales("B")[["D05", "D06", "D09"]]
    sen["COPIA"] = sen["D05"]
    t = vara.mcs(sen, "B", tramo=vara.tramos("B")["H2"], n_boot=200)
    assert t.loc["COPIA", "duplicado_de"] == "D05"
    assert t.loc["COPIA", "incluido"] == t.loc["D05", "incluido"]
    assert set(vara.BASELINES) <= set(t.index) and (t["bloque"] == 50).all()
    assert (t["n_sesiones"] == 2030).all()


@pytest.mark.datos
@pytest.mark.resultados
@pytest.mark.skipif(not _hay_paneles(), reason="faltan los paneles OOS locales")
@pytest.mark.parametrize("pista", ["A", "B"])
def test_mcs_reproduce_mcs_nupr(pista) -> None:
    """§5.1: con solo los individuales, :func:`vara.mcs` es el MCS publicado del ADR-010 §9.2
    (``results/detectores/pool/mcs_NUPR_<P>.csv``): misma pérdida, mismo p-valor, mismo veredicto."""
    ruta = rutas.RESULTS_DETECTORES / "pool" / f"mcs_NUPR_{pista}.csv"
    if not ruta.exists():
        pytest.skip(f"falta {ruta}")
    oficial = pd.read_csv(ruta, index_col="modelo")
    t = vara.mcs(vara.senales(pista), pista)
    assert set(t.index) == set(oficial.index) and t["duplicado_de"].isna().all()
    t = t.loc[oficial.index]
    assert np.allclose(t["perdida_media"], oficial["perdida_media"], rtol=0, atol=1e-12)
    assert np.allclose(t["p_mcs"], oficial["p_mcs"], rtol=0, atol=0)
    assert (t["incluido"].astype(bool) == oficial["incluido"].astype(bool)).all()
