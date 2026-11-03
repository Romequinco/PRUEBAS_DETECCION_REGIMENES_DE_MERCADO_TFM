"""Variantes detector × menú (ADR-008 §3) y su evaluación sobre un panel dado."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimenes.benchmark import cache
from regimenes.detectores.menus import ADMITIDOS, columnas_menu, variant_specs
from regimenes.detectores.registry import detector_specs


@pytest.mark.parametrize("pista, n_termometros", [("A", 7), ("B", 10)])
def test_menu_T_usa_un_termometro_por_bloque(pista, n_termometros):
    specs = variant_specs(pista, "T")
    assert [s.detector_id for s in specs] == [f"{d}-T" for d in ADMITIDOS["T"]]
    for s in specs:
        assert s.features == columnas_menu(pista, "T")
        assert len(s.features) == n_termometros
        if "features" in s.params:
            assert s.params["features"] == list(s.features)
    d02 = next(s for s in specs if s.detector_id == "D02-T")
    assert set(d02.params["signs"].values()) == {1}


def test_variantes_conservan_hiperparametros():
    originales = {s.detector_id: s for s in detector_specs("B")}
    for menu in ("U", "T"):
        for v in variant_specs("B", menu):
            o = originales[v.detector_id.split("-")[0]]
            assert (v.step, v.expanding, v.class_name) == (o.step, o.expanding, o.class_name)
            assert {k: x for k, x in v.params.items() if k not in ("features", "signs")} == \
                   {k: x for k, x in o.params.items() if k not in ("features", "signs")}


def test_menu_N_son_los_12_originales_y_U_es_el_retorno():
    assert variant_specs("A", "N") == detector_specs("A")
    assert all(s.features == ("SP500_ret",) for s in variant_specs("A", "U"))
    with pytest.raises(ValueError):
        variant_specs("A", "Z")


def test_menus_no_entran_en_la_huella_de_la_cache():
    rutas = {p.name for p in cache._cache_code_paths()}
    assert "menus.py" not in rutas and "variantes.py" not in rutas


def test_evaluar_variante_en_panel_sintetico():
    from regimenes.benchmark.variantes import evaluar_variante
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2007-04-11", "2016-12-30")
    cols = columnas_menu("B", "T")
    panel = pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=list(cols))
    panel["SP500_ret"] = rng.normal(0, 0.01, len(idx))
    spec = next(s for s in variant_specs("B", "T") if s.detector_id == "D02-T")
    metrics, wf = evaluar_variante("B", spec, panel, idx)
    assert metrics.loc[0, "id"] == "D02-T"
    assert {"score_deteccion", "det_precision", "det_event_recall"} <= set(metrics.columns)
    assert len(wf) == len(idx) - 252 * 3
