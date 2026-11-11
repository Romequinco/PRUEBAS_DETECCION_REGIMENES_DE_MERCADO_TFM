"""Veredicto de admision: reglas de ``configs/sinteticos.yaml`` sobre tablas sinteticas de juguete."""
from __future__ import annotations

import pandas as pd
import pytest
import yaml

from regimenes import rutas
from regimenes.sinteticos.validacion import veredicto

CFG = yaml.safe_load(rutas.SINTETICOS_CONFIG.read_text(encoding="utf-8"))["validacion"]
COLS = ["regimen", "columna", "tipo", "metrica", "real", "sintetico", "banda_inf", "banda_sup", "cociente", "en_banda"]


def _fila(regimen, columna, tipo, metrica, sint=1.0, cociente=1.0, en_banda=True, banda_sup=1.0, real=1.0):
    return dict(zip(COLS, [regimen, columna, tipo, metrica, real, sint, 0.0, banda_sup, cociente, en_banda]))


def _tablas(cociente_desv=1.0, auc=0.6, cociente_nn=1.2, tstr=0.5):
    marg, dep = [], []
    for r in ("calma", "crisis"):
        for c in ("SP500_ret", "FF_MKT_z"):
            marg.append(_fila(r, c, "modelada", "desviacion", cociente=cociente_desv, en_banda=False))
        marg.append(_fila(r, "SP500_drawdown", "re-derivada", "desviacion", cociente=3.0, en_banda=False))
        for q in ("q01", "q99"):
            marg.append(_fila(r, "SP500_ret", "modelada", q))
        dep += [_fila(r, "SP500_ret", "modelada", m) for m in CFG["reglas"]["dependencia"]["ret"]]
        dep += [_fila(r, "persistentes", "—", m) for m in CFG["reglas"]["dependencia"]["persistentes"]]
        dep.append(_fila(r, "mensuales", "—", "escalones", cociente=0.0, en_banda=False))
    corr = [_fila(r, "—", "—", "frobenius", sint=0.15, banda_sup=0.1, cociente=float("nan")) for r in ("calma", "crisis")]
    cond = [_fila("todos", "SP500_ret", "modelada", "separacion_vol", cociente=1.05)]
    disc = [_fila("todos", "—", "—", "auc", sint=auc, cociente=float("nan"), en_banda=pd.NA)]
    mem = [_fila(r, "modeladas", "modelada", "cociente_nn", cociente=cociente_nn) for r in ("todos", "calma", "crisis")]
    util = []
    for det in ("D01", "D03", "D06"):
        util.append({**_fila("todos", "—", "—", "score_deteccion", sint=tstr, real=0.55), "detector": det})
        util.append({**_fila("todos", "—", "—", "score_nulo_p95", sint=0.40), "detector": det})
    df = lambda filas: pd.DataFrame(filas).astype({"en_banda": "boolean"})
    return {"marginal": df(marg), "dependencia": df(dep), "correlaciones": df(corr), "condicionamiento": df(cond),
            "discriminador": df(disc), "memorizacion": df(mem), "utilidad": df(util)}


def _veredicto(tablas, **kw):
    return veredicto(tablas, CFG["umbrales"], reglas=CFG["reglas"], niveles=CFG["niveles"], **kw)


def test_generador_que_cumple_todo_es_apto_en_ambos_niveles():
    v = _veredicto({"bueno": _tablas()})
    assert bool(v.loc["bueno", "apto_laboratorio"]) and bool(v.loc["bueno", "apto_aumento"])
    # las filas no clave (re-derivadas, escalones) no deciden
    assert v.loc["bueno", "fallos_marginal"] == ""


def test_cada_criterio_falla_por_su_umbral():
    v = _veredicto({
        "escala": _tablas(cociente_desv=1.4),
        "separable": _tablas(auc=0.9),
        "copia": _tablas(cociente_nn=0.5),
        "inutil": _tablas(tstr=0.3),
    })
    assert not v.loc["escala", "marginal"] and not v.loc["escala", "apto_laboratorio"]
    assert not v.loc["separable", "discriminador"]
    assert not v.loc["copia", "memorizacion"] and not v.loc["copia", "apto_aumento"]
    assert bool(v.loc["copia", "apto_laboratorio"])
    assert not v.loc["inutil", "utilidad"]


def test_correlaciones_con_factor_sobre_p95():
    assert bool(_veredicto({"g": _tablas()}).loc["g", "correlaciones"])   # 0,15 <= 2 x 0,10


def test_criterios_fuera_de_los_niveles_son_informativos():
    v = _veredicto({"g": _tablas(tstr=0.0)})
    assert not v.loc["g", "utilidad"]          # se calcula y se publica ...
    for nivel, criterios in CFG["niveles"].items():
        if "utilidad" not in criterios:        # ... pero no decide ningun nivel que no la use
            assert "utilidad" not in criterios


def test_faltan_filas_clave_es_error():
    t = _tablas()
    t["dependencia"] = t["dependencia"].iloc[:2]
    with pytest.raises(ValueError):
        _veredicto({"g": t})
