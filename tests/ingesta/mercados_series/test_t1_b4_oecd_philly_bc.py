"""Bloque 4: OCDE (ediciones), Philadelphia Fed RTDSM y bancos centrales, sin red (fixtures sintéticas
con el formato real de cada fuente) y una prueba ``live``."""
from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from ingesta.mercados_series import catalogo
from ingesta.mercados_series.contrato import validar_series
from ingesta.mercados_series.fuentes import b4_comun, bancos_centrales, oecd, philly_rtdsm


@pytest.fixture(scope="module")
def pubs():
    return b4_comun.publicaciones_b4()


# --- Philly RTDSM ----------------------------------------------------------------------------------
def test_siglo_employ64_es_1964_regresion():
    """Regresión: la regla «yy < 65 → 20yy» convertía EMPLOY64M12 en 2064."""
    assert philly_rtdsm.anios_vintage(["EMPLOY64M12", "EMPLOY65M1", "EMPLOY65M2"]) == [1964, 1965, 1965]
    assert philly_rtdsm.fechas_vintage(["EMPLOY64M12"])[0] == pd.Timestamp("1964-12-31")


def test_siglo_cambia_al_pasar_de_99_a_00():
    cols = ["ROUTPUT99Q3", "ROUTPUT99Q4", "ROUTPUT00Q1", "ROUTPUT26Q3"]
    assert philly_rtdsm.anios_vintage(cols) == [1999, 1999, 2000, 2026]
    assert philly_rtdsm.anios_vintage(["PCPI98M11", "PCPI26M9"]) == [1998, 2026]


def test_fecha_vintage_trimestral_es_fin_del_mes_central():
    f = philly_rtdsm.fechas_vintage(["ROUTPUT65Q4", "ROUTPUT66Q1"])
    assert list(f) == [pd.Timestamp("1965-11-30"), pd.Timestamp("1966-02-28")]


def test_columna_invalida():
    with pytest.raises(ValueError):
        philly_rtdsm.partes_columna("ROUTPUT65Q5")
    with pytest.raises(ValueError):
        philly_rtdsm.partes_columna("DATE")


def _matriz_routput():
    return pd.DataFrame({
        "DATE": ["1965:Q2", "1965:Q3", "1965:Q4"],
        "ROUTPUT65Q4": [600.0, 610.0, "#N/A"],
        "ROUTPUT66Q1": [600.0, 612.0, 620.0],
        "ROUTPUT92Q1": [3000.0, 3010.0, 3020.0],   # posterior a ALFRED: fuera
    })


def test_tabla_philly_corta_en_alfred_y_compacta(pubs):
    t = philly_rtdsm.tabla_philly("ROUTPUT", _matriz_routput(), primer_alfred="1991-12-04", publicaciones=pubs)
    validar_series(t)
    assert set(t["vintage"]) == {"1965-11", "1966-02"}
    # 1965T2 no cambia en 66Q1: una sola fila; 1965T3 se revisa; 1965T4 aparece en 66Q1.
    assert len(t) == 4
    assert (t["fecha_publicacion"] > pd.Timestamp("1965-11-30")).all()
    rev = t[t["fecha_referencia"] == pd.Timestamp("1965-07-01")].sort_values("vintage")
    assert list(rev["valor"]) == [610.0, 612.0]
    assert (t["serie"] == "philly:ROUTPUT").all() and (t["metodo_publicacion"] == "vintage_exacto").all()


def test_mapeo_philly_a_fred():
    assert {v: m["fred"] for v, m in philly_rtdsm.MAPEO.items()} == {
        "ROUTPUT": "fred:GDPC1", "PCPI": "fred:CPIAUCSL", "EMPLOY": "fred:PAYEMS", "RUC": "fred:UNRATE"}


# --- OCDE ------------------------------------------------------------------------------------------
_CSV_OECD = """DATAFLOW,REF_AREA,FREQ,MEASURE,UNIT_MEASURE,ACTIVITY,EDITION,TIME_PERIOD,OBS_VALUE,OBS_STATUS,UNIT_MULT,DECIMALS,BASE_PER
OECD.SDD.STES:DSD_STES_REVISIONS@DF_STES_REVISIONS(4.0),USA,M,LI_AA,IX,_T,202608,2026-06,100.10,A,0,2,
OECD.SDD.STES:DSD_STES_REVISIONS@DF_STES_REVISIONS(4.0),USA,M,LI_AA,IX,_T,202609,2026-06,100.10,A,0,2,
OECD.SDD.STES:DSD_STES_REVISIONS@DF_STES_REVISIONS(4.0),USA,M,LI_AA,IX,_T,202609,2026-07,100.30,A,0,2,
OECD.SDD.STES:DSD_STES_REVISIONS@DF_STES_REVISIONS(4.0),USA,M,LI_AA,IX,_T,202610,2026-06,100.05,A,0,2,
OECD.SDD.STES:DSD_STES_REVISIONS@DF_STES_REVISIONS(4.0),USA,M,LI_AA,IX,_T,202610,2026-07,100.30,A,0,2,
OECD.SDD.STES:DSD_STES_REVISIONS@DF_STES_REVISIONS(4.0),USA,M,LI_TR,GR,_T,202610,2026-07,0.5,A,0,2,
OECD.SDD.STES:DSD_STES_REVISIONS@DF_STES_REVISIONS(4.0),USA,Q,LI_AA,IX,_T,202610,2026-Q2,100.0,A,0,2,
"""


def test_oecd_ediciones_como_vintages(pubs):
    crudo = oecd.leer_csv_revisiones(_CSV_OECD.encode())
    assert set(crudo["medida"]) == {"LI_AA", "LI_TR"}          # la trimestral queda fuera
    t = oecd.tabla_oecd("USA", "LI_AA", crudo, publicaciones=pubs)
    validar_series(t)
    # 2026-06: 202608 (nuevo), 202610 (revisado); 2026-07: 202609 (nuevo). 202609 de junio se funde.
    assert sorted(zip(t["fecha_referencia"].dt.strftime("%Y-%m"), t["vintage"])) == [
        ("2026-06", "2026-08"), ("2026-06", "2026-10"), ("2026-07", "2026-09")]
    # Conservador: visible tras el fin del mes de la edición (31-ago 23:59 París → 31-ago, 16:30 NY ya pasó → 1-sep).
    f = t.set_index("vintage")["fecha_publicacion"]
    assert f["2026-08"] == pd.Timestamp("2026-08-31") or f["2026-08"] > pd.Timestamp("2026-08-31")
    assert (t["fecha_publicacion"] > t["fecha_referencia"]).all()


def test_oecd_dbnomics_plan_b(pubs):
    doc = {"series": {"docs": [{"series_code": "USA.M.LI.IX._Z.AA.IX._Z.H", "dimensions": {"REF_AREA": "USA"},
                                "period": ["2026-07", "2026-08", "2026-09"], "value": [100.1, "NA", 100.4]}]}}
    d = oecd.leer_dbnomics(doc)
    assert len(d) == 2
    t = oecd.tabla_dbnomics("USA", d, "2026-10-13", publicaciones=pubs)
    validar_series(t)
    assert set(t["vintage"]) == {"2026-10"} and t["nota"].str.contains("fuga de revisiones").all()


# --- Bancos centrales ------------------------------------------------------------------------------
def test_lectores_bancos():
    boc = json.dumps({"observations": [{"d": "2026-10-08", "V39079": {"v": "2.25"}},
                                       {"d": "2026-10-09", "V39079": {"v": "2.25"}},
                                       {"d": "2026-10-10", "V39079": {}}]}).encode()
    s = bancos_centrales.leer_boc(boc, "V39079")
    assert list(s.index.strftime("%Y-%m-%d")) == ["2026-10-08", "2026-10-09"]
    boj = json.dumps({"RESULTSET": [{"SERIES_CODE": "STRDCLUCON", "VALUES": {
        "SURVEY_DATES": [20261001, 20261002, 20261003], "VALUES": [0.477, 0.478, None]}}]}).encode()
    assert list(bancos_centrales.leer_boj(boj, "STRDCLUCON")) == [0.477, 0.478]
    bbk = ('﻿"";BBSIS.X;BBSIS.X_FLAGS\n"";Zinsstrukturkurve;\nEinheit;Prozent;\n'
           "2026-10-11;.;Kein Wert vorhanden\n2026-10-12;3,53;\n2026-10-13;3,49;\n").encode("utf-8")
    assert list(bancos_centrales.leer_bundesbank(bbk)) == [3.53, 3.49]
    rba = ("﻿F1 INTEREST RATES\nTitle,Cash Rate Target,Otro\nSeries ID,FIRMMCRTD,FOTRO\n"
           "09-Oct-2026,3.60,1\n12-Oct-2026,3.60,1,extra\n13-Oct-2026,,1\n").encode("utf-8")
    assert list(bancos_centrales.leer_rba(rba, "FIRMMCRTD")) == [3.6, 3.6]


def test_tabla_banco_visible_el_habil_siguiente(pubs):
    s = pd.Series([2.25, 2.25], index=pd.to_datetime(["2026-10-08", "2026-10-09"]))   # jueves y viernes
    t = bancos_centrales.tabla_banco("boc:V39079", s, publicaciones=pubs)
    validar_series(t)
    assert list(t["fecha_publicacion"].dt.strftime("%Y-%m-%d")) == ["2026-10-09", "2026-10-12"]
    assert (t["vintage"] == "unico").all() and (t["metodo_publicacion"] == "calendario").all()


# --- Fragmento -------------------------------------------------------------------------------------
def test_fragmento_valido_con_reglas_del_catalogo(pubs):
    # Fragmento fundido por el integrador: las series están en catalogo_series.yaml.
    cat = catalogo.cargar_catalogo()
    assert {"oecd", "philly", "boc", "boj", "bbk", "rba"} <= set(catalogo.PREFIJOS)
    propias = {s: e for s, e in cat.items() if s.split(":", 1)[0] in ("oecd", "philly", "boj", "bbk", "rba")}
    usados = {e["publicador"] for e in propias.values()}
    assert usados <= set(pubs["publicadores"])
    assert set(bancos_centrales.SERIES) <= set(cat)
    assert {f"philly:{v}" for v in philly_rtdsm.MAPEO} <= set(cat)


# --- Red -------------------------------------------------------------------------------------------
@pytest.mark.live
def test_live_boc_valet(pubs):
    from ingesta.comun.http import get
    r = get("https://www.bankofcanada.ca/valet/observations/V39079/json", params={"recent": 5}, timeout=60)
    s = bancos_centrales.leer_boc(r.content, "V39079")
    assert len(s) >= 1
    validar_series(bancos_centrales.tabla_banco("boc:V39079", s, publicaciones=pubs))
