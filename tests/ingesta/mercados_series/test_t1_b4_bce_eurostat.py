"""Tests de ``ingesta.mercados_series.fuentes.bce`` y ``...fuentes.eurostat`` (bloque 4).

Sin red: CSV SINTÉTICOS con el formato real (``format=csvdata`` del Data Portal del BCE, con y sin
``includeHistory``; SDMX-CSV 2.1 de Eurostat) servidos por un transporte falso; caché en
``tmp_path``. Ningún valor real de las fuentes. ``@pytest.mark.live``: descarga real mínima
(``INGESTA_LIVE=1``).
"""
from __future__ import annotations

import copy
import re

import pandas as pd
import pytest
import yaml

from ingesta.mercados_series import catalogo as cat_mod
from ingesta.mercados_series import visibilidad
from ingesta.mercados_series.contrato import validar_series
from ingesta.mercados_series.fuentes import bce, eurostat

# --- CSV sintéticos (cabeceras reales, valores inventados) ---------------------------------------
CAB_EXR = ("KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE,OBS_STATUS,OBS_CONF,"
           "TITLE,UNIT,UNIT_MULT")
CAB_DATOS = "KEY,FREQ,TIME_PERIOD,OBS_VALUE"
CAB_HIST = "KEY,FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE,ACTION,VALID_FROM,VALID_TO"
CAB_ESTAT = "DATAFLOW,LAST UPDATE,freq,unit,coicop18,geo,TIME_PERIOD,OBS_VALUE,OBS_FLAG,CONF_STATUS"


def _exr_ultimo() -> str:
    return "\n".join([
        CAB_EXR,
        "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2026-10-12,1.5,A,F,US dollar/Euro ECB reference exchange rate,USD,0",
        "EXR.D.ZZZ.EUR.SP00.A,D,ZZZ,EUR,SP00,A,2001-12-31,9.0,A,F,Divisa retirada/Euro,ZZZ,0",
    ])


def _exr_todo() -> str:
    return "\n".join([
        CAB_DATOS,
        "EXR.D.USD.EUR.SP00.A,D,2026-10-09,1.4",
        "EXR.D.USD.EUR.SP00.A,D,2026-10-12,1.5",
        "EXR.D.USD.EUR.SP00.A,D,2026-10-13,",
        "EXR.D.ZZZ.EUR.SP00.A,D,2001-12-31,9.0",
    ])


def _diaria(clave: str) -> str:
    return "\n".join([CAB_DATOS, f"{clave},B,2026-04-02,1.0", f"{clave},B,2026-10-09,1.1"])


def _historial(clave: str, mensual: bool = True) -> str:
    if mensual:
        filas = [f"{clave},M,U2,2010-01,1.0,Replace,2026-02-04T10:00:04.000+01:00,",
                 f"{clave},M,U2,2026-09,2.0,Replace,2026-10-01T11:07:14.000+02:00,",
                 f"{clave},M,U2,1996-06,,Replace,2026-02-04T10:00:04.000+01:00,",
                 f"{clave},M,U2,1996-05,,Delete,,2026-02-04T10:00:04.000+01:00"]
    else:
        filas = [f"{clave},Q,I10,2010-Q1,0.5,Replace,2026-09-07T11:19:29.000+02:00,"]
    return "\n".join([CAB_HIST, *filas])


def transporte_bce(url: str, params: dict) -> str:
    ruta = url.split("/service/data/", 1)[1]
    if ruta.startswith("EXR/"):
        return _exr_ultimo() if "lastNObservations" in params else _exr_todo()
    if params.get("includeHistory") == "true":
        clave = ruta.replace("/", ".")
        return _historial(clave, mensual=not ruta.startswith("MNA/"))
    return _diaria(ruta.replace("/", "."))


def _estat(conj: str) -> str:
    filas = [
        f"ESTAT:{conj.upper()}(1.0),02/10/26 11:00:00,M,RCH_A,TOTAL,DE,2026-07,2.0,,",
        f"ESTAT:{conj.upper()}(1.0),02/10/26 11:00:00,M,RCH_A,TOTAL,DE,2026-08,2.1,p,",
        f"ESTAT:{conj.upper()}(1.0),02/10/26 11:00:00,M,RCH_A,TOTAL,DE,2026-09,,c,",
        f"ESTAT:{conj.upper()}(1.0),02/10/26 11:00:00,M,RCH_A,TOTAL,EA,2026-08,1.9,,",
        f"ESTAT:{conj.upper()}(1.0),02/10/26 11:00:00,M,RCH_A,TOTAL,XX,2026-08,5.0,,",
    ]
    if conj == "namq_10_gdp":
        filas = ["ESTAT:NAMQ_10_GDP(1.0),09/10/26 23:00:00,Q,CLV,SCA,DE,2026-Q2,1.0,p,"]
    if conj == "prc_hicp_manr":
        filas = ["ESTAT:PRC_HICP_MANR(1.0),06/02/26 23:00:00,M,RCH_A,CP00,DE,2026-07,1.9,,"]
    return "\n".join([CAB_ESTAT, *filas])


def transporte_estat(url: str, params: dict) -> str:
    return _estat(url.split("/data/", 1)[1].split("/", 1)[0])


# --- Parseo y fechas -----------------------------------------------------------------------------
def test_t1_b4_parsear_csv_y_periodos():
    df = bce.parsear_csv(_historial("HICP.M.U2.N.000000.4D0.ANR"))
    assert list(df.columns[:3]) == ["KEY", "TIME_PERIOD", "OBS_VALUE"]
    assert str(df["VALID_FROM"].dtype) == "datetime64[ns, UTC]"
    assert df["OBS_VALUE"].isna().sum() == 2
    f = bce.fecha_periodo(pd.Series(["2026-10-12", "2026-09", "2026-Q2", "2025"]))
    assert list(f) == [pd.Timestamp(x) for x in ("2026-10-12", "2026-09-01", "2026-04-01", "2025-01-01")]
    assert bce.parsear_csv("").empty


def test_t1_b4_id_fx_coincide_con_divisas():
    from ingesta.mercados_series.divisas import SERIE_FX
    assert bce.id_fx("EXR.D.USD.EUR.SP00.A") == SERIE_FX.format(divisa="USD") == "ecb:EXR.D.USD.EUR"


def test_t1_b4_fx_mismo_dia_y_estr_habil_target(tmp_path):
    obs = bce.parsear_csv(_exr_todo())
    cli = bce.ClienteBce(transporte=transporte_bce, carpeta=tmp_path, hoy="2026-10-13")
    e = bce.entradas_bce(bce.series_fx(cli))["ecb:EXR.D.USD.EUR"]
    t = bce.tabla_calendario("ecb:EXR.D.USD.EUR", obs[obs["KEY"].str.contains("USD")], e)
    validar_series(t)
    assert len(t) == 2  # el día sin valor se descarta
    fila = t[t["fecha_referencia"] == "2026-10-12"].iloc[0]
    assert fila["fecha_publicacion"] == pd.Timestamp("2026-10-12")
    assert fila["disponible_utc"] == pd.Timestamp("2026-10-12 14:00", tz="UTC")  # 16:00 Berlín (CEST)
    estr = bce.tabla_calendario("ecb:ESTR", bce.parsear_csv(_diaria("EST.B.EU000A2X2A25.WT")), bce.SERIES_BCE["ecb:ESTR"])
    pub = dict(zip(estr["fecha_referencia"], estr["fecha_publicacion"]))
    # Jueves Santo 2026-04-02: Viernes Santo y Lunes de Pascua son festivos TARGET → martes 7.
    assert pub[pd.Timestamp("2026-04-02")] == pd.Timestamp("2026-04-07")
    assert pub[pd.Timestamp("2026-10-09")] == pd.Timestamp("2026-10-12")  # viernes → lunes


def test_t1_b4_tipos_oficiales_visibles_en_fecha_de_efecto():
    t = bce.tabla_calendario("ecb:DFR", bce.parsear_csv(_diaria("FM.B.U2.EUR.4F.KR.DFR.LEV")), bce.SERIES_BCE["ecb:DFR"])
    assert (t["fecha_publicacion"] == t["fecha_referencia"]).all()


def test_t1_b4_versiones_regla_previa_y_valid_from():
    e = bce.SERIES_BCE["ecb:HICP.M.U2.N.000000.4D0.ANR"]
    hist = bce.parsear_csv(_historial("HICP.M.U2.N.000000.4D0.ANR"))
    t = validar_series(bce.tabla_versiones("ecb:HICP.M.U2.N.000000.4D0.ANR", hist, e))
    assert len(t) == 2  # versiones sin valor y Delete fuera
    vieja = t[t["fecha_referencia"] == "2010-01-01"].iloc[0]
    # HICP ECOICOP v2: la retropolación no existía antes de su primera versión (2026-02-04 en la fixture).
    assert vieja["metodo_publicacion"] == "lag_fijo" and vieja["nota"] == bce.NOTA_POSTERIOR_BCE
    assert vieja["fecha_publicacion"] == pd.Timestamp("2026-02-04") and pd.isna(vieja["disponible_utc"])
    nueva = t[t["fecha_referencia"] == "2026-09-01"].iloc[0]
    assert nueva["metodo_publicacion"] == "vintage_exacto" and nueva["vintage"] == "2026-10-01"
    assert nueva["disponible_utc"] == pd.Timestamp("2026-10-01 09:07:14", tz="UTC")
    assert nueva["fecha_publicacion"] == pd.Timestamp("2026-10-01")


def test_t1_b4_tabla_bce_completa_y_cache(tmp_path):
    cli = bce.ClienteBce(transporte=transporte_bce, carpeta=tmp_path, hoy="2026-10-13")
    df, med = bce.tabla_bce(cli)
    assert cli.peticiones == 2 + 4 + 4
    assert {"ecb:EXR.D.USD.EUR", "ecb:EXR.D.ZZZ.EUR", "ecb:ESTR", "ecb:EONIA", "ecb:DFR", "ecb:MRO"} <= set(df["serie"])
    assert med["series"]["ecb:HICP.M.U2.N.000000.4D0.ANR"]["include_history"]["acciones"] == {"Replace": 3, "Delete": 1}
    espec = bce.entradas_bce(bce.series_fx(cli))
    assert espec["ecb:EXR.D.ZZZ.EUR"]["estado"] == "retirada" and espec["ecb:EXR.D.USD.EUR"]["estado"] == "activa"
    assert len(list(tmp_path.rglob("*.csv"))) == 10
    sin_red = bce.ClienteBce(carpeta=tmp_path, sin_red=True, hoy="2026-10-20")
    df2, _ = bce.tabla_bce(sin_red)
    assert sin_red.peticiones == 0
    pd.testing.assert_frame_equal(df, df2)


def test_t1_b4_include_history_sin_versiones_multiples(tmp_path):
    cli = bce.ClienteBce(transporte=transporte_bce, carpeta=tmp_path, hoy="2026-10-13")
    r = bce.medir_include_history(cli, "HICP/M.U2.N.000000.4D0.ANR")
    assert r["devuelve_historia"] is False and r["versiones_por_obs_max"] == 1


# --- Eurostat ------------------------------------------------------------------------------------
def test_t1_b4_eurostat_lag_banderas_y_huecos():
    obs = eurostat.parsear_sdmx_csv(_estat("une_rt_m"))
    e = eurostat.entradas_eurostat({"une_rt_m": ["DE"]})
    sid = eurostat.id_eurostat("une_rt_m", eurostat.CONJUNTOS["une_rt_m"]["clave"], "DE")
    assert sid == "eurostat:une_rt_m.M.SA.TOTAL.PC_ACT.T.DE"
    t = validar_series(eurostat.tabla_conjunto(sid, obs[obs["geo"] == "DE"], e[sid]))
    assert len(t) == 3
    ago = t[t["fecha_referencia"] == "2026-08-01"].iloc[0]
    assert ago["fecha_publicacion"] == pd.Timestamp("2026-10-31")  # EUROSTAT_PARO: 2M desde el fin de mes
    assert "provisional" in ago["nota"] and cat_mod.es_nota_revisable(ago["nota"])
    sep = t[t["fecha_referencia"] == "2026-09-01"].iloc[0]
    assert pd.isna(sep["valor"]) and "hueco" in sep["nota"]
    assert (t["metodo_publicacion"] == "lag_fijo").all() and (t["vintage"] == "unico").all()


def test_t1_b4_tabla_eurostat_y_ruptura(tmp_path):
    cli = eurostat.ClienteEurostat(transporte=transporte_estat, carpeta=tmp_path, hoy="2026-10-13")
    df, med = eurostat.tabla_eurostat(cli)
    assert cli.peticiones == len(eurostat.CONJUNTOS)
    assert not df["serie"].str.endswith(".XX").any()  # zona fuera de GEOS
    assert med["conjuntos"]["prc_hicp_manr"]["ultima_actualizacion"].startswith("2026-02-06")
    r = med["conjuntos"]["ruptura_hicp"]["DE"]
    assert r["meses_comunes"] == 1 and r["dif_abs_max"] == pytest.approx(0.1)
    espec = med["espec"]
    manr = eurostat.id_eurostat("prc_hicp_manr", "M.RCH_A.CP00.", "DE")
    assert espec[manr]["estado"] == "retirada" and espec[manr]["sucesor"].startswith("eurostat:prc_hicp_minr")


# --- Fragmento del catálogo ------------------------------------------------------------------------
def test_t1_b4_fragmento_valida_con_el_catalogo():
    # Fragmento fundido por el integrador: las series del BCE están en catalogo_series.yaml.
    cat = cat_mod.cargar_catalogo()
    pubs = bce.publicaciones_b4()
    assert "ecb_tipos_oficiales" in pubs["publicadores"]
    assert {"ecb", "eurostat"} <= set(cat_mod.PREFIJOS)
    assert set(bce.SERIES_BCE) <= set(cat)
    assert "ecb:EXR.D.USD.EUR" in cat and "ecb:ESTR" in cat
    assert {e["publicador"] for s, e in cat.items() if s.startswith("ecb:") and e.get("publicador")}         <= set(visibilidad.cargar_publicaciones()["publicadores"])


# --- Red (INGESTA_LIVE=1) ----------------------------------------------------------------------------
@pytest.mark.live
def test_t1_b4_live_bce_y_eurostat(tmp_path):
    cli = bce.ClienteBce(carpeta=tmp_path)
    obs = cli.datos("EXR/D.USD.EUR.SP00.A", detail="dataonly", lastNObservations=5)
    t = bce.tabla_calendario("ecb:EXR.D.USD.EUR", obs, bce.entradas_bce()["ecb:ESTR"] | {
        "unidad": "USD por 1 EUR", "divisa": "USD", "publicador": "ecb_fx"})
    assert len(validar_series(t)) >= 3
    ce = eurostat.ClienteEurostat(carpeta=tmp_path)
    df = ce.datos("une_rt_m", "M.SA.TOTAL.PC_ACT.T.DE", lastNObservations=2)
    assert len(df) == 2 and df["OBS_VALUE"].notna().all()
