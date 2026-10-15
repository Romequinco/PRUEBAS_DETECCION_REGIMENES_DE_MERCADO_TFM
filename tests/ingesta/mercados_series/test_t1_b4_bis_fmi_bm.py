"""Bloque 4: BIS, FMI (WEO por edición) y Banco Mundial. Sin red salvo el test ``live``."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from ingesta.mercados_series import catalogo as cat_mod
from ingesta.mercados_series.contrato import validar_series
from ingesta.mercados_series.fuentes import _organismos as org
from ingesta.mercados_series.fuentes import bis, imf
from ingesta.mercados_series.fuentes import worldbank as wb

# --- Fixtures sintéticas con el formato real ------------------------------------------------------
CSV_GAP = """FREQ,BORROWERS_CTY,TC_BORROWERS,TC_LENDERS,CG_DTYPE,TIME_PERIOD,OBS_VALUE
Q,US,P,A,C,2023-Q1,-7.5
Q,US,P,A,C,2023-Q2,NaN
Q,US,P,A,C,2023-Q3,-9.0
Q,XM,P,A,C,2023-Q1,-2.0
"""
CSV_CBPOL = """FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE
M,US,2024-01,5.375
M,US,2024-02,5.375
"""


def _dbnomics() -> dict:
    def serie(pais, concepto, valores):
        return {"dimensions": {"weo-country": pais, "weo-subject": concepto, "unit": "pcent_change"},
                "indexed_at": "2019-10-16T02:00:00Z", "period": ["2017", "2018", "2019", "2020"], "value": valores}
    return {"series": {"num_found": 3, "docs": [
        serie("USA", "NGDP_RPCH", [2.4, 2.9, 2.4, 2.1]),
        serie("USA", "LUR", [4.4, 3.9, "NA", 3.5]),
        serie("ESP", "NGDP_RPCH", [3.0, 2.4, 2.2, 1.8])]}}


def _sdmx() -> dict:
    return {"data": {"structures": [{
        "dimensions": {"series": [{"id": "COUNTRY", "values": [{"id": "USA"}]},
                                  {"id": "INDICATOR", "values": [{"id": "NGDP_RPCH"}]},
                                  {"id": "FREQUENCY", "values": [{"id": "A"}]}],
                       "observation": [{"id": "TIME_PERIOD", "values": [{"value": "2025"}, {"value": "2026"},
                                                                         {"value": "2027"}]}]},
        "attributes": {"series": [{"id": "SCALE"}, {"id": "DECIMALS_DISPLAYED"}, {"id": "OVERLAP"},
                                  {"id": "COUNTRY_UPDATE_DATE"}]}}],
        "dataSets": [{"series": {"0:0:0": {"attributes": [0, 0, 0, "9/30/2026"],
                                           "observations": {"0": ["2.1"], "1": ["1.9"], "2": ["2.0"]}}}}]}}


PAISES_WB = json.dumps([{"page": 1, "pages": 1}, [
    {"id": "ESP", "name": "Spain", "region": {"id": "ECS", "value": "Europe"}, "incomeLevel": {"value": "High"}},
    {"id": "EUU", "name": "European Union", "region": {"id": "NA", "value": "Aggregates"},
     "incomeLevel": {"value": "Aggregates"}}]])
IND_WB = json.dumps([{"page": 1, "pages": 1, "total": 6, "lastupdated": "2026-10-08"}, [
    {"country": {"id": "ES"}, "countryiso3code": "ESP", "date": str(a), "value": v}
    for a, v in [(2025, None), (2024, 3.2), (2023, None), (2022, 6.2)]] + [
    {"country": {"id": "EU"}, "countryiso3code": "EUU", "date": "2024", "value": 1.0}]])


# --- BIS ------------------------------------------------------------------------------------------
def test_bis_brecha_trimestral_lag_y_huecos():
    t = bis.tabla_flujo(bis.leer_csv(CSV_GAP, "WS_CREDIT_GAP"), "WS_CREDIT_GAP")
    validar_series(t)
    us = t[t["serie"] == "bis:WS_CREDIT_GAP.Q.US.P.A.C"].set_index("fecha_referencia")
    assert list(us.index) == list(pd.to_datetime(["2023-01-01", "2023-04-01", "2023-07-01"]))
    # 6 fines de mes tras el fin del trimestre (2023-03-31) -> 2023-09-30
    assert us.loc["2023-01-01", "fecha_publicacion"] == pd.Timestamp("2023-09-30")
    assert pd.isna(us.loc["2023-04-01", "valor"]) and org.NOTA_HUECO in us.loc["2023-04-01", "nota"]
    assert us["nota"].str.contains(cat_mod.NOTA_REVISABLE_EDICION, regex=False).all()
    assert (t["vintage"] == "unico").all() and (t["metodo_publicacion"] == "lag_fijo").all()


def test_bis_cbpol_no_revisable():
    t = bis.tabla_flujo(bis.leer_csv(CSV_CBPOL, "WS_CBPOL"), "WS_CBPOL")
    assert t["serie"].unique().tolist() == ["bis:WS_CBPOL.M.US"]
    assert t["fecha_publicacion"].iloc[0] == pd.Timestamp("2024-02-07")
    assert t["nota"].isna().all()


def test_bis_url_y_pais():
    assert bis.url_flujo("WS_DSR", ["US", "XM"]).endswith("/WS_DSR/Q.US+XM.P/all")
    assert bis.pais_de("WS_EER", "M.R.B.CN") == "CN"
    with pytest.raises(ValueError):
        bis.leer_csv("A,B\n1,2\n", "WS_CBPOL")


# --- FMI ------------------------------------------------------------------------------------------
def test_weo_edicion_separa_dato_y_proyecciones():
    datos, meta = imf.leer_dbnomics(json.dumps(_dbnomics()))
    assert meta["indexed_at_min"] == "2019-10-16T02:00:00Z"
    t, cuenta = imf.tabla_edicion(datos, "2019-10")
    validar_series(t)
    dato = t[~t["serie"].map(imf.es_proyeccion)]
    proy = t[t["serie"].map(imf.es_proyeccion)]
    # Las series de dato solo traen años ANTERIORES al de la edición (ESP fuera del alcance).
    assert set(dato["serie"]) == {"imf:WEO.USA.NGDP_RPCH", "imf:WEO.USA.LUR"}
    assert dato["fecha_referencia"].dt.year.max() == 2018 and dato["nota"].isna().all()
    # Año en curso (H0) y proyecciones (H1...) en series propias, fechadas en el año de la edición.
    assert {"imf:WEO.USA.NGDP_RPCH.PROY.H0", "imf:WEO.USA.NGDP_RPCH.PROY.H1"} <= set(proy["serie"])
    assert (proy["fecha_referencia"] == pd.Timestamp("2019-01-01")).all()
    assert proy["nota"].str.startswith("proyección del WEO (no es un dato)").all()
    assert proy.loc[proy["serie"].str.endswith(".H0"), "nota"].str.contains(imf.NOTA_ESTIMACION, regex=False).all()
    assert cuenta["proyecciones"] == len(proy) and cuenta["filas"] == len(dato)
    assert (t["vintage"] == "2019-10").all() and (t["fecha_publicacion"] == pd.Timestamp("2019-10-15")).all()
    assert imf.disponible("2019-10") == pd.Timestamp("2019-10-15 13:00", tz="UTC")


def test_weo_hora_medida_2026_10_y_fechas():
    # La versión 9.0.0 ya estaba en la API a las 07:46 de Nueva York: nunca después de lo observado.
    assert imf.disponible("2026-10") == pd.Timestamp("2026-10-13 11:46", tz="UTC")
    assert imf.FECHAS_EDICION["2026-04"] == "2026-04-14" and imf.FECHAS_EDICION["2026-10"] == "2026-10-13"
    assert imf.id_proyeccion("USA", "LUR", 3) == "imf:WEO.USA.LUR.PROY.H3"
    ids = imf.ids_proyeccion(["imf:WEO.USA.LUR"])
    assert len(ids) == imf.HORIZONTE_MAX + 1 and all(imf.es_proyeccion(i) for i in ids)
    frag = imf.fragmento(["imf:WEO.USA.LUR", *ids])
    assert "PROYECCIÓN, no dato" in frag["imf:WEO.USA.LUR.PROY.H3"]["notas"][0]


def test_weo_sdmx_edicion_por_fecha():
    datos, ed = imf.leer_sdmx(_sdmx())
    assert ed == "2026-10" and len(datos) == 3
    assert imf.edicion_de_fecha("3/20/2026") == "2026-04"
    t, cuenta = imf.tabla_edicion(datos, ed)
    assert cuenta["filas"] + cuenta["proyecciones"] + cuenta["proyecciones_descartadas"] == 3
    assert t["fecha_referencia"].max() <= t["fecha_publicacion"].min()
    with pytest.raises(ValueError):
        imf.disponible("2030-04")


def test_weo_edicion_fallida_en_dbnomics_no_descarta_la_sdmx(monkeypatch):
    """Regresión (revisión): una edición que DBnomics no pudo servir (robots.txt) contaba como «vista» y
    la misma edición servida por la API SDMX se descartaba como repetida."""
    monkeypatch.setattr(imf, "descargar_dbnomics", lambda ed, **k: (None, {"error": f"robots.txt no permite {ed}"}))
    monkeypatch.setattr(imf, "descargar_sdmx",
                        lambda v, **k: (_sdmx(), {"bytes": 1}) if v == imf.VERSIONES_SDMX[0] else (None, {"error": "x"}))
    sin, _ = imf.tabla_weo([])
    con, informe = imf.tabla_weo(["2026-10"])
    assert len(sin) > 0 and len(con) == len(sin)
    assert informe["2026-10"]["origen"].startswith("sdmx") and "robots" in informe["2026-10"]["error_dbnomics"]
    # Una edición que DBnomics sí sirve sigue mandando: la SDMX de la misma edición se descarta.
    monkeypatch.setattr(imf, "descargar_dbnomics", lambda ed, **k: (_dbnomics(), {"bytes": 1}))
    _, informe2 = imf.tabla_weo(["2026-10"])
    assert informe2["2026-10"]["origen"] == "dbnomics"


# --- Banco Mundial --------------------------------------------------------------------------------
def test_wb_paises_sin_agregados_y_lag():
    paises = wb.leer_paises(PAISES_WB)
    assert paises.set_index("iso3")["es_pais"].to_dict() == {"ESP": True, "EUU": False}
    datos, cab = wb.leer_indicador(IND_WB)
    assert cab["lastupdated"] == "2026-10-08"
    t = wb.tabla_indicador(datos, "NY.GDP.MKTP.KD.ZG", paises.loc[paises["es_pais"], "iso3"])
    validar_series(t)
    assert t["serie"].unique().tolist() == ["wb:ESP.NY.GDP.MKTP.KD.ZG"]
    assert t["fecha_referencia"].dt.year.tolist() == [2022, 2023, 2024]          # 2025 nulo final recortado
    assert t.set_index("fecha_referencia").loc["2024-01-01", "fecha_publicacion"] == pd.Timestamp("2025-12-31")
    assert org.NOTA_HUECO in t.set_index("fecha_referencia").loc["2023-01-01", "nota"]


def test_wb_paginado_falla():
    with pytest.raises(ValueError):
        wb.leer_indicador(json.dumps([{"pages": 2}, []]))


# --- Utilidades -----------------------------------------------------------------------------------
def test_descargar_respeta_robots(tmp_path, monkeypatch):
    monkeypatch.setattr(org.http, "robots_permite", lambda url: False)
    contenido, meta = org.descargar("bis", "x.csv", "https://ejemplo.invalid/x", raiz=tmp_path)
    assert contenido is None and "robots" in meta["error"]
    assert not (tmp_path / "bis" / f"{org.hoy_ny()}_x.csv").exists()


def test_medir_y_metadatos(tmp_path):
    t = bis.tabla_flujo(bis.leer_csv(CSV_CBPOL, "WS_CBPOL"), "WS_CBPOL")
    m = org.medir(t, {"bis:WS_CBPOL.M.US": "mensual"}, hoy=pd.Timestamp("2024-03-10"))
    assert m.loc[0, "retraso_observado_dias"] == 10 and m.loc[0, "lag_mediano_dias"] == 7
    ruta = org.guardar_metadatos(m, "bis", tmp_path / "meta.csv")
    assert "valor" not in pd.read_csv(ruta).columns


def test_fragmentos_coherentes():
    frag = bis.fragmento(["bis:WS_CREDIT_GAP.Q.US.P.A.C", "bis:WS_CBPOL.M.US"])
    assert frag["bis:WS_CREDIT_GAP.Q.US.P.A.C"]["notas"][0] == cat_mod.NOTA_REVISABLE_EDICION
    assert set(bis.lags_fragmento()) == {f"BIS_{f}" for f in bis.FLUJOS}
    assert all(e["lag_clave"] in wb.lags_fragmento() for e in wb.patrones_fragmento().values())
    assert imf.fragmento(["imf:WEO.USA.LUR"])["imf:WEO.USA.LUR"]["metodo"] == "vintage_exacto"


@pytest.mark.live
def test_live_bis_cbpol_us(tmp_path):
    contenido, meta = org.descargar("bis", "cbpol.csv", bis.url_flujo("WS_CBPOL", ["US"]),
                                    params={"format": "csv", "detail": "dataonly", "startPeriod": "2024-01"},
                                    raiz=tmp_path)
    assert contenido is not None, meta.get("error")
    t = bis.tabla_flujo(bis.leer_csv(contenido, "WS_CBPOL"), "WS_CBPOL")
    assert len(t) >= 12
