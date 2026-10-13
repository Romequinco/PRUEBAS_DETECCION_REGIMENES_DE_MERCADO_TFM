"""Tests de ``ingesta.mercados_series.fuentes.curvas`` (bloque 6: curvas oficiales).

Sin red: ficheros SINTÉTICOS con el formato real (CSV anual del Tesoro con plazos que cambian,
CSV SDMX del BCE, CSV del MoF con título, ``-`` y nota al pie, CSV de Valet con el marcador
``OBSERVATIONS``, ZIP del BoE con su xlsx). Ningún valor real de las fuentes.
``@pytest.mark.live``: descarga real (``INGESTA_LIVE=1``).
"""
from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pandas as pd
import pytest

from ingesta.mercados_series.contrato import validar_series
from ingesta.mercados_series.fuentes import curvas as cv


# --- Fixtures sintéticas ------------------------------------------------------------------------
def _csv_ust_1990() -> str:
    return ('Date,"3 Mo","6 Mo","1 Yr","30 Yr"\n'
            "01/03/1990,7.1,7.2,7.3,8.1\n"
            "01/02/1990,7.0,7.1,7.2,8.0\n")


def _csv_ust_2025() -> str:
    return ('Date,"1 Mo","1.5 Month","2 Mo","3 Mo","4 Mo","6 Mo","1 Yr","30 Yr"\n'
            "02/19/2025,4.1,4.15,4.2,4.3,4.25,4.2,4.1,4.9\n"
            "02/18/2025,4.0,4.05,4.1,4.2,4.15,4.1,4.0,\n"
            "02/14/2025,4.0,,4.1,4.2,4.15,4.1,4.0,4.8\n")


def _csv_bce() -> str:
    filas = ["KEY,FREQ,REF_AREA,CURRENCY,PROVIDER_FM,INSTRUMENT_FM,PROVIDER_FM_ID,DATA_TYPE_FM,TIME_PERIOD,OBS_VALUE"]
    for fecha, base in (("2026-10-08", 2.0), ("2026-10-09", 2.1), ("2026-10-12", 2.2)):
        for inst, extra in (("G_N_A", 0.0), ("G_N_C", 0.3)):
            for plazo, inc in (("SR_1Y", 0.0), ("SR_10Y", 1.0)):
                filas.append(f"YC.B.U2.EUR.4F.{inst}.SV_C_YM.{plazo},B,U2,EUR,4F,{inst},SV_C_YM,{plazo},"
                             f"{fecha},{base + extra + inc:.4f}")
    return "\n".join(filas) + "\n"


def _csv_mof() -> bytes:
    texto = ("Interest Rate,,,,,(Unit : %)\n"
             "Date,1Y,2Y,10Y,40Y\n"
             "1974/9/24,10.3,9.3,8.1,-\n"
             "1974/9/25,10.4,9.4,8.2,-\n"
             "2026/10/9,1.6,1.9,3.0,4.0\n"
             "2026/10/13,1.7,2.0,3.1,4.1\n"
             ",,,,\n"
             "\"  ※If you cannot download the latest csv data, please clear the browser's cache and download "
             "again.\",,,,\n")
    return texto.encode("utf-8")


def _csv_boc() -> str:
    return ('﻿"TERMS AND CONDITIONS"\n"https://www.bankofcanada.ca/terms/"\n\n"NAME"\n"Selected benchmark bond '
            'yields"\n\n"SERIES"\n"id","label","description"\n"BD.CDN.2YR.DQ.YLD","2-year","x"\n'
            '"BD.CDN.LONG.DQ.YLD","long","y"\n\n"OBSERVATIONS"\n'
            '"date","BD.CDN.2YR.DQ.YLD","BD.CDN.LONG.DQ.YLD"\n'
            '"2026-10-07","3.20","4.30"\n"2026-10-08","","4.29"\n"2026-10-09","3.22","4.28"\n')


def _zip_boe() -> bytes:
    filas = [[None, "UK nominal spot curve", None, None, None],
             [None, None, None, None, None],
             ["Maturity", None, None, None, None],
             ["years:", 0.5, 1.0, 1.5, 10.0],
             [None, None, None, None, None],
             [pd.Timestamp("2026-10-08"), 4.1, 4.3, 4.4, 4.9],
             [pd.Timestamp("2026-10-09"), 4.2, 4.4, 4.5, 5.0]]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame([["Bank of England UK yield curve data"]]).to_excel(w, sheet_name="info", header=False,
                                                                           index=False)
        pd.DataFrame(filas).to_excel(w, sheet_name=cv.HOJA_BOE, header=False, index=False)
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("GLC Nominal daily data current month.xlsx", buf.getvalue())
        zf.writestr("OIS daily data current month.xlsx", b"no es un xlsx")
    return z.getvalue()


# --- Plazos -------------------------------------------------------------------------------------
@pytest.mark.parametrize(("cab", "plazo"), [("1 Mo", "1M"), ("1.5 Month", "6W"), ("4 Mo", "4M"), ("1 Yr", "1Y"),
                                            ("30 Yr", "30Y"), ("Date", None), ("Extra", None)])
def test_plazo_ust(cab, plazo):
    assert cv.plazo_ust(cab) == plazo


def test_plazo_anios():
    assert [cv.plazo_anios(a) for a in (0.5, 1, 1.5, 40.0)] == ["6M", "1Y", "18M", "40Y"]


# --- Tesoro: cambio de plazos entre años ---------------------------------------------------------
def test_ust_cambio_de_plazos_y_huecos():
    ancho = pd.concat([cv.leer_ust(_csv_ust_1990()), cv.leer_ust(_csv_ust_2025())]).sort_index()
    assert list(ancho.index[:2]) == [pd.Timestamp("1990-01-02"), pd.Timestamp("1990-01-03")]
    assert {"1M", "6W", "2M", "4M"} <= set(ancho.columns)
    tabla = cv.tabla_ust(ancho)
    validar_series(tabla)
    # 1M no existe en 1990: su tramo empieza en 2025 (no se rellenan nulos antes de existir).
    m1 = tabla[tabla["serie"] == "ust:PAR.1M"]
    assert m1["fecha_referencia"].min() == pd.Timestamp("2025-02-14")
    # 6W empieza el primer día con valor; el hueco del 30Y dentro de su tramo es nulo con nota.
    assert tabla[tabla["serie"] == "ust:PAR.6W"]["fecha_referencia"].min() == pd.Timestamp("2025-02-18")
    hueco = tabla[(tabla["serie"] == "ust:PAR.30Y") & (tabla["fecha_referencia"] == "2025-02-18")]
    assert hueco["valor"].isna().all() and (hueco["nota"] == cv.NOTA_HUECO).all()
    # 18:00 NY del mismo día → decisión del día siguiente.
    fila = tabla[(tabla["serie"] == "ust:PAR.3M") & (tabla["fecha_referencia"] == "2025-02-14")].iloc[0]
    assert fila["fecha_publicacion"] == pd.Timestamp("2025-02-15")
    assert fila["disponible_utc"] == pd.Timestamp("2025-02-14 23:00", tz="UTC")


# --- BCE ----------------------------------------------------------------------------------------
def test_bce_ids_y_habil_siguiente():
    ancho = cv.leer_bce(_csv_bce())
    assert set(ancho.columns) == {"AAA.1Y", "AAA.10Y", "ALL.1Y", "ALL.10Y"}
    tabla = cv.tabla_bce(ancho)
    validar_series(tabla)
    assert set(tabla["serie"]) == {"ecb:YC.AAA.1Y", "ecb:YC.AAA.10Y", "ecb:YC.ALL.1Y", "ecb:YC.ALL.10Y"}
    f = tabla[tabla["serie"] == "ecb:YC.AAA.10Y"].set_index("fecha_referencia")
    # Viernes 9 → sale el lunes 12 a las 12:00 Berlín (10:00 UTC) → decisión del lunes 12.
    assert f.loc["2026-10-09", "disponible_utc"] == pd.Timestamp("2026-10-12 10:00", tz="UTC")
    assert f.loc["2026-10-09", "fecha_publicacion"] == pd.Timestamp("2026-10-12")
    # Último dato: el siguiente día L-V.
    assert f.loc["2026-10-12", "fecha_publicacion"] == pd.Timestamp("2026-10-13")


# --- MoF: cabecera de título, '-' y nota al pie ---------------------------------------------------
def test_mof_nota_al_pie_y_faltantes():
    ancho = cv.leer_mof(_csv_mof())
    assert len(ancho) == 4  # la fila vacía y la nota sobre la caché se descartan
    assert list(ancho.columns) == ["1Y", "2Y", "10Y", "40Y"]
    assert ancho.loc["1974-09-24", "40Y"] != ancho.loc["1974-09-24", "40Y"]  # '-' → NaN
    tabla = cv.tabla_mof(ancho)
    validar_series(tabla)
    # 40Y empieza en su primer valor (no hay nulos antes de existir el plazo).
    assert tabla[tabla["serie"] == "mof:JGB.40Y"]["fecha_referencia"].min() == pd.Timestamp("2026-10-09")
    # Viernes 9 → siguiente hábil con dato = martes 13 (lunes 12 festivo en Japón) 08:30 JST
    # = lunes 12 23:30 UTC = 19:30 NY → decisión del 13.
    f = tabla[tabla["serie"] == "mof:JGB.10Y"].set_index("fecha_referencia")
    assert f.loc["2026-10-09", "disponible_utc"] == pd.Timestamp("2026-10-12 23:30", tz="UTC")
    assert f.loc["2026-10-09", "fecha_publicacion"] == pd.Timestamp("2026-10-13")


def test_mof_cp932():
    texto = _csv_mof().decode("utf-8").replace("※", "")
    datos = texto.encode("cp932") + "※".encode("cp932")
    assert len(cv.leer_mof(datos)) == 4


# --- BoC: marcador OBSERVATIONS -----------------------------------------------------------------
def test_boc_marcador_observations():
    ancho = cv.leer_boc(_csv_boc())
    assert list(ancho.columns) == ["2Y", "LONG"]
    assert np.isnan(ancho.loc["2026-10-08", "2Y"])
    tabla = cv.tabla_boc(ancho)
    validar_series(tabla)
    hueco = tabla[(tabla["serie"] == "boc:BOND.2Y") & (tabla["fecha_referencia"] == "2026-10-08")]
    assert (hueco["nota"] == cv.NOTA_HUECO).all()
    # 18:00 Toronto del hábil siguiente (> 16:30 NY) → decisión del día posterior.
    f = tabla[tabla["serie"] == "boc:BOND.LONG"].set_index("fecha_referencia")
    assert f.loc["2026-10-07", "fecha_publicacion"] == pd.Timestamp("2026-10-09")


def test_boc_sin_marcador():
    with pytest.raises(ValueError, match="OBSERVATIONS"):
        cv.leer_boc('"date","BD.CDN.2YR.DQ.YLD"\n"2026-10-07","3.2"\n')


# --- BoE ----------------------------------------------------------------------------------------
def test_boe_zip():
    pytest.importorskip("openpyxl")
    ancho = cv.leer_boe_zip(_zip_boe())
    assert list(ancho.columns) == ["6M", "1Y", "10Y"]  # 1.5 años no está en PLAZOS_BOE
    tabla = cv.tabla_boe(ancho)
    validar_series(tabla)
    f = tabla[tabla["serie"] == "boe:NOM.10Y"].set_index("fecha_referencia")
    # Jueves 8 → viernes 9 12:00 Londres (11:00 UTC) → decisión del 9.
    assert f.loc["2026-10-08", "disponible_utc"] == pd.Timestamp("2026-10-09 11:00", tz="UTC")
    assert f.loc["2026-10-08", "fecha_publicacion"] == pd.Timestamp("2026-10-09")


# --- Crudos, resumen y fichas -------------------------------------------------------------------
def test_boe_hoja_antigua_y_festivos():
    pytest.importorskip("openpyxl")
    filas = [[None, "UK nominal spot curve", None], ["years:", 1.0, 10.0],
             [pd.Timestamp("1979-01-01"), None, None],      # festivo listado en blanco
             [pd.Timestamp("1979-01-02"), 11.4, 12.0], [pd.Timestamp("1979-01-03"), 11.5, 12.1]]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame(filas).to_excel(w, sheet_name="4. nominal spot curve", header=False, index=False)
    ancho = cv.leer_boe_xlsx(buf.getvalue())
    assert list(ancho.index) == [pd.Timestamp("1979-01-02"), pd.Timestamp("1979-01-03")]
    assert list(ancho.columns) == ["1Y", "10Y"]


def test_tabla_desde_crudos_y_fichas(tmp_path):
    (cv.ruta_crudo("ust_curva", tmp_path) / "20261013_ust_1990.csv").write_text(_csv_ust_1990(), encoding="utf-8")
    (cv.ruta_crudo("ust_curva", tmp_path) / "20261013_ust_2025.csv").write_text(_csv_ust_2025(), encoding="utf-8")
    (cv.ruta_crudo("mof_jgb", tmp_path) / "20261013_jgbcme_all.csv").write_bytes(_csv_mof())
    (cv.ruta_crudo("boc_curva", tmp_path) / "20261013_bond_yields_benchmark.csv").write_text(_csv_boc(),
                                                                                             encoding="utf-8")
    tabla = cv.tabla_curvas(("ust_curva", "mof_jgb", "boc_curva"), raiz=tmp_path)
    assert {"ust_curva", "mof_jgb", "boc_curva"} == set(tabla["fuente"])
    res = cv.resumen(tabla)
    assert {"inicio", "fin", "n_obs", "n_huecos", "retraso_dias_mediana"} <= set(res.columns)
    fichas = {f.fuente: f for f in cv.fichas_curvas(tabla, raiz=tmp_path)}
    assert fichas["mof_jgb"].cobertura_inicio == "1974-09-24"
    assert fichas["boe_curvas"].estado_sonda == "no_probada"
    assert "medido" in fichas["ust_curva"].retraso_publicacion
    p = cv.escribir(tabla, tmp_path / "curvas.parquet")
    assert len(pd.read_parquet(p)) == len(tabla)


def test_fragmento_yaml_coherente():
    # Fragmento fundido por el integrador: publicadores en publicaciones.yaml y series en el catálogo.
    from ingesta.mercados_series import catalogo, visibilidad

    pubs = visibilidad.cargar_publicaciones()["publicadores"]
    for clave, p in cv.PUBLICADORES.items():
        pub = pubs[p["clave"]]
        assert (pub["hora"].strftime("%H:%M"), pub["huso"]) == (p["hora"], p["huso"]), clave
    propias = {s: e for s, e in catalogo.cargar_catalogo().items() if e["fuente"] in cv.FUENTES}
    assert len(propias) == 70
    for sid, e in propias.items():
        assert sid == f"{sid.split(':', 1)[0]}:{e['id_origen']}"
        assert e["fuente"] in cv.FUENTES


@pytest.mark.live
def test_live_mof_mes():
    if os.environ.get("INGESTA_LIVE") != "1":
        pytest.skip("INGESTA_LIVE=1 para la descarga real")
    from ingesta.comun import http

    ancho = cv.leer_mof(http.get(cv.URL_MOF_MES, timeout=60).content)
    assert {"1Y", "10Y", "40Y"} <= set(ancho.columns) and len(ancho) > 0
