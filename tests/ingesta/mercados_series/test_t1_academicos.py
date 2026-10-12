"""Tests de ``ingesta.mercados_series.fuentes.academicos`` (Ken French, Shiller, Goyal-Welch, OFR, JST).

Sin red: ficheros SINTÉTICOS con el formato real, generados en ``tmp_path`` (zip CSV de French con
cabecera y pie, rejilla de Shiller con fechas ``1871.1``, xlsx de Goyal-Welch y JST, ``fsi.csv``).
Ningún valor real de las fuentes. ``@pytest.mark.datos``: ``data/raw`` o los crudos locales.
``@pytest.mark.live``: descarga real (``INGESTA_LIVE=1``).
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ingesta.mercados_series import catalogo as cat_mod
from ingesta.mercados_series.contrato import ErrorContrato, validar_series
from ingesta.mercados_series.fuentes import academicos as ac

ROOT = Path(__file__).resolve().parents[3]


# --- Generadores de ficheros sintéticos ---------------------------------------------------------
def _zip_french(nombre: str = "F-F_Research_Data_Factors_daily", *, mensual: bool = False,
                columnas: tuple[str, ...] = ("Mkt-RF", "SMB", "HML", "RF"), dias: int = 6,
                crsp: str = "202608", faltante: bool = True) -> bytes:
    """Zip con un CSV como los de la Data Library: cabecera de texto, bloque, bloque anual y pie."""
    lineas = [f"This file was created by using the {crsp} CRSP database.",
              "The Tbill return is the simple daily rate (texto sintético).", "",
              "," + ",".join(columnas)]
    fechas = (pd.date_range("2026-01-01", periods=dias, freq="MS") if mensual
              else pd.bdate_range("2026-08-24", periods=dias))
    for k, f in enumerate(fechas):
        vals = [0.01 * (k + 1) * (j + 1) for j in range(len(columnas))]
        if faltante and k == 1:
            vals[1] = -99.99
        etiqueta = f.strftime("%Y%m") if mensual else f.strftime("%Y%m%d")
        lineas.append(f"{etiqueta}," + ",".join(f"{v:8.2f}" for v in vals))
    lineas += ["", " Annual Factors: January-December ", "," + ",".join(columnas),
               "  2025," + ",".join("   1.00" for _ in columnas), "",
               "Copyright 2026 Eugene F. Fama and Kenneth R. French", ""]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        info = zipfile.ZipInfo(f"{nombre}.csv", date_time=(2026, 9, 24, 9, 9, 46))
        z.writestr(info, "\r\n".join(lineas).encode("latin-1"))
    return buf.getvalue()


def _rejilla_shiller(meses: int = 24, *, texto: bool = True, provisional: bool = False) -> pd.DataFrame:
    """Rejilla de la hoja Data: 7 filas de títulos, fila 'Date', datos AAAA.MM y nota al pie."""
    ncol = 14
    titulos = [["" for _ in range(ncol)] for _ in range(7)]
    titulos[1][0] = 'Stock Market Data Used in "Irrational Exuberance" (sintético)'
    cab = ["Date", "P", "D", "E", "CPI", "Fraction", "Rate GS10", "Price", "Dividend", "Price", "Earnings",
           "Earnings", "CAPE", ""]
    filas = []
    for k in range(meses + (1 if provisional else 0)):
        anio, mes = 1871 + k // 12, k % 12 + 1
        fecha: object = f"{anio}.{mes:02d}" if texto else anio + mes / 100
        if texto and mes == 10:
            fecha = f"{anio}.1"          # así lo escribe Shiller: octubre = .1
        cape = "NA" if k < 3 else f"{10 + k * 0.1:.3f}"
        fila = [fecha, f"{4 + k * 0.01:.2f}", "0.26", "0.4", "12.5", "", f"{5 + k * 0.01:.2f}", "1", "1", "1",
                "1", "1", cape, ""]
        filas.append(fila if texto else [fecha, *[pd.to_numeric(v, errors="coerce") if v not in ("", "NA")
                                                  else np.nan for v in fila[1:]]])
    nota = ["", "Sept price is Sept 1st close", "", "", "CPI estimated", *[""] * (ncol - 5)]
    g = pd.DataFrame(titulos + [cab] + filas + [nota])
    return g


def _xlsx(hojas: dict[str, pd.DataFrame]) -> bytes:
    pytest.importorskip("openpyxl")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for nombre, df in hojas.items():
            df.to_excel(w, sheet_name=nombre, index=False, header=nombre != "Data")
    return buf.getvalue()


def _gw_mensual(meses: int = 30) -> pd.DataFrame:
    fechas = pd.date_range("2023-01-01", periods=meses, freq="MS")
    return pd.DataFrame({"yyyymm": [int(f.strftime("%Y%m")) for f in fechas],
                         "Index": np.linspace(4000, 4500, meses), "tbl": np.linspace(0.04, 0.05, meses),
                         "b/m": np.linspace(0.2, 0.18, meses)})


def _fsi(dias: int = 10, inicio: str = "2026-09-28") -> bytes:
    fechas = pd.bdate_range(inicio, periods=dias)
    df = pd.DataFrame({"Date": fechas.strftime("%Y-%m-%d"), "OFR FSI": np.round(np.linspace(-2, -1, dias), 3),
                       "Credit": 0.1, "Funding": 0.2, "Volatility": 0.3, "United States": 0.4})
    return df.to_csv(index=False).encode()


def _crudo(raiz: Path, fuente: str, nombre: str, contenido: bytes, *, dia: str = "20261012",
           last_modified: str | None = "Mon, 12 Oct 2026 14:00:18 GMT",
           descargado: str = "2026-10-12T16:00:00+00:00") -> Path:
    carpeta = raiz / fuente
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{dia}_{nombre}"
    ruta.write_bytes(contenido)
    cab = {"Last-Modified": last_modified} if last_modified else {}
    meta = {"fuente": fuente, "url": "https://ejemplo.invalid/" + nombre, "descargado_utc": descargado,
            "estado": 200, "cabeceras": cab, "bytes": len(contenido),
            "sha256": hashlib.sha256(contenido).hexdigest()}
    ruta.with_name(ruta.name + ".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return ruta


def _entrada(lag: str = "45D", fin_periodo: str = "mes", frecuencia: str = "diaria", revisable: bool = True,
             fuente: str = "ken_french") -> dict:
    return {"fuente": fuente, "frecuencia": frecuencia, "metodo": "fichero", "lag": lag,
            "fin_periodo": fin_periodo, "revisable": revisable, "unidad": "%", "divisa": None}


def _ed(fecha: str, visto: str | None = None, origen: str = "crudo") -> ac.Edicion:
    t = pd.Timestamp(fecha, tz="UTC")
    return ac.Edicion(origen, Path("x"), t, "last_modified", pd.Timestamp(visto, tz="UTC") if visto else t)


# --- Ken French --------------------------------------------------------------------------------------
def test_t1_french_primer_bloque_cabecera_y_faltantes():
    df, info = ac.leer_french(_zip_french())
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 6 and df.index[0] == pd.Timestamp("2026-08-24")
    assert np.isnan(df.iloc[1]["SMB"]) and df.isna().sum().sum() == 1     # -99.99 -> nulo
    assert info["crsp"] == "202608" and info["creado_zip"].startswith("2026-09-24")
    assert "Copyright" in info["copyright"]
    assert df.index.max() < pd.Timestamp("2026-09-01")                     # no lee el bloque anual


def test_t1_french_mensual_y_errores():
    df, _ = ac.leer_french(_zip_french("F-F_Research_Data_Factors", mensual=True, faltante=False))
    assert df.index[0] == pd.Timestamp("2026-01-01") and df.index.freqstr is None and len(df) == 6
    malo = io.BytesIO()
    with zipfile.ZipFile(malo, "w") as z:
        z.writestr("x.csv", "texto\r\n,A,B\r\n20260102,1.0\r\n")
    with pytest.raises(ValueError, match="valores con 2 columnas"):
        ac.leer_french(malo.getvalue())
    vacio = io.BytesIO()
    with zipfile.ZipFile(vacio, "w") as z:
        z.writestr("x.csv", "solo texto\r\n")
    with pytest.raises(ValueError, match="vacío"):
        ac.leer_french(vacio.getvalue())


# --- Shiller: regresión del bug de octubres -------------------------------------------------------
def test_t1_shiller_octubre_no_es_enero_regresion():
    idx = ac.fechas_shiller(["1871.1", "1871.01", "1871.10", 1871.1, 1871.01, 1871.12, 2026.09, "Date", "", None,
                             1871.13])
    esperado = ["1871-10-01", "1871-01-01", "1871-10-01", "1871-10-01", "1871-01-01", "1871-12-01", "2026-09-01"]
    assert list(idx[:7]) == [pd.Timestamp(x) for x in esperado]
    assert idx[7:].isna().all()
    # El parseo de la Fase 1 por texto ('1871.1' -> '1871-1') convertía octubre en enero:
    bug = pd.to_datetime(pd.Series(["1871.01", "1871.1"]).str.replace(".", "-", regex=False), format="%Y-%m")
    assert bug.duplicated().any() and not ac.fechas_shiller(["1871.01", "1871.1"]).duplicated().any()


@pytest.mark.parametrize("texto", [True, False])
def test_t1_shiller_rejilla_sin_meses_repetidos(texto):
    df = ac.leer_shiller(_rejilla_shiller(24, texto=texto))
    assert len(df) == 24 and not df.index.has_duplicates and df.index.is_monotonic_increasing
    octubres = df.index[df.index.month == 10]
    assert list(octubres) == [pd.Timestamp("1871-10-01"), pd.Timestamp("1872-10-01")]
    assert {"P", "D", "E", "CPI", "Rate GS10", "CAPE"} <= set(df.columns)
    assert "Price" not in df.columns and "Earnings" not in df.columns     # nombres repetidos: fuera
    assert df["CAPE"].iloc[:3].isna().all() and df["CAPE"].iloc[3:].notna().all()   # 'NA' -> nulo


def test_t1_shiller_csv_y_errores(tmp_path):
    ruta = tmp_path / "ie_data.csv"
    _rejilla_shiller(15).to_csv(ruta, header=False, index=False)
    assert len(ac.leer_shiller(ruta)) == 15
    g = _rejilla_shiller(3)
    with pytest.raises(ValueError, match="cabecera"):
        ac.leer_shiller(g.iloc[8:])
    rep = pd.concat([g.iloc[:9], g.iloc[8:9]], ignore_index=True)
    with pytest.raises(ValueError, match="repetidos"):
        ac.leer_shiller(rep)


# --- Goyal-Welch, OFR y JST --------------------------------------------------------------------------
def test_t1_goyal_welch_xlsx(tmp_path):
    ruta = tmp_path / "gw.xlsx"
    ruta.write_bytes(_xlsx({"Monthly": _gw_mensual(), "Annual": pd.DataFrame({"yyyy": [2024, 2025], "tbl": [0.05, 0.04]})}))
    m = ac.leer_goyal_welch(ruta)
    assert m.index[0] == pd.Timestamp("2023-01-01") and list(m.columns) == ["Index", "tbl", "b/m"]
    a = ac.leer_goyal_welch(ruta, "Annual")
    assert list(a.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")]


def test_t1_ofr_csv(tmp_path):
    ruta = tmp_path / "fsi.csv"
    ruta.write_bytes(_fsi())
    df = ac.leer_ofr(ruta)
    assert "OFR FSI" in df.columns and len(df) == 10 and df.index[0] == pd.Timestamp("2026-09-28")
    with pytest.raises(ValueError, match="Date"):
        ac.leer_ofr(pd.DataFrame({"Fecha": ["2026-01-01"]}))


def test_t1_jst_tabla_salida_del_fichero(tmp_path):
    panel = pd.DataFrame({"year": [2006, 2007, 2008, 2007], "country": ["US", "US", "US", "Spain"],
                          "iso": ["USA", "USA", "USA", "ESP"], "crisisJST": [0, 1, 0, 1], "stir": [5.0, 4.5, 1.5, 4.0],
                          "ltrate": [4.8, 4.6, 3.7, 4.3], "cpi": [95.0, 98.0, 101.0, 100.0], "unemp": [4.6, 4.6, 5.8, 8.2]})
    _crudo(tmp_path, "jst_macrohistory", "JSTdatasetR6.xlsx", _xlsx({"Sheet1": panel}),
           last_modified="Wed, 10 Jul 2024 08:29:43 GMT")
    usa = ac.leer_jst(next((tmp_path / "jst_macrohistory").glob("*.xlsx")))
    assert len(usa) == 3 and usa.loc["2007-01-01", "crisisJST"] == 1
    t = ac.tabla_jst(raiz_crudo=tmp_path)
    assert set(t["serie"]) == {f"jst:USA.{c}" for c in ac.COLUMNAS_JST}
    assert (t["fecha_publicacion"] == pd.Timestamp("2024-07-10")).all()   # 08:29 UTC < 16:30 NY
    assert (t["metodo_publicacion"] == "fichero").all() and t["nota"].str.contains(cat_mod.NOTA_REVISABLE_EDICION, regex=False).all()
    assert len(ac.tabla_jst(raiz_crudo=tmp_path / "vacio")) == 0


# --- Ediciones -----------------------------------------------------------------------------------------
def test_t1_ediciones_por_contenido(tmp_path):
    z1, z2 = _zip_french(dias=5), _zip_french(dias=6)
    lm = "Fri, 25 Sep 2026 21:55:02 GMT"
    _crudo(tmp_path, "ken_french", "F-F_Research_Data_Factors_daily_CSV.zip", z1, dia="20261001",
           last_modified=lm, descargado="2026-10-01T15:00:00+00:00")
    _crudo(tmp_path, "ken_french", "F-F_Research_Data_Factors_daily_CSV.zip", z1, dia="20261005",
           last_modified=lm, descargado="2026-10-05T15:00:00+00:00")
    _crudo(tmp_path, "ken_french", "F-F_Research_Data_Factors_daily_CSV.zip", z2, dia="20261026",
           last_modified="Fri, 23 Oct 2026 20:00:00 GMT", descargado="2026-10-26T15:00:00+00:00")
    eds = ac.ediciones("ken_french", "F-F_Research_Data_Factors_daily", tmp_path)
    assert len(eds) == 2
    assert eds[0].fecha_edicion == pd.Timestamp("2026-09-25 21:55:02", tz="UTC")
    assert eds[0].visto == pd.Timestamp("2026-10-05 15:00", tz="UTC")      # última descarga con ese contenido
    assert eds[1].fuente_fecha == "last_modified"


def test_t1_ediciones_xlsx_sin_last_modified(tmp_path):
    _crudo(tmp_path, "goyal_welch", "gw_predictores.xlsx", _xlsx({"Monthly": _gw_mensual()}), last_modified=None,
           descargado="2099-01-01T00:00:00+00:00")
    (e,) = ac.ediciones("goyal_welch", "gw_predictores.xlsx", tmp_path)
    assert e.fuente_fecha == "xlsx_modificado" and e.fecha_edicion < e.visto
    _crudo(tmp_path, "ofr_fsi", "fsi.csv", _fsi(), last_modified=None)
    (o,) = ac.ediciones("ofr_fsi", "fsi.csv", tmp_path)
    assert o.fuente_fecha == "descarga" and o.fecha_edicion == o.visto


# --- Fechado: regla, cota de edición, provisionales y huecos -------------------------------------------
def _serie(fechas, valores=None) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(fechas))
    return pd.Series(np.arange(1.0, len(idx) + 1) if valores is None else valores, index=idx)


def test_t1_regla_fichero_french_45_dias():
    s = _serie(pd.bdate_range("2026-08-03", "2026-08-31"))
    t = validar_series(ac.tabla_serie("ff:X", _entrada(), [(_ed("2026-09-25 21:55"), s)]))
    assert (t["fecha_publicacion"] == pd.Timestamp("2026-10-15")).all()     # 31-ago + 45 días
    assert t["disponible_utc"].isna().all() and (t["vintage"] == "unico").all()
    assert t["nota"].str.contains(cat_mod.NOTA_REVISABLE_EDICION, regex=False).all()
    assert t["nota"].str.contains("CRSP").all()


def test_t1_cota_inferior_de_edicion_vista_sin_el_periodo():
    """Shiller: una edición vista el 12-oct sin septiembre cerrado -> septiembre no antes del 12-oct."""
    e = _entrada(lag="1D", frecuencia="mensual", fuente="shiller")
    vieja = _serie(["2026-07-01", "2026-08-01", "2026-09-01"])        # septiembre provisional (día 1)
    nueva = _serie(["2026-07-01", "2026-08-01", "2026-09-01", "2026-10-01"])
    info = ac._Info("shiller:X", "shiller", "ie_data.xls", "CAPE")
    t = ac.tabla_serie("shiller:X", e, [(_ed("2026-09-02 17:51", "2026-10-12 16:05"), vieja),
                                        (_ed("2026-10-14 15:00"), nueva)], info)
    pub = t.set_index("fecha_referencia")["fecha_publicacion"]
    assert pub["2026-08-01"] == pd.Timestamp("2026-09-01")        # regla (la vieja ya lo tenía)
    assert pub["2026-09-01"] == pd.Timestamp("2026-10-12")        # cota: vista sin él el 12-oct 12:05 NY
    assert "2026-10-01" not in pub.index.strftime("%Y-%m-%d")     # octubre no cerrado el 14-oct
    assert info.con_cota == 1 and info.provisionales == 1 and info.ediciones == 2
    assert info.ultimo_por_edicion[0]["ultimo"] == "2026-08-01"


def test_t1_cota_no_adelanta_la_regla():
    """La copia de data/raw vista el 20-jul sin junio no cambia la regla de 45 días (más tardía)."""
    vieja = _serie(pd.bdate_range("2026-05-01", "2026-05-29"))
    nueva = _serie(pd.bdate_range("2026-05-01", "2026-08-31"))
    t = ac.tabla_serie("ff:X", _entrada(), [(_ed("2026-07-20 19:37", origen="fase1"), vieja),
                                            (_ed("2026-09-25 21:55"), nueva)])
    pub = t.set_index("fecha_referencia")["fecha_publicacion"]
    assert pub["2026-06-15"] == pd.Timestamp("2026-08-14") and pub["2026-05-29"] == pd.Timestamp("2026-07-15")


def test_t1_valores_de_la_edicion_vista_mas_reciente():
    vieja = _serie(["2025-06-01", "2025-07-01"], [1.0, 2.0])
    nueva = _serie(["2025-06-01", "2025-07-01"], [1.5, 2.0])
    # data/raw (mtime posterior al Last-Modified) vista antes que el crudo: manda el crudo.
    t = ac.tabla_serie("gw:X", _entrada("7M1D", "anio", "mensual", fuente="goyal_welch"),
                       [(_ed("2026-07-20 19:00", origen="fase1"), vieja), (_ed("2026-07-01", "2026-10-12"), nueva)])
    assert t["valor"].tolist() == [1.5, 2.0]
    assert (t["fecha_publicacion"] == pd.Timestamp("2026-08-01")).all()


def test_t1_huecos_mensuales_con_nota_y_no_revisable():
    s = _serie(["2025-01-01", "2025-02-01", "2025-04-01"])
    t = validar_series(ac.tabla_serie("x", _entrada("1D", frecuencia="mensual", revisable=False),
                                      [(_ed("2026-01-10"), s)]))
    hueco = t[t["valor"].isna()]
    assert len(t) == 4 and hueco["fecha_referencia"].tolist() == [pd.Timestamp("2025-03-01")]
    assert hueco["nota"].iloc[0] == ac.NOTA_HUECO and t["nota"].isna().sum() == 3


def test_t1_sin_lag_o_sin_versiones_es_error():
    with pytest.raises(ValueError, match="sin versiones"):
        ac.tabla_serie("x", _entrada(), [])
    e = _entrada() | {"metodo": "calendario"}
    with pytest.raises(ValueError, match="sin lag"):
        ac.tabla_serie("x", e, [(_ed("2026-01-01"), _serie(["2025-01-02"]))])


# --- Tabla completa con el catálogo real (ficheros sintéticos) -------------------------------------
IDS = ("ff:F3D.Mkt-RF", "ff:F3D.SMB", "shiller:CAPE", "shiller:LONGRATE", "gw:M.tbl", "gw:M.bm", "ofr:FSI")


def _catalogo(ids=IDS) -> dict:
    cat = cat_mod.cargar_catalogo()
    return {k: cat[k] for k in ids}


def _crudos_sinteticos(raiz: Path) -> None:
    _crudo(raiz, "ken_french", "F-F_Research_Data_Factors_daily_CSV.zip", _zip_french(dias=8),
           last_modified="Fri, 25 Sep 2026 21:55:02 GMT")   # 24-ago a 2-sep: septiembre sin cerrar
    data = _rejilla_shiller(24, texto=False, provisional=False)
    # Último mes = mes de la edición (provisional): 1872-12 con edición del 2-dic-1872 no existe;
    # se usa una edición «antigua» para que diciembre de 1872 sea provisional.
    _crudo(raiz, "shiller", "ie_data.xls", _xlsx({"Data": data}), last_modified="Mon, 02 Dec 1872 17:00:00 GMT")
    _crudo(raiz, "goyal_welch", "gw_predictores.xlsx", _xlsx({"Monthly": _gw_mensual(36)}), last_modified=None)
    _crudo(raiz, "ofr_fsi", "fsi.csv", _fsi())


def test_t1_tabla_academicos_sintetica(tmp_path):
    raiz = tmp_path / "crudo"
    _crudos_sinteticos(raiz)
    cat = _catalogo()
    t = ac.tabla_academicos(catalogo=cat, raiz_crudo=raiz, raiz_fase1=tmp_path / "sin_raw")
    cat_mod.validar_contra_catalogo(t, cat)
    assert set(t["serie"]) == set(IDS)
    por = t.groupby("serie")
    assert por.size()["ff:F3D.Mkt-RF"] == 6                                 # 1 y 2 de septiembre: fuera
    assert por.size()["ff:F3D.SMB"] == 5                                    # diaria: el -99.99 no es fila
    assert (t.loc[t["serie"] == "ff:F3D.Mkt-RF", "fecha_publicacion"] == pd.Timestamp("2026-10-15")).all()
    cape = t[t["serie"] == "shiller:CAPE"]
    assert cape["fecha_referencia"].max() == pd.Timestamp("1872-11-01")     # diciembre provisional: fuera
    assert (cape["fecha_referencia"].dt.month == 10).sum() == 2             # 1871.1 y 1872.1 son octubres
    gw = t[t["serie"] == "gw:M.tbl"]
    assert gw["fecha_publicacion"].max() == pd.Timestamp("2026-08-01")      # GW_ANUAL: 1-ago del año siguiente
    assert (t["vintage"] == "unico").all() and t["disponible_utc"].isna().all()
    res = ac.resumen_academicos(catalogo=cat, raiz_crudo=raiz, raiz_fase1=tmp_path / "sin_raw")
    fila = res.set_index("id").loc["shiller:CAPE"]
    assert fila["provisionales_descartados"] == 1 and fila["origen"] == "crudo"
    assert res.set_index("id").loc["ff:F3D.Mkt-RF", "provisionales_descartados"] == 2
    assert res.set_index("id").loc["ff:F3D.Mkt-RF", "retraso_edicion_dias"] == 25
    # SHILLER_FICHERO (1M1D) nunca adelanta la edición (con la regla 1D de la Fase 1 salía < 0).
    assert res.set_index("id").loc["shiller:CAPE", "regla_vs_edicion_dias"] > 0


def test_t1_shiller_columnas_propuestas_p_d_e_cpi(tmp_path):
    """P, D, E y CPI (aún sin entrada en el catálogo) salen con una entrada como la propuesta."""
    raiz = tmp_path / "crudo"
    _crudos_sinteticos(raiz)
    base = _catalogo(("shiller:CAPE",))["shiller:CAPE"]
    cat = {f"shiller:{c}": base | {"id_origen": f"ie_data.xls (col {c})", "alias": [], "fichero_fase1": [],
                                    "lag": "1M1D", "unidad": "u", "divisa": "USD" if c in "PDE" else None}
           for c in ("P", "D", "E", "CPI")}
    t = ac.tabla_academicos(catalogo=cat, raiz_crudo=raiz, raiz_fase1=tmp_path)
    cat_mod.validar_contra_catalogo(t, cat)
    assert t.groupby("serie").size().to_dict() == {f"shiller:{c}": 23 for c in ("CPI", "D", "E", "P")}
    assert t.loc[t["fecha_referencia"] == pd.Timestamp("1871-01-01"), "fecha_publicacion"].eq(
        pd.Timestamp("1871-03-01")).all()


def test_t1_tabla_academicos_rechaza_unidad_distinta(tmp_path):
    raiz = tmp_path / "crudo"
    _crudos_sinteticos(raiz)
    cat = _catalogo(("ofr:FSI",))
    t = ac.tabla_academicos(catalogo=cat, raiz_crudo=raiz, raiz_fase1=tmp_path)
    t.loc[0, "unidad"] = "otra"
    with pytest.raises(ErrorContrato):
        cat_mod.validar_contra_catalogo(t, cat)


def test_t1_ofr_sin_crudo_usa_data_raw(tmp_path):
    pytest.importorskip("pyarrow")
    raw = tmp_path / "raw" / "ofr"
    raw.mkdir(parents=True)
    idx = pd.bdate_range("2026-07-01", "2026-07-16")
    pd.DataFrame({"OFR_FSI": np.linspace(-1, 1, len(idx))}, index=idx).to_parquet(raw / "OFR_FSI.parquet")
    cat = _catalogo(("ofr:FSI",))
    t = ac.tabla_academicos(catalogo=cat, raiz_crudo=tmp_path / "crudo", raiz_fase1=tmp_path / "raw")
    assert len(t) == len(idx) and (t["metodo_publicacion"] == "lag_fijo").all()
    assert t["fecha_publicacion"].iloc[0] == pd.Timestamp("2026-07-06")      # 5D desde la fecha
    res = ac.resumen_academicos(catalogo=cat, raiz_crudo=tmp_path / "crudo", raiz_fase1=tmp_path / "raw")
    assert res["origen"].iloc[0] == "fase1"


def test_t1_localizar_entradas_del_catalogo():
    cat = cat_mod.cargar_catalogo()
    for sid in cat_mod.filtrar(cat, fuente=ac.FUENTES_CATALOGO):
        clave, col, _ = ac.localizar(cat[sid])
        assert clave in ac.FICHEROS[cat[sid]["fuente"]] and col
    assert ac.localizar(cat["shiller:LONGRATE"])[1] == "Rate GS10"
    assert ac.localizar(cat["gw:M.bm"]) == ("gw_predictores.xlsx", "b/m", "Monthly")
    with pytest.raises(ValueError):
        ac.localizar({"fuente": "fred_alfred", "id_origen": "DGS10"})


# --- Descarga (red simulada) ------------------------------------------------------------------------------
class _Resp:
    def __init__(self, contenido: bytes, estado: int = 200, tipo: str = "application/x-zip-compressed"):
        self.content, self.status_code, self.url = contenido, estado, "https://ejemplo.invalid/f"
        self.headers = {"Content-Type": tipo, "Last-Modified": "Fri, 25 Sep 2026 21:55:02 GMT", "Set-Cookie": "x"}
        self.text = contenido.decode("latin-1")


def test_t1_descargar_cache_403_y_html(tmp_path, monkeypatch):
    import requests

    llamadas = []

    def falso_get(url, **kw):
        llamadas.append(url)
        if "fsi.csv" in url:
            e = requests.HTTPError("403 Forbidden")
            e.response = _Resp(b"<html>bloqueado</html>", 403, "text/html")
            raise e
        if "docs.google" in url:
            return _Resp(b"<html>reto</html>", 200, "text/html; charset=utf-8")
        return _Resp(_zip_french())

    monkeypatch.setattr(ac.http, "get", falso_get)
    monkeypatch.setattr(ac.http, "robots_permite", lambda url: True)
    (m,) = ac.descargar("ken_french", claves=["F-F_Research_Data_Factors"], raiz=tmp_path)
    assert m["estado"] == 200 and "error" not in m and "Set-Cookie" not in m["cabeceras"]
    assert len(ac.crudos("ken_french", "F-F_Research_Data_Factors", tmp_path)) == 1
    ac.descargar("ken_french", claves=["F-F_Research_Data_Factors"], raiz=tmp_path)
    assert len(llamadas) == 1                                                # mismo día: caché
    (o,) = ac.descargar("ofr_fsi", raiz=tmp_path)
    assert o["estado"] == 403 and "no se esquiva" in o["error"] and not ac.crudos("ofr_fsi", "fsi.csv", tmp_path)
    (g,) = ac.descargar("goyal_welch", raiz=tmp_path)
    assert "HTML" in g["error"] and not ac.crudos("goyal_welch", "gw_predictores.xlsx", tmp_path)
    monkeypatch.setattr(ac.http, "robots_permite", lambda url: False)
    (j,) = ac.descargar("jst_macrohistory", raiz=tmp_path)
    assert "robots" in j["error"]
    with pytest.raises(ValueError):
        ac.ruta_crudo("otra", tmp_path)


def test_t1_fichas_sinteticas(tmp_path):
    from ingesta.comun.fichas import guardar_ficha

    raiz = tmp_path / "crudo"
    _crudos_sinteticos(raiz)
    res = ac.resumen_academicos(catalogo=_catalogo(), raiz_crudo=raiz, raiz_fase1=tmp_path)
    fichas = {f.fuente: f for f in ac.fichas_academicos(res, raiz_crudo=raiz)}
    assert set(fichas) == set(ac.FUENTES)
    assert fichas["ken_french"].estado_sonda == "ok" and fichas["jst_macrohistory"].estado_sonda == "no_probada"
    assert fichas["ken_french"].mediciones["ediciones"]["F-F_Research_Data_Factors_daily"][0]["crsp"] == "202608"
    assert fichas["ofr_fsi"].cobertura_inicio == "2026-09-28"
    ruta = guardar_ficha(fichas["shiller"], tmp_path / "fichas")
    assert json.loads(ruta.read_text(encoding="utf-8"))["fuente"] == "shiller"


# --- Datos locales ----------------------------------------------------------------------------------------
@pytest.mark.datos
def test_t1_bug_shiller_longrate_en_data_raw():
    ruta = ROOT / "data" / "raw" / "academico" / "SHILLER_LONGRATE.parquet"
    if not ruta.exists():
        pytest.skip("falta data/raw/academico/SHILLER_LONGRATE.parquet")
    pytest.importorskip("pyarrow")
    idx = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(ruta).index))
    assert idx.duplicated().sum() == 153 and (idx.month == 10).sum() == 0   # octubres leídos como enero


@pytest.mark.datos
def test_t1_tabla_academicos_crudos_locales():
    if not any(ac.crudos("ken_french", "F-F_Research_Data_Factors_daily")):
        pytest.skip("sin crudos locales de Ken French (python -m ingesta.mercados_series.fuentes.academicos "
                    "--descargar)")
    t = ac.tabla_academicos()
    assert set(t["fuente"]) == set(ac.FUENTES_CATALOGO)
    cape = t[t["serie"] == "shiller:CAPE"]
    assert not cape["fecha_referencia"].duplicated().any()
    assert (t["fecha_publicacion"] >= t["fecha_referencia"]).all()


# --- Red (INGESTA_LIVE=1) ----------------------------------------------------------------------------------
@pytest.mark.live
def test_t1_live_french_mensual(tmp_path):
    (m,) = ac.descargar("ken_french", claves=["F-F_Research_Data_Factors"], raiz=tmp_path)
    assert m["estado"] == 200, m.get("error")
    df, info = ac.leer_french(ac.crudos("ken_french", "F-F_Research_Data_Factors", tmp_path)[-1][0])
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"] and info["crsp"] is not None
    assert m["cabeceras"].get("Last-Modified")
    assert (df.index.max() + pd.offsets.MonthEnd(1)) < pd.Timestamp.now()


@pytest.mark.live
def test_t1_live_ofr_sin_bloqueo(tmp_path):
    (m,) = ac.descargar("ofr_fsi", raiz=tmp_path)
    if m.get("estado") == 403:
        pytest.skip("la OFR devuelve 403 al User-Agent identificable: se usa data/raw (documentado)")
    assert m["estado"] == 200, m.get("error")
    df = ac.leer_ofr(ac.crudos("ofr_fsi", "fsi.csv", tmp_path)[-1][0])
    assert {"OFR FSI", "Credit", "Funding", "Volatility", "United States"} <= set(df.columns)
    ed = ac.ediciones("ofr_fsi", "fsi.csv", tmp_path)[-1]
    assert 0 <= (ed.fecha_edicion - pd.Timestamp(df.index.max(), tz="UTC")).days <= 10
    assert str(ed.fecha_edicion.tz) == "UTC"
