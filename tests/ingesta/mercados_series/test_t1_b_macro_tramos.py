"""Tema 1 · macro internacional y calendarios: tramos de lags por época, empalmes, WEO, OCDE y curvas.

Sin red: fixtures sintéticas y la configuración del repositorio (lags.yaml, catálogo).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from ingesta.mercados_series import base, catalogo as c
from ingesta.mercados_series.contrato import validar_series
from ingesta.mercados_series.fuentes import curvas, eurostat, oecd

TRAMOS_MIN = """\
version: 1
lag_publicacion_fase1: {}
adicionales:
  ZZ_TRAMOS:
    lag: 2M
    fin_periodo: mes
    verificado: false
    justificacion: prueba
    tramos:
    - desde: '2012-01-01'
      lag: 1M10D
      habiles: target
      verificado: true
      fuente: prueba
      muestras:
      - {periodo: '2013-11-01', publicado: '2014-01-08', fuente: prueba}
"""


def _lags(tmp_path: Path, texto: str = TRAMOS_MIN) -> dict:
    r = tmp_path / "lags.yaml"
    r.write_text(texto, encoding="utf-8")
    return c.cargar_lags(r)


# --- Tramos por época ---------------------------------------------------------------------------
def test_t1_b_tramos_eligen_regla_por_fecha_de_referencia(tmp_path):
    e = _lags(tmp_path)["ZZ_TRAMOS"]
    fechas = pd.Series(pd.to_datetime(["2011-11-01", "2011-12-01", "2012-01-01", "2013-11-01"]), index=[5, 6, 7, 8])
    pub = c.fecha_publicacion_clave(fechas, e)
    assert list(pub.index) == [5, 6, 7, 8]                      # mismo índice que la entrada
    assert pub[5] == pd.Timestamp("2012-01-31")                # antes del tramo: 2M (fin de M+2)
    assert pub[6] == pd.Timestamp("2012-02-29")
    assert pub[7] == pd.Timestamp("2012-03-12")                # 1M10D = sábado 10-mar -> lunes TARGET
    assert pub[8] == pd.Timestamp("2014-01-10")


def test_t1_b_sin_tramos_es_identico_a_fecha_publicacion_lag():
    fechas = pd.Series(pd.date_range("2000-01-01", periods=40, freq="MS"))
    a = c.fecha_publicacion_tramos(fechas, "1M1D", "mes", habiles="us_federal")
    b = c.fecha_publicacion_lag(fechas, "1M1D", "mes", habiles="us_federal")
    pd.testing.assert_series_equal(a, b)


@pytest.mark.parametrize("cambio, mensaje", [
    (("      lag: 1M10D\n", "      lag: 1X\n"), "lag"),
    (("    - desde: '2012-01-01'\n", "    - hasta: '2011-01-01'\n      desde: '2012-01-01'\n"), "anterior"),
    (("      habiles: target\n", "      habiles: luna\n"), "habiles"),
    (("periodo: '2013-11-01'", "periodo: '2010-11-01'"), "fuera del tramo"),
    (("      verificado: true\n", ""), "verificado"),
])
def test_t1_b_tramos_mal_formados(tmp_path, cambio, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        _lags(tmp_path, TRAMOS_MIN.replace(*cambio))


def test_t1_b_tramos_solapados(tmp_path):
    texto = TRAMOS_MIN + """\
    - desde: '2011-06-01'
      lag: 1M
      verificado: false
      fuente: prueba
"""
    with pytest.raises(ValueError, match="ordenados"):
        _lags(tmp_path, texto)


def test_t1_b_muestra_que_adelanta_la_publicacion_falla(tmp_path):
    lags = _lags(tmp_path, TRAMOS_MIN.replace("publicado: '2014-01-08'", "publicado: '2014-01-20'"))
    with pytest.raises(ValueError, match="anteriores a la publicación real"):
        c.validar_muestras_tramos(lags)


def test_t1_b_reglas_reales_nunca_antes_de_la_publicacion_real():
    """lags.yaml real: ninguna muestra archivada (notas de prensa de Eurostat) queda antes de la regla,
    tampoco en las fronteras de tramo (primera y última de cada tramo)."""
    filas = c.validar_muestras_tramos()
    claves = {f["clave"] for f in filas}
    assert {"EUROSTAT_PARO", "EUROSTAT_PIB"} <= claves
    assert all(f["holgura_dias"] >= 0 for f in filas)
    por = {(f["clave"], f["periodo"]) for f in filas}
    # Fronteras del PIB: último trimestre t+45 (2015T4) y primero t+30 (2016T1); primera flash (2003T1).
    assert {("EUROSTAT_PIB", "2015-10-01"), ("EUROSTAT_PIB", "2016-01-01"), ("EUROSTAT_PIB", "2003-01-01")} <= por
    # Noviembre (el más tardío del año) en el paro.
    assert ("EUROSTAT_PARO", "2023-11-01") in por


def test_t1_b_frontera_de_tramo_paro_y_pib():
    lags = c.cargar_lags()
    paro = c.fecha_publicacion_clave(pd.Series(pd.to_datetime(["2011-11-01", "2011-12-01", "2012-01-01"])),
                                     "EUROSTAT_PARO", lags)
    # Noviembre de 2011 (tramo conservador) no se ve antes de su publicación real (2012-01-05 o más tarde).
    assert paro[0] == pd.Timestamp("2012-01-31") and paro[2] == pd.Timestamp("2012-03-12")
    pib = c.fecha_publicacion_clave(pd.Series(pd.to_datetime(["2002-10-01", "2003-01-01", "2015-10-01",
                                                              "2016-01-01"])), "EUROSTAT_PIB", lags)
    assert list(pib) == [pd.Timestamp("2003-03-10"), pd.Timestamp("2003-05-19"), pd.Timestamp("2016-02-18"),
                         pd.Timestamp("2016-05-05")]
    # Países: nunca la flash t+30 de los agregados.
    paises = c.fecha_publicacion_clave(pd.Series(pd.to_datetime(["2020-10-01"])), "EUROSTAT_PIB_PAISES", lags)
    assert paises[0] == pd.Timestamp("2021-02-18")


def test_t1_b_catalogo_sin_lags_de_ee_uu_en_eurostat_bce():
    cat = c.cargar_catalogo()
    for sid, e in cat.items():
        if sid.startswith(("eurostat:une_rt_m", "eurostat:namq_10_gdp")):
            assert e["lag_clave"].startswith("EUROSTAT_"), sid
            assert e["lag_tramos"] or e["lag_clave"] == "EUROSTAT_PARO_UK", sid
    assert cat["ecb:LFSI.M.U2.S.UNEHRT.TOTAL0.15_74.T"]["regla_previa_alfred"]["lag_clave"] == "EUROSTAT_PARO"
    assert cat["ecb:MNA.Q.Y.I10.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.GY"]["regla_previa_alfred"]["lag_tramos"]
    # La entrada previa conserva los tramos (fecha_publicacion_regla los aplica).
    previa = c.entrada_previa(cat["ecb:LFSI.M.U2.S.UNEHRT.TOTAL0.15_74.T"])
    assert c.fecha_publicacion_regla(pd.Series(pd.to_datetime(["2013-11-01"])), previa)[0] == pd.Timestamp("2014-01-10")


def test_t1_b_alias_zona_euro_por_defecto():
    cat = c.cargar_catalogo()
    for a, sid in eurostat.ALIAS_ZONA_EURO.items():
        assert c.resolver(a, cat) == sid and sid in cat
    assert c.por_id("HICP_EA", cat)["lag_clave"] == "CPIAUCSL"
    assert eurostat.CREACION_GEO == {"EA20": "2023-03-01", "EA21": "2026-03-01"}


# --- HICP empalmado -------------------------------------------------------------------------------
def _hicp(conj: str, geo: str, meses: list[str], valores: list[float]) -> pd.DataFrame:
    c_ = eurostat.CONJUNTOS[conj]
    sid = eurostat.id_eurostat(conj, c_["clave"], geo)
    obs = pd.DataFrame({"geo": geo, "TIME_PERIOD": meses, "OBS_VALUE": valores, "OBS_FLAG": None})
    return eurostat.tabla_conjunto(sid, obs, {"clave": f"{conj}/{c_['clave']}{geo}", "lag_clave": "CPIAUCSL",
                                              "unidad": c_["unidad"]})


def test_t1_b_hicp_empalmado_v1_luego_v2():
    v1 = _hicp("prc_hicp_manr", "EA", ["2025-10", "2025-11", "2025-12"], [2.1, 2.1, 2.0])
    v2 = _hicp("prc_hicp_minr", "EA", ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02"], [2.2, 2.0, 2.0, 1.7, 1.9])
    df = validar_series(pd.concat([v1, v2], ignore_index=True))
    emp = eurostat.empalmar_hicp(df)
    assert set(emp["serie"]) == {"eurostat:HICP_EMP.EA"}
    assert len(emp) == 5 and not emp.duplicated(["fecha_referencia"]).any()
    v1_meses = emp[emp["nota"].str.contains("tramo v1")]
    assert list(v1_meses["valor"]) == [2.1, 2.1, 2.0]                 # todos los periodos de v1, con su valor
    assert emp["nota"].str.contains(eurostat.NOTA_EMPALME_HICP, regex=False).all()
    nuevos = emp[emp["fecha_referencia"] >= "2026-01-01"]
    assert nuevos["nota"].str.contains("tramo v2").all()
    # v2 no existía antes de su creación: sus filas nunca antes de no_antes_de.
    assert (nuevos["fecha_publicacion"] >= pd.Timestamp(eurostat.CONJUNTOS["prc_hicp_minr"]["no_antes_de"])).all()
    # Con frontera explícita, solo cuenta como v1 lo publicado antes de ella.
    emp2 = eurostat.empalmar_hicp(df, frontera="2026-01-15")
    assert emp2.loc[emp2["fecha_referencia"] == "2025-12-01", "nota"].str.contains("tramo v2").all()


# --- Empalme PIB real de EE. UU. ------------------------------------------------------------------
def _fila(serie, ref, pub, vintage, valor, metodo="vintage_exacto"):
    return {"serie": serie, "fecha_referencia": pd.Timestamp(ref), "fecha_publicacion": pd.Timestamp(pub),
            "disponible_utc": pd.NaT, "vintage": vintage, "valor": valor, "unidad": "u", "divisa": "USD",
            "fuente": "x", "metodo_publicacion": metodo, "nota": None}


def test_t1_b_empalme_pib_real_usa_por_vintage():
    filas = [_fila("philly:ROUTPUT", "1991-04-01", "1991-09-01", "1991-08", 4800.0),
             _fila("philly:ROUTPUT", "1991-07-01", "1991-12-01", "1991-11", 4820.0),
             _fila("fred:GDPC1", "1950-01-01", "1950-05-01", "1991-12-04", 1700.0, "lag_fijo"),
             _fila("fred:GDPC1", "1991-07-01", "1991-12-04", "1991-12-04", 4870.0),
             _fila("fred:GDPC1", "1991-10-01", "1992-01-31", "1992-01-31", 4890.0)]
    b = pd.DataFrame(filas)
    b["disponible_utc"] = pd.to_datetime(b["disponible_utc"], utc=True)
    cat = c.cargar_catalogo()
    u, n = base.empalmar_pib_real_usa(b, cat)
    e = u[u["serie"] == base.EMPALME_PIB_USA]
    assert n == 4 == len(e)                                     # la fila lag_fijo previa a ALFRED, fuera
    assert set(e["vintage"]) == {"1991-08-31", "1991-11-30", "1991-12-04", "1992-01-31"}
    assert e["nota"].str.contains("tramo PNB").sum() == 2 and e["nota"].str.contains("tramo PIB").sum() == 2
    validar_series(e.reset_index(drop=True))
    c.validar_contra_catalogo(e.reset_index(drop=True), cat)
    # Sin uno de los tramos no se inventa nada.
    assert base.empalmar_pib_real_usa(b[b["serie"] != "fred:GDPC1"], cat)[1] == 0


# --- OCDE ---------------------------------------------------------------------------------------
def test_t1_b_oecd_g4e_no_es_la_zona_euro():
    assert "G4E" in oecd.AREAS
    e = c.por_id("oecd:LI_AA.G4E")
    assert any("NO es la zona euro" in n for n in e["notas"])
    assert not c.por_id("oecd:LI_AA.USA")["reestimada"]          # revisión parcial, no entera (medido)
    assert any("re-estimado entero" in n for n in c.por_id("oecd:LI_AA.USA")["notas"])


# --- Curvas: hábil siguiente con festivos reales --------------------------------------------------
def test_t1_b_curvas_habil_siguiente_target():
    assert curvas.habil_siguiente("2026-04-02", "target") == pd.Timestamp("2026-04-07")   # Viernes Santo y Lunes de Pascua
    assert curvas.habil_siguiente("2026-12-24", "target") == pd.Timestamp("2026-12-28")
    assert curvas.habil_siguiente("2026-10-09", "semana") == pd.Timestamp("2026-10-12")
    disp = curvas.disponible_habil_siguiente(pd.DatetimeIndex(["2026-04-01", "2026-04-02"]), "12:00",
                                             "Europe/Berlin", "target")
    assert disp.iloc[-1] == pd.Timestamp("2026-04-07 10:00", tz="UTC")


@pytest.mark.parametrize("cal, dia, esperado", [
    ("XTKS", "2026-09-18", "2026-09-24"),     # Japón: 21-23 de septiembre festivos
    ("XTSE", "2026-06-30", "2026-07-02"),     # Canadá: Canada Day
    ("XTSE", "2026-10-09", "2026-10-13"),     # Canadá: Acción de Gracias
    ("XLON", "2026-08-28", "2026-09-01"),     # Reino Unido: Summer bank holiday
])
def test_t1_b_curvas_habil_siguiente_bolsas(cal, dia, esperado):
    pytest.importorskip("exchange_calendars")
    assert curvas.habil_siguiente(dia, cal) == pd.Timestamp(esperado)


def test_t1_b_curvas_calendario_por_publicador():
    assert curvas.CALENDARIO_HABILES == {"bce_curvas": "target", "mof_jgb": "XTKS", "boc_curva": "XTSE",
                                         "boe_curvas": "XLON"}


def test_t1_b_lags_yaml_muestras_con_fuente():
    crudo = yaml.safe_load(c.RUTA_LAGS.read_text(encoding="utf-8"))
    for clave in ("EUROSTAT_PARO", "EUROSTAT_PIB"):
        for t in crudo["adicionales"][clave]["tramos"]:
            assert t["verificado"] is True and len(t["muestras"]) >= 2
            assert all(m["fuente"].strip() for m in t["muestras"])


def test_t1_c_construir_base_con_empalme_pib_resume_sin_error(tmp_path):
    """Regresión (cierre): el empalme entraba en la base pero el resumen fallaba con KeyError
    (sin cargador asignado) después de escribirla, y la CLI no rehacía la tabla de cobertura."""
    cat = c.cargar_catalogo()

    def fila(serie, ref, pub, vintage, valor):
        e = cat[serie]
        return {"serie": serie, "fecha_referencia": pd.Timestamp(ref), "fecha_publicacion": pd.Timestamp(pub),
                "disponible_utc": pd.NaT, "vintage": vintage, "valor": valor, "unidad": e["unidad"],
                "divisa": e.get("divisa"), "fuente": e["fuente"], "metodo_publicacion": "vintage_exacto",
                "nota": None}

    def tabla(filas):
        t = pd.DataFrame(filas)
        t["disponible_utc"] = pd.to_datetime(t["disponible_utc"], utc=True)
        return validar_series(t)

    tablas = {"philly_rtdsm": tabla([fila("philly:ROUTPUT", "1991-04-01", "1991-09-01", "1991-08-31", 4800.0)]),
              "alfred": tabla([fila("fred:GDPC1", "1991-07-01", "1991-12-04", "1991-12-04", 4870.0)])}
    res, t = base.construir_base(tablas=tablas, catalogo=cat, ruta=tmp_path / "b.parquet", escribir=False)
    assert base.EMPALME_PIB_USA in set(t["serie"])
    assert res.por_cargador.get(base.CARGADOR_DERIVADAS, 0) >= 1


def test_t1_c_mof_viernes_antes_del_dia_del_deporte():
    """Regresión (cierre): con lunes-viernes, la JGB del viernes 2026-10-09 se veía el lunes 12, festivo
    en Japón (Día del Deporte); MoF publica el martes 13. Corregidas 15 filas de curvas_b6."""
    pytest.importorskip("exchange_calendars")
    pub = curvas.CALENDARIO_HABILES["mof_jgb"]
    assert curvas.habil_siguiente("2026-10-09", pub) == pd.Timestamp("2026-10-13")
