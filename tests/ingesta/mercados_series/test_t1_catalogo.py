"""Tests del catálogo de series del Tema 1 (bloques 2 y 3): ``catalogo``, ``config/lags.yaml`` y la fusión
con ``publicaciones.yaml`` (bloque 3: vintages de ALFRED, fuente viva, derivadas de los diferenciales de FRED). Sin red; los que leen ``data/raw`` llevan ``@pytest.mark.datos`` y se saltan
si falta. Datos de prueba sintéticos (ningún valor real de FRED/ICE).
"""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import pytest

from ingesta.comun.rutas import ROOT
from ingesta.mercados_series import catalogo as c
from ingesta.mercados_series import sonda_horas, tools
from ingesta.mercados_series import visibilidad as v
from ingesta.mercados_series.contrato import ErrorContrato, validar_series

RAW = ROOT / "data" / "raw"
requiere_raw = pytest.mark.skipif(not (RAW / "provenance.json").exists(), reason="falta data/raw (gitignored)")
ETF_AJUSTADOS = ("SPY", "RSP", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "IWM",
                 "IWD", "IWF", "GLD", "HYG", "IEF", "LQD", "TLT", "SPHB", "SPLV", "MTUM")


def _utc(texto: str) -> pd.Timestamp:
    return pd.Timestamp(texto, tz="UTC")


# --- Catálogo real ------------------------------------------------------------------------------
def test_t1_catalogo_real_valido_y_coherente():
    cat = c.cargar_catalogo()
    assert len(cat) >= 200
    for sid, e in cat.items():
        assert e["estado"] in c.ESTADOS and e["frecuencia"] in c.FRECUENCIAS, sid
        assert sid.split(":", 1)[0] in c.PREFIJOS, sid
        if e["metodo"] in ("lag_fijo", "fichero"):
            assert e["lag"] and e["fin_periodo"], sid
        if e["revisable"] and e["metodo"] != "vintage_exacto":
            assert any(c.nota_revisable(e) in n for n in e["notas"]), sid
        if e["estado"] == "duplicada":
            assert e["duplicado_de"] in cat, sid
    resumen = c.resumen()
    assert sum(resumen["estado"].values()) == len(cat)


def test_t1_catalogo_alias_de_la_fase1():
    assert c.resolver("VIX") == "fred:VIXCLS"
    assert c.resolver("VIX_VALIDATION_REF") == "fred:VIXCLS"       # copia con el mismo id de fuente
    assert c.resolver("SP500") == c.resolver("SP500_DAILY_DRAWDOWN_SRC") == "yf:^GSPC"
    assert c.resolver("MOODYS_BAA_AAA_SPREAD") == "deriv:DBAA-DAAA"
    assert c.resolver("FF_FACTORS_3_DAILY") == "ff:F3D.Mkt-RF"     # columna alias de Ken French
    assert c.resolver("fred:DGS10") == "fred:DGS10"                # un id se devuelve tal cual
    assert c.por_id("UNRATE")["publicador"] == "bls"
    assert c.resolver("BAA10Y") == "deriv:DBAA-DGS10"                  # bloque 3: la derivada propia
    with pytest.raises(KeyError, match="catálogo"):
        c.por_id("NO_EXISTE")


def test_t1_catalogo_estados_decididos():
    for tk in ETF_AJUSTADOS:
        e = c.por_id(f"yf:{tk}")
        assert e["estado"] == "excluida_ajustada" and e["metodo"] == "cierre_mercado", tk
    assert len(c.filtrar(estado="excluida_ajustada")) == len(ETF_AJUSTADOS)
    for indice in ("yf:^GSPC", "yf:^VIX9D", "yf:^TNX", "yf:DX-Y.NYB", "yf:GC=F"):
        assert c.por_id(indice)["estado"] == "activa", indice
    assert c.por_id("fred:BAMLH0A0HYM2")["estado"] == "capada"
    assert c.por_id("fred:WEI")["estado"] == "activa"                  # bloque 3: ya se carga de ALFRED
    assert c.por_id("fred:DTWEXM")["sucesor"] == "fred:DTWEXAFEGS"
    assert c.por_id("fred:IRLTLT01USM156N")["duplicado_de"] == "fred:GS10"
    assert c.por_id("shiller:LONGRATE")["fase1_utilizable"] is False
    deriv = c.por_id("deriv:DBAA-DAAA")
    assert deriv["metodo"] == "derivada" and deriv["componentes"] == ["fred:DBAA", "fred:DAAA"]


@pytest.mark.parametrize("sid, unidad, divisa", [
    ("fred:DEXJPUS", "JPY por 1 USD", None),           # tipo de cambio: no es un importe (bloque 2)
    ("fred:DEXUSAL", "USD por 1 AUD", None),           # cotización invertida
    ("fred:DEXUSEU", "USD por 1 EUR", None),
    ("fred:DGS10", "% anual", None),
    ("gw:M.tbl", "tanto por uno anual", None),         # Goyal-Welch en decimales
    ("fred:TB3MS", "% anual (base descuento)", None),
    ("yf:GC=F", "USD por onza troy", "USD"),
])
def test_t1_catalogo_unidades_y_divisas(sid, unidad, divisa):
    e = c.por_id(sid)
    assert (e["unidad"], e["divisa"]) == (unidad, divisa)


def test_t1_catalogo_publicacion_por_grupo():
    assert c.por_id("fred:DGS10")["publicador"] == "fed_h15"
    assert c.por_id("fred:DEXJPUS")["publicador"] == "fed_h10"
    assert c.por_id("fred:DBAA")["publicador"] == "moodys"
    assert c.por_id("fred:BAA10Y")["publicador"] == "fred_calculadas_dia_siguiente"
    assert c.por_id("fred:T10Y2Y")["publicador"] == "fred_tesoro_calculadas"
    assert c.por_id("fred:VIXCLS")["mercado"] == "CBOE_IDX"
    assert c.por_id("yf:^GSPC")["mercado"] == "XNYS"
    assert c.por_id("yf:DX-Y.NYB")["mercado"] == "ICE_DXY"
    assert c.por_id("yf:GC=F")["mercado"] == "CME"
    ff = c.por_id("ff:F3D.SMB")
    assert (ff["metodo"], ff["lag"], ff["fin_periodo"], ff["revisable"]) == ("fichero", "45D", "mes", True)
    gw = c.por_id("gw:M.Index")
    assert (gw["metodo"], gw["fin_periodo"]) == ("fichero", "anio")


@requiere_raw
@pytest.mark.datos
def test_t1_catalogo_cubre_cada_columna_de_data_raw_una_vez():
    pq = pytest.importorskip("pyarrow.parquet")
    columnas = set()
    for p in sorted(RAW.glob("*/*.parquet")):
        nombres = [n for n in pq.read_schema(p).names if not n.startswith("__")]
        columnas |= {f"{p.parent.name}/{p.name}:{col}" for col in nombres if col not in ("Date", "yyyymm")}
    declaradas = [f for e in c.cargar_catalogo().values() for f in e["fichero_fase1"]]
    assert len(declaradas) == len(set(declaradas))
    assert set(declaradas) == columnas


# --- lags.yaml ----------------------------------------------------------------------------------
def test_t1_lags_copia_congelada_de_lag_publicacion():
    lags_f1 = pytest.importorskip("regimenes.features.lags").LAG_PUBLICACION
    lags = c.cargar_lags()
    congeladas = {k: e for k, e in lags.items() if e["seccion"] == "lag_publicacion_fase1"}
    assert set(congeladas) == set(lags_f1), "la copia congelada no coincide con LAG_PUBLICACION de la Fase 1"
    for k, e in lags_f1.items():
        assert {x: congeladas[k][x] for x in ("meses", "dias", "tipo", "fuente")} == e, k


def _referencias(fin_periodo: str) -> pd.DatetimeIndex:
    if fin_periodo == "mes":
        return pd.date_range("1990-01-01", "2030-12-01", freq="MS")
    if fin_periodo == "trimestre":
        return pd.date_range("1990-01-01", "2030-10-01", freq="QS")
    return pd.date_range("1990-01-06", "2030-12-31", freq="7D")    # sábados / viernes de fin de semana


def test_t1_lags_conversion_exacta_desde_el_fin_del_periodo():
    """inicio + meses + días (Fase 1) == fin del periodo + lag (Tema 1), en 40 años de fechas."""
    for clave, e in c.cargar_lags().items():
        if e["seccion"] != "lag_publicacion_fase1":
            continue
        ref = pd.Series(_referencias(e["fin_periodo"]))
        esperado = ref + pd.DateOffset(months=e["meses"], days=e["dias"])
        obtenido = c.fecha_publicacion_lag(ref, e["lag"], e["fin_periodo"])
        assert (obtenido == esperado).all(), clave


def test_t1_lags_adicionales_conservadores_y_sin_verificar():
    lags = c.cargar_lags()
    pedidas = {"FF_MENSUAL", "IRLTLT01USM156N", "TWEXMMTH", "TWEXAFEGSMTH", "NBER_NY_COMMERCIAL_PAPER",
               "LIBOR_3M_QUARTERLY", "SHILLER_LONGRATE", "GH_US10Y_MONTHLY", "SP500_MONTHLY_DEEP_SRC", "ANFCI",
               "NFCI_RISK", "NFCI_CREDIT", "NFCI_LEVERAGE", "KCFSI", "NBER_RECESSION_MONTHLY",
               "RECESSION_PROB_SMOOTHED", "SAHM_RULE_REALTIME", "SAHM_RULE_CURRENT", "GDP_RECESSION_HAMILTON"}
    adicionales = {k for k, e in lags.items() if e["seccion"] == "adicionales"}
    assert pedidas <= adicionales
    for k in adicionales:
        assert lags[k]["verificado"] is False and lags[k]["justificacion"].strip(), k
        # Conservador: nunca antes del fin del periodo.
        ref = pd.Series(_referencias(lags[k]["fin_periodo"] if lags[k]["fin_periodo"] != "anio" else "mes"))
        fin = c.fin_de_periodo(ref, lags[k]["fin_periodo"])
        assert (c.fecha_publicacion_lag(ref, lags[k]["lag"], lags[k]["fin_periodo"]) > fin).all(), k


@pytest.mark.parametrize("ref, lag, fin, esperado", [
    ("2024-01-01", "1D", "mes", "2024-02-01"),
    ("2024-01-01", "1M1D", "mes", "2024-03-01"),
    ("2024-01-01", "11D", "mes", "2024-02-11"),        # WTI: 1 mes + 10 días desde el día 1
    ("2024-02-01", "1M1D", "mes", "2024-04-01"),       # febrero bisiesto
    ("2024-01-01", "1M1D", "trimestre", "2024-05-01"), # PIB trimestral: inicio + 4 meses
    ("2024-01-01", "24M", "mes", "2026-01-31"),
    ("2024-08-15", "45D", "mes", "2024-10-15"),        # FF diario: fin de agosto + 45 días
    ("2025-03-01", "7M1D", "anio", "2026-08-01"),      # Goyal-Welch: edición anual
    ("2024-03-09", "5D", "fecha", "2024-03-14"),       # semanal (sábado → jueves)
])
def test_t1_fecha_publicacion_lag_ejemplos(ref, lag, fin, esperado):
    assert c.fecha_publicacion_lag([ref], lag, fin).iloc[0] == pd.Timestamp(esperado)


def test_t1_fecha_publicacion_lag_errores_y_nulos():
    with pytest.raises(ValueError, match="meses"):
        c.fecha_publicacion_lag(["2024-01-01"], "1M", "fecha")
    with pytest.raises(ValueError, match="lag"):
        c.fecha_publicacion_lag(["2024-01-01"], "1S", "mes")
    with pytest.raises(ValueError, match="fin_periodo"):
        c.fin_de_periodo(["2024-01-01"], "semana")
    assert c.fecha_publicacion_lag([None, "2024-01-01"], "1D", "mes").isna().tolist() == [True, False]


def test_t1_fecha_publicacion_lag_propiedad():
    hypothesis = pytest.importorskip("hypothesis")
    st = pytest.importorskip("hypothesis.strategies")

    @hypothesis.settings(max_examples=200, deadline=None, database=None)
    @hypothesis.given(st.dates(min_value=pd.Timestamp("1900-01-01").date(), max_value=pd.Timestamp("2090-12-31").date()),
                      st.integers(0, 30), st.integers(0, 60), st.sampled_from(["mes", "trimestre", "anio"]))
    def propiedad(d, meses, dias, fin):
        lag = (f"{meses}M" if meses else "") + (f"{dias}D" if dias or not meses else "")
        pub = c.fecha_publicacion_lag([d], lag, fin).iloc[0]
        ultimo = c.fin_de_periodo([d], fin).iloc[0]
        assert pub >= ultimo >= pd.Timestamp(d)
        assert pub == pub.normalize()

    propiedad()


# --- Fusión con publicaciones.yaml --------------------------------------------------------------
def test_t1_publicaciones_fusiona_el_catalogo():
    p = v.cargar_publicaciones()
    assert "fred:DGS10" in p["series"] and "ecb:EXR.D.USD.EUR" in p["series"]
    assert p["alias"]["VIX"] == "fred:VIXCLS" and p["alias"]["EURUSD"] == "ecb:EXR.D.USD.EUR"
    assert p["series"]["fred:UNRATE"]["metodo"] == "vintage_exacto" and "lag" not in p["series"]["fred:UNRATE"]
    assert c.por_id("UNRATE")["regla_previa_alfred"]["lag"] == "1M1D"   # periodos anteriores al primer vintage
    assert "alias" not in p["series"]["fred:DGS10"]           # solo la parte de publicación
    sin = v.cargar_publicaciones(v.RUTA_PUBLICACIONES, catalogo=False)
    assert "fred:DGS10" not in sin["series"] and "VIX" not in sin["alias"]
    assert tools.resolver_id("SP500") == "yf:^GSPC"


def _pub_tmp(tmp_path, serie: str = "", alias: str = '{EURUSD: "ecb:EXR.D.USD.EUR"}') -> object:
    base = v.RUTA_PUBLICACIONES.read_text(encoding="utf-8")
    viejo = ('alias:   # los alias de la Fase 1 (VIX, SP500...) están en catalogo_series.yaml\n'
             '  EURUSD: "ecb:EXR.D.USD.EUR"\n')
    assert viejo in base
    texto = base.replace(viejo, f"alias: {alias}\n")
    if serie:
        texto = texto.replace("series:   # las del Tema 1 están en catalogo_series.yaml\n",
                              "series:\n" + serie)
    ruta = tmp_path / "publicaciones.yaml"
    ruta.write_text(texto, encoding="utf-8")
    return ruta


def test_t1_publicaciones_rechaza_ids_y_alias_duplicados(tmp_path):
    dup = '  "fred:DGS10":\n    fuente: x\n    metodo: calendario\n    publicador: fed_h15\n    verificado: false\n'
    with pytest.raises(ValueError, match="dos ficheros"):
        v.cargar_publicaciones(_pub_tmp(tmp_path, serie=dup), catalogo=True)
    with pytest.raises(ValueError, match="alias 'VIX'"):
        v.cargar_publicaciones(_pub_tmp(tmp_path, alias='{VIX: "fred:VIXCLS"}'), catalogo=True)
    # Sin pedir el catálogo, una ruta propia no lo fusiona (compatibilidad con el bloque 1).
    assert "fred:DGS10" in v.cargar_publicaciones(_pub_tmp(tmp_path, serie=dup))["series"]


def test_t1_lag_compuesto_valido_en_publicaciones(tmp_path):
    serie = ('  "sint:X":\n    fuente: x\n    metodo: lag_fijo\n    lag: "1M1D"\n    verificado: false\n')
    assert v.cargar_publicaciones(_pub_tmp(tmp_path, serie=serie))["series"]["sint:X"]["lag"] == "1M1D"
    with pytest.raises(ValueError, match="lag"):
        v.cargar_publicaciones(_pub_tmp(tmp_path, serie=serie.replace("1M1D", "1D1M")))


# --- Publicadores medidos (sondas del bloque 2) -------------------------------------------------
@pytest.mark.parametrize("clave, fecha, esperado_utc, decision", [
    ("fed_h15", "2026-09-18", "2026-09-21 20:15", "2026-09-21"),   # viernes → lunes (ALFRED)
    ("fed_h15", "2026-09-19", "2026-09-22 20:15", "2026-09-22"),   # DFF del sábado → martes (ALFRED)
    ("fed_h15", "2026-09-20", "2026-09-22 20:15", "2026-09-22"),   # DFF del domingo → martes (ALFRED)
    ("fed_h10", "2026-08-31", "2026-09-08 20:15", "2026-09-08"),   # lunes 7 festivo → martes (ALFRED)
    ("fed_h10", "2026-09-18", "2026-09-21 20:15", "2026-09-21"),
    ("moodys", "2026-10-08", "2026-10-09 16:00", "2026-10-09"),
    ("fred_calculadas_dia_siguiente", "2026-10-08", "2026-10-09 21:05", "2026-10-10"),  # BAA10Y: D+2
    ("fred_tesoro_calculadas", "2026-10-09", "2026-10-09 21:05", "2026-10-10"),         # T10Y2Y: D+1
    ("nyfed_sofr", "2026-10-08", "2026-10-09 12:00", "2026-10-09"),
    ("nyfed_effr", "2026-10-08", "2026-10-09 13:00", "2026-10-09"),
])
def test_t1_publicadores_medidos(clave, fecha, esperado_utc, decision):
    disp = v.disponible_publicador(clave, fecha)
    assert disp == _utc(esperado_utc)
    assert v.primera_decision(disp) == pd.Timestamp(decision)


# --- Errores de validación del catálogo ---------------------------------------------------------
_CAT_OK = """
version: 1
series:
  "fred:AAA1":
    alias: ZZ_A
    nombre: Serie de prueba diaria
    fuente: fred_alfred
    id_origen: AAA1
    frecuencia: diaria
    unidad: "% anual"
    divisa: null
    metodo: calendario
    publicador: fed_h15
    revisable: false
    rol_fase1: null
    licencia: prueba
    estado: activa
    verificado: false
  "fred:BBB1":
    alias: [ZZ_B]
    nombre: Serie de prueba mensual revisable
    fuente: fred_alfred
    id_origen: BBB1
    frecuencia: mensual
    unidad: índice
    divisa: null
    metodo: lag_fijo
    lag_clave: UNRATE
    revisable: true
    rol_fase1: core
    licencia: prueba
    estado: activa
    notas: ["valores del último vintage: posible fuga de revisiones (ALFRED en el bloque 3)"]
    verificado: false
  "deriv:A-B":
    alias: null
    nombre: Derivada de prueba
    fuente: calculo_propio
    id_origen: A - B
    frecuencia: diaria
    unidad: puntos porcentuales
    divisa: null
    metodo: derivada
    componentes: ["fred:AAA1", "fred:BBB1"]
    revisable: false
    rol_fase1: null
    licencia: prueba
    estado: activa
    verificado: false
"""


def _cat_tmp(tmp_path, texto: str):
    ruta = tmp_path / "catalogo.yaml"
    ruta.write_text(textwrap.dedent(texto), encoding="utf-8")
    return ruta


def test_t1_catalogo_minimo_valido(tmp_path):
    cat = c.cargar_catalogo(_cat_tmp(tmp_path, _CAT_OK))
    assert cat["fred:BBB1"]["lag"] == "1M1D" and cat["fred:BBB1"]["alias"] == ["ZZ_B"]
    assert c.alias(cat) == {"ZZ_A": "fred:AAA1", "ZZ_B": "fred:BBB1"}
    assert c.filtrar(cat, metodo=("derivada", "lag_fijo")) == ["deriv:A-B", "fred:BBB1"]
    assert c.parte_publicacion(cat["fred:BBB1"])["lag"] == "1M1D"
    assert "lag" not in c.parte_publicacion(cat["fred:AAA1"])
    t = c.tabla(cat)
    assert list(t["id"]) == sorted(cat) and t.loc[t["id"] == "deriv:A-B", "componentes"].item() == "fred:AAA1 | fred:BBB1"


@pytest.mark.parametrize("cambio, mensaje", [
    (("estado: activa\n    verificado: false\n  \"fred:BBB1\"", "estado: viva\n    verificado: false\n  \"fred:BBB1\""),
     "estado"),
    (('notas: ["valores del último vintage: posible fuga de revisiones (ALFRED en el bloque 3)"]', "notas: []"),
     "nota obligatoria"),
    (("lag_clave: UNRATE", "lag_clave: NO_EXISTE"), "lag_clave"),
    (('componentes: ["fred:AAA1", "fred:BBB1"]', 'componentes: ["fred:AAA1", "fred:ZZZ"]'), "componente"),
    (("alias: [ZZ_B]", "alias: [ZZ_A]"), "repetido"),
    (("alias: [ZZ_B]", "alias: [fred:AAA1]"), "coincide con un id"),
    (('"fred:AAA1":\n    alias: ZZ_A', '"fred:AAA2":\n    alias: ZZ_A'), "id_origen"),
    (('"deriv:A-B"', '"calc:A-B"'), "prefijo"),
    (('divisa: null\n    metodo: calendario', 'divisa: usd\n    metodo: calendario'), "divisa"),
    (("frecuencia: mensual", "frecuencia: quincenal"), "frecuencia"),
    (("publicador: fed_h15", "publicador: nadie"), "publicador"),
    (("revisable: false\n    rol_fase1: null\n    licencia: prueba\n    estado: activa\n    verificado: false\n  \"fred:BBB1\"",
      "revisable: true\n    rol_fase1: null\n    licencia: prueba\n    estado: activa\n    verificado: false\n  \"fred:BBB1\""),
     "revisable"),
    (("estado: activa\n    verificado: false\n  \"fred:BBB1\"", "estado: duplicada\n    verificado: false\n  \"fred:BBB1\""),
     "duplicado_de"),
    (("estado: activa\n    verificado: false\n  \"fred:BBB1\"",
      "estado: excluida_ajustada\n    verificado: false\n  \"fred:BBB1\""), "excluida_ajustada"),
    (("lag_clave: UNRATE", "lag_clave: UNRATE\n    sinsentido: 1"), "desconocidas"),
    (("    metodo: calendario\n    publicador: fed_h15", "    metodo: calendario\n    publicador: fed_h15\n    lag_clave: UNRATE"),
     "lag_clave"),
])
def test_t1_catalogo_errores(tmp_path, cambio, mensaje):
    viejo, nuevo = cambio
    assert viejo in _CAT_OK, viejo
    with pytest.raises(ValueError, match=mensaje):
        c.cargar_catalogo(_cat_tmp(tmp_path, _CAT_OK.replace(viejo, nuevo, 1)))


# --- Coherencia de filas con el catálogo y aviso de get_serie -----------------------------------
def _filas_unrate(nota: str | None, unidad: str = "% de la población activa") -> pd.DataFrame:
    """Tabla sintética con la forma de UNRATE cargada desde data/raw (valores inventados)."""
    meses = pd.date_range("2024-01-01", "2024-06-01", freq="MS")
    pub = c.fecha_publicacion_lag(pd.Series(meses), "1M1D", "mes")
    return validar_series(pd.DataFrame({
        "serie": "fred:UNRATE", "fecha_referencia": meses, "fecha_publicacion": pub.to_numpy(),
        "disponible_utc": pd.NaT, "vintage": "unico", "valor": np.linspace(3.0, 3.5, len(meses)),
        "unidad": unidad, "divisa": None, "fuente": "fred_alfred", "metodo_publicacion": "lag_fijo", "nota": nota,
    }))


def test_t1_validar_contra_catalogo():
    c.validar_contra_catalogo(_filas_unrate(c.NOTA_REVISABLE))
    with pytest.raises(ErrorContrato, match="nota obligatoria"):
        c.validar_contra_catalogo(_filas_unrate(None))
    with pytest.raises(ErrorContrato, match="unidad"):
        c.validar_contra_catalogo(_filas_unrate(c.NOTA_REVISABLE, unidad="porcentaje"))
    otra = _filas_unrate(c.NOTA_REVISABLE).assign(serie="sint:NO_CATALOGADA")
    with pytest.raises(ErrorContrato, match="no está en el catálogo"):
        c.validar_contra_catalogo(otra)
    c.validar_contra_catalogo(otra, solo_catalogadas=False)
    etf = _filas_unrate(None).assign(serie="yf:SPY", unidad="USD por participación (ajustado a posteriori)",
                                     divisa="USD", fuente="yfinance", metodo_publicacion="cierre_mercado")
    with pytest.raises(ErrorContrato, match="no entra en la base"):
        c.validar_contra_catalogo(etf)


def test_t1_get_serie_avisa_de_revisables_con_ultimo_vintage():
    base = _filas_unrate(c.NOTA_REVISABLE)
    r = tools.get_serie("UNRATE", "2024-12-31", base=base)          # alias de la Fase 1 vía catálogo
    assert r.serie == "fred:UNRATE"
    assert any("posible fuga de revisiones" in a for a in r.avisos)
    sin = tools.get_serie("UNRATE", "2024-12-31", base=_filas_unrate(None))
    assert not any("posible fuga de revisiones" in a for a in sin.avisos)
    # Truncado: el aviso solo cuenta los periodos visibles.
    r = tools.get_serie("UNRATE", "2024-03-01", base=base)
    assert any("en 1 periodos" in a for a in r.avisos)


# --- Sondas (sin red) ---------------------------------------------------------------------------
def test_t1_sonda_primeras_publicaciones_sin_valores(monkeypatch):
    """ALFRED sintético: dos vintages de D y una observación truncada en rt_desde."""
    respuesta = {"observations": [
        {"date": "2026-09-17", "realtime_start": "2026-09-10", "realtime_end": "9999-12-31", "value": "1.0"},
        {"date": "2026-09-18", "realtime_start": "2026-09-21", "realtime_end": "2026-09-22", "value": "1.0"},
        {"date": "2026-09-18", "realtime_start": "2026-09-23", "realtime_end": "9999-12-31", "value": "1.1"},
        {"date": "2026-09-19", "realtime_start": "2026-09-22", "realtime_end": "9999-12-31", "value": "."},
    ]}
    monkeypatch.setattr(sonda_horas, "_fred", lambda *a, **k: respuesta)
    filas = sonda_horas.primeras_publicaciones("SINT", "2026-09-15", "2026-09-19", "2026-09-10")
    assert len(filas) == 1
    f = filas[0]
    assert (f["fecha_referencia"], f["fecha_publicacion_fred"], f["dias_habiles_us"]) == ("2026-09-18", "2026-09-21", 1)
    assert set(f) == set(sonda_horas.COLUMNAS)


def test_t1_sonda_csv_guardado_solo_fechas():
    ruta = sonda_horas.ruta_csv()
    if not ruta.exists():
        pytest.skip("sondas del bloque 2 sin ejecutar")
    t = pd.read_csv(ruta)
    assert list(t.columns) == list(sonda_horas.COLUMNAS)
    assert not any("value" in col or "valor" in col for col in t.columns)
    h15 = t[(t["sonda"] == "h15_realtime_start") & (t["serie"] == "DGS10")]
    assert len(h15) and (h15["dias_habiles_us"] == 1).all()


@pytest.mark.live
def test_t1_sonda_live_h15_dia_siguiente():
    filas = sonda_horas.primeras_publicaciones("DGS10", "2026-09-21", "2026-09-25", "2026-09-14")
    assert filas and all(f["dias_habiles_us"] == 1 for f in filas)


def test_t1_shiller_reglas_de_fichero_medidas():
    """Shiller (bloque 2): la regla 1D de la Fase 1 adelantaba agosto-2026 un día a la edición; P, D, E y CPI entran."""
    cat = c.cargar_catalogo()
    lags = c.cargar_lags()
    assert lags["SHILLER_FICHERO"]["lag"] == "1M1D" and lags["SHILLER_DE"]["lag"] == "3M1D"
    assert lags["SHILLER_SP500_CAPE"]["lag"] == "1D"                     # la copia congelada no cambia
    for sid, lag in (("shiller:CAPE", "1M1D"), ("shiller:LONGRATE", "1M1D"), ("shiller:P", "1M1D"),
                     ("shiller:D", "3M1D"), ("shiller:E", "3M1D"), ("shiller:CPI", "1M1D")):
        e = cat[sid]
        assert e["metodo"] == "fichero" and e["lag"] == lag, sid
        assert e["revisable"] == (sid != "shiller:LONGRATE"), sid      # el tipo largo es la media de GS10
    # Agosto-2026 (edición del 2026-09-02): visible el 2026-10-01, nunca antes de la edición.
    assert c.fecha_publicacion_lag(["2026-08-01"], "1M1D", "mes").iloc[0] == pd.Timestamp("2026-10-01")
    # Ninguna nota dice ya que la OFR exige un User-Agent de navegador (medido: HTTP 200 con el identificable).
    assert not any("User-Agent de navegador (catálogo" in n for e in cat.values() for n in e["notas"])


# --- Bloque 3: vintages de ALFRED, fuente viva y derivadas -------------------------------------
LISTA_ALFRED = ROOT / "results" / "ingesta" / "mercados_series" / "tablas" / "alfred_lista_propuesta.csv"
SIN_REGLA_PREVIA = {"fred:WEI", "fred:GDPNOW", "fred:FEDTARMD", "fred:UNRATEMD", "fred:PCECTPIMD"}
FIN_POR_FRECUENCIA = {"diaria": "fecha", "semanal": "fecha", "mensual": "mes", "trimestral": "trimestre",
                      "anual": "anio"}


@pytest.fixture(scope="module")
def lista_alfred():
    if not LISTA_ALFRED.exists():
        pytest.skip("falta la lista de ALFRED del bloque 3")
    return pd.read_csv(LISTA_ALFRED, dtype=str, keep_default_na=False)


PRIMER_VINTAGE_MEDIDO = {"fred:USREC": "2011-05-02"}


def test_t1_catalogo_lista_alfred_completa(lista_alfred):
    cat = c.cargar_catalogo()
    assert len(lista_alfred) == 161 and (lista_alfred["en_catalogo"] == "False").sum() == 128
    # Las vintage_exacto de ALFRED (las de los bloques 4-6 tienen su propio prefijo: ecb:, oecd:, imf:...).
    assert {s for s in c.filtrar(cat, metodo="vintage_exacto") if s.startswith("fred:")}         == {f"fred:{i}" for i in lista_alfred["id"]}
    pubs = v.cargar_publicaciones(catalogo=False)["publicadores"]
    for f in lista_alfred.itertuples(index=False):
        sid = f"fred:{f.id}"
        e = cat[sid]
        assert (e["fuente"], e["revisable"], e["fuente_viva"]) == ("fred_alfred", True, "fred"), sid
        assert c.origenes(e) == ("alfred",), sid
        assert pubs[e["publicador"]]["dia"] == "mismo", sid                # la vintage es el día de publicación
        assert (e["frecuencia"], e["grupo"], e["release"]) == (f.frecuencia, f.grupo, f.release), sid
        assert e["agentes"] == [a.strip() for a in f.agentes.split(",")], sid
        assert not any(c.NOTA_REVISABLE in n for n in e["notas"]), sid      # ya no es el último vintage
        assert e.get("lag_clave") is None and e["lag"] is None, sid
        previa = e["regla_previa_alfred"]
        if sid in SIN_REGLA_PREVIA:
            assert previa is None, sid
        else:
            # USREC: el primer vintage medido en las observaciones (2011-05-02) es anterior al de series/vintagedates.
            esperado = PRIMER_VINTAGE_MEDIDO.get(sid, f.primer_vintage)
            assert previa["primer_vintage"] == esperado and previa["metodo"] == "lag_fijo", sid
            assert previa["fin_periodo"] == FIN_POR_FRECUENCIA[e["frecuencia"]], sid
        if f.en_catalogo == "False":
            assert e["alias"] == [] and e["fichero_fase1"] == [] and e["estado"] == "activa", sid
            assert e["unidad"] and e["licencia"].endswith("vía FRED: solo local, sin redistribuir"), sid
        assert ("se reestima entera" in f.motivo) <= e["reestimada"], sid


def test_t1_catalogo_avisos_de_la_lista_alfred():
    cat = c.cargar_catalogo()
    assert set(c.filtrar(cat, cobertura_variable=True)) == {"fred:PCEC96", "fred:PNFIC1", "fred:PRFIC1"}
    assert cat["fred:A191RL1Q225SBEA"]["coherente_con"] == ["fred:GDPC1"]
    assert cat["fred:CFNAIMA3"]["coherente_con"] == ["fred:CFNAI"]
    assert cat["fred:SAHMREALTIME"]["coherente_con"] == cat["fred:SAHMCURRENT"]["coherente_con"] == ["fred:UNRATE"]
    for sid in ("fred:NFCI", "fred:ANFCI", "fred:STLFSI4", "fred:CFSI", "fred:KCFSI", "fred:RECPROUSM156N",
                "fred:USALOLITOAASTSAM", "fred:BBKMLEIX", "fred:USPHCI", "fred:FRBKCLMCILA"):
        assert cat[sid]["reestimada"], sid
    stl = cat["fred:STLFSI4"]
    assert stl["regla_previa_alfred"]["primer_vintage"] == stl["publicado_desde"] == "2022-11-10"
    assert any("STLFSI2" in n for n in stl["notas"])
    assert {s: e["proyecciones"] for s, e in cat.items() if e.get("proyecciones")} == {
        "fred:FEDTARMD": "conservar", "fred:UNRATEMD": "conservar", "fred:PCECTPIMD": "conservar",
        "fred:NROU": "recortar", "fred:GDPPOT": "recortar"}
    assert cat["fred:GDPC1"]["unidad_por_vintage"] and cat["fred:INDPRO"]["unidad_por_vintage"]
    assert not cat["fred:CPIAUCSL"]["unidad_por_vintage"]                  # base 1982-84 fija
    # Las que eran revisable: false y ALFRED sí revisa.
    assert cat["fred:UMCSENT"]["revisable"] and cat["fred:JHDUSRGDPBR"]["revisable"]


@pytest.mark.parametrize("clave, fecha, decision", [
    ("bls", "2026-10-02", "2026-10-02"),                 # Employment Situation 08:30: el mismo día
    ("census_10", "2026-09-24", "2026-09-24"),           # vivienda nueva 10:00
    ("jolts", "2026-09-29", "2026-09-29"),
    ("fed_h8", "2026-10-09", "2026-10-10"),              # viernes 16:15 + 30 min: el sábado (conservador)
    ("fed_h41", "2026-10-08", "2026-10-09"),             # jueves 16:30 + 5 min: el viernes
    ("fed_z1", "2026-09-11", "2026-09-11"),
    ("fed_sep", "2026-09-16", "2026-09-16"),
    ("oecd_mei", "2026-09-15", "2026-09-16"),            # hora desconocida: el día siguiente
    ("fred_calculo", "2026-10-02", "2026-10-03"),
])
def test_t1_publicadores_del_bloque3(clave, fecha, decision):
    assert v.primera_decision(v.disponible_publicador(clave, fecha)) == pd.Timestamp(decision)


def test_t1_publicadores_del_bloque3_conservadores_sin_verificar():
    pubs = v.cargar_publicaciones(catalogo=False)["publicadores"]
    usados = {e["publicador"] for e in c.cargar_catalogo().values()
              if e["metodo"] == "vintage_exacto" and e["fuente"] == "fred_alfred"}
    assert usados <= set(pubs)
    for clave in usados:
        p = pubs[clave]
        assert p["huso"] == "America/New_York" and p["fuente"].startswith("https://"), clave
        if p["hora"].strftime("%H:%M") == "23:59":       # hora desconocida: siempre el día siguiente
            assert p["verificado"] is False and any("conservador" in n for n in p["notas"]), clave


def test_t1_derivadas_de_los_diferenciales_de_fred():
    cat = c.cargar_catalogo()
    esperadas = {"BAA10Y": ("deriv:DBAA-DGS10", ["fred:DBAA", "fred:DGS10"]),
                 "T10YFF": ("deriv:DGS10-DFF", ["fred:DGS10", "fred:DFF"]),
                 "T5YFF": ("deriv:DGS5-DFF", ["fred:DGS5", "fred:DFF"]),
                 "CPFF": ("deriv:DCPF3M-DFF", ["fred:DCPF3M", "fred:DFF"])}
    for fred_id, (deriv, comp) in esperadas.items():
        d, orig = cat[deriv], cat[f"fred:{fred_id}"]
        assert (d["metodo"], d["componentes"], d["fuente"], d["estado"]) == ("derivada", comp, "calculo_propio", "activa")
        assert d["id_origen"] == f"{comp[0]} - {comp[1]}" and d["verificado"] is True
        assert (orig["estado"], orig["duplicado_de"], orig["alias"]) == ("duplicada", deriv, [])
        assert orig["publicador"] == "fred_calculadas_dia_siguiente"     # la de FRED sigue fechada en D+2
        assert c.origenes(d) == ("derivadas",)
        if fred_id != "CPFF":
            assert c.resolver(fred_id) == deriv                          # el alias de la Fase 1 pasa a la derivada
    assert cat["fred:TEDRATE"]["sucesor"] == cat["fred:LIOR3M"]["sucesor"] == "deriv:DCPF3M-DFF"


def test_t1_fuente_viva_y_origenes():
    cat = c.cargar_catalogo()
    for sid, e in cat.items():
        viva = sid.startswith("fred:") and e["fuente"] == "fred_alfred"
        assert e["fuente_viva"] == ("fred" if viva else None), sid
    assert c.origenes(cat["fred:DGS10"]) == ("fred", "fase1")              # calendario: FRED vivo y data/raw
    assert c.origenes(cat["fred:BAMLH0A0HYM2"]) == ("fred",)               # capada, sin fichero de la Fase 1
    assert c.origenes(cat["fred:INDPRO"]) == ("alfred",)
    assert c.origenes(cat["yf:^GSPC"]) == ("fase1",)
    assert c.origenes(cat["ff:F3D.SMB"]) == ("academicos", "fase1")
    assert c.origenes(cat["shiller:LONGRATE"]) == ("academicos",)          # fase1_utilizable: false
    assert c.origenes(cat["deriv:DBAA-DAAA"]) == ("derivadas", "fase1")
    for sid, e in cat.items():
        o = c.origenes(e)
        assert list(o) == sorted(o, key=c.PRIORIDAD_ORIGENES.index), sid
    assert c.resumen()["fuente_viva"]["fred"] == len(c.filtrar(cat, fuente="fred_alfred"))


def test_t1_regla_previa_y_vista_del_bloque2():
    cat = c.cargar_catalogo()
    e = cat["fred:UNRATE"]
    previa = c.entrada_previa(e)
    assert (previa["metodo"], previa["lag"], previa["fin_periodo"]) == ("lag_fijo", "1M1D", "mes")
    assert c.fecha_publicacion_regla(["2024-01-01"], previa).iloc[0] == pd.Timestamp("2024-03-01")
    assert e["metodo"] == "vintage_exacto"                                # la copia no toca el catálogo
    assert c.nota_previa_alfred(e) == f"{c.NOTA_PREVIA_ALFRED} (1960-03-15)"
    assert c.metodos_admitidos(e) == {"vintage_exacto", "lag_fijo"}
    with pytest.raises(ValueError, match="regla_previa_alfred"):
        c.entrada_previa(cat["fred:WEI"])
    vista = c.vista_regla_previa(cat)
    assert vista["fred:UNRATE"]["metodo"] == "lag_fijo" and vista["fred:WEI"]["metodo"] == "vintage_exacto"
    assert vista["fred:DGS10"] == cat["fred:DGS10"]
    solo = c.vista_regla_previa(cat, ["fred:CPIAUCSL"])
    assert solo["fred:CPIAUCSL"]["metodo"] == "lag_fijo" and solo["fred:UNRATE"]["metodo"] == "vintage_exacto"
    t = c.tabla(cat).set_index("id")
    assert (t.loc["fred:UNRATE", "lag_previo_alfred"], t.loc["fred:UNRATE", "primer_vintage_alfred"]) == (
        "1M1D", "1960-03-15")
    assert t.loc["fred:UNRATE", "fuente_viva"] == "fred" and t.loc["fred:UNRATE", "agentes"] == "macro | risk"


def test_t1_validar_contra_catalogo_filas_previas_a_alfred():
    """Filas de UNRATE anteriores al primer vintage (lag_fijo) y de ALFRED (vintage_exacto): ambas valen."""
    previas = _filas_unrate(None).assign(vintage=lambda d: d["fecha_publicacion"].dt.strftime("%Y-%m-%d"))
    alfred = previas.assign(metodo_publicacion="vintage_exacto")
    c.validar_contra_catalogo(previas)
    c.validar_contra_catalogo(alfred)
    with pytest.raises(ErrorContrato, match="metodo_publicacion"):
        c.validar_contra_catalogo(previas.assign(metodo_publicacion="calendario"))
    wei = previas.assign(serie="fred:WEI", unidad=c.por_id("fred:WEI")["unidad"])
    with pytest.raises(ErrorContrato, match="metodo_publicacion"):         # WEI no tiene regla previa
        c.validar_contra_catalogo(wei)


_CAT_B3 = _CAT_OK + """
  "fred:CCC1":
    alias: null
    nombre: Serie de prueba de ALFRED
    fuente: fred_alfred
    id_origen: CCC1
    frecuencia: mensual
    unidad: índice
    divisa: null
    metodo: vintage_exacto
    publicador: bls
    regla_previa_alfred: {lag_clave: UNRATE, primer_vintage: "2010-01-08"}
    proyecciones: recortar
    revisable: true
    rol_fase1: null
    licencia: prueba
    estado: activa
    grupo: empleo
    agentes: [macro, risk]
    release: Employment Situation
    reestimada: true
    coherente_con: ["fred:BBB1"]
    verificado: false
"""


def test_t1_catalogo_bloque3_minimo_valido(tmp_path):
    cat = c.cargar_catalogo(_cat_tmp(tmp_path, _CAT_B3))
    e = cat["fred:CCC1"]
    assert e["regla_previa_alfred"] == {"metodo": "lag_fijo", "lag_clave": "UNRATE", "lag": "1M1D",
                                        "fin_periodo": "mes", "lag_habiles": None, "lag_n_habiles": None,
                                        "primer_vintage": "2010-01-08"}
    assert (e["fuente_viva"], e["grupo"], e["agentes"], e["reestimada"]) == ("fred", "empleo", ["macro", "risk"], True)
    assert cat["deriv:A-B"]["fuente_viva"] is None and cat["fred:AAA1"]["fuente_viva"] == "fred"
    assert e["cobertura_variable"] is False and e["unidad_por_vintage"] is False


_PREVIA_CALENDARIO = '    regla_previa_alfred: {lag_clave: UNRATE, primer_vintage: "2010-01-08"}'


@pytest.mark.parametrize("cambio, mensaje", [
    (("lag_clave: UNRATE, primer_vintage", "lag_clave: NO_EXISTE, primer_vintage"), "regla_previa_alfred"),
    (('primer_vintage: "2010-01-08"', 'primer_vintage: "enero"'), "AAAA-MM-DD"),
    (('primer_vintage: "2010-01-08"}', 'primer_vintage: "2010-01-08", lag: 1D}'), "desconocidas"),
    (("proyecciones: recortar", "proyecciones: quizas"), "proyecciones"),
    (("grupo: empleo", "grupo: deportes"), "grupo"),
    (("agentes: [macro, risk]", "agentes: [macro, macro]"), "agentes"),
    (("agentes: [macro, risk]", "agentes: [trader]"), "agentes"),
    (("reestimada: true", "reestimada: si"), "reestimada"),
    (('coherente_con: ["fred:BBB1"]', 'coherente_con: ["fred:NADA"]'), "coherente_con"),
    (("    publicador: bls\n    regla_previa", "    regla_previa"), "publicador"),
    (("grupo: empleo", "grupo: empleo\n    fuente_viva: null"), "fuente_viva"),
    (("grupo: empleo", "grupo: empleo\n    fuente_viva: alfred"), "fuente_viva"),
    (("    metodo: derivada\n", "    metodo: derivada\n    fuente_viva: fred\n"), "fuente_viva"),
    (("    metodo: calendario\n    publicador: fed_h15", "    metodo: calendario\n    publicador: fed_h15\n"
      + _PREVIA_CALENDARIO), "solo vale con metodo vintage_exacto"),
])
def test_t1_catalogo_bloque3_errores(tmp_path, cambio, mensaje):
    viejo, nuevo = cambio
    assert viejo in _CAT_B3, viejo
    with pytest.raises(ValueError, match=mensaje):
        c.cargar_catalogo(_cat_tmp(tmp_path, _CAT_B3.replace(viejo, nuevo, 1)))
