"""Tests del cargador de ``data/raw`` de la Fase 1 (``fuentes.fase1``, bloque 2 del Tema 1).

Sin red. Un ``data/raw`` SINTÉTICO mínimo en ``tmp_path`` imita los formatos reales (parquet de una
columna con índice de fechas, ``Date`` como nombre del índice en yfinance, ICSA entero y en sábado,
Ken French multicolumna con la columna alias duplicada) con valores aleatorios: ningún dato real.
El test sobre el ``data/raw`` real lleva ``@pytest.mark.datos`` y se salta si falta.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from ingesta.comun.rutas import ROOT
from ingesta.mercados_series import catalogo as c
from ingesta.mercados_series.contrato import ErrorContrato, validar_series
from ingesta.mercados_series.fuentes import fase1

RAW = ROOT / "data" / "raw"
NY = "America/New_York"
DESCARGA = pd.Timestamp("2024-03-20 15:37", tz=NY)       # barra del 2024-03-20 intradía
IDS = ("yf:^GSPC", "yf:SPY", "fred:DGS10", "fred:ICSA", "fred:UNRATE", "ff:F3D.Mkt-RF", "ff:F3D.SMB",
       "ff:F3D.HML", "ff:F3D.RF", "deriv:DED3-DTB3", "fred:DED3", "fred:DTB3", "deriv:DBAA-DAAA", "fred:DBAA",
       "fred:DAAA", "deriv:VOLREAL21_GSPC")


def _ny(texto: str) -> pd.Timestamp:
    return pd.Timestamp(texto, tz=NY).tz_convert("UTC")


def _guardar(raiz, rel: str, df: pd.DataFrame, mtime: pd.Timestamp | None = None) -> None:
    ruta = raiz / rel
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ruta)
    if mtime is not None:
        os.utime(ruta, (mtime.timestamp(), mtime.timestamp()))


def _col(nombre: str, fechas, rng, entero: bool = False, indice: str | None = None) -> pd.DataFrame:
    v = rng.normal(100, 5, len(fechas))
    v = np.round(v * 1000).astype("int64") if entero else np.round(v, 2)
    idx = pd.DatetimeIndex(fechas, name=indice)
    return pd.DataFrame({nombre: v}, index=idx)


@pytest.fixture
def raw(tmp_path):
    """``data/raw`` sintético con los formatos de la Fase 1."""
    rng = np.random.default_rng(7)
    raiz = tmp_path / "raw"
    habiles = pd.bdate_range("2024-01-02", "2024-03-20")
    # yfinance: índice (con copia de validación idéntica), ETF ajustada y vol. realizada (derivada).
    spx = _col("SP500", habiles, rng, indice="Date")
    _guardar(raiz, "yfinance/SP500.parquet", spx, DESCARGA)
    _guardar(raiz, "yfinance/SP500_DAILY_DRAWDOWN_SRC.parquet",
             spx.rename(columns={"SP500": "SP500_DAILY_DRAWDOWN_SRC"}), DESCARGA)
    _guardar(raiz, "yfinance/SP500_CAPWEIGHT_ETF.parquet", _col("SP500_CAPWEIGHT_ETF", habiles, rng, indice="Date"),
             DESCARGA)
    # La vol. realizada se rederivó después (como en data/raw): su última fecha no tiene ^GSPC.
    _guardar(raiz, "yfinance/REALIZED_VOL_SP500.parquet", _col("REALIZED_VOL_SP500", habiles[21:], rng),
             DESCARGA + pd.Timedelta(days=5))
    # FRED diaria H.15 con una interrupción larga (1-abr a 15-may) y sin el 4 de julio.
    dgs = pd.bdate_range("2024-01-02", "2024-07-10")
    dgs = dgs[~((dgs >= "2024-04-01") & (dgs <= "2024-05-15")) & (dgs != "2024-07-04")]
    _guardar(raiz, "fred/DGS10.parquet", _col("DGS10", dgs, rng))
    _guardar(raiz, "fred/DTB3.parquet", _col("DTB3", habiles, rng))
    _guardar(raiz, "fred/EURODOLLAR_3M.parquet", _col("EURODOLLAR_3M", habiles, rng))
    _guardar(raiz, "fred/EURODOLLAR_TBILL_SPREAD.parquet", _col("EURODOLLAR_TBILL_SPREAD", habiles, rng))
    _guardar(raiz, "fred/MOODYS_BAA_AAA_SPREAD.parquet", _col("MOODYS_BAA_AAA_SPREAD", habiles, rng))
    # Semanal (sábados, entero) y mensual (día 1, sin febrero: hueco permanente).
    _guardar(raiz, "fred/ICSA.parquet", _col("ICSA", pd.date_range("2024-01-06", "2024-03-16", freq="W-SAT"), rng,
                                             entero=True))
    _guardar(raiz, "fred/UNRATE.parquet", _col("UNRATE", ["2023-12-01", "2024-01-01", "2024-03-01"], rng))
    # Ken French diario: multicolumna con la columna alias (= Mkt-RF).
    ff = pd.DataFrame({c: np.round(rng.normal(0, 1, len(habiles)), 2) for c in ("Mkt-RF", "SMB", "HML", "RF")},
                      index=habiles)
    ff.insert(0, "FF_FACTORS_3_DAILY", ff["Mkt-RF"])
    _guardar(raiz, "academico/FF_FACTORS_3_DAILY.parquet", ff)
    return raiz


# Bloque 3: las revisables de ALFRED (vintage_exacto) en su forma del bloque 2 (regla previa, lag_fijo):
# estos tests ejercitan las reglas de publicación con entradas reales del catálogo.
@pytest.fixture(scope="module")
def cat():
    real = c.vista_regla_previa()
    return {k: real[k] for k in IDS}


@pytest.fixture
def tabla(raw, cat):
    return fase1.tabla_fase1(raw, cat)


def _serie(t: pd.DataFrame, sid: str) -> pd.DataFrame:
    return t[t["serie"] == sid].set_index("fecha_referencia")


# --- Reglas del catálogo de las que dependen los resultados esperados ----------------------------
def test_t1_fase1_reglas_de_partida(cat):
    assert cat["yf:^GSPC"]["mercado"] == "XNYS"
    assert cat["fred:DGS10"]["publicador"] == cat["fred:DED3"]["publicador"] == "fed_h15"
    assert cat["fred:DBAA"]["publicador"] == "moodys" and not cat["fred:DBAA"]["fichero_fase1"]
    assert (cat["fred:ICSA"]["lag"], cat["fred:ICSA"]["fin_periodo"]) == ("5D", "fecha")
    assert (cat["fred:UNRATE"]["lag"], cat["fred:UNRATE"]["fin_periodo"]) == ("1M1D", "mes")
    assert (cat["ff:F3D.SMB"]["lag"], cat["ff:F3D.SMB"]["fin_periodo"]) == ("45D", "mes")
    assert cat["yf:SPY"]["estado"] == "excluida_ajustada"


# --- Tabla sintética ----------------------------------------------------------------------------
def test_t1_fase1_valida_y_series_esperadas(tabla):
    assert validar_series(tabla).equals(tabla)
    c.validar_contra_catalogo(tabla, None)       # todas catalogadas en el catálogo real
    esperadas = {"yf:^GSPC", "fred:DGS10", "fred:ICSA", "fred:UNRATE", "ff:F3D.Mkt-RF", "ff:F3D.SMB",
                 "ff:F3D.HML", "ff:F3D.RF", "deriv:DED3-DTB3", "fred:DED3", "fred:DTB3", "deriv:DBAA-DAAA",
                 "deriv:VOLREAL21_GSPC"}
    assert set(tabla["serie"]) == esperadas
    assert (tabla["vintage"] == "unico").all()
    assert not tabla.duplicated(["serie", "fecha_referencia", "vintage"]).any()


def test_t1_fase1_excluye_etf_ajustada_y_alias(tabla, raw, cat):
    assert "yf:SPY" not in set(tabla["serie"])
    assert not tabla["serie"].isin(["SP500_CAPWEIGHT_ETF", "FF_FACTORS_3_DAILY", "SP500"]).any()
    res = fase1.resumen_fase1(raw, cat).set_index("id")
    assert not res.loc["yf:SPY", "incluida"] and res.loc["yf:SPY", "motivo"] == "estado excluida_ajustada"
    assert res.loc["ff:F3D.Mkt-RF", "copias_distintas"] == 0
    assert res.loc["yf:^GSPC", "copias_distintas"] == 0


def test_t1_fase1_cierre_xnys_y_barra_intradia(tabla):
    g = _serie(tabla, "yf:^GSPC")
    assert g.index.max() == pd.Timestamp("2024-03-19")          # el 2024-03-20 (intradía) fuera
    assert (g["fecha_publicacion"] == g.index).all()            # 16:00 + 30 min = 16:30 NY → mismo día
    assert g.loc["2024-03-15", "disponible_utc"] == _ny("2024-03-15 16:30")
    assert (g["metodo_publicacion"] == "cierre_mercado").all() and g["nota"].isna().all()


def test_t1_fase1_resumen_mide_descarte_y_lags(raw, cat):
    res = fase1.resumen_fase1(raw, cat).set_index("id")
    assert res.loc["yf:^GSPC", "barras_descartadas"] == 1
    assert res.loc["yf:^GSPC", "dia_descarga"] == "2024-03-20"
    assert res.loc["yf:^GSPC", "desfase_descarga_dias"] == 1
    assert res.loc["yf:^GSPC", "lag_mediano_dias"] == 0 and bool(res.loc["yf:^GSPC", "con_hora"])
    assert res.loc["fred:ICSA", "lag_min_dias"] == res.loc["fred:ICSA", "lag_max_dias"] == 5
    assert not bool(res.loc["fred:ICSA", "con_hora"])
    assert res.loc["fred:UNRATE", "n_huecos"] == 1 and res.loc["fred:UNRATE", "n_obs"] == 3
    assert res.loc["deriv:DBAA-DAAA", "componentes_por_regla"] == "fred:DBAA | fred:DAAA"
    assert res.loc["deriv:VOLREAL21_GSPC", "filas_sin_componentes"] == 1      # vol del 03-20 sin ^GSPC
    for col in ("id", "n_obs", "inicio", "fin", "metodo", "lag_mediano_dias"):
        assert col in res.reset_index().columns


def test_t1_fase1_h15_dia_habil_siguiente(tabla):
    g = _serie(tabla, "fred:DGS10")
    assert g.loc["2024-03-15", "fecha_publicacion"] == pd.Timestamp("2024-03-18")   # viernes → lunes
    assert g.loc["2024-03-15", "disponible_utc"] == _ny("2024-03-18 16:15")
    assert g.loc["2024-07-03", "fecha_publicacion"] == pd.Timestamp("2024-07-05")   # 4 de julio festivo
    assert (g["fecha_publicacion"] > g.index).all()


def test_t1_fase1_hueco_diario_largo(tabla):
    g = _serie(tabla, "fred:DGS10")
    nulos = g[g["valor"].isna()]
    assert list(nulos.index) == [pd.Timestamp("2024-04-01")]       # primer laborable tras el 03-29
    assert nulos["nota"].iloc[0].startswith(fase1.NOTA_HUECO_DIARIO)
    assert "2024-03-30 a 2024-05-15" in nulos["nota"].iloc[0]
    assert g.loc["2024-04-01", "fecha_publicacion"] == pd.Timestamp("2024-04-02")
    # Un día suelto sin dato (festivo) no es un hueco.
    assert pd.Timestamp("2024-07-04") not in g.index


def test_t1_fase1_semanal_y_mensual_con_lag(tabla):
    icsa = _serie(tabla, "fred:ICSA")
    assert (icsa.index.weekday == 5).all()
    assert icsa.loc["2024-03-16", "fecha_publicacion"] == pd.Timestamp("2024-03-21")   # sábado + 5D = jueves
    assert icsa["disponible_utc"].isna().all() and (icsa["metodo_publicacion"] == "lag_fijo").all()
    un = _serie(tabla, "fred:UNRATE")
    assert un.loc["2024-01-01", "fecha_publicacion"] == pd.Timestamp("2024-03-01")   # fin de ene + 1M1D
    assert un.loc["2023-12-01", "fecha_publicacion"] == pd.Timestamp("2024-02-01")
    assert np.isnan(un.loc["2024-02-01", "valor"])
    assert un.loc["2024-02-01", "nota"].startswith(fase1.NOTA_HUECO)


def test_t1_fase1_nota_obligatoria_en_revisables(tabla):
    for sid in ("fred:ICSA", "fred:UNRATE", "ff:F3D.SMB"):
        g = _serie(tabla, sid)
        assert g["nota"].str.contains(c.nota_revisable(c.por_id(sid)), regex=False).all(), sid
    assert _serie(tabla, "fred:DGS10").loc["2024-03-15", "nota"] is None


def test_t1_fase1_multicolumna_ken_french(tabla, raw):
    ff = pd.read_parquet(raw / "academico/FF_FACTORS_3_DAILY.parquet")
    for col in ("Mkt-RF", "SMB", "HML", "RF"):
        g = _serie(tabla, f"ff:F3D.{col}")
        assert np.allclose(g["valor"].to_numpy(), ff[col].to_numpy())
        assert (g["metodo_publicacion"] == "fichero").all() and (g["fuente"] == "ken_french").all()
    g = _serie(tabla, "ff:F3D.SMB")
    assert g.loc["2024-01-15", "fecha_publicacion"] == pd.Timestamp("2024-03-16")     # 31-ene + 45D
    assert g.loc["2024-03-01", "fecha_publicacion"] == pd.Timestamp("2024-05-15")


def test_t1_fase1_derivadas(tabla):
    ed = _serie(tabla, "deriv:DED3-DTB3")
    assert ed.loc["2024-03-15", "fecha_publicacion"] == pd.Timestamp("2024-03-18")
    assert ed.loc["2024-03-15", "disponible_utc"] == _ny("2024-03-18 16:15")
    mo = _serie(tabla, "deriv:DBAA-DAAA")                       # componentes de Moody's fechados por regla
    assert mo.loc["2024-03-15", "fecha_publicacion"] == pd.Timestamp("2024-03-18")
    assert mo.loc["2024-03-15", "disponible_utc"] == _ny("2024-03-18 12:00")
    vol = _serie(tabla, "deriv:VOLREAL21_GSPC")
    spx = _serie(tabla, "yf:^GSPC")
    assert vol.index.max() == pd.Timestamp("2024-03-19")
    assert (vol["fecha_publicacion"] == spx.loc[vol.index, "fecha_publicacion"]).all()
    assert (tabla.loc[tabla["serie"].str.startswith("deriv:"), "metodo_publicacion"] == "derivada").all()


def test_t1_fase1_copia_distinta_se_mide(raw, cat):
    ruta = raw / "yfinance/SP500_DAILY_DRAWDOWN_SRC.parquet"
    df = pd.read_parquet(ruta)
    df.iloc[3, 0] += 1.0
    df.to_parquet(ruta)
    os.utime(ruta, (DESCARGA.timestamp(), DESCARGA.timestamp()))
    res = fase1.resumen_fase1(raw, cat).set_index("id")
    assert res.loc["yf:^GSPC", "copias_distintas"] == 1


def test_t1_fase1_columna_inexistente(raw, cat):
    malo = {k: dict(v) for k, v in cat.items()}
    malo["fred:DGS10"]["fichero_fase1"] = ["fred/DGS10.parquet:NO_ESTA"]
    with pytest.raises(KeyError, match="NO_ESTA"):
        fase1.tabla_fase1(raw, malo)


def test_t1_fase1_validacion_contra_catalogo(raw, cat):
    malo = {k: dict(v) for k, v in cat.items()}
    malo["fred:DGS10"]["unidad"] = "otra"
    t, _, _ = fase1._cargar(raw, cat)
    with pytest.raises(ErrorContrato, match="unidad"):
        c.validar_contra_catalogo(t, malo)


def test_t1_fase1_catalogo_vacio_da_tabla_vacia(raw):
    t = fase1.tabla_fase1(raw, {})
    assert t.empty and list(t.columns) == list(validar_series(t).columns)


# --- Normalización de fechas --------------------------------------------------------------------
def test_t1_fase1_normalizar_fechas():
    fin_mes = pd.DatetimeIndex(["2024-01-31", "2024-02-29"])
    assert list(fase1.normalizar_fechas(fin_mes, "mensual")) == [pd.Timestamp("2024-01-01"),
                                                               pd.Timestamp("2024-02-01")]
    trim = pd.DatetimeIndex(["2024-03-31", "2024-05-15"])
    assert list(fase1.normalizar_fechas(trim, "trimestral")) == [pd.Timestamp("2024-01-01"),
                                                               pd.Timestamp("2024-04-01")]
    jueves = pd.DatetimeIndex(["2024-03-14", "2024-03-21"])
    assert list(fase1.normalizar_fechas(jueves, "semanal", 5).weekday) == [5, 5]       # → sábado
    assert fase1.normalizar_fechas(pd.DatetimeIndex(["2024-03-16"]), "semanal", 5)[0] == pd.Timestamp("2024-03-16")
    with pytest.raises(ValueError, match="DIA_SEMANAL"):
        fase1.normalizar_fechas(pd.DatetimeIndex(["2024-03-14", "2024-03-22"]), "semanal")
    diaria = pd.DatetimeIndex(["2024-03-14 00:00"])
    assert fase1.normalizar_fechas(diaria, "diaria").equals(diaria)
    assert fase1.DIA_SEMANAL["fred:ICSA"] == 5 and fase1.DIA_SEMANAL["fred:NFCI"] == 4


def test_t1_fase1_mensual_fuera_de_rejilla(raw, cat):
    # Dos fechas del mismo mes tras normalizar: error, no se pisan.
    _guardar(raw, "fred/UNRATE.parquet", pd.DataFrame({"UNRATE": [4.0, 4.1]},
                                                      index=pd.DatetimeIndex(["2024-01-01", "2024-01-31"])))
    with pytest.raises(ValueError, match="fechas repetidas"):
        fase1.tabla_fase1(raw, cat)


# --- Ficha --------------------------------------------------------------------------------------
def test_t1_fase1_ficha(raw, cat, tmp_path):
    from ingesta.comun.fichas import guardar_ficha

    tabla = fase1.tabla_fase1(raw, cat)
    res = fase1.resumen_fase1(raw, cat)
    ficha = fase1.ficha_fase1(res, len(tabla), raw)
    assert ficha.fuente == "fase1_raw" and ficha.tema == "mercados_series"
    assert ficha.cobertura_inicio == "2023-12-01" and ficha.cobertura_fin == "2024-07-10"
    assert ficha.mediciones["series_cargadas"] == 13
    assert ficha.mediciones["excluidas"] == {"estado excluida_ajustada": 1}
    assert ficha.mediciones["barras_intradia_descartadas"] == 1
    assert set(ficha.mediciones["retraso_por_metodo_regla"]) == {"cierre_mercado", "calendario", "lag_fijo", "fichero",
                                                           "derivada"}
    assert "not endorsed or certified" in ficha.licencia_y_terminos
    ruta = guardar_ficha(ficha, tmp_path / "fichas")
    assert ruta.name == "fase1_raw.json"


# --- data/raw real ------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tabla_real():
    if not (RAW / "provenance.json").exists():
        pytest.skip("falta data/raw (gitignored)")
    return fase1.tabla_fase1(catalogo=c.vista_regla_previa())


@pytest.mark.datos
def test_t1_fase1_datos_reales(tabla_real):
    t = tabla_real
    assert validar_series(t).equals(t)
    assert not t.duplicated(["serie", "fecha_referencia", "vintage"]).any()
    cat = c.vista_regla_previa()
    esperadas = {sid for sid, e in cat.items() if e["fichero_fase1"] and e["estado"] in c.ESTADOS_CARGABLES
                 and e["fase1_utilizable"]}
    assert set(t["serie"]) == esperadas
    assert not t["serie"].map(lambda s: cat[s]["estado"]).eq("excluida_ajustada").any()
    diarias_cierre = t["metodo_publicacion"].eq("cierre_mercado")
    assert (t.loc[diarias_cierre, "fecha_publicacion"] >= t.loc[diarias_cierre, "fecha_referencia"]).all()
    dgs10 = t[t["serie"] == "fred:DGS10"]
    assert len(dgs10) > 15000 and (dgs10["fecha_publicacion"] > dgs10["fecha_referencia"]).all()
    # Ninguna barra de yfinance del día de la descarga (2026-07-20, intradía).
    yf = t[t["fuente"] == "yfinance"]
    assert yf["fecha_referencia"].max() < pd.Timestamp("2026-07-20")
    # Revisables con la nota obligatoria; huecos medidos por el catalogador.
    rev = t["serie"].map(lambda s: cat[s]["revisable"])
    assert c.es_nota_revisable(t.loc[rev, "nota"]).all()
    un = t[(t["serie"] == "fred:UNRATE") & t["valor"].isna()]
    assert list(un["fecha_referencia"]) == [pd.Timestamp("2025-10-01")]
    assert set(t.loc[t["nota"].fillna("").str.startswith(fase1.NOTA_HUECO_DIARIO), "serie"]) == {
        "fred:DGS20", "fred:DTB1YR"}
