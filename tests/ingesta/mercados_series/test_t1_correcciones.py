"""Regresiones de la revisión del bloque 2 del Tema 1 (fugas de fecha y fallos de la base).

Sin red. Tablas y ficheros SINTÉTICOS con las entradas reales del catálogo (valores inventados).
Las fechas de publicación son las reales (calendario de BLS, BEA, Chicago Fed, SIFMA...), que son
metadatos públicos. Los tests que leen ``config/excepciones_publicacion.csv`` (generado con ALFRED)
se saltan si aún no existe.
"""
from __future__ import annotations

import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from ingesta.comun.rutas import ROOT
from ingesta.mercados_series import almacen, base, tools, visibilidad
from ingesta.mercados_series import catalogo as c
from ingesta.mercados_series import __main__ as cli
from ingesta.mercados_series.contrato import ErrorContrato, validar_series
from ingesta.mercados_series.fuentes import fase1, fred

NOTEBOOK = ROOT / "notebooks" / "fase2" / "mercados_series" / "02_base_fase1.ipynb"


@pytest.fixture(scope="module")
def cat():
    return c.cargar_catalogo()


def _filas(sid: str, e: dict, fechas, pub, *, disp=None, valores=None, nota=None) -> pd.DataFrame:
    fechas = pd.DatetimeIndex(pd.to_datetime(list(fechas)))
    n = len(fechas)
    return pd.DataFrame({
        "serie": sid, "fecha_referencia": fechas, "fecha_publicacion": pd.to_datetime(list(pub)),
        "disponible_utc": pd.Series(disp if disp is not None else [pd.NaT] * n, dtype="datetime64[ns, UTC]"),
        "vintage": "unico", "valor": valores if valores is not None else np.linspace(1, 2, n),
        "unidad": e["unidad"], "divisa": e["divisa"], "fuente": e["fuente"], "metodo_publicacion": e["metodo"],
        "nota": nota if nota is not None else (c.nota_revisable(e) if e["revisable"] else None),
    })


def _excepciones(filas) -> pd.DataFrame:
    df = pd.DataFrame(filas, columns=["serie", "fecha_referencia", "fecha_publicacion"])
    for col in ("fecha_referencia", "fecha_publicacion"):
        df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
    return df[list(c.COLUMNAS_EXCEPCIONES)]


# --- Excepciones de publicación (cierres del Gobierno) -------------------------------------------
def test_t1_excepcion_unrate_septiembre_2025_visible_el_20_de_noviembre(cat):
    """Regresión: la regla 1M1D daba el paro de sep-2025 el 2025-11-01; BLS lo publicó el 2025-11-20."""
    e = cat["fred:UNRATE"]
    meses = pd.date_range("2025-07-01", "2025-09-01", freq="MS")
    t = _filas("fred:UNRATE", e, meses, c.fecha_publicacion_regla(pd.Series(meses), e))
    assert t.loc[2, "fecha_publicacion"] == pd.Timestamp("2025-11-01")          # la regla sola fugaba
    exc = _excepciones([("fred:UNRATE", "2025-09-01", "2025-11-20")])
    out, n = c.aplicar_excepciones(validar_series(t), exc, catalogo=cat)
    assert n == 1
    r19 = tools.get_serie("fred:UNRATE", "2025-11-19", base=out)
    r20 = tools.get_serie("fred:UNRATE", "2025-11-20", base=out)
    assert r19.ultimo()["fecha_referencia"] == "2025-08-01"
    assert r20.ultimo()["fecha_referencia"] == "2025-09-01"
    assert c.NOTA_EXCEPCION in r20.ultimo()["nota"] and "2025-11-01" in r20.ultimo()["nota"]
    assert any("ALFRED" in a for a in r20.avisos)


def test_t1_excepcion_pib_q3_2025_no_visible_antes_del_23_de_diciembre(cat):
    e = cat["fred:GDPC1"]
    trimestres = pd.date_range("2025-04-01", "2025-07-01", freq="QS")
    t = validar_series(_filas("fred:GDPC1", e, trimestres, c.fecha_publicacion_regla(pd.Series(trimestres), e)))
    out, _ = c.aplicar_excepciones(t, _excepciones([("fred:GDPC1", "2025-07-01", "2025-12-23")]), catalogo=cat)
    assert tools.get_serie("fred:GDPC1", "2025-12-22", base=out).ultimo()["fecha_referencia"] == "2025-04-01"
    assert tools.get_serie("fred:GDPC1", "2025-12-23", base=out).ultimo()["fecha_referencia"] == "2025-07-01"


def test_t1_excepciones_nunca_adelantan_y_son_idempotentes(cat):
    e = cat["fred:PAYEMS"]
    meses = pd.date_range("2025-01-01", "2025-12-01", freq="MS")
    regla = c.fecha_publicacion_regla(pd.Series(meses), e)
    t = validar_series(_filas("fred:PAYEMS", e, meses, regla))
    rng = np.random.default_rng(3)
    exc = _excepciones([("fred:PAYEMS", m, r + pd.Timedelta(days=int(d)))
                        for m, r, d in zip(meses, regla, rng.integers(-40, 40, len(meses)))])
    out, n = c.aplicar_excepciones(t, exc, catalogo=cat)
    assert (out["fecha_publicacion"].to_numpy() >= t["fecha_publicacion"].to_numpy()).all()
    esperado = np.maximum(t["fecha_publicacion"].to_numpy(), exc["fecha_publicacion"].to_numpy())
    assert (out["fecha_publicacion"].to_numpy() == esperado).all()
    assert n == int((exc["fecha_publicacion"].to_numpy() > t["fecha_publicacion"].to_numpy()).sum())
    otra, n2 = c.aplicar_excepciones(out, exc, catalogo=cat)
    assert n2 == 0
    pd.testing.assert_frame_equal(otra.reset_index(drop=True), out.reset_index(drop=True))


def test_t1_excepcion_con_hora_de_publicador_y_derivada_hereda(cat, tmp_path, monkeypatch):
    """Moody's fuera de FRED (oct-2016 a mar-2017): DBAA y la derivada Baa−Aaa no antes del 2017-03-22."""
    dias = pd.bdate_range("2016-11-01", "2016-11-04")
    tablas = {}
    for sid in ("fred:DBAA", "fred:DAAA"):
        e = cat[sid]
        disp = pd.Series([visibilidad.disponible_publicador("moodys", d) for d in dias], dtype="datetime64[ns, UTC]")
        tablas[sid] = _filas(sid, e, dias, [visibilidad.primera_decision(x) for x in disp], disp=disp)
    exc = _excepciones([(sid, d, "2017-03-22") for sid in tablas for d in dias])
    monkeypatch.setattr(c, "RUTA_EXCEPCIONES", tmp_path / "no_existe.csv")
    res, tabla = base.construir_base(tablas={"fred": validar_series(pd.concat(tablas.values()))},
                                     catalogo=cat, escribir=False, excepciones=exc)
    assert res.excepciones_aplicadas == 8
    dbaa = tabla[tabla["serie"] == "fred:DBAA"]
    assert (dbaa["fecha_publicacion"] == pd.Timestamp("2017-03-22")).all()       # 12:00 NY < 16:30
    assert (dbaa["disponible_utc"] == pd.Timestamp("2017-03-22 16:00", tz="UTC")).all()
    spread = tabla[tabla["serie"] == "deriv:DBAA-DAAA"]
    assert len(spread) == 4 and (spread["fecha_publicacion"] >= pd.Timestamp("2017-03-22")).all()


def test_t1_excepcion_compara_con_la_hora_del_publicador(cat):
    """Regresión: el día de ALFRED se comparaba sin su hora; a las 17:05 la decisión es el día siguiente."""
    e = cat["fred:T10YFF"]
    d = pd.Timestamp("2024-01-10")
    disp = visibilidad.disponible_publicador(e["publicador"], d)
    t = validar_series(_filas("fred:T10YFF", e, [d], [visibilidad.primera_decision(disp)],
                              disp=pd.Series([disp], dtype="datetime64[ns, UTC]")))
    assert t["fecha_publicacion"].iloc[0] == pd.Timestamp("2024-01-12")
    out, n = c.aplicar_excepciones(t, _excepciones([("fred:T10YFF", d, "2024-01-12")]), catalogo=cat)
    assert n == 1 and out["fecha_publicacion"].iloc[0] == pd.Timestamp("2024-01-13")
    assert out["disponible_utc"].iloc[0] == pd.Timestamp("2024-01-12 22:05", tz="UTC")


@pytest.mark.skipif(not c.RUTA_EXCEPCIONES.exists(), reason="sin config/excepciones_publicacion.csv")
def test_t1_excepciones_de_config_coherentes(cat):
    exc = c.cargar_excepciones()
    assert set(exc["serie"]) <= set(cat)
    assert (exc["fecha_publicacion"] > exc["fecha_referencia"]).all()
    # Posteriores a la regla vigente (excepción > regla) en las lag_fijo.
    lag = exc[exc["serie"].map(lambda s: cat[s]["metodo"] == "lag_fijo")]
    for sid, g in lag.groupby("serie"):
        regla = c.fecha_publicacion_regla(g["fecha_referencia"].reset_index(drop=True), cat[sid])
        assert (g["fecha_publicacion"].to_numpy() > regla.to_numpy()).all(), sid
    clave = exc.set_index(["serie", "fecha_referencia"])["fecha_publicacion"]
    # Cierres del Gobierno de 2025 medidos en ALFRED (publicaciones de BLS y BEA).
    assert clave[("fred:UNRATE", pd.Timestamp("2025-09-01"))] == pd.Timestamp("2025-11-20")
    assert clave[("fred:PAYEMS", pd.Timestamp("2025-10-01"))] == pd.Timestamp("2025-12-16")
    assert clave[("fred:GDPC1", pd.Timestamp("2025-07-01"))] == pd.Timestamp("2025-12-23")
    # Solo fechas: ningún valor de las series en el CSV del repositorio.
    assert list(pd.read_csv(c.RUTA_EXCEPCIONES, nrows=1).columns) == list(c.COLUMNAS_EXCEPCIONES)


# --- Reglas de publicación corregidas ------------------------------------------------------------
def test_t1_shiller_e_y_d_por_trimestre(cat):
    """Regresión: E de abril (interpolado hacia junio) se veía dos meses antes que el de junio."""
    for sid in ("shiller:E", "shiller:D"):
        pub = c.fecha_publicacion_regla(pd.Series(pd.to_datetime(["2026-04-01", "2026-05-01", "2026-06-01"])), cat[sid])
        assert pub.nunique() == 1 and pub.iloc[0] == pd.Timestamp("2026-10-01"), sid
    cape = c.fecha_publicacion_regla(pd.Series([pd.Timestamp("2026-04-01")]), cat["shiller:CAPE"])
    assert cape.iloc[0] == pd.Timestamp("2026-06-01")                             # el CAPE sigue mensual


@pytest.mark.parametrize("ref, esperado", [
    ("2025-05-01", "2025-06-02"),          # 1 de junio de 2025 en domingo: el lunes
    ("2025-12-01", "2026-01-02"),          # 1 de enero festivo
    ("2026-06-01", "2026-07-01"),          # día 1 hábil: sin cambio
])
def test_t1_medias_mensuales_h15_al_primer_habil(cat, ref, esperado):
    for sid in ("fred:GS10", "fred:TB3MS", "fred:BAA"):
        assert c.fecha_publicacion_regla(pd.Series([pd.Timestamp(ref)]), cat[sid]).iloc[0] == pd.Timestamp(esperado)


@pytest.mark.parametrize("viernes, esperado", [
    ("2024-12-20", "2024-12-27"),          # Nochebuena (cierre extraordinario) y Navidad: tercer hábil
    ("2025-10-10", "2025-10-16"),          # lunes festivo (Columbus): jueves
    ("2025-10-03", "2025-10-08"),          # semana normal: miércoles
])
def test_t1_nfci_tercer_habil(cat, viernes, esperado):
    for sid in ("fred:NFCI", "fred:ANFCI", "fred:NFCICREDIT"):
        assert c.fecha_publicacion_regla(pd.Series([pd.Timestamp(viernes)]), cat[sid]).iloc[0] == pd.Timestamp(esperado)


def test_t1_fecha_publicacion_lag_habiles_valida():
    with pytest.raises(ValueError, match="n_habiles"):
        c.fecha_publicacion_lag(["2024-01-05"], "5D", "fecha", n_habiles=3)
    assert c.fecha_publicacion_lag([None], "1D", "mes", habiles="us_federal").isna().all()


def test_t1_sofr_viernes_santo():
    """Regresión: SOFR del Jueves Santo de 2026 salió el lunes (SIFMA cierra en Viernes Santo)."""
    t = visibilidad.disponible_publicador("nyfed_sofr", "2026-04-02")
    assert visibilidad.primera_decision(t) == pd.Timestamp("2026-04-06")
    # EFFR (us_federal, sin festivos móviles) sí sale el Viernes Santo.
    assert visibilidad.primera_decision(visibilidad.disponible_publicador("nyfed_effr", "2026-04-02")) == \
        pd.Timestamp("2026-04-03")


def test_t1_t10y2y_antes_de_2019_no_antes_que_sus_componentes(cat):
    """Regresión: el diferencial del viernes 2008-09-12 se veía el sábado y DGS10 el lunes."""
    viernes = pd.Timestamp("2008-09-12")
    pub = lambda p, d: visibilidad.primera_decision(visibilidad.disponible_publicador(p, d))  # noqa: E731
    assert pub("fred_tesoro_calculadas", viernes) == pd.Timestamp("2008-09-15") == pub("fed_h15", viernes)
    for d in pd.bdate_range("2010-01-04", "2019-06-20", freq="7B")[:80]:
        assert pub("fred_tesoro_calculadas", d) >= pub("fed_h15", d)
    # Desde el 2019-06-21, la hora medida: el viernes se ve el sábado.
    assert pub("fred_tesoro_calculadas", "2020-01-10") == pd.Timestamp("2020-01-11")


def test_t1_publicador_previo_y_festivos_validados(tmp_path):
    import yaml

    crudo = yaml.safe_load(visibilidad.RUTA_PUBLICACIONES.read_text(encoding="utf-8"))
    crudo["publicadores"]["fred_tesoro_calculadas"]["previo"] = {"hasta": "2019-06-20", "publicador": "no_existe"}
    ruta = tmp_path / "pub.yaml"
    ruta.write_text(yaml.safe_dump(crudo, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="previo"):
        visibilidad.cargar_publicaciones(ruta)
    crudo["publicadores"]["fred_tesoro_calculadas"]["previo"]["publicador"] = "fed_h15"
    crudo["publicadores"]["nyfed_sofr"]["festivos_moviles"] = ["carnaval"]
    ruta.write_text(yaml.safe_dump(crudo, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="festivos_moviles"):
        visibilidad.cargar_publicaciones(ruta)


# --- Índices reconstruidos hacia atrás -----------------------------------------------------------
def test_t1_reconstruidas_anotadas_y_avisadas(cat):
    e = cat["fred:NFCI"]
    assert e["publicado_desde"] == "2011-05-25"
    # El viernes 2011-05-20 sale el miércoles 2011-05-25: es la primera publicación real.
    viernes = pd.to_datetime(["2008-10-03", "2011-05-13", "2011-05-20", "2011-06-03"])
    t = validar_series(_filas("fred:NFCI", e, viernes, c.fecha_publicacion_regla(pd.Series(viernes), e)))
    out = c.anotar_reconstruidas(t, cat)
    rec = out["nota"].str.contains(c.NOTA_RECONSTRUIDA, regex=False)
    assert rec.tolist() == [True, True, False, False]
    assert c.es_nota_revisable(out["nota"]).all()                                # la nota obligatoria sigue
    pd.testing.assert_frame_equal(c.anotar_reconstruidas(out, cat), out)       # idempotente
    r = tools.get_serie("fred:NFCI", "2012-01-01", base=out)
    assert any("reconstruidos" in a and "2011-05-25" in a for a in r.avisos)


# --- Base: revisiones acotadas y huérfanas -------------------------------------------------------
def _ff(cat, valores) -> pd.DataFrame:
    e = cat["ff:F3D.SMB"]
    dias = pd.bdate_range("2024-01-02", periods=len(valores))
    return validar_series(_filas("ff:F3D.SMB", e, dias, c.fecha_publicacion_regla(pd.Series(dias), e),
                                 valores=np.asarray(valores, float)))


def _vix(cat, valores) -> pd.DataFrame:
    e = cat["fred:VIXCLS"]
    dias = pd.bdate_range("2024-01-02", periods=len(valores))
    disp = visibilidad.disponible_cierre_serie(pd.Series(dias), e["mercado"])
    return validar_series(_filas("fred:VIXCLS", e, dias, visibilidad.fecha_publicacion_cierre_serie(pd.Series(dias),
                                                                                                    e["mercado"]),
                                 disp=disp.to_numpy(), valores=np.asarray(valores, float)))


def test_t1_revision_academica_pasa_y_la_de_fred_no(cat, tmp_path, monkeypatch):
    """Regresión: una edición nueva de French obligaba a --reemplazar (sin protección para todas)."""
    monkeypatch.setattr(c, "RUTA_EXCEPCIONES", tmp_path / "no_existe.csv")
    ruta = tmp_path / "base.parquet"
    base.construir_base(tablas={"academicos": _ff(cat, [1, 2, 3]), "fred": _vix(cat, [15, 16, 17])},
                        catalogo=cat, ruta=ruta)
    res, _ = base.construir_base(tablas={"academicos": _ff(cat, [1, 2.5, 3, 4]), "fred": _vix(cat, [15, 16, 17])},
                                 catalogo=cat, ruta=ruta)
    assert res.revisables == 1 and almacen.leer_series("ff:F3D.SMB", ruta)["valor"].tolist() == [1, 2.5, 3, 4]
    with pytest.raises(ErrorContrato, match="valor"):
        base.construir_base(tablas={"academicos": _ff(cat, [1, 2.5, 3, 4]), "fred": _vix(cat, [15, 16.5, 17])},
                            catalogo=cat, ruta=ruta)


def test_t1_revisable_no_puede_cambiar_fecha(cat, tmp_path):
    ruta = tmp_path / "base.parquet"
    t = _ff(cat, [1, 2, 3])
    almacen.escribir_series(t, ruta)
    otra = t.assign(fecha_publicacion=t["fecha_publicacion"] + pd.Timedelta(days=1))
    with pytest.raises(ErrorContrato, match="fecha_publicacion"):
        almacen.escribir_series(otra, ruta, revisables=["ff:F3D.SMB"])
    almacen.escribir_series(t.assign(valor=[1, 2, 9]), ruta, revisables=["ff:F3D.SMB"])


def test_t1_huerfanas_fallan_o_se_purgan(cat, tmp_path, monkeypatch):
    """Regresión: una serie que deja de construirse seguía en el parquet y get_serie la servía."""
    monkeypatch.setattr(c, "RUTA_EXCEPCIONES", tmp_path / "no_existe.csv")
    ruta = tmp_path / "base.parquet"
    base.construir_base(tablas={"academicos": _ff(cat, [1, 2]), "fred": _vix(cat, [15, 16])}, catalogo=cat, ruta=ruta)
    with pytest.raises(ErrorContrato, match="huérfanas"):
        base.construir_base(tablas={"academicos": _ff(cat, [1, 2])}, catalogo=cat, ruta=ruta)
    res, _ = base.construir_base(tablas={"academicos": _ff(cat, [1, 2])}, catalogo=cat, ruta=ruta, purgar=True)
    assert res.purgadas == ["fred:VIXCLS"] and set(almacen.leer_series(ruta=ruta)["serie"]) == {"ff:F3D.SMB"}


def test_t1_cli_sin_escribir_no_toca_results(cat, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(c, "RUTA_EXCEPCIONES", tmp_path / "no_existe.csv")
    monkeypatch.setattr(almacen, "corpus_dir", lambda tema=None: tmp_path / "corpus")
    escritas = []
    monkeypatch.setattr(base, "escribir_cobertura", lambda *a, **k: escritas.append(a))
    monkeypatch.setattr(base, "escribir_anomalias", lambda *a, **k: escritas.append(a))
    monkeypatch.setattr(base, "tablas_fuentes", lambda **kw: {"fred": _vix(cat, [15, 16])})
    assert cli.main(["construir-base", "--sin-red", "--sin-escribir"]) == 0
    assert escritas == [] and not (tmp_path / "corpus").exists()
    assert "ensayo" in capsys.readouterr().out


# --- Anomalías ------------------------------------------------------------------------------------
def test_t1_anomalias_pico_fin_de_semana_y_racha(cat):
    e = cat["fred:VXDCLS"]
    dias = list(pd.bdate_range("2021-07-01", "2021-07-30"))
    valores = np.full(len(dias), 15.0) + np.linspace(0, 1, len(dias))
    valores[8] = 2.71                                                             # pico aislado sintético
    dias.append(pd.Timestamp("2021-07-25"))                                       # domingo
    valores = np.r_[valores, 15.5]
    t = _filas("fred:VXDCLS", e, dias, [d + pd.Timedelta(days=1) for d in dias], valores=valores)
    out, anom = base.anotar_anomalias(validar_series(t), cat)
    notas = out.set_index("fecha_referencia")["nota"]
    assert base.NOTA_PICO in notas[dias[8]] and base.NOTA_FIN_DE_SEMANA in notas[pd.Timestamp("2021-07-25")]
    assert set(anom["tipo"]) == {"pico", "fin_de_semana"} and "valor" not in anom.columns
    assert (out["valor"].to_numpy() == validar_series(t)["valor"].to_numpy()).all()  # nunca toca valores
    e2 = cat["fred:DGS10"]
    plano = _filas("fred:DGS10", e2, pd.bdate_range("2013-01-01", periods=130),
                   pd.bdate_range("2013-01-02", periods=130), valores=np.full(130, 0.28))
    out2, anom2 = base.anotar_anomalias(validar_series(plano), cat)
    assert (anom2["tipo"] == "racha").all() and out2["nota"].str.contains("racha plana de 130").all()


# --- data/raw: descarga declarada y periodos provisionales ---------------------------------------
def test_t1_descarga_declarada_si_cambia_el_mtime(tmp_path):
    """Regresión: con data/raw copiado (mtime nuevo) la barra intradía entraba en silencio."""
    raiz = tmp_path / "raw"
    ruta = raiz / "yfinance" / "SP500.parquet"
    ruta.parent.mkdir(parents=True)
    dias = pd.bdate_range("2024-03-14", "2024-03-20")
    pd.DataFrame({"SP500": np.linspace(100, 105, len(dias))}, index=pd.DatetimeIndex(dias, name="Date")).to_parquet(ruta)
    copia = pd.Timestamp("2024-05-02 09:00", tz="America/New_York")
    os.utime(ruta, (copia.timestamp(), copia.timestamp()))
    declaradas = {"por_defecto": pd.Timestamp("2024-03-20 15:37", tz="America/New_York").tz_convert("UTC"),
                  "ficheros": {}, "tolerancia": pd.Timedelta(minutes=10)}
    sub = {"yf:^GSPC": c.por_id("yf:^GSPC")}
    sub["yf:^GSPC"]["fichero_fase1"] = ["yfinance/SP500.parquet:SP500"]
    with pytest.warns(UserWarning, match="declarada"):
        t = fase1.tabla_fase1(raiz, sub, descargas=declaradas)
    assert t["fecha_referencia"].max() == pd.Timestamp("2024-03-19")              # sin la barra intradía
    # Sin declaración (carpeta ajena a data/raw) manda el mtime: es justo el riesgo que se corrige.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert fase1.tabla_fase1(raiz, sub)["fecha_referencia"].max() == pd.Timestamp("2024-03-20")


def test_t1_descargas_declaradas_de_config():
    d = fase1.descargas_declaradas()
    assert d["por_defecto"].tz_convert("America/New_York").date() == pd.Timestamp("2026-07-20").date()
    assert "academico/SHILLER_SP500_CAPE.parquet" in d["ficheros"]


def test_t1_fase1_descarta_mes_provisional_de_shiller(tmp_path):
    raiz = tmp_path / "raw"
    ruta = raiz / "academico" / "SHILLER_SP500_CAPE.parquet"
    ruta.parent.mkdir(parents=True)
    meses = pd.date_range("2026-06-01", "2026-09-01", freq="MS")
    pd.DataFrame({"SHILLER_SP500_CAPE": [30.0, 30.5, 31.0, 31.2]}, index=meses).to_parquet(ruta)
    descarga = pd.Timestamp("2026-09-29 04:21", tz="America/New_York")
    os.utime(ruta, (descarga.timestamp(), descarga.timestamp()))
    e = c.por_id("shiller:CAPE")
    e["fichero_fase1"] = ["academico/SHILLER_SP500_CAPE.parquet:SHILLER_SP500_CAPE"]
    e["fase1_utilizable"] = True
    t = fase1.tabla_fase1(raiz, {"shiller:CAPE": e})
    assert t["fecha_referencia"].max() == pd.Timestamp("2026-08-01")              # septiembre, provisional


# --- get_serie ------------------------------------------------------------------------------------
def test_t1_get_serie_agrupa_huecos(cat):
    """Regresión: un aviso por fila nula (CPFF: 1011 avisos, ~127 KB por llamada)."""
    e = cat["fred:UMCSENT"]
    meses = pd.date_range("1960-01-01", "1979-12-01", freq="MS")
    valores = np.where(np.arange(len(meses)) % 3 == 0, 90.0, np.nan)
    nota = [None if v == v else f"{fred.NOTA_PUNTO}" for v in valores]
    t = validar_series(_filas("fred:UMCSENT", e, meses, c.fecha_publicacion_regla(pd.Series(meses), e),
                              valores=valores, nota=nota))
    r = tools.get_serie("fred:UMCSENT", "2000-01-01", base=t)
    assert len(r.avisos) < 5
    assert any(re.search(r"\d+ periodos con valor nulo", a) for a in r.avisos)


def test_t1_get_serie_tipo_de_cambio_no_se_convierte(cat):
    e = cat["fred:DEXJPUS"]
    assert e["divisa"] is None
    dias = pd.bdate_range("2024-02-12", "2024-02-16")
    disp = pd.Series([visibilidad.disponible_publicador(e["publicador"], d) for d in dias], dtype="datetime64[ns, UTC]")
    t = validar_series(_filas("fred:DEXJPUS", e, dias, [visibilidad.primera_decision(x) for x in disp], disp=disp,
                              valores=np.full(len(dias), 150.0)))
    r = tools.get_serie("fred:DEXJPUS", "2024-03-01", base=t, divisa="USD")
    assert r.datos["valor"].iloc[-1] == 150.0 and pd.isna(r.datos["fx_usd"].iloc[-1])
    assert any("no es monetaria" in a for a in r.avisos)


def test_t1_get_serie_id_sin_distinguir_mayusculas(cat):
    assert tools.resolver_id("gw:M.TBL") == "gw:M.tbl"
    assert tools.resolver_id("gw:M.tbl") == "gw:M.tbl"
    assert tools.resolver_id("vix") == "fred:VIXCLS"
    assert tools.resolver_id("sint:NO_EXISTE") == "sint:NO_EXISTE"
    assert c.resolver_caja("gw:m.TBL") == "gw:M.tbl"


def test_t1_cache_publicador_invalida_al_editar_yaml():
    fred._disponible_publicador_vec("fed_h15", pd.Series([pd.Timestamp("2024-01-05")]), None)
    claves = [k for k in fred._cache_publicador if k[0] == "fed_h15"]
    assert claves and all(len(k) == 3 and k[2] == visibilidad.RUTA_PUBLICACIONES.stat().st_mtime_ns for k in claves)


# --- Notas de revisables --------------------------------------------------------------------------
def test_t1_nota_revisable_segun_fuente(cat):
    assert c.nota_revisable(cat["fred:UNRATE"]) == c.NOTA_REVISABLE
    for sid in ("ff:F3D.SMB", "gw:M.tbl", "shiller:E", "ofr:FSI"):
        assert c.nota_revisable(cat[sid]) == c.NOTA_REVISABLE_EDICION, sid
        assert "ALFRED en el bloque 3" not in c.nota_revisable(cat[sid])


# --- Notebook sin valores de series ---------------------------------------------------------------
@pytest.mark.skipif(not NOTEBOOK.exists(), reason="sin el notebook 02")
def test_t1_notebook_02_sin_valores_en_salidas():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for i, celda in enumerate(nb["cells"]):
        for salida in celda.get("outputs", []):
            datos = salida.get("data", {})
            html = "".join(datos.get("text/html", []))
            assert "<th>valor</th>" not in html, f"celda {i}: tabla con columna 'valor'"
            # Tablas impresas (repr de DataFrame): la cabecera es la primera línea.
            texto = "".join(datos.get("text/plain", []))
            cabecera = texto.splitlines()[0].split() if texto.strip() else []
            assert "valor" not in cabecera, f"celda {i}: tabla con columna 'valor'"
            for linea in "".join(salida.get("text", [])).splitlines():   # print(DataFrame)
                assert linea.split()[:1] != ["serie"] or "valor" not in linea.split(), f"celda {i}: {linea!r}"
    assert NOTEBOOK.stat().st_size < 2_000_000
