"""Tests de la base de series del Tema 1 (``base``, ``__main__`` y ``tools.get_serie`` con el catálogo).

Sin red: los cargadores se sustituyen por tablas SINTÉTICAS con las entradas reales del catálogo
(valores aleatorios, ningún dato real), fechadas con las reglas del tema
(``fuentes.fred.tabla_serie`` y ``catalogo.fecha_publicacion_lag``). El test sobre la base real
(``<corpus>/mercados_series/series/base_series.parquet``) lleva ``@pytest.mark.datos`` y se salta
si no se ha construido (``python -m ingesta.mercados_series construir-base --sin-red``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ingesta.mercados_series import almacen, base, timegate, tools
from ingesta.mercados_series import catalogo as c
from ingesta.mercados_series import __main__ as cli
from ingesta.mercados_series.contrato import ErrorContrato, fecha_publicacion_derivada, validar_series
from ingesta.mercados_series.fuentes import alfred, fred

HABILES = pd.bdate_range("2024-01-02", "2024-03-29")


# Bloque 3: las revisables de ALFRED (vintage_exacto) en su forma del bloque 2 (regla previa, lag_fijo):
# estos tests ejercitan las reglas de publicación con entradas reales del catálogo.
@pytest.fixture(scope="module")
def cat():
    return c.vista_regla_previa()


@pytest.fixture
def corpus_en_tmp(tmp_path, monkeypatch):
    """Las rutas por defecto del almacén y de results apuntan a ``tmp_path`` (nunca al corpus real)."""
    def corpus_dir(tema=None):
        p = tmp_path / "corpus" / tema if tema else tmp_path / "corpus"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resultados_dir(tema):
        p = tmp_path / "results" / tema
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(almacen, "corpus_dir", corpus_dir)
    monkeypatch.setattr(base, "resultados_dir", resultados_dir)
    # Sin las excepciones medidas de config: los datos sintéticos no son los reales.
    monkeypatch.setattr(c, "RUTA_EXCEPCIONES", tmp_path / "sin_excepciones.csv")
    return tmp_path


def _obs(fechas, rng, *, ausentes=()) -> pd.DataFrame:
    fechas = pd.DatetimeIndex(fechas)
    valor = np.round(rng.normal(3, 0.5, len(fechas)), 2)
    ausente = fechas.isin(pd.DatetimeIndex(list(ausentes)))
    valor[ausente] = np.nan
    return pd.DataFrame({"fecha_referencia": fechas, "valor": valor, "ausente": ausente})


def _fred(sid, cat, fechas, rng, **kw) -> pd.DataFrame:
    """Filas del contrato fechadas con la regla del catálogo (calendario, cierre o lag_fijo)."""
    return fred.tabla_serie(sid, _obs(fechas, rng, **kw), cat[sid])


def _fichero(sid, cat, fechas, rng) -> pd.DataFrame:
    e = cat[sid]
    fechas = pd.Series(pd.DatetimeIndex(fechas))
    return pd.DataFrame({
        "serie": sid, "fecha_referencia": fechas, "fecha_publicacion": c.fecha_publicacion_regla(fechas, e),
        "disponible_utc": pd.NaT, "vintage": "unico", "valor": rng.normal(0, 1, len(fechas)), "unidad": e["unidad"],
        "divisa": e["divisa"], "fuente": e["fuente"], "metodo_publicacion": e["metodo"], "nota": c.nota_revisable(e),
    })


def _copia_derivada(sid, cat, comp: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Derivada «de data/raw»: A − B con la fecha de sus componentes (como la carga fuentes.fase1)."""
    e = cat[sid]
    comp = comp[comp["valor"].notna()]
    f = fecha_publicacion_derivada(comp, [a, b]).set_index("fecha_referencia")
    va = comp[comp["serie"] == a].set_index("fecha_referencia")["valor"].reindex(f.index)
    vb = comp[comp["serie"] == b].set_index("fecha_referencia")["valor"].reindex(f.index)
    return pd.DataFrame({
        "serie": sid, "fecha_referencia": f.index, "fecha_publicacion": f["fecha_publicacion"].to_numpy(),
        "disponible_utc": f["disponible_utc"].to_numpy(), "vintage": "unico", "valor": (va - vb).to_numpy(),
        "unidad": e["unidad"], "divisa": e["divisa"], "fuente": e["fuente"], "metodo_publicacion": "derivada",
        "nota": None,
    })


@pytest.fixture
def tablas(cat):
    """Tablas sintéticas por cargador con un conflicto (ff:F3D.SMB en fase1 y académicos)."""
    rng = np.random.default_rng(11)
    vix = _fred("fred:VIXCLS", cat, HABILES, rng)
    cpi = _fred("fred:CPIAUCSL", cat, pd.date_range("2023-01-01", "2024-02-01", freq="MS"), rng)
    sofr = _fred("fred:SOFR", cat, HABILES, rng)
    effr = _fred("fred:EFFR", cat, HABILES, rng)
    dbaa = _fred("fred:DBAA", cat, HABILES, rng, ausentes=[pd.Timestamp("2024-01-15")])
    daaa = _fred("fred:DAAA", cat, HABILES, rng)
    # La copia de data/raw del diferencial acaba antes que los componentes de FRED.
    copia = _copia_derivada("deriv:DBAA-DAAA", cat, pd.concat([dbaa, daaa]), "fred:DBAA", "fred:DAAA")
    copia = copia[copia["fecha_referencia"] <= pd.Timestamp("2024-02-29")]
    smb_f1 = _fichero("ff:F3D.SMB", cat, HABILES[:30], rng)
    smb_ac = _fichero("ff:F3D.SMB", cat, HABILES, rng)
    return {
        "fase1": validar_series(pd.concat([vix, cpi, sofr, effr, copia, smb_f1], ignore_index=True)),
        "fred": validar_series(pd.concat([dbaa, daaa], ignore_index=True)),
        "academicos": validar_series(smb_ac),
    }


# --- Unión y conflictos -------------------------------------------------------------------------
def test_t1_unir_una_fuente_por_serie(tablas):
    tabla, elegido, conflictos = base.unir(tablas)
    assert conflictos == {"ff:F3D.SMB": {"cargadores": ["fase1", "academicos"], "elegido": "academicos"}}
    assert elegido["ff:F3D.SMB"] == "academicos" and elegido["fred:DBAA"] == "fred"
    smb = tabla[tabla["serie"] == "ff:F3D.SMB"]
    assert len(smb) == len(tablas["academicos"])                         # solo la versión académica
    assert not tabla.duplicated(["serie", "fecha_referencia", "vintage"]).any()


def test_t1_unir_rechaza_cargador_desconocido(tablas):
    with pytest.raises(ValueError, match="cargadores desconocidos"):
        base.unir({"otro": tablas["fred"]})


def test_t1_prioridad_academicos_sobre_fred_y_fase1():
    assert base.PRIORIDAD["academicos"] > base.PRIORIDAD["fred"] > base.PRIORIDAD["fase1"]
    assert tuple(base.PRIORIDAD) == base.CARGADORES


# --- Derivadas ----------------------------------------------------------------------------------
def test_t1_derivadas_recalculadas_con_sus_componentes(tablas, cat):
    union, _, _ = base.unir(tablas)
    out, info = base.derivar_diferencias(union, cat)
    d = info["deriv:DBAA-DAAA"]
    assert d["recalculada"] and d["componentes"] == ["fred:DBAA", "fred:DAAA"]
    copia = d["copia_fase1"]
    assert copia["valores_distintos"] == 0 and copia["publicacion_distinta"] == 0 and copia["solo_copia"] == 0
    assert copia["solo_calculo"] == int((HABILES > pd.Timestamp("2024-02-29")).sum())   # tramo nuevo de FRED
    spread = out[out["serie"] == "deriv:DBAA-DAAA"].set_index("fecha_referencia")
    assert pd.Timestamp("2024-01-15") not in spread.index             # DBAA sin valor ('.'): sin fila
    a = union[union["serie"] == "fred:DBAA"].set_index("fecha_referencia")
    b = union[union["serie"] == "fred:DAAA"].set_index("fecha_referencia")
    dia = pd.Timestamp("2024-03-28")
    assert spread.loc[dia, "valor"] == pytest.approx(a.loc[dia, "valor"] - b.loc[dia, "valor"])
    assert spread.loc[dia, "fecha_publicacion"] == max(a.loc[dia, "fecha_publicacion"], b.loc[dia, "fecha_publicacion"])
    # Moody's publica en D+1 a las 12:00: el jueves 28 se ve el viernes 29 (Viernes Santo no es festivo federal).
    assert spread.loc[dia, "fecha_publicacion"] == pd.Timestamp("2024-03-29")
    # SOFR − EFFR no estaba en ningún cargador: se crea desde sus componentes.
    nueva = out[out["serie"] == "deriv:SOFR-EFFR"]
    assert info["deriv:SOFR-EFFR"]["recalculada"] and "copia_fase1" not in info["deriv:SOFR-EFFR"]
    assert len(nueva) == len(HABILES) and (nueva["metodo_publicacion"] == "derivada").all()
    assert (nueva["fecha_publicacion"] >= nueva["fecha_referencia"]).all()


def test_t1_derivada_sin_componentes_queda_como_estaba(tablas, cat):
    sin_fred = {k: v for k, v in tablas.items() if k != "fred"}
    union, _, _ = base.unir(sin_fred)
    out, info = base.derivar_diferencias(union, cat)
    assert info["deriv:DBAA-DAAA"] == {"recalculada": False, "motivo": "componentes sin datos: fred:DAAA, fred:DBAA"}
    pd.testing.assert_frame_equal(out[out["serie"] == "deriv:DBAA-DAAA"].reset_index(drop=True),
                                  union[union["serie"] == "deriv:DBAA-DAAA"].reset_index(drop=True))


# --- Construcción -------------------------------------------------------------------------------
def test_t1_construir_base_escribe_y_resume(tablas, cat, tmp_path):
    ruta = tmp_path / "base_series.parquet"
    res, tabla = base.construir_base(tablas=tablas, catalogo=cat, ruta=ruta)
    assert res.escrita and ruta.exists() and res.ruta == str(ruta)
    guardada = almacen.leer_series(ruta=ruta)
    pd.testing.assert_frame_equal(guardada, tabla)
    assert res.n_series == 9 and res.n_filas == len(tabla)
    assert res.por_cargador == {"academicos": 1, "derivadas": 2, "fase1": 4, "fred": 2}
    assert res.n_huecos == 0                                             # el '.' suelto de DBAA (festivo) no es un hueco
    assert set(res.conflictos) == {"ff:F3D.SMB"}
    assert "fred:DGS10" in res.sin_datos and "yf:SPY" not in res.sin_datos   # ETF: no cargable
    assert res.por_metodo["derivada"] == 2 and sum(res.por_estado.values()) == res.n_series
    # Reconstruir con lo mismo no cambia nada; con un valor cambiado, la base que solo crece lo impide.
    base.construir_base(tablas=tablas, catalogo=cat, ruta=ruta)
    otra = dict(tablas)
    vix = otra["fase1"].copy()
    vix.loc[vix["serie"] == "fred:VIXCLS", "valor"] += 1.0
    otra["fase1"] = vix
    with pytest.raises(ErrorContrato, match="solo crece"):
        base.construir_base(tablas=otra, catalogo=cat, ruta=ruta)
    res2, _ = base.construir_base(tablas=otra, catalogo=cat, ruta=ruta, permitir_perdidas=True)
    assert res2.escrita


def test_t1_construir_base_sin_escribir(tablas, cat, tmp_path):
    res, tabla = base.construir_base(tablas=tablas, catalogo=cat, ruta=tmp_path / "x.parquet", escribir=False)
    assert not res.escrita and not (tmp_path / "x.parquet").exists() and len(tabla) == res.n_filas


def test_t1_construir_base_valida_contra_catalogo(tablas, cat, tmp_path):
    malas = dict(tablas)
    f = malas["fred"].copy()
    f.loc[f["serie"] == "fred:DAAA", "unidad"] = "otra unidad"
    malas["fred"] = f
    with pytest.raises(ErrorContrato, match="unidad"):
        base.construir_base(tablas=malas, catalogo=cat, ruta=tmp_path / "x.parquet")
    assert not (tmp_path / "x.parquet").exists()


def test_t1_catalogo_efectivo_wei(cat):
    assert cat["fred:WEI"]["estado"] == "activa"                       # bloque 3: ya se carga de ALFRED
    pendiente = {**cat, "fred:WEI": dict(cat["fred:WEI"], estado="pendiente_alfred")}
    ef = base.catalogo_efectivo(pendiente)
    assert ef["fred:WEI"]["estado"] == "activa"
    assert ef["yf:SPY"]["estado"] == "excluida_ajustada"


# --- Cobertura (solo metadatos) -----------------------------------------------------------------
def test_t1_cobertura_solo_metadatos(tablas, cat, tmp_path):
    _, tabla = base.construir_base(tablas=tablas, catalogo=cat, escribir=False)
    cob = base.cobertura(tabla, cat)
    assert list(cob.columns) == list(base.COLUMNAS_COBERTURA) and "valor" not in cob.columns
    assert len(cob) == tabla["serie"].nunique()
    fila = cob.set_index("id").loc["fred:CPIAUCSL"]
    assert fila["metodo"] == "lag_fijo" and fila["inicio"] == "2023-01-01" and fila["fin"] == "2024-02-01"
    assert fila["lag_mediano_dias"] == pytest.approx(60, abs=2)          # 1M1D desde el fin del mes
    dbaa = cob.set_index("id").loc["fred:DBAA"]
    assert dbaa["n_huecos"] == 0 and dbaa["n_obs"] == len(HABILES) - 1
    vix = cob.set_index("id").loc["fred:VIXCLS"]
    assert vix["alias"] == "VIX | VIX_VALIDATION_REF" and vix["lag_min_dias"] >= 1   # CBOE: t+1
    ruta = base.escribir_cobertura(cob, tmp_path / "cob.csv")
    assert pd.read_csv(ruta).shape == cob.shape


def test_t1_resumen_base(tablas, cat):
    _, tabla = base.construir_base(tablas=tablas, catalogo=cat, escribir=False)
    r = base.resumen_base(tabla, cat)
    assert r["n_series"] == 9 and r["no_catalogadas"] == [] and sum(r["metodo"].values()) == 9


# --- CLI ----------------------------------------------------------------------------------------
def test_t1_cli_construir_y_resumen(tablas, corpus_en_tmp, monkeypatch, capsys):
    monkeypatch.setattr(base, "tablas_fuentes", lambda **kw: tablas)
    assert cli.main(["resumen"]) == 1                                     # sin base todavía
    assert cli.main(["construir-base", "--sin-red"]) == 0
    ruta = corpus_en_tmp / "corpus" / "mercados_series" / "series" / "base_series.parquet"
    assert ruta.exists()
    cob = corpus_en_tmp / "results" / "mercados_series" / "tablas" / "cobertura_series.csv"
    assert cob.exists() and "valor" not in pd.read_csv(cob).columns
    assert cli.main(["resumen", "--csv"]) == 0
    salida = capsys.readouterr().out
    assert '"n_series": 9' in salida and "not endorsed" in salida


# --- get_serie con el catálogo ------------------------------------------------------------------
def test_t1_get_serie_alias_del_catalogo_por_defecto(tablas, cat):
    _, tabla = base.construir_base(tablas=tablas, catalogo=cat, escribir=False)
    assert tools.alias_por_defecto()["VIX"] == "fred:VIXCLS"
    assert tools.alias_por_defecto()["EURUSD"].startswith("ecb:")         # alias de publicaciones.yaml
    r = tools.get_serie("VIX", "2024-03-15", base=tabla)
    assert r.serie == "fred:VIXCLS"
    assert r.ultimo()["fecha_referencia"] == "2024-03-14"                  # Cboe 16:15 NY: t+1
    d = tools.get_serie("MOODYS_BAA_AAA_SPREAD", "2024-03-15", base=tabla)
    assert d.serie == "deriv:DBAA-DAAA" and d.ultimo()["fecha_referencia"] == "2024-03-14"


def test_t1_get_serie_avisos_revisable_y_lag_fijo(tablas, cat):
    _, tabla = base.construir_base(tablas=tablas, catalogo=cat, escribir=False)
    r = tools.get_serie("CPIAUCSL", "2024-03-15", base=tabla)
    assert r.ultimo()["fecha_referencia"] == "2024-01-01"                  # febrero: el 1 de abril
    assert any("lag_fijo" in a for a in r.avisos) and any("revisable" in a for a in r.avisos)
    smb = tools.get_serie("ff:F3D.SMB", "2024-03-29", base=tabla)
    assert not any("lag_fijo" in a for a in smb.avisos) and any("revisable" in a for a in smb.avisos)
    vix = tools.get_serie("VIX", "2024-03-15", base=tabla)
    assert vix.avisos == []


@pytest.mark.parametrize("id", ["SP500_CAPWEIGHT_ETF", "yf:SPY", "SPY", "SPDR_XLK"])
def test_t1_get_serie_etf_excluida_remite_a_get_precios(id, tablas, cat):
    _, tabla = base.construir_base(tablas=tablas, catalogo=cat, escribir=False)
    with pytest.raises(KeyError, match="get_precios"):
        tools.get_serie(id, "2024-03-15", base=tabla)


def test_t1_get_serie_desconocida_sigue_siendo_neutra(tablas, cat):
    _, tabla = base.construir_base(tablas=tablas, catalogo=cat, escribir=False)
    with pytest.raises(KeyError, match="desconocida o aún no publicada"):
        tools.get_serie("fred:NO_EXISTE", "2024-03-15", base=tabla)
    with pytest.raises(KeyError, match="desconocida o aún no publicada"):
        tools.get_serie("VIX", "2023-12-01", base=tabla)                 # aún sin datos: mismo mensaje


# --- Bloque 3: ALFRED, fuente viva y reconstrucción desde cero -----------------------------------
@pytest.fixture(scope="module")
def cat3():
    """Catálogo real del bloque 3 (vintage_exacto con regla previa)."""
    return c.cargar_catalogo()


def _vint(fecha: str, valor: str, rs: str, re: str = "9999-12-31") -> dict:
    """Fila de ALFRED (output_type=1) SINTÉTICA."""
    return {"date": fecha, "value": valor, "realtime_start": rs, "realtime_end": re}


def _alfred(sid, cat3, filas) -> pd.DataFrame:
    t, _ = alfred.tabla_serie_alfred(sid, alfred.parsear_vintages(filas), cat3[sid])
    return validar_series(t)


@pytest.fixture
def unrate_alfred(cat3):
    """UNRATE sintético: 1959-1960 llega con el primer vintage (1960-03-15); abril de 2020 sale el 8 de mayo."""
    v0 = cat3["fred:UNRATE"]["regla_previa_alfred"]["primer_vintage"]
    filas = [_vint(f"1959-{m:02d}-01", f"{5 + m / 10:.1f}", v0) for m in range(1, 13)]
    filas += [_vint("1960-01-01", "5.2", v0), _vint("1960-02-01", "4.8", v0),
              _vint("2020-03-01", "4.4", "2020-04-03", "2020-05-07"), _vint("2020-03-01", "4.5", "2020-05-08"),
              _vint("2020-04-01", "14.7", "2020-05-08", "2020-06-04"), _vint("2020-04-01", "14.8", "2020-06-05")]
    return _alfred("fred:UNRATE", cat3, filas)


def test_t1_prioridad_bloque3_alfred_primero():
    assert base.PRIORIDAD["alfred"] > base.PRIORIDAD["academicos"] > base.PRIORIDAD["fred"] > base.PRIORIDAD["fase1"]
    orden = [o for o in c.PRIORIDAD_ORIGENES if o in base.CARGADORES]
    assert sorted(base.CARGADORES, key=base.PRIORIDAD.__getitem__, reverse=True) == orden


def test_t1_unir_alfred_sobre_fase1_y_fechas_de_respaldo(cat, unrate_alfred):
    rng = np.random.default_rng(5)
    # La copia de data/raw (bloque 2: lag_fijo con el último vintage) trae además 1958, que ALFRED no tiene.
    f1 = _fred("fred:UNRATE", cat, pd.date_range("1958-01-01", "1960-02-01", freq="MS"), rng)
    tablas = {"fase1": validar_series(f1), "alfred": unrate_alfred}
    tabla, elegido, conflictos = base.unir(tablas)
    assert elegido["fred:UNRATE"] == "alfred"
    assert conflictos["fred:UNRATE"] == {"cargadores": ["fase1", "alfred"], "elegido": "alfred"}
    assert len(tabla) == len(unrate_alfred)
    assert base.fechas_solo_respaldo(tablas, elegido) == {"fred:UNRATE": 12}       # los 12 meses de 1958


def test_t1_construir_base_desde_cero_con_respaldo(tablas, cat, cat3, unrate_alfred, tmp_path):
    ruta = tmp_path / "series" / "base_series.parquet"
    base.construir_base(tablas=tablas, catalogo=cat, ruta=ruta)
    antes = ruta.read_bytes()
    nuevas = {"fred": tablas["fred"], "alfred": unrate_alfred}
    # Sin desde_cero, las series que ya no se construyen son huérfanas: error.
    with pytest.raises(ErrorContrato, match="huérfanas"):
        base.construir_base(tablas=nuevas, catalogo=cat3, ruta=ruta)
    res, _ = base.construir_base(tablas=nuevas, catalogo=cat3, ruta=ruta, desde_cero=True)
    copia = ruta.with_name(base.RESPALDO)
    assert res.desde_cero and res.copia_previa == str(copia) and copia.read_bytes() == antes
    assert set(almacen.leer_series(ruta=ruta)["serie"]) == {"fred:DBAA", "fred:DAAA", "fred:UNRATE",
                                                             "deriv:DBAA-DAAA"}
    assert "fred:VIXCLS" in res.purgadas and res.bytes_disco == ruta.stat().st_size
    assert res.por_cargador["alfred"] == 1
    # Una segunda reconstrucción no pisa el respaldo del bloque 2, pero copia la base vigente (rota).
    vigente = ruta.read_bytes()
    res2, _ = base.construir_base(tablas={"alfred": unrate_alfred}, catalogo=cat3, ruta=ruta, desde_cero=True)
    assert copia.read_bytes() == antes and set(almacen.leer_series(ruta=ruta)["serie"]) == {"fred:UNRATE"}
    previa = ruta.with_name(f"base_series{base.SUFIJO_PREVIA}")
    assert res2.copia_previa == str(previa) and previa.read_bytes() == vigente


def test_t1_desde_cero_no_escribe_si_falta_un_archivo_de_alfred(tablas, cat, cat3, unrate_alfred, tmp_path,
                                                                 monkeypatch):
    """Regresión: con el crudo de ALFRED incompleto, desde cero sustituía la base por otra sin esas series."""
    ruta = tmp_path / "series" / "base_series.parquet"
    base.construir_base(tablas=tablas, catalogo=cat, ruta=ruta)
    antes = ruta.read_bytes()

    def parcial(**kw):
        kw["informe_alfred"]["fred:PAYEMS"] = {"serie": "fred:PAYEMS", "filas": 0, "avisos": [alfred.AVISO_SIN_ARCHIVO]}
        return {"alfred": unrate_alfred}

    monkeypatch.setattr(base, "tablas_fuentes", parcial)
    with pytest.raises(ErrorContrato, match="sin descarga completa archivada"):
        base.construir_base(catalogo=cat3, ruta=ruta, desde_cero=True)
    assert ruta.read_bytes() == antes and not ruta.with_name(base.RESPALDO).exists()
    # Forzado explícito (--reemplazar): escribe.
    res, _ = base.construir_base(catalogo=cat3, ruta=ruta, desde_cero=True, permitir_perdidas=True)
    assert res.escrita and res.alfred


def test_t1_lectura_filtrada_por_grupos(tablas, cat, tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    monkeypatch.setattr(almacen, "FILAS_POR_GRUPO_SERIES", 50)
    ruta = tmp_path / "b.parquet"
    _, tabla = base.construir_base(tablas=tablas, catalogo=cat, ruta=ruta)
    assert pq.ParquetFile(ruta).metadata.num_row_groups > 10
    vix = almacen.leer_series("fred:VIXCLS", ruta=ruta)
    pd.testing.assert_frame_equal(vix, tabla[tabla["serie"] == "fred:VIXCLS"].reset_index(drop=True))


def test_t1_leer_ultimas_igual_que_as_of(cat3, unrate_alfred, corpus_en_tmp):
    nfci = _alfred("fred:NFCI", cat3, [_vint("2020-01-03", "-0.5", "2020-01-08", "2020-01-14"),
                                      _vint("2020-01-03", "-0.48", "2020-01-15"),
                                      _vint("2020-01-10", "-0.4", "2020-01-15")])
    almacen.escribir_series(pd.concat([unrate_alfred, nfci], ignore_index=True))
    for sid, fechas in (("fred:UNRATE", ["1959-03-01", "1960-03-15", "2020-05-07", "2020-05-08", "2026-01-01"]),
                        ("fred:NFCI", ["2020-01-08", "2020-01-14", "2020-01-15", "2020-02-01"])):
        completa = almacen.leer_series(sid)
        for f in fechas:
            esperado = timegate.as_of_series(completa, sid, f)
            pd.testing.assert_frame_equal(almacen.leer_ultimas(sid, f), esperado)
            # get_serie desde el almacén (sin base) = con la base en memoria.
            if len(esperado):
                desde_disco = tools.get_serie(sid, f)
                en_memoria = tools.get_serie(sid, f, base=completa)
                pd.testing.assert_frame_equal(desde_disco.datos, en_memoria.datos)
                assert desde_disco.avisos == en_memoria.avisos
    assert almacen.leer_ultimas("fred:NO_EXISTE", "2020-01-01").empty
    sin_validar = almacen.leer_series("fred:UNRATE", hasta="2020-05-07", validar=False)
    assert sin_validar["fecha_publicacion"].max() <= pd.Timestamp("2020-05-07")
    assert list(sin_validar.dtypes.astype(str)) == list(almacen.leer_series("fred:UNRATE").dtypes.astype(str))


def test_t1_get_serie_avisos_alfred(unrate_alfred):
    r = tools.get_serie("UNRATE", "2020-05-07", base=unrate_alfred)
    assert r.ultimo()["fecha_referencia"] == "2020-03-01" and r.ultimo()["valor"] == pytest.approx(4.4)
    previas = [a for a in r.avisos if "anteriores a ALFRED" in a]
    assert len(previas) == 1 and "1960-03-15" in previas[0]
    assert not any("lag_fijo: retraso" in a for a in r.avisos)            # las previas no repiten el aviso genérico
    r8 = tools.get_serie("UNRATE", "2020-05-08", base=unrate_alfred, desde="2020-01-01")
    assert r8.ultimo()["fecha_referencia"] == "2020-04-01" and r8.avisos == []    # vintage_exacto: sin avisos
    assert r8.ultimo()["vintage"] == "2020-05-08" and r8.metodo_publicacion == "vintage_exacto"
    # Truncado: la respuesta con la base truncada es la misma.
    trunc = unrate_alfred[unrate_alfred["fecha_publicacion"] <= pd.Timestamp("2020-05-08")]
    pd.testing.assert_frame_equal(tools.get_serie("UNRATE", "2020-05-08", base=trunc).datos,
                                  tools.get_serie("UNRATE", "2020-05-08", base=unrate_alfred).datos)


def test_t1_get_serie_avisos_hueco_reestimada_y_unidad(cat3):
    cpi = _alfred("fred:CPIAUCSL", cat3, [_vint("2025-09-01", "324.4", "2025-10-24"),
                                          _vint("2025-10-01", ".", "2025-12-18"),
                                          _vint("2025-11-01", "325.0", "2025-12-18")])
    assert tools.get_serie("CPIAUCSL", "2025-11-15", base=cpi).ultimo()["fecha_referencia"] == "2025-09-01"
    r = tools.get_serie("CPIAUCSL", "2025-12-18", base=cpi)
    hueco = r.datos[r.datos["fecha_referencia"] == pd.Timestamp("2025-10-01")]
    assert hueco["valor"].isna().all()
    assert any("2025-10-01: valor nulo" in a and "ALFRED sin valor" in a for a in r.avisos)
    # NFCI: re-estimado entero en cada vintage.
    nfci = _alfred("fred:NFCI", cat3, [_vint("2020-01-03", "-0.5", "2020-01-08", "2020-01-14"),
                                      _vint("2020-01-03", "-0.48", "2020-01-15"),
                                      _vint("2020-01-10", "-0.4", "2020-01-15")])
    n = tools.get_serie("NFCI", "2020-01-20", base=nfci)
    assert any("re-estimado entero" in a and "2020-01-15" in a for a in n.avisos)
    gdp = _alfred("fred:GDPC1", cat3, [_vint("2020-01-01", "19000.0", "2020-04-29")])
    assert any("unidad del vintage vigente" in a for a in tools.get_serie("GDPC1", "2020-05-01", base=gdp).avisos)


def test_t1_get_serie_avisos_ediciones_academicas(cat3):
    e = cat3["ff:F3M.Mkt-RF"]
    nota1 = f"{c.NOTA_PRIMERA_EDICION} (2026-07); Ken French recalcula toda la historia con cada base CRSP"
    filas = pd.DataFrame({
        "serie": "ff:F3M.Mkt-RF",
        "fecha_referencia": pd.to_datetime(["2026-04-01", "2026-05-01", "2026-05-01", "2026-06-01"]),
        "fecha_publicacion": pd.to_datetime(["2026-06-15", "2026-07-15", "2026-09-26", "2026-09-26"]),
        "disponible_utc": pd.NaT, "vintage": ["2026-07", "2026-07", "2026-09", "2026-09"],
        "valor": [1.0, 2.0, 2.1, 3.0], "unidad": e["unidad"], "divisa": e["divisa"], "fuente": e["fuente"],
        "metodo_publicacion": "fichero",
        "nota": [nota1, nota1, "edición 2026-09: revisa el valor de la edición 2026-07",
                 "edición 2026-09: periodo nuevo"],
    })
    t = validar_series(filas)
    r = tools.get_serie("ff:F3M.Mkt-RF", "2026-09-30", base=t)
    ed = [a for a in r.avisos if "por edición" in a]
    assert len(ed) == 1 and "2026-07, 2026-09" in ed[0] and "1 periodos con el valor de la primera edición" in ed[0]
    assert not any("último vintage o edición" in a for a in r.avisos)       # la primera edición no es «último vintage»
    antes = tools.get_serie("ff:F3M.Mkt-RF", "2026-08-01", base=t)
    assert antes.ultimo()["valor"] == pytest.approx(2.0)
    assert any("la última visible en la fecha es 2026-07" in a for a in antes.avisos)
    assert c.es_nota_revisable(nota1) and tools._sin_notas_comunes(nota1).startswith("Ken French")


# --- CLI de descargas por tandas ----------------------------------------------------------------
def test_t1_cli_descargar_alfred_reintenta_y_registra(corpus_en_tmp, monkeypatch):
    llamadas = []

    def falso(ids=None, **kw):
        llamadas.append(ids)
        if ids is None:
            return pd.DataFrame({"serie": ["fred:A", "fred:B", "fred:C"], "origen": ["A", "B", "C"],
                                 "estado": ["descargada", "error", "pendiente"], "dia": None, "filas": 0,
                                 "peticiones": 1, "segundos": 1.0, "error": [None, "500 sin_clave", None]})
        return pd.DataFrame({"serie": ids, "origen": ids, "estado": "descargada", "dia": None, "filas": 0,
                             "peticiones": 1, "segundos": 1.0, "error": None})

    monkeypatch.setattr(alfred, "descargar_vintages", falso)
    monkeypatch.setattr(alfred, "cliente_alfred", lambda **kw: type("C", (), {"peticiones": 4})())
    assert cli.main(["descargar-alfred", "--limite-segundos", "60"]) == 2           # queda una pendiente
    assert llamadas == [None, ["fred:B"]]                                            # la fallida, una vez
    log = pd.read_csv(corpus_en_tmp / "corpus" / "mercados_series" / "logs" / "descargas.csv")
    assert log.iloc[-1]["orden"] == "descargar-alfred" and log.iloc[-1]["reintentadas"] == 1
    assert log.iloc[-1]["errores"] == 0 and log.iloc[-1]["pendientes"] == 1


def test_t1_cli_descargar_fred_vivo_codigos(corpus_en_tmp, monkeypatch):
    estado = {"n": 0}

    def falso(ids=None, **kw):
        estado["n"] += 1
        return {"hechas": [], "cache": [], "errores": {"fred:X": "500"}, "pendientes": [], "peticiones": 1,
                "segundos": 1.0, "detalle": []}

    monkeypatch.setattr(fred, "descargar_fred_vivo", falso)
    monkeypatch.setattr(fred, "ClienteFred", lambda **kw: type("C", (), {"peticiones": 2})())
    assert cli.main(["descargar-fred-vivo", "--ids", "fred:X"]) == 1                 # sigue con error tras reintentar
    assert estado["n"] == 2


def test_t1_cli_construir_desde_cero(tablas, corpus_en_tmp, monkeypatch, capsys):
    monkeypatch.setattr(base, "tablas_fuentes", lambda **kw: tablas)
    assert cli.main(["construir-base", "--sin-red"]) == 0
    assert cli.main(["construir-base", "--sin-red", "--desde-cero"]) == 0
    carpeta = corpus_en_tmp / "corpus" / "mercados_series" / "series"
    assert (carpeta / base.RESPALDO).exists() and (carpeta / "base_series.parquet").exists()
    assert '"desde_cero": true' in capsys.readouterr().out


# --- Base real (local) --------------------------------------------------------------------------
# Leen el almacén real (<corpus>/mercados_series/series/base_series.parquet) por series, sin cargar
# los 8,6 millones de filas salvo en la validación completa. Solo comparan fechas y valores entre sí.
@pytest.fixture(scope="module")
def ruta_real():
    ruta = almacen.ruta_series()
    if not ruta.exists():
        pytest.skip(f"falta la base real {ruta} (python -m ingesta.mercados_series construir-base --sin-red)")
    if "fred:PAYEMS" not in set(almacen.leer_ultimas("fred:PAYEMS", "2026-10-13")["serie"]):
        pytest.skip("base guardada anterior al bloque 3 (sin PAYEMS de ALFRED): construir-base --desde-cero")
    return ruta


@pytest.mark.datos
def test_t1_base_real_valida_contra_catalogo(ruta_real):
    cat = c.cargar_catalogo()                                          # el catálogo real (bloque 3)
    real = almacen.leer_series(validar=False)
    c.validar_contra_catalogo(real, base.catalogo_efectivo(cat))
    ids = set(real["serie"])
    assert not any(cat[s]["estado"] == "excluida_ajustada" for s in ids)
    cargables = {s for s, e in base.catalogo_efectivo(cat).items() if e["estado"] in c.ESTADOS_CARGABLES}
    assert ids == cargables
    assert (real["fecha_publicacion"] >= real["fecha_referencia"]).all()
    # Las 161 de ALFRED solo con vintage_exacto o su regla previa; las de la fuente viva de FRED, sin data/raw.
    alf = real[real["serie"].isin(alfred.ids_alfred(cat))]
    assert alf["serie"].nunique() == 161 and set(alf["metodo_publicacion"]) == {"vintage_exacto", "lag_fijo"}
    assert (alf["vintage"] != "unico").all()


@pytest.mark.datos
def test_t1_base_real_consultas(ruta_real):
    # VIX (Cboe 16:15 NY → visible en la decisión de t+1): el 2008-10-10 se ve el cierre del 09.
    vix = tools.get_serie("VIX", "2008-10-10")
    assert vix.serie == "fred:VIXCLS" and vix.ultimo()["fecha_referencia"] == "2008-10-09"
    fila_10 = almacen.leer_series("fred:VIXCLS")
    fila_10 = fila_10[fila_10["fecha_referencia"] == pd.Timestamp("2008-10-10")]
    assert fila_10["fecha_publicacion"].iloc[0] == pd.Timestamp("2008-10-11")   # t+1 natural (decisión diaria)
    # DGS10 (H.15, hábil siguiente a las 16:15), ahora de la fuente viva de FRED.
    dgs = tools.get_serie("fred:DGS10", "2020-03-16")
    assert dgs.ultimo()["fecha_referencia"] == "2020-03-13" and dgs.ultimo()["fecha_publicacion"] == "2020-03-16"
    # Alias de la Fase 1 y truncado: la respuesta con la base truncada es la misma.
    completa = almacen.leer_series("yf:^GSPC")
    sp = tools.get_serie("SP500", "2008-09-15", base=completa)
    assert sp.serie == "yf:^GSPC" and sp.ultimo()["fecha_referencia"] == "2008-09-15"   # XNYS 16:00: mismo día
    truncada = completa[completa["fecha_publicacion"] <= pd.Timestamp("2008-09-15")]
    pd.testing.assert_frame_equal(tools.get_serie("SP500", "2008-09-15", base=truncada).datos, sp.datos)
    with pytest.raises(KeyError, match="get_precios"):
        tools.get_serie("SP500_CAPWEIGHT_ETF", "2020-03-16")


@pytest.mark.datos
def test_t1_base_real_payems_primera_estimacion(ruta_real):
    # Noviembre de 2008 tal como se publicó el viernes 2008-12-05 (primera estimación), no el dato revisado.
    r = tools.get_serie("PAYEMS", "2008-12-10")
    nov = r.datos.set_index("fecha_referencia").loc[pd.Timestamp("2008-11-01")]
    assert nov["fecha_publicacion"] == pd.Timestamp("2008-12-05") and nov["vintage"] == "2008-12-05"
    assert nov["metodo_publicacion"] == "vintage_exacto" and r.ultimo()["fecha_referencia"] == "2008-11-01"
    hoy = tools.get_serie("PAYEMS", "2026-10-13").datos.set_index("fecha_referencia").loc[pd.Timestamp("2008-11-01")]
    assert hoy["valor"] != nov["valor"] and hoy["vintage"] > "2008-12-05"
    assert not any("revisable" in a or "lag_fijo" in a for a in r.avisos)        # vintage exacto: sin avisos
    assert tools.get_serie("PAYEMS", "2008-12-04").ultimo()["fecha_referencia"] == "2008-10-01"


@pytest.mark.datos
def test_t1_base_real_unrate_abril_2020(ruta_real):
    assert tools.get_serie("UNRATE", "2020-05-07").ultimo()["fecha_referencia"] == "2020-03-01"
    u = tools.get_serie("UNRATE", "2020-05-08").ultimo()
    assert u["fecha_referencia"] == "2020-04-01" and u["fecha_publicacion"] == "2020-05-08"
    assert u["vintage"] == "2020-05-08" and u["metodo_publicacion"] == "vintage_exacto"


@pytest.mark.datos
def test_t1_base_real_cpi_hueco_octubre_2025(ruta_real):
    r = tools.get_serie("CPIAUCSL", "2025-11-15")
    assert r.ultimo()["fecha_referencia"] == "2025-09-01"
    assert not (r.datos["fecha_referencia"] == pd.Timestamp("2025-10-01")).any()     # octubre aún no existe
    despues = tools.get_serie("CPIAUCSL", "2026-01-15")
    octubre = despues.datos[despues.datos["fecha_referencia"] == pd.Timestamp("2025-10-01")]
    assert len(octubre) == 1 and octubre["valor"].isna().all()                        # hueco permanente del cierre
    assert any("2025-10-01: valor nulo" in a and "ALFRED sin valor" in a for a in despues.avisos)


@pytest.mark.datos
def test_t1_base_real_cierre_del_gobierno_2013(ruta_real):
    # El empleo de septiembre de 2013 se retrasó al 2013-10-22 por el cierre del gobierno (1-16 de octubre).
    for sid in ("PAYEMS", "UNRATE"):
        assert tools.get_serie(sid, "2013-10-21").ultimo()["fecha_referencia"] == "2013-08-01"
        u = tools.get_serie(sid, "2013-10-22").ultimo()
        assert u["fecha_referencia"] == "2013-09-01" and u["fecha_publicacion"] == "2013-10-22"


@pytest.mark.datos
def test_t1_base_real_previas_reestimadas_y_ediciones(ruta_real):
    # UNRATE de 1950 en 1955: anterior a ALFRED (primer vintage 1960-03-15), con la regla previa y su aviso.
    u = tools.get_serie("UNRATE", "1955-01-10")
    assert u.datos["metodo_publicacion"].eq("lag_fijo").all()
    assert any("anteriores a ALFRED (primer vintage 1960-03-15)" in a for a in u.avisos)
    nfci = tools.get_serie("NFCI", "2020-03-25")
    assert any("re-estimado entero" in a for a in nfci.avisos) and nfci.ultimo()["fecha_referencia"] == "2020-03-20"
    mom = tools.get_serie("ff:MOMD.Mom", "2026-10-13")
    assert any("por edición" in a for a in mom.avisos)


@pytest.mark.datos
def test_t1_base_real_derivadas_y_duplicados(ruta_real):
    spread = tools.get_serie("deriv:DBAA-DAAA", "2026-10-12").datos
    assert spread["fecha_referencia"].max() > pd.Timestamp("2026-07-20")         # tramo de FRED tras data/raw
    sofr_effr = almacen.leer_series("deriv:SOFR-EFFR")
    assert len(sofr_effr) > 2000 and sofr_effr["fecha_referencia"].min() == pd.Timestamp("2018-04-03")
    # Alias de la Fase 1 BAA10Y → derivada propia (D+1); la de FRED queda como duplicada con su id.
    assert tools.get_serie("BAA10Y", "2026-10-12").serie == "deriv:DBAA-DGS10"
    ids = {"fred:EFFR", "fred:DFF", "datahub:US10Y_MENSUAL", "fred:GS10", "fred:BAA10Y", "deriv:DGS10-DFF"}
    assert set(almacen.leer_series(sorted(ids))["serie"]) == ids


@pytest.mark.datos
def test_t1_base_real_h6_semanal_dia_siguiente(ruta_real):
    """Regresión: la vintage del jueves 2020-02-27 de M2SL (H.6 a las 16:30) no se ve ese día."""
    tabla = almacen.leer_series("fred:M2SL", validar=False)
    if not (tabla["vintage"] == "2020-02-27").any():
        pytest.skip("la base no tiene la vintage 2020-02-27 de M2SL")
    filas = tabla[tabla["vintage"] == "2020-02-27"]
    assert (filas["fecha_publicacion"] == pd.Timestamp("2020-02-28")).all()
    assert "2020-02-27" not in set(timegate.as_of_series(tabla, "fred:M2SL", "2020-02-27")["vintage"])
    assert "2020-02-27" in set(timegate.as_of_series(tabla, "fred:M2SL", "2020-02-28")["vintage"])
    con_previa = [s for s, e in c.cargar_catalogo().items() if e.get("regla_previa_alfred")]
    previas = almacen.leer_series(con_previa, validar=False)
    previas = previas[previas["nota"].fillna("").str.contains(c.NOTA_PREVIA_ALFRED, regex=False)]
    citado = pd.to_datetime(previas["nota"].str.extract(r"\((\d{4}-\d{2}-\d{2})\)")[0])
    assert (previas["fecha_publicacion"] < citado).all()

