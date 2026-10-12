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
from ingesta.mercados_series.fuentes import fred

HABILES = pd.bdate_range("2024-01-02", "2024-03-29")


@pytest.fixture(scope="module")
def cat():
    return c.cargar_catalogo()


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
    ef = base.catalogo_efectivo(cat)
    assert cat["fred:WEI"]["estado"] == "pendiente_alfred" and ef["fred:WEI"]["estado"] == "activa"
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


# --- Base real (local) --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def base_real():
    ruta = almacen.ruta_series()
    if not ruta.exists():
        pytest.skip(f"falta la base real {ruta} (python -m ingesta.mercados_series construir-base --sin-red)")
    return almacen.leer_series()


@pytest.mark.datos
def test_t1_base_real_valida_contra_catalogo(base_real, cat):
    c.validar_contra_catalogo(base_real, base.catalogo_efectivo(cat))
    ids = set(base_real["serie"])
    assert not any(cat[s]["estado"] == "excluida_ajustada" for s in ids)
    cargables = {s for s, e in base.catalogo_efectivo(cat).items() if e["estado"] in c.ESTADOS_CARGABLES}
    assert ids == cargables
    assert (base_real["fecha_publicacion"] >= base_real["fecha_referencia"]).all()


@pytest.mark.datos
def test_t1_base_real_consultas(base_real, cat):
    # VIX (Cboe 16:15 NY → visible en la decisión de t+1): el 2008-10-10 se ve el cierre del 09.
    vix = tools.get_serie("VIX", "2008-10-10", base=base_real)
    assert vix.serie == "fred:VIXCLS" and vix.ultimo()["fecha_referencia"] == "2008-10-09"
    fila_10 = base_real[(base_real["serie"] == "fred:VIXCLS") & (base_real["fecha_referencia"] == "2008-10-10")]
    assert fila_10["fecha_publicacion"].iloc[0] == pd.Timestamp("2008-10-11")   # t+1 natural (decisión diaria)
    # DGS10 (H.15, hábil siguiente a las 16:15): el lunes 2020-03-16 se ve el viernes 13.
    dgs = tools.get_serie("fred:DGS10", "2020-03-16", base=base_real)
    assert dgs.ultimo()["fecha_referencia"] == "2020-03-13" and dgs.ultimo()["fecha_publicacion"] == "2020-03-16"
    # CPI (lag_fijo de lags.yaml): el último visible es el que cumple la regla.
    e = cat["fred:CPIAUCSL"]
    cpi = tools.get_serie("CPIAUCSL", "2020-03-16", base=base_real)
    ultimo = pd.Timestamp(cpi.ultimo()["fecha_referencia"])
    regla = c.fecha_publicacion_lag([ultimo, ultimo + pd.offsets.MonthBegin(1)], e["lag"], e["fin_periodo"])
    assert regla.iloc[0] <= pd.Timestamp("2020-03-16") < regla.iloc[1]
    assert ultimo == pd.Timestamp("2020-01-01")                             # febrero sale el 2020-04-01 (1M1D)
    assert any("revisable" in a for a in cpi.avisos) and any("lag_fijo" in a for a in cpi.avisos)
    # Alias de la Fase 1 y truncado: la respuesta con la base truncada es la misma.
    sp = tools.get_serie("SP500", "2008-09-15", base=base_real)
    assert sp.serie == "yf:^GSPC" and sp.ultimo()["fecha_referencia"] == "2008-09-15"   # XNYS 16:00: mismo día
    truncada = base_real[base_real["fecha_publicacion"] <= pd.Timestamp("2008-09-15")]
    pd.testing.assert_frame_equal(tools.get_serie("SP500", "2008-09-15", base=truncada).datos, sp.datos)
    # ETF de la Fase 1 excluida → KeyError que remite a get_precios.
    with pytest.raises(KeyError, match="get_precios"):
        tools.get_serie("SP500_CAPWEIGHT_ETF", "2020-03-16", base=base_real)


@pytest.mark.datos
def test_t1_base_real_derivadas_y_duplicados(base_real):
    spread = timegate.as_of_series(base_real, "deriv:DBAA-DAAA", "2026-10-12")
    assert spread["fecha_referencia"].max() > pd.Timestamp("2026-07-20")   # tramo de FRED tras data/raw
    sofr_effr = base_real[base_real["serie"] == "deriv:SOFR-EFFR"]
    assert len(sofr_effr) > 2000 and sofr_effr["fecha_referencia"].min() == pd.Timestamp("2018-04-03")
    # Duplicados con su propio id: EFFR (duplicado de DFF) y la copia de GS10 de datahub.
    assert {"fred:EFFR", "fred:DFF", "datahub:US10Y_MENSUAL", "fred:GS10"} <= set(base_real["serie"])
