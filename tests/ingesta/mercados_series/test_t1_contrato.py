"""Tests del contrato del Tema 1 (``ingesta.mercados_series.contrato``): solo pandas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ingesta.mercados_series.contrato import (
    CLAVE_PRECIOS,
    CLAVE_SERIES,
    COLUMNAS_PRECIOS,
    COLUMNAS_PROHIBIDAS,
    COLUMNAS_SERIES,
    METODOS_PUBLICACION,
    ErrorContrato,
    descartar_velas_incompletas,
    es_columna_prohibida,
    fecha_publicacion_derivada,
    primera_decision_vec,
    vacio_precios,
    vacio_series,
    validar_precios,
    validar_series,
)


def _series(n: int = 3, **cambios) -> pd.DataFrame:
    """Serie mensual válida: publicada el día 5 del mes siguiente a las 08:30 Nueva York."""
    ref = pd.date_range("2024-01-01", periods=n, freq="MS")
    pub = ref + pd.offsets.MonthBegin(1) + pd.Timedelta(days=4)
    disp = (pub + pd.Timedelta(hours=8, minutes=30)).tz_localize("America/New_York").tz_convert("UTC")
    base = {
        "serie": ["fred:TEST"] * n,
        "fecha_referencia": ref,
        "fecha_publicacion": pub,
        "disponible_utc": disp,
        "vintage": pub.strftime("%Y-%m-%d"),
        "valor": np.arange(n, dtype=float) + 1.0,
        "unidad": ["porcentaje"] * n,
        "divisa": [None] * n,
        "fuente": ["fred"] * n,
        "metodo_publicacion": ["vintage_exacto"] * n,
        "nota": [None] * n,
    }
    base.update(cambios)
    return pd.DataFrame(base)


def _precios(n: int = 3, **cambios) -> pd.DataFrame:
    """Precios XNYS válidos: visibles a las 16:30 Nueva York del mismo día."""
    ses = pd.bdate_range("2024-01-02", periods=n)
    disp = (ses + pd.Timedelta(hours=16, minutes=30)).tz_localize("America/New_York").tz_convert("UTC")
    base = {
        "ticker": ["ZZ"] * n, "sesion": ses, "mercado": ["XNYS"] * n,
        "open": [10.0] * n, "high": [11.0] * n, "low": [9.0] * n, "close": [10.5] * n,
        "volume": [1000.0] * n, "dividendo": [0.0] * n, "split": [1.0] * n, "divisa": ["USD"] * n,
        "fecha_publicacion": ses, "disponible_utc": disp, "fuente": ["prueba"] * n,
        "descargado": [pd.Timestamp("2024-02-01", tz="UTC")] * n,
    }
    base.update(cambios)
    return pd.DataFrame(base)


def _problema(excinfo) -> str:
    return " | ".join(excinfo.value.problemas)


# --- Constantes y regla del TimeGate ----------------------------------------------------------
def test_t1_constantes_del_contrato():
    assert list(COLUMNAS_SERIES)[:3] == ["serie", "fecha_referencia", "fecha_publicacion"]
    assert set(CLAVE_SERIES) <= set(COLUMNAS_SERIES) and set(CLAVE_PRECIOS) <= set(COLUMNAS_PRECIOS)
    assert set(METODOS_PUBLICACION) == {"vintage_exacto", "calendario", "lag_fijo", "cierre_mercado",
                                        "derivada", "fichero"}
    assert not any(es_columna_prohibida(c) for c in [*COLUMNAS_SERIES, *COLUMNAS_PRECIOS])
    assert all(es_columna_prohibida(c) for c in COLUMNAS_PROHIBIDAS)


@pytest.mark.parametrize("instante, esperada", [
    ("2024-01-15 21:30", "2024-01-15"),   # 16:30 EST: el borde es visible ese día
    ("2024-01-15 21:31", "2024-01-16"),   # 16:31 EST: al día siguiente
    ("2024-07-15 20:30", "2024-07-15"),   # 16:30 EDT
    ("2024-07-15 20:31", "2024-07-16"),
    ("2024-03-10 03:00", "2024-03-10"),   # 22:00 EST del sábado 9 → la decisión del domingo 10
    ("2024-03-10 21:00", "2024-03-11"),   # 17:00 EDT (ya con DST) → lunes
    ("2024-03-15 06:30", "2024-03-15"),   # 02:30 EDT (cierre de Tokio + 30 min)
])
def test_t1_primera_decision_vec_borde_y_dst(instante, esperada):
    s = pd.Series([pd.Timestamp(instante, tz="UTC")])
    out = primera_decision_vec(s)
    assert str(out.dtype) == "datetime64[ns]"
    assert out.iloc[0] == pd.Timestamp(esperada)


def test_t1_primera_decision_vec_nulos_y_naive():
    s = pd.Series([pd.Timestamp("2024-01-15 12:00", tz="UTC"), pd.NaT], dtype="datetime64[ns, UTC]")
    out = primera_decision_vec(s)
    assert out.iloc[0] == pd.Timestamp("2024-01-15") and pd.isna(out.iloc[1])
    with pytest.raises(TypeError):
        primera_decision_vec(pd.Series([pd.Timestamp("2024-01-15 12:00")]))


# --- Normalización ----------------------------------------------------------------------------
def test_t1_series_normaliza_dtypes_y_orden():
    df = _series()
    desordenado = df[list(reversed(df.columns))].iloc[::-1].copy()
    desordenado["fecha_referencia"] = desordenado["fecha_referencia"].dt.strftime("%Y-%m-%d")
    desordenado["fecha_publicacion"] = desordenado["fecha_publicacion"].dt.strftime("%Y-%m-%d")
    desordenado["disponible_utc"] = desordenado["disponible_utc"].dt.tz_convert("Europe/Madrid")
    desordenado["valor"] = desordenado["valor"].astype(int)
    out = validar_series(desordenado)
    assert list(out.columns) == list(COLUMNAS_SERIES)
    assert {c: str(t) for c, t in out.dtypes.items()} == COLUMNAS_SERIES
    assert out["fecha_referencia"].is_monotonic_increasing and list(out.index) == [0, 1, 2]
    assert str(out["disponible_utc"].dt.tz) == "UTC"
    pd.testing.assert_frame_equal(out, validar_series(df))


def test_t1_precios_normaliza_dtypes_y_no_modifica_entrada():
    df = _precios()
    df["volume"] = df["volume"].astype(int)
    copia = df.copy()
    out = validar_precios(df.iloc[::-1])
    assert list(out.columns) == list(COLUMNAS_PRECIOS)
    assert {c: str(t) for c, t in out.dtypes.items()} == COLUMNAS_PRECIOS
    assert out["sesion"].is_monotonic_increasing
    pd.testing.assert_frame_equal(df, copia)


def test_t1_disponible_utc_todo_nulo_es_valido():
    df = _series(disponible_utc=[None, None, None])
    out = validar_series(df)
    assert str(out["disponible_utc"].dtype) == "datetime64[ns, UTC]" and out["disponible_utc"].isna().all()


def test_t1_tablas_vacias():
    for vacio, cols, validar in ((vacio_series(), COLUMNAS_SERIES, validar_series),
                                 (vacio_precios(), COLUMNAS_PRECIOS, validar_precios)):
        assert len(vacio) == 0 and {c: str(t) for c, t in vacio.dtypes.items()} == cols
        assert len(validar(vacio)) == 0


def test_t1_texto_no_str_falla():
    with pytest.raises(ErrorContrato, match="'unidad' debe ser texto"):
        validar_series(_series(unidad=[1, 2, 3]))


# --- Estructura --------------------------------------------------------------------------------
def test_t1_faltan_columnas():
    with pytest.raises(ErrorContrato, match=r"faltan columnas: \['vintage'\]"):
        validar_series(_series().drop(columns="vintage"))


def test_t1_columnas_fuera_del_contrato():
    with pytest.raises(ErrorContrato, match="fuera del contrato"):
        validar_precios(_precios().assign(extra=1))


@pytest.mark.parametrize("columna", ["adj_close", "Adj Close", "ADJ_CLOSE", "close_adj", "AdjClose", "adj_factor"])
def test_t1_columnas_prohibidas(columna):
    with pytest.raises(ErrorContrato, match="columnas prohibidas") as e:
        validar_precios(_precios().assign(**{columna: 1.0}))
    assert e.value.tabla == "precios"
    with pytest.raises(ErrorContrato, match="columnas prohibidas"):
        validar_series(_series().assign(**{columna: 1.0}))


# --- Series: contenido ---------------------------------------------------------------------------
def test_t1_series_valor_nulo_sin_nota_falla_y_con_nota_pasa():
    with pytest.raises(ErrorContrato, match="valores nulos sin nota"):
        validar_series(_series(valor=[1.0, np.nan, 3.0]))
    out = validar_series(_series(valor=[1.0, np.nan, 3.0], nota=[None, "hueco permanente", None]))
    assert out["valor"].isna().sum() == 1


def test_t1_series_nulos_en_fecha_publicacion():
    df = _series(disponible_utc=[None] * 3)
    df.loc[1, "fecha_publicacion"] = pd.NaT
    with pytest.raises(ErrorContrato, match="nulos en 'fecha_publicacion'"):
        validar_series(df)


def test_t1_series_metodo_invalido():
    with pytest.raises(ErrorContrato, match="metodo_publicacion inválido"):
        validar_series(_series(metodo_publicacion=["vintage_exacto", "a_ojo", "vintage_exacto"]))


def test_t1_series_duplicados_de_clave():
    df = _series()
    df = pd.concat([df, df.iloc[[0]].assign(valor=99.0)], ignore_index=True)
    with pytest.raises(ErrorContrato, match="clave .* repetida"):
        validar_series(df)


def test_t1_series_dos_vintages_del_mismo_periodo_pasan():
    df = _series(n=1)
    rev = df.assign(fecha_publicacion=pd.Timestamp("2024-03-08"), vintage="2024-03-08", valor=1.1,
                    disponible_utc=pd.Timestamp("2024-03-08 13:30", tz="UTC"))
    out = validar_series(pd.concat([rev, df], ignore_index=True))
    assert out["vintage"].tolist() == ["2024-02-05", "2024-03-08"]


@pytest.mark.parametrize("vintage", ["2024/02/05", "v1", "", "2024-2-5"])
def test_t1_series_vintage_con_formato_invalido(vintage):
    with pytest.raises(ErrorContrato, match="'vintage' con formato inválido"):
        validar_series(_series(n=1, vintage=[vintage]))


def test_t1_series_mezcla_formatos_de_vintage():
    with pytest.raises(ErrorContrato, match="mezclan formatos de vintage"):
        validar_series(_series(vintage=["2024-02-05", "2024-03", "unico"], disponible_utc=[None] * 3))


def test_t1_series_unidad_debe_ser_constante():
    with pytest.raises(ErrorContrato, match="'unidad' cambia"):
        validar_series(_series(unidad=["porcentaje", "puntos", "porcentaje"]))


def test_t1_series_fecha_con_hora_falla():
    df = _series()
    df["fecha_referencia"] = df["fecha_referencia"] + pd.Timedelta(hours=1)
    with pytest.raises(ErrorContrato, match="medianoche"):
        validar_series(df)


def test_t1_series_disponible_sin_huso_falla():
    df = _series()
    df["disponible_utc"] = df["disponible_utc"].dt.tz_localize(None)
    with pytest.raises(ErrorContrato, match="debe llevar huso"):
        validar_series(df)


def test_t1_series_fecha_publicacion_incoherente_con_disponible():
    df = _series()
    df.loc[0, "fecha_publicacion"] = df.loc[0, "fecha_publicacion"] - pd.Timedelta(days=1)  # adelanta el dato
    with pytest.raises(ErrorContrato, match="incoherente con disponible_utc"):
        validar_series(df)
    assert len(validar_series(df, comprobar_disponible=False)) == 3


def test_t1_series_valor_infinito_falla():
    with pytest.raises(ErrorContrato, match="infinitos"):
        validar_series(_series(valor=[1.0, np.inf, 2.0]))


def test_t1_error_contrato_acumula_problemas():
    df = _series(metodo_publicacion=["x"] * 3, valor=[np.nan] * 3)
    with pytest.raises(ErrorContrato) as e:
        validar_series(df)
    assert len(e.value.problemas) >= 2 and isinstance(e.value, ValueError)


# --- Precios: contenido -------------------------------------------------------------------------
@pytest.mark.parametrize("cambio, mensaje", [
    ({"split": [1.0, 0.0, 1.0]}, "split <= 0"),
    ({"split": [1.0, -2.0, 1.0]}, "split <= 0"),
    ({"dividendo": [0.0, -0.1, 0.0]}, "dividendo < 0"),
    ({"dividendo": [0.0, np.nan, 0.0]}, "nulos en 'dividendo'"),
    ({"close": [10.0, np.nan, 10.0]}, "nulos en 'close'"),
    ({"mercado": ["XNYS", "XLON", "XNYS"]}, "'mercado' cambia"),
    ({"divisa": ["USD", "US Dollar", "USD"]}, "'divisa' con formato inválido"),
])
def test_t1_precios_contenido_invalido(cambio, mensaje):
    with pytest.raises(ErrorContrato, match=mensaje):
        validar_precios(_precios(**cambio))


def test_t1_precios_fecha_publicacion_anterior_a_sesion():
    df = _precios()
    df["fecha_publicacion"] = df["sesion"] - pd.Timedelta(days=1)
    with pytest.raises(ErrorContrato, match="anterior a la sesión"):
        validar_precios(df, comprobar_disponible=False)


def test_t1_precios_vela_incompleta():
    df = _precios(descargado=[pd.Timestamp("2024-01-03 12:00", tz="UTC")] * 3)
    with pytest.raises(ErrorContrato, match="velas incompletas"):
        validar_precios(df)


def test_t1_precios_duplicados_de_clave():
    df = _precios()
    with pytest.raises(ErrorContrato, match=r"\('ticker', 'sesion'\) repetida"):
        validar_precios(pd.concat([df, df.iloc[[1]]], ignore_index=True))


def test_t1_precios_negativos_y_subunidades_permitidos():
    out = validar_precios(_precios(close=[-37.63, 10.0, 10.5], divisa=["GBp"] * 3))
    assert out["close"].iloc[0] == -37.63 and set(out["divisa"]) == {"GBp"}


def test_t1_precios_ohlv_nulos_permitidos():
    out = validar_precios(_precios(open=[np.nan] * 3, volume=[np.nan] * 3))
    assert out["open"].isna().all()


def test_t1_series_fecha_publicacion_anterior_a_fecha_referencia():
    """Regresión: una fila de junio publicada en enero pasaba y get_serie la devolvía en enero."""
    df = _series(n=1, fecha_referencia=[pd.Timestamp("2024-06-01")], fecha_publicacion=[pd.Timestamp("2024-01-01")],
                 disponible_utc=[None], vintage=["unico"], metodo_publicacion=["lag_fijo"])
    with pytest.raises(ErrorContrato, match="anterior a fecha_referencia"):
        validar_series(df)
    with pytest.raises(ErrorContrato, match="anterior a fecha_referencia"):
        validar_series(df, comprobar_disponible=False)
    # el mismo día de referencia sí vale (cierre diario visible esa tarde)
    assert len(validar_series(df.assign(fecha_publicacion=pd.Timestamp("2024-06-01")))) == 1


@pytest.mark.parametrize("mercado, hora_utc", [
    ("CRYPTO", "2024-01-09 20:00"),     # la vela diaria UTC cierra a medianoche: visible en t+1
    ("CME", "2024-01-09 20:30"),        # 17:00 NY + 30 min
    ("CBOE_IDX", "2024-01-09 20:30"),   # 16:15 NY + 30 min = 16:45 > 16:30
    ("XLON", "2024-01-09 16:00"),       # 16:30 Londres + 30 min = 17:00 UTC
])
def test_t1_precios_disponible_anterior_al_cierre_del_mercado(mercado, hora_utc):
    """Regresión: una vela con disponible_utc antes del cierre de su mercado pasaba la validación."""
    disp = pd.Timestamp(hora_utc, tz="UTC")
    pub = primera_decision_vec(pd.Series([disp])).iloc[0]
    df = _precios(n=1, sesion=[pd.Timestamp("2024-01-09")], mercado=[mercado], disponible_utc=[disp],
                  fecha_publicacion=[pub])
    with pytest.raises(ErrorContrato, match="antes del cierre de su mercado"):
        validar_precios(df)
    assert len(validar_precios(df, comprobar_mercado=False)) == 1


def test_t1_precios_mercado_desconocido_y_disponible_conservador():
    with pytest.raises(ErrorContrato, match="mercados.yaml"):
        validar_precios(_precios(mercado=["XXXX"] * 3))
    # más tarde que el cierre + 30 min es conservador: vale
    tarde = _precios()
    tarde["disponible_utc"] = tarde["disponible_utc"] + pd.Timedelta(days=1)
    tarde["fecha_publicacion"] = tarde["sesion"] + pd.Timedelta(days=1)
    assert len(validar_precios(tarde)) == 3


@pytest.mark.parametrize("vintage", ["unico\n", "2024-02-05\n", " unico"])
def test_t1_series_vintage_con_salto_de_linea_falla(vintage):
    """Regresión: 'unico\\n' pasaba (``$`` encaja antes de un salto final) y creaba claves gemelas."""
    with pytest.raises(ErrorContrato, match="'vintage' con formato inválido"):
        validar_series(_series(n=1, vintage=[vintage], disponible_utc=[None]))


@pytest.mark.parametrize("divisa, valida", [
    ("USD", True), ("GBp", True), ("GBX", True), ("ZAc", True), ("ILA", True),
    ("usd", False), ("Usd", False), ("gbp", False), ("USD\n", False), ("US", False),
])
def test_t1_divisa_mayusculas_y_subunidades(divisa, valida):
    """Regresión: 'usd' pasaba y luego a_usd no lo reconocía como USD (NaN)."""
    df = _precios(divisa=[divisa] * 3)
    if valida:
        assert set(validar_precios(df)["divisa"]) == {divisa}
    else:
        with pytest.raises(ErrorContrato, match="'divisa' con formato inválido"):
            validar_precios(df)


def test_t1_descartar_velas_incompletas():
    df = _precios()
    df.loc[2, "descargado"] = pd.Timestamp("2024-01-04 12:00", tz="UTC")   # antes del cierre del 4
    out = descartar_velas_incompletas(df)
    assert list(out.index) == [0, 1] and len(validar_precios(out)) == 2
    pd.testing.assert_frame_equal(df, _precios().assign(descargado=df["descargado"]))   # no modifica


def test_t1_fecha_publicacion_derivada():
    a = validar_series(_series(n=3))
    b = a.assign(serie="fred:OTRA", fecha_publicacion=a["fecha_publicacion"] + pd.Timedelta(days=2),
                 disponible_utc=a["disponible_utc"] + pd.Timedelta(days=2),
                 vintage=(a["fecha_publicacion"] + pd.Timedelta(days=2)).dt.strftime("%Y-%m-%d"))
    b = b.iloc[1:]   # le falta el primer periodo
    rev = a.iloc[[1]].assign(fecha_publicacion=pd.Timestamp("2024-06-03"), vintage="2024-06-03",
                             disponible_utc=pd.Timestamp("2024-06-03 12:30", tz="UTC"), valor=9.0)
    base = validar_series(pd.concat([a, b, rev], ignore_index=True))
    d = fecha_publicacion_derivada(base, ["fred:TEST", "fred:OTRA"])
    assert d["fecha_referencia"].tolist() == list(b["fecha_referencia"])
    assert d["fecha_publicacion"].tolist() == list(b["fecha_publicacion"])   # la máxima; la revisión no cuenta
    assert (d["disponible_utc"] == b["disponible_utc"].to_numpy()).all()
    with pytest.raises(KeyError):
        fecha_publicacion_derivada(base, ["fred:TEST", "fred:NO"])


# --- Rendimiento ------------------------------------------------------------------------------
def test_t1_validar_series_vectorizado():
    """200 000 filas se validan en poco tiempo (antes, bucles Python por fila en textos y patrones)."""
    import time

    n = 200_000
    i = np.arange(n)
    ref = pd.Timestamp("1990-01-01") + pd.to_timedelta(i % 10_000, unit="D")
    df = pd.DataFrame({"serie": [f"fred:X{k}" for k in i // 10_000], "fecha_referencia": ref, "fecha_publicacion": ref,
                       "disponible_utc": pd.NaT, "vintage": "unico", "valor": 1.0, "unidad": "u",
                       "divisa": "USD", "fuente": "f", "metodo_publicacion": "calendario", "nota": None})
    inicio = time.perf_counter()
    validar_series(df)
    assert time.perf_counter() - inicio < 5.0


# --- pandera (opcional) -------------------------------------------------------------------------
def test_t1_esquemas_pandera():
    pytest.importorskip("pandera.pandas")
    from ingesta.mercados_series.contrato import esquema_pandera_precios, esquema_pandera_series

    esquema_pandera_series().validate(validar_series(_series()))
    esquema_pandera_precios().validate(validar_precios(_precios()))
    malo = validar_series(_series(valor=[1.0, np.nan, 3.0], nota=["x", "y", "z"])).assign(nota=None)
    with pytest.raises(Exception):
        esquema_pandera_series().validate(malo)
