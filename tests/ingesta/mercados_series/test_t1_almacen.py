"""Tests del almacén del Tema 1 (``ingesta.mercados_series.almacen``): parquet en ``tmp_path``.

Nunca escriben en el corpus real: las rutas por defecto se redirigen a ``tmp_path``.
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ingesta.mercados_series import almacen
from ingesta.mercados_series import sinteticos as s
from ingesta.mercados_series.almacen import (
    CARACTERES_SEGUROS,
    escribir_precios,
    escribir_series,
    leer_precios,
    leer_series,
    nombre_fichero_ticker,
    ticker_de_fichero,
    tickers_guardados,
)
from ingesta.mercados_series.contrato import (
    COLUMNAS_PRECIOS,
    COLUMNAS_SERIES,
    ErrorContrato,
    validar_series,
)


@pytest.fixture(autouse=True)
def corpus_en_tmp(tmp_path, monkeypatch):
    """Las rutas por defecto del almacén apuntan a ``tmp_path`` (nunca al corpus real)."""
    def corpus_dir(tema=None):
        p = tmp_path / "corpus" / tema if tema else tmp_path / "corpus"
        p.mkdir(parents=True, exist_ok=True)
        return p
    monkeypatch.setattr(almacen, "corpus_dir", corpus_dir)
    return tmp_path / "corpus"


@pytest.fixture(scope="module")
def vintages():
    return s.serie_con_vintages()


@pytest.fixture(scope="module")
def fx():
    return s.fx_ecb_sintetico(desde="2024-01-02", hasta="2024-06-28")


@pytest.fixture(scope="module")
def multi():
    return s.precios_multimercado()


def _dtypes(df: pd.DataFrame) -> dict[str, str]:
    return {c: str(t) for c, t in df.dtypes.items()}


# --- Nombres de fichero -------------------------------------------------------------------------
@pytest.mark.parametrize("ticker, nombre", [
    ("^GSPC", "~5EGSPC"),
    ("ES=F", "ES~3DF"),
    ("BTC-USD", "BTC-USD"),
    ("EURUSD=X", "EURUSD~3DX"),
    ("BTC/USD", "BTC~2FUSD"),
    ("BRK.B", "BRK.B"),
    ("7203.T", "7203.T"),
    ("~X", "~7EX"),
    ("CON", "~43ON"),
    ("con.DE", "~63on.DE"),
    ("LPT1", "~4CPT1"),
    ("COMP", "COMP"),
    (".", "~2E"),
    ("..", "~2E~2E"),
    ("ABC.", "ABC~2E"),
    ("A B", "A~20B"),
    ("ÑU", "~C3~91U"),
])
def test_t1_nombre_fichero_ejemplos(ticker, nombre):
    assert nombre_fichero_ticker(ticker) == nombre
    assert ticker_de_fichero(nombre) == ticker


def test_t1_nombre_fichero_reversible_y_seguro():
    rng = np.random.default_rng(0)
    alfabeto = list(CARACTERES_SEGUROS + "^=/\\:*?\"<>| ~%&$#@!+,;'()[]{}") + ["é", "€", "日", "\U0001F4C8"]
    tickers = {"".join(rng.choice(alfabeto, size=rng.integers(1, 12))) for _ in range(2000)}
    nombres = {}
    for tk in tickers:
        nombre = nombre_fichero_ticker(tk)
        assert set(nombre) <= set(CARACTERES_SEGUROS + "~"), tk
        assert not nombre.startswith(".") and not nombre.endswith("."), tk
        assert ticker_de_fichero(nombre) == tk
        nombres[nombre] = tk
    assert len(nombres) == len(tickers)   # inyectiva


@pytest.mark.parametrize("ticker", ["", None, 5])
def test_t1_nombre_fichero_vacio_es_error(ticker):
    with pytest.raises(ValueError):
        nombre_fichero_ticker(ticker)


@pytest.mark.parametrize("nombre", ["", "~5e", "~ZZ", "~5", "A B", "A^B", "~41BC", "CON", "~FF"])
def test_t1_ticker_de_fichero_rechaza_nombres_no_canonicos(nombre):
    with pytest.raises(ValueError):
        ticker_de_fichero(nombre)


def test_t1_rutas_por_defecto(corpus_en_tmp):
    assert almacen.ruta_series() == corpus_en_tmp / "mercados_series" / "series" / "base_series.parquet"
    assert almacen.ruta_series().parent.is_dir()
    assert (almacen.ruta_precios("yahoo", "^GSPC")
            == corpus_en_tmp / "mercados_series" / "precios" / "yahoo" / "~5EGSPC.parquet")
    for malo in ("", "../x", "a/b", ".oculto"):
        with pytest.raises(ValueError):
            almacen.ruta_precios(malo, "ZZ")


# --- Series -------------------------------------------------------------------------------------
def test_t1_series_ida_y_vuelta(tmp_path, vintages, fx):
    ruta = tmp_path / "series.parquet"
    base = validar_series(pd.concat([vintages, fx], ignore_index=True))
    assert escribir_series(base, ruta) == ruta
    leido = leer_series(ruta=ruta)
    pd.testing.assert_frame_equal(leido, base)
    assert _dtypes(leido) == COLUMNAS_SERIES
    # El hueco permanente sobrevive: valor nulo con su nota.
    hueco = leido[(leido["serie"] == s.SERIE_VINTAGES) & (leido["fecha_referencia"] == s.MES_HUECO)]
    assert len(hueco) == 1 and hueco["valor"].isna().all() and hueco["nota"].iloc[0] == s.NOTA_HUECO


def test_t1_series_esquema_parquet_fijo(tmp_path):
    ruta = tmp_path / "series.parquet"
    escribir_series(s.serie_mercado_diaria(hasta="2023-01-31"), ruta)   # divisa y nota todo nulos
    esquema = pq.read_schema(ruta)
    assert esquema.names == list(COLUMNAS_SERIES)
    assert esquema.field("nota").type == pa.string()
    assert esquema.field("divisa").type == pa.string()
    assert esquema.field("fecha_referencia").type == pa.timestamp("ns")
    assert esquema.field("disponible_utc").type == pa.timestamp("ns", tz="UTC")


def test_t1_leer_series_filtra(tmp_path, vintages, fx):
    ruta = tmp_path / "series.parquet"
    escribir_series(pd.concat([vintages, fx], ignore_index=True), ruta)
    gbp = leer_series("ecb:EXR.D.GBP.EUR", ruta)
    assert set(gbp["serie"]) == {"ecb:EXR.D.GBP.EUR"}
    pd.testing.assert_frame_equal(gbp, fx[fx["serie"] == "ecb:EXR.D.GBP.EUR"].reset_index(drop=True))
    dos = leer_series(["ecb:EXR.D.USD.EUR", s.SERIE_VINTAGES], ruta)
    assert set(dos["serie"]) == {"ecb:EXR.D.USD.EUR", s.SERIE_VINTAGES}
    for vacio in (leer_series("no:EXISTE", ruta), leer_series([], ruta)):
        assert vacio.empty and _dtypes(vacio) == COLUMNAS_SERIES


def test_t1_leer_series_sin_fichero(tmp_path):
    vacio = leer_series(ruta=tmp_path / "no_existe.parquet")
    assert vacio.empty and _dtypes(vacio) == COLUMNAS_SERIES
    assert leer_series().empty   # ruta por defecto (en tmp_path)


def test_t1_series_ruta_por_defecto(vintages, corpus_en_tmp):
    ruta = escribir_series(vintages)
    assert ruta == corpus_en_tmp / "mercados_series" / "series" / "base_series.parquet"
    pd.testing.assert_frame_equal(leer_series(), vintages)


def test_t1_series_reemplaza_solo_las_presentes(tmp_path, vintages, fx):
    ruta = tmp_path / "series.parquet"
    escribir_series(pd.concat([vintages, fx], ignore_index=True), ruta)
    # Nueva vintage de sint:PARO: el resto de series no se toca.
    extra = vintages[vintages["fecha_referencia"] == vintages["fecha_referencia"].max()].tail(1).copy()
    extra["vintage"] = "2026-06-05"
    extra["disponible_utc"] = pd.Timestamp("2026-06-05 12:30", tz="UTC")
    extra["fecha_publicacion"] = pd.Timestamp("2026-06-05")
    extra["valor"] = 4.2
    escribir_series(pd.concat([vintages, extra], ignore_index=True), ruta)
    leido = leer_series(ruta=ruta)
    pd.testing.assert_frame_equal(leer_series(list(s.SERIES_FX), ruta), fx)
    assert len(leido) == len(vintages) + len(fx) + 1
    assert ((leido["vintage"] == "2026-06-05") & (leido["valor"] == 4.2)).sum() == 1
    # Reescribir lo mismo (con el hueco nulo) no es una pérdida.
    escribir_series(leer_series(s.SERIE_VINTAGES, ruta), ruta)
    assert len(leer_series(ruta=ruta)) == len(leido)
    # Una serie nueva se añade sin tocar las demás.
    escribir_series(s.serie_mercado_diaria(hasta="2023-02-28"), ruta)
    assert set(leer_series(ruta=ruta)["serie"]) == {s.SERIE_VINTAGES, s.SERIE_DIARIA, *s.SERIES_FX}


@pytest.mark.parametrize("cambio", ["falta_clave", "valor", "fecha_publicacion", "nota", "hueco_rellenado",
                                    # regresión: los metadatos que justifican la publicación también
                                    "disponible_utc", "metodo_publicacion", "fuente", "unidad"])
def test_t1_series_solo_crece(tmp_path, vintages, fx, cambio):
    ruta = tmp_path / "series.parquet"
    escribir_series(pd.concat([vintages, fx], ignore_index=True), ruta)
    antes = ruta.read_bytes()
    df = vintages.copy()
    i = df.index[(df["fecha_referencia"] == s.MES_RETRASADO) & df["nota"].notna()][0]
    if cambio == "falta_clave":
        df = df.drop(index=df.index[0])
    elif cambio == "valor":
        df.loc[i, "valor"] += 0.1
    elif cambio == "fecha_publicacion":
        df.loc[i, "fecha_publicacion"] += pd.Timedelta(days=1)
        df.loc[i, "disponible_utc"] += pd.Timedelta(days=1)
    elif cambio == "nota":
        df.loc[i, "nota"] = None
    elif cambio == "disponible_utc":
        df.loc[i, "disponible_utc"] = pd.NaT
    elif cambio == "metodo_publicacion":
        df.loc[i, "metodo_publicacion"] = "lag_fijo"
    elif cambio == "fuente":
        df.loc[i, "fuente"] = "otra"
    elif cambio == "unidad":
        df["unidad"] = "otra unidad"
    else:
        j = df.index[df["valor"].isna()][0]
        df.loc[j, "valor"] = 4.0
    with pytest.raises(ErrorContrato, match="solo crece"):
        escribir_series(df, ruta)
    assert ruta.read_bytes() == antes
    escribir_series(df, ruta, permitir_perdidas=True)
    leido = leer_series(s.SERIE_VINTAGES, ruta)
    pd.testing.assert_frame_equal(leido, validar_series(df))
    pd.testing.assert_frame_equal(leer_series(list(s.SERIES_FX), ruta), fx)


def test_t1_series_invalida_no_toca_el_fichero(tmp_path, vintages):
    ruta = tmp_path / "series.parquet"
    escribir_series(vintages, ruta)
    antes = ruta.read_bytes()
    malo = vintages.assign(adj_close=1.0)
    with pytest.raises(ErrorContrato):
        escribir_series(malo, ruta)
    malo = vintages.copy()
    malo.loc[0, "metodo_publicacion"] = "inventado"
    with pytest.raises(ErrorContrato):
        escribir_series(malo, ruta)
    assert ruta.read_bytes() == antes


# --- Atomicidad ---------------------------------------------------------------------------------
def test_t1_escritura_atomica_si_falla_replace(tmp_path, vintages, fx, monkeypatch):
    ruta = tmp_path / "series.parquet"
    escribir_series(vintages, ruta)
    antes = ruta.read_bytes()

    def falla(*args, **kwargs):
        raise OSError("disco lleno (simulado)")
    monkeypatch.setattr(almacen.os, "replace", falla)
    with pytest.raises(OSError, match="simulado"):
        escribir_series(pd.concat([vintages, fx], ignore_index=True), ruta)
    assert ruta.read_bytes() == antes
    assert sorted(p.name for p in tmp_path.iterdir()) == ["series.parquet"]   # sin temporales


def test_t1_escritura_atomica_si_falla_a_medias(tmp_path, vintages, fx, monkeypatch):
    ruta = tmp_path / "series.parquet"
    escribir_series(vintages, ruta)
    antes = ruta.read_bytes()

    def a_medias(tabla, destino, **kwargs):
        with open(destino, "wb") as f:
            f.write(b"PAR1 a medias")
        raise OSError("corte (simulado)")
    monkeypatch.setattr(almacen.pq, "write_table", a_medias)
    with pytest.raises(OSError, match="simulado"):
        escribir_series(pd.concat([vintages, fx], ignore_index=True), ruta)
    assert ruta.read_bytes() == antes
    assert sorted(p.name for p in tmp_path.iterdir()) == ["series.parquet"]
    monkeypatch.undo()
    pd.testing.assert_frame_equal(leer_series(ruta=ruta), vintages)


def test_t1_replace_reintenta_si_el_destino_esta_ocupado(tmp_path, vintages, monkeypatch):
    ruta = tmp_path / "series.parquet"
    real = almacen.os.replace
    llamadas = []

    def ocupado_una_vez(origen, destino):
        llamadas.append(destino)
        if len(llamadas) == 1:
            raise PermissionError("abierto por otro proceso (simulado)")
        return real(origen, destino)
    monkeypatch.setattr(almacen.os, "replace", ocupado_una_vez)
    monkeypatch.setattr(almacen.time, "sleep", lambda _: None)
    escribir_series(vintages, ruta)
    assert len(llamadas) == 2
    pd.testing.assert_frame_equal(leer_series(ruta=ruta), vintages)


# --- Precios ------------------------------------------------------------------------------------
def _renombrar(multi: pd.DataFrame) -> pd.DataFrame:
    """Tickers reales con caracteres conflictivos (los datos siguen siendo sintéticos)."""
    nombres = {"ZZNY": "^GSPC", "ZZ=F": "ES=F", "ZZ-USD": "BTC-USD"}
    return multi.assign(ticker=multi["ticker"].replace(nombres))


def test_t1_precios_ida_y_vuelta_y_nombres(tmp_path, multi):
    carpeta = tmp_path / "yahoo"
    df = _renombrar(multi)
    rutas = escribir_precios(df, "yahoo", carpeta)
    nombres = sorted(p.name for p in carpeta.iterdir())
    assert sorted(p.name for p in rutas) == nombres
    for esperado in ("~5EGSPC.parquet", "ES~3DF.parquet", "BTC-USD.parquet", "ZZLN.L.parquet",
                     "ZZDE.DE.parquet", "9999.T.parquet"):
        assert esperado in nombres
    assert tickers_guardados("yahoo", carpeta) == sorted(df["ticker"].unique())
    for tk in df["ticker"].unique():
        leido = leer_precios(tk, "yahoo", carpeta)
        esperado = df[df["ticker"] == tk].sort_values("sesion").reset_index(drop=True)
        pd.testing.assert_frame_equal(leido, esperado)
        assert _dtypes(leido) == COLUMNAS_PRECIOS
    assert leer_precios("ZZLN.L", "yahoo", carpeta)["divisa"].eq("GBp").all()   # subunidad intacta


def test_t1_precios_ruta_por_defecto(multi, corpus_en_tmp):
    df = _renombrar(multi)
    rutas = escribir_precios(df[df["ticker"] == "^GSPC"], "yahoo")
    assert rutas == [corpus_en_tmp / "mercados_series" / "precios" / "yahoo" / "~5EGSPC.parquet"]
    assert len(leer_precios("^GSPC")) == (df["ticker"] == "^GSPC").sum()
    assert tickers_guardados() == ["^GSPC"]
    assert leer_precios("NO-EXISTE").empty
    assert tickers_guardados("otro") == []


def test_t1_precios_sin_fichero(tmp_path):
    vacio = leer_precios("ZZ", "yahoo", tmp_path / "nada")
    assert vacio.empty and _dtypes(vacio) == COLUMNAS_PRECIOS


def test_t1_precios_fusion_solo_crece_y_conserva_lo_guardado(tmp_path):
    carpeta = tmp_path / "yahoo"
    accion = s.accion_con_eventos()
    primera = accion[accion["sesion"] < "2024-07-01"]
    escribir_precios(primera, "yahoo", carpeta)
    # Segunda descarga, posterior: solapa un mes con los mismos valores y añade sesiones nuevas.
    nueva_desc = pd.Timestamp("2025-02-01", tz="UTC")
    segunda = accion[accion["sesion"] >= "2024-06-01"].assign(descargado=nueva_desc)
    segunda.loc[segunda.index[0], "close"] *= 1 + 1e-12    # ruido de coma flotante (des-ajuste): no es cambio
    escribir_precios(segunda, "yahoo", carpeta)
    leido = leer_precios(s.TICKER_ACCION, "yahoo", carpeta)
    assert len(leido) == len(accion)
    # Las sesiones ya guardadas conservan la primera descarga (valores y 'descargado').
    pd.testing.assert_frame_equal(leido[leido["sesion"] < "2024-07-01"].reset_index(drop=True),
                                  primera.reset_index(drop=True))
    assert (leido.loc[leido["sesion"] >= "2024-07-01", "descargado"] == nueva_desc).all()


@pytest.mark.parametrize("cambio", ["close", "dividendo", "split", "volume"])
def test_t1_precios_correccion_tardia_es_error_y_no_cambia_el_pasado(tmp_path, cambio):
    """Regresión: una descarga posterior que corrige una vela vieja no puede reescribir el pasado.

    La vela conserva la fecha_publicacion de su sesión: si la corrección entrase, get_precios en
    fechas pasadas vería datos que entonces no se conocían (fuga hacia atrás).
    """
    from ingesta.mercados_series.tools import get_precios

    carpeta = tmp_path / "yahoo"
    accion = s.accion_con_eventos()
    escribir_precios(accion, "yahoo", carpeta)
    antes = (carpeta / "ZZAC.parquet").read_bytes()
    base = leer_precios(s.TICKER_ACCION, "yahoo", carpeta)
    vistas = {(f, aj): get_precios(s.TICKER_ACCION, f, ajuste=aj, divisa="original", base=base).datos
              for f in ("2022-03-02", "2022-06-30") for aj in ("ninguno", "total")}
    corr = accion[accion["sesion"] == "2022-03-01"].assign(descargado=pd.Timestamp("2026-10-01", tz="UTC"))
    corr[cambio] = {"close": 999.0, "dividendo": 5.0, "split": 2.0, "volume": 1.0}[cambio]
    with pytest.raises(ErrorContrato, match="solo crece"):
        escribir_precios(corr, "yahoo", carpeta)
    assert (carpeta / "ZZAC.parquet").read_bytes() == antes
    base = leer_precios(s.TICKER_ACCION, "yahoo", carpeta)
    for (f, aj), esperado in vistas.items():
        pd.testing.assert_frame_equal(
            get_precios(s.TICKER_ACCION, f, ajuste=aj, divisa="original", base=base).datos, esperado)
    # Corregir un error propio es posible, pero explícito.
    escribir_precios(corr, "yahoo", carpeta, permitir_correcciones=True)
    fila = leer_precios(s.TICKER_ACCION, "yahoo", carpeta).set_index("sesion").loc["2022-03-01"]
    assert fila[cambio] == corr[cambio].iloc[0]


def test_t1_precios_revision_tolerada_conserva_lo_guardado_y_la_apunta(tmp_path):
    """Regresión (revisión): una revisión de Yahoo del volumen de la última sesión (MSFT, NVDA) o de un
    dividendo antiguo (ITX.MC, MC.PA, VOD.L) rechazaba el ticker entero y lo congelaba en cada refresco.
    Con ``revisiones_toleradas`` se añaden las sesiones nuevas, la vela guardada no cambia (sin fuga hacia
    atrás) y la discrepancia queda en el registro. Otra columna sigue siendo error."""
    carpeta = tmp_path / "yahoo"
    accion = s.accion_con_eventos()
    primera = accion[accion["sesion"] < "2024-07-01"]
    escribir_precios(primera, "yahoo", carpeta)
    antes = leer_precios(s.TICKER_ACCION, "yahoo", carpeta)
    segunda = accion[accion["sesion"] >= "2024-06-03"].assign(descargado=pd.Timestamp("2025-02-01", tz="UTC"))
    ultima_guardada = primera["sesion"].max()
    segunda.loc[segunda["sesion"] == ultima_guardada, "volume"] += 12345.0
    with pytest.raises(ErrorContrato, match="solo crece"):
        escribir_precios(segunda, "yahoo", carpeta)
    escribir_precios(segunda, "yahoo", carpeta, revisiones_toleradas=("volume", "dividendo"))
    leido = leer_precios(s.TICKER_ACCION, "yahoo", carpeta)
    assert len(leido) == len(accion)                                        # las sesiones nuevas entran
    pd.testing.assert_frame_equal(leido[leido["sesion"] < "2024-07-01"].reset_index(drop=True),
                                  antes.reset_index(drop=True))             # lo guardado, intacto
    registro = pd.read_csv(almacen.ruta_revisiones_precios("yahoo", carpeta))
    assert list(registro["columna"]) == ["volume"] and registro["ticker"].iloc[0] == s.TICKER_ACCION
    assert pd.Timestamp(registro["sesion"].iloc[0]) == ultima_guardada
    otra = segunda.copy()
    otra.loc[otra["sesion"] == ultima_guardada, "close"] *= 1.01
    with pytest.raises(ErrorContrato, match="close"):
        escribir_precios(otra, "yahoo", carpeta, revisiones_toleradas=("volume", "dividendo"))


def test_t1_precios_invalidos_no_se_escriben(tmp_path):
    carpeta = tmp_path / "yahoo"
    accion = s.accion_con_eventos()
    with pytest.raises(ErrorContrato):
        escribir_precios(accion.assign(**{"Adj Close": accion["close"]}), "yahoo", carpeta)
    malo = accion.copy()
    malo.loc[malo.index[-1], "descargado"] = pd.Timestamp("2024-12-31 12:00", tz="UTC")   # vela incompleta
    with pytest.raises(ErrorContrato):
        escribir_precios(malo, "yahoo", carpeta)
    assert not carpeta.exists() or not any(carpeta.iterdir())
    escribir_precios(accion, "yahoo", carpeta)
    # Cambiar la divisa de parte de un ticker ya guardado rompe el contrato al fusionar.
    cola = accion.tail(5).assign(divisa="EUR", descargado=pd.Timestamp("2025-03-01", tz="UTC"))
    with pytest.raises(ErrorContrato, match="divisa"):
        escribir_precios(cola, "yahoo", carpeta)
    pd.testing.assert_frame_equal(leer_precios(s.TICKER_ACCION, "yahoo", carpeta), accion)


def test_t1_precios_choque_de_mayusculas(tmp_path):
    carpeta = tmp_path / "yahoo"
    accion = s.accion_con_eventos()
    escribir_precios(accion.assign(ticker="zzac"), "yahoo", carpeta)
    with pytest.raises(ErrorContrato, match="mayúsculas"):
        escribir_precios(accion, "yahoo", carpeta)
    dos = pd.concat([accion.assign(ticker="AB"), accion.assign(ticker="ab")], ignore_index=True)
    with pytest.raises(ErrorContrato, match="mayúsculas"):
        escribir_precios(dos, "yahoo", tmp_path / "otro")


def test_t1_precios_proveedor_invalido(tmp_path):
    with pytest.raises(ValueError):
        escribir_precios(s.accion_con_eventos(), "../fuera", tmp_path)


def test_t1_precios_fichero_con_otro_ticker(tmp_path):
    carpeta = tmp_path / "yahoo"
    [ruta] = escribir_precios(s.accion_con_eventos(), "yahoo", carpeta)
    shutil.copy(ruta, carpeta / f"{nombre_fichero_ticker('OTRO')}.parquet")
    with pytest.raises(ErrorContrato, match="otros tickers"):
        leer_precios("OTRO", "yahoo", carpeta)


# --- duckdb (opcional) --------------------------------------------------------------------------
def test_t1_duckdb_lee_los_parquet(tmp_path, vintages, fx, multi):
    duckdb = pytest.importorskip("duckdb")
    ruta = tmp_path / "series.parquet"
    escribir_series(pd.concat([vintages, fx], ignore_index=True), ruta)
    con = duckdb.connect()
    filas = con.execute(
        "SELECT serie, count(*) FROM read_parquet(?) GROUP BY serie ORDER BY serie", [str(ruta)]).fetchall()
    esperado = pd.concat([vintages, fx]).groupby("serie").size()
    assert filas == list(esperado.items())
    nulos = con.execute("SELECT count(*) FROM read_parquet(?) WHERE valor IS NULL AND nota IS NOT NULL",
                        [str(ruta)]).fetchone()[0]
    assert nulos == 1
    carpeta = tmp_path / "yahoo"
    escribir_precios(_renombrar(multi), "yahoo", carpeta)
    n = con.execute("SELECT count(*) FROM read_parquet(?) WHERE ticker = '^GSPC'",
                    [str(carpeta / "~5EGSPC.parquet")]).fetchone()[0]
    assert n == (_renombrar(multi)["ticker"] == "^GSPC").sum()
    con.close()
