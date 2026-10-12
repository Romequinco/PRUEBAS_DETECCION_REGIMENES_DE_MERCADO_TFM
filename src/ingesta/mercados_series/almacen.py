"""Almacén en disco del Tema 1 (parquet bajo ``corpus_dir('mercados_series')``, fuera del repo).

Estructura::

    <corpus>/mercados_series/series/series.parquet                 tabla larga de todas las series
    <corpus>/mercados_series/precios/<proveedor>/<ticker>.parquet  un fichero por ticker

Toda escritura valida con ``contrato`` y es atómica (``.tmp`` + ``os.replace``). La base solo
crece: reescribir una serie no puede perder ni cambiar filas ya guardadas, y una descarga de
precios no puede cambiar una vela ya guardada (su ``fecha_publicacion`` es la de la sesión, así
que una corrección tardía se vería hacia atrás), salvo que se pida para corregir errores propios.
Los nombres de fichero de ticker son seguros en Windows y reversibles (``nombre_fichero_ticker``).

Los parquet llevan el esquema del contrato fijado (texto → ``string``, fechas → ``timestamp[ns]``,
instantes → ``timestamp[ns, UTC]``), también con columnas enteras nulas: se leen igual con
pyarrow, pandas o duckdb. No hay bloqueo entre procesos: un solo escritor a la vez.
"""
from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ingesta.comun.rutas import corpus_dir
from ingesta.mercados_series.contrato import (
    CLAVE_PRECIOS,
    CLAVE_SERIES,
    COLUMNAS_PRECIOS,
    COLUMNAS_SERIES,
    FECHA,
    REAL,
    TEXTO,
    UTC,
    ErrorContrato,
    vacio_precios,
    vacio_series,
    validar_precios,
    validar_series,
)

TEMA = "mercados_series"

# Caracteres permitidos tal cual en el nombre de fichero de un ticker; el resto se codifica
# como ``~XX`` (hexadecimal del byte UTF-8): '^GSPC' → '~5EGSPC', 'ES=F' → 'ES~3DF', 'BTC/USD' →
# 'BTC~2FUSD'. Windows no distingue mayúsculas: dos tickers que solo difieren en la caja chocan
# (se detecta al escribir y es un error).
CARACTERES_SEGUROS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-_"
_SEGUROS = frozenset(CARACTERES_SEGUROS)
_HEX = frozenset("0123456789ABCDEF")
ESCAPE = "~"

# Nombres reservados de Windows (también con extensión: 'CON.DE.parquet' no se puede crear).
# Si la parte anterior al primer punto es uno de ellos, se codifica su primer carácter:
# 'CON.DE' → '~43ON.DE'.
_RESERVADOS_WINDOWS = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)})

_REINTENTOS_REPLACE = 5   # os.replace falla en Windows si otro proceso tiene abierto el destino


# --- Nombres de fichero -------------------------------------------------------------------------
def _escapar(caracter: str) -> str:
    return "".join(f"{ESCAPE}{b:02X}" for b in caracter.encode("utf-8"))


def nombre_fichero_ticker(ticker: str) -> str:
    """Nombre de fichero (sin extensión) seguro y reversible para ``ticker``.

    ``'^GSPC'`` → ``'~5EGSPC'``; ``'EURUSD=X'`` → ``'EURUSD~3DX'``. ``ValueError`` si es vacío.

    Codificación: cada carácter fuera de ``CARACTERES_SEGUROS`` (incluido ``~``) pasa a ``~XX``
    por byte UTF-8, hexadecimal en mayúsculas. Además se codifican un punto inicial (ficheros
    ocultos, ``..``), un punto final (Windows lo elimina) y el primer carácter de un nombre
    reservado de Windows (``CON``, ``NUL``, ``COM1``...). Es inyectiva: ``ticker_de_fichero``
    la invierte exactamente.
    """
    if not isinstance(ticker, str) or not ticker:
        raise ValueError(f"ticker vacío o no textual: {ticker!r}")
    partes = [c if c in _SEGUROS else _escapar(c) for c in ticker]
    if ticker[0] == ".":
        partes[0] = _escapar(".")
    if ticker[-1] == "." and len(ticker) > 1:
        partes[-1] = _escapar(".")
    if ticker.split(".", 1)[0].upper() in _RESERVADOS_WINDOWS:
        partes[0] = _escapar(ticker[0])
    return "".join(partes)


def ticker_de_fichero(nombre: str) -> str:
    """Inversa de ``nombre_fichero_ticker``.

    ``nombre`` es el nombre sin extensión. ``ValueError`` si no es un nombre que haya podido
    producir ``nombre_fichero_ticker`` (caracteres no seguros, ``~`` sin dos dígitos hexadecimales
    en mayúsculas o bytes que no forman UTF-8).
    """
    if not isinstance(nombre, str) or not nombre:
        raise ValueError(f"nombre de fichero vacío o no textual: {nombre!r}")
    datos = bytearray()
    i = 0
    while i < len(nombre):
        c = nombre[i]
        if c == ESCAPE:
            hexa = nombre[i + 1:i + 3]
            if len(hexa) != 2 or not set(hexa) <= _HEX:
                raise ValueError(f"escape inválido en {nombre!r} (posición {i}): se espera '~XX'")
            datos.append(int(hexa, 16))
            i += 3
        elif c in _SEGUROS:
            datos.extend(c.encode("ascii"))
            i += 1
        else:
            raise ValueError(f"carácter no permitido {c!r} en el nombre de fichero {nombre!r}")
    try:
        ticker = datos.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"{nombre!r} no decodifica a UTF-8 ({e})") from None
    if nombre_fichero_ticker(ticker) != nombre:
        raise ValueError(f"{nombre!r} no es la codificación canónica de {ticker!r}")
    return ticker


def _validar_proveedor(proveedor: str) -> str:
    if (not isinstance(proveedor, str) or not proveedor or proveedor.startswith(".")
            or not set(proveedor) <= _SEGUROS):
        raise ValueError(f"proveedor inválido {proveedor!r}: solo {CARACTERES_SEGUROS!r}, sin punto inicial")
    return proveedor


# --- Rutas --------------------------------------------------------------------------------------
def ruta_series() -> Path:
    """``corpus_dir('mercados_series') / 'series' / 'series.parquet'`` (crea la carpeta)."""
    carpeta = corpus_dir(TEMA) / "series"
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta / "series.parquet"


def _carpeta_precios(proveedor: str) -> Path:
    return corpus_dir(TEMA) / "precios" / _validar_proveedor(proveedor)


def ruta_precios(proveedor: str, ticker: str) -> Path:
    """``corpus_dir('mercados_series') / 'precios' / proveedor / f'{nombre_fichero_ticker(ticker)}.parquet'``."""
    return _carpeta_precios(proveedor) / f"{nombre_fichero_ticker(ticker)}.parquet"


# --- Parquet con el esquema del contrato --------------------------------------------------------
_TIPOS_ARROW = {
    TEXTO: pa.string(),
    FECHA: pa.timestamp("ns"),
    UTC: pa.timestamp("ns", tz="UTC"),
    REAL: pa.float64(),
}


def _esquema(columnas: dict[str, str]) -> pa.Schema:
    return pa.schema([pa.field(c, _TIPOS_ARROW[t]) for c, t in columnas.items()])


ESQUEMA_SERIES = _esquema(COLUMNAS_SERIES)
ESQUEMA_PRECIOS = _esquema(COLUMNAS_PRECIOS)


def _escribir_atomico(df: pd.DataFrame, esquema: pa.Schema, ruta: Path) -> Path:
    """Escribe ``df`` en un temporal de la misma carpeta y lo mueve sobre ``ruta`` (``os.replace``).

    Si algo falla, ``ruta`` queda como estaba y el temporal se borra.
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_name(f"{ruta.name}.{uuid.uuid4().hex[:12]}.tmp")
    try:
        tabla = pa.Table.from_pandas(df, schema=esquema, preserve_index=False)
        pq.write_table(tabla, tmp)
        for intento in range(_REINTENTOS_REPLACE):
            try:
                os.replace(tmp, ruta)
                break
            except PermissionError:
                if intento == _REINTENTOS_REPLACE - 1:
                    raise
                time.sleep(0.1 * (intento + 1))
    finally:
        if tmp.exists():
            tmp.unlink()
    return ruta


def _leer_parquet(ruta: Path, filtros=None) -> pd.DataFrame:
    return pq.read_table(ruta, filters=filtros).to_pandas()


# --- Series -------------------------------------------------------------------------------------
def _iguales(a: pd.Series, b: pd.Series) -> pd.Series:
    """Igualdad elemento a elemento en la que dos nulos son iguales."""
    return (a == b) | (a.isna() & b.isna())


# Columnas que una reescritura no puede cambiar en una clave ya guardada: el valor y todo lo que
# justifica su fecha de publicación o describe la serie.
VIGILADAS_SERIES = ("valor", "fecha_publicacion", "disponible_utc", "metodo_publicacion", "nota",
                    "unidad", "divisa", "fuente")
# Columnas de una vela ya guardada que una descarga posterior no puede cambiar.
VIGILADAS_PRECIOS = ("mercado", "open", "high", "low", "close", "volume", "dividendo", "split", "divisa",
                     "fecha_publicacion", "disponible_utc")
# Tolerancia relativa al comparar velas: el des-ajuste de una descarga de Yahoo (precio ajustado ×
# splits) no reproduce el as-traded bit a bit.
TOLERANCIA_PRECIOS = 1e-9


def _cambios(guardado: pd.DataFrame, nuevo: pd.DataFrame, clave: list[str], columnas: tuple[str, ...],
             *, faltan_es_problema: bool = True, rtol: float = 0.0) -> list[str]:
    """Claves guardadas ausentes en ``nuevo`` (si ``faltan_es_problema``) o con ``columnas`` distintas.

    Dos nulos son iguales; con ``rtol`` los reales se comparan con esa tolerancia relativa.
    """
    cruce = guardado.merge(nuevo, on=clave, how="left", suffixes=("", "_nuevo"), indicator=True)
    falta = cruce["_merge"] == "left_only"
    problemas = []
    if faltan_es_problema and falta.any():
        ej = cruce.loc[falta, clave].head(5).astype(str).agg(", ".join, axis=1).tolist()
        problemas.append(f"{int(falta.sum())} filas guardadas desaparecerían (p. ej. {ej})")
    presentes = cruce[~falta]
    for col in columnas:
        a, b = presentes[col], presentes[f"{col}_nuevo"]
        iguales = _iguales(a, b)
        if rtol and pd.api.types.is_float_dtype(a):
            iguales |= pd.Series(np.isclose(a.to_numpy(float), b.to_numpy(float), rtol=rtol, atol=0.0),
                                 index=a.index)
        cambia = ~iguales
        if cambia.any():
            ej = presentes.loc[cambia, clave].head(5).astype(str).agg(", ".join, axis=1).tolist()
            problemas.append(f"{int(cambia.sum())} filas guardadas cambiarían de {col!r} (p. ej. {ej})")
    return problemas


def _perdidas_series(guardado: pd.DataFrame, nuevo: pd.DataFrame) -> list[str]:
    """Claves guardadas que faltan en ``nuevo`` o cambian en alguna de ``VIGILADAS_SERIES``."""
    problemas = _cambios(guardado, nuevo, list(CLAVE_SERIES), VIGILADAS_SERIES)
    if problemas:
        problemas.append("la base de series solo crece: usa permitir_perdidas=True solo para corregir "
                         "errores propios")
    return problemas


def _leer_guardado(ruta: Path, columnas: dict[str, str], vacio) -> pd.DataFrame:
    """Parquet propio (esquema del contrato fijado) con los dtypes del contrato, sin revalidar."""
    if not ruta.exists():
        return vacio()
    df = _leer_parquet(ruta)
    return df[list(columnas)].astype(columnas)


def escribir_series(df: pd.DataFrame, ruta: Path | None = None, *, permitir_perdidas: bool = False) -> Path:
    """Valida ``df`` y reemplaza en el almacén las series presentes en él; el resto no se toca.

    Base que solo crece: si para alguna serie de ``df`` falta una clave
    ``(serie, fecha_referencia, vintage)`` ya guardada, o una clave guardada cambia en alguna de
    ``VIGILADAS_SERIES`` (valor, fecha de publicación, ``disponible_utc``, método, nota, unidad,
    divisa o fuente), lanza ``ErrorContrato`` (salvo ``permitir_perdidas=True``, para corregir
    errores propios). La tabla resultante se valida entera una vez. Escritura atómica. Devuelve la
    ruta escrita.
    """
    nuevo = validar_series(df)
    ruta = Path(ruta) if ruta is not None else ruta_series()
    guardado = _leer_guardado(ruta, COLUMNAS_SERIES, vacio_series)
    presentes = set(nuevo["serie"])
    afectado = guardado["serie"].isin(presentes)
    if not permitir_perdidas and afectado.any():
        problemas = _perdidas_series(guardado[afectado], nuevo)
        if problemas:
            raise ErrorContrato("series", problemas)
    total = validar_series(pd.concat([guardado[~afectado], nuevo], ignore_index=True))
    return _escribir_atomico(total, ESQUEMA_SERIES, ruta)


def leer_series(series: Iterable[str] | str | None = None, ruta: Path | None = None) -> pd.DataFrame:
    """Tabla larga validada (todas las series o solo ``series``). Sin fichero: tabla vacía del contrato.

    El filtro por ``series`` se aplica al leer el parquet (``filters`` de pyarrow).
    """
    ruta = Path(ruta) if ruta is not None else ruta_series()
    if not ruta.exists():
        return vacio_series()
    filtros = None
    if series is not None:
        lista = [series] if isinstance(series, str) else list(series)
        if not lista:
            return vacio_series()
        filtros = [("serie", "in", lista)]
    return validar_series(_leer_parquet(ruta, filtros))


# --- Precios ------------------------------------------------------------------------------------
def _choque_de_caja(carpeta: Path, nombres: Iterable[str]) -> list[str]:
    """Nombres que solo difieren en la caja de otro (en el disco o entre sí): mismo fichero en Windows."""
    vistos: dict[str, str] = {}
    if carpeta.exists():
        vistos = {p.stem.casefold(): p.stem for p in carpeta.glob("*.parquet")}
    problemas = []
    for nombre in nombres:
        otro = vistos.setdefault(nombre.casefold(), nombre)
        if otro != nombre:
            problemas.append(f"el fichero {nombre!r} choca con {otro!r} (Windows no distingue mayúsculas)")
    return problemas


def escribir_precios(df: pd.DataFrame, proveedor: str, ruta: Path | None = None, *,
                     permitir_correcciones: bool = False) -> list[Path]:
    """Valida ``df`` (uno o varios tickers) y lo fusiona con lo guardado, un fichero por ticker.

    Base que solo crece: las sesiones nuevas se añaden y las ya guardadas se conservan tal cual
    (con su ``descargado`` original). Si una descarga trae una sesión ya guardada con OHLCV,
    dividendo, split, mercado, divisa o publicación distintos (``VIGILADAS_PRECIOS``; reales con
    tolerancia relativa ``TOLERANCIA_PRECIOS``), lanza ``ErrorContrato`` y no escribe nada: la
    ``fecha_publicacion`` de la vela es la de su sesión, así que una corrección o un dividendo
    añadido tarde se vería como conocido desde la sesión original (fuga hacia atrás).
    ``permitir_correcciones=True`` (solo para errores propios) deja ganar a la fila de ``df`` en
    esas sesiones, sabiendo que la corrección se presenta como conocida desde la sesión original.
    ``ruta`` sustituye a la carpeta ``precios/<proveedor>`` (tests). Escritura atómica por
    fichero. Devuelve las rutas.
    """
    _validar_proveedor(proveedor)
    nuevo = validar_precios(df)
    carpeta = Path(ruta) if ruta is not None else _carpeta_precios(proveedor)
    tickers = list(dict.fromkeys(nuevo["ticker"]))
    nombres = {tk: nombre_fichero_ticker(tk) for tk in tickers}
    choques = _choque_de_caja(carpeta, nombres.values())
    if choques:
        raise ErrorContrato("precios", choques)
    clave = list(CLAVE_PRECIOS)
    totales = {}
    for tk in tickers:
        guardado = leer_precios(tk, proveedor, ruta=carpeta)
        propio = nuevo[nuevo["ticker"] == tk]
        if not permitir_correcciones and len(guardado):
            problemas = _cambios(guardado, propio, clave, VIGILADAS_PRECIOS, faltan_es_problema=False,
                                 rtol=TOLERANCIA_PRECIOS)
            if problemas:
                raise ErrorContrato("precios", [f"{tk}: {p}" for p in problemas] + [
                    "la base de precios solo crece: una vela guardada no se corrige con una descarga "
                    "posterior (permitir_correcciones=True solo para errores propios)"])
        # Sin correcciones gana lo guardado (primera descarga); con correcciones, lo nuevo.
        orden = [propio, guardado] if permitir_correcciones else [guardado, propio]
        total = pd.concat([t for t in orden if len(t)], ignore_index=True).drop_duplicates(clave, keep="first")
        totales[tk] = validar_precios(total if len(total) else vacio_precios())
    # Todo validado antes de escribir: un ticker con problemas no deja a los demás a medias.
    return [_escribir_atomico(totales[tk], ESQUEMA_PRECIOS, carpeta / f"{nombres[tk]}.parquet")
            for tk in tickers]


def leer_precios(ticker: str, proveedor: str = "yahoo", ruta: Path | None = None) -> pd.DataFrame:
    """Precios as-traded validados de ``ticker``. Sin fichero: tabla vacía del contrato.

    ``ruta`` sustituye a la carpeta ``precios/<proveedor>``. ``ErrorContrato`` si el fichero
    contiene otro ticker (fichero renombrado a mano).
    """
    carpeta = Path(ruta) if ruta is not None else _carpeta_precios(proveedor)
    fichero = carpeta / f"{nombre_fichero_ticker(ticker)}.parquet"
    if not fichero.exists():
        return vacio_precios()
    out = validar_precios(_leer_parquet(fichero))
    otros = sorted(set(out["ticker"]) - {ticker})
    if otros:
        raise ErrorContrato("precios", [f"{fichero.name} contiene otros tickers: {otros[:5]}"])
    return out


def tickers_guardados(proveedor: str = "yahoo", ruta: Path | None = None) -> list[str]:
    """Tickers con fichero en ``precios/<proveedor>`` (o en ``ruta``), ordenados."""
    carpeta = Path(ruta) if ruta is not None else _carpeta_precios(proveedor)
    if not carpeta.exists():
        return []
    return sorted(ticker_de_fichero(p.stem) for p in carpeta.glob("*.parquet"))


__all__ = [
    "CARACTERES_SEGUROS", "ESQUEMA_PRECIOS", "ESQUEMA_SERIES", "nombre_fichero_ticker",
    "ticker_de_fichero", "ruta_series", "ruta_precios", "escribir_series", "leer_series",
    "escribir_precios", "leer_precios", "tickers_guardados",
]
