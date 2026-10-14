"""Almacén en disco del Tema 1 (parquet bajo ``corpus_dir('mercados_series')``, fuera del repo).

Estructura::

    <corpus>/mercados_series/series/base_series.parquet            tabla larga de todas las series
    <corpus>/mercados_series/precios/<proveedor>/<ticker>.parquet  un fichero por ticker

Toda escritura valida con ``contrato`` y es atómica (``.tmp`` + ``os.replace``). La base solo
crece: reescribir una serie no puede perder ni cambiar filas ya guardadas, y una descarga de
precios no puede cambiar una vela ya guardada (su ``fecha_publicacion`` es la de la sesión, así
que una corrección tardía se vería hacia atrás), salvo que se pida para corregir errores propios. Las
revisiones del proveedor en columnas declaradas tolerables (volumen provisional, dividendo corregido)
no rechazan el ticker: se conserva la vela guardada y se apunta la discrepancia
(``ruta_revisiones_precios``).
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
from typing import Any

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
# Nombre del parquet de la base de series (bloque 2: ``base.construir_base`` lo escribe entero).
FICHERO_SERIES = "base_series.parquet"
# Row groups de la base de series (bloque 3: millones de filas con los vintages de ALFRED). Ordenada por
# serie, una consulta de una serie lee solo sus grupos (``leer_series`` filtra al leer).
FILAS_POR_GRUPO_SERIES = 65_536

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
    """``corpus_dir('mercados_series') / 'series' / FICHERO_SERIES`` (crea la carpeta)."""
    carpeta = corpus_dir(TEMA) / "series"
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta / FICHERO_SERIES


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


def _escribir_atomico(df: pd.DataFrame, esquema: pa.Schema, ruta: Path, *,
                      filas_por_grupo: int | None = None) -> Path:
    """Escribe ``df`` en un temporal de la misma carpeta y lo mueve sobre ``ruta`` (``os.replace``).

    ``filas_por_grupo``: tamaño de los row groups (con la tabla ordenada por serie, la lectura
    filtrada salta los grupos de otras series por sus estadísticas). Si algo falla, ``ruta`` queda
    como estaba y el temporal se borra.
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_name(f"{ruta.name}.{uuid.uuid4().hex[:12]}.tmp")
    try:
        tabla = pa.Table.from_pandas(df, schema=esquema, preserve_index=False)
        pq.write_table(tabla, tmp, row_group_size=filas_por_grupo)
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


def escribir_series(df: pd.DataFrame, ruta: Path | None = None, *, permitir_perdidas: bool = False,
                    revisables: Iterable[str] = (), purgar: Iterable[str] = ()) -> Path:
    """Valida ``df`` y reemplaza en el almacén las series presentes en él; el resto no se toca.

    Base que solo crece: si para alguna serie de ``df`` falta una clave
    ``(serie, fecha_referencia, vintage)`` ya guardada, o una clave guardada cambia en alguna de
    ``VIGILADAS_SERIES`` (valor, fecha de publicación, ``disponible_utc``, método, nota, unidad,
    divisa o fuente), lanza ``ErrorContrato`` (salvo ``permitir_perdidas=True``, para corregir
    errores propios). ``revisables``: ids cuyo ``valor`` puede cambiar (revisiones de la última
    edición de una serie con vintage ``unico``, declaradas con su nota); el resto de columnas y
    las claves siguen vigiladas. ``purgar``: ids que se quitan del almacén (series que ya no se
    construyen). La tabla resultante se valida entera una vez. Escritura atómica. Devuelve la
    ruta escrita.
    """
    nuevo = validar_series(df)
    ruta = Path(ruta) if ruta is not None else ruta_series()
    guardado = _leer_guardado(ruta, COLUMNAS_SERIES, vacio_series)
    purgar = set(purgar)
    if purgar:
        guardado = guardado[~guardado["serie"].isin(purgar)].reset_index(drop=True)
    presentes = set(nuevo["serie"])
    afectado = guardado["serie"].isin(presentes)
    if not permitir_perdidas and afectado.any():
        rev = set(revisables)
        g = guardado[afectado]
        fijas = g[~g["serie"].isin(rev)]
        problemas = _perdidas_series(fijas, nuevo) if len(fijas) else []
        revisadas = g[g["serie"].isin(rev)]
        if len(revisadas):
            sin_valor = tuple(c for c in VIGILADAS_SERIES if c != "valor")
            extra = _cambios(revisadas, nuevo, list(CLAVE_SERIES), sin_valor)
            if extra:
                problemas += [f"revisables: {p}" for p in extra] + [
                    "una revisión solo puede cambiar el valor (fechas, método y nota quedan fijos)"]
        if problemas:
            raise ErrorContrato("series", problemas)
    total = validar_series(pd.concat([guardado[~afectado], nuevo], ignore_index=True))
    return _escribir_atomico(total, ESQUEMA_SERIES, ruta, filas_por_grupo=FILAS_POR_GRUPO_SERIES)


def leer_series(series: Iterable[str] | str | None = None, ruta: Path | None = None, *,
                hasta: Any = None, validar: bool = True) -> pd.DataFrame:
    """Tabla larga (todas las series o solo ``series``). Sin fichero: tabla vacía del contrato.

    Los filtros se aplican al leer el parquet (``filters`` de pyarrow: con la base ordenada por serie
    y en row groups, solo se leen los grupos de las series pedidas). ``hasta``: solo filas con
    ``fecha_publicacion <= hasta`` (truncado al leer: lo publicado después ni se lee).
    ``validar=False`` (consultas de ``tools``): sin ``contrato.validar_series``, solo los dtypes
    del contrato; el parquet propio ya se validó entero al escribirlo (``escribir_series``) y
    validar un índice re-estimado de un millón de filas cuesta segundos.
    """
    ruta = Path(ruta) if ruta is not None else ruta_series()
    if not ruta.exists():
        return vacio_series()
    filtros = []
    if series is not None:
        lista = [series] if isinstance(series, str) else list(series)
        if not lista:
            return vacio_series()
        filtros.append(("serie", "in", lista))
    if hasta is not None:
        filtros.append(("fecha_publicacion", "<=", pd.Timestamp(hasta).to_datetime64()))
    df = _leer_parquet(ruta, filtros or None)
    if validar:
        return validar_series(df)
    return df[list(COLUMNAS_SERIES)].astype(COLUMNAS_SERIES).reset_index(drop=True)


def leer_ultimas(serie: str, hasta: Any, ruta: Path | None = None) -> pd.DataFrame:
    """Por ``fecha_referencia``, la fila de ``serie`` con mayor ``(fecha_publicacion, vintage)`` entre
    las publicadas hasta ``hasta`` (la regla de ``timegate.as_of_series``), resuelta en Arrow.

    Para las consultas de ``tools.get_serie`` sobre series de un millón de filas (índices
    re-estimados de ALFRED): solo pasa a pandas una fila por periodo. Sin validar (ver ``leer_series``).
    """
    ruta = Path(ruta) if ruta is not None else ruta_series()
    if not ruta.exists():
        return vacio_series()
    filtros = [("serie", "==", serie), ("fecha_publicacion", "<=", pd.Timestamp(hasta).to_datetime64())]
    tabla = pq.read_table(ruta, filters=filtros)
    if tabla.num_rows == 0:
        return vacio_series()
    tabla = tabla.sort_by([("fecha_referencia", "ascending"), ("fecha_publicacion", "ascending"),
                           ("vintage", "ascending")])
    ref = tabla.column("fecha_referencia").to_numpy()
    ultima = np.r_[ref[1:] != ref[:-1], True]
    df = tabla.take(pa.array(np.flatnonzero(ultima))).to_pandas()
    return df[list(COLUMNAS_SERIES)].astype(COLUMNAS_SERIES).reset_index(drop=True)


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


def _revisiones_precios(guardado: pd.DataFrame, nuevo: pd.DataFrame, columnas: Iterable[str]) -> pd.DataFrame:
    """Sesiones guardadas cuya ``columna`` (de ``columnas``) trae otro valor en ``nuevo``.

    Columnas ``ticker, sesion, columna, guardado, nuevo`` (los valores, solo para el registro local).
    """
    clave = list(CLAVE_PRECIOS)
    cols = [c for c in columnas if c in VIGILADAS_PRECIOS]
    cruce = guardado[clave + cols].merge(nuevo[clave + cols], on=clave, suffixes=("", "_nuevo"))
    filas = []
    for col in cols:
        a, b = cruce[col], cruce[f"{col}_nuevo"]
        iguales = _iguales(a, b)
        if pd.api.types.is_float_dtype(a):
            iguales |= pd.Series(np.isclose(a.to_numpy(float), b.to_numpy(float), rtol=TOLERANCIA_PRECIOS, atol=0.0),
                                 index=a.index)
        cambia = cruce[~iguales]
        filas += [{"ticker": r["ticker"], "sesion": r["sesion"], "columna": col, "guardado": r[col],
                   "nuevo": r[f"{col}_nuevo"]} for _, r in cambia.iterrows()]
    return pd.DataFrame(filas, columns=["ticker", "sesion", "columna", "guardado", "nuevo"])


def ruta_revisiones_precios(proveedor: str, ruta: Path | None = None) -> Path:
    """Registro de revisiones toleradas: ``<corpus>/mercados_series/logs/revisiones_precios_<proveedor>.csv``
    (con ``ruta``, la carpeta de precios de los tests: ``<ruta>/_revisiones_<proveedor>.csv``)."""
    if ruta is not None:
        return Path(ruta) / f"_revisiones_{proveedor}.csv"
    return corpus_dir(TEMA) / "logs" / f"revisiones_precios_{proveedor}.csv"


def escribir_precios(df: pd.DataFrame, proveedor: str, ruta: Path | None = None, *,
                     permitir_correcciones: bool = False,
                     revisiones_toleradas: Iterable[str] = ()) -> list[Path]:
    """Valida ``df`` (uno o varios tickers) y lo fusiona con lo guardado, un fichero por ticker.

    Base que solo crece: las sesiones nuevas se añaden y las ya guardadas se conservan tal cual
    (con su ``descargado`` original). Si una descarga trae una sesión ya guardada con OHLCV,
    dividendo, split, mercado, divisa o publicación distintos (``VIGILADAS_PRECIOS``; reales con
    tolerancia relativa ``TOLERANCIA_PRECIOS``), lanza ``ErrorContrato`` y no escribe nada: la
    ``fecha_publicacion`` de la vela es la de su sesión, así que una corrección o un dividendo
    añadido tarde se vería como conocido desde la sesión original (fuga hacia atrás).
    ``permitir_correcciones=True`` (solo para errores propios) deja ganar a la fila de ``df`` en
    esas sesiones, sabiendo que la corrección se presenta como conocida desde la sesión original.
    ``revisiones_toleradas`` (p. ej. ``('volume', 'dividendo')``: revisiones del proveedor, como el
    volumen provisional de la última sesión o un dividendo corregido): un cambio SOLO en esas columnas
    no rechaza el ticker; la vela guardada se conserva tal cual (no hay fuga hacia atrás), las
    sesiones nuevas se añaden y cada discrepancia se apunta en ``ruta_revisiones_precios`` (ticker,
    sesión, columna, valor guardado y nuevo, detección). Un cambio en otra columna sigue siendo error.
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
    toleradas = tuple(c for c in revisiones_toleradas if c in VIGILADAS_PRECIOS)
    estrictas = tuple(c for c in VIGILADAS_PRECIOS if c not in toleradas)
    totales = {}
    revisiones = []
    for tk in tickers:
        guardado = leer_precios(tk, proveedor, ruta=carpeta)
        propio = nuevo[nuevo["ticker"] == tk]
        if not permitir_correcciones and len(guardado):
            problemas = _cambios(guardado, propio, clave, estrictas, faltan_es_problema=False,
                                 rtol=TOLERANCIA_PRECIOS)
            if not problemas and toleradas:
                revisiones.append(_revisiones_precios(guardado, propio, toleradas))
            if problemas:
                raise ErrorContrato("precios", [f"{tk}: {p}" for p in problemas] + [
                    "la base de precios solo crece: una vela guardada no se corrige con una descarga "
                    "posterior (permitir_correcciones=True solo para errores propios)"])
        # Sin correcciones gana lo guardado (primera descarga); con correcciones, lo nuevo.
        orden = [propio, guardado] if permitir_correcciones else [guardado, propio]
        total = pd.concat([t for t in orden if len(t)], ignore_index=True).drop_duplicates(clave, keep="first")
        totales[tk] = validar_precios(total if len(total) else vacio_precios())
    # Todo validado antes de escribir: un ticker con problemas no deja a los demás a medias.
    rutas = [_escribir_atomico(totales[tk], ESQUEMA_PRECIOS, carpeta / f"{nombres[tk]}.parquet")
             for tk in tickers]
    revisadas = pd.concat([r for r in revisiones if len(r)], ignore_index=True) if any(
        len(r) for r in revisiones) else None
    if revisadas is not None:
        destino = ruta_revisiones_precios(proveedor, ruta)
        destino.parent.mkdir(parents=True, exist_ok=True)
        revisadas = revisadas.assign(proveedor=proveedor,
                                     detectado=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"))
        revisadas.to_csv(destino, mode="a", header=not destino.exists(), index=False, encoding="utf-8",
                         lineterminator="\n")
    return rutas


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
    "CARACTERES_SEGUROS", "ESQUEMA_PRECIOS", "FICHERO_SERIES", "ESQUEMA_SERIES", "nombre_fichero_ticker",
    "ticker_de_fichero", "ruta_series", "ruta_precios", "escribir_series", "leer_series",
    "escribir_precios", "leer_precios", "tickers_guardados", "ruta_revisiones_precios",
]
