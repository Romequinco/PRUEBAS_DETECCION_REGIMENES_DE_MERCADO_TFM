"""TimeGate del Tema 1: qué se sabía en una fecha de decisión.

Regla única: una fila es visible en ``fecha`` si ``fecha_publicacion <= fecha`` (``fecha`` es un
día natural; ``fecha_publicacion`` ya codifica la hora de decisión, ver ``contrato``). La base solo
crece: lo que se sabía en ``fecha`` nunca cambia al añadir filas publicadas después.

- ``fecha_decision``    normaliza una fecha de decisión (día natural, sin huso, formato estricto).
- ``as_of_series``      valor de una serie tal como se conocía en ``fecha`` (última vintage visible).
- ``visibles_precios``  sesiones de precios visibles en ``fecha``.
- ``ajustar_a_fecha``   precios ajustados con SOLO los splits y dividendos visibles en ``fecha``.
- ``des_ajustar_yahoo`` inversa del ajuste por splits de Yahoo: de su vista a precios as-traded.

Las tres consultas filtran por visibilidad ANTES de calcular nada: el truncado (añadir filas con
``fecha_publicacion > fecha`` no cambia el resultado) se cumple por construcción.
"""
from __future__ import annotations

import re
import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd

from ingesta.mercados_series.contrato import COLUMNAS_PRECIOS, COLUMNAS_SERIES, FECHA, REAL

FechaLike = date | str | pd.Timestamp

# Único formato de texto admitido para una fecha de decisión: ISO 'AAAA-MM-DD'. '02/01/2024' es
# 2 de enero en España y 1 de febrero en EE. UU.: un texto ambiguo no se adivina.
_PATRON_FECHA = re.compile(r"\d{4}-\d{2}-\d{2}")

# Columnas que añade ``ajustar_a_fecha`` (en este orden, tras las del contrato de precios).
COLUMNAS_AJUSTE: tuple[str, ...] = (
    "open_aj_splits", "high_aj_splits", "low_aj_splits", "close_aj_splits", "volume_aj_splits",
    "dividendo_aj_splits", "open_aj_total", "high_aj_total", "low_aj_total", "close_aj_total",
    "factor_splits", "factor_dividendos",
)
_OHLC = ("open", "high", "low", "close")


def fecha_decision(fecha: FechaLike) -> pd.Timestamp:
    """``fecha`` como día natural: ``pd.Timestamp`` naive a medianoche.

    Admite ``date``, ``datetime``, ``pd.Timestamp`` o ``np.datetime64`` a medianoche y sin huso,
    y textos ISO estrictos ``'AAAA-MM-DD'``. ``ValueError`` si es nula, lleva huso (un instante
    con huso no identifica un día de decisión: se pasa la fecha local de Nueva York), trae una
    hora distinta de 00:00 (p. ej. ``utcnow()``: podría adelantar casi un día; pasar ``.date()``)
    o es de otro tipo o formato: enteros, ``'now'``/``'today'``, ``'02/01/2024'`` (ambiguo),
    ``'Jan 2024'``...
    """
    if fecha is None:
        raise ValueError("fecha de decisión nula")
    if isinstance(fecha, str):
        if not _PATRON_FECHA.fullmatch(fecha):
            raise ValueError(f"fecha de decisión inválida: {fecha!r}; se espera texto ISO 'AAAA-MM-DD'")
        try:
            return pd.Timestamp(date.fromisoformat(fecha))
        except ValueError as e:
            raise ValueError(f"fecha de decisión inválida: {fecha!r} ({e})") from None
    if isinstance(fecha, (pd.Timestamp, datetime, np.datetime64)):
        ts = pd.Timestamp(fecha)
    elif isinstance(fecha, date):
        ts = pd.Timestamp(fecha)
    else:
        raise ValueError(f"fecha de decisión de tipo no admitido {type(fecha).__name__}: {fecha!r}; "
                         f"se espera date, Timestamp o texto 'AAAA-MM-DD'")
    if pd.isna(ts):
        raise ValueError("fecha de decisión nula")
    if ts.tzinfo is not None:
        raise ValueError(f"la fecha de decisión es un día natural, sin huso: {fecha!r}")
    if ts != ts.normalize():
        raise ValueError(f"la fecha de decisión es un día natural, sin hora: {fecha!r} "
                         f"(pasa la fecha local de Nueva York, p. ej. con .date())")
    return ts


def _comprobar_columnas(df: pd.DataFrame, columnas, tabla: str, fechas: tuple[str, ...] = ()) -> None:
    """Columnas del contrato presentes y ``fechas`` como ``datetime64`` naive (tabla validada)."""
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(f"a la tabla de {tabla} le faltan columnas del contrato: {faltan}")
    malas = [c for c in fechas if not (pd.api.types.is_datetime64_dtype(df[c])
                                       and not isinstance(df[c].dtype, pd.DatetimeTZDtype))]
    if malas:
        raise ValueError(f"en la tabla de {tabla}, {malas} deben ser datetime64 sin huso "
                         f"(pásala antes por contrato.validar_{tabla})")


def as_of_series(base: pd.DataFrame, serie: str, fecha: FechaLike) -> pd.DataFrame:
    """Valores de ``serie`` tal como se conocían en ``fecha`` (tabla del contrato de series).

    Filtra ``base`` a la serie y a ``fecha_publicacion <= fecha``; para cada ``fecha_referencia``
    se queda con la fila de mayor ``fecha_publicacion`` (empate: mayor ``vintage``). Los huecos
    permanentes (``valor`` nulo con ``nota``) se devuelven tal cual. Resultado ordenado por
    ``fecha_referencia``, con las columnas y dtypes del contrato; vacío si nada es visible o la
    serie no existe. No modifica ``base``.
    """
    f = fecha_decision(fecha)
    _comprobar_columnas(base, COLUMNAS_SERIES, "series", ("fecha_referencia", "fecha_publicacion"))
    columnas = list(COLUMNAS_SERIES)
    visibles = base.loc[(base["serie"] == serie) & (base["fecha_publicacion"] <= f), columnas]
    # Dentro de una serie el formato de vintage es único (contrato): el orden de texto es el temporal.
    orden = visibles.sort_values(["fecha_referencia", "fecha_publicacion", "vintage"], kind="mergesort")
    ultimas = orden.drop_duplicates("fecha_referencia", keep="last")
    return ultimas.reset_index(drop=True).astype(COLUMNAS_SERIES)


def visibles_precios(precios: pd.DataFrame, fecha: FechaLike) -> pd.DataFrame:
    """Filas de ``precios`` con ``fecha_publicacion <= fecha``, ordenadas por (ticker, sesion).

    Conserva las columnas del contrato de precios (as-traded, sin ajustar). No modifica ``precios``.
    """
    f = fecha_decision(fecha)
    _comprobar_columnas(precios, COLUMNAS_PRECIOS, "precios", ("sesion", "fecha_publicacion"))
    visibles = precios.loc[precios["fecha_publicacion"] <= f]
    return visibles.sort_values(["ticker", "sesion"], kind="mergesort").reset_index(drop=True)


def _producto_posteriores(x: np.ndarray) -> np.ndarray:
    """``out[i] = prod(x[j] for j > i)`` (1 en la última posición)."""
    if len(x) == 0:
        return np.ones(0)
    return np.r_[np.cumprod(x[::-1])[::-1][1:], 1.0]


def _multiplicadores_dividendo(dividendo: np.ndarray, cierre: np.ndarray, sesiones: pd.Series,
                               ticker: str) -> tuple[np.ndarray, list[str]]:
    """``m[i] = 1 - D_i / C_{i-1}`` en cada sesión ex ``i > 0`` (1 sin evento) y avisos.

    Un multiplicador no finito o <= 0 (dividendo >= cierre previo, cierre previo nulo o negativo)
    es un dato erróneo: ese evento no ajusta y se devuelve un aviso que lo explica.
    """
    m = np.ones(len(dividendo))
    if len(dividendo) < 2:
        return m, []
    previo = cierre[:-1]
    d = dividendo[1:]
    con_evento = d > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        cand = np.where(con_evento, 1.0 - d / previo, 1.0)
    malos = con_evento & ~(np.isfinite(cand) & (cand > 0))
    avisos = []
    if malos.any():
        ejemplos = [str(s.date()) for s in sesiones.iloc[1:][malos].head(5)]
        avisos.append(f"{ticker}: dividendos que no se pueden ajustar (cierre previo nulo, <= 0 o "
                      f"menor que el dividendo) en {ejemplos}; se ignoran en el ajuste total")
        cand = np.where(malos, 1.0, cand)
    m[1:] = cand
    return m, avisos


def ajustar_a_fecha(precios_ticker: pd.DataFrame, fecha: FechaLike, *, avisar: bool = True) -> pd.DataFrame:
    """Precios de UN ticker ajustados con lo que se sabía en ``fecha``.

    1. Se quedan las filas visibles (``fecha_publicacion <= fecha``) y solo sus eventos.
    2. Splits: cada fila se multiplica por ``1 / P``, con ``P`` = producto de ``split`` de las
       sesiones visibles POSTERIORES a la fila → ``open_aj_splits``, ``high_aj_splits``,
       ``low_aj_splits``, ``close_aj_splits``, ``dividendo_aj_splits``; ``volume_aj_splits`` =
       ``volume * P``.
    3. Dividendos (hacia atrás): en cada sesión ex ``i`` con ``dividendo_aj_splits`` ``D_i > 0``,
       las filas anteriores a ``i`` se multiplican por ``1 - D_i / C_{i-1}`` (``C_{i-1}`` =
       ``close_aj_splits`` de la sesión visible previa) → ``open_aj_total``, ``high_aj_total``,
       ``low_aj_total``, ``close_aj_total``. Un dividendo en la primera fila visible no ajusta nada.
    4. Se añaden ``factor_splits`` (``1 / P``) y ``factor_dividendos``.

    Devuelve las columnas del contrato más las ajustadas, ordenado por ``sesion``. ``ValueError``
    si hay más de un ticker. Propiedades: (a) truncado: añadir filas o eventos con
    ``fecha_publicacion > fecha`` no cambia el resultado; (b) el retorno total diario
    (``close_aj_total``) no cambia por un split; (c) con la última fecha, coincide con la vista de
    Yahoo de hoy (``sinteticos.vista_yahoo_hoy``: ``Close`` ↔ ``close_aj_splits``, ``Adj Close`` ↔
    ``close_aj_total``).

    Un dividendo imposible de aplicar (``D_i >= C_{i-1}`` o cierre previo <= 0) no ajusta: el
    texto del aviso va en ``out.attrs['avisos']`` (siempre presente, lista) y, con ``avisar=True``
    (uso interactivo), también se emite un ``RuntimeWarning``. Las tools usan ``avisar=False`` y
    leen ``attrs``: no tocan el estado global de ``warnings`` (seguro entre hilos).
    ``ValueError`` si hay tickers nulos.
    """
    f = fecha_decision(fecha)
    _comprobar_columnas(precios_ticker, COLUMNAS_PRECIOS, "precios", ("sesion", "fecha_publicacion"))
    if precios_ticker["ticker"].isna().any():
        raise ValueError("ajustar_a_fecha: hay filas con ticker nulo (tabla sin validar)")
    tickers = precios_ticker["ticker"].unique()
    if len(tickers) > 1:
        raise ValueError(f"ajustar_a_fecha espera un único ticker; hay {len(tickers)}: {list(tickers)[:5]}")
    vis = precios_ticker.loc[precios_ticker["fecha_publicacion"] <= f]
    if vis["sesion"].duplicated().any():
        raise ValueError("ajustar_a_fecha: sesiones visibles repetidas (clave (ticker, sesion) del contrato)")
    out = vis.sort_values("sesion", kind="mergesort").reset_index(drop=True)

    split = out["split"].to_numpy(dtype=REAL)
    p_posterior = _producto_posteriores(split)
    factor_splits = 1.0 / p_posterior
    for c in _OHLC:
        out[f"{c}_aj_splits"] = out[c].to_numpy(dtype=REAL) * factor_splits
    out["volume_aj_splits"] = out["volume"].to_numpy(dtype=REAL) * p_posterior
    out["dividendo_aj_splits"] = out["dividendo"].to_numpy(dtype=REAL) * factor_splits

    ticker = str(tickers[0]) if len(tickers) else ""
    m, avisos = _multiplicadores_dividendo(out["dividendo_aj_splits"].to_numpy(),
                                           out["close_aj_splits"].to_numpy(), out["sesion"], ticker)
    if avisos and avisar:
        for texto in avisos:
            warnings.warn(texto, RuntimeWarning, stacklevel=2)
    factor_dividendos = _producto_posteriores(m)
    for c in _OHLC:
        out[f"{c}_aj_total"] = out[f"{c}_aj_splits"].to_numpy() * factor_dividendos
    out["factor_splits"] = factor_splits
    out["factor_dividendos"] = factor_dividendos

    columnas = [c for c in out.columns if c not in COLUMNAS_AJUSTE] + list(COLUMNAS_AJUSTE)
    out = out[columnas].astype({c: REAL for c in COLUMNAS_AJUSTE})
    out.attrs = {"avisos": avisos}
    return out


# Columnas de la vista de Yahoo (``auto_adjust=False``) que lee ``des_ajustar_yahoo``.
COLUMNAS_YAHOO: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits")


def des_ajustar_yahoo(vista: pd.DataFrame) -> pd.DataFrame:
    """Precios as-traded a partir de la vista de Yahoo de UN ticker (``auto_adjust=False``).

    Yahoo da ``Open/High/Low/Close`` y ``Dividends`` divididos por el producto de los splits
    POSTERIORES a cada fila conocidos al descargar, y ``Volume`` multiplicado por él;
    ``Stock Splits`` es el ratio nuevo/antiguo (0 sin evento). Esta función deshace ese ajuste
    con los mismos splits de la vista: OHLC y dividendo × producto, volumen ÷ producto, y
    ``split`` = ``Stock Splits`` (0 → 1.0). ``Adj Close`` se ignora (nunca se guarda).

    ``vista``: índice de fechas de sesión (``Date``, sin huso o con el huso del mercado) y las
    columnas ``COLUMNAS_YAHOO``. Devuelve ``sesion, open, high, low, close, volume, dividendo,
    split`` ordenado por sesión; el extractor añade el resto del contrato. Es exacta si la vista
    está completa desde el primer split; los dividendos de valores en subunidades (GBp) que Yahoo
    da en la unidad (GBP) los corrige el extractor. ``ValueError`` si faltan columnas, hay fechas
    repetidas o splits negativos.
    """
    faltan = [c for c in COLUMNAS_YAHOO if c not in vista.columns]
    if faltan:
        raise ValueError(f"a la vista de Yahoo le faltan columnas: {faltan}")
    indice = pd.DatetimeIndex(vista.index)
    if indice.tz is not None:
        indice = indice.tz_localize(None)
    if indice.duplicated().any():
        raise ValueError("des_ajustar_yahoo: fechas repetidas en la vista")
    v = vista.set_axis(indice.normalize()).sort_index()
    split = v["Stock Splits"].to_numpy(dtype=REAL)
    if (split < 0).any() or not np.isfinite(split).all():
        raise ValueError("des_ajustar_yahoo: 'Stock Splits' con valores negativos o no finitos")
    split = np.where(split == 0.0, 1.0, split)
    posteriores = _producto_posteriores(split)
    out = pd.DataFrame({"sesion": pd.Series(v.index, dtype=FECHA)})
    for c, cy in (("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close")):
        out[c] = v[cy].to_numpy(dtype=REAL) * posteriores
    out["volume"] = v["Volume"].to_numpy(dtype=REAL) / posteriores
    out["dividendo"] = v["Dividends"].to_numpy(dtype=REAL) * posteriores
    out["split"] = split
    return out
