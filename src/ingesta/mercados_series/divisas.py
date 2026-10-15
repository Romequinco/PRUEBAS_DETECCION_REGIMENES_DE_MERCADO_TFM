"""Divisas: subunidades de cotización y conversión a USD AL LEER con el FX del BCE visible en fecha.

Se guarda siempre la divisa original; la conversión se hace en las tools. El BCE publica unidades
de X por 1 EUR (series ``ecb:EXR.D.<X>.EUR``, ~16:00 Europe/Berlin), luego::

    USD por 1 X = (USD/EUR) / (X/EUR)      (X = EUR → USD/EUR; X = USD → 1)

El TimeGate se aplica a los fixings: solo cuentan los visibles en la fecha de decisión
(``timegate.as_of_series``). Cada fila se convierte con el último fixing visible cuya
``fecha_referencia`` sea <= la de la fila (festivos TARGET: el anterior).

Cobertura (límite conocido): los tipos de referencia del BCE empiezan el 1999-01-04 y solo cubren
unas 30 divisas (no TWD, aunque XTAI está en ``config/mercados.yaml``; algunas empiezan después de
1999 y el RUB está suspendido desde 2022). Fuera de esa cobertura la conversión deja ``NaN`` y un
aviso, nunca un FX inventado. PENDIENTE: FX secundario (Fed H.10 / FRED ``DEX*``, con su calendario
``fed_h10``) para antes de 1999 y para divisas que el BCE no publica.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd

from ingesta.mercados_series import timegate

FechaLike = date | str | pd.Timestamp

# Subunidad de cotización → (divisa ISO, unidades de subunidad por unidad).
SUBUNIDADES: dict[str, tuple[str, float]] = {
    "GBp": ("GBP", 100.0),   # peniques (Londres)
    "GBX": ("GBP", 100.0),   # alias de GBp en algunas fuentes
    "ZAc": ("ZAR", 100.0),   # céntimos de rand (Johannesburgo)
    "ILA": ("ILS", 100.0),   # agorot (Tel Aviv)
}
SERIE_FX = "ecb:EXR.D.{divisa}.EUR"
SERIE_USD_EUR = SERIE_FX.format(divisa="USD")


def normalizar_subunidad(valores: pd.Series | np.ndarray | float, divisa: str):
    """``(valores / factor, divisa ISO)`` si ``divisa`` es una subunidad; si no, ``(valores, divisa)``.

    Distingue mayúsculas (``GBp`` ≠ ``GBP``). No modifica la entrada.
    """
    if divisa in SUBUNIDADES:
        iso, factor = SUBUNIDADES[divisa]
        return valores / factor, iso
    return valores, divisa


def _fixings(base: pd.DataFrame, serie: str, fecha: FechaLike) -> pd.Series:
    """Fixings de ``serie`` visibles en ``fecha`` (sin huecos), indexados por ``fecha_referencia``."""
    visibles = timegate.as_of_series(base, serie, fecha)
    visibles = visibles[visibles["valor"].notna()]
    return pd.Series(visibles["valor"].to_numpy(float),
                     index=pd.DatetimeIndex(visibles["fecha_referencia"], name="fecha_referencia"))


def _existe(base: pd.DataFrame, serie: str) -> bool:
    return bool(len(base)) and bool((base["serie"] == serie).any())


def fx_usd_por_unidad(base_series: pd.DataFrame, divisa: str, fecha: FechaLike) -> pd.Series:
    """USD por 1 unidad de ``divisa`` según el BCE, tal como se conocía en ``fecha``.

    Usa ``timegate.as_of_series`` sobre ``ecb:EXR.D.USD.EUR`` y ``ecb:EXR.D.<divisa>.EUR`` y las
    alinea por ``fecha_referencia`` (solo días con ambas). Índice: ``fecha_referencia``; nombre:
    ``f'USD/{divisa}'``. ``USD`` → serie de 1.0 en las fechas de USD/EUR; ``EUR`` → USD/EUR.
    Una subunidad es ``ValueError`` (normalizar antes); una divisa sin serie del BCE, ``KeyError``.
    Los fixings nulos (huecos declarados) se descartan.
    """
    if divisa in SUBUNIDADES:
        raise ValueError(f"{divisa!r} es una subunidad: normaliza antes con normalizar_subunidad")
    nombre = f"USD/{divisa}"
    if not _existe(base_series, SERIE_USD_EUR):
        if divisa == "USD":
            vacio = pd.DatetimeIndex([], name="fecha_referencia")
            return pd.Series([], index=vacio, dtype=float, name=nombre)
        raise KeyError(f"no hay serie {SERIE_USD_EUR!r} en la base de FX")
    usd_eur = _fixings(base_series, SERIE_USD_EUR, fecha)
    if divisa == "USD":
        return pd.Series(1.0, index=usd_eur.index, name=nombre)
    if divisa == "EUR":
        return usd_eur.rename(nombre)
    serie = SERIE_FX.format(divisa=divisa)
    if not _existe(base_series, serie):
        raise KeyError(f"no hay serie del BCE {serie!r} para {divisa!r}")
    x_eur = _fixings(base_series, serie, fecha)
    comunes = usd_eur.index.intersection(x_eur.index)
    return (usd_eur.loc[comunes] / x_eur.loc[comunes]).rename(nombre)


def a_usd(df: pd.DataFrame, columnas: Sequence[str], columna_divisa_o_valor: str,
          base_fx: pd.DataFrame, fecha: FechaLike, columna_fecha: str) -> pd.DataFrame:
    """Copia de ``df`` con ``columnas`` convertidas a USD con el FX visible en ``fecha``.

    ``columna_divisa_o_valor``: nombre de una columna de ``df`` con la divisa de cada fila, o un
    código de divisa común a todas. Primero se normalizan subunidades; después, para la fecha de
    cada fila (``columna_fecha``) se usa el último fixing con ``fecha_referencia <=`` esa fecha
    entre los visibles en ``fecha`` (festivos del BCE: el anterior). Sin fixing previo → ``NaN``
    y la fila se cuenta en ``df.attrs['avisos']``. Añade ``divisa_original`` y ``fx_usd`` (factor
    aplicado) y deja ``divisa = 'USD'``. No modifica ``df``.

    Detalles: ``fx_usd`` es el multiplicador total (incluye la subunidad: GBp → USD/GBP / 100);
    ``fx_fecha`` es la ``fecha_referencia`` del fixing usado (NaT para USD y sin fixing). Si la
    fecha de una fila es posterior al último fixing visible (su fixing aún no se conocía en
    ``fecha``, o fin de semana), se usa ese último y se avisa. Una divisa sin serie del BCE y una
    fila con fecha nula (salvo en USD) dejan ``NaN`` y un aviso. ``avisos`` es una lista de textos
    (se conservan los que ya tuviera ``df``).
    """
    faltan = [c for c in [*columnas, columna_fecha] if c not in df.columns]
    if faltan:
        raise KeyError(f"columnas que no están en df: {faltan}")
    out = df.copy()
    avisos: list[str] = list(df.attrs.get("avisos", []))
    es_columna = columna_divisa_o_valor in df.columns
    divisas = (df[columna_divisa_o_valor] if es_columna
               else pd.Series(columna_divisa_o_valor, index=df.index, dtype=object))
    if divisas.isna().any():
        raise ValueError(f"{int(divisas.isna().sum())} filas sin divisa: no se pueden convertir a USD")
    fechas = pd.to_datetime(df[columna_fecha])
    if isinstance(fechas.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{columna_fecha!r} debe ser una fecha sin huso (sesión o periodo)")
    fechas = fechas.dt.normalize()
    fecha_decision = timegate.fecha_decision(fecha)

    fx = pd.Series(np.nan, index=df.index, dtype=float)
    fx_fecha = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for original in pd.unique(divisas):
        filas = divisas == original
        uno, iso = normalizar_subunidad(1.0, original)
        if iso == "USD":
            fx[filas] = uno
            continue
        try:
            tabla = fx_usd_por_unidad(base_fx, iso, fecha_decision)
        except KeyError as e:
            avisos.append(f"{int(filas.sum())} filas en {original} sin FX del BCE ({e.args[0]}): valores NaN")
            continue
        d = fechas[filas].to_numpy("datetime64[ns]")
        sin_fecha = np.isnat(d)
        # searchsorted coloca NaT al final (daría el último fixing): las fechas nulas no tienen FX.
        pos = tabla.index.to_numpy("datetime64[ns]").searchsorted(d, side="right") - 1
        con_fx = (pos >= 0) & ~sin_fecha
        valores = np.full(len(d), np.nan)
        usadas = np.full(len(d), np.datetime64("NaT"), dtype="datetime64[ns]")
        valores[con_fx] = tabla.to_numpy(float)[pos[con_fx]] * uno
        usadas[con_fx] = tabla.index.to_numpy("datetime64[ns]")[pos[con_fx]]
        fx[filas] = valores
        fx_fecha[filas] = usadas
        if sin_fecha.any():
            avisos.append(f"{int(sin_fecha.sum())} filas en {original} con {columna_fecha!r} nula: "
                          f"sin FX, valores NaN")
        sin_fx = int((~con_fx & ~sin_fecha).sum())
        if sin_fx:
            avisos.append(f"{sin_fx} filas en {original} sin fixing del BCE visible en "
                          f"{fecha_decision.date()} anterior o igual a su fecha: valores NaN")
        if len(tabla):
            ultimo = tabla.index.max()
            tarde = int((d[~sin_fecha] > ultimo.to_datetime64()).sum())
            if tarde:
                avisos.append(f"{tarde} filas en {original} posteriores al último fixing visible en "
                              f"{fecha_decision.date()} ({ultimo.date()}): se usa ese fixing")

    for c in columnas:
        out[c] = pd.to_numeric(df[c], errors="raise").astype(float) * fx
    out["divisa_original"] = divisas.astype(object).to_numpy()
    out["fx_usd"] = fx
    out["fx_fecha"] = fx_fecha
    out["divisa"] = "USD"
    if es_columna and columna_divisa_o_valor != "divisa":
        out[columna_divisa_o_valor] = "USD"
    out.attrs["avisos"] = avisos
    return out
