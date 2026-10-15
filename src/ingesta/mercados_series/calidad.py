"""Controles de calidad de precios as-traded (bloque 6).

``incidencias_precios`` marca, sobre la tabla del contrato de precios de uno o varios tickers:

- ``salto``: rentabilidad diaria con el split deshecho (``close_t * split_t / close_{t-1} - 1``)
  mayor que ``UMBRAL_SALTO`` en valor absoluto en una sesión SIN split (un split mal registrado,
  un cambio de unidad o un error del proveedor; también caídas reales extremas).
- ``volumen_cero``: volumen 0 en un instrumento con volumen (no índices, divisas ni fondos).
- ``cierre_repetido``: cierre idéntico al de la sesión previa (precio congelado o día sin negociación
  relleno); en letras del Tesoro y fondos monetarios es normal.
- ``dividendo_alto``: dividendo mayor que ``UMBRAL_DIVIDENDO`` del cierre previo (dividendo
  extraordinario, dividendo en subunidad equivocada o retorno de capital).

``resumen_calidad`` cuenta las incidencias por ticker. Solo marca: no corrige nada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UMBRAL_SALTO = 0.40
UMBRAL_DIVIDENDO = 0.20
# Clases (instrumentType de Yahoo) sin volumen significativo.
CLASES_SIN_VOLUMEN = frozenset({"INDEX", "CURRENCY", "MUTUALFUND"})
TIPOS = ("salto", "volumen_cero", "cierre_repetido", "dividendo_alto")


def _incidencias_ticker(p: pd.DataFrame, con_volumen: bool) -> pd.DataFrame:
    p = p.sort_values("sesion", kind="mergesort").reset_index(drop=True)
    close = p["close"].to_numpy(dtype=float)
    split = p["split"].to_numpy(dtype=float)
    previo = np.r_[np.nan, close[:-1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = close * split / previo - 1.0
        div_rel = p["dividendo"].to_numpy(dtype=float) / previo
    marcas = {
        "salto": (np.abs(ret) > UMBRAL_SALTO) & (split == 1.0),
        "volumen_cero": (p["volume"].to_numpy(dtype=float) == 0.0) if con_volumen else np.zeros(len(p), bool),
        "cierre_repetido": close == previo,
        "dividendo_alto": div_rel > UMBRAL_DIVIDENDO,
    }
    filas = []
    for tipo, m in marcas.items():
        m = np.asarray(m, dtype=bool) & np.isfinite(close)
        if m.any():
            valor = ret if tipo == "salto" else (div_rel if tipo == "dividendo_alto" else close)
            filas.append(pd.DataFrame({"ticker": p["ticker"].to_numpy()[m], "sesion": p["sesion"].to_numpy()[m],
                                       "tipo": tipo, "valor": valor[m]}))
    if not filas:
        return pd.DataFrame(columns=["ticker", "sesion", "tipo", "valor"])
    return pd.concat(filas, ignore_index=True)


def incidencias_precios(precios: pd.DataFrame, clases: dict[str, str] | None = None) -> pd.DataFrame:
    """Incidencias de calidad (``ticker, sesion, tipo, valor``) de una tabla de precios as-traded.

    ``clases``: ``{ticker: instrumentType}``; los de ``CLASES_SIN_VOLUMEN`` no se marcan por
    volumen cero. ``valor``: rentabilidad (salto), cierre (repetido, volumen cero) o dividendo
    sobre el cierre previo.
    """
    clases = clases or {}
    partes = [_incidencias_ticker(g, clases.get(tk, "") not in CLASES_SIN_VOLUMEN)
              for tk, g in precios.groupby("ticker", sort=True)]
    partes = [x for x in partes if len(x)]
    if not partes:
        return pd.DataFrame(columns=["ticker", "sesion", "tipo", "valor"])
    return pd.concat(partes, ignore_index=True).sort_values(["ticker", "sesion", "tipo"]).reset_index(drop=True)


def resumen_calidad(precios: pd.DataFrame, clases: dict[str, str] | None = None) -> pd.DataFrame:
    """Número de incidencias por ticker y tipo (índice ``ticker``, una columna por tipo y ``sesiones``)."""
    inc = incidencias_precios(precios, clases)
    tabla = (inc.groupby(["ticker", "tipo"]).size().unstack(fill_value=0)
             if len(inc) else pd.DataFrame())
    tabla = tabla.reindex(columns=list(TIPOS), fill_value=0)
    sesiones = precios.groupby("ticker").size().rename("sesiones")
    return tabla.reindex(sesiones.index, fill_value=0).join(sesiones).astype(int)


__all__ = ["UMBRAL_SALTO", "UMBRAL_DIVIDENDO", "CLASES_SIN_VOLUMEN", "TIPOS", "incidencias_precios",
           "resumen_calidad"]
