"""Tipo libre de riesgo del Tema 1 (bloque 8).

Tres usos, tres series:

- **Agentes (punto en el tiempo)**: ``fred:DTB3`` (letra a 3 meses, mercado secundario, % anual en
  base descuento; publicada en T+1). En cada fecha de decisión ``d`` vale el último dato con
  ``fecha_publicacion <= d`` (la misma regla que ``timegate.as_of_series``), pasado a rendimiento
  equivalente de bono (``rendimiento_equivalente``) y devengado por días naturales (act/365) con
  el tipo conocido al principio de cada tramo.
- **Evaluación ex post**: ``ff:F3D.RF`` (RF diario de Ken French, la letra a 1 mes). Es la última
  edición del fichero: sirve para medir resultados a posteriori, NUNCA como entrada de un agente.
- **EUR**: EONIA (``ecb:EONIA``, hasta 2019-10-01) y €STR (``ecb:ESTR``, desde que es visible).
  Son ids del bloque 4 (``fuentes/bce.py``); si aún no están en la base, se devuelve la tabla con
  nulos y un aviso en ``attrs['avisos']``. Euribor 3M no está en el catálogo del bloque 4: no hay
  sensibilidad con Euribor (pendiente).

Todo se lee de la base del contrato (``almacen.leer_series``) o de una tabla que se pase, y se
resuelve con ``ultimo_visible`` (vectorizado y equivalente a ``as_of_series`` fecha a fecha).
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ingesta.mercados_series import almacen, timegate

SERIE_DTB3 = "fred:DTB3"
SERIE_FRENCH_RF = "ff:F3D.RF"
SERIE_EONIA = "ecb:EONIA"
SERIE_ESTR = "ecb:ESTR"
# Primer día de €STR (publicado el 2019-10-02 por la mañana) y último de EONIA (2021-12-31).
INICIO_ESTR = pd.Timestamp("2019-10-01")
DIAS_LETRA_3M = 91

COLUMNAS_TASA = ("tasa", "fecha_referencia", "fecha_publicacion")


def _fechas(fechas: Iterable) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([timegate.fecha_decision(f) for f in fechas])


def ultimo_visible(tabla: pd.DataFrame, fechas: Iterable) -> pd.DataFrame:
    """Para cada fecha de decisión, la fila visible con mayor ``(fecha_referencia, fecha_publicacion)``.

    ``tabla``: filas del contrato de series de UNA serie. Devuelve un DataFrame indexado por las
    fechas con ``tasa`` (el ``valor``), ``fecha_referencia`` y ``fecha_publicacion`` (nulos si no
    hay nada visible). Es ``timegate.as_of_series(...).iloc[-1]`` en cada fecha, sin bucle: se
    ordena por publicación y se acumula el máximo de la clave (referencia, publicación).
    """
    idx = _fechas(fechas)
    out = pd.DataFrame(index=idx, columns=list(COLUMNAS_TASA)).astype(
        {"tasa": float, "fecha_referencia": "datetime64[ns]", "fecha_publicacion": "datetime64[ns]"})
    if len(tabla) == 0 or len(idx) == 0:
        return out
    if tabla["serie"].nunique() > 1:
        raise ValueError(f"ultimo_visible espera una serie; hay {sorted(tabla['serie'].unique())[:5]}")
    t = tabla.sort_values(["fecha_publicacion", "fecha_referencia", "vintage"], kind="mergesort").reset_index(drop=True)
    ref = t["fecha_referencia"].to_numpy("datetime64[D]").astype(np.int64)
    pub = t["fecha_publicacion"].to_numpy("datetime64[D]").astype(np.int64)
    clave = (ref + 100_000) * 1_000_000 + (pub + 100_000)
    maximo = np.maximum.accumulate(clave)
    pos = np.maximum.accumulate(np.where(clave == maximo, np.arange(len(t)), 0))
    i = np.searchsorted(t["fecha_publicacion"].to_numpy("datetime64[ns]"), idx.to_numpy("datetime64[ns]"),
                        side="right") - 1
    ok = i >= 0
    filas = t.iloc[pos[i[ok]]]
    out.loc[idx[ok], "tasa"] = filas["valor"].to_numpy(float)
    out.loc[idx[ok], "fecha_referencia"] = filas["fecha_referencia"].to_numpy()
    out.loc[idx[ok], "fecha_publicacion"] = filas["fecha_publicacion"].to_numpy()
    return out


def rendimiento_equivalente(descuento_pct: pd.Series | np.ndarray | float, dias: int = DIAS_LETRA_3M):
    """Rendimiento equivalente de bono (decimal) de una letra cotizada en % de descuento.

    ``y = 365·d / (360 − d·dias)`` con ``d`` en decimal (convención del Tesoro de EE. UU.).
    """
    d = np.asarray(descuento_pct, dtype=float) / 100.0
    return 365.0 * d / (360.0 - d * dias)


def tasa_dtb3(fechas: Iterable, base: pd.DataFrame | None = None) -> pd.DataFrame:
    """DTB3 visible en cada fecha de decisión (punto en el tiempo) y su rendimiento equivalente.

    Columnas: ``tasa`` (% descuento), ``fecha_referencia``, ``fecha_publicacion`` y
    ``rendimiento`` (decimal anual, ``rendimiento_equivalente``). ``base``: tabla del contrato
    (por defecto, ``almacen.leer_series('fred:DTB3')``).
    """
    tabla = almacen.leer_series([SERIE_DTB3]) if base is None else base[base["serie"] == SERIE_DTB3]
    out = ultimo_visible(tabla, fechas)
    out["rendimiento"] = rendimiento_equivalente(out["tasa"].to_numpy(float))
    return out


def indice_devengo(rendimiento: pd.Series) -> pd.Series:
    """Índice de una cuenta al tipo libre de riesgo (base 1 en la primera fecha).

    Entre dos fechas consecutivas ``d0 < d1`` se devenga ``y(d0) · (d1 − d0)/365``: el tipo que se
    conocía al principio del tramo (sin mirar al futuro). Un tipo nulo devenga 0.
    """
    r = rendimiento.sort_index()
    dias = np.diff(r.index.to_numpy("datetime64[D]").astype(np.int64))
    tramo = 1.0 + np.nan_to_num(r.to_numpy(float)[:-1]) * dias / 365.0
    return pd.Series(np.r_[1.0, np.cumprod(tramo)], index=r.index, name="indice_rf")


def rf_french(base: pd.DataFrame | None = None) -> pd.Series:
    """RF diario de Ken French (decimal por sesión), ÚLTIMA edición: solo para evaluar ex post.

    Índice: ``fecha_referencia`` (sesiones). No es punto en el tiempo (se publica con meses de
    retraso y la edición puede revisar el pasado): no se le pasa nunca a un agente.
    """
    tabla = almacen.leer_series([SERIE_FRENCH_RF]) if base is None else base[base["serie"] == SERIE_FRENCH_RF]
    if tabla.empty:
        return pd.Series(dtype=float, name="rf_french")
    ult = tabla.sort_values(["fecha_referencia", "fecha_publicacion", "vintage"]).drop_duplicates(
        "fecha_referencia", keep="last")
    return pd.Series(ult["valor"].to_numpy(float) / 100.0, index=pd.DatetimeIndex(ult["fecha_referencia"]),
                     name="rf_french")


def tasa_eur(fechas: Iterable, base: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tipo a un día en EUR visible en cada fecha: EONIA hasta que €STR es visible, después €STR.

    Columnas de ``ultimo_visible`` más ``serie`` (de dónde sale cada fila). Se empalma sin el
    diferencial oficial (EONIA = €STR + 8,5 pb desde 2019-10-02): se declara en el aviso. Sin las
    series del BCE en la base, todo nulo y aviso en ``attrs['avisos']``.
    """
    if base is None:
        try:
            base = almacen.leer_series([SERIE_EONIA, SERIE_ESTR])
        except Exception as e:  # base sin las series o contrato aún sin integrar
            base = None
            error = f"{type(e).__name__}: {e}"
        else:
            error = None
    else:
        error = None
    idx = _fechas(fechas)
    vacio = pd.DataFrame(index=idx, columns=[*COLUMNAS_TASA, "serie"])
    if base is None or base[base["serie"].isin([SERIE_EONIA, SERIE_ESTR])].empty:
        vacio.attrs["avisos"] = [f"{SERIE_EONIA}/{SERIE_ESTR} no están en la base de series (bloque 4 sin "
                                 f"integrar: python -m ingesta.mercados_series.fuentes.bce --sin-red --escribir)"
                                 + (f"; {error}" if error else "")]
        return vacio
    eonia = ultimo_visible(base[base["serie"] == SERIE_EONIA], idx).assign(serie=SERIE_EONIA)
    estr = ultimo_visible(base[base["serie"] == SERIE_ESTR], idx).assign(serie=SERIE_ESTR)
    usar_estr = estr["tasa"].notna()
    out = eonia.where(~usar_estr, estr)
    out.attrs["avisos"] = ["empalme EONIA→€STR en la primera fecha con €STR visible, sin sumar los 8,5 pb "
                           "del diferencial oficial"]
    return out


__all__ = [
    "INICIO_ESTR", "SERIE_DTB3", "SERIE_EONIA", "SERIE_ESTR", "SERIE_FRENCH_RF", "indice_devengo",
    "rendimiento_equivalente", "rf_french", "tasa_dtb3", "tasa_eur", "ultimo_visible",
]
