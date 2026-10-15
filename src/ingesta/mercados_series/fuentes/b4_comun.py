"""Utilidades comunes del bloque 4 (OCDE, Philadelphia Fed RTDSM y bancos centrales).

- ``cargar_fragmento``: lee ``config/fragmentos/b4_oecd_philly_bc.yaml`` (secciones ``series``,
  ``publicadores`` y ``mercados``, mismo esquema que ``catalogo_series.yaml`` / ``publicaciones.yaml``)
  hasta que el integrador lo funda con la configuración del tema.
- ``publicaciones_b4``: los publicadores del fragmento validados con las reglas de
  ``visibilidad`` (se pueden pasar a ``visibilidad.disponible_publicador``).
- ``tabla_contrato``: monta filas del contrato de series (``contrato.validar_series``) con
  ``disponible_utc`` = hora del publicador el día de publicación y ``fecha_publicacion`` = primera
  decisión de las 16:30 de Nueva York a partir de ahí.
- ``compactar_vintages``: de una matriz de vintages (una fila por periodo y vintage) deja solo los
  valores nuevos o revisados (dos vintages seguidos con el mismo valor se funden).
- ``guardar_crudo``: copia cruda de cada descarga en ``<corpus>/mercados_series/crudo/<fuente>/``.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ingesta.comun.rutas import corpus_dir
from ingesta.mercados_series import visibilidad
from ingesta.mercados_series.contrato import COLUMNAS_SERIES, UTC, primera_decision_vec, validar_series

TEMA = "mercados_series"
RUTA_FRAGMENTO = Path(__file__).resolve().parents[1] / "config" / "fragmentos" / "b4_oecd_philly_bc.yaml"

__all__ = ["TEMA", "RUTA_FRAGMENTO", "cargar_fragmento", "publicaciones_b4", "disponible", "tabla_contrato",
           "compactar_vintages", "ruta_crudo", "guardar_crudo"]


@lru_cache(maxsize=2)
def _fragmento(ruta: str, _mtime: int) -> dict[str, Any]:
    with open(ruta, encoding="utf-8") as f:
        crudo = yaml.safe_load(f) or {}
    for seccion in ("series", "publicadores", "mercados"):
        crudo.setdefault(seccion, {})
        crudo[seccion] = crudo[seccion] or {}
    return crudo


def cargar_fragmento(ruta: Path | None = None) -> dict[str, Any]:
    """Fragmento del bloque 4 tal cual (diccionario con ``series``, ``publicadores`` y ``mercados``)."""
    p = Path(ruta or RUTA_FRAGMENTO)
    return _fragmento(str(p), p.stat().st_mtime_ns)


def publicaciones_b4(ruta: Path | None = None) -> dict[str, Any]:
    """Publicadores del fragmento validados como en ``publicaciones.yaml`` (sin series ni alias).

    Fundido ya el fragmento por el integrador (el fichero no existe), ``publicaciones.yaml``.
    """
    if not Path(ruta or RUTA_FRAGMENTO).exists():
        return visibilidad.cargar_publicaciones()
    frag = cargar_fragmento(ruta)
    crudo = {"alias": {}, "publicadores": frag["publicadores"], "series": {}}
    return visibilidad._validar_publicaciones(crudo, Path(ruta or RUTA_FRAGMENTO).name)


def disponible(publicador: str, dias: pd.Series, publicaciones: dict[str, Any] | None = None) -> pd.Series:
    """Instante UTC de publicación de cada día (una llamada por fecha distinta)."""
    pubs = publicaciones if publicaciones is not None else publicaciones_b4()
    dias = pd.Series(pd.to_datetime(dias)).reset_index(drop=True)
    codigos, unicos = pd.factorize(dias)
    horas = pd.array([visibilidad.disponible_publicador(publicador, d, pubs) for d in unicos], dtype=UTC)
    return pd.Series(horas.take(codigos, allow_fill=True), dtype=UTC)


def tabla_contrato(serie: str, fecha_referencia: Any, dia_publicacion: Any, valor: Any, *, vintage: Any,
                   unidad: str, divisa: str | None, fuente: str, metodo: str, publicador: str,
                   nota: Any = None, publicaciones: dict[str, Any] | None = None) -> pd.DataFrame:
    """Filas del contrato de series de una serie (validadas).

    ``dia_publicacion``: día (local del publicador) en que el dato es público; ``vintage``: texto o
    serie de textos (``AAAA-MM``, ``AAAA-MM-DD`` o ``unico``). Los valores nulos necesitan ``nota``.
    """
    fr = pd.Series(pd.to_datetime(fecha_referencia)).reset_index(drop=True)
    n = len(fr)
    if n == 0:
        return validar_series(pd.DataFrame({c: pd.Series(dtype=t) for c, t in COLUMNAS_SERIES.items()}))
    disp = disponible(publicador, pd.Series(pd.to_datetime(dia_publicacion)).reset_index(drop=True), publicaciones)

    def _col(x: Any) -> Any:
        return pd.Series(x).reset_index(drop=True) if isinstance(x, (pd.Series, np.ndarray, list)) else [x] * n

    df = pd.DataFrame({
        "serie": serie, "fecha_referencia": fr.dt.normalize(), "fecha_publicacion": primera_decision_vec(disp),
        "disponible_utc": disp, "vintage": _col(vintage),
        "valor": pd.to_numeric(pd.Series(valor).reset_index(drop=True), errors="coerce").astype(float),
        "unidad": unidad, "divisa": divisa, "fuente": fuente, "metodo_publicacion": metodo, "nota": _col(nota)})
    return validar_series(df)


def compactar_vintages(largo: pd.DataFrame) -> pd.DataFrame:
    """De ``(periodo, orden_vintage, valor)`` deja la primera aparición y cada revisión de cada periodo.

    ``largo`` necesita las columnas ``periodo``, ``orden`` (orden cronológico del vintage) y ``valor``
    (sin nulos). Dos vintages seguidos con el mismo valor (redondeo a 10 decimales) se funden.
    """
    d = largo.sort_values(["periodo", "orden"]).reset_index(drop=True)
    previo = d.groupby("periodo")["valor"].shift()
    cambia = previo.isna() | (np.round(d["valor"] - previo, 10) != 0)
    return d[cambia].reset_index(drop=True)


def ruta_crudo(fuente: str) -> Path:
    """Carpeta de la copia cruda de una fuente (fuera del repositorio)."""
    p = corpus_dir(TEMA) / "crudo" / fuente
    p.mkdir(parents=True, exist_ok=True)
    return p


def guardar_crudo(fuente: str, nombre: str, contenido: bytes) -> Path:
    """Guarda una descarga cruda (atómico) y devuelve su ruta."""
    destino = ruta_crudo(fuente) / nombre
    tmp = destino.with_suffix(destino.suffix + ".tmp")
    tmp.write_bytes(contenido)
    tmp.replace(destino)
    return destino
