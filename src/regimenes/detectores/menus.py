"""Variantes detector × menú de datos (ADR-008 §3).

Un **menú** es un conjunto fijo de columnas de entrada, igual para todos los detectores
que lo admiten:

- ``N`` · núcleo: la configuración a priori de :mod:`regimenes.detectores.registry`
  (los 12 detectores originales, sin tocar).
- ``U`` · univariante: solo ``SP500_ret``.
- ``T`` · termómetros: una columna por bloque (``configs/bloques.yaml``).
- ``P`` · pool completo y ``R`` · rentabilidades multiactivo: pendientes (Fase 2).

Este módulo NO se importa desde ``regimenes.detectores`` ni entra en la huella de la
caché del benchmark (``benchmark.cache._cache_code_paths``): añadir variantes no
invalida los 24 resultados oficiales. Cada variante es un :class:`DetectorSpec` con id
``Dnn-M`` y los mismos hiperparámetros que su original; solo cambian las columnas.
"""

from __future__ import annotations

from dataclasses import replace

from regimenes.detectores.registry import DetectorSpec, detector_specs
from regimenes.features.bloques import cargar_bloques

MENUS = ("N", "U", "T")

# Qué detectores admite cada menú nuevo (ADR-008 §3; D11/D12 solo en sus menús
# actuales y los univariantes no van a T). N son los 12 originales.
ADMITIDOS = {
    "U": ("D03", "D04", "D08", "D09"),
    "T": ("D02", "D03", "D04", "D08", "D09", "D10"),
}


def columnas_menu(track: str, menu: str) -> tuple[str, ...]:
    """Columnas de entrada del menú ``U`` o ``T`` en la pista ``track``."""
    menu = menu.upper()
    if menu == "U":
        return ("SP500_ret",)
    if menu == "T":
        return tuple(cargar_bloques(track))
    raise ValueError(f"menú sin columnas fijas: {menu!r} (U | T)")


def _con_columnas(spec: DetectorSpec, menu: str, cols: tuple[str, ...]) -> DetectorSpec:
    params = dict(spec.params)
    if "features" in params:                 # D04, D08, D10
        params["features"] = list(cols)
    if "signs" in params:                    # D02: los termómetros ya suben con el estrés
        params["signs"] = {c: +1 for c in cols}
    return replace(spec, detector_id=f"{spec.detector_id}-{menu}", features=cols, params=params,
                   variant=f"menú {menu}: {len(cols)} columnas")


def variant_specs(track: str, menu: str) -> list[DetectorSpec]:
    """Especificaciones de la pista para un menú. ``N`` devuelve los 12 originales."""
    menu = menu.upper()
    originales = detector_specs(track)
    if menu == "N":
        return originales
    if menu not in ADMITIDOS:
        raise ValueError(f"menú desconocido: {menu!r} ({' | '.join(MENUS)})")
    cols = columnas_menu(track, menu)
    return [_con_columnas(s, menu, cols) for s in originales if s.detector_id in ADMITIDOS[menu]]
