"""Proveedores de precios del Tema 1 (bloque 6): interfaz común.

Un proveedor descarga la historia diaria de un ticker y la devuelve AS-TRADED con las columnas del
contrato de precios (``contrato.COLUMNAS_PRECIOS``), ya validada con ``contrato.validar_precios``.
Nunca entrega precios ajustados: el ajuste se reconstruye a fecha con ``timegate.ajustar_a_fecha``.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ProveedorPrecios(Protocol):
    """Interfaz mínima de un proveedor de precios diarios."""

    nombre: str   # nombre de la carpeta en <corpus>/mercados_series/precios/<nombre>

    def descargar(self, ticker: str) -> pd.DataFrame:
        """Historia completa de ``ticker`` as-traded, con las columnas del contrato de precios."""
        ...


__all__ = ["ProveedorPrecios"]
