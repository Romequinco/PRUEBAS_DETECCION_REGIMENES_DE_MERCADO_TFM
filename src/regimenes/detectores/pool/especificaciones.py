"""Especificaciones de las variantes de la Fase 2 (ADR-010 §3).

``SpecPool`` es un ``DetectorSpec`` cuya clase vive en ``regimenes.detectores.pool``; así no hace
falta tocar ``registry.py`` (huella de la caché v2). Las especificaciones de cada grupo de
detectores las aporta su módulo:

- ``especificaciones_pool_crudo(track)``: D10-P, D10-R, D14 y D18 (``pool_crudo``).
- ``especificaciones_shu_supervisados(track)``: D15, D16 y D17 (``shu_supervisados``).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from regimenes.detectores.registry import DetectorSpec

PAQUETE = "regimenes.detectores.pool"


@dataclass(frozen=True)
class SpecPool(DetectorSpec):
    """Variante de la Fase 2: ``module`` es el nombre del módulo dentro de ``regimenes.detectores.pool``."""

    menu: str = ""

    def factory(self):
        cls = getattr(import_module(f"{PAQUETE}.{self.module}"), self.class_name)
        return cls(**self.params)


def specs_pool(track: str) -> list[SpecPool]:
    """Todas las variantes de la Fase 2 de la pista ``A`` o ``B``, en orden de id."""
    from regimenes.detectores.pool.pool_crudo import especificaciones_pool_crudo
    from regimenes.detectores.pool.shu_supervisados import especificaciones_shu_supervisados

    specs = [*especificaciones_pool_crudo(track.upper()), *especificaciones_shu_supervisados(track.upper())]
    return sorted(specs, key=lambda s: s.detector_id)
