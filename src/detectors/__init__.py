"""Registro activo de detectores para la Fase D v2.

Las implementaciones matemáticas permanecen congeladas en
``capa1_exploracion/detectors``. Este paquete solo aporta el adaptador declarativo
que las conecta con los paneles y el benchmark v2.
"""

from .registry import DetectorSpec, detector_specs, specs_table

__all__ = ["DetectorSpec", "detector_specs", "specs_table"]
