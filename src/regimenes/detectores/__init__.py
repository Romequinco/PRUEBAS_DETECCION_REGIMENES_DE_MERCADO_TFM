"""Detectores de regimen.

- ``base``      interfaz unica ``RegimeDetector`` (antes src/detector_base.py y
                su copia identica de la Capa 1, eliminada; ver
                docs/revisiones/registro_eliminaciones.md).
- ``registry``  ``DetectorSpec``, ``detector_specs``, ``specs_table``
                (antes src/detectors/registry.py): adaptador declarativo que
                conecta las implementaciones con los paneles y el benchmark.
- ``f1_reglas`` .. ``f7_deep``: implementaciones matematicas congeladas de la
                Capa 1, agrupadas por familia. Se importan de forma perezosa via
                ``DetectorSpec.factory()``.
"""

from regimenes.detectores.base import RegimeDetector
from regimenes.detectores.registry import DetectorSpec, detector_specs, specs_table

__all__ = ["DetectorSpec", "RegimeDetector", "detector_specs", "specs_table"]
