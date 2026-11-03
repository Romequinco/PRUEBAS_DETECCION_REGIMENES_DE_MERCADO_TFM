"""Detectores de la Fase 2 sobre el pool sin comprimir (ADR-010).

Esta carpeta NO empieza por ``f`` a propósito: ``benchmark.cache._cache_code_paths`` hashea
``detectores/f*/*.py`` (y ``registry.py``) en la huella de la caché v2, y a través de ella en la
del notebook 21. Nada de aquí se importa desde ``regimenes.detectores`` ni desde ``registry.py``.

Contrato común (ADR-010 §3-§4):

- Cada detector es un ``RegimeDetector`` (``regimenes.detectores.base``) con estado de crisis
  canónico ``n_states - 1``.
- Cada variante es un :class:`~regimenes.detectores.pool.especificaciones.SpecPool`, que resuelve
  su clase por ruta completa del módulo en vez del mapa de familias de ``registry.py``.
- ``especificaciones.specs_pool(track)`` reúne las variantes de la Fase 2 de la pista.
"""
