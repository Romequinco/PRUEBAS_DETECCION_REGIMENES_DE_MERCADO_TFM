"""regimenes: deteccion de regimenes de mercado (TFM).

Paquete unico del TFM (estructura en docs/decisions/ADR-004-unificacion.md):

- ``regimenes.rutas``        rutas centralizadas del repositorio (unica fuente).
- ``regimenes.datos``        descarga dirigida por ``configs/catalog.yaml``.
- ``regimenes.features``     primitivas causales, lags de publicacion y paneles.
- ``regimenes.detectores``   interfaz ``RegimeDetector``, registro y 12 detectores (7 familias).
- ``regimenes.evaluacion``   el juez: walk-forward, metricas y ranking de deteccion (ADR-003).
- ``regimenes.benchmark``    ejecucion reanudable con cache por huella (``python -m regimenes.benchmark``).
- ``regimenes.fusion``       maquina causal normal/vigilancia/confirmado.
- ``regimenes.viz``          estilo de casa para figuras.
- ``regimenes.sinteticos``   generadores de escenarios sinteticos (esqueleto).
"""

__version__ = "0.4.0"

__all__ = ["__version__"]
