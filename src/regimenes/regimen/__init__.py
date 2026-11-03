"""Tool ``get_regimen(fecha)`` de la Fase 1 del TFM (versión PROVISIONAL).

ADR-005 §2.1: la Fase 1 se cierra publicando ``get_regimen(fecha)`` → estado,
probabilidad y días en el estado, usando solo información publicada hasta
``fecha``. El detector final y su conjunto de estados los fijará la futura
ADR-006 (notebook ``22_decision_final``); hasta entonces este subpaquete expone
una versión provisional con el detector **intercambiable**. La implementación
vive en :mod:`regimenes.regimen.tool`.
"""

from regimenes.regimen.tool import (
    CONFIG_REGIMEN,
    ESTADOS_PROVISIONALES,
    FechaFueraDeRango,
    cargar_config,
    get_regimen,
    limpiar_cache,
)

__all__ = [
    "CONFIG_REGIMEN",
    "ESTADOS_PROVISIONALES",
    "FechaFueraDeRango",
    "cargar_config",
    "get_regimen",
    "limpiar_cache",
]
