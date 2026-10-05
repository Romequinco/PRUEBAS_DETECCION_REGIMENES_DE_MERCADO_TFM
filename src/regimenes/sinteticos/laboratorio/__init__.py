"""Laboratorio de detectores con regimen conocido (notebook 17).

Simulacion controlada: los tres generadores parametricos con mecanismo explicito
(``gaussiano_regimen``, ``var_regimen``, ``garch_regimen``), ya ajustados en el
notebook 15 con el tramo de train, generan trayectorias con una secuencia de
regimen IMPUESTA y una intensidad de crisis controlada; los detectores del
benchmark se ejecutan sobre ellas con el MISMO protocolo walk-forward y las
mismas especificaciones, y se puntuan contra la verdad-terreno exacta.

- ``escenarios``  configuracion pre-registrada, rejilla de celdas, secuencias de
                  regimen, atenuacion de intensidad y ventanas de verdad-terreno.
- ``metricas``    metricas ADR-003 por trayectoria + retraso, exactitud
                  equilibrada y falsas alarmas por ano.
- ``ejecucion``   generacion cacheada de trayectorias y ejecucion paralela y
                  reanudable de (celda, trayectoria, detector) con cache por huella.
- ``analisis``    agregados para el notebook: matriz, degradacion, concordancia
                  con el ranking real (H3) y circularidad (H4).

Configuracion en ``configs/sinteticos.yaml`` (seccion ``laboratorio``). No edita
nada de ``evaluacion``, ``detectores`` ni ``benchmark``: solo los importa.
CLI: ``python -m regimenes.sinteticos.laboratorio --help``.
"""

from regimenes.sinteticos.laboratorio.escenarios import (
    Celda,
    atenuar,
    celdas,
    config_laboratorio,
    detectores_pista,
    secuencias,
    ventanas_verdad,
)
from regimenes.sinteticos.laboratorio.metricas import metricas_trayectoria

__all__ = [
    "Celda",
    "atenuar",
    "celdas",
    "config_laboratorio",
    "detectores_pista",
    "secuencias",
    "ventanas_verdad",
    "metricas_trayectoria",
]
