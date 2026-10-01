"""Validacion de trayectorias sinteticas (notebook 16).

Compara cada generador con el **tramo de entrenamiento real** con el que se
ajusto (nunca con el tramo posterior a ``fin_train``) en varias dimensiones; los
umbrales de admision viven en ``configs/sinteticos.yaml`` (``validacion.umbrales``)
y se fijaron antes de ver resultados; los cambios posteriores de reglas y
niveles estan declarados y fechados en los comentarios de esa seccion.

- ``fidelidad``     marginales por regimen, dependencia temporal y cruzada dentro
                    de regimen (hechos estilizados) y regimenes/condicionamiento.
- ``discriminador`` test de dos muestras con clasificador real frente a sintetico.
- ``utilidad``      TSTR (ajustar el detector en sintetico, evaluar en real)
                    frente a TRTR, con las metricas por evento del benchmark.
- ``memorizacion``  distancia al vecino real mas cercano con indice integro.
- ``veredicto``     combinacion con los umbrales en los niveles de ``validacion.niveles``
                    (apto_laboratorio, apto_aumento, laboratorio_condicionado).

Importar este subpaquete no importa torch ni el benchmark.
"""

from regimenes.sinteticos.validacion.discriminador import discriminador
from regimenes.sinteticos.validacion.fidelidad import (
    condicionamiento,
    distancia_correlaciones,
    fidelidad_dependencia,
    fidelidad_marginal,
    fidelidad_regimenes,
)
from regimenes.sinteticos.validacion.memorizacion import memorizacion
from regimenes.sinteticos.validacion.utilidad import referencia_trtr, unir_utilidad, utilidad_tstr
from regimenes.sinteticos.validacion.veredicto import veredicto

__all__ = [
    "fidelidad_marginal",
    "fidelidad_dependencia",
    "fidelidad_regimenes",
    "distancia_correlaciones",
    "condicionamiento",
    "discriminador",
    "utilidad_tstr",
    "referencia_trtr",
    "unir_utilidad",
    "memorizacion",
    "veredicto",
]
