"""Generadores de escenarios sinteticos de mercado con regimen conocido.

Objetivo: generar trayectorias multivariantes de features con la secuencia de
regimen conocida para (a) estresar los detectores con verdad-terreno exacta y
(b) ampliar la escasa muestra de crisis. Estructura:

- ``base``         interfaz abstracta ``Generador`` (fit/sample/name).
- ``comun``        ``GeneradorBase``: fontaneria comun; cada generador solo
                   implementa ``_fit`` y ``_sample`` sobre matrices estandarizadas.
- ``datos``        panel de ajuste, regimen de referencia (ventanas de crisis,
                   nunca un detector), rachas y cadena de Markov.
- ``espacio``      espacio de trabajo frente a publico: las features deterministas
                   del S&P 500 se re-derivan, no se modelan.
- ``bloques``      troceo y encadenado de bloques para los neuronales.
- ``persistencia`` trayectorias (parquet largo) y fichas de ajuste.
- ``registry``     registro con catalogo perezoso nombre -> modulo.
- ``parametricos`` / ``neuronales``  un fichero por generador (los neuronales
                   requieren el extra ``[deep]``).
- ``validacion``   metricas de fidelidad/utilidad/memorizacion (notebook 16).

Configuracion en ``configs/sinteticos.yaml``; salidas en ``data/sinteticos``
y ``results/sinteticos`` (ver ``regimenes.rutas``). Importar este paquete no
importa torch ni el benchmark (imports perezosos).
"""

from regimenes.sinteticos.base import Generador
from regimenes.sinteticos.comun import GeneradorBase
from regimenes.sinteticos.registry import crear, disponibles, registrar

__all__ = ["Generador", "GeneradorBase", "registrar", "crear", "disponibles"]
