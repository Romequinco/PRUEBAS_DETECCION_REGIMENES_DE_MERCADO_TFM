"""Generadores de escenarios sinteticos de mercado (ESQUELETO, sin logica).

Objetivo: generar trayectorias multivariantes de features con regimenes
conocidos para (a) estresar los detectores con verdad-terreno exacta y
(b) ampliar la muestra de crisis. Estructura:

- ``base``         interfaz abstracta ``Generador`` (fit/sample/name).
- ``registry``     registro nombre -> clase de generador (vacio).
- ``parametricos`` generadores parametricos (p. ej. MS-VAR, bootstrap por regimen).
- ``neuronales``   generadores neuronales (p. ej. TimeGAN, difusion).
- ``validacion``   metricas de fidelidad/utilidad/memorizacion de lo generado.

Configuracion en ``configs/sinteticos.yaml``; salidas en ``data/sinteticos``
y ``results/sinteticos`` (ver ``regimenes.rutas``).
"""

from regimenes.sinteticos.base import Generador

__all__ = ["Generador"]
