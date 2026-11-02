"""Aumento de datos real + sintetico (notebook 21).

Pregunta: entrenar los detectores del benchmark con real + sintetico mejora su
deteccion de las crisis REALES fuera de muestra? Todo lo que se fija a priori
(alcance, anclas, brazos, proporciones, contrastes y regla para el 22) esta en
``configs/sinteticos.yaml`` -> ``aumento`` (pre-registro del 2026-10-15).

Modulos:

- ``config``: lectura de la seccion ``aumento``, rutas y semillas estables.
- ``anclas``: generadores reentrenados por ancla anual sin fuga (crisis
  reconocidas, recorte en ventanas abiertas, validacion por ancla) y registro.
- ``muestras``: reservorios de trayectorias por ancla y brazo, y el bloque
  sintetico de cada pliegue (proporciones anidadas).
- ``envoltorio``: ``DetectorAumentado``, que ajusta el detector con
  [bloque sintetico] + [train real] y delega todo lo demas.
- ``ejecucion``: trabajos, cache por huella, piloto de coste y ejecucion
  paralela con presupuesto de horas (CLI ``python -m regimenes.sinteticos.aumento``).
- ``analisis``: senal de voto, segmentos, exceso sobre el nulo, permutacion
  exacta con Holm, bootstrap pareado e hipotesis H1-H5; lecturas del notebook
  (cobertura, anclas, mapa, dosis-respuesta, diagnosticos) sobre resultados parciales.
- ``figuras``: figuras del notebook 21 (diseno, H1, mapa, H2, H3-H5, diagnosticos).

No modifica ``regimenes.evaluacion`` ni los detectores: ``walk_forward`` se usa
tal cual con una fabrica que devuelve el envoltorio, asi que la huella del
benchmark no cambia.
"""

from regimenes.sinteticos.aumento.config import config_aumento, detectores_pista, fuentes

__all__ = ["config_aumento", "detectores_pista", "fuentes"]
