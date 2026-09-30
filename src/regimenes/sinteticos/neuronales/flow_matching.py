"""Flow matching condicional del siguiente bloque (caminos rectos).

Que hace
--------
Aprende el campo de velocidades que transporta ruido gaussiano hasta el
siguiente bloque de ``largo_bloque`` sesiones, condicionado a las
``largo_contexto`` sesiones anteriores y al regimen de cada dia del bloque.
Para entrenar no hay que integrar nada: por cada ventana se sortean un ruido
``x_0 ~ N(0, I)`` y un instante ``t ~ U(0, 1)``, se toma el punto del segmento
``x_t = (1 - t) x_0 + t x_1`` y se hace regresion de minimos cuadrados sobre la
velocidad del segmento, ``x_1 - x_0`` (Lipman et al. 2023; Liu et al. 2023).
Muestrear es soltar una particula en ``x_0`` e integrar ``dx/dt = v(x, t)`` de
0 a 1; el bloque generado pasa a ser contexto del siguiente (``encadenar``).

Es un port a torch puro del generador del taller previo (Keras con bucle de
torch), con tres cambios: (1) modelo de siguiente bloque en lugar de ventana
suelta de 60 x 20 con una etiqueta (la dimension baja de 1201 a 126 en la pista
A y 231 en la B, sin PCA); (2) red residual con FiLM y atajo de ganancia
compartida con ``difusion`` (``_redes_flujo.RedCondicional``) en lugar de un MLP
con todo concatenado; (3) espacio del modelo con ancla de nivel y compresion
``arcsinh`` de las colas. Se conservan las lecciones del taller: AdamW con
decaimiento coseno, recorte de gradiente a norma 1, EMA 0,999 de los pesos (con
calentamiento) y validacion con numeros aleatorios comunes. Todo lo comun
(transformacion, tope de seguridad, equilibrio de lotes, validacion, historial)
esta documentado en ``_redes_flujo``.

Que supone
----------
- Que el siguiente bloque depende del pasado solo a traves de las ultimas
  ``largo_contexto`` sesiones y del regimen del propio bloque.
- Que el regimen es exogeno: no se genera, se impone (o lo simula la base con
  la matriz de transicion de train).
- Un MLP sobre el bloque aplanado: cada (dia, columna) es una coordenada mas,
  sin sesgo inductivo temporal.

Que reproduce y que no (pista A hasta 2006, 50 x 2520, semilla 42)
-----------------------------------------------------------------
(Cifras del humo de desarrollo, no las vigentes: las del muestreo de
``configs/sinteticos.yaml`` (100 x 2520, cadena estacionaria compartida) salen de
``notebooks/15_sinteticos_generadores`` y quedan en
``results/sinteticos/generadores/sanidad_generadores.csv``; si difieren, manda el notebook.)

Reproduce: desviacion por columna y regimen (sintetico / real entre 0,88 y
1,08 en las seis columnas de trabajo; ``SP500_ret`` 0,97 en calma y 1,08 en
crisis), cuantiles extremos de ``SP500_ret`` en crisis (0,1 %: -7,5 sigmas
frente a -7,3; 99,9 %: 6,9 frente a 5,6), persistencia de los escalones
mensuales (autocorrelacion a 1 dia 0,996-0,997 frente a 0,998-0,999) y
estabilidad al encadenar 120 bloques (la desviacion de los dos ultimos anos es
0,81-0,94 veces la historica; 2e-5 de las coordenadas tocan el tope).
No reproduce: las colas en calma quedan cortas (curtosis 2,2 frente a 4,7;
cuantil 0,1 % -3,3 frente a -4,0) y ningun evento del tamano de octubre de 1987
(minimo -13,8 sigmas frente a -24,4: una sola observacion no se aprende); la
agrupacion de volatilidad se queda a medias (autocorrelacion de ``|ret|`` 0,11
frente a 0,22) y con ella la volatilidad realizada derivada (``SP500_vol_z``
0,76-0,88 de su desviacion); los escalones mensuales salen como series suaves,
no como escalones (95 % de dias sin cambio en el real, 0 % en el sintetico); la
media por regimen de las columnas persistentes se desvia hasta 0,56 sigmas.

Como leer la curva de convergencia
----------------------------------
La perdida es un MSE por coordenada sobre datos de varianza 1, con dos anclas
(``anclas_``): **2,0** es el predictor trivial ``v = 0`` (la red recien creada;
``perdida_val_inicial`` lo mide: 1,94 = 1 + varianza de la cola de validacion) y
**pi/2 = 1,571** es el suelo de datos sin ninguna estructura. Por debajo de
1,571 se esta aprendiendo dependencia real (entre dias, entre columnas, con el
contexto y el regimen). La perdida NO tiende a cero: el minimo es la varianza
condicional de ``x_1 - x_0`` dado ``x_t``. En la pista A: 1,69 -> 0,97 en ajuste
y 1,48 -> 1,04 en validacion (minimo en la epoca 35 de 40, que es la que se
conserva); por regimen, 0,85 en calma y 1,34 en crisis. Una perdida baja no
garantiza buenas muestras (el error se integra a lo largo de la trayectoria y
luego de 120 bloques encadenados): la calidad se mide en los notebooks 16-18.
Mas epocas bajan la perdida de ajuste y empeoran las muestras (ver
``restaurar_mejor`` en ``_redes_flujo``).

Parametros propios: ``n_pasos`` y ``metodo_integracion`` (``"heun"`` o
``"euler"``). El integrador no es el cuello de botella: Heun con 25 pasos, Heun
con 10 y Euler con 50 dan sd de ``SP500_ret`` 0,97/1,08, 0,99/1,08 y 0,93/1,03
(calma/crisis); el error que queda es del campo aprendido.

Coste (CPU, un hilo): ~0,78 M de parametros, 40 epocas x 38 pasos; ajuste de la
pista A en 70-100 s y muestreo de 50 x 2520 en 14-24 s.

Referencias: Lipman, Chen, Ben-Hamu, Nickel y Le (2023), "Flow Matching for
Generative Modeling", ICLR; Liu, Gong y Liu (2023), "Flow Straight and Fast:
Learning to Generate and Transfer Data with Rectified Flow", ICLR.
"""

from __future__ import annotations

import math

from regimenes.sinteticos.neuronales._redes_flujo import PARAMS_COMUNES, GeneradorRuidoABloque
from regimenes.sinteticos.registry import registrar


@registrar
class FlowMatching(GeneradorRuidoABloque):
    """Campo de velocidades ``v(x_t, t | contexto, regimen)`` del siguiente bloque (CFM, camino recto)."""

    nombre = "flow_matching"
    familia = "neuronales"
    PARAMS = {**PARAMS_COMUNES, "epocas": 40, "n_pasos": 25, "metodo_integracion": "heun"}
    PARAMS_RAPIDOS = {
        "epocas": 40, "tam_lote": 64, "ancho": 64, "n_bloques": 2, "dim_contexto": 32,
        "dim_tiempo": 16, "n_pasos": 8,
    }

    def _anclas(self) -> dict[str, float]:
        return {"predictor_trivial": 2.0, "sin_estructura": math.pi / 2.0}

    def _objetivo(self, x1, ruido, u):
        # el objetivo CFM: punto del segmento ruido -> dato y su velocidad (constante en t)
        return (1.0 - u) * ruido + u * x1, u, x1 - ruido

    def _integrar(self, prediccion, ruido, tope):
        """Integra ``dx/dt = v(x, t)`` de t=0 (ruido) a t=1 (dato) con Euler o Heun."""
        metodo = str(self.metodo_integracion).lower()
        if metodo not in {"euler", "heun"}:
            raise ValueError("metodo_integracion debe ser 'euler' o 'heun'.")
        n_pasos = max(int(self.n_pasos), 1)
        dt = 1.0 / n_pasos
        x = ruido
        for k in range(n_pasos):
            t0 = k * dt
            v1 = prediccion(x, t0)
            if metodo == "euler":
                x = x + dt * v1
            else:
                # Heun (RK2): paso de Euler y correccion con la media de las dos velocidades
                v2 = prediccion(x + dt * v1, min(t0 + dt, 1.0))
                x = x + 0.5 * dt * (v1 + v2)
        return x
