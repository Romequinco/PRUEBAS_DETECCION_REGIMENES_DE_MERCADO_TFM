"""Difusion condicional del siguiente bloque (prediccion de ruido, muestreo DDIM).

Que hace
--------
Aprende a quitar ruido al siguiente bloque de ``largo_bloque`` sesiones,
condicionado a las ``largo_contexto`` anteriores y al regimen de cada dia del
bloque. El proceso directo tiene forma cerrada (Ho, Jain y Abbeel 2020)::

    x_t = sqrt(alfa_barra_t) x_0 + sqrt(1 - alfa_barra_t) eps,   eps ~ N(0, I)

con ``pasos_difusion`` = 1000 niveles y planificador coseno; la red predice el
``eps`` inyectado (MSE). Se muestrea con DDIM determinista (Song, Meng y Ermon
2021) sobre una subsecuencia de ``n_pasos`` = 50 niveles: en cada paso se
despeja ``x_0`` de la prediccion de ruido y se recombina con el nivel de ruido
siguiente. El regimen se anula (one-hot a cero) en una fraccion
``prob_dropout_condicion`` de los ejemplos de cada lote, de modo que la misma
red aprende el modelo con y sin regimen y ``guiado`` puede extrapolar entre
ambos al muestrear (classifier-free guidance, Ho y Salimans 2022); el contexto
no se anula nunca. ``guiado = 1`` (defecto) es el condicional puro y no cuesta
la segunda pasada.

Es un port a torch puro del generador del taller previo (Keras), que alli fue
mediocre y muy sensible al recorte de ``x_0``. Cambios: (1) modelo de siguiente
bloque sin PCA (126 dimensiones en la pista A, 231 en la B, en lugar de 1201);
(2) EMA 0,999 de los pesos, que el original no tenia; (3) la cadena inversa
empieza en ``senal_minima`` en lugar de en ruido puro, y al acotar ``x_0`` se
recalcula ``eps``; (4) red con atajo de ganancia, compartida con
``flow_matching``; (5) espacio del modelo con ancla de nivel y compresion
``arcsinh``. Todo lo comun esta documentado en ``_redes_flujo``.

El recorte, y por que ya no importa (mini-barrido, pista A, semilla 42)
-----------------------------------------------------------------------
Los primeros pasos de la cadena inversa dividen por ``sqrt(alfa_barra)``; en el
ultimo nivel del planificador coseno eso es ~1/0,003, asi que cualquier error
de la red se multiplica por 300 y hay que acotar ``x_0``. Ese era el origen de
la sensibilidad del taller (sin recorte la volatilidad salia x7,7; con +-3
recortaba colas). Cociente de desviaciones sintetico / real de ``SP500_ret``
(calma / crisis):

==============  ================  ==========  ==========  ==========
senal_minima    recorte = None    1,0         1,5         3,0
==============  ================  ==========  ==========  ==========
0 (ruido puro)  no finito         2,37/2,92   3,03/4,29   5,70/7,63
0,02            --                1,30/1,91   1,35/2,37   --
0,05            --                --          1,01/1,12   --
0,1 (defecto)   0,99/0,94         0,99/0,94   0,99/0,94   0,99/0,94
0,2             --                --          0,94/0,82   --
==============  ================  ==========  ==========  ==========

La decision que importa no es el valor del recorte sino donde empieza la
cadena: arrancando en el nivel cuya tasa de senal ``sqrt(alfa_barra)`` es 0,1
(la amplificacion maxima baja de 300 a 10 y ``x_t`` sigue siendo ruido al 99 %
en varianza) el resultado es el mismo con cualquier recorte, incluido
ninguno. ``recorte = 1,5`` queda como red de seguridad que no mutila nada: deja
pasar hasta 1,5 veces el extremo historico de cada coordenada y lo tocan 3e-4
de las coordenadas (casi todas por el tope de nivel). Con 0,2 se pierde
varianza (la cadena arranca demasiado tarde para un ``x_T`` gaussiano); con
0,05 las colas de crisis engordan (curtosis 14 frente a 3) a costa de minimos
de -30 sigmas.

Que supone
----------
Lo mismo que ``flow_matching``: dependencia del pasado solo a traves del
contexto, regimen exogeno e impuesto, y un MLP sobre el bloque aplanado.

Que reproduce y que no (pista A hasta 2006, 50 x 2520, semilla 42)
-----------------------------------------------------------------
(Cifras del humo de desarrollo, no las vigentes: las del muestreo de
``configs/sinteticos.yaml`` (100 x 2520, cadena estacionaria compartida) salen de
``notebooks/15_sinteticos_generadores`` y quedan en
``results/sinteticos/generadores/sanidad_generadores.csv``; si difieren, manda el notebook.)

Reproduce: desviacion por columna y regimen (0,78-1,20 en las seis columnas de
trabajo; ``SP500_ret`` 0,99 en calma y 0,94 en crisis), persistencia de los
escalones mensuales (autocorrelacion a 1 dia 0,996-0,997) y estabilidad al
encadenar 120 bloques (desviacion de los dos ultimos anos 0,92-1,09 veces la
historica).
No reproduce: colas cortas en los dos regimenes (curtosis de ``SP500_ret`` 1,6
y 3,0 frente a 4,7 y 47,8; cuantil 0,1 % en crisis -5,9 sigmas frente a -7,3),
agrupacion de volatilidad debil (autocorrelacion de ``|ret|`` 0,09 frente a
0,22), escalones suavizados y medias por regimen de las columnas persistentes
desviadas hasta 0,44 sigmas. En la pista B (2702 x 11, una sola gran crisis y
validacion sin crisis) queda claramente por debajo de ``flow_matching``: la
volatilidad en crisis sale a 0,58-0,85 de la real en la mayoria de columnas y
el nivel del VIX en crisis 0,8 sigmas por debajo. Subir ``guiado`` lo compensa
solo en parte (1,5 -> 0,76 en ``SP500_ret``) y en la pista A lo exagera (2,0 ->
1,45 y curtosis 50): se deja en 1.

Como leer la curva de convergencia
----------------------------------
La perdida es el MSE de ``eps`` por coordenada. El ancla es **1,0**: la red
recien creada predice cero y ``E[eps^2] = 1`` (``perdida_val_inicial`` lo mide:
0,9996). No tiende a cero: en los niveles de mucho ruido ``x_t`` apenas informa
de ``eps`` y el mejor predictor posible sigue fallando; unos datos sin ninguna
estructura se quedarian en la media de ``alfa_barra`` (~0,5). En la pista A:
0,62 -> 0,30 en ajuste y 0,44 -> 0,30 en validacion (0,27 en calma, 0,35 en
crisis), monotona y sin senal de sobreajuste en 60 epocas (en una prueba con
120 bajo 0,001 mas). Dos avisos: la perdida de difusion y la de flow matching no son
comparables entre si, y esta curva no dice nada de los parametros de muestreo
(``senal_minima``, ``n_pasos``, ``guiado``), que son los que deciden la calidad
(ver la tabla de arriba: misma red, de 0,99 a no finito).

Parametros propios: ``pasos_difusion``, ``n_pasos`` (20 pasos dan sd 0,93/0,88;
200 dan 1,02/0,96), ``senal_minima``, ``prob_dropout_condicion`` y ``guiado``.

Coste (CPU, un hilo): ~0,78 M de parametros, 60 epocas x 38 pasos; ajuste de la
pista A en 100-140 s y muestreo de 50 x 2520 en 13-20 s (el doble con guiado).

Referencias: Ho, Jain y Abbeel (2020), "Denoising Diffusion Probabilistic
Models", NeurIPS; Song, Meng y Ermon (2021), "Denoising Diffusion Implicit
Models", ICLR; Ho y Salimans (2022), "Classifier-Free Diffusion Guidance";
Nichol y Dhariwal (2021), "Improved Denoising Diffusion Probabilistic Models"
(planificador coseno).
"""

from __future__ import annotations

import math

import numpy as np

from regimenes.sinteticos.neuronales._redes_flujo import PARAMS_COMUNES, GeneradorRuidoABloque, torch
from regimenes.sinteticos.registry import registrar


def planificador_coseno(pasos: int, desplazamiento: float = 0.008) -> np.ndarray:
    """``alfa_barra`` (longitud ``pasos``) del planificador coseno de Nichol y Dhariwal (2021).

    ``alfa_barra[0]`` ~ 1 (casi sin ruido) y ``alfa_barra[-1]`` ~ 0 (ruido puro).
    Se pasa por las betas para acotarlas (la ultima tenderia a 1) y se recorta
    por abajo para que ``1 / sqrt(alfa_barra)`` no explote al estimar ``x_0``.
    """
    t = np.arange(pasos + 1, dtype=np.float64) / pasos
    f = np.cos((t + desplazamiento) / (1.0 + desplazamiento) * math.pi / 2.0) ** 2
    betas = np.clip(1.0 - f[1:] / f[:-1], 1e-8, 0.999)
    return np.clip(np.cumprod(1.0 - betas), 1e-5, 1.0)


@registrar
class Difusion(GeneradorRuidoABloque):
    """Denoiser ``eps(x_t, t | contexto, regimen)`` del siguiente bloque, muestreo DDIM."""

    nombre = "difusion"
    familia = "neuronales"
    PARAMS = {
        **PARAMS_COMUNES,
        "pasos_difusion": 1000,
        "n_pasos": 50,
        "senal_minima": 0.1,
        "prob_dropout_condicion": 0.1,
        "guiado": 1.0,
    }
    PARAMS_RAPIDOS = {
        "epocas": 40, "tam_lote": 64, "ancho": 64, "n_bloques": 2, "dim_contexto": 32,
        "dim_tiempo": 16, "n_pasos": 12,
    }

    def _anclas(self) -> dict[str, float]:
        return {"red_nula": 1.0}

    def _preparar(self) -> None:
        if int(self.n_pasos) > int(self.pasos_difusion):
            raise ValueError("n_pasos (muestreo) no puede superar pasos_difusion (entrenamiento).")
        self.alfa_barra_ = torch.as_tensor(planificador_coseno(int(self.pasos_difusion)), dtype=torch.float32)

    def _objetivo(self, x1, ruido, u):
        # proceso directo en forma cerrada: x_t = sqrt(ab) x_0 + sqrt(1 - ab) eps; se regresa eps
        T = len(self.alfa_barra_)
        paso = torch.clamp((u * T).long(), max=T - 1)
        alfa = self.alfa_barra_[paso]
        return torch.sqrt(alfa) * x1 + torch.sqrt(1.0 - alfa) * ruido, (paso + 1).float() / T, ruido

    def _integrar(self, prediccion, ruido, tope):
        """DDIM determinista (eta = 0) sobre una subsecuencia de ``n_pasos`` de los ``T`` pasos."""
        T = len(self.alfa_barra_)
        senal = np.sqrt(self.alfa_barra_.numpy())
        # la cadena inversa arranca en el nivel mas ruidoso cuya tasa de senal es >= senal_minima
        validos = np.flatnonzero(senal >= float(self.senal_minima or 0.0))
        inicio = int(validos.max()) if len(validos) else 0
        pasos = np.unique(np.round(np.linspace(0, inicio, int(self.n_pasos))).astype(int))[::-1]
        x = ruido
        for i, paso in enumerate(pasos):
            alfa = float(self.alfa_barra_[paso])
            alfa_previa = float(self.alfa_barra_[pasos[i + 1]]) if i + 1 < len(pasos) else 1.0
            eps = prediccion(x, (paso + 1) / T)
            x0 = (x - math.sqrt(1.0 - alfa) * eps) / math.sqrt(alfa)
            if tope is not None:
                # los primeros pasos dividen por sqrt(alfa) ~ 0 y amplifican el error de la red: se acota
                # x_0 y se recalcula eps para que (x_0, eps) sigan siendo coherentes con x_t; si solo se
                # acotara x_0, un eps desbocado seguiria entrando en el paso siguiente
                x0 = torch.maximum(torch.minimum(x0, tope), -tope)
                eps = (x - math.sqrt(alfa) * x0) / math.sqrt(1.0 - alfa)
            x = math.sqrt(alfa_previa) * x0 + math.sqrt(1.0 - alfa_previa) * eps
        return x
