"""Base comun de los generadores de "ruido -> bloque": flow matching y difusion.

Por que existe: flow matching y difusion son el MISMO transporte de ruido a
datos con dos parametrizaciones distintas (camino recto y velocidad frente a
camino de varianza preservada y ruido). Para que la comparacion entre ambos en
los notebooks 16-18 mida solo esa diferencia, todo lo demas es literalmente el
mismo codigo y vive aqui: la red, la transformacion de los datos, el reparto
ajuste/validacion, el equilibrio de lotes, la media movil de pesos (EMA), el
bucle de entrenamiento y el encadenado de bloques. Cada generador aporta solo
dos metodos: ``_objetivo`` (como se mezcla dato y ruido y que se regresa) y
``_integrar`` (como se va de ruido a dato al muestrear).

Las cifras de este docstring son del humo con datos reales: pista A hasta
2006-12-31 (11267 x 6 columnas de trabajo, 9500 ventanas de ajuste), 50
trayectorias de 2520 sesiones con la cadena de regimenes simulada, semilla 42.
"sd" es el cociente de desviaciones sintetico / real de ``SP500_ret`` dentro de
cada regimen. No son las cifras vigentes del proyecto: las del muestreo de
``configs/sinteticos.yaml`` (100 trayectorias, cadena estacionaria compartida)
salen de ``notebooks/15_sinteticos_generadores`` (§5 y §6); si difieren, manda
el notebook.

Formulacion ("siguiente bloque")
--------------------------------
Se modela ``p(bloque | contexto, regimen del bloque)``: el bloque son las
``largo_bloque`` sesiones siguientes (aplanadas, ``L_blk * d`` coordenadas), el
contexto las ``largo_contexto`` anteriores y el regimen un valor por dia del
bloque (asi una transicion calma -> crisis a mitad de bloque es una condicion
mas, no un caso especial). Las trayectorias largas salen de ``encadenar``.

Espacio del modelo (transformacion invertible, ajustada solo con train)
-----------------------------------------------------------------------
1. **Ancla de nivel** en las columnas persistentes (autocorrelacion a 1 dia de
   train > ``umbral_persistencia``: escalones mensuales, niveles de VIX...): se
   modela ``bloque - ultima fila del contexto``. Una red de ruido genera cada
   coordenada con un error pequeno pero independiente; sobre el NIVEL de una
   serie que casi no se mueve ese error destruiria la autocorrelacion, sobre la
   DESVIACION respecto al ultimo valor conocido queda reescalado por la
   desviacion tipica (pequena) de esa desviacion. Autocorrelacion a 1 dia de
   los escalones mensuales: 0,998-0,999 real, 0,996-0,997 con ancla, 0,987-0,992
   sin ella (medido con ``a = 2`` y 150 epocas).
2. **Colas** (``colas``), por coordenada (dia del bloque x columna):

   - ``"arcsinh"`` (defecto): estandarizar, ``y -> a * arcsinh(y / a)`` con
     ``a = escala_colas`` y volver a estandarizar. Es la identidad para
     ``|y| << a`` y logaritmica en las colas: con ``a = 3`` el -24 sigmas de
     octubre de 1987 queda en ~-8 (antes de reestandarizar) y deja de dominar el MSE, pero NO se recorta
     (la inversa ``a * sinh`` lo devuelve intacto). El precio es que la inversa
     amplifica el error de la red en la cola, tanto mas cuanto menor es ``a``:
     ``a`` es el mando entre colas fieles y cadena estable. Flow matching,
     cuantil 0,1 % de ``SP500_ret`` en crisis (real -7,3 sigmas) y en calma
     (real -4,0): ``a = 2`` -> -11,8 / -3,6 (sd en crisis 1,24); ``a = 3`` ->
     -7,5 / -3,3 (1,08); ``a = 4`` -> -7,9 / -3,2 (1,13); ``a = 8`` -> -6,5 /
     -3,1 (2, 4 y 8 medidos con ``equilibrio = 0,5``). Se toma 3: colas de
     crisis en su sitio, las de calma algo cortas.
   - ``"cuantil"``: gaussianizacion por rangos con inversa = cuantil empirico
     (empates repartidos al azar dentro de su intervalo de rangos). Sobre el
     papel es la opcion limpia (cada coordenada es N(0, 1) exacta); en la
     practica NO se recomienda: la funcion cuantil es casi vertical en el
     extremo (de ~7 a 24 sigmas entre z = 3,7 y z = 3,9) y la red genera
     ``|z| > 3,88`` con frecuencia 6,7e-4 donde lo exacto es 1e-4, asi que
     fabrica un crash de 1987 cada pocos anos: sd 1,23 en calma y 2,51 en
     crisis, curtosis por encima de 200. Se conserva como ablacion documentada.
   - ``"ninguna"``: solo estandarizacion. Las colas dominan el MSE y las
     desviaciones respecto al ancla (casi todas cero mas algun salto) se
     aprenden mal: sd de los escalones mensuales hasta x3,0.

El contexto entra en la red comprimido con el mismo ``arcsinh`` (sin ancla);
ahi no hace falta inversa.

Tope de seguridad, declarado y contado
--------------------------------------
``recorte`` (factor f, parametro de MUESTREO: se puede cambiar tras ``fit``)
acota, por cada lado, (a) cada coordenada generada a ``f`` veces el extremo
historico de esa coordenada en train (desviacion respecto al ancla en las
persistentes) y (b) el nivel resultante a ``f`` veces el extremo historico de
la columna. Con ``f >= 1`` ningun valor observado en train queda fuera, asi que
no mutila las colas reales; solo impide que un bloque desbocado, o un nivel que
deriva bloque tras bloque, saque el contexto de lo que la red ha visto (sin (b),
en la pista B una trayectoria de 50 llego a un nivel de 16 sigmas y la difusion
devolvio NaN). ``None`` lo desactiva. La fraccion de coordenadas que tocan el
tope queda en ``diagnostico_muestreo_["fraccion_recortada"]`` tras cada
``sample`` (del orden de 1e-5 a 3e-4 en la pista A y de 1e-3 a 2e-3 en la B,
casi todo por (b)): si crece, el modelo no ha aprendido el soporte.

Equilibrio de lotes
-------------------
Cada ventana se clasifica en "pura de un regimen" o "mixta" (contiene una
transicion). Las ventanas con crisis o transicion son pocas (8 episodios en la
pista A: 7787 de calma, 1433 de crisis y 280 mixtas), asi que cada epoca se
sortea con probabilidad por categoria proporcional a ``frecuencia ** (1 -
equilibrio)``: 0 = muestreo natural, 1 = todas las categorias por igual. Como
despues se genera CONDICIONANDO en el regimen, cambiar la mezcla no sesga lo
que se aprende de cada categoria; lo que si hace es repetir mas veces las
mismas pocas ventanas de crisis, y eso se paga en memorizacion. Flow matching,
minimo de ``perdida_val``: 1,031 con 0, 1,038 con 0,25 y 1,074 con 0,5 (con
0,5 la perdida de validacion en crisis vuelve a subir desde la epoca ~21). Se
toma 0,25: equilibrio suave.

Validacion interna y seleccion de pesos
---------------------------------------
Se reserva la cola cronologica de las ventanas (``frac_validacion``) con un
hueco de ``largo_bloque + largo_contexto`` ventanas para que ninguna ventana de
validacion comparta dias con una de ajuste. ``perdida_val`` se mide con los
pesos EMA y con numeros aleatorios comunes (mismos ruidos y mismos instantes en
todas las epocas), de modo que lo unico que cambia entre epocas son los pesos.
Con ``restaurar_mejor`` se conservan los pesos EMA de la epoca de menor
``perdida_val`` (``epoca_restaurada_``): con ventanas muy solapadas la red
memoriza pronto (con 150 epocas, flow matching bajaba la perdida de ajuste a
0,82 mientras la de validacion subia de 1,07 a 1,43 y la volatilidad generada
en crisis se iba a x2,7). Es un indicador de convergencia y de sobreajuste, no
una medida de calidad de las muestras. Dos contrapartidas: esa cola NO se usa
para ajustar (``frac_validacion=0`` la recupera a cambio de quedarse sin
``perdida_val`` y sin seleccion de pesos), y puede no contener crisis (pista B
hasta 2017: las 399 ventanas de validacion son de calma, asi que alli la
seleccion de pesos no ve el regimen que mas importa).

Historial (una fila por epoca): ``epoca``, ``perdida`` (media de los lotes,
pesos en curso, mezcla equilibrada), ``perdida_val``, ``perdida_val_calma``,
``perdida_val_crisis`` (ventanas sin / con algun dia de crisis) y ``lr``.
``anclas_`` guarda las referencias para leer la curva, incluida
``perdida_val_inicial`` (la red recien creada, que predice cero).

Reproducibilidad: ajuste y muestreo corren con un hilo de torch mientras duran
(``_torch.GeneradorNeuronal``, que restaura despues los hilos del proceso), asi
que una misma semilla da el mismo resultado bit a bit con cualquier numero de
nucleos.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

from regimenes.sinteticos.bloques import construir_bloques, encadenar
from regimenes.sinteticos.neuronales._torch import (
    GeneradorNeuronal,
    a_numpy,
    a_tensor,
    generador_torch,
    importar_torch,
    sembrar,
)

torch = importar_torch()
nn = torch.nn

# Desfase de la semilla de los numeros aleatorios comunes de validacion.
_DESFASE_VALIDACION = 9973
# Nudos de la funcion cuantil empirica y tramo (en z) con el que se mide la pendiente de la cola.
_NUDOS_CUANTIL = 257
_TRAMO_COLA = 0.5
# Margen maximo (en z) del tope de seguridad mas alla del extremo historico.
_MARGEN_TOPE = 2.0

PARAMS_COMUNES = {
    # formulacion
    "largo_bloque": 21,
    "largo_contexto": 21,
    # espacio del modelo
    "umbral_persistencia": 0.9,
    "colas": "arcsinh",
    "escala_colas": 3.0,
    "recorte": 1.5,
    # red
    "ancho": 256,
    "n_bloques": 3,
    "dim_contexto": 128,
    "dim_tiempo": 64,
    # optimizacion
    "epocas": 60,
    "tam_lote": 256,
    "lr": 1e-3,
    "decaimiento_pesos": 1e-4,
    "recorte_gradiente": 1.0,
    "ema": 0.999,
    "equilibrio": 0.25,
    # validacion interna
    "frac_validacion": 0.15,
    "niveles_validacion": 8,
    "max_muestras_validacion": 1024,
    "restaurar_mejor": True,
}


# ------------------------------------------------------------------ transformaciones


def _limites(minimo: np.ndarray, maximo: np.ndarray, recorte: float) -> tuple[np.ndarray, np.ndarray]:
    """Tope por lado: ``recorte`` veces el extremo historico de ese lado (por coordenada)."""
    return minimo - (recorte - 1.0) * np.abs(minimo), maximo + (recorte - 1.0) * np.abs(maximo)


def _acotar(x: np.ndarray, minimo: np.ndarray, maximo: np.ndarray, recorte: float | None):
    """Aplica el tope de seguridad y cuenta las coordenadas que lo tocan."""
    if not recorte:
        return x, 0
    bajo, alto = _limites(minimo, maximo, float(recorte))
    tocadas = int(((x < bajo) | (x > alto)).sum())
    return np.clip(x, bajo, alto), tocadas


class ColasCuantil:
    """Gaussianizacion por rangos de cada coordenada, con inversa = cuantil empirico.

    ``rangos`` devuelve, por valor, el intervalo de probabilidad ``[p, p + ancho]``
    que ocupa en la distribucion empirica de ajuste (``ancho = k/n`` si el valor
    aparece ``k`` veces); ``z = Phi^-1(p + U * ancho)`` es N(0, 1) exacta. La
    inversa interpola la funcion cuantil en ``_NUDOS_CUANTIL`` nudos equiespaciados
    en ``z`` y, fuera del rango historico, sigue recta con la pendiente de la cola.
    """

    def ajustar(self, y: np.ndarray) -> None:
        n = len(y)
        self.n_ = n
        self.orden_ = np.sort(y, axis=0)
        self.z_max_ = float(ndtri(1.0 - 0.5 / n))
        self.z_ = np.linspace(-self.z_max_, self.z_max_, _NUDOS_CUANTIL)
        indices = np.clip(np.ceil(ndtr(self.z_) * n).astype(int) - 1, 0, n - 1)
        self.x_ = self.orden_[indices]  # (nudos, D): cuantil empirico (funcion escalon) en cada nudo
        salto = max(int(round(_TRAMO_COLA / (self.z_[1] - self.z_[0]))), 1)
        self.pendiente_ = np.stack([
            (self.x_[salto] - self.x_[0]) / (self.z_[salto] - self.z_[0]),
            (self.x_[-1] - self.x_[-1 - salto]) / (self.z_[-1] - self.z_[-1 - salto]),
        ])

    def rangos(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        bajo = np.empty(y.shape)
        alto = np.empty(y.shape)
        for j in range(y.shape[1]):
            bajo[:, j] = np.searchsorted(self.orden_[:, j], y[:, j], side="left")
            alto[:, j] = np.searchsorted(self.orden_[:, j], y[:, j], side="right")
        return bajo / self.n_, (alto - bajo) / self.n_

    def a_modelo(self, base, ancho, u):
        """Tensores ``base``/``ancho`` (float64) y ``u`` uniforme -> z (float32)."""
        minimo = 0.5 / self.n_
        p = torch.clamp(base + u * ancho, minimo, 1.0 - minimo)
        return torch.special.ndtri(p).float()

    def inversa(self, z: np.ndarray, recorte: float | None) -> tuple[np.ndarray, int]:
        x = np.empty(z.shape)
        for j in range(z.shape[1]):
            x[:, j] = np.interp(z[:, j], self.z_, self.x_[:, j])
        x += np.minimum(z + self.z_max_, 0.0) * self.pendiente_[0]
        x += np.maximum(z - self.z_max_, 0.0) * self.pendiente_[1]
        return _acotar(x, self.x_[0], self.x_[-1], recorte)

    def tope(self, recorte: float | None) -> np.ndarray | None:
        """|z| a partir del cual la inversa queda acotada por ``recorte`` (por coordenada)."""
        if not recorte:
            return None
        exceso = (float(recorte) - 1.0) * np.abs(np.stack([self.x_[0], self.x_[-1]]))
        margen = np.where(self.pendiente_ > 0, exceso / np.maximum(self.pendiente_, 1e-12), _MARGEN_TOPE)
        return self.z_max_ + np.minimum(margen, _MARGEN_TOPE).max(axis=0)


class ColasArcsinh:
    """Estandarizar, ``a * arcsinh(y / a)`` (``a = None``: identidad) y volver a estandarizar."""

    def __init__(self, escala: float | None) -> None:
        self.a = float(escala) if escala else None

    def _comprimir(self, y: np.ndarray) -> np.ndarray:
        return y if self.a is None else self.a * np.arcsinh(y / self.a)

    def _directa(self, y: np.ndarray) -> np.ndarray:
        return (self._comprimir((y - self.media1_) / self.escala1_) - self.media2_) / self.escala2_

    def ajustar(self, y: np.ndarray) -> None:
        self.media1_ = y.mean(axis=0)
        escala = y.std(axis=0)
        self.escala1_ = np.where(escala > 1e-8, escala, 1.0)
        z = (y - self.media1_) / self.escala1_
        self.minimo_, self.maximo_ = z.min(axis=0), z.max(axis=0)
        comprimido = self._comprimir(z)
        self.media2_ = comprimido.mean(axis=0)
        escala = comprimido.std(axis=0)
        self.escala2_ = np.where(escala > 1e-8, escala, 1.0)

    def rangos(self, y: np.ndarray) -> tuple[np.ndarray, None]:
        return self._directa(y), None

    def a_modelo(self, base, ancho, u):
        return base.float()

    def inversa(self, z: np.ndarray, recorte: float | None) -> tuple[np.ndarray, int]:
        y = z * self.escala2_ + self.media2_
        if self.a is not None:
            # sinh desborda float64 a partir de ~710: se acota antes; el tope de abajo decide el valor
            y = self.a * np.sinh(np.clip(y / self.a, -50.0, 50.0))
        y, tocadas = _acotar(y, self.minimo_, self.maximo_, recorte)
        return y * self.escala1_ + self.media1_, tocadas

    def tope(self, recorte: float | None) -> np.ndarray | None:
        if not recorte:
            return None
        limite = np.stack(_limites(self.minimo_, self.maximo_, float(recorte)))
        return np.abs((self._comprimir(limite) - self.media2_) / self.escala2_).max(axis=0)


# ------------------------------------------------------------------------------ red


class EmbeddingTiempo(nn.Module):
    """Features de Fourier del instante ``t`` en [0, 1] (senos y cosenos).

    Un escalar entrando directo en una capa densa da una respuesta casi lineal
    en ``t``; el banco de frecuencias geometricas (de 1 a ``frecuencia_max``
    oscilaciones por intervalo) da resolucion a varias escalas a la vez.
    """

    def __init__(self, dimension: int = 64, frecuencia_max: float = 1000.0) -> None:
        super().__init__()
        if dimension % 2:
            raise ValueError("dim_tiempo debe ser par.")
        frecuencias = torch.exp(torch.linspace(0.0, math.log(frecuencia_max), dimension // 2))
        self.register_buffer("frecuencias", 2.0 * math.pi * frecuencias)

    def forward(self, t):
        angulos = t * self.frecuencias[None, :]
        return torch.cat([torch.sin(angulos), torch.cos(angulos)], dim=1)


class BloqueFiLM(nn.Module):
    """Bloque residual cuyo estado normalizado se modula con el contexto (FiLM).

    La condicion (tiempo + contexto + regimen) se reinyecta en cada bloque como
    escala y desplazamiento; arrancan en 1 y 0 (capa a cero), de modo que el
    bloque empieza sin modular.
    """

    def __init__(self, ancho: int, dim_contexto: int) -> None:
        super().__init__()
        self.norma = nn.LayerNorm(ancho)
        self.film = nn.Linear(dim_contexto, 2 * ancho)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.densa1 = nn.Linear(ancho, ancho)
        self.densa2 = nn.Linear(ancho, ancho)
        self.activacion = nn.SiLU()

    def forward(self, h, contexto):
        escala, desplazamiento = self.film(contexto).chunk(2, dim=1)
        z = self.norma(h) * (1.0 + escala) + desplazamiento
        return h + self.densa2(self.activacion(self.densa1(self.activacion(z))))


class RedCondicional(nn.Module):
    """MLP residual ``(x_t, t, condicion) -> vector de la dimension de x_t``.

    La condicion entra dos veces: concatenada a ``x_t`` en la primera capa (el
    contexto es continuo y de dimension alta; la red debe poder combinarlo
    linealmente con el estado) y, junto con el tiempo, como modulacion FiLM de
    cada bloque. La salida es lineal (el objetivo no esta acotado) y suma un
    atajo ``ganancia(tiempo, condicion) * x_t``: la solucion de datos sin
    estructura es exactamente ``x_t`` por un escalar que depende del instante (y
    la volatilidad de un regimen, una ganancia que depende de la condicion), y
    el tronco, que pasa por ``LayerNorm`` y puede ser mas estrecho que ``x_t``,
    no representa bien esa identidad reescalada (sin el atajo la difusion
    generaba con volatilidad x3-6 y sin distinguir regimenes en el panel de
    juguete). Salida y ganancia arrancan a cero: la primera prediccion es el
    vector nulo y la perdida inicial coincide con su ancla teorica.
    """

    def __init__(
        self,
        dim_x: int,
        dim_condicion: int,
        ancho: int = 256,
        n_bloques: int = 3,
        dim_contexto: int = 128,
        dim_tiempo: int = 64,
    ) -> None:
        super().__init__()
        self.emb_tiempo = nn.Sequential(
            EmbeddingTiempo(dim_tiempo), nn.Linear(dim_tiempo, dim_contexto), nn.SiLU()
        )
        self.emb_condicion = nn.Sequential(nn.Linear(dim_condicion, dim_contexto), nn.SiLU())
        self.mezcla = nn.Sequential(nn.Linear(2 * dim_contexto, dim_contexto), nn.SiLU())
        self.entrada = nn.Linear(dim_x + dim_condicion, ancho)
        self.bloques = nn.ModuleList([BloqueFiLM(ancho, dim_contexto) for _ in range(n_bloques)])
        self.norma = nn.LayerNorm(ancho)
        self.salida = nn.Linear(ancho, dim_x)
        self.ganancia = nn.Linear(dim_contexto, dim_x)
        for capa in (self.salida, self.ganancia):
            nn.init.zeros_(capa.weight)
            nn.init.zeros_(capa.bias)

    def forward(self, x, t, condicion):
        contexto = self.mezcla(torch.cat([self.emb_tiempo(t), self.emb_condicion(condicion)], dim=1))
        h = self.entrada(torch.cat([x, condicion], dim=1))
        for bloque in self.bloques:
            h = bloque(h, contexto)
        return self.salida(self.norma(h)) + self.ganancia(contexto) * x


# ------------------------------------------------------------------------ generador


class GeneradorRuidoABloque(GeneradorNeuronal):
    """Base de flow matching y difusion (ver docstring del modulo).

    Las subclases definen ``nombre``, ``PARAMS`` (``PARAMS_COMUNES`` + los suyos),
    ``_objetivo`` e ``_integrar``; opcionalmente ``_preparar`` y ``_anclas``.
    Parametros opcionales de la subclase: ``prob_dropout_condicion`` (dropout
    del regimen al entrenar) y ``guiado`` (classifier-free guidance al muestrear).

    Parametros comunes (``PARAMS_COMUNES``)
    ---------------------------------------
    largo_bloque, largo_contexto : sesiones generadas por bloque y sesiones de contexto.
    umbral_persistencia : autocorrelacion a 1 dia a partir de la cual una columna usa ancla
        de nivel (``None``: ninguna).
    colas, escala_colas : transformacion de colas y su escala ``a`` (tambien comprime el contexto).
    recorte : factor del tope de seguridad sobre el extremo historico (``None``: sin tope).
    ancho, n_bloques, dim_contexto, dim_tiempo : tamano de la red.
    epocas, tam_lote, lr, decaimiento_pesos, recorte_gradiente : AdamW con decaimiento coseno
        de ``lr`` a ``0,02 lr`` y recorte de la norma del gradiente.
    ema : factor de la media movil de pesos (con calentamiento); los pesos EMA son los que quedan.
    equilibrio : 0 = lotes con la mezcla natural de ventanas, 1 = categorias equiprobables.
    frac_validacion, niveles_validacion, max_muestras_validacion : cola de validacion y rejilla
        de numeros aleatorios comunes.
    restaurar_mejor : conservar los pesos EMA de la epoca de menor ``perdida_val``.

    Attributes (tras ``fit``)
    -------------------------
    red_ : RedCondicional            red con los pesos EMA, en modo evaluacion
    persistentes_ : ndarray (d,)     columnas modeladas respecto al ancla de nivel
    colas_ : ColasArcsinh | ColasCuantil   transformacion ajustada (L_blk * d coordenadas)
    nivel_min_, nivel_max_ : ndarray (d,)  extremos historicos del nivel (tope de seguridad)
    anclas_ : dict                   referencias de la perdida (ver cada generador)
    reparto_ : dict                  ventanas de ajuste / validacion y categorias
    epoca_restaurada_ : int          epoca cuyos pesos EMA se conservan
    n_parametros_ : int
    diagnostico_muestreo_ : dict     campos comunes de ``GeneradorBase`` mas ``fraccion_recortada``
                                     y ``coordenadas`` del ultimo ``sample``
    """

    familia = "neuronales"

    # ------------------------------------------------------------------ hooks
    def _preparar(self) -> None:
        """Precalculos de la parametrizacion (p. ej. el planificador de ruido)."""

    def _objetivo(self, x1, ruido, u):
        """``(x_t, t, objetivo)`` dado el dato ``x1``, el ``ruido`` y ``u`` uniforme en [0, 1)."""
        raise NotImplementedError

    def _integrar(self, prediccion, ruido, tope):
        """De ``ruido`` a dato con ``prediccion(x, t: float)``; devuelve el bloque en espacio del modelo."""
        raise NotImplementedError

    def _anclas(self) -> dict[str, float]:
        """Referencias teoricas de la perdida de esta parametrizacion."""
        return {}

    # --------------------------------------------------------- espacio modelo
    def _ancla_nivel(self, contextos: np.ndarray) -> np.ndarray:
        """Ultima fila del contexto en las columnas persistentes (0 en el resto), por dia del bloque."""
        n, d = len(contextos), contextos.shape[2]
        ancla = np.zeros((n, d))
        if contextos.shape[1] and self.persistentes_.any():
            ancla[:, self.persistentes_] = contextos[:, -1, self.persistentes_]
        return np.tile(ancla[:, None, :], (1, int(self.largo_bloque), 1)).reshape(n, -1)

    def _de_modelo(self, z: np.ndarray, contextos: np.ndarray) -> tuple[np.ndarray, int]:
        """Espacio del modelo ``(n, L_blk * d)`` -> bloques ``(n, L_blk, d)`` y coordenadas acotadas."""
        y, tocadas = self.colas_.inversa(z, self.recorte)
        y = (y + self._ancla_nivel(contextos)).reshape(len(y), int(self.largo_bloque), -1)
        # el tope anterior acota la desviacion respecto al ancla; este, el NIVEL que se va acumulando
        y, en_nivel = _acotar(y, self.nivel_min_, self.nivel_max_, self.recorte)
        return y, tocadas + en_nivel

    def _condicion(self, contextos: np.ndarray, reg_bloque: np.ndarray) -> np.ndarray:
        """``[contexto comprimido aplanado, regimen por dia en one-hot]`` (float32)."""
        n = len(contextos)
        uno = np.zeros((n, reg_bloque.shape[1], self.n_regimenes_), dtype=np.float32)
        np.put_along_axis(uno, reg_bloque[:, :, None], 1.0, axis=2)
        a = self.escala_colas
        ctx = contextos if not a else a * np.arcsinh(contextos / a)
        return np.concatenate([ctx.reshape(n, -1).astype(np.float32), uno.reshape(n, -1)], axis=1)

    # ------------------------------------------------------------ preparacion
    def _columnas_persistentes(self, X: np.ndarray) -> np.ndarray:
        d = X.shape[1]
        persistentes = np.zeros(d, dtype=bool)
        if int(self.largo_contexto) < 1 or len(X) < 3 or self.umbral_persistencia is None:
            return persistentes
        for c in range(d):
            a, b = X[:-1, c], X[1:, c]
            if a.std() > 0 and b.std() > 0:
                persistentes[c] = np.corrcoef(a, b)[0, 1] > float(self.umbral_persistencia)
        return persistentes

    def _repartir(self, m: int) -> tuple[np.ndarray, np.ndarray]:
        """Indices de ventanas de ajuste y de validacion (cola cronologica con hueco)."""
        hueco = int(self.largo_bloque) + int(self.largo_contexto)
        n_val = int(round(m * float(self.frac_validacion or 0.0)))
        n_ajuste = m - n_val - hueco
        if n_val < 32 or n_ajuste < 64:
            return np.arange(m), np.arange(0)
        return np.arange(n_ajuste), np.arange(m - n_val, m)

    def _categorias(self, reg_bloque: np.ndarray) -> np.ndarray:
        """Categoria de cada ventana: su regimen si es pura, ``n_regimenes_`` si es mixta."""
        pura = (reg_bloque == reg_bloque[:, :1]).all(axis=1)
        return np.where(pura, reg_bloque[:, 0], self.n_regimenes_)

    def _pesos_equilibrio(self, categorias: np.ndarray) -> np.ndarray | None:
        equilibrio = float(self.equilibrio or 0.0)
        if equilibrio <= 0:
            return None
        conteo = np.bincount(categorias, minlength=self.n_regimenes_ + 1).astype(float)
        frecuencia = conteo / conteo.sum()
        masa = np.where(conteo > 0, frecuencia ** (1.0 - equilibrio), 0.0)
        pesos = masa[categorias] / np.maximum(conteo[categorias], 1.0)
        return pesos / pesos.sum()

    def _crear_colas(self):
        modo = str(self.colas or "ninguna").lower()
        if modo == "cuantil":
            return ColasCuantil()
        if modo in {"arcsinh", "ninguna"}:
            return ColasArcsinh(self.escala_colas if modo == "arcsinh" else None)
        raise ValueError("colas debe ser 'cuantil', 'arcsinh' o 'ninguna'.")

    # -------------------------------------------------------------------- fit
    def _fit(self, X: np.ndarray, reg: np.ndarray, fechas: pd.DatetimeIndex) -> None:
        gen = sembrar(self.random_state)
        rng = self.rng_ajuste()
        self._preparar()
        lb, lc = int(self.largo_bloque), int(self.largo_contexto)
        contextos, bloques, reg_bloque = construir_bloques(X, reg, lb, lc, fechas=fechas)
        idx_aj, idx_val = self._repartir(len(bloques))
        n_aj = len(idx_aj)

        # --- espacio del modelo. Solo la transformacion de colas (``colas_``) se ajusta con las
        # ventanas de ajuste; ``persistentes_`` y ``nivel_min_``/``nivel_max_`` usan TODO train,
        # incluida la cola de validacion interna. No hay fuga posterior al corte (todo es train):
        # el unico efecto es un optimismo ligero de ``perdida_val`` (la cola ha influido en que
        # columnas llevan ancla), y el tope de nivel DEBE cubrir todo train porque es un tope de
        # muestreo, no un estadistico del ajuste.
        self.persistentes_ = self._columnas_persistentes(X)
        self.nivel_min_, self.nivel_max_ = X.min(axis=0), X.max(axis=0)
        crudo = bloques.reshape(len(bloques), -1) - self._ancla_nivel(contextos)
        self.colas_ = self._crear_colas()
        self.colas_.ajustar(crudo[idx_aj])
        base, ancho = self.colas_.rangos(crudo)
        self.colas_.orden_ = None  # solo hacia falta para los rangos; no se guarda con el modelo
        dim_x = crudo.shape[1]

        def tensores(indices):
            b = torch.as_tensor(base[indices], dtype=torch.float64)
            a = None if ancho is None else torch.as_tensor(ancho[indices], dtype=torch.float64)
            return b, a

        def desempate(forma, generador):
            return None if ancho is None else torch.rand(forma, generator=generador, dtype=torch.float64)

        C = self._condicion(contextos, reg_bloque)
        dim_reg = lb * self.n_regimenes_
        categorias = self._categorias(reg_bloque)
        pesos = self._pesos_equilibrio(categorias[idx_aj])
        base_aj, ancho_aj = tensores(idx_aj)
        C_aj = a_tensor(C[idx_aj])
        tam_lote = min(int(self.tam_lote), n_aj)

        # --- validacion con numeros aleatorios comunes
        gen_val = torch.Generator().manual_seed(int(self.random_state or 0) + _DESFASE_VALIDACION)
        validacion = None
        if len(idx_val):
            m_val = min(len(idx_val), int(self.max_muestras_validacion))
            sel = idx_val[np.unique(np.linspace(0, len(idx_val) - 1, m_val).round().astype(int))]
            niveles = int(self.niveles_validacion)
            b, a = tensores(sel)
            validacion = {
                "y": self.colas_.a_modelo(b, a, desempate(b.shape, gen_val)),
                "c": a_tensor(C[sel]),
                "ruido": [torch.randn((len(sel), dim_x), generator=gen_val) for _ in range(niveles)],
                "u": [(k + 0.5) / niveles for k in range(niveles)],
                "crisis": torch.as_tensor(reg_bloque[sel].max(axis=1) > 0),
            }

        red = RedCondicional(
            dim_x, C.shape[1], int(self.ancho), int(self.n_bloques),
            int(self.dim_contexto), int(self.dim_tiempo),
        )
        red_ema = copy.deepcopy(red).requires_grad_(False).eval()
        self.n_parametros_ = int(sum(p.numel() for p in red.parameters()))

        # varianza por coordenada del dato en el espacio del modelo (1 = las anclas son exactas)
        y_aj = self.colas_.a_modelo(base_aj, ancho_aj, desempate(base_aj.shape, gen_val))
        segundo = (y_aj ** 2).mean(dim=1).numpy().astype(float)
        self.anclas_ = {
            **self._anclas(),
            "varianza_ajuste": float(segundo.mean()),
            "varianza_ajuste_equilibrada": float(segundo.mean() if pesos is None else segundo @ pesos),
            "varianza_validacion": float((validacion["y"] ** 2).mean()) if validacion else float("nan"),
        }
        if validacion is not None:
            self.anclas_["perdida_val_inicial"] = self._perdida_validacion(red_ema, validacion)[0]
        n_categorias = self.n_regimenes_ + 1
        self.reparto_ = {
            "ventanas_ajuste": int(n_aj),
            "ventanas_validacion": int(len(idx_val)),
            "categorias_ajuste": np.bincount(categorias[idx_aj], minlength=n_categorias).tolist(),
            "categorias_validacion": np.bincount(categorias[idx_val], minlength=n_categorias).tolist(),
            "pasos_por_epoca": -(-n_aj // tam_lote),
        }

        epocas = int(self.epocas)
        parametros = list(red.parameters())
        optim = torch.optim.AdamW(parametros, lr=float(self.lr), weight_decay=float(self.decaimiento_pesos))
        total_pasos = max(epocas * self.reparto_["pasos_por_epoca"], 1)
        plan = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=total_pasos, eta_min=0.02 * float(self.lr)
        )
        p_drop = float(getattr(self, "prob_dropout_condicion", 0.0) or 0.0)
        ema = float(self.ema or 0.0)
        mejor = (float("inf"), epocas, None)
        paso = 0
        for epoca in range(1, epocas + 1):
            if pesos is None:
                orden = rng.permutation(n_aj)
            else:
                orden = rng.choice(n_aj, size=n_aj, replace=True, p=pesos)
            suma, n_lotes = 0.0, 0
            for ini in range(0, n_aj, tam_lote):
                idx = torch.as_tensor(orden[ini:ini + tam_lote])
                b = base_aj[idx]
                a = None if ancho_aj is None else ancho_aj[idx]
                x1 = self.colas_.a_modelo(b, a, desempate(b.shape, gen))
                c = C_aj[idx]
                if p_drop > 0:
                    # dropout del REGIMEN (one-hot a cero = "sin etiqueta"); el contexto se conserva
                    nulo = torch.rand((len(idx), 1), generator=gen) < p_drop
                    c = torch.cat([c[:, :-dim_reg], c[:, -dim_reg:] * (~nulo)], dim=1)
                ruido = torch.randn(x1.shape, generator=gen)
                u = torch.rand((len(idx), 1), generator=gen)
                x_t, t, objetivo = self._objetivo(x1, ruido, u)
                perdida = torch.mean((red(x_t, t, c) - objetivo) ** 2)
                optim.zero_grad(set_to_none=True)
                perdida.backward()
                if self.recorte_gradiente:
                    torch.nn.utils.clip_grad_norm_(parametros, float(self.recorte_gradiente))
                optim.step()
                plan.step()
                paso += 1
                # EMA con calentamiento: con pocos pasos no se queda pegada a la inicializacion
                factor = min(ema, (1.0 + paso) / (10.0 + paso))
                with torch.no_grad():
                    for sombra, p in zip(red_ema.parameters(), parametros):
                        sombra.mul_(factor).add_(p.detach(), alpha=1.0 - factor)
                suma += float(perdida.detach())
                n_lotes += 1
            media = suma / max(n_lotes, 1)
            if not np.isfinite(media):
                raise FloatingPointError(f"{self.name}: la perdida diverge en la epoca {epoca}.")
            val = (float("nan"),) * 3 if validacion is None else self._perdida_validacion(red_ema, validacion)
            self.registrar(
                epoca=epoca, perdida=media, perdida_val=val[0], perdida_val_calma=val[1],
                perdida_val_crisis=val[2], lr=float(optim.param_groups[0]["lr"]),
            )
            if self.restaurar_mejor and validacion is not None and val[0] < mejor[0]:
                mejor = (val[0], epoca, copy.deepcopy(red_ema.state_dict()))
        if mejor[2] is not None:
            red_ema.load_state_dict(mejor[2])
        self.epoca_restaurada_ = int(mejor[1])
        self.red_ = red_ema

    def _perdida_validacion(self, red, validacion: dict) -> tuple[float, float, float]:
        """Perdida (total, ventanas en calma, ventanas con crisis) sobre la rejilla fija."""
        y, c, crisis = validacion["y"], validacion["c"], validacion["crisis"]
        por_muestra = torch.zeros(len(y))
        with torch.no_grad():
            for ruido, nivel in zip(validacion["ruido"], validacion["u"]):
                x_t, t, objetivo = self._objetivo(y, ruido, torch.full((len(y), 1), nivel))
                por_muestra += ((red(x_t, t, c) - objetivo) ** 2).mean(dim=1)
        por_muestra /= len(validacion["u"])

        def media(mascara) -> float:
            return float(por_muestra[mascara].mean()) if bool(mascara.any()) else float("nan")

        return float(por_muestra.mean()), media(~crisis), media(crisis)

    # ----------------------------------------------------------------- sample
    def _sample(self, reg: np.ndarray, rng: np.random.Generator, contexto: np.ndarray) -> np.ndarray:
        gen = generador_torch(rng)
        lb, lc = int(self.largo_bloque), int(self.largo_contexto)
        dim_reg = lb * self.n_regimenes_
        guiado = float(getattr(self, "guiado", 1.0) or 1.0)
        tope = self.colas_.tope(self.recorte)
        tope = None if tope is None else a_tensor(tope)[None, :]
        red = self.red_
        cuenta = {"tocadas": 0, "total": 0}

        def generar_bloque(ctx: np.ndarray, reg_blk: np.ndarray, _rng) -> np.ndarray:
            condicion = a_tensor(self._condicion(ctx, reg_blk))
            nula = None
            if abs(guiado - 1.0) > 1e-8:
                nula = torch.cat([condicion[:, :-dim_reg], torch.zeros((len(condicion), dim_reg))], dim=1)

            def prediccion(x, t: float):
                instante = torch.full((len(x), 1), float(t))
                salida = red(x, instante, condicion)
                if nula is None:
                    return salida
                base = red(x, instante, nula)
                return base + guiado * (salida - base)

            ruido = torch.randn((len(condicion), lb * self.d_), generator=gen)
            x = self._integrar(prediccion, ruido, tope)
            bloque, tocadas = self._de_modelo(a_numpy(x), ctx)
            cuenta["tocadas"] += tocadas
            cuenta["total"] += x.numel()
            return bloque

        with torch.no_grad():
            salida = encadenar(generar_bloque, reg, contexto, lb, lc, rng)
        self.diagnostico_muestreo_.update(
            fraccion_recortada=cuenta["tocadas"] / max(cuenta["total"], 1),
            coordenadas=cuenta["total"],
        )
        return salida
