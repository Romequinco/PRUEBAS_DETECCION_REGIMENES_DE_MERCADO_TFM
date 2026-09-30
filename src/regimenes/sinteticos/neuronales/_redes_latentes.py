"""Piezas compartidas por los generadores de variable latente (CVAE y CGAN).

Por que existe: el CVAE y la CGAN resuelven el mismo problema ("siguiente
bloque": ``p(bloque | contexto, regimen por dia)``) y solo difieren en la red y
en la perdida. Todo lo que ocurre ANTES de la red (como se representan el
bloque y la condicion) y el muestreo de lotes debe ser identico en los dos para
que la comparacion de los notebooks 16-18 mida el modelo y no el preprocesado.
No define ningun generador: solo ``PreparadorBloques``, el muestreo equilibrado
de lotes, un constructor de MLP y dos diagnosticos de dispersion.

Representacion (cinco decisiones; la salida nunca se recorta):

1. **Ancla de las columnas persistentes**: en las columnas cuya autocorrelacion
   a 1 dia supera ``umbral_ancla`` la red no modela el nivel sino la diferencia
   con la ultima fila del contexto. Tres columnas del nucleo son escalones
   mensuales (el 95 % de sus variaciones diarias es exactamente 0): modelar el
   nivel obligaria a la red a copiar el contexto con precision de maquina;
   modelar el incremento convierte "no cambia" en "emite 0". Es una
   reparametrizacion (el contexto es entrada de la red), no una restriccion.
   El incremento se divide por su desviacion en train (a horizontes 1..L_blk)
   para que pese en la perdida lo mismo que el resto de columnas: sin ello una
   distancia euclidea (el critico de la GAN) apenas lo ve.
2. **Colas**: ``u = c * arcsinh(x / c)`` y re-estandarizacion (media y
   desviacion de ``u`` en train), SOLO en las columnas sin ancla cuyo exceso de
   curtosis supera ``curtosis_colas``. Es la identidad cerca de 0 y logaritmica
   en las colas: con ``c = 2`` el -24,4 de octubre de 1987 pasa a -6,4. No se
   recorta nada: la inversa ``x = c * sinh(u / c)`` devuelve exactamente el
   valor original y una cola generada vuelve a ser una cola. El precio es que
   el ruido de la red en ``u`` se amplifica exponencialmente al deshacerla
   (``sinh`` de una normal estandar con ``c = 1`` ya tiene exceso de curtosis
   33; con ``c = 2``, 1,5); por eso ``c`` (``escala_colas``) es un
   hiperparametro, ``None`` la desactiva y no se aplica a columnas de cola
   ligera ni a las persistentes (alli el error se acumularia al encadenar).
3. **Blanqueo de las columnas sin ancla**: rotacion y escala de componentes
   principales (de rango completo, sin reducir dimension) estimada en train y
   aplicada dia a dia. Dos columnas del nucleo (``FF_MKT_z`` y ``SP500_ret``)
   tienen correlacion 0,99: un modelo con ruido independiente por dimension
   (el decoder del CVAE) la destruiria; en la base rotada el ruido
   independiente reproduce la correlacion contemporanea.
4. **Condicion**: contexto aplanado en ``u`` (``L_ctx * d``) seguido del regimen
   de cada dia del bloque en one-hot sin la primera categoria
   (``L_blk * (K - 1)``; con dos regimenes son ``L_blk`` valores 0/1).
5. **Saturacion del contexto** (solo de la ENTRADA de la red): el contexto pasa
   por ``L * tanh(u / L)`` con ``L`` = maximo absoluto de la columna en train.
   Es monotona (no pierde informacion dentro del rango de train) y acotada:
   al encadenar, la red nunca recibe un contexto fuera del rango en el que se
   entreno. Sin ella un generador autoregresivo puede realimentarse (un
   retorno extremo generado -> contexto fuera de rango -> la red extrapola un
   bloque aun mas extremo -> ``sinh`` lo amplifica) y divergir; ocurrio con la
   CGAN en la pista A (valores de 1e120 tras 80 bloques). La salida no se
   toca: un bloque generado puede superar el maximo historico.

Muestreo de lotes: los bloques se clasifican en "todo calma", "todo crisis" y
"mixto" (contiene una transicion). Los mixtos son ~3 % de los bloques de la
pista A. ``probabilidades_lote`` mezcla la frecuencia empirica con la uniforme
entre clases; como el modelo es condicional, reponderar las condiciones no
sesga ``p(bloque | condicion)``, solo reparte el gradiente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regimenes.sinteticos.bloques import construir_bloques
from regimenes.sinteticos.neuronales._torch import importar_torch

torch = importar_torch()
nn = torch.nn


class PreparadorBloques:
    """Paso entre el espacio estandarizado ``X`` y la representacion que ve la red.

    Parameters
    ----------
    largo_bloque, largo_contexto : ``L_blk`` (>= 1) y ``L_ctx`` (>= 0).
    n_regimenes : numero de regimenes ``K``.
    escala_colas : ``c`` de ``c * arcsinh(x / c)``; ``None`` o 0 = sin transformar.
    umbral_ancla : autocorrelacion a 1 dia por encima de la cual una columna se
        modela en diferencias respecto a la ultima fila del contexto; ``None`` o
        un valor >= 1 lo desactiva. Sin contexto no hay ancla.
    curtosis_colas : exceso de curtosis a partir del cual una columna sin ancla
        recibe la compresion arcsinh.
    blanquear : rotar y escalar las columnas sin ancla a componentes principales.
    saturar_contexto : acotar el contexto de entrada con ``L * tanh(u / L)``.

    Attributes (tras ``ajustar``)
    -----------------------------
    autocorr_, curtosis_ : (d,) autocorrelacion a 1 dia y exceso de curtosis de ``X``.
    ancla_ : (d,) bool, columnas modeladas en diferencias.
    colas_ : (d,) bool, columnas con compresion arcsinh.
    media_, desv_ : (d,) estadisticos de la re-estandarizacion en ``u``.
    limite_contexto_ : (d,) ``L`` de la saturacion (maximo absoluto de ``u`` en train).
    escala_ancla_ : (d,) desviacion del incremento de las columnas con ancla (1 en el resto).
    rotacion_, rotacion_inversa_ : matrices del blanqueo de las columnas sin ancla.
    dim_salida, dim_condicion : dimensiones de la salida y de la condicion.
    """

    def __init__(
        self,
        largo_bloque: int,
        largo_contexto: int,
        n_regimenes: int = 2,
        escala_colas: float | None = 2.0,
        umbral_ancla: float | None = 0.9,
        curtosis_colas: float = 3.0,
        blanquear: bool = True,
        saturar_contexto: bool = True,
    ) -> None:
        self.lb, self.lc = int(largo_bloque), int(largo_contexto)
        if self.lb < 1 or self.lc < 0:
            raise ValueError("largo_bloque >= 1 y largo_contexto >= 0.")
        self.k = max(int(n_regimenes), 2)
        self.escala = float(escala_colas) if escala_colas else None
        self.umbral_ancla = umbral_ancla
        self.curtosis_colas = float(curtosis_colas)
        self.blanquear = bool(blanquear)
        self.saturar_contexto = bool(saturar_contexto)

    # ------------------------------------------------------------------ colas
    def _comprimir(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.escala is None or not self.colas_.any():
            return X
        return np.where(self.colas_, self.escala * np.arcsinh(X / self.escala), X)

    def a_u(self, X: np.ndarray) -> np.ndarray:
        """``X`` estandarizado -> representacion ``u`` (colas comprimidas, re-estandarizada)."""
        return (self._comprimir(X) - self.media_) / self.desv_

    def a_x(self, U: np.ndarray) -> np.ndarray:
        """Inversa exacta de ``a_u``."""
        V = np.asarray(U, dtype=float) * self.desv_ + self.media_
        if self.escala is None or not self.colas_.any():
            return V
        with np.errstate(over="ignore"):
            return np.where(self.colas_, self.escala * np.sinh(V / self.escala), V)

    # ----------------------------------------------------------------- ajuste
    def ajustar(self, X: np.ndarray) -> "PreparadorBloques":
        """Decide ancla y colas por columna y calcula re-estandarizacion y blanqueo."""
        X = np.asarray(X, dtype=float)
        self.d = X.shape[1]
        a, b = X[:-1] - X[:-1].mean(axis=0), X[1:] - X[1:].mean(axis=0)
        denom = np.sqrt((a**2).sum(axis=0) * (b**2).sum(axis=0))
        self.autocorr_ = np.divide((a * b).sum(axis=0), denom, out=np.zeros(self.d), where=denom > 0)
        con_ancla = self.lc > 0 and self.umbral_ancla is not None
        self.ancla_ = (self.autocorr_ > float(self.umbral_ancla)) if con_ancla else np.zeros(self.d, dtype=bool)
        centrada = X - X.mean(axis=0)
        m2 = (centrada**2).mean(axis=0)
        self.curtosis_ = np.divide((centrada**4).mean(axis=0), m2**2, out=np.full(self.d, 3.0), where=m2 > 0) - 3.0
        self.colas_ = (self.curtosis_ > self.curtosis_colas) & ~self.ancla_ & (self.escala is not None)
        V = self._comprimir(X)
        self.media_ = V.mean(axis=0)
        desv = V.std(axis=0)
        self.desv_ = np.where(desv > 0, desv, 1.0)
        Un = (V - self.media_) / self.desv_
        limite = np.abs(Un).max(axis=0)
        self.limite_contexto_ = np.where(limite > 0, limite, 1.0)
        self.escala_ancla_ = np.ones(self.d)
        if self.ancla_.any() and len(X) > self.lb:
            cuadrados = [((Un[h:] - Un[:-h]) ** 2).mean(axis=0) for h in range(1, self.lb + 1)]
            escala = np.sqrt(np.mean(cuadrados, axis=0))
            self.escala_ancla_ = np.where(self.ancla_ & (escala > 0), escala, 1.0)
        libres = ~self.ancla_
        n_libres = int(libres.sum())
        self.rotacion_ = self.rotacion_inversa_ = np.eye(n_libres)
        if self.blanquear and n_libres > 1:
            valores, vectores = np.linalg.eigh(np.cov(Un[:, libres], rowvar=False))
            raiz = np.sqrt(np.maximum(valores, 1e-10 * valores.max()))
            self.rotacion_ = vectores / raiz
            self.rotacion_inversa_ = (vectores * raiz).T
        return self

    @property
    def dim_salida(self) -> int:
        return self.lb * self.d

    @property
    def dim_condicion(self) -> int:
        return self.lc * self.d + self.lb * (self.k - 1)

    # ------------------------------------------------------------ conversiones
    def _ctx_u(self, ctx: np.ndarray, n: int) -> np.ndarray:
        """Ultimas ``L_ctx`` filas del contexto en ``u``: (n, L_ctx, d)."""
        if not self.lc:
            return np.zeros((n, 0, self.d))
        ctx = np.asarray(ctx, dtype=float)
        return self.a_u(ctx[:, ctx.shape[1] - self.lc:])

    def _ancla(self, ctx_u: np.ndarray) -> np.ndarray:
        """(n, 1, d): ultima fila del contexto en las columnas con ancla, 0 en el resto."""
        if not self.lc:
            return np.zeros((len(ctx_u), 1, self.d))
        return ctx_u[:, -1:, :] * self.ancla_

    def condicion(self, ctx: np.ndarray, reg_bloque: np.ndarray) -> np.ndarray:
        """``[contexto en u aplanado, regimen por dia]`` -> (n, dim_condicion) float32."""
        reg_bloque = np.asarray(reg_bloque, dtype=int)
        n = len(reg_bloque)
        una_caliente = (reg_bloque[:, :, None] == np.arange(1, self.k)[None, None, :])
        ctx_u = self._ctx_u(ctx, n)
        if self.saturar_contexto:
            ctx_u = self.limite_contexto_ * np.tanh(ctx_u / self.limite_contexto_)
        partes = [ctx_u.reshape(n, -1), una_caliente.reshape(n, -1)]
        return np.concatenate(partes, axis=1).astype(np.float32)

    def objetivo(self, ctx: np.ndarray, bloque: np.ndarray) -> np.ndarray:
        """Bloque real -> lo que la red debe producir, (n, dim_salida) float32."""
        n = len(bloque)
        res = (self.a_u(bloque) - self._ancla(self._ctx_u(ctx, n))) / self.escala_ancla_
        libres = ~self.ancla_
        res[..., libres] = res[..., libres] @ self.rotacion_
        return res.reshape(n, -1).astype(np.float32)

    def a_bloque(self, salida: np.ndarray, ctx: np.ndarray) -> np.ndarray:
        """Salida de la red (n, dim_salida) -> bloque (n, L_blk, d) en el espacio ``X``."""
        n = len(salida)
        res = np.array(salida, dtype=float).reshape(n, self.lb, self.d)
        libres = ~self.ancla_
        res[..., libres] = res[..., libres] @ self.rotacion_inversa_
        return self.a_x(res * self.escala_ancla_ + self._ancla(self._ctx_u(ctx, n)))

    def pares(
        self, X: np.ndarray, reg: np.ndarray, fechas: pd.DatetimeIndex | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pares de entrenamiento ``(condicion, objetivo, regimen del bloque)`` de un tramo."""
        ctx, blk, reg_blk = construir_bloques(X, reg, self.lb, self.lc, fechas=fechas)
        return self.condicion(ctx, reg_blk), self.objetivo(ctx, blk), reg_blk


def partir_validacion(
    n: int, fraccion: float, hueco: int, minimo: int
) -> tuple[slice, slice] | None:
    """Cabeza de ajuste y cola cronologica de validacion separadas por ``hueco`` filas.

    Devuelve ``None`` si ``fraccion`` es 0 o si alguna de las dos partes tendria
    menos de ``minimo`` filas (no cabe ni un par contexto + bloque).
    """
    if not fraccion or fraccion <= 0:
        return None
    n_val = int(round(n * float(fraccion)))
    corte = n - n_val
    if n_val < minimo or corte - hueco < minimo:
        return None
    return slice(0, corte - hueco), slice(corte, n)


def clase_bloque(reg_bloque: np.ndarray) -> np.ndarray:
    """0 = todo calma, 1 = todo en regimenes de crisis (>= 1), 2 = mixto (hay transicion)."""
    crisis = np.asarray(reg_bloque) > 0
    todo, alguno = crisis.all(axis=1), crisis.any(axis=1)
    return np.where(todo, 1, np.where(alguno, 2, 0))


def probabilidades_lote(reg_bloque: np.ndarray, equilibrio: float) -> np.ndarray:
    """Probabilidad de muestreo de cada bloque.

    ``equilibrio = 0``: uniforme sobre bloques (frecuencia empirica de las
    clases); ``equilibrio = 1``: las clases presentes (calma, crisis, mixto)
    pesan lo mismo; valores intermedios interpolan linealmente.
    """
    clases = clase_bloque(reg_bloque)
    a = float(np.clip(equilibrio, 0.0, 1.0))
    presentes, conteo = np.unique(clases, return_counts=True)
    p = np.empty(len(clases), dtype=float)
    for c, n_c in zip(presentes, conteo):
        masa = (1.0 - a) * n_c / len(clases) + a / len(presentes)
        p[clases == c] = masa / n_c
    return p / p.sum()


def mlp(entrada: int, ocultas, salida: int, activacion: str = "silu", dropout: float = 0.0):
    """MLP ``entrada -> ocultas... -> salida`` con salida lineal y SIN normalizacion por lote."""
    capas: list = []
    previo = int(entrada)
    for ancho in ocultas:
        capas.append(nn.Linear(previo, int(ancho)))
        capas.append(nn.SiLU() if activacion == "silu" else nn.LeakyReLU(0.2))
        if dropout:
            capas.append(nn.Dropout(float(dropout)))
        previo = int(ancho)
    capas.append(nn.Linear(previo, int(salida)))
    return nn.Sequential(*capas)


def cociente_dispersion(sintetico, real) -> float:
    """Media sobre dimensiones de ``desviacion(sintetico) / desviacion(real)`` entre muestras.

    1 = misma dispersion; < 1 sostenido = infradispersion (media condicional en
    un VAE, colapso de modos en una GAN). Se ignoran dimensiones constantes.
    """
    ds, dr = sintetico.std(dim=0), real.std(dim=0)
    validas = dr > 1e-6
    return float((ds[validas] / dr[validas]).mean()) if bool(validas.any()) else float("nan")


def submuestra_diagnostico(n: int, tam: int, semilla: int | None) -> np.ndarray:
    """Indices fijos (ordenados) para los diagnosticos por epoca; RNG propio, no consume el del ajuste."""
    if n <= tam:
        return np.arange(n)
    sorteo = np.random.default_rng(0 if semilla is None else semilla)
    return np.sort(sorteo.choice(n, size=int(tam), replace=False))
