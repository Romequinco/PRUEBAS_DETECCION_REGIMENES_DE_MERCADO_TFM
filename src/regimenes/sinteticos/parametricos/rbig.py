"""RBIG por regimen sobre bloques temporales, condicionado al contexto en el espacio gaussianizado.

Que es
------
RBIG (*Rotation-Based Iterative Gaussianization*; Laparra, Camps-Valls y Malo
2011, sobre la gaussianizacion de Chen y Gopinath 2000) es un flujo normalizante
que no se entrena por gradiente. Cada capa encadena dos operaciones cerradas:

1. **Gaussianizacion marginal**: cada dimension pasa por su CDF empirica
   (tabulada) y por la funcion probit. Las marginales quedan N(0, 1) pero la
   dependencia entre dimensiones sigue intacta.
2. **Rotacion PCA**: redistribuye esa dependencia para que la siguiente
   gaussianizacion marginal la vea como no-gaussianidad marginal y la elimine.

Iterando, la multi-informacion (dependencia total) se destruye capa a capa hasta
que el vector es gaussiano factorizado. Las dos operaciones son invertibles, asi
que generar es muestrear ``z ~ N(0, I)`` y desandar las capas.

El problema: RBIG modela un vector, no una serie
------------------------------------------------
RBIG no tiene nocion de tiempo ni sitio donde inyectar una condicion. Aqui se
resuelve asi, por regimen ``k``:

- **Bloques**: el vector modelado es un bloque de ``largo_bloque`` sesiones
  consecutivas aplanado (``largo_bloque * d`` dimensiones); la dinamica y la
  dependencia cruzada DENTRO del bloque son dependencia entre dimensiones.
- **Un modelo por regimen**: la etiqueta no entra en el vector. Cada regimen
  se ajusta solo con bloques PUROS (todas sus sesiones en ese regimen); su
  contexto (las ``largo_contexto`` sesiones previas) puede ser de cualquier
  regimen, asi que las entradas en crisis desde calma estan en la muestra.
  Si un regimen no llega a ``MIN_BLOQUES_PUROS`` se usan los bloques de
  mayoria, y si no alcanza ``MIN_BLOQUES`` hereda el modelo del regimen con
  mas muestras (queda anotado en ``sustituidos_``).
- **Capa 0, por columna**: contexto y bloque se gaussianizan marginalmente en
  las coordenadas ORIGINALES (una tabla de CDF por columna, comun a todas las
  sesiones). Es una capa element-wise mas del flujo y es la que fija la
  marginal de cada variable: las colas de un retorno diario se diluyen en
  cualquier componente principal de un bloque y las rotaciones PCA tardarian
  muchas capas en reencontrarlas.
- **Condicion (via "a")**: en ese espacio gaussianizado ``g`` se toma la
  condicional gaussiana del bloque dado el contexto por complemento de Schur::

      E[g_blk | g_ctx] = m_b + A (g_ctx - m_c),      A = S_bc S_cc^-1

  y lo que RBIG gaussianiza es la INNOVACION ``r = g_blk - E[g_blk | g_ctx]``
  (gaussianizada por dimension y reducida con PCA a ``n_componentes``). Es un
  flujo condicional con acoplamiento afin: la condicion desplaza el bloque, y
  toda la estructura no gaussiana de lo que el contexto no explica (colas
  conjuntas, agrupamiento de volatilidad dentro del bloque, dependencia
  cruzada no lineal) la modela RBIG. Generar es ``z ~ N(0, I)`` -> RBIG
  inverso -> ``r`` -> sumar la media condicional -> inversa de la capa 0.
- **Encadenado**: ``bloques.encadenar``; el contexto de cada bloque son las
  ultimas sesiones GENERADAS (el primero, las ultimas reales de train).

Por que la condicional se toma tras la capa 0 y no en la cima del flujo: se
probo gaussianizar por separado contexto y bloque con dos RBIG y aplicar Schur
entre los dos latentes. La dependencia lineal entre latentes pierde precision
(cada flujo mezcla no linealmente niveles y ruido) y en las columnas que son
escalones mensuales el salto en la costura era 3-5 veces el salto diario. En el
espacio de la capa 0 la correlacion entre la ultima sesion del contexto y la
primera del bloque se conserva integra. Por que no bloques independientes (via
"b"): cada costura seria un salto de nivel del tamano de la desviacion del
regimen. ``largo_contexto = 0`` recupera esa via.

Bloques de transicion al generar: si el regimen impuesto cambia dentro de un
bloque, se genera un bloque con el modelo de cada regimen presente (ambos
condicionados al mismo contexto) y cada sesion se toma del modelo de SU
regimen. El regimen se respeta dia a dia; la costura intra-bloque es el precio.

Reduccion de dimension
----------------------
Un bloque de 21 x 6 tiene 126 dimensiones y el regimen de crisis ~2000
ventanas solapadas de 8 episodios (muchas menos efectivas). La innovacion se
proyecta con PCA a ``n_componentes`` (acotado a ``n / MUESTRAS_POR_DIM``); al
generar se deshace la proyeccion y, con ``ruido_residual``, se repone la
varianza descartada como ruido gaussiano independiente por dimension, para que
la covarianza sintetica no sea singular.

Convergencia
------------
No hay perdida. El equivalente es la reduccion de multi-informacion por capa
(nats), ``dI = -1/2 sum log var_d + sum negentropia_d`` tras la rotacion: el
primer termino es exacto y el segundo se estima por espaciados (Vasicek). Se
registra en ``historial`` por regimen y capa como ``reduccion_info`` y
``reduccion_acumulada``. Se para cuando una capa reduce menos de
``tolerancia * k`` nats o tras ``paciencia`` capas sin mejorar, nunca antes de
``min_capas`` ni despues de ``max_capas``. La ALTURA de la curva solo es
comparable a igual numero de muestras (el estimador se sesga al alza con
pocas); lo interpretable es donde se aplana. Mide la dependencia de la
innovacion: la lineal con el pasado ya la quito la condicional.

Que reproduce y que no
----------------------
Reproduce: la marginal de cada columna por regimen (colas y asimetria
incluidas), la dependencia cruzada y la dinamica dentro del bloque, y la
continuidad de las columnas persistentes a traves de las costuras.

No reproduce / limites:

- **Soporte acotado**: la inversa de la capa 0 satura en el minimo y el maximo
  de cada columna en train para ese regimen. No inventa un dia peor que el
  peor observado. Esa saturacion es tambien la que impide que el encadenado
  explote: el contexto realimentado siempre esta dentro del soporte.
- **Condicion lineal**: el pasado solo desplaza la media del bloque (en el
  espacio gaussianizado); la dispersion de la innovacion no depende del
  contexto. La volatilidad cambia con el regimen y dentro del bloque, pero no
  se hereda de un bloque al siguiente (sin efecto GARCH entre bloques).
- **Nivel heredado entre regimenes**: en las columnas persistentes el nivel
  lo arrastra el contexto (coeficiente ~1), asi que al entrar en crisis se
  parte del nivel de la calma previa y la media por regimen de train (fruto de
  8 episodios) no se reproduce; si su dispersion y su rango.
- **Escalones**: las columnas mensuales son constantes a trozos; un modelo
  continuo las convierte en derivas suaves con la misma autocorrelacion (mas
  cambios pequenos diarios, menos saltos mensuales).
- **Ruido residual gaussiano e independiente**: lo que la PCA descarta vuelve
  sin estructura.
- **Pocas muestras efectivas en crisis**: ventanas solapadas de pocos
  episodios; el modelo de crisis interpola entre ellos.
- La duracion de los regimenes no la modela este generador: la pone la cadena
  de Markov de ``GeneradorBase`` o el regimen impuesto.

Implementacion: numpy y scipy puros (la PCA es una descomposicion espectral de
la covarianza, determinista). ``fit`` solo usa numeros aleatorios, sembrados con
``random_state``, para tabular la calibracion de salida de cada flujo.

Referencias
-----------
- Laparra, V., Camps-Valls, G. y Malo, J. (2011). "Iterative Gaussianization:
  from ICA to Random Rotations". IEEE Transactions on Neural Networks 22(4).
- Chen, S. S. y Gopinath, R. A. (2000). "Gaussianization". NeurIPS 13.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

from regimenes.sinteticos.bloques import construir_bloques, encadenar
from regimenes.sinteticos.comun import GeneradorBase
from regimenes.sinteticos.registry import registrar

# Maximo de puntos interiores con los que se tabula la CDF empirica de una dimension.
PUNTOS_CDF_MAXIMOS = 512
# Estadisticos de orden extremos que entran siempre en la tabla, por cada cola.
PUNTOS_COLA = 64
# Bloques puros minimos para ajustar un regimen solo con ellos.
MIN_BLOQUES_PUROS = 50
# Por debajo de esto un regimen no tiene modelo propio (usa el de otro regimen).
MIN_BLOQUES = 8
# Muestras (ventanas) por dimension reducida que se exigen como minimo.
MUESTRAS_POR_DIM = 2
# Regularizacion relativa de la covarianza del contexto en la condicional.
RIDGE = 1e-2
# Muestras del propio modelo con las que se tabula la calibracion de cada flujo.
N_CALIBRACION = 20_000
# Entropia diferencial de una normal estandar, 1/2 log(2 pi e).
ENTROPIA_NORMAL = 0.5 * float(np.log(2.0 * np.pi * np.e))
_TINY = np.finfo(np.float64).tiny


# --------------------------------------------------------------------------- piezas


def _forzar_creciente(soporte: np.ndarray) -> np.ndarray:
    """Hace estrictamente creciente cada columna de la tabla de cuantiles.

    Los empates (valores repetidos) dejarian tramos planos que rompen la
    inversa por interpolacion; se acumula el maximo y se suma una rampa
    despreciable frente a la escala del dato.
    """
    soporte = np.maximum.accumulate(np.asarray(soporte, dtype=np.float64), axis=0)
    paso = np.maximum(np.ptp(soporte, axis=0), 1.0) * 1e-9
    return soporte + np.arange(soporte.shape[0], dtype=np.float64)[:, None] * paso[None, :]


@dataclass
class _Marginal:
    """CDF empirica tabulada por dimension: ``z = probit(F(x))`` y su inversa.

    ``probabilidades`` (m,) es comun a todas las dimensiones y nunca toca 0 ni 1
    (posiciones de Hazen ``(i + 0.5) / n``); ``soporte`` (m, k) son los
    estadisticos de orden e incluye siempre el minimo y el maximo de train.
    Directa e inversa interpolan sobre la misma tabla: se deshacen exactamente
    dentro del rango observado y SATURAN fuera de el.
    """

    probabilidades: np.ndarray
    soporte: np.ndarray

    @classmethod
    def ajustar(cls, X: np.ndarray) -> "_Marginal":
        n = X.shape[0]
        n_puntos = int(min(max(n, 2), PUNTOS_CDF_MAXIMOS))
        rejilla = np.round(np.linspace(0, n - 1, n_puntos)).astype(int)
        # Las colas no se submuestrean: interpolar linealmente entre un cuantil
        # interior y el extremo regalaria masa a valores que casi nunca ocurren.
        cola = np.arange(min(PUNTOS_COLA, n))
        posiciones = np.unique(np.concatenate([rejilla, cola, n - 1 - cola]))
        probabilidades = (posiciones + 0.5) / n
        return cls(probabilidades, _forzar_creciente(np.sort(X, axis=0)[posiciones]))

    def aplicar(self, X: np.ndarray) -> np.ndarray:
        p = np.empty_like(X)
        for j in range(X.shape[1]):
            p[:, j] = np.interp(X[:, j], self.soporte[:, j], self.probabilidades)
        return ndtri(p)

    def invertir(self, Z: np.ndarray) -> np.ndarray:
        p = ndtr(Z)
        X = np.empty_like(p)
        for j in range(Z.shape[1]):
            X[:, j] = np.interp(p[:, j], self.probabilidades, self.soporte[:, j])
        return X


def _por_columna(marginal: _Marginal, X: np.ndarray, inversa: bool = False) -> np.ndarray:
    """Aplica (o invierte) la capa 0: la tabla de cada columna a todas las sesiones de ``X``."""
    plano = X.reshape(-1, marginal.soporte.shape[1])
    return (marginal.invertir(plano) if inversa else marginal.aplicar(plano)).reshape(X.shape)


def _pca(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA por descomposicion espectral: ``(media, componentes (k, D), varianzas (D,))``.

    Determinista (sin solver aleatorio); el signo de cada componente se fija
    para que su coordenada de mayor modulo sea positiva.
    """
    media = X.mean(axis=0)
    cov = np.atleast_2d(np.cov(X - media, rowvar=False))
    valores, vectores = np.linalg.eigh(cov)
    orden = np.argsort(valores)[::-1]
    valores, vectores = np.maximum(valores[orden], 0.0), vectores[:, orden]
    comp = vectores[:, :k].T
    signo = np.sign(comp[np.arange(comp.shape[0]), np.abs(comp).argmax(axis=1)])
    return media, comp * np.where(signo == 0, 1.0, signo)[:, None], valores


def _regresion(Gc: np.ndarray, Gb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complemento de Schur: ``(m_c, m_b, A)`` con ``A = S_bc S_cc^-1`` (ridge minimo)."""
    m_c, m_b = Gc.mean(axis=0), Gb.mean(axis=0)
    Cc, Cb = Gc - m_c, Gb - m_b
    S_cc = Cc.T @ Cc / len(Gc)
    S_cb = Cc.T @ Cb / len(Gc)
    ridge = RIDGE * float(np.trace(S_cc)) / S_cc.shape[0]
    return m_c, m_b, np.linalg.solve(S_cc + ridge * np.eye(S_cc.shape[0]), S_cb).T


def _entropias_marginales(X: np.ndarray) -> np.ndarray:
    """Entropia diferencial de cada columna por espaciados muestrales (Vasicek).

    Su sesgo depende solo de ``n`` y del solape ``m``, no de la forma de la
    distribucion, asi que no contamina la comparacion entre capas.
    """
    n = X.shape[0]
    m = max(1, int(round(np.sqrt(n))))
    ordenado = np.sort(X, axis=0)
    pos = np.arange(n)
    espaciados = ordenado[np.minimum(pos + m, n - 1)] - ordenado[np.maximum(pos - m, 0)]
    return np.log(np.maximum(espaciados, _TINY) * n / (2 * m)).mean(axis=0)


def _reduccion_informacion(rotado: np.ndarray) -> float:
    """Multi-informacion destruida por una capa, en nats.

    ``dI = k h_N - sum_d H(x_d)`` tras la rotacion (la gaussianizacion marginal
    no altera la multi-informacion y la rotacion conserva la entropia conjunta).
    Se separa en dispersion de varianzas (exacta) mas negentropia marginal
    (estimada y recortada a >= 0): suma de dos terminos no negativos, de modo
    que la curva acumulada es monotona.
    """
    varianzas = rotado.var(axis=0)
    norm = varianzas / max(float(varianzas.mean()), _TINY)
    dispersion = -0.5 * float(np.log(np.maximum(norm, 1e-300)).sum())
    gauss = ENTROPIA_NORMAL + 0.5 * np.log(np.maximum(varianzas, _TINY))
    negentropia = np.maximum(gauss - _entropias_marginales(rotado), 0.0)
    return dispersion + float(negentropia.sum())


@dataclass
class _Flujo:
    """RBIG de una matriz (n, D).

    Cadena directa: (1) gaussianizacion marginal de cada dimension ORIGINAL
    (``entrada``, D tablas); (2) reduccion PCA a ``k``; (3) capas RBIG (marginal
    + rotacion) en el espacio reducido; (4) cierre: una capa que no rota
    (``rotacion is None``) y deja la cima con marginales N(0, 1) exactas sobre
    train, que es lo que asume el muestreo.

    El paso (1) es la primera gaussianizacion marginal del RBIG canonico, hecha
    antes de reducir: asi la inversa termina en la CDF empirica de cada
    dimension y su marginal (por pesada que sea la cola) no depende de cuantas
    capas se ajusten. Reducir primero deja esa marginal en manos de componentes
    principales de cola muy pesada, y el muestreo infla su varianza.

    ``calibracion`` cierra el lazo al generar. Con un numero finito de capas el
    latente no es exactamente gaussiano conjunto, y al muestrear ``N(0, I)`` la
    salida del paso (2) inverso tiene marginales solo aproximadamente N(0, 1)
    (algo de curtosis de mas); pasada por la inversa de una CDF de cola pesada,
    esa desviacion pequena infla la varianza un 15-25 %. Se tabula, con
    ``N_CALIBRACION`` muestras del propio modelo, la CDF que el flujo produce en
    cada dimension y se re-gaussianiza con ella antes de invertir ``entrada``:
    la marginal generada de cada dimension es asi la empirica de train.
    """

    entrada: _Marginal
    centro: np.ndarray
    componentes: np.ndarray
    sigma_residual: np.ndarray
    varianza_explicada: float
    calibracion: _Marginal | None = None
    capas: list[tuple[_Marginal, np.ndarray | None, np.ndarray | None]] = field(default_factory=list)
    info: list[float] = field(default_factory=list)

    @property
    def k(self) -> int:
        return self.componentes.shape[0]

    @classmethod
    def ajustar(cls, X: np.ndarray, k: int, max_capas: int, min_capas: int, tolerancia: float,
                paciencia: int, rng: np.random.Generator) -> "_Flujo":
        entrada = _Marginal.ajustar(X)
        U = entrada.aplicar(X)
        centro, comp, valores = _pca(U, k)
        actual = (U - centro) @ comp.T
        total = float(valores.sum())
        explicada = float(valores[:k].sum() / total) if total > 0 else 1.0
        flujo = cls(entrada, centro, comp, (U - centro - actual @ comp).std(axis=0), explicada)
        umbral, mejor, estancadas = tolerancia * k, np.inf, 0
        for capa in range(1, max(int(max_capas), 1) + 1):
            marginal = _Marginal.ajustar(actual)
            gauss = marginal.aplicar(actual)
            centro, rotacion, _ = _pca(gauss, k)
            actual = (gauss - centro) @ rotacion.T
            reduccion = _reduccion_informacion(actual)
            flujo.capas.append((marginal, centro, rotacion))
            flujo.info.append(reduccion)
            # Parada como la de una perdida de validacion: o reduccion despreciable,
            # o estancada en el suelo de ruido del estimador.
            if reduccion < mejor:
                mejor, estancadas = reduccion, 0
            else:
                estancadas += 1
            if capa >= min_capas and (reduccion < umbral or estancadas >= paciencia):
                break
        flujo.capas.append((_Marginal.ajustar(actual), None, None))
        flujo.calibracion = _Marginal.ajustar(flujo._gaussiano(
            rng.standard_normal((N_CALIBRACION, k)), rng.standard_normal((N_CALIBRACION, X.shape[1]))
        ))
        return flujo

    def _gaussiano(self, Z: np.ndarray, ruido: np.ndarray | None) -> np.ndarray:
        """Latente (n, k) -> espacio gaussianizado por dimension (n, D), sin calibrar."""
        for marginal, centro, rotacion in reversed(self.capas):
            if rotacion is not None:
                Z = Z @ rotacion + centro
            Z = marginal.invertir(Z)
        U = Z @ self.componentes + self.centro
        return U if ruido is None else U + ruido * self.sigma_residual

    def inverso(self, Z: np.ndarray, ruido: np.ndarray | None = None) -> np.ndarray:
        """Latente (n, k) -> (n, D); ``ruido`` (n, D) normal repone la varianza descartada."""
        return self.entrada.invertir(self.calibracion.aplicar(self._gaussiano(Z, ruido)))


@dataclass
class _ModeloRegimen:
    """Capa 0 (contexto y bloque), condicional gaussiana y RBIG de la innovacion."""

    entrada_bloque: _Marginal            # una tabla por columna original
    entrada_contexto: _Marginal | None
    m_c: np.ndarray | None               # (L_ctx * d,)
    m_b: np.ndarray                      # (L_blk * d,)
    A: np.ndarray | None                 # (L_blk * d, L_ctx * d)
    flujo: _Flujo | None = None
    r2_contexto: float = 0.0             # varianza de g_blk explicada por el contexto

    def media_condicional(self, ctx: np.ndarray) -> np.ndarray:
        """``ctx`` (m, L_ctx * d) en espacio de trabajo -> E[g_blk | g_ctx] (m, L_blk * d)."""
        if self.A is None:
            return np.broadcast_to(self.m_b, (ctx.shape[0], self.m_b.size))
        return self.m_b + (_por_columna(self.entrada_contexto, ctx) - self.m_c) @ self.A.T


# --------------------------------------------------------------------------- generador


@registrar
class RBIG(GeneradorBase):
    """RBIG por regimen sobre bloques, condicionado al contexto (ver modulo).

    Parametros (``PARAMS``)
    -----------------------
    largo_bloque : sesiones por bloque (dimension del vector = ``largo_bloque * d``).
    largo_contexto : sesiones previas que condicionan el bloque; 0 = bloques
        independientes por regimen.
    n_componentes : dimension reducida de la innovacion del bloque (PCA previa);
        ``None`` = sin tope propio. Siempre se acota a
        ``min(D, n - 1, n // MUESTRAS_POR_DIM)``.
    max_capas, min_capas : tope y minimo de capas RBIG (sin contar el cierre).
    tolerancia : umbral de parada en nats por dimension reducida.
    paciencia : capas seguidas sin mejorar la menor reduccion antes de parar.
    ruido_residual : repone la varianza descartada por la PCA como ruido gaussiano.
    paso : desplazamiento entre ventanas de train (1 = todas las ventanas).

    Atributos tras ``fit``
    ----------------------
    modelos_ : dict[int, _ModeloRegimen]   modelo usado por cada regimen
    sustituidos_ : dict[int, int]          regimen sin modelo propio -> regimen donante
    resumen_regimenes_ : pandas.DataFrame  bloques, dimension, varianza explicada,
        capas, reduccion total y R2 del contexto por regimen

    ``historial``: una fila por (regimen, capa) con ``reduccion_info`` y
    ``reduccion_acumulada`` (nats), mas ``n_muestras`` y ``n_componentes``.
    """

    nombre = "rbig"
    familia = "parametricos"
    PARAMS = {
        "largo_bloque": 21,
        "largo_contexto": 5,
        "n_componentes": 64,
        "max_capas": 30,
        "min_capas": 3,
        "tolerancia": 0.01,
        "paciencia": 3,
        "ruido_residual": True,
        "paso": 1,
    }
    PARAMS_RAPIDOS = {"largo_bloque": 5, "largo_contexto": 2, "n_componentes": 12, "max_capas": 6}

    # ------------------------------------------------------------------ ajuste
    def _dimension(self, n: int, D: int) -> int:
        tope = D if self.n_componentes is None else int(self.n_componentes)
        return int(np.clip(tope, 1, max(1, min(D, n - 1, n // MUESTRAS_POR_DIM))))

    def _ajustar_regimen(self, k: int, ctx: np.ndarray, blk: np.ndarray,
                         rng: np.random.Generator) -> _ModeloRegimen:
        """``ctx`` (n, L_ctx * d) y ``blk`` (n, L_blk * d) de los bloques del regimen ``k``."""
        d = self.d_
        entrada_b = _Marginal.ajustar(blk.reshape(-1, d))
        Gb = _por_columna(entrada_b, blk)
        if ctx.shape[1]:
            entrada_c = _Marginal.ajustar(ctx.reshape(-1, d))
            m_c, m_b, A = _regresion(_por_columna(entrada_c, ctx), Gb)
            modelo = _ModeloRegimen(entrada_b, entrada_c, m_c, m_b, A)
        else:
            modelo = _ModeloRegimen(entrada_b, None, None, Gb.mean(axis=0), None)
        innovacion = Gb - modelo.media_condicional(ctx)
        modelo.r2_contexto = 1.0 - float(innovacion.var(axis=0).sum() / max(Gb.var(axis=0).sum(), _TINY))
        n_comp = self._dimension(*innovacion.shape)
        modelo.flujo = _Flujo.ajustar(innovacion, n_comp, int(self.max_capas), int(self.min_capas),
                                      float(self.tolerancia), int(self.paciencia), rng)
        acumulada = 0.0
        for capa, reduccion in enumerate(modelo.flujo.info, start=1):
            acumulada += reduccion
            self.registrar(regimen=k, capa=capa, n_muestras=len(blk), n_componentes=n_comp,
                           reduccion_info=reduccion, reduccion_acumulada=acumulada)
        return modelo

    def _fit(self, X: np.ndarray, reg: np.ndarray, fechas: pd.DatetimeIndex) -> None:
        lb, lc = int(self.largo_bloque), int(self.largo_contexto)
        if lb < 1 or lc < 0:
            raise ValueError("rbig: largo_bloque >= 1 y largo_contexto >= 0.")
        ctx, blk, reg_blk = construir_bloques(X, reg, lb, lc, paso=int(self.paso), fechas=fechas)
        ctx, blk = ctx.reshape(len(ctx), -1), blk.reshape(len(blk), -1)
        self.modelos_: dict[int, _ModeloRegimen] = {}
        self.sustituidos_: dict[int, int] = {}
        filas: list[dict] = []
        tamanos: dict[int, int] = {}
        rng = self.rng_ajuste()  # solo para la tabla de calibracion de cada flujo
        for k in range(self.n_regimenes_):
            usar = (reg_blk == k).all(axis=1)
            criterio = "puros"
            if usar.sum() < MIN_BLOQUES_PUROS:
                # Pocos bloques puros (episodios cortos frente al bloque): bloques de mayoria.
                usar = (reg_blk == k).mean(axis=1) > 0.5
                criterio = "mayoria"
            n = int(usar.sum())
            if n < MIN_BLOQUES:
                continue
            modelo = self._ajustar_regimen(k, ctx[usar], blk[usar], rng)
            self.modelos_[k] = modelo
            tamanos[k] = n
            filas.append({
                "regimen": k, "criterio": criterio, "n_bloques": n, "dim_bloque": blk.shape[1],
                "n_componentes": modelo.flujo.k, "varianza_explicada": modelo.flujo.varianza_explicada,
                "capas": len(modelo.flujo.info), "reduccion_total": float(np.sum(modelo.flujo.info)),
                "r2_contexto": modelo.r2_contexto,
            })
        if not self.modelos_:
            raise ValueError("rbig: ningun regimen tiene bloques suficientes; reduce largo_bloque.")
        donante = max(tamanos, key=tamanos.get)
        for k in range(self.n_regimenes_):
            if k not in self.modelos_:
                self.modelos_[k] = self.modelos_[donante]
                self.sustituidos_[k] = donante
        self.resumen_regimenes_ = pd.DataFrame(filas)

    # ---------------------------------------------------------------- muestreo
    def _bloque_regimen(self, k: int, ctx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """``ctx`` (m, L_ctx, d) -> bloque (m, L_blk, d) del regimen ``k``."""
        modelo = self.modelos_[k]
        m = ctx.shape[0]
        z = rng.standard_normal((m, modelo.flujo.k))
        ruido = rng.standard_normal((m, modelo.m_b.size))
        innovacion = modelo.flujo.inverso(z, ruido if self.ruido_residual else None)
        g = modelo.media_condicional(ctx.reshape(m, -1)) + innovacion
        return _por_columna(modelo.entrada_bloque, g, inversa=True).reshape(m, int(self.largo_bloque), -1)

    def _generar_bloque(self, ctx: np.ndarray, reg_bloque: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        salida = np.empty((ctx.shape[0], reg_bloque.shape[1], ctx.shape[2]))
        for k in np.unique(reg_bloque):
            presente = reg_bloque == k
            filas = np.flatnonzero(presente.any(axis=1))
            bloque = self._bloque_regimen(int(k), ctx[filas], rng)
            # Bloque de transicion: cada sesion sale del modelo de su regimen.
            salida[filas] = np.where(presente[filas][:, :, None], bloque, salida[filas])
        return salida

    def _sample(self, reg: np.ndarray, rng: np.random.Generator, contexto: np.ndarray) -> np.ndarray:
        lc = int(self.largo_contexto)
        if lc and len(contexto) < lc:
            # train mas corto que el contexto: se repite la primera fila disponible
            contexto = np.vstack([np.repeat(contexto[:1], lc - len(contexto), axis=0), contexto])
        return encadenar(self._generar_bloque, reg, contexto, int(self.largo_bloque), lc, rng)
