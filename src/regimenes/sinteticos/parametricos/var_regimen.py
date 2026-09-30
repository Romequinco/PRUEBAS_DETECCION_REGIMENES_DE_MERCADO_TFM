"""VAR(p) por regimen observado (Markov-switching VAR con regimen conocido).

Que hace
--------
Dentro de cada regimen ``k`` (0 = calma, 1 = crisis) el panel estandarizado
sigue un vector autorregresivo de orden ``p`` con coeficientes propios:

    x_t = c_k + A_k1 x_{t-1} + ... + A_kp x_{t-p} + e_t,      s_t = k.

Es el MS-VAR de Hamilton (1989) y Krolzig (1997) con una diferencia que lo
simplifica todo: aqui el regimen NO es latente (es el regimen de referencia de
``regimenes.sinteticos.datos``), asi que no hace falta el filtro de Hamilton ni
EM; la verosimilitud se separa por regimen y el estimador es minimos cuadrados
en forma cerrada, ecuacion por ecuacion, con las filas de cada regimen.

Ajuste
------
- Muestra de cada regimen: los dias ``t`` tales que ``t, t-1, ..., t-p`` son
  filas consecutivas de train y TODAS del regimen ``k``. Asi no entra ningun
  par que cruce una frontera de regimen (el primer dia de una crisis no se
  explica con coeficientes de crisis a partir de un dia de calma) y los
  coeficientes describen solo la dinamica interna del regimen.
- MCO con una penalizacion ridge pequena (``ridge * n_obs`` sobre los
  coeficientes de los retardos CENTRADOS; la constante no se penaliza). Su
  funcion es numerica: las columnas mensuales son escalones casi colineales con
  su propio retardo y, en crisis, hay pocos dias efectivos.
- Salvaguarda de estabilidad: se calcula el radio espectral de la matriz
  companera. Si es ``>= radio_max`` (raiz unitaria o explosiva, frecuente con
  columnas de nivel en muestras cortas) los coeficientes se contraen con
  ``A_kj <- f**j A_kj``, ``f = radio_max_efectivo / radio``, que multiplica
  TODOS los autovalores de la companera por ``f`` exactamente; la constante y
  los residuos se recalculan con los coeficientes contraidos. Queda registrado
  (``contraido``, ``factor_contraccion``, ``radio_original``). Sin esto una raiz
  de 1.002 multiplica el estado por ~150 en 2520 pasos.
- Constante (``anclar_media``), siempre tras la contraccion:

  - ``True`` (defecto): ``c_k = (I - A_k1 - ... - A_kp) m_k`` con ``m_k`` la
    media muestral de TODOS los dias del regimen ``k`` en train. El punto fijo
    del VAR del regimen es entonces exactamente ``m_k``.
  - ``False`` (constante de MCO): ``c_k = media(Y) - A_k media(Z)`` sobre la
    muestra de la regresion. Solo garantiza residuos de media cero; NO fija el
    punto fijo en la media del regimen. Con raices de 0,997-0,998,
    ``(I - A)^-1`` amplifica ~500 veces la diferencia (del orden de 1e-3) entre
    la media de ``Y`` y la de ``Z``: en la pista A el punto fijo de crisis de
    ``credit_BaaAaa_mensual_z`` salia en 1,72 (media muestral -0,235) y
    ``term_spread_hist_z`` sintetico en crisis valia +0,39 frente a -0,73 real.

  En ambos casos los residuos se centran antes de guardarse (las innovaciones
  tienen media cero; con ``anclar_media`` la media del residuo de MCO, de orden
  1e-3, es justo la deriva que se elimina).

  Medido en el humo de desarrollo (las cifras vigentes, con la semilla y la
  cadena compartida del proyecto, son las de la tabla de ``anclar_media`` del
  notebook ``15_sinteticos_generadores`` §4.4; difieren en valor, no en signo)
  (100 x 2520 sesiones, arranque estacionario, media de 3 semillas;
  error de media = ``|media sintetica - real|`` por regimen en desviaciones
  reales del regimen, error de dispersion = ``|log(sd sintetica / sd real)|``,
  ambos promediados sobre las columnas de trabajo; calma / crisis)::

                      error de media            error de dispersion
      pista A  MCO      0,104 / 0,271             0,021 / 0,053
      pista A  anclada  0,094 / 0,098             0,031 / 0,071
      pista B  MCO      0,112 / 0,086             0,066 / 0,118
      pista B  anclada  0,048 / 0,145             0,188 / 0,170

  En la pista A (la principal) anclar corrige la media de crisis
  (``term_spread_hist_z``: real -0,73, MCO +0,39, anclada +0,04) con un coste
  pequeno en dispersion. En la pista B empeora: la constante de MCO recoge la
  DERIVA media dentro de la racha (en una crisis corta los niveles de VIX o de
  diferenciales suben dia a dia; no oscilan alrededor de su media), y al
  quitarla las columnas persistentes se mueven menos (cociente de desviaciones
  de ``credit_BaaAaa_mensual_z`` 0,93 -> 0,67) y tardan mas en llegar al nivel
  de crisis. Sumando ambas pistas las dos opciones empatan; se deja ``True``
  por defecto porque en la pista principal es claramente mejor y porque el
  punto fijo de MCO (hasta 4,9 desviaciones de la media del regimen en la
  pista B) no es interpretable. ``anclar_media=False`` recupera el
  comportamiento anterior.
- Un regimen con menos de ``d * p + d + 2`` observaciones validas (o ausente)
  hereda el VAR estimado con todos los pares consecutivos de train
  (``respaldo=True``).

Innovaciones
------------
- ``"bootstrap"`` (defecto): en cada paso se remuestrea con reemplazo una FILA
  ENTERA de residuos reales del regimen vigente. Conserva la dependencia
  cruzada contemporanea, las colas y la asimetria de las innovaciones, y en
  particular el caracter de las columnas mensuales: sus residuos son casi
  siempre ~0 y de vez en cuando un salto, de modo que la simulacion produce
  escalones irregulares en vez de un AR gaussiano suave.
- ``"gaussiana"``: ``e_t ~ N(0, Sigma_k)`` con la covarianza muestral de los
  residuos (Cholesky con jitter). Pierde colas y saltos; sirve de contraste.

Simulacion
----------
Recursiva y vectorizada sobre trayectorias: el estado inicial son las ultimas
``p`` filas reales de train (``contexto``), de modo que cada trayectoria es una
continuacion del ultimo dia observado; en cada paso se aplican los coeficientes
del regimen de ese dia, y al cambiar el regimen cambian constante, matrices e
innovaciones mientras los retardos siguen siendo los simulados.

Que reproduce
-------------
- Media por regimen SOLO como punto fijo (``anclar_media``): con raices de
  ~0,998 la semivida del ajuste es de ~350 sesiones, mas que la duracion media
  de una crisis, asi que tras un cambio de regimen las columnas persistentes
  van hacia la media del nuevo regimen sin llegar a alcanzarla. La media
  sintetica DENTRO de regimen de esas columnas queda entre la del regimen
  anterior y la propia (no es la muestral), y tanto menos cuanto mas corta la
  racha. En las columnas poco persistentes (retornos, cambios) si coincide.
- Persistencia lineal: autocorrelaciones y correlaciones cruzadas retardadas
  de las columnas de nivel y mensuales, que el gaussiano i.i.d. destruye.
- Dependencia cruzada contemporanea de las innovaciones y, con bootstrap, sus
  colas (incluidos los dias extremos del regimen donde cayeron).
- Transiciones suaves: tras un cambio de regimen las columnas persistentes
  derivan hacia la media del nuevo regimen en lugar de saltar.

Que NO reproduce
----------------
- Agrupamiento de volatilidad DENTRO de regimen: las innovaciones son i.i.d.
  dado el regimen (homocedasticas); toda la heterocedasticidad viene del cambio
  de regimen. Para eso esta ``garch_regimen``.
- No linealidades ni asimetrias en la dinamica (la media condicional es lineal
  en los retardos).
- Con bootstrap los saltos de las columnas mensuales llegan en dias al azar,
  no cada ~21 sesiones: se conserva su frecuencia media, no su calendario.
- La dispersion incondicional de una columna casi integrada depende de un
  coeficiente cercano a 1 estimado con error (y de la contraccion si actua):
  el cociente de desviaciones frente al real es menos fiable en esas columnas
  que en ``SP500_ret``.
- La estabilidad esta garantizada regimen a regimen (radio < ``radio_max``),
  no para productos arbitrarios de matrices de regimenes distintos; en la
  practica los regimenes son largos y el producto hereda la contraccion.

Historial (una fila por regimen)
--------------------------------
``regimen``, ``n_obs`` (observaciones validas de la regresion), ``n_dias``
(dias del regimen en train), ``respaldo``, ``radio_original``,
``radio_espectral`` (tras la salvaguarda), ``contraido``,
``factor_contraccion``, ``r2_medio`` y ``r2_min`` (R2 por columna, tras la
contraccion), ``desv_residuo_media``, ``anclar_media``,
``desvio_punto_fijo_max`` (mayor distancia, en desviaciones de train, entre el
punto fijo del VAR del regimen y su media muestral: 0 con ``anclar_media``),
``deriva_constante_max`` (mayor diferencia entre la constante anclada y la de
MCO) y ``saltos_cholesky``.

Referencias
-----------
- Hamilton (1989), "A new approach to the economic analysis of nonstationary
  time series and the business cycle", Econometrica.
- Krolzig (1997), "Markov-Switching Vector Autoregressions", Springer.
- Lutkepohl (2005), "New Introduction to Multiple Time Series Analysis",
  Springer (forma companera, estabilidad y estimacion MCO del VAR).
- Hoerl y Kennard (1970), "Ridge regression", Technometrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regimenes.sinteticos.comun import GeneradorBase
from regimenes.sinteticos.parametricos.gaussiano_regimen import cholesky_con_jitter
from regimenes.sinteticos.registry import registrar

INNOVACIONES = ("bootstrap", "gaussiana")
# Margen bajo ``radio_max`` al contraer, para que el radio recalculado quede
# estrictamente por debajo pese al redondeo.
MARGEN_CONTRACCION = 1e-6


def matriz_companera(A: np.ndarray, orden: int) -> np.ndarray:
    """Matriz companera ``(d*p, d*p)`` de ``A`` ``(d, d*p)`` = ``[A_1 ... A_p]``."""
    d = A.shape[0]
    comp = np.zeros((d * orden, d * orden))
    comp[:d] = A
    if orden > 1:
        comp[d:, : d * (orden - 1)] = np.eye(d * (orden - 1))
    return comp


def radio_espectral(A: np.ndarray, orden: int) -> float:
    """Mayor modulo de los autovalores de la matriz companera."""
    return float(np.abs(np.linalg.eigvals(matriz_companera(A, orden))).max())


@registrar
class VarRegimen(GeneradorBase):
    """VAR(p) con coeficientes e innovaciones propios de cada regimen observado.

    Parameters
    ----------
    orden:
        Numero de retardos ``p`` (>= 1). Es tambien el largo del contexto real
        del que arranca la simulacion.
    ridge:
        Penalizacion relativa de los coeficientes de los retardos
        (``ridge * n_obs`` sobre regresores centrados y estandarizados en train).
    innovaciones:
        ``"bootstrap"`` (filas enteras de residuos reales del regimen) o
        ``"gaussiana"``.
    radio_max:
        Radio espectral maximo admitido de la companera (en (0, 1)).
    anclar_media:
        ``True`` (defecto): la constante fija el punto fijo del VAR de cada
        regimen en la media muestral del regimen, ``c = (I - sum_j A_j) m_k``.
        ``False``: constante de MCO (ver docstring del modulo).

    Attributes (tras ``fit``)
    -------------------------
    c_ : numpy.ndarray (n_regimenes, d)             constantes
    A_ : numpy.ndarray (n_regimenes, d, d * orden)  ``[A_1 ... A_p]`` por regimen
    residuos_ : list[numpy.ndarray]                 residuos (n_k, d) por regimen
    chol_ : numpy.ndarray (n_regimenes, d, d)       Cholesky de la covarianza residual
    radio_ : numpy.ndarray (n_regimenes,)           radio espectral final
    """

    nombre = "var_regimen"
    familia = "parametricos"
    PARAMS = {
        "orden": 1, "ridge": 1e-4, "innovaciones": "bootstrap", "radio_max": 0.999, "anclar_media": True,
    }
    PARAMS_RAPIDOS: dict = {}

    @property
    def largo_contexto_efectivo(self) -> int:
        """El contexto de arranque son las ultimas ``orden`` filas reales."""
        return max(int(self.orden), 1)

    # ------------------------------------------------------------------ ajuste
    def _validar_params(self) -> None:
        if int(self.orden) != self.orden or self.orden < 1:
            raise ValueError("orden debe ser un entero >= 1.")
        if not self.ridge >= 0:
            raise ValueError("ridge debe ser >= 0.")
        if self.innovaciones not in INNOVACIONES:
            raise ValueError(f"innovaciones debe estar en {INNOVACIONES}; llego {self.innovaciones!r}.")
        if not 0 < self.radio_max < 1:
            raise ValueError("radio_max debe estar en (0, 1).")

    @staticmethod
    def _retardos(X: np.ndarray, t: np.ndarray, orden: int) -> np.ndarray:
        """Regresores ``[x_{t-1}, ..., x_{t-p}]`` apilados: ``(len(t), d * p)``."""
        return np.concatenate([X[t - j] for j in range(1, orden + 1)], axis=1)

    def _mco(self, Y: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ridge sin penalizar la constante: ``(A (d, d*p), media_Y, media_Z)``."""
        media_y, media_z = Y.mean(axis=0), Z.mean(axis=0)
        Yc, Zc = Y - media_y, Z - media_z
        gram = Zc.T @ Zc + self.ridge * len(Z) * np.eye(Z.shape[1])
        try:
            B = np.linalg.solve(gram, Zc.T @ Yc)
        except np.linalg.LinAlgError:
            B = np.linalg.lstsq(gram, Zc.T @ Yc, rcond=None)[0]
        return B.T, media_y, media_z

    def _fit(self, X: np.ndarray, reg: np.ndarray, fechas: pd.DatetimeIndex) -> None:
        self._validar_params()
        p = int(self.orden)
        n, d = X.shape
        minimo = d * p + d + 2
        t_todos = np.arange(p, n)
        if len(t_todos) < minimo:
            raise ValueError(f"{self.name}: {n} filas no bastan para un VAR({p}) en {d} dimensiones.")
        # dias cuyo tramo [t-p, t] es homogeneo en regimen
        homogeneo = np.ones(n, dtype=bool)
        homogeneo[:p] = False
        for j in range(1, p + 1):
            homogeneo[j:] &= reg[j:] == reg[:-j]

        self.c_ = np.zeros((self.n_regimenes_, d))
        self.A_ = np.zeros((self.n_regimenes_, d, d * p))
        self.chol_ = np.zeros((self.n_regimenes_, d, d))
        self.radio_ = np.zeros(self.n_regimenes_)
        self.residuos_ = []
        for k in range(self.n_regimenes_):
            t = np.flatnonzero(homogeneo & (reg == k))
            n_obs = len(t)
            respaldo = n_obs < minimo
            if respaldo:  # regimen ausente o casi vacio: VAR del conjunto de train
                t = t_todos
            Y, Z = X[t], self._retardos(X, t, p)
            A, media_y, media_z = self._mco(Y, Z)
            radio_original = radio_espectral(A, p)
            factor = 1.0
            if radio_original >= self.radio_max:
                factor = (self.radio_max - MARGEN_CONTRACCION) / radio_original
                # A_j <- f**j A_j escala por f todos los autovalores de la companera
                A = A * np.repeat(factor ** np.arange(1, p + 1), d)[None, :]
            radio = radio_espectral(A, p)
            c_mco = media_y - A @ media_z
            # media muestral del regimen (todos sus dias; la de train si el regimen no aparece)
            media_regimen = X[reg == k].mean(axis=0) if (reg == k).any() else X.mean(axis=0)
            suma_A = A.reshape(d, p, d).sum(axis=1)
            c_anclada = (np.eye(d) - suma_A) @ media_regimen
            c = c_anclada if self.anclar_media else c_mco
            try:
                punto_fijo = np.linalg.solve(np.eye(d) - suma_A, c)
                desvio = float(np.abs(punto_fijo - media_regimen).max())
            except np.linalg.LinAlgError:
                desvio = float("nan")
            residuos = Y - c - Z @ A.T
            residuos = residuos - residuos.mean(axis=0)
            sst = ((Y - media_y) ** 2).sum(axis=0)
            r2 = 1.0 - (residuos**2).sum(axis=0) / np.where(sst > 0, sst, 1.0)
            cov = np.cov(residuos, rowvar=False).reshape(d, d)
            chol, saltos = cholesky_con_jitter(cov)
            self.c_[k], self.A_[k], self.chol_[k], self.radio_[k] = c, A, chol, radio
            self.residuos_.append(residuos)
            self.registrar(
                regimen=k,
                n_obs=int(n_obs),
                n_dias=int((reg == k).sum()),
                respaldo=bool(respaldo),
                radio_original=radio_original,
                radio_espectral=radio,
                contraido=bool(factor < 1.0),
                factor_contraccion=float(factor),
                r2_medio=float(r2.mean()),
                r2_min=float(r2.min()),
                desv_residuo_media=float(residuos.std(axis=0).mean()),
                anclar_media=bool(self.anclar_media),
                desvio_punto_fijo_max=desvio,
                deriva_constante_max=float(np.abs(c_anclada - c_mco).max()),
                saltos_cholesky=int(saltos),
            )

    # ---------------------------------------------------------------- muestreo
    def _innovaciones(self, reg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Innovaciones ``(n_paths, length, d)`` segun el regimen de cada dia."""
        if self.innovaciones == "gaussiana":
            z = rng.standard_normal(reg.shape + (self.d_,))
            eps = np.empty_like(z)
            for k in range(self.n_regimenes_):
                mascara = reg == k
                if mascara.any():
                    eps[mascara] = z[mascara] @ self.chol_[k].T
            return eps
        # bootstrap: una fila entera del banco de residuos del regimen del dia
        tamanos = np.array([len(r) for r in self.residuos_])
        desplaz = np.r_[0, np.cumsum(tamanos)[:-1]]
        banco = np.concatenate(self.residuos_, axis=0)
        u = rng.random(reg.shape)
        fila = np.minimum((u * tamanos[reg]).astype(int), tamanos[reg] - 1)
        return banco[desplaz[reg] + fila]

    def _sample(self, reg: np.ndarray, rng: np.random.Generator, contexto: np.ndarray) -> np.ndarray:
        p, d = int(self.orden), self.d_
        n_paths, length = reg.shape
        eps = self._innovaciones(reg, rng)
        ctx = np.asarray(contexto, dtype=float).reshape(-1, d)
        if len(ctx) < p:  # train mas corto que el orden: se repite la fila mas antigua
            ctx = np.vstack([np.repeat(ctx[:1], p - len(ctx), axis=0), ctx])
        # estado: retardos del mas reciente al mas antiguo, (n_paths, p * d)
        estado = np.tile(ctx[::-1][:p].reshape(1, p * d), (n_paths, 1))
        salida = np.empty((n_paths, length, d))
        for t in range(length):
            k = reg[:, t]
            x = self.c_[k] + np.einsum("nij,nj->ni", self.A_[k], estado) + eps[:, t]
            salida[:, t] = x
            if p > 1:
                estado[:, d:] = estado[:, :-d].copy()
            estado[:, :d] = x
        return salida
