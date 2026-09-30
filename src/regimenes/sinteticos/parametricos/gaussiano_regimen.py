"""Normal multivariante i.i.d. por regimen: la linea base "solo segundo orden".

Que hace
--------
Para cada regimen ``k`` (0 = calma, 1 = crisis) estima, con las filas de train
de ese regimen, un vector de medias ``mu_k`` y una matriz de covarianzas
``Sigma_k``; cada dia sintetico es una extraccion independiente

    x_t | s_t = k  ~  N(mu_k, Sigma_k),

en el espacio de trabajo estandarizado de ``GeneradorBase``. El muestreo usa el
factor de Cholesky guardado (``x = mu_k + L_k z``), de modo que generar es una
multiplicacion matricial. La secuencia de regimenes la resuelve la base (cadena
de Markov empirica o regimen impuesto); este generador solo decide que se
extrae dado el regimen del dia.

Covarianza: Ledoit-Wolf y puerta dura de Cholesky
-------------------------------------------------
La covarianza por defecto es el estimador de contraccion de Ledoit y Wolf
(2004): combinacion convexa de la covarianza muestral y una diagonal escalada,
con el peso (``shrinkage``) estimado de los propios datos. Aqui la dimension es
pequena (d ~ 6 frente a miles de dias), asi que la contraccion es casi nula y
el resultado coincide practicamente con la covarianza muestral; se mantiene
porque (a) el regimen de crisis tiene pocos dias EFECTIVAMENTE independientes
(las columnas mensuales son escalones de ~21 sesiones y las de nivel son muy
persistentes), (b) el mismo codigo sirve para paneles mas anchos (``features``
ampliadas) y (c) permite registrar cuanto mejora el condicionamiento frente a
la muestral. ``estimador="muestral"`` usa la covarianza empirica sin contraer.

La factorizacion de Cholesky es una puerta dura: se intenta con un jitter
diagonal ``jitter * traza_media * 10**salto`` para ``salto = 0..9`` y, si nada
factoriza, el ajuste falla con ``LinAlgError`` en vez de generar basura. El
numero de saltos necesarios queda en el historial.

Un regimen con menos de ``d + 2`` filas (o ausente, p. ej. ``regimes=None``) no
permite estimar una covarianza: hereda la media y la covarianza del conjunto de
train y se marca con ``respaldo=True`` en el historial.

Que reproduce
-------------
- Media y desviacion tipica de cada columna dentro de cada regimen (por tanto
  la diferencia de nivel y de volatilidad entre calma y crisis).
- Correlaciones cruzadas contemporaneas dentro de cada regimen.
- La mezcla entre regimenes da colas mas gruesas que una normal en la
  distribucion incondicional (mezcla de gaussianas), pero solo por esa via.

Que NO reproduce (por diseno)
-----------------------------
- Curtosis 3 y asimetria 0 DENTRO de cada regimen: los extremos reales
  (octubre de 1987 ronda -24 desviaciones de ``SP500_ret``) no aparecen nunca;
  en su lugar inflan la varianza gaussiana del regimen donde cayeron.
- Ninguna dependencia temporal dentro de regimen: ni agrupamiento de
  volatilidad (la autocorrelacion de |x_t| es nula salvo por el cambio de
  regimen) ni persistencia. Las columnas de nivel y las mensuales (escalones
  constantes ~21 sesiones en el panel real) salen como ruido blanco alrededor
  de la media del regimen: su autocorrelacion a un dia, ~1 en datos reales, es
  ~0 aqui. Un detector que explote esa persistencia vera un panel irreal.
- La trayectoria no continua desde el ultimo dia real: ``contexto`` se ignora.

Esa pobreza es el proposito: cuantifica cuanto de la fidelidad y de la utilidad
aguas abajo se explica solo con medias y covarianzas por regimen; todo lo que
los demas generadores ganen sobre este es atribuible a dinamica o a colas.

Historial (una fila por regimen)
--------------------------------
``regimen``, ``n_obs``, ``respaldo``, ``shrinkage`` (peso de Ledoit-Wolf; 0 con
el estimador muestral), ``cond_ledoit_wolf`` y ``cond_muestral`` (numeros de
condicion), ``autovalor_min`` (de la covarianza usada) y ``saltos_cholesky``.

Referencias
-----------
- Ledoit y Wolf (2004), "A well-conditioned estimator for large-dimensional
  covariance matrices", Journal of Multivariate Analysis.
- Hamilton (1989), "A new approach to the economic analysis of nonstationary
  time series and the business cycle", Econometrica (mezcla gaussiana con
  regimen de Markov; aqui el regimen es observado).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regimenes.sinteticos.comun import GeneradorBase
from regimenes.sinteticos.registry import registrar

ESTIMADORES = ("ledoit_wolf", "muestral")
MAX_SALTOS = 10


def cholesky_con_jitter(cov: np.ndarray, jitter: float = 1e-10) -> tuple[np.ndarray, int]:
    """Factor de Cholesky de ``cov`` con jitter diagonal creciente (puerta dura).

    Intenta ``cov + jitter * traza_media * 10**salto * I`` para ``salto`` de 0 a
    ``MAX_SALTOS - 1``. Devuelve ``(L, salto)``; si ningun intento factoriza
    lanza ``numpy.linalg.LinAlgError``.
    """
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    d = len(cov)
    escala = float(np.trace(cov)) / d
    if not np.isfinite(escala) or escala <= 0:
        escala = 1.0
    for salto in range(MAX_SALTOS):
        try:
            return np.linalg.cholesky(cov + jitter * escala * (10.0**salto) * np.eye(d)), salto
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError(
        f"La covarianza no es factorizable ni con jitter {jitter * 10.0 ** (MAX_SALTOS - 1):.1e} x traza media."
    )


def numero_condicion(cov: np.ndarray) -> tuple[float, float]:
    """``(numero de condicion, autovalor minimo)`` de una matriz simetrica."""
    autovalores = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    minimo = float(autovalores.min())
    return float(autovalores.max() / max(minimo, 1e-300)), minimo


@registrar
class GaussianoRegimen(GeneradorBase):
    """Normal multivariante i.i.d. por regimen con covarianza Ledoit-Wolf.

    Parameters
    ----------
    estimador:
        ``"ledoit_wolf"`` (defecto) o ``"muestral"``.
    jitter:
        Jitter diagonal inicial de la factorizacion, como fraccion de la
        varianza media; se multiplica por 10 en cada reintento.

    Attributes (tras ``fit``)
    -------------------------
    mu_ : numpy.ndarray (n_regimenes, d)        medias por regimen
    cov_ : numpy.ndarray (n_regimenes, d, d)    covarianzas usadas
    chol_ : numpy.ndarray (n_regimenes, d, d)   factores de Cholesky inferiores
    """

    nombre = "gaussiano_regimen"
    familia = "parametricos"
    PARAMS = {"estimador": "ledoit_wolf", "jitter": 1e-10}
    PARAMS_RAPIDOS: dict = {}

    def _covarianzas(self, Xk: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """``(covarianza usada, covarianza muestral, shrinkage)`` de un bloque de filas."""
        d = Xk.shape[1]
        muestral = np.cov(Xk, rowvar=False).reshape(d, d)
        if self.estimador == "muestral":
            return muestral, muestral, 0.0
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf(assume_centered=False).fit(Xk)
        return np.asarray(lw.covariance_, dtype=float), muestral, float(lw.shrinkage_)

    def _fit(self, X: np.ndarray, reg: np.ndarray, fechas: pd.DatetimeIndex) -> None:
        if self.estimador not in ESTIMADORES:
            raise ValueError(f"estimador debe estar en {ESTIMADORES}; llego {self.estimador!r}.")
        if not self.jitter >= 0:
            raise ValueError("jitter debe ser >= 0.")
        n, d = X.shape
        if n < d + 2:
            raise ValueError(f"{self.name}: {n} filas no bastan para una covarianza en {d} dimensiones.")
        self.mu_ = np.zeros((self.n_regimenes_, d))
        self.cov_ = np.zeros((self.n_regimenes_, d, d))
        self.chol_ = np.zeros((self.n_regimenes_, d, d))
        for k in range(self.n_regimenes_):
            Xk = X[reg == k]
            n_obs = len(Xk)
            respaldo = n_obs < d + 2
            if respaldo:  # regimen ausente o casi vacio: estadisticos del conjunto de train
                Xk = X
            cov, muestral, shrinkage = self._covarianzas(Xk)
            chol, saltos = cholesky_con_jitter(cov, self.jitter)
            cond, autovalor_min = numero_condicion(cov)
            cond_muestral, _ = numero_condicion(muestral)
            self.mu_[k] = Xk.mean(axis=0)
            self.cov_[k] = cov
            self.chol_[k] = chol
            self.registrar(
                regimen=k,
                n_obs=int(n_obs),
                respaldo=bool(respaldo),
                shrinkage=shrinkage,
                cond_ledoit_wolf=cond if self.estimador == "ledoit_wolf" else float("nan"),
                cond_muestral=cond_muestral,
                autovalor_min=autovalor_min,
                saltos_cholesky=int(saltos),
            )

    def _sample(self, reg: np.ndarray, rng: np.random.Generator, contexto: np.ndarray) -> np.ndarray:
        eps = rng.standard_normal(reg.shape + (self.d_,))
        salida = np.empty_like(eps)
        for k in range(self.n_regimenes_):
            mascara = reg == k
            if mascara.any():
                salida[mascara] = self.mu_[k] + eps[mascara] @ self.chol_[k].T
        return salida
