"""Bootstrap estacionario condicionado a regimen: recombina tramos reales.

Que hace
--------
No ajusta ningun modelo: guarda la matriz de entrenamiento y, para cada regimen,
el conjunto ("pool") de sus dias y donde termina cada racha real contigua. Para
generar, recorre la secuencia de regimen pedida y la rellena con BLOQUES
contiguos de dias reales de ese mismo regimen:

1. al empezar, al cambiar el regimen pedido, al agotarse el bloque o al llegar al
   final de la racha real de la que se esta copiando, se elige un nuevo dia de
   inicio, uniforme entre todos los dias reales de ese regimen;
2. la longitud del bloque es geometrica de media ``longitud_media`` (bootstrap
   estacionario de Politis y Romano, 1994), lo que evita el artefacto de
   periodicidad de los bloques de longitud fija (Kunsch, 1989);
3. dentro del bloque se avanza dia a dia por la historia real.

A diferencia del bootstrap estacionario original no se "da la vuelta" al final de
la muestra (wrap-around): un bloque nunca cruza el final de una racha real, porque
el dia siguiente pertenece a otro regimen. Por eso la longitud de bloque EFECTIVA
es menor que ``longitud_media``; su esperanza exacta, para un inicio uniforme en
el pool, es ``mean_i[(1 - (1-p)^r_i) / p]`` con ``p = 1/longitud_media`` y ``r_i``
los dias que quedan de racha real desde el dia ``i``. Se registra en el historial.

Que supone
----------
- Que, condicionado al regimen, el proceso es estacionario: un dia de la crisis
  de 1974 es intercambiable con uno de la de 2001. Es la hipotesis fuerte.
- Que la dependencia temporal relevante cabe en un bloque: lo que ocurra a mas de
  ``longitud_media`` sesiones se pierde en las costuras.
- Que el regimen pedido es exogeno: el bootstrap no modela la transicion, solo
  rellena la secuencia que le dan (la cadena la simula ``GeneradorBase``).

Que reproduce
-------------
- Las distribuciones marginales y la dependencia transversal (correlaciones,
  colas conjuntas) de cada regimen, EXACTAMENTE: cada fila sintetica es una fila
  real completa.
- La dependencia temporal dentro de cada bloque (autocorrelacion, agrupamiento
  de volatilidad, escalones de las columnas mensuales) hasta el orden del bloque.

Que no reproduce (limites, a documentar en los notebooks 16-18)
---------------------------------------------------------------
- NO crea dias nuevos: solo recombina. El soporte de lo generado es el de train;
  nunca sale un dia peor que el peor dia observado. Como fuente de escenarios
  extremos es, por construccion, esteril, y en las metricas de memorizacion
  aparece como copia literal (distancia 0 al vecino real mas cercano).
- Discontinuidad en las costuras: entre el ultimo dia de un bloque y el primero
  del siguiente las columnas persistentes (z-scores de nivel, escalones
  mensuales) saltan de golpe. Con 21 sesiones de media hay un salto cada ~mes.
  La primera fila tampoco continua el ultimo dia real de train.
- La dinamica propia del cambio de regimen (el deterioro previo al pico, la
  recuperacion tras el suelo): un bloque de crisis puede empezar en mitad de un
  episodio real, no en su principio.
- Con pocos episodios (8 crisis en la pista A) la diversidad de tramos de crisis
  es escasa: trayectorias largas repiten los mismos tramos.
- Un regimen sin dias en train no se puede generar (error explicito).

Referencias
-----------
- Politis, D. N. y Romano, J. P. (1994). The stationary bootstrap. Journal of the
  American Statistical Association, 89(428), 1303-1313.
- Kunsch, H. R. (1989). The jackknife and the bootstrap for general stationary
  observations. The Annals of Statistics, 17(3), 1217-1241.
- Politis, D. N. y White, H. (2004). Automatic block-length selection for the
  dependent bootstrap. Econometric Reviews, 23(1), 53-70.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from regimenes.sinteticos.comun import GeneradorBase
from regimenes.sinteticos.registry import registrar


@registrar
class BootstrapRegimen(GeneradorBase):
    """Bootstrap estacionario por bloques dentro de cada regimen.

    Parameters (``PARAMS``)
    -----------------------
    longitud_media : float
        Media de la longitud geometrica de bloque, en sesiones (>= 1). 21 ~ un mes.

    Attributes (tras ``fit``)
    -------------------------
    X_ : numpy.ndarray (n, d)            historia de train estandarizada
    reg_train_ : numpy.ndarray (n,)      regimen de cada fila de ``X_``
    fin_racha_ : numpy.ndarray (n,)      True si la fila es la ultima de su racha real
    orden_ : numpy.ndarray (n,)          indices de train ordenados por regimen (pools concatenados)
    inicio_pool_, tam_pool_ : (K,)       posicion y tamano del pool de cada regimen en ``orden_``
    """

    nombre = "bootstrap_regimen"
    familia = "parametricos"
    PARAMS = {"longitud_media": 21}
    PARAMS_RAPIDOS: dict[str, Any] = {}

    # ------------------------------------------------------------------ ajuste
    def _preparar(self, X: np.ndarray, reg: np.ndarray) -> list[dict[str, Any]]:
        """Guarda la historia y los pools por regimen; devuelve una ficha por regimen."""
        if not float(self.longitud_media) >= 1.0:
            raise ValueError("longitud_media debe ser >= 1.")
        n = len(X)
        self.X_ = X
        self.reg_train_ = reg
        self.fin_racha_ = np.r_[reg[1:] != reg[:-1], True]
        self.orden_ = np.argsort(reg, kind="stable")
        self.tam_pool_ = np.bincount(reg, minlength=self.n_regimenes_)
        self.inicio_pool_ = np.r_[0, np.cumsum(self.tam_pool_)[:-1]]
        # dias que quedan de racha real desde cada fila (incluida ella)
        finales = np.flatnonzero(self.fin_racha_)
        restantes = finales[np.searchsorted(finales, np.arange(n))] - np.arange(n) + 1
        p = 1.0 / float(self.longitud_media)
        esperado = (1.0 - (1.0 - p) ** restantes) / p
        fichas = []
        for k in range(self.n_regimenes_):
            es_k = reg == k
            n_rachas = int((self.fin_racha_ & es_k).sum())
            fichas.append({
                "regimen": k,
                "n_obs": int(es_k.sum()),
                "n_rachas": n_rachas,
                "racha_media": float(es_k.sum() / n_rachas) if n_rachas else float("nan"),
                "longitud_media": float(self.longitud_media),
                "bloque_medio_efectivo": float(esperado[es_k].mean()) if n_rachas else float("nan"),
            })
        return fichas

    def _fit(self, X: np.ndarray, reg: np.ndarray, fechas: pd.DatetimeIndex) -> None:
        for ficha in self._preparar(X, reg):
            self.registrar(**ficha)

    # ---------------------------------------------------------------- muestreo
    def muestrear_indices(self, reg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Indices de fila de train ``(n_paths, length)`` que rellenan ``reg``.

        Vectorizado sobre trayectorias; el bucle es solo sobre el tiempo. Consume
        del ``rng`` exactamente dos matrices ``(n_paths, length)`` (uniformes para
        los inicios y geometricas para las longitudes), se usen o no.
        """
        n_paths, length = reg.shape
        vacios = [int(k) for k in np.unique(reg) if self.tam_pool_[k] == 0]
        if vacios:
            raise ValueError(
                f"{self.name}: no hay dias del regimen {vacios} en train; "
                "el bootstrap no puede generar un regimen que no vio."
            )
        u = rng.random((n_paths, length))
        largos = rng.geometric(1.0 / float(self.longitud_media), size=(n_paths, length))
        tam = self.tam_pool_[reg]
        # dia de inicio candidato en cada (trayectoria, t): solo se usa donde toca saltar
        candidatos = self.orden_[self.inicio_pool_[reg] + np.minimum((u * tam).astype(np.int64), tam - 1)]
        indices = np.empty((n_paths, length), dtype=np.int64)
        actual = candidatos[:, 0].copy()
        quedan = largos[:, 0].copy()
        indices[:, 0] = actual
        for t in range(1, length):
            saltar = (quedan <= 1) | (reg[:, t] != reg[:, t - 1]) | self.fin_racha_[actual]
            actual = np.where(saltar, candidatos[:, t], actual + 1)
            quedan = np.where(saltar, largos[:, t], quedan - 1)
            indices[:, t] = actual
        return indices

    def _sample(self, reg: np.ndarray, rng: np.random.Generator, contexto: np.ndarray) -> np.ndarray:
        return self.X_[self.muestrear_indices(reg, rng)]
