"""Jitter: tramos reales mas ruido gaussiano. Suelo de comparacion y control de memorizacion.

Que hace
--------
Es el modelo simple "datos reales + ruido" (el cuarto modelo del taller previo
del equipo), adaptado a trayectorias con regimen:

1. copia tramos reales CONTIGUOS del regimen pedido, con la misma mecanica que
   ``bootstrap_regimen`` (inicio uniforme en los dias de ese regimen, longitud
   geometrica, sin cruzar el final de una racha real) pero con bloques largos
   (``longitud_media`` 63 sesiones ~ un trimestre);
2. suma a cada dia ruido gaussiano independiente de desviacion ``sigma`` veces la
   desviacion tipica de cada columna en train.

``sigma`` es RELATIVA a la escala de cada columna, no absoluta: el panel mezcla
retornos diarios y z-scores de escalas muy distintas y un ruido absoluto ahogaria
unas columnas sin tocar otras. La escala se mide sobre todo train y no por
regimen: la crisis tiene pocos dias y su desviacion seria inestable justo donde
mas importa. Como el espacio de trabajo ya esta estandarizado, esa escala vale 1
en cada columna (0 en una columna constante, que se deja sin ruido).

Que supone
----------
Nada sobre la distribucion: no aprende, perturba. Solo que un punto a distancia
``sigma`` de un dia real sigue siendo un dia plausible.

Que reproduce y que no
----------------------
- Reproduce casi todo lo que reproduce el bootstrap (marginales, dependencia
  transversal y dependencia temporal dentro del bloque), ligeramente emborronado:
  la varianza de cada columna crece en ``sigma^2`` (x1,005 en desviacion con 0,1
  sobre una columna de varianza unidad; mas en las columnas y regimenes cuya
  desviacion propia es menor que la global).
- MEMORIZA POR DISENO. Cada dia sintetico esta a una distancia ~``sigma*sqrt(d)``
  (en desviaciones tipicas) de un dia real concreto, y con el vecino temporal
  correcto. Es el control positivo de las metricas de memorizacion del notebook
  16: si una metrica no marca al jitter como copia, la metrica no sirve; si un
  generador neuronal no mejora al jitter en utilidad, no ha aportado nada mas que
  regularizacion por ruido.
- No genera configuraciones de mercado nuevas: solo puntos en una bola alrededor
  de los observados. Hereda las costuras del bootstrap entre bloques.
- El ruido es blanco e independiente entre columnas: rompe los escalones de las
  columnas mensuales (dejan de ser constantes dentro del mes) y anade una
  componente sin autocorrelacion a las columnas persistentes. El ruido sobre
  ``SP500_ret`` se acumula en el precio y por tanto en las columnas re-derivadas
  (drawdown, momentum, volatilidad realizada).
- Eleccion de ``sigma``: por debajo de ~0,05 son copias casi exactas; por encima
  de ~0,5 se destruyen la estructura temporal y las correlaciones.

Referencias
-----------
- Politis, D. N. y Romano, J. P. (1994). The stationary bootstrap. Journal of the
  American Statistical Association, 89(428), 1303-1313.
- Um, T. T. et al. (2017). Data augmentation of wearable sensor data for
  Parkinson's disease monitoring using convolutional neural networks. ICMI 2017.
- Iwana, B. K. y Uchida, S. (2021). An empirical survey of data augmentation for
  time series classification with neural networks. PLOS ONE, 16(7), e0254841.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from regimenes.sinteticos.parametricos.bootstrap_regimen import BootstrapRegimen
from regimenes.sinteticos.registry import registrar


@registrar
class Jitter(BootstrapRegimen):
    """Tramos reales del regimen pedido mas ruido gaussiano relativo.

    Parameters (``PARAMS``)
    -----------------------
    sigma : float
        Desviacion del ruido en fraccion de la desviacion tipica de cada columna
        en train (>= 0; 0 equivale al bootstrap puro).
    longitud_media : float
        Media de la longitud geometrica de los tramos copiados, en sesiones.

    Attributes (tras ``fit``)
    -------------------------
    escala_ruido_ : numpy.ndarray (d,)   desviacion por columna de train (espacio de trabajo)
    (mas los de ``BootstrapRegimen``)
    """

    nombre = "jitter"
    familia = "parametricos"
    PARAMS = {"sigma": 0.1, "longitud_media": 63}
    PARAMS_RAPIDOS: dict[str, Any] = {}

    def _fit(self, X: np.ndarray, reg: np.ndarray, fechas: pd.DatetimeIndex) -> None:
        if not float(self.sigma) >= 0.0:
            raise ValueError("sigma debe ser >= 0.")
        fichas = self._preparar(X, reg)
        self.escala_ruido_ = X.std(axis=0)
        ruido = float(self.sigma) * self.escala_ruido_
        for ficha in fichas:
            Xk = X[reg == ficha["regimen"]]
            propia = Xk.std(axis=0) if len(Xk) > 1 else np.full(X.shape[1], np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                cociente = np.where(propia > 0, ruido / propia, np.nan)
            self.registrar(
                **ficha,
                sigma=float(self.sigma),
                # distancia RMS de un dia perturbado a su original, en desviaciones de train
                desviacion_relativa=float(np.sqrt((ruido**2).sum())),
                # ruido frente a la desviacion propia del regimen (media y maximo entre columnas)
                ruido_sobre_regimen=float(np.nanmean(cociente)) if np.isfinite(cociente).any() else float("nan"),
                ruido_sobre_regimen_max=float(np.nanmax(cociente)) if np.isfinite(cociente).any() else float("nan"),
            )

    def _sample(self, reg: np.ndarray, rng: np.random.Generator, contexto: np.ndarray) -> np.ndarray:
        base = self.X_[self.muestrear_indices(reg, rng)]
        return base + float(self.sigma) * self.escala_ruido_ * rng.standard_normal(base.shape)
