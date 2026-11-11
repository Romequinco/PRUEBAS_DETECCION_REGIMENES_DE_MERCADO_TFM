"""Espacio de generacion ("trabajo") frente al espacio publico de features.

Por que existe: cuatro columnas del nucleo (``SP500_ret_z``, ``SP500_vol_z``,
``SP500_momentum``, ``SP500_drawdown``) son funciones DETERMINISTAS de la senda
del S&P 500. Si un generador las modelara como variables libres produciria
paneles imposibles (un drawdown que no corresponde a los retornos generados, una
volatilidad realizada incoherente con ellos) y los detectores verian una
inconsistencia que no existe en datos reales. Por eso:

- **Espacio de trabajo**: ``SP500_ret`` crudo + el resto de columnas del panel
  tal cual, estandarizadas con media/desviacion SOLO del tramo de entrenamiento
  (los generadores ven columnas ~N(0,1) en escala).
- **Vuelta a publico**: se deshace la estandarizacion y las cuatro columnas se
  RE-DERIVAN con las mismas primitivas causales de ``regimenes.features`` sobre
  [historia real del S&P 500 hasta ``fin_train``] + [retornos sinteticos],
  quedandose con el tramo sintetico. Cada trayectoria es asi una continuacion
  hipotetica del mercado tras ``fin_train``: su drawdown parte del maximo
  historico real y su z-score expanding de la historia real.

Si el panel no contiene ninguna de esas cuatro columnas (paneles de juguete de
los tests) el espacio es la identidad mas la estandarizacion.

Calendario de las trayectorias (limitacion declarada): ``indice_sintetico`` fecha
con ``pandas.bdate_range`` (lunes a viernes), que NO es el calendario de la NYSE:
incluye los festivos de mercado (unas 9 sesiones al ano que en los datos reales
no existen). Las fechas sinteticas son solo una etiqueta ordenada de sesiones
posteriores al corte: NO deben cruzarse por fecha con datos reales (ni
``join``/``reindex`` contra un panel real ni contra otra serie con calendario
de mercado); la sesion k-esima sintetica no es "el mismo dia" que la fecha que
lleva. Para comparar con el tramo real posterior se compara por posicion o por
distribucion.

No se retiene futuro: al ajustar, la historia de precios se recorta a
``<= fin_train`` y la serie de entrada se descarta, asi que ni el espacio ni el
pickle del generador contienen precios posteriores al corte.

``error_rederivacion_`` guarda, por columna derivada, el error absoluto maximo
entre la columna real del panel de entrenamiento y su re-derivacion desde el
precio crudo: es la prueba de que la vuelta a publico usa la misma receta que
``notebooks/03_preprocesado``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regimenes.features.transformaciones import (
    causal_zscore,
    drawdown,
    log_returns,
    momentum,
    realized_vol,
)
from regimenes.sinteticos.datos import COL_RET

DERIVADAS: tuple[str, ...] = ("SP500_ret_z", "SP500_vol_z", "SP500_momentum", "SP500_drawdown")


def derivar_sp500(precios: pd.Series) -> pd.DataFrame:
    """Las cuatro features deterministas de la senda del S&P 500.

    Misma receta que ``regimenes.features.paneles.construir_diarias`` (espina
    equity): z expanding del log-retorno, z expanding de la volatilidad
    realizada a 21 sesiones anualizada, momentum 12M-1M y drawdown corriente.
    """
    retorno = log_returns(precios)
    return pd.DataFrame({
        "SP500_ret_z": causal_zscore(retorno),
        "SP500_vol_z": causal_zscore(realized_vol(retorno, window=21, annualize=True)),
        "SP500_momentum": momentum(precios),
        "SP500_drawdown": drawdown(precios),
    })


def cargar_historia_sp500(fin: pd.Timestamp | str | None = None) -> pd.Series:
    """Precio crudo del S&P 500 (``data/raw/<fuente>/SP500.parquet``) hasta ``fin``.

    Es la misma serie de la que ``load_track_panel`` deriva ``SP500_ret``.
    """
    from regimenes.benchmark.cache import _sp500_path

    precios = pd.read_parquet(_sp500_path())["SP500"].sort_index().astype(float).dropna()
    precios.index = pd.DatetimeIndex(precios.index).astype("datetime64[ns]")
    if fin is not None:
        precios = precios.loc[precios.index <= pd.Timestamp(fin)]
    return precios.rename("SP500")


class EspacioGeneracion:
    """Transformacion panel publico <-> matriz de trabajo estandarizada.

    Parameters
    ----------
    historia_sp500:
        Serie de PRECIOS del S&P 500 (indice de fechas). Solo se usa si el panel
        contiene columnas derivadas; ``None`` la lee de ``data/raw``. Se recorta
        siempre a ``<= fin_train`` al ajustar (nunca entra futuro) y el objeto
        NO conserva la serie de entrada: tras ``ajustar`` solo queda
        ``historia_`` (``<= fin_train``).

    Attributes
    ----------
    columnas_publicas_, columnas_trabajo_, derivadas_ : list[str]
    media_, escala_ : numpy.ndarray (d,)
        Estadisticos de estandarizacion del espacio de trabajo (solo train).
    fin_train_ : pandas.Timestamp
        Ultima fecha del panel de entrenamiento.
    historia_ : pandas.Series | None
        Precios reales hasta ``fin_train_`` (``None`` si el espacio es identidad).
    error_rederivacion_ : dict[str, float]
        Error absoluto maximo de re-derivacion por columna derivada, sobre train.
    """

    def __init__(self, historia_sp500: pd.Series | None = None) -> None:
        self._historia_entrada = historia_sp500

    # ------------------------------------------------------------------ ajuste
    def ajustar(self, train: pd.DataFrame) -> np.ndarray:
        """Ajusta el espacio con ``train`` y devuelve ``X`` (n, d) estandarizada."""
        self.columnas_publicas_ = list(train.columns)
        self.derivadas_ = [c for c in self.columnas_publicas_ if c in DERIVADAS]
        self.columnas_trabajo_ = [c for c in self.columnas_publicas_ if c not in DERIVADAS]
        self.fin_train_ = pd.Timestamp(train.index[-1])
        self.historia_ = None
        self.error_rederivacion_ = {}
        if self.derivadas_:
            if COL_RET not in self.columnas_trabajo_:
                raise ValueError(
                    f"El panel tiene columnas derivadas del S&P 500 {self.derivadas_} pero no "
                    f"{COL_RET!r}: sin el retorno crudo no se pueden re-derivar."
                )
            self.historia_ = self._historia_hasta(self.fin_train_)
            derivado = derivar_sp500(self.historia_).reindex(train.index)
            self.error_rederivacion_ = {
                c: float((derivado[c] - train[c]).abs().max()) for c in self.derivadas_
            }
            ret_hist = log_returns(self.historia_).reindex(train.index)
            self.error_rederivacion_[COL_RET] = float((ret_hist - train[COL_RET]).abs().max())
        # la serie de entrada puede traer precios posteriores al corte: no se guarda
        self._historia_entrada = None
        valores = train[self.columnas_trabajo_].to_numpy(dtype=float)
        self.media_ = valores.mean(axis=0)
        escala = valores.std(axis=0, ddof=0)
        self.escala_ = np.where(escala > 0, escala, 1.0)
        return (valores - self.media_) / self.escala_

    def recortar_historia(self, precios: pd.Series) -> pd.Series:
        """``precios`` ordenados, sin NaN y recortados a ``<= fin_train_``."""
        precios = precios.sort_index().astype(float).dropna()
        return precios.loc[precios.index <= self.fin_train_]

    def _historia_hasta(self, fin: pd.Timestamp) -> pd.Series:
        if self._historia_entrada is None:
            precios = cargar_historia_sp500(fin)
        else:
            precios = self.recortar_historia(self._historia_entrada)
        if precios.empty or pd.Timestamp(precios.index[-1]) != fin:
            ultimo = None if precios.empty else pd.Timestamp(precios.index[-1]).date()
            raise ValueError(
                f"La historia del S&P 500 debe llegar exactamente a fin_train={fin.date()} "
                f"(ultimo precio disponible: {ultimo})."
            )
        return precios.rename("SP500")

    # ------------------------------------------------------------ propiedades
    @property
    def d(self) -> int:
        """Dimension del espacio de trabajo."""
        return len(self.columnas_trabajo_)

    def indice_sintetico(self, length: int) -> pd.DatetimeIndex:
        """Dias habiles (lunes a viernes, NO calendario NYSE) desde el siguiente a ``fin_train``.

        Etiqueta ordenada de sesiones, no fechas de mercado reales: no cruzar por
        fecha con datos reales (ver docstring del modulo).
        """
        inicio = self.fin_train_ + pd.offsets.BDay(1)
        return pd.DatetimeIndex(pd.bdate_range(inicio, periods=int(length))).astype("datetime64[ns]")

    # ------------------------------------------------------------ transformar
    def a_trabajo(self, panel: pd.DataFrame) -> np.ndarray:
        """Panel publico -> matriz de trabajo estandarizada (n, d)."""
        return (panel[self.columnas_trabajo_].to_numpy(dtype=float) - self.media_) / self.escala_

    def desestandarizar(self, Z: np.ndarray) -> np.ndarray:
        """Matriz de trabajo estandarizada -> unidades originales (mismas columnas)."""
        return np.asarray(Z, dtype=float) * self.escala_ + self.media_

    def a_publico(self, Z: np.ndarray, index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
        """Una trayectoria de trabajo ``(length, d)`` -> panel publico.

        Las columnas derivadas se recalculan sobre historia real + retornos
        sinteticos; el resultado tiene las columnas publicas en su orden original.
        """
        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2 or Z.shape[1] != self.d:
            raise ValueError(f"Se esperaba una matriz (length, {self.d}); llego {Z.shape}.")
        index = self.indice_sintetico(len(Z)) if index is None else index
        trabajo = pd.DataFrame(self.desestandarizar(Z), index=index, columns=self.columnas_trabajo_)
        if not self.derivadas_:
            return trabajo[self.columnas_publicas_]
        with np.errstate(over="ignore", invalid="ignore"):
            precio_sint = float(self.historia_.iloc[-1]) * np.exp(np.cumsum(trabajo[COL_RET].to_numpy()))
        senda = pd.concat([self.historia_, pd.Series(precio_sint, index=index)])
        derivado = derivar_sp500(senda).loc[index]
        for col in self.derivadas_:
            trabajo[col] = derivado[col].to_numpy()
        return trabajo[self.columnas_publicas_]
