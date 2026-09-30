"""Clase base comun de los generadores: toda la fontaneria, ningun modelo.

Por que existe: los diez generadores (parametricos y neuronales) comparten
validacion de entradas, espacio de generacion, cadena de regimenes, semillas,
vuelta al espacio publico, historial de convergencia y persistencia. Si cada
uno lo reimplementara, las diferencias entre generadores que midan los
notebooks 16-18 mezclarian el modelo con la fontaneria. ``GeneradorBase``
implementa ``fit`` y ``sample`` de la interfaz ``Generador`` y deja a cada
generador solo dos metodos sobre matrices numpy estandarizadas:

- ``_fit(X, reg, fechas)``: ``X`` (n, d) float64 en espacio de trabajo
  estandarizado (media 0, desviacion 1 por columna en train), ``reg`` (n,) int,
  ``fechas`` el ``DatetimeIndex`` de train.
- ``_sample(reg, rng, contexto)``: ``reg`` (n_paths, length) int ya resuelto,
  ``rng`` un ``numpy.random.Generator`` ya sembrado, ``contexto`` (L_ctx, d) las
  ultimas filas reales de train; devuelve (n_paths, length, d) en el mismo
  espacio estandarizado.

Declaracion de un generador (sin ``__init__``)::

    @registrar
    class Jitter(GeneradorBase):
        nombre = "jitter"                 # clave del registro y de las rutas
        familia = "parametricos"
        PARAMS = {"sigma": 0.1}           # hiperparametros y sus valores por defecto
        PARAMS_RAPIDOS = {}               # sobrescrituras para el test de contrato

Los hiperparametros de ``PARAMS`` quedan como atributos (``self.sigma``) y en
``self.params``; una clave desconocida en el constructor es un ``TypeError``.
Convencion: el estado ajustado se guarda en atributos terminados en ``_``.

Garantias anti-fuga: ``fit`` solo ve ``train``; la estandarizacion, la matriz de
transicion y el contexto de arranque se calculan con ``train``; las trayectorias
se fechan a partir del dia habil siguiente a la ultima fecha de ``train``. Tras
``fit`` el generador solo conserva la historia del S&P 500 ``<= fin_train``
(aunque se le pasara mas larga): el pickle de un generador ajustado no contiene
ningun precio posterior al corte.

Cadena de regimenes al muestrear: ``sample(..., regimes=None)`` simula la cadena
de Markov de train continuando el ultimo estado (comportamiento por defecto).
``regimes`` admite ademas un ``dict`` de opciones, ``{"inicial": "ultimo" |
"estacionaria", "duraciones": "geometricas" | "empiricas"}`` (ver
``datos.simular_regimenes``), o la matriz ya simulada. Cada ``sample`` deja un
diagnostico en ``diagnostico_muestreo_`` (``diagnostico_muestreo()``) sin tocar
nada que cambie muestras posteriores.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regimenes.sinteticos.base import Generador
from regimenes.sinteticos.datos import (
    COL_REGIMEN,
    DURACIONES,
    INICIALES,
    matriz_transicion,
    rachas,
    simular_regimenes,
)
from regimenes.sinteticos.espacio import EspacioGeneracion

SEED = 42
FICHERO_MODELO = "generador.pkl"


class GeneradorBase(Generador):
    """Base concreta de los generadores (ver docstring del modulo).

    Parameters
    ----------
    random_state:
        Semilla del AJUSTE (inicializaciones, entrenamiento neuronal). La del
        muestreo se pasa a ``sample``.
    historia_sp500:
        Precios del S&P 500 para re-derivar las columnas deterministas; ``None``
        los lee de ``data/raw`` (solo si el panel tiene esas columnas). ``fit``
        la recorta a ``<= fin_train`` y descarta el resto: para reajustar el
        mismo objeto con un corte POSTERIOR hay que crear otro generador.
    **params:
        Hiperparametros declarados en ``PARAMS`` de la subclase.

    Attributes (tras ``fit``)
    -------------------------
    columnas_ : list[str]            columnas publicas (las de train)
    columnas_trabajo_ : list[str]    columnas del espacio de trabajo (orden de ``X``)
    fin_train_ : pandas.Timestamp    ultima fecha de train
    n_train_, d_ : int               filas de train y dimension de trabajo
    n_regimenes_ : int               numero de regimenes (>= 2)
    P_ : numpy.ndarray               matriz de transicion empirica (train)
    rachas_ : pandas.DataFrame       rachas del regimen de train
    regimen_final_ : int             regimen de la ultima fila de train
    contexto_ : numpy.ndarray        ultimas ``largo_contexto`` filas de ``X``
    reg_contexto_ : numpy.ndarray    regimen de esas filas
    espacio_ : EspacioGeneracion
    tiempo_ajuste_ : float           segundos de ``_fit``
    diagnostico_muestreo_ : dict     diagnostico del ultimo ``sample`` (vacio antes)
    """

    nombre: str = ""
    familia: str = ""
    PARAMS: dict[str, Any] = {}
    PARAMS_RAPIDOS: dict[str, Any] = {}
    # False si el generador, por diseno, no diferencia la volatilidad por regimen
    # (el test de contrato omite entonces esa comprobacion).
    condiciona_volatilidad: bool = True

    def __init__(
        self,
        *,
        random_state: int | None = SEED,
        historia_sp500: pd.Series | None = None,
        **params: Any,
    ) -> None:
        desconocidos = sorted(set(params) - set(self.PARAMS))
        if desconocidos:
            raise TypeError(
                f"{type(self).__name__}: parametros desconocidos {desconocidos}; "
                f"admitidos: {sorted(self.PARAMS)}."
            )
        self.random_state = random_state
        self.params: dict[str, Any] = {**self.PARAMS, **params}
        for clave, valor in self.params.items():
            setattr(self, clave, valor)
        self._historia_sp500 = historia_sp500
        self._historial: list[dict[str, Any]] = []
        self._ajustado = False
        self.diagnostico_muestreo_: dict[str, Any] = {}

    # ------------------------------------------------------------------ hooks
    @property
    def name(self) -> str:
        """Identificador estable: el atributo de clase ``nombre``."""
        if not type(self).nombre:
            raise NotImplementedError(f"{type(self).__name__} debe definir el atributo de clase `nombre`.")
        return type(self).nombre

    def _fit(self, X: np.ndarray, reg: np.ndarray, fechas: pd.DatetimeIndex) -> None:
        """Ajusta el modelo sobre ``X`` (n, d) estandarizada y ``reg`` (n,) int."""
        raise NotImplementedError

    def _sample(self, reg: np.ndarray, rng: np.random.Generator, contexto: np.ndarray) -> np.ndarray:
        """Genera ``(n_paths, length, d)`` estandarizado dado ``reg`` (n_paths, length).

        Puede anadir claves propias al diagnostico del muestreo (contadores de
        recorte, ...) con ``self.diagnostico_muestreo_.update(...)``: la base lo
        vacia antes de llamar y anade despues los campos comunes. No debe
        modificar ningun otro atributo (dos ``sample`` con la misma semilla
        tienen que dar lo mismo).
        """
        raise NotImplementedError

    # ------------------------------------------------------------ utilidades
    @property
    def largo_contexto_efectivo(self) -> int:
        """Filas de contexto que recibe ``_sample`` (param ``largo_contexto``; 1 si no existe)."""
        return max(int(getattr(self, "largo_contexto", 1) or 0), 0)

    def rng_ajuste(self) -> np.random.Generator:
        """``numpy.random.Generator`` sembrado con ``random_state`` para usar en ``_fit``."""
        return np.random.default_rng(self.random_state)

    def registrar(self, **columnas: Any) -> None:
        """Anade una fila al historial de convergencia (p. ej. ``epoca=3, perdida=0.41``)."""
        self._historial.append(dict(columnas))

    @property
    def historial(self) -> pd.DataFrame:
        """Historial de convergencia acumulado durante el ultimo ``fit``."""
        return pd.DataFrame(self._historial)

    def _exigir_ajuste(self) -> None:
        if not getattr(self, "_ajustado", False):
            raise RuntimeError(f"{self.name}: hay que llamar a fit() antes.")

    # -------------------------------------------------------------------- fit
    def fit(self, train: pd.DataFrame, regimes: pd.Series | None = None) -> "GeneradorBase":
        """Ajusta el generador SOLO con ``train`` (y ``regimes`` alineado).

        ``regimes=None`` equivale a un unico regimen (todo 0): el generador queda
        incondicional y la cadena simulada permanece en 0.
        """
        train = self._validar_train(train)
        reg = self._validar_regimen_train(regimes, train)
        self._ajustado = False
        self._historial = []
        self.diagnostico_muestreo_ = {}
        self.espacio_ = EspacioGeneracion(self._historia_sp500)
        X = self.espacio_.ajustar(train)
        # no se retiene historia posterior al corte (ni aqui ni en el espacio)
        if self._historia_sp500 is not None:
            self._historia_sp500 = self.espacio_.recortar_historia(self._historia_sp500)
        self.columnas_ = list(train.columns)
        self.columnas_trabajo_ = list(self.espacio_.columnas_trabajo_)
        self.fin_train_ = self.espacio_.fin_train_
        self.n_train_, self.d_ = X.shape
        self.n_regimenes_ = max(2, int(reg.max()) + 1)
        self.P_ = matriz_transicion(reg, n=self.n_regimenes_)
        self.rachas_ = rachas(pd.Series(reg, index=train.index))
        self.regimen_final_ = int(reg[-1])
        self.frecuencia_regimen_ = np.bincount(reg, minlength=self.n_regimenes_) / reg.size
        largo = min(self.largo_contexto_efectivo, len(X))
        self.contexto_ = X[len(X) - largo:].copy()
        self.reg_contexto_ = reg[len(reg) - largo:].copy()
        inicio = time.perf_counter()
        self._fit(X.copy(), reg.copy(), train.index)
        self.tiempo_ajuste_ = time.perf_counter() - inicio
        self._ajustado = True
        return self

    @staticmethod
    def _validar_train(train: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(train, pd.DataFrame) or train.empty:
            raise ValueError("train debe ser un DataFrame no vacio.")
        if not isinstance(train.index, pd.DatetimeIndex):
            raise ValueError("train debe tener un DatetimeIndex.")
        if not train.index.is_monotonic_increasing or train.index.has_duplicates:
            raise ValueError("El indice de train debe ser creciente y sin duplicados.")
        if COL_REGIMEN in train.columns:
            raise ValueError(f"train no puede llevar la columna {COL_REGIMEN!r}; pasala en `regimes`.")
        if train.columns.has_duplicates:
            raise ValueError("train tiene columnas duplicadas.")
        valores = train.to_numpy(dtype=float)
        if not np.isfinite(valores).all():
            raise ValueError("train contiene NaN o infinitos; recorta el warm-up antes de ajustar.")
        return train.astype(float)

    @staticmethod
    def _validar_regimen_train(regimes, train: pd.DataFrame) -> np.ndarray:
        if regimes is None:
            return np.zeros(len(train), dtype=int)
        if isinstance(regimes, pd.Series):
            if not regimes.index.equals(train.index):
                raise ValueError("`regimes` debe tener exactamente el mismo indice que train.")
            valores = regimes.to_numpy()
        else:
            valores = np.asarray(regimes)
        if valores.shape != (len(train),):
            raise ValueError(f"`regimes` debe tener longitud {len(train)}; llego {valores.shape}.")
        if pd.isna(valores).any():
            raise ValueError("`regimes` contiene NaN.")
        enteros = valores.astype(int)
        if not np.array_equal(enteros, valores) or enteros.min() < 0:
            raise ValueError("`regimes` debe contener enteros no negativos.")
        return enteros

    # ----------------------------------------------------------------- sample
    def resolver_regimenes(self, n_paths: int, length: int, regimes, rng: np.random.Generator) -> np.ndarray:
        """Matriz ``(n_paths, length)`` de regimenes segun el contrato de ``sample``."""
        if regimes is None or isinstance(regimes, dict):
            opciones = self._opciones_cadena(regimes)
            return simular_regimenes(
                n_paths, length, rng, self.P_, self.rachas_, ultimo=self.regimen_final_,
                frecuencia=self.frecuencia_regimen_, **opciones,
            )
        if isinstance(regimes, str):
            raise TypeError(
                "`regimes` no admite cadenas de texto; para elegir como se simula la cadena pasa "
                "un dict, p. ej. regimes={'inicial': 'estacionaria', 'duraciones': 'empiricas'}."
            )
        valores = np.asarray(regimes.to_numpy() if isinstance(regimes, (pd.Series, pd.DataFrame)) else regimes)
        if pd.isna(valores).any():
            raise ValueError("`regimes` contiene NaN.")
        enteros = valores.astype(int)
        if not np.array_equal(enteros, valores):
            raise ValueError("`regimes` debe contener enteros.")
        if enteros.shape == (length,):
            enteros = np.tile(enteros, (n_paths, 1))
        elif enteros.shape != (n_paths, length):
            raise ValueError(
                f"`regimes` debe tener forma ({length},) o ({n_paths}, {length}); llego {enteros.shape}."
            )
        if enteros.min() < 0 or enteros.max() >= self.n_regimenes_:
            raise ValueError(f"`regimes` debe estar en [0, {self.n_regimenes_ - 1}].")
        return np.ascontiguousarray(enteros)

    @staticmethod
    def _opciones_cadena(regimes: dict | None) -> dict[str, str]:
        """``{"inicial", "duraciones"}`` con sus valores por defecto; clave desconocida = error."""
        opciones = {"inicial": INICIALES[0], "duraciones": DURACIONES[0]}
        if regimes:
            desconocidas = sorted(set(regimes) - set(opciones))
            if desconocidas:
                raise ValueError(
                    f"Opciones de cadena desconocidas {desconocidas}; admitidas: {sorted(opciones)}."
                )
            opciones.update({k: v for k, v in regimes.items() if v is not None})
        return opciones

    def sample(
        self,
        n_paths: int,
        length: int,
        regimes: np.ndarray | pd.Series | dict | None = None,
        *,
        random_state: int | None = None,
    ) -> list[pd.DataFrame]:
        """Genera ``n_paths`` trayectorias de ``length`` sesiones.

        ``regimes``:

        - ``None``: se simula una cadena de Markov por trayectoria con la matriz
          empirica de train, continuando el ultimo regimen de train.
        - ``dict`` con ``inicial`` (``"ultimo"`` | ``"estacionaria"``) y/o
          ``duraciones`` (``"geometricas"`` | ``"empiricas"``): se simula la
          cadena con esas opciones (ver ``datos.simular_regimenes``; las claves
          que falten toman el valor por defecto, asi que ``{}`` equivale a
          ``None``). Ejemplo: ``regimes={"inicial": "estacionaria"}``.
        - vector de longitud ``length``: se impone a todas las trayectorias.
        - matriz ``(n_paths, length)``: se usa tal cual (p. ej. la salida de
          ``datos.simular_regimenes``).

        Cada trayectoria es un DataFrame con las columnas de train mas
        ``regime`` (int), fechado con dias habiles a partir del siguiente a la
        ultima fecha de train. Misma ``random_state`` => mismas trayectorias.
        El diagnostico de la llamada queda en ``diagnostico_muestreo()``.
        """
        self._exigir_ajuste()
        n_paths, length = int(n_paths), int(length)
        if n_paths < 1 or length < 1:
            raise ValueError("n_paths y length deben ser >= 1.")
        rng = np.random.default_rng(random_state)
        reg = self.resolver_regimenes(n_paths, length, regimes, rng)
        self.diagnostico_muestreo_ = {}
        Z = np.asarray(self._sample(reg.copy(), rng, self.contexto_.copy()), dtype=float)
        esperado = (n_paths, length, self.d_)
        if Z.shape != esperado:
            raise ValueError(f"{self.name}._sample devolvio {Z.shape}; se esperaba {esperado}.")
        if not np.isfinite(Z).all():
            raise FloatingPointError(f"{self.name}._sample devolvio valores no finitos.")
        indice = self.espacio_.indice_sintetico(length)
        trayectorias = []
        for k in range(n_paths):
            tray = self.espacio_.a_publico(Z[k], indice)
            if not np.isfinite(tray.to_numpy()).all():
                raise FloatingPointError(
                    f"{self.name}: la trayectoria {k} no es finita en el espacio publico "
                    "(retornos sinteticos fuera de escala al re-derivar el S&P 500)."
                )
            tray[COL_REGIMEN] = reg[k]
            trayectorias.append(tray)
        simulada = regimes is None or isinstance(regimes, dict)
        self.diagnostico_muestreo_ = {
            **self._diagnostico_regimenes(reg),
            "regimen_impuesto": not simulada,
            **(self._opciones_cadena(regimes) if simulada else {"inicial": None, "duraciones": None}),
            "random_state": random_state,
            **self.diagnostico_muestreo_,
        }
        return trayectorias

    def _diagnostico_regimenes(self, reg: np.ndarray) -> dict[str, Any]:
        """Campos comunes del diagnostico: tamanos y reparto de regimenes de la llamada."""
        una_clase = (reg == reg[:, :1]).all(axis=1)
        fraccion = np.bincount(reg.ravel(), minlength=self.n_regimenes_) / reg.size
        return {
            "n_paths": int(reg.shape[0]),
            "length": int(reg.shape[1]),
            "fraccion_regimen_1": float(fraccion[1]),
            "fraccion_por_regimen": [float(x) for x in fraccion],
            "n_trayectorias_una_clase": int(una_clase.sum()),
            "n_trayectorias_sin_crisis": int((reg == 0).all(axis=1).sum()),
        }

    def diagnostico_muestreo(self) -> dict[str, Any]:
        """Copia del diagnostico del ultimo ``sample`` (``{}`` si aun no se ha muestreado).

        Campos comunes: ``n_paths``, ``length``, ``fraccion_regimen_1`` (dias en
        regimen 1 sobre el total), ``fraccion_por_regimen``,
        ``n_trayectorias_una_clase`` (trayectorias con un solo regimen),
        ``n_trayectorias_sin_crisis`` (todo regimen 0), ``regimen_impuesto``,
        ``inicial`` y ``duraciones`` (``None`` si el regimen se impuso) y
        ``random_state``. Cada generador anade los suyos (p. ej. fracciones de
        celdas recortadas por sus topes de seguridad).
        """
        return dict(getattr(self, "diagnostico_muestreo_", {}))

    # ---------------------------------------------------------------- resumen
    def resumen(self) -> dict[str, Any]:
        """Ficha del ajuste: identidad, parametros, tamanos, tiempo y ``final_<col>`` del historial."""
        self._exigir_ajuste()
        ficha: dict[str, Any] = {
            "nombre": self.name,
            "familia": type(self).familia,
            "params": dict(self.params),
            "random_state": self.random_state,
            "n_train": int(self.n_train_),
            "d": int(self.d_),
            "d_publico": len(self.columnas_),
            "fin_train": str(self.fin_train_.date()),
            "pct_crisis_train": float(100.0 * self.frecuencia_regimen_[1:].sum()),
            "tiempo_ajuste_s": float(self.tiempo_ajuste_),
            "n_historial": len(self._historial),
        }
        if self._historial:
            for col, valor in self._historial[-1].items():
                ficha[f"final_{col}"] = valor.item() if isinstance(valor, np.generic) else valor
        return ficha

    # ------------------------------------------------------------ persistencia
    def guardar(self, directorio: str | Path) -> Path:
        """Guarda el generador ajustado en ``directorio/generador.pkl`` (pickle).

        Sobrescribible en generadores con estado no serializable (p. ej. guardar
        ``state_dict`` de torch); ``cargar`` debe sobrescribirse a la vez.
        """
        directorio = Path(directorio)
        directorio.mkdir(parents=True, exist_ok=True)
        destino = directorio / FICHERO_MODELO
        with destino.open("wb") as fh:
            pickle.dump(self, fh)
        return destino

    @classmethod
    def cargar(cls, directorio: str | Path) -> "GeneradorBase":
        """Carga un generador guardado con ``guardar`` (solo ficheros propios: es pickle)."""
        with (Path(directorio) / FICHERO_MODELO).open("rb") as fh:
            objeto = pickle.load(fh)
        if not isinstance(objeto, cls):
            raise TypeError(f"El fichero contiene {type(objeto).__name__}, no {cls.__name__}.")
        return objeto
