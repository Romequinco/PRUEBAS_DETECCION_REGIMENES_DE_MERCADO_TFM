"""Datos de ajuste de los generadores y aritmetica del regimen de referencia.

Por que existe: todos los generadores deben ajustarse con EXACTAMENTE el mismo
panel y la misma etiqueta de regimen, y esa etiqueta no puede salir de un
detector del benchmark (circularidad: el detector jugaria en casa al evaluarse
sobre lo generado). Aqui se fija una unica definicion:

- **Regimen de referencia** binario: 1 = crisis si el dia cae dentro de alguna
  ventana ``[pico, suelo]`` (extremos incluidos) de ``crisis_windows`` de la
  pista en ``configs/benchmark_spec.yaml``; 0 = calma en otro caso.
- **Panel de ajuste**: nucleo de la pista (``CORE_A`` / ``CORE_B``) mas el
  log-retorno crudo ``SP500_ret``, recortado a ``<= fin_train`` y sin NaN
  (se elimina el warm-up de las features causales).
- ``fin_train`` no puede partir un episodio de crisis: un episodio a medias
  sesgaria duraciones y transiciones. ``cargar_entrenamiento`` lo comprueba.

Las utilidades de cadena (``rachas``, ``matriz_transicion``, ``simular_cadena``,
``distribucion_estacionaria``, ``simular_regimenes``) y ``momentos_por_regimen``
son numpy/pandas puro y no leen disco. Los imports de ``regimenes.benchmark`` y
``regimenes.evaluacion`` son perezosos para que importar ``regimenes.sinteticos``
no arrastre todo el benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COL_RET = "SP500_ret"
COL_REGIMEN = "regime"

# Opciones de la cadena de regimenes simulada (ver ``simular_regimenes``).
INICIALES = ("ultimo", "estacionaria")
DURACIONES = ("geometricas", "empiricas")


def nucleo_pista(pista: str) -> list[str]:
    """Columnas del nucleo de la pista (``CORE_A`` o ``CORE_B``), en su orden."""
    from regimenes.detectores.registry import CORE_A, CORE_B

    pista = pista.upper()
    if pista not in {"A", "B"}:
        raise ValueError("pista debe ser 'A' o 'B'.")
    return list(CORE_A if pista == "A" else CORE_B)


def ventanas_crisis(pista: str) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Ventanas ``[pico, suelo]`` congeladas de la pista, como ``Timestamp``."""
    from regimenes.evaluacion.ranking import track_crisis_windows

    return {
        nombre: (pd.Timestamp(ini), pd.Timestamp(fin))
        for nombre, (ini, fin) in track_crisis_windows(pista.upper()).items()
    }


def regimen_referencia(index: pd.DatetimeIndex, pista: str) -> pd.Series:
    """Regimen binario (0 calma, 1 crisis) de cada fecha de ``index``.

    Un dia es crisis si esta dentro de ``[pico, suelo]`` (inclusive) de alguna
    ventana de ``crisis_windows`` de la pista. No usa ningun detector.
    """
    index = pd.DatetimeIndex(index)
    reg = np.zeros(len(index), dtype=int)
    for ini, fin in ventanas_crisis(pista).values():
        reg[(index >= ini) & (index <= fin)] = 1
    return pd.Series(reg, index=index, name=COL_REGIMEN)


def episodio_cruzado(pista: str, fin_train: str | pd.Timestamp) -> str | None:
    """Nombre del episodio de crisis que ``fin_train`` parte en dos (o ``None``)."""
    corte = pd.Timestamp(fin_train)
    for nombre, (ini, fin) in ventanas_crisis(pista).items():
        if ini <= corte < fin:
            return nombre
    return None


def corte_sin_cruce(pista: str, fin_train: str | pd.Timestamp) -> pd.Timestamp:
    """Fecha mas cercana a ``fin_train`` que no parte ningun episodio de crisis.

    Si ``fin_train`` ya es valida se devuelve tal cual; si cae dentro de un
    episodio se elige entre el dia anterior al pico y el propio suelo, el mas
    cercano en dias naturales.
    """
    corte = pd.Timestamp(fin_train)
    nombre = episodio_cruzado(pista, corte)
    if nombre is None:
        return corte
    ini, fin = ventanas_crisis(pista)[nombre]
    antes = ini - pd.Timedelta(days=1)
    return antes if (corte - antes) <= (fin - corte) else fin


def cargar_entrenamiento(
    pista: str,
    fin_train: str | pd.Timestamp,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Panel de ajuste y regimen de referencia de una pista.

    Parameters
    ----------
    pista:
        ``"A"`` o ``"B"``.
    fin_train:
        Ultima fecha (inclusive) que entra en el ajuste.
    features:
        Columnas del panel; ``None`` o lista vacia = nucleo de la pista. Siempre
        se anade ``SP500_ret`` (al final) si no esta.

    Returns
    -------
    (panel, regimen)
        ``panel`` sin NaN, indice ``datetime64[ns]`` ``<= fin_train``;
        ``regimen`` Serie int (0/1) alineada con ``panel``.

    Raises
    ------
    ValueError
        Si ``fin_train`` parte un episodio de crisis (el mensaje propone la
        fecha valida mas cercana) o si no queda ninguna fila completa.
    """
    from regimenes.benchmark.ejecucion import load_track_panel

    pista = pista.upper()
    corte = pd.Timestamp(fin_train)
    cruzado = episodio_cruzado(pista, corte)
    if cruzado is not None:
        propuesta = corte_sin_cruce(pista, corte).date()
        raise ValueError(
            f"fin_train={corte.date()} parte el episodio {cruzado!r} de la pista {pista}; "
            f"usa {propuesta} (fecha mas cercana que no cruza ningun episodio)."
        )
    columnas = list(features) if features else nucleo_pista(pista)
    if COL_RET not in columnas:
        columnas.append(COL_RET)
    completo = load_track_panel(pista)
    faltan = [c for c in columnas if c not in completo.columns]
    if faltan:
        raise KeyError(f"La pista {pista} no tiene las columnas {faltan}.")
    panel = completo.loc[completo.index <= corte, columnas].dropna().astype(float)
    if panel.empty:
        raise ValueError(f"Sin filas completas en la pista {pista} hasta {corte.date()}.")
    panel.index = pd.DatetimeIndex(panel.index).astype("datetime64[ns]")
    return panel, regimen_referencia(panel.index, pista)


# --------------------------------------------------------------------------- cadena


def _como_regimen(reg) -> np.ndarray:
    valores = np.asarray(reg)
    if valores.ndim != 1:
        raise ValueError("El regimen debe ser un vector 1-D.")
    enteros = valores.astype(int)
    if not np.array_equal(enteros, valores):
        raise ValueError("El regimen debe contener enteros.")
    if enteros.size and enteros.min() < 0:
        raise ValueError("El regimen no admite etiquetas negativas.")
    return enteros


def rachas(reg) -> pd.DataFrame:
    """Tabla de rachas del regimen: ``inicio``, ``fin``, ``regimen``, ``duracion``.

    ``inicio``/``fin`` son etiquetas del indice si ``reg`` es una Serie y
    posiciones enteras si es un array; ``fin`` es inclusivo y ``duracion`` se
    mide en observaciones.
    """
    etiquetas = reg.index if isinstance(reg, pd.Series) else None
    valores = _como_regimen(reg)
    columnas = ["inicio", "fin", "regimen", "duracion"]
    if valores.size == 0:
        return pd.DataFrame(columns=columnas)
    cortes = np.flatnonzero(np.diff(valores) != 0) + 1
    ini = np.r_[0, cortes]
    fin = np.r_[cortes - 1, valores.size - 1]
    tabla = pd.DataFrame({
        "inicio": ini if etiquetas is None else etiquetas[ini],
        "fin": fin if etiquetas is None else etiquetas[fin],
        "regimen": valores[ini],
        "duracion": fin - ini + 1,
    })
    return tabla[columnas]


def matriz_transicion(reg, n: int = 2) -> np.ndarray:
    """Matriz de transicion empirica ``P[i, j] = P(s_{t+1}=j | s_t=i)`` (n x n).

    Estimador de maxima verosimilitud por conteo de transiciones consecutivas.
    Una fila sin transiciones observadas (estado nunca visitado, o visitado solo
    en la ultima observacion) se deja absorbente (``P[i, i] = 1``) para que la
    matriz siga siendo estocastica.
    """
    valores = _como_regimen(reg)
    if valores.size and valores.max() >= n:
        raise ValueError(f"El regimen tiene etiquetas >= n={n}.")
    conteo = np.zeros((n, n), dtype=float)
    if valores.size > 1:
        np.add.at(conteo, (valores[:-1], valores[1:]), 1.0)
    total = conteo.sum(axis=1, keepdims=True)
    P = np.divide(conteo, total, out=np.zeros_like(conteo), where=total > 0)
    vacias = np.flatnonzero(total.ravel() == 0)
    P[vacias, vacias] = 1.0
    return P


def simular_cadena(
    P: np.ndarray,
    length: int,
    rng: np.random.Generator,
    inicial: int | np.ndarray = 0,
    n_paths: int | None = None,
) -> np.ndarray:
    """Simula una cadena de Markov de matriz ``P`` como CONTINUACION de ``inicial``.

    ``inicial`` es el estado de la ultima observacion conocida (no forma parte
    de la salida): el primer estado simulado se extrae de ``P[inicial]``. Puede
    ser un entero (el mismo para todas las trayectorias) o un vector
    ``(n_paths,)`` con un estado por trayectoria.

    Returns
    -------
    numpy.ndarray
        ``(length,)`` si ``n_paths`` es ``None``; ``(n_paths, length)`` si no
        (una cadena independiente por trayectoria).
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P debe ser una matriz cuadrada.")
    if np.any(P < 0) or not np.allclose(P.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("P debe ser estocastica por filas.")
    m = 1 if n_paths is None else int(n_paths)
    actual = np.asarray(inicial, dtype=int)
    if actual.ndim == 0:
        actual = np.full(m, int(actual), dtype=int)
    elif actual.shape != (m,):
        raise ValueError(f"`inicial` debe ser un entero o un vector ({m},).")
    if actual.min() < 0 or actual.max() >= P.shape[0]:
        raise ValueError("Estado inicial fuera de rango.")
    acumulada = np.cumsum(P, axis=1)
    acumulada[:, -1] = 1.0
    u = rng.random((m, int(length)))
    estados = np.empty((m, int(length)), dtype=int)
    for t in range(int(length)):
        actual = (u[:, t, None] >= acumulada[actual]).sum(axis=1)
        estados[:, t] = actual
    return estados[0] if n_paths is None else estados


def distribucion_estacionaria(P: np.ndarray, respaldo: np.ndarray | None = None) -> np.ndarray:
    """Distribucion estacionaria ``pi`` de ``P`` (``pi P = pi``, suma 1).

    Si la cadena es irreducible ``pi`` es unica y se obtiene del sistema lineal.
    Si no lo es (p. ej. ``P`` identidad cuando train tiene un solo regimen) hay
    infinitas: se devuelve el limite de ``respaldo @ P^n`` (media de dos
    potencias consecutivas, por si hay periodicidad), con ``respaldo`` la
    distribucion de partida (por defecto uniforme; ``GeneradorBase`` pasa la
    frecuencia de cada regimen en train, de modo que un regimen nunca visto
    recibe probabilidad 0).
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P debe ser una matriz cuadrada.")
    n = P.shape[0]
    unitarios = int((np.abs(np.linalg.eigvals(P) - 1.0) < 1e-10).sum())
    if unitarios == 1:
        sistema = np.vstack([P.T - np.eye(n), np.ones((1, n))])
        pi = np.linalg.lstsq(sistema, np.r_[np.zeros(n), 1.0], rcond=None)[0]
    else:
        w = np.full(n, 1.0 / n) if respaldo is None else np.asarray(respaldo, dtype=float)
        potencia = np.linalg.matrix_power(P, 2**20)
        pi = 0.5 * (w @ potencia + w @ potencia @ P)
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


def _duraciones_por_regimen(tabla: pd.DataFrame | None, n: int) -> list[np.ndarray]:
    if tabla is None:
        raise ValueError("duraciones='empiricas' necesita `rachas` (tabla de rachas de train).")
    return [tabla.loc[tabla["regimen"] == k, "duracion"].to_numpy(dtype=int) for k in range(n)]


def simular_regimenes(
    n_paths: int,
    length: int,
    rng: np.random.Generator,
    P: np.ndarray,
    rachas: pd.DataFrame | None = None,
    *,
    inicial: str = "ultimo",
    duraciones: str = "geometricas",
    ultimo: int = 0,
    frecuencia: np.ndarray | None = None,
) -> np.ndarray:
    """Matriz ``(n_paths, length)`` de regimenes simulados (una cadena por trayectoria).

    Es lo que ``GeneradorBase.sample`` usa cuando no se le impone el regimen; la
    matriz devuelta se puede pasar tal cual como ``regimes`` a ``sample``.

    Parameters
    ----------
    P:
        Matriz de transicion empirica de train (``matriz_transicion``).
    rachas:
        Tabla de ``rachas`` de train; solo hace falta con ``duraciones="empiricas"``.
    inicial:
        - ``"ultimo"`` (defecto): la cadena continua el estado ``ultimo`` (el de
          la ultima fila de train). Todas las trayectorias arrancan igual, asi
          que con una calma larga por delante la fraccion de crisis de un
          horizonte finito queda por debajo de la estacionaria.
        - ``"estacionaria"``: el estado de partida de cada trayectoria se sortea
          de la distribucion estacionaria (de ``P`` con duraciones geometricas;
          del proceso de renovacion, proporcional a frecuencia de visita x
          duracion media, con duraciones empiricas). La fraccion esperada de
          cada regimen es entonces la estacionaria en cualquier horizonte.
    duraciones:
        - ``"geometricas"`` (defecto): cadena de Markov de primer orden con
          ``P``; la duracion de cada racha es geometrica de media ``1/(1-P[k,k])``.
        - ``"empiricas"``: semi-Markov. Cada racha dura una de las duraciones
          REALES de las rachas de train de su regimen, remuestreada con
          reemplazo; al acabar se pasa a otro regimen con las probabilidades de
          salto de ``P`` (su diagonal a cero y renormalizada; con dos regimenes
          se alternan). La primera racha es una racha EN CURSO: su duracion se
          sortea con probabilidad proporcional a la duracion (una racha larga es
          mas facil de encontrar empezada) y de ella queda un tiempo residual
          uniforme en ``1..duracion``. Se usan todas las rachas de train,
          incluidas la primera y la ultima, que estan censuradas por los
          extremos de la muestra (sesgo a la baja pequeno en su duracion).
    ultimo:
        Estado de la ultima fila de train (solo con ``inicial="ultimo"``).
    frecuencia:
        Frecuencia de cada regimen en train; desempata la distribucion
        estacionaria cuando ``P`` es reducible.

    El consumo de ``rng`` con los valores por defecto es identico al de
    ``simular_cadena(P, length, rng, inicial=ultimo, n_paths=n_paths)``.
    """
    if inicial not in INICIALES:
        raise ValueError(f"inicial debe estar en {INICIALES}; llego {inicial!r}.")
    if duraciones not in DURACIONES:
        raise ValueError(f"duraciones debe estar en {DURACIONES}; llego {duraciones!r}.")
    P = np.asarray(P, dtype=float)
    n_paths, length, K = int(n_paths), int(length), P.shape[0]
    if duraciones == "geometricas":
        if inicial == "ultimo":
            return simular_cadena(P, length, rng, inicial=int(ultimo), n_paths=n_paths)
        pi = distribucion_estacionaria(P, frecuencia)
        partida = rng.choice(K, size=n_paths, p=pi)
        return simular_cadena(P, length, rng, inicial=partida, n_paths=n_paths)

    bancos = _duraciones_por_regimen(rachas, K)
    # cadena de saltos: a donde se va al terminar una racha (nunca al mismo regimen)
    salto = P.copy()
    np.fill_diagonal(salto, 0.0)
    total = salto.sum(axis=1, keepdims=True)
    salto = np.divide(salto, total, out=np.zeros_like(salto), where=total > 0)
    sin_salida = total.ravel() <= 0
    media = np.array([b.mean() if b.size else 0.0 for b in bancos])
    if inicial == "estacionaria":
        visitables = ~sin_salida & (media > 0)
        if visitables.sum() >= 2:
            sub = salto[np.ix_(visitables, visitables)]
            sub = sub / sub.sum(axis=1, keepdims=True)
            peso = np.zeros(K)
            peso[visitables] = distribucion_estacionaria(sub) * media[visitables]
        else:  # cadena degenerada: se reparte segun la frecuencia de train
            peso = np.full(K, 1.0 / K) if frecuencia is None else np.asarray(frecuencia, dtype=float)
        partida = rng.choice(K, size=n_paths, p=peso / peso.sum())
    else:
        partida = np.full(n_paths, int(ultimo), dtype=int)

    def banco_de(k: int) -> np.ndarray:
        if bancos[k].size == 0:
            raise ValueError(f"El regimen {k} no tiene rachas en train: no hay duraciones que remuestrear.")
        return bancos[k]

    salida = np.empty((n_paths, length), dtype=int)
    for i in range(n_paths):
        k = int(partida[i])
        banco = banco_de(k)
        # racha en curso: duracion sesgada por longitud y tiempo residual uniforme
        duracion = int(rng.choice(banco, p=banco / banco.sum()))
        queda = int(rng.integers(1, duracion + 1))
        t = 0
        while True:
            salida[i, t:t + queda] = k
            t += queda
            if t >= length:
                break
            if sin_salida[k]:  # regimen absorbente en train: no se sale
                salida[i, t:] = k
                break
            k = int(rng.choice(K, p=salto[k]))
            queda = int(rng.choice(banco_de(k)))
    return salida


# --------------------------------------------------------------------------- momentos


def momentos_por_regimen(
    datos: pd.DataFrame | list[pd.DataFrame],
    regimen: pd.Series | np.ndarray | None = None,
    columnas: list[str] | None = None,
) -> pd.DataFrame:
    """Media y desviacion por columna y regimen (tabla comun real frente a sintetico).

    Parameters
    ----------
    datos:
        Un panel (real o una trayectoria) o una lista de trayectorias de
        ``sample`` (se apilan: los momentos son sobre todos los dias de todas).
    regimen:
        Regimen alineado con ``datos`` cuando este no lleva la columna
        ``regime`` (caso del panel real). Si ``datos`` ya la lleva, no se pasa.
    columnas:
        Subconjunto de columnas; por defecto todas menos ``regime``.

    Returns
    -------
    pandas.DataFrame
        Indice ``(regimen, columna)`` y columnas ``n`` (dias), ``media`` y
        ``desviacion`` (``ddof=1``). Dos tablas (real y sintetica) se comparan
        uniendolas por indice.
    """
    if isinstance(datos, pd.DataFrame):
        tabla = datos
    else:
        if not len(datos):
            raise ValueError("No hay trayectorias.")
        tabla = pd.concat(list(datos), ignore_index=True)
    if regimen is None:
        if COL_REGIMEN not in tabla.columns:
            raise ValueError(f"Falta el regimen: pasa `regimen` o incluye la columna {COL_REGIMEN!r}.")
        reg = tabla[COL_REGIMEN].to_numpy()
    else:
        reg = np.asarray(regimen.to_numpy() if isinstance(regimen, pd.Series) else regimen)
        if reg.shape != (len(tabla),):
            raise ValueError(f"`regimen` debe tener longitud {len(tabla)}; llego {reg.shape}.")
    columnas = [c for c in tabla.columns if c != COL_REGIMEN] if columnas is None else list(columnas)
    reg = reg.astype(int)
    filas = []
    for k in np.unique(reg):
        bloque = tabla.loc[reg == k, columnas]
        for col in columnas:
            filas.append({
                "regimen": int(k), "columna": col, "n": int(len(bloque)),
                "media": float(bloque[col].mean()), "desviacion": float(bloque[col].std(ddof=1)),
            })
    return pd.DataFrame(filas).set_index(["regimen", "columna"])
