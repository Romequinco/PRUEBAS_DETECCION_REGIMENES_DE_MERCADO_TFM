"""Utilidades comunes de la validacion: separar real y sintetico, rachas y episodios.

Convenciones que comparten todos los modulos de ``validacion``:

- **Real**: el panel del tramo de entrenamiento (``datos.cargar_entrenamiento``),
  con el regimen de referencia como columna ``regime`` o pasado aparte.
- **Sintetico**: lista de trayectorias de ``persistencia.cargar_trayectorias``
  (una por ``path_id``, todas con las mismas fechas y columna ``regime``).
  Nunca se cruzan con el real por fecha: se comparan por distribucion o por
  posicion dentro de cada trayectoria.
- **Episodio**: racha contigua de un regimen. En crisis el n efectivo es el
  numero de episodios del tramo real, no el de dias; las bandas de crisis se
  calculan por episodio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regimenes.sinteticos.datos import COL_REGIMEN


def separar_real(real: pd.DataFrame, regimen_real=None) -> tuple[pd.DataFrame, np.ndarray]:
    """Panel real sin la columna de regimen y su regimen como array de enteros.

    ``regimen_real`` (Serie o array alineado) tiene prioridad; si es ``None`` se
    usa la columna ``regime`` de ``real``.
    """
    if regimen_real is None:
        if COL_REGIMEN not in real.columns:
            raise ValueError(f"Falta el regimen real: pasa `regimen_real` o incluye la columna {COL_REGIMEN!r}.")
        reg = real[COL_REGIMEN].to_numpy()
    else:
        reg = np.asarray(regimen_real.to_numpy() if isinstance(regimen_real, pd.Series) else regimen_real)
    if reg.shape != (len(real),):
        raise ValueError(f"El regimen real debe tener longitud {len(real)}; llego {reg.shape}.")
    return real.drop(columns=[COL_REGIMEN], errors="ignore"), reg.astype(int)


def separar_sintetico(sintetico: list[pd.DataFrame]) -> list[tuple[pd.DataFrame, np.ndarray]]:
    """Cada trayectoria como (panel sin ``regime``, regimen entero)."""
    if not len(sintetico):
        raise ValueError("No hay trayectorias sinteticas.")
    salida = []
    for i, tray in enumerate(sintetico):
        if COL_REGIMEN not in tray.columns:
            raise ValueError(f"La trayectoria {i} no tiene columna {COL_REGIMEN!r}.")
        salida.append((tray.drop(columns=[COL_REGIMEN]), tray[COL_REGIMEN].to_numpy().astype(int)))
    return salida


def tramos(reg: np.ndarray) -> list[tuple[int, int, int]]:
    """Rachas contiguas como ``(inicio, fin_exclusivo, regimen)`` en posiciones."""
    reg = np.asarray(reg).astype(int)
    if reg.size == 0:
        return []
    cortes = np.flatnonzero(np.diff(reg) != 0) + 1
    ini = np.r_[0, cortes]
    fin = np.r_[cortes, reg.size]
    return [(int(a), int(b), int(reg[a])) for a, b in zip(ini, fin)]


def id_episodio(reg: np.ndarray) -> np.ndarray:
    """Identificador de racha por dia (0, 1, 2, ... en orden), comun a ambos regimenes."""
    reg = np.asarray(reg).astype(int)
    out = np.empty(reg.size, dtype=int)
    for j, (a, b, _) in enumerate(tramos(reg)):
        out[a:b] = j
    return out


def estandarizador(real: pd.DataFrame, columnas: list[str] | None = None):
    """Funcion que estandariza con media y desviacion del panel real (solo train)."""
    columnas = list(real.columns) if columnas is None else list(columnas)
    media = real[columnas].mean()
    escala = real[columnas].std(ddof=1).replace(0.0, 1.0)

    def aplicar(panel: pd.DataFrame) -> np.ndarray:
        return ((panel[columnas] - media) / escala).to_numpy(dtype=float)

    return aplicar
