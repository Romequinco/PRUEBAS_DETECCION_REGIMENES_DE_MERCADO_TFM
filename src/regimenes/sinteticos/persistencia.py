"""Persistencia de trayectorias y fichas de ajuste de los generadores.

Por que existe: los notebooks 16-18 consumen las trayectorias generadas en el 15
sin volver a ajustar nada, asi que el formato en disco es parte del contrato:

- Trayectorias: UN parquet "largo" por generador y pista en
  ``rutas.DATA_SINTETICOS/<nombre>/pista<X>/trayectorias.parquet`` con columnas
  ``path_id``, ``date``, features y ``regime`` (gitignored: se regeneran).
- Ficha y convergencia: ``<nombre>_pista<X>_resumen.json`` y
  ``<nombre>_pista<X>_historial.csv`` en ``rutas.RESULTS_SINTETICOS/'generadores'``
  (pequenos, versionables).

Todas las funciones aceptan ``base`` para redirigir la raiz (tests con ``tmp_path``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regimenes import rutas
from regimenes.sinteticos.datos import COL_REGIMEN

FICHERO_TRAYECTORIAS = "trayectorias.parquet"


def dir_trayectorias(nombre: str, pista: str, base: Path | None = None) -> Path:
    """``<base o DATA_SINTETICOS>/<nombre>/pista<X>``."""
    return Path(base or rutas.DATA_SINTETICOS) / nombre / f"pista{pista.upper()}"


def dir_resumenes(base: Path | None = None) -> Path:
    """``<base o RESULTS_SINTETICOS>/generadores``."""
    return Path(base or rutas.RESULTS_SINTETICOS) / "generadores"


def trayectorias_a_largo(trayectorias: list[pd.DataFrame]) -> pd.DataFrame:
    """Lista de trayectorias -> tabla larga (``path_id``, ``date``, features, ``regime``)."""
    if not trayectorias:
        raise ValueError("No hay trayectorias que guardar.")
    piezas = []
    for k, tray in enumerate(trayectorias):
        pieza = tray.reset_index(names="date")
        pieza.insert(0, "path_id", k)
        piezas.append(pieza)
    largo = pd.concat(piezas, ignore_index=True)
    largo["path_id"] = largo["path_id"].astype("int32")
    largo[COL_REGIMEN] = largo[COL_REGIMEN].astype("int8")
    return largo


def largo_a_trayectorias(largo: pd.DataFrame) -> list[pd.DataFrame]:
    """Inversa de ``trayectorias_a_largo`` (``regime`` vuelve a int64)."""
    trayectorias = []
    for _, pieza in largo.groupby("path_id", sort=True):
        tray = pieza.drop(columns="path_id").set_index("date")
        tray.index = pd.DatetimeIndex(tray.index).astype("datetime64[ns]")
        tray.index.name = None
        tray[COL_REGIMEN] = tray[COL_REGIMEN].astype(np.int64)
        trayectorias.append(tray)
    return trayectorias


def guardar_trayectorias(
    trayectorias: list[pd.DataFrame], nombre: str, pista: str, base: Path | None = None
) -> Path:
    """Escribe el parquet largo de un generador y devuelve su ruta."""
    destino = dir_trayectorias(nombre, pista, base) / FICHERO_TRAYECTORIAS
    destino.parent.mkdir(parents=True, exist_ok=True)
    trayectorias_a_largo(trayectorias).to_parquet(destino, index=False)
    return destino


def cargar_trayectorias(nombre: str, pista: str, base: Path | None = None) -> list[pd.DataFrame]:
    """Lee las trayectorias de un generador como lista de DataFrames."""
    origen = dir_trayectorias(nombre, pista, base) / FICHERO_TRAYECTORIAS
    if not origen.exists():
        raise FileNotFoundError(f"No existe {origen}; ejecuta antes 15_sinteticos_generadores.")
    return largo_a_trayectorias(pd.read_parquet(origen))


def _serializable(valor: Any) -> Any:
    if isinstance(valor, dict):
        return {str(k): _serializable(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_serializable(v) for v in valor]
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if isinstance(valor, np.generic):
        return valor.item()
    if isinstance(valor, (pd.Timestamp, Path)):
        return str(valor)
    return valor


def guardar_resumen(generador, pista: str, base: Path | None = None, **extra: Any) -> Path:
    """Guarda ``generador.resumen()`` (+ ``extra``) en JSON y su historial en CSV.

    Devuelve la ruta del JSON. El CSV de historial solo se escribe si el
    generador registro alguna fila de convergencia.
    """
    carpeta = dir_resumenes(base)
    carpeta.mkdir(parents=True, exist_ok=True)
    raiz = f"{generador.name}_pista{pista.upper()}"
    ficha = {**generador.resumen(), "pista": pista.upper(), **extra}
    destino = carpeta / f"{raiz}_resumen.json"
    destino.write_text(
        json.dumps(_serializable(ficha), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    historial = generador.historial
    if not historial.empty:
        historial.to_csv(carpeta / f"{raiz}_historial.csv", index=False)
    return destino


def cargar_resumenes(base: Path | None = None) -> pd.DataFrame:
    """Tabla con todas las fichas ``*_resumen.json`` guardadas (una fila por generador y pista)."""
    filas = [
        json.loads(ruta.read_text(encoding="utf-8"))
        for ruta in sorted(dir_resumenes(base).glob("*_resumen.json"))
    ]
    return pd.DataFrame(filas)
