"""rentabilidades.py — Menú R: rentabilidades log diarias multiactivo de la pista B (ADR-010 §2.2).

Es el dato que usan Kritzman y Li (2010) para la turbulencia y Kritzman et al. (2011) para el
absorption ratio: 22 series de activos, sin z-score.

- **Series:** :data:`SERIES_R` (renta variable, renta fija, crédito, materias primas y divisas).
- **Transformación:** ``log(p_t / p_{t-1})`` entre observaciones consecutivas de la serie
  cruda (su propio calendario, como :func:`regimenes.features.transformaciones.logret`), y
  después reindexada al índice de ``pistaB_diaria.parquet``. Los días sin cotización quedan en
  NaN, sin rellenar; el día siguiente lleva la rentabilidad desde la última cotización.
- **Divisas:** rentabilidad del dólar. ``DXY``, ``DEXJPUS`` (yenes por dólar) y ``DEXSZUS``
  (francos por dólar) ya lo son; ``DEXUSAL`` (dólares por dólar australiano) se invierte.
  Sin retraso de publicación, como el resto de series diarias de mercado (ADR-003 §2.1).
- **Causal:** el valor de ``t`` solo usa cotizaciones ``<= t`` (prueba de prefijo en los tests).

Este módulo NO lo importa ``regimenes.features`` (``__init__``) ni entra en ninguna huella de la
caché v2. El panel se escribe en ``data/processed/pistaB_rentabilidades.parquet`` (archivo nuevo;
no modifica los paneles existentes ni ``_meta.json``) con su sha256 en
``pistaB_rentabilidades.json``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from regimenes import rutas

SERIES_R: dict[str, tuple[str, ...]] = {
    "renta_variable": ("SPDR_XLB", "SPDR_XLE", "SPDR_XLF", "SPDR_XLI", "SPDR_XLK", "SPDR_XLP",
                       "SPDR_XLU", "SPDR_XLV", "SPDR_XLY", "SP500", "NASDAQ_COMP", "RUSSELL2000"),
    "renta_fija": ("TLT_TREASURY", "IEF_TREASURY"),
    "credito": ("LQD_IGCREDIT", "HYG_CREDIT"),
    "materias_primas": ("GOLD_GLD", "SPGSCI"),
    "divisas": ("DXY", "DEXJPUS", "DEXSZUS", "DEXUSAL"),
}
SERIES: tuple[str, ...] = tuple(s for grupo in SERIES_R.values() for s in grupo)
INVERTIDAS: tuple[str, ...] = ("DEXUSAL",)        # dólares por AUD -> rentabilidad del dólar
SUFIJO = "_logret"

PANEL_R: Path = rutas.DATA_PROCESSED / "pistaB_rentabilidades.parquet"
META_R: Path = rutas.DATA_PROCESSED / "pistaB_rentabilidades.json"


def columna(serie: str) -> str:
    """Nombre de la columna del menú R de una serie cruda (``SP500`` -> ``SP500_logret``)."""
    return f"{serie}{SUFIJO}"


def columnas_r() -> tuple[str, ...]:
    """Las 22 columnas del menú R en el orden de :data:`SERIES_R`."""
    return tuple(columna(s) for s in SERIES)


def rentabilidad_log(precios: pd.Series, *, invertir: bool = False) -> pd.Series:
    """Rentabilidad log entre cotizaciones consecutivas (sin NaN ni precios <= 0)."""
    p = pd.Series(precios, dtype=float).sort_index()
    p = p[np.isfinite(p) & (p > 0)]
    r = np.log(p / p.shift(1))
    return -r if invertir else r


def panel_rentabilidades(crudos: Mapping[str, pd.Series], indice: pd.DatetimeIndex) -> pd.DataFrame:
    """Panel del menú R sobre ``indice`` a partir de los precios crudos de :data:`SERIES`.

    Falla si falta alguna serie (el menú tiene exactamente las 22)."""
    faltan = [s for s in SERIES if s not in crudos]
    if faltan:
        raise KeyError(f"menú R: faltan series crudas {faltan}")
    indice = pd.DatetimeIndex(indice)
    cols = {columna(s): rentabilidad_log(crudos[s], invertir=s in INVERTIDAS).reindex(indice)
            for s in SERIES}
    out = pd.DataFrame(cols, index=indice)
    out.index.name = indice.name
    return out


def _fuentes() -> dict[str, str]:
    cov = pd.read_csv(rutas.DATA_RAW / "coverage_report.csv")
    return dict(zip(cov["nombre"], cov["fuente"]))


def rutas_crudas(raiz: Path | None = None) -> dict[str, Path]:
    """Ruta en ``data/raw/<fuente>/<serie>.parquet`` de cada serie del menú R."""
    raiz = Path(raiz) if raiz is not None else rutas.DATA_RAW
    fuente = _fuentes() if raiz == rutas.DATA_RAW else {}
    out = {}
    for s in SERIES:
        candidatos = [raiz / str(fuente[s]) / f"{s}.parquet"] if s in fuente else sorted(raiz.glob(f"*/{s}.parquet"))
        candidatos = [c for c in candidatos if c.exists()]
        if len(candidatos) != 1:
            raise FileNotFoundError(f"menú R: no se encuentra de forma unívoca {s}.parquet en {raiz}")
        out[s] = candidatos[0]
    return out


def cargar_crudos(raiz: Path | None = None) -> dict[str, pd.Series]:
    """Precios crudos de las 22 series desde ``data/raw`` (no descarga nada)."""
    from regimenes.features.paneles import seleccionar_raw

    return {s: seleccionar_raw(pd.read_parquet(r), s) for s, r in rutas_crudas(raiz).items()}


def sha256(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def indice_pista_b() -> pd.DatetimeIndex:
    """Índice de ``data/processed/pistaB_diaria.parquet`` (solo lo lee)."""
    idx = pd.read_parquet(rutas.DATA_PROCESSED / "pistaB_diaria.parquet", columns=[]).index
    return pd.DatetimeIndex(idx).sort_values()


def construir_panel_r(raiz: Path | None = None, indice: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Panel R desde ``data/raw`` sobre el índice de la pista B."""
    return panel_rentabilidades(cargar_crudos(raiz), indice_pista_b() if indice is None else indice)


def escribir_panel_r(ruta: Path = PANEL_R, meta: Path = META_R) -> dict:
    """Construye y escribe el panel R y su ficha (sha256 del panel y de cada crudo).

    Escritura atómica; devuelve la ficha."""
    from regimenes.benchmark.cache import _write_atomic

    panel = construir_panel_r()
    _write_atomic(Path(ruta), lambda tmp: panel.to_parquet(tmp))
    crudos = rutas_crudas()
    ficha = {
        "archivo": Path(ruta).name, "sha256": sha256(ruta), "n_filas": len(panel),
        "inicio": str(panel.index[0].date()), "fin": str(panel.index[-1].date()),
        "columnas": list(panel.columns), "invertidas": [columna(s) for s in INVERTIDAS],
        "nan_por_columna": {c: int(v) for c, v in panel.isna().sum().items()},
        "primer_dato": {c: str(panel[c].first_valid_index().date()) for c in panel},
        "crudos_sha256": {s: sha256(r) for s, r in crudos.items()},
        "regla": "log(p_t/p_{t-1}) entre cotizaciones consecutivas, reindexada al índice de "
                 "pistaB_diaria.parquet; NaN sin rellenar; DEXUSAL invertida; sin retraso",
    }
    texto = json.dumps(ficha, ensure_ascii=False, indent=1)
    _write_atomic(Path(meta), lambda tmp: Path(tmp).write_text(texto, encoding="utf-8"))
    return ficha


def leer_panel_r(ruta: Path = PANEL_R, meta: Path = META_R) -> pd.DataFrame:
    """Lee el panel R verificando su sha256 frente a la ficha."""
    ficha = json.loads(Path(meta).read_text(encoding="utf-8"))
    real = sha256(ruta)
    if ficha.get("sha256") != real:
        raise ValueError(f"{Path(ruta).name}: sha256 {real} no coincide con {Path(meta).name}; "
                         "reconstrúyelo con escribir_panel_r()")
    panel = pd.read_parquet(ruta)
    panel.index = pd.DatetimeIndex(panel.index)
    if list(panel.columns) != list(columnas_r()):
        raise ValueError(f"{Path(ruta).name}: columnas distintas de las del menú R")
    return panel
