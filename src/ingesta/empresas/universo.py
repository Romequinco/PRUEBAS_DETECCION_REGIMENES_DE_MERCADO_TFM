"""Universo de empresas del Tema 3: correspondencia ticker <-> CIK y muestra de la sonda.

La SEC identifica cada empresa por su **CIK** (Central Index Key), un número que no cambia
nunca. Los tickers sí cambian (fusiones, cambios de nombre) y se reutilizan, así que el
identificador interno del tema es el CIK y el ticker es solo una etiqueta.

- ``tabla_tickers()``: correspondencia actual ticker -> CIK publicada por la SEC
  (``company_tickers.json``). Solo contiene empresas que hoy presentan informes: las
  desaparecidas (Lehman, Enron...) no están y hay que fijar su CIK a mano.
- ``MUESTRA_SONDA``: las 10 empresas de la prueba pequeña (8 del S&P 500 actual de sectores
  distintos + 2 desaparecidas para medir el sesgo de supervivencia).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from ingesta.comun.http import get
from ingesta.comun.rutas import corpus_dir

URL_TICKERS = "https://www.sec.gov/files/company_tickers.json"


@dataclass(frozen=True)
class Empresa:
    ticker: str          # etiqueta legible (el de la época si ya no cotiza)
    cik: int             # identificador estable de la SEC
    nombre: str
    sector: str
    activa: bool = True  # False = deslistada / quebrada (no aparece en company_tickers.json)


# 8 empresas grandes y antiguas de sectores distintos (historia larga en EDGAR) + 2 desaparecidas.
MUESTRA_SONDA: tuple[Empresa, ...] = (
    Empresa("AAPL", 320193, "Apple Inc.", "Tecnología"),
    Empresa("MSFT", 789019, "Microsoft Corp.", "Tecnología"),
    Empresa("JPM", 19617, "JPMorgan Chase & Co.", "Financiero"),
    Empresa("XOM", 34088, "Exxon Mobil Corp.", "Energía"),
    Empresa("JNJ", 200406, "Johnson & Johnson", "Salud"),
    Empresa("WMT", 104169, "Walmart Inc.", "Consumo básico"),
    Empresa("KO", 21344, "Coca-Cola Co.", "Consumo básico"),
    Empresa("GE", 40545, "General Electric Co.", "Industria"),
    Empresa("LEH", 806085, "Lehman Brothers Holdings Inc.", "Financiero", activa=False),
    Empresa("ENRNQ", 1024401, "Enron Corp.", "Energía", activa=False),
)


def cik10(cik: int | str) -> str:
    """CIK con 10 dígitos y ceros a la izquierda, como lo piden las APIs de ``data.sec.gov``."""
    return str(int(cik)).zfill(10)


def tabla_tickers(usar_cache: bool = True) -> pd.DataFrame:
    """Correspondencia actual ticker -> CIK de la SEC (``ticker``, ``cik``, ``nombre``).

    Se guarda una copia en el corpus para no pedirla en cada ejecución.
    """
    cache = corpus_dir("empresas") / "sec" / "company_tickers.json"
    if usar_cache and cache.exists():
        datos = json.loads(cache.read_text(encoding="utf-8"))
    else:
        datos = get(URL_TICKERS, por_segundo=5).json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(datos), encoding="utf-8")
    return tickers_a_tabla(datos)


def tickers_a_tabla(datos: dict) -> pd.DataFrame:
    """Convierte ``company_tickers.json`` (``{"0": {"cik_str", "ticker", "title"}, ...}``) en tabla."""
    filas = [{"ticker": v["ticker"], "cik": int(v["cik_str"]), "nombre": v["title"]} for v in datos.values()]
    return pd.DataFrame(filas, columns=["ticker", "cik", "nombre"])


def resolver_cik(ticker: str, tabla: pd.DataFrame) -> int:
    """CIK de un ticker activo. Lanza ``KeyError`` si no está (p. ej. empresa deslistada)."""
    fila = tabla.loc[tabla["ticker"].str.upper() == ticker.upper()]
    if fila.empty:
        raise KeyError(f"{ticker} no está en company_tickers.json (¿deslistada? fija su CIK a mano)")
    return int(fila["cik"].iloc[0])
