"""
sources.py — Descargadores por fuente para la capa de datos v2.

Cada `fetch_*` devuelve una `pd.Series` diaria/mensual indexada por fecha (o lanza
excepcion). SIN imputar: cada serie arranca en su fecha real. La orquestacion
(download.py) las cachea a data/raw/<fuente>/<nombre>.parquet y registra procedencia.

Fuentes soportadas (todo gratis): fred (API con key), yfinance, ofr (CSV publico),
github (raw CSV de datahub), academico (Ken French via pandas_datareader; Shiller xls).
stooq queda bloqueado por su challenge JS (se registra, no se descarga).
"""
from __future__ import annotations

import io
import json
import time
import urllib.request
import warnings
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) regime-tfm/2.0"}


# --------------------------------------------------------------------------- #
# Credenciales
# --------------------------------------------------------------------------- #
def fred_key() -> str | None:
    """Lee FRED_API_KEY de .env (no se imprime)."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("FRED_API_KEY="):
                return line.split("=", 1)[1].strip()
    import os
    return os.environ.get("FRED_API_KEY")


def _get(url: str, timeout: int = 30, retries: int = 3) -> bytes:
    """GET con User-Agent y reintentos cortos."""
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (k + 1))
    raise last  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# FRED (API JSON con key)
# --------------------------------------------------------------------------- #
def fetch_fred(series_id: str, start: str = "1776-07-04") -> pd.Series:
    key = fred_key()
    if not key:
        raise RuntimeError("FRED_API_KEY no disponible en .env")
    url = (
        f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}"
        f"&api_key={key}&file_type=json&observation_start={start}"
    )
    data = json.loads(_get(url))
    obs = data.get("observations", [])
    if not obs:
        raise RuntimeError(f"FRED {series_id}: sin observaciones")
    idx = pd.to_datetime([o["date"] for o in obs])
    val = pd.to_numeric([o["value"] for o in obs], errors="coerce")  # "." -> NaN
    s = pd.Series(val, index=idx, name=series_id).dropna()
    if s.empty:
        raise RuntimeError(f"FRED {series_id}: todo NaN")
    return s


# --------------------------------------------------------------------------- #
# yfinance
# --------------------------------------------------------------------------- #
def fetch_yahoo(ticker: str) -> pd.Series:
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(
            ticker, period="max", auto_adjust=True, progress=False, multi_level_index=False
        )
    if df is None or df.empty or "Close" not in df:
        raise RuntimeError(f"yfinance {ticker}: vacio")
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index)
    s.name = ticker
    return s.dropna()


# --------------------------------------------------------------------------- #
# OFR Financial Stress Index (CSV publico, multi-columna)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _ofr_fsi_csv() -> pd.DataFrame:
    url = "https://www.financialresearch.gov/financial-stress-index/data/fsi.csv"
    df = pd.read_csv(io.BytesIO(_get(url)))
    df.columns = [c.strip() for c in df.columns]
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col)


def fetch_ofr(spec: str) -> pd.Series:
    """spec = 'fsi.csv:<Columna>' (p.ej. 'fsi.csv:OFR FSI')."""
    col = spec.split(":", 1)[1] if ":" in spec else spec
    df = _ofr_fsi_csv()
    if col not in df.columns:
        raise RuntimeError(f"OFR: columna {col!r} no esta en {list(df.columns)}")
    return pd.to_numeric(df[col], errors="coerce").dropna().rename(col)


# --------------------------------------------------------------------------- #
# GitHub / datahub (raw CSV)
# --------------------------------------------------------------------------- #
def fetch_github_csv(spec: str) -> pd.Series:
    """spec = '<owner/repo>:<path/al.csv>'. Toma 1a col de fecha + 1a col numerica."""
    if ":" not in spec:
        raise RuntimeError(f"github spec sin ':': {spec}")
    repo, path = spec.split(":", 1)
    raw = None
    for branch in ("main", "master"):
        try:
            raw = _get(f"https://raw.githubusercontent.com/{repo}/{branch}/{path}")
            break
        except Exception:  # noqa: BLE001
            continue
    if raw is None:
        raise RuntimeError(f"github {spec}: no accesible (main/master)")
    df = pd.read_csv(io.BytesIO(raw))
    date_cols = [c for c in df.columns if str(c).lower() in ("date", "date_time", "time", "day")]
    if not date_cols:
        raise RuntimeError(f"github {spec}: no es serie temporal (cols={list(df.columns)[:6]})")
    dc = date_cols[0]
    df[dc] = pd.to_datetime(df[dc], errors="coerce")
    df = df.dropna(subset=[dc]).set_index(dc).sort_index()
    num = df.select_dtypes("number")
    if num.empty:
        raise RuntimeError(f"github {spec}: sin columna numerica")
    return num.iloc[:, 0].dropna().rename(num.columns[0])


# --------------------------------------------------------------------------- #
# Academico: Ken French (pandas_datareader) y Shiller (xls)
# --------------------------------------------------------------------------- #
def fetch_academico(spec: str) -> pd.Series:
    """Ken French: spec = nombre del dataset famafrench (p.ej. 'F-F_Research_Data_Factors_daily').
    Shiller: spec que empiece por 'shiller'."""
    if spec.lower().startswith("shiller"):
        # Shiller ie_data.xls, hoja 'Data'; devuelve el indice S&P (columna 'P') mensual.
        url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
        raw = _get(url, timeout=60)
        xl = pd.read_excel(io.BytesIO(raw), sheet_name="Data", skiprows=7)
        xl = xl.rename(columns={xl.columns[0]: "Date"})
        # La col Date de Shiller es YYYY.MM (float). Reconstruir fecha mensual.
        d = xl["Date"].astype(str).str.replace(".", "-", regex=False)
        idx = pd.to_datetime(d, format="%Y-%m", errors="coerce")
        s = pd.Series(pd.to_numeric(xl["P"], errors="coerce").values, index=idx, name="SHILLER_P")
        return s.dropna()
    # Ken French via pandas_datareader
    from pandas_datareader import data as pdr

    d = pdr.DataReader(spec, "famafrench", start="1926-01-01")
    df = d[0]  # primera tabla
    if hasattr(df.index, "to_timestamp"):
        df.index = df.index.to_timestamp()
    return df.iloc[:, 0].dropna().rename(f"{spec}::{df.columns[0]}")


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
FETCHERS = {
    "fred": fetch_fred,
    "yfinance": fetch_yahoo,
    "ofr": fetch_ofr,
    "github": fetch_github_csv,
    "academico": fetch_academico,
}


def fetch(fuente: str, series_id: str) -> pd.Series:
    if fuente == "stooq":
        raise RuntimeError("stooq bloqueado por challenge JS (proof-of-work); no descargable")
    fn = FETCHERS.get(fuente)
    if fn is None:
        raise RuntimeError(f"fuente desconocida: {fuente}")
    return fn(series_id)
