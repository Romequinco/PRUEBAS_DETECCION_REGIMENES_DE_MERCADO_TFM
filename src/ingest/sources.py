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
def fetch_github_csv(spec: str, columna: str | None = None) -> pd.Series:
    """spec = '<owner/repo>:<path/al.csv>'. Toma 1a col de fecha + 1a col numerica
    (o la columna `columna` si se indica; ahi un 0.0 se trata como faltante, que es como
    datahub marca los meses sin dato en p. ej. PE10)."""
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
    if columna is not None:
        if columna not in num.columns:
            raise RuntimeError(f"github {spec}: sin columna {columna!r} (cols={list(num.columns)})")
        return num[columna].mask(num[columna] == 0.0).dropna().rename(columna)
    return num.iloc[:, 0].dropna().rename(num.columns[0])


# --------------------------------------------------------------------------- #
# FRED derivado: spread A - B  (id = "A,B")
# --------------------------------------------------------------------------- #
def fetch_fred_spread(spec: str) -> pd.Series:
    a, b = [x.strip() for x in spec.split(",")[:2]]
    sa, sb = fetch_fred(a), fetch_fred(b)
    common = sa.index.intersection(sb.index)
    if len(common) == 0:
        raise RuntimeError(f"fred_spread {spec}: sin fechas comunes")
    return (sa.loc[common] - sb.loc[common]).dropna().rename(f"{a}-{b}")


# --------------------------------------------------------------------------- #
# Academico: enruta por id/url. Devuelve Series o DataFrame (paneles multi-columna).
# --------------------------------------------------------------------------- #
def _shiller_dates(col: pd.Series) -> pd.DatetimeIndex:
    """Convierte la columna Date de Shiller (float ``AAAA.MM``) en fechas de inicio de mes.

    Ojo: Shiller codifica octubre como ``1871.1`` (float). Un parseo por texto
    (``'1871.1' -> '1871-1'``) lo convierte en ENERO y duplica eneros / pierde octubres
    (bug de la v1: SHILLER_LONGRATE quedo con 153 fechas duplicadas). Aqui se parsea
    numericamente: mes = round((v - floor(v)) * 100).
    """
    v = pd.to_numeric(col, errors="coerce")
    year = v.floordiv(1)
    month = ((v - year) * 100).round()
    ok = v.notna() & month.between(1, 12)
    out = pd.Series(pd.NaT, index=col.index, dtype="datetime64[ns]")
    out[ok] = pd.to_datetime(
        {"year": year[ok].astype(int), "month": month[ok].astype(int), "day": 1}
    )
    return pd.DatetimeIndex(out)


# Orden de preferencia de ie_data.xls. El enlace con UUID es el que publica hoy
# https://shillerdata.com/ (actualizado mensualmente; CAPE hasta el mes en curso). Los dos
# siguientes son espejos historicos que se quedaron congelados (2024-09 y 2023-09).
SHILLER_XLS_URLS = (
    "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/"
    "70fec4f5-727f-4e53-b5f1-179af109c5fa/ie_data.xls",
    "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/ie_data.xls",
    "http://www.econ.yale.edu/~shiller/data/ie_data.xls",
)


def _shiller_xls() -> pd.DataFrame:
    for url in SHILLER_XLS_URLS:
        try:
            raw = _get(url, timeout=60)
            xl = pd.read_excel(io.BytesIO(raw), sheet_name="Data", skiprows=7)
            xl.index = _shiller_dates(xl.iloc[:, 0])
            xl = xl[xl.index.notna()]
            return xl[~xl.index.duplicated(keep="first")]
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("Shiller ie_data.xls no accesible")


def _gw_quarter_index(raw: pd.Series) -> pd.DatetimeIndex:
    """Goyal-Welch trimestral codifica el trimestre como ``AAAAQ`` (p. ej. ``18711``).

    ``PeriodIndex('18711')`` lo lee como el ano 18711 (bug v1: GW_PREDICTORS_QUARTERLY
    en ERROR con 'year 18711 is out of range'). Se separa ano (4 digitos) y trimestre.
    """
    s = raw.astype(str).str.strip()
    ok = s.str.fullmatch(r"\d{4}[1-4]")
    per = pd.PeriodIndex(s[ok].str[:4] + "Q" + s[ok].str[4], freq="Q")
    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    out[ok] = per.to_timestamp()
    return pd.DatetimeIndex(out)


MISSING_FRENCH = (-99.99, -999.0)


def _french_zip(name: str) -> pd.Series | pd.DataFrame:
    """Primer bloque diario de un CSV de Ken French con **TODAS sus columnas** (ADR-003).

    - Fichero de una sola columna (p. ej. ``F-F_Momentum_Factor_daily`` -> ``Mom``):
      devuelve ``pd.Series`` (mismo formato en disco que antes).
    - Fichero multi-columna: devuelve ``pd.DataFrame`` con los nombres originales de la
      cabecera, p. ej.
        - F-F_Research_Data_Factors_daily       -> Mkt-RF, SMB, HML, RF
        - F-F_Research_Data_5_Factors_2x3_daily -> Mkt-RF, SMB, HML, RMW, CMA, RF
        - 5_Industry_Portfolios_daily           -> Cnsmr, Manuf, HiTec, Hlth, Other
          (primer bloque = value-weighted; el bloque equal-weighted se ignora)
      ``download.py`` antepone una columna ALIAS con el ``nombre_interno`` (= primera
      columna de datos) para conservar el contrato de carga
      ``pd.read_parquet(...)[nombre_interno]`` (antes solo se guardaba esa columna).
    Los faltantes de Ken French (-99.99 / -999) pasan a NaN.
    """
    import re
    import zipfile

    url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
    raw = _get(url, timeout=60)
    z = zipfile.ZipFile(io.BytesIO(raw))
    txt = z.read(z.namelist()[0]).decode("latin-1")
    rows: list[list[float]] = []
    idx: list[pd.Timestamp] = []
    header: list[str] | None = None
    last_header: list[str] | None = None
    for ln in txt.splitlines():
        m = re.match(r"^\s*(\d{8})\s*,", ln)
        if m:
            if header is None:
                header = last_header
            parts = [p.strip() for p in ln.split(",")]
            try:
                vals = [float(p) for p in parts[1:]]
            except ValueError:
                continue
            idx.append(pd.Timestamp(m.group(1)))
            rows.append(vals)
        elif rows:
            break  # fin del primer bloque diario
        elif ln.lstrip().startswith(","):
            last_header = [c.strip() for c in ln.split(",")[1:]]
    if not rows:
        raise RuntimeError(f"Ken French {name}: bloque diario vacio")
    ncol = len(rows[0])
    cols = header if header is not None and len(header) == ncol else [f"c{i}" for i in range(ncol)]
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=cols)
    df = df.mask(df.isin(MISSING_FRENCH))
    if ncol == 1:
        return df.iloc[:, 0].rename(name)
    return df


def fetch_academico(spec: str, url: str | None = None):
    s = spec.lower()
    # -- Shiller (ie_data.xls): columna por hint del id --
    if "ie_data" in s or "shiller" in s:
        xl = _shiller_xls()
        if "cape" in s:  # PE10 de Shiller (columna exacta 'CAPE', no 'TR CAPE')
            if "CAPE" not in xl.columns:
                raise RuntimeError(f"Shiller ie_data.xls sin columna CAPE: {list(xl.columns)}")
            col = "CAPE"
        elif "long interest rate" in s or "gs10" in s or "rate" in s:
            col = next((c for c in xl.columns if "GS10" in str(c) or "Long" in str(c)
                        or "Rate" in str(c)), xl.columns[6])
        else:
            col = "P" if "P" in xl.columns else xl.columns[1]
        return pd.to_numeric(xl[col], errors="coerce").dropna().rename(f"SHILLER::{col}")
    # -- Goyal-Welch (Google Sheet xlsx, hoja Monthly/Quarterly): panel completo --
    if "googlesheet" in s or (url and "docs.google.com" in url):
        gurl = "https://docs.google.com/spreadsheets/d/1qwpl2R_DNujpU5YUkk8lacP1tTeMb9iJ/export?format=xlsx"
        sheet = "Quarterly" if "quarter" in s else "Monthly"
        df = pd.read_excel(io.BytesIO(_get(gurl, timeout=60)), sheet_name=sheet)
        dc = df.columns[0]  # yyyymm (Monthly) o yyyyQ (Quarterly)
        raw = df[dc].astype(str).str.replace(r"\.0$", "", regex=True)
        if sheet == "Monthly":
            df.index = pd.to_datetime(raw, format="%Y%m", errors="coerce")
        else:
            df.index = _gw_quarter_index(raw)
        return df.drop(columns=[dc])[df.index.notna()]
    # -- JST macrohistory (xlsx panel; filtra USA) --
    if "jst" in s:
        for jurl in (
            "https://www.macrohistory.net/app/download/9834512569/JSTdatasetR6.xlsx",
            "https://www.macrohistory.net/app/download/9834512569/JSTdatasetR5.xlsx",
        ):
            try:
                df = pd.read_excel(io.BytesIO(_get(jurl, timeout=90)), sheet_name="Data")
                break
            except Exception:  # noqa: BLE001
                df = None
        if df is None:
            raise RuntimeError("JST macrohistory xlsx no accesible")
        ccol = next((c for c in df.columns if str(c).lower() in ("iso", "country")), None)
        if ccol is not None:
            df = df[df[ccol].astype(str).str.upper().isin(["USA", "UNITED STATES"])]
        ycol = next((c for c in df.columns if str(c).lower() == "year"), df.columns[0])
        df.index = pd.to_datetime(df[ycol].astype(int).astype(str) + "-12-31", errors="coerce")
        if "crisis" in s:  # solo la columna crisisJST
            cc = next((c for c in df.columns if "crisis" in str(c).lower()), None)
            if cc is None:
                raise RuntimeError("JST: sin columna crisisJST")
            return pd.to_numeric(df[cc], errors="coerce").dropna().rename("crisisJST_USA")
        return df.select_dtypes("number")
    # -- Philadelphia Fed anxious index (xlsx) --
    if "anxious" in s or "spf-anxious" in s:
        purl = "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/survey-of-professional-forecasters/data-files/files/anxious_index_chart.xlsx"
        df = pd.read_excel(io.BytesIO(_get(purl, timeout=60)), skiprows=3)
        df = df.dropna(subset=[df.columns[0]])
        df.index = pd.PeriodIndex(df.iloc[:, 0].astype(int).astype(str) + "Q"
                                  + df.iloc[:, 1].astype(int).astype(str), freq="Q").to_timestamp()
        col = next((c for c in df.columns if "anxious" in str(c).lower()), df.columns[2])
        return pd.to_numeric(df[col], errors="coerce").dropna().rename("ANXIOUS_INDEX")
    # -- Ken French (zip directo; fallback pandas_datareader) --
    try:
        return _french_zip(spec)
    except Exception:  # noqa: BLE001
        from pandas_datareader import data as pdr

        d = pdr.DataReader(spec, "famafrench", start="1926-01-01")
        df = d[0]
        if hasattr(df.index, "to_timestamp"):
            df.index = df.index.to_timestamp()
        return df.iloc[:, 0].dropna().rename(f"{spec}::{df.columns[0]}")


# --------------------------------------------------------------------------- #
# stooq (bloqueado por challenge JS; se intenta el mirror y si no, se declara)
# --------------------------------------------------------------------------- #
def fetch_stooq(ticker: str) -> pd.Series:
    tk = ticker.lstrip("^").lower()
    for url in (
        f"https://stooq.com/q/d/l/?s={ticker}&i=d",
        f"https://stooq.pl/q/d/l/?s={ticker}&i=d",
    ):
        try:
            raw = _get(url, timeout=30)
            if raw[:15].lstrip().lower().startswith(b"<!doctype") or b"<html" in raw[:200].lower():
                continue  # gate JS
            df = pd.read_csv(io.BytesIO(raw))
            if "Date" in df and "Close" in df:
                df["Date"] = pd.to_datetime(df["Date"])
                return df.set_index("Date")["Close"].dropna().rename(tk)
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("stooq bloqueado por challenge JS (proof-of-work); no descargable")


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def es_ken_french(spec: str) -> bool:
    """True si `fetch_academico(spec)` enruta a la Data Library de Ken French."""
    s = spec.lower()
    return not any(k in s for k in ("ie_data", "shiller", "googlesheet", "jst", "anxious"))


def fetch(fuente: str, series_id: str, url: str | None = None, columna: str | None = None):
    if columna is not None and fuente != "github":
        raise RuntimeError(f"`columna` solo se admite para fuente github (no {fuente})")
    if fuente == "github":
        return fetch_github_csv(series_id, columna=columna)
    if fuente == "fred":
        return fetch_fred_spread(series_id) if "," in series_id else fetch_fred(series_id)
    if fuente == "fred_spread":
        return fetch_fred_spread(series_id)
    if fuente == "yfinance":
        return fetch_yahoo(series_id)
    if fuente == "ofr":
        return fetch_ofr(series_id)
    if fuente == "academico":
        return fetch_academico(series_id, url=url)
    if fuente == "stooq":
        return fetch_stooq(series_id)
    raise RuntimeError(f"fuente desconocida: {fuente}")
