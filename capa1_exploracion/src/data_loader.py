"""
data_loader.py — Descarga de datos crudos (yfinance + FRED), SIN imputar.

Política de datos (decisión de proyecto)
----------------------------------------
- Rango: LO MÁS LARGO POSIBLE POR SERIE. Cada serie arranca en su fecha real de
  inicio; NO se rellenan NaN artificialmente.
- Se documenta la fecha de inicio efectiva de cada serie y la ventana común
  resultante (ver `coverage_report`).
- Crudos se guardan en data/raw/ (gitignored). Las features causales se calculan
  aparte en features.py y se guardan en data/processed/.

Universo de datos — SET AMPLIADO (y procedencia real usada)
-----------------------------------------------------------
Mercado / riesgo (yfinance):
    ^GSPC  S&P 500
    ^VIX   volatilidad implícita de equity (nivel de miedo)
    ^MOVE  volatilidad implícita de tipos  (DISPONIBLE en yfinance desde 2002-11)
    TLT    Treasuries largos
    IEF    Treasuries medios
    HYG    high yield (crédito)
    GLD    oro (refugio)
Dólar (yfinance):
    DX-Y.NYB  índice dólar ICE (DXY clásico). Se elige como PRIMARIA frente a la
              serie FRED DTWEXBGS porque FRED resultó inaccesible en el entorno
              de desarrollo (timeouts sistemáticos) y DX-Y.NYB es la alternativa
              explícitamente contemplada y más estándar.
Curva de tipos (FRED con fallback yfinance):
    T10Y3M (FRED)  pendiente 10Y-3M, predictor de recesión (Estrella-Mishkin).
                   Si FRED no responde, se construye el PROXY real:
                   YIELD_10Y_3M = ^TNX (yield 10Y) - ^IRX (yield 3M T-bill),
                   ambos de yfinance. Se registra qué fuente se usó en provenance.
Crédito (FRED, opcional/ampliable):
    BAMLH0A0HYM2 (FRED)  OAS high yield. Si FRED no responde, se OMITE y se
                         declara: el riesgo de crédito queda cubierto por HYG y
                         por el spread HYG-IEF en features.py.

Nota: FRED se descarga vía el endpoint CSV público (sin API key). En entornos sin
acceso a fred.stlouisfed.org, el loader aplica los fallbacks anteriores y deja
constancia en data/raw/provenance.json. Nada se inventa ni se omite en silencio.
"""

from __future__ import annotations

import io
import json
import time
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore", category=FutureWarning)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# --- Series de yfinance: nombre interno -> ticker -------------------------- #
YF_TICKERS: dict[str, str] = {
    "SP500": "^GSPC",
    "VIX": "^VIX",
    "MOVE": "^MOVE",
    "TLT": "TLT",
    "IEF": "IEF",
    "HYG": "HYG",
    "GOLD": "GLD",
    "DXY": "DX-Y.NYB",   # DXY clásico (ICE), primaria sobre FRED DTWEXBGS
    "TNX": "^TNX",       # yield 10Y (para construir la curva si FRED falla)
    "IRX": "^IRX",       # yield 3M T-bill (idem)
}

# --- Series de FRED: nombre interno -> id FRED ----------------------------- #
# Solo se intenta la curva de tipos. HY_OAS (BAMLH0A0HYM2) se EXCLUYE a propósito:
# FRED resultó inaccesible en este entorno y su única respuesta llegó truncada
# (solo 2023+ en lugar de su histórico desde 1996), lo que corrompería la ventana
# común. El riesgo de crédito queda cubierto por HYG y el spread HYG-IEF.
FRED_SERIES: dict[str, str] = {
    "YIELD_10Y_3M": "T10Y3M",      # core (con fallback yfinance ^TNX - ^IRX)
}

# Mínimo de filas para aceptar una respuesta de FRED como íntegra (no truncada).
FRED_MIN_ROWS = 500

DEFAULT_START = "1985-01-01"


# --------------------------------------------------------------------------- #
# yfinance
# --------------------------------------------------------------------------- #
def download_yfinance(
    tickers: dict[str, str] | None = None,
    start: str = DEFAULT_START,
    save: bool = True,
) -> pd.DataFrame:
    """Descarga niveles/precios diarios de yfinance, SIN imputar.

    Descarga ticker a ticker (robusto a fallos individuales) y toma la columna
    'Close' con auto_adjust=True (precio ajustado para ETFs; nivel para índices
    y yields). Cada columna empieza en su fecha real; sin relleno.

    Returns
    -------
    pd.DataFrame indexado por fecha (DatetimeIndex), una columna por nombre
    interno. Unión de fechas; los huecos previos al inicio de cada serie quedan
    como NaN.
    """
    import yfinance as yf

    tickers = tickers or YF_TICKERS
    cols: dict[str, pd.Series] = {}
    for name, tk in tickers.items():
        df = yf.download(
            tk, start=start, progress=False, auto_adjust=True, multi_level_index=False
        )
        if df is None or df.empty:
            warnings.warn(f"yfinance vacío para {name} ({tk})")
            continue
        s = df["Close"].copy()
        s.name = name
        cols[name] = s
    panel = pd.concat(cols.values(), axis=1)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    panel.index.name = "date"
    if save:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(RAW_DIR / "yfinance_raw.parquet")
    return panel


# --------------------------------------------------------------------------- #
# FRED (CSV público, sin API key) con reintentos cortos
# --------------------------------------------------------------------------- #
def _fred_csv(series_id: str, start: str, timeout: int = 20, retries: int = 2) -> pd.Series | None:
    """Descarga una serie FRED vía fredgraph.csv. Devuelve Series o None si falla.

    Reintentos cortos con backoff para fallar rápido en entornos sin acceso a
    FRED en lugar de colgar la ejecución.
    """
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        f"&cosd={start}"
    )
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    for k in range(retries):
        try:
            r = sess.get(url, timeout=timeout)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")  # "." -> NaN
            s = df.set_index("date")["value"].dropna()
            # Guardia anti-truncamiento: una respuesta throttled puede llegar
            # parcial y parsear sin error. Rechazar si es sospechosamente corta.
            if len(s) < FRED_MIN_ROWS:
                warnings.warn(
                    f"FRED {series_id}: respuesta truncada ({len(s)} filas < {FRED_MIN_ROWS}); descartada"
                )
                return None
            s.name = series_id
            return s
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"FRED {series_id} intento {k}: {type(e).__name__}")
            time.sleep(2 * (k + 1))
    return None


def download_fred(
    series: dict[str, str] | None = None,
    start: str = DEFAULT_START,
    save: bool = True,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Descarga series de FRED, SIN imputar. Devuelve (panel, disponibilidad).

    `disponibilidad[nombre] = True/False` indica si FRED respondió, para que
    `load_raw` aplique fallbacks y registre provenance.
    """
    series = series or FRED_SERIES
    cols: dict[str, pd.Series] = {}
    available: dict[str, bool] = {}
    for name, sid in series.items():
        s = _fred_csv(sid, start)
        available[name] = s is not None
        if s is not None:
            s.name = name
            cols[name] = s
    panel = pd.concat(cols.values(), axis=1) if cols else pd.DataFrame()
    if not panel.empty:
        panel.index = pd.to_datetime(panel.index)
        panel = panel.sort_index()
        panel.index.name = "date"
        if save:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            panel.to_parquet(RAW_DIR / "fred_raw.parquet")
    return panel, available


# --------------------------------------------------------------------------- #
# Orquestación: panel crudo completo + provenance
# --------------------------------------------------------------------------- #
def load_raw(start: str = DEFAULT_START, save: bool = True) -> pd.DataFrame:
    """Descarga TODO el set ampliado (yfinance + FRED) y lo une por fecha.

    No imputa ni alinea a ventana común (eso es decisión explícita y se reporta
    en `coverage_report`). Aplica fallbacks documentados:
      - DXY: ya viene de yfinance (DX-Y.NYB).
      - Curva 10Y-3M: FRED T10Y3M; si no, ^TNX - ^IRX (proxy real).
      - HY OAS: FRED; si no, se omite (declarado).
    Guarda data/raw/raw_panel.parquet y data/raw/provenance.json.

    Returns
    -------
    pd.DataFrame indexado por fecha con las columnas finales del set.
    """
    provenance: dict[str, str] = {}

    yf_panel = download_yfinance(start=start, save=save)
    for c in yf_panel.columns:
        provenance[c] = f"yfinance:{YF_TICKERS[c]}"

    fred_panel, fred_ok = download_fred(start=start, save=save)

    # --- Curva 10Y-3M: FRED o proxy yfinance ------------------------------- #
    if fred_ok.get("YIELD_10Y_3M") and "YIELD_10Y_3M" in fred_panel:
        yield_slope = fred_panel["YIELD_10Y_3M"].copy()
        provenance["YIELD_10Y_3M"] = "FRED:T10Y3M"
    else:
        # Proxy real: yield 10Y (^TNX) - yield 3M T-bill (^IRX), ambos en %.
        yield_slope = (yf_panel["TNX"] - yf_panel["IRX"]).rename("YIELD_10Y_3M")
        provenance["YIELD_10Y_3M"] = "proxy:yfinance(^TNX - ^IRX) [FRED inaccesible]"

    # --- HY OAS: opcional --------------------------------------------------- #
    parts = [yf_panel.drop(columns=["TNX", "IRX"]), yield_slope]
    if fred_ok.get("HY_OAS") and "HY_OAS" in fred_panel:
        parts.append(fred_panel["HY_OAS"])
        provenance["HY_OAS"] = "FRED:BAMLH0A0HYM2"
    else:
        provenance["HY_OAS"] = "OMITIDO [FRED inaccesible]; crédito cubierto por HYG y spread HYG-IEF"

    panel = pd.concat(parts, axis=1).sort_index()
    panel.index = pd.to_datetime(panel.index)
    panel.index.name = "date"

    if save:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(RAW_DIR / "raw_panel.parquet")
        (RAW_DIR / "provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return panel


# --------------------------------------------------------------------------- #
# Reporte de cobertura (output, no comentario)
# --------------------------------------------------------------------------- #
def coverage_report(raw: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Tabla de cobertura por serie + ventana común. Es un OUTPUT versionable.

    Para cada serie: fecha de inicio efectiva (primer dato no NaN), fecha de fin,
    nº de observaciones válidas y % de NaN dentro de su propio rango. Añade dos
    filas resumen: la ventana común (intersección = max de inicios -> min de
    fines) y su nº de filas sin NaN en ninguna serie.

    Guarda data/raw/coverage_report.csv.
    """
    rows = []
    starts, ends = [], []
    for col in raw.columns:
        s = raw[col].dropna()
        if s.empty:
            rows.append({"serie": col, "inicio": None, "fin": None, "n_obs": 0, "pct_nan_en_rango": None})
            continue
        start, end = s.index.min(), s.index.max()
        starts.append(start)
        ends.append(end)
        full = raw[col].loc[start:end]
        pct_nan = 100.0 * full.isna().mean()
        rows.append(
            {
                "serie": col,
                "inicio": start.date().isoformat(),
                "fin": end.date().isoformat(),
                "n_obs": int(s.shape[0]),
                "pct_nan_en_rango": round(float(pct_nan), 3),
            }
        )
    report = pd.DataFrame(rows).sort_values("inicio", na_position="last").reset_index(drop=True)

    # Ventana común (intersección)
    common_start = max(starts) if starts else None
    common_end = min(ends) if ends else None
    n_common = 0
    if common_start is not None and common_end is not None:
        sub = raw.loc[common_start:common_end].dropna(how="any")
        n_common = int(sub.shape[0])
    meta = pd.DataFrame(
        [
            {
                "serie": "VENTANA_COMUN",
                "inicio": common_start.date().isoformat() if common_start is not None else None,
                "fin": common_end.date().isoformat() if common_end is not None else None,
                "n_obs": n_common,
                "pct_nan_en_rango": 0.0,
            }
        ]
    )
    report = pd.concat([report, meta], ignore_index=True)

    if save:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        report.to_csv(RAW_DIR / "coverage_report.csv", index=False)
    return report


if __name__ == "__main__":  # pragma: no cover
    raw = load_raw()
    rep = coverage_report(raw)
    print(rep.to_string(index=False))
