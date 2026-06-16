"""
features.py — Features CAUSALES para los detectores de régimen.

Hallazgo clave de la tarea previa: estandarizar con media/desviación de TODA la
muestra mete información del futuro (look-ahead sutil en el z-score). Aquí TODA
estandarización es causal: en `t` solo se usan estadísticos calculados con datos
`<= t` (expanding o rolling). NINGÚN estadístico de muestra completa.

Convención de causalidad
------------------------
`causal_zscore` usa `expanding`/`rolling` `.mean()` y `.std()`, que por
construcción solo agregan observaciones hasta `t` inclusive. Incluir la propia
observación `t` en su normalización NO es look-ahead (ese dato está disponible en
`t`); lo prohibido es usar datos de `t+1..T`. Quien quiera normalización estricta
one-step-ahead puede pasar `lag=1` (usa estadísticos hasta `t-1`).

La causalidad se VERIFICA explícitamente en `assert_causal` (y en 00_eda.ipynb):
truncar la entrada en T y recomputar debe dar exactamente los mismos valores que
computar sobre la serie completa y recortar en T.

Features construidas (set base, ampliable según estado del arte en FASE 2)
--------------------------------------------------------------------------
  SP500_ret_z        retorno log S&P 500, z causal
  SP500_vol_z        volatilidad realizada 21d (anualizada), z causal
  VIX_level_z        nivel de VIX, z causal  (el nivel, no el retorno: "miedo")
  VIX_change_z       ΔVIX diario, z causal   (velocidad del deterioro)
  MOVE_level_z       nivel de MOVE (vol implícita de tipos), z causal
  TLT_ret_z          retorno log Treasuries largos, z causal
  IEF_ret_z          retorno log Treasuries medios, z causal
  HYG_ret_z          retorno log high yield, z causal
  credit_spread_z    proxy spread crédito = ret(HYG) - ret(IEF), z causal
  yield_slope_z      pendiente 10Y-3M (nivel), z causal
  DXY_change_z       Δlog DXY, z causal
  GOLD_ret_z         retorno log oro, z causal
  corr_spx_bond      correlación rolling 60d S&P500/Treasuries (∈[-1,1], crudo;
                     cambia de signo entre regímenes, Gulko 2002)
  SP500_drawdown     drawdown corriente del S&P 500 (máximo expanding; ∈[-1,0])
  SP500_momentum     momentum 12M-1M del S&P 500 (ret 252d - ret 21d)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Primitivas causales
# --------------------------------------------------------------------------- #
def causal_zscore(
    s: pd.Series,
    method: str = "expanding",
    window: int = 252,
    min_periods: int = 60,
    lag: int = 0,
) -> pd.Series:
    """Z-score CAUSAL: en `t` usa solo media/std de datos `<= t` (o `<= t-lag`).

    Parameters
    ----------
    method : {'expanding', 'rolling'}
        'expanding' usa toda la historia hasta t (estable a largo plazo).
        'rolling' usa una ventana de `window` días (adaptativo a cambios de nivel).
    window : int
        Ventana para method='rolling'.
    min_periods : int
        Mínimo de observaciones antes de emitir un z-score (antes -> NaN, sin
        imputar).
    lag : int
        0 = media/std incluyen t (estándar, causal). 1 = normalización estricta
        one-step-ahead (estadísticos hasta t-1).

    Returns
    -------
    pd.Series alineada con `s`.
    """
    if method == "expanding":
        roll = s.expanding(min_periods=min_periods)
    elif method == "rolling":
        roll = s.rolling(window=window, min_periods=min_periods)
    else:
        raise ValueError(f"method desconocido: {method!r}")
    mu = roll.mean()
    sigma = roll.std(ddof=1)
    if lag > 0:
        mu = mu.shift(lag)
        sigma = sigma.shift(lag)
    z = (s - mu) / sigma.replace(0.0, np.nan)
    return z.rename(f"{s.name}_z" if s.name else "z")


def log_returns(prices: pd.Series) -> pd.Series:
    """Retorno logarítmico diario."""
    return np.log(prices / prices.shift(1))


def realized_vol(returns: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    """Volatilidad realizada rolling causal (anualizada ×√252 si annualize)."""
    vol = returns.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 60) -> pd.Series:
    """Correlación rolling causal entre dos series."""
    return a.rolling(window=window, min_periods=max(20, window // 2)).corr(b)


def drawdown(prices: pd.Series) -> pd.Series:
    """Drawdown corriente respecto al máximo histórico CAUSAL (expanding max)."""
    peak = prices.expanding(min_periods=1).max()
    return prices / peak - 1.0


def momentum(prices: pd.Series, long_w: int = 252, short_w: int = 21) -> pd.Series:
    """Momentum 12M-1M: retorno log a `long_w` días menos retorno a `short_w`."""
    long_ret = np.log(prices / prices.shift(long_w))
    short_ret = np.log(prices / prices.shift(short_w))
    return long_ret - short_ret


# --------------------------------------------------------------------------- #
# Construcción del set completo
# --------------------------------------------------------------------------- #
def build_features(
    raw: pd.DataFrame,
    method: str = "expanding",
    save: bool = True,
    dropna: bool = True,
) -> pd.DataFrame:
    """Construye la matriz de features CAUSALES a partir del panel crudo.

    NO imputa. Si `dropna`, descarta filas iniciales sin todas las features
    (efecto de min_periods y de los distintos inicios de serie) y deja el panel
    listo para los detectores. Guarda data/processed/features.parquet.

    Returns
    -------
    pd.DataFrame de features causales, indexado por fecha.
    """
    f = {}

    # Retornos log
    spx_ret = log_returns(raw["SP500"])
    tlt_ret = log_returns(raw["TLT"])
    ief_ret = log_returns(raw["IEF"])
    hyg_ret = log_returns(raw["HYG"])
    gold_ret = log_returns(raw["GOLD"])
    dxy_chg = log_returns(raw["DXY"])

    f["SP500_ret_z"] = causal_zscore(spx_ret.rename("SP500_ret"), method=method)
    f["SP500_vol_z"] = causal_zscore(
        realized_vol(spx_ret).rename("SP500_vol"), method=method
    )
    f["VIX_level_z"] = causal_zscore(raw["VIX"].rename("VIX_level"), method=method)
    f["VIX_change_z"] = causal_zscore(
        raw["VIX"].diff().rename("VIX_change"), method=method
    )
    f["MOVE_level_z"] = causal_zscore(raw["MOVE"].rename("MOVE_level"), method=method)
    f["TLT_ret_z"] = causal_zscore(tlt_ret.rename("TLT_ret"), method=method)
    f["IEF_ret_z"] = causal_zscore(ief_ret.rename("IEF_ret"), method=method)
    f["HYG_ret_z"] = causal_zscore(hyg_ret.rename("HYG_ret"), method=method)
    f["credit_spread_z"] = causal_zscore(
        (hyg_ret - ief_ret).rename("credit_spread"), method=method
    )
    f["yield_slope_z"] = causal_zscore(
        raw["YIELD_10Y_3M"].rename("yield_slope"), method=method
    )
    f["DXY_change_z"] = causal_zscore(dxy_chg.rename("DXY_change"), method=method)
    f["GOLD_ret_z"] = causal_zscore(gold_ret.rename("GOLD_ret"), method=method)

    # Features ya acotadas / interpretables: se dejan crudas (no z).
    f["corr_spx_bond"] = rolling_correlation(spx_ret, tlt_ret, window=60)
    f["SP500_drawdown"] = drawdown(raw["SP500"])
    f["SP500_momentum"] = momentum(raw["SP500"])

    feats = pd.DataFrame(f)
    feats.index = pd.to_datetime(feats.index)
    feats = feats.sort_index()
    feats.index.name = "date"
    if dropna:
        feats = feats.dropna(how="any")
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(PROCESSED_DIR / "features.parquet")
    return feats


# --------------------------------------------------------------------------- #
# Verificación de causalidad (test de no look-ahead)
# --------------------------------------------------------------------------- #
def assert_causal(
    raw: pd.DataFrame, cut: str = "2015-01-01", tol: float = 1e-9
) -> pd.DataFrame:
    """Verifica que NINGUNA feature usa información futura.

    Computa las features (a) sobre toda la muestra y (b) sobre la muestra
    truncada en `cut`. Si una feature es causal, sus valores hasta `cut` deben
    coincidir en ambas (el futuro no puede alterar el pasado). Devuelve, por
    feature, la discrepancia máxima absoluta; debe ser ~0.

    Returns
    -------
    pd.DataFrame con columnas [feature, max_abs_diff, causal_ok].
    """
    full = build_features(raw, save=False, dropna=False)
    truncated = build_features(raw.loc[:cut], save=False, dropna=False)
    common_idx = truncated.index.intersection(full.index)
    common_idx = common_idx[common_idx <= pd.Timestamp(cut)]
    rows = []
    for col in full.columns:
        a = full.loc[common_idx, col]
        b = truncated.loc[common_idx, col]
        diff = (a - b).abs()
        max_diff = float(np.nanmax(diff.values)) if len(diff) else 0.0
        rows.append(
            {"feature": col, "max_abs_diff": max_diff, "causal_ok": max_diff <= tol}
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover
    from src import data_loader as dl

    raw = dl.load_raw(save=False)
    feats = build_features(raw)
    print("Features:", feats.shape, "| ventana:", feats.index.min().date(), "->", feats.index.max().date())
    print(assert_causal(raw).to_string(index=False))
