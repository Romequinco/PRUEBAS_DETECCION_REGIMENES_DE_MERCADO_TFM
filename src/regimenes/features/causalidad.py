"""causalidad.py — Verificación de causalidad (test de truncado).

Sección 5 del antiguo ``src/features.py`` (ADR-004). Ver el docstring de
``regimenes.features.transformaciones`` ("Alcance del test de truncado").
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from regimenes.features.transformaciones import RawDict


# --------------------------------------------------------------------------- #
# 5. Verificación de causalidad (test de truncado)
# --------------------------------------------------------------------------- #
def truncar(raw: RawDict, cut: str | pd.Timestamp) -> dict[str, pd.Series]:
    """Copia de `raw` con cada serie cortada en `cut` (inclusive, fecha de FUENTE)."""
    return {k: v.loc[:cut] for k, v in raw.items()}


def truncation_max_abs_diff(full: pd.Series | pd.DataFrame,
                            trunc: pd.Series | pd.DataFrame,
                            cut: str | pd.Timestamp) -> pd.Series:
    """max|full - trunc| por columna en las fechas comunes <= cut.

    Además de diferencias numéricas, cuenta como discrepancia (valor inf) que una
    celda sea NaN en una versión y no en la otra (un NaN 'que aparece' también es
    información del futuro alterando el pasado).
    """
    full_df = full.to_frame() if isinstance(full, pd.Series) else full
    trunc_df = trunc.to_frame() if isinstance(trunc, pd.Series) else trunc
    cut = pd.Timestamp(cut)
    idx = full_df.index.intersection(trunc_df.index)
    idx = idx[idx <= cut]
    a = full_df.loc[idx, trunc_df.columns]
    b = trunc_df.loc[idx]
    diff = (a - b).abs().max()
    nan_mismatch = (a.isna() != b.isna()).any()
    return diff.fillna(0.0).where(~nan_mismatch, np.inf)


def assert_causal(builder: Callable[[RawDict], pd.Series | pd.DataFrame | None],
                  raw: RawDict, cut: str = "2015-01-01", tol: float = 1e-9) -> pd.DataFrame:
    """Verifica que `builder(raw)` no usa información futura (test de truncado).

    Computa `builder` sobre la muestra completa y sobre la muestra truncada en `cut`;
    los valores <= cut deben coincidir. Devuelve [feature, max_abs_diff, causal_ok].
    """
    full = builder(raw)
    trunc = builder(truncar(raw, cut))
    if full is None or trunc is None:
        return pd.DataFrame(columns=['feature', 'max_abs_diff', 'causal_ok'])
    mad = truncation_max_abs_diff(full, trunc, cut)
    return pd.DataFrame({'feature': mad.index.astype(str), 'max_abs_diff': mad.values,
                         'causal_ok': (mad.values <= tol)})
