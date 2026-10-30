"""Línea base VOL_P90 (A2): percentil expansivo causal de la volatilidad."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from regimenes.evaluacion import ranking as rk

INDEX = pd.bdate_range("2000-01-03", "2010-12-31")
WINDOWS = {"c1": ("2003-03-03", "2003-06-30"), "c2": ("2008-09-01", "2009-03-31")}
SPEC = {"crisis_windows": {"pista_A": {k: list(v) for k, v in WINDOWS.items()}},
        "false_positive_windows": {}}


def _vol() -> pd.Series:
    rng = np.random.default_rng(0)
    vol = pd.Series(np.abs(rng.normal(1, 0.2, len(INDEX))), index=INDEX)
    for a, b in WINDOWS.values():
        vol.loc[a:b] += 2.0
    return vol


class ExpandingQuantileTests(unittest.TestCase):
    def test_uses_only_past_observations(self) -> None:
        vol = _vol()
        thr = rk.expanding_quantile_causal(vol, 0.9, min_obs=252)
        self.assertTrue(thr.iloc[:252].isna().all())
        for t in (252, 1000, len(vol) - 1):
            self.assertAlmostEqual(thr.iloc[t], vol.iloc[:t].quantile(0.9))

    def test_truncation_invariance(self) -> None:
        vol = _vol()
        full = rk.vol_percentile_flags(vol)
        cut = 1500
        short = rk.vol_percentile_flags(vol.iloc[:cut])
        pd.testing.assert_series_equal(full.iloc[:cut], short)
        bumped = vol.copy()
        bumped.iloc[cut:] *= 10  # el futuro no cambia el pasado
        pd.testing.assert_series_equal(rk.vol_percentile_flags(bumped).iloc[:cut], full.iloc[:cut])

    def test_not_a_full_sample_percentile(self) -> None:
        # con una tendencia creciente el percentil de toda la muestra no marcaría
        # el principio; el expansivo marca días en toda la muestra.
        trend = pd.Series(np.linspace(0, 10, len(INDEX)), index=INDEX)
        flags = rk.vol_percentile_flags(trend, min_obs=50)
        self.assertTrue(flags.iloc[60:].all())
        self.assertFalse(flags.iloc[:50].any())


class TrivialBaselinesTests(unittest.TestCase):
    def test_without_vol_nothing_changes(self) -> None:
        kw = dict(rate=0.1, mean_run=10, benchmark=SPEC, n_sims=10, seed=3)
        default = rk.trivial_baselines("A", INDEX, **kw)
        explicit = rk.trivial_baselines("A", INDEX, vol=None, **kw)
        pd.testing.assert_frame_equal(default, explicit)
        self.assertEqual(list(default["id"]),
                         ["SIEMPRE_CRISIS", "SIEMPRE_CALMA", "AZAR_IID", "AZAR_PERSISTENTE"])

    def test_vol_p90_row_is_appended(self) -> None:
        kw = dict(rate=0.1, mean_run=10, benchmark=SPEC, n_sims=10, seed=3)
        default = rk.trivial_baselines("A", INDEX, **kw)
        with_vol = rk.trivial_baselines("A", INDEX, vol=_vol(), **kw)
        self.assertEqual(list(with_vol["id"])[-1], "VOL_P90")
        pd.testing.assert_frame_equal(with_vol.iloc[:4][default.columns], default)
        row = with_vol.set_index("id").loc["VOL_P90"]
        self.assertEqual(row["det_n_detectados"], 2)  # la vol plantada detecta ambas
        self.assertGreater(row["det_lift_precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
