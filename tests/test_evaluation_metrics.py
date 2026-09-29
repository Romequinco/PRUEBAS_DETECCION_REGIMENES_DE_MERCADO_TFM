"""Regresiones del juez: causalidad del walk-forward y definición de métricas."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src import evaluation as ev
from src.detector_base import RegimeDetector


class _RecordingDetector(RegimeDetector):
    """Detector trivial que registra qué fechas ve en fit y en predict."""

    log: list[dict] = []

    def __init__(self) -> None:
        super().__init__(n_states=2)

    @property
    def name(self) -> str:
        return "recording"

    @property
    def bibliography(self) -> list[str]:
        return []

    def fit(self, X_train: pd.DataFrame) -> "_RecordingDetector":
        self._threshold = float(X_train["x"].median())
        self._train_end = X_train.index.max()
        self._train_start = X_train.index.min()
        self._is_fitted = True
        return self

    def _predict_states(self, X: pd.DataFrame) -> np.ndarray:
        return (X["x"].values > self._threshold).astype(int)

    def predict_online(self, X: pd.DataFrame, refit: bool = False) -> np.ndarray:
        _RecordingDetector.log.append({
            "train_start": self._train_start,
            "train_end": self._train_end,
            "test_start": X.index.min(),
            "test_end": X.index.max(),
        })
        return super().predict_online(X)


def _toy_frame(n: int = 300) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2000-01-03", periods=n)
    x = pd.Series(rng.normal(size=n), index=index)
    returns = pd.Series(-0.01 * x.values + rng.normal(scale=0.001, size=n), index=index)
    return pd.DataFrame({"x": x}), returns


class WalkForwardCausalityTests(unittest.TestCase):
    def setUp(self) -> None:
        _RecordingDetector.log = []

    def test_every_fold_trains_strictly_before_its_block(self) -> None:
        X, returns = _toy_frame()
        panel = ev.walk_forward(
            _RecordingDetector, X, market_returns=returns,
            train_size=100, min_train=100, step=20,
        )
        # Solo las llamadas del bloque propio (las re-predicciones del bloque
        # previo, para label_stability, son diagnóstico y no causales).
        own = [row for row in _RecordingDetector.log if row["test_start"] > row["train_end"]]
        self.assertGreater(len(own), 5)
        for row in own:
            self.assertLess(row["train_end"], row["test_start"])
        # Cada fecha OOS aparece exactamente una vez y empieza tras el train inicial.
        self.assertTrue(panel.index.is_unique)
        self.assertEqual(panel.index[0], X.index[100])
        self.assertEqual(len(panel), len(X) - 100)

    def test_rolling_window_keeps_fixed_train_length(self) -> None:
        X, returns = _toy_frame()
        ev.walk_forward(
            _RecordingDetector, X, market_returns=returns,
            train_size=100, min_train=100, step=20, expanding=False,
        )
        own = [row for row in _RecordingDetector.log if row["test_start"] > row["train_end"]]
        for row in own:
            n_train = len(X.loc[row["train_start"]:row["train_end"]])
            self.assertEqual(n_train, 100)

    def test_oos_labels_do_not_depend_on_future_data(self) -> None:
        X, returns = _toy_frame()
        full = ev.walk_forward(
            _RecordingDetector, X, market_returns=returns,
            train_size=100, min_train=100, step=20,
        )
        cut = X.index[200]
        truncated = ev.walk_forward(
            _RecordingDetector, X.loc[:cut], market_returns=returns.loc[:cut],
            train_size=100, min_train=100, step=20,
        )
        common = truncated.index
        pd.testing.assert_series_equal(full.loc[common, "state"], truncated["state"])

    def test_stability_panel_lives_only_in_attrs(self) -> None:
        X, returns = _toy_frame()
        panel = ev.walk_forward(
            _RecordingDetector, X, market_returns=returns,
            train_size=100, min_train=100, step=20,
        )
        self.assertEqual(list(panel.columns), ["state", "p_crisis", "fold"])
        self.assertIn("stability_panel", panel.attrs)


class MetricDefinitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.bdate_range("2020-01-01", periods=10)

    def test_coverage_is_nan_outside_oos_and_fraction_inside(self) -> None:
        states = pd.Series([0, 1, 1, 0, 0, 0, 0, 0, 0, 0], index=self.index)
        windows = {
            "dentro": (str(self.index[0].date()), str(self.index[3].date())),
            "fuera": ("1990-01-01", "1990-02-01"),
        }
        cov = ev.crisis_coverage(states, 1, windows)
        self.assertAlmostEqual(cov["dentro"], 0.5)
        self.assertTrue(np.isnan(cov["fuera"]))

    def test_false_alarm_rate_is_share_of_flags_outside_windows(self) -> None:
        # 4 días marcados: 1 dentro de la ventana, 3 fuera -> 0.75 (1 - precisión).
        states = pd.Series([1, 0, 0, 0, 0, 1, 1, 1, 0, 0], index=self.index)
        windows = {"w": (str(self.index[0].date()), str(self.index[1].date()))}
        self.assertAlmostEqual(ev.false_alarm_rate(states, 1, windows), 0.75)

    def test_false_alarm_rate_is_nan_when_never_flagging(self) -> None:
        states = pd.Series(0, index=self.index)
        windows = {"w": (str(self.index[0].date()), str(self.index[1].date()))}
        self.assertTrue(np.isnan(ev.false_alarm_rate(states, 1, windows)))

    def test_switching_and_duration(self) -> None:
        states = pd.Series([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], index=self.index)
        self.assertAlmostEqual(ev.switching_rate(states), 2 / 10)
        self.assertAlmostEqual(ev.mean_regime_duration(states), 10 / 3)

    def test_label_stability_is_bounded_by_one_half_with_two_estimates(self) -> None:
        # Con dos estimaciones por fecha (la propia y la del fold siguiente),
        # el acuerdo por fecha solo puede ser 1 o 0.5: la métrica vive en [0.5, 1].
        panel = pd.DataFrame(
            {0: [0, 1, 1, 0], 1: [1, 0, 0, 1]},
            index=pd.bdate_range("2020-01-01", periods=4),
        )
        self.assertAlmostEqual(ev.label_stability(panel), 0.5)
        panel[1] = panel[0]
        self.assertAlmostEqual(ev.label_stability(panel), 1.0)

    def test_lead_lag_negative_when_signal_precedes_trough(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=30)
        p = pd.Series(0.0, index=index)
        p.iloc[10:25] = 1.0
        trough = str(index[20].date())
        out = ev.lead_lag(p, {"evento": trough}, persist=3)
        self.assertEqual(out["evento"], -10.0)


if __name__ == "__main__":
    unittest.main()
