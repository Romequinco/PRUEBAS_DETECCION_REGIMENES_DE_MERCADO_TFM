"""ADR-003: propagación causal de estado entre bloques del walk-forward.

`walk_forward` predice cada bloque sobre ``[cola del train] + bloque`` y se queda
solo con el bloque. Estos tests fijan tres propiedades:

1. Causalidad: truncar el futuro no cambia ninguna predicción pasada.
2. Un autómata de histéresis ya NO se reinicia en calma en el borde de bloque
   (con ``context=0``, el protocolo anterior, sí lo hacía).
3. Un detector sin estado da exactamente el mismo panel que antes, y los que ya
   anteponían el train como burn-in (D07) no duplican historia.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from regimenes import evaluacion as ev
from regimenes.detectores.base import RegimeDetector


class _HysteresisAutomaton(RegimeDetector):
    """Autómata 0/1 con umbrales fijos: entra si x > 2, sale si x < 0."""

    TAU_IN = 2.0
    TAU_OUT = 0.0

    def __init__(self) -> None:
        super().__init__(n_states=2)

    @property
    def name(self) -> str:
        return "toy_hysteresis"

    @property
    def bibliography(self) -> list[str]:
        return []

    def fit(self, X_train: pd.DataFrame) -> "_HysteresisAutomaton":
        self._canonical_order = np.array([0, 1])
        self._is_fitted = True
        return self

    @classmethod
    def run(cls, x: np.ndarray) -> np.ndarray:
        out = np.zeros(len(x), dtype=int)
        state = 0  # arranca en calma, como D01/D02/D06/D10
        for t, v in enumerate(x):
            if state == 0 and v > cls.TAU_IN:
                state = 1
            elif state == 1 and v < cls.TAU_OUT:
                state = 0
            out[t] = state
        return out

    def _predict_states(self, X: pd.DataFrame) -> np.ndarray:
        return self.run(X["x"].values)


class _Stateless(RegimeDetector):
    """Umbral por fila (mediana del train): sin memoria entre días."""

    def __init__(self) -> None:
        super().__init__(n_states=2)

    @property
    def name(self) -> str:
        return "toy_stateless"

    @property
    def bibliography(self) -> list[str]:
        return []

    def fit(self, X_train: pd.DataFrame) -> "_Stateless":
        self._thr = float(X_train["x"].median())
        self._canonical_order = np.array([0, 1])
        self._is_fitted = True
        return self

    def _predict_states(self, X: pd.DataFrame) -> np.ndarray:
        return (X["x"].values > self._thr).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-(X["x"].values - self._thr)))
        return np.column_stack([1.0 - p, p])


def _plateau_frame(n: int = 400) -> pd.DataFrame:
    """Calma, un pico (entra en crisis) y una meseta en la banda muerta (0, 2)
    que atraviesa varios bordes de bloque: el estado correcto es crisis."""
    rng = np.random.default_rng(1)
    x = rng.uniform(-1.0, -0.5, size=n)
    x[130] = 3.0          # entrada a crisis
    x[131:251] = 1.0      # banda muerta: se mantiene crisis 120 días (6 bloques)
    index = pd.bdate_range("2001-01-01", periods=n)
    return pd.DataFrame({"x": x}, index=index)


def _toy_frame(n: int = 320) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2000-01-03", periods=n)
    x = pd.Series(rng.normal(size=n), index=index)
    returns = pd.Series(-0.01 * x.values + rng.normal(scale=0.001, size=n), index=index)
    return pd.DataFrame({"x": x}), returns


WF = dict(train_size=100, min_train=100, step=20)


class ResolveContextTests(unittest.TestCase):
    def test_auto_is_one_year_in_the_index_frequency(self) -> None:
        daily = pd.bdate_range("2000-01-03", periods=600)
        monthly = pd.date_range("1990-01-01", periods=300, freq="MS")
        weekly = pd.date_range("1990-01-05", periods=300, freq="W-FRI")
        self.assertEqual(ev.resolve_context("auto", daily), 252)
        self.assertEqual(ev.resolve_context("auto", monthly), 12)
        self.assertEqual(ev.resolve_context("auto", weekly), 52)

    def test_explicit_values(self) -> None:
        idx = pd.bdate_range("2000-01-03", periods=10)
        self.assertIsNone(ev.resolve_context(None, idx))
        self.assertEqual(ev.resolve_context(0, idx), 0)
        self.assertEqual(ev.resolve_context(63, idx), 63)
        with self.assertRaises(ValueError):
            ev.resolve_context(-1, idx)
        with self.assertRaises(ValueError):
            ev.resolve_context("year", idx)


class StatePropagationTests(unittest.TestCase):
    def test_hysteresis_no_longer_resets_at_block_borders(self) -> None:
        X = _plateau_frame()
        truth = pd.Series(_HysteresisAutomaton.run(X["x"].values), index=X.index)
        new = ev.walk_forward(_HysteresisAutomaton, X, **WF)
        old = ev.walk_forward(_HysteresisAutomaton, X, context=0, **WF)
        # Con contexto, el walk-forward reproduce la ejecución continua.
        pd.testing.assert_series_equal(
            new["state"], truth.loc[new.index], check_names=False
        )
        # El protocolo anterior perdía la crisis en el primer borde de bloque.
        plateau = X.index[140:251]
        self.assertTrue((new.loc[plateau, "state"] == 1).all())
        self.assertLess(old.loc[plateau, "state"].mean(), 0.2)
        first_border = new.index[(new["fold"].diff() != 0)].intersection(plateau)[0]
        self.assertEqual(old.loc[first_border, "state"], 0)
        self.assertEqual(new.loc[first_border, "state"], 1)

    def test_truncating_the_future_does_not_change_past_predictions(self) -> None:
        X = _plateau_frame()
        full = ev.walk_forward(_HysteresisAutomaton, X, **WF)
        for cut_pos in (150, 207, 260, 333):
            cut = X.index[cut_pos]
            trunc = ev.walk_forward(_HysteresisAutomaton, X.loc[:cut], **WF)
            pd.testing.assert_frame_equal(full.loc[trunc.index], trunc)

    def test_truncation_invariance_with_market_returns_and_rolling(self) -> None:
        X, returns = _toy_frame()
        for expanding in (True, False):
            full = ev.walk_forward(
                _HysteresisAutomaton, X, market_returns=returns,
                expanding=expanding, **WF,
            )
            cut = X.index[230]
            trunc = ev.walk_forward(
                _HysteresisAutomaton, X.loc[:cut], market_returns=returns.loc[:cut],
                expanding=expanding, **WF,
            )
            pd.testing.assert_frame_equal(full.loc[trunc.index], trunc)

    def test_stateless_detector_is_identical_to_previous_protocol(self) -> None:
        X, returns = _toy_frame()
        for context in ("auto", None, 5):
            new = ev.walk_forward(_Stateless, X, market_returns=returns, context=context, **WF)
            old = ev.walk_forward(_Stateless, X, market_returns=returns, context=0, **WF)
            pd.testing.assert_frame_equal(new, old)
            pd.testing.assert_frame_equal(
                new.attrs["stability_panel"], old.attrs["stability_panel"]
            )

    def test_changepoint_d07_does_not_double_count_its_burn_in(self) -> None:
        """D07 ya antepone todo el train como burn-in; con contexto el resultado
        debe ser idéntico (recorta su burn-in a las fechas previas a la entrada)."""
        from regimenes.detectores import detector_specs

        spec = next(s for s in detector_specs("A") if s.detector_id == "D07")
        rng = np.random.default_rng(7)
        n = 700
        vol = np.where((np.arange(n) > 380) & (np.arange(n) < 520), 0.03, 0.008)
        r = rng.standard_t(df=5, size=n) * vol
        index = pd.bdate_range("1995-01-02", periods=n)
        X = pd.DataFrame({"SP500_ret": r}, index=index)
        mr = X["SP500_ret"]
        kw = dict(market_returns=mr, train_size=300, min_train=300, step=21)
        new = ev.walk_forward(spec.factory, X, **kw)
        old = ev.walk_forward(spec.factory, X, context=0, **kw)
        pd.testing.assert_frame_equal(new, old)


if __name__ == "__main__":
    unittest.main()
