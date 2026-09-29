"""Pruebas de causalidad e interpretación para la fusión de detectores."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.fusion import (
    FusionConfig,
    assert_prefix_causal,
    confirmation_episode_table,
    ablation_scorecard,
    fuse_early_warning,
    warning_ground_truth_table,
    warning_episode_table,
    warning_summary,
)


def _panel(states: list[int], probabilities: list[float]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(states), freq="B")
    return pd.DataFrame(
        {"state": states, "p_crisis": probabilities},
        index=index,
    )


class FusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.alert = _panel(
            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0.1, 0.2, 0.7, 0.8, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1],
        )
        self.confirmation = _panel(
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0.1, 0.1, 0.2, 0.3, 0.8, 0.9, 0.9, 0.8, 0.2, 0.1],
        )
        self.config = FusionConfig(
            alert_crisis_state=1,
            confirmation_persistence=2,
            warning_horizon=6,
        )

    def test_three_state_decision_and_lead_are_interpretable(self) -> None:
        fused = fuse_early_warning(self.alert, self.confirmation, self.config)

        self.assertEqual(fused["decision_code"].tolist(), [0, 0, 1, 1, 1, 2, 2, 2, 1, 0])
        warnings = warning_episode_table(fused, self.config.warning_horizon)
        confirmations = confirmation_episode_table(fused, self.config.warning_horizon)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings.loc[0, "lead_sessions"], 3)
        self.assertTrue(confirmations.loc[0, "preceded_by_alert"])
        self.assertEqual(confirmations.loc[0, "lead_sessions"], 3)

    def test_warning_expires_without_using_future_information(self) -> None:
        no_confirmation = self.confirmation.assign(state=0, p_crisis=0.0)
        config = FusionConfig(alert_crisis_state=1, confirmation_persistence=2, warning_horizon=2)
        fused = fuse_early_warning(self.alert, no_confirmation, config)

        self.assertEqual(fused["decision_code"].tolist(), [0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
        summary = warning_summary(fused, config.warning_horizon)
        self.assertEqual(summary["false_or_unconfirmed_warnings"], 1.0)
        self.assertTrue(np.isnan(summary["confirmation_recall"]))

    def test_prefix_invariance_proves_causal_construction(self) -> None:
        self.assertTrue(
            assert_prefix_causal(
                self.alert,
                self.confirmation,
                self.config,
                fractions=(0.4, 0.7, 0.9),
            )
        )

    def test_alert_during_confirmation_is_not_credited_as_early_warning(self) -> None:
        alert = _panel(
            [0, 0, 1, 1, 0, 0],
            [0.1, 0.1, 0.8, 0.8, 0.1, 0.1],
        )
        confirmation = _panel(
            [0, 1, 1, 0, 1, 1],
            [0.1, 0.9, 0.9, 0.1, 0.9, 0.9],
        )
        config = FusionConfig(alert_crisis_state=1, confirmation_persistence=1, warning_horizon=4)
        fused = fuse_early_warning(alert, confirmation, config)
        episodes = confirmation_episode_table(fused, config.warning_horizon)

        self.assertEqual(len(episodes), 2)
        self.assertFalse(episodes["preceded_by_alert"].any())
        self.assertFalse(fused["warning_onset"].any())
        self.assertFalse(fused["warning_window"].any())

    def test_one_warning_cannot_explain_two_confirmations(self) -> None:
        alert = _panel([0, 1, 1, 1, 1, 1, 1], [0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        confirmation = _panel([0, 0, 1, 0, 0, 1, 1], [0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.9])
        config = FusionConfig(alert_crisis_state=1, confirmation_persistence=1, warning_horizon=6)
        fused = fuse_early_warning(alert, confirmation, config)

        confirmations = confirmation_episode_table(fused, config.warning_horizon)
        self.assertEqual(confirmations["preceded_by_alert"].sum(), 1)

    def test_ground_truth_distinguishes_anticipation_and_reaction(self) -> None:
        config = FusionConfig(alert_crisis_state=1, confirmation_persistence=1, warning_horizon=3)
        confirmation = _panel([0] * 10, [0.0] * 10)
        windows = {"evento": ("2020-01-08", "2020-01-10")}

        before = fuse_early_warning(
            _panel([0, 0, 1, 1, 0, 0, 0, 0, 0, 0], [0.0] * 10),
            confirmation,
            config,
        )
        during = fuse_early_warning(
            _panel([0, 0, 0, 0, 0, 1, 1, 0, 0, 0], [0.0] * 10),
            confirmation,
            config,
        )

        self.assertEqual(
            warning_ground_truth_table(before, 3, windows).loc[0, "phase"],
            "anticipacion",
        )
        self.assertEqual(
            warning_ground_truth_table(during, 3, windows).loc[0, "phase"],
            "durante_crisis",
        )

    def test_ablation_contains_parallel_and_sequential_architectures(self) -> None:
        fused = fuse_early_warning(self.alert, self.confirmation, self.config)
        table = ablation_scorecard(fused, self.config.warning_horizon)
        self.assertEqual(
            set(table["arquitectura"]),
            {"D7_solo", "D8_solo", "fusion_paralela", "confirmacion_D8_tras_D7", "fusion_secuencial"},
        )


if __name__ == "__main__":
    unittest.main()
