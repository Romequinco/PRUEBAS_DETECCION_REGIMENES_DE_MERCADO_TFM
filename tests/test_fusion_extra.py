"""Pruebas adicionales de la fusión: causalidad del estado de crisis, utilidad,
veredicto, línea base de fases y selección en un tramo / evaluación en otro."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src import evaluation as ev
from src.fusion import (
    FusionConfig,
    ablation_scorecard,
    assert_prefix_causal,
    crisis_state_from_metrics,
    event_scorecard,
    fuse_early_warning,
    fusion_verdict,
    load_panel,
    operational_utility,
    per_crisis_scorecard,
    period_scorecard,
    select_then_evaluate,
    warning_ground_truth_table,
    warning_phase_base_rates,
)


def _panel(states, probabilities, start="2020-01-01"):
    index = pd.date_range(start, periods=len(states), freq="B")
    return pd.DataFrame({"state": states, "p_crisis": probabilities}, index=index)


class _Windows:
    """Sustituye temporalmente las labels globales del juez."""

    def __init__(self, crisis, false_positive):
        self.crisis, self.false_positive = crisis, false_positive

    def __enter__(self):
        self.saved = (dict(ev.CRISIS_WINDOWS), dict(ev.FALSE_POSITIVE_WINDOWS))
        ev.CRISIS_WINDOWS.clear(); ev.CRISIS_WINDOWS.update(self.crisis)
        ev.FALSE_POSITIVE_WINDOWS.clear(); ev.FALSE_POSITIVE_WINDOWS.update(self.false_positive)
        return self

    def __exit__(self, *exc):
        ev.CRISIS_WINDOWS.clear(); ev.CRISIS_WINDOWS.update(self.saved[0])
        ev.FALSE_POSITIVE_WINDOWS.clear(); ev.FALSE_POSITIVE_WINDOWS.update(self.saved[1])
        return False


class CrisisStateCausalityTests(unittest.TestCase):
    def test_state_source_requires_explicit_crisis_state(self):
        alert = _panel([0, 1, 0], [0.1, 0.9, 0.1])
        with self.assertRaises(ValueError):
            fuse_early_warning(alert, alert, FusionConfig())

    def test_crisis_state_is_not_inferred_from_future_maximum(self):
        # Detector de 3 estados que solo alcanza el estado 2 (crisis) al final.
        # Con la inferencia antigua (state.max()) los días en estado 1 contaban
        # como "crisis" en el prefijo truncado y dejaban de hacerlo con el panel
        # completo: una dependencia del futuro.
        alert = _panel([0, 1, 1, 0, 0, 1, 0, 0, 2, 2], [0.0] * 10)
        confirmation = _panel([0] * 10, [0.0] * 10)
        config = FusionConfig(alert_crisis_state=2, confirmation_persistence=1, warning_horizon=3)
        fused = fuse_early_warning(alert, confirmation, config)
        self.assertEqual(fused["alert_active"].tolist(), [False] * 8 + [True, True])
        self.assertTrue(assert_prefix_causal(alert, confirmation, config, fractions=(0.3, 0.5, 0.8)))

    def test_states_above_declared_crisis_state_are_rejected(self):
        alert = _panel([0, 3, 0], [0.1, 0.9, 0.1])
        with self.assertRaises(ValueError):
            fuse_early_warning(alert, alert, FusionConfig(alert_crisis_state=1))

    def test_probability_source_ignores_state(self):
        alert = _panel([0, 0, 0, 0], [0.1, 0.8, 0.8, 0.1])
        confirmation = _panel([0] * 4, [0.0] * 4)
        config = FusionConfig(alert_source="probability", alert_threshold=0.5,
                              confirmation_persistence=1, warning_horizon=2)
        fused = fuse_early_warning(alert, confirmation, config)
        self.assertEqual(fused["alert_onset"].tolist(), [False, True, False, False])

    def test_crisis_state_from_metrics(self):
        metrics = pd.DataFrame({"pista": ["A", "B"], "id": ["D08", "D08"], "n_states": [4, 2]})
        self.assertEqual(crisis_state_from_metrics(metrics, "a", "d08"), 3)
        with self.assertRaises(KeyError):
            crisis_state_from_metrics(metrics, "A", "D02")


class LoadPanelTests(unittest.TestCase):
    def test_load_panel_rejects_nan_and_disorder(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "panels").mkdir()
            good = _panel([0, 1], [0.1, 0.9])
            good.to_parquet(Path(tmp) / "panels" / "A_D01.parquet")
            self.assertEqual(len(load_panel(Path(tmp), "a", "d01")), 2)
            good.iloc[::-1].to_parquet(Path(tmp) / "panels" / "A_D02.parquet")
            with self.assertRaises(ValueError):
                load_panel(Path(tmp), "A", "D02")
            bad = good.assign(p_crisis=[np.nan, 0.5])
            bad.to_parquet(Path(tmp) / "panels" / "A_D03.parquet")
            with self.assertRaises(ValueError):
                load_panel(Path(tmp), "A", "D03")


class UtilityAndVerdictTests(unittest.TestCase):
    def test_operational_utility_formula(self):
        self.assertAlmostEqual(operational_utility(0.6, 0.5, 0.1, 0.02), -0.1)

    def test_verdict_covers_four_cases(self):
        self.assertEqual(fusion_verdict(0.0, -1.0, -1.0), "mejora_ambos")
        self.assertEqual(fusion_verdict(0.0, 1.0, -1.0), "mejora_confirmador_pero_no_alerta")
        self.assertEqual(fusion_verdict(0.0, -1.0, 1.0), "mejora_alerta_pero_no_confirmador")
        self.assertEqual(fusion_verdict(0.0, 1.0, 1.0), "no_mejora")

    def test_scorecards_share_utility_and_labels(self):
        alert = _panel([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], [0.0] * 10)
        confirmation = _panel([0] * 10, [0, 0, 0, 0.9, 0.9, 0.9, 0.9, 0, 0, 0])
        config = FusionConfig(alert_crisis_state=1, confirmation_persistence=1, warning_horizon=3)
        with _Windows({"c": ("2020-01-03", "2020-01-08")}, {"t": ("2020-01-13", "2020-01-14")}):
            fused = fuse_early_warning(alert, confirmation, config)
            ablation = ablation_scorecard(fused, 3, alert_label="D2", confirmation_label="D6")
            events = event_scorecard(fused).set_index("signal")
            crisis = per_crisis_scorecard(fused, alert_label="D2", confirmation_label="D6")
        self.assertIn("confirmacion_D6_tras_D2", set(ablation["arquitectura"]))
        self.assertEqual(list(crisis.columns), ["crisis", "D2_alerta", "D6_confirmacion", "fusion_paralela"])
        joint = ablation.set_index("arquitectura").loc["fusion_paralela"]
        self.assertAlmostEqual(joint["operational_utility"], events.loc["accionable", "operational_utility"])
        expected = operational_utility(joint["mean_crisis_coverage"], joint["false_alarm_rate"],
                                       joint["mean_trap_activation"], joint["switching_rate"])
        self.assertAlmostEqual(joint["operational_utility"], expected)


class BaseRateAndSplitTests(unittest.TestCase):
    def test_phase_base_rates_match_ground_truth_classification(self):
        index = pd.date_range("2020-01-01", periods=20, freq="B")
        windows = {"evento": (str(index[8].date()), str(index[10].date()))}
        rates = warning_phase_base_rates(index, 3, windows)
        self.assertAlmostEqual(rates.sum(), 1.0)
        # 3 sesiones antes, 3 durante, 3 después; 11 restantes = falsa alarma.
        self.assertAlmostEqual(rates["anticipacion"], 3 / 20)
        self.assertAlmostEqual(rates["durante_crisis"], 3 / 20)
        self.assertAlmostEqual(rates["posterior"], 3 / 20)
        self.assertAlmostEqual(rates["falsa_alarma"], 11 / 20)
        # Coherencia con la tabla de avisos: un aviso en la posición 6 anticipa.
        states = [0] * 20
        states[6] = 1
        fused = fuse_early_warning(
            _panel(states, [0.0] * 20), _panel([0] * 20, [0.0] * 20),
            FusionConfig(alert_crisis_state=1, confirmation_persistence=1, warning_horizon=3),
        )
        table = warning_ground_truth_table(fused, 3, windows)
        self.assertEqual(table.loc[0, "phase"], "anticipacion")
        self.assertEqual(table.loc[0, "lead_to_crisis_start_sessions"], 2)

    def test_period_scorecard_slices_without_recomputing_state(self):
        alert = _panel([0, 1, 1, 0, 0, 0, 1, 1, 0, 0], [0.0] * 10)
        confirmation = _panel([0] * 10, [0.0] * 10)
        config = FusionConfig(alert_crisis_state=1, confirmation_persistence=1, warning_horizon=1)
        with _Windows({"c1": ("2020-01-02", "2020-01-03"), "c2": ("2020-01-09", "2020-01-10")},
                      {"t": ("2020-01-14", "2020-01-14")}):
            fused = fuse_early_warning(alert, confirmation, config)
            table = period_scorecard(fused, {"h1": ("2020-01-01", "2020-01-07"),
                                             "h2": ("2020-01-08", "2020-01-14")}).set_index("periodo")
        self.assertEqual(table["n_sesiones"].sum(), 10)
        self.assertAlmostEqual(table.loc["h1", "mean_crisis_coverage"], 1.0)
        self.assertAlmostEqual(table.loc["h2", "mean_crisis_coverage"], 1.0)
        # La trampa (14-ene) solo cae en h2: en h1 la utilidad omite ese término.
        self.assertFalse(table.loc["h1", "trampas_en_tramo"])
        self.assertTrue(table.loc["h2", "trampas_en_tramo"])
        self.assertTrue(np.isfinite(table.loc["h1", "operational_utility"]))

    def test_select_then_evaluate_reports_out_of_sample_rank(self):
        table = pd.DataFrame({
            "alerta": ["X", "X", "Y", "Y", "Z", "Z"],
            "periodo": ["seleccion", "evaluacion"] * 3,
            "utilidad": [0.5, -0.4, 0.2, 0.1, 0.9, 0.9],
            "elegible": [True, True, True, True, False, False],
        })
        out = select_then_evaluate(table)
        self.assertTrue(out.loc["X", "elegido_en_seleccion"])
        self.assertEqual(out.loc["X", "puesto_seleccion"], 1)
        self.assertEqual(out.loc["X", "puesto_evaluacion"], 2)
        self.assertTrue(np.isnan(out.loc["Z", "puesto_seleccion"]))
        self.assertFalse(out.loc["Z", "elegido_en_seleccion"])


if __name__ == "__main__":
    unittest.main()
