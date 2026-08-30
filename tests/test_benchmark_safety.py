"""Regresiones de seguridad para configuración, ranking y caché del benchmark."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest import mock
import json

import pandas as pd

from src import evaluation as ev
from src import viz
from src import benchmark as bm


class EventWindowConfigurationTests(unittest.TestCase):
    def test_configuration_updates_existing_import_references(self) -> None:
        originals = (
            ev.CRISIS_WINDOWS.copy(),
            ev.FALSE_POSITIVE_WINDOWS.copy(),
            ev.DRAWDOWN_TROUGHS.copy(),
        )
        crisis_reference = viz.CRISIS_WINDOWS
        false_positive_reference = viz.FALSE_POSITIVE_WINDOWS
        trough_reference = viz.DRAWDOWN_TROUGHS
        try:
            ev.configure_event_windows(
                {"evento": ("2020-01-01", "2020-01-31")},
                {"trampa": ("2021-01-01", "2021-01-31")},
                {"evento": "2020-01-15"},
            )
            self.assertIs(crisis_reference, ev.CRISIS_WINDOWS)
            self.assertIs(false_positive_reference, ev.FALSE_POSITIVE_WINDOWS)
            self.assertIs(trough_reference, ev.DRAWDOWN_TROUGHS)
            self.assertEqual(tuple(viz.CRISIS_WINDOWS), ("evento",))
            self.assertEqual(tuple(viz.FALSE_POSITIVE_WINDOWS), ("trampa",))
            self.assertEqual(viz.DRAWDOWN_TROUGHS, {"evento": "2020-01-15"})
        finally:
            ev.configure_event_windows(*originals)


class RankingTests(unittest.TestCase):
    def test_mean_duration_is_descriptive_not_double_counted(self) -> None:
        metrics = pd.DataFrame([
            {
                "pista": "A",
                "id": "D01",
                "cov_evento": 0.9,
                "false_alarm_rate": 0.1,
                "fa_trampa": 0.1,
                "switching_rate": 0.01,
                "mean_regime_duration": 1.0,
                "label_stability": 0.9,
            },
            {
                "pista": "A",
                "id": "D02",
                "cov_evento": 0.1,
                "false_alarm_rate": 0.9,
                "fa_trampa": 0.9,
                "switching_rate": 0.10,
                "mean_regime_duration": 1000.0,
                "label_stability": 0.1,
            },
        ])

        ranked = bm.rank_within_track(metrics).set_index("id")

        self.assertNotIn("rank_mean_regime_duration", ranked.columns)
        self.assertEqual(ranked.loc["D01", "rank_medio"], 1.0)
        self.assertEqual(ranked.loc["D02", "rank_medio"], 2.0)


class CacheTests(unittest.TestCase):
    def test_external_output_path_can_be_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            external = Path(temp) / "metric.csv"
            self.assertEqual(bm._display_path(external), str(external.resolve()))

    def test_cache_requires_panel_and_exact_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metric = root / "metric.csv"
            panel = root / "panel.parquet"
            pd.DataFrame({"cache_fingerprint": ["correcta"]}).to_csv(metric, index=False)

            self.assertFalse(bm._cache_matches(metric, panel, "correcta"))
            panel.write_bytes(b"panel")
            self.assertTrue(bm._cache_matches(metric, panel, "correcta"))
            self.assertFalse(bm._cache_matches(metric, panel, "otra"))

    def test_legacy_cache_without_fingerprint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metric = root / "metric.csv"
            panel = root / "panel.parquet"
            pd.DataFrame({"id": ["D01"]}).to_csv(metric, index=False)
            panel.write_bytes(b"panel")

            self.assertFalse(bm._cache_matches(metric, panel, "cualquiera"))

    def test_metric_loader_rejects_stale_results_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            metrics_dir = output / "metrics"
            panels_dir = output / "panels"
            metrics_dir.mkdir()
            panels_dir.mkdir()
            pd.DataFrame({
                "pista": ["A"],
                "id": ["D01"],
                "train_days": [bm.DEFAULT_TRAIN_DAYS["A"]],
            }).to_csv(metrics_dir / "A_D01.csv", index=False)
            (panels_dir / "A_D01.parquet").write_bytes(b"panel")

            with mock.patch.object(bm, "_cache_fingerprint", return_value="actual"):
                with self.assertRaisesRegex(RuntimeError, "obsoletas"):
                    bm.load_metrics(output)
            loaded = bm.load_metrics(output, require_current=False)
            self.assertEqual(len(loaded), 1)

    def test_metric_loader_uses_recorded_not_reader_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            metrics_dir = output / "metrics"
            panels_dir = output / "panels"
            metrics_dir.mkdir()
            panels_dir.mkdir()
            pd.DataFrame({
                "pista": ["A"],
                "id": ["D01"],
                "train_days": [bm.DEFAULT_TRAIN_DAYS["A"]],
                "cache_fingerprint": ["válida"],
            }).to_csv(metrics_dir / "A_D01.csv", index=False)
            (panels_dir / "A_D01.parquet").write_bytes(b"panel")
            recorded_runtime = {"python": "entorno-productor"}
            (output / "manifest.json").write_text(json.dumps({
                "cache_schema_version": bm.CACHE_SCHEMA_VERSION,
                "runtime_versions": recorded_runtime,
            }), encoding="utf-8")

            with mock.patch.object(
                bm, "_cache_fingerprint", return_value="válida"
            ) as fingerprint:
                loaded = bm.load_metrics(output)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(
                fingerprint.call_args.kwargs["runtime_versions"],
                recorded_runtime,
            )


if __name__ == "__main__":
    unittest.main()
