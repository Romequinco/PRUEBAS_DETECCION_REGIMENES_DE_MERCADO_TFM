"""Regresiones del ranking descriptivo, del modo solo-caché y de la trazabilidad."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from regimenes import rutas
from regimenes.benchmark import cache as bm_cache
from regimenes.benchmark import ejecucion as bm_run
from regimenes.evaluacion import ranking as rk


RESULTS = rutas.RESULTS_BENCHMARK


def _row(track: str, det: str, cov: float, far: float, trap: float,
         switching: float, stability: float) -> dict:
    return {
        "pista": track, "id": det, "cov_evento": cov, "false_alarm_rate": far,
        "fa_trampa": trap, "switching_rate": switching,
        "mean_regime_duration": 1 / max(switching, 1e-9), "label_stability": stability,
    }


class RankWithinTrackTests(unittest.TestCase):
    def test_tracks_are_ranked_independently(self) -> None:
        metrics = pd.DataFrame([
            _row("A", "D01", 0.9, 0.1, 0.0, 0.01, 0.99),
            _row("A", "D02", 0.1, 0.9, 0.5, 0.10, 0.50),
            _row("B", "D01", 0.1, 0.9, 0.5, 0.10, 0.50),
            _row("B", "D02", 0.9, 0.1, 0.0, 0.01, 0.99),
        ])
        ranked = rk.rank_within_track(metrics).set_index(["pista", "id"])
        self.assertEqual(ranked.loc[("A", "D01"), "puesto_pista"], 1)
        self.assertEqual(ranked.loc[("B", "D02"), "puesto_pista"], 1)

    def test_ties_share_average_rank_and_min_place(self) -> None:
        metrics = pd.DataFrame([
            _row("A", "D01", 0.5, 0.5, 0.0, 0.01, 0.9),
            _row("A", "D02", 0.5, 0.5, 0.0, 0.01, 0.9),
            _row("A", "D03", 0.1, 0.9, 0.9, 0.50, 0.5),
        ])
        ranked = rk.rank_within_track(metrics).set_index("id")
        self.assertEqual(ranked.loc["D01", "rank_mean_crisis_coverage"], 1.5)
        self.assertEqual(ranked.loc["D01", "puesto_pista"], 1)
        self.assertEqual(ranked.loc["D02", "puesto_pista"], 1)
        self.assertEqual(ranked.loc["D03", "puesto_pista"], 3)

    def test_nan_false_alarm_rate_goes_to_bottom(self) -> None:
        metrics = pd.DataFrame([
            _row("A", "D01", 0.5, 0.8, 0.0, 0.01, 0.9),
            _row("A", "SILENCIOSO", 0.0, np.nan, 0.0, 0.0, 1.0),
        ])
        ranked = rk.rank_within_track(metrics).set_index("id")
        self.assertEqual(ranked.loc["SILENCIOSO", "rank_false_alarm_rate"], 2)

    def test_coverage_mean_skips_crises_outside_oos(self) -> None:
        metrics = pd.DataFrame([{
            "pista": "B", "id": "D01", "cov_gfc": np.nan, "cov_gfc_lo": 0.0,
            "cov_gfc_hi": 1.0, "cov_covid": 0.8, "cov_svb": 0.2, "fa_taper": 0.0,
        }])
        out = rk.add_comparison_metrics(metrics)
        self.assertAlmostEqual(out.loc[0, "mean_crisis_coverage"], 0.5)


@unittest.skipUnless((RESULTS / "metrics_master_v2.csv").exists(), "sin resultados versionados")
class VersionedArtifactsTests(unittest.TestCase):
    """Los CSV derivados deben poder regenerarse exactamente desde metrics/."""

    def test_ranking_is_reproducible_from_metric_files(self) -> None:
        metrics = bm_cache.load_metrics(RESULTS, require_current=False)
        rebuilt = rk.rank_within_track(metrics).reset_index(drop=True)
        stored = pd.read_csv(RESULTS / "metrics_master_v2.csv")
        self.assertEqual(len(rebuilt), 24)
        key = ["pista", "id"]
        merged = stored.merge(rebuilt, on=key, suffixes=("_csv", "_new"))
        for col in ("mean_crisis_coverage", "mean_trap_activation", "rank_medio", "puesto_pista"):
            np.testing.assert_allclose(merged[f"{col}_csv"], merged[f"{col}_new"], rtol=0, atol=1e-12)

    def test_every_track_has_twelve_detectors_on_one_oos_window(self) -> None:
        metrics = bm_cache.load_metrics(RESULTS, require_current=False)
        for _, part in metrics.groupby("pista"):
            self.assertEqual(sorted(part["id"]), [f"D{i:02d}" for i in range(1, 13)])
            self.assertEqual(part["oos_start"].nunique(), 1)
            self.assertEqual(part["oos_end"].nunique(), 1)
            self.assertEqual(part["n_oos"].nunique(), 1)

    def test_metric_files_match_manifest_fingerprints(self) -> None:
        provenance = bm_cache.metrics_provenance(RESULTS)
        self.assertEqual(len(provenance), 24)
        self.assertTrue(provenance["coincide_manifest"].all())


class CacheOnlyModeTests(unittest.TestCase):
    def test_cache_only_never_fits_nor_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with mock.patch.object(bm_run, "run_one") as run_one, \
                 mock.patch.object(bm_cache, "_cache_fingerprint", return_value="x"):
                metrics, status = bm_run.run_benchmark(
                    tracks=("A",), detector_ids=("D01",), output_dir=output, cache_only=True,
                )
            run_one.assert_not_called()
            self.assertTrue(metrics.empty)
            self.assertEqual(status["estado"].tolist(), ["sin_cache"])
            self.assertEqual(list(output.iterdir()), [])

    def test_cache_only_reads_verified_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "metrics").mkdir()
            (output / "panels").mkdir()
            pd.DataFrame({"pista": ["A"], "id": ["D01"], "cache_fingerprint": ["x"]}).to_csv(
                output / "metrics" / "A_D01.csv", index=False)
            (output / "panels" / "A_D01.parquet").write_bytes(b"panel")
            with mock.patch.object(bm_cache, "_cache_fingerprint", return_value="x"):
                metrics, status = bm_run.run_benchmark(
                    tracks=("A",), detector_ids=("D01",), output_dir=output, cache_only=True,
                )
            self.assertEqual(status["estado"].tolist(), ["cache"])
            self.assertEqual(len(metrics), 1)

    def test_force_and_cache_only_are_incompatible(self) -> None:
        with self.assertRaises(ValueError):
            bm_run.run_benchmark(tracks=("A",), cache_only=True, force=True)


class ProvenanceTests(unittest.TestCase):
    def test_provenance_flags_mismatch_without_local_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "metrics").mkdir()
            pd.DataFrame({"pista": ["A"], "id": ["D01"], "cache_fingerprint": ["x"]}).to_csv(
                output / "metrics" / "A_D01.csv", index=False)
            pd.DataFrame({"pista": ["B"], "id": ["D01"], "cache_fingerprint": ["y"]}).to_csv(
                output / "metrics" / "B_D01.csv", index=False)
            (output / "manifest.json").write_text(json.dumps({
                "cache_fingerprints": {"A": {"D01": "x"}, "B": {"D01": "otra"}},
            }), encoding="utf-8")
            prov = bm_cache.metrics_provenance(output).set_index("pista")
            self.assertTrue(prov.loc["A", "coincide_manifest"])
            self.assertFalse(prov.loc["B", "coincide_manifest"])
            self.assertFalse(prov["panel_oos_presente"].any())


if __name__ == "__main__":
    unittest.main()
