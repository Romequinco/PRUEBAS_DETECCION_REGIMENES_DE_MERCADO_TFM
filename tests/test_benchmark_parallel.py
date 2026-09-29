"""Ejecución paralela / por detector y huella de caché por contenido."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src import benchmark as bm


class ContentFingerprintTests(unittest.TestCase):
    def test_parquet_hash_depends_on_content_not_bytes(self) -> None:
        frame = pd.DataFrame({"x": np.arange(50, dtype=float), "y": list("ab" * 25)},
                             index=pd.bdate_range("2020-01-01", periods=50))
        with tempfile.TemporaryDirectory() as temp:
            a, b, c = (Path(temp) / name for name in ("a.parquet", "b.parquet", "c.parquet"))
            frame.to_parquet(a, compression="snappy")
            frame.to_parquet(b, compression="gzip")
            changed = frame.copy()
            changed.iloc[3, 0] = 99.0
            changed.to_parquet(c)
            self.assertNotEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(bm._sha256_file(a), bm._sha256_file(b))
            self.assertNotEqual(bm._sha256_file(a), bm._sha256_file(c))

    def test_text_hash_ignores_line_endings_and_yaml_formatting(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            lf, crlf = Path(temp) / "lf.py", Path(temp) / "crlf.py"
            lf.write_bytes(b"x = 1\ny = 2\n")
            crlf.write_bytes(b"x = 1\r\ny = 2\r\n")
            self.assertEqual(bm._sha256_file(lf), bm._sha256_file(crlf))
            y1, y2 = Path(temp) / "a.yaml", Path(temp) / "b.yaml"
            y1.write_text("a: 1  # comentario\nb: [1, 2]\n", encoding="utf-8")
            y2.write_text("b:\n  - 1\n  - 2\na: 1\n", encoding="utf-8")
            self.assertEqual(bm._sha256_file(y1), bm._sha256_file(y2))

    def test_fingerprint_hashes_the_detector_base_actually_imported(self) -> None:
        spec = bm.detector_specs("A")[0]
        spec.factory()  # antepone capa1_exploracion a sys.path, como en el benchmark
        imported = Path(sys.modules["src.detector_base"].__file__).resolve()
        self.assertEqual(bm._resolved_detector_base(), imported)
        self.assertIn(imported, [p.resolve() for p in bm._cache_code_paths()])


class JobAndConsolidationTests(unittest.TestCase):
    def test_run_job_writes_only_its_own_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            fake = ({}, {"pista": "A", "id": "D01", "estado": "ok", "segundos": 1.0, "detalle": "x"})
            with mock.patch.object(bm, "run_one", return_value=fake):
                status = bm.run_job("a", "d01", output_dir=output)
            self.assertEqual(status["estado"], "ok")
            self.assertFalse((output / "manifest.json").exists())
            self.assertFalse((output / "run_status.csv").exists())
            self.assertTrue((output / "status" / "A_D01.json").is_file())

    def test_run_job_records_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(bm, "run_one", side_effect=ValueError("boom")):
                status = bm.run_job("A", "D02", output_dir=Path(temp))
            self.assertEqual(status["estado"], "error")
            saved = json.loads((Path(temp) / "status" / "A_D02.json").read_text(encoding="utf-8"))
            self.assertIn("boom", saved["detalle"])

    def test_consolidate_builds_manifest_and_sorted_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "status").mkdir()
            for track, det in (("B", "D02"), ("A", "D05"), ("B", "D01")):
                (output / "status" / f"{track}_{det}.json").write_text(json.dumps(
                    {"pista": track, "id": det, "estado": "ok", "segundos": 1.0, "detalle": ""}),
                    encoding="utf-8")
            with mock.patch.object(bm, "_cache_fingerprint", return_value="h"):
                bm.consolidate_run(output, tracks=("B",))
            status = pd.read_csv(output / "run_status.csv")
            self.assertEqual(status["id"].tolist(), ["D01", "D02"])
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["cache_fingerprints"]), {"B"})

    def test_clean_sys_path_removes_legacy_root_temporarily(self) -> None:
        sys.path.insert(0, str(bm.LEGACY_ROOT))
        try:
            with bm._clean_sys_path_for_workers():
                self.assertNotIn(str(bm.LEGACY_ROOT), sys.path)
                self.assertIn(str(bm.ROOT), sys.path)
            self.assertEqual(sys.path[0], str(bm.LEGACY_ROOT))
        finally:
            sys.path.remove(str(bm.LEGACY_ROOT))


@unittest.skipUnless(bm.processed_available("B"), "sin data/processed de la pista B")
class ParallelIntegrationTests(unittest.TestCase):
    def test_two_fast_detectors_in_parallel_outside_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            metrics, status = bm.run_benchmark(
                tracks=("B",), detector_ids=("D01", "D02"), output_dir=output, n_jobs=2)
            self.assertEqual(sorted(status["estado"]), ["ok", "ok"])
            self.assertEqual(status["pid"].nunique(), 2)
            self.assertEqual(sorted(metrics["id"]), ["D01", "D02"])
            self.assertTrue((metrics["det_precision"] - (1 - metrics["false_alarm_rate"])).abs().max() < 1e-12)
            run_status = pd.read_csv(output / "run_status.csv")
            self.assertEqual(run_status["id"].tolist(), ["D01", "D02"])
            self.assertTrue((output / "manifest.json").is_file())
            # segunda pasada: caché verificada, sin reajustar
            _, again = bm.run_benchmark(
                tracks=("B",), detector_ids=("D01", "D02"), output_dir=output, n_jobs=2)
            self.assertEqual(sorted(again["estado"]), ["cache", "cache"])


if __name__ == "__main__":
    unittest.main()
