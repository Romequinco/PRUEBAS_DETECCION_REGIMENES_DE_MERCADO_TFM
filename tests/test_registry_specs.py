"""Contrato del registro de detectores v2 y de la huella de caché."""

from __future__ import annotations

from pathlib import Path
import unittest

import yaml

from src import benchmark as bm
from src.detectors import detector_specs


ROOT = Path(__file__).resolve().parents[1]


class RegistryContractTests(unittest.TestCase):
    def test_twelve_official_detectors_per_track(self) -> None:
        for track in ("A", "B"):
            ids = [spec.detector_id for spec in detector_specs(track)]
            self.assertEqual(ids, [f"D{i:02d}" for i in range(1, 13)])

    def test_same_model_family_and_class_in_both_tracks(self) -> None:
        a = {s.detector_id: (s.module, s.class_name, s.step, s.expanding) for s in detector_specs("A")}
        b = {s.detector_id: (s.module, s.class_name, s.step, s.expanding) for s in detector_specs("B")}
        self.assertEqual(a, b)

    def test_track_a_does_not_use_track_b_only_signals(self) -> None:
        b_only = {"VIX_level_z", "VIX_change_z", "MOVE_level_z", "credit_BAA10Y_z",
                  "DFII10_change_z", "corr_spx_bond", "DXY_change_z", "slope_10y2y_z"}
        for spec in detector_specs("A"):
            self.assertFalse(set(spec.features) & b_only, spec.detector_id)

    def test_declared_params_reference_declared_features(self) -> None:
        for track in ("A", "B"):
            for spec in detector_specs(track):
                params = spec.params
                used = set()
                if "feature" in params:
                    used.add(params["feature"])
                if "features" in params:
                    used.update(params["features"])
                if "signs" in params:
                    used.update(params["signs"])
                self.assertTrue(used <= set(spec.features), (track, spec.detector_id, used))

    def test_factories_return_fresh_instances(self) -> None:
        spec = detector_specs("A")[0]
        self.assertIsNot(spec.factory(), spec.factory())

    @unittest.skipUnless(bm.processed_available(), "sin paneles data/processed locales")
    def test_features_exist_in_frozen_track_panels(self) -> None:
        for track in ("A", "B"):
            columns = set(bm.load_track_panel(track).columns)
            for spec in detector_specs(track):
                self.assertTrue(set(spec.features) <= columns, (track, spec.detector_id))

    def test_track_windows_match_frozen_spec(self) -> None:
        spec = yaml.safe_load((ROOT / "data" / "benchmark_spec.yaml").read_text(encoding="utf-8"))
        self.assertEqual(len(spec["crisis_windows"]["pista_A"]), 18)
        self.assertEqual(len(spec["crisis_windows"]["pista_B"]), 10)


class CacheFingerprintScopeTests(unittest.TestCase):
    def test_hashed_detector_base_is_identical_to_the_one_imported(self) -> None:
        """Los detectores importan ``src.detector_base`` (copia v2), pero la huella
        hashea ``capa1_exploracion/src/detector_base.py``. Mientras ambas copias
        sean idénticas la huella sigue siendo válida; si divergen, este test avisa
        de que la caché dejaría de detectar el cambio."""
        import src.detector_base as imported

        hashed = ROOT / "capa1_exploracion" / "src" / "detector_base.py"
        self.assertIn(hashed.resolve(), [p.resolve() for p in bm._cache_code_paths()])
        normalize = lambda p: Path(p).read_bytes().replace(b"\r\n", b"\n")  # noqa: E731
        self.assertEqual(normalize(imported.__file__), normalize(hashed))

    def test_model_fingerprint_covers_the_numeric_pipeline(self) -> None:
        # Si se renombra una de estas funciones, la huella debe fallar ruidosamente.
        self.assertEqual(len(bm._benchmark_model_sha256()), 64)


if __name__ == "__main__":
    unittest.main()
