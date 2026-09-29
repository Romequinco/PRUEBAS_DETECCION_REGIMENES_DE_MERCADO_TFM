"""Criterio de ranking de detección (ADR-003): métricas por evento, niveles y líneas base."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src import benchmark as bm
from src import evaluation as ev


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmark_v2"
BASELINES = {"SIEMPRE_CRISIS", "SIEMPRE_CALMA", "AZAR_IID", "AZAR_PERSISTENTE"}

WINDOWS = {"corta": ("2020-01-06", "2020-01-10"), "larga": ("2020-02-03", "2020-02-28"),
           "fuera": ("2019-01-01", "2019-01-31")}
INDEX = pd.bdate_range("2019-12-02", "2020-04-30")


def _flags(on: list[tuple[str, str]]) -> pd.Series:
    flags = pd.Series(False, index=INDEX)
    for start, end in on:
        flags.loc[start:end] = True
    return flags


class EventDetectionTests(unittest.TestCase):
    def test_requires_consecutive_run_inside_window(self) -> None:
        # dos días sueltos en "larga" no bastan; tres consecutivos en "corta" sí
        flags = _flags([("2020-01-07", "2020-01-09"), ("2020-02-04", "2020-02-04"),
                        ("2020-02-06", "2020-02-06")])
        table = bm.event_detection_table(flags, WINDOWS, min_run=3)
        self.assertEqual(table.loc["corta", "detectada"], 1.0)
        self.assertEqual(table.loc["larga", "detectada"], 0.0)
        self.assertEqual(table.loc["larga", "racha_max"], 1)
        self.assertTrue(np.isnan(table.loc["fuera", "detectada"]))

    def test_short_window_needs_all_its_days(self) -> None:
        windows = {"mini": ("2020-01-06", "2020-01-07")}
        self.assertEqual(bm.event_detection_table(
            _flags([("2020-01-06", "2020-01-07")]), windows)["detectada"].iloc[0], 1.0)
        self.assertEqual(bm.event_detection_table(
            _flags([("2020-01-06", "2020-01-06")]), windows)["detectada"].iloc[0], 0.0)

    def test_summary_matches_judge_false_alarm_rate(self) -> None:
        flags = _flags([("2020-01-06", "2020-01-10"), ("2020-03-02", "2020-03-20")])
        summary = bm.detection_summary(flags, WINDOWS)
        windows = {k: v for k, v in WINDOWS.items()}
        far = ev.false_alarm_rate(flags.astype(int), 1, windows)
        self.assertAlmostEqual(summary["det_precision"], 1 - far)
        self.assertEqual(summary["det_n_eventos"], 2)  # "fuera" no se evalúa
        self.assertEqual(summary["det_event_recall"], 0.5)
        inside = (INDEX >= "2020-01-06") & (INDEX <= "2020-01-10") | \
                 (INDEX >= "2020-02-03") & (INDEX <= "2020-02-28")
        self.assertAlmostEqual(summary["det_base_rate"], inside.mean())
        self.assertAlmostEqual(summary["det_lift_precision"],
                               summary["det_precision"] / summary["det_base_rate"])

    def test_never_marking_has_nan_precision_and_zero_score(self) -> None:
        summary = bm.detection_summary(_flags([]), WINDOWS)
        self.assertTrue(np.isnan(summary["det_precision"]))
        self.assertEqual(bm.detection_score(summary["det_precision"], 0.0), 0.0)

    def test_score_is_f_beta(self) -> None:
        self.assertAlmostEqual(bm.detection_score(0.5, 1.0, 1.0), 2 * 0.5 / 1.5)
        self.assertAlmostEqual(bm.detection_score(0.5, 1.0, 2.0), 5 * 0.5 / (4 * 0.5 + 1))


class CoverageApproximationTests(unittest.TestCase):
    def test_panel_and_coverage_sources_agree_on_consecutive_signal(self) -> None:
        spec = {"crisis_windows": {"pista_A": {k: list(v) for k, v in WINDOWS.items()}},
                "false_positive_windows": {}}
        flags = _flags([("2020-01-06", "2020-01-10"), ("2020-02-10", "2020-02-14"),
                        ("2020-03-02", "2020-03-06")])
        states = flags.astype(int)
        row = {"pista": "A", "id": "D01", "n_states": 2, "n_oos": len(INDEX),
               "false_alarm_rate": ev.false_alarm_rate(states, 1, WINDOWS),
               "oos_start": str(INDEX[0].date()), "oos_end": str(INDEX[-1].date())}
        row.update({f"cov_{k}": v for k, v in ev.crisis_coverage(states, 1, WINDOWS).items()})
        metrics = pd.DataFrame([row])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            approx = bm.add_detection_metrics(metrics, output, benchmark=spec)
            (output / "panels").mkdir()
            pd.DataFrame({"state": states, "p_crisis": states.astype(float), "fold": 0}).to_parquet(
                output / "panels" / "A_D01.parquet")
            exact = bm.add_detection_metrics(metrics, output, benchmark=spec)
        self.assertEqual(approx.loc[0, "fuente_deteccion"], "cobertura_aprox_bdate")
        self.assertEqual(exact.loc[0, "fuente_deteccion"], "panel")
        for col in ("det_event_recall", "det_precision", "det_base_rate", "det_marked_rate"):
            self.assertAlmostEqual(approx.loc[0, col], exact.loc[0, col], msg=col)

    def test_csv_columns_take_priority(self) -> None:
        metrics = pd.DataFrame([{"pista": "A", "id": "D01", "det_min_run": 3,
                                 "det_event_recall": 0.25, "det_precision": 0.5}])
        out = bm.add_detection_metrics(metrics, Path("no-existe"))
        self.assertEqual(out.loc[0, "fuente_deteccion"], "csv")
        self.assertEqual(out.loc[0, "det_event_recall"], 0.25)


def _det_row(track: str, det: str, recall: float, precision: float, base: float,
             switching: float, duration: float, stability: float = 0.99,
             marked: float = 0.2, crisis_run: float = 20.0) -> dict:
    return {
        "pista": track, "id": det, "cov_evento": recall, "fa_trampa": 0.0,
        "false_alarm_rate": 1 - precision if not np.isnan(precision) else np.nan,
        "switching_rate": switching, "mean_regime_duration": duration,
        "label_stability": stability, "det_min_run": 3, "det_event_recall": recall,
        "det_precision": precision, "det_base_rate": base,
        "det_lift_precision": precision / base, "det_marked_rate": marked,
        "det_mean_crisis_run": crisis_run,
    }


class RankDetectionTests(unittest.TestCase):
    def test_levels_order_before_score(self) -> None:
        metrics = pd.DataFrame([
            _det_row("A", "BUENO", 0.8, 0.5, 0.2, 0.01, 50),
            _det_row("A", "MEDIOCRE", 0.3, 0.25, 0.2, 0.01, 50),
            _det_row("A", "PEOR_QUE_AZAR", 1.0, 0.19, 0.2, 0.01, 50),
            _det_row("A", "PARPADEO", 1.0, 0.6, 0.2, 0.40, 2.5),
            _det_row("A", "ALARMAS_SUELTAS", 1.0, 0.6, 0.2, 0.05, 20, crisis_run=1.2),
            _det_row("A", "CONSTANTE", 1.0, 0.2, 0.2, 0.0, 1000, marked=1.0),
        ])
        ranked = bm.rank_detection(metrics, output_dir=Path("no-existe")).set_index("id")
        self.assertEqual(list(ranked.sort_values("puesto_deteccion").index),
                         ["BUENO", "MEDIOCRE", "PEOR_QUE_AZAR", "ALARMAS_SUELTAS", "PARPADEO",
                          "CONSTANTE"])
        self.assertEqual(ranked.loc["PARPADEO", "nivel_etiqueta"], "parpadeo")
        self.assertEqual(ranked.loc["CONSTANTE", "nivel_etiqueta"], "degenerado")
        self.assertIn("rank_medio_legacy", ranked)
        self.assertIn("puesto_legacy", ranked)

    def test_persistence_only_breaks_ties(self) -> None:
        metrics = pd.DataFrame([
            _det_row("A", "NERVIOSO", 0.8, 0.5, 0.2, 0.05, 20),
            _det_row("A", "TRANQUILO", 0.8, 0.5, 0.2, 0.01, 100),
            _det_row("A", "MEJOR_SCORE_NERVIOSO", 0.9, 0.5, 0.2, 0.15, 6),
        ])
        ranked = bm.rank_detection(metrics, output_dir=Path("no-existe")).set_index("id")
        self.assertEqual(ranked.loc["MEJOR_SCORE_NERVIOSO", "puesto_deteccion"], 1)
        self.assertEqual(ranked.loc["TRANQUILO", "puesto_deteccion"], 2)
        self.assertEqual(ranked.loc["NERVIOSO", "puesto_deteccion"], 3)

    def test_full_ties_share_place_and_tracks_are_independent(self) -> None:
        metrics = pd.DataFrame([
            _det_row("A", "X", 0.8, 0.5, 0.2, 0.01, 50),
            _det_row("A", "Y", 0.8, 0.5, 0.2, 0.01, 50),
            _det_row("B", "Z", 0.1, 0.25, 0.2, 0.01, 50),
        ])
        ranked = bm.rank_detection(metrics, output_dir=Path("no-existe")).set_index("id")
        self.assertEqual(ranked.loc["X", "puesto_deteccion"], 1)
        self.assertEqual(ranked.loc["Y", "puesto_deteccion"], 1)
        self.assertEqual(ranked.loc["Z", "puesto_deteccion"], 1)

    def test_pareto_mask(self) -> None:
        frame = pd.DataFrame({"r": [1.0, 0.5, 0.4, 0.9], "p": [0.2, 0.5, 0.4, 0.2]})
        self.assertEqual(bm.pareto_mask(frame, ["r", "p"]).tolist(), [True, True, False, False])


class BaselinesAtBottomTests(unittest.TestCase):
    """Las líneas base triviales deben quedar al fondo con el criterio nuevo."""

    def _rank_with_baselines(self, metrics: pd.DataFrame, output: Path) -> pd.DataFrame:
        ranked = bm.rank_detection(metrics, output_dir=output)
        rows = []
        for track, part in ranked.groupby("pista"):
            index, _ = bm.track_oos_index(track, metrics, output)
            rate = float(part["det_marked_rate"].median())
            mean_run = float((2 * part["det_marked_rate"] * part["mean_regime_duration"]).median())
            rows.append(bm.trivial_baselines(track, index, rate=rate, mean_run=mean_run, n_sims=40))
        return bm.rank_detection(pd.concat([metrics, *rows], ignore_index=True), output_dir=output)

    def _assert_bottom(self, ranked: pd.DataFrame) -> None:
        for track, part in ranked.groupby("pista"):
            real = part[~part["id"].isin(BASELINES)]
            base = part[part["id"].isin(BASELINES)].set_index("id")
            worst_real = real["puesto_deteccion"].max()
            for name in ("SIEMPRE_CRISIS", "SIEMPRE_CALMA", "AZAR_IID"):
                self.assertGreater(base.loc[name, "puesto_deteccion"], worst_real,
                                   f"{track}: {name} no queda al fondo")
            eligible_real = real.loc[real["elegible"], "puesto_deteccion"]
            self.assertFalse(bool(base.loc["AZAR_PERSISTENTE", "elegible"]))
            if len(eligible_real):
                self.assertGreater(base.loc["AZAR_PERSISTENTE", "puesto_deteccion"],
                                   eligible_real.max())
            self.assertEqual(base.loc["SIEMPRE_CRISIS", "nivel_etiqueta"], "degenerado")
            self.assertEqual(base.loc["SIEMPRE_CALMA", "nivel_etiqueta"], "degenerado")
            self.assertEqual(base.loc["AZAR_IID", "nivel_etiqueta"], "parpadeo")

    @unittest.skipUnless((RESULTS / "metrics").is_dir(), "sin métricas versionadas")
    def test_versioned_metrics(self) -> None:
        metrics = bm.load_metrics(RESULTS, require_current=False)
        self._assert_bottom(self._rank_with_baselines(metrics, RESULTS))

    def test_synthetic_detectors_even_when_always_crisis_scores_higher(self) -> None:
        # SIEMPRE_CRISIS tiene F1 = 2·0,19/(1,19) ≈ 0,32, por encima de estos dos
        # detectores flojos; aun así debe quedar por debajo de ellos.
        rng = np.random.default_rng(1)
        index = pd.bdate_range("2000-01-03", "2010-12-31")
        windows = {"c1": ("2001-03-01", "2001-10-31"), "c2": ("2002-05-01", "2002-10-31"),
                   "c3": ("2007-10-01", "2008-03-31"), "c4": ("2008-09-01", "2009-03-31")}
        spec = {"crisis_windows": {"pista_A": {k: list(v) for k, v in windows.items()}},
                "false_positive_windows": {}}
        rows = []
        for name, periods in {
            # recall 1/4 y precisión 0,25 (lift > 1): F1 ≈ 0,25 < F1(SIEMPRE_CRISIS)
            "DEBIL": [("2001-04-02", "2001-04-13"), ("2005-01-03", "2005-02-07")],
            "OTRO": [("2008-10-01", "2008-10-14"), ("2003-01-06", "2003-02-10")],
        }.items():
            flags = pd.Series(False, index=index)
            for start, end in periods:
                flags.loc[start:end] = True
            states = flags.astype(int)
            row = {"pista": "A", "id": name, "n_states": 2, "n_oos": len(index),
                   "false_alarm_rate": ev.false_alarm_rate(states, 1, windows),
                   "switching_rate": ev.switching_rate(states),
                   "mean_regime_duration": ev.mean_regime_duration(states),
                   "label_stability": 0.99, "fa_x": 0.0}
            row.update({f"cov_{k}": v for k, v in ev.crisis_coverage(states, 1, windows).items()})
            row.update(bm.detection_summary(flags, windows))
            rows.append(row)
        metrics = pd.DataFrame(rows)
        base = bm.trivial_baselines("A", index, rate=0.05, mean_run=10, benchmark=spec,
                                    n_sims=20, seed=int(rng.integers(1000)))
        ranked = bm.rank_detection(pd.concat([metrics, base], ignore_index=True),
                                   output_dir=Path("no-existe"), benchmark=spec)
        crisis = ranked.set_index("id").loc["SIEMPRE_CRISIS"]
        self.assertGreater(crisis["score_deteccion"],
                           ranked.set_index("id").loc["DEBIL", "score_deteccion"])
        self._assert_bottom(ranked)


@unittest.skipUnless((RESULTS / "ranking_v2.csv").exists(), "sin ranking versionado")
class VersionedRankingTests(unittest.TestCase):
    def test_ranking_csv_has_new_and_legacy_columns(self) -> None:
        stored = pd.read_csv(RESULTS / "ranking_v2.csv")
        for col in ("puesto_deteccion", "score_deteccion", "det_event_recall",
                    "det_precision", "nivel_etiqueta", "rank_medio_legacy", "puesto_legacy"):
            self.assertIn(col, stored)
        self.assertFalse(stored["id"].isin(BASELINES).any())

    def test_ranking_csv_reproducible_when_metrics_carry_detection(self) -> None:
        metrics = bm.load_metrics(RESULTS, require_current=False)
        if "det_event_recall" not in metrics or metrics["det_event_recall"].isna().any():
            self.skipTest("métricas previas a ADR-003: el ranking depende de los paneles locales")
        rebuilt = bm.rank_detection(metrics, output_dir=RESULTS)
        stored = pd.read_csv(RESULTS / "ranking_v2.csv")
        merged = stored.merge(rebuilt, on=["pista", "id"], suffixes=("_csv", "_new"))
        self.assertEqual(len(merged), len(stored))
        np.testing.assert_array_equal(merged["puesto_deteccion_csv"], merged["puesto_deteccion_new"])
        np.testing.assert_allclose(merged["score_deteccion_csv"], merged["score_deteccion_new"],
                                   atol=1e-12)


if __name__ == "__main__":
    unittest.main()
