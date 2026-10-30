"""Curva detección / falsas alarmas / retraso a igual tasa de FA al año (A3)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from regimenes import evaluacion as ev
from regimenes.evaluacion import curvas

INDEX = pd.bdate_range("2000-01-03", periods=2520)  # ~10 años
LAG = 5          # retraso plantado (sesiones desde el pico)
CRISIS_LEN = 40
CRISIS_POS = [300, 600, 900, 1200, 1500, 1800, 2100, 2400]


def _toy() -> tuple[pd.Series, dict[str, tuple[str, str]]]:
    """Ruido bajo + 40 «sustos» de 5 días en calma con alturas distintas (1,0–2,8)
    + crisis con señal 3 desde ``pico + LAG``."""
    rng = np.random.default_rng(0)
    values = rng.normal(0, 0.1, len(INDEX))
    windows = {}
    for k, p in enumerate(CRISIS_POS):
        windows[f"c{k}"] = (str(INDEX[p].date()), str(INDEX[p + CRISIS_LEN - 1].date()))
        values[p + LAG:p + CRISIS_LEN] += 3.0
    calm = [p for p in range(20, 2500, 55)
            if all(not (c - 10 <= p <= c + CRISIS_LEN + 10) for c in CRISIS_POS)][:40]
    for h, p in zip(np.linspace(1.0, 2.8, len(calm)), rng.permutation(calm)):
        values[p:p + 5] += h
    return pd.Series(values, index=INDEX), windows


class EpisodeTests(unittest.TestCase):
    def test_false_alarm_is_episode_starting_outside_windows(self) -> None:
        flags = pd.Series(False, index=INDEX[:100])
        windows = {"c": (str(INDEX[50].date()), str(INDEX[70].date()))}
        flags.iloc[10:12] = True   # 2 días: no es alarma
        flags.iloc[20:25] = True   # falsa
        flags.iloc[48:60] = True   # empieza 2 días antes del pico: falsa…
        flags.iloc[80:90] = True   # tras el suelo: falsa
        ep = ev.episodios_alarma(flags, windows)
        self.assertEqual(list(ep["falsa"]), [True, True, True])
        # …salvo que se admita margen previo
        ep = ev.episodios_alarma(flags, windows, margen_previo=5)
        self.assertEqual(list(ep["falsa"]), [True, False, True])


class ExPostCurveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.score, self.windows = _toy()
        self.table = ev.curva_deteccion(self.score, self.windows, n_grid=3000)

    def test_detects_all_with_planted_delay(self) -> None:
        for _, row in self.table.iterrows():
            self.assertEqual(row["n_detectadas"], len(CRISIS_POS))
            self.assertEqual(row["tasa_deteccion"], 1.0)
            self.assertEqual(row["retraso_mediano"], LAG)
            self.assertEqual(row["leadlag_suelo_mediano"], LAG - (CRISIS_LEN - 1))
            self.assertLess(row["ratio_ruido_senal"], 0.1)

    def test_realised_false_alarms_match_target(self) -> None:
        for _, row in self.table.iterrows():
            step = 1 / row["anios_calma"]  # resolución: una alarma
            self.assertLessEqual(row["fa_anio"], row["fa_objetivo"] + 1e-12)
            self.assertGreater(row["fa_anio"], row["fa_objetivo"] - step)
        # más FA permitidas → umbral más bajo
        self.assertTrue(self.table["umbral"].is_monotonic_decreasing)

    def test_noise_to_signal_definition(self) -> None:
        thr = float(self.table["umbral"].iloc[1])
        summary, _ = ev.metricas_umbral(self.score, thr, self.windows)
        flags = self.score >= thr
        inside = np.zeros(len(INDEX), bool)
        for a, b in self.windows.values():
            inside |= (INDEX >= a) & (INDEX <= b)
        expected = flags[~inside].mean() / flags[inside].mean()
        self.assertAlmostEqual(summary["ratio_ruido_senal"], expected)

    def test_detection_matches_event_table_and_lead_lag(self) -> None:
        detail = self.table.attrs["detalle"][1.0]
        self.assertTrue((detail["retraso"] == LAG).all())
        self.assertTrue(detail["detectada"].eq(1.0).all())

    def test_trap_windows_are_counted(self) -> None:
        trap = {"t": (str(INDEX[0].date()), str(INDEX[299].date()))}
        table = ev.curva_deteccion(self.score, self.windows, fp_windows=trap, n_grid=3000)
        self.assertTrue((table["n_fa_trampas"] <= table["n_falsas_alarmas"]).all())


class ExpandingCurveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.score, self.windows = _toy()
        self.kw = dict(min_obs=504, paso=126, n_grid=500)

    def test_truncation_does_not_change_the_past(self) -> None:
        full = ev.umbral_expansivo(self.score, self.windows, **self.kw)
        cut = 1700
        short = ev.umbral_expansivo(self.score.iloc[:cut], self.windows, **self.kw)
        pd.testing.assert_frame_equal(full.iloc[:cut], short)
        self.assertTrue(full.iloc[:504].isna().all().all())
        self.assertTrue(full.iloc[504:].notna().all().all())

    def test_threshold_ignores_current_and_future_scores(self) -> None:
        bumped = self.score.copy()
        bumped.iloc[1000:] += 50.0  # cambiar el día de recalibración y el futuro
        a = ev.umbral_expansivo(self.score, self.windows, **self.kw)
        b = ev.umbral_expansivo(bumped, self.windows, **self.kw)
        pd.testing.assert_frame_equal(a.iloc[:1008], b.iloc[:1008])  # 1008 = 504 + 4·126

    def test_curve_runs_after_burn_in(self) -> None:
        table = ev.curva_deteccion(self.score, self.windows, modo="expansivo", **self.kw)
        self.assertEqual(list(table["dias_burn_in"]), [504] * 3)
        # crisis antes del burn-in (posición 300) no se evalúa
        self.assertTrue((table["n_eventos"] == len(CRISIS_POS) - 1).all())
        self.assertTrue((table["retraso_mediano"] == LAG).all())

    def test_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            curvas.curva_deteccion(self.score, self.windows, modo="otro")


if __name__ == "__main__":
    unittest.main()
