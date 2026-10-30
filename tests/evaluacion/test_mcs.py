"""Model Confidence Set (A1): pérdidas sintéticas conocidas y matriz preliminar real."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import pytest

from regimenes import evaluacion as ev
from regimenes.evaluacion import mcs


def _losses(means: dict[str, float], n: int = 200, sd: float = 0.3, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.normal(0, sd, size=n)  # pérdidas correlacionadas entre modelos
    return pd.DataFrame({k: m + common + rng.normal(0, sd, size=n) for k, m in means.items()})


GOOD_BAD = {"bueno1": 0.0, "bueno2": 0.0, "malo1": 1.0, "malo2": 1.2, "malo3": 1.5}


class BootstrapIndicesTests(unittest.TestCase):
    def test_iid_and_blocks_have_right_shape_and_range(self) -> None:
        for metodo in ("circular", "estacionario"):
            idx = mcs.bootstrap_indices(50, 30, block_len=5, metodo=metodo, seed=0)
            self.assertEqual(idx.shape, (30, 50))
            self.assertTrue(((idx >= 0) & (idx < 50)).all())

    def test_circular_blocks_are_consecutive(self) -> None:
        idx = mcs.bootstrap_indices(20, 4, block_len=5, metodo="circular", seed=0)
        steps = (np.diff(idx[:, :5], axis=1)) % 20
        self.assertTrue((steps == 1).all())


class ModelConfidenceSetTests(unittest.TestCase):
    def test_keeps_good_and_drops_bad(self) -> None:
        for stat in ("T_max", "T_R"):
            res = ev.model_confidence_set(_losses(GOOD_BAD), alpha=0.10, estadistico=stat,
                                          n_boot=500, seed=3)
            self.assertEqual(set(res.supervivientes), {"bueno1", "bueno2"}, stat)
            self.assertEqual(res.tabla["p_mcs"].max(), 1.0)
            # los malos salen primero
            order = res.tabla["orden_eliminacion"]
            self.assertTrue(order[["malo1", "malo2", "malo3"]].max() < order[["bueno1", "bueno2"]].min())

    def test_equal_models_are_all_kept(self) -> None:
        kept = []
        for seed in range(5):
            losses = _losses({f"m{i}": 0.0 for i in range(4)}, seed=seed)
            kept.append(len(ev.model_confidence_set(losses, n_boot=400, seed=seed).supervivientes))
        self.assertGreaterEqual(np.mean(kept), 3.5)  # aprox.: casi nunca elimina

    def test_identical_losses_never_eliminate(self) -> None:
        col = np.random.default_rng(0).normal(size=50)
        losses = pd.DataFrame({"a": col, "b": col, "c": col})
        res = ev.model_confidence_set(losses, n_boot=200)
        self.assertEqual(len(res.supervivientes), 3)

    def test_reproducible_with_seed(self) -> None:
        losses = _losses(GOOD_BAD)
        a = ev.model_confidence_set(losses, n_boot=300, seed=7).tabla
        b = ev.model_confidence_set(losses, n_boot=300, seed=7).tabla
        pd.testing.assert_frame_equal(a, b)

    def test_block_bootstrap_works(self) -> None:
        for metodo in ("circular", "estacionario"):
            res = ev.model_confidence_set(_losses(GOOD_BAD, n=300), block_len=10, metodo=metodo,
                                          n_boot=400, seed=2)
            self.assertEqual(set(res.supervivientes), {"bueno1", "bueno2"}, metodo)
            self.assertEqual(res.block_len, 10)

    def test_p_mcs_is_monotone_in_elimination_order(self) -> None:
        tabla = ev.model_confidence_set(_losses(GOOD_BAD), n_boot=300).tabla
        p = tabla.sort_values("orden_eliminacion")["p_mcs"].to_numpy()
        self.assertTrue((np.diff(p) >= 0).all())

    def test_nan_rows_are_dropped(self) -> None:
        losses = _losses(GOOD_BAD)
        losses.iloc[0, 2] = np.nan
        res = ev.model_confidence_set(losses, n_boot=100)
        self.assertEqual(res.n_obs, len(losses) - 1)
        self.assertEqual(res.info["filas_descartadas_nan"], 1)


class PreliminaryLossTests(unittest.TestCase):
    def test_calm_day_rate_from_detection_columns(self) -> None:
        frame = pd.DataFrame({"det_marked_rate": [0.3, 0.0, 1.0], "det_precision": [0.5, np.nan, 0.2],
                              "det_base_rate": [0.2, 0.2, 0.2]})
        rate = mcs.tasa_dias_calma_marcados(frame)
        # 0.3 marcados, la mitad fuera → 0.15 / 0.8 de días de calma
        np.testing.assert_allclose(rate, [0.15 / 0.8, 0.0, 1.0])


@pytest.mark.resultados
class VersionedMetricsTests(unittest.TestCase):
    def test_matrix_from_versioned_csv(self) -> None:
        for track, n_crisis in (("A", 17), ("B", 9)):  # README: 17 y 9 crisis evaluadas OOS
            losses = mcs.perdidas_por_crisis(track)
            self.assertEqual(losses.shape, (n_crisis, 12), track)
            self.assertEqual(sorted(losses.columns), [f"D{i:02d}" for i in range(1, 13)])
            self.assertFalse(losses.isna().any().any())
            self.assertTrue(losses.attrs["preliminar"])
            # λ = 0 es solo la cobertura perdida
            zero = mcs.perdidas_por_crisis(track, lam=0.0)
            self.assertTrue(((zero >= 0) & (zero <= 1)).all().all())
            fa = pd.Series(losses.attrs["fa_por_detector"])
            pd.testing.assert_frame_equal(losses - zero, zero * 0 + fa.reindex(zero.columns).to_numpy(),
                                          check_names=False)

    def test_baselines_enter_and_uninformative_loss_is_one(self) -> None:
        losses = mcs.perdidas_por_crisis("B", incluir_baselines=True)
        for name in ("SIEMPRE_CALMA", "SIEMPRE_CRISIS", "AZAR_IID", "AZAR_PERSISTENTE"):
            self.assertIn(name, losses.columns)
        # con λ = 1 y FA de días de calma, las salidas constantes pierden exactamente 1
        np.testing.assert_allclose(losses["SIEMPRE_CALMA"], 1.0)
        np.testing.assert_allclose(losses["SIEMPRE_CRISIS"], 1.0)
        res = ev.model_confidence_set(losses, n_boot=300, seed=0)
        self.assertNotIn("SIEMPRE_CALMA", res.supervivientes)


if __name__ == "__main__":
    unittest.main()
