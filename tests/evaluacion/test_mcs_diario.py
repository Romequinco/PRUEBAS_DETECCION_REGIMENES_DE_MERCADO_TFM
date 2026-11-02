"""MCS definitivo (ADR-008 §4): pérdidas diarias desde paneles, con datos de juguete."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regimenes import evaluacion as ev
from regimenes.evaluacion import curvas, mcs

INDEX = pd.bdate_range("2000-01-03", periods=2520)  # ~10 años
CRISIS_POS = [300, 700, 1100, 1500, 1900, 2300]
CRISIS_LEN = [40, 20, 60, 30, 80, 50]
WINDOWS = {
    f"c{k}": (str(INDEX[p].date()), str(INDEX[p + n - 1].date()))
    for k, (p, n) in enumerate(zip(CRISIS_POS, CRISIS_LEN))
}
SPEC = {"crisis_windows": {"pista_A": {k: list(v) for k, v in WINDOWS.items()}},
        "false_positive_windows": {}}


def _dentro() -> np.ndarray:
    mask = np.zeros(len(INDEX), dtype=bool)
    for p, n in zip(CRISIS_POS, CRISIS_LEN):
        mask[p:p + n] = True
    return mask


def _senales(seed: int = 0) -> pd.DataFrame:
    """bueno: cubre el 90 % de cada crisis y marca el 2 % de la calma; ruido: azar."""
    rng = np.random.default_rng(seed)
    dentro = _dentro()
    bueno = np.where(dentro, rng.random(len(INDEX)) < 0.9, rng.random(len(INDEX)) < 0.02)
    return pd.DataFrame({
        "bueno": bueno,
        "ruido": rng.random(len(INDEX)) < 0.2,
        "calma": np.zeros(len(INDEX), bool),
        "crisis": np.ones(len(INDEX), bool),
        "perfecto": dentro,
    }, index=INDEX)


class PerdidasDiariasTests(unittest.TestCase):
    def test_mean_is_crisis_miss_plus_false_alarm_rate(self) -> None:
        senales = _senales()
        L = mcs.perdidas_diarias(senales, WINDOWS, lam=1.0)
        dentro = _dentro()
        for col in senales:
            s = senales[col]
            cov = ev.crisis_coverage(s.astype(int), 1, WINDOWS)
            esperado = np.mean([1 - v for v in cov.values()]) + s[~dentro].mean()
            self.assertAlmostEqual(L[col].mean(), esperado, places=12)

    def test_uninformative_detectors_lose_one_and_perfect_zero(self) -> None:
        L = mcs.perdidas_diarias(_senales(), WINDOWS)
        self.assertAlmostEqual(L["calma"].mean(), 1.0, places=12)
        self.assertAlmostEqual(L["crisis"].mean(), 1.0, places=12)
        self.assertAlmostEqual(L["perfecto"].mean(), 0.0, places=12)
        self.assertAlmostEqual(L["ruido"].mean(), 1.0, delta=0.08)  # azar: 1 en esperanza

    def test_every_crisis_weighs_the_same(self) -> None:
        # fallar solo la crisis más corta o solo la más larga cuesta lo mismo: 1/K
        dentro = _dentro()
        corta = pd.Series(dentro.copy(), index=INDEX)
        larga = pd.Series(dentro.copy(), index=INDEX)
        corta.iloc[700:720] = False
        larga.iloc[1900:1980] = False
        L = mcs.perdidas_diarias(pd.DataFrame({"corta": corta, "larga": larga}), WINDOWS)
        self.assertAlmostEqual(L["corta"].mean(), 1 / len(WINDOWS), places=12)
        self.assertAlmostEqual(L["larga"].mean(), 1 / len(WINDOWS), places=12)

    def test_decomposition_and_lambda(self) -> None:
        senales = _senales()
        L0 = mcs.perdidas_diarias(senales, WINDOWS, lam=0.0)
        L2 = mcs.perdidas_diarias(senales, WINDOWS, lam=2.0)
        d = mcs.descomponer_perdida(L2, WINDOWS)
        np.testing.assert_allclose(d["fallo"], L0.mean(), atol=1e-12)
        np.testing.assert_allclose(d["perdida_media"], L2.mean(), atol=1e-12)
        np.testing.assert_allclose(d["falsa_alarma"], 2.0 * d["fpr"], atol=1e-12)

    def test_windows_outside_oos_are_skipped_and_errors(self) -> None:
        fuera = dict(WINDOWS, vieja=("1990-01-01", "1990-03-01"))
        L = mcs.perdidas_diarias(_senales(), fuera)
        self.assertEqual(L.attrs["n_crisis"], len(WINDOWS))
        self.assertEqual(L.attrs["crisis_sin_dias_oos"], ["vieja"])
        solapada = dict(WINDOWS, otra=(WINDOWS["c0"][0], WINDOWS["c0"][1]))
        with self.assertRaises(ValueError):
            mcs.perdidas_diarias(_senales(), solapada)
        with self.assertRaises(ValueError):
            mcs.perdidas_diarias(_senales(), {"todo": (str(INDEX[0].date()), str(INDEX[-1].date()))})

    def test_block_length_is_median_window_length(self) -> None:
        self.assertEqual(mcs.longitud_bloque_crisis(INDEX, WINDOWS), int(np.median(CRISIS_LEN)))


class BootstrapPorTandasTests(unittest.TestCase):
    def test_batched_means_match_the_direct_ones_in_distribution(self) -> None:
        rng = np.random.default_rng(3)
        L = rng.normal(size=(3000, 4))
        directo = mcs._medias_bootstrap(L, 400, block_len=10, metodo="estacionario", seed=0,
                                        memoria=10 ** 9)
        tandas = mcs._medias_bootstrap(L, 400, block_len=10, metodo="estacionario", seed=0,
                                       memoria=10 ** 6)
        self.assertEqual(tandas.shape, (400, 4))
        np.testing.assert_allclose(tandas.mean(axis=0), L.mean(axis=0), atol=0.02)
        np.testing.assert_allclose(tandas.std(axis=0), directo.std(axis=0), rtol=0.25)
        again = mcs._medias_bootstrap(L, 400, block_len=10, metodo="estacionario", seed=0,
                                      memoria=10 ** 6)
        np.testing.assert_array_equal(tandas, again)

    def test_small_problem_keeps_the_original_stream(self) -> None:
        rng = np.random.default_rng(4)
        L = rng.normal(size=(50, 3))
        idx = mcs.bootstrap_indices(50, 100, block_len=3, metodo="circular", seed=7)
        out = mcs._medias_bootstrap(L, 100, block_len=3, metodo="circular", seed=7)
        np.testing.assert_allclose(out, L[idx].mean(axis=1))


class ReplicasRazonTests(unittest.TestCase):
    def test_constant_and_perfect_detectors_have_no_bootstrap_noise(self) -> None:
        R = mcs.replicas_perdida(_senales(), WINDOWS, n_boot=200, block_len=35, seed=0)
        cols = list(_senales().columns)
        self.assertEqual(R.shape, (200, len(cols)))
        np.testing.assert_allclose(R[:, cols.index("calma")], 1.0, atol=1e-12)
        np.testing.assert_allclose(R[:, cols.index("crisis")], 1.0, atol=1e-12)
        np.testing.assert_allclose(R[:, cols.index("perfecto")], 0.0, atol=1e-12)
        # con pesos fijos, la media remuestreada de SIEMPRE_CALMA sí varía (el artefacto)
        L = mcs.perdidas_diarias(_senales(), WINDOWS).to_numpy()
        lin = mcs._medias_bootstrap(L, 200, block_len=35, metodo="estacionario", seed=0)
        self.assertGreater(lin[:, cols.index("calma")].std(), 0.05)

    def test_replicas_center_on_the_point_loss(self) -> None:
        senales = _senales()
        L = mcs.perdidas_diarias(senales, WINDOWS)
        R = mcs.replicas_perdida(senales, WINDOWS, n_boot=400, block_len=1, seed=1)
        np.testing.assert_allclose(R.mean(axis=0), L.mean().to_numpy(), atol=0.02)

    def test_ratio_replicas_match_a_brute_force_resample(self) -> None:
        """Cada réplica = 1 − J recalculada a mano sobre los días sacados (con repetición)."""
        senales = _senales(2)
        n_boot, bl, seed = 30, 35, 5
        R = mcs.replicas_perdida(senales, WINDOWS, n_boot=n_boot, block_len=bl, seed=seed)
        # Mismos índices que usa replicas_perdida (una sola tanda con este tamaño).
        ((_, _, cuentas),) = list(mcs._tandas_recuentos(len(INDEX), n_boot, block_len=bl,
                                                         metodo="estacionario", seed=seed))
        dentro = _dentro()
        S = senales.to_numpy(dtype=float)
        for b in range(n_boot):
            idx = np.repeat(np.arange(len(INDEX)), cuentas[b].astype(int))
            fallos = []
            for p, n in zip(CRISIS_POS, CRISIS_LEN):
                sel = idx[(idx >= p) & (idx < p + n)]
                if sel.size:
                    fallos.append(1 - S[sel].mean(axis=0))
            calma = idx[~dentro[idx]]
            esperado = np.mean(fallos, axis=0) + S[calma].mean(axis=0)
            np.testing.assert_allclose(R[b], esperado, atol=1e-12)

    def test_mcs_engine_matches_arch_with_the_same_bootstrap(self) -> None:
        """Con las MISMAS réplicas que ``arch.bootstrap.MCS``, mismos p-valores (max y R)."""
        arch_mcs = pytest.importorskip("arch.bootstrap").MCS
        rng = np.random.default_rng(11)
        L = pd.DataFrame(rng.normal(size=(300, 6)) + np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.3]),
                         columns=[f"m{i}" for i in range(6)])
        for metodo, est in (("max", "T_max"), ("R", "T_R")):
            a = arch_mcs(L, size=0.1, reps=300, block_size=10, method=metodo,
                         bootstrap="stationary", seed=3)
            a.compute()
            idx = np.asarray(a._bootstrap_indices)
            nuestro = mcs.model_confidence_set(L, estadistico=est, replicas=L.to_numpy()[idx].mean(axis=1))
            p_arch = a.pvalues["Pvalue"].reindex(nuestro.tabla.index)
            np.testing.assert_allclose(nuestro.tabla["p_mcs"], p_arch, atol=1e-12, err_msg=est)

    def test_mcs_accepts_replicas_and_checks_their_shape(self) -> None:
        senales = _senales()
        L, res = mcs.mcs_senales(senales, WINDOWS, n_boot=300, seed=0)
        self.assertEqual(res.info["remuestreo"], "razon")
        self.assertEqual(res.n_boot, 300)
        self.assertIn("perfecto", res.supervivientes)
        for fuera in ("calma", "crisis", "ruido"):
            self.assertNotIn(fuera, res.supervivientes)
        _, lin = mcs.mcs_senales(senales, WINDOWS, n_boot=300, seed=0, remuestreo="lineal")
        self.assertEqual(lin.info["remuestreo"], "lineal")
        with self.assertRaises(ValueError):
            mcs.model_confidence_set(L, replicas=np.zeros((10, 2)))
        with self.assertRaises(ValueError):
            mcs.mcs_senales(senales, WINDOWS, remuestreo="otro")


class SenalesYPanelesTests(unittest.TestCase):
    def _panel(self, flags: np.ndarray, n_states: int = 2) -> pd.DataFrame:
        state = np.where(flags, n_states - 1, 0)
        return pd.DataFrame({"state": state, "p_crisis": flags * 0.9 + 0.05,
                             "fold": 0}, index=INDEX)

    def test_crisis_state_is_the_top_state(self) -> None:
        dentro = _dentro()
        sen = mcs.senales_crisis({"X": self._panel(dentro, n_states=3)})
        np.testing.assert_array_equal(sen["X"].to_numpy(), dentro)
        malo = self._panel(dentro)
        malo["p_crisis"] = 1 - malo["p_crisis"]   # el estado máximo no es el de crisis
        with self.assertRaises(ValueError):
            mcs.senales_crisis({"X": malo})
        fijado = mcs.senales_crisis({"X": malo}, estados_crisis={"X": 0})
        np.testing.assert_array_equal(fijado["X"].to_numpy(), ~dentro)

    def test_detector_that_never_visits_its_crisis_state(self) -> None:
        """Un detector que nunca marca en el OOS no puede leerse como SIEMPRE_CRISIS."""
        nunca = pd.DataFrame({"state": np.zeros(len(INDEX), dtype=int), "p_crisis": 0.01,
                              "fold": 0}, index=INDEX)
        with self.assertRaises(ValueError):            # ambiguo sin n_states
            mcs.senales_crisis({"X": nunca})
        sen = mcs.senales_crisis({"X": nunca}, n_states={"X": 2})
        self.assertFalse(sen["X"].any())
        # Tres estados, el de crisis (2) sin visitar: el máximo observado (1) NO es la crisis.
        tres = self._panel(_dentro())                    # estados 0 y 1
        sen = mcs.senales_crisis({"X": tres}, n_states={"X": 3})
        self.assertFalse(sen["X"].any())
        with self.assertRaises(ValueError):
            mcs.senales_crisis({"X": self._panel(_dentro(), n_states=3)}, n_states={"X": 2})

    def test_n_states_from_metric_csvs(self) -> None:
        senales = _senales()
        with tempfile.TemporaryDirectory() as tmp:
            panels = Path(tmp) / "panels"
            panels.mkdir()
            (Path(tmp) / "metrics").mkdir()
            self._panel(senales["bueno"].to_numpy()).to_parquet(panels / "A_bueno.parquet")
            nunca = self._panel(np.zeros(len(INDEX), bool))
            nunca.to_parquet(panels / "A_nunca.parquet")
            pd.DataFrame([{"id": "bueno", "n_states": 2}]).to_csv(Path(tmp) / "metrics" / "A_bueno.csv",
                                                                  index=False)
            pd.DataFrame([{"id": "nunca", "n_states": 2}]).to_csv(panels / "A_nunca.csv", index=False)
            self.assertEqual(mcs.n_states_metricas("A", panels, ["bueno", "nunca", "otro"]),
                             {"bueno": 2, "nunca": 2})
            L, res = mcs.mcs_diario("A", panels_dir=panels, benchmark=SPEC, n_boot=300, seed=0)
        # «nunca» pierde 1 (siempre calma), no 1 por marcar todo: su FPR es 0.
        d = mcs.descomponer_perdida(L, WINDOWS)
        self.assertAlmostEqual(d.loc["nunca", "fpr"], 0.0)
        self.assertAlmostEqual(d.loc["nunca", "fallo"], 1.0)

    def test_baselines_have_the_requested_rate(self) -> None:
        vol = pd.Series(np.random.default_rng(0).normal(size=len(INDEX)), index=INDEX)
        base = mcs.senales_baselines("A", INDEX, rate=0.2, mean_run=10.0, vol=vol)
        self.assertEqual(list(base.columns), list(mcs.BASELINES_TRIVIALES))
        self.assertAlmostEqual(base["AZAR_IID"].mean(), 0.2, delta=0.03)
        self.assertAlmostEqual(base["AZAR_PERSISTENTE"].mean(), 0.2, delta=0.1)
        self.assertGreater(mcs._racha_media(base["AZAR_PERSISTENTE"].to_numpy()), 4)
        self.assertFalse(base["VOL_P90"].iloc[:252].any())   # burn-in del percentil causal

    def test_end_to_end_from_a_panels_folder(self) -> None:
        senales = _senales()
        with tempfile.TemporaryDirectory() as tmp:
            for k in ("bueno", "ruido"):
                self._panel(senales[k].to_numpy()).to_parquet(Path(tmp) / f"A_{k}.parquet")
            paneles = mcs.cargar_paneles("A", tmp)
            self.assertEqual(sorted(paneles), ["bueno", "ruido"])
            L, res = mcs.mcs_diario("A", panels_dir=tmp, benchmark=SPEC, n_boot=500, seed=0)
        self.assertEqual(res.block_len, int(np.median(CRISIS_LEN)))
        self.assertIn("bueno", res.supervivientes)
        for fuera in ("ruido", "SIEMPRE_CALMA", "SIEMPRE_CRISIS", "AZAR_IID"):
            self.assertNotIn(fuera, res.supervivientes)
        self.assertNotIn("VOL_P90", L.columns)       # sin vol no hay VOL_P90

    def test_panels_with_different_index_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._panel(_dentro()).to_parquet(Path(tmp) / "A_X.parquet")
            self._panel(_dentro()).iloc[1:].to_parquet(Path(tmp) / "A_Y.parquet")
            with self.assertRaises(ValueError):
                mcs.cargar_paneles("A", tmp)


class CurvasDetectoresTests(unittest.TestCase):
    def test_binary_and_native_point(self) -> None:
        senales = _senales()
        rng = np.random.default_rng(1)
        continua = pd.Series(np.where(_dentro(), 2.0, 0.0) + rng.normal(0, 0.5, len(INDEX)),
                             index=INDEX)
        tabla = curvas.curvas_detectores(
            {"binaria": senales["bueno"].astype(float), "continua": continua}, WINDOWS,
            modos=("ex_post",), senales_nativas={"binaria": senales["bueno"]})
        self.assertEqual(len(tabla), 2 * len(curvas.FA_OBJETIVO))
        b = tabla[tabla["id"] == "binaria"]
        self.assertTrue(b["binaria"].all())
        self.assertFalse(tabla.loc[tabla["id"] == "continua", "binaria"].any())
        # la binaria solo tiene su punto de operación o «nada»
        self.assertTrue(set(b["umbral"]) <= {1.0, np.inf})
        alcanzable = b[b["fa_objetivo"] >= b["fa_anio_nativa"]]
        np.testing.assert_allclose(alcanzable["fa_anio"], alcanzable["fa_anio_nativa"])
        self.assertTrue(tabla.loc[tabla["id"] == "continua", "fa_anio_nativa"].isna().all())


if __name__ == "__main__":
    unittest.main()
