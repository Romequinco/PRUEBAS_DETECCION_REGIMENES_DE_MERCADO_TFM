"""Ejecución oficial del menú T (``regimenes.benchmark.menus_oficial``) con datos de juguete.

Cubre lo que no depende de los datos reales: índice común con termómetros, esquema de la
fila N tomada de v2, reanudación (el CSV es la marca de terminado), orden LPT y la
comparación celda a celda con v2. La reproducción real de v2 (B/D02) se comprueba con
``--verificar-n`` y, si hay datos locales, con el test marcado ``datos``.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from regimenes import rutas
from regimenes.benchmark import menus_oficial as mo
from regimenes.detectores.menus import variant_specs


def _fila_v2(track: str = "B", detector: str = "D02") -> pd.DataFrame:
    """Fila mínima con el esquema de ``results/benchmark/metrics`` (columnas de cabeza y de
    detección; las del medio no cambian el tratamiento)."""
    return pd.DataFrame([{
        "id": detector, "pista": track, "familia": "f1", "variante_v2": "v", "n_features_input": 3,
        "features_input": "a|b|c", "train_days": 756, "refit_step": 21, "expanding": True,
        "cache_fingerprint": "abc", "elapsed_seconds": 4.2, "detector": "x", "n_states": 2,
        "oos_start": "2010-04-12", "n_oos": 4059, "det_precision": 0.4, "det_event_recall": 0.6,
        "det_n_detectados": 6,
    }])


def _escribir_v2(carpeta: Path, fila: pd.DataFrame, panel: pd.DataFrame | None = None) -> None:
    track, detector = fila.loc[0, "pista"], fila.loc[0, "id"]
    (carpeta / "metrics").mkdir(parents=True, exist_ok=True)
    fila.to_csv(carpeta / "metrics" / f"{track}_{detector}.csv", index=False)
    if panel is not None:
        (carpeta / "panels").mkdir(parents=True, exist_ok=True)
        panel.to_parquet(carpeta / "panels" / f"{track}_{detector}.parquet")


class IndiceTests(unittest.TestCase):
    def _panel(self) -> tuple[pd.DataFrame, list[str]]:
        idx = pd.bdate_range("2000-01-03", periods=900)
        rng = np.random.default_rng(0)
        cols = sorted({c for s in variant_specs("B", "N") for c in s.features})
        panel = pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols)
        panel.iloc[:50, 0] = np.nan                       # calentamiento de N
        panel["T1"] = rng.normal(size=len(idx))
        panel.loc[panel.index[:30], "T1"] = np.nan         # calentamiento de T dentro del de N
        return panel, ["T1"]

    def test_mismo_indice_si_los_termometros_no_tienen_nan_en_el_de_n(self) -> None:
        panel, cols_t = self._panel()
        indice, informe = mo.indice_oficial("B", panel, cols_t)
        self.assertEqual(len(indice), 850)
        self.assertTrue(informe["mismo_indice_que_v2"])
        self.assertEqual(informe["n_oos"], 850 - 756)
        self.assertEqual(informe["n_dias_perdidos_por_termometros"], 0)

    def test_falla_y_no_recorta_si_los_termometros_pierden_dias(self) -> None:
        panel, cols_t = self._panel()
        panel.loc[panel.index[100:103], "T1"] = np.nan
        with self.assertRaisesRegex(ValueError, "3 días"):
            mo.indice_oficial("B", panel, cols_t)


class FilaNV2Tests(unittest.TestCase):
    def test_esquema_de_menus_y_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            _escribir_v2(Path(temp), _fila_v2())
            fila = mo.fila_n_v2("B", "D02", Path(temp))
        self.assertEqual(list(fila.columns[:4]), ["base", "menu", "id", "pista"])
        self.assertEqual(fila.columns[-1], "score_deteccion")
        self.assertFalse(set(mo._SOLO_V2) & set(fila.columns))
        self.assertEqual((fila.loc[0, "base"], fila.loc[0, "menu"]), ("D02", "N"))
        self.assertAlmostEqual(fila.loc[0, "score_deteccion"], 2 * 0.4 * 0.6 / 1.0)

    def test_esquema_coincide_con_el_de_humo(self) -> None:
        """La fila N de v2 tiene exactamente las columnas que escribía la prueba de humo."""
        humo = rutas.RESULTS_DETECTORES / "menus_humo" / "B_D02_N.csv"
        v2 = rutas.RESULTS_BENCHMARK / "metrics" / "B_D02.csv"
        if not (humo.exists() and v2.exists()):
            pytest.skip("faltan results/detectores/menus_humo o results/benchmark/metrics")
        self.assertEqual(list(mo.fila_n_v2("B", "D02").columns), list(pd.read_csv(humo).columns))

    def test_escribir_n_v2_es_reanudable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            v2, salida = Path(temp) / "v2", Path(temp) / "menus"
            _escribir_v2(v2, _fila_v2())
            (ruta,) = mo.escribir_n_v2(["B"], ["D02"], salida, v2)
            ruta.write_text("no se toca", encoding="utf-8")
            mo.escribir_n_v2(["B"], ["D02"], salida, v2)
            self.assertEqual(ruta.read_text(encoding="utf-8"), "no se toca")


class TrabajoTests(unittest.TestCase):
    def _falso_evaluar(self, track, spec, panel, index):
        metrics = pd.DataFrame([{"id": spec.detector_id, "pista": track, "elapsed_seconds": 1.0,
                                 "score_deteccion": 0.5}])
        wf = pd.DataFrame({"state": np.zeros(len(index), dtype=int)}, index=index)
        return metrics, wf

    def test_escribe_panel_y_csv_y_salta_lo_hecho(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            entrada = temp / "B.parquet"
            pd.DataFrame({"x": [1.0, 2.0]}, index=pd.bdate_range("2020-01-01", periods=2)).to_parquet(entrada)
            with mock.patch.object(mo, "evaluar_variante", side_effect=self._falso_evaluar) as ev:
                estado = mo.ejecutar_trabajo("B", "D02", "T", entrada, temp)
                self.assertEqual(estado["estado"], "ok")
                csv = mo.ruta_trabajo(temp, "B", "D02", "T")
                fila = pd.read_csv(csv)
                self.assertEqual(list(fila.columns[:3]), ["base", "menu", "id"])
                self.assertEqual(fila.loc[0, "id"], "D02-T")
                self.assertTrue(csv.with_suffix(".parquet").exists())
                self.assertEqual(mo.ejecutar_trabajo("B", "D02", "T", entrada, temp)["estado"], "ya_estaba")
                self.assertEqual(ev.call_count, 1)

    def test_un_fallo_no_deja_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            entrada = temp / "B.parquet"
            pd.DataFrame({"x": [1.0]}, index=pd.bdate_range("2020-01-01", periods=1)).to_parquet(entrada)
            with mock.patch.object(mo, "evaluar_variante", side_effect=RuntimeError("memoria")):
                estado = mo._trabajo_seguro(("B", "D02", "T", str(entrada), str(temp)))
            self.assertEqual(estado["estado"], "error")
            self.assertFalse(mo.ruta_trabajo(temp, "B", "D02", "T").exists())

    def test_orden_lpt(self) -> None:
        trabajos = [("B", "D02", "T"), ("A", "D04", "T"), ("B", "D04", "T"), ("A", "D99", "T")]
        tiempos = {("B", "D02"): 4.0, ("A", "D04"): 6562.0, ("B", "D04"): 781.0}
        orden = mo.ordenar_lpt(trabajos, tiempos)
        self.assertEqual([d for _, d, _ in orden[:2]], ["D99", "D04"])   # desconocido delante
        self.assertEqual(orden[-1], ("B", "D02", "T"))


class ComparacionV2Tests(unittest.TestCase):
    def test_detecta_igualdad_y_diferencias(self) -> None:
        idx = pd.bdate_range("2020-01-01", periods=5)
        panel = pd.DataFrame({"state": [0, 1, 1, 0, 0]}, index=idx)
        with tempfile.TemporaryDirectory() as temp:
            _escribir_v2(Path(temp), _fila_v2(), panel)
            rerun = mo.fila_n_v2("B", "D02", Path(temp))
            rerun.loc[0, "elapsed_seconds"] = 99.0           # el tiempo no cuenta
            ok = mo.comparar_con_v2("B", "D02", rerun, panel, Path(temp))
            self.assertTrue(ok["reproduce_v2"])
            rerun.loc[0, "det_precision"] = 0.41
            mal = mo.comparar_con_v2("B", "D02", rerun, panel.assign(state=[0, 1, 0, 0, 0]), Path(temp))
            self.assertFalse(mal["reproduce_v2"])
            self.assertEqual([c for c, *_ in mal["columnas_distintas"]],
                             ["det_precision"])
            self.assertFalse(mal["panel_identico"])


@pytest.mark.datos
def test_entrada_oficial_tiene_el_indice_de_v2() -> None:
    """Con datos locales: el índice oficial (con termómetros) da el n_oos de v2 en las dos pistas."""
    if not all((rutas.DATA_PROCESSED / f"pista{p}_diaria.parquet").exists() for p in "AB"):
        pytest.skip("faltan data/processed")
    for track in "AB":
        try:
            panel, indice, informe = mo.panel_oficial(track)
        except FileNotFoundError as exc:
            pytest.skip(str(exc))
        v2 = rutas.RESULTS_BENCHMARK / "metrics" / f"{track}_D02.csv"
        assert informe["mismo_indice_que_v2"]
        assert informe["n_oos"] == int(pd.read_csv(v2).loc[0, "n_oos"])
