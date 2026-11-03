"""Tests ligeros de ``regimenes.fusion.figuras_online`` (notebook 17): paleta, tablas y figuras.

Los de paleta no leen nada. Los de tablas y figuras leen ``results/fusion/online/`` y se saltan si
no existe (en la CI no se ejecuta la fusión online)."""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from regimenes.benchmark import panorama as pn  # noqa: E402
from regimenes.fusion import figuras_online as fo  # noqa: E402
from regimenes.fusion import online as on  # noqa: E402

HAY_RESULTADOS = all((on.directorio_resultados() / f"criterio_{p}.json").is_file() for p in fo.PISTAS) and (
    on.directorio_resultados() / "resumen.json").is_file()


class TestPaleta(unittest.TestCase):
    IDS = ["D01", "D02", "D10", "D10-P", "D10-R", "D14-R", "D05", "D06", "D11", "D15-U"]

    def test_un_color_distinto_por_detector(self):
        c = fo.colores_detectores(self.IDS)
        self.assertEqual(set(c), set(self.IDS))
        self.assertEqual(len(set(c.values())), len(self.IDS))

    def test_detector_solo_en_su_familia_lleva_el_color_de_panorama(self):
        c = fo.colores_detectores(self.IDS)
        self.assertEqual(c["D05"].lower(), pn.COLOR_FAMILIA["F4"].lower())

    def test_orden_agrupa_por_familia(self):
        orden = fo.orden_detectores(self.IDS)
        fams = [pn.familia_de(d) for d in orden]
        self.assertEqual(fams, sorted(fams))

    def test_variante_pool(self):
        self.assertTrue(fo.es_variante_pool("D10-P"))
        self.assertFalse(fo.es_variante_pool("D10"))


@unittest.skipUnless(HAY_RESULTADOS, "faltan results/fusion/online (python -m regimenes.fusion.online ejecutar)")
class TestConResultados(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_tabla_comparadores_coincide_con_el_criterio(self):
        for p in fo.PISTAS:
            t = fo.tabla_comparadores(p)       # lanza AssertionError si no coincide con criterio_{P}.json
            crit = on.leer_criterio_pista(p)
            self.assertEqual(t.loc["I*", "id_oos"], crit["I*"])
            self.assertEqual(t.loc["I*", "id_H2"], crit["I*_H1"])
            self.assertEqual(bool(t.loc["O", "mcs_oos"]), bool(crit["ii"]))

    def test_tabla_criterio_reproduce_el_veredicto(self):
        t = fo.tabla_criterio()
        res = on.leer_resumen()
        for p in fo.PISTAS:
            fila = t.loc[(p, "→ la pista cumple (i)+(ii)+(iii)"), "cumple"]
            self.assertEqual(fila == "✓", bool(res["cumple_por_pista"][p]))

    def test_pesos_suman_uno(self):
        for p in fo.PISTAS:
            S = fo.serie_pesos(p)
            cols = [c for c in S.columns if c != "n_efectivo"]
            self.assertTrue(np.allclose(S[cols].sum(axis=1), 1.0))
            self.assertTrue(S.index.is_monotonic_increasing)

    def test_conclusiones(self):
        c = fo.conclusiones(fo.hechos())
        self.assertTrue(3 <= len(c) <= 5)
        self.assertIn(on.leer_resumen()["veredicto"], c[0])

    def test_figuras_se_construyen(self):
        figs = [fo.figura_resumen(), fo.figura_mecanismo("A"), fo.figura_pesos("B"), fo.figura_eta(),
                fo.figura_sensibilidades(), fo.figura_mcs("oos")]
        for f in figs:
            self.assertGreater(len(f.axes), 0)


if __name__ == "__main__":
    unittest.main()
