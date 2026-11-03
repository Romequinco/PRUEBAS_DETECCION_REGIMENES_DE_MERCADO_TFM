"""Tests de la fusión online (ADR-011 §7.1): fuga de etiquetas, arranque, η = 0, entrada binaria y prefijo."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from regimenes.fusion.online import (
    PRINCIPAL,
    _orden_eta,
    Variante,
    calendario,
    combinar,
    fusion_online,
    historia_perdidas,
    perdida_en_tau,
    pesos_desde_perdidas,
    veredicto,
    voto_mayoria,
)


def _datos(n_det: int = 5, seed: int = 0):
    """Dos años de sesiones, crisis sintéticas y detectores con distinta calidad."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2001-01-01", "2002-12-31")
    crisis = {
        "c1": ("2001-03-05", "2001-03-30"),
        "c2": ("2001-09-03", "2001-10-19"),
        "c3": ("2002-06-03", "2002-07-31"),
    }
    trampas = [("2002-02-04", "2002-02-15")]
    verdad = np.zeros(len(idx), dtype=bool)
    for a, b in crisis.values():
        verdad |= (idx >= a) & (idx <= b)
    cols = [f"D{i:02d}" for i in range(1, n_det + 1)]
    ruido = np.linspace(0.05, 0.45, n_det)   # D01 bueno, el último casi azar
    flags = np.column_stack([verdad ^ (rng.random(len(idx)) < r) for r in ruido])
    senales = pd.DataFrame(flags, index=idx, columns=cols)
    probs = pd.DataFrame(np.clip(flags * 0.6 + rng.random(flags.shape) * 0.4, 0, 1),
                         index=idx, columns=cols)
    return probs, senales, crisis, trampas


class TestCalendario(unittest.TestCase):
    def test_tau_ultima_sesion_y_vigencia_siguiente(self) -> None:
        probs, *_ = _datos()
        cal = calendario(probs.index)
        self.assertEqual(len(cal), 24)
        self.assertEqual(cal["tau"].iloc[0], pd.Timestamp("2001-01-31"))
        self.assertEqual(cal["vigencia"].iloc[0], pd.Timestamp("2001-02-01"))
        self.assertTrue(pd.isna(cal["vigencia"].iloc[-1]))


class TestFugaDeEtiquetas(unittest.TestCase):
    def test_crisis_sin_terminar_en_tau_no_cambia_los_pesos(self) -> None:
        probs, senales, crisis, trampas = _datos()
        cal = calendario(senales.index)
        pos = int(cal.loc[pd.Period("2001-09"), "pos_tau"])   # τ = 2001-09-28, c2 sin suelo
        base, _ = perdida_en_tau(senales, pos, crisis, trampas)
        sin_c2 = {k: v for k, v in crisis.items() if k != "c2"}
        otra_c2 = dict(crisis, c2=("2001-09-03", "2001-12-31"))
        for alternativa in (sin_c2, otra_c2):
            L, _ = perdida_en_tau(senales, pos, alternativa, trampas)
            pd.testing.assert_series_equal(base, L)
            pd.testing.assert_series_equal(pesos_desde_perdidas(base, 10.0),
                                           pesos_desde_perdidas(L, 10.0))
        # Y el futuro de las señales tampoco entra.
        futuro = senales.copy()
        futuro.iloc[pos + 1:] = ~futuro.iloc[pos + 1:]
        L, _ = perdida_en_tau(futuro, pos, crisis, trampas)
        pd.testing.assert_series_equal(base, L)

    def test_perdida_es_uno_menos_youden_equiponderado(self) -> None:
        _, senales, crisis, trampas = _datos()
        pos = len(senales) - 1
        L, info = perdida_en_tau(senales, pos, crisis, trampas)
        idx = senales.index
        dentro = np.zeros(len(idx), dtype=bool)
        fallo = 0.0
        for a, b in crisis.values():
            m = (idx >= a) & (idx <= b)
            dentro |= m
            fallo += 1 - senales.loc[m].mean()
        esperado = fallo / len(crisis) + senales.loc[~dentro].mean()
        pd.testing.assert_series_equal(L, esperado.astype(float), check_names=False)
        self.assertEqual(info["K"], 3)


class TestArranque(unittest.TestCase):
    def test_pesos_iguales_hasta_la_primera_crisis_terminada(self) -> None:
        probs, senales, crisis, trampas = _datos()
        res = fusion_online(probs, senales, crisis, trampas, eta=10.0)
        W, meta = res["pesos"], res["meta"]
        primero = meta.index[meta["aprendido"]][0]
        self.assertEqual(primero, pd.Timestamp("2001-03-30"))          # suelo de c1
        n = W.shape[1]
        self.assertTrue(np.allclose(W.loc[W.index < primero].to_numpy(), 1 / n))
        self.assertFalse(np.allclose(W.loc[primero].to_numpy(), 1 / n))
        # p̄ antes de la vigencia de esos pesos = media simple.
        vig = meta.loc[primero, "vigencia"]
        antes = probs.index < vig
        np.testing.assert_allclose(res["pbar"][antes], probs.loc[antes].mean(axis=1))
        self.assertFalse(np.allclose(res["pbar"][~antes], probs.loc[~antes].mean(axis=1)))

    def test_min_dias_retrasa_el_arranque(self) -> None:
        probs, senales, crisis, trampas = _datos()
        res = fusion_online(probs, senales, crisis, trampas, variante=Variante("m", min_dias=21))
        primero = res["meta"].index[res["meta"]["aprendido"]][0]
        # c1 tiene 20 días (< 21); c2 termina el 2001-10-19, así que el primer τ es el de octubre.
        self.assertEqual(res["meta"].loc[pd.Timestamp("2001-09-28"), "dias_crisis"], 20)
        self.assertEqual(primero, pd.Timestamp("2001-10-31"))


class TestComparadores(unittest.TestCase):
    def test_eta_cero_es_la_media_simple(self) -> None:
        probs, senales, crisis, trampas = _datos()
        res = fusion_online(probs, senales, crisis, trampas, eta=0.0)
        np.testing.assert_allclose(res["pbar"], probs.mean(axis=1))
        pd.testing.assert_series_equal(res["senal"], probs.mean(axis=1) >= 0.5, check_names=False)

    def test_eta_cero_binaria_es_voto_mayoria(self) -> None:
        for n in (5, 7):
            probs, senales, crisis, trampas = _datos(n_det=n, seed=n)
            res = fusion_online(probs, senales, crisis, trampas, eta=0.0,
                                variante=Variante("binaria", binaria=True))
            pd.testing.assert_series_equal(res["senal"], voto_mayoria(senales), check_names=False)
            # Y con la regla de las fijas que construye VOTO_mayoria (§3.2).
            from regimenes.fusion import fijas

            pd.testing.assert_series_equal(res["senal"], fijas.votacion(senales, fijas.k_mayoria(n)),
                                           check_names=False)

    def test_eta_infinito_sigue_al_lider_con_empates(self) -> None:
        L = pd.Series({"a": 0.3, "b": 0.3, "c": 0.5})
        w = pesos_desde_perdidas(L, math.inf)
        self.assertEqual(w.to_dict(), {"a": 0.5, "b": 0.5, "c": 0.0})

    def test_nan_sale_de_la_suma_y_renormaliza(self) -> None:
        idx = pd.bdate_range("2020-01-01", periods=2)
        P = pd.DataFrame({"a": [0.2, np.nan], "b": [0.8, 0.9]}, index=idx)
        W = pd.DataFrame({"a": [0.75, 0.75], "b": [0.25, 0.25]}, index=idx)
        np.testing.assert_allclose(combinar(P, W), [0.35, 0.9])


class TestPrefijo(unittest.TestCase):
    def test_pbar_calculado_hasta_t_es_el_de_todo_recortado(self) -> None:
        probs, senales, crisis, trampas = _datos()
        variantes = (PRINCIPAL, Variante("r", retraso=63), Variante("e", excluir_sin_terminar=True),
                     Variante("b", binaria=True))
        for var in variantes:
            completo = fusion_online(probs, senales, crisis, trampas, eta=10.0, variante=var)["pbar"]
            for t in ("2001-03-29", "2001-04-02", "2001-10-10", "2002-06-28", "2002-07-01"):
                corto = fusion_online(probs.loc[:t], senales.loc[:t], crisis, trampas,
                                      eta=10.0, variante=var)["pbar"]
                pd.testing.assert_series_equal(corto, completo.loc[:t], check_freq=False)

    def test_historia_reutilizada_da_lo_mismo(self) -> None:
        probs, senales, crisis, trampas = _datos()
        h = historia_perdidas(senales, crisis, trampas)
        a = fusion_online(probs, senales, crisis, trampas, eta=5.0, historia=h)["pbar"]
        b = fusion_online(probs, senales, crisis, trampas, eta=5.0)["pbar"]
        pd.testing.assert_series_equal(a, b)


class TestVeredicto(unittest.TestCase):
    def test_veredicto_y_los_pesos_aprenden(self) -> None:
        f = {"cumple": True, "O_gt_M_oos": True, "O_gt_M_H2": True}
        self.assertEqual(veredicto({"A": f, "B": f})["veredicto"], "aporta")
        self.assertTrue(veredicto({"A": f, "B": f})["los_pesos_aprenden"])
        r = veredicto({"A": f, "B": dict(f, cumple=False, O_gt_M_H2=False)})
        self.assertEqual(r["veredicto"], "indicio")
        self.assertFalse(r["los_pesos_aprenden"])
        no = dict(f, cumple=False)
        self.assertEqual(veredicto({"A": no, "B": no})["veredicto"], "no aporta")
        # con una sola pista no hay veredicto
        self.assertTrue(veredicto({"A": f})["veredicto"].startswith("pendiente"))
        self.assertFalse(veredicto({"A": f})["los_pesos_aprenden"])

    def test_eta_de_h1_desempata_por_el_menor(self) -> None:
        t = pd.DataFrame({"eta": [10.0, 2.0, 5.0, math.inf], "nivel": [0, 0, 0, 1],
                          "score_deteccion": [0.6004, 0.6001, 0.59, 0.99],
                          "switching_rate": [0.01, 0.01, 0.0, 0.0]})
        # 10 y 2 empatan a 3 decimales y en switching: gana el η menor; nivel 1 va al final
        self.assertEqual(list(_orden_eta(t)["eta"]), [2.0, 10.0, 5.0, math.inf])


if __name__ == "__main__":
    unittest.main()
