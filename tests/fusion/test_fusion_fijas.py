"""Tests de las fusiones fijas (``regimenes.fusion.fijas``, ADR-011 §3, §6.1 y §7.1).

Causalidad por prefijo de las tres familias (incluida «confirmación tras aviso»), confirmador
binario, votación (k = n es AND, k = 1 es OR), identificadores, criterio y veredicto, con datos de
juguete. El recuento del §3.4 lee los paneles locales (marcadores ``datos`` y ``resultados``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimenes.fusion import assert_prefix_causal, fijas, vara

FRACCIONES = (0.3, 0.55, 0.8)


def _juguete(n: int = 400, k: int = 4, seed: int = 7):
    """Paneles con rachas (persistencia) y p_crisis que NO coincide con la señal binaria."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    pan, sen = {}, {}
    for j in range(k):
        cambios = rng.random(n) < 0.04
        s = (np.cumsum(cambios) % 2).astype(bool)
        d = f"D{j + 1:02d}"
        ruido = rng.uniform(-0.4, 0.4, n)
        pan[d] = pd.DataFrame({"state": s.astype(int), "p_crisis": np.clip(s * 0.6 + 0.2 + ruido, 0, 1),
                               "fold": 0}, index=idx)
        sen[d] = pd.Series(s, index=idx)
    return pan, pd.DataFrame(sen), {d: 2 for d in pan}


def _corta(pan: dict, stop: int) -> dict:
    return {d: p.iloc[:stop] for d, p in pan.items()}


def test_prefijo_familia_a_con_tras_aviso() -> None:
    pan, sen, ns = _juguete()
    cols = ["actionable", "tras_aviso", "confirmation_active", "warning_window", "decision_code"]
    for x, y in [("D01", "D02"), ("D03", "D01"), ("D02", "D04")]:
        completa = fijas.fusionar_pareja(pan[x], pan[y], sen[y], ns[x])
        assert completa["tras_aviso"].any()                 # el juguete ejercita la sensibilidad
        for f in FRACCIONES:
            stop = int(len(sen) * f)
            corta = fijas.fusionar_pareja(pan[x].iloc[:stop], pan[y].iloc[:stop], sen[y].iloc[:stop], ns[x])
            pd.testing.assert_frame_equal(completa.iloc[:stop][cols], corta[cols])
        conf = fijas.panel_confirmador(pan[y], sen[y])
        assert assert_prefix_causal(pan[x], conf, fijas.config_pareja(ns[x]), FRACCIONES)


def test_prefijo_familias_b_y_c() -> None:
    _, sen, _ = _juguete()
    orden = list(sen.columns)
    b, _ = fijas.familia_b(sen, orden)
    c = fijas.familia_c(sen)
    for f in FRACCIONES:
        stop = int(len(sen) * f)
        bc, _ = fijas.familia_b(sen.iloc[:stop], orden)
        pd.testing.assert_frame_equal(b.iloc[:stop], bc)
        pd.testing.assert_frame_equal(c.iloc[:stop], fijas.familia_c(sen.iloc[:stop]))


def test_confirmador_binario_es_persistencia_3() -> None:
    pan, sen, ns = _juguete()
    # p_crisis ≥ 0,5 discrepa de la binaria: el confirmador debe usar la binaria
    assert (pan["D02"]["p_crisis"].ge(0.5) != sen["D02"]).any()
    f = fijas.fusionar_pareja(pan["D01"], pan["D02"], sen["D02"], ns["D01"])
    esperado = sen["D02"].astype(int).rolling(3, min_periods=3).sum().eq(3)
    assert f["confirmation_active"].equals(esperado.rename("confirmation_active"))
    assert fijas.persistencia(sen["D02"], 3).equals(esperado)
    # la alerta es la binaria de X (estado n_states − 1, sin persistencia)
    assert (f["alert_active"] == sen["D01"]).all()
    # accionable = vigilancia ∪ confirmado
    assert (f["actionable"] == (f["warning_window"] | f["confirmation_active"])).all()


def test_votacion_and_or_y_mayoria() -> None:
    _, sen, _ = _juguete(k=5)
    n = sen.shape[1]
    assert fijas.votacion(sen, n).equals(sen.all(axis=1))
    assert fijas.votacion(sen, 1).equals(sen.any(axis=1))
    assert fijas.k_mayoria(15) == 8 and fijas.k_mayoria(17) == 9
    # con n impar, «al menos ⌈n/2⌉» = «media de las binarias ≥ 0,50» (§3.2)
    assert fijas.votacion(sen, fijas.k_mayoria(n)).equals(sen.astype(float).mean(axis=1).ge(0.5))
    b, miembros = fijas.familia_b(sen, ["D05", "D03", "D01", "D02", "D04"])
    assert miembros["VOTO_2de3"] == ["D05", "D03", "D01"]
    assert b["VOTO_2de3"].equals(sen[["D05", "D03", "D01"]].sum(axis=1).ge(2))
    assert b["VOTO_3de5"].equals(sen.sum(axis=1).ge(3))


def test_identificadores_y_recuento_de_juguete() -> None:
    pan, sen, ns = _juguete(k=4)
    sen = sen.rename(columns={"D04": "D01-P"})
    pan["D01-P"], ns["D01-P"] = pan.pop("D04"), ns.pop("D04")
    a, tras, conf = fijas.familia_a(pan, sen, ns)
    c = fijas.familia_c(sen)
    assert a.shape[1] == tras.shape[1] == 4 * 3 and conf.shape[1] == 4
    assert c.shape[1] == 2 * 6
    assert "PAR_D01-P>D02" in a and "TRAS_D02>D01-P" in tras and "CONF_D01-P" in conf
    # X < Y como cadenas: "D01" < "D01-P" < "D02"
    assert "OR_D01+D01-P" in c and "AND_D01-P+D02" in c and "OR_D02+D01-P" not in c
    assert (c["OR_D01+D02"] == (sen["D01"] | sen["D02"])).all()
    assert (c["AND_D01+D02"] == (sen["D01"] & sen["D02"])).all()


def _ev_juguete(score_f_oos: float, score_f_h2: float, nivel_f_h2: int = 0) -> pd.DataFrame:
    filas = []
    for tramo, sf, nf, si, sih1 in (("oos", score_f_oos, 0, 0.6, 0.5), ("H1", 0.9, 0, 0.5, 0.7),
                                    ("H2", score_f_h2, nivel_f_h2, 0.55, 0.65)):
        filas += [
            {"tramo": tramo, "grupo": "principal", "familia": "c", "id": "OR_D01+D02", "puesto": 1,
             "nivel": nf, "score_deteccion": sf},
            {"tramo": tramo, "grupo": "individual", "familia": "individual", "id": "D01",
             "puesto": 1 if tramo == "oos" else 2, "nivel": 0, "score_deteccion": si},
            {"tramo": tramo, "grupo": "individual", "familia": "individual", "id": "D02",
             "puesto": 2 if tramo == "oos" else 1, "nivel": 0, "score_deteccion": sih1},
        ]
    return pd.DataFrame(filas).assign(pista="A")


def test_criterio_estricto_y_veredicto() -> None:
    elec = {"pista": "A", "f_star": "OR_D01+D02", "i_star": "D01", "i_star_h1": "D02"}
    mcs_dentro = pd.DataFrame({"incluido": [True], "duplicado_de": [None]}, index=["OR_D01+D02"])
    ok = fijas.criterio_pista(_ev_juguete(0.61, 0.66), elec, mcs_dentro)
    assert ok["cumple_i"] and ok["cumple_ii"] and ok["cumple_iii"] and ok["cumple_pista"]
    # empate exacto en (i) y en (iii) no cumple; nivel ≠ 0 en H2 tampoco
    empate = fijas.criterio_pista(_ev_juguete(0.6, 0.65), elec, mcs_dentro)
    assert not empate["cumple_i"] and not empate["cumple_iii"]
    assert not fijas.criterio_pista(_ev_juguete(0.61, 0.9, nivel_f_h2=2), elec, mcs_dentro)["cumple_iii"]
    sin_f = fijas.criterio_pista(_ev_juguete(0.61, 0.66), {**elec, "f_star": None}, None)
    assert not any(sin_f[k] for k in ("cumple_i", "cumple_ii", "cumple_iii", "cumple_pista"))

    crit = pd.DataFrame({"pista": ["A", "B"], "cumple_pista": [True, True]})
    assert fijas.veredicto(crit) == "aporta"
    assert fijas.veredicto(crit.assign(cumple_pista=[True, False])).startswith("indicio")
    assert fijas.veredicto(crit.assign(cumple_pista=False)) == fijas.NO_APORTA
    assert fijas.veredicto(crit.iloc[:1]).startswith("pendiente")


def test_eleccion_usa_solo_h1() -> None:
    """F* e I*_H1 salen del orden de H1 (§5.2); I* del de todo el OOS."""
    ev = _ev_juguete(0.61, 0.66)
    extra = ev[ev["id"] == "OR_D01+D02"].assign(id="AND_D01+D02")
    # AND es la primera en todo el OOS y en H2, pero tiene nivel 1 en H1: no puede ser F*
    extra.loc[extra["tramo"] == "H1", "nivel"] = 1
    extra.loc[extra["tramo"] != "H1", "score_deteccion"] = 0.99
    ev = pd.concat([ev, extra], ignore_index=True)
    ev.loc[ev["grupo"] == "principal", "puesto"] = ev[ev["grupo"] == "principal"].groupby("tramo")[
        "id"].transform(lambda s: [2, 1] if s.name != "H1" else [1, 2])
    cat = pd.DataFrame({"id": ["OR_D01+D02", "AND_D01+D02", "D01", "D02"],
                        "grupo": ["principal", "principal", "individual", "individual"],
                        "miembros": ["D01|D02", "D01|D02", "D01", "D02"]})
    e = fijas.eleccion(ev, cat, "A")
    assert e["f_star"] == "OR_D01+D02" and e["f_star_familia"] == "c"
    assert e["i_star"] == "D01" and e["i_star_h1"] == "D02" and e["n_nivel0_H1"] == 1


def test_f_star_es_la_primera_de_nivel_0() -> None:
    t = pd.DataFrame({"puesto": [1, 2, 3], "nivel": [1, 0, 0]}, index=["X", "Y", "Z"])
    assert fijas.primera_nivel0(t) == "Y"
    assert fijas.primera_nivel0(t.assign(nivel=2)) is None


def _hay_paneles() -> bool:
    return all((vara.PANELES_V2 / f"{p}_D01.parquet").exists()
               and (vara.PANELES_POOL / f"{p}_D15-U.parquet").exists() for p in vara.PISTAS)


@pytest.mark.datos
@pytest.mark.resultados
@pytest.mark.skipif(not _hay_paneles(), reason="faltan los paneles OOS locales")
@pytest.mark.parametrize("pista,n,n_sesiones,par,k_mayoria", [("A", 15, 14133, 210, 8),
                                                             ("B", 17, 4059, 272, 9)])
def test_recuento_del_adr(pista, n, n_sesiones, par, k_mayoria) -> None:
    """§3.4: 423 fusiones principales en A (210 + 3 + 210) y 547 en B (272 + 3 + 272), con
    tantas de «tras aviso» como parejas ordenadas."""
    todas, cat = fijas.construir(pista)
    r = fijas.recuento(cat)
    assert r[("principal", "a")] == par and r[("principal", "b")] == 3 and r[("principal", "c")] == par
    assert int((cat["grupo"] == "principal").sum()) == {"A": 423, "B": 547}[pista]
    assert r[("tras_aviso", "a")] == par and r[("confirmado", "a")] == n and r[("individual", "individual")] == n
    assert todas.shape == (n_sesiones, 2 * par + 3 + par + n + n) and todas.dtypes.eq(bool).all()
    assert cat.loc[cat["id"] == "VOTO_mayoria", "k"].iloc[0] == k_mayoria
