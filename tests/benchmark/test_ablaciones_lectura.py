"""Lectura de las ablaciones del menú T (``regimenes.benchmark.ablaciones_lectura``).

CSV de juguete escritos en una carpeta temporal con el esquema pactado; no lee
``results/`` ni ejecuta detectores.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from regimenes.benchmark import ablaciones_lectura as al

DET = al.DETECTORES
AZAR = [f"AZAR-{i:03d}" for i in range(al.N_AZAR)]


def _tipo(variante: str) -> str:
    if variante.startswith("REF"):
        return "ref"
    if variante.startswith("SIN"):
        return "sin_bloque"
    return "media" if variante == "MEDIA" else "azar"


def _escribe(carpeta, pista, base, variante, score, bloque=None):
    fila = {c: np.nan for c in al.COLUMNAS}
    fila.update(base=base, pista=pista, variante=variante, tipo=_tipo(variante), bloque_quitado=bloque,
                semilla=int(variante[-3:]) if variante.startswith("AZAR") else np.nan, step=252,
                score_deteccion=score, det_precision=score, det_event_recall=score,
                det_n_detectados=1, det_n_eventos=2, n_oos=100, elapsed_seconds=1.0)
    pd.DataFrame([fila]).to_csv(carpeta / f"{pista}_{base}_{variante}.csv", index=False)


VARIANTES = {}


def _variantes(carpeta, pista, nombres):
    """Variantes esperadas en el test (el diseño real sale de bloques.yaml)."""
    VARIANTES[pista] = list(nombres)


def _cargar(carpeta):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return al.cargar(carpeta, pistas=("A",), variantes=VARIANTES)


def _azar(carpeta, medianas, quitar=()):
    """Una variante AZAR por mediana: los 6 detectores con ese mismo score."""
    for nombre, m in zip(AZAR, medianas):
        for d in DET:
            if (nombre, d) not in quitar and nombre not in quitar:
                _escribe(carpeta, "A", d, nombre, m)


def _ref_t(carpeta, scores):
    for d, s in zip(DET, scores):
        _escribe(carpeta, "A", d, "REF-T", s)


# --------------------------------------------------------------------------- #
def test_cargar_valida_esquema_y_avisa_de_faltantes(tmp_path):
    _variantes(tmp_path, "A", ["REF-N", "REF-T"])
    for d in DET:
        _escribe(tmp_path, "A", d, "REF-T", 0.5)
    with pytest.warns(UserWarning, match="faltan 6 trabajos"):
        tabla = al.cargar(tmp_path, pistas=("A",), variantes=VARIANTES)
    assert len(tabla) == 6
    assert tabla.attrs["faltan"] == [f"A_{d}_REF-N" for d in DET]

    pd.read_csv(tmp_path / "A_D02_REF-T.csv").drop(columns="score_deteccion").to_csv(
        tmp_path / "A_D02_REF-T.csv", index=False)
    with pytest.raises(ValueError, match="faltan columnas"):
        _cargar(tmp_path)


def test_cargar_cuenta_frente_al_diseno_y_marca_ajenas(tmp_path):
    (tmp_path / "variantes_A.json").write_text(json.dumps({"variantes": {"REF-T": {}, "RARA": {}}}),
                                               encoding="utf-8")
    _escribe(tmp_path, "A", "D02", "REF-T", 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tabla = al.cargar(tmp_path, pistas=("A",))
    diseno = al.variantes_esperadas("A")
    assert len(diseno) == 2 + 7 + 1 + al.N_AZAR and "SIN-T1b" in diseno
    assert len(tabla.attrs["faltan"]) == len(diseno) * len(DET) - 1
    assert tabla.attrs["fuera_de_diseno"] == ["A_RARA"]


def test_cargar_rechaza_nombre_incoherente(tmp_path):
    _escribe(tmp_path, "A", "D02", "REF-T", 0.5)
    (tmp_path / "A_D02_REF-T.csv").rename(tmp_path / "A_D03_REF-T.csv")
    with pytest.raises(ValueError, match="no casa con el nombre"):
        _cargar(tmp_path)


def test_criterio_b_p90_conocido(tmp_path):
    _variantes(tmp_path, "A", ["REF-T", *AZAR])
    _azar(tmp_path, [i / 100 for i in range(100)])          # medianas 0,00 … 0,99
    _ref_t(tmp_path, [0.95] * 6)
    r = al.criterio_b(_cargar(tmp_path)).iloc[0]
    assert r["p90_azar"] == pytest.approx(0.891)             # lineal: entre la 90.ª y la 91.ª
    assert r["mediana_ref_t"] == pytest.approx(0.95)
    assert r["percentil_ref_t"] == pytest.approx(95.5)       # 95 por debajo y 1 empate
    assert r["b_supera_p90"] and r["veredicto_b"] == "cumple" and r["definitivo"]
    assert r["n_azar_completas"] == 100 and r["n_azar_incompletas"] == 0


def test_criterio_b_empate_no_cumple(tmp_path):
    _variantes(tmp_path, "A", ["REF-T", *AZAR])
    _azar(tmp_path, [0.5] * 100)
    _ref_t(tmp_path, [0.4, 0.4, 0.5, 0.5, 0.6, 0.6])        # mediana 0,5 = p90
    r = al.criterio_b(_cargar(tmp_path)).iloc[0]
    assert r["p90_azar"] == pytest.approx(0.5)
    assert r["percentil_ref_t"] == pytest.approx(50.0)
    assert not r["b_supera_p90"] and r["veredicto_b"] == "no cumple"


def test_criterio_b_incompletas_salen_sin_reponer(tmp_path):
    _variantes(tmp_path, "A", ["REF-T", *AZAR])
    # AZAR-099 no está y AZAR-098 tiene solo 5 detectores: 98 completas (≥ 95).
    _azar(tmp_path, [i / 100 for i in range(100)], quitar={"AZAR-099", ("AZAR-098", "D10")})
    _ref_t(tmp_path, [0.89] * 6)
    tabla = _cargar(tmp_path)
    az = al.medianas_azar(tabla, "A")
    assert az.set_index("variante").loc["AZAR-098", "n_detectores"] == 5
    assert np.isnan(az.set_index("variante").loc["AZAR-098", "mediana"])
    r = al.criterio_b(tabla).iloc[0]
    assert (r["n_azar_completas"], r["n_azar_incompletas"], r["n_azar_ausentes"]) == (98, 1, 1)
    assert r["azar_incompletas"] == "AZAR-098"
    assert r["p90_azar"] == pytest.approx(np.quantile(np.arange(98) / 100, 0.9))   # solo las 98
    assert r["b_supera_p90"] and r["veredicto_b"] == "cumple" and r["definitivo"]
    # Las cotas son sensibilidad: aquí no coinciden, pero no cambian el veredicto.
    assert r["p90_cota_baja"] < 0.89 < r["p90_cota_alta"]


@pytest.mark.parametrize("n_completas, evaluable", [(94, False), (95, True)])
def test_criterio_b_minimo_95_semillas(tmp_path, n_completas, evaluable):
    _variantes(tmp_path, "A", ["REF-T", *AZAR])
    _azar(tmp_path, [0.1] * n_completas)                       # el resto de semillas no está
    _ref_t(tmp_path, [0.9] * 6)
    r = al.criterio_b(_cargar(tmp_path)).iloc[0]
    assert r["n_azar_completas"] == n_completas and r["evaluable"] == evaluable
    assert r["definitivo"] == evaluable and r["b_supera_p90"] == evaluable
    assert r["veredicto_b"] == ("cumple" if evaluable else "no evaluable (cuenta como no cumplido)")


def test_criterio_b_sin_ref_t_completo(tmp_path):
    _variantes(tmp_path, "A", ["REF-T", *AZAR])
    _azar(tmp_path, [0.1] * 100)
    for d in DET[:5]:
        _escribe(tmp_path, "A", d, "REF-T", 0.9)
    r = al.criterio_b(_cargar(tmp_path)).iloc[0]
    assert r["n_ref_t"] == 5 and np.isnan(r["mediana_ref_t"])
    assert r["veredicto_b"] == "no evaluable (cuenta como no cumplido)"
    assert not r["b_supera_p90"] and not r["definitivo"]


def test_criterio_b_sin_ref_t(tmp_path):
    _variantes(tmp_path, "A", ["REF-T", *AZAR])
    _azar(tmp_path, [0.1] * 100)
    r = al.criterio_b(_cargar(tmp_path)).iloc[0]
    assert r["n_ref_t"] == 0 and not r["b_supera_p90"]
    assert r["veredicto_b"] == "no evaluable (cuenta como no cumplido)"


def test_tabla_sin_bloque_ordena_y_marca(tmp_path):
    _variantes(tmp_path, "A", ["REF-T", "SIN-T1", "SIN-T3"])
    _ref_t(tmp_path, [0.5] * 6)
    for i, d in enumerate(DET):
        _escribe(tmp_path, "A", d, "SIN-T1", 0.3, bloque="T1_renta_variable")       # quitarlo empeora
        _escribe(tmp_path, "A", d, "SIN-T3", 0.6 if i < 4 else 0.4, bloque="T3_velocidad_tipos")
    t = al.tabla_sin_bloque(_cargar(tmp_path))
    assert list(t["bloque"]) == ["T3", "T1"]                   # primero el que más estorba
    t = t.set_index("bloque")
    assert t.loc["T3", "efecto"] == "estorba" and t.loc["T3", "n_mejoran"] == 4
    assert t.loc["T3", "mediana_delta"] == pytest.approx(0.1)
    assert t.loc["T1", "efecto"] == "ayuda" and t.loc["T1", "delta_D02"] == pytest.approx(-0.2)
    assert t.loc["T1", "delta_mediana"] == pytest.approx(-0.2)


def test_tabla_media(tmp_path):
    _variantes(tmp_path, "A", ["REF-T", "MEDIA"])
    _ref_t(tmp_path, [0.5] * 6)
    for i, d in enumerate(DET):
        _escribe(tmp_path, "A", d, "MEDIA", 0.6 if i < 2 else 0.5)
    t = al.tabla_media(_cargar(tmp_path)).set_index("base")
    assert t.loc["D02", "delta"] == pytest.approx(0.1) and t.loc["D02", "media_mejora"]
    assert t.loc["D10", "media_mejora"] is False
    assert t.loc["mediana", "media_mejora"] == "2 de 6"


def test_validez_atajo_signos(tmp_path):
    _variantes(tmp_path, "A", ["REF-N", "REF-T"])
    for i, d in enumerate(DET):
        _escribe(tmp_path, "A", d, "REF-N", 0.5)
        _escribe(tmp_path, "A", d, "REF-T", 0.6 if i % 2 == 0 else 0.4)   # D02 +, D03 −, …
    oficial = pd.DataFrame({"pista": "A", "base": list(DET), "score_deteccion_N": 0.5,
                            "score_deteccion_T": [0.7, 0.3, 0.7, 0.3, 0.3, 0.3]})
    v = al.validez_atajo(_cargar(tmp_path), oficial).set_index("base")
    assert v.loc["D02", "coincide"] and v.loc["D03", "coincide"]
    assert not v.loc["D09", "coincide"]                        # anual +, oficial −
    assert v.attrs["coinciden"] == "5 de 6"


def test_percentil_rango_medio():
    assert al.percentil_rango_medio(2.0, np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(37.5)
    assert np.isnan(al.percentil_rango_medio(np.nan, np.array([1.0])))


def test_consolidado_trabajos_equivale_a_los_csv_sueltos(tmp_path):
    """Un clon limpio solo tiene ``trabajos.csv``: se lee igual que los CSV sueltos."""
    sueltos, clon = tmp_path / "sueltos", tmp_path / "clon"
    sueltos.mkdir(), clon.mkdir()
    _variantes(sueltos, "A", ["REF-T", *AZAR[:3]])
    _ref_t(sueltos, [0.5, 0.4, 0.3, 0.2, 0.6, 0.7])
    _azar(sueltos, [0.1, 0.2, 0.3])
    consolidado = al._csv(sueltos / al.TRABAJOS, al._tabla_trabajos(sueltos))
    (clon / al.TRABAJOS).write_bytes(consolidado.read_bytes())
    a, b = _cargar(sueltos), _cargar(clon)
    pd.testing.assert_frame_equal(a, b)
    assert len(b) == 6 * 4 and b.attrs["faltan"] == a.attrs["faltan"]
    # Si hay CSV sueltos mandan ellos (ejecución local en curso), no el consolidado.
    _escribe(clon, "A", "D02", "REF-T", 0.99)
    assert _cargar(clon)["score_deteccion"].tolist() == [0.99]


def test_consolidado_valida_esquema(tmp_path):
    pd.DataFrame({"pista": ["A"]}).to_csv(tmp_path / al.TRABAJOS, index=False)
    with pytest.raises(ValueError, match="faltan columnas"):
        _cargar(tmp_path)
