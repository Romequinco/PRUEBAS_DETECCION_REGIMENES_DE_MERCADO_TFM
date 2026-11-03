"""Tests de las ablaciones y controles del pool de termómetros (``benchmark.ablaciones``).

Datos de juguete: dos factores persistentes con columnas orientadas a favor y en contra y
una columna que empieza tarde. Nada lee ``data/`` salvo ``configs/bloques.yaml``.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from regimenes.benchmark import ablaciones as ab
from regimenes.features.bloques import cargar_bloques

MIN_OBS = 60


def _panel(n: int = 900, semilla: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    idx = pd.bdate_range("2000-01-03", periods=n)

    def ar() -> np.ndarray:
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.98 * x[t - 1] + rng.normal()
        return x

    f1, f2 = ar(), ar()
    ruido = lambda: rng.normal(scale=0.5, size=n)  # noqa: E731
    p = pd.DataFrame({"a1": f1 + ruido(), "a2": -f1 + ruido(), "a3": f1 + ruido(),
                      "b1": f2 + ruido(), "b2": f2 + ruido(), "b3": -f2 + ruido(),
                      "c1": rng.normal(size=n)}, index=idx)
    p.iloc[:300, p.columns.get_loc("a3")] = np.nan          # empieza tarde
    p.attrs["f1"], p.attrs["f2"] = f1, f2
    return p


BLOQUES = {
    "T1_x": {"columnas": ["a1", "a2", "a3"], "ancla": {"columna": "a1", "signo": 1}},
    "T2_y": {"columnas": ["b1", "b2", "b3"], "ancla": {"columna": "b1", "signo": 1}},
    "T3_z": {"columnas": ["c1"], "ancla": {"columna": "c1", "signo": 1}},
}


# --------------------------------------------------------------------------- #
# Composición
# --------------------------------------------------------------------------- #
def test_reparto_conserva_columnas_y_tamanos_y_es_reproducible() -> None:
    pool = ab.pool_columnas(BLOQUES)
    r = ab.reparto_aleatorio(BLOQUES, np.random.default_rng(5))
    assert sorted(c for cols in r.values() for c in cols) == sorted(pool)
    assert [len(c) for c in r.values()] == [3, 3, 1]
    assert r == ab.reparto_aleatorio(BLOQUES, np.random.default_rng(5))
    distintos = {json.dumps(ab.reparto_aleatorio(BLOQUES, np.random.default_rng(s))) for s in range(10)}
    assert len(distintos) > 1


def test_reparto_real_de_las_dos_pistas() -> None:
    for pista, n_cols, n_bloques in (("A", 35, 7), ("B", 77, 10)):
        bq = cargar_bloques(pista)
        r = ab.reparto_aleatorio(bq, np.random.default_rng(0))
        assert sorted(c for cols in r.values() for c in cols) == sorted(ab.pool_columnas(bq))
        assert len(ab.pool_columnas(bq)) == n_cols and len(r) == n_bloques
        assert [len(c) for c in r.values()] == [len(b["columnas"]) for b in bq.values()]


def test_nombres_y_sin_bloque_quitan_exactamente_una_columna() -> None:
    for pista, n in (("A", 7), ("B", 10)):
        bq = cargar_bloques(pista)
        nombres = ab.nombres_variantes(bq)
        assert len(nombres) == 2 + n + 1 + 100 and len(set(nombres)) == len(nombres)
        sin = [v for v in nombres if ab.tipo_de(v) == "sin_bloque"]
        assert len(sin) == n
        for v in sin:
            cols = ab.columnas_sin(bq, v)
            quitado = ab.bloque_quitado(v, bq)
            assert len(cols) == n - 1 and quitado not in cols
            assert set(bq) - set(cols) == {quitado}
    assert ab.bloque_quitado("SIN-T1b", cargar_bloques("B")) == "T1b_rentabilidad_diaria"
    assert ab.semilla_de("AZAR-042") == 42 and ab.semilla_de("MEDIA") is None


# --------------------------------------------------------------------------- #
# Causalidad y regla de signo
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("corte", [250, 420, 611, 899])
def test_media_es_causal(corte: int) -> None:
    p = _panel()
    completo = ab.termometros_media(p, BLOQUES, min_obs=MIN_OBS)
    t = p.index[corte]
    recortado = ab.termometros_media(p.loc[:t], BLOQUES, min_obs=MIN_OBS)
    pd.testing.assert_frame_equal(recortado, completo.loc[:t], rtol=1e-12, check_freq=False)


@pytest.mark.parametrize("corte", [70, 250, 420, 611, 899])
def test_azar_es_causal(corte: int) -> None:
    p = _panel()
    for semilla in (0, 3):
        completo, _ = ab.termometros_azar(p, BLOQUES, semilla, umbral_var=0.3, min_obs=MIN_OBS)
        t = p.index[corte]
        recortado, _ = ab.termometros_azar(p.loc[:t], BLOQUES, semilla, umbral_var=0.3, min_obs=MIN_OBS)
        pd.testing.assert_frame_equal(recortado, completo.loc[:t], rtol=1e-12, check_freq=False)


def test_media_no_usa_la_pc1() -> None:
    from regimenes.features import bloques as bl

    p = _panel()
    _, diag = bl.construir_termometros(p, BLOQUES, umbral_var=ab.UMBRAL_SOLO_MEDIA, min_obs=MIN_OBS)
    for nombre in ("T1_x", "T2_y"):
        assert set(diag[nombre]["metodo"]) <= {"media", "sin_ancla"}
        cargas = diag[nombre].dropna(subset=["cargas"])["cargas"].iloc[-1]
        assert np.allclose(np.abs(list(cargas.values())), 1 / np.sqrt(len(cargas)))


def test_regla_de_signo_sigue_la_orientacion_del_bloque_original() -> None:
    p = _panel()
    anclas = ab.anclas_originales(BLOQUES)
    f1, f2 = p.attrs["f1"], p.attrs["f2"]
    # a2 = -f1: sola, se orienta con -1 frente a su ancla original a1 y el termómetro sube con f1
    serie, ficha = ab.termometro_azar(p, ["a2"], anclas, umbral_var=0.3, min_obs=MIN_OBS)
    assert ficha["ancla"] == {"columna": "a2", "signo_inicial": -1} and ficha["cambios_signo"] == []
    assert ficha["fecha_fijacion"] is not None
    assert serie.loc[:ficha["fecha_fijacion"]].isna().all()
    ok = serie.notna().to_numpy()
    assert np.corrcoef(serie[ok], f1[ok])[0, 1] > 0.8
    # bloque mezclado: b3 (= -f2) y a2 (= -f1) suben con el estrés de su bloque original
    serie, ficha = ab.termometro_azar(p, ["b3", "a2", "c1"], anclas, umbral_var=0.3, min_obs=MIN_OBS)
    assert ficha["ancla"]["signo_inicial"] == -1 and ficha["ancla"]["columna"] in ("b3", "a2")
    factor = f2 if ficha["ancla"]["columna"] == "b3" else f1
    ok = serie.notna().to_numpy()
    assert np.corrcoef(serie[ok], factor[ok])[0, 1] > 0.5
    # una ancla original sola no lleva máscara: su signo no depende de datos
    serie, ficha = ab.termometro_azar(p, ["c1"], anclas, umbral_var=0.3, min_obs=MIN_OBS)
    assert ficha["fecha_fijacion"] is None and serie.notna().all()


def _nombres(code) -> set[str]:
    """Nombres globales y atributos que usa una función (y sus funciones internas)."""
    out = set(code.co_names)
    for c in code.co_consts:
        if inspect.iscode(c):
            out |= _nombres(c)
    return out


def test_la_regla_de_signo_no_usa_etiquetas() -> None:
    for f in (ab._signo_columna, ab.termometro_azar, ab.termometros_azar, ab.termometros_media,
              ab.reparto_aleatorio, ab.construir_variante):
        nombres = " ".join(_nombres(f.__code__)).lower()
        assert "crisis" not in nombres and "evaluacion" not in nombres and "ev" not in nombres.split(), f.__name__
    # cambiar las ventanas de crisis del juez no cambia nada
    from regimenes import evaluacion as ev

    p = _panel()
    antes, _ = ab.termometros_azar(p, BLOQUES, 1, umbral_var=0.3, min_obs=MIN_OBS)
    with mock.patch.object(ev, "CRISIS_WINDOWS", {}):
        despues, _ = ab.termometros_azar(p, BLOQUES, 1, umbral_var=0.3, min_obs=MIN_OBS)
    pd.testing.assert_frame_equal(antes, despues)


def test_reparto_no_factible_se_descarta_de_forma_reproducible() -> None:
    p = _panel()
    indice = p.index[200:]
    # semilla cuyo primer reparto deja a3 (empieza en la fila 300) sola en Z03
    semilla = next(s for s in range(200)
                   if ab.reparto_aleatorio(BLOQUES, np.random.default_rng(s))["Z03"] == ["a3"])
    t, info = ab.termometros_azar(p, BLOQUES, semilla, umbral_var=0.3, min_obs=MIN_OBS, indice=indice)
    assert info["intentos"] >= 2 and not t.reindex(indice).isna().any().any()
    t2, info2 = ab.termometros_azar(p, BLOQUES, semilla, umbral_var=0.3, min_obs=MIN_OBS, indice=indice)
    assert info == info2
    pd.testing.assert_frame_equal(t, t2)


# --------------------------------------------------------------------------- #
# Trabajos: reanudación y esquema del CSV
# --------------------------------------------------------------------------- #
PACTADAS = ["base", "pista", "variante", "tipo", "bloque_quitado", "semilla", "step", "score_deteccion",
            "det_precision", "det_event_recall", "det_n_detectados", "det_n_eventos", "n_oos",
            "elapsed_seconds", "n_states"]


def _entorno(carpeta: Path) -> pd.DatetimeIndex:
    """Entrada base de juguete de la pista B y su ``variantes_B.json`` con SIN-T1 y AZAR-007."""
    bq = cargar_bloques("B")
    idx = pd.bdate_range("2005-01-03", periods=900, name="date")
    rng = np.random.default_rng(1)
    base = pd.DataFrame(rng.normal(size=(900, len(bq) + 1)), index=idx, columns=[*bq, "SP500_ret"])
    ruta = ab.ruta_base(carpeta, "B")
    ruta.parent.mkdir(parents=True)
    base.to_parquet(ruta)
    azar = pd.DataFrame(rng.normal(size=(900, 2)), index=idx, columns=["Z01", "Z02"])
    ruta_azar = ab.ruta_parquet_variante(carpeta, "B", "AZAR-007")
    azar.to_parquet(ruta_azar)
    comp = {"pista": "B", "entrada_base": {"archivo": ruta.name, "sha256": ab.sha256(ruta)},
            "huella_codigo": ab.huella_codigo(),
            "variantes": {**{v: ab.ficha_simple("B", v, bq) for v in ("SIN-T1", "SIN-T2", "SIN-T3")},
                          "AZAR-007": {"tipo": "azar", "columnas": ["Z01", "Z02"],
                                       "archivo": ruta_azar.name, "sha256": ab.sha256(ruta_azar)}}}
    ab.escribir_composicion(carpeta, "B", comp)
    return idx


def _falso_evaluar(llamadas: list):
    def evaluar(track, spec, panel, index):
        llamadas.append(spec)
        oos = index[756:]
        metrics = pd.DataFrame([{"id": spec.detector_id, "pista": track, "n_states": 2,
                                 "n_oos": len(oos), "det_n_eventos": 9, "det_n_detectados": 3,
                                 "det_event_recall": 1 / 3, "det_precision": 0.5,
                                 "score_deteccion": 0.4, "elapsed_seconds": 1.0}])
        wf = pd.DataFrame({"state": 0, "p_crisis": 0.1, "fold": 0}, index=oos)
        return metrics, wf
    return evaluar


def test_csv_lleva_las_columnas_pactadas(tmp_path: Path) -> None:
    _entorno(tmp_path)
    llamadas: list = []
    with mock.patch.object(ab, "evaluar_variante", _falso_evaluar(llamadas)):
        for v in ("SIN-T1", "AZAR-007"):
            assert ab.ejecutar_trabajo("B", "D02", v, tmp_path)["estado"] == "ok"
    sin = pd.read_csv(ab.ruta_trabajo(tmp_path, "B", "D02", "SIN-T1"))
    azar = pd.read_csv(ab.ruta_trabajo(tmp_path, "B", "D02", "AZAR-007"))
    for fila in (sin, azar):
        assert set(PACTADAS) <= set(fila.columns)
        assert list(fila.columns[:6]) == list(ab.COLUMNAS_EXTRA)
        assert fila.loc[0, "step"] == 252
    assert sin.loc[0, "tipo"] == "sin_bloque" and sin.loc[0, "bloque_quitado"] == "T1_renta_variable"
    assert pd.isna(sin.loc[0, "semilla"])
    assert azar.loc[0, "tipo"] == "azar" and azar.loc[0, "semilla"] == 7
    assert pd.isna(azar.loc[0, "bloque_quitado"])
    panel = pd.read_parquet(ab.ruta_trabajo(tmp_path, "B", "D02", "AZAR-007").with_suffix(".parquet"))
    assert {"state", "p_crisis"} <= set(panel.columns)
    # specs: columnas de la variante, signos +1 en D02 y reajuste anual
    assert len(llamadas[0].features) == 9 and "T1_renta_variable" not in llamadas[0].features
    assert llamadas[1].features == ("Z01", "Z02")
    assert llamadas[1].params["signs"] == {"Z01": 1, "Z02": 1}
    assert all(s.step == 252 for s in llamadas)


def test_reanudacion_no_recalcula_un_csv_existente(tmp_path: Path) -> None:
    _entorno(tmp_path)
    destino = ab.ruta_trabajo(tmp_path, "B", "D02", "SIN-T1")
    destino.write_text("ya\n", encoding="utf-8")
    llamadas: list = []
    with mock.patch.object(ab, "evaluar_variante", _falso_evaluar(llamadas)):
        assert ab.ejecutar_trabajo("B", "D02", "SIN-T1", tmp_path)["estado"] == "ya_estaba"
    assert llamadas == [] and destino.read_text(encoding="utf-8") == "ya\n"
    listos, _ = ab.trabajos(["B"], ["D02"], ["SIN-T1", "AZAR-007"], tmp_path)
    assert listos == [("B", "D02", "AZAR-007")]


def test_entrada_alterada_falla_sin_dejar_csv(tmp_path: Path) -> None:
    _entorno(tmp_path)
    ruta = ab.ruta_parquet_variante(tmp_path, "B", "AZAR-007")
    t = pd.read_parquet(ruta)
    (t * 2).to_parquet(ruta)
    with mock.patch.object(ab, "evaluar_variante", _falso_evaluar([])):
        estado = ab._trabajo_seguro(("B", "D02", "AZAR-007", str(tmp_path)))
    assert estado["estado"] == "error" and "sha256" in estado["detalle"]
    assert not ab.ruta_trabajo(tmp_path, "B", "D02", "AZAR-007").exists()


def test_orden_lpt_lentos_primero() -> None:
    orden = ab.ordenar_lpt([("B", "D02", "MEDIA"), ("A", "D08", "MEDIA"), ("B", "D04", "MEDIA")])
    assert [t[:2] for t in orden] == [("A", "D08"), ("B", "D04"), ("B", "D02")]


# --------------------------------------------------------------------------- #
# Signo re-estimado en cada reajuste
# --------------------------------------------------------------------------- #
def _panel_que_cambia(n: int = 1500) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """``d`` y ``d2`` siguen a ``f1`` hasta la fila 300 y van en contra después."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2000-01-03", periods=n)
    f1 = np.zeros(n)
    for t in range(1, n):
        f1[t] = 0.98 * f1[t - 1] + rng.normal()
    signo = np.where(np.arange(n) < 300, 1.0, -1.0)
    p = pd.DataFrame({"a1": f1 + rng.normal(scale=0.5, size=n),
                      "d": signo * f1 + rng.normal(scale=0.5, size=n),
                      "d2": signo * f1 + rng.normal(scale=0.7, size=n),
                      "c1": rng.normal(size=n)}, index=idx)
    anclas = {"a1": ("a1", 1), "d": ("a1", 1), "d2": ("a1", 1), "c1": ("c1", 1)}
    return p, f1, anclas


@pytest.mark.parametrize("cols", [["d"], ["d", "d2", "c1"]])
def test_el_signo_de_la_ancla_sigue_al_vigente(cols: list[str]) -> None:
    p, f1, anclas = _panel_que_cambia()
    serie, ficha = ab.termometro_azar(p, cols, anclas, umbral_var=0.3, min_obs=MIN_OBS)
    assert ficha["ancla"]["columna"] in ("d", "d2") and ficha["ancla"]["signo_inicial"] == 1
    assert ficha["cambios_signo"] and ficha["cambios_signo"][-1][1] == -1
    # tras el cambio el termómetro vuelve a subir con f1 (con el signo fijo en r0 bajaría)
    cambio = pd.Timestamp(ficha["cambios_signo"][-1][0])
    tarde = np.asarray(serie.index > cambio) & serie.notna().to_numpy()
    assert np.corrcoef(serie[tarde], f1[tarde])[0, 1] > 0.5


@pytest.mark.parametrize("corte", [200, 450, 700, 1499])
def test_azar_con_cambio_de_signo_es_causal(corte: int) -> None:
    p, _, anclas = _panel_que_cambia()
    t = p.index[corte]
    for cols in (["d"], ["d", "d2", "c1"]):
        completo, _ = ab.termometro_azar(p, cols, anclas, umbral_var=0.3, min_obs=MIN_OBS)
        recortado, _ = ab.termometro_azar(p.loc[:t], cols, anclas, umbral_var=0.3, min_obs=MIN_OBS)
        pd.testing.assert_series_equal(recortado, completo.loc[:t], rtol=1e-12, check_freq=False)


# --------------------------------------------------------------------------- #
# Huella de código y procesos que mueren
# --------------------------------------------------------------------------- #
def test_huella_distinta_se_niega_a_seguir(tmp_path: Path) -> None:
    _entorno(tmp_path)
    comp = ab.leer_composicion(tmp_path, "B")
    comp["huella_codigo"]["ablaciones.py"] = "otro"
    ab.escribir_composicion(tmp_path, "B", comp)
    with pytest.raises(ValueError, match="variantes_B.json"):
        ab.ejecutar_trabajo("B", "D02", "SIN-T1", tmp_path)
    with pytest.raises(ValueError, match="el código cambió"):
        ab.ejecutar(["B"], ["D02"], ["SIN-T1"], salida=tmp_path)


HIJO_FALSO = '''
import os, time
from pathlib import Path

def trabajo(args):
    track, d, v, salida = args
    marca = Path(salida) / f"{v}.murio"
    if v == "SIN-T3" or (v == "SIN-T1" and not marca.exists()):
        marca.touch()
        os._exit(1)
    time.sleep(4.0 if v == "SIN-T2" else 0.2)   # SIN-T2 sigue en vuelo cuando SIN-T1 muere
    (Path(salida) / f"{track}_{d}_{v}.csv").write_text("ok", encoding="utf-8")
    return {"trabajo": f"{track}_{d}_{v}", "estado": "ok"}
'''


def test_un_hijo_que_muere_no_rompe_la_ejecucion(tmp_path: Path, monkeypatch) -> None:
    import os
    import sys

    modulos = tmp_path / "modulos"
    modulos.mkdir()
    (modulos / "hijo_falso_ablaciones.py").write_text(HIJO_FALSO, encoding="utf-8")
    monkeypatch.syspath_prepend(str(modulos))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(modulos), *[p for p in sys.path if p]]))
    import hijo_falso_ablaciones as hf

    salida = tmp_path / "salida"
    _entorno(salida)
    # dos en vuelo; SIN-T1 muere una vez. Si el pool detecta la rotura con los dos en vuelo, vuelven a
    # pendientes y acaban bien. CPython (max_tasks_per_child) a veces no la detecta hasta que acaba SIN-T2:
    # entonces SIN-T1 es el único en vuelo y queda en _errores. Lo garantizado: la ejecución sigue,
    # SIN-T2 acaba bien y SIN-T1 acaba bien o queda registrado como error.
    estados = ab.ejecutar(["B"], ["D02"], ["SIN-T1", "SIN-T2"], salida=salida, jobs=2, trabajo=hf.trabajo)
    final = {e["trabajo"]: e["estado"] for e in estados}
    assert final["B_D02_SIN-T2"] == "ok"
    assert final["B_D02_SIN-T1"] == "ok" or ab.ruta_error(salida, "B", "D02", "SIN-T1").exists()
    # uno solo en vuelo que siempre muere: es el causante y queda en _errores; el resto sigue
    estados = ab.ejecutar(["B"], ["D02"], ["SIN-T3", "AZAR-007"], salida=salida, jobs=1, trabajo=hf.trabajo)
    assert {e["trabajo"]: e["estado"] for e in estados} == {"B_D02_SIN-T3": "error", "B_D02_AZAR-007": "ok"}
    assert ab.ruta_error(salida, "B", "D02", "SIN-T3").exists()
    assert not ab.ruta_trabajo(salida, "B", "D02", "SIN-T3").exists()