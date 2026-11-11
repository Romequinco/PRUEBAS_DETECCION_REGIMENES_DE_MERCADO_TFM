"""Tests de ``regimenes.regimen.get_regimen`` (tool provisional de la Fase 1).

Con paneles de juguete en ``tmp_path``: causalidad (truncar el panel no cambia
el pasado; ``dias_en_estado`` solo mira hacia atrás), fines de semana, fechas
fuera de rango, cambio de detector y coherencia con el panel. Al final, unas
comprobaciones con los paneles reales del benchmark v2 (se saltan si faltan).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regimenes import rutas
from regimenes.regimen import FechaFueraDeRango, cargar_config, get_regimen, limpiar_cache
from regimenes.regimen.tool import _rachas


CONFIG_JUGUETE = {
    "provisional": True,
    "directorio_benchmark": "no-se-usa",
    "detector_por_pista": {"A": "X1", "B": "X2"},
    "pista_preferente": "B",
    "pista_respaldo": "A",
    "estados": ["calma", "crisis"],
}


def _escribir(
    directorio: Path,
    pista: str,
    detector: str,
    index: pd.DatetimeIndex,
    state: np.ndarray,
    *,
    n_states: int = 2,
    con_prob: bool = True,
) -> pd.DataFrame:
    (directorio / "panels").mkdir(parents=True, exist_ok=True)
    (directorio / "metrics").mkdir(parents=True, exist_ok=True)
    panel = pd.DataFrame({"state": np.asarray(state, dtype=np.int64)}, index=index)
    panel.index.name = "date"
    if con_prob:
        panel["p_crisis"] = np.linspace(0.01, 0.99, len(index))
    panel["fold"] = np.arange(len(index)) // 21
    panel.to_parquet(directorio / "panels" / f"{pista}_{detector}.parquet")
    pd.DataFrame(
        [{"id": detector, "pista": pista, "n_states": n_states, "cache_fingerprint": "abc123"}]
    ).to_csv(directorio / "metrics" / f"{pista}_{detector}.csv", index=False)
    return panel


@pytest.fixture()
def bench(tmp_path: Path) -> dict:
    """Pista A desde 2000-01-03 y B desde 2000-03-01, ambas hasta 2000-06-30."""
    limpiar_cache()
    rng = np.random.default_rng(0)
    idx_a = pd.bdate_range("2000-01-03", "2000-06-30")
    idx_b = pd.bdate_range("2000-03-01", "2000-06-30")
    # Rachas largas y aleatorias para que dias_en_estado sea informativo.
    estados_a = np.repeat(rng.integers(0, 2, size=len(idx_a) // 7 + 1), 7)[: len(idx_a)]
    estados_b = np.repeat(rng.integers(0, 2, size=len(idx_b) // 5 + 1), 5)[: len(idx_b)]
    paneles = {
        ("A", "X1"): _escribir(tmp_path, "A", "X1", idx_a, estados_a),
        ("B", "X2"): _escribir(tmp_path, "B", "X2", idx_b, estados_b),
        ("B", "X3"): _escribir(tmp_path, "B", "X3", idx_b, 1 - estados_b, con_prob=False),
        ("A", "X4"): _escribir(tmp_path, "A", "X4", idx_a, np.arange(len(idx_a)) % 3, n_states=3),
    }
    yield {"dir": tmp_path, "paneles": paneles}
    limpiar_cache()


def _q(bench: dict, fecha, **kw) -> dict:
    return get_regimen(fecha, directorio=bench["dir"], config=CONFIG_JUGUETE, **kw)


# --------------------------------------------------------------------------- #
# Contrato y coherencia con el panel
# --------------------------------------------------------------------------- #
def test_claves_y_coherencia_con_el_panel(bench: dict) -> None:
    panel = bench["paneles"][("B", "X2")]
    for fecha in panel.index[[0, 10, 37, -1]]:
        r = _q(bench, fecha)
        assert r["pista"] == "B" and r["detector"] == "X2"
        assert r["fecha_dato"] == fecha.date().isoformat()
        assert r["estado_detector"] == int(panel.loc[fecha, "state"])
        assert r["estado"] == ("crisis" if panel.loc[fecha, "state"] == 1 else "calma")
        assert r["prob_crisis"] == pytest.approx(float(panel.loc[fecha, "p_crisis"]))
        assert r["dias_en_estado"] >= 1
        assert r["fuente"]["sha256"] and r["fuente"]["huella_cache"] == "abc123"
        assert r["fuente"]["panel"].endswith("panels/B_X2.parquet")
        assert r["provisional"] is True
        assert r["estados_posibles"] == ["calma", "crisis"]


def test_dias_en_estado_cuenta_la_racha_hacia_atras(bench: dict) -> None:
    panel = bench["paneles"][("A", "X1")]
    s = panel["state"].to_numpy()
    for i in (0, 5, 6, 7, 30, len(s) - 1):
        esperado = 1
        while i - esperado >= 0 and s[i - esperado] == s[i]:
            esperado += 1
        r = _q(bench, panel.index[i], pista="A")
        assert r["dias_en_estado"] == esperado
        assert r["dias_en_estado_censurado"] == (esperado == i + 1)


def test_rachas_basico() -> None:
    np.testing.assert_array_equal(
        _rachas(np.array([0, 0, 1, 1, 1, 0], dtype=bool)), [1, 2, 1, 2, 3, 1]
    )


def test_multiestado_solo_el_ultimo_es_crisis(bench: dict) -> None:
    panel = bench["paneles"][("A", "X4")]
    for fecha in panel.index[:6]:
        r = _q(bench, fecha, pista="A", detector="X4")
        assert r["estado"] == ("crisis" if panel.loc[fecha, "state"] == 2 else "calma")
    # Estados 0 y 1 son ambos «calma»: la racha no se corta entre ellos.
    assert _q(bench, panel.index[1], pista="A", detector="X4")["dias_en_estado"] == 2


# --------------------------------------------------------------------------- #
# Causalidad
# --------------------------------------------------------------------------- #
def test_truncar_el_panel_no_cambia_el_pasado(bench: dict, tmp_path_factory) -> None:
    completo = {f: _q(bench, f) for f in pd.date_range("2000-01-03", "2000-06-30", freq="D")}
    corte = pd.Timestamp("2000-04-14")
    trunc = tmp_path_factory.mktemp("truncado")
    for (pista, det), panel in bench["paneles"].items():
        p = panel.loc[:corte]
        _escribir(trunc, pista, det, p.index, p["state"].to_numpy(),
                  n_states=3 if det == "X4" else 2, con_prob="p_crisis" in p)
        if "p_crisis" in p:  # misma probabilidad que el original en las fechas comunes
            p.to_parquet(trunc / "panels" / f"{pista}_{det}.parquet")
    for fecha, r in completo.items():
        if fecha > corte:
            continue
        r2 = get_regimen(fecha, directorio=trunc, config=CONFIG_JUGUETE)
        claves = ("fecha_dato", "pista", "detector", "estado", "prob_crisis",
                  "dias_en_estado", "dias_en_estado_censurado")
        assert {k: r[k] for k in claves} == {k: r2[k] for k in claves}, fecha


def test_dias_en_estado_no_mira_hacia_delante(bench: dict) -> None:
    # Cambiar todo lo posterior a t no altera la respuesta en t.
    d = bench["dir"]
    panel = bench["paneles"][("B", "X2")]
    t = panel.index[40]
    antes = _q(bench, t)
    s = panel["state"].to_numpy().copy()
    s[41:] = 1 - s[41:]
    _escribir(d, "B", "X2", panel.index, s)
    limpiar_cache()
    despues = _q(bench, t)
    assert despues["fuente"]["sha256"] != antes["fuente"]["sha256"]  # sí releyó el panel
    antes.pop("fuente"), despues.pop("fuente")
    assert despues == antes


# --------------------------------------------------------------------------- #
# Fechas: fines de semana, regla de pista y rango
# --------------------------------------------------------------------------- #
def test_fin_de_semana_usa_el_ultimo_dia_habil(bench: dict) -> None:
    viernes = _q(bench, "2000-04-14")
    for f in ("2000-04-15", "2000-04-16"):
        r = _q(bench, f)
        assert r["fecha"] == f and r["fecha_dato"] == "2000-04-14"
        assert {k: v for k, v in r.items() if k != "fecha"} == {
            k: v for k, v in viernes.items() if k != "fecha"
        }


def test_regla_de_pista_por_fecha(bench: dict) -> None:
    assert _q(bench, "2000-02-29")["pista"] == "A"
    assert _q(bench, "2000-02-29")["detector"] == "X1"
    assert _q(bench, "2000-03-01")["pista"] == "B"
    # Con pista explícita no hay regla: A sigue disponible después del inicio de B.
    assert _q(bench, "2000-05-02", pista="A")["pista"] == "A"


def test_fecha_fuera_de_rango(bench: dict) -> None:
    with pytest.raises(FechaFueraDeRango, match="anterior al inicio del OOS"):
        _q(bench, "1999-12-31")
    with pytest.raises(FechaFueraDeRango, match="anterior al inicio del OOS"):
        _q(bench, "2000-02-15", pista="B")
    with pytest.raises(FechaFueraDeRango, match="posterior al último dato"):
        _q(bench, "2000-07-03")  # lunes sin dato
    # Último dato viernes 30-06; el sábado y el domingo siguientes sí responden.
    assert _q(bench, "2000-07-02")["fecha_dato"] == "2000-06-30"


# --------------------------------------------------------------------------- #
# Detector intercambiable
# --------------------------------------------------------------------------- #
def test_cambio_de_detector_por_parametro(bench: dict) -> None:
    r2 = _q(bench, "2000-05-02")
    r3 = _q(bench, "2000-05-02", detector="x3")
    assert (r2["detector"], r3["detector"]) == ("X2", "X3")
    assert r2["estado"] != r3["estado"]  # X3 es el complemento de X2
    assert r3["prob_crisis"] is None     # X3 no trae p_crisis
    assert r2["fuente"]["sha256"] != r3["fuente"]["sha256"]
    with pytest.raises(FileNotFoundError, match="X9"):
        _q(bench, "2000-05-02", detector="X9")


def test_config_rechaza_estados_no_provisionales() -> None:
    with pytest.raises(ValueError, match="alerta"):
        cargar_config({**CONFIG_JUGUETE, "estados": ["calma", "alerta", "crisis"]})


# --------------------------------------------------------------------------- #
# Paneles reales del benchmark v2 (se saltan si no están en disco)
# --------------------------------------------------------------------------- #
_PANELES = rutas.RESULTS_BENCHMARK / "panels"
_MCS = rutas.RESULTS_BENCHMARK / "vara" / "mcs_diario.csv"
_reales = pytest.mark.skipif(
    not ((_PANELES / "A_D06.parquet").is_file() and (_PANELES / "B_D05.parquet").is_file()),
    reason="faltan los paneles reales de results/benchmark/panels",
)


@pytest.mark.resultados
@pytest.mark.skipif(not _MCS.is_file(), reason="falta results/benchmark/vara/mcs_diario.csv")
def test_config_real_coincide_con_lider_del_mcs() -> None:
    mcs = pd.read_csv(_MCS)
    cfg = cargar_config()
    for pista, det in cfg["detector_por_pista"].items():
        lider = mcs[mcs["pista"] == pista].sort_values("orden_eliminacion").iloc[-1]["modelo"]
        assert det == lider, f"pista {pista}: config {det} y MCS {lider}"


@pytest.mark.resultados
@_reales
def test_reales_coherentes_con_el_panel() -> None:
    limpiar_cache()
    for fecha, pista, det in (("2008-10-10", "A", "D06"), ("2020-03-16", "B", "D05"),
                              ("2015-06-01", "B", "D05")):
        r = get_regimen(fecha)
        assert (r["pista"], r["detector"]) == (pista, det)
        panel = pd.read_parquet(_PANELES / f"{pista}_{det}.parquet")
        fila = panel.loc[pd.Timestamp(r["fecha_dato"])]
        assert r["estado_detector"] == int(fila["state"])
        assert r["prob_crisis"] == pytest.approx(float(fila["p_crisis"]))


@pytest.mark.resultados
@_reales
def test_reales_crisis_conocidas_y_cambio_de_detector() -> None:
    assert get_regimen("2008-10-10")["estado"] == "crisis"
    assert get_regimen("2020-03-14")["fecha_dato"] == "2020-03-13"  # sábado
    with pytest.raises(FechaFueraDeRango):
        get_regimen("1960-01-04")
    # Cambio de detector: D10 no se versiona (solo A_D06 y B_D05, ADR-012); se salta en un clon limpio.
    if not all((_PANELES / f"{p}_D10.parquet").is_file() for p in "AB"):
        pytest.skip("faltan los paneles D10 locales (python -m regimenes.benchmark --track A B)")
    r = get_regimen("2020-03-16", detector="D10")
    assert r["detector"] == "D10" and r["pista"] == "B"


# --------------------------------------------------------------------------- #
# Contexto: consenso de detectores y fusión online (opción 3 de CP2)
# --------------------------------------------------------------------------- #
def _config_contexto(bench: dict, prob: float = 0.8) -> dict:
    """Escribe señales y probabilidad de juguete para la pista B y devuelve la config."""
    idx = bench["paneles"][("B", "X2")].index
    dir_ctx = bench["dir"] / "contexto"
    dir_ctx.mkdir(exist_ok=True)
    senales = pd.DataFrame(
        {"X2": bench["paneles"][("B", "X2")]["state"].astype(bool), "Y1": True, "Y2": False,
         "PAR_Y1>Y2": True},  # una fusión: no cuenta como detector
        index=idx,
    )
    senales.to_parquet(dir_ctx / "senales_B.parquet")
    pd.DataFrame({"C": prob}, index=idx).to_parquet(dir_ctx / "pbar_B.parquet")
    return {
        **CONFIG_JUGUETE,
        "contexto": {
            "senales_detectores": str(dir_ctx / "senales_{pista}.parquet"),
            "prob_consenso": str(dir_ctx / "pbar_{pista}.parquet"),
            "columna_consenso": "C",
            "umbral_consenso": 0.5,
        },
    }


@pytest.mark.parametrize("prob", [0.8, 0.2])
def test_contexto_cuenta_acuerdo_y_lectura(bench: dict, prob: float) -> None:
    cfg = _config_contexto(bench, prob)
    for fecha in pd.bdate_range("2000-03-01", "2000-06-30")[::9]:
        r = get_regimen(fecha, directorio=bench["dir"], config=cfg)
        ctx = r["contexto"]
        crisis = r["estado"] == "crisis"
        # X2 (el principal) e Y1 en crisis según toque; Y2 nunca; la fusión no cuenta.
        assert ctx["detectores_total"] == 3
        assert ctx["ids_en_crisis"] == (["X2", "Y1"] if crisis else ["Y1"])
        assert ctx["confianza"] == pytest.approx(2 / 3)
        assert ctx["estado_consenso"] == ("crisis" if prob >= 0.5 else "calma")
        assert ctx["coincide_consenso"] == (crisis == (prob >= 0.5))
        assert ctx["lectura"] == {
            (True, True): "crisis respaldada", (True, False): "crisis sin respaldo",
            (False, True): "calma con aviso del consenso", (False, False): "calma respaldada",
        }[(crisis, prob >= 0.5)]


def test_contexto_opcional_y_ausente(bench: dict) -> None:
    cfg = _config_contexto(bench)
    assert get_regimen("2000-05-02", directorio=bench["dir"], config=cfg, contexto=False)["contexto"] is None
    # Pista A: no hay ficheros de contexto → respuesta normal con contexto None.
    assert get_regimen("2000-01-10", directorio=bench["dir"], config=cfg)["contexto"] is None
    # Sin bloque en la config (la de juguete) → None.
    assert _q(bench, "2000-05-02")["contexto"] is None


_CONTEXTO_REAL = all(
    (rutas.RESULTS / "fusion" / sub / f"{nombre}_{p}.parquet").is_file()
    for sub, nombre in (("fijas", "senales"), ("online", "pbar")) for p in ("A", "B")
)


@pytest.mark.resultados
@_reales
@pytest.mark.skipif(not _CONTEXTO_REAL, reason="faltan results/fusion/{fijas,online}")
def test_reales_contexto_usa_los_candidatos_del_adr011() -> None:
    from regimenes.fusion import vara

    limpiar_cache()
    for fecha, pista in (("2008-10-10", "A"), ("2020-03-16", "B"), ("2015-06-01", "B")):
        ctx = get_regimen(fecha)["contexto"]
        assert ctx is not None
        assert ctx["detectores_total"] == len(vara.candidatos(pista))
        assert set(ctx["ids_en_crisis"]) <= set(vara.candidatos(pista))
        assert 0.0 <= ctx["prob_crisis_consenso"] <= 1.0
        assert 0.0 <= ctx["confianza"] <= 1.0
    assert get_regimen("2008-10-10")["contexto"]["lectura"] == "crisis respaldada"
