"""Tests sobre datos reales del preprocesado v2 (notebooks/03_preprocesado.ipynb).

Se saltan si `data/raw/` (gitignored) no está descargado. Cubren:
- coherencia del banco congelado (`configs/benchmark_spec.yaml`, ADR-002);
- que `regimenes.features.construir_paneles` reproduce EXACTAMENTE `data/processed/*` (los paneles
  que consumió el benchmark);
- causalidad del pipeline COMPLETO por truncado (features + rejilla NYSE + `_edad_dias`);
- regla anti-fuga (ninguna serie rol=validation como feature);
- causalidad de CALENDARIO (ADR-003): tabla `LAG_PUBLICACION` completa y correcta, y
  truncado "por fecha de publicación" del pipeline completo;
- regresiones de los defectos corregidos por ADR-003 (antes `xfail(strict)`): ICSA_z
  vacía en pista B y mensuales "media del mes" usadas sin lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from regimenes import features as ft
from regimenes import rutas

RAW = rutas.DATA_RAW
PROCESSED = rutas.DATA_PROCESSED
SPEC = yaml.safe_load(rutas.BENCHMARK_SPEC.read_text(encoding="utf-8"))
GW_COLUMN = ft.GW_COLUMN  # fijada en 03 §1

_cov_path = RAW / "coverage_report.csv"
HAY_RAW = _cov_path.exists() and (RAW / "yfinance" / "SP500.parquet").exists()
necesita_raw = pytest.mark.skipif(not HAY_RAW, reason="data/raw no descargado")
necesita_processed = pytest.mark.skipif(
    not (PROCESSED / "pistaA_diaria.parquet").exists(), reason="data/processed no generado"
)


def _load_any(nombre: str, src: dict[str, str]):
    """Misma convención de carga que 03 (`ft.seleccionar_raw`)."""
    return ft.seleccionar_raw(pd.read_parquet(RAW / src[nombre] / f"{nombre}.parquet"), nombre)


@pytest.fixture(scope="module")
def raw_por_pista() -> dict[str, dict[str, pd.Series]]:
    cov = pd.read_csv(_cov_path)
    src = dict(zip(cov.nombre, cov.fuente))
    return {
        p: {n: _load_any(n, src) for n in SPEC[f"pista_{p}"]["series_features"]}
        for p in ("A", "B")
    }


@pytest.fixture(scope="module")
def paneles(raw_por_pista) -> dict[str, dict[str, pd.DataFrame]]:
    out = {}
    for p, raw in raw_por_pista.items():
        spec = SPEC[f"pista_{p}"]
        out[p] = ft.construir_paneles(raw, spec["ventana_inicio"], spec["ventana_fin"])
    return out


# --------------------------------------------------------------------------- #
# Coherencia del banco congelado (no necesita data/raw)
# --------------------------------------------------------------------------- #
def test_benchmark_spec_coherente_con_adr002():
    a, b = SPEC["pista_A"], SPEC["pista_B"]
    assert len(a["series_features"]) == a["n_series"] == 41
    assert len(b["series_features"]) == b["n_series"] == 106
    assert len(set(b["series_features"])) == 106  # sin duplicados
    assert set(a["series_features"]) <= set(b["series_features"])
    assert a["ventana_fin"] == b["ventana_fin"] == "2026-05-29"
    assert (a["ventana_inicio"], b["ventana_inicio"]) == ("1962-01-02", "2007-04-11")
    assert len(SPEC["crisis_windows"]["pista_A"]) == a["n_crisis_en_ventana"] == 18
    assert len(SPEC["crisis_windows"]["pista_B"]) == b["n_crisis_en_ventana"] == 10
    for p in ("A", "B"):
        ini = pd.Timestamp(SPEC[f"pista_{p}"]["ventana_inicio"])
        fin = pd.Timestamp(SPEC[f"pista_{p}"]["ventana_fin"])
        for evento, (pico, suelo) in SPEC["crisis_windows"][f"pista_{p}"].items():
            assert ini <= pd.Timestamp(pico) <= pd.Timestamp(suelo) <= fin, evento
            assert SPEC["drawdown_troughs"][evento] == suelo, evento


def test_ninguna_serie_validation_es_feature():
    cat = yaml.safe_load(rutas.CATALOG.read_text(encoding="utf-8"))
    validation = {
        s["nombre_interno"]
        for bloque in cat.values() if isinstance(bloque, dict) and "series" in bloque
        for s in bloque["series"] if s.get("rol") == "validation"
    }
    assert validation, "no se encontraron series rol=validation en catalog.yaml"
    assert not validation & set(SPEC["pista_B"]["series_features"])


# --------------------------------------------------------------------------- #
# Reproducibilidad y causalidad del pipeline completo (datos reales)
# --------------------------------------------------------------------------- #
@necesita_raw
@necesita_processed
@pytest.mark.parametrize("pista", ["A", "B"])
@pytest.mark.parametrize("freq", ["diaria", "mensual"])
def test_pipeline_reproduce_paneles_en_disco(paneles, pista, freq):
    disco = pd.read_parquet(PROCESSED / f"pista{pista}_{freq}.parquet")
    nuevo = paneles[pista][freq].copy()
    nuevo.index.name = "date"
    pd.testing.assert_frame_equal(disco, nuevo, check_exact=True, check_freq=False)


@necesita_raw
@pytest.mark.parametrize(
    "pista,cut",
    [("A", "1974-10-03"), ("A", "1987-10-19"), ("A", "2020-03-02"),
     ("B", "2008-09-15"), ("B", "2013-06-20"), ("B", "2020-03-02")],
)
def test_pipeline_completo_causal_por_truncado(raw_por_pista, paneles, pista, cut):
    """Truncar TODAS las series crudas en `cut` y rehacer features+rejilla+edad no puede
    cambiar ninguna celda <= cut de ningún panel (incluidas las columnas _edad_dias)."""
    spec = SPEC[f"pista_{pista}"]
    trunc = ft.construir_paneles(ft.truncar(raw_por_pista[pista], cut),
                                 spec["ventana_inicio"], spec["ventana_fin"])
    for freq in ("diaria", "mensual"):
        full = paneles[pista][freq]
        assert list(trunc[freq].columns) == list(full.columns)
        assert trunc[freq].index.max() <= pd.Timestamp(cut)
        mad = ft.truncation_max_abs_diff(full, trunc[freq], cut)
        assert mad.max() == 0.0, mad[mad > 0].to_dict()


@necesita_raw
def test_anidamiento_columnas_A_en_B(paneles):
    base = lambda p: {c for f in ("diaria", "mensual") for c in paneles[p][f].columns}  # noqa: E731
    assert base("A") <= base("B")


@necesita_raw
def test_paneles_dentro_de_ventana(paneles):
    for p in ("A", "B"):
        spec = SPEC[f"pista_{p}"]
        for f in ("diaria", "mensual"):
            idx = paneles[p][f].index
            assert idx.min() >= pd.Timestamp(spec["ventana_inicio"])
            assert idx.max() == pd.Timestamp(spec["ventana_fin"])
            assert idx.is_monotonic_increasing and idx.is_unique


# --------------------------------------------------------------------------- #
# Regresiones de los defectos corregidos por ADR-003 (antes xfail(strict))
# --------------------------------------------------------------------------- #
@necesita_raw
def test_ninguna_columna_diaria_vacia(paneles):
    """ICSA (fechada en sábado) quedaba entera en NaN con la alineación v1."""
    for p in ("A", "B"):
        vacias = [c for c in paneles[p]["diaria"] if paneles[p]["diaria"][c].isna().all()]
        assert not vacias, (p, vacias)


@necesita_raw
def test_icsa_visible_el_jueves_de_publicacion(raw_por_pista, paneles):
    """El dato de la semana que acaba el sábado S aparece el jueves S+5, no antes."""
    icsa = raw_por_pista["B"]["ICSA"]
    z_pub = ft.causal_zscore(ft.lag_publicacion(icsa, 0, 5))
    grid = paneles["B"]["diaria"].index
    esperado = z_pub.reindex(grid, method="ffill")
    np.testing.assert_allclose(paneles["B"]["diaria"]["ICSA_z"].values, esperado.values, equal_nan=True)
    sabado = pd.Timestamp("2020-03-21")  # semana de la explosión de peticiones (covid)
    assert paneles["B"]["diaria"].loc["2020-03-25", "ICSA_z"] < 5  # miércoles: aún no publicado
    assert paneles["B"]["diaria"].loc["2020-03-26", "ICSA_z"] == z_pub.loc[sabado + pd.Timedelta(days=5)]


@necesita_raw
def test_gs10_mensual_es_media_del_mes_fechada_el_dia_1(raw_por_pista):
    """Evidencia del look-ahead de calendario de las mensuales 'mercado' sin lag: el valor
    fechado M-01 coincide con la MEDIA de los diarios de todo el mes M (no con el último
    dato conocido el día 1)."""
    raw = raw_por_pista["A"]
    gs10, dgs10 = raw["GS10"].loc["1990":], raw["DGS10"].loc["1990":]
    media_mes = dgs10.resample("MS").mean()
    ultimo_conocido = dgs10.resample("MS").last().shift(1)  # último dato del mes anterior
    j = pd.concat({"m": gs10, "media": media_mes, "prev": ultimo_conocido}, axis=1).dropna()
    mae_media = (j.m - j.media).abs().mean()
    mae_prev = (j.m - j.prev).abs().mean()
    assert mae_media < 0.01 < mae_prev


@necesita_raw
@pytest.mark.parametrize("mensual,diaria", [
    ("GS10", "DGS10"), ("GS5", "DGS5"), ("GS1", "DGS1"), ("TB3MS", "DTB3"), ("FEDFUNDS", "DFF"),
    (("MOODYS_BAA", "MOODYS_AAA"), "MOODYS_BAA_AAA_SPREAD"),
])
def test_ninguna_serie_media_del_mes_sin_lag(raw_por_pista, mensual, diaria):
    """Si el dato mensual es (empíricamente) la media del mes M, la tabla de lags debe
    retrasarlo al menos 1 mes (hasta el cierre de M)."""
    raw = raw_por_pista["B"]
    if isinstance(mensual, tuple):
        m = raw[mensual[0]] - raw[mensual[1]]
        nombres = mensual
    else:
        m, nombres = raw[mensual], (mensual,)
    d = raw[diaria].loc["1990":]
    j = pd.concat({"m": m.loc["1990":], "media": d.resample("MS").mean()}, axis=1).dropna()
    assert (j.m - j.media).abs().mean() < 0.02  # es media del mes
    for n in nombres:
        e = ft.LAG_PUBLICACION[n]
        assert e["tipo"] == "media_mes" and e["meses"] >= 1, n


@necesita_raw
def test_tabla_lags_cubre_toda_serie_no_diaria(raw_por_pista):
    """Toda serie cruda de la pista con frecuencia semanal o menor tiene entrada en
    LAG_PUBLICACION (nadie entra a una feature con su fecha de referencia)."""
    sin_lag = []
    for n, s in raw_por_pista["B"].items():
        paso = s.dropna(how="all").index.to_series().diff().median()
        if paso > pd.Timedelta(days=4) and n not in ft.LAG_PUBLICACION:
            sin_lag.append(n)
    assert not sin_lag, sin_lag
    assert all(ft.LAG_PUBLICACION[n]["meses"] >= 1 for n in raw_por_pista["B"]
               if n in ft.LAG_PUBLICACION and ft.LAG_PUBLICACION[n]["tipo"] != "semanal")


@necesita_raw
def test_mensuales_mercado_solo_visibles_tras_cerrar_el_mes(raw_por_pista, paneles):
    raw = raw_por_pista["A"]
    spread = (raw["MOODYS_BAA"] - raw["MOODYS_AAA"]).rename("credit_BaaAaa_mensual")
    correcto = ft.causal_zscore(ft.lag_publicacion(spread, 1))
    grid = paneles["A"]["mensual"].index
    esperado = correcto.reindex(grid, method="ffill")
    actual = paneles["A"]["mensual"]["credit_BaaAaa_mensual_z"]
    np.testing.assert_allclose(actual.values, esperado.values, equal_nan=True)


@necesita_raw
@pytest.mark.parametrize(
    "pista,t",
    [("A", "1987-10-19"), ("A", "2008-10-15"), ("B", "2008-10-15"), ("B", "2020-03-20")],
)
def test_pipeline_causal_por_fecha_de_publicacion(raw_por_pista, paneles, pista, t):
    """Truncado de CALENDARIO: rehacer los paneles solo con lo PUBLICADO a fecha t (cada
    serie cortada en t - lag de publicación de `LAG_PUBLICACION`) no puede cambiar ninguna
    celda <= t. Si el pipeline ignorara la tabla (regla v1: mensuales sin lag, macro con 1
    mes) este test falla; que la tabla sea correcta lo cubren los tests de arriba."""
    t = pd.Timestamp(t)
    publicado = {}
    for n, s in raw_por_pista[pista].items():
        lag = ft.lag_de(n)
        publicado[n] = s.loc[: (t - lag) if lag is not None else t]
    spec = SPEC[f"pista_{pista}"]
    trunc = ft.construir_paneles(publicado, spec["ventana_inicio"], spec["ventana_fin"])
    for freq in ("diaria", "mensual"):
        mad = ft.truncation_max_abs_diff(paneles[pista][freq], trunc[freq], t)
        assert mad.max() == 0.0, (freq, mad[mad > 0].to_dict())
