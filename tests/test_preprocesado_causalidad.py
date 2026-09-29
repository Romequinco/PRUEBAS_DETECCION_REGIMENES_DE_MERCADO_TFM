"""Tests sobre datos reales del preprocesado v2 (notebooks/03_preprocesado.ipynb).

Se saltan si `data/raw/` (gitignored) no está descargado. Cubren:
- coherencia del banco congelado (`data/benchmark_spec.yaml`, ADR-002);
- que `src.features.construir_paneles` reproduce EXACTAMENTE `data/processed/*` (los paneles
  que consumió el benchmark);
- causalidad del pipeline COMPLETO por truncado (features + rejilla NYSE + `_edad_dias`);
- regla anti-fuga (ninguna serie rol=validation como feature);
- defectos conocidos documentados como `xfail(strict=True)`: si alguien los corrige (y
  regenera los paneles), el test pasará a XPASS y obligará a retirar la marca.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src import features as ft

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SPEC = yaml.safe_load((ROOT / "data" / "benchmark_spec.yaml").read_text(encoding="utf-8"))
GW_COLUMN = "b/m"  # fijada en 03 §1

_cov_path = RAW / "coverage_report.csv"
HAY_RAW = _cov_path.exists() and (RAW / "yfinance" / "SP500.parquet").exists()
necesita_raw = pytest.mark.skipif(not HAY_RAW, reason="data/raw no descargado")
necesita_processed = pytest.mark.skipif(
    not (PROCESSED / "pistaA_diaria.parquet").exists(), reason="data/processed no generado"
)


def _load_any(nombre: str, src: dict[str, str]) -> pd.Series:
    df = pd.read_parquet(RAW / src[nombre] / f"{nombre}.parquet")
    if nombre == "GW_PREDICTORS_MONTHLY":
        return df[GW_COLUMN].sort_index().rename(f"{nombre}_{GW_COLUMN}")
    return df[nombre].sort_index()


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
    cat = yaml.safe_load((ROOT / "data" / "catalog.yaml").read_text(encoding="utf-8"))
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
# Defectos conocidos (documentados; corregirlos cambia los paneles del benchmark)
# --------------------------------------------------------------------------- #
@necesita_raw
@pytest.mark.xfail(strict=True, reason=(
    "alinear_diaria v1 pierde columnas fechadas fuera de la rejilla NYSE: ICSA (sábados) "
    "queda entera en NaN en pistaB_diaria. Fix: alinear_diaria(..., asof_por_columna=True)."))
def test_ninguna_columna_diaria_vacia(paneles):
    vacias = [c for c in paneles["B"]["diaria"] if paneles["B"]["diaria"][c].isna().all()]
    assert not vacias, vacias


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
@pytest.mark.xfail(strict=True, reason=(
    "Regla v1 de 02 §3.1: credit_BaaAaa_mensual_z/term_spread_hist_z (y resto de NOLAG_MENSUAL) "
    "se usan sin lag aunque son medias del mes fechadas el día 1 -> look-ahead de hasta ~1 mes. "
    "Corregirlo cambia los paneles consumidos por el benchmark 04 (ver 02 §3.2)."))
def test_mensuales_mercado_solo_visibles_tras_cerrar_el_mes(raw_por_pista, paneles):
    raw = raw_por_pista["A"]
    spread = (raw["MOODYS_BAA"] - raw["MOODYS_AAA"]).rename("credit_BaaAaa_mensual")
    correcto = ft.causal_zscore(ft.lag_publicacion(spread, 1))
    grid = paneles["A"]["mensual"].index
    esperado = correcto.reindex(grid, method="ffill")
    actual = paneles["A"]["mensual"]["credit_BaaAaa_mensual_z"]
    np.testing.assert_allclose(actual.values, esperado.values, equal_nan=True)
