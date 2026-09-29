"""Tests offline de regimenes.datos (antes src/ingest; sin red y sin tocar data/raw real).

Cubren los bugs corregidos en la revision de 00/01:
  - fechas de Shiller (octubre = AAAA.1 se leia como enero -> fechas duplicadas);
  - trimestre Goyal-Welch AAAAQ ('18711' se leia como el ano 18711);
  - guarda anti-degradacion: una re-descarga casi vacia no sobrescribe el cache;
  - `only` no borra las filas ERROR del coverage_report previo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from regimenes.datos import catalogo
from regimenes.datos import descarga as download, fuentes as sources


def test_shiller_dates_october_is_not_january():
    col = pd.Series([1871.01, 1871.09, 1871.1, 1871.11, 1871.12, 1872.01, None, "x"])
    idx = sources._shiller_dates(col)
    got = [d.strftime("%Y-%m") if pd.notna(d) else None for d in idx]
    assert got == ["1871-01", "1871-09", "1871-10", "1871-11", "1871-12", "1872-01", None, None]
    assert not idx.dropna().duplicated().any()


def test_gw_quarter_index_parses_yyyyq():
    raw = pd.Series(["18711", "18714", "20253", "bad"])
    idx = sources._gw_quarter_index(raw)
    assert [d.strftime("%Y-%m") if pd.notna(d) else None for d in idx] == [
        "1871-01", "1871-10", "2025-07", None]


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """Catalogo minimo + data/raw aislados en tmp_path."""
    raw = tmp_path / "raw"
    cat = {
        "pista_A": {"series": [
            {"nombre_interno": "GOOD", "fuente": "fred", "id": "G", "pista": "A", "rol": "core"},
            {"nombre_interno": "BAD", "fuente": "fred", "id": "B", "pista": "A", "rol": "enricher"},
        ]},
        "pista_B": {"series": []},
        "validacion_externa": {"series": []},
    }
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(yaml.safe_dump(cat), encoding="utf-8")
    monkeypatch.setattr(download, "RAW", raw)
    monkeypatch.setattr(catalogo, "CATALOG", catalog)  # load_catalog lo lee de catalogo
    return raw


def _series(n, name):
    return pd.Series(range(n), index=pd.bdate_range("2000-01-03", periods=n), name=name, dtype=float)


def test_degraded_redownload_keeps_cache(fake_repo, monkeypatch):
    calls = {"n": 100}

    def fetch(fuente, sid, url=None):
        if sid == "B":
            raise RuntimeError("fuente caida")
        return _series(calls["n"], sid)

    monkeypatch.setattr(sources, "fetch", fetch)
    rep = download.download_all()
    assert rep.set_index("nombre").loc["GOOD", "status"] == "OK"
    assert rep.set_index("nombre").loc["BAD", "status"] == "ERROR"

    # la fuente devuelve 1 sola fila (caso real yfinance ^VIX9D) -> no sobrescribe
    calls["n"] = 1
    rep2 = download.download_all(force=True).set_index("nombre")
    assert rep2.loc["GOOD", "status"] == "CACHE"
    assert "aviso" in rep2.columns and isinstance(rep2.loc["GOOD", "aviso"], str)
    kept = pd.read_parquet(fake_repo / "fred" / "GOOD.parquet")
    assert len(kept) == 100

    # recorte moderado (> MIN_FRAC_VS_CACHE) si se acepta
    calls["n"] = 90
    rep3 = download.download_all(force=True).set_index("nombre")
    assert rep3.loc["GOOD", "status"] == "OK" and rep3.loc["GOOD", "n_obs"] == 90


def test_only_preserves_previous_error_rows(fake_repo, monkeypatch):
    def fetch(fuente, sid, url=None):
        if sid == "B":
            raise RuntimeError("fuente caida")
        return _series(50, sid)

    monkeypatch.setattr(sources, "fetch", fetch)
    download.download_all()
    rep = download.download_all(force=True, only=["GOOD"]).set_index("nombre")
    assert set(rep.index) == {"GOOD", "BAD"}
    assert rep.loc["BAD", "status"] == "ERROR"


# --------------------------------------------------------------------------- #
# ADR-003: ingesta de Ken French multi-columna, vol realizada, columna github, offline
# --------------------------------------------------------------------------- #
def _zip_bytes(txt: str) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.CSV", txt)
    return buf.getvalue()


_FRENCH_IND = """This file was created using the 202608 CRSP database.
Missing data are indicated by -99.99 or -999.

  Average Value Weighted Returns -- Daily
,Cnsmr,Manuf,HiTec,Hlth,Other
19260701,   0.10,   0.20, -99.99,   0.40,   0.50
19260702,   0.11,   0.21,   0.31,   0.41,   0.51

  Average Equal Weighted Returns -- Daily
,Cnsmr,Manuf,HiTec,Hlth,Other
19260701,   9.00,   9.00,   9.00,   9.00,   9.00
"""

_FRENCH_MOM = """Momentum
,Mom
19261103,   0.56
19261104,  -0.50
"""


def test_french_zip_guarda_todas_las_columnas_del_primer_bloque(monkeypatch):
    monkeypatch.setattr(sources, "_get", lambda url, timeout=60: _zip_bytes(_FRENCH_IND))
    df = sources._french_zip("5_Industry_Portfolios_daily")
    assert list(df.columns) == ["Cnsmr", "Manuf", "HiTec", "Hlth", "Other"]
    assert len(df) == 2 and pd.isna(df.loc["1926-07-01", "HiTec"])  # -99.99 -> NaN
    assert df.loc["1926-07-02", "Other"] == 0.51  # bloque value-weighted, no el equal-weighted


def test_french_zip_una_columna_sigue_siendo_series(monkeypatch):
    monkeypatch.setattr(sources, "_get", lambda url, timeout=60: _zip_bytes(_FRENCH_MOM))
    s = sources._french_zip("F-F_Momentum_Factor_daily")
    assert isinstance(s, pd.Series) and list(s.values) == [0.56, -0.50]


def test_github_csv_columna_concreta_trata_cero_como_faltante(monkeypatch):
    csv = b"Date,SP500,PE10\n2023-08-01,4457.3,30.47\n2023-09-01,4515.7,30.81\n2023-10-01,4200.0,0.0\n"
    monkeypatch.setattr(sources, "_get", lambda url, timeout=30, retries=3: csv)
    s = sources.fetch_github_csv("datasets/s-and-p-500:data/data.csv", columna="PE10")
    assert list(s.values) == [30.47, 30.81]
    assert sources.fetch_github_csv("datasets/s-and-p-500:data/data.csv").name == "SP500"


@pytest.fixture
def fake_repo_ff(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    cat = {
        "pista_A": {"series": [
            {"nombre_interno": "FF_5_INDUSTRY", "fuente": "academico", "id": "5_Industry_Portfolios_daily",
             "pista": "A", "rol": "enricher"},
            {"nombre_interno": "REALIZED_VOL_SP500", "fuente": "yfinance", "id": "^GSPC",
             "pista": "A", "rol": "spine"},
            {"nombre_interno": "MISSING", "fuente": "fred", "id": "M", "pista": "A", "rol": "enricher"},
        ]},
        "pista_B": {"series": []},
        "validacion_externa": {"series": []},
    }
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(yaml.safe_dump(cat), encoding="utf-8")
    monkeypatch.setattr(download, "RAW", raw)
    monkeypatch.setattr(catalogo, "CATALOG", catalog)  # load_catalog lo lee de catalogo
    return raw


def test_download_ff_alias_y_vol_realizada(fake_repo_ff, monkeypatch):
    idx = pd.bdate_range("2000-01-03", periods=300)
    ind = pd.DataFrame({"Cnsmr": 1.0, "Manuf": 2.0, "HiTec": 3.0, "Hlth": 4.0, "Other": 5.0}, index=idx)
    precio = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 300))), index=idx)

    def fetch(fuente, sid, url=None):
        if sid == "M":
            raise RuntimeError("no disponible")
        return ind if fuente == "academico" else precio

    monkeypatch.setattr(sources, "fetch", fetch)
    rep = download.download_all(only=["FF_5_INDUSTRY", "REALIZED_VOL_SP500", "MISSING"],
                                force=True).set_index("nombre")
    ff = pd.read_parquet(fake_repo_ff / "academico" / "FF_5_INDUSTRY.parquet")
    assert list(ff.columns) == ["FF_5_INDUSTRY", "Cnsmr", "Manuf", "HiTec", "Hlth", "Other"]
    assert (ff["FF_5_INDUSTRY"] == ff["Cnsmr"]).all() and rep.loc["FF_5_INDUSTRY", "n_cols"] == 6
    rv = pd.read_parquet(fake_repo_ff / "yfinance" / "REALIZED_VOL_SP500.parquet")["REALIZED_VOL_SP500"]
    from regimenes import features as ft
    esperado = ft.realized_vol(ft.log_returns(precio)).dropna()
    np.testing.assert_allclose(rv.values, esperado.values)
    assert rv.max() < 1.0  # una vol anualizada, no un precio


def test_download_offline_no_llama_a_la_red(fake_repo_ff, monkeypatch):
    def fetch(*a, **k):
        raise AssertionError("offline no debe descargar")

    idx = pd.bdate_range("2000-01-03", periods=50)
    (fake_repo_ff / "yfinance").mkdir(parents=True)
    pd.DataFrame({"REALIZED_VOL_SP500": np.linspace(0.1, 0.2, 50)}, index=idx).to_parquet(
        fake_repo_ff / "yfinance" / "REALIZED_VOL_SP500.parquet")
    monkeypatch.setattr(sources, "fetch", fetch)
    rep = download.download_all(offline=True, force=True).set_index("nombre")
    assert rep.loc["REALIZED_VOL_SP500", "status"] == "CACHE"
    assert rep.loc["FF_5_INDUSTRY", "status"] == "ERROR" and rep.loc["MISSING", "status"] == "ERROR"
