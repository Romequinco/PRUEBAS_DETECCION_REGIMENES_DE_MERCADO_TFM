"""Tests offline de src/ingest (sin red y sin tocar data/raw real).

Cubren los bugs corregidos en la revision de 00/01:
  - fechas de Shiller (octubre = AAAA.1 se leia como enero -> fechas duplicadas);
  - trimestre Goyal-Welch AAAAQ ('18711' se leia como el ano 18711);
  - guarda anti-degradacion: una re-descarga casi vacia no sobrescribe el cache;
  - `only` no borra las filas ERROR del coverage_report previo.
"""
from __future__ import annotations

import pandas as pd
import pytest
import yaml

from src.ingest import download, sources


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
    monkeypatch.setattr(download, "CATALOG", catalog)
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
