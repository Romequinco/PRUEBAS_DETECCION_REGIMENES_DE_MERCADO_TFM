"""
download.py — Orquestador de descarga v2, dirigido por data/catalog.yaml.

Recorre TODAS las series del catalogo (pista_A + pista_B + validacion_externa),
descarga cada una por su fuente, la guarda SIN imputar en
data/raw/<fuente>/<nombre_interno>.parquet, y escribe:
  - data/raw/provenance.json     (por serie: fuente, id, status, checksum, ventana)
  - data/raw/coverage_report.csv (tabla de cobertura: inicio/fin/n_obs por serie)

Resiliente: un fallo por serie se registra (status=ERROR) y NO aborta el resto.
Cacheo: si el parquet ya existe y force=False, se salta (usa el cache).
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import pandas as pd
import yaml

from . import sources

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CATALOG = ROOT / "data" / "catalog.yaml"


def load_catalog() -> dict:
    return yaml.safe_load(CATALOG.read_text(encoding="utf-8"))


def _iter_series(cat: dict):
    """Genera (nombre_interno, entrada) deduplicando por nombre (1a aparicion gana)."""
    seen: set[str] = set()
    for pista in ("pista_A", "pista_B", "validacion_externa"):
        block = cat.get(pista) or {}
        for s in block.get("series", []) or []:
            n = s.get("nombre_interno")
            if not n or n in seen:
                continue
            seen.add(n)
            yield n, s


def _checksum(s: pd.Series) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(s, index=True).values.tobytes()).hexdigest()[:16]


def download_all(force: bool = False, only: list[str] | None = None) -> pd.DataFrame:
    """Descarga todo el catalogo. Devuelve el coverage_report como DataFrame."""
    cat = load_catalog()
    RAW.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, dict] = {}
    rows: list[dict] = []

    series_list = list(_iter_series(cat))
    total = len(series_list)
    for i, (nombre, entry) in enumerate(series_list, 1):
        fuente = entry.get("fuente", "?")
        sid = entry.get("id", "")
        pista = entry.get("pista", "?")
        rol = entry.get("rol", "?")
        if only and nombre not in only:
            continue
        outdir = RAW / fuente
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / f"{nombre}.parquet"

        rec = {"nombre": nombre, "fuente": fuente, "id": sid, "pista": pista, "rol": rol}
        if out.exists() and not force:
            try:
                s = pd.read_parquet(out).iloc[:, 0]
                rec.update(_meta_from_series(s, cached=True))
                provenance[nombre] = {**rec, "status": "CACHE", "checksum": _checksum(s)}
                rows.append(rec | {"status": "CACHE"})
                print(f"[{i}/{total}] CACHE  {nombre:28} ({fuente})")
                continue
            except Exception:  # noqa: BLE001  (cache corrupto -> re-descarga)
                pass
        try:
            obj = sources.fetch(fuente, sid, url=entry.get("url"))
            obj = obj.sort_index()
            if isinstance(obj, pd.DataFrame):
                df_out = obj                          # panel multi-columna (GW, JST)
                rep = obj.iloc[:, 0]
                rec["n_cols"] = obj.shape[1]
            else:
                obj.name = nombre
                df_out = pd.DataFrame({nombre: obj})  # serie simple
                rep = obj
            df_out.to_parquet(out)
            rec.update(_meta_from_series(rep, cached=False))
            provenance[nombre] = {**rec, "status": "OK", "checksum": _checksum(rep), "url": entry.get("url")}
            rows.append(rec | {"status": "OK"})
            print(f"[{i}/{total}] OK     {nombre:28} ({fuente}) {rec['inicio']}->{rec['fin']} n={rec['n_obs']}")
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            provenance[nombre] = {**rec, "status": "ERROR", "error": msg, "url": entry.get("url")}
            rows.append(rec | {"status": "ERROR", "n_obs": 0, "inicio": None, "fin": None, "error": msg})
            print(f"[{i}/{total}] ERROR  {nombre:28} ({fuente}) -> {msg[:70]}")

    (RAW / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = pd.DataFrame(rows)
    cols = ["nombre", "fuente", "id", "pista", "rol", "status", "granularidad", "inicio", "fin", "n_obs"]
    report = report.reindex(columns=[c for c in cols if c in report.columns] +
                            [c for c in report.columns if c not in cols])
    report.to_csv(RAW / "coverage_report.csv", index=False, encoding="utf-8")
    return report


def _meta_from_series(s: pd.Series, cached: bool) -> dict:
    freq = "mensual" if len(s) > 1 and s.index.to_series().diff().median() > pd.Timedelta(days=20) else "diaria"
    return {
        "granularidad": freq,
        "inicio": s.index.min().date().isoformat(),
        "fin": s.index.max().date().isoformat(),
        "n_obs": int(len(s)),
    }


if __name__ == "__main__":  # pragma: no cover
    import sys

    force = "--force" in sys.argv
    rep = download_all(force=force)
    ok = (rep["status"].isin(["OK", "CACHE"])).sum()
    print(f"\n=== RESUMEN: {ok}/{len(rep)} series disponibles ===")
    print(rep["status"].value_counts().to_string())
