"""
download.py — Orquestador de descarga v2, dirigido por data/catalog.yaml.

Recorre TODAS las series del catalogo (pista_A + pista_B + validacion_externa),
descarga cada una por su fuente, la guarda SIN imputar en
data/raw/<fuente>/<nombre_interno>.parquet, y escribe:
  - data/raw/provenance.json     (por serie: fuente, id, status, checksum, ventana)
  - data/raw/coverage_report.csv (tabla de cobertura: inicio/fin/n_obs por serie)

Resiliente: un fallo por serie se registra (status=ERROR) y NO aborta el resto.
Cacheo: si el parquet ya existe y force=False, se salta (usa el cache).
Guarda anti-degradacion: con force=True, una descarga que devuelve MUCHA menos historia
que el cache en disco (p. ej. yfinance devolviendo 1 sola fila para ^VIX9D) NO lo
sobrescribe: se conserva el cache y se registra un aviso (status=CACHE, campo `aviso`).
`only`: acota que se (re)descarga; las series no pedidas y ausentes de disco conservan
su fila del coverage_report anterior (asi un `only` nunca borra los ERROR declarados).
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

# Una re-descarga con menos de esta fraccion de las observaciones del cache se
# considera degradada (fallo silencioso de la fuente) y no sobrescribe el cache.
MIN_FRAC_VS_CACHE = 0.5
MIN_OBS = 2


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
    prev_rows = _previous_report()
    prev_prov = _previous_provenance()
    for i, (nombre, entry) in enumerate(series_list, 1):
        fuente = entry.get("fuente", "?")
        sid = entry.get("id", "")
        pista = entry.get("pista", "?")
        rol = entry.get("rol", "?")
        outdir = RAW / fuente
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / f"{nombre}.parquet"

        rec = {"nombre": nombre, "fuente": fuente, "id": sid, "pista": pista, "rol": rol}
        # `only` acota QUE se (re)descarga, pero las demas series ya en disco SIGUEN
        # registrandose como CACHE -> `only` nunca vacia el coverage_report.
        targeted = (not only) or (nombre in only)
        cached = None
        if out.exists():
            try:
                cached = pd.read_parquet(out).iloc[:, 0]
            except Exception:  # noqa: BLE001  (cache corrupto -> re-descarga)
                cached = None
        if cached is not None and (not force or not targeted):
            _register_cache(rec, cached, provenance, rows, entry)
            print(f"[{i}/{total}] CACHE  {nombre:28} ({fuente})")
            continue
        if not targeted:
            # no esta en disco y no se ha pedido -> conserva su fila previa (si la habia)
            if nombre in prev_rows:
                rows.append(prev_rows[nombre])
            if nombre in prev_prov:
                provenance[nombre] = prev_prov[nombre]
            continue
        try:
            obj = sources.fetch(fuente, sid, url=entry.get("url"))
            obj = obj.sort_index()
            rep_new = obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj
            n_new = int(rep_new.dropna().shape[0])
            if n_new < MIN_OBS:
                raise RuntimeError(f"descarga degenerada: {n_new} observaciones")
            if cached is not None and n_new < MIN_FRAC_VS_CACHE * len(cached):
                aviso = (f"re-descarga degradada ({n_new} obs vs {len(cached)} en cache): "
                         "se conserva el cache")
                _register_cache(rec, cached, provenance, rows, entry, aviso=aviso)
                print(f"[{i}/{total}] AVISO  {nombre:28} ({fuente}) -> {aviso}")
                continue
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
            if cached is not None:  # fallo al refrescar, pero hay cache valido -> se usa
                _register_cache(rec, cached, provenance, rows, entry,
                                aviso=f"re-descarga fallida ({msg[:80]}): se conserva el cache")
                print(f"[{i}/{total}] AVISO  {nombre:28} ({fuente}) -> {msg[:70]} (se usa cache)")
                continue
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


def _register_cache(rec: dict, s: pd.Series, provenance: dict, rows: list,
                    entry: dict, aviso: str | None = None) -> None:
    """Registra una serie servida desde el parquet en disco (status=CACHE)."""
    rec.update(_meta_from_series(s, cached=True))
    extra = {"aviso": aviso} if aviso else {}
    provenance[rec["nombre"]] = {**rec, "status": "CACHE", "checksum": _checksum(s),
                                 "url": entry.get("url"), **extra}
    rows.append(rec | {"status": "CACHE", **extra})


def _previous_report() -> dict[str, dict]:
    """Filas del coverage_report.csv anterior (para no perderlas en un `only`)."""
    f = RAW / "coverage_report.csv"
    if not f.exists():
        return {}
    try:
        prev = pd.read_csv(f)
    except Exception:  # noqa: BLE001
        return {}
    prev = prev.astype(object).where(prev.notna(), None)
    return {r["nombre"]: {k: v for k, v in r.items() if v is not None}
            for r in prev.to_dict("records")}


def _previous_provenance() -> dict[str, dict]:
    f = RAW / "provenance.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _meta_from_series(s: pd.Series, cached: bool) -> dict:
    # Heuristica binaria diaria/mensual (espaciado mediano > 20 dias -> mensual).
    # Limitacion conocida: las series SEMANALES (NFCI y subindices, STLFSI4, ICSA) quedan
    # etiquetadas 'diaria'; 01_eda §1 las detecta y 02_diseno_preprocesado las trata
    # aparte filtrando granularidad=='diaria' -> no se cambia aqui para no romper ese
    # contrato aguas abajo.
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
