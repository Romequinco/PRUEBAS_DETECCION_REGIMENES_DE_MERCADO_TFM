"""Foto de referencia (golden baseline) previa a la unificación (ADR-004).

Genera en esta carpeta:
- results_hashes.json   hash de contenido de cada CSV/JSON versionado en results/
- panels_hashes.json    hash de contenido de cada panel OOS (gitignored) y de data/processed
- notebooks_inventory.json  por notebook: títulos, nº celdas, nº figuras, tablas, texto de markdown
- api_inventory.json    funciones y clases públicas de src/ y capa1_exploracion/detectors
- tests_inventory.txt   ids de pytest
- docs_links.json       enlaces relativos de los .md y si resuelven

Uso: python docs/revisiones/baseline/make_baseline.py [--out DIR]
El mismo script, re-ejecutado tras la unificación con --out a otro sitio, permite comparar.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def content_hash_frame(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()


def hash_file(path: Path) -> str:
    if path.suffix == ".csv":
        return content_hash_frame(pd.read_csv(path))
    if path.suffix == ".parquet":
        return content_hash_frame(pd.read_parquet(path))
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def results_hashes(results: Path) -> dict:
    out = {}
    for p in sorted(results.rglob("*")):
        if p.is_file() and p.suffix in {".csv", ".json"}:
            out[p.relative_to(results).as_posix()] = hash_file(p)
    return out


def panels_hashes(results: Path, processed: Path) -> dict:
    out = {}
    for base in (results, processed):
        for p in sorted(base.rglob("*.parquet")):
            out[f"{base.name}/{p.relative_to(base).as_posix()}"] = hash_file(p)
    return out


def notebook_inventory(nb_dir: Path) -> dict:
    inv = {}
    for nb_path in sorted(nb_dir.glob("*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        cells = nb["cells"]
        headings, md_texts = [], []
        n_img = n_tables = n_err = 0
        for c in cells:
            src = "".join(c.get("source", []))
            if c["cell_type"] == "markdown":
                md_texts.append(src)
                headings += [l.strip() for l in src.splitlines() if l.lstrip().startswith("#")]
            for o in c.get("outputs", []):
                data = o.get("data", {})
                n_img += int("image/png" in data or "image/svg+xml" in data)
                n_tables += int("text/html" in data)
                n_err += int(o.get("output_type") == "error")
        inv[nb_path.name] = {
            "n_cells": len(cells),
            "n_code": sum(c["cell_type"] == "code" for c in cells),
            "n_markdown": sum(c["cell_type"] == "markdown" for c in cells),
            "n_figures": n_img,
            "n_html_tables": n_tables,
            "n_errors": n_err,
            "headings": headings,
            "markdown": md_texts,
        }
    return inv


def api_inventory(paths: list[Path]) -> dict:
    out = {}
    for base in paths:
        for p in sorted(base.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            tree = ast.parse(p.read_text(encoding="utf-8"))
            names = [
                n.name for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and not n.name.startswith("_")
            ]
            consts = [
                t.id for n in tree.body if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name) and t.id.isupper()
            ]
            out[p.relative_to(ROOT).as_posix()] = {"defs": names, "constants": consts}
    return out


LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def docs_links() -> dict:
    out = {}
    for md in sorted(ROOT.rglob("*.md")):
        if any(part in {".git", "node_modules"} for part in md.parts):
            continue
        links = []
        for target in LINK.findall(md.read_text(encoding="utf-8", errors="replace")):
            if target.startswith(("http", "mailto:")):
                continue
            links.append({"target": target, "ok": (md.parent / target).resolve().exists()})
        if links:
            out[md.relative_to(ROOT).as_posix()] = links
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent))
    ap.add_argument("--results", default="results")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--notebooks", default="notebooks")
    ap.add_argument("--src", nargs="+", default=["src", "capa1_exploracion/detectors"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results, processed = ROOT / args.results, ROOT / args.processed

    dump = lambda name, obj: (out / name).write_text(
        json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    dump("results_hashes.json", results_hashes(results))
    dump("panels_hashes.json", panels_hashes(results, processed))
    dump("notebooks_inventory.json", notebook_inventory(ROOT / args.notebooks))
    dump("api_inventory.json", api_inventory([ROOT / s for s in args.src]))
    dump("docs_links.json", docs_links())
    tests = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                           cwd=ROOT, capture_output=True, text=True).stdout
    (out / "tests_inventory.txt").write_text(
        "\n".join(l for l in tests.splitlines() if "::" in l) + "\n", encoding="utf-8")
    print(f"baseline escrito en {out}")


if __name__ == "__main__":
    main()
