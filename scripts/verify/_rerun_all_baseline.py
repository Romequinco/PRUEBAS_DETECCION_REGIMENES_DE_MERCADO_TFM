"""TANDA 1 — Re-ejecuta los 14 notebooks en orden y verifica reproducibilidad.

Respalda results/*.csv, re-ejecuta 00->13 en orden, y compara cada CSV trackeado
contra el backup reportando deriva numérica. No depende de red (lee parquet local).

Uso:  python -m scripts.verify._rerun_all_baseline
Salida: _baseline_report.md en la raíz + códigos de error por notebook.
"""
import shutil
import time
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[2]
NB_DIR = ROOT / "notebooks"
RES = ROOT / "results"
BACKUP = ROOT / "_baseline_backup"

ORDER = [
    "00_eda.ipynb",
    "01_rule_vix_threshold.ipynb",
    "02_rule_composite_riskoff.ipynb",
    "03_clustering_gmm.ipynb",
    "04_hmm_gaussian_2s.ipynb",
    "05_markov_switching_var.ipynb",
    "06_garch_t_vol.ipynb",
    "07_changepoint_online.ipynb",
    "08_hmm_tstudent.ipynb",
    "09_jump_model.ipynb",
    "10_turbulence_mahalanobis.ipynb",
    "11_msgarch_regime.ipynb",
    "12_deep_ae_regime.ipynb",
    "13_comparison.ipynb",
]


def backup_csvs() -> None:
    BACKUP.mkdir(exist_ok=True)
    for csv in RES.glob("*.csv"):
        shutil.copy2(csv, BACKUP / csv.name)


def run_nb(name: str) -> tuple[int, float]:
    t0 = time.time()
    nb = nbf.read(NB_DIR / name, as_version=4)
    ep = ExecutePreprocessor(timeout=2400, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    nbf.write(nb, NB_DIR / name)
    errs = sum(
        1 for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"
    )
    return errs, time.time() - t0


def diff_csv(name: str) -> list[str]:
    old_p, new_p = BACKUP / name, RES / name
    if not old_p.exists():
        return [f"   (NUEVO csv, sin backup)"]
    old, new = pd.read_csv(old_p), pd.read_csv(new_p)
    out = []
    if list(old.columns) != list(new.columns):
        out.append(f"   columnas cambiaron: {set(new.columns) ^ set(old.columns)}")
    common = [c for c in old.columns if c in new.columns]
    n = min(len(old), len(new))
    for i in range(n):
        for c in common:
            ov, nv = old.iloc[i][c], new.iloc[i][c]
            if isinstance(ov, str) or isinstance(nv, str):
                same = str(ov) == str(nv)
            else:
                same = (pd.isna(ov) and pd.isna(nv)) or np.isclose(
                    float(ov), float(nv), rtol=1e-6, atol=1e-9
                )
            if not same:
                out.append(f"   fila{i} {c:24} {ov}  ->  {nv}")
    return out


def main() -> None:
    backup_csvs()
    report = ["# TANDA 1 — Baseline de re-ejecución\n"]
    total_err = 0
    for name in ORDER:
        try:
            errs, dt = run_nb(name)
            total_err += errs
            report.append(f"- **{name}** — {dt:5.1f}s, errores={errs}")
            print(f"[OK] {name} {dt:.1f}s err={errs}", flush=True)
        except Exception as e:  # noqa: BLE001
            total_err += 1
            report.append(f"- **{name}** — FALLO: {type(e).__name__}: {e}")
            print(f"[FAIL] {name}: {e}", flush=True)

    report.append("\n## Deriva de métricas (CSV trackeados)\n")
    any_drift = False
    for csv in sorted(RES.glob("*.csv")):
        diffs = diff_csv(csv.name)
        if diffs:
            any_drift = True
            report.append(f"### {csv.name}")
            report.extend(diffs)
    if not any_drift:
        report.append("**Deriva CERO en todos los CSV — reproducibilidad confirmada.**")
    report.append(f"\n## Total errores de celda: {total_err}")
    (ROOT / "_baseline_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n=== DONE total_err={total_err} any_drift={any_drift} ===", flush=True)


if __name__ == "__main__":
    main()
