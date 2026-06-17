"""Re-ejecuta los notebooks de la Tanda 1 con el núcleo corregido (Arreglos 1-3) y
compara las métricas con el backup en /tmp/t1_backup. Reporta qué columnas cambian.
"""
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NBS = ["01_rule_vix_threshold.ipynb", "03_clustering_gmm.ipynb", "04_hmm_gaussian_2s.ipynb"]
BACKUP = Path("/tmp/t1_backup")


def run_nb(name: str) -> int:
    nb = nbf.read(ROOT / "notebooks" / name, as_version=4)
    ep = ExecutePreprocessor(timeout=1200, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(ROOT / "notebooks")}})
    nbf.write(nb, ROOT / "notebooks" / name)
    return sum(1 for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error")


def diff_csv(csv_name: str) -> None:
    new = pd.read_csv(ROOT / "results" / csv_name)
    old = pd.read_csv(BACKUP / csv_name)
    print(f"\n### {csv_name}")
    changed = False
    for c in old.columns:
        ov, nv = old.iloc[0][c], new.iloc[0][c]
        if isinstance(ov, str) or isinstance(nv, str):
            same = str(ov) == str(nv)
        else:
            same = (pd.isna(ov) and pd.isna(nv)) or np.isclose(float(ov), float(nv), rtol=1e-9, atol=1e-12)
        if not same:
            changed = True
            print(f"   {c:24} {ov}  ->  {nv}")
    if not changed:
        print("   (sin cambios)")


for name in NBS:
    err = run_nb(name)
    print(f"[exec] {name}: errores={err}")

for csv in ["metrics_01_rule_vix_threshold.csv", "metrics_03_clustering_gmm.csv",
            "metrics_04_hmm_gaussian_2s.csv", "metrics_04_hmm_gaussian_2s_insample.csv"]:
    diff_csv(csv)

# Tabla maestra actualizada
import glob
dfs = [pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "results" / "metrics_0[134]_*.csv")))]
master = pd.concat(dfs, ignore_index=True)
master.to_csv(ROOT / "results" / "metrics_master.csv", index=False)
print("\nmaster regenerado:", master.shape)
