"""Verificación de NO-REGRESIÓN del Arreglo 4 (_economic_state_order vol-primario)
+ limpieza del override de D6.

- Re-ejecuta los notebooks 01,02,03,04,06 con el núcleo nuevo y compara las filas de
  results/ con el backup en _t12_backup/. Espera: TODOS idénticos salvo, como mucho,
  D6 (que tras quitar su override debe quedar IGUAL).
- D5 (MS, ~33 min) no se re-ejecuta: se comprueba que el núcleo nuevo da crisis=alta
  varianza en varias ventanas de fold => orden canónico idéntico => métricas iguales.
"""
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
BK = ROOT / "_t12_backup"


def run_nb(name: str) -> int:
    nb = nbf.read(ROOT / "notebooks" / name, as_version=4)
    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(ROOT / "notebooks")}})
    nbf.write(nb, ROOT / "notebooks" / name)
    return sum(1 for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error")


def diff_csv(csv_name: str) -> bool:
    new = pd.read_csv(ROOT / "results" / csv_name)
    old = pd.read_csv(BK / csv_name)
    changed = []
    for c in old.columns:
        ov, nv = old.iloc[0][c], new.iloc[0][c]
        same = (str(ov) == str(nv)) if (isinstance(ov, str) or isinstance(nv, str)) else (
            (pd.isna(ov) and pd.isna(nv)) or np.isclose(float(ov), float(nv), rtol=1e-9, atol=1e-12))
        if not same:
            changed.append(f"{c}: {ov} -> {nv}")
    print(f"### {csv_name}: {'IDÉNTICO' if not changed else 'CAMBIA'}", flush=True)
    for c in changed:
        print("    ", c, flush=True)
    return not changed


NBS = {
    "01_rule_vix_threshold.ipynb": "metrics_01_rule_vix_threshold.csv",
    "02_rule_composite_riskoff.ipynb": "metrics_02_rule_composite_riskoff.csv",
    "03_clustering_gmm.ipynb": "metrics_03_clustering_gmm.csv",
    "04_hmm_gaussian_2s.ipynb": "metrics_04_hmm_gaussian_2s.csv",
    "06_garch_t_vol.ipynb": "metrics_06_garch_t_vol.csv",
}
allok = True
for nb_name, csv in NBS.items():
    err = run_nb(nb_name)
    print(f"[exec] {nb_name}: errores={err}", flush=True)
    allok &= diff_csv(csv)

# --- D5: prueba de equivalencia de orden (sin re-run de 33 min) -------------- #
print("\n### D5 markov_switching_var — equivalencia de orden (crisis=alta varianza)", flush=True)
import sys
sys.path.insert(0, str(ROOT))
from detectors.markov_switching_var import MarkovSwitchingVar  # noqa: E402
raw = pd.read_parquet(ROOT / "data" / "raw" / "raw_panel.parquet")
spx_ret = np.log(raw["SP500"] / raw["SP500"].shift(1)).rename("SP500_ret")
X = pd.DataFrame({"SP500_ret": spx_ret}).dropna()
mr = X["SP500_ret"]
ok5 = True
for cut in (252 * 8, 252 * 16, 252 * 24, len(X)):
    Xtr = X.iloc[:cut]
    det = MarkovSwitchingVar(n_states=2).fit(Xtr)
    det.label_states_economically(Xtr, market_returns=mr.reindex(Xtr.index))
    v = det.variances_canonical()
    crisis_is_high_var = bool(v[det.crisis_state] == v.max())
    ok5 &= crisis_is_high_var
    print(f"   train[:{cut}] vars_canon={np.round(v,3)} crisis=alta_var:{crisis_is_high_var}", flush=True)
print(f"D5 orden equivalente (crisis=alta varianza en todas las ventanas): {ok5}", flush=True)

print(f"\nNO-REGRESIÓN GLOBAL (01-04,06 idénticos): {allok} | D5 orden OK: {ok5}", flush=True)
