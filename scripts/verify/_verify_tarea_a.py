"""Verifica que el etiquetado refinado (market_returns explícito en walk_forward)
NO cambia las métricas de la Tanda 1. Re-ejecuta D1/D3/D4 por la ruta nueva y
compara con los CSV guardados en results/.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from src import features as ft, evaluation as ev  # noqa: E402
from detectors.rule_vix_threshold import RuleVixThreshold  # noqa: E402
from detectors.clustering_gmm import ClusteringGMM  # noqa: E402
from detectors.hmm_gaussian_2s import HMMGaussian2S, BRIDGE_FEATURES  # noqa: E402

raw = pd.read_parquet(ROOT / "data" / "raw" / "raw_panel.parquet")
feats = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
mkt_full = np.log(raw["SP500"] / raw["SP500"].shift(1))


def compare(name, new_row: pd.DataFrame, csv_path: Path) -> bool:
    old = pd.read_csv(csv_path)
    new = new_row.reset_index(drop=True)
    cols = list(old.columns)
    ok = True
    diffs = []
    for c in cols:
        ov, nv = old.iloc[0][c], new.iloc[0][c]
        if isinstance(ov, str) or isinstance(nv, str):
            same = str(ov) == str(nv)
        else:
            same = (pd.isna(ov) and pd.isna(nv)) or np.isclose(float(ov), float(nv), rtol=1e-9, atol=1e-12)
        if not same:
            ok = False
            diffs.append(f"{c}: old={ov} new={nv}")
    print(f"[{'IDÉNTICO' if ok else 'DIFERENCIAS'}] {name}")
    for d in diffs:
        print("    ", d)
    return ok


# --- D1 ---------------------------------------------------------------------
vix = raw["VIX"].dropna()
vix_z = ft.causal_zscore(vix.rename("VIX_level"))
spx_ret = np.log(raw["SP500"] / raw["SP500"].shift(1)).rename("SP500_ret")
X1 = pd.DataFrame({"VIX_level_z": vix_z, "SP500_ret": spx_ret.reindex(vix_z.index)}).dropna().sort_index()
panel1 = ev.walk_forward(lambda: RuleVixThreshold(q_in=0.90, q_out=0.70, min_dwell=5),
                         X1, market_returns=X1["SP500_ret"], train_size=252 * 8, step=21)
det1 = RuleVixThreshold(q_in=0.90, q_out=0.70, min_dwell=5).fit(X1)
res1 = ev.evaluate(det1, panel1, market_returns=X1["SP500_ret"].reindex(panel1.index), X_full=X1)
ok1 = compare("D1 rule_vix_threshold", ev.results_table([res1]), ROOT / "results" / "metrics_01_rule_vix_threshold.csv")

# --- D3 ---------------------------------------------------------------------
X3 = feats.copy()
mr3 = mkt_full.reindex(X3.index)
panel3 = ev.walk_forward(lambda: ClusteringGMM(n_states=3), X3, market_returns=mr3, train_size=252 * 8, step=21)
det3 = ClusteringGMM(n_states=3).fit(X3)
res3 = ev.evaluate(det3, panel3, market_returns=mr3, X_full=X3)
ok3 = compare("D3 clustering_gmm_k3", ev.results_table([res3]), ROOT / "results" / "metrics_03_clustering_gmm.csv")

# --- D4 (causal) ------------------------------------------------------------
X4 = feats[BRIDGE_FEATURES].copy()
mr4 = mkt_full.reindex(X4.index)
panel4 = ev.walk_forward(lambda: HMMGaussian2S(n_states=2, n_init=5), X4, market_returns=mr4, train_size=252 * 5, step=21)
det4 = HMMGaussian2S(n_states=2, n_init=5).fit(X4)
det4.label_states_economically(X4, market_returns=mr4)
res4 = ev.evaluate(det4, panel4, market_returns=mr4, X_full=X4)
ok4 = compare("D4 hmm_gaussian_2s (causal)", ev.results_table([res4]), ROOT / "results" / "metrics_04_hmm_gaussian_2s.csv")

print("\nRESULTADO:", "TODAS IDÉNTICAS" if (ok1 and ok3 and ok4) else "HAY DIFERENCIAS (revisar arriba)")
