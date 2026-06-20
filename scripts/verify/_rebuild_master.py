"""Reconstruye results/metrics_master.csv desde los 12 CSV individuales de detector.

Fuente de verdad = cada results/metrics_NN_*.csv (esquema unificado tras la
re-ejecución). Evita la lógica incremental frágil (solo 6 notebooks mantenían el
master) que dejaba filas con esquemas mezclados. Excluye variantes no canónicas
(_insample, pca_gmm_baseline). Mantiene el orden D1..D12.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"

FILES = [
    "metrics_01_rule_vix_threshold.csv",
    "metrics_02_rule_composite_riskoff.csv",
    "metrics_03_clustering_gmm.csv",
    "metrics_04_hmm_gaussian_2s.csv",
    "metrics_05_markov_switching_var.csv",
    "metrics_06_garch_t_vol.csv",
    "metrics_07_changepoint_online.csv",
    "metrics_08_hmm_tstudent.csv",
    "metrics_09_jump_model.csv",
    "metrics_10_turbulence_mahalanobis.csv",
    "metrics_11_msgarch_regime.csv",
    "metrics_12_deep_ae_regime.csv",
]

EXCLUDE_SUBSTR = ("baseline", "INSAMPLE", "pca_gmm")


def main() -> None:
    rows = []
    for f in FILES:
        df = pd.read_csv(RES / f)
        df = df[~df["detector"].astype(str).str.contains("|".join(EXCLUDE_SUBSTR), case=False)]
        if len(df) == 0:
            raise SystemExit(f"{f}: sin fila canónica tras excluir variantes")
        rows.append(df.iloc[[0]])
    master = pd.concat(rows, ignore_index=True)
    out = RES / "metrics_master.csv"
    master.to_csv(out, index=False)
    print(f"master reconstruido: {master.shape} -> {out}")
    print("detectores:", list(master["detector"]))
    n_new = [c for c in master.columns if c.endswith(("_lo", "_hi")) or c == "silhouette"]
    print(f"columnas nuevas presentes: {len(n_new)} | total cols: {master.shape[1]}")


if __name__ == "__main__":
    main()
