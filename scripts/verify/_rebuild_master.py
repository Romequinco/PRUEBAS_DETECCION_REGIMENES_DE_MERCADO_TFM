"""Reconstruye results/metrics_master.csv (ESQUEMA CANÓNICO UNIFICADO) desde los
12 CSV individuales de detector + metadatos estáticos.

CONTEXTO (saneamiento Ola 0)
----------------------------
Antes convivían DOS masters con esquemas complementarios:
  · metrics_master.csv        (32 cols): silhouette + IC bootstrap de cobertura.
  · metrics_master_final.csv  (34 cols): clase, coste, vio_2008_oos, *_estres, nota.
Se unifican en UN ÚNICO master canónico = SUPERSET de ambos (43 columnas).
`results/metrics_master_final.csv` queda archivado en results/_archive/ y el
ÚNICO canónico pasa a ser results/metrics_master.csv (lo leen TODOS los scripts
de figuras: build_pdf_figs.py y build_synthesis_figs.py).

ESQUEMA CANÓNICO (43 columnas), por grupo y procedencia:
  Identidad / equidad (metadatos):
    detector, n_states            -> de cada metrics_NN_*.csv
    clase                         -> dict CLASE  (estático, copiado de _build_13)
    coste                         -> dict COSTE  (estático, copiado de _build_13)
    vio_2008_oos                  -> derivado: oos_start < 2008-09-01
    ventana_eval, oos_start, oos_end, n_oos   -> de cada metrics_NN_*.csv
  Tasas globales:
    false_alarm_rate              -> de cada metrics_NN_*.csv
    false_alarm_rate_estres       -> binarios: = false_alarm_rate ; multi: MULTI_ESTRES
  Dinámica / separación:
    switching_rate, mean_regime_duration, label_stability, silhouette  -> CSV
  Ajuste (solo generativos):
    log_likelihood, aic, bic      -> de cada metrics_NN_*.csv
  Cobertura crisis estricta + IC bootstrap por bloques:
    cov_{W}, cov_{W}_lo, cov_{W}_hi  (W = GFC_2008, EuroDebt_2011, COVID_2020,
                                      Inflation_2022)  -> de cada CSV
  Cobertura ESTRÉS agregado (severidad alta = unión de los 2 estados más severos):
    cov_estres_{W}                -> binarios (n_states==2): copia de cov_{W};
                                     multi-estado (D3,D8,D12): dict MULTI_ESTRES
                                     (recomputado en _build_13 con walk-forward; aquí
                                     se incrusta como metadato para NO re-ejecutar).
  Falsas alarmas en trampas (2013/2018), estricta y estrés:
    fa_{F}, fa_estres_{F}         (F = TaperTantrum_2013, Selloff_Q4_2018)
                                     -> fa_{F} del CSV; fa_estres_{F} = fa_{F} en
                                        binarios, MULTI_ESTRES en multi-estado.
  Lead/lag por evento:
    leadlag_{W}                   -> de cada metrics_NN_*.csv
  Nota:
    nota                          -> derivado: nota_estres(name, n_states)

IMPORTANTE: este script NO re-ejecuta detectores ni walk-forward. Lee los
metrics_NN_*.csv que YA existen. Las columnas de ESTRÉS de los 3 multi-estado
(D3 clustering_gmm_k3, D8 hmm_tstudent_4s, D12 deep_ae_regime) son metadatos
incrustados (MULTI_ESTRES) provenientes del recompute walk-forward de _build_13.
Si en una re-ejecución (Ola 2) cambian esos números, refrescar MULTI_ESTRES.

Excluye variantes no canónicas (_insample, pca_gmm_baseline). Mantiene D1..D12.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"

# Un CSV por detector (la primera fila canónica de cada uno; orden D1..D12).
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

CRISIS = ["GFC_2008", "EuroDebt_2011", "COVID_2020", "Inflation_2022"]
FP = ["TaperTantrum_2013", "Selloff_Q4_2018"]
TROUGH = ["GFC_2008", "EuroDebt_2011", "COVID_2020", "Inflation_2022"]

# --- Metadatos estáticos (copiados VERBATIM de scripts/builders/_build_13.py) --- #
CLASE = {
    "rule_vix_threshold": "baseline", "rule_composite_riskoff": "baseline",
    "clustering_gmm_k3": "baseline", "hmm_gaussian_2s": "baseline",
    "markov_switching_var_2s": "avanzado", "garch_t_vol": "avanzado",
    "changepoint_online": "avanzado", "hmm_tstudent_4s": "avanzado",
    "turbulence_mahalanobis": "avanzado", "jump_model": "avanzado",
    "msgarch_regime": "exploratorio-negativo", "deep_ae_regime": "exploratorio-negativo",
}
COSTE = {
    "rule_vix_threshold": "bajo", "rule_composite_riskoff": "bajo",
    "clustering_gmm_k3": "medio", "hmm_gaussian_2s": "medio",
    "markov_switching_var_2s": "alto", "garch_t_vol": "medio",
    "changepoint_online": "bajo", "hmm_tstudent_4s": "alto",
    "turbulence_mahalanobis": "bajo", "jump_model": "medio",
    "msgarch_regime": "alto", "deep_ae_regime": "medio",
}

# Estrés agregado de los 3 MULTI-ESTADO (recompute walk-forward de _build_13;
# incrustado aquí para reconstruir el master SIN re-ejecutar detectores).
# Claves NaN (GFC/EuroDebt/Taper) = ventana corta: no las vio / no aplica.
MULTI_ESTRES = {
    "clustering_gmm_k3": {
        "cov_estres": {"COVID_2020": 0.96, "Inflation_2022": 0.8653846153846154},
        "fa_estres": {"Selloff_Q4_2018": 0.7288135593220338},
        "false_alarm_rate_estres": 0.8280542986425339,
    },
    "hmm_tstudent_4s": {
        "cov_estres": {"COVID_2020": 0.96, "Inflation_2022": 0.9038461538461539},
        "fa_estres": {"TaperTantrum_2013": 0.0, "Selloff_Q4_2018": 0.8135593220338984},
        "false_alarm_rate_estres": 0.794066317626527,
    },
    "deep_ae_regime": {
        "cov_estres": {"COVID_2020": 0.96, "Inflation_2022": 0.7884615384615384},
        "fa_estres": {"Selloff_Q4_2018": 0.1864406779661017},
        "false_alarm_rate_estres": 0.7239583333333334,
    },
}


def nota_estres(name: str, n_states: int) -> str:
    if n_states == 2:
        return "binario: estres = crisis (state==1)"
    if name == "hmm_tstudent_4s":
        return "k=4: estres = {correccion(2), crisis(3)}"
    return "k=3: estres = {correccion(1), crisis(2)}"


# Orden canónico de columnas del master unificado (43).
CANON_ORDER = (
    ["detector", "n_states", "clase", "coste", "vio_2008_oos",
     "ventana_eval", "oos_start", "oos_end", "n_oos",
     "false_alarm_rate", "false_alarm_rate_estres",
     "switching_rate", "mean_regime_duration", "label_stability", "silhouette",
     "log_likelihood", "aic", "bic"]
    + sum([[f"cov_{w}", f"cov_{w}_lo", f"cov_{w}_hi"] for w in CRISIS], [])
    + [f"cov_estres_{w}" for w in CRISIS]
    + [f"fa_{f}" for f in FP]
    + [f"fa_estres_{f}" for f in FP]
    + [f"leadlag_{w}" for w in TROUGH]
    + ["nota"]
)


def _augment(row: dict) -> dict:
    """Añade metadatos + columnas de estrés a una fila base (crisis estricta)."""
    name = row["detector"]
    n = int(row["n_states"])
    row["clase"] = CLASE.get(name, "?")
    row["coste"] = COSTE.get(name, "?")
    row["vio_2008_oos"] = pd.Timestamp(row["oos_start"]) < pd.Timestamp("2008-09-01")
    row["nota"] = nota_estres(name, n)
    if n == 2:  # binario: estrés = crisis (copia directa)
        for w in CRISIS:
            row[f"cov_estres_{w}"] = row.get(f"cov_{w}", float("nan"))
        for f in FP:
            row[f"fa_estres_{f}"] = row.get(f"fa_{f}", float("nan"))
        row["false_alarm_rate_estres"] = row.get("false_alarm_rate", float("nan"))
    else:       # multi-estado: estrés incrustado (recompute de _build_13)
        m = MULTI_ESTRES[name]
        for w in CRISIS:
            row[f"cov_estres_{w}"] = m["cov_estres"].get(w, float("nan"))
        for f in FP:
            row[f"fa_estres_{f}"] = m["fa_estres"].get(f, float("nan"))
        row["false_alarm_rate_estres"] = m["false_alarm_rate_estres"]
    return row


def main() -> None:
    rows = []
    for f in FILES:
        df = pd.read_csv(RES / f)
        df = df[~df["detector"].astype(str).str.contains("|".join(EXCLUDE_SUBSTR), case=False)]
        if len(df) == 0:
            raise SystemExit(f"{f}: sin fila canónica tras excluir variantes")
        rows.append(_augment(df.iloc[0].to_dict()))

    master = pd.DataFrame(rows)
    # Reindexa al esquema canónico (columnas ausentes -> NaN; p. ej. D5 sin silhouette/IC).
    master = master.reindex(columns=CANON_ORDER)

    out = RES / "metrics_master.csv"
    master.to_csv(out, index=False)
    print(f"master reconstruido: {master.shape} -> {out}")
    print("detectores:", list(master["detector"]))
    missing = [c for c in CANON_ORDER if c not in master.columns]
    print(f"columnas canónicas: {master.shape[1]} (esperado {len(CANON_ORDER)}) | faltan: {missing}")


if __name__ == "__main__":
    main()
