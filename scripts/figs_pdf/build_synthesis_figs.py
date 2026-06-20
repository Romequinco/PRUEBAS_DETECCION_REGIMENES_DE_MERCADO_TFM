#!/usr/bin/env python
"""Figuras de síntesis (Tanda 4) generadas VERBATIM desde results/metrics_master.csv.

NO re-ejecuta detectores. Produce figuras nuevas con prefijo `synth_`:
  - synth_coverage_ci.png : cobertura COVID-2020 (ventana OOS común a los 12) con
    las bandas de IC bootstrap de bloques (método nuevo) como barras de error.
  - synth_scorecard.png   : heatmap de los 6 ejes normalizados "mejor-para-qué".
  - synth_silhouette.png  : separación de regímenes en feature-space (silhouette).

Honestidad: la cobertura se muestra para COVID-2020 porque es la única ventana de
crisis OOS compartida por TODOS los detectores (las de ventana corta no vieron
2008/2011). El IC es intra-evento (días autocorrelados), no entre eventos.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import viz  # noqa: E402

viz.use_house_style()
RES = ROOT / "results"
M = pd.read_csv(RES / "metrics_master.csv")

# nombre corto legible D1..D12
SHORT = {
    "rule_vix_threshold": "D1 vix", "rule_composite_riskoff": "D2 riskoff",
    "clustering_gmm_k3": "D3 gmm", "hmm_gaussian_2s": "D4 hmm-g",
    "markov_switching_var_2s": "D5 msvar", "garch_t_vol": "D6 garch",
    "changepoint_online": "D7 cusum", "hmm_tstudent_4s": "D8 hmm-t",
    "jump_model": "D9 jump", "turbulence_mahalanobis": "D10 turb",
    "msgarch_regime": "D11 msgarch", "deep_ae_regime": "D12 ae",
}
M["short"] = M["detector"].map(SHORT).fillna(M["detector"])


def fig_coverage_ci() -> None:
    d = M.sort_values("cov_COVID_2020", ascending=True)
    y = np.arange(len(d))
    cov = d["cov_COVID_2020"].values
    lo = d["cov_COVID_2020_lo"].values
    hi = d["cov_COVID_2020_hi"].values
    # barras de error asimétricas (NaN -> sin barra)
    err_lo = np.where(np.isnan(lo), 0, cov - lo)
    err_hi = np.where(np.isnan(hi), 0, hi - cov)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y, cov, color=viz.C_LONG, alpha=0.85, zorder=2)
    has_ci = ~np.isnan(lo)
    ax.errorbar(cov[has_ci], y[has_ci], xerr=[err_lo[has_ci], err_hi[has_ci]],
                fmt="none", ecolor="#333", elinewidth=1.2, capsize=3, zorder=3)
    # marcar el que no tiene IC (no re-ejecutado)
    for yi, hci in zip(y, has_ci):
        if not hci:
            ax.text(cov[yi] + 0.01, yi, "sin IC", va="center", fontsize=8, color=viz.C_NEG)
    ax.set_yticks(y); ax.set_yticklabels(d["short"])
    ax.set_xlabel("cobertura de COVID-2020 (fracción de días en crisis)")
    ax.set_xlim(0, 1.05)
    ax.set_title("Cobertura de crisis con IC bootstrap de bloques (95%)\n"
                 "COVID-2020 = única ventana OOS común a los 12 detectores")
    fig.savefig(RES / "synth_coverage_ci.png")
    plt.close(fig)


def _norm(s: pd.Series, invert: bool = False) -> pd.Series:
    v = s.astype(float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or hi == lo:
        return pd.Series(np.where(np.isnan(v), np.nan, 0.5), index=s.index)
    n = (v - lo) / (hi - lo)
    return 1 - n if invert else n


def fig_scorecard() -> None:
    fa_mean = M[["fa_TaperTantrum_2013", "fa_Selloff_Q4_2018"]].mean(axis=1)
    # construir con el índice nativo de M (evita desalineación -> NaN); etiquetas aparte
    axes = pd.DataFrame({
        "Cobertura\n(COVID)": _norm(M["cov_COVID_2020"]),
        "Especificidad\n(trampas)": _norm(fa_mean, invert=True),
        "Persistencia\n(log dur)": _norm(np.log10(M["mean_regime_duration"].clip(lower=1))),
        "Pocas falsas\nalarmas": _norm(M["false_alarm_rate"], invert=True),
        "Separación\n(silhouette)": _norm(M["silhouette"]),
        "Ajuste\n(-BIC)": _norm(M["bic"], invert=True),
    })
    col_labels = list(axes.columns)
    row_labels = list(M["short"].values)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.grid(False)
    data = axes.values.astype(float)
    im = ax.imshow(np.ma.masked_invalid(data), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="#222")
            else:
                ax.text(j, i, "·", ha="center", va="center", color=viz.C_NEG)
    ax.set_title("Scorecard 'mejor-para-qué': 6 ejes normalizados [0,1] por columna\n"
                 "(verde = mejor en ese eje; ningún detector domina todos)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="rango relativo")
    fig.savefig(RES / "synth_scorecard.png")
    plt.close(fig)


def fig_silhouette() -> None:
    d = M.dropna(subset=["silhouette"]).sort_values("silhouette")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = [viz.C_LONG if v > 0.15 else viz.C_SHORT for v in d["silhouette"]]
    ax.barh(d["short"], d["silhouette"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("silhouette medio (separación de regímenes en feature-space)")
    ax.set_title("Separación de regímenes en el espacio de features\n"
                 "(>0 = regímenes distinguibles; útil sobre todo para clustering)")
    fig.savefig(RES / "synth_silhouette.png")
    plt.close(fig)


def main() -> None:
    fig_coverage_ci()
    fig_scorecard()
    fig_silhouette()
    print("Generadas: synth_coverage_ci.png, synth_scorecard.png, synth_silhouette.png")


if __name__ == "__main__":
    main()
