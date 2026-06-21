#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Curaduría de figuras para el PDF ejecutivo del TFM (detección de regímenes).

TODAS las figuras se generan VERBATIM desde `results/metrics_master.csv` (master canónico).
NO re-ejecuta detectores ni walk-forward; NO toca CSVs ni las figuras `fase4_*`.
Las figuras nuevas se escriben con prefijo NUEVO `results/pdf_*.png` (DPI>=200).

Uso:
    python scripts/figs_pdf/build_pdf_figs.py

Decisiones de equidad (revisión académica):
  - La cobertura SIEMPRE se presenta separada por grupo de ventana (`vio_2008_oos`):
    los detectores de ventana larga vieron la GFC-2008 fuera de muestra; los de
    ventana corta NO. Comparar su cobertura "sistémica" junta es injusto.
  - El eje de SENSIBILIDAD común a los 12 es la cobertura de COVID-2020 (ventana
    OOS compartida por todos). La GFC-2008 se ranquea SOLO dentro de la ventana larga.
  - Lead/lag se calcula con `lookback=252` días (ver src/evaluation.py): un valor en
    el borde (|d| >= 249) está CENSURADO (la señal ya estaba activa al inicio de la
    ventana) y se marca como tal; no es anticipación genuina.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import viz  # noqa: E402  (mapeo de nombres cortos centralizado y robusto a la K)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
# Master canónico ÚNICO (superset unificado en Ola 0). Antes leía metrics_master_final.csv,
# ya archivado en results/_archive/; ahora todas las columnas (silhouette, IC bootstrap,
# clase, coste, vio_2008_oos, *_estres, nota) viven en metrics_master.csv.
MASTER = ROOT / "results" / "metrics_master.csv"
OUT = ROOT / "results"

DPI = 200
LOOKBACK = 252          # src/evaluation.lead_lag(lookback=252)
CENSOR_THR = 249        # |leadlag| >= 249 ~ censurado (señal ya activa al inicio)

# Paleta sobria y consistente (un color por grupo de ventana; nada de rojos chillones)
C_LONG = "#2b5d8a"      # ventana larga  (vio 2008 OOS)  -> azul sobrio
C_SHORT = "#c98a2b"     # ventana corta  (no vio 2008)   -> ámbar sobrio
C_NEG = "#7a7a7a"       # exploratorio-negativo          -> gris
C_NA = "#e8e8e8"        # no aplica / fuera de ventana

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# Labels cortos VARIANTE INFORME (marcan los negativos D11/D12 con "(-)" y usan
# abreviaturas propias del PDF). Keyed por PREFIJO canónico (sin la K) para no
# romperse si cambia el K elegido por BIC; la normalización del sufijo de K vive en
# src/viz.canonical_detector (fuente de verdad única; ver viz.SHORT para el D-N base).
SHORT_PDF = {
    "rule_vix_threshold": "D1 vix", "rule_composite_riskoff": "D2 comp",
    "clustering_gmm": "D3 gmm", "hmm_gaussian": "D4 hmm",
    "markov_switching_var": "D5 msvar", "garch_t_vol": "D6 garch",
    "changepoint_online": "D7 cusum", "hmm_tstudent": "D8 hmm-t",
    "jump_model": "D9 jump", "turbulence_mahalanobis": "D10 turb",
    "msgarch_regime": "D11 msg(-)", "deep_ae_regime": "D12 ae(-)",
}


def SHORT(name: str) -> str:
    """Label corto (variante informe) resuelto por prefijo canónico."""
    return SHORT_PDF.get(viz.canonical_detector(name), str(name))
TROUGH = ["GFC_2008", "EuroDebt_2011", "COVID_2020", "Inflation_2022"]
TROUGH_ES = ["GFC 2008", "EuroDeuda 2011", "COVID 2020", "Inflación 2022"]


def load() -> pd.DataFrame:
    df = pd.read_csv(MASTER)
    df["vio_2008_oos"] = df["vio_2008_oos"].astype(str).str.lower().eq("true")
    df["is_neg"] = df["clase"].eq("exploratorio-negativo")
    # Ejes derivados (idénticos a la lógica de la fase 4)
    df["eje_especif_estricta"] = 1 - df[["fa_TaperTantrum_2013", "fa_Selloff_Q4_2018"]].mean(axis=1)
    df["eje_especif_estres"] = 1 - df[["fa_estres_TaperTantrum_2013", "fa_estres_Selloff_Q4_2018"]].mean(axis=1)
    df["eje_persistencia"] = df["mean_regime_duration"]
    df["eje_switching"] = df["switching_rate"]
    df["eje_leadlag"] = df[[f"leadlag_{w}" for w in TROUGH]].mean(axis=1, skipna=True)
    df["coste_num"] = df["coste"].map({"bajo": 1, "medio": 2, "alto": 3})
    df["short"] = df["detector"].map(SHORT)
    return df


def color_for(row) -> str:
    if row["is_neg"]:
        return C_NEG
    return C_LONG if row["vio_2008_oos"] else C_SHORT


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- #
# FIG A: rank_heatmap CORREGIDO (cobertura separada por grupo de ventana)
# --------------------------------------------------------------------------- #
def fig_rank_heatmap(df: pd.DataFrame):
    """Heatmap de rangos por eje, con la corrección de equidad:
       - 'Cob. GFC 08' se ranquea SOLO entre los de ventana larga (n/a para los cortos).
       - 'Cob. COVID 20' (ventana común a los 12) se ranquea entre todos.
       - Filas agrupadas en dos bloques por ventana, con separador.
    """
    d = df.copy()

    # Orden de filas: bloque ventana larga arriba, corta abajo; dentro por cobertura COVID desc
    d_long = d[d["vio_2008_oos"]].sort_values("cov_COVID_2020", ascending=False)
    d_short = d[~d["vio_2008_oos"]].sort_values("cov_COVID_2020", ascending=False)
    order = pd.concat([d_long, d_short])
    d = d.set_index("detector").loc[order["detector"]]
    long_mask = d["vio_2008_oos"]

    # (label, columna, ascending(True=menor mejor), scope)
    cols = [
        ("Cob. GFC 08\n(solo v. larga)", "cov_GFC_2008", False, "long"),
        ("Cob. COVID 20\n(común a 12)", "cov_COVID_2020", False, "all"),
        ("Especif.\nestricta", "eje_especif_estricta", False, "all"),
        ("Especif.\nestrés", "eje_especif_estres", False, "all"),
        ("Persist.\n(duración)", "eje_persistencia", False, "all"),
        ("Anti-\nflicker", "eje_switching", True, "all"),
        ("Lead/lag †\n(media)", "eje_leadlag", True, "all"),
        ("BIC\n(generat.)", "bic", True, "gen"),
        ("Coste", "coste_num", True, "all"),
    ]

    n_rows, n_cols = len(d), len(cols)
    ranks = np.full((n_rows, n_cols), np.nan)
    norm = np.full((n_rows, n_cols), np.nan)   # 1=mejor (oscuro) .. 0=peor (claro)

    for j, (_, col, asc, scope) in enumerate(cols):
        s = d[col].copy()
        if scope == "long":
            s = s.where(long_mask)                        # solo ventana larga
        # rank dentro del scope (NaN se ignora)
        r = s.rank(ascending=asc, method="min")
        ranks[:, j] = r.values
        valid = r.dropna()
        if len(valid) > 1:
            rmin, rmax = valid.min(), valid.max()
            nrm = 1 - (r - rmin) / (rmax - rmin)         # mejor -> 1 -> oscuro
        else:
            nrm = r * 0 + 1.0
        norm[:, j] = nrm.values

    fig, ax = plt.subplots(figsize=(11.5, 7.8))
    cmap = plt.get_cmap("Blues")
    cmap.set_bad(C_NA)
    masked = np.ma.masked_invalid(norm)
    ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    for i in range(n_rows):
        for j in range(n_cols):
            v = ranks[i, j]
            if not np.isnan(v):
                txt_dark = norm[i, j] >= 0.55
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=9,
                        color="white" if txt_dark else "#222222")
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.5, color="#999999")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([c[0] for c in cols], fontsize=8.5)
    ax.set_yticks(range(n_rows))
    ylabels = [SHORT(idx) for idx in d.index]
    ax.set_yticklabels(ylabels, fontsize=9.5)
    # colorear etiquetas Y por grupo de ventana
    for tick, idx in zip(ax.get_yticklabels(), d.index):
        if d.loc[idx, "is_neg"]:
            tick.set_color(C_NEG)
        else:
            tick.set_color(C_LONG if d.loc[idx, "vio_2008_oos"] else C_SHORT)

    # separador entre bloques de ventana
    sep = len(d_long) - 0.5
    ax.axhline(sep, color="black", lw=2)
    ax.text(-1.15, (len(d_long) - 1) / 2, "VENTANA LARGA\n(vio 2008 OOS)", rotation=90,
            va="center", ha="center", fontsize=8.5, color=C_LONG, fontweight="bold")
    ax.text(-1.15, len(d_long) + (len(d_short) - 1) / 2, "VENTANA CORTA\n(no vio 2008)",
            rotation=90, va="center", ha="center", fontsize=8.5, color=C_SHORT, fontweight="bold")

    ax.set_title("Ranking por eje (1 = mejor; más oscuro = mejor rango)\n"
                 "Cobertura SEPARADA por ventana: GFC-2008 solo ranquea entre los de ventana larga",
                 fontsize=11)
    ax.text(0.0, 1.0, "", transform=ax.transAxes)
    fig.text(0.5, 0.005,
             "† Lead/lag censurado en ±252 d (ver figura de lead/lag); su rango medio puede estar inflado por detectores 'always-on'.",
             ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0.02, 0.03, 1, 1))
    out = OUT / "pdf_rank_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# FIG B: sensibilidad (cobertura COVID común) vs especificidad
# --------------------------------------------------------------------------- #
def fig_sens_espec(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10.5, 7))
    for _, row in df.iterrows():
        x = row["eje_especif_estricta"]
        y = row["cov_COVID_2020"]                  # ventana OOS común a los 12 -> justo
        c = color_for(row)
        mk = "X" if row["is_neg"] else ("o" if row["vio_2008_oos"] else "s")
        ax.scatter(x, y, s=180, c=c, marker=mk, edgecolor="black", linewidth=0.8, zorder=3)
        ax.annotate(row["short"], (x, y), textcoords="offset points",
                    xytext=(7, 5), fontsize=9)
    ax.set_xlabel("Especificidad = 1 − media(FA 2013, FA 2018)   (alto = no dispara en trampas)")
    ax.set_ylabel("Sensibilidad = cobertura COVID-2020 (OOS común a los 12)   (alto = sensible)")
    ax.set_title("Plano sensibilidad ↔ especificidad\n"
                 "Sensibilidad medida en la ventana OOS común (COVID-2020) para comparación justa")
    ax.set_ylim(0, 1.05)
    style_axes(ax)
    ax.legend(handles=[
        Patch(facecolor=C_LONG, edgecolor="black", label="Ventana larga (vio 2008 OOS)"),
        Patch(facecolor=C_SHORT, edgecolor="black", label="Ventana corta (no vio 2008)"),
        Patch(facecolor=C_NEG, edgecolor="black", label="Exploratorio-negativo (D11/D12)"),
    ], loc="lower left", fontsize=9)
    fig.tight_layout()
    out = OUT / "pdf_sensibilidad_especificidad.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# FIG C: persistencia (anti-flicker) vs sensibilidad (cobertura COVID común)
# --------------------------------------------------------------------------- #
def fig_persist_sens(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10.5, 7))
    for _, row in df.iterrows():
        x = row["eje_persistencia"]
        y = row["cov_COVID_2020"]
        c = color_for(row)
        mk = "X" if row["is_neg"] else ("o" if row["vio_2008_oos"] else "s")
        ax.scatter(x, y, s=180, c=c, marker=mk, edgecolor="black", linewidth=0.8, zorder=3)
        ax.annotate(row["short"], (x, y), textcoords="offset points",
                    xytext=(7, 5), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Persistencia = duración media de régimen (días, escala log)   (alto = no flickea)")
    ax.set_ylabel("Sensibilidad = cobertura COVID-2020 (OOS común a los 12)")
    ax.set_title("Plano persistencia ↔ sensibilidad\n"
                 "D3/D12 flickean (baja duración); D7 y las reglas son muy persistentes")
    ax.set_ylim(0, 1.05)
    style_axes(ax)
    ax.legend(handles=[
        Patch(facecolor=C_LONG, edgecolor="black", label="Ventana larga (vio 2008 OOS)"),
        Patch(facecolor=C_SHORT, edgecolor="black", label="Ventana corta (no vio 2008)"),
        Patch(facecolor=C_NEG, edgecolor="black", label="Exploratorio-negativo (D11/D12)"),
    ], loc="lower right", fontsize=9)
    fig.tight_layout()
    out = OUT / "pdf_persistencia_sensibilidad.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# FIG D: BIC de modelos generativos (paleta sobria)
# --------------------------------------------------------------------------- #
def fig_bic(df: pd.DataFrame):
    s = df.dropna(subset=["bic"]).sort_values("bic")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [C_NEG if r["is_neg"] else (C_LONG if r["vio_2008_oos"] else C_SHORT)
              for _, r in s.iterrows()]
    ax.bar(s["short"], s["bic"], color=colors, edgecolor="black", linewidth=0.6)
    for i, (_, r) in enumerate(s.iterrows()):
        ax.text(i, r["bic"], f"{r['bic']:.0f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("BIC (menor = mejor ajuste)")
    ax.set_title("BIC de modelos generativos\n"
                 "Aviso: el BIC solo es estrictamente comparable sobre las MISMAS features/ventana (p. ej. D4 vs D8)")
    style_axes(ax)
    ax.grid(axis="x", alpha=0)
    plt.xticks(rotation=25, ha="right")
    ax.legend(handles=[
        Patch(facecolor=C_LONG, edgecolor="black", label="Ventana larga"),
        Patch(facecolor=C_SHORT, edgecolor="black", label="Ventana corta"),
        Patch(facecolor=C_NEG, edgecolor="black", label="Negativo"),
    ], loc="upper left", fontsize=8.5)
    fig.tight_layout()
    out = OUT / "pdf_bic.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# FIG E: lead/lag por evento, con CENSURA marcada
# --------------------------------------------------------------------------- #
def fig_leadlag(df: pd.DataFrame):
    # mismo orden que el heatmap (ventana larga arriba)
    d_long = df[df["vio_2008_oos"]].sort_values("eje_leadlag")
    d_short = df[~df["vio_2008_oos"]].sort_values("eje_leadlag")
    order = pd.concat([d_long, d_short]).set_index("detector")

    M = np.array([[order.loc[n, f"leadlag_{w}"] for w in TROUGH] for n in order.index], dtype=float)

    fig, ax = plt.subplots(figsize=(11, 6.8))
    # paleta sobria divergente azul (anticipa) <-> gris (retrasa); sin rojos chillones
    cmap = plt.get_cmap("PuBu_r")           # más oscuro = más negativo = anticipa
    cmap.set_bad(C_NA)
    masked = np.ma.masked_invalid(M)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-LOOKBACK, vmax=0)

    for i, n in enumerate(order.index):
        for j in range(len(TROUGH)):
            v = M[i, j]
            if np.isnan(v):
                continue
            censored = abs(v) >= CENSOR_THR
            label = f"{v:.0f}*" if censored else f"{v:.0f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8.5,
                    color="white" if v <= -120 else "#222222",
                    fontweight="bold" if censored else "normal")
            if censored:   # marca de censura: borde discontinuo
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", lw=1.4, linestyle=(0, (2, 2))))

    ax.set_xticks(range(len(TROUGH)))
    ax.set_xticklabels(TROUGH_ES, rotation=15, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT(n) for n in order.index])
    for tick, n in zip(ax.get_yticklabels(), order.index):
        if order.loc[n, "is_neg"]:
            tick.set_color(C_NEG)
        else:
            tick.set_color(C_LONG if order.loc[n, "vio_2008_oos"] else C_SHORT)
    sep = len(d_long) - 0.5
    ax.axhline(sep, color="black", lw=2)
    ax.set_title("Lead/lag por evento (días; negativo = anticipa el suelo del drawdown)\n"
                 "* = CENSURADO en ±252 d (señal ya activa al inicio de la ventana; no es anticipación genuina)",
                 fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="días (− = anticipa)")
    fig.tight_layout()
    out = OUT / "pdf_leadlag.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# FIG F: crisis estricta vs estrés (multi-estado) — paleta sobria
# --------------------------------------------------------------------------- #
def fig_estres(df: pd.DataFrame):
    multi = ["clustering_gmm_k3", "hmm_tstudent_4s", "deep_ae_regime"]
    wins = ["cov_COVID_2020", "cov_Inflation_2022"]
    wins_es = ["COVID 2020", "Inflación 2022"]
    d = df.set_index("detector")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    C_ESTR = "#3a4a5a"      # estricta -> pizarra oscura
    C_ESTRES = "#c98a2b"    # estrés   -> ámbar sobrio
    for axx, name in zip(axes, multi):
        r = d.loc[name]
        estr = [r[w] for w in wins]
        estres = [r[w.replace("cov_", "cov_estres_")] for w in wins]
        xx = np.arange(len(wins)); w = 0.38
        b1 = axx.bar(xx - w / 2, estr, w, label="Crisis estricta (cola extrema)", color=C_ESTR)
        b2 = axx.bar(xx + w / 2, estres, w, label="Estrés agregado (corrección+crisis)", color=C_ESTRES)
        axx.set_xticks(xx); axx.set_xticklabels(wins_es)
        neg = "  (NEGATIVO)" if r["is_neg"] else ""
        axx.set_title(SHORT(name) + neg)
        axx.set_ylim(0, 1.08); axx.grid(alpha=0.25, axis="y"); axx.grid(axis="x", alpha=0)
        style_axes(axx)
        for b in list(b1) + list(b2):
            axx.text(b.get_x() + b.get_width() / 2, b.get_height(),
                     f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8.5)
    axes[0].set_ylabel("Cobertura")
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Multi-estado: cobertura con crisis ESTRICTA vs ESTRÉS agregado\n"
                 "(ventanas OOS comunes COVID-2020 e Inflación-2022)", y=1.02)
    fig.tight_layout()
    out = OUT / "pdf_estres_vs_estricta.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    df = load()
    outs = [
        fig_rank_heatmap(df),
        fig_sens_espec(df),
        fig_persist_sens(df),
        fig_bic(df),
        fig_leadlag(df),
        fig_estres(df),
    ]
    print("Figuras generadas:")
    for o in outs:
        print("  ", o.relative_to(ROOT))


if __name__ == "__main__":
    main()
