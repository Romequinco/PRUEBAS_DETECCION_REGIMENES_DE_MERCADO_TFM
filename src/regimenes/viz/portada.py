"""Figura de portada del README: lo que ve un agente con ``get_regimen`` en la pista B.

Solo lee ficheros versionados (panel ``B_<detector>`` de ``results/benchmark/panels``, consenso de
``results/fusion/online``, ventanas de crisis de ``configs/benchmark_spec.yaml``) y el S&P 500 de
``data/raw``. Uso::

    python -m regimenes.viz.portada            # escribe docs/img/portada_regimen.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from regimenes import rutas
from regimenes.regimen import tool
from regimenes.viz.figuras import use_house_style

SALIDA = rutas.DOCS / "img" / "portada_regimen.png"
C_PRECIO, C_CRISIS, C_CATALOGO, C_CONSENSO = "#1f3b5c", "#c8414b", "#9aa0a6", "#e09f3e"


def _tramos(serie: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Tramos consecutivos en los que ``serie`` es True."""
    bloque = (serie != serie.shift()).cumsum()
    return [(g.index[0], g.index[-1]) for _, g in serie[serie].groupby(bloque[serie])]


def dibujar(salida: Path = SALIDA, pista: str = "B") -> Path:
    import matplotlib.pyplot as plt

    use_house_style()
    cfg = tool.cargar_config()
    det = cfg["detector_por_pista"][pista]
    panel = tool._cargar_panel(rutas.ROOT / cfg["directorio_benchmark"], pista, det)
    crisis = pd.Series(panel.es_crisis, index=panel.index)
    p_det = pd.Series(panel.p_crisis, index=panel.index)
    consenso = pd.read_parquet(rutas.ROOT / cfg["contexto"]["prob_consenso"].format(pista=pista))
    consenso = consenso[cfg["contexto"].get("columna_consenso", "ONLINE_principal_eta10")]
    spec = yaml.safe_load(rutas.BENCHMARK_SPEC.read_text(encoding="utf-8"))
    ventanas = spec["crisis_windows"][f"pista_{pista}"]
    sp = pd.read_parquet(next(rutas.DATA_RAW.glob("*/SP500.parquet")))["SP500"]
    sp = sp.loc[crisis.index[0]:crisis.index[-1]]

    fig, (ax, ax_c, ax_p) = plt.subplots(3, 1, figsize=(12, 5.6), dpi=150, sharex=True,
                                         gridspec_kw={"height_ratios": [3, 0.35, 1.4], "hspace": 0.08})
    for ini, fin in ventanas.values():
        if pd.Timestamp(fin) >= sp.index[0]:
            for a in (ax, ax_p):
                a.axvspan(pd.Timestamp(ini), pd.Timestamp(fin), color=C_CATALOGO, alpha=0.18, lw=0)
    ax.semilogy(sp.index, sp.values, color=C_PRECIO, lw=1.0)
    ax.set_ylabel("S&P 500 (log)")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    ax.yaxis.set_major_locator(FixedLocator([1000, 1500, 2000, 3000, 4000, 6000, 8000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(f"Lo que ve un agente: get_regimen(fecha) en la pista {pista} ({det}), sin mirar el futuro",
                 loc="left", fontsize=11)

    for ini, fin in _tramos(crisis):
        ax_c.axvspan(ini, fin, color=C_CRISIS, lw=0)
    ax_c.set_yticks([]), ax_c.grid(False)
    ax_c.set_ylabel("crisis", rotation=0, ha="right", va="center", fontsize=9)

    ax_p.plot(p_det.index, p_det.values, color=C_CRISIS, lw=0.8, label=f"prob. de crisis de {det}")
    ax_p.plot(consenso.index, consenso.values, color=C_CONSENSO, lw=1.0, label="consenso de detectores (fusión online)")
    ax_p.set_ylim(-0.03, 1.03), ax_p.set_ylabel("probabilidad")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_CATALOGO, alpha=0.35, label="crisis del catálogo (referencia)"),
                       Patch(color=C_CRISIS, label="estado «crisis» de get_regimen")],
              loc="upper left", frameon=False, fontsize=8.5)
    ax_p.legend(loc="upper left", bbox_to_anchor=(0, -0.18), frameon=False, fontsize=8.5, ncol=2)
    fig.text(0.99, 0.005, "Walk-forward fuera de muestra · fuente: results/benchmark y results/fusion",
             ha="right", fontsize=7, color="#666")
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, bbox_inches="tight")
    plt.close(fig)
    return salida


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--salida", type=Path, default=SALIDA)
    parser.add_argument("--pista", default="B", choices=["A", "B"])
    args = parser.parse_args(argv)
    print(dibujar(args.salida, args.pista))


if __name__ == "__main__":
    main()
