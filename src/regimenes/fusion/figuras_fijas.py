"""Tablas y figuras del notebook 15 (fusiones fijas, ADR-011 §3, §5 y §6).

**Solo lee y dibuja.** Todo sale de lo que ya escribió :mod:`regimenes.fusion.fijas` en
``results/fusion/fijas/`` (criterio, elección, evaluación en formato largo, MCS-F, lecturas sin
peso y señales) y de los resultados versionados de las fusiones anteriores
(``results/fusion/d07_d08/`` y ``results/fusion/d02_d06/``, antecedentes sin peso). No recalcula
ninguna fusión, ni el MCS, ni toca paneles: las ventanas de crisis vienen de la especificación del
juez (``ranking.track_crisis_windows``) y las mitades se cortan sobre el índice de las señales
guardadas (:func:`regimenes.fusion.vara.mitades`).

Paleta: los individuales con el color de su familia de detectores
(:data:`regimenes.benchmark.panorama.COLOR_FAMILIA`, el mismo del notebook 14); las fusiones con
un color por familia de fusión (:data:`COLOR_FUSION`); los baselines en gris.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from regimenes import rutas
from regimenes.fusion import fijas as fj
from regimenes.fusion.vara import mitades

PISTAS = ("A", "B")
FIJAS: Path = rutas.RESULTS_FUSION_FIJAS
D07_D08: Path = rutas.RESULTS_FUSION_D07_D08
D02_D06: Path = rutas.RESULTS_FUSION_D02_D06

# Filas de la figura resumen: (clave, etiqueta). «c» se parte en OR y AND.
FILAS = (
    ("individual", "Individuales"),
    ("a", "(a) alerta → confirmación"),
    ("b", "(b) votación k-de-n"),
    ("c_or", "(c) OR de parejas"),
    ("c_and", "(c) AND de parejas"),
)
COLOR_FUSION = {"a": "#5b8fb0", "b": "#c98a2b", "c_or": "#7a9a3a", "c_and": "#6a3d9a"}
COLOR_INDIVIDUAL = "#333333"
COLOR_CRISIS = "#b5341f"
NOMBRE_NIVEL = {0: "elegible", 1: "precisión ≤ azar", 2: "parpadeo", 3: "degenerada"}
ANTECEDENTES = {"D7+D8": ("D07", "D08"), "D7+D6": ("D07", "D06"), "D2+D6": ("D02", "D06")}


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def num(x, d: int = 3) -> str:
    """Número con coma decimal (``0,597``); vacío si falta."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x):.{d}f}".replace(".", ",")


def fila_de(i: str) -> str:
    """Fila de la figura resumen de un id: ``individual``, ``a``, ``b``, ``c_or`` o ``c_and``."""
    i = str(i)
    if i.startswith("PAR_"):
        return "a"
    if i.startswith("VOTO_"):
        return "b"
    if i.startswith("OR_"):
        return "c_or"
    if i.startswith("AND_"):
        return "c_and"
    return "individual"


def color_de(i: str) -> str:
    """Color de un id: familia de detectores (individual o baseline) o familia de fusión."""
    f = fila_de(i)
    if f != "individual":
        return COLOR_FUSION[f]
    from regimenes.benchmark import panorama as pn

    try:
        return pn.color_de(i)
    except KeyError:
        return COLOR_INDIVIDUAL


def _plt():
    import matplotlib.pyplot as plt
    from regimenes import viz

    viz.use_house_style()
    return plt


def _sel(ev: pd.DataFrame, pista: str, tramo: str, grupos: Sequence[str] = ("principal", "individual")
         ) -> pd.DataFrame:
    t = ev[(ev["pista"] == pista) & (ev["tramo"] == tramo) & ev["grupo"].isin(grupos)]
    return t.set_index("id")


def _crit(crit: pd.DataFrame, pista: str) -> pd.Series:
    return crit.set_index("pista").loc[pista]


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #
def cargar(salida: Path = FIJAS) -> dict:
    """Todo lo que usa el notebook, leído de ``results/fusion/fijas/``.

    Claves: ``criterio``, ``eleccion``, ``catalogo``, ``evaluacion``, ``mcs`` y ``mcs_H2`` (dict
    por pista), ``ganadores``, ``optimismo``, ``topn``, ``variantes``, ``anticipacion`` (puede
    venir vacía) y ``utilidad``. Lanza ``FileNotFoundError`` si falta el criterio."""
    crit = fj.leer_criterio(salida)
    if crit is None:
        raise FileNotFoundError(f"Faltan resultados en {salida}: python -m regimenes.fusion.fijas ejecutar")
    antic = fj.leer_lectura("anticipacion", salida)
    return {
        "criterio": crit,
        "eleccion": fj.leer_eleccion(salida),
        "catalogo": fj.leer_catalogo(salida),
        "evaluacion": fj.leer_evaluacion(salida=salida),
        "mcs": {p: fj.leer_mcs(p, "oos", salida) for p in PISTAS},
        "mcs_H2": {p: fj.leer_mcs(p, "H2", salida) for p in PISTAS},
        "ganadores": fj.leer_lectura("ganadores_familia", salida),
        "optimismo": fj.leer_lectura("optimismo", salida),
        "topn": fj.leer_lectura("topn", salida),
        "variantes": fj.leer_lectura("variantes", salida),
        "anticipacion": antic if antic is not None else pd.DataFrame(columns=fj.COLUMNAS_ANTICIPACION),
        "utilidad": fj.leer_lectura("utilidad", salida),
    }


def ventanas(pista: str) -> dict[str, tuple[str, str]]:
    """Ventanas ``[pico, suelo]`` del juez (especificación congelada; no lee paneles)."""
    from regimenes.evaluacion import ranking as rk

    return rk.track_crisis_windows(pista)


def indice(pista: str, salida: Path = FIJAS) -> pd.DatetimeIndex:
    """Índice OOS de la pista, tomado de las señales guardadas (una sola columna)."""
    s = fj.leer_senales(pista, columnas=[_primer_individual(pista, salida)], salida=salida)
    return pd.DatetimeIndex(s.index)


def _primer_individual(pista: str, salida: Path = FIJAS) -> str:
    cat = fj.leer_tabla(f"catalogo_{pista}", salida)
    return str(cat.loc[cat["grupo"] == "individual", "id"].iloc[0])


# --------------------------------------------------------------------------- #
# 1 · Antecedentes (sin peso)
# --------------------------------------------------------------------------- #
def tabla_antecedentes(d07: Path = D07_D08, d02: Path = D02_D06) -> pd.DataFrame:
    """Utilidad operativa de las tres parejas de los notebooks antiguos frente a sus sensores.

    Lee ``d02_d06/pair_verdict.csv`` (que incluye D7+D8, D7+D6 y D2+D6) y añade los avisos que
    anticipan una crisis frente a los esperables por azar (``warning_phase_summary`` de D7+D8 y
    ``warning_phase_vs_chance`` de D2+D6)."""
    pv = pd.read_csv(Path(d02) / "pair_verdict.csv")
    out = pv.rename(columns={"utilidad_alerta_sola": "alerta_sola", "utilidad_confirmador_solo":
                             "confirmador_solo", "utilidad_fusion": "fusion"})
    out["mejora_a_los_dos"] = out["fusion"] > out[["alerta_sola", "confirmador_solo"]].max(axis=1)
    ant = {}
    ph78 = pd.read_csv(Path(d07) / "warning_phase_summary.csv").set_index("pista")
    for p, r in ph78.iterrows():
        ant[("D7+D8", p)] = (int(r["anticipacion"]), int(r[["anticipacion", "durante_crisis", "posterior",
                                                              "falsa_alarma"]].sum()))
    ph26 = pd.read_csv(Path(d02) / "warning_phase_vs_chance.csv")
    for p, g in ph26.groupby("pista"):
        g = g.set_index("fase")
        ant[("D2+D6", p)] = (int(g.loc["anticipacion", "avisos"]), int(g["avisos"].sum()))
    out["avisos_que_anticipan"] = [ant.get((a, p), (np.nan, np.nan))[0] for a, p in zip(out["pareja"], out["pista"])]
    out["avisos"] = [ant.get((a, p), (np.nan, np.nan))[1] for a, p in zip(out["pareja"], out["pista"])]
    return out[["pista", "pareja", "alerta_sola", "confirmador_solo", "fusion", "veredicto",
                "mejora_a_los_dos", "avisos", "avisos_que_anticipan"]].sort_values(["pista", "pareja"])


def tabla_mitad_antigua(d02: Path = D02_D06) -> pd.DataFrame:
    """Control por mitades del notebook 16 antiguo: alerta elegida en la 1.ª mitad y su puesto en
    la 2.ª, frente a D02 (la alerta elegida con toda la muestra)."""
    sh = pd.read_csv(Path(d02) / "split_half_selection.csv")
    filas = []
    for ambito, g in sh.groupby("ambito", sort=False):
        el = g[g["elegido_en_seleccion"]].iloc[0]
        d2 = g[g["alerta"] == "D02"].iloc[0]
        n = int(g["elegible"].sum())
        filas.append({"ambito": ambito, "elegida_en_1a_mitad": el["alerta"],
                      "su_puesto_en_2a": f"{int(el['puesto_evaluacion'])} de {n}",
                      "puesto_D02_1a": f"{int(d2['puesto_seleccion'])} de {n}" if d2["elegible"] else "no elegible",
                      "puesto_D02_2a": f"{int(d2['puesto_evaluacion'])} de {n}" if d2["elegible"] else "no elegible"})
    return pd.DataFrame(filas)


def tabla_antecedentes_vara(ev: pd.DataFrame, crit: pd.DataFrame) -> pd.DataFrame:
    """Las parejas antiguas medidas con la vara nueva (familia (a), todo el OOS)."""
    filas = []
    for p in PISTAS:
        oos = _sel(ev, p, "oos", ("principal",))
        c = _crit(crit, p)
        for nombre, (x, y) in ANTECEDENTES.items():
            i = fj.id_pareja(x, y)
            r = oos.loc[i]
            filas.append({"pista": p, "pareja": nombre, "id": i, "score_oos": r["score_deteccion"],
                          "puesto": f"{int(r['puesto'])} de {len(oos)}",
                          "mejor_individual": f"{c['i_star']} {num(c['score_i_star_oos'])}",
                          "diferencia": r["score_deteccion"] - c["score_i_star_oos"]})
    return pd.DataFrame(filas)


def figura_antecedentes(tabla: pd.DataFrame):
    """Utilidad operativa de cada pareja antigua frente a su alerta y su confirmador solos."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharex=False)
    estilos = {"alerta_sola": ("alerta sola", "o", "#5b8fb0"), "confirmador_solo": ("confirmador solo", "s", "#c98a2b"),
               "fusion": ("fusión", "*", "#333333")}
    for ax, p in zip(axes, PISTAS):
        t = tabla[tabla["pista"] == p].reset_index(drop=True)
        for y, r in t.iterrows():
            vals = [r[c] for c in estilos]
            ax.plot([min(vals), max(vals)], [y, y], color="#cccccc", lw=2, zorder=1)
            for c, (lab, m, col) in estilos.items():
                ax.scatter(r[c], y, marker=m, s=160 if m == "*" else 60, color=col, zorder=3,
                           label=lab if y == 0 else None, edgecolor="white" if m != "*" else col)
        ax.set_yticks(range(len(t)), t["pareja"])
        ax.set_ylim(len(t) - 0.5, -0.5)
        ax.set_xlabel("utilidad operativa (más a la derecha, mejor) →")
        ax.set_title(f"Pista {p}")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False, ncol=3, fontsize=9)
    fig.suptitle("Antecedentes: cada pareja se medía contra sus propios sensores, con la utilidad operativa",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


# --------------------------------------------------------------------------- #
# 2 · Diseño
# --------------------------------------------------------------------------- #
def tabla_recuento(cat: pd.DataFrame) -> pd.DataFrame:
    """Señales por pista y familia (§3.4): 423 principales en A y 547 en B, más las de lectura."""
    nombres = {("individual", "individual"): "individuales (candidatos)",
               ("principal", "a"): "(a) parejas alerta→confirmación", ("principal", "b"): "(b) votación k-de-n",
               ("principal", "c"): "(c) OR y AND de parejas",
               ("tras_aviso", "a"): "sensibilidad (a) «tras aviso» (sin peso)",
               ("confirmado", "a"): "referencia «confirmado» puro (sin peso)"}
    t = cat.groupby(["grupo", "familia", "pista"]).size().unstack("pista")
    t.index = [nombres.get(i, " ".join(i)) for i in t.index]
    orden = [nombres[k] for k in nombres]
    t = t.reindex(orden)
    total = cat[cat["grupo"] == "principal"].groupby("pista").size()
    t.loc["total principal (entra en la selección de F*)"] = total
    return t.astype(int)


def tramos_pista(pista: str, salida: Path = FIJAS) -> pd.DataFrame:
    """Las dos mitades del §5.2 con sus fechas, sesiones y crisis con días en cada una."""
    idx = indice(pista, salida)
    win = ventanas(pista)
    filas = []
    for nombre, t in zip(("H1 (selección)", "H2 (evaluación)"), mitades(idx)):
        crisis = [c for c, (a, b) in win.items()
                  if pd.Timestamp(b) >= t[0] and pd.Timestamp(a) <= t[-1]]
        filas.append({"pista": pista, "mitad": nombre, "desde": t[0].date(), "hasta": t[-1].date(),
                      "sesiones": len(t), "crisis": len(crisis)})
    return pd.DataFrame(filas)


def figura_mitades(salida: Path = FIJAS):
    """Las dos mitades de cada pista sobre el calendario, con las ventanas de crisis."""
    import matplotlib.dates as mdates

    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(13, 3.4))
    for ax, p in zip(axes, PISTAS):
        idx = indice(p, salida)
        h1, h2 = mitades(idx)
        for t, col, lab in ((h1, "#dde8f0", "H1 · se elige"), (h2, "#f3e6d3", "H2 · se evalúa")):
            ax.axvspan(t[0], t[-1], color=col, zorder=0)
            n = sum(1 for a, b in ventanas(p).values() if pd.Timestamp(b) >= t[0] and pd.Timestamp(a) <= t[-1])
            ax.text(t[0] + (t[-1] - t[0]) / 2, 0.80, f"{lab} · {len(t):,} sesiones · {n} crisis".replace(",", "."),
                    ha="center", va="center", fontsize=9, transform=ax.get_xaxis_transform())
        ax.xaxis.set_major_locator(mdates.YearLocator(5 if p == "A" else 2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for a, b in ventanas(p).values():
            a, b = max(pd.Timestamp(a), idx[0]), pd.Timestamp(b)
            if b >= idx[0]:
                ax.axvspan(a, b, ymin=0.1, ymax=0.55, color=COLOR_CRISIS, alpha=0.85, lw=0)
        ax.axvline(h2[0], color="#333", lw=1.2, ls="--")
        ax.set_yticks([])
        ax.set_ylabel(f"Pista {p}", rotation=0, ha="right", va="center")
        ax.grid(False)
        ax.set_xlim(idx[0] - pd.Timedelta(days=60), idx[-1] + pd.Timedelta(days=60))
    fig.suptitle("Elegir en la primera mitad y evaluar en la segunda (rojo: ventanas de crisis)", fontsize=12,
                 weight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 0 · Resumen
# --------------------------------------------------------------------------- #
def tabla_puntos(ev: pd.DataFrame, tramo: str = "oos") -> pd.DataFrame:
    """Score de todas las fusiones principales y los individuales en el tramo, con su fila."""
    t = ev[(ev["tramo"] == tramo) & ev["grupo"].isin(["principal", "individual"])].copy()
    t["fila"] = t["id"].map(fila_de)
    return t


def figura_resumen(ev: pd.DataFrame, crit: pd.DataFrame, opt: pd.DataFrame | None = None):
    """Un punto por señal y pista sobre el eje del score en todo el OOS, una fila por familia.

    Línea punteada: el mejor individual (I*). Estrella: F* (elegida en H1). Rombo: la mejor fusión
    si se eligiera con todo el OOS (lectura del optimismo). Hueco: no elegible (nivel > 0)."""
    plt = _plt()
    from matplotlib.lines import Line2D

    pts = tabla_puntos(ev)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    rng = np.random.default_rng(0)
    for ax, p in zip(axes, PISTAS):
        c = _crit(crit, p)
        part = pts[pts["pista"] == p]
        for y, (clave, _) in enumerate(FILAS):
            q = part[part["fila"] == clave]
            jit = rng.uniform(-0.22, 0.22, len(q)) if len(q) > 3 else np.zeros(len(q))
            cols = [color_de(i) for i in q["id"]]
            llenos = (q["nivel"] == 0).to_numpy()
            ax.scatter(q["score_deteccion"][llenos], y + jit[llenos], s=22 if clave != "individual" else 55,
                       color=np.array(cols)[llenos], alpha=0.75 if clave != "individual" else 1, lw=0, zorder=2)
            ax.scatter(q["score_deteccion"][~llenos], y + jit[~llenos], s=22, facecolor="white",
                       edgecolor=np.array(cols)[~llenos], lw=0.9, zorder=2)
            if clave in ("individual", "b") and len(q):
                k = 0
                for (_, r), j in zip(q.sort_values("score_deteccion").iterrows(), jit):
                    if r["id"] == c["f_star"]:
                        continue
                    if clave == "b" or r["id"] == c["i_star"]:
                        ax.annotate(r["id"].replace("VOTO_", ""), (r["score_deteccion"], y + j),
                                    xytext=(0, 9 if k % 2 == 0 else -13), textcoords="offset points",
                                    ha="center", fontsize=8)
                        k += 1
            n_sup = int(((q["score_deteccion"] > c["score_i_star_oos"]) & (q["nivel"] == 0)).sum())
            if clave != "individual":
                ax.text(1.0, y, f"{n_sup} de {len(q)} > I*", transform=ax.get_yaxis_transform(), ha="right",
                        va="center", fontsize=8, color="#555")
        ax.axvline(c["score_i_star_oos"], color="#333", ls=":", lw=1.2)
        f = c["f_star"]
        if isinstance(f, str):
            yf = [k for k, _ in FILAS].index(fila_de(f))
            ax.scatter(c["score_f_star_oos"], yf, marker="*", s=420, color=color_de(f), edgecolor="black", lw=1.2,
                       zorder=5)
            ax.annotate(f"F* {f}\n{num(c['score_f_star_oos'])}", (c["score_f_star_oos"], yf), xytext=(-12, -26),
                        textcoords="offset points", ha="right", fontsize=9, weight="bold")
        if opt is not None:
            o = opt.set_index("pista").loc[p]
            yo = [k for k, _ in FILAS].index(fila_de(o["mejor_oos"]))
            ax.scatter(o["score_mejor_oos"], yo, marker="D", s=70, facecolor="none", edgecolor="black", lw=1.4,
                       zorder=5)
        ax.set_yticks(range(len(FILAS)), [lab for _, lab in FILAS])
        ax.set_ylim(len(FILAS) - 0.5, -0.6)
        ax.set_xlim(-0.02, 0.8)
        ax.set_xlabel("score de detección en todo el OOS →")
        ax.set_title(f"Pista {p} · línea punteada: mejor individual {c['i_star']} ({num(c['score_i_star_oos'])})")
    extra = [Line2D([], [], marker="*", ls="", ms=15, color="#999", mec="black", label="F* (elegida en H1)"),
             Line2D([], [], marker="D", ls="", ms=7, mfc="none", mec="black",
                    label="mejor fusión eligiendo con todo el OOS (optimista)"),
             Line2D([], [], marker="o", ls="", ms=6, mfc="white", mec="#555", label="no elegible (nivel > 0)")]
    fig.legend(handles=extra, loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False, ncol=3, fontsize=9)
    n = int(crit["cumple_i"].astype(bool).sum())
    titulo = ("La fusión elegida en H1 (F*) no supera al mejor detector solo en ninguna pista" if n == 0 else
              f"La fusión elegida en H1 (F*) supera al mejor detector solo en {n} pista(s)")
    fig.suptitle(titulo, fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def hechos(d: dict, antecedentes_vara: pd.DataFrame | None = None) -> dict:
    """Cifras que citan las conclusiones del notebook (todas de los CSV)."""
    crit, ev, opt = d["criterio"], d["evaluacion"], d["optimismo"]
    h = {"veredicto": str(crit["veredicto"].iloc[0]), "pistas": {}}
    for p in PISTAS:
        c = _crit(crit, p)
        oos = _sel(ev, p, "oos", ("principal",))
        h1 = _sel(ev, p, "H1", ("principal",))["score_deteccion"]
        h2 = _sel(ev, p, "H2", ("principal",))["score_deteccion"]
        o = opt.set_index("pista").loc[p]
        mcs = d["mcs"][p]
        h["pistas"][p] = {
            **c.to_dict(),
            "n_principales": len(oos),
            "n_superan_i_star": int(((oos["score_deteccion"] > c["score_i_star_oos"]) & (oos["nivel"] == 0)).sum()),
            "rho_h1_h2": float(h1.corr(h2.reindex(h1.index), method="spearman")),
            "optimismo": float(o["optimismo"]), "mejor_oos": o["mejor_oos"],
            "score_mejor_oos": float(o["score_mejor_oos"]),
            "p_mcs_f_star": float(mcs.loc[c["f_star"], "p_mcs"]) if isinstance(c["f_star"], str) else np.nan,
            "n_mcs": int(mcs["incluido"].sum()), "n_mcs_total": len(mcs),
            "score_i_star_H2": float(_sel(ev, p, "H2", ("individual",)).loc[c["i_star"], "score_deteccion"]),
            "score_mayoria_oos": float(oos.loc["VOTO_mayoria", "score_deteccion"]),
        }
    if antecedentes_vara is not None:
        h["antecedentes"] = antecedentes_vara
    return h


def conclusiones(h: dict) -> list[str]:
    """Conclusiones del resumen, redactadas a partir de :func:`hechos`."""
    A, B = h["pistas"]["A"], h["pistas"]["B"]
    fallan = {k: [p for p in PISTAS if not bool(h["pistas"][p][f"cumple_{k}"])] for k in ("i", "ii", "iii")}
    resumen = "; ".join(f"({k}) falla en {' y '.join(v)}" for k, v in fallan.items() if v) or "se cumple todo"
    out = [f"**Veredicto (ADR-011 §6.1): «{h['veredicto']}».** El criterio exige (i), (ii) y (iii) en las dos "
           f"pistas: {resumen}."]
    for p, x in (("A", A), ("B", B)):
        dentro = "dentro" if x["f_star_en_mcs"] else "fuera"
        out.append(
            f"**Pista {p}:** F\\* = `{x['f_star']}` puntúa {num(x['score_f_star_oos'])} en todo el OOS frente a "
            f"{num(x['score_i_star_oos'])} de {x['i_star']} (i ✗); queda {dentro} del MCS-F "
            f"(p = {num(x['p_mcs_f_star'], 3)}; (ii) {'✓' if x['cumple_ii'] else '✗'}) y en H2 gana a I\\*_H1 = "
            f"{x['i_star_h1']} ({num(x['score_f_star_H2'])} frente a {num(x['score_i_star_h1_H2'])}; (iii) "
            f"{'✓' if x['cumple_iii'] else '✗'}).")
    out.append(
        f"**Haberlas, las hay, pero no se pueden elegir de antemano.** Con todo el OOS, {A['n_superan_i_star']} de "
        f"{A['n_principales']} fusiones superan a {A['i_star']} en A y {B['n_superan_i_star']} de "
        f"{B['n_principales']} a {B['i_star']} en B. Eligiendo con todo el OOS se ganaría "
        f"{num(A['optimismo'])} (A) y {num(B['optimismo'])} (B) de score: ese es el optimismo de la selección.")
    out.append(
        f"**El orden de H1 apenas predice el de H2:** correlación de Spearman entre el score de las fusiones en las "
        f"dos mitades de {num(A['rho_h1_h2'], 2)} en A y {num(B['rho_h1_h2'], 2)} en B. (iii) se cumple porque "
        f"I\\*_H1 también se hunde en H2, no porque F\\* sea mejor que el mejor detector: en H2, {A['i_star']} "
        f"puntúa {num(A['score_i_star_H2'])} (F\\* {num(A['score_f_star_H2'])}) en A y {B['i_star']} "
        f"{num(B['score_i_star_H2'])} (F\\* {num(B['score_f_star_H2'])}) en B.")
    gana = {p: h["pistas"][p]["score_mayoria_oos"] > h["pistas"][p]["score_i_star_oos"] for p in PISTAS}
    out.append(
        f"**La votación por mayoría, que no elige a nadie**, puntúa {num(A['score_mayoria_oos'])} en A "
        f"({'por encima' if gana['A'] else 'por debajo'} de {A['i_star']}) y {num(B['score_mayoria_oos'])} en B "
        f"({'por encima' if gana['B'] else 'por debajo'} de {B['i_star']})"
        + (": no se sostiene en las dos pistas." if gana["A"] != gana["B"] else ".")
        + " Es la referencia V de la fusión online (notebook 17).")
    return out


# --------------------------------------------------------------------------- #
# 3 · Resultados por familia
# --------------------------------------------------------------------------- #
def matriz_parejas(ev: pd.DataFrame, cat: pd.DataFrame, pista: str, familia: str, tramo: str = "oos"
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matrices (score, nivel) candidato × candidato.

    ``familia="a"``: filas = alerta X, columnas = confirmador Y. ``familia="c"``: triángulo superior
    = OR y triángulo inferior = AND. La diagonal es el individual."""
    ids = list(cat[(cat["pista"] == pista) & (cat["grupo"] == "individual")]["id"])
    t = _sel(ev, pista, tramo)
    sc = pd.DataFrame(np.nan, index=ids, columns=ids)
    nv = pd.DataFrame(np.nan, index=ids, columns=ids)
    for x in ids:
        sc.loc[x, x], nv.loc[x, x] = t.loc[x, "score_deteccion"], t.loc[x, "nivel"]
    for i in ids:
        for j in ids:
            if i == j:
                continue
            if familia == "a":
                k = fj.id_pareja(i, j)
            else:
                a, b = sorted((i, j))
                pre = "OR" if ids.index(i) < ids.index(j) else "AND"
                k = f"{pre}_{a}+{b}"
            if k in t.index:
                sc.loc[i, j], nv.loc[i, j] = t.loc[k, "score_deteccion"], t.loc[k, "nivel"]
    return sc, nv


def figura_mapas(ev: pd.DataFrame, cat: pd.DataFrame, crit: pd.DataFrame, tramo: str = "oos"):
    """Mapas de calor del score de las parejas menos el del mejor individual, por pista.

    Rojo = mejor que I*; azul = peor; gris = no elegible (nivel > 0). Diagonal: los individuales.
    Recuadro negro: F* (si es una pareja)."""
    plt = _plt()
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(2, 2, figsize=(14, 13.5), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=0.06)
    im = None
    for r, p in enumerate(PISTAS):
        c = _crit(crit, p)
        ref = float(c["score_i_star_oos"] if tramo == "oos" else c["score_i_star_h1_H2"])
        for k, fam in enumerate(("a", "c")):
            ax = axes[r, k]
            sc, nv = matriz_parejas(ev, cat, p, fam, tramo)
            delta = (sc - ref).where(nv == 0)
            cmap = plt.get_cmap("RdBu_r").copy()
            cmap.set_bad("#d9d9d9")
            im = ax.imshow(delta.to_numpy(dtype=float), cmap=cmap, norm=norm, aspect="equal")
            n = len(sc)
            ax.set_xticks(range(n), sc.columns, rotation=90, fontsize=8)
            ax.set_yticks(range(n), sc.index, fontsize=8)
            ax.grid(False)
            for d in range(n):
                ax.add_patch(Rectangle((d - 0.5, d - 0.5), 1, 1, fill=False, ec="#333", lw=0.8))
            f = c["f_star"]
            if isinstance(f, str) and fila_de(f)[0] == fam:
                if fam == "a":
                    x, y = f[4:].split(">")
                    ii, jj = sc.index.get_loc(x), sc.columns.get_loc(y)
                else:
                    a, b = f.split("_", 1)[1].split("+")
                    ia, ib = sc.index.get_loc(a), sc.index.get_loc(b)
                    ii, jj = (min(ia, ib), max(ia, ib)) if f.startswith("OR_") else (max(ia, ib), min(ia, ib))
                ax.add_patch(Rectangle((jj - 0.5, ii - 0.5), 1, 1, fill=False, ec="black", lw=2.5))
                ax.annotate("F*", (jj, ii), ha="center", va="center", fontsize=8, weight="bold")
            for (ii, jj), v in np.ndenumerate(delta.to_numpy(dtype=float)):
                if np.isfinite(v) and v > 0:
                    ax.text(jj, ii, "+", ha="center", va="center", fontsize=9, color="white", weight="bold")
            if fam == "a":
                ax.set_title(f"Pista {p} · (a) filas: alerta X · columnas: confirmador Y")
                ax.set_xlabel("confirmador Y")
                ax.set_ylabel("alerta X")
            else:
                ax.set_title(f"Pista {p} · (c) arriba: OR · abajo: AND")
    cb = fig.colorbar(im, ax=axes, shrink=0.5, location="right", pad=0.02)
    cb.set_label(f"score {'en todo el OOS' if tramo == 'oos' else 'en H2'} − score del mejor individual  (+ = lo supera)")
    fig.suptitle("Score de cada pareja frente al mejor detector solo (diagonal: los individuales; gris: no elegible)",
                 fontsize=13, weight="bold")
    return fig


def tabla_votaciones(ev: pd.DataFrame, elec: pd.DataFrame, crit: pd.DataFrame) -> pd.DataFrame:
    """Las tres votaciones de cada pista: miembros, k, score en H1, H2 y OOS, frente a I*."""
    filas = []
    for p in PISTAS:
        e = elec.set_index("pista").loc[p]
        c = _crit(crit, p)
        for v in ("VOTO_2de3", "VOTO_3de5", "VOTO_mayoria"):
            m = str(e[f"miembros_{v}"]).split("|")
            fila = {"pista": p, "votación": v, "miembros": " · ".join(m) if len(m) <= 5 else f"los {len(m)} candidatos"}
            for t in ("H1", "H2", "oos"):
                r = _sel(ev, p, t, ("principal",)).loc[v]
                fila[f"score_{t}"] = r["score_deteccion"]
            fila["supera_I*_oos"] = fila["score_oos"] > c["score_i_star_oos"]
            filas.append(fila)
    return pd.DataFrame(filas)


def tabla_ganadores(gan: pd.DataFrame) -> pd.DataFrame:
    """Ganador de cada familia en H1 y su puesto en H2 (entre todas y dentro de su familia)."""
    t = gan.copy()
    t["familia"] = t["familia"].map({"a": "(a) alerta→conf.", "b": "(b) votación", "c": "(c) OR/AND"})
    return t[["pista", "familia", "ganador_H1", "puesto_H1", "score_H1", "score_H2", "puesto_H2",
              "puesto_H2_familia", "score_oos"]]


# --------------------------------------------------------------------------- #
# 4 · Selección en H1 y evaluación en H2
# --------------------------------------------------------------------------- #
def figura_h1_h2(ev: pd.DataFrame, crit: pd.DataFrame):
    """Score en H1 frente a score en H2 de todas las fusiones principales (y los individuales).

    Línea horizontal: score de I*_H1 en H2 (la vara de (iii)). Diagonal: H1 = H2."""
    plt = _plt()
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, p in zip(axes, PISTAS):
        c = _crit(crit, p)
        h1 = _sel(ev, p, "H1")
        h2 = _sel(ev, p, "H2").reindex(h1.index)
        fusion = h1["grupo"] == "principal"
        for clave in ("a", "c_or", "c_and", "b"):
            m = fusion & (h1.index.map(fila_de) == clave)
            ax.scatter(h1.loc[m, "score_deteccion"], h2.loc[m, "score_deteccion"], s=14 if clave != "b" else 60,
                       color=COLOR_FUSION[clave], alpha=0.55 if clave != "b" else 1, lw=0)
        ind = h1.index[~fusion]
        ax.scatter(h1.loc[ind, "score_deteccion"], h2.loc[ind, "score_deteccion"], s=45, marker="s",
                   color=[color_de(i) for i in ind], edgecolor="black", lw=0.6, zorder=3)
        ax.plot([0, 1], [0, 1], color="#bbb", ls="--", lw=1)
        ax.axhline(c["score_i_star_h1_H2"], color="#333", ls=":", lw=1.2)
        f = c["f_star"]
        if isinstance(f, str):
            ax.scatter(h1.loc[f, "score_deteccion"], h2.loc[f, "score_deteccion"], marker="*", s=420,
                       color=color_de(f), edgecolor="black", lw=1.2, zorder=5)
            ax.annotate(f"F* {f}", (h1.loc[f, "score_deteccion"], h2.loc[f, "score_deteccion"]), xytext=(-10, 12),
                        textcoords="offset points", ha="right", fontsize=9, weight="bold")
        i = c["i_star_h1"]
        ax.annotate(f"I*_H1 {i}", (h1.loc[i, "score_deteccion"], h2.loc[i, "score_deteccion"]), xytext=(6, -14),
                    textcoords="offset points", fontsize=9)
        rho = h1.loc[fusion, "score_deteccion"].corr(h2.loc[fusion, "score_deteccion"], method="spearman")
        ax.set_xlim(0, 0.95)
        ax.set_ylim(0, 0.8)
        ax.set_xlabel("score en H1 (donde se elige) →")
        ax.set_ylabel("score en H2 (donde se evalúa) →")
        ax.set_title(f"Pista {p} · Spearman H1–H2 de las fusiones = {num(rho, 2)}")
    handles = [Line2D([], [], marker="o", ls="", color=COLOR_FUSION[k], label=lab)
               for k, lab in FILAS[1:] if k in COLOR_FUSION]
    handles += [Line2D([], [], marker="s", ls="", color="#888", mec="black", label="individual"),
                Line2D([], [], ls=":", color="#333", label="score de I*_H1 en H2 (vara de iii)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False, ncol=6, fontsize=9)
    fig.suptitle("Lo que gana en la primera mitad no es lo que gana en la segunda", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def tabla_seleccion(d: dict) -> pd.DataFrame:
    """Escalera de la selección por pista: I*, F*, la elegida con todo el OOS, los ganadores de
    cada familia en H1 y las variantes sin peso, con su score en todo el OOS y en H2."""
    crit, ev, opt, gan, var = d["criterio"], d["evaluacion"], d["optimismo"], d["ganadores"], d["variantes"]
    filas = []
    for p in PISTAS:
        c = _crit(crit, p)
        o = opt.set_index("pista").loc[p]
        oos, h2 = _sel(ev, p, "oos", fj.GRUPOS), _sel(ev, p, "H2", fj.GRUPOS)

        def fila(papel, i, como):
            filas.append({"pista": p, "papel": papel, "id": i, "cómo se elige": como,
                          "score_oos": oos.loc[i, "score_deteccion"], "score_H2": h2.loc[i, "score_deteccion"],
                          "nivel_oos": int(oos.loc[i, "nivel"])})

        fila("I* (mejor individual)", c["i_star"], "todo el OOS (ranking v3)")
        fila("I*_H1", c["i_star_h1"], "H1")
        fila("F* (criterio)", c["f_star"], "H1, las 3 familias juntas")
        fila("mejor fusión optimista", o["mejor_oos"], "todo el OOS (sin control)")
        for _, g in gan[gan["pista"] == p].iterrows():
            if isinstance(g["ganador_H1"], str):
                fila(f"ganador familia ({g['familia']})", g["ganador_H1"], "H1, dentro de su familia")
        for _, v in var[(var["pista"] == p) & (var["lectura"] == "mejor_H1")].iterrows():
            if isinstance(v["id"], str):
                fila("«tras aviso» (sin peso)" if v["grupo"] == "tras_aviso" else "«confirmado» (sin peso)",
                     v["id"], "H1, dentro de su grupo")
    return pd.DataFrame(filas)


def figura_seleccion(tabla: pd.DataFrame):
    """Escalera de :func:`tabla_seleccion`: score en todo el OOS (lleno) y en H2 (hueco)."""
    plt = _plt()
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=False)
    for ax, p in zip(axes, PISTAS):
        t = tabla[tabla["pista"] == p].reset_index(drop=True)
        ref = t.loc[t["papel"].str.startswith("I* "), "score_oos"].iloc[0]
        for y, r in t.iterrows():
            col = color_de(r["id"]) if not str(r["id"]).startswith(("TRAS_", "CONF_")) else "#999999"
            ax.plot([r["score_H2"], r["score_oos"]], [y, y], color="#ddd", lw=2, zorder=1)
            ax.scatter(r["score_oos"], y, s=80 if "F*" not in r["papel"] else 300,
                       marker="*" if "F*" in r["papel"] else "o", color=col, edgecolor="black", lw=0.6, zorder=3)
            ax.scatter(r["score_H2"], y, s=60, facecolor="white", edgecolor=col, lw=1.5, zorder=3)
            ax.annotate(r["id"], (max(r["score_oos"], r["score_H2"]), y), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8)
        ax.axvline(ref, color="#333", ls=":", lw=1.2)
        ax.set_yticks(range(len(t)), t["papel"], fontsize=9)
        ax.set_ylim(len(t) - 0.5, -0.5)
        ax.set_xlim(0.25, 0.85)
        ax.set_xlabel("score de detección →")
        ax.set_title(f"Pista {p} · línea punteada: I* en todo el OOS")
    handles = [Line2D([], [], marker="o", ls="", color="#777", label="todo el OOS"),
               Line2D([], [], marker="o", ls="", mfc="white", mec="#777", label="solo H2")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False, ncol=2, fontsize=9)
    fig.suptitle("Cuánto vale cada elección: con todo el OOS se gana, eligiendo en H1 no", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


# --------------------------------------------------------------------------- #
# 5 · Criterio y MCS-F
# --------------------------------------------------------------------------- #
def tabla_criterio(crit: pd.DataFrame) -> pd.DataFrame:
    """(i), (ii) y (iii) del §6.1 por pista, con las cifras que se comparan y ✓/✗."""
    filas = []
    for p in PISTAS:
        c = _crit(crit, p)
        ok = lambda b: "✓" if bool(b) else "✗"  # noqa: E731
        filas += [
            {"pista": p, "condición": "(i) nivel 0 y score OOS > mejor individual",
             "F*": f"{c['f_star']} {num(c['score_f_star_oos'])} (nivel {int(c['nivel_f_star_oos'])})",
             "frente a": f"{c['i_star']} {num(c['score_i_star_oos'])}", "cumple": ok(c["cumple_i"])},
            {"pista": p, "condición": "(ii) dentro del MCS-F al 10 % (todo el OOS)",
             "F*": "dentro" if c["f_star_en_mcs"] else "fuera", "frente a": "MCS-F", "cumple": ok(c["cumple_ii"])},
            {"pista": p, "condición": "(iii) en H2: nivel 0 y score > I*_H1",
             "F*": f"{num(c['score_f_star_H2'])} (nivel {int(c['nivel_f_star_H2'])})",
             "frente a": f"{c['i_star_h1']} {num(c['score_i_star_h1_H2'])}", "cumple": ok(c["cumple_iii"])},
            {"pista": p, "condición": "**pista**", "F*": "", "frente a": "", "cumple": ok(c["cumple_pista"])},
        ]
    return pd.DataFrame(filas).set_index(["pista", "condición"])


def figura_mcs(mcs: dict[str, pd.DataFrame], crit: pd.DataFrame, titulo: str = "todo el OOS"):
    """Pérdida media (1 − J de Youden) de cada modelo del MCS-F; opaco = dentro del MCS al 10 %."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    for ax, p in zip(axes, PISTAS):
        t = mcs[p].sort_values("perdida_media")
        f = _crit(crit, p)["f_star"]
        cols = ["black" if i == f else color_de(i) for i in t.index]
        alfa = [1.0 if inc else 0.28 for inc in t["incluido"]]
        y = np.arange(len(t))
        for yi, (i, r), col, a in zip(y, t.iterrows(), cols, alfa):
            ax.barh(yi, r["perdida_media"], color=col, alpha=a, edgecolor="black" if i == f else "none", lw=1.5)
            ax.text(r["perdida_media"] + 0.005, yi, f"p = {num(r['p_mcs'], 3)}", va="center", fontsize=7.5,
                    color="#444")
        ax.set_yticks(y, [f"★ {i}" if i == f else i for i in t.index], fontsize=8.5)
        ax.set_ylim(len(t) - 0.5, -0.5)
        ax.set_xlim(0.4, 1.12)
        ax.set_xlabel("pérdida media 1 − J (menos es mejor) →")
        n_in = int(t["incluido"].sum())
        ax.set_title(f"Pista {p} · {n_in} de {len(t)} dentro del MCS · F* {'dentro' if t.loc[f, 'incluido'] else 'fuera'}")
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"MCS-F al 10 % ({titulo}): individuales, baselines (gris) y F* (negro); claro = fuera",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 6 · Lecturas sin peso
# --------------------------------------------------------------------------- #
def tabla_variantes(var: pd.DataFrame) -> pd.DataFrame:
    """«Confirmación tras aviso» y «confirmado» puro: mejor de H1 y mejor de todo el OOS."""
    t = var.copy()
    t["grupo"] = t["grupo"].map({"tras_aviso": "«tras aviso»", "confirmado": "«confirmado» puro"})
    t["lectura"] = t["lectura"].map({"mejor_H1": "elegida en H1", "mejor_oos": "elegida con todo el OOS",
                                     "de_f_star": "la de F*"})
    return t.set_index(["pista", "grupo", "lectura"])


def tabla_utilidad(util: pd.DataFrame, tramos: Sequence[str] = ("oos", "H2")) -> pd.DataFrame:
    """Utilidad operativa (lectura de los notebooks antiguos) de F*, V, I* e I*_H1."""
    t = util[util["tramo"].isin(tramos)]
    cols = ["mean_crisis_coverage", "false_alarm_rate", "switching_rate", "operational_utility"]
    out = t.set_index(["pista", "tramo", "papel", "id"])[cols]
    return out.rename(columns={"mean_crisis_coverage": "cobertura", "false_alarm_rate": "falsas_alarmas",
                               "switching_rate": "cambios", "operational_utility": "utilidad"})


def tabla_topn(topn: pd.DataFrame) -> pd.DataFrame:
    t = topn.copy()
    for c in ("top_v3_oos", "top_H1", "solo_en_v3", "solo_en_H1"):
        t[c] = t[c].fillna("").str.replace("|", " · ", regex=False)
    return t[["pista", "n", "top_v3_oos", "top_H1", "solo_en_v3", "solo_en_H1"]].set_index(["pista", "n"])


# --------------------------------------------------------------------------- #
# 7 · Cronograma
# --------------------------------------------------------------------------- #
def _rachas(s: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Intervalos ``[inicio, fin]`` de las rachas ``True`` de una serie booleana."""
    s = pd.Series(s).astype(bool)
    if not s.any():
        return []
    v = s.to_numpy()
    d = np.diff(np.r_[0, v.astype(int), 0])
    ini, fin = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    return [(s.index[a], s.index[b]) for a, b in zip(ini, fin)]


def filas_cronograma(crit: pd.DataFrame, elec: pd.DataFrame, pista: str) -> list[str]:
    """F*, sus miembros, I* e I*_H1 (sin repetir)."""
    c, e = _crit(crit, pista), elec.set_index("pista").loc[pista]
    f = c["f_star"]
    miembros = str(e["f_star_miembros"]).split("|") if isinstance(f, str) else []
    return list(dict.fromkeys([f, *miembros, c["i_star"], c["i_star_h1"]] if isinstance(f, str)
                              else [c["i_star"], c["i_star_h1"]]))


def figura_cronograma(crit: pd.DataFrame, elec: pd.DataFrame, pista: str, salida: Path = FIJAS):
    """Días de crisis de F*, sus miembros, I* e I*_H1 sobre las ventanas de crisis de la pista."""
    import matplotlib.dates as mdates

    plt = _plt()
    c = _crit(crit, pista)
    ids = filas_cronograma(crit, elec, pista)
    sen = fj.leer_senales(pista, columnas=ids, salida=salida)
    h1, h2 = mitades(sen.index)
    fig, ax = plt.subplots(figsize=(14, 0.55 * len(ids) + 1.6))
    for a, b in ventanas(pista).values():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if b >= sen.index[0]:
            ax.axvspan(max(a, sen.index[0]), b, color=COLOR_CRISIS, alpha=0.18, lw=0, zorder=0)
    for y, i in enumerate(ids):
        tramos = [(mdates.date2num(a), max(mdates.date2num(b) - mdates.date2num(a), 1.0))
                  for a, b in _rachas(sen[i])]
        ax.broken_barh(tramos, (y - 0.32, 0.64), color="black" if i == c["f_star"] else color_de(i), lw=0)
    ax.axvline(h2[0], color="#333", ls="--", lw=1.2)
    ax.text(h2[0], -0.75, "  H2 →", fontsize=9, va="center")
    ax.text(h1[-1], -0.75, "← H1  ", fontsize=9, va="center", ha="right")
    etiquetas = []
    for i in ids:
        papel = ("F*" if i == c["f_star"] else "I*" if i == c["i_star"] else "I*_H1" if i == c["i_star_h1"]
                 else "miembro")
        if i == c["i_star"] and i == c["i_star_h1"]:
            papel = "I* = I*_H1"
        if i == c["i_star_h1"] and papel == "miembro":
            papel = "I*_H1"
        etiquetas.append(f"{i}  ({papel})")
    ax.set_yticks(range(len(ids)), etiquetas, fontsize=9)
    ax.set_ylim(len(ids) - 0.4, -1.1)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator(5 if pista == "A" else 2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(sen.index[0], sen.index[-1])
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Pista {pista} · días marcados como crisis (barras) sobre las ventanas de crisis (rojo claro)",
                 fontsize=11)
    fig.tight_layout()
    return fig


def cobertura_por_crisis(crit: pd.DataFrame, elec: pd.DataFrame, pista: str, salida: Path = FIJAS
                         ) -> pd.DataFrame:
    """Fracción de días de cada ventana de crisis (con días OOS) que marca cada señal del cronograma."""
    ids = filas_cronograma(crit, elec, pista)
    sen = fj.leer_senales(pista, columnas=ids, salida=salida)
    filas = {}
    for nombre, (a, b) in ventanas(pista).items():
        dias = sen.loc[pd.Timestamp(a):pd.Timestamp(b)]
        if len(dias):
            filas[nombre.replace("_", " ")] = dias.mean()
    out = pd.DataFrame(filas).T
    out["días OOS"] = [len(sen.loc[pd.Timestamp(a):pd.Timestamp(b)]) for a, b in ventanas(pista).values()
                       if len(sen.loc[pd.Timestamp(a):pd.Timestamp(b)])]
    return out
