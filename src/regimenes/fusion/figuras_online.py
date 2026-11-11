"""Tablas y figuras del notebook 16 (fusión online C1, ADR-011 §4, §5 y §6.2).

**Solo lee y dibuja.** Todo sale de lo que :mod:`regimenes.fusion.online` dejó en
``results/fusion/online/`` (señales, ``p̄``, pesos, evaluación con la vara, MCS-O, rejilla de η,
sensibilidades y utilidad) y, para el cronograma, de las señales y ``p_crisis`` de los
candidatos (:mod:`regimenes.fusion.vara`). No recalcula pesos ni MCS ni escribe en disco.

Paleta: los pesos de cada detector usan el color de su familia de
:mod:`regimenes.benchmark.panorama` (un tono por miembro de la familia; rayado = variante del
pool). Los comparadores del §4.5 tienen color propio: O negro, M gris medio, V gris claro, F*
verde azulado e I* el color de la familia del detector.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from regimenes.benchmark import panorama as pn
from regimenes.fusion import online as on
from regimenes.fusion import vara

PISTAS: tuple[str, ...] = on.PISTAS
ID_O = on.id_senal(on.PRINCIPAL, on.ETA_PRINCIPAL)
ID_M = on.id_senal(on.PRINCIPAL, 0.0)
COMPARADORES: tuple[str, ...] = ("O", "M", "V", "F*", "I*")
COLOR_COMP = {"O": "#111111", "M": "#8f8f8f", "V": "#cdcdcd", "F*": "#2a9d8f"}
NOMBRE_COMP = {
    "O": "O · online (η = 10)",
    "M": "M · media simple (η = 0)",
    "V": "V · votación por mayoría",
    "F*": "F* · mejor fija (elegida en H1)",
    "I*": "I* · mejor individual",
}
NOMBRE_VARIANTE = {
    "ONLINE_principal_eta10": "principal: p_crisis, η = 10",
    "ONLINE_binaria_eta10": "entrada binaria (η = 10)",
    "ONLINE_retraso63_eta10": "63 sesiones de retraso de reconocimiento",
    "ONLINE_min21_eta10": "pesos iguales hasta 21 días de crisis",
    "ONLINE_excluir_sin_terminar_eta10": "sin días de crisis aún sin terminar",
    "ONLINE_binaria_eta0": "binaria con η = 0 (= V)",
}
COLOR_TRAMO = {"oos": "#111111", "H1": "#E69F00", "H2": "#0072B2"}
NOMBRE_TRAMO = {"oos": "todo el OOS", "H1": "H1 (primera mitad)", "H2": "H2 (segunda mitad)"}


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _plt():
    return pn._plt()


def _si(valor: bool) -> str:
    return "✓" if bool(valor) else "✗"


def _fmt(x: float, d: int = 3) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def _eta_txt(eta: float) -> str:
    return "∞" if math.isinf(float(eta)) else f"{float(eta):g}"


def colores_detectores(ids: Iterable[str]) -> dict[str, str]:
    """Color de cada detector: el de su familia (``panorama.COLOR_FAMILIA``), en tonos distintos
    para los miembros de una misma familia (más oscuro el primero por id, más claro el último)."""
    from matplotlib.colors import to_hex, to_rgb

    grupos: dict[str, list[str]] = {}
    for d in ids:
        grupos.setdefault(pn.familia_de(d), []).append(str(d))
    out: dict[str, str] = {}
    for fam, ds in grupos.items():
        base = np.array(to_rgb(pn.COLOR_BASELINE if fam == "baseline" else pn.COLOR_FAMILIA[fam]))
        tonos = [0.0] if len(ds) == 1 else np.linspace(-0.35, 0.55, len(ds))
        for d, t in zip(sorted(ds), tonos):
            c = base * (1 + t) if t < 0 else base + (1 - base) * t
            out[d] = to_hex(np.clip(c, 0, 1))
    return out


def orden_detectores(ids: Iterable[str]) -> list[str]:
    """Por familia y, dentro, por id (los tonos de una familia quedan juntos en el área apilada)."""
    return sorted(map(str, ids), key=lambda d: (pn.familia_de(d), d))


def es_variante_pool(detector_id: str) -> bool:
    return "-" in str(detector_id)


def color_comparador(comp: str, detector_id: str | None = None) -> str:
    if comp == "I*" or comp not in COLOR_COMP:
        return pn.color_de(detector_id) if detector_id else "#555555"
    return COLOR_COMP[comp]


def _ventanas_oos(pista: str) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Ventanas ``[pico, suelo]`` recortadas al OOS (solo las que tienen días OOS)."""
    idx = vara.indice(pista)
    out = {}
    for c, (a, b) in vara.ventanas(pista).items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if ((idx >= a) & (idx <= b)).any():
            out[c] = (max(a, idx[0]), min(b, idx[-1]))
    return out


def _trampas_oos(pista: str) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    from regimenes.evaluacion.ranking import track_false_positive_windows

    idx = vara.indice(pista)
    out = {}
    for c, (a, b) in track_false_positive_windows().items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if ((idx >= a) & (idx <= b)).any():
            out[c] = (a, b)
    return out


def _sombrear(ax, pista: str, *, etiquetas: bool = False, alpha: float = 0.10) -> None:
    from regimenes.viz import C_FP, C_LONG

    for i, (c, (a, b)) in enumerate(_ventanas_oos(pista).items()):
        ax.axvspan(a, b, color=C_LONG, alpha=alpha, lw=0,
                   label="crisis [pico, suelo]" if etiquetas and i == 0 else None)
    for i, (a, b) in enumerate(_trampas_oos(pista).values()):
        ax.axvspan(a, b, color=C_FP, alpha=0.25, lw=0, label="trampa (no crisis)" if etiquetas and i == 0 else None)


# --------------------------------------------------------------------------- #
# Tablas
# --------------------------------------------------------------------------- #
def tabla_comparadores(pista: str) -> pd.DataFrame:
    """O, M, V, F* e I* con score, nivel y MCS en todo el OOS y en H2.

    I* es el puesto 1 de todo el OOS en la columna ``oos`` y el mejor de H1 (I*_H1) en la
    columna ``H2``: son las referencias de los criterios (i) y (iii). Comprueba que los scores
    coinciden con ``criterio_{P}.json``."""
    p = pista.upper()
    crit = on.leer_criterio_pista(p)
    ev = {t: on.leer_evaluacion(p, t) for t in ("oos", "H2")}
    mcs = {t: on.leer_mcs(p, t) for t in ("oos", "H2")}
    filas = []
    for c in COMPARADORES:
        if c == "F*" and not crit.get("F*"):
            continue
        id_oos = crit["I*"] if c == "I*" else c
        id_h2 = crit["I*_H1"] if c == "I*" else c
        filas.append({
            "comparador": c,
            "id_oos": crit["F*"] if c == "F*" else id_oos,
            "id_H2": crit["F*"] if c == "F*" else id_h2,
            "score_oos": float(ev["oos"].loc[id_oos, "score_deteccion"]),
            "score_H2": float(ev["H2"].loc[id_h2, "score_deteccion"]),
            "nivel_oos": int(ev["oos"].loc[id_oos, "nivel"]),
            "nivel_H2": int(ev["H2"].loc[id_h2, "nivel"]),
            "recall_oos": float(ev["oos"].loc[id_oos, "det_event_recall"]),
            "precision_oos": float(ev["oos"].loc[id_oos, "det_precision"]),
            "mcs_oos": bool(mcs["oos"].loc[id_oos, "incluido"]),
            "p_mcs_oos": float(mcs["oos"].loc[id_oos, "p_mcs"]),
            "mcs_H2": bool(mcs["H2"].loc[id_h2, "incluido"]),
        })
    t = pd.DataFrame(filas).set_index("comparador")
    claves = {"O": ("score_O", "score_O_H2"), "M": ("score_M", "score_M_H2"), "V": ("score_V", "score_V_H2"),
              "F*": ("score_F*", "score_F*_H2"), "I*": ("score_I*", "score_I*_H1_en_H2")}
    for c in t.index:
        a, b = claves[c]
        if not (np.isclose(t.loc[c, "score_oos"], crit[a], atol=1e-12)
                and np.isclose(t.loc[c, "score_H2"], crit[b], atol=1e-12)):
            raise AssertionError(f"[{p}] {c}: la evaluación y criterio_{p}.json no coinciden.")
    return t


def tabla_criterio() -> pd.DataFrame:
    """Criterio del §6.2 por pista: (i), (ii) y (iii) con sus cifras, y si la pista cumple."""
    filas = []
    for p in PISTAS:
        c = on.leer_criterio_pista(p)
        mcs = on.leer_mcs(p, "oos")
        filas += [
            {"pista": p, "criterio": "(i) O > I* en todo el OOS, nivel 0",
             "O": c["score_O"], "referencia": c["score_I*"], "ref": f"I* = {c['I*']}", "cumple": c["i"]},
            {"pista": p, "criterio": "(ii) O dentro del MCS-O al 10 % (todo el OOS)",
             "O": float(mcs.loc["O", "p_mcs"]), "referencia": 0.10, "ref": "p_MCS > α", "cumple": c["ii"]},
            {"pista": p, "criterio": "(iii) O > I*_H1 en H2, nivel 0",
             "O": c["score_O_H2"], "referencia": c["score_I*_H1_en_H2"], "ref": f"I*_H1 = {c['I*_H1']}",
             "cumple": c["iii"]},
            {"pista": p, "criterio": "→ la pista cumple (i)+(ii)+(iii)", "O": np.nan, "referencia": np.nan,
             "ref": "", "cumple": c["cumple"]},
        ]
    t = pd.DataFrame(filas)
    t["cumple"] = t["cumple"].map(_si)
    return t.set_index(["pista", "criterio"])


def tabla_aprenden() -> pd.DataFrame:
    """Veredicto secundario «los pesos aprenden»: O frente a M en todo el OOS y en H2."""
    filas = []
    for p in PISTAS:
        c = on.leer_criterio_pista(p)
        for t, so, sm, ok in (("oos", c["score_O"], c["score_M"], c["O_gt_M_oos"]),
                              ("H2", c["score_O_H2"], c["score_M_H2"], c["O_gt_M_H2"])):
            filas.append({"pista": p, "tramo": t, "O": so, "M": sm, "O − M": so - sm, "O > M": _si(ok)})
    return pd.DataFrame(filas).set_index(["pista", "tramo"])


def tabla_eta() -> pd.DataFrame:
    t = pd.concat([on.leer_eta(p) for p in PISTAS], ignore_index=True)
    t["η"] = t["eta"].map(_eta_txt)
    cols = ["pista", "η", "puesto_H1", "nivel_oos", "score_deteccion_oos", "nivel_H1", "score_deteccion_H1",
            "nivel_H2", "score_deteccion_H2"]
    return t[cols].set_index(["pista", "η"])


def tabla_sensibilidades() -> pd.DataFrame:
    """Score de la principal y de las cuatro sensibilidades del §6.3 (más binaria η = 0), por tramo."""
    filas = []
    for p in PISTAS:
        c = on.leer_criterio_pista(p)
        eta = on.leer_eta(p).set_index("id")
        filas.append({"pista": p, "id": ID_O, **{t: float(eta.loc[ID_O, f"score_deteccion_{t}"])
                                                 for t in ("oos", "H1", "H2")}})
        for sid, v in c["sensibilidades"].items():
            filas.append({"pista": p, "id": sid, **{t: float(v[t]) for t in ("oos", "H1", "H2")}})
    t = pd.DataFrame(filas)
    t["variante"] = t["id"].map(lambda s: NOMBRE_VARIANTE.get(s, s))
    base = t[t["id"] == ID_O].set_index("pista")
    for tr in ("oos", "H2"):
        t[f"Δ {tr} vs principal"] = t[tr] - t["pista"].map(base[tr])
    return t.set_index(["pista", "variante"]).drop(columns="id")


def tabla_utilidad() -> pd.DataFrame:
    """Utilidad operativa (sin peso, §5.1) de O, M, V, F* e I* en todo el OOS y en H2."""
    filas = []
    for p in PISTAS:
        u = on.leer_utilidad(p)
        u = u[u["tramo"].isin(["oos", "H2"])].reset_index().assign(pista=p)
        filas.append(u)
    t = pd.concat(filas, ignore_index=True)
    cols = ["pista", "tramo", "id", "mean_crisis_coverage", "false_alarm_rate", "mean_duration", "operational_utility"]
    return t[cols].set_index(["pista", "tramo", "id"])


def resumen_pesos(pista: str) -> dict[str, object]:
    """Concentración de los pesos de O: n efectivo (``exp`` de la entropía) y peso del líder."""
    W = on.leer_pesos(pista)
    cols = [c for c in W.columns if c not in on._COLS_META_PESOS]
    apr = W[W["aprendido"]]
    lider = apr[cols].idxmax(axis=1)
    peso_lider = apr[cols].max(axis=1)
    finales = on.pesos_finales(pista, top=None)
    return {
        "n": len(cols),
        "n_meses": len(W),
        "n_meses_aprendidos": int(W["aprendido"].sum()),
        "n_efectivo_medio": float(apr["entropia_efectiva"].mean()),
        "n_efectivo_min": float(apr["entropia_efectiva"].min()),
        "n_efectivo_final": float(W["entropia_efectiva"].iloc[-1]),
        "frac_meses_lider_mayoria": float((peso_lider > 0.5).mean()),
        "lider_mas_frecuente": str(lider.value_counts().index[0]),
        "frac_lider_mas_frecuente": float(lider.value_counts(normalize=True).iloc[0]),
        "top3_final": [(str(k), float(v)) for k, v in finales.head(3).items()],
        "primer_tau": str(pd.Timestamp(apr["tau"].iloc[0]).date()),
        "primera_vigencia": str(pd.Timestamp(apr["vigencia"].iloc[0]).date()),
    }


def hechos() -> dict[str, object]:
    """Cifras para el resumen y las conclusiones (todas leídas de ``results/fusion/online``)."""
    res = on.leer_resumen()
    h: dict[str, object] = {"veredicto": res["veredicto"], "aprenden": bool(res["los_pesos_aprenden"]),
                            "cumple": res["cumple_por_pista"]}
    for p in PISTAS:
        c = on.leer_criterio_pista(p)
        c["mcs_p_O"] = float(on.leer_mcs(p, "oos").loc["O", "p_mcs"])
        c["pesos"] = resumen_pesos(p)
        h[p] = c
    return h


def _fallos(c: dict) -> list[str]:
    return [k for k in ("i", "ii", "iii") if not c[k]]


def conclusiones(h: dict) -> list[str]:
    """3–5 conclusiones con las cifras de :func:`hechos` (el texto se adapta a los resultados)."""
    out = []
    si = [p for p in PISTAS if h["cumple"][p]]
    no = [p for p in PISTAS if not h["cumple"][p]]
    txt = f"**Veredicto del §6.2: «{h['veredicto']}».** "
    for p in si:
        c = h[p]
        txt += (f"En {p}, O cumple los tres criterios: {c['score_O']:.3f} frente a {c['score_I*']:.3f} de "
                f"{c['I*']} en todo el OOS, dentro del MCS-O (p = {c['mcs_p_O']:.2f}) y {c['score_O_H2']:.3f} "
                f"frente a {c['score_I*_H1_en_H2']:.3f} de {c['I*_H1']} en H2. ")
    for p in no:
        c = h[p]
        partes = []
        if not c["i"]:
            partes.append(f"(i): {c['score_O']:.3f} frente a {c['score_I*']:.3f} de {c['I*']}")
        if not c["ii"]:
            partes.append(f"(ii): fuera del MCS-O (p = {c['mcs_p_O']:.2f})")
        if not c["iii"]:
            partes.append(f"(iii): {c['score_O_H2']:.3f} frente a {c['score_I*_H1_en_H2']:.3f} de {c['I*_H1']}")
        cumple = [k for k in ("i", "ii", "iii") if c[k]]
        txt += (f"En {p} falla {'; '.join(partes)}"
                + (f" (sí cumple {', '.join(f'({k})' for k in cumple)})." if cumple else "."))
    out.append(txt)

    a = []
    for p in PISTAS:
        c = h[p]
        a.append(f"{p}: O {c['score_O']:.3f} vs M {c['score_M']:.3f} en todo el OOS y "
                 f"{c['score_O_H2']:.3f} vs {c['score_M_H2']:.3f} en H2")
    pierde = [p for p in PISTAS if not (h[p]["O_gt_M_oos"] and h[p]["O_gt_M_H2"])]
    gana = [p for p in PISTAS if p not in pierde]
    out.append(f"**Los pesos aprenden: {'sí' if h['aprenden'] else 'no'}** (hace falta O > M en las dos pistas y "
               "en los dos tramos). " + "; ".join(a) + ". "
               + (f"En {', '.join(pierde)} la media simple, con pesos iguales, iguala o supera a O en algún tramo, "
                  "así que allí la ventaja de O no se puede atribuir a aprender los pesos"
                  + (f"; en {', '.join(gana)}, en cambio, O supera a M en los dos tramos." if gana else ".")
                  if pierde else "O supera a M en todos los tramos."))

    fs = []
    for p in PISTAS:
        c = h[p]
        if c.get("F*"):
            gana = "O" if c["score_O"] > c["score_F*"] else "F*"
            fs.append(f"en {p} gana {gana} ({c['score_O']:.3f} O vs {c['score_F*']:.3f} F* = {c['F*']})")
    if fs:
        out.append("**Frente a la mejor fusión fija (F\\*, todo el OOS):** " + "; ".join(fs) + ".")

    e = []
    for p in PISTAS:
        c = h[p]
        if float(c["eta_H1"]) == on.ETA_PRINCIPAL:
            e.append(f"en {p} gana en H1 el mismo η = 10 que se fijó a priori")
        else:
            e.append(f"en {p} gana η = {_eta_txt(c['eta_H1'])} en H1, que en H2 da {c['score_eta_H1_en_H2']:.3f} "
                     f"frente a {c['score_O_H2']:.3f} de η = 10")
    out.append("**Tasa de aprendizaje (sin peso):** " + "; ".join(e) + ".")

    w = []
    for p in PISTAS:
        r = h[p]["pesos"]
        top = ", ".join(f"{d} {v:.0%}" for d, v in r["top3_final"])
        w.append(f"en {p}, n efectivo medio {r['n_efectivo_medio']:.1f} de {r['n']} y al final {top}")
    out.append("**Los pesos no se quedan con un solo detector:** " + "; ".join(w) + ".")
    return out


# --------------------------------------------------------------------------- #
# Figura 0 · resumen
# --------------------------------------------------------------------------- #
def figura_resumen():
    """Score de O, M, V, F* e I* en todo el OOS (barra llena) y en H2 (rayada), por pista."""
    plt = _plt()
    from matplotlib.patches import Patch

    res = on.leer_resumen()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharex=True)
    for ax, p in zip(axes, PISTAS):
        t = tabla_comparadores(p)
        crit = on.leer_criterio_pista(p)
        n = len(t)
        for k, (c, r) in enumerate(t.iterrows()):
            col = color_comparador(c, r["id_oos"])
            col2 = color_comparador(c, r["id_H2"])
            ax.barh(k - 0.19, r["score_oos"], height=0.36, color=col, edgecolor="#333", lw=0.6)
            ax.barh(k + 0.19, r["score_H2"], height=0.36, color=col2, alpha=0.45, hatch="///", edgecolor="#333",
                    lw=0.6)
            marca = "●" if r["mcs_oos"] else "○"
            ax.text(r["score_oos"] + 0.008, k - 0.19, f"{r['score_oos']:.3f} {marca}", va="center", fontsize=9,
                    weight="bold" if c == "O" else None)
            ax.text(r["score_H2"] + 0.008, k + 0.19, f"{r['score_H2']:.3f}", va="center", fontsize=8.5,
                    color="#444")
        etiquetas = []
        for c, r in t.iterrows():
            if c == "I*":
                etiquetas.append(f"I* · {r['id_oos']}\n(H2: I*_H1 · {r['id_H2']})")
            elif c == "F*":
                etiquetas.append(f"F* · {r['id_oos']}")
            else:
                etiquetas.append(NOMBRE_COMP[c])
        ax.set_yticks(range(n), etiquetas, fontsize=9.5)
        ax.set_ylim(n - 0.4, -0.6)
        ax.axvline(t.loc["I*", "score_oos"], color="#333", ls="--", lw=1)
        ax.axvline(t.loc["I*", "score_H2"], color="#333", ls=":", lw=1)
        ax.set_xlim(0, 0.85)
        ax.set_xlabel("score de detección →")
        ax.set_title(f"Pista {p}: {'cumple' if crit['cumple'] else 'no cumple'}  ·  (i) {_si(crit['i'])}  "
                     f"(ii) {_si(crit['ii'])}  (iii) {_si(crit['iii'])}", fontsize=11.5)
    handles = [Patch(facecolor="#777", edgecolor="#333", label="todo el OOS"),
               Patch(facecolor="#777", alpha=0.45, hatch="///", edgecolor="#333", label="H2 (segunda mitad)"),
               plt.Line2D([], [], color="#333", ls="--", label="umbral (i): I* en todo el OOS"),
               plt.Line2D([], [], color="#333", ls=":", label="umbral (iii): I*_H1 en H2"),
               Patch(facecolor="none", edgecolor="none", label="● dentro / ○ fuera del MCS-O (todo el OOS)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=5, frameon=False, fontsize=9)
    cumple = [p for p in PISTAS if res["cumple_por_pista"][p]]
    fig.suptitle(f"Fusión online: «{res['veredicto']}» — cumple en {', '.join(cumple) or 'ninguna pista'}; "
                 f"los pesos aprenden (O > M en OOS y H2, las dos pistas): {'sí' if res['los_pesos_aprenden'] else 'no'}",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# --------------------------------------------------------------------------- #
# Figura 1 · la idea
# --------------------------------------------------------------------------- #
def figura_mecanismo(pista: str = "A"):
    """(a) Calendario de los pesos sobre un esquema de meses; (b) de la pérdida al peso con η.

    En (b), la banda gris es la diferencia de pérdida entre el mejor y el peor candidato en el
    último τ de ``pista``."""
    plt = _plt()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 4.9), gridspec_kw={"width_ratios": [1.6, 1]})

    # (a) esquema
    meses = 9
    for m in range(meses + 1):
        ax.axvline(m, color="#bbb", lw=0.8, zorder=0)
    for m in range(meses):
        ax.text(m + 0.5, -0.32, f"mes {m + 1}", ha="center", fontsize=8.5, color="#555")
        ax.plot(m + 1, 0, marker="|", ms=14, color="#333")
    ax.text(meses + 0.05, 0.05, "τ = cierre\nde cada mes", fontsize=8, color="#333", va="bottom")
    ax.axvspan(1.3, 3.6, ymin=0.25, ymax=0.62, color="#2b5d8a", alpha=0.18, lw=0)
    ax.text(2.45, 0.55, "crisis [pico, suelo]", ha="center", fontsize=9, color="#2b5d8a", weight="bold")
    ax.text(2.45, 0.38, "mientras dura, sus días\ncuentan como calma en L", ha="center", fontsize=8, color="#2b5d8a")
    ax.annotate("suelo: la crisis\nya ha terminado", xy=(3.6, 0.25), xytext=(4.3, 0.42), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#333"))
    ax.fill_between([0, 4], 1.0, 1.25, color="#cfcfcf", lw=0)
    ax.text(2.0, 1.12, "pesos iguales (aún no hay crisis terminada)", ha="center", va="center", fontsize=9)
    ax.fill_between([4, meses], 1.0, 1.25, color="#111", alpha=0.85, lw=0)
    ax.text(6.5, 1.12, "w_i ∝ exp(−η·L_i), L_i = 1 − J con las crisis ya terminadas", ha="center",
            va="center", fontsize=9, color="white")
    ax.annotate("", xy=(4.5, 0.95), xytext=(4.0, 0.05),
                arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.6, connectionstyle="arc3,rad=-0.3"))
    ax.text(4.65, 0.72, "se calculan en τ\n(cierre del mes 4)\ny rigen todo el mes 5", fontsize=8.5,
            color="#D55E00")
    ax.annotate("", xy=(7.5, 0.95), xytext=(7.0, 0.05),
                arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.0, connectionstyle="arc3,rad=-0.3"))
    ax.text(7.6, 0.6, "se recalculan\nenteros cada\nmes", fontsize=8, color="#D55E00")
    ax.set_xlim(-0.1, meses + 0.9)
    ax.set_ylim(-0.45, 1.6)
    ax.axis("off")
    ax.set_title("(a) Cada día: p̄_t = Σ w_i · p_crisis_i,t   →   crisis si p̄_t ≥ 0,50 (sin persistencia)",
                 fontsize=11, loc="left")

    # (b) de la pérdida al peso
    W = on.leer_pesos(pista)
    L = on.leer_perdidas(pista)
    L = L[(L["retraso"] == 0) & (~L["excluir_sin_terminar"])]
    cols = [c for c in W.columns if c not in on._COLS_META_PESOS]
    ult = L.iloc[-1][cols].astype(float)
    rango = float(ult.max() - ult.min())
    x = np.linspace(0, 0.5, 200)
    cmap = plt.get_cmap("viridis")
    etas = [e for e in on.REJILLA_ETA if 0 < e < math.inf]
    for k, e in enumerate(etas):
        lw, col = (2.6, "#111") if e == on.ETA_PRINCIPAL else (1.2, cmap(k / max(len(etas) - 1, 1)))
        bx.plot(x, np.exp(-e * x), color=col, lw=lw, label=f"η = {e:g}" + (" (principal)" if e == on.ETA_PRINCIPAL
                                                                             else ""))
    bx.axhline(1, color="#8f8f8f", lw=1.2, ls="--", label="η = 0 (media simple M)")
    bx.axvspan(0, rango, color="#999", alpha=0.12, lw=0)
    bx.text(rango, 0.92, f"pista {pista}, último τ:\nmejor-peor ΔL = {rango:.2f}", ha="right", fontsize=8.5)
    bx.plot([0.1], [math.exp(-on.ETA_PRINCIPAL * 0.1)], "o", color="#D55E00")
    bx.annotate(f"ΔL = 0,1 → peso ×{math.exp(-on.ETA_PRINCIPAL * 0.1):.2f}", (0.1, math.exp(-on.ETA_PRINCIPAL * 0.1)),
                xytext=(0.17, 0.5), fontsize=8.5, arrowprops=dict(arrowstyle="->", color="#D55E00"))
    bx.set_xlabel("ΔL = pérdida del detector − pérdida del mejor")
    bx.set_ylabel("peso relativo al mejor")
    bx.set_ylim(0, 1.08)
    bx.legend(fontsize=8, ncol=2, loc="upper right", bbox_to_anchor=(1, 0.8), frameon=False)
    bx.set_title("(b) De la pérdida al peso: η es una temperatura (η = ∞: todo al líder)", fontsize=11, loc="left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figura 2 · los pesos
# --------------------------------------------------------------------------- #
def serie_pesos(pista: str) -> pd.DataFrame:
    """Pesos de O vigentes desde cada fecha (escalonados): fila 0 = inicio del OOS con pesos
    iguales; cada τ rige desde su ``vigencia``; la última fila repite los pesos hasta el fin del OOS."""
    W = on.leer_pesos(pista)
    cols = orden_detectores(c for c in W.columns if c not in on._COLS_META_PESOS)
    idx = vara.indice(pista)
    v = W[W["vigencia"].notna()]
    S = pd.DataFrame(v[cols].to_numpy(float), index=pd.DatetimeIndex(v["vigencia"]), columns=cols)
    S["n_efectivo"] = v["entropia_efectiva"].to_numpy(float)
    ini = pd.DataFrame([[1.0 / len(cols)] * len(cols) + [float(len(cols))]], index=[idx[0]], columns=S.columns)
    S = pd.concat([ini, S])
    S = S[~S.index.duplicated(keep="last")]
    fin = S.iloc[[-1]].copy()
    fin.index = [idx[-1]]
    return pd.concat([S, fin])


def figura_pesos(pista: str):
    """Área apilada de los pesos de O (η = 10) por detector y su número efectivo de detectores."""
    plt = _plt()
    from matplotlib.patches import Patch

    S = serie_pesos(pista)
    cols = [c for c in S.columns if c != "n_efectivo"]
    colores = colores_detectores(cols)
    r = resumen_pesos(pista)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(15, 6.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.1]})
    polys = ax.stackplot(S.index, S[cols].to_numpy().T, colors=[colores[c] for c in cols], step="post", lw=0)
    for poly, c in zip(polys, cols):
        if es_variante_pool(c):
            poly.set_hatch("////")
            poly.set_edgecolor("white")
    W = on.leer_pesos(pista)
    suelos = W.loc[W["K"].diff().fillna(W["K"]) > 0, "tau"]
    win = _ventanas_oos(pista)
    for c, (a, b) in win.items():
        for z in (ax, bx):
            z.axvspan(a, b, color="#2b5d8a", alpha=0.08, lw=0)
        ax.plot([b], [1.015], marker="v", color="#2b5d8a", ms=6, clip_on=False)
    for z in (ax, bx):
        z.axvline(pd.Timestamp(r["primera_vigencia"]), color="#D55E00", lw=1.2, ls="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("peso en p̄")
    ax.margins(x=0)
    handles = [Patch(facecolor=colores[c], hatch="////" if es_variante_pool(c) else None, edgecolor="white", label=c)
               for c in cols]
    handles.append(plt.Line2D([], [], ls="", marker="v", color="#2b5d8a", label="suelo: crisis terminada"))
    handles.append(plt.Line2D([], [], ls="--", color="#D55E00", label=f"primeros pesos aprendidos\n(rigen desde {r['primera_vigencia'][:7]})"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8.5, frameon=False,
              title="color = familia", title_fontsize=8.5)
    top = ", ".join(f"{d} {v:.0%}" for d, v in r["top3_final"])
    ax.set_title(f"Pista {pista}: los pesos de O se mueven con cada crisis que termina ({len(suelos)} cambios de K); "
                 f"al final {top}", fontsize=11.5, loc="left", pad=14)
    bx.step(S.index, S["n_efectivo"], where="post", color="#111", lw=1.3)
    bx.axhline(len(cols), color="#8f8f8f", ls="--", lw=1)
    bx.axhline(1, color="#8f8f8f", ls=":", lw=1)
    bx.text(1.005, len(cols), f"{len(cols)} = pesos iguales", va="center", fontsize=8.5, color="#555",
            transform=bx.get_yaxis_transform())
    bx.text(1.005, 1, "1 = un solo líder", va="center", fontsize=8.5, color="#555", transform=bx.get_yaxis_transform())
    bx.set_ylim(0, len(cols) + 1)
    bx.set_ylabel("n efectivo\nexp(entropía)")
    bx.set_title(f"Número efectivo de detectores: medio {r['n_efectivo_medio']:.1f}, mínimo "
                 f"{r['n_efectivo_min']:.1f}; el líder pesa más del 50 % en el "
                 f"{r['frac_meses_lider_mayoria']:.0%} de los meses aprendidos", fontsize=10, loc="left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figura 3 · p̄ y señal frente a las crisis
# --------------------------------------------------------------------------- #
def senales_comparadores(pista: str) -> pd.DataFrame:
    """Señales de O, M, V, F* y del mejor individual (columna ``I*=<id>``) sobre el OOS."""
    crit = on.leer_criterio_pista(pista)
    sen = on.leer_senales(pista)
    comp = [c for c in ("O", "M", "V", "F*") if c in sen]
    out = sen[comp].astype(bool).copy()
    out[f"I*={crit['I*']}"] = vara.senales(pista)[crit["I*"]].astype(bool)
    return out


def tabla_crisis(pista: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(detectada 1/0, cobertura) comparador × crisis con días OOS (racha mínima de la vara)."""
    from regimenes.evaluacion import metricas as mt
    from regimenes.evaluacion import ranking as rk

    sen = senales_comparadores(pista)
    win = vara.ventanas(pista)
    det, cov = {}, {}
    for c in sen:
        t = mt.event_detection_table(sen[c], win, min_run=int(rk.DETECTION_DEFAULTS["min_run"]))
        t = t[t["dias_oos"] > 0]
        det[c] = t["detectada"]
        cov[c] = t["dias_marcados"] / t["dias_oos"]
    return pd.DataFrame(det).T, pd.DataFrame(cov).T


def figura_cronograma(pista: str):
    """p̄ de O y de M (con la p_crisis del mejor individual) y las señales de los comparadores
    sobre las crisis del catálogo."""
    plt = _plt()
    crit = on.leer_criterio_pista(pista)
    pbar = on.leer_pbar(pista)
    istar = crit["I*"]
    p_i = vara.probabilidades(pista)[istar]
    sen = senales_comparadores(pista)
    det, _ = tabla_crisis(pista)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(15, 6.4), sharex=True, gridspec_kw={"height_ratios": [2, 1.5]})
    _sombrear(ax, pista, etiquetas=True)
    ax.plot(p_i.index, p_i.to_numpy(), color=pn.color_de(istar), lw=0.6, alpha=0.55, label=f"p_crisis de I* = {istar}")
    ax.plot(pbar.index, pbar[ID_M].to_numpy(), color=COLOR_COMP["M"], lw=0.8, label="p̄ de M (pesos iguales)")
    ax.plot(pbar.index, pbar[ID_O].to_numpy(), color=COLOR_COMP["O"], lw=0.9, label="p̄ de O (η = 10)")
    ax.axhline(on.UMBRAL, color="#D55E00", ls="--", lw=1, label="umbral 0,50")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("probabilidad de crisis")
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1), fontsize=8.5, frameon=False)
    ax.margins(x=0)
    filas = list(sen.columns)
    _sombrear(bx, pista)
    for k, c in enumerate(filas):
        clave = "I*" if c.startswith("I*") else c
        col = color_comparador(clave, istar)
        if clave == "V":
            col = "#9a9a9a"
        s = sen[c].to_numpy()
        bx.fill_between(sen.index, k - 0.36, k + 0.36, where=s, color=col, lw=0, step="post")
    n_det = det.sum(axis=1).astype(int)
    n_ev = det.notna().sum(axis=1).astype(int)
    bx.set_yticks(range(len(filas)), [f"{c}  ({n_det[c]}/{n_ev[c]})" for c in filas], fontsize=9)
    bx.set_ylim(len(filas) - 0.5, -0.5)
    bx.grid(False)
    bx.set_title("Señal binaria (crisis = barra); entre paréntesis, crisis detectadas", fontsize=10, loc="left")
    o_no = [c for c in det.columns if det.loc["O", c] == 0 and det.loc[f"I*={istar}", c] == 1]
    extra = f"; O pierde frente a {istar}: {', '.join(pn._corto(c) for c in o_no)}" if o_no else ""
    ax.set_title(f"Pista {pista}: O detecta {n_det['O']}/{n_ev['O']} crisis e I* = {istar} {n_det[f'I*={istar}']}/"
                 f"{n_ev[f'I*={istar}']}{extra}", fontsize=11.5, loc="left")
    fig.tight_layout()
    return fig


def figura_crisis():
    """Comparador × crisis, por pista: color = cobertura de la ventana, ✓ = detectada."""
    plt = _plt()
    mats = {p: tabla_crisis(p) for p in PISTAS}
    anchos = [mats[p][0].shape[1] for p in PISTAS]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": anchos})
    for ax, p in zip(axes, PISTAS):
        det, cov = mats[p]
        im = ax.imshow(cov.to_numpy(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(det.shape[0]):
            for j in range(det.shape[1]):
                if det.iat[i, j] == 1:
                    ax.text(j, i, "✓", ha="center", va="center", fontsize=9,
                            color="white" if cov.iat[i, j] > 0.55 else "#08306b")
        ax.set_xticks(range(det.shape[1]), [pn._corto(c) for c in det.columns], rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(det.shape[0]), [f"{c} ({int(det.loc[c].sum())}/{int(det.loc[c].notna().sum())})"
                                            for c in det.index], fontsize=9)
        ax.grid(False)
        ax.set_title(f"Pista {p}", fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="cobertura: fracción de la crisis marcada")
    fig.suptitle("Qué crisis ve cada comparador (✓ = detectada: ≥ 3 sesiones seguidas en crisis dentro de [pico, suelo])",
                 fontsize=12, weight="bold")
    return fig


# --------------------------------------------------------------------------- #
# Figura 4 · MCS-O
# --------------------------------------------------------------------------- #
def figura_mcs(tramo: str = "oos"):
    """Pérdida media de cada modelo del MCS-O (orden de menor a mayor); hueco = fuera del MCS."""
    plt = _plt()
    tablas = {p: on.leer_mcs(p, tramo).sort_values("perdida_media") for p in PISTAS}
    alto = max(len(t) for t in tablas.values()) * 0.27 + 1.6
    fig, axes = plt.subplots(1, 2, figsize=(15, alto))
    for ax, p in zip(axes, PISTAS):
        t = tablas[p]
        crit = on.leer_criterio_pista(p)
        for k, (m, r) in enumerate(t.iterrows()):
            col = COLOR_COMP.get(m) if m in COLOR_COMP else pn.color_de(m)
            dentro = bool(r["incluido"])
            ax.barh(k, r["perdida_media"], color=col if dentro else "white", edgecolor=col if m != "V" else "#888",
                    lw=1.4, height=0.72)
            ax.text(r["perdida_media"] + 0.006, k, f"p = {r['p_mcs']:.2f}", va="center", fontsize=7.5, color="#444")
        ax.set_yticks(range(len(t)), [f"{m} (I*)" if m == crit["I*"] else m for m in t.index], fontsize=8.5)
        for lab, m in zip(ax.get_yticklabels(), t.index):
            if m in COLOR_COMP:
                lab.set_weight("bold")
        ax.set_ylim(len(t) - 0.4, -0.6)
        ax.set_xlim(0.4, 1.15)
        ax.set_xlabel("pérdida media 1 − J (menor es mejor)")
        n_in = int(t["incluido"].sum())
        ax.set_title(f"Pista {p}: {n_in} de {len(t)} dentro; O {'dentro' if t.loc['O', 'incluido'] else 'fuera'} "
                     f"(p = {t.loc['O', 'p_mcs']:.2f})", fontsize=11)
    fig.suptitle(f"MCS-O al 10 % ({NOMBRE_TRAMO[tramo]}): barra llena = dentro, hueca = fuera; "
                 "en negrita los comparadores", fontsize=12, weight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figura 5 · η
# --------------------------------------------------------------------------- #
def figura_eta():
    """Score frente a η (rejilla del §4.4) en todo el OOS, H1 y H2; hueco = nivel ≠ 0."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, p in zip(axes, PISTAS):
        t = on.leer_eta(p)
        crit = on.leer_criterio_pista(p)
        x = np.arange(len(t))
        for tr in ("oos", "H1", "H2"):
            y = t[f"score_deteccion_{tr}"].to_numpy(float)
            ok = t[f"nivel_{tr}"].to_numpy() == 0
            ax.plot(x, y, color=COLOR_TRAMO[tr], lw=1.4, label=NOMBRE_TRAMO[tr])
            ax.scatter(x[ok], y[ok], color=COLOR_TRAMO[tr], s=36, zorder=3)
            ax.scatter(x[~ok], y[~ok], facecolor="white", edgecolor=COLOR_TRAMO[tr], s=36, zorder=3)
        k10 = int(np.flatnonzero(t["eta"].to_numpy() == on.ETA_PRINCIPAL)[0])
        kh1 = int(np.flatnonzero(t["eta"].to_numpy() == crit["eta_H1"])[0])
        ax.axvspan(k10 - 0.3, k10 + 0.3, color="#D55E00", alpha=0.10, lw=0)
        ax.text(k10, 0.995, "η pre-registrado", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8.5, color="#D55E00")
        ax.scatter([kh1], [t.loc[kh1, "score_deteccion_H1"]], marker="*", s=260, color=COLOR_TRAMO["H1"],
                   edgecolor="#333", zorder=4, label="η que gana en H1")
        ax.axhline(crit["score_I*"], color=COLOR_TRAMO["oos"], ls="--", lw=0.9, label=f"I* = {crit['I*']} (OOS)")
        ax.axhline(crit["score_I*_H1_en_H2"], color=COLOR_TRAMO["H2"], ls=":", lw=1.1,
                   label=f"I*_H1 = {crit['I*_H1']} en H2")
        ax.set_xticks(x, [_eta_txt(e) for e in t["eta"]])
        ax.set_xlabel("η (0 = media simple, ∞ = seguir al líder)")
        ax.set_title(f"Pista {p}: en H1 gana η = {_eta_txt(crit['eta_H1'])}; en H2 da "
                     f"{crit['score_eta_H1_en_H2']:.3f} frente a {crit['score_O_H2']:.3f} de η = 10", fontsize=10.5)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    axes[0].set_ylabel("score de detección")
    fig.suptitle("Tasa de aprendizaje (sin peso en el veredicto): punto hueco = nivel ≠ 0 en ese tramo",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figura 6 · sensibilidades
# --------------------------------------------------------------------------- #
def figura_sensibilidades():
    """Score de la principal y de las sensibilidades del §6.3 en todo el OOS, H1 y H2."""
    plt = _plt()
    t = tabla_sensibilidades()
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.9), sharey=True)
    marc = {"oos": "o", "H1": "s", "H2": "D"}
    for ax, p in zip(axes, PISTAS):
        s = t.loc[p]
        crit = on.leer_criterio_pista(p)
        y = np.arange(len(s))
        for tr in ("oos", "H1", "H2"):
            ax.scatter(s[tr], y, marker=marc[tr], color=COLOR_TRAMO[tr], s=46, label=NOMBRE_TRAMO[tr], zorder=3)
        for k, (_, r) in enumerate(s.iterrows()):
            ax.plot([r[["oos", "H1", "H2"]].min(), r[["oos", "H1", "H2"]].max()], [k, k], color="#ccc", lw=1, zorder=1)
        ax.axvline(crit["score_I*"], color=COLOR_TRAMO["oos"], ls="--", lw=0.9)
        ax.axvline(crit["score_I*_H1_en_H2"], color=COLOR_TRAMO["H2"], ls=":", lw=1.1)
        ax.set_yticks(y, s.index, fontsize=9)
        ax.set_ylim(len(s) - 0.5, -0.5)
        ax.set_xlabel("score de detección")
        ax.set_title(f"Pista {p} (-- I* en OOS, ··· I*_H1 en H2)", fontsize=11)
    axes[1].legend(fontsize=8.5, loc="lower right", frameon=False)
    fig.suptitle("Sensibilidades del §6.3 (sin peso): la fila de arriba es O", fontsize=12, weight="bold")
    fig.tight_layout()
    return fig


def figuras_disponibles() -> Sequence[str]:
    return ("figura_resumen", "figura_mecanismo", "figura_pesos", "figura_cronograma", "figura_crisis",
            "figura_mcs", "figura_eta", "figura_sensibilidades")
