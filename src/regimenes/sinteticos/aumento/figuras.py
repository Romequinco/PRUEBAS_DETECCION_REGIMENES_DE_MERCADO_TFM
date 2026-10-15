"""Figuras del notebook 18 (aumento real + sintetico).

Cada funcion recibe tablas ya calculadas por ``analisis`` (o el registro de
anclas) y devuelve una ``Figure`` de matplotlib; no lee resultados por su cuenta
salvo el indice de la pista en :func:`fig_diseno`. Con tablas vacias (resultados
parciales) dibujan el panel con el texto «sin datos» en lugar de fallar.
Estilo de casa: ``regimenes.viz``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from regimenes.sinteticos.aumento.analisis import CATEGORIAS_ANCLA, categoria_ancla
from regimenes.sinteticos.aumento.config import config_aumento
from regimenes.viz import C_CRISIS, C_LONG, C_NA, C_NEG, C_SHORT, SHORT

C_PISTA = {"A": C_LONG, "B": C_SHORT}
C_MEJORA = "#3a7d44"
C_VEREDICTO = {"mejora": C_MEJORA, "empeora": C_CRISIS, "sin efecto detectable": C_NEG, "falta": C_NA}
C_GRUPO = {"aprende_estado": C_LONG, "umbral": C_SHORT, "mixto": C_NEG, "secuenciales": C_CRISIS}
C_CATEGORIA = dict(zip(CATEGORIAS_ANCLA, [C_LONG, C_NA, C_SHORT, C_CRISIS, "#7a1f12", "#222222"]))
ETIQUETA_CATEGORIA = dict(zip(CATEGORIAS_ANCLA, ["válida", "pocas crisis", "no pasa memorización",
                                                  "no pasa condicionamiento", "no pasa ambas", "error"]))


def etiqueta(det: str) -> str:
    """'D04' -> 'D4 hmm-g' (nombres cortos de ``regimenes.viz``)."""
    try:
        n = int(str(det)[1:])
    except ValueError:
        return str(det)
    return next((v for v in SHORT.values() if v.split()[0] == f"D{n}"), str(det))


def _sin_datos(ax, texto: str = "sin datos todavía") -> None:
    ax.text(0.5, 0.5, texto, ha="center", va="center", transform=ax.transAxes, color=C_NEG)
    ax.set_xticks([])
    ax.set_yticks([])


def _forest(ax, tabla: pd.DataFrame, col: str, lo: str, hi: str, filas: list[str], color=None) -> None:
    """Punto + IC por fila (filas en orden de arriba abajo); color por signo del IC si no se da."""
    y = {f: i for i, f in enumerate(filas)}
    for _, r in tabla.iterrows():
        if r["_fila"] not in y or not np.isfinite(r.get(col, np.nan)):
            continue
        c = color(r) if callable(color) else (
            C_LONG if r[lo] > 0 else C_CRISIS if r[hi] < 0 else C_NEG)
        ax.plot([r[lo], r[hi]], [y[r["_fila"]]] * 2, color=c, lw=1.6)
        ax.plot(r[col], y[r["_fila"]], "o", color=c, ms=5)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(filas)), [etiqueta(f) for f in filas])
    ax.set_ylim(len(filas) - 0.5, -0.5)


# --------------------------------------------------------------------------- 2. diseno
def fig_diseno(registros: dict[str, pd.DataFrame], cfg: dict | None = None, pista: str = "B",
               fin_train: str = "2020-06-30", fuente: str | None = None, step: int = 21):
    """Linea temporal de un pliegue (bloques sinteticos delante, train real, OOS, anclas) y mapa de anclas.

    Arriba, un pliegue real de ``pista`` (train expansivo que acaba en ``fin_train``):
    bloques del brazo completas con las proporciones anidadas y del brazo crisis,
    crisis del train (reconocidas o no en el ancla), candidatas anuales de ``fuente``
    y el margen de reconocimiento. Debajo, por pista, la categoria de cada ancla
    (fuente x ano) del registro.
    """
    from regimenes.sinteticos.aumento import anclas as an
    from regimenes.sinteticos.aumento import muestras as mu
    from regimenes.sinteticos.aumento.ejecucion import _indice_pista
    from regimenes.sinteticos.aumento.envoltorio import fechas_previas
    from regimenes.sinteticos.datos import ventanas_crisis

    cfg = config_aumento() if cfg is None else cfg
    fuente = fuente or cfg["generadores"][0]
    margen = int(cfg["anclas"]["margen_reconocimiento"])
    pistas = [p for p in cfg["pistas"] if p in registros]
    alturas = [2.6] + [0.9 + 0.12 * registros[p]["fuente"].nunique() for p in pistas]
    fig = plt.figure(figsize=(12, sum(alturas) + 1.2))
    gs = fig.add_gridspec(len(alturas), 1, height_ratios=alturas, hspace=0.55)

    # -- linea temporal de un pliegue
    ax = fig.add_subplot(gs[0])
    idx = _indice_pista(pista)
    t = idx[idx <= pd.Timestamp(fin_train)][-1]
    train = idx[: idx.get_loc(t) + 1]
    oos = idx[idx.get_loc(t) + 1: idx.get_loc(t) + 1 + step]
    limite = an.limite_reconocimiento(train, margen)
    reg = registros.get(pista)
    validas = an.anclas_validas(reg, fuente) if reg is not None else pd.DatetimeIndex([])
    tau = an.ancla_para(validas, limite)
    d = pd.Timedelta(days=1)

    def barra(y, ini, fin, **kw):
        ax.broken_barh([(ini, fin - ini + d)], (y - 0.32, 0.64), **kw)

    ps = sorted(float(p) for p in cfg["proporciones"])
    for k, p in enumerate(ps):
        f = fechas_previas(train[0], mu.n_sintetico(p, len(train)))
        ax.plot([f[0], f[-1]], [4.45 + 0.18 * k] * 2, color=C_SHORT, lw=2.2, solid_capstyle="butt")
        ax.text(f[0], 4.45 + 0.18 * k, f"p = {p:g} ", ha="right", va="center", fontsize=8, color=C_SHORT)
    barra(4, f[0], f[-1], color=C_SHORT, alpha=0.6, lw=0)          # bloque de la mayor proporcion
    if tau is not None:
        n_cri = mu.n_sintetico(1.0, mu.base_proporcion("crisis", train, pista, tau))
        if n_cri:
            f = fechas_previas(train[0], n_cri)
            barra(3, f[0], f[-1], color=C_CRISIS, alpha=0.5, lw=0)
            ax.text(f[0], 3, "brazo crisis (p = 1) ", ha="right", va="center", fontsize=8, color=C_CRISIS)
    barra(4, train[0], train[-1], color=C_LONG, alpha=0.85, lw=0)
    ancho_oos = max(oos[-1] - oos[0], pd.Timedelta(days=150)) if len(oos) else pd.Timedelta(days=150)
    ax.broken_barh([(oos[0] if len(oos) else t + d, ancho_oos)], (3.68, 0.64), color="k", alpha=0.8, lw=0)
    ax.text(train[0] + (train[-1] - train[0]) / 2, 4, f"train real ({len(train)} sesiones)", ha="center",
            va="center", color="w", fontsize=9)
    ax.text(train[0] - d * 30, 4, "bloque sintético (fechas ficticias) ", ha="right", va="center", fontsize=8)
    ax.annotate(f"OOS: {step} sesiones\n(ampliado)", (t + ancho_oos, 4), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=8)
    # crisis del train: reconocidas en tau (suelo <= tau) o no
    for nombre, (pico, suelo) in ventanas_crisis(pista).items():
        if pico > t or suelo < train[0]:
            continue
        reconocida = tau is not None and suelo <= tau
        barra(2, max(pico, train[0]), min(suelo, t), color=C_CRISIS, alpha=0.85 if reconocida else 0.2,
              hatch=None if reconocida else "////", lw=0)
    # anclas candidatas de la fuente
    if reg is not None:
        r = reg[(reg["fuente"] == fuente) & (pd.to_datetime(reg["tau"]) <= t)]
        v = r["valida"].astype(bool)
        ax.plot(pd.to_datetime(r.loc[v, "tau"]), [1] * int(v.sum()), "o", color=C_LONG, ms=6)
        ax.plot(pd.to_datetime(r.loc[~v, "tau"]), [1] * int((~v).sum()), "o", mfc="w", mec=C_NEG, ms=6)
    if limite is not None:
        ax.axvspan(limite, t, ymin=0.0, ymax=0.62, color=C_NA, alpha=0.9, zorder=0)
        ax.text(limite, 2.55, f"margen de {margen} sesiones ", ha="right", va="center", fontsize=8)
    if tau is not None:
        ax.plot(tau, 1, "*", color=C_CRISIS, ms=15, zorder=5)
        ax.annotate(f"ancla τ = {tau.date()}\n(ajuste con datos ≤ τ)", (tau, 1), xytext=(-12, -20),
                    textcoords="offset points", ha="right", fontsize=8)
    ax.axvline(t, color="k", lw=0.8, ls=":")
    ax.set_yticks([4.6, 4, 3, 2, 1], ["proporciones\n(anidadas)", "pliegue", "brazo crisis",
                                       "crisis en el train", f"anclas\n({fuente})"], fontsize=8)
    ax.set_ylim(-0.2, 5.2)
    ax.grid(False)
    ax.set_title(f"Pista {pista}: pliegue con train hasta {t.date()} (brazo completas, p = 1). Crisis lisas = "
                 "reconocidas en τ; rayadas = aún no (calma para el generador)", fontsize=10)
    leyenda = [Patch(color=C_LONG, label="ancla válida"), Patch(facecolor="w", edgecolor=C_NEG, label="no válida"),
               Patch(color=C_NA, label="margen de reconocimiento")]
    ax.legend(handles=leyenda, loc="upper left", fontsize=7, ncol=3, frameon=False, bbox_to_anchor=(0, -0.1))

    # -- mapa de anclas por pista
    cmap = plt.matplotlib.colors.ListedColormap([C_CATEGORIA[c] for c in CATEGORIAS_ANCLA])
    for k, p in enumerate(pistas):
        axm = fig.add_subplot(gs[k + 1])
        reg = registros[p].assign(categoria=categoria_ancla(registros[p]),
                                  ano=pd.to_datetime(registros[p]["tau"]).dt.year)
        m = (reg.pivot(index="fuente", columns="ano", values="categoria")
             .reindex([f for f in [*cfg["generadores"], cfg["control"]] if f in set(reg["fuente"])]))
        codigos = m.apply(lambda s: s.map({c: i for i, c in enumerate(CATEGORIAS_ANCLA)})).to_numpy(float)
        axm.imshow(np.ma.masked_invalid(codigos), cmap=cmap, vmin=-0.5, vmax=len(CATEGORIAS_ANCLA) - 0.5,
                   aspect="auto", interpolation="none")
        axm.set_yticks(range(len(m)), m.index, fontsize=8)
        paso = 5 if m.shape[1] > 30 else 2
        cols = list(m.columns)
        axm.set_xticks([i for i, a in enumerate(cols) if a % paso == 0], [a for a in cols if a % paso == 0],
                       fontsize=8)
        axm.grid(False)
        validas_n = (reg["categoria"] == "valida").sum()
        axm.set_title(f"Pista {p}: anclas por fuente y año ({validas_n} de {len(reg)} válidas)", fontsize=10)
    if pistas:
        fig.legend(handles=[Patch(color=C_CATEGORIA[c], label=ETIQUETA_CATEGORIA[c]) for c in CATEGORIAS_ANCLA], loc="lower center",
                   ncol=len(CATEGORIAS_ANCLA), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    return fig


# --------------------------------------------------------------------------- 3. H1
def fig_h1(t1: pd.DataFrame, cfg: dict | None = None):
    """Forest plot de H1: Delta(voto) con IC 95 % por detector y pista, veredicto y suelo de ruido."""
    cfg = config_aumento() if cfg is None else cfg
    minima = float(cfg["contrastes"]["mejora_minima"])
    pistas = [p for p in cfg["pistas"] if not t1.empty and p in set(t1["pista"])]
    n = max([int((t1["pista"] == p).sum()) for p in pistas] or [4])
    fig, axs = plt.subplots(1, max(len(pistas), 1), figsize=(6 * max(len(pistas), 1), 0.38 * n + 1.9),
                            squeeze=False)
    if not pistas:
        _sin_datos(axs[0, 0])
        return fig
    for ax, p in zip(axs[0], pistas):
        g = t1[t1["pista"] == p].reset_index(drop=True)
        for i, r in g.iterrows():
            s = r.get("suelo_ruido", np.nan)
            if pd.notna(s):
                ax.plot([-s, s], [i, i], color=C_NA, lw=7, solid_capstyle="butt", zorder=0)
            if r["veredicto"] == "falta" or pd.isna(r.get("delta", np.nan)):
                ax.text(0, i, "falta", ha="center", va="center", fontsize=8, color=C_NEG)
                continue
            c = C_VEREDICTO[r["veredicto"]]
            ax.plot([r["ic_inf"], r["ic_sup"]], [i, i], color=c, lw=1.8)
            ax.plot(r["delta"], i, "o", color=c, mfc=c if r["holm"] else "w", ms=6, zorder=3)
            ax.annotate(f"p={r['p']:.3f}", (1.0, i), xycoords=("axes fraction", "data"), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=7, color=C_NEG)
        ax.axvline(0, color="k", lw=0.8)
        ax.axvline(minima, color=C_MEJORA, lw=0.9, ls="--")
        ax.set_yticks(range(len(g)), [etiqueta(x) for x in g["detector"]])
        ax.set_ylim(len(g) - 0.5, -0.5)
        ax.set_xlabel("Δ del exceso sobre el nulo (voto, completas, p = 1)")
        ax.set_title(f"Pista {p}")
    leyenda = [Patch(color=c, label=v) for v, c in C_VEREDICTO.items()]
    leyenda += [Patch(color=C_NA, label="suelo de ruido (±)"),
                plt.Line2D([], [], color=C_MEJORA, ls="--", label=f"mejora mínima ({minima:g})"),
                plt.Line2D([], [], marker="o", color=C_NEG, mfc="w", ls="", label="no significativo (Holm)"),
                plt.Line2D([], [], marker="o", color=C_NEG, ls="", label="significativo (Holm)")]
    fig.legend(handles=leyenda, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- 4. mapa
def fig_mapa(mapa: pd.DataFrame, cfg: dict | None = None):
    """Mapa detector x fuente de Delta (exploratorio); el voto separado por lineas blancas."""
    cfg = config_aumento() if cfg is None else cfg
    orden = [*cfg["generadores"], "voto", cfg["control"]]
    pistas = [p for p in cfg["pistas"] if not mapa.empty and p in set(mapa["pista"])]
    fig, axs = plt.subplots(1, max(len(pistas), 1), figsize=(6.2 * max(len(pistas), 1), 5.4), squeeze=False)
    if not pistas or mapa["delta"].isna().all():
        _sin_datos(axs[0, 0])
        return fig
    vmax = float(np.nanmax(np.abs(mapa["delta"]))) or 1.0
    for ax, p in zip(axs[0], pistas):
        m = mapa[mapa["pista"] == p].pivot(index="detector", columns="fuente", values="delta").reindex(columns=orden)
        im = ax.imshow(m.to_numpy(float), cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        for (i, j), v in np.ndenumerate(m.to_numpy(float)):
            ax.text(j, i, "." if np.isnan(v) else f"{v:+.2f}", ha="center", va="center", fontsize=7,
                    color="w" if abs(v) > 0.6 * vmax else "k")
        k = orden.index("voto")
        for x in (k - 0.5, k + 0.5):
            ax.axvline(x, color="w", lw=3)
        ax.set_xticks(range(len(orden)), [o.replace("_regimen", "") for o in orden], rotation=35, ha="right",
                      fontsize=8)
        ax.set_yticks(range(len(m)), [etiqueta(x) for x in m.index], fontsize=8)
        ax.grid(False)
        ax.set_title(f"Pista {p}")
    fig.colorbar(im, ax=axs[0].tolist(), shrink=0.8, label="Δ del exceso sobre el nulo")
    return fig


# --------------------------------------------------------------------------- 5. H2
def fig_dosis(dosis: pd.DataFrame, cfg: dict | None = None):
    """Delta(voto) frente a la proporcion: detectores (finas) y media por grupo de H4 con IC (gruesas)."""
    cfg = config_aumento() if cfg is None else cfg
    pistas = [p for p in cfg["pistas"] if not dosis.empty and p in set(dosis["pista"])]
    brazos = ["completas", "crisis"]
    fig, axs = plt.subplots(max(len(pistas), 1), 2, figsize=(11, 3.4 * max(len(pistas), 1)), squeeze=False,
                            sharex=True)
    if not pistas:
        _sin_datos(axs[0, 0])
        _sin_datos(axs[0, 1])
        return fig
    ps = [0.0, *sorted(float(p) for p in cfg["proporciones"])]
    for i, p in enumerate(pistas):
        for j, brazo in enumerate(brazos):
            ax = axs[i, j]
            g = dosis[(dosis["pista"] == p) & (dosis["brazo"] == brazo)]
            if g.empty:
                _sin_datos(ax)
                continue
            for (grupo, det), h in g.groupby(["grupo", "detector"]):
                h = h.sort_values("p")
                c = C_GRUPO.get(grupo, C_NEG)
                if det == "(media)":
                    ax.fill_between(h["p"], h["ic_inf"], h["ic_sup"], color=c, alpha=0.15, lw=0)
                    ax.plot(h["p"], h["delta"], "o-", color=c, lw=2.4,
                            label=f"{grupo} (n = {int(h['n_detectores'].iloc[-1])})")
                else:
                    ax.plot(h["p"], h["delta"], "-", color=c, lw=0.8, alpha=0.45)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(ps, [f"{x:g}" for x in ps])
            ax.set_title(f"Pista {p}, brazo {brazo}", fontsize=10)
            if j == 0:
                ax.set_ylabel("Δ (voto)")
            if i == len(pistas) - 1:
                ax.set_xlabel("proporción p")
            ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- 6. H3 y H5 (H4 en el titulo)
def fig_contrastes(t3: pd.DataFrame, t5: pd.DataFrame, t4: pd.DataFrame | None = None, cfg: dict | None = None):
    """H3 (voto y media de generadores frente al control) y H5 (crisis - completas) por detector; H4 en el titulo."""
    cfg = config_aumento() if cfg is None else cfg
    paneles = [("H3: voto − control", t3, "voto_menos_control", "ic_inf", "ic_sup"),
               ("H3: media de generadores − control", t3, "media_gen_menos_control", "media_ic_inf", "media_ic_sup"),
               ("H5: crisis − completas", t5, "crisis_menos_completas", "ic_inf", "ic_sup")]
    pistas = [p for p in cfg["pistas"] if any(not t.empty and p in set(t["pista"]) for t in (t3, t5))]
    n = max([len(set(pd.concat([t3, t5])["detector"][pd.concat([t3, t5])["pista"] == p])) for p in pistas] or [4])
    fig, axs = plt.subplots(max(len(pistas), 1), 3, figsize=(13, (0.3 * n + 1.4) * max(len(pistas), 1)),
                            squeeze=False, sharex=False)
    if not pistas:
        for ax in axs[0]:
            _sin_datos(ax)
        return fig
    for i, p in enumerate(pistas):
        dets = sorted(set(pd.concat([t3, t5]).query("pista == @p")["detector"]))
        for j, (titulo, t, col, lo, hi) in enumerate(paneles):
            ax = axs[i, j]
            if t.empty or col not in t or not (t["pista"] == p).any():
                _sin_datos(ax)
                ax.set_title(f"{titulo} ({p})", fontsize=9)
                continue
            _forest(ax, t[t["pista"] == p].assign(_fila=lambda x: x["detector"]), col, lo, hi, dets)
            ax.set_title(f"{titulo} ({p})", fontsize=9)
            if j:
                ax.set_yticklabels([])
        if t4 is not None and not t4.empty and (t4["pista"] == p).any():
            r = t4[t4["pista"] == p].iloc[0]
            axs[i, 0].annotate(f"H4 ({p}): aprende estado − umbral = {r['diferencia']:+.3f} "
                               f"[{r['ic_inf']:+.3f}, {r['ic_sup']:+.3f}]", (0, 1.16), xycoords="axes fraction",
                               fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- 7. diagnosticos
def fig_diagnosticos(dm: pd.DataFrame, ocup: pd.DataFrame, cfg: dict | None = None):
    """(a) recalibracion frente a aprendizaje, (b) score crudo frente a exceso, (c) estados solo sinteticos."""
    cfg = config_aumento() if cfg is None else cfg
    p1 = float(cfg["contrastes"]["h1"]["proporcion"])
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    voto = dm[(dm["tipo"] == "voto") & (dm["rep"].fillna(0) == 0)] if not dm.empty else dm
    h1v = voto[(voto["brazo"] == "completas") & np.isclose(voto["proporcion"].astype(float), p1)] \
        if not voto.empty else voto
    ax = axs[0]
    if h1v.empty:
        _sin_datos(ax)
    else:
        fu = dm[(dm["tipo"] == "fuente") & (dm["brazo"] == "completas")
                & np.isclose(dm["proporcion"].astype(float), p1) & (dm["rep"].fillna(0) == 0)]
        ax.scatter(fu["d_tasa"], fu["d_recall"], s=10, color=C_NA, label="cada generador")
        for p, g in h1v.groupby("pista"):
            ax.scatter(g["d_tasa"], g["d_recall"], s=30, color=C_PISTA.get(p, C_NEG), label=f"voto, pista {p}")
            for _, r in g.iterrows():
                ax.annotate(etiqueta(r["detector"]).split()[0], (r["d_tasa"], r["d_recall"]), fontsize=7,
                            xytext=(3, 2), textcoords="offset points")
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("Δ tasa de marcado")
        ax.set_ylabel("Δ recall por evento")
        ax.legend(fontsize=7, frameon=False)
    ax.set_title("(a) recall frente a cuánto se marca\n(completas, p = 1)", fontsize=10)
    ax = axs[1]
    if voto.empty:
        _sin_datos(ax)
    else:
        for p, g in voto.groupby("pista"):
            ax.scatter(g["d_score"], g["d_exceso"], s=14, alpha=0.7, color=C_PISTA.get(p, C_NEG), label=f"pista {p}")
        lim = float(np.nanmax(np.abs(voto[["d_score", "d_exceso"]].to_numpy()))) * 1.1 or 0.1
        ax.plot([-lim, lim], [-lim, lim], color=C_NEG, lw=0.8, ls="--", label="exceso = crudo")
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("Δ score crudo")
        ax.set_ylabel("Δ exceso sobre el nulo")
        ax.legend(fontsize=7, frameon=False)
    ax.set_title("(b) crudo frente a exceso\n(voto, todos los brazos y proporciones)", fontsize=10)
    ax = axs[2]
    o = ocup[(ocup["brazo"] == "completas") & np.isclose(ocup["proporcion"].astype(float), p1)] \
        if not ocup.empty else ocup
    if o.empty:
        _sin_datos(ax)
    else:
        m = o.groupby(["detector", "pista"])["frac_solo_sint"].mean().unstack("pista")
        y = np.arange(len(m))
        alto = 0.8 / max(m.shape[1], 1)
        for k, p in enumerate(m.columns):
            ax.barh(y + k * alto, m[p], height=alto, color=C_PISTA.get(p, C_NEG), label=f"pista {p}")
        ax.set_yticks(y + alto * (m.shape[1] - 1) / 2, [etiqueta(x) for x in m.index], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("fracción de pliegues con un estado solo sintético")
        ax.legend(fontsize=7, frameon=False)
    ax.set_title("(c) estados que solo ocupa el sintético\n(media de fuentes, completas, p = 1)", fontsize=10)
    fig.tight_layout()
    return fig


def fig_robustez(t1: pd.DataFrame, post: pd.DataFrame, suelo: pd.DataFrame, rep: pd.DataFrame):
    """(a) H1 entero frente a solo crisis posteriores al ajuste del 15, (b) replica de B, (c) suelo de ruido."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax = axs[0]
    t = t1[t1["veredicto"] != "falta"] if not t1.empty else t1
    if t.empty or post.empty:
        _sin_datos(ax)
    else:
        a = t.assign(_fila=t["pista"] + " " + t["detector"])
        b = post.assign(_fila=post["pista"] + " " + post["detector"])
        filas = sorted(set(a["_fila"]) | set(b["_fila"]))
        y = {f: i for i, f in enumerate(filas)}
        for tab, dy, c, lab in ((a, -0.15, C_NEG, "todas las crisis"), (b, 0.15, C_LONG, "solo posteriores")):
            ys = [y[f] + dy for f in tab["_fila"]]
            ax.hlines(ys, tab["ic_inf"], tab["ic_sup"], color=c, lw=1.4)
            ax.plot(tab["delta"], ys, "o", color=c, ms=4, label=lab)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(filas)), [f.split()[0] + " " + etiqueta(f.split()[1]) for f in filas], fontsize=7)
        ax.set_ylim(len(filas) - 0.5, -0.5)
        ax.legend(fontsize=7, frameon=False)
        ax.set_xlabel("Δ H1")
    ax.set_title("(a) sensibilidad: crisis posteriores\nal ajuste del 15 + margen", fontsize=10)
    ax = axs[1]
    if rep.empty:
        _sin_datos(ax)
    else:
        for brazo, g in rep.groupby("brazo"):
            ax.scatter(g["delta_r0"], g["delta_r1"], s=18, alpha=0.8,
                       color=C_LONG if brazo == "completas" else C_CRISIS, label=f"brazo {brazo}")
        lim = float(np.nanmax(np.abs(rep[["delta_r0", "delta_r1"]].to_numpy()))) * 1.1 or 0.1
        ax.plot([-lim, lim], [-lim, lim], color=C_NEG, lw=0.8, ls="--")
        r = np.corrcoef(rep["delta_r0"], rep["delta_r1"])[0, 1] if len(rep) > 2 else np.nan
        ax.set_xlabel("Δ (repetición 0)")
        ax.set_ylabel("Δ (repetición 1)")
        ax.legend(fontsize=7, frameon=False, title=f"r = {r:.2f}, n = {len(rep)}", title_fontsize=7)
    ax.set_title("(b) réplica: muestras nuevas\n(voto, todas las proporciones)", fontsize=10)
    ax = axs[2]
    s = suelo.dropna(subset=["suelo"]) if not suelo.empty else suelo
    if s.empty:
        _sin_datos(ax)
    else:
        s = s.assign(_fila=s["pista"] + " " + s["detector"]).reset_index(drop=True)
        ax.barh(range(len(s)), s["suelo"], color=C_NA, label="suelo (máx. |Δ| entre semillas)")
        if not t.empty:
            d1 = t.assign(_fila=t["pista"] + " " + t["detector"]).set_index("_fila")["delta"].abs()
            ax.plot(s["_fila"].map(d1), range(len(s)), "o", color=C_LONG, label="|Δ H1|")
        ax.set_yticks(range(len(s)), [f.split()[0] + " " + etiqueta(f.split()[1]) for f in s["_fila"]], fontsize=8)
        ax.invert_yaxis()
        ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title("(c) suelo de ruido del optimizador (p = 0)", fontsize=10)
    fig.tight_layout()
    return fig
