"""
viz.py — Visualizaciones estándar reutilizables por todos los detectores.

Para que la comparación sea honesta, todos los detectores se visualizan IGUAL.
Estas funciones se llaman desde cada notebook de detector y desde la síntesis
comparativa (FASE 4), consolidando los helpers que antes se redefinían sueltos en
cada notebook (`shade_regime`, `episode_durations`, etc.).

Estilo de casa
--------------
Paleta sobria y consistente, heredada de la curaduría de figuras del informe
ejecutivo (`scripts/figs_pdf/build_pdf_figs.py`): azul para "ventana larga / calma",
ámbar para "ventana corta", gris para "exploratorio-negativo", rojo sobrio solo
para resaltar tramos de crisis. Llama a `use_house_style()` al inicio de un
notebook para fijar rcParams homogéneos (tamaños de fuente, grid tenue, DPI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Constantes de evaluación (fuente de verdad única de ventanas/troughs).
try:  # import robusto tanto si se usa como paquete (src.viz) como suelto
    from src.evaluation import CRISIS_WINDOWS, FALSE_POSITIVE_WINDOWS, DRAWDOWN_TROUGHS
except Exception:  # pragma: no cover
    from evaluation import CRISIS_WINDOWS, FALSE_POSITIVE_WINDOWS, DRAWDOWN_TROUGHS

# --------------------------------------------------------------------------- #
# Paleta y estilo de casa
# --------------------------------------------------------------------------- #
C_LONG = "#2b5d8a"      # ventana larga (vio 2008 OOS) / régimen calma  -> azul sobrio
C_SHORT = "#c98a2b"     # ventana corta (no vio 2008)                   -> ámbar sobrio
C_NEG = "#7a7a7a"       # exploratorio-negativo                         -> gris
C_NA = "#e8e8e8"        # no aplica / fuera de ventana
C_CRISIS = "#b5341f"    # resaltado de tramos de crisis                 -> rojo sobrio
C_FP = "#8a8a8a"        # ventanas trampa (falsos positivos)            -> gris medio

# Colores por estado canónico 0=calma .. n-1=crisis (gradiente azul->rojo sobrio).
REGIME_COLORS = ["#2b5d8a", "#5b8fb0", "#c98a2b", "#b5341f", "#7a1f12"]

HOUSE_RC = {
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def use_house_style() -> None:
    """Fija los rcParams de matplotlib al estilo de casa (llamar una vez por notebook)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(HOUSE_RC)


def regime_color(state: int, n_states: int) -> str:
    """Color canónico para un estado (interpola si n_states > paleta base)."""
    if n_states <= 1:
        return REGIME_COLORS[0]
    if state >= len(REGIME_COLORS):
        return C_CRISIS if state == n_states - 1 else C_SHORT
    # Mapear 0..n-1 a extremos calma(azul)->crisis(rojo) de la paleta base.
    idx = int(round(state / (n_states - 1) * (len(REGIME_COLORS) - 1)))
    return REGIME_COLORS[idx]


# --------------------------------------------------------------------------- #
# Helpers de bajo nivel (compatibles con los que se usaban sueltos en notebooks)
# --------------------------------------------------------------------------- #
def _as_series(states) -> pd.Series:
    return states if isinstance(states, pd.Series) else pd.Series(np.asarray(states))


def shade_regime(ax, states, crisis_state, color=C_CRISIS, alpha=0.22) -> None:
    """Sombrea en `ax` los tramos en que `states == crisis_state` (episodios de crisis).

    Reemplazo único del helper que antes se copiaba en ~6 notebooks. Firma
    compatible: shade_regime(ax, states, crisis_state, color, alpha).
    """
    s = _as_series(states)
    v = (s == crisis_state).astype(int).values
    idx = s.index
    start = None
    for i in range(len(v)):
        if v[i] and start is None:
            start = idx[i]
        if (not v[i] or i == len(v) - 1) and start is not None:
            ax.axvspan(start, idx[i], color=color, alpha=alpha, lw=0)
            start = None


def episode_durations(states, n_states: int | None = None) -> dict[int, list[int]]:
    """Duraciones (en días) de cada racha consecutiva, agrupadas por estado.

    Devuelve {estado: [longitudes_de_racha]}. Generaliza el helper previo a K
    estados (antes asumía {0,1}). Útil para histogramas de persistencia/flickering.
    """
    s = _as_series(states)
    v = s.values
    if n_states is None:
        n_states = int(np.nanmax(v)) + 1 if len(v) else 0
    out: dict[int, list[int]] = {k: [] for k in range(n_states)}
    if len(v) == 0:
        return out
    run = 1
    for i in range(1, len(v)):
        if v[i] == v[i - 1]:
            run += 1
        else:
            out.setdefault(int(v[i - 1]), []).append(run)
            run = 1
    out.setdefault(int(v[-1]), []).append(run)
    return out


def _shade_windows(ax, windows: dict, color: str, alpha: float, label: str | None) -> None:
    first = True
    for a, b in windows.values():
        ax.axvspan(
            pd.Timestamp(a), pd.Timestamp(b),
            color=color, alpha=alpha, lw=0,
            label=(label if first else None),
        )
        first = False


# --------------------------------------------------------------------------- #
# Figuras canónicas
# --------------------------------------------------------------------------- #
def plot_price_by_regime(
    prices, states, crisis_state, ax=None, *,
    logy: bool = True, mark_crisis_windows: bool = True,
    mark_fp_windows: bool = True, title: str | None = None,
):
    """S&P 500 (u otro precio) con los tramos de crisis del detector sombreados.

    Visualización canónica: precio en escala log con bandas rojas donde el
    detector marca crisis, y (opcional) las ventanas de crisis/trampa conocidas
    como referencia para inspeccionar cobertura y falsos positivos.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(15, 5))
    s = _as_series(states)
    px = (prices if isinstance(prices, pd.Series) else pd.Series(prices)).reindex(s.index)
    ax.plot(px.index, px.values, color="black", lw=0.7, zorder=3)
    if logy:
        ax.set_yscale("log")
    shade_regime(ax, s, crisis_state)
    if mark_crisis_windows:
        for a, b in CRISIS_WINDOWS.values():
            ax.axvline(pd.Timestamp(a), color=C_LONG, ls="--", lw=0.7, alpha=0.5)
    if mark_fp_windows:
        _shade_windows(ax, FALSE_POSITIVE_WINDOWS, C_FP, 0.12, "trampa (no-crisis)")
    ax.set_ylabel("precio (log)" if logy else "precio")
    ax.set_title(title or "Precio coloreado por régimen (rojo = crisis detectada, CAUSAL OOS)")
    ax.legend(loc="upper left", framealpha=0.9)
    return ax


def plot_regime_timeline(states, n_states: int | None = None, ax=None, *, title: str | None = None):
    """Timeline horizontal de regímenes (banda de color por día) a lo largo del tiempo."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if ax is None:
        _, ax = plt.subplots(figsize=(15, 1.4))
    s = _as_series(states)
    if n_states is None:
        n_states = int(np.nanmax(s.values)) + 1 if len(s) else 1
    cmap = ListedColormap([regime_color(k, n_states) for k in range(n_states)])
    arr = s.values.reshape(1, -1)
    ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=max(1, n_states - 1),
              extent=[0, len(s), 0, 1], interpolation="nearest")
    # Etiquetas de fecha en el eje x.
    ticks = np.linspace(0, len(s) - 1, min(8, len(s))).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(s.index[t])[:7] for t in ticks], rotation=0)
    ax.set_yticks([])
    ax.set_title(title or "Línea temporal de regímenes (0=calma … n-1=crisis)")
    return ax


def plot_crisis_probability(p_crisis, crisis_windows: dict | None = None, ax=None, *,
                            threshold: float = 0.5, title: str | None = None):
    """Probabilidad continua de crisis con ventanas de crisis sombreadas y umbral."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(15, 4))
    p = _as_series(p_crisis)
    ax.plot(p.index, p.values, color=C_CRISIS, lw=0.8, label="P(crisis)")
    ax.axhline(threshold, color="black", ls=":", lw=0.8, alpha=0.6,
               label=f"umbral {threshold:g}")
    _shade_windows(ax, crisis_windows or CRISIS_WINDOWS, C_LONG, 0.10, "crisis conocida")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("P(crisis)")
    ax.set_title(title or "Probabilidad de crisis (CAUSAL OOS)")
    ax.legend(loc="upper left", framealpha=0.9)
    return ax


def plot_duration_histogram(states, n_states: int | None = None, ax=None, *, bins: int = 30,
                            title: str | None = None):
    """Histograma de duraciones de episodios por régimen (colas cortas = flickering)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    s = _as_series(states)
    if n_states is None:
        n_states = int(np.nanmax(s.values)) + 1 if len(s) else 1
    dur = episode_durations(s, n_states)
    labels = {0: "calma", n_states - 1: "crisis"}
    for k in range(n_states):
        if dur.get(k):
            ax.hist(dur[k], bins=bins, alpha=0.6,
                    color=regime_color(k, n_states),
                    label=f"{labels.get(k, f'estado {k}')} (n={len(dur[k])})")
    ax.set_xlabel("duración del episodio (días)")
    ax.set_ylabel("frecuencia")
    ax.set_title(title or "Duración de episodios (CAUSAL OOS) — colas cortas = flickering")
    ax.legend(framealpha=0.9)
    return ax


def plot_transition_matrix(matrix, labels=None, ax=None, *, title: str | None = None):
    """Heatmap de la matriz de transición (detectores con dinámica explícita)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    M = np.asarray(matrix, dtype=float)
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    n = M.shape[0]
    labels = labels or ([f"S{i}" for i in range(n - 1)] + ["crisis"] if n > 1 else ["S0"])
    ax.set_xticks(range(n)); ax.set_xticklabels(labels)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("hacia"); ax.set_ylabel("desde")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] > 0.5 else "black", fontsize=9)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title or "Matriz de transición")
    return ax


def plot_metric_comparison(results_df: pd.DataFrame, metric: str, ax=None, *,
                           name_col: str = "detector", title: str | None = None,
                           ascending: bool = False):
    """Barras comparando una métrica entre detectores (FASE 4)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    df = results_df[[name_col, metric]].dropna().sort_values(metric, ascending=ascending)
    ax.barh(df[name_col], df[metric], color=C_LONG)
    ax.set_xlabel(metric)
    ax.set_title(title or f"Comparación de detectores — {metric}")
    ax.invert_yaxis()
    return ax
