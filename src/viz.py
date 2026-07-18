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

import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Nombres cortos de detector (D1..D12) — FUENTE DE VERDAD ÚNICA
# --------------------------------------------------------------------------- #
# Las figuras/notebooks de detectores necesitan un nombre corto legible por detector
# (se reutilizará en la Fase D). El nombre canónico incluye la K
# elegida por BIC (p. ej. `hmm_tstudent_4s`, `clustering_gmm_k3`); si esa K cambia
# en una re-ejecución, un dict keyed por el nombre completo se rompería (devolvería
# NaN). Para robustez, el mapeo se resuelve por PREFIJO: `canonical_detector()`
# normaliza el sufijo de K y `detector_short()` mapea ese prefijo al label D-N.
SHORT = {
    "rule_vix_threshold": "D1 vix",
    "rule_composite_riskoff": "D2 riskoff",
    "clustering_gmm": "D3 gmm",
    "hmm_gaussian": "D4 hmm-g",
    "markov_switching_var": "D5 msvar",
    "garch_t_vol": "D6 garch",
    "changepoint_online": "D7 cusum",
    "hmm_tstudent": "D8 hmm-t",
    "jump_model": "D9 jump",
    "turbulence_mahalanobis": "D10 turb",
    "msgarch_regime": "D11 msgarch",
    "deep_ae_regime": "D12 ae",
}

# Sufijos que codifican la K seleccionada por BIC: `_k3`, `_4s`, `_2s`, ...
_K_SUFFIX = re.compile(r"_(k\d+|\d+s)$")


def canonical_detector(name: str) -> str:
    """Normaliza el nombre de detector quitando el sufijo de K (robusto a cambios de K).

    'hmm_tstudent_4s' -> 'hmm_tstudent' ; 'clustering_gmm_k3' -> 'clustering_gmm' ;
    'deep_ae_regime'  -> 'deep_ae_regime' (sin cambios).
    """
    return _K_SUFFIX.sub("", str(name))


def detector_short(name: str) -> str:
    """Label corto D-N del detector, resuelto por prefijo (estable ante cambios de K)."""
    return SHORT.get(canonical_detector(name), str(name))

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


# --------------------------------------------------------------------------- #
# Figuras adicionales (Ola 1) — reutilizables por los builders de notebooks
# --------------------------------------------------------------------------- #
# Estas funciones AMPLÍAN el catálogo canónico sin tocar ninguna de las de arriba.
# Todas reutilizan `regime_color`/paleta de casa para que el color de cada régimen
# sea IDÉNTICO al de `plot_price_by_regime`/`plot_regime_timeline`. Matplotlib se
# importa de forma perezosa dentro de cada función (igual que el resto del módulo)
# y, cuando crean su propia figura, aplican `fig.tight_layout()` antes de devolver.
def _regime_label(state: int, n_states: int, labels=None) -> str:
    """Etiqueta legible de un estado: usa `labels` si se da, si no calma/crisis/estado k."""
    if labels is not None:
        try:
            return str(labels[state])
        except (KeyError, IndexError, TypeError):
            pass
    if n_states <= 1:
        return "estado 0"
    if state == 0:
        return "calma"
    if state == n_states - 1:
        return "crisis"
    if n_states == 3 and state == 1:
        return "corrección"
    return f"estado {state}"


def plot_distribution_by_regime(
    values, states, *, crisis_state=None, labels=None, kind: str = "violin",
    ax=None, title: str | None = None, xlabel: str | None = None, bins: int = 40,
):
    """Distribución de una variable (VIX, retorno, turbulencia…) separada POR régimen.

    Compara cómo se distribuye `values` dentro de cada estado canónico (0=calma …
    n-1=crisis), usando el MISMO color de régimen que el resto de figuras. Útil para
    mostrar que la variable de cribado (p. ej. el nivel del VIX) tiene soportes muy
    distintos en calma vs crisis.

    Parameters
    ----------
    values : array-like | pd.Series
        Variable continua alineada 1:1 con `states` (misma longitud / índice).
    states : array-like | pd.Series
        Estado canónico por observación (entero 0..n-1).
    crisis_state : int, opcional
        Estado a destacar como "crisis"; solo afecta al texto del título por defecto.
    labels : dict | list, opcional
        Etiquetas legibles por estado (p. ej. {0: "calma", 1: "crisis"}).
    kind : {"violin", "hist", "box"}
        "violin" (por defecto) = violines lado a lado; "hist" = histogramas
        solapados; "box" = diagrama de cajas.
    ax : matplotlib Axes, opcional
        Eje donde dibujar; si None se crea una figura nueva (y se hace tight_layout).
    title, xlabel : str, opcional
        Título y etiqueta del eje de la variable.
    bins : int
        Nº de bins cuando `kind="hist"`.

    Returns
    -------
    matplotlib.axes.Axes

    Example
    -------
    >>> from src import viz; viz.use_house_style()
    >>> viz.plot_distribution_by_regime(vix_level, states, crisis_state=1,
    ...     kind="violin", xlabel="nivel VIX", title="D1 — VIX por régimen")
    """
    import matplotlib.pyplot as plt

    created = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    s = _as_series(states)
    v = (values if isinstance(values, pd.Series) else pd.Series(np.asarray(values)))
    v = v.reindex(s.index) if v.index.equals(s.index) is False and len(v) == len(s) else v
    vals = np.asarray(v.values, dtype=float)
    st = np.asarray(s.values)
    present = sorted(int(k) for k in np.unique(st[~np.isnan(st.astype(float))]))
    n_states = (int(np.nanmax(st)) + 1) if len(st) else 1
    data, colors, ticklabels = [], [], []
    for k in present:
        d = vals[(st == k) & ~np.isnan(vals)]
        if len(d) == 0:
            continue
        data.append(d)
        colors.append(regime_color(k, n_states))
        ticklabels.append(f"{_regime_label(k, n_states, labels)} (n={len(d)})")
    positions = list(range(1, len(data) + 1))
    if kind == "hist":
        for d, c, lab in zip(data, colors, ticklabels):
            ax.hist(d, bins=bins, alpha=0.55, color=c, label=lab)
        ax.set_xlabel(xlabel or "valor"); ax.set_ylabel("frecuencia")
        ax.legend(framealpha=0.9)
    elif kind == "box":
        bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.65)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_xticks(positions); ax.set_xticklabels(ticklabels, rotation=0)
        ax.set_ylabel(xlabel or "valor")
    else:  # violin (por defecto)
        if data:
            parts = ax.violinplot(data, positions=positions, showmedians=True,
                                  showextrema=False)
            for body, c in zip(parts["bodies"], colors):
                body.set_facecolor(c); body.set_edgecolor("black")
                body.set_alpha(0.65)
            if "cmedians" in parts:
                parts["cmedians"].set_color("black")
        ax.set_xticks(positions); ax.set_xticklabels(ticklabels, rotation=0)
        ax.set_ylabel(xlabel or "valor")
    ax.set_title(title or "Distribución de la variable por régimen")
    if created:
        ax.figure.tight_layout()
    return ax


def plot_feature_space_scatter(
    X, states, *, dims=None, use_pca: bool = True, crisis_state=None,
    feature_names=None, labels=None, ax=None, title: str | None = None,
    max_points: int | None = 6000,
):
    """Scatter 2D del espacio de features coloreado por régimen (PCA si hay >2 dims).

    Proyecta la matriz de features `X` a 2D y colorea cada punto por su estado
    canónico (misma paleta que el resto). Si `X` tiene más de 2 columnas, usa PCA a
    2 componentes (sklearn) salvo que se pasen `dims=(i, j)` explícitas. Para
    inspeccionar separabilidad de regímenes en el espacio de features (D3 GMM,
    D9 jump, D12 autoencoder).

    Parameters
    ----------
    X : array-like (n, d) | pd.DataFrame
        Matriz de features (filas alineadas con `states`).
    states : array-like | pd.Series
        Estado canónico por fila.
    dims : tuple[int, int] | tuple[str, str], opcional
        Par de columnas a usar como ejes (índices enteros o nombres si `X` es
        DataFrame). Si se da, NO se aplica PCA.
    use_pca : bool
        Si True (defecto) y hay >2 dims sin `dims` explícitas, proyecta con PCA(2).
        Si False y hay >2 dims, usa las 2 primeras columnas.
    crisis_state : int, opcional
        Solo afecta a la etiqueta de leyenda destacada.
    feature_names : list[str], opcional
        Nombres de las features (si `X` no es DataFrame) para rotular ejes.
    labels : dict | list, opcional
        Etiquetas legibles por estado.
    ax : matplotlib Axes, opcional
    title : str, opcional
    max_points : int, opcional
        Submuestreo aleatorio (semilla fija) para no saturar el scatter; None = todo.

    Returns
    -------
    matplotlib.axes.Axes

    Example
    -------
    >>> viz.plot_feature_space_scatter(features_df, states, use_pca=True,
    ...     title="D3 — espacio de features (PCA) por régimen")
    """
    import matplotlib.pyplot as plt

    created = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
        Xv = X.values.astype(float)
    else:
        Xv = np.asarray(X, dtype=float)
        if Xv.ndim == 1:
            Xv = Xv.reshape(-1, 1)
        names = list(feature_names) if feature_names is not None else [
            f"x{i}" for i in range(Xv.shape[1])]

    st = np.asarray(_as_series(states).values)
    n_states = (int(np.nanmax(st)) + 1) if len(st) else 1

    # Resolver coordenadas 2D y etiquetas de eje.
    xlabel = ylabel = None
    if dims is not None:
        i, j = dims
        if isinstance(i, str):
            i = names.index(i)
        if isinstance(j, str):
            j = names.index(j)
        XY = np.column_stack([Xv[:, i], Xv[:, j]])
        xlabel, ylabel = names[i], names[j]
    elif Xv.shape[1] <= 2:
        XY = Xv if Xv.shape[1] == 2 else np.column_stack([Xv[:, 0], np.zeros(len(Xv))])
        xlabel = names[0]
        ylabel = names[1] if Xv.shape[1] == 2 else ""
    elif use_pca:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        Z = StandardScaler().fit_transform(Xv)
        pca = PCA(n_components=2, random_state=0)
        XY = pca.fit_transform(Z)
        ev = pca.explained_variance_ratio_
        xlabel = f"PC1 ({ev[0]:.0%} var)"
        ylabel = f"PC2 ({ev[1]:.0%} var)"
    else:
        XY = np.column_stack([Xv[:, 0], Xv[:, 1]])
        xlabel, ylabel = names[0], names[1]

    # Submuestreo reproducible.
    idx = np.arange(len(XY))
    if max_points is not None and len(idx) > max_points:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))

    for k in sorted(int(u) for u in np.unique(st)):
        m = (st[idx] == k)
        if not m.any():
            continue
        lab = _regime_label(k, n_states, labels)
        ax.scatter(XY[idx][m, 0], XY[idx][m, 1], s=10, alpha=0.55,
                   color=regime_color(k, n_states), edgecolors="none", label=lab)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title or "Espacio de features coloreado por régimen")
    ax.legend(framealpha=0.9, markerscale=1.6)
    if created:
        ax.figure.tight_layout()
    return ax


def plot_regime_correlation_heatmaps(
    X, states, *, feature_names=None, regimes=None, max_regimes: int = 3,
    precision: bool = False, labels=None, title: str | None = None,
):
    """Rejilla de heatmaps de correlación (o de precisión Σ⁻¹) calculados POR régimen.

    Para cada estado seleccionado calcula la matriz de correlación (o la matriz de
    precisión Σ⁻¹, si `precision=True`) de las features SOLO con las observaciones de
    ese régimen, y la pinta como heatmap. Visualiza cómo cambia la estructura de
    covarianza entre calma y crisis — el mecanismo central de D10 (turbulencia de
    Mahalanobis: Σ⁻¹ es lo que "ve" el colapso de correlaciones).

    Parameters
    ----------
    X : array-like (n, d) | pd.DataFrame
        Matriz de features alineada con `states`.
    states : array-like | pd.Series
        Estado canónico por fila.
    feature_names : list[str], opcional
        Nombres de feature (si `X` no es DataFrame).
    regimes : list[int], opcional
        Estados concretos a pintar; por defecto los `max_regimes` con más muestras.
    max_regimes : int
        Máximo de paneles si no se pasan `regimes` explícitos.
    precision : bool
        False (defecto) = correlación; True = matriz de precisión Σ⁻¹ (estandarizada
        a correlación parcial, signo invertido en fuera-diagonal para legibilidad).
    labels : dict | list, opcional
        Etiquetas legibles por estado para los títulos de cada panel.
    title : str, opcional
        Título global (suptitle).

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------
    >>> fig = viz.plot_regime_correlation_heatmaps(X_maha, states, precision=True,
    ...     title="D10 — estructura de covarianza por régimen (Σ⁻¹)")
    """
    import matplotlib.pyplot as plt

    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
        Xv = X.values.astype(float)
    else:
        Xv = np.asarray(X, dtype=float)
        if Xv.ndim == 1:
            Xv = Xv.reshape(-1, 1)
        names = list(feature_names) if feature_names is not None else [
            f"x{i}" for i in range(Xv.shape[1])]

    st = np.asarray(_as_series(states).values)
    n_states = (int(np.nanmax(st)) + 1) if len(st) else 1
    if regimes is None:
        uniq, counts = np.unique(st[~np.isnan(st.astype(float))].astype(int),
                                 return_counts=True)
        order = uniq[np.argsort(-counts)]
        regimes = sorted(order[:max_regimes].tolist())
    regimes = [int(r) for r in regimes]

    npan = max(1, len(regimes))
    fig, axes = plt.subplots(1, npan, figsize=(4.2 * npan, 4.0), squeeze=False)
    axes = axes[0]
    d = Xv.shape[1]
    im = None
    for axx, k in zip(axes, regimes):
        sub = Xv[st == k]
        kind = "Σ⁻¹ (precisión)" if precision else "correlación"
        if sub.shape[0] > d:
            C = np.corrcoef(sub, rowvar=False)
            if precision:
                try:
                    P = np.linalg.inv(C)
                    dinv = np.sqrt(np.clip(np.diag(P), 1e-12, None))
                    # Correlación parcial: -P_ij / sqrt(P_ii P_jj); diagonal = 1.
                    M = -P / np.outer(dinv, dinv)
                    np.fill_diagonal(M, 1.0)
                except np.linalg.LinAlgError:
                    M = C
            else:
                M = C
        else:
            M = np.full((d, d), np.nan)
        im = axx.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
        axx.set_xticks(range(d)); axx.set_xticklabels(names, rotation=45, ha="right",
                                                       fontsize=8)
        axx.set_yticks(range(d)); axx.set_yticklabels(names, fontsize=8)
        if d <= 8:
            for i in range(d):
                for j in range(d):
                    if M[i, j] == M[i, j]:
                        axx.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                                 fontsize=7,
                                 color="white" if abs(M[i, j]) > 0.6 else "black")
        axx.set_title(f"{_regime_label(k, n_states, labels)}  (n={int((st == k).sum())})\n{kind}",
                      fontsize=10)
    if im is not None:
        fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02,
                     label="parcial" if precision else "ρ")
    if title:
        fig.suptitle(title, y=1.04)
    fig.tight_layout()
    return fig


def plot_fold_panel(
    fold_index, values, *, ylabel: str | None = None, ref_line: float | None = None,
    highlight=None, ax=None, title: str | None = None, baseline_ok: bool = True,
):
    """Métrica por fold del walk-forward (diagnóstico de degeneración por bloque).

    Barras de una métrica medida en cada fold/bloque (p. ej. nº de regímenes distintos
    por fold, cobertura por fold, switching por fold) para localizar QUÉ fold colapsa
    (p. ej. D11 MS-GARCH cae a un único régimen en algún bloque). Permite resaltar
    folds problemáticos en rojo crisis.

    Parameters
    ----------
    fold_index : array-like
        Identificadores de fold (enteros, fechas o etiquetas) para el eje x.
    values : array-like
        Valor de la métrica por fold (misma longitud que `fold_index`).
    ylabel : str, opcional
        Etiqueta del eje y (la métrica).
    ref_line : float, opcional
        Línea horizontal de referencia (p. ej. 1 régimen = degeneración, o 0.5).
    highlight : callable | array-like[bool] | list[int], opcional
        Folds a resaltar en rojo: una máscara booleana, una lista de posiciones, o
        un predicado `f(valor) -> bool` (p. ej. `lambda v: v <= 1`).
    ax : matplotlib Axes, opcional
    title : str, opcional
    baseline_ok : bool
        Si True, las barras no resaltadas se pintan en azul calma (C_LONG); si False
        en gris (C_NEG).

    Returns
    -------
    matplotlib.axes.Axes

    Example
    -------
    >>> viz.plot_fold_panel(range(n_folds), n_regimes_per_fold, ylabel="nº regímenes",
    ...     ref_line=1, highlight=lambda v: v <= 1, title="D11 — folds que colapsan")
    """
    import matplotlib.pyplot as plt

    created = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.2))
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    pos = np.arange(n)
    # Resolver máscara de resaltado.
    mask = np.zeros(n, dtype=bool)
    if callable(highlight):
        mask = np.array([bool(highlight(v)) for v in vals])
    elif highlight is not None:
        hl = np.asarray(highlight)
        if hl.dtype == bool and len(hl) == n:
            mask = hl
        else:  # lista de posiciones
            mask[hl.astype(int)] = True
    base_c = C_LONG if baseline_ok else C_NEG
    colors = [C_CRISIS if m else base_c for m in mask]
    ax.bar(pos, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(pos)
    ax.set_xticklabels([str(f) for f in fold_index], rotation=45, ha="right",
                       fontsize=8)
    if ref_line is not None:
        ax.axhline(ref_line, color="black", ls="--", lw=0.8, alpha=0.7)
    ax.set_xlabel("fold (bloque walk-forward)")
    ax.set_ylabel(ylabel or "métrica")
    ax.set_title(title or "Métrica por fold (walk-forward)")
    if created:
        ax.figure.tight_layout()
    return ax


def render_table_figure(
    df, *, title: str | None = None, max_rows: int | None = None,
    highlight_cols=None, fmt=None, col_width: float | None = None,
):
    """Renderiza un DataFrame pequeño como FIGURA de tabla limpia (embebible en LaTeX).

    Sustituye los `df.style`/Stylers de pandas (que no se exportan bien a PNG/PDF) por
    una tabla matplotlib sobria con la cabecera en azul de casa. Pensado para tablas
    compactas de comparación (caso D4 y cualquier tabla de métricas por ventana).

    Parameters
    ----------
    df : pd.DataFrame
        Tabla a renderizar (se incluye el índice como primera columna).
    title : str, opcional
        Título sobre la tabla.
    max_rows : int, opcional
        Trunca a las primeras `max_rows` filas (añade una fila "…" si se truncó).
    highlight_cols : list[str], opcional
        Columnas cuyo fondo se tiñe suavemente (ámbar) para destacarlas.
    fmt : str | dict | callable, opcional
        Formato de celdas numéricas: cadena tipo "{:.2f}" o ".2f", un dict
        {columna: fmt}, o una función `f(valor) -> str`.
    col_width : float, opcional
        Ancho relativo por columna (se autoescala el tamaño de la figura).

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------
    >>> fig = viz.render_table_figure(cmp.round(3), title="D4 — cobertura por ventana",
    ...     highlight_cols=["cobertura_OOS"], fmt="{:.1%}")
    """
    import matplotlib.pyplot as plt

    d = df.copy()
    truncated = False
    if max_rows is not None and len(d) > max_rows:
        d = d.iloc[:max_rows]
        truncated = True

    def _fmt_cell(col, val):
        if isinstance(fmt, dict):
            f = fmt.get(col)
        elif callable(fmt) or isinstance(fmt, str):
            f = fmt
        else:
            f = None
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        if f is None:
            if isinstance(val, float):
                return f"{val:.3g}"
            return str(val)
        if callable(f):
            return str(f(val))
        f = f if "{" in f else "{:" + f + "}"
        try:
            return f.format(val)
        except (ValueError, KeyError):
            return str(val)

    cols = list(d.columns)
    index_name = d.index.name or ""
    header = [index_name] + [str(c) for c in cols]
    cell_text = []
    for idx, row in d.iterrows():
        cell_text.append([str(idx)] + [_fmt_cell(c, row[c]) for c in cols])
    if truncated:
        cell_text.append(["…"] * len(header))

    ncol = len(header)
    nrow = len(cell_text) + 1
    fig_w = max(4.0, 1.5 * ncol)
    fig_h = max(1.2, 0.45 * nrow + 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=header, cellLoc="center",
                     loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    if col_width is not None:
        for cell in table.get_celld().values():
            cell.set_width(col_width)
    hl = set(highlight_cols or [])
    hl_pos = {i + 1 for i, c in enumerate(cols) if c in hl}  # +1 por la col índice
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:  # cabecera
            cell.set_facecolor(C_LONG)
            cell.set_text_props(color="white", fontweight="bold")
        elif c in hl_pos:
            cell.set_facecolor("#f6e6c8")  # ámbar muy suave
    if title:
        ax.set_title(title, fontsize=11, pad=10)
    # Las tablas de matplotlib no son compatibles con tight_layout (avisa por
    # warning); usamos subplots_adjust para dejar margen homogéneo sin ruido.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90 if title else 0.98,
                        bottom=0.02)
    return fig


def plot_grouped_bars(
    categories, series_dict, *, ylabel: str | None = None, title: str | None = None,
    colors=None, ax=None, rotation: float = 0.0, value_labels: bool = False,
):
    """Barras agrupadas genéricas para comparativas (varias series por categoría).

    Comparador flexible cuando `plot_metric_comparison` (1 métrica entre detectores)
    se queda corto: aquí cada categoría del eje x tiene N barras (una por serie del
    dict). Para comparar p. ej. D9 vs D3 en varias métricas, AE vs PCA, o robusto vs
    gaussiano por ventana.

    Nota: si solo necesitas UNA métrica ordenada entre detectores, usa la función
    existente `plot_metric_comparison`. Esta es para >1 serie por categoría.

    Parameters
    ----------
    categories : list[str]
        Etiquetas del eje x (las categorías comunes).
    series_dict : dict[str, array-like]
        {nombre_serie: valores}; cada array tiene len == len(categories).
    ylabel, title : str, opcional
    colors : list, opcional
        Colores por serie (por defecto azul/ámbar/gris/rojo de la paleta de casa).
    ax : matplotlib Axes, opcional
    rotation : float
        Rotación de las etiquetas del eje x.
    value_labels : bool
        Si True, anota el valor encima de cada barra.

    Returns
    -------
    matplotlib.axes.Axes

    Example
    -------
    >>> viz.plot_grouped_bars(["2013", "2018"],
    ...     {"robust": [0.0, 0.05], "gaussian": [1.0, 1.0]},
    ...     ylabel="falsa alarma", title="CP2 — robusto vs gaussiano en trampas")
    """
    import matplotlib.pyplot as plt

    created = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5))
    cats = list(categories)
    series = {k: np.asarray(v, dtype=float) for k, v in series_dict.items()}
    nser = len(series)
    palette = colors or [C_LONG, C_SHORT, C_NEG, C_CRISIS, "#5b8fb0"]
    x = np.arange(len(cats))
    total_w = 0.8
    w = total_w / max(1, nser)
    for i, (name, vals) in enumerate(series.items()):
        offset = (i - (nser - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, label=name,
                      color=palette[i % len(palette)], edgecolor="black",
                      linewidth=0.5)
        if value_labels:
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{b.get_height():.2f}", ha="center", va="bottom",
                        fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=rotation, ha="center" if rotation == 0 else "right")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title or "Comparativa por categoría")
    ax.legend(framealpha=0.9)
    if created:
        ax.figure.tight_layout()
    return ax
