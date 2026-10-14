"""Figuras del Tema 4 (notebooks t4_01 y t4_02). Solo matplotlib, estilo de la casa de la Fase 1.

Convenciones comunes a todas las figuras:

- Colores por familia de fuente (``C_FAMILIA``), por tipo de dato (``C_TIPO``) y por estado (``C_ESTADO``).
- Crisis de la Fase 1 sombreadas en rojo sobrio (``sombrear_crisis``), días testigo como líneas verticales
  punteadas (``marcar_testigos``).
- Cada función recibe tablas de ``results/ingesta/noticias_alt`` (o series del corpus) y devuelve ``(fig, ax)``;
  no hace ``plt.show()`` (lo hace el notebook) ni guarda nada.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ingesta.noticias_alt.muestra import dias_testigo

try:  # estilo de la Fase 1 si está el paquete
    from regimenes.viz.figuras import C_CRISIS, use_house_style
except Exception:  # noqa: BLE001
    C_CRISIS = "#b5341f"

    def use_house_style() -> None:  # type: ignore[misc]
        plt.style.use("default")

C_FAMILIA = {
    "texto_historico": "#2b5d8a",
    "gdelt_wiki": "#5b8fb0",
    "vivo_redes": "#3a9e5c",
    "divulgaciones": "#c98a2b",
    "prediccion": "#8e5cc2",
    "indices_ews": "#b5341f",
}
C_TIPO = {
    "texto": "#2b5d8a", "titular": "#5b8fb0", "serie": "#c98a2b", "evento": "#8e5cc2",
    "instantanea": "#3a9e5c", "ficha": "#bdbdbd",
}
C_ESTADO = {"ok": "#3a9e5c", "parcial": "#e0a526", "fallo": "#b5341f", "no_probada": "#9e9e9e"}
C_SEMAFORO = {"apta_backtest": "#3a9e5c", "solo_vivo": "#e0a526", "descartar": "#b5341f", "contexto": "#5b8fb0"}


def estilo(dpi: int = 90) -> None:
    """Estilo común (el de la Fase 1) y dpi contenido para que el notebook pese poco."""
    use_house_style()
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi


def sombrear_crisis(ax, crisis: Mapping[str, tuple[str, str]] | None = None, *, desde=None, hasta=None,
                    etiquetas: bool = False, alpha: float = 0.15) -> None:
    """Sombrea las ventanas de crisis de la Fase 1 que caen en el eje (por defecto, pista A)."""
    if crisis is None:
        try:
            from ingesta.noticias_alt.muestra import crisis_fase1

            crisis = crisis_fase1("A")
        except Exception:  # noqa: BLE001
            return
    x0, x1 = (pd.Timestamp(desde) if desde is not None else None), (pd.Timestamp(hasta) if hasta is not None else None)
    primera = True
    for nombre, (ini, fin) in crisis.items():
        a, b = pd.Timestamp(ini), pd.Timestamp(fin)
        if (x0 is not None and b < x0) or (x1 is not None and a > x1):
            continue
        ax.axvspan(a, b, color=C_CRISIS, alpha=alpha, lw=0, label="crisis (Fase 1)" if primera else None)
        primera = False
        if etiquetas:
            ax.text(a, 1.0, nombre.split("_")[0], transform=ax.get_xaxis_transform(), fontsize=6,
                    rotation=90, va="top", ha="right", color=C_CRISIS)


def marcar_testigos(ax, *, con_texto: bool = True, incluir_hoy: bool = False) -> None:
    """Líneas verticales en los días testigo."""
    for d in dias_testigo(incluir_hoy=incluir_hoy):
        ax.axvline(pd.Timestamp(d.fecha), color="#333333", ls=":", lw=0.9)
        if con_texto:
            ax.text(pd.Timestamp(d.fecha), 1.01, d.clave, transform=ax.get_xaxis_transform(), fontsize=7,
                    rotation=45, ha="left", va="bottom", color="#333333")


def eje_anios(ax, cada: int = 5) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(cada))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def gantt_cobertura(cob: pd.DataFrame, *, desde: str = "1979-01-01", hasta=None, ax=None):
    """Una barra por fuente (inicio → fin), coloreada por tipo de dato; rombo si trae hora.

    ``cob``: tabla ``cobertura`` (``base.COLUMNAS_COBERTURA``). Las fuentes sin fechas medidas se listan
    en gris a la izquierda.
    """
    c = cob.copy()
    c["inicio"] = pd.to_datetime(c["inicio"], errors="coerce")
    c["fin"] = pd.to_datetime(c["fin"], errors="coerce")
    hasta = pd.Timestamp(hasta) if hasta is not None else pd.Timestamp.today().normalize()
    c["fin"] = c["fin"].fillna(hasta)
    orden = {f: i for i, f in enumerate(C_FAMILIA)}
    c = c.assign(_o=c["familia"].map(orden).fillna(99)).sort_values(["_o", "inicio"]).reset_index(drop=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, max(4, 0.28 * len(c) + 1)))
    else:
        fig = ax.figure
    for i, r in c.iterrows():
        col = C_TIPO.get(r["tipo"], "#999999")
        if pd.isna(r["inicio"]):
            ax.plot([pd.Timestamp(desde)], [i], marker="x", color="#9e9e9e")
            continue
        ax.plot([max(r["inicio"], pd.Timestamp(desde)), r["fin"]], [i, i], color=col, lw=5, solid_capstyle="butt",
                alpha=1.0 if r.get("backtest", True) else 0.45)
        if bool(r.get("hora", False)):
            ax.plot(r["fin"], i, marker="D", ms=3, color="black")
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels(c["fuente"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(pd.Timestamp(desde), hasta + pd.Timedelta(days=200))
    eje_anios(ax)
    sombrear_crisis(ax, desde=desde)
    marcar_testigos(ax)
    for t, col in C_TIPO.items():
        ax.plot([], [], color=col, lw=5, label=t)
    ax.plot([], [], marker="D", ls="", color="black", ms=3, label="trae hora")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.set_title("Cobertura medida por fuente (barra tenue: no sirve para backtest)", pad=28)
    return fig, ax


def mapa_epocas(cob: pd.DataFrame, epocas: Iterable[tuple[str, str, str]] | None = None, ax=None):
    """Mapa de calor época × fuente: 0 nada, 1 serie/evento, 2 titular, 3 texto completo."""
    epocas = list(epocas or [
        ("1979-1999", "1979-01-01", "1999-12-31"), ("2000-2006", "2000-01-01", "2006-12-31"),
        ("2007-2015", "2007-01-01", "2015-12-31"), ("2016-2019", "2016-01-01", "2019-12-31"),
        ("2020-2023", "2020-01-01", "2023-12-31"), ("2024-hoy", "2024-01-01", "2100-01-01"),
    ])
    nivel = {"texto": 3, "titular": 2, "serie": 1, "evento": 1, "instantanea": 1, "ficha": 0}
    c = cob.copy()
    c["inicio"] = pd.to_datetime(c["inicio"], errors="coerce")
    c["fin"] = pd.to_datetime(c["fin"], errors="coerce").fillna(pd.Timestamp.today())
    m = np.zeros((len(c), len(epocas)))
    for i, r in enumerate(c.itertuples()):
        if pd.isna(r.inicio):
            continue
        for j, (_, a, b) in enumerate(epocas):
            if r.inicio <= pd.Timestamp(b) and r.fin >= pd.Timestamp(a):
                m[i, j] = nivel.get(r.tipo, 1)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.26 * len(c) + 1)))
    else:
        fig = ax.figure
    cmap = plt.matplotlib.colors.ListedColormap(["#f2f2f2", "#c98a2b", "#5b8fb0", "#2b5d8a"])
    ax.imshow(m, aspect="auto", cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks(range(len(epocas)))
    ax.set_xticklabels([e[0] for e in epocas], fontsize=8)
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels(c["fuente"], fontsize=7)
    for k, (t, col) in enumerate(zip(["nada", "serie / evento", "titular", "texto completo"],
                                     ["#f2f2f2", "#c98a2b", "#5b8fb0", "#2b5d8a"])):
        ax.plot([], [], "s", color=col, label=t)
    # Fuentes no medidas (no_probada / fallo): cobertura solo documentada -> rayado.
    if "estado" in c.columns:
        for i, est in enumerate(c["estado"].astype(str)):
            if est in ("no_probada", "fallo"):
                for j in range(len(epocas)):
                    if m[i, j] > 0:
                        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="////",
                                                   edgecolor="white", lw=0))
        ax.add_patch(plt.Rectangle((0, 0), 0, 0, facecolor="#2b5d8a", hatch="////", edgecolor="white",
                                   label="documentada, no medida"))
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Qué hay en cada época")
    return fig, ax


def barras_estado(registro: pd.DataFrame, ax=None):
    """Peticiones por fuente apiladas por resultado (ok / error)."""
    g = registro.groupby(["fuente", "ok"]).size().unstack(fill_value=0)
    g = g.reindex(columns=[True, False], fill_value=0).sort_values(True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, max(3, 0.25 * len(g) + 1)))
    else:
        fig = ax.figure
    ax.barh(g.index, g[True], color=C_ESTADO["ok"], label="ok")
    ax.barh(g.index, g[False], left=g[True], color=C_ESTADO["fallo"], label="error")
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7)
    ax.set_title("Peticiones de la sonda por fuente")
    return fig, ax


def paneles_series(series: Mapping[str, pd.Series], *, ncol: int = 2, desde=None, alto: float = 2.3,
                   titulo: str | None = None):
    """Pequeños múltiplos: una serie por panel con las crisis sombreadas.

    Los NaN interiores se conservan (la línea se corta en los huecos); solo se recortan los de los extremos.
    """
    n = len(series)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.3 * ncol, alto * nrow), squeeze=False)
    for ax, (nombre, s) in zip(axes.ravel(), series.items()):
        s = s.loc[s.first_valid_index():s.last_valid_index()] if s.notna().any() else s.dropna()
        if desde is not None:
            s = s[s.index >= pd.Timestamp(desde)]
        ax.plot(s.index, s.values, lw=0.8, color="#2b5d8a")
        sombrear_crisis(ax, desde=s.index.min() if len(s) else None)
        ax.set_title(nombre, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    if titulo:
        fig.suptitle(titulo)
    fig.tight_layout()
    return fig, axes


# =========================================================================== utilidades internas

DESTINOS_ESQUEMA = {
    "cajon": "Cajón (RAG de cajón)",
    "base": "RAG base",
    "serie": "Serie TimeGate",
    "vivo": "Solo en vivo",
    "ninguno": "Descartada",
}
_VERDE = {"ok", "si", "sí", "pasa", "apta", "verde", "true", "limpio", "sin_look_ahead", "cumple"}
_AMBAR = {"parcial", "revisable", "ambar", "ámbar", "dudoso", "aviso", "riesgo", "vintage_actual_revisable"}
_ROJO = {"fallo", "no", "falla", "rojo", "false", "look-ahead", "look_ahead", "lookahead", "no_cumple"}
C_GRIS = "#9e9e9e"
C_BASE = "#2b5d8a"
# Escala continua del semáforo (rojo -> ámbar -> verde) con tonos medios: el texto negro se lee encima.
_CMAP_SEMAFORO = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "semaforo", [C_ESTADO["fallo"], C_ESTADO["parcial"], C_ESTADO["ok"]])


def _fig_ax(ax, figsize):
    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.figure, ax


def _vacio(ax, texto: str = "sin datos") -> None:
    ax.text(0.5, 0.5, texto, transform=ax.transAxes, ha="center", va="center", color=C_GRIS, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _fechas(s) -> pd.Series:
    """Fechas (str, date, Timestamp con o sin zona) a ``datetime64`` sin zona (UTC)."""
    s = pd.Series(s).map(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)
    t = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    return t.dt.tz_convert(None)


def _num(s) -> pd.Series:
    return pd.to_numeric(pd.Series(s), errors="coerce")


def _bool(s) -> pd.Series:
    return pd.Series(s).astype(str).str.lower().isin(["true", "1", "1.0"])


def _leyenda(ax, **kw) -> None:
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=kw.pop("fontsize", 7), **kw)


def _a_series(datos, *, col_serie: str = "serie", col_fecha: str = "fecha_referencia",
              col_valor: str = "valor") -> dict[str, pd.Series]:
    """Mapa ``{nombre: Series}`` o tabla larga (``COLUMNAS_SERIE``) a ``{nombre: Series}`` indexada por fecha."""
    if isinstance(datos, Mapping):
        return dict(datos)
    if datos is None or len(datos) == 0:
        return {}
    d = datos.assign(_f=_fechas(datos[col_fecha]).values, _v=_num(datos[col_valor]).values)
    return {n: g.set_index("_f")["_v"].sort_index() for n, g in d.groupby(col_serie, sort=False)}


def _matriz(df: pd.DataFrame, fila: str, columna: str | None, valor: str | None) -> pd.DataFrame:
    """Tabla larga (fila, columna, valor) o ancha (fila + una columna por criterio) a matriz fila × columna."""
    if columna is not None and columna in df.columns and valor is not None and valor in df.columns:
        return df.pivot_table(index=fila, columns=columna, values=valor, aggfunc="first", sort=False)
    return df.set_index(fila) if fila in df.columns else df


def color_valor(v, *, vmin: float = 0.0, vmax: float = 1.0) -> str:
    """Color de semáforo de un valor: texto (ok/parcial/fallo, sí/no, veredictos), booleano o número en [vmin, vmax]
    (verde = alto). Blanco si el texto no se reconoce; gris claro si falta."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "#f2f2f2"
    if isinstance(v, (bool, np.bool_)):
        return C_ESTADO["ok"] if v else C_ESTADO["fallo"]
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = (float(v) - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        return plt.matplotlib.colors.to_hex(_CMAP_SEMAFORO(min(max(x, 0.0), 1.0)))
    t = str(v).strip().lower().replace(" ", "_")
    if t in C_SEMAFORO:
        return C_SEMAFORO[t]
    if t in _VERDE:
        return C_ESTADO["ok"]
    if t in _AMBAR:
        return C_ESTADO["parcial"]
    if t in _ROJO:
        return C_ESTADO["fallo"]
    return "#ffffff"


def _texto_celda(v, ancho: int = 22) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (float, np.floating)):
        if float(v).is_integer():
            return f"{int(v)}"
        return f"{v:.0f}" if abs(v) >= 100 else f"{v:.3g}"
    t = str(v)
    return t if len(t) <= ancho else t[: ancho - 1] + "…"


def _orden_familia(df: pd.DataFrame) -> pd.DataFrame:
    orden = {f: i for i, f in enumerate(C_FAMILIA)}
    fam = df["familia"] if "familia" in df else pd.Series("", index=df.index)
    return df.assign(_o=fam.map(orden).fillna(99)).sort_values(["_o", "fuente"]).reset_index(drop=True)


# =========================================================================== t4_01 · sonda

def esquema_destinos(cob: pd.DataFrame, ax=None):
    """Esquema fuentes → destino (cajón, RAG base, serie, vivo, descartada) con cajas y flechas.

    ``cob``: tabla ``cobertura`` (``fuente, familia, destino, estado``). ``base+cajon`` dibuja dos flechas.
    Flecha continua si la fuente está ``ok``; discontinua si no.
    """
    c = _orden_familia(cob)
    n = max(len(c), 1)
    fig, ax = _fig_ax(ax, (10, max(4.0, 0.2 * n + 1.2)))
    ax.set_xlim(0, 10)
    ax.set_ylim(n - 0.2, -0.8)
    ax.axis("off")
    destinos = list(DESTINOS_ESQUEMA)
    y_dest = {d: (k + 0.5) * n / len(destinos) - 0.5 for k, d in enumerate(destinos)}
    caja = dict(boxstyle="round,pad=0.25", lw=0.6)
    for d, y in y_dest.items():
        ax.text(8.3, y, DESTINOS_ESQUEMA[d], ha="center", va="center", fontsize=9, fontweight="bold",
                bbox=dict(caja, fc="#f2f2f2", ec="#555555"))
    for i, r in c.iterrows():
        col = C_FAMILIA.get(r.get("familia"), C_GRIS)
        ax.text(2.2, i, str(r["fuente"]), ha="right", va="center", fontsize=6.5, color="white",
                bbox=dict(caja, fc=col, ec=col))
        ok = str(r.get("estado", "ok")) == "ok"
        destino = r.get("destino")
        partes = [x for x in str(destino if isinstance(destino, str) and destino else "ninguno").split("+") if x]
        for d in partes or ["ninguno"]:
            d = d if d in y_dest else "ninguno"
            ax.annotate("", xy=(7.0, y_dest[d]), xytext=(2.35, i),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=0.7, alpha=0.8,
                                        ls="-" if ok else "--", shrinkA=0, shrinkB=2))
    for fam, col in C_FAMILIA.items():
        ax.plot([], [], "s", color=col, label=fam)
    ax.plot([], [], color="#555555", ls="--", lw=0.8, label="estado ≠ ok")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False)
    ax.set_title("Destino de cada fuente en el sistema")
    return fig, ax


def estado_fuentes(cob: pd.DataFrame, registro: pd.DataFrame | None = None, ax=None):
    """Una barra por fuente: % de peticiones correctas de la sonda (rayada a 100 si no hay registro), color = estado."""
    c = cob.copy()
    if registro is not None and len(registro):
        pct = registro.assign(_ok=_bool(registro["ok"]).values).groupby("fuente")["_ok"].mean().mul(100)
        c["pct_ok"] = c["fuente"].map(pct)
    else:
        c["pct_ok"] = np.nan
    c = _orden_familia(c)
    fig, ax = _fig_ax(ax, (8, max(3.0, 0.2 * len(c) + 1)))
    ax.barh(range(len(c)), c["pct_ok"].fillna(100), color=[C_ESTADO.get(e, C_GRIS) for e in c["estado"]],
            hatch=["//" if pd.isna(p) else "" for p in c["pct_ok"]], edgecolor="white", lw=0.3)
    for i, (p, e) in enumerate(zip(c["pct_ok"], c["estado"])):
        if p == 0:
            ax.plot(0.8, i, "x", color=C_ESTADO.get(e, C_GRIS), ms=5)
            ax.text(2, i, "0 %", va="center", fontsize=6.5, color=C_ESTADO.get(e, C_GRIS))
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels(c["fuente"], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("% de peticiones correctas (rayado: sin peticiones registradas)")
    for e, col in C_ESTADO.items():
        ax.plot([], [], "s", color=col, label=e)
    _leyenda(ax, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Estado medido por fuente")
    return fig, ax


def barras_idiomas(df: pd.DataFrame, *, col: str = "idioma", peso: str | None = None, top: int = 12,
                   titulo: str = "Idiomas", ax=None):
    """Barras horizontales del % por idioma (cuenta filas o suma la columna ``peso``)."""
    fig, ax = _fig_ax(ax, (5, 3.5))
    if df is None or len(df) == 0 or col not in df:
        _vacio(ax)
        return fig, ax
    s = (df.groupby(col)[peso].sum() if peso else df[col].fillna("und").value_counts()).astype(float)
    s = (100 * s / s.sum()).sort_values(ascending=False)
    if len(s) > top:
        s = pd.concat([s.iloc[:top], pd.Series({"resto": s.iloc[top:].sum()})])
    ax.barh(s.index.astype(str), s.values, color=C_BASE)
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("% del total")
    ax.set_title(titulo, fontsize=10)
    return fig, ax


def ccnews_tokens_idiomas(art: pd.DataFrame, idiomas: pd.DataFrame | None = None, *, umbral: int | None = None):
    """CC-NEWS: histograma de tokens por respuesta (escala log) e idiomas.

    ``art``: ``ccnews_articulos``. ``idiomas``: ``hf_ccnews_idiomas`` (suma ``filas`` por idioma); si falta,
    se cuentan los idiomas de ``art``.
    """
    if umbral is None:
        try:
            from ingesta.noticias_alt.ccnews import MIN_TOKENS as umbral
        except Exception:  # noqa: BLE001
            umbral = 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1.4, 1]))
    tok = _num(art["n_tokens"]).dropna() if art is not None and "n_tokens" in art else pd.Series(dtype=float)
    tok = tok[tok > 0]
    if len(tok):
        bins = np.logspace(0, np.log10(max(tok.max(), 10)) + 0.1, 30)
        axes[0].hist(tok, bins=bins, color=C_BASE)
        axes[0].set_xscale("log")
        axes[0].axvline(umbral, color=C_CRISIS, ls="--", lw=1, label=f"umbral de artículo ({umbral} tokens)")
        axes[0].set_xlabel("tokens por respuesta (escala log)")
        axes[0].set_ylabel("respuestas")
        axes[0].set_title(f"Tamaño de las respuestas (n = {len(tok)}; mediana {tok.median():.0f})", fontsize=10)
        _leyenda(axes[0])
    else:
        _vacio(axes[0])
    if idiomas is not None and len(idiomas):
        barras_idiomas(idiomas, peso="filas", titulo="Idiomas (HF ccnews, filas)", ax=axes[1])
    else:
        barras_idiomas(art, titulo="Idiomas (muestra CC-NEWS)", ax=axes[1])
    fig.tight_layout()
    return fig, axes


def fecha_articulo_vs_rastreo(art: pd.DataFrame, *, col_pub: str = "fecha_publicacion", col_rastreo: str = "warc_date",
                              col_panel: str = "dia", col_color: str = "regla"):
    """Fecha del artículo frente a la de rastreo (WARC), un panel por día; la diagonal es «sin retraso»."""
    d = art.assign(_p=_fechas(art[col_pub]).values, _r=_fechas(art[col_rastreo]).values).dropna(subset=["_p", "_r"])
    if col_panel not in d:
        d[col_panel] = "todo"
    d[col_color] = d[col_color].fillna("?").astype(str) if col_color in d else "?"
    paneles = list(dict.fromkeys(d[col_panel]))
    k = max(len(paneles), 1)
    fig, axes = plt.subplots(1, k, figsize=(min(3.4 * k, 12), 3.4), squeeze=False)
    axes = axes.ravel()
    paleta = dict(zip(dict.fromkeys(d[col_color]), [C_BASE, "#c98a2b", "#3a9e5c", "#8e5cc2", C_GRIS]))
    if d.empty:
        _vacio(axes[0])
    for ax, p in zip(axes, paneles):
        g = d[d[col_panel] == p]
        for reg, h in g.groupby(col_color):
            ax.scatter(h["_p"], h["_r"], s=12, color=paleta.get(reg, C_GRIS), label=str(reg), alpha=0.8)
        lo, hi = min(g["_p"].min(), g["_r"].min()), max(g["_p"].max(), g["_r"].max())
        pad = max((hi - lo) * 0.05, pd.Timedelta(minutes=15))
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color="#333333", lw=0.8, ls="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        for eje in (ax.xaxis, ax.yaxis):
            loc = mdates.AutoDateLocator()
            eje.set_major_locator(loc)
            eje.set_major_formatter(mdates.ConciseDateFormatter(loc, show_offset=False))
        ax.tick_params(labelsize=6)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(str(p), fontsize=9)
        ax.set_xlabel("fecha del artículo", fontsize=8)
    axes[0].set_ylabel("fecha de rastreo", fontsize=8)
    _leyenda(axes[0], title="regla", title_fontsize=7)
    for ax in axes[len(paneles):]:
        ax.set_visible(False)
    fig.suptitle("CC-NEWS: fecha del artículo frente a fecha de rastreo (sobre la diagonal = retraso)", fontsize=10)
    fig.tight_layout()
    return fig, axes


def filas_anio_datasets(hf: pd.DataFrame, ax=None):
    """Filas por año de cada dataset de Hugging Face (barras agrupadas, escala log). ``hf_filas_anio``."""
    fig, ax = _fig_ax(ax, (10, 3.5))
    if hf is None or len(hf) == 0:
        _vacio(ax)
        return fig, ax
    m = hf.pivot_table(index="anio", columns="dataset", values="filas", aggfunc="sum").sort_index()
    k = m.shape[1]
    ancho = 0.8 / max(k, 1)
    paleta = [C_BASE, "#c98a2b", "#3a9e5c", "#8e5cc2", "#b5341f", "#5b8fb0"]
    for j, ds in enumerate(m.columns):
        ax.bar(np.arange(len(m)) + (j - (k - 1) / 2) * ancho, m[ds].values, width=ancho,
               color=paleta[j % len(paleta)], label=ds)
    ax.set_xticks(range(len(m)))
    ax.set_xticklabels(m.index.astype(int), fontsize=7, rotation=45)
    ax.set_yscale("log")
    ax.set_ylabel("filas (escala log)")
    _leyenda(ax, ncol=2)
    ax.set_title("Datasets de Hugging Face: filas por año")
    return fig, ax


def gdelt_tono_temas(tono: pd.DataFrame, temas: pd.DataFrame | None = None, *, grupo: str = "economia", top: int = 12):
    """GDELT GKG: tono medio por hora UTC en cada día testigo (líneas) y temas más citados (barras, color = tono).

    ``tono``: ``gdelt_tono_intradia``; ``temas``: ``gdelt_temas`` (se filtra ``grupo`` y se suma ``n`` entre días).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1.2, 1]))
    if tono is not None and len(tono):
        for clave, g in tono.groupby("clave", sort=False):
            g = g.assign(_h=_num(g["hora_utc"]).values, _t=_num(g["tono_medio"]).values).sort_values("_h")
            axes[0].plot(g["_h"], g["_t"], marker="o", ms=3, lw=1, label=str(clave))
        axes[0].axhline(0, color="#333333", lw=0.6)
        axes[0].set_xlabel("hora UTC")
        axes[0].set_ylabel("tono medio GKG")
        axes[0].set_xlim(-0.5, 23.5)
        _leyenda(axes[0])
        axes[0].set_title("Tono intradía en los días testigo", fontsize=10)
    else:
        _vacio(axes[0])
    if temas is not None and len(temas):
        t = temas[temas["grupo"] == grupo] if "grupo" in temas else temas
        if t.empty:
            t = temas
        a = t.groupby("tema").agg(n=("n", "sum"), tono=("tono_medio", "mean")).nlargest(top, "n")
        lim = max(float(np.nanmax(np.abs(a["tono"]))) if a["tono"].notna().any() else 1.0, 0.1)
        cols = [plt.get_cmap("RdBu")((x / lim + 1) / 2) if pd.notna(x) else C_GRIS for x in a["tono"]]
        axes[1].barh(a.index, a["n"], color=cols)
        axes[1].invert_yaxis()
        axes[1].tick_params(axis="y", labelsize=6.5)
        axes[1].set_xlabel("artículos que citan el tema")
        axes[1].set_title(f"Temas más citados ({grupo}; rojo = tono negativo)", fontsize=10)
    else:
        _vacio(axes[1])
    fig.tight_layout()
    return fig, axes


def gdelt_desfase_eventos(ev: pd.DataFrame, ax=None):
    """Desfase SQLDATE → DATEADDED de GDELT Events.

    Con una fila por evento (columna ``lag_dias``) dibuja el histograma; con el resumen ``gdelt_eventos_lag``
    (mediana, p90 y máximo por día testigo) dibuja barras agrupadas. Escala symlog: hay ceros.
    """
    fig, ax = _fig_ax(ax, (8, 3.4))
    if ev is None or len(ev) == 0:
        _vacio(ax)
        return fig, ax
    if "lag_dias" in ev:
        lag = _num(ev["lag_dias"]).dropna().astype(float)
        bordes = np.unique(np.r_[-0.5, 0.5, np.logspace(0, np.log10(max(lag.max(), 1) + 1), 30)])
        ax.hist(lag, bins=bordes, color=C_BASE)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("log")
        ax.set_xlabel("días entre el hecho (SQLDATE) y su alta (DATEADDED), symlog")
        ax.set_ylabel("eventos (log)")
        ax.set_title(f"GDELT Events: desfase SQLDATE → DATEADDED (n = {len(lag)}; "
                     f"{100 * (lag > 0).mean():.1f} % con retraso)")
        return fig, ax
    cols = [c for c in ("lag_dias_mediana", "lag_dias_p90", "lag_dias_max") if c in ev]
    version = ev["version"].astype(str) if "version" in ev else pd.Series("", index=ev.index)
    etiquetas = (ev["clave"].astype(str) + "\n" + version).values
    ancho = 0.8 / max(len(cols), 1)
    for j, (c, col) in enumerate(zip(cols, [C_BASE, "#c98a2b", C_CRISIS])):
        ax.bar(np.arange(len(ev)) + (j - (len(cols) - 1) / 2) * ancho, _num(ev[c]).fillna(0), width=ancho,
               color=col, label=c.replace("lag_dias_", ""))
    ax.set_xticks(range(len(ev)))
    ax.set_xticklabels(etiquetas, fontsize=7)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("días (symlog)")
    _leyenda(ax)
    ax.set_title("GDELT Events: desfase SQLDATE → DATEADDED por día testigo")
    return fig, ax


def wikipedia_hechos_visitas(ce: pd.DataFrame, pageviews, picos: pd.DataFrame | None = None):
    """Wikipedia: hechos de Current Events por día testigo (al cierre del día y hoy) y visitas con picos anotados.

    ``ce``: ``wiki_current_events``; ``pageviews``: serie larga ``wiki_pageviews`` o ``{artículo: Series}``;
    ``picos``: ``wiki_pageviews_picos``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1, 1.8]))
    if ce is not None and len(ce):
        x = np.arange(len(ce))
        axes[0].bar(x - 0.2, _num(ce["n_hechos"]).fillna(0), width=0.4, color=C_BASE, label="al cierre del día")
        if "n_hechos_actual" in ce:
            axes[0].bar(x + 0.2, _num(ce["n_hechos_actual"]).fillna(0), width=0.4, color="#c98a2b",
                        label="versión actual")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(ce["clave"], fontsize=7, rotation=30)
        axes[0].set_ylabel("hechos")
        _leyenda(axes[0])
        axes[0].set_title("Current Events: hechos por día testigo", fontsize=10)
    else:
        _vacio(axes[0])
    series = {k: v.dropna() for k, v in _a_series(pageviews).items()}
    series = {k: v for k, v in series.items() if len(v)}
    if series:
        for nombre, s in series.items():
            axes[1].plot(s.index, s.clip(lower=1).values, lw=0.6, label=str(nombre).split(":")[-1])
        axes[1].set_yscale("log")
        if picos is not None and len(picos):
            for r in picos.itertuples():
                f = pd.Timestamp(r.fecha_pico)
                axes[1].plot(f, r.visitas_pico, "v", color=C_CRISIS, ms=4)
                axes[1].annotate(f"{r.articulo}\n{f:%Y-%m-%d}", (f, r.visitas_pico), fontsize=6,
                                 xytext=(4, -2), textcoords="offset points", va="top")
        sombrear_crisis(axes[1], desde=min(s.index.min() for s in series.values()))
        axes[1].set_ylabel("visitas/día (log)")
        _leyenda(axes[1], ncol=2, loc="upper left")
        axes[1].set_title("Visitas diarias a artículos de estrés (picos anotados)", fontsize=10)
    else:
        _vacio(axes[1])
    fig.tight_layout()
    return fig, axes


def rss_feeds(feeds: pd.DataFrame):
    """RSS: feeds permitidos / bloqueados por robots / con fallo por país y profundidad del feed (horas, log)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    if feeds is None or len(feeds) == 0:
        _vacio(axes[0])
        _vacio(axes[1])
        return fig, axes
    f = feeds.copy()
    f["pais"] = (f["pais"] if "pais" in f else pd.Series("?", index=f.index)).fillna("?").replace("", "?")
    robots = _bool(f["robots_ok"]) if "robots_ok" in f else pd.Series(True, index=f.index)
    f["clase"] = np.where(~robots.values, "bloqueado (robots)",
                          np.where(f["estado"].astype(str).values == "ok", "permitido", "fallo"))
    t = f.groupby(["pais", "clase"]).size().unstack(fill_value=0)
    abajo = np.zeros(len(t))
    for clase, col in (("permitido", C_ESTADO["ok"]), ("bloqueado (robots)", C_ESTADO["parcial"]),
                       ("fallo", C_ESTADO["fallo"])):
        if clase in t:
            axes[0].bar(t.index.astype(str), t[clase], bottom=abajo, color=col, label=clase)
            abajo += t[clase].values
    axes[0].set_ylabel("feeds")
    axes[0].tick_params(axis="x", labelsize=7)
    _leyenda(axes[0])
    axes[0].set_title("Feeds por país", fontsize=10)
    prof = f.assign(_p=_num(f["profundidad_h"]).values if "profundidad_h" in f else np.nan).dropna(subset=["_p"])
    prof = prof[prof["_p"] > 0]
    if len(prof):
        paises = list(dict.fromkeys(prof["pais"]))
        y = prof["pais"].map({p: i for i, p in enumerate(paises)}).values \
            + np.random.default_rng(0).uniform(-.15, .15, len(prof))
        fu = prof["fuente"].astype(str) if "fuente" in prof else pd.Series("rss", index=prof.index)
        paleta = dict(zip(dict.fromkeys(fu), [C_BASE, "#c98a2b", "#3a9e5c", "#8e5cc2", C_GRIS]))
        for nombre, col in paleta.items():
            m = (fu == nombre).values
            axes[1].scatter(prof["_p"].values[m], y[m], s=18, color=col, label=nombre, alpha=0.85)
        axes[1].set_xscale("log")
        axes[1].set_yticks(range(len(paises)))
        axes[1].set_yticklabels(paises, fontsize=7)
        for h, txt in ((24, "1 día"), (24 * 7, "1 semana"), (24 * 30, "1 mes")):
            axes[1].axvline(h, color=C_GRIS, ls=":", lw=0.8)
            axes[1].text(h, 1.0, txt, transform=axes[1].get_xaxis_transform(), fontsize=6, ha="center", va="bottom")
        axes[1].set_xlabel("profundidad: horas hasta el ítem más antiguo (log)")
        _leyenda(axes[1])
        axes[1].set_title("Profundidad por feed", fontsize=10, pad=12)
    else:
        _vacio(axes[1])
    fig.tight_layout()
    return fig, axes


def embudo_duplicados(emb: pd.DataFrame, ax=None, *, titulo: str = "Embudo de duplicados"):
    """Barras horizontales decrecientes: documentos que quedan en cada etapa (``ccnews_embudo``)."""
    fig, ax = _fig_ax(ax, (7, 2.8))
    if emb is None or len(emb) == 0:
        _vacio(ax)
        return fig, ax
    e = emb.sort_values("orden") if "orden" in emb else emb
    n = _num(e["n"]).fillna(0).values
    ax.barh(e["etapa"].astype(str), n, color=[plt.get_cmap("Blues")(max(0.85 - 0.12 * i, 0.2)) for i in range(len(e))])
    ax.invert_yaxis()
    pct = _num(e["pct_restante"]).values if "pct_restante" in e else 100 * n / max(n[0], 1)
    for i, (v, p) in enumerate(zip(n, pct)):
        ax.text(v, i, f" {int(v)} ({float(p):.0f} %)", va="center", fontsize=7)
    ax.set_xlim(0, max(n.max(), 1) * 1.25)
    ax.set_xlabel("documentos")
    ax.set_title(titulo)
    return fig, ax


def congreso_ptr(anio: pd.DataFrame, ops: pd.DataFrame | None = None, *, plazo_dias: int = 45):
    """Congreso (House PTR): declaraciones por año (electrónicas frente a escaneadas) y lag operación → publicación.

    ``anio``: ``congreso_ptr_anio``; ``ops``: ``congreso_operaciones`` (``lag_dias``). Línea en el plazo legal.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    if anio is not None and len(anio):
        a = anio.sort_values("anio")
        ele, esc = _num(a["n_electronicos"]).fillna(0).values, _num(a["n_escaneados"]).fillna(0).values
        axes[0].bar(a["anio"].astype(int), ele, color=C_BASE, label="electrónicos")
        axes[0].bar(a["anio"].astype(int), esc, bottom=ele, color=C_GRIS, label="escaneados (PDF imagen)")
        axes[0].set_ylabel("PTR presentados")
        axes[0].tick_params(axis="x", labelsize=7, rotation=45)
        _leyenda(axes[0])
        axes[0].set_title("PTR por año", fontsize=10)
    else:
        _vacio(axes[0])
    lag = _num(ops["lag_dias"]).dropna().astype(float) if ops is not None and "lag_dias" in ops else \
        pd.Series(dtype=float)
    if len(lag):
        tope = max(float(lag.quantile(0.99)), plazo_dias + 10)
        axes[1].hist(lag.clip(upper=tope), bins=40, color=C_BASE)
        axes[1].axvline(plazo_dias, color=C_CRISIS, ls="--", lw=1, label=f"plazo legal ({plazo_dias} d)")
        axes[1].set_xlabel("días de la operación a la publicación (cola recortada en p99)")
        axes[1].set_ylabel("operaciones")
        _leyenda(axes[1])
        axes[1].set_title(f"Lag operación → publicación (n = {len(lag)}; mediana {lag.median():.0f} d; "
                          f"{100 * (lag > plazo_dias).mean():.0f} % fuera de plazo)", fontsize=9)
    else:
        _vacio(axes[1])
    fig.tight_layout()
    return fig, axes


def form4_lag_codigos(ops: pd.DataFrame, *, plazo_dias: int = 2):
    """Form 4 (insiders): lag transacción → publicación (días) y códigos de transacción. ``insiders_operaciones``."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    if ops is None or len(ops) == 0:
        _vacio(axes[0])
        _vacio(axes[1])
        return fig, axes
    lag = _num(ops["lag_dias"]).dropna().astype(float) if "lag_dias" in ops else pd.Series(dtype=float)
    if len(lag):
        tope = max(float(lag.quantile(0.99)), plazo_dias + 5)
        axes[0].hist(lag.clip(upper=tope), bins=np.arange(-0.5, tope + 1.5, 1), color=C_BASE)
        axes[0].axvline(plazo_dias + 0.5, color=C_CRISIS, ls="--", lw=1, label=f"plazo legal ({plazo_dias} días hábiles)")
        axes[0].axvspan(plazo_dias + 0.5, plazo_dias + 2.5, color=C_CRISIS, alpha=0.08, lw=0,
                        label="mismo plazo con fin de semana")
        axes[0].set_xlabel("días de la transacción a la publicación (cola recortada en p99)")
        axes[0].set_ylabel("transacciones")
        _leyenda(axes[0])
        axes[0].set_title(f"Lag del Form 4 (n = {len(lag)}; mediana {lag.median():.0f} d)", fontsize=10)
    else:
        _vacio(axes[0])
    if "codigo" in ops:
        cod = ops["codigo"].fillna("?").astype(str).value_counts()
        axes[1].bar(cod.index, cod.values,
                    color=["#3a9e5c" if c == "P" else (C_CRISIS if c == "S" else C_BASE) for c in cod.index])
        axes[1].set_ylabel("transacciones")
        axes[1].set_xlabel("código (P compra, S venta, A concesión, M ejercicio...)")
        axes[1].set_title("Códigos de transacción", fontsize=10)
    else:
        _vacio(axes[1])
    fig.tight_layout()
    return fig, axes


def prediccion_liquidez(pm: pd.DataFrame | None, ka: pd.DataFrame | None):
    """Liquidez por año: Polymarket (USD) y Kalshi (contratos), escala log; rayado = muestra truncada."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, df, col, uni, nombre, color in (
        (axes[0], pm, "volumen_usd", "USD", "Polymarket", "#8e5cc2"),
        (axes[1], ka, "volumen_contratos", "contratos", "Kalshi", "#5b8fb0"),
    ):
        if df is None or len(df) == 0 or col not in df:
            _vacio(ax)
            continue
        d = df.sort_values("anio")
        trunc = _bool(d["truncado"]) if "truncado" in d else pd.Series(False, index=d.index)
        ax.bar(d["anio"].astype(int), _num(d[col]).fillna(0).clip(lower=1), color=color,
               hatch=["//" if t else "" for t in trunc], edgecolor="white")
        ax.set_yscale("log")
        ax.set_ylabel(f"volumen ({uni}, log)")
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(f"{nombre}: volumen por año" + (" (rayado: truncado)" if trunc.any() else ""), fontsize=10)
    fig.tight_layout()
    return fig, axes


def serie_con_saltos(serie, saltos: pd.DataFrame | None = None, *, nombre: str | None = None, ax=None):
    """Probabilidad de un mercado (precio del «Yes») con los saltos marcados (``polymarket_saltos``)."""
    series = _a_series(serie)
    if nombre is None and series:
        nombre = next(iter(series))
    fig, ax = _fig_ax(ax, (10, 3.2))
    s = series.get(nombre, pd.Series(dtype=float)).dropna() if series else pd.Series(dtype=float)
    if s.empty:
        _vacio(ax)
        return fig, ax
    ax.plot(s.index, s.values, color="#8e5cc2", lw=1)
    if saltos is not None and len(saltos):
        sa = saltos[saltos["serie"] == nombre] if "serie" in saltos else saltos
        for r in sa.itertuples():
            f = pd.Timestamp(r.fecha)
            sube = float(r.cambio_pp) > 0
            ax.plot(f, r.a, "^" if sube else "v", color="#3a9e5c" if sube else C_CRISIS, ms=7)
            ax.annotate(f"{float(r.cambio_pp):+.0f} pp\n{f:%d-%m-%Y}", (f, r.a), fontsize=6.5,
                        xytext=(5, 6 if sube else -6), textcoords="offset points", va="bottom" if sube else "top")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("probabilidad implícita")
    ax.set_title(f"{nombre}: probabilidad diaria y saltos marcados")
    return fig, ax


def indices_paneles(series, *, ncol: int = 2, desde=None, log: Iterable[str] = (), testigos: bool = True,
                    max_puntos: int | None = 2500):
    """Índices (GPR, EPU, SKEW, put/call, short ratio, VIX) en pequeños múltiplos con crisis y días testigo.

    ``series``: ``{nombre: Series}`` o tabla larga; ``log``: nombres con eje en escala log. Las series con más de
    ``max_puntos`` observaciones se dibujan como media semanal (figura más ligera; None = sin remuestrear).
    """
    s = _a_series(series)
    if max_puntos is not None:
        s = {k: (v.dropna().resample("W").mean() if len(v.dropna()) > max_puntos else v) for k, v in s.items()}
    fig, axes = paneles_series(s, ncol=ncol, desde=desde, alto=2.0,
                               titulo="Índices alternativos con las crisis de la Fase 1 sombreadas")
    log = set(log)
    for ax, nombre in zip(axes.ravel(), s):
        if nombre in log:
            ax.set_yscale("log")
        if testigos:
            lim = ax.get_xlim()
            marcar_testigos(ax, con_texto=False)
            ax.set_xlim(lim)
    return fig, axes


def ews_z(tz: pd.DataFrame, *, umbrales: Iterable[float] = (2.35,), ax=None):
    """EWS: z publicado (referencia con datos posteriores) frente a z causal, con los umbrales de alarma."""
    fig, ax = _fig_ax(ax, (11, 3.2))
    if tz is None or len(tz) == 0:
        _vacio(ax)
        return fig, ax
    t = _fechas(tz["ts"]).values
    ax.plot(t, _num(tz["z_publicado"]).values, lw=0.5, color=C_BASE, label="z publicado")
    ax.plot(t, _num(tz["z_causal"]).values, lw=0.5, color="#c98a2b", alpha=0.85, label="z causal (solo pasado)")
    for u in umbrales:
        ax.axhline(u, color=C_CRISIS, ls="--", lw=0.8)
        ax.text(1.0, u, f" {u:g}", transform=ax.get_yaxis_transform(), fontsize=7, color=C_CRISIS, va="center")
    ax.set_ylabel("z")
    _leyenda(ax, loc="upper left")
    ax.set_title("EWS: z publicado frente a z causal")
    return fig, ax


def volumen_fuentes(df: pd.DataFrame, *, col_fuente: str = "fuente", col_valor: str = "n",
                    col_color: str | None = "familia", unidad: str = "documentos o filas", ax=None):
    """Volumen por fuente (barras horizontales, escala log), color por familia si está la columna."""
    d = df.assign(_v=_num(df[col_valor]).values).dropna(subset=["_v"])
    d = d[d["_v"] > 0].sort_values("_v")
    fig, ax = _fig_ax(ax, (8, max(3.0, 0.2 * len(d) + 1)))
    if d.empty:
        _vacio(ax)
        return fig, ax
    con_color = bool(col_color) and col_color in d
    cols = [C_FAMILIA.get(x, C_GRIS) for x in d[col_color]] if con_color else C_BASE
    ax.barh(d[col_fuente].astype(str), d["_v"], color=cols)
    ax.set_xscale("log")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel(f"{unidad} (escala log)")
    if con_color:
        for fam in dict.fromkeys(d[col_color]):
            ax.plot([], [], "s", color=C_FAMILIA.get(fam, C_GRIS), label=fam)
        _leyenda(ax, loc="lower right")
    ax.set_title("Volumen por fuente")
    return fig, ax


def lags_publicacion(tg: pd.DataFrame, cob: pd.DataFrame | None = None, *, ax=None):
    """Lag hecho → publicación por fuente (horas, symlog): cajas y puntos con los ejemplos de ``timegate`` y
    rombos con ``lag_mediana_h`` de ``cobertura`` para las fuentes sin ejemplos."""
    if tg is not None and len(tg):
        d = tg.assign(_l=_num(tg["lag_h"]).values).dropna(subset=["_l"])
    else:
        d = pd.DataFrame({"fuente": pd.Series(dtype=str), "_l": pd.Series(dtype=float)})
    fuentes = list(d.groupby("fuente")["_l"].median().sort_values().index)
    extra = pd.Series(dtype=float)
    if cob is not None and len(cob) and "lag_mediana_h" in cob:
        extra = cob.assign(_l=_num(cob["lag_mediana_h"]).values).dropna(subset=["_l"]).set_index("fuente")["_l"]
        extra = extra[~extra.index.isin(fuentes)].sort_values()
    todas = fuentes + list(extra.index)
    fig, ax = _fig_ax(ax, (9, max(3.0, 0.25 * len(todas) + 1)))
    if not todas:
        _vacio(ax)
        return fig, ax
    if fuentes:
        ax.boxplot([d.loc[d["fuente"] == f, "_l"].values for f in fuentes], positions=range(len(fuentes)),
                   orientation="horizontal", widths=0.6, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#dce6f0", lw=0.6), medianprops=dict(color=C_BASE))
        for i, f in enumerate(fuentes):
            v = d.loc[d["fuente"] == f, "_l"].values
            ax.scatter(v, i + np.random.default_rng(i).uniform(-.18, .18, len(v)), s=6, color=C_BASE, alpha=0.6)
    if len(extra):
        ax.scatter(extra.values, range(len(fuentes), len(todas)), marker="D", s=22, color="#c98a2b",
                   label="mediana declarada (cobertura)")
    ax.set_yticks(range(len(todas)))
    ax.set_yticklabels(todas, fontsize=7)
    ax.set_ylim(len(todas) - 0.5, -0.5)
    ax.set_xscale("symlog", linthresh=1)
    valores = np.r_[d["_l"].values.astype(float), extra.values.astype(float)]
    vmin, vmax = float(np.nanmin(valores)), float(np.nanmax(valores))
    ax.set_xlim(vmin * 2 if vmin < 0 else -0.5, max(vmax * 2, 24 * 60))
    for h, txt in ((1, "1 h"), (24, "1 día"), (24 * 7, "1 sem."), (24 * 45, "45 días")):
        ax.axvline(h, color=C_GRIS, ls=":", lw=0.7)
        ax.text(h, 1.0, txt, transform=ax.get_xaxis_transform(), fontsize=6, ha="center", va="bottom")
    ax.axvline(0, color="#333333", lw=0.6)
    ax.set_xlabel("horas del hecho a la publicación (symlog; < 0 = antes del día de referencia)")
    _leyenda(ax, loc="lower right")
    ax.set_title("Lag de publicación por fuente", pad=12)
    return fig, ax


# =========================================================================== t4_02 · validez

CRITERIOS_CALIDAD = ("pct_huecos", "pct_atipicos", "n_rupturas", "pct_repetidos", "racha_max_constante",
                     "lag_pub_mediana_dias", "n_pub_antes_ref", "revisable")
CRITERIOS_SEMAFORO = ("historia", "calidad", "lookahead", "senal", "acceso")
ORDEN_VEREDICTO = ("apta_backtest", "solo_vivo", "contexto", "descartar")


def _col(df: pd.DataFrame, *candidatas: str) -> str | None:
    """Primera columna de ``candidatas`` presente en ``df`` (None si ninguna)."""
    return next((c for c in candidatas if c is not None and c in df.columns), None)


def _a_numero(m: pd.DataFrame) -> pd.DataFrame:
    """Matriz a números (booleanos como 0/1); lo que no es número queda NaN."""
    return m.apply(lambda c: pd.to_numeric(c.map(lambda x: float(x) if isinstance(x, (bool, np.bool_)) else x),
                                           errors="coerce"))


def _rejilla(ax, m: pd.DataFrame, colores, textos=None, *, fs: float = 6.5, rot: int = 30) -> None:
    """Pinta una matriz de celdas coloreadas (``colores[i][j]``) con texto opcional y ejes etiquetados."""
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color=colores[i][j], lw=0))
            if textos is not None:
                ax.text(j, i, textos[i][j], ha="center", va="center", fontsize=fs)
    ax.set_xlim(-.5, m.shape[1] - .5)
    ax.set_ylim(m.shape[0] - .5, -.5)
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels([str(c).replace("_", " ") for c in m.columns], fontsize=7, rotation=rot,
                       ha="right" if rot else "center")
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels(m.index.astype(str), fontsize=6.5)
    ax.grid(False)


def calor_calidad(df: pd.DataFrame, *, fila: str = "serie", columna: str | None = None, valor: str | None = None,
                  criterios: Iterable[str] | None = CRITERIOS_CALIDAD, mayor_es_mejor: bool = False,
                  por_columna: bool = True, vmin: float | None = None, vmax: float | None = None,
                  titulo: str = "Calidad por serie y criterio (verde = mejor)", ax=None):
    """Mapa de calor serie × criterio de calidad.

    ``df``: ``validez_calidad`` (ancha; se usan los ``criterios`` presentes) o larga (``fila, columna, valor``).
    Números: verde = mejor (``mayor_es_mejor=False``: huecos, atípicos, rupturas, lag... cuanto menos, mejor),
    escalados por columna si ``por_columna``. Textos (ok/parcial/fallo...): colores de semáforo.
    """
    m = _matriz(df, fila, columna, valor)
    if columna is None or columna not in df.columns:
        crit = [c for c in (criterios or []) if c in m.columns]
        if crit:
            m = m[crit]
    fig, ax = _fig_ax(ax, (min(max(5.0, 0.8 * m.shape[1] + 3), 12), max(2.5, 0.24 * m.shape[0] + 1.4)))
    if m.empty:
        _vacio(ax)
        return fig, ax
    nums = _a_numero(m)
    es_num = nums.notna().sum().sum() >= 0.9 * m.notna().sum().sum()
    cmap = _CMAP_SEMAFORO
    colores, textos = [], []
    for i in range(m.shape[0]):
        fc, ft = [], []
        for j in range(m.shape[1]):
            if es_num:
                v = nums.iat[i, j]
                col = nums.iloc[:, j]
                lo = vmin if vmin is not None else float(col.min() if por_columna else np.nanmin(nums.values))
                hi = vmax if vmax is not None else float(col.max() if por_columna else np.nanmax(nums.values))
                if pd.isna(v):
                    fc.append("#f2f2f2")
                else:
                    x = (float(v) - lo) / (hi - lo) if hi > lo else 0.0
                    fc.append(plt.matplotlib.colors.to_hex(cmap(min(max(x if mayor_es_mejor else 1 - x, 0), 1))))
                orig = m.iat[i, j]
                ft.append(("sí" if orig else "no") if isinstance(orig, (bool, np.bool_)) else _texto_celda(v, 8))
            else:
                v = m.iat[i, j]
                fc.append(color_valor(v))
                ft.append(_texto_celda(v, 8))
        colores.append(fc)
        textos.append(ft)
    _rejilla(ax, m, colores, textos, fs=6)
    ax.set_title(titulo)
    return fig, ax


def lookahead_semaforo(df: pd.DataFrame, *, fila: str = "fuente", columna: str = "comprobacion",
                       valor: str = "resultado", solo_con_riesgo: bool = False, ax=None):
    """Comprobaciones de look-ahead como semáforo: fuente × comprobación coloreada por resultado
    (``validez_lookahead``: ok verde, riesgo ámbar, fallo rojo; gris si la comprobación no aplica)."""
    d = df
    if solo_con_riesgo:
        malas = df.loc[df[valor].astype(str).str.lower() != "ok", fila].unique()
        d = df[df[fila].isin(malas)]
    m = d.pivot_table(index=fila, columns=columna, values=valor, aggfunc="first", sort=False)
    fig, ax = _fig_ax(ax, (min(max(6.0, 0.55 * m.shape[1] + 3), 12), max(2.5, 0.17 * m.shape[0] + 2.0)))
    if m.empty:
        _vacio(ax)
        return fig, ax
    colores = [[color_valor(v) if pd.notna(v) else "#f7f7f7" for v in fila_m] for fila_m in m.values]
    _rejilla(ax, m, colores, rot=60)
    ax.tick_params(axis="y", labelsize=6)
    for v, col in (("ok", C_ESTADO["ok"]), ("riesgo", C_ESTADO["parcial"]), ("fallo", C_ESTADO["fallo"]),
                   ("no aplica", "#f7f7f7")):
        ax.plot([], [], "s", color=col, mec="#bbbbbb", label=v)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    n_riesgo = int((d[valor].astype(str).str.lower() != "ok").sum())
    ax.set_title(f"Comprobaciones de look-ahead ({len(d)} comprobaciones; {n_riesgo} con riesgo)")
    return fig, ax


def tabla_semaforo(df: pd.DataFrame, *, columnas: list[str] | None = None, titulo: str = "Comprobación de look-ahead",
                   colorear: Iterable[str] | None = None, invertir: Iterable[str] = (), ancho: int = 26, ax=None):
    """Tabla como figura. Las celdas de texto o booleanas de ``colorear`` (por defecto todas las columnas) se
    pintan con ``color_valor``; los números y los textos no reconocidos quedan en blanco. En las columnas de
    ``invertir`` (p. ej. «revisable») «sí»/True es malo: verde y rojo se intercambian."""
    d = df[list(columnas)] if columnas else df
    colorear = set(colorear if colorear is not None else d.columns)
    invertir = set(invertir)
    cambio = {C_ESTADO["ok"]: C_ESTADO["fallo"], C_ESTADO["fallo"]: C_ESTADO["ok"]}
    fig, ax = _fig_ax(ax, (min(1.6 * d.shape[1] + 1, 12), 0.28 * len(d) + 0.9))
    ax.axis("off")
    if d.empty:
        _vacio(ax)
        return fig, ax

    def _color(c, v):
        if c not in colorear:
            return "#ffffff"
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)):
            return "#ffffff"
        col = color_valor(v)
        if c in invertir:
            col = cambio.get(col, col)
        return "#ffffff" if col == "#f2f2f2" else col

    filas = list(d.itertuples(index=False))
    texto = [[_texto_celda(v, ancho) for v in f] for f in filas]
    colores = [[_color(c, v) for c, v in zip(d.columns, f)] for f in filas]
    t = ax.table(cellText=texto, colLabels=[str(c) for c in d.columns], cellColours=colores, loc="center",
                 cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(7)
    t.scale(1, 1.25)
    for (i, _), celda in t.get_celld().items():
        celda.set_linewidth(0.3)
        if i == 0:
            celda.set_facecolor("#e6e6e6")
            celda.set_text_props(fontweight="bold")
    ax.set_title(titulo)
    return fig, ax


def estudio_eventos(df: pd.DataFrame, *, panel: str = "serie", x: str = "dia_rel", media: str = "media",
                    ic_inf: str = "lo", ic_sup: str = "hi", grupo: str | None = None, series: Iterable[str] | None = None,
                    ncol: int = 5, etiqueta_y: str = "z medio", nivel_ic: float = 0.90):
    """Estudio de eventos en pequeños múltiplos (``validez_eventos``): media ± IC alrededor del pico del mercado
    (día 0, inicio de la ventana pico -> suelo), línea vertical en el día 0 y horizontal en 0. ``series`` limita
    los paneles; ``nivel_ic`` es el nivel del intervalo de la tabla (``validez.NIVEL_IC``), solo para el título."""
    ic_inf = _col(df, ic_inf, "ic_inf", "lo")
    ic_sup = _col(df, ic_sup, "ic_sup", "hi")
    paneles = list(dict.fromkeys(df[panel])) if len(df) else []
    if series is not None:
        paneles = [p for p in series if p in set(paneles)]
    n = max(len(paneles), 1)
    ncol = min(ncol, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(min(2.6 * ncol, 13), 1.9 * nrow + 0.6), squeeze=False,
                             sharex=True)
    if not paneles:
        _vacio(axes[0, 0])
    paleta = [C_BASE, "#c98a2b", "#3a9e5c", "#8e5cc2"]
    for ax, p in zip(axes.ravel(), paneles):
        g = df[df[panel] == p]
        grupos = list(g.groupby(grupo, sort=False)) if grupo and grupo in g else [(None, g)]
        for k, (nom, h) in enumerate(grupos):
            h = h.sort_values(x)
            col = paleta[k % len(paleta)]
            ax.plot(h[x], _num(h[media]).values, color=col, lw=0.9, label=nom)
            if ic_inf and ic_sup:
                ax.fill_between(h[x], _num(h[ic_inf]).values, _num(h[ic_sup]).values, color=col, alpha=0.2, lw=0)
        ax.axvline(0, color=C_CRISIS, lw=0.8)
        ax.axhline(0, color="#333333", lw=0.5)
        ax.set_title(str(p), fontsize=7.5)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(paneles):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("días hábiles respecto al pico", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel(etiqueta_y, fontsize=7)
    if grupo:
        _leyenda(axes[0, 0], fontsize=6.5)
    fig.suptitle(f"Estudio de eventos: media ± IC {nivel_ic:.0%} alrededor del pico del mercado (día 0)", fontsize=10)
    fig.tight_layout()
    return fig, axes


def auc_vs_vix(df: pd.DataFrame, *, serie: str = "serie", adelanto: str = "adelanto", auc: str = "auc",
               ref_col: str = "auc_vix_mismo_periodo", ref: str = "vix_cierre", adelantos: Iterable[int] | None = None):
    """AUC de cada serie para anticipar crisis, un panel por adelanto (0/20/60 días), con la referencia del VIX y
    la línea de 0,5 (azar).

    ``validez_auc`` trae ``auc_vix_mismo_periodo`` (AUC del VIX en el periodo de cada serie): se marca con una
    raya roja en cada barra. Si no está, la referencia es la fila ``serie == ref`` (o la que contenga «vix»).
    Barra verde si supera a su VIX (o a 0,5 si no hay VIX). Con ``ic_inf``/``ic_sup`` dibuja el intervalo.
    """
    d = df.assign(_a=_num(df[auc]).values).dropna(subset=["_a"])
    ads = list(adelantos) if adelantos is not None else sorted(d[adelanto].dropna().unique())
    es_ref = d[serie] == ref
    if not es_ref.any():
        es_ref = d[serie].astype(str).str.lower().str.contains("vix")
    con_ref_col = ref_col in d.columns
    n_series = d.loc[~es_ref, serie].nunique()
    k = max(len(ads), 1)
    fig, axes = plt.subplots(1, k, figsize=(min(4 * k, 12), max(2.8, 0.2 * n_series + 1.4)), squeeze=False)
    axes = axes.ravel()
    if not ads:
        _vacio(axes[0])
    for ax, a in zip(axes, ads):
        sel = d[adelanto] == a
        r = d[sel & es_ref]
        g = d[sel & ~es_ref].sort_values("_a")
        vix = _num(g[ref_col]).values if con_ref_col else np.full(len(g), float(r["_a"].iloc[0]) if len(r) else np.nan)
        umbral = np.where(np.isnan(vix), 0.5, vix)
        ax.barh(g[serie].astype(str), g["_a"], color=[C_ESTADO["ok"] if v > u else C_GRIS
                                                      for v, u in zip(g["_a"], umbral)])
        if "ic_inf" in g and "ic_sup" in g:
            ax.errorbar(g["_a"], range(len(g)), xerr=[(g["_a"] - _num(g["ic_inf"])).clip(lower=0),
                                                      (_num(g["ic_sup"]) - g["_a"]).clip(lower=0)],
                        fmt="none", ecolor="#333333", lw=0.6)
        ax.axvline(0.5, color="#333333", ls=":", lw=0.8, label="azar (0,5)")
        if con_ref_col:
            ax.scatter(vix, range(len(g)), marker="|", s=60, color=C_CRISIS, lw=1.5, label="VIX, mismo periodo")
        elif len(r):
            ax.axvline(float(r["_a"].iloc[0]), color=C_CRISIS, ls="--", lw=1, label=f"VIX ({float(r['_a'].iloc[0]):.2f})")
        ax.set_xlim(0, 1)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_xlabel("AUC")
        ax.set_title(f"adelanto {a} días", fontsize=9)
        _leyenda(ax, loc="lower right", fontsize=6)
    fig.suptitle("Capacidad de anticipar crisis: AUC por serie frente al VIX (verde: supera al VIX)", fontsize=10)
    fig.tight_layout()
    return fig, axes


def calor_lead_lag(df: pd.DataFrame, *, serie: str = "serie", lag: str = "lag", valor: str = "corr",
                   filtro: Mapping[str, object] | None = None, ax=None):
    """Mapa de calor serie × lag (divergente, centrado en 0); punto negro: máximo |valor| de cada serie.

    ``validez_leadlag`` trae varias ``referencia``/``metrica``/``rejilla``: ``filtro`` elige una (p. ej.
    ``{"referencia": "vix", "metrica": "cambio_5d", "rejilla": "referencia"}``); sin filtro se toma la primera
    de cada columna.
    """
    d = df
    sel = dict(filtro or {})
    for c in ("referencia", "metrica", "rejilla"):
        if c in d.columns and c not in sel and d[c].nunique() > 1:
            sel[c] = d[c].iloc[0]
    for c, v in sel.items():
        d = d[d[c] == v]
    m = d.pivot_table(index=serie, columns=lag, values=valor, aggfunc="mean", sort=False)
    m = m.reindex(columns=sorted(m.columns))
    fig, ax = _fig_ax(ax, (min(max(6.0, 0.3 * m.shape[1] + 3), 12), max(2.5, 0.2 * m.shape[0] + 1.4)))
    if m.empty:
        _vacio(ax)
        return fig, ax
    lim = float(np.nanmax(np.abs(m.values))) or 1.0
    im = ax.imshow(m.values, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    for i in range(m.shape[0]):
        fila = np.abs(m.values[i])
        if np.isfinite(fila).any():
            ax.plot(int(np.nanargmax(fila)), i, "o", color="black", ms=2.5)
    cols = list(m.columns)
    paso = 1 if len(cols) <= 25 else 10
    marcas = [j for j, c in enumerate(cols) if float(c) % paso == 0]
    ax.set_xticks(marcas)
    ax.set_xticklabels([f"{cols[j]:g}" if isinstance(cols[j], (int, float, np.number)) else str(cols[j])
                        for j in marcas], fontsize=6.5)
    if 0 in cols:
        ax.axvline(cols.index(0), color="#333333", lw=0.6)
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels(m.index.astype(str), fontsize=6.5)
    ax.set_xlabel("lag en días (> 0: la serie adelanta)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=7)
    extra = ", ".join(f"{k} = {v}" for k, v in sel.items())
    ax.set_title(f"Lead-lag: {valor} por serie y lag" + (f" ({extra})" if extra else ""), fontsize=10)
    return fig, ax


def _orden_periodos(periodos: Iterable[str]) -> list[str]:
    """Décadas en orden y después los cortes antes/desde (``antes_2015`` < ``desde_2015``)."""
    import re

    def clave(p: str):
        n = re.search(r"(\d{4})", p)
        return (0 if "decada" in p else 1, int(n.group(1)) if n else 9999, 0 if "antes" in p else 1, p)

    return sorted(dict.fromkeys(str(p) for p in periodos), key=clave)


def estabilidad_subperiodos(df: pd.DataFrame, *, serie: str = "serie", subperiodo: str = "periodo",
                            valor: str = "auc", referencia: float | None = 0.5, como: str = "auto",
                            orden: Iterable[str] | None = None, ax=None):
    """Métrica por subperiodo (``validez_estabilidad``).

    ``como="lineas"``: puntos unidos por serie (estable si son planas); ``"calor"``: matriz serie × periodo
    coloreada alrededor de ``referencia``. ``"auto"`` elige calor con más de 10 series.
    """
    if df is None or len(df) == 0:
        fig, ax = _fig_ax(ax, (9, 3.6))
        _vacio(ax)
        return fig, ax
    sub = subperiodo if subperiodo in df else _col(df, "subperiodo", "periodo")
    subs = list(orden) if orden is not None else _orden_periodos(df[sub])
    n_series = df[serie].nunique()
    if como == "calor" or (como == "auto" and n_series > 10):
        m = df.pivot_table(index=serie, columns=sub, values=valor, aggfunc="first", sort=False)
        m = m.reindex(columns=[s for s in subs if s in m.columns])
        fig, ax = _fig_ax(ax, (min(max(5.0, 0.9 * m.shape[1] + 3), 12), max(2.5, 0.2 * m.shape[0] + 1.4)))
        ref = referencia if referencia is not None else float(np.nanmean(m.values))
        dist = float(np.nanmax(np.abs(m.values - ref))) or 1.0
        colores = [[color_valor(v, vmin=ref - dist, vmax=ref + dist) for v in f] for f in m.values]
        _rejilla(ax, m, colores, [[_texto_celda(v, 6) for v in f] for f in m.values], fs=5.5)
        ax.set_title(f"Estabilidad por subperiodo ({valor}; amarillo = {ref:g})", fontsize=10)
        return fig, ax
    fig, ax = _fig_ax(ax, (9, 3.6))
    pos = {s: i for i, s in enumerate(subs)}
    for k, (nom, g) in enumerate(df.groupby(serie, sort=False)):
        g = g.assign(_x=g[sub].astype(str).map(pos)).dropna(subset=["_x"]).sort_values("_x")
        ax.plot(g["_x"], _num(g[valor]).values, marker="o", ms=3.5, lw=0.9, label=str(nom),
                color=plt.get_cmap("tab10")(k % 10))
    if referencia is not None:
        ax.axhline(referencia, color="#333333", ls=":", lw=0.8)
    ax.set_xticks(range(len(subs)))
    ax.set_xticklabels([s.replace("_", " ") for s in subs], fontsize=7, rotation=20)
    ax.set_ylabel(valor)
    _leyenda(ax, fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title(f"Estabilidad por subperiodo ({valor})")
    return fig, ax


def fiabilidad_calibracion(bins: pd.DataFrame, resumen: pd.DataFrame | None = None, *,
                           plataforma: str = "plataforma", horizonte: str | None = None, p: str = "p_media",
                           frec: str | None = None, n: str = "n", brier: str = "brier", liquidez: str = "todos"):
    """Diagrama de fiabilidad por plataforma (paneles) y horizonte (líneas); área del punto ∝ nº de mercados.

    ``bins``: ``validez_fiabilidad_bins`` (``horizonte_dias, p_media, frec_obs``) o
    ``prediccion_calibracion_resumen`` (``horizonte, p_media, frec_yes, brier``). ``resumen``:
    ``validez_calibracion`` para el Brier de la leyenda (fila ``liquidez``).
    """
    if bins is None or len(bins) == 0:
        fig, ax = plt.subplots(figsize=(4, 3.8))
        _vacio(ax)
        return fig, np.array([ax])
    hz = horizonte or _col(bins, "horizonte", "horizonte_dias")
    fr = frec or _col(bins, "frec_yes", "frec_obs")
    briers: dict = {}
    if resumen is not None and len(resumen):
        r = resumen[resumen["liquidez"] == liquidez] if "liquidez" in resumen else resumen
        hz_r = _col(r, hz, "horizonte", "horizonte_dias")
        briers = {(a, str(b)): v for a, b, v in zip(r[plataforma], r[hz_r], _num(r[brier]))}
    elif brier in bins:
        briers = {(a, str(b)): v for (a, b), v in bins.groupby([plataforma, hz])[brier].first().items()}
    plats = list(dict.fromkeys(bins[plataforma]))
    k = len(plats)
    fig, axes = plt.subplots(1, k, figsize=(min(4 * k, 12), 3.8), squeeze=False)
    axes = axes.ravel()
    paleta = [C_BASE, "#c98a2b", "#3a9e5c", "#8e5cc2"]
    for ax, pl in zip(axes, plats):
        g = bins[bins[plataforma] == pl]
        for j, (h, t) in enumerate(g.groupby(hz, sort=True)):
            t = t.assign(_p=_num(t[p]).values, _f=_num(t[fr]).values).dropna(subset=["_p", "_f"]).sort_values("_p")
            b = briers.get((pl, str(h)), np.nan)
            col = paleta[j % len(paleta)]
            et = f"{h} d" if isinstance(h, (int, np.integer)) else str(h)
            ax.plot(t["_p"], t["_f"], color=col, lw=0.9, label=f"{et} (Brier {b:.3f})" if pd.notna(b) else et)
            tam = 8 + 4 * (_num(t[n]).fillna(1).clip(upper=60).values if n in t else 1)
            ax.scatter(t["_p"], t["_f"], s=tam, color=col, alpha=0.7)
        ax.plot([0, 1], [0, 1], color="#333333", ls="--", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("probabilidad media del tramo")
        ax.set_title(str(pl), fontsize=10)
        _leyenda(ax, title="horizonte", title_fontsize=7, fontsize=6.5, loc="upper left")
    axes[0].set_ylabel("frecuencia observada del «Yes»")
    fig.suptitle("Calibración de los mercados de predicción", fontsize=10)
    fig.tight_layout()
    return fig, axes


def car_grupos(df: pd.DataFrame, *, grupo: str = "grupo", horizonte: str = "horizonte",
               desde: str = "referencia_fecha", car: str = "car_medio", ic_inf: str = "lo", ic_sup: str = "hi",
               escala: float = 100.0, unidad: str = "%", nivel_ic: float = 0.90):
    """CAR medio con IC por grupo (paneles) y horizonte, contado desde la publicación y desde la operación.

    ``validez_car``: ``grupo, referencia_fecha, horizonte, car_medio, lo, hi`` (en tanto por uno: ``escala``
    = 100 lo pasa a %). Acepta también ``desde, car, ic_inf, ic_sup``. ``nivel_ic``: nivel del intervalo de la
    tabla (``validez.NIVEL_IC``), solo para el título.
    """
    desde = _col(df, desde, "desde", "referencia_fecha")
    car = _col(df, car, "car", "car_medio")
    ic_inf = _col(df, ic_inf, "ic_inf", "lo")
    ic_sup = _col(df, ic_sup, "ic_sup", "hi")
    grupos = list(dict.fromkeys(df[grupo])) if len(df) else []
    k = max(len(grupos), 1)
    fig, axes = plt.subplots(1, k, figsize=(min(3.8 * k, 12), 3.3), squeeze=False, sharey=True)
    axes = axes.ravel()
    if not grupos:
        _vacio(axes[0])
    colores = {"publicacion": C_BASE, "publicación": C_BASE, "operacion": "#c98a2b", "operación": "#c98a2b"}
    for ax, gr in zip(axes, grupos):
        g = df[df[grupo] == gr]
        hs = sorted(g[horizonte].unique())
        pos = {h: i for i, h in enumerate(hs)}
        origenes = list(dict.fromkeys(g[desde])) if desde else [None]
        for j, o in enumerate(origenes):
            t = g[g[desde] == o] if o is not None else g
            x = t[horizonte].map(pos).values + (j - (len(origenes) - 1) / 2) * 0.18
            y = escala * _num(t[car]).values
            err = None
            if ic_inf and ic_sup:
                err = [np.clip(y - escala * _num(t[ic_inf]).values, 0, None),
                       np.clip(escala * _num(t[ic_sup]).values - y, 0, None)]
            nombre = {"publicacion": "publicación", "operacion": "operación"}.get(str(o), str(o))
            ax.errorbar(x, y, yerr=err, fmt="o", ms=4, capsize=2, lw=0.8, color=colores.get(str(o), C_GRIS),
                        label=f"desde la {nombre}" if o is not None else None)
        ax.axhline(0, color="#333333", lw=0.6)
        ax.set_xticks(range(len(hs)))
        ax.set_xticklabels([str(h) for h in hs], fontsize=7)
        ax.set_xlabel("horizonte (días hábiles)")
        ax.set_title(str(gr), fontsize=9)
    axes[0].set_ylabel(f"CAR medio ({unidad})")
    _leyenda(axes[0], fontsize=6.5)
    fig.suptitle(f"Rentabilidad anormal acumulada (media ± IC {nivel_ic:.0%})", fontsize=10)
    fig.tight_layout()
    return fig, axes


def semaforo_final(df: pd.DataFrame, *, fila: str = "candidato", columna: str | None = None, valor: str | None = None,
                   criterios: Iterable[str] | None = CRITERIOS_SEMAFORO,
                   veredicto: str | pd.Series | Mapping | None = "veredicto", ordenar: bool = True, ax=None):
    """Semáforo final: matriz candidato × criterio coloreada y una última columna con el veredicto
    (``apta_backtest``, ``solo_vivo``, ``contexto``, ``descartar``).

    ``df``: ``validez_semaforo`` (ancha; puntos 0-2 en ``criterios``) o larga (``fila, columna, valor`` con
    textos ok/parcial/fallo). ``veredicto``: columna (en la larga, el mismo valor en todas las filas de una fila)
    o mapa fila → veredicto. ``ordenar``: agrupa por veredicto.
    """
    fila = fila if fila in df.columns else (_col(df, "candidato", "fuente", "serie") or fila)
    ver = None
    if isinstance(veredicto, str) and veredicto in df.columns:
        ver = df.groupby(fila, sort=False)[veredicto].first()
        df = df.drop(columns=[veredicto])
    elif isinstance(veredicto, (Mapping, pd.Series)):
        ver = pd.Series(veredicto)
    if columna is not None and columna in df.columns:
        m = _matriz(df, fila, columna, valor)
    else:
        m = df.groupby(fila, sort=False).first()
        crit = [c for c in (criterios or []) if c in m.columns]
        if crit:
            m = m[crit]
    if ver is not None:
        m = m.assign(veredicto=ver.reindex(m.index).values)
        if ordenar:
            rango = {v: i for i, v in enumerate(ORDEN_VEREDICTO)}
            m = m.iloc[np.argsort([rango.get(v, 99) for v in m["veredicto"]], kind="stable")]
    fig, ax = _fig_ax(ax, (min(max(6.0, 1.0 * m.shape[1] + 3.5), 12), max(2.5, 0.19 * m.shape[0] + 1.4)))
    if m.empty:
        _vacio(ax)
        return fig, ax
    crits = m.drop(columns=["veredicto"]) if ver is not None else m
    nums = _a_numero(crits)
    es_num = nums.notna().sum().sum() >= 0.9 * crits.notna().sum().sum() and nums.notna().any().any()
    lo, hi = (float(np.nanmin(nums.values)), float(np.nanmax(nums.values))) if es_num else (0.0, 1.0)
    colores, textos = [], []
    for i in range(m.shape[0]):
        fc, ft = [], []
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            es_ver = ver is not None and j == m.shape[1] - 1
            if not es_ver and es_num:
                v = nums.iat[i, j]
                fc.append(color_valor(v, vmin=lo, vmax=hi) if pd.notna(v) else "#f2f2f2")
            else:
                fc.append(color_valor(v))
            ft.append(_texto_celda(v, 14).replace("_", " "))
        colores.append(fc)
        textos.append(ft)
    _rejilla(ax, m, colores, textos, fs=6)
    if ver is not None:
        ax.axvline(m.shape[1] - 1.5, color="black", lw=1.5)
    for v, col in C_SEMAFORO.items():
        ax.plot([], [], "s", color=col, label=v.replace("_", " "))
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), title="veredicto", title_fontsize=7)
    if ver is not None:
        cuenta = pd.Series(m["veredicto"]).value_counts()
        resumen = "; ".join(f"{k.replace('_', ' ')} {int(cuenta[k])}" for k in ORDEN_VEREDICTO if k in cuenta)
        ax.set_title(f"Semáforo final ({resumen})", fontsize=10)
    else:
        ax.set_title("Semáforo final", fontsize=10)
    return fig, ax
