"""Panorama de todos los detectores para el notebook 14 (comparativa).

Reúne en tablas y figuras lo que ya está calculado en ``results/``: los 12 detectores de v2
(``results/benchmark``), las variantes de la Fase 2 (``results/detectores/pool``, ADR-010), el
menú T descartado y sus ablaciones (``results/detectores/menus`` y ``ablaciones``, ADR-009) y
la vara de medir (``results/benchmark/vara``). **Solo lee y dibuja**: no ajusta detectores ni
vuelve a calcular el MCS. Las únicas escrituras son las figuras PNG (``results/detectores/
comparativa/``, ignoradas por git) y ``ranking_v2.csv`` (:func:`exportar_ranking_v2`, el mismo
fichero que escribía la versión anterior del notebook y que leen los notebooks de familia).

Envuelve a :mod:`regimenes.benchmark.pool_lectura` y a :mod:`regimenes.evaluacion.ranking` sin
modificarlos. :func:`ranking_principal` reproduce :func:`pool_lectura.ranking_v3` (mismos
detectores y puestos; lo comprueba el test) pero conserva todas las columnas del criterio.

Paleta: un color por familia (F1–F8), el mismo en todas las figuras; los baselines en gris.
Marcador por menú: N círculo, U cuadrado, P triángulo, R rombo, T triángulo invertido.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import warnings

import numpy as np
import pandas as pd

from regimenes import rutas
from regimenes.benchmark import pool_lectura as pl

V2 = rutas.RESULTS_BENCHMARK
POOL = pl.SALIDA
MENUS_T = rutas.RESULTS_DETECTORES / "menus"
ABLACIONES = rutas.RESULTS_DETECTORES / "ablaciones"
VARA = rutas.RESULTS_BENCHMARK / "vara"
FIGURAS = rutas.RESULTS_DETECTORES / "comparativa"
PISTAS = ("A", "B")

# --------------------------------------------------------------------------- #
# Familias, colores y menús
# --------------------------------------------------------------------------- #
FAMILIAS = {
    "F1": "F1 Reglas", "F2": "F2 Clustering", "F3": "F3 HMM", "F4": "F4 Markov-switching",
    "F5": "F5 GARCH", "F6": "F6 Change-point", "F7": "F7 Deep learning", "F8": "F8 Supervisados",
}
FAMILIA_DE_BASE = {
    "D01": "F1", "D02": "F1", "D10": "F1", "D14": "F1",
    "D03": "F2", "D09": "F2", "D15": "F2", "D18": "F2",
    "D04": "F3", "D08": "F3", "D13": "F3",
    "D05": "F4",
    "D06": "F5", "D11": "F5",
    "D07": "F6",
    "D12": "F7",
    "D16": "F8", "D17": "F8",
}
# Okabe-Ito (legible para daltónicos) con un marrón y un morado en lugar del amarillo.
COLOR_FAMILIA = {
    "F1": "#0072B2", "F2": "#E69F00", "F3": "#009E73", "F4": "#CC79A7",
    "F5": "#D55E00", "F6": "#56B4E9", "F7": "#8C6D31", "F8": "#6A3D9A",
}
COLOR_BASELINE = "#8a8a8a"
MARCADOR_MENU = {"N": "o", "U": "s", "P": "^", "R": "D", "T": "v"}
NOMBRE_MENU = {"N": "N núcleo", "U": "U univariante", "P": "P pool entero", "R": "R rentabilidades",
               "T": "T termómetros (descartado)"}
NOTEBOOK_FAMILIA = {
    "F1": "06_familia_F1_reglas.ipynb", "F2": "07_familia_F2_clustering.ipynb",
    "F3": "08_familia_F3_hmm.ipynb", "F4": "09_familia_F4_markov_switching.ipynb",
    "F5": "10_familia_F5_garch.ipynb", "F6": "11_familia_F6_changepoint.ipynb",
    "F7": "12_familia_F7_deep.ipynb", "F8": "13_familia_F8_supervisado.ipynb",
}
BASELINES = pl.BASELINES
SUPERVISADOS = ("D16", "D17")
GRUPOS = ("Núcleo N y U", "Pool sin comprimir (P, R)", "Menú T (descartado)", "Supervisados (F8)",
          "Referencias triviales")


def base_de(detector_id: str) -> str:
    return str(detector_id).replace("_", "-").split("-")[0]


def es_baseline(detector_id: str) -> bool:
    return str(detector_id) in BASELINES


def es_supervisado(detector_id: str) -> bool:
    return base_de(detector_id) in SUPERVISADOS


def menu_de(detector_id: str) -> str:
    """Menú de un id: ``D10`` → N, ``D06`` → U, ``D10-R`` → R, ``D16-P-excl`` → P, ``D02_T`` → T."""
    d = str(detector_id)
    if es_baseline(d):
        return "baseline"
    partes = d.replace("_", "-").split("-")
    return partes[1].upper() if len(partes) > 1 else pl.MENU_V2.get(d, "N")


def familia_de(detector_id: str) -> str:
    """Código de familia (``F1``…``F8``) o ``baseline``."""
    if es_baseline(detector_id):
        return "baseline"
    return FAMILIA_DE_BASE[base_de(detector_id)]


def color_de(detector_id: str) -> str:
    f = familia_de(detector_id)
    return COLOR_BASELINE if f == "baseline" else COLOR_FAMILIA[f]


def grupo_de(detector_id: str) -> str:
    if es_baseline(detector_id):
        return GRUPOS[4]
    if es_supervisado(detector_id):
        return GRUPOS[3]
    menu = menu_de(detector_id)
    return {"N": GRUPOS[0], "U": GRUPOS[0], "P": GRUPOS[1], "R": GRUPOS[1], "T": GRUPOS[2]}[menu]


def _anotar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["menu"] = df["id"].map(menu_de)
    df["familia_tfm"] = df["id"].map(familia_de)
    df["grupo"] = df["id"].map(grupo_de)
    return df


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #
def cargar_v2(output_dir: Path = V2) -> tuple[pd.DataFrame, dict]:
    """Métricas de los 12 detectores de v2 y cómo se ha verificado su procedencia.

    Intenta el modo estricto (huella de código, datos y entorno locales); si falla, usa las
    métricas versionadas y comprueba solo que cada CSV lleva la huella del ``manifest.json``."""
    from regimenes.benchmark import load_metrics, metrics_provenance

    info = {"verificacion_local": False, "motivo": ""}
    try:
        metrics = load_metrics(output_dir)
        info["verificacion_local"] = True
    except RuntimeError as exc:
        info["motivo"] = str(exc)[:160]
        metrics = load_metrics(output_dir, require_current=False)
    prov = metrics_provenance(output_dir)
    info["manifest_utc"] = prov.attrs.get("manifest_generated_utc")
    info["csv_con_huella_del_manifiesto"] = f"{int(prov['coincide_manifest'].sum())}/{len(prov)}"
    info["trazable"] = bool(len(prov) and prov["coincide_manifest"].all())
    return _anotar(metrics), info


def cargar_pool(salida: Path = POOL) -> pd.DataFrame:
    """Métricas de las 20 variantes de la Fase 2 (ADR-010), supervisadas incluidas."""
    tabla = pl.variantes(salida)
    return _anotar(tabla) if len(tabla) else tabla


def cargar_menu_t(menus: Path = MENUS_T) -> pd.DataFrame:
    """Métricas oficiales (reajuste mensual) de los 6 multivariantes con el menú T (ADR-008)."""
    partes = [pd.read_csv(p) for p in sorted(Path(menus).glob("[AB]_D*_T.csv"))]
    return _anotar(pd.concat(partes, ignore_index=True)) if partes else pd.DataFrame()


def metricas_todas(v2: pd.DataFrame | None = None) -> pd.DataFrame:
    """Una fila por pista y detector o variante: v2, pool (con supervisados) y menú T.

    Pasa por ``rank_detection`` solo para completar nivel y score con el criterio del ADR-003;
    los puestos de esta tabla no significan nada (mezcla menús que no compiten)."""
    from regimenes.evaluacion import ranking as rk

    v2 = cargar_v2()[0] if v2 is None else v2
    partes = [v2, cargar_pool(), cargar_menu_t()]
    m = pd.concat([p for p in partes if len(p)], ignore_index=True)
    out = rk.rank_detection(m, output_dir=V2, **rk.DETECTION_DEFAULTS)
    out = _anotar(out.drop(columns=["menu", "familia_tfm", "grupo"], errors="ignore"))
    return out.drop(columns=["puesto_deteccion", "rank_medio_legacy", "puesto_legacy"])


# --------------------------------------------------------------------------- #
# Ranking principal v3
# --------------------------------------------------------------------------- #
def menus_principales(salida: Path = POOL) -> dict:
    """Menús del ranking principal v3: N y U, más P y R si el §5 del ADR-010 se cumple en las
    dos pistas (misma lógica que la versión anterior del notebook, que leía ``criterio_adr010``)."""
    crit = pl.leer_tabla("criterio_adr010", salida)
    aporta = bool(crit is not None and not crit.attrs.get("desfasada")
                  and crit["cumple_pista"].astype(str).str.lower().eq("true").all())
    return {"menus": ("N", "U", "P", "R") if aporta else ("N", "U"), "aporta": aporta,
            "criterio": crit, "aviso": pl.aviso_tabla("criterio_adr010.csv", crit)}


def tabla_mcs(track: str, salida: Path = POOL) -> pd.DataFrame | None:
    """MCS al 10 % de N ∪ U ∪ P ∪ R con los baselines (``mcs_NUPR_<P>.csv``), indexado por modelo."""
    return pl.leer_tabla(f"mcs_NUPR_{track.upper()}", salida, index_col="modelo")


def ranking_principal(menus: Sequence[str] = ("N", "U", "P", "R"), v2: pd.DataFrame | None = None,
                      salida: Path = POOL) -> pd.DataFrame:
    """Ranking de detección (ADR-003) de los no supervisados de ``menus``, con todas las columnas.

    Mismos detectores y puestos que ``pool_lectura.ranking_v3(menus=menus)``. Añade menú,
    familia y, si existe la tabla, si cada uno queda dentro del MCS (``en_mcs``, ``perdida_mcs``)."""
    from regimenes.evaluacion import ranking as rk

    v2 = cargar_v2()[0] if v2 is None else v2
    pool = cargar_pool(salida)
    m = v2.copy()
    if len(pool):
        extra = pool[~pool["id"].map(es_supervisado) & pool["menu"].isin(list(menus))]
        m = pd.concat([m, extra], ignore_index=True)
    m = m[m["menu"].isin(list(menus))]
    out = rk.rank_detection(m, output_dir=V2, **rk.DETECTION_DEFAULTS)
    out = _anotar(out.drop(columns=["menu", "familia_tfm", "grupo"], errors="ignore"))
    out["en_mcs"] = pd.Series([pd.NA] * len(out), dtype="boolean")
    out["perdida_mcs"] = np.nan
    for t in PISTAS:
        mcs = tabla_mcs(t, salida)
        if mcs is None:
            continue
        sel = out["pista"] == t
        out.loc[sel, "en_mcs"] = out.loc[sel, "id"].map(mcs["incluido"].astype(bool)).astype("boolean")
        out.loc[sel, "perdida_mcs"] = out.loc[sel, "id"].map(mcs["perdida_media"])
    return out.sort_values(["pista", "puesto_deteccion", "id"]).reset_index(drop=True)


COLUMNAS_RANKING = ["pista", "puesto_deteccion", "id", "menu", "familia_tfm", "nivel_etiqueta",
                    "score_deteccion", "det_event_recall", "det_n_detectados", "det_n_eventos",
                    "det_precision", "det_lift_precision", "en_mcs", "perdida_mcs"]


def baselines(salida: Path = POOL) -> pd.DataFrame:
    """Score de las referencias triviales (``vara/baselines_score.csv``) y si entran en el MCS."""
    b = pd.read_csv(VARA / "baselines_score.csv")
    b["en_mcs"] = pd.Series([pd.NA] * len(b), dtype="boolean")
    for t in PISTAS:
        mcs = tabla_mcs(t, salida)
        if mcs is not None:
            sel = b["pista"] == t
            b.loc[sel, "en_mcs"] = b.loc[sel, "id"].map(mcs["incluido"].astype(bool)).astype("boolean")
    return b


def exportar_ranking_v2(v2: pd.DataFrame, output_dir: Path = V2) -> tuple[Path, bool]:
    """Escribe ``ranking_v2.csv`` (criterio ADR-003 sobre los 12 de v2, con columnas legacy).

    Es el fichero que leen los notebooks de familia (``informes.buscar_ranking``). Devuelve la
    ruta y si el contenido ha cambiado respecto al que había."""
    from regimenes.evaluacion import ranking as rk

    ruta = Path(output_dir) / "ranking_v2.csv"
    previo = ruta.read_bytes() if ruta.exists() else b""
    tabla = rk.rank_detection(v2.drop(columns=["menu", "familia_tfm", "grupo"], errors="ignore"),
                              output_dir=output_dir, **rk.DETECTION_DEFAULTS)
    tabla[rk.RANKING_COLUMNS].to_csv(ruta, index=False)
    return ruta, ruta.read_bytes() != previo


# --------------------------------------------------------------------------- #
# Inventario, gate y crisis evaluadas
# --------------------------------------------------------------------------- #
def inventario(todas: pd.DataFrame, principal: pd.DataFrame) -> pd.DataFrame:
    """Una fila por detector o variante: familia, menú, supervisado, si entra en el ranking
    principal, nivel del criterio y score en cada pista (vacío si no existe en esa pista)."""
    en_ranking = set(zip(principal["pista"], principal["id"]))
    filas = []
    for det_id, part in todas.groupby("id", sort=False):
        fila = {"id": det_id, "familia": FAMILIAS[familia_de(det_id)], "menu": menu_de(det_id),
                "supervisado": es_supervisado(det_id),
                "en_ranking_principal": any((t, det_id) in en_ranking for t in part["pista"]),
                "detector": str(part["detector"].iloc[0]) if "detector" in part else ""}
        for t in PISTAS:
            p = part[part["pista"] == t]
            fila[f"nivel_{t}"] = p["nivel_etiqueta"].iloc[0] if len(p) else ""
            fila[f"score_{t}"] = float(p["score_deteccion"].iloc[0]) if len(p) else np.nan
        filas.append(fila)
    inv = pd.DataFrame(filas)
    orden_menu = {"N": 0, "U": 1, "P": 2, "R": 3, "T": 4}
    inv["_f"] = inv["id"].map(familia_de)
    inv["_m"] = inv["menu"].map(orden_menu)
    inv["_b"] = inv["id"].map(base_de)
    return inv.sort_values(["supervisado", "_f", "_b", "_m", "id"]).drop(columns=["_f", "_m", "_b"]) \
        .reset_index(drop=True)


def resumen_inventario(inv: pd.DataFrame) -> pd.DataFrame:
    """Recuento por familia y menú (para la tabla compacta del notebook)."""
    t = inv.pivot_table(index="familia", columns="menu", values="id", aggfunc="count", fill_value=0)
    t = t.reindex(columns=[c for c in ("N", "U", "P", "R", "T") if c in t.columns])
    t["ids"] = inv.groupby("familia")["id"].apply(lambda s: ", ".join(s))
    return t


def gate(v2: pd.DataFrame, pool: pd.DataFrame | None = None, menu_t: pd.DataFrame | None = None
         ) -> tuple[pd.DataFrame, bool]:
    """Gate de comparabilidad: dentro de cada pista, todos (v2, pool y T) con el mismo OOS.

    Comprueba inicio, fin y longitud OOS comunes, ventana de entrenamiento común y ausencia de
    duplicados ``(pista, id)``. Devuelve la tabla por pista y si se cumple."""
    partes = [v2] + [p for p in (pool, menu_t) if p is not None and len(p)]
    m = pd.concat(partes, ignore_index=True)
    filas = []
    for t, part in m.groupby("pista"):
        filas.append({
            "pista": t, "filas": len(part), "de_v2": int((part["id"].isin(v2["id"])).sum()),
            "duplicados": int(part.duplicated(["pista", "id"]).sum()),
            "inicios_oos": part["oos_start"].nunique(), "finales_oos": part["oos_end"].nunique(),
            "longitudes_oos": part["n_oos"].nunique(), "train_configs": part["train_days"].nunique(),
            "oos": f"{part['oos_start'].iloc[0]} → {part['oos_end'].iloc[0]} (n={int(part['n_oos'].iloc[0])})",
        })
    tabla = pd.DataFrame(filas).set_index("pista")
    ok = bool((tabla["duplicados"] == 0).all()
              and (tabla[["inicios_oos", "finales_oos", "longitudes_oos", "train_configs"]] == 1).all().all()
              and set(v2.groupby("pista")["id"].nunique()) == {12})
    return tabla, ok


def ventanas_crisis(v2: pd.DataFrame) -> pd.DataFrame:
    """Cada ventana ``[pico, suelo]`` frente al OOS común: completa, parcial o fuera.

    Días = sesiones lunes–viernes aproximadas (informativo). Las de fuera no cuentan (NaN)."""
    from regimenes.evaluacion import ranking as rk

    filas = []
    for t in PISTAS:
        part = v2[v2["pista"] == t].iloc[0]
        ini, fin = pd.Timestamp(part["oos_start"]), pd.Timestamp(part["oos_end"])
        for nombre, (a, b) in rk.track_crisis_windows(t).items():
            a, b = pd.Timestamp(a), pd.Timestamp(b)
            dias_oos = len(pd.bdate_range(max(a, ini), min(b, fin))) if b >= ini else 0
            estado = "fuera" if dias_oos == 0 else ("completa" if a >= ini else "parcial")
            filas.append({"pista": t, "crisis": nombre, "pico": a.date(), "suelo": b.date(),
                          "dias_oos_aprox": dias_oos, "estado": estado})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# Figuras: utilidades
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib.pyplot as plt
    from regimenes import viz

    viz.use_house_style()
    return plt


def guardar(fig, nombre: str, carpeta: Path = FIGURAS) -> Path:
    """``<carpeta>/<nombre>.png`` con el estilo de casa (``informes.guardar_figura``)."""
    from regimenes.informes import guardar_figura

    return guardar_figura(fig, carpeta, nombre, verbose=False)


def _leyenda_familias(fig, familias: Iterable[str], *, menus: Iterable[str] = (), extra=(), y: float = -0.02,
                      ncol: int | None = None) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [Patch(color=COLOR_FAMILIA[f], label=FAMILIAS[f]) for f in sorted(set(familias))
               if f in COLOR_FAMILIA]
    handles += [Line2D([], [], ls="", marker=MARCADOR_MENU[m], color="#444", label=f"menú {NOMBRE_MENU[m]}")
                for m in menus]
    handles += list(extra)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), frameon=False,
               ncol=ncol or min(len(handles), 5), fontsize=9)


def _marca(valor) -> str:
    """● dentro del MCS, ○ fuera, vacío si no hay dato."""
    if valor is None or valor is pd.NA or (isinstance(valor, float) and np.isnan(valor)):
        return ""
    return "●" if bool(valor) else "○"


def _corto(crisis: str) -> str:
    return crisis.replace("_", " ")


# --------------------------------------------------------------------------- #
# Figura 0 · resumen
# --------------------------------------------------------------------------- #
def tabla_resumen(principal: pd.DataFrame, todas: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Score de detección de todos, agrupados en los cinco grupos de la figura resumen."""
    filas = [principal[["pista", "id", "score_deteccion", "nivel_etiqueta"]]]
    otros = todas[todas["grupo"].isin([GRUPOS[2], GRUPOS[3]])]
    filas.append(otros[["pista", "id", "score_deteccion", "nivel_etiqueta"]])
    filas.append(base[["pista", "id", "score_deteccion"]].assign(nivel_etiqueta="referencia"))
    r = pd.concat(filas, ignore_index=True)
    r["grupo"] = r["id"].map(grupo_de)
    return r


def figura_resumen(resumen: pd.DataFrame):
    """Un punto por detector y pista sobre el eje del score; una fila por grupo."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    rng = np.random.default_rng(0)
    cabezas = {}
    for ax, t in zip(axes, PISTAS):
        part = resumen[resumen["pista"] == t]
        for y, g in enumerate(GRUPOS):
            p = part[part["grupo"] == g].sort_values("score_deteccion", ascending=False)
            if not len(p):
                continue
            jit = rng.uniform(-0.18, 0.18, len(p))
            for (_, r), j in zip(p.iterrows(), jit):
                hueco = r["nivel_etiqueta"] not in ("elegible", "referencia")
                ax.scatter(r["score_deteccion"], y + j, s=70, marker=MARCADOR_MENU.get(menu_de(r["id"]), "X"),
                           color="white" if hueco else color_de(r["id"]), edgecolor=color_de(r["id"]),
                           linewidth=1.4, zorder=3)
            mejor = p.iloc[0]
            ax.annotate(f"{mejor['id']} {mejor['score_deteccion']:.2f}", (mejor["score_deteccion"], y),
                        xytext=(0, 13), textcoords="offset points", ha="center", fontsize=9, weight="bold")
            if g == GRUPOS[0]:
                cabezas[t] = mejor
                ax.axvline(mejor["score_deteccion"], color="#333", ls=":", lw=1)
        ax.set_yticks(range(len(GRUPOS)), GRUPOS)
        ax.set_ylim(len(GRUPOS) - 0.5, -0.6)
        ax.set_xlim(-0.02, 0.75)
        ax.set_xlabel("score de detección (F1 de recall por evento y precisión) →")
        ax.set_title(f"Pista {t}: la línea punteada es el mejor del núcleo ({cabezas[t]['id']})")
    fams = {familia_de(i) for i in resumen["id"] if not es_baseline(i)}
    _leyenda_familias(fig, fams, menus=("N", "U", "P", "R", "T"), y=0.0, ncol=7)
    fig.suptitle("Ningún grupo supera al mejor detector del núcleo N/U en ninguna pista "
                 if _nucleo_gana(resumen) else "Score de detección de todos los detectores, por grupo",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def _nucleo_gana(resumen: pd.DataFrame) -> bool:
    for t in PISTAS:
        p = resumen[resumen["pista"] == t]
        nucleo = p.loc[p["grupo"] == GRUPOS[0], "score_deteccion"].max()
        if (p.loc[p["grupo"] != GRUPOS[0], "score_deteccion"] >= nucleo).any():
            return False
    return True


# --------------------------------------------------------------------------- #
# Figura 1 · ranking v3
# --------------------------------------------------------------------------- #
def figura_ranking(principal: pd.DataFrame, base: pd.DataFrame):
    """Barras horizontales del score por pista, color por familia, MCS y referencias triviales."""
    plt = _plt()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    alto = max(principal.groupby("pista").size()) * 0.38 + 1.8
    fig, axes = plt.subplots(1, 2, figsize=(14, alto))
    estilos = {"VOL_P90": ("-", "#444"), "SIEMPRE_CRISIS": ("--", COLOR_BASELINE),
               "AZAR_PERSISTENTE": (":", COLOR_BASELINE)}
    for ax, t in zip(axes, PISTAS):
        p = principal[principal["pista"] == t].sort_values("puesto_deteccion")
        y = np.arange(len(p))
        for yi, (_, r) in zip(y, p.iterrows()):
            elegible = r["nivel_etiqueta"] == "elegible"
            ax.barh(yi, r["score_deteccion"], color=color_de(r["id"]), alpha=1 if elegible else 0.35,
                    hatch="///" if r["menu"] in ("P", "R") else None, edgecolor="white", height=0.75)
            marca = _marca(r["en_mcs"])
            nota = "" if elegible else f"  ({r['nivel_etiqueta'].replace('_', ' ')})"
            ax.text(r["score_deteccion"] + 0.008, yi, f"{r['score_deteccion']:.3f} {marca}{nota}",
                    va="center", fontsize=8.5)
        ax.set_yticks(y, [f"{int(r.puesto_deteccion)}. {r.id}" for r in p.itertuples()])
        ax.invert_yaxis()
        b = base[base["pista"] == t].set_index("id")
        for k, (bid, (ls, col)) in enumerate(estilos.items()):
            if bid in b.index:
                x = b.loc[bid, "score_deteccion"]
                ax.axvline(x, ls=ls, color=col, lw=1.3)
                dentro = b.loc[bid, "en_mcs"]
                ax.text(x, -0.75 - 0.55 * (k % 2), f" {bid} {x:.2f} {_marca(dentro)} ",
                        ha="right" if k == 1 else "left", va="center", fontsize=7.5, color=col,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
        ax.set_xlim(0, 0.8)
        ax.set_ylim(len(p) - 0.4, -1.9)
        ax.set_xlabel("score de detección →")
        n_mcs = int(p["en_mcs"].fillna(False).sum())
        top = p.iloc[0]
        ax.set_title(f"Pista {t}: {top['id']} primero; {n_mcs} de {len(p)} detectores dentro del MCS",
                     fontsize=11)
        ax.grid(axis="y", visible=False)
    extra = [Patch(facecolor="white", edgecolor="#444", hatch="///", label="variante del pool (P o R)"),
             Patch(color="white", label="● dentro del MCS al 10 %   ○ fuera"),
             Line2D([], [], ls="-", color="#444", label="VOL_P90: un umbral sobre la volatilidad"),
             Line2D([], [], ls="--", color=COLOR_BASELINE, label="SIEMPRE_CRISIS"),
             Line2D([], [], ls=":", color=COLOR_BASELINE, label="AZAR_PERSISTENTE")]
    _leyenda_familias(fig, principal["familia_tfm"], extra=extra, y=0.0, ncol=5)
    tops = principal[principal["puesto_deteccion"] == 1]
    nucleo = tops["menu"].isin(["N", "U"]).all()
    fig.suptitle(f"Ranking v3 ({', '.join(sorted(principal['menu'].unique(), key='NUPR'.index))}): "
                 + ("la cabeza es del núcleo N/U en las dos pistas" if nucleo else "cabeza por pista"),
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


# --------------------------------------------------------------------------- #
# Figura 2 · recall frente a precisión
# --------------------------------------------------------------------------- #
def figura_frontera(todas: pd.DataFrame, principal: pd.DataFrame, base: pd.DataFrame):
    """Recall por evento frente a precisión diaria de todos (T y supervisados en hueco)."""
    plt = _plt()
    from matplotlib.lines import Line2D
    from regimenes.evaluacion import ranking as rk

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    fronteras = {}
    for ax, t in zip(axes, PISTAS):
        # Curvas de igual F1 para leer el score sin tabla.
        pp = np.linspace(0.01, 1, 200)
        for f in (0.2, 0.4, 0.6):
            rr = f * pp / (2 * pp - f)
            ok = (rr > 0) & (rr <= 1.05)
            ax.plot(pp[ok], rr[ok], color="#dddddd", lw=0.9, zorder=0)
            x_et = 0.6
            ax.text(x_et, f * x_et / (2 * x_et - f), f"score {f}", fontsize=7, color="#999999", ha="right",
                    va="bottom")
        part = todas[(todas["pista"] == t) & ~todas["id"].map(es_baseline)]
        for _, r in part.iterrows():
            hueco = r["grupo"] in (GRUPOS[2], GRUPOS[3])
            ax.scatter(r["det_precision"], r["det_event_recall"], s=60,
                       marker=MARCADOR_MENU.get(r["menu"], "o"), color="white" if hueco else color_de(r["id"]),
                       edgecolor=color_de(r["id"]), linewidth=1.3, zorder=3, alpha=0.95)
        p = principal[principal["pista"] == t]
        mask = rk.pareto_mask(p, ["det_event_recall", "det_precision"])
        front = p[mask].sort_values("det_precision")
        fronteras[t] = list(front["id"])
        ax.plot(front["det_precision"], front["det_event_recall"], color="#b5341f", lw=1.2, zorder=2)
        for _, r in p.iterrows():
            if r["id"] in fronteras[t] or r["puesto_deteccion"] <= 3:
                ax.annotate(r["id"], (r["det_precision"], r["det_event_recall"]), xytext=(4, 4),
                            textcoords="offset points", fontsize=8.5,
                            weight="bold" if r["id"] in fronteras[t] else "normal")
        b = base[(base["pista"] == t) & base["det_precision"].notna()]
        ax.scatter(b["det_precision"], b["det_event_recall"], marker="X", s=60, color=COLOR_BASELINE, zorder=3)
        for _, r in b.iterrows():
            ax.annotate(r["id"], (r["det_precision"], r["det_event_recall"]), xytext=(4, -10),
                        textcoords="offset points", fontsize=7.5, color="#666")
        tasa = float(p["det_base_rate"].iloc[0])
        ax.axvline(tasa, ls="--", color="#999", lw=1)
        ax.text(tasa, 0.02, " tasa base\n (= azar)", fontsize=7.5, color="#777")
        ax.set_xlim(0, 0.62)
        ax.set_ylim(-0.03, 1.07)
        ax.set_xlabel("precisión diaria (días marcados que caen en crisis) →")
        ax.set_ylabel("recall por evento (crisis detectadas) ↑")
        ax.set_title(f"Pista {t}: frontera (rojo) = {', '.join(fronteras[t])}", fontsize=11)
    extra = [Line2D([], [], ls="", marker="o", mfc="white", mec="#444", label="hueco: menú T o supervisado"),
             Line2D([], [], ls="", marker="X", color=COLOR_BASELINE, label="referencia trivial"),
             Line2D([], [], color="#b5341f", label="frontera de Pareto (ranking principal)")]
    _leyenda_familias(fig, todas.loc[~todas["id"].map(es_baseline), "familia_tfm"],
                      menus=("N", "U", "P", "R", "T"), extra=extra, y=0.0, ncol=6)
    fig.suptitle("Recall por evento frente a precisión diaria: arriba a la derecha es mejor "
                 "(curvas grises = mismo score)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.fronteras = fronteras
    return fig


# --------------------------------------------------------------------------- #
# Figura 3 · detector × crisis
# --------------------------------------------------------------------------- #
def matriz_crisis(principal: pd.DataFrame, track: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(detectada 0/1, cobertura) detector × crisis evaluada, filas en orden del ranking.

    Sale de las columnas ``det_ev_<crisis>`` y ``cov_<crisis>`` de cada CSV de métricas (v2 y
    pool), calculadas por la misma evaluación del benchmark sobre el panel OOS."""
    from regimenes.evaluacion import ranking as rk

    p = principal[principal["pista"] == track].sort_values("puesto_deteccion").set_index("id")
    nombres = [c for c in rk.track_crisis_windows(track) if f"det_ev_{c}" in p and p[f"det_ev_{c}"].notna().any()]
    det = p[[f"det_ev_{c}" for c in nombres]].set_axis(nombres, axis=1).astype(float)
    cov = p[[f"cov_{c}" for c in nombres]].set_axis(nombres, axis=1).astype(float)
    return det, cov


def figura_crisis(principal: pd.DataFrame):
    """Mapa de calor detector × crisis por pista: color = cobertura, ✓ = detectada."""
    plt = _plt()
    mats = {t: matriz_crisis(principal, t) for t in PISTAS}
    anchos = [mats[t][0].shape[1] for t in PISTAS]
    alto = max(m[0].shape[0] for m in mats.values()) * 0.36 + 2.6
    fig, axes = plt.subplots(1, 2, figsize=(15, alto), gridspec_kw={"width_ratios": anchos})
    dificiles = {}
    for ax, t in zip(axes, PISTAS):
        det, cov = mats[t]
        im = ax.imshow(cov.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(det.shape[0]):
            for j in range(det.shape[1]):
                if det.iat[i, j] == 1:
                    ax.text(j, i, "✓", ha="center", va="center", fontsize=9,
                            color="white" if cov.iat[i, j] > 0.55 else "#08306b")
        n = det.notna().sum()
        k = det.sum()
        ax.set_xticks(range(det.shape[1]), [f"{_corto(c)} ({int(k[c])}/{int(n[c])})" for c in det.columns],
                      rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(det.shape[0]), det.index, fontsize=8.5)
        for lab in ax.get_yticklabels():
            lab.set_color(color_de(lab.get_text()))
        ax.grid(False)
        frac = (k / n).sort_values()
        dificiles[t] = list(frac.index[:3])
        ax.set_title(f"Pista {t}: la que menos detectan es {_corto(frac.index[0])} "
                     f"({int(k[frac.index[0]])}/{int(n[frac.index[0]])})", fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01, label="cobertura: fracción de la crisis marcada")
    fig.suptitle("Las grandes crisis las ve casi todo el mundo; el ranking se decide en las cortas o atípicas\n"
                 "(✓ = detectada: ≥ 3 sesiones seguidas en crisis dentro de [pico, suelo]; filas en orden del ranking;"
                 " color del nombre = familia)", fontsize=12, weight="bold")
    fig.dificiles = dificiles
    return fig


# --------------------------------------------------------------------------- #
# Figura 4 · persistencia, estabilidad y coste
# --------------------------------------------------------------------------- #
COLUMNAS_ESTABILIDAD = ["pista", "puesto_deteccion", "id", "menu", "nivel_etiqueta", "mean_regime_duration",
                        "det_mean_crisis_run", "switching_rate", "label_stability", "elapsed_seconds"]


def tabla_estabilidad(principal: pd.DataFrame) -> pd.DataFrame:
    t = principal[COLUMNAS_ESTABILIDAD].copy()
    t["coste_min"] = t.pop("elapsed_seconds") / 60
    return t


def figura_estabilidad(principal: pd.DataFrame):
    """Duración media del régimen frente al coste de cómputo; tamaño = episodio medio de crisis."""
    plt = _plt()
    from regimenes.evaluacion import ranking as rk

    umbral = rk.DETECTION_DEFAULTS["min_mean_duration"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    rhos, ordenes = {}, {}
    for ax, t in zip(axes, PISTAS):
        p = principal[principal["pista"] == t]
        for _, r in p.iterrows():
            elegible = r["nivel_etiqueta"] == "elegible"
            tam = 25 + 6 * np.sqrt(max(float(r["det_mean_crisis_run"]) if pd.notna(r["det_mean_crisis_run"]) else 0, 0))
            ax.scatter(r["mean_regime_duration"], r["elapsed_seconds"], s=tam, marker=MARCADOR_MENU[r["menu"]],
                       color=color_de(r["id"]) if elegible else "white", edgecolor=color_de(r["id"]),
                       linewidth=1.3, alpha=0.9, zorder=3)
            ax.annotate(r["id"], (r["mean_regime_duration"], r["elapsed_seconds"]), xytext=(5, 2),
                        textcoords="offset points", fontsize=8)
        ax.axvline(umbral, ls="--", color="#999", lw=1)
        ax.text(umbral, p["elapsed_seconds"].min(), " filtro de parpadeo",
                fontsize=7.5, color="#777", rotation=90, va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("duración media de un régimen (sesiones, log)")
        ax.set_ylabel("coste del walk-forward (segundos, log)")
        parpadeo = p.loc[p["nivel_etiqueta"] == "parpadeo", "id"].tolist()
        horas = p["elapsed_seconds"].max() / 3600
        rho = p[["elapsed_seconds", "score_deteccion"]].corr("spearman").iloc[0, 1]
        rhos[t] = rho
        ordenes[t] = np.log10(p["elapsed_seconds"].max() / p["elapsed_seconds"].min())
        ax.set_title(f"Pista {t}: parpadeo: {', '.join(parpadeo) or 'ninguno'}; el más caro tarda {horas:.1f} h;"
                     f" Spearman(coste, score) = {rho:+.2f}", fontsize=10.5)
    _leyenda_familias(fig, principal["familia_tfm"], menus=sorted(principal["menu"].unique(), key="NUPR".index),
                      y=0.0, ncol=6)
    fig.suptitle(f"El coste varía unos {max(ordenes.values()):.0f} órdenes de magnitud y no compra detección "
                 f"(Spearman coste-score {rhos['A']:+.2f} en A, {rhos['B']:+.2f} en B)\n"
                 "(hueco = no elegible por el criterio; tamaño = duración media de los episodios de crisis)",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


# --------------------------------------------------------------------------- #
# Figura 5 · ¿sirve el pool? (P/R frente a N; menú T)
# --------------------------------------------------------------------------- #
def tabla_pool(principal: pd.DataFrame, salida: Path = POOL) -> pd.DataFrame:
    """Criterio (a) (cada variante frente a su par en N) con (b) (¿dentro del MCS?) al lado."""
    comp = pl.comparacion(salida)
    if comp.empty:
        return comp
    comp = comp[~comp["id"].map(es_supervisado)].copy()
    clave = principal.set_index(["pista", "id"])
    comp["b_en_mcs"] = [bool(clave.loc[(r.pista, r.id), "en_mcs"]) if (r.pista, r.id) in clave.index
                        and pd.notna(clave.loc[(r.pista, r.id), "en_mcs"]) else False for r in comp.itertuples()]
    comp["puesto_v3"] = [int(clave.loc[(r.pista, r.id), "puesto_deteccion"]) if (r.pista, r.id) in clave.index
                         else np.nan for r in comp.itertuples()]
    comp["cumple_a_y_b"] = comp["a_mejora_par"].astype(bool) & comp["b_en_mcs"] & comp["cuenta_para_veredicto"].astype(bool)
    return comp


def figura_pool(tabla: pd.DataFrame, principal: pd.DataFrame, criterio: pd.DataFrame | None):
    """Flecha de cada par en N a su variante P/R; color = criterio (a); relleno = criterio (b)."""
    plt = _plt()
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, t in zip(axes, PISTAS):
        p = tabla[tabla["pista"] == t].sort_values("score")
        for y, (_, r) in enumerate(p.iterrows()):
            if not r["cuenta_para_veredicto"]:
                col = COLOR_BASELINE
            else:
                col = "#1a9850" if r["a_mejora_par"] else "#d73027"
            ax.annotate("", xy=(r["score"], y), xytext=(r["score_par"], y),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2, mutation_scale=14))
            ax.scatter(r["score_par"], y, s=45, color=color_de(r["par_v2"]), zorder=3)
            ax.scatter(r["score"], y, s=90, marker=MARCADOR_MENU.get(r["menu"], "o"),
                       color=color_de(r["id"]) if r["b_en_mcs"] else "white", edgecolor=color_de(r["id"]),
                       linewidth=1.5, zorder=4)
            ax.text(max(r["score"], r["score_par"]) + 0.015, y,
                    f"{r['delta_score']:+.3f}" + ("  (sin peso)" if not r["cuenta_para_veredicto"] else ""),
                    va="center", fontsize=8.5, color=col)
        ax.set_yticks(range(len(p)), [f"{r.id}  ←  {r.par_v2}" for r in p.itertuples()])
        cab = principal[(principal["pista"] == t) & principal["menu"].isin(["N", "U"])].sort_values("puesto_deteccion").iloc[0]
        ax.axvline(cab["score_deteccion"], color="#333", ls=":", lw=1.2)
        ax.text(cab["score_deteccion"], len(p) - 0.45, f" cabeza N/U: {cab['id']} {cab['score_deteccion']:.2f}",
                fontsize=8, va="bottom")
        ax.set_xlim(-0.02, 0.78)
        ax.set_ylim(-0.6, len(p) - 0.1)
        ax.set_xlabel("score de detección →")
        cumple = ""
        if criterio is not None:
            c = criterio.set_index("pista")
            if t in c.index:
                cumple = str(c.loc[t, "cumplen_a_y_b"]) if pd.notna(c.loc[t, "cumplen_a_y_b"]) else "ninguna"
                cumple = cumple.replace("|", ", ")
        ax.set_title(f"Pista {t}: cumplen (a) y (b): {cumple or 'ninguna'}", fontsize=11)
        ax.grid(axis="y", visible=False)
    extra = [Line2D([], [], color="#1a9850", lw=2, label="(a) mejora a su par en N"),
             Line2D([], [], color="#d73027", lw=2, label="(a) empeora"),
             Line2D([], [], color=COLOR_BASELINE, lw=2, label="D15-U: se reporta sin peso"),
             Line2D([], [], ls="", marker="^", color="#444", label="relleno: (b) dentro del MCS; hueco: fuera")]
    _leyenda_familias(fig, list(tabla["id"].map(familia_de)) + list(tabla["par_v2"].map(familia_de)),
                      extra=extra, y=0.0, ncol=4)
    veredicto = criterio["veredicto"].iloc[0] if criterio is not None and len(criterio) else "pendiente"
    max_var = tabla.loc[tabla["cuenta_para_veredicto"].astype(bool), "score"].groupby(tabla["pista"]).max()
    cabezas = principal[principal["menu"].isin(["N", "U"])].groupby("pista")["score_deteccion"].max()
    desplaza = bool((max_var.reindex(cabezas.index) >= cabezas).any())
    fig.suptitle(f"¿Sirve el pool sin comprimir? Veredicto del ADR-010 §5: «{veredicto}»"
                 + ("" if desplaza else ", pero ninguna variante alcanza a la cabeza de N/U"),
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


def tablas_menu_t() -> dict[str, pd.DataFrame]:
    """CSV del ADR-009 (menú T): N frente a T oficial, criterio (b), media frente a PCA, quitar un
    bloque y la distribución de las 100 reparticiones al azar."""
    def leer(ruta: Path) -> pd.DataFrame:
        return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()

    return {"n_vs_t": leer(MENUS_T / "comparacion_N_T.csv"),
            "criterios_adr008": leer(MENUS_T / "criterios_adr008.csv"),
            "criterio_b": leer(ABLACIONES / "criterio_b.csv"),
            "azar": leer(ABLACIONES / "medianas_azar.csv"),
            "media": leer(ABLACIONES / "media.csv"),
            "sin_bloque": leer(ABLACIONES / "sin_bloque.csv"),
            "mcs_t": pd.concat([leer(ABLACIONES / f"mcs_menu_T_{t}.csv") for t in PISTAS], ignore_index=True)}


def figura_menu_t(tablas: dict[str, pd.DataFrame]):
    """Cuatro paneles del ADR-009: N→T, bloques frente al azar, media frente a PCA, quitar un bloque."""
    plt = _plt()
    fig, axs = plt.subplots(2, 2, figsize=(14, 9.5))
    # (1) N → T por detector y pista
    ax = axs[0, 0]
    nt = tablas["n_vs_t"].sort_values(["pista", "base"])
    for y, r in enumerate(nt.itertuples()):
        col = "#1a9850" if r.delta_score > 0 else "#d73027"
        ax.annotate("", xy=(r.score_deteccion_T, y), xytext=(r.score_deteccion_N, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, mutation_scale=12))
        ax.scatter(r.score_deteccion_N, y, s=40, color=color_de(r.base), zorder=3)
        ax.scatter(r.score_deteccion_T, y, s=60, marker="v", color="white", edgecolor=color_de(r.base), zorder=3)
    ax.set_yticks(range(len(nt)), [f"{r.pista} · {r.base}" for r in nt.itertuples()])
    ax.invert_yaxis()
    mejoran = nt.groupby("pista")["delta_score"].apply(lambda s: f"{int((s > 0).sum())}/{len(s)}")
    ax.set_title("① Con T mejoran " + " y ".join(f"{v} en {k}" for k, v in mejoran.items())
                 + " (● N → ▽ T)", fontsize=10.5)
    ax.set_xlabel("score de detección")
    # (2) criterio (b): bloques económicos frente a 100 reparticiones al azar
    ax = axs[0, 1]
    cb = tablas["criterio_b"].set_index("pista")
    az = tablas["azar"]
    rng = np.random.default_rng(1)
    for y, t in enumerate(PISTAS):
        v = az.loc[az["pista"] == t, "mediana"].to_numpy()
        ax.scatter(v, y + rng.uniform(-0.17, 0.17, len(v)), s=12, color=COLOR_BASELINE, alpha=0.6)
        ax.plot([cb.loc[t, "p90_azar"]] * 2, [y - 0.3, y + 0.3], color="#333", lw=2)
        ax.scatter(cb.loc[t, "mediana_ref_t"], y, s=160, marker="*", color="#b5341f", zorder=4)
        ax.text(cb.loc[t, "mediana_ref_t"], y - 0.33, f"percentil {cb.loc[t, 'percentil_ref_t']:.0f} · "
                f"{cb.loc[t, 'veredicto_b']}", ha="center", fontsize=8.5)
    ax.set_yticks(range(len(PISTAS)), [f"Pista {t}" for t in PISTAS])
    ax.set_ylim(len(PISTAS) - 0.5, -0.6)
    ax.set_xlabel("mediana del score de los 6 multivariantes")
    ax.set_title("② Bloques económicos (★) frente a 100 repartos al azar (·); barra = p90", fontsize=10.5)
    # (3) media simple frente a PCA
    ax = axs[1, 0]
    me = tablas["media"]
    me = me[me["base"] != "mediana"].sort_values(["pista", "base"])
    for y, r in enumerate(me.itertuples()):
        col = "#1a9850" if r.delta > 0 else "#d73027"
        ax.annotate("", xy=(r.score_media, y), xytext=(r.score_ref_t, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, mutation_scale=12))
        ax.scatter(r.score_ref_t, y, s=40, color=color_de(r.base), zorder=3)
    ax.set_yticks(range(len(me)), [f"{r.pista} · {r.base}" for r in me.itertuples()])
    ax.invert_yaxis()
    gana = me.groupby("pista")["delta"].apply(lambda s: f"{int((s > 0).sum())}/{len(s)}")
    ax.set_title("③ La media simple gana a la PCA en " + " y ".join(f"{v} en {k}" for k, v in gana.items()),
                 fontsize=10.5)
    ax.set_xlabel("score de detección (● PCA → punta: media)")
    # (4) quitar un bloque
    ax = axs[1, 1]
    sb = tablas["sin_bloque"].copy()
    sb["etiqueta"] = sb["pista"] + " · " + sb["bloque"]
    sb = sb.sort_values(["pista", "mediana_delta"], ascending=[True, False])
    cols = ["#d73027" if v > 0 else "#1a9850" for v in sb["mediana_delta"]]
    ax.barh(range(len(sb)), sb["mediana_delta"], color=cols)
    ax.set_yticks(range(len(sb)), sb["etiqueta"], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("mediana del cambio de score al quitar el bloque (+ = estorba)")
    ax.set_title("④ Quitar un bloque: rojo estorba, verde ayuda", fontsize=10.5)
    mt = tablas.get("mcs_t", pd.DataFrame())
    vol_dentro = sorted(mt.loc[(mt["modelo"] == "VOL_P90") & mt["incluido"].astype(bool), "pista"]) if len(mt) else []
    fig.suptitle("Menú T (termómetros por bloque): " + ", ".join(f"mejora a N en {v} en {k}" for k, v in mejoran.items())
                 + (f"; VOL_P90 sigue dentro de su MCS en {' y '.join(vol_dentro)}" if vol_dentro else "")
                 + " → descartado (ADR-009)", fontsize=12.5, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# --------------------------------------------------------------------------- #
# Figura 6 · techo de los supervisados
# --------------------------------------------------------------------------- #
def tabla_techo(salida: Path = POOL) -> pd.DataFrame:
    t = pl.leer_tabla("techo_supervisados", salida)
    if t is None or t.attrs.get("desfasada"):
        t = pl.techo(salida)
    if len(t):
        t = t.copy()
        t["menu"] = t["id"].map(lambda i: "P-excl" if i.endswith("-excl") else menu_de(i))
    return t


def figura_techo(techo: pd.DataFrame):
    plt = _plt()
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    alfa = {"N": 1.0, "P-excl": 0.6, "P": 0.3}
    for ax, t in zip(axes, PISTAS):
        p = techo[techo["pista"] == t].sort_values("score", ascending=False)
        for y, r in enumerate(p.itertuples()):
            ax.barh(y, r.score, color=COLOR_FAMILIA["F8"], alpha=alfa.get(r.menu, 0.5),
                    hatch="///" if r.menu != "N" else None, edgecolor="white", height=0.7)
            dentro = getattr(r, "en_mcs_conjunto", None)
            marca = " " + _marca(dentro)
            ax.text(r.score + 0.008, y, f"{r.score:.3f}{marca}", va="center", fontsize=8.5)
        ax.set_yticks(range(len(p)), p["id"])
        ax.invert_yaxis()
        mejor, sm = p["mejor_no_supervisado"].iloc[0], p["score_mejor"].iloc[0]
        ax.axvline(sm, color=color_de(mejor), lw=2)
        ax.text(sm - 0.005, -0.75, f"mejor no supervisado: {mejor} {sm:.3f}", ha="right", fontsize=8.5,
                color=color_de(mejor), bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
        ax.set_xlim(0, 0.75)
        ax.set_ylim(len(p) - 0.4, -1.1)
        ax.set_xlabel("score de detección →")
        top = p.iloc[0]
        ax.set_title(f"Pista {t}: el mejor supervisado ({top['id']}, {top['score']:.2f}) queda "
                     f"{sm - top['score']:.2f} por debajo de {mejor}", fontsize=11)
        ax.grid(axis="y", visible=False)
    extra = [Patch(color=COLOR_FAMILIA["F8"], alpha=a, label=f"con {m}") for m, a in alfa.items()]
    extra.append(Patch(color="white", label="● dentro del MCS conjunto  ○ fuera"))
    fig.legend(handles=extra, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False)
    ninguno = not techo["mejora_al_mejor"].astype(bool).any()
    fig.suptitle("Supervisados (F8, con etiquetas de crisis): " +
                 ("ni con etiquetas se supera al mejor no supervisado, y con el pool P rinden peor que con N"
                  if ninguno else "alguno mejora al mejor no supervisado"), fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


# --------------------------------------------------------------------------- #
# Robustez del criterio
# --------------------------------------------------------------------------- #
VARIANTES_CRITERIO = {
    "beta=0.5": {"beta": 0.5}, "beta=2": {"beta": 2.0},
    "min_run=1": {"min_run": 1}, "min_run=5": {"min_run": 5},
    "lift_min=1.5": {"lift_min": 1.5},
    "sin_filtros_persistencia": {"min_mean_duration": 0.0, "min_crisis_run": 0.0},
    "min_crisis_run=5": {"min_crisis_run": 5.0},
}


def _con_min_run(m: pd.DataFrame, min_run: int, salida: Path = POOL) -> pd.DataFrame:
    """Recalcula ``det_*`` de las variantes del pool con otro ``min_run`` desde su panel OOS.

    Las de v2 las recalcula ``rank_detection`` desde ``results/benchmark/panels``; las del pool
    tienen el panel en otra carpeta. Sin panel, ``rank_detection`` usa la aproximación por
    cobertura y lo marca en ``fuente_deteccion``."""
    from regimenes.evaluacion import ranking as rk

    out = m.copy()
    for i, r in out.iterrows():
        if "-" not in str(r["id"]):
            continue
        ruta = Path(salida) / f"{r['pista']}_{r['id']}.parquet"
        if not ruta.exists():
            out.at[i, "det_min_run"] = np.nan
            out.at[i, "det_event_recall"] = np.nan
            continue
        flags = pd.read_parquet(ruta, columns=["state"])["state"].eq(int(r["n_states"]) - 1)
        for k, v in rk.detection_summary(flags, rk.track_crisis_windows(r["pista"]), min_run=min_run).items():
            out.at[i, k] = v
        out.at[i, "fuente_deteccion"] = "panel"
    return out


def metricas_principal(menus: Sequence[str], v2: pd.DataFrame) -> pd.DataFrame:
    """Métricas en bruto de los detectores del ranking principal (entrada de ``rank_detection``)."""
    pool = cargar_pool()
    m = pd.concat([v2, pool[~pool["id"].map(es_supervisado)]], ignore_index=True)
    return m[m["menu"].isin(list(menus))].drop(columns=["familia_tfm", "grupo"], errors="ignore")


def sensibilidad(m: pd.DataFrame) -> pd.DataFrame:
    """Puesto de cada detector con el criterio oficial y con cada variante de parámetros."""
    from regimenes.evaluacion import ranking as rk

    puestos = {"oficial": rk.rank_detection(m, output_dir=V2, **rk.DETECTION_DEFAULTS)
               .set_index(["pista", "id"])["puesto_deteccion"]}
    for nombre, cambio in VARIANTES_CRITERIO.items():
        params = {**rk.DETECTION_DEFAULTS, **cambio}
        entrada = _con_min_run(m, int(params["min_run"])) if params["min_run"] != rk.DETECTION_DEFAULTS["min_run"] else m
        puestos[nombre] = rk.rank_detection(entrada, output_dir=V2, **params).set_index(["pista", "id"])["puesto_deteccion"]
    return pd.DataFrame(puestos)


def loco(m: pd.DataFrame) -> pd.DataFrame:
    """Puesto al quitar una crisis cada vez del recall por evento (leave-one-crisis-out)."""
    from regimenes.evaluacion import ranking as rk

    cols = {}
    for t in PISTAS:
        part = m[m["pista"] == t]
        ev = [c for c in part if c.startswith("det_ev_") and part[c].notna().any()]
        for c in ev:
            alt = part.copy()
            alt["det_event_recall"] = alt[[x for x in ev if x != c]].mean(axis=1, skipna=True)
            r = rk.rank_detection(alt, output_dir=V2, **rk.DETECTION_DEFAULTS)
            cols.setdefault(c.removeprefix("det_ev_"), []).append(r.set_index(["pista", "id"])["puesto_deteccion"])
    return pd.DataFrame({k: pd.concat(v) for k, v in cols.items()})


def nulo_persistente(principal: pd.DataFrame, m: pd.DataFrame, n_sims: int = 200) -> pd.DataFrame:
    """p-valor del score frente a una señal de Markov independiente de las crisis con la misma
    tasa de marcado y la misma duración de episodios que el detector (informativo)."""
    from regimenes.evaluacion import ranking as rk

    filas = []
    for t in PISTAS:
        indice, _ = rk.track_oos_index(t, m, V2)
        ventanas = rk.track_crisis_windows(t)
        for i, r in enumerate(principal[principal["pista"] == t].itertuples()):
            if r.nivel_etiqueta == "degenerado" or not np.isfinite(r.det_mean_crisis_run):
                continue
            nulo = rk.persistent_random_null(indice, ventanas, rate=r.det_marked_rate, mean_run=r.det_mean_crisis_run,
                                             n_sims=n_sims, seed=1000 + i)
            filas.append({"pista": t, "id": r.id, "score": r.score_deteccion,
                          "score_nulo_p95": nulo["score_deteccion"].quantile(0.95),
                          "p_valor": (1 + (nulo["score_deteccion"] >= r.score_deteccion).sum()) / (1 + n_sims)})
    return pd.DataFrame(filas)


def tabla_robustez(principal: pd.DataFrame, sens: pd.DataFrame, lo: pd.DataFrame, nulo: pd.DataFrame
                   ) -> pd.DataFrame:
    """Por detector: puesto oficial, rango con parámetros, rango sin una crisis y p-valor."""
    t = principal.set_index(["pista", "id"])[["puesto_deteccion"]].copy()
    t["param_min"], t["param_max"] = sens.min(axis=1), sens.max(axis=1)
    t["loco_min"], t["loco_max"] = lo.min(axis=1), lo.max(axis=1)
    t = t.join(nulo.set_index(["pista", "id"])["p_valor"])
    return t.reset_index().sort_values(["pista", "puesto_deteccion"])


def figura_robustez(rob: pd.DataFrame, sens: pd.DataFrame):
    """Rango de puestos (parámetros y sin una crisis) y p-valor frente al azar persistente."""
    plt = _plt()
    from matplotlib.lines import Line2D

    alto = max(rob.groupby("pista").size()) * 0.36 + 1.8
    fig, axes = plt.subplots(1, 2, figsize=(14, alto))
    for ax, t in zip(axes, PISTAS):
        p = rob[rob["pista"] == t].reset_index(drop=True)
        for y, r in p.iterrows():
            col = color_de(r["id"])
            ax.plot([r["param_min"], r["param_max"]], [y - 0.12] * 2, color=col, lw=3, solid_capstyle="round")
            ax.plot([r["loco_min"], r["loco_max"]], [y + 0.12] * 2, color=col, lw=1.4, ls="--")
            ax.scatter(r["puesto_deteccion"], y, s=50, color=col, edgecolor="black", zorder=3)
            pv = r["p_valor"]
            if pd.notna(pv):
                ax.text(len(p) + 0.8, y, f"p={pv:.3f}" + (" *" if pv < 0.05 else ""), va="center", fontsize=8,
                        color="#222" if pv < 0.05 else "#999")
        ax.set_yticks(range(len(p)), p["id"])
        ax.invert_yaxis()
        ax.set_xlim(0.5, len(p) + 3)
        ax.set_xticks(range(1, len(p) + 1))
        ax.set_xlabel("puesto (1 = mejor)")
        s = sens.loc[t]
        uno = s.index[(s == 1).all(axis=1)]
        ax.set_title(f"Pista {t}: nº 1 en todas las variantes de parámetros: {', '.join(uno) or 'ninguno'}",
                     fontsize=11)
        ax.grid(axis="y", visible=False)
    extra = [Line2D([], [], color="#444", lw=3, label="rango con 7 variantes de parámetros"),
             Line2D([], [], color="#444", lw=1.4, ls="--", label="rango quitando una crisis cada vez"),
             Line2D([], [], ls="", marker="o", mfc="#444", mec="black", label="puesto oficial"),
             Line2D([], [], ls="", marker="$p$", color="#444",
                    label="p-valor frente al azar persistente (* < 0,05)")]
    fig.legend(handles=extra, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False)
    primeros = rob[rob["puesto_deteccion"] == 1]
    firmes = primeros[(primeros["param_max"] == 1) & (primeros["loco_max"] == 1)]
    texto = (f"el nº 1 ({', '.join(firmes['pista'] + ': ' + firmes['id'])}) no se mueve con ninguna variante"
             if len(firmes) == len(primeros) else "el nº 1 cambia con alguna variante")
    fig.suptitle(f"Robustez del orden: {texto}; el centro de la tabla sí se mueve", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def baselines_frente(principal: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Para cada referencia trivial: su score, si entra en el MCS y qué detectores del ranking
    principal puntúan por debajo (los que la regla trivial supera)."""
    filas = []
    for r in base.itertuples():
        p = principal[principal["pista"] == r.pista]
        debajo = p[p["score_deteccion"] < r.score_deteccion]
        filas.append({"pista": r.pista, "referencia": r.id, "score": r.score_deteccion,
                      "en_mcs": _marca(r.en_mcs), "detectores_por_debajo": len(debajo),
                      "de": len(p), "cuales": ", ".join(debajo["id"])})
    return pd.DataFrame(filas)


def baselines_legacy(v2: pd.DataFrame) -> pd.DataFrame:
    """Puesto de las cuatro líneas base triviales entre los 12 de v2 con el criterio ADR-003 y con
    el ``rank_medio`` legacy (como la versión anterior de este notebook; ADR-003 §1 lo cita).

    Las líneas base usan la tasa de marcado y la duración de episodio medianas de la pista."""
    from regimenes.evaluacion import ranking as rk

    m = v2.drop(columns=["menu", "familia_tfm", "grupo"], errors="ignore")
    r = rk.rank_detection(m, output_dir=V2, **rk.DETECTION_DEFAULTS)
    partes = []
    for t in PISTAS:
        p = r[r["pista"] == t]
        indice, _ = rk.track_oos_index(t, m, V2)
        partes.append(rk.trivial_baselines(t, indice, rate=float(p["det_marked_rate"].median()),
                                           mean_run=float(p["det_mean_crisis_run"].median()),
                                           min_run=int(rk.DETECTION_DEFAULTS["min_run"])))
    base = pd.concat(partes, ignore_index=True)
    con = rk.rank_detection(pd.concat([m, base], ignore_index=True), output_dir=V2, **rk.DETECTION_DEFAULTS)
    filas_por_pista = con.groupby("pista").size()
    con = con[con["id"].isin(set(base["id"]))].copy()
    con["de"] = con["pista"].map(filas_por_pista)
    return con[["pista", "id", "nivel_etiqueta", "score_deteccion", "puesto_deteccion", "puesto_legacy", "de"]] \
        .reset_index(drop=True)


def comparacion_legacy(v2: pd.DataFrame) -> pd.DataFrame:
    """Spearman entre el puesto ADR-003 y el ``rank_medio`` anterior (solo v2) por pista."""
    from regimenes.evaluacion import ranking as rk

    r = rk.rank_detection(v2.drop(columns=["menu", "familia_tfm", "grupo"], errors="ignore"),
                          output_dir=V2, **rk.DETECTION_DEFAULTS)
    filas = []
    for t, p in r.groupby("pista"):
        filas.append({"pista": t, "spearman": p[["puesto_deteccion", "puesto_legacy"]].corr("spearman").iloc[0, 1],
                      "primero_adr003": p.loc[p["puesto_deteccion"].idxmin(), "id"],
                      "primero_legacy": p.loc[p["puesto_legacy"].idxmin(), "id"]})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# Conclusiones del resumen (texto con cifras calculadas)
# --------------------------------------------------------------------------- #
def hechos(principal: pd.DataFrame, base: pd.DataFrame, pool: pd.DataFrame, criterio: pd.DataFrame | None,
           tablas_t: dict[str, pd.DataFrame], techo: pd.DataFrame) -> dict:
    """Cifras de las cinco conclusiones del resumen ejecutivo, todas leídas de las tablas."""
    h: dict = {}
    for t in PISTAS:
        p = principal[principal["pista"] == t].sort_values("puesto_deteccion")
        b = base[base["pista"] == t].set_index("id")
        top = p.iloc[0]
        mcs = p.dropna(subset=["perdida_mcs"]).sort_values("perdida_mcs")
        vol = float(b.loc["VOL_P90", "score_deteccion"])
        pool_p = p[p["menu"].isin(["P", "R"])]
        mejor_pool = pool_p.iloc[0] if len(pool_p) else None
        tp = techo[techo["pista"] == t].sort_values("score", ascending=False)
        sup = {}
        for base_id in SUPERVISADOS:
            for menu in ("N", "P"):
                fila = tp[tp["id"] == f"{base_id}-{menu}"]
                sup[f"{base_id}-{menu}"] = float(fila["score"].iloc[0]) if len(fila) else np.nan
        nt = tablas_t["n_vs_t"]
        cb = tablas_t["criterio_b"].set_index("pista") if len(tablas_t["criterio_b"]) else None
        h[t] = {
            "primero": top["id"], "score_primero": float(top["score_deteccion"]),
            "n": len(p), "menu_primero": top["menu"],
            "primero_mcs": mcs["id"].iloc[0] if len(mcs) else None,
            "perdida_primero_mcs": float(mcs["perdida_mcs"].iloc[0]) if len(mcs) else np.nan,
            "n_mcs": int(p["en_mcs"].fillna(False).sum()),
            "vol_p90_score": vol, "vol_p90_en_mcs": _marca(b.loc["VOL_P90", "en_mcs"]) == "●",
            "vol_p90_puesto_hipotetico": int((p["score_deteccion"] > vol).sum() + 1),
            "mejor_pool": None if mejor_pool is None else mejor_pool["id"],
            "puesto_mejor_pool": None if mejor_pool is None else int(mejor_pool["puesto_deteccion"]),
            "score_mejor_pool": np.nan if mejor_pool is None else float(mejor_pool["score_deteccion"]),
            "cumplen_a_y_b": (str(criterio.set_index("pista").loc[t, "cumplen_a_y_b"]).replace("|", ", ")
                              if criterio is not None and t in set(criterio["pista"]) else ""),
            "t_mejoran": int((nt.loc[nt["pista"] == t, "delta_score"] > 0).sum()),
            "t_total": int((nt["pista"] == t).sum()),
            "t_percentil_azar": float(cb.loc[t, "percentil_ref_t"]) if cb is not None else np.nan,
            "mejor_supervisado": tp["id"].iloc[0], "score_mejor_supervisado": float(tp["score"].iloc[0]),
            "mejor_no_supervisado": tp["mejor_no_supervisado"].iloc[0],
            "score_mejor_no_supervisado": float(tp["score_mejor"].iloc[0]),
            "supervisados_mejoran": int(tp["mejora_al_mejor"].astype(bool).sum()),
            **{f"score_{k}": v for k, v in sup.items()},
        }
    h["veredicto_pool"] = criterio["veredicto"].iloc[0] if criterio is not None and len(criterio) else "pendiente"
    return h


def conclusiones(h: dict) -> list[str]:
    """Las cinco conclusiones del resumen, redactadas a partir de :func:`hechos`."""
    A, B = h["A"], h["B"]
    c1 = (f"**El primero depende de la pista y del criterio.** Por score de detección encabeza "
          f"{A['primero']} en A ({A['score_primero']:.3f}) y {B['primero']} en B ({B['score_primero']:.3f}); "
          f"por pérdida del MCS el primero es {A['primero_mcs']} en A ({A['perdida_primero_mcs']:.3f}) y "
          f"{B['primero_mcs']} en B ({B['perdida_primero_mcs']:.3f})."
          + (" Todos son del núcleo N/U." if all(menu_de(x) in ("N", "U") for x in
             (A['primero'], B['primero'], A['primero_mcs'], B['primero_mcs'])) else ""))
    c2 = (f"**Los datos no separan a la mayoría.** Dentro del MCS al 10 % quedan {A['n_mcs']} de {A['n']} "
          f"detectores en A y {B['n_mcs']} de {B['n']} en B. La regla trivial `VOL_P90` (un umbral sobre la "
          f"volatilidad) tiene score {A['vol_p90_score']:.3f} en A — sería el puesto "
          f"{A['vol_p90_puesto_hipotetico']} — y queda {'dentro' if A['vol_p90_en_mcs'] else 'fuera'} del MCS; "
          f"en B queda {'dentro' if B['vol_p90_en_mcs'] else 'fuera'} (score {B['vol_p90_score']:.3f}).")
    c3 = (f"**El pool sin comprimir aporta solo en lo formal.** Veredicto del ADR-010 §5: «{h['veredicto_pool']}» "
          f"(cumplen (a) y (b): {A['cumplen_a_y_b']} en A; {B['cumplen_a_y_b']} en B). Pero la mejor variante "
          f"del pool queda {A['puesto_mejor_pool']}.ª en A ({A['mejor_pool']}, {A['score_mejor_pool']:.3f}) y "
          f"{B['puesto_mejor_pool']}.ª en B ({B['mejor_pool']}, {B['score_mejor_pool']:.3f})"
          + (": no desplaza a N." if min(A['puesto_mejor_pool'], B['puesto_mejor_pool']) > 1 else "."))
    c4 = (f"**Comprimir el pool en termómetros (menú T) no sirve.** Con T mejoran {A['t_mejoran']} de "
          f"{A['t_total']} multivariantes en A y {B['t_mejoran']} de {B['t_total']} en B; los bloques económicos "
          f"quedan en el percentil {A['t_percentil_azar']:.0f} (A) y {B['t_percentil_azar']:.0f} (B) de 100 "
          f"repartos al azar. Descartado (ADR-009).")
    n_sup = A['supervisados_mejoran'] + B['supervisados_mejoran']
    c5 = (f"**Ni con etiquetas se gana a lo simple.** {'Ningún supervisado mejora' if n_sup == 0 else f'{n_sup} supervisados mejoran'}"
          f" al mejor no supervisado: en A el mejor es {A['mejor_supervisado']} "
          f"({A['score_mejor_supervisado']:.3f} frente a {A['score_mejor_no_supervisado']:.3f} de "
          f"{A['mejor_no_supervisado']}) y en B {B['mejor_supervisado']} ({B['score_mejor_supervisado']:.3f} "
          f"frente a {B['score_mejor_no_supervisado']:.3f} de {B['mejor_no_supervisado']}). Con el pool P "
          f"rinden peor que con N (D16: {A['score_D16-P']:.2f} frente a {A['score_D16-N']:.2f} en A; "
          f"{B['score_D16-P']:.2f} frente a {B['score_D16-N']:.2f} en B).")
    return [c1, c2, c3, c4, c5]
