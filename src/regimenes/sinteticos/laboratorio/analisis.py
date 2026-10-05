"""Agregados del laboratorio para el notebook 17 (hipotesis H1-H4 del pre-registro).

Todas las funciones reciben la tabla de filas (``ejecucion.consolidar``): una fila
por (pista, celda, trayectoria, detector). Los intervalos son bootstrap por
percentiles remuestreando TRAYECTORIAS (la unidad independiente).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

METRICA = "score_deteccion"


def ic_media(valores, n_boot: int = 1000, semilla: int = 42, alfa: float = 0.05) -> tuple[float, float, float]:
    """Media e IC bootstrap por percentiles (NaN se descartan)."""
    v = np.asarray(valores, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(semilla)
    medias = rng.choice(v, size=(int(n_boot), len(v)), replace=True).mean(axis=1)
    return (float(v.mean()), float(np.quantile(medias, alfa / 2)), float(np.quantile(medias, 1 - alfa / 2)))


def resumen_celdas(filas: pd.DataFrame, metrica: str = METRICA, n_boot: int = 1000) -> pd.DataFrame:
    """Media e IC de ``metrica`` por (pista, generador, duracion, intensidad, detector)."""
    claves = ["pista", "generador", "duracion", "intensidad", "id"]
    out = []
    for k, g in filas.groupby(claves, sort=True):
        m, lo, hi = ic_media(g[metrica], n_boot)
        out.append({**dict(zip(claves, k)), "media": m, "ic_lo": lo, "ic_hi": hi, "n": int(g[metrica].notna().sum())})
    return pd.DataFrame(out)


def ranking_laboratorio(filas: pd.DataFrame, pista: str, metrica: str = METRICA) -> pd.DataFrame:
    """Score medio de cada detector sobre TODAS las celdas de la pista (media de celdas) y puesto."""
    f = filas[filas.pista == pista]
    por_celda = f.groupby(["celda", "id"])[metrica].mean().unstack("id")
    tabla = por_celda.mean().rename("score_lab").to_frame()
    tabla["puesto_lab"] = tabla["score_lab"].rank(ascending=False, method="min").astype(int)
    return tabla.sort_values("puesto_lab")


def h1_recuperacion(filas: pd.DataFrame, duracion: int = 252, intensidad: float = 1.0) -> pd.DataFrame:
    """H1: por pista, generador y detector, en la celda facil: lift medio, recall medio y si
    supera el azar (lift > 1) y detecta >= la mitad de las crisis."""
    f = filas[(filas.duracion == duracion) & (filas.intensidad == intensidad)]
    t = f.groupby(["pista", "generador", "id"]).agg(
        lift=("det_lift_precision", "mean"), recall=("det_event_recall", "mean"), score=(METRICA, "mean"))
    t["cumple"] = (t["lift"] > 1) & (t["recall"] >= 0.5)
    return t.reset_index()


def h2_degradacion(filas: pd.DataFrame, n_boot: int = 1000, semilla: int = 42) -> pd.DataFrame:
    """H2: efecto sobre el score de cada detector de acortar la crisis (D max -> D min) y de
    atenuarla (lambda max -> lambda min), con IC bootstrap.

    - Acortar: las duraciones usan trayectorias distintas, asi que es una diferencia de medias por
      (generador, intensidad) promediada; el IC remuestrea por separado las trayectorias de cada
      grupo. Mezcla dificultad y prevalencia (la tasa base cambia con la duracion).
    - Atenuar: diseno pareado (misma trayectoria bruta con las dos intensidades); se remuestrean
      las diferencias por trayectoria. Las trayectorias tienen semillas independientes, pero dentro
      de un generador comparten el mismo ajuste del 15 (la incertidumbre del ajuste no se recoge).
    """
    rng = np.random.default_rng(semilla)
    out = []
    dmax, dmin = filas.duracion.max(), filas.duracion.min()
    imax, imin = filas.intensidad.max(), filas.intensidad.min()
    for (pista, det), g in filas.groupby(["pista", "id"]):
        grupos = []
        for _, gg in g.groupby(["generador", "intensidad"]):
            a = gg.loc[gg.duracion == dmin, METRICA].dropna().to_numpy()
            b = gg.loc[gg.duracion == dmax, METRICA].dropna().to_numpy()
            if len(a) and len(b):
                grupos.append((a, b))
        if grupos:
            acortar = float(np.mean([a.mean() - b.mean() for a, b in grupos]))
            boot = [np.mean([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for a, b in grupos])
                    for _ in range(int(n_boot))]
            ac_lo, ac_hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
        else:
            acortar = ac_lo = ac_hi = np.nan
        p = g.pivot_table(index=["generador", "duracion", "path_id"], columns="intensidad", values=METRICA)
        dif_int = (p[imin] - p[imax]).to_numpy() if {imin, imax} <= set(p.columns) else np.array([])
        m, lo, hi = ic_media(dif_int, n_boot)
        out.append({"pista": pista, "id": det, "efecto_acortar": acortar, "acortar_ic_lo": ac_lo,
                    "acortar_ic_hi": ac_hi, "efecto_atenuar": m, "atenuar_ic_lo": lo, "atenuar_ic_hi": hi})
    return pd.DataFrame(out)


def h3_concordancia(ranking_lab: pd.DataFrame, ranking_real: pd.DataFrame, n_perm: int = 10000,
                    semilla: int = 42) -> dict[str, float]:
    """H3: Spearman entre score de laboratorio y score real ADR-003 (detectores comunes) y p
    bilateral por permutacion."""
    real = ranking_real.set_index("id")["score_deteccion"]
    comun = ranking_lab.index.intersection(real.index)
    x, y = ranking_lab.loc[comun, "score_lab"].to_numpy(), real.loc[comun].to_numpy()
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(semilla)
    nulos = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(int(n_perm))])
    return {"n_detectores": int(len(comun)), "rho": rho, "p_perm": float((np.abs(nulos) >= abs(rho)).mean())}


def _diferencias_generador(f: pd.DataFrame, gen: str) -> pd.Series:
    """Por detector: score medio en ``gen`` menos media de sus scores medios en los otros generadores."""
    por_gen = f.groupby(["id", "generador"])[METRICA].mean().unstack("generador")
    otros = [o for o in por_gen.columns if o != gen]
    return por_gen[gen] - por_gen[otros].mean(axis=1)


def h4_circularidad(filas: pd.DataFrame, misma_familia: dict[str, list[str]], n_boot: int = 1000,
                    semilla: int = 42) -> pd.DataFrame:
    """H4: para cada (generador, detector «de casa»).

    - ``ventaja_casa`` (pre-registrada): score en su generador menos su score medio en los otros.
    - ``ventaja_relativa`` (EXPLORATORIA, anadida tras ver los resultados; ver yaml): la misma
      diferencia menos su media en los detectores que no juegan en casa en ese generador, porque un
      generador puede ser mas facil para todos.

    IC bootstrap remuestreando ``path_id`` dentro de cada (pista, generador, duracion, intensidad), con
    el MISMO remuestreo para todos los detectores (las trayectorias son comunes a todos ellos).
    """
    rng = np.random.default_rng(semilla)
    out = []
    for pista, f in filas.groupby("pista"):
        celdas_paths = {k: g["path_id"].unique() for k, g in f.groupby("celda")}

        def remuestra() -> pd.DataFrame:
            piezas = []
            for celda, ids in celdas_paths.items():
                elegidos = rng.choice(ids, len(ids))
                sub = f[f.celda == celda].set_index("path_id")
                piezas.append(sub.loc[elegidos].reset_index())
            return pd.concat(piezas, ignore_index=True)

        boots = [remuestra() for _ in range(int(n_boot))]
        for gen, dets in (misma_familia or {}).items():
            if gen not in set(f.generador):
                continue
            ajenos_de = lambda d_todos: [d for d in d_todos.index if d not in set(dets)]  # noqa: E731
            d_obs = _diferencias_generador(f, gen)
            d_boot = [_diferencias_generador(b, gen) for b in boots]
            for det in dets:
                if det not in d_obs.index:
                    continue
                casa_b = np.array([d[det] for d in d_boot])
                rel_b = np.array([d[det] - d.loc[ajenos_de(d)].mean() for d in d_boot])
                out.append({
                    "pista": pista, "generador": gen, "id": det,
                    "ventaja_casa": float(d_obs[det]),
                    "ic_lo": float(np.quantile(casa_b, 0.025)), "ic_hi": float(np.quantile(casa_b, 0.975)),
                    "ventaja_resto": float(d_obs.loc[ajenos_de(d_obs)].mean()),
                    "ventaja_relativa": float(d_obs[det] - d_obs.loc[ajenos_de(d_obs)].mean()),
                    "rel_ic_lo": float(np.quantile(rel_b, 0.025)), "rel_ic_hi": float(np.quantile(rel_b, 0.975)),
                })
    return pd.DataFrame(out)


__all__ = ["ic_media", "resumen_celdas", "ranking_laboratorio", "h1_recuperacion", "h2_degradacion",
           "h3_concordancia", "h4_circularidad"]
