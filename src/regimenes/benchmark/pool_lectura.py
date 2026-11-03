"""Lectura para los notebooks de la Fase 2 · pool sin comprimir (ADR-010).

Solo lee lo que escribe :mod:`regimenes.benchmark.pool` en ``results/detectores/pool/`` y lo que
ya está en ``results/benchmark/`` (v2); no ejecuta detectores ni el MCS. Vive aparte de
``pool.py`` porque ese módulo entra en su propia huella de código (``pool.huella_codigo``):
tocarlo con el cálculo en marcha haría que ``ejecutar`` se negara a seguir. Este módulo no entra
en ninguna huella (ni en la de la caché v2 ni en la del notebook 21).

Todo tolera resultados a medias: si falta un fichero devuelve una tabla vacía o ``None`` y el
notebook avisa (como :mod:`regimenes.benchmark.ablaciones_lectura` en la sección 13 del 04).

- :func:`estado`: hechos y pendientes por pista (sin costes ni paneles; para eso, ``pool estado``).
- :func:`leer_tabla`: un CSV de ``pool leer`` (``comparacion_pares``, ``mcs_NUPR_<P>``,
  ``criterio_adr010``…), marcado como desfasado si es anterior al último resultado terminado.
- :func:`comparacion`: criterio (a) del §5 calculado al vuelo con lo terminado (barato, sin MCS).
- :func:`senales`: señal diaria de crisis de variantes del pool y de detectores de v2.
- :func:`delta_ar_b`: serie ΔAR de D14 en la pista B (causal; es la misma que usa el detector).
- :func:`lambdas_d15`, :func:`reajustes`: fichas por reajuste.
- :func:`importancias`: coeficientes (D16) o importancias (D17) del ajuste final, ilustrativos.
- :func:`techo`: §6 sin MCS (score frente al mejor no supervisado).
- :func:`ranking_v3`: ranking de detección (ADR-003) del núcleo N con U (ADR-009 §6–7).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import warnings

import numpy as np
import pandas as pd

from regimenes import rutas
from regimenes.benchmark import pool as bp

SALIDA = bp.SALIDA
V2 = bp.V2
PISTAS = bp.PISTAS
PENDIENTE = "PENDIENTE DE RESULTADOS"
# Menú de los 12 detectores de v2 (ADR-008 §3): los univariantes van en U; el resto, en N.
MENU_V2 = {"D01": "U", "D05": "U", "D06": "U", "D07": "U", "D11": "U",
           "D02": "N", "D03": "N", "D04": "N", "D08": "N", "D09": "N", "D10": "N", "D12": "N"}
BASELINES = ("SIEMPRE_CRISIS", "SIEMPRE_CALMA", "AZAR_IID", "AZAR_PERSISTENTE", "VOL_P90")


def rel(ruta: Path | str) -> str:
    ruta = Path(ruta)
    return ruta.relative_to(rutas.ROOT).as_posix() if ruta.is_relative_to(rutas.ROOT) else ruta.as_posix()


# --------------------------------------------------------------------------- #
# Estado y tablas
# --------------------------------------------------------------------------- #
def ids_esperados(track: str) -> list[str]:
    """Ids de las variantes de la Fase 2 de la pista (``specs_pool``)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [s.detector_id for s in bp.todas_las_specs(track.upper())]


def estado(salida: Path = SALIDA) -> pd.DataFrame:
    """Una fila por pista: variantes terminadas (CSV de métricas presente) y pendientes."""
    salida = Path(salida)
    filas = []
    for track in PISTAS:
        ids = ids_esperados(track)
        hechos = [i for i in ids if bp.ruta_trabajo(salida, track, i).exists()]
        filas.append({"pista": track, "hechos": len(hechos), "esperados": len(ids),
                      "pendientes": "|".join(i for i in ids if i not in hechos)})
    return pd.DataFrame(filas)


def variantes(salida: Path = SALIDA) -> pd.DataFrame:
    """Métricas de las variantes terminadas (``pool.cargar``), indexables por (pista, id)."""
    return bp.cargar(Path(salida))


def _ultimo_resultado(salida: Path) -> float:
    tiempos = [p.stat().st_mtime for p in Path(salida).glob("[AB]_D*.csv")]
    return max(tiempos) if tiempos else 0.0


def leer_tabla(nombre: str, salida: Path = SALIDA, *, index_col: str | None = None) -> pd.DataFrame | None:
    """CSV ``<salida>/<nombre>`` de ``pool leer``, o ``None`` si falta.

    ``attrs['desfasada']`` es ``True`` si el CSV es anterior al último resultado terminado (hay
    que volver a lanzar ``python -m regimenes.benchmark.pool leer``)."""
    ruta = Path(salida) / (nombre if nombre.endswith(".csv") else f"{nombre}.csv")
    if not ruta.exists():
        return None
    tabla = pd.read_csv(ruta, index_col=index_col)
    tabla.attrs["ruta"] = ruta
    tabla.attrs["desfasada"] = ruta.stat().st_mtime < _ultimo_resultado(Path(salida))
    return tabla


def aviso_tabla(nombre: str, tabla: pd.DataFrame | None) -> str | None:
    """Texto de aviso de una tabla de :func:`leer_tabla` (``None`` si está al día)."""
    if tabla is None:
        return f"{PENDIENTE}: falta {nombre} (python -m regimenes.benchmark.pool leer)."
    if tabla.attrs.get("desfasada"):
        return (f"AVISO: {rel(tabla.attrs['ruta'])} es anterior al último resultado; vuelve a lanzar "
                "python -m regimenes.benchmark.pool leer.")
    return None


def comparacion(salida: Path = SALIDA, v2: Path = V2, ids: Iterable[str] | None = None) -> pd.DataFrame:
    """Criterio (a) del §5 con lo terminado (``pool.comparacion_pares``), opcionalmente filtrado."""
    columnas = ["pista", "id", "menu", "par_v2", "score", "score_par", "delta_score",
                "a_mejora_par", "cuenta_para_veredicto"]
    tabla = variantes(salida)
    comp = bp.comparacion_pares(tabla, v2) if len(tabla) else pd.DataFrame()
    if comp.empty:
        return pd.DataFrame(columns=columnas)
    if ids is not None:
        comp = comp[comp["id"].isin(list(ids))]
    return comp.reset_index(drop=True)


def criterio_provisional(comp: pd.DataFrame) -> pd.DataFrame:
    """Solo la parte (a) del veredicto del §5, por pista, mientras falta el MCS."""
    filas = []
    for track in PISTAS:
        c = comp[(comp["pista"] == track) & comp["cuenta_para_veredicto"].astype(bool)] if len(comp) else comp
        filas.append({"pista": track, "candidatas": "|".join(c["id"]) if len(c) else "",
                      "a_mejoran_par": "|".join(c.loc[c["a_mejora_par"].astype(bool), "id"]) if len(c) else "",
                      "b_en_mcs": PENDIENTE})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# Series diarias
# --------------------------------------------------------------------------- #
def senales(track: str, ids_pool: Sequence[str] = (), ids_v2: Sequence[str] = (), *,
            salida: Path = SALIDA, v2: Path = V2) -> pd.DataFrame:
    """Señal diaria de crisis (``state == n_states − 1``) en el OOS, solo de lo que existe.

    Las variantes del pool sin panel se omiten; las columnas de v2 llevan su id (``D10``)."""
    from regimenes.evaluacion import mcs

    track = track.upper()
    partes = []
    hechos = [i for i in ids_pool if bp.ruta_trabajo(salida, track, i).with_suffix(".parquet").exists()
              and bp.ruta_trabajo(salida, track, i).exists()]
    if hechos:
        pan = mcs.cargar_paneles(track, Path(salida), hechos)
        partes.append(mcs.senales_crisis(pan, n_states=mcs.n_states_metricas(track, Path(salida), hechos)))
    if ids_v2:
        paneles = Path(v2) / "panels"
        pan = mcs.cargar_paneles(track, paneles, list(ids_v2))
        partes.append(mcs.senales_crisis(pan, n_states=mcs.n_states_metricas(track, paneles, list(ids_v2))))
    return pd.concat(partes, axis=1) if partes else pd.DataFrame()


def dias_en_crisis(sen: pd.DataFrame, track: str) -> pd.DataFrame:
    """Días marcados, fracción del OOS y fracción dentro y fuera de las ventanas de crisis."""
    from regimenes.evaluacion import ranking as rk

    dentro = pd.Series(False, index=sen.index)
    for a, b in rk.track_crisis_windows(track).values():
        dentro |= (sen.index >= pd.Timestamp(a)) & (sen.index <= pd.Timestamp(b))
    return pd.DataFrame({
        "dias_crisis": sen.sum().astype(int),
        "fraccion_oos": sen.mean(),
        "cobertura_dias_crisis": sen[dentro].mean(),
        "marcados_en_calma": sen[~dentro].mean(),
    })


def delta_ar_b() -> pd.Series:
    """ΔAR de D14 sobre el índice de la pista B (panel R; ADR-010 §2.2 y §3).

    Es causal (cada día usa solo filas ``<= t``), así que coincide con la que ve el detector en el
    walk-forward. Cuesta unos segundos (un ``eigvalsh`` 22 × 22 por día)."""
    from regimenes.detectores.pool.absorption_ratio import AbsorptionRatio
    from regimenes.detectores.pool.pool_crudo import D14_PARAMS
    from regimenes.features import rentabilidades as rent

    R = rent.leer_panel_r()[list(rent.columnas_r())]
    return AbsorptionRatio(**D14_PARAMS).serie_delta_ar(R).rename("delta_ar")


# --------------------------------------------------------------------------- #
# Reajustes
# --------------------------------------------------------------------------- #
def reajustes(track: str, detector_id: str, salida: Path = SALIDA) -> pd.DataFrame:
    """``reajustes/<P>_<id>.csv`` (vacío si falta)."""
    ruta = bp.ruta_reajustes(salida, track, detector_id)
    if not ruta.exists():
        return pd.DataFrame()
    r = pd.read_csv(ruta)
    if "fin_train" in r:
        r["fin_train"] = pd.to_datetime(r["fin_train"])
    return r


def lambdas_d15(track: str, salida: Path = SALIDA) -> pd.DataFrame:
    """λ elegida por reajuste de D15-U, con la discrepancia con la regla del paper."""
    r = reajustes(track, "D15-U", salida)
    cols = [c for c in ("reajuste", "fin_train", "lambda_elegida", "discrepa_regla_paper",
                        "n_historia", "n_validacion") if c in r]
    return r[cols] if len(r) else r


def informe_d18(track: str, salida: Path = SALIDA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frecuencia por columna y por bloque de las columnas con peso > 0 (``pool.informe_d18``)."""
    return bp.informe_d18(track, Path(salida))


# --------------------------------------------------------------------------- #
# Supervisados
# --------------------------------------------------------------------------- #
def importancias(track: str, detector_id: str, salida: Path = SALIDA) -> pd.Series | None:
    """Coeficientes (D16) o importancias (D17) del ajuste final con toda la muestra.

    Es el mismo ajuste final que hace ``pool.evaluar_pool`` (todas las crisis terminadas): una
    lectura **ilustrativa** de qué usa el modelo, no el modelo de cada reajuste. ``None`` si falta
    la entrada de la pista (``pool preparar``). D17 tarda unos segundos (500 árboles, un hilo)."""
    track = track.upper()
    prep = bp.leer_preparacion(salida, track)
    if not prep or not bp.ruta_entrada(salida, track).exists():
        return None
    panel, indice = bp.leer_entrada(Path(salida), track, prep)
    spec = bp.spec_de(track, detector_id)
    X = bp.matriz(track, spec, panel, indice)
    modelo = spec.factory()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo.fit(X)
    base = getattr(modelo, "base", modelo)
    clf = getattr(base, "_clf", None)
    columnas = list(getattr(base, "_columnas", []))
    if clf is None:
        return pd.Series(0.0, index=columnas, name=detector_id)
    valores = clf.coef_[0] if hasattr(clf, "coef_") else clf.feature_importances_
    return pd.Series(valores, index=columnas, name=detector_id)


def techo(salida: Path = SALIDA, v2: Path = V2) -> pd.DataFrame:
    """§6 sin MCS: cada supervisado terminado frente al mejor no supervisado de su pista (v2 + pool)."""
    from regimenes.benchmark import menus_oficial as mo

    tabla = variantes(salida)
    filas = []
    for track in PISTAS:
        t = tabla[tabla["pista"] == track] if len(tabla) else tabla
        if not len(t):
            continue
        sup = t[t["id"].map(bp.es_supervisado)]
        if not len(sup):
            continue
        scores = {d: float(mo.fila_n_v2(track, d, v2).loc[0, "score_deteccion"]) for d in MENU_V2}
        scores.update({r.id: float(r.score_deteccion) for r in t.itertuples() if not bp.es_supervisado(r.id)})
        mejor = max(scores, key=scores.get)
        for r in sup.itertuples():
            filas.append({"pista": track, "id": r.id, "menu": r.menu, "score": float(r.score_deteccion),
                          "mejor_no_supervisado": mejor, "score_mejor": scores[mejor],
                          "mejora_al_mejor": bool(float(r.score_deteccion) > scores[mejor])})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# Ranking v3
# --------------------------------------------------------------------------- #
def ranking_v3(salida: Path = SALIDA, v2: Path = V2, *, menus: Sequence[str] = ("N", "U")) -> pd.DataFrame:
    """Ranking de detección (ADR-003) del núcleo N con los univariantes en U (ADR-009 §6–7).

    Los 12 detectores de v2 (métricas versionadas) más las variantes no supervisadas del pool
    terminadas cuyo menú esté en ``menus`` (por defecto solo D15-U). Con ``menus=("N", "U", "P",
    "R")`` sale la lectura ampliada con el pool, que no es el ranking principal."""
    from regimenes.benchmark import load_metrics
    from regimenes.evaluacion import ranking as rk

    m = load_metrics(Path(v2), require_current=False).copy()
    m["menu"] = m["id"].map(MENU_V2).fillna("N")
    tabla = variantes(salida)
    if len(tabla):
        extra = tabla[~tabla["id"].map(bp.es_supervisado) & tabla["menu"].isin(list(menus))].copy()
        if len(extra):
            if "familia" not in extra:
                extra["familia"] = "Fase 2 (ADR-010)"
            m = pd.concat([m, extra], ignore_index=True)
    m = m[m["menu"].isin(list(menus))]
    out = rk.rank_detection(m, output_dir=Path(v2), **rk.DETECTION_DEFAULTS)
    cols = ["pista", "puesto_deteccion", "id", "menu", *[c for c in rk.RANKING_COLUMNS[3:12]]]
    return out[[c for c in cols if c in out]]
