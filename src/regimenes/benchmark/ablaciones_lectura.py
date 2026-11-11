"""Lectura de las ablaciones y controles del menú T (ADR-008 §5, puntos 4–6; plan 2.5).

El punto 4 (relevancia) ya está en la sección 10 del notebook 04; aquí van el 5 y el 6.

Solo lee. Los trabajos los ejecuta :mod:`regimenes.benchmark.ablaciones` (reajuste ANUAL,
``step = 252``, en todas las variantes) y deja en ``results/detectores/ablaciones/`` un CSV
por trabajo ``<P>_<D>_<VARIANTE>.csv`` con su panel OOS ``<P>_<D>_<VARIANTE>.parquet``
(``state``, ``p_crisis``) y ``variantes_<P>.json`` con la composición de cada variante.
Variantes: ``REF-N`` y ``REF-T`` (los menús N y T con reajuste anual), ``SIN-T<k>`` (menú T
sin el bloque ``T<k>``; 7 en A y 10 en B), ``MEDIA`` (media simple de las columnas
orientadas en vez de la PC1) y ``AZAR-000`` … ``AZAR-099`` (100 reparticiones aleatorias de
columnas a bloques). Detectores: los 6 multivariantes del menú T.

Funciones:

- :func:`cargar`: tabla larga de todos los CSV, con el esquema validado y aviso de los
  trabajos que faltan.
- :func:`validez_atajo`: signo de ``REF-T − REF-N`` frente al signo oficial (reajuste
  mensual, ``results/detectores/menus/comparacion_N_T.csv``). Es el criterio de validez del
  atajo anual: si el signo no se reproduce, el resto se lee con cautela.
- :func:`criterio_b`: criterio (b) del ADR-008 §6 (mediana de los 6 scores con ``REF-T``
  frente al percentil 90 de las 100 medianas ``AZAR``).
- :func:`tabla_sin_bloque` y :func:`tabla_media`: quitar un bloque cada vez y media simple
  frente a PCA, en ``score_deteccion`` frente a ``REF-T``.
- :func:`mcs_por_menu`: punto 6 (MCS dentro de cada menú con los baselines) sobre los
  paneles OFICIALES (reajuste mensual), no sobre los del atajo.
- :func:`escribir_tablas`: la única función que escribe (escritura atómica).

**Criterio (b) y trabajos incompletos (errata del ADR-008 §5.1).** Una semilla ``AZAR``
entra en la distribución solo si tiene los 6 detectores con ``score_deteccion``. Una semilla
incompleta sale entera del cálculo y no se repone. (b) exige al menos :data:`MIN_AZAR`
(95) semillas completas y ``REF-T`` con los 6 detectores. Si falta algo de eso, (b) no se
puede evaluar y **cuenta como no cumplido** (``veredicto_b`` = «no evaluable (cuenta como
no cumplido)», ``b_supera_p90`` = False). Si se puede evaluar, «cumple» si la mediana de
``REF-T`` supera estrictamente al p90 de las completas y «no cumple» en otro caso.
``definitivo`` = ``REF-T`` completo y ≥ 95 completas. Las columnas ``p90_cota_baja`` y
``p90_cota_alta`` son solo **sensibilidad**, sin peso en el veredicto: el p90 de las 100 con
los scores que faltan rellenos con el mínimo o el máximo que ese detector dio en las
completas.

**Percentil 90.** ``numpy.quantile(…, 0.9)`` con interpolación lineal (con 100 medianas,
entre la 90.ª y la 91.ª ordenadas). El criterio exige superarlo **estrictamente**: un empate
no cumple. El percentil que ocupa ``REF-T`` es el de rango medio:
100 · (n_menores + n_iguales / 2) / n.

Uso (desde la raíz del repo)::

    python -m regimenes.benchmark.ablaciones_lectura          # MCS por menú + tablas
    python -m regimenes.benchmark.ablaciones_lectura --sin-mcs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from regimenes import rutas
from regimenes.benchmark.cache import _write_atomic

SALIDA = rutas.RESULTS_DETECTORES / "ablaciones"
MENUS = rutas.RESULTS_DETECTORES / "menus"
V2 = rutas.RESULTS_BENCHMARK
OFICIAL = MENUS / "comparacion_N_T.csv"
VARA = rutas.RESULTS_BENCHMARK / "vara" / "mcs_diario.csv"

DETECTORES = ("D02", "D03", "D04", "D08", "D09", "D10")
PISTAS = ("A", "B")
N_AZAR = 100
MIN_AZAR = 95           # semillas AZAR completas que exige (b) (errata del ADR-008 §5.1)
TIPOS = ("ref", "sin_bloque", "media", "azar")
COLUMNAS = (
    "base", "pista", "variante", "tipo", "bloque_quitado", "semilla", "step",
    "score_deteccion", "det_precision", "det_event_recall", "det_n_detectados",
    "det_n_eventos", "n_oos", "elapsed_seconds",
)
# Vara de medir definitiva (ADR-008 §4; notebook 05 §10): las mismas opciones que mcs_diario.csv.
MCS_KW = {"lam": 1.0, "alpha": 0.10, "estadistico": "T_max", "metodo": "estacionario",
          "n_boot": 5000, "seed": 0, "remuestreo": "razon"}
_NOMBRE = re.compile(r"^(?P<pista>[AB])_(?P<base>D\d\d)_(?P<variante>.+)$")
# Consolidado versionado: una fila por trabajo, con las mismas columnas que los CSV sueltos.
# Los CSV sueltos ``<P>_<D>_<VARIANTE>.csv`` los escribe ``ablaciones.py`` (son su marca de
# «trabajo hecho») y quedan solo en local (.gitignore); en un clon limpio se lee este fichero.
TRABAJOS = "trabajos.csv"


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #
def nombre_corto(bloque: str) -> str:
    """``T1b_rentabilidad_diaria`` → ``T1b`` (y ``T1b`` se queda igual)."""
    return str(bloque).split("_", 1)[0]


def variantes_esperadas(pista: str) -> list[str]:
    """Variantes del diseño completo de la pista (``REF-N``, ``REF-T``, ``SIN-T<k>`` por bloque
    de ``configs/bloques.yaml``, ``MEDIA`` y las 100 ``AZAR``).

    No se leen de ``variantes_<P>.json``: ese fichero describe lo que se ha lanzado (puede
    ser un subconjunto de prueba) y aquí se cuenta lo que falta frente al diseño.
    """
    from regimenes.features.bloques import cargar_bloques

    sin = [f"SIN-{nombre_corto(b)}" for b in cargar_bloques(pista.upper())]
    return ["REF-N", "REF-T", *sin, "MEDIA", *[f"AZAR-{i:03d}" for i in range(N_AZAR)]]


def variantes_lanzadas(pista: str, salida: Path = SALIDA) -> list[str] | None:
    """Variantes de ``variantes_<P>.json`` (``None`` si no existe o no se reconoce)."""
    ruta = Path(salida) / f"variantes_{pista.upper()}.json"
    if not ruta.exists():
        return None
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        datos = datos["variantes"] if isinstance(datos, dict) and "variantes" in datos else datos
        return [str(k) for k in datos] if isinstance(datos, dict) else [str(v) for v in datos]
    except (ValueError, KeyError, TypeError):
        return None


def trabajos_faltantes(tabla: pd.DataFrame, pistas: tuple[str, ...] = PISTAS,
                       variantes: dict[str, list[str]] | None = None) -> list[str]:
    """``<P>_<D>_<VARIANTE>`` del diseño (o de ``variantes``) que no están en ``tabla``."""
    variantes = variantes or {}
    hechos = set(zip(tabla["pista"], tabla["base"], tabla["variante"])) if len(tabla) else set()
    return [f"{p}_{d}_{v}" for p in pistas for v in variantes.get(p, variantes_esperadas(p))
            for d in DETECTORES if (p, d, v) not in hechos]


def _leer_trabajo(ruta: Path) -> pd.DataFrame:
    fila = pd.read_csv(ruta)
    faltan = [c for c in COLUMNAS if c not in fila.columns]
    if faltan:
        raise ValueError(f"{ruta.name}: faltan columnas del esquema {faltan}")
    if len(fila) != 1:
        raise ValueError(f"{ruta.name}: {len(fila)} filas; se espera una por trabajo")
    m = _NOMBRE.match(ruta.stem)
    if m is None:
        raise ValueError(f"{ruta.name}: nombre fuera del patrón <P>_<D>_<VARIANTE>.csv")
    for campo in ("pista", "base", "variante"):
        if str(fila.loc[0, campo]) != m[campo]:
            raise ValueError(f"{ruta.name}: {campo} = {fila.loc[0, campo]!r} no casa con el nombre")
    if fila.loc[0, "tipo"] not in TIPOS:
        raise ValueError(f"{ruta.name}: tipo {fila.loc[0, 'tipo']!r} fuera de {TIPOS}")
    return fila


def _rutas_trabajo(salida: Path) -> list[Path]:
    return sorted(salida.glob("[AB]_D[0-9][0-9]_*.csv")) if salida.exists() else []


def _tabla_trabajos(salida: Path) -> pd.DataFrame:
    """Todos los trabajos de ``salida``: de los CSV sueltos o, si no hay, de :data:`TRABAJOS`."""
    rutas_csv = _rutas_trabajo(salida)
    if rutas_csv:
        return pd.concat([_leer_trabajo(r) for r in rutas_csv], ignore_index=True)
    consolidado = salida / TRABAJOS
    if not consolidado.is_file():
        return pd.DataFrame(columns=list(COLUMNAS))
    tabla = pd.read_csv(consolidado)
    faltan = [c for c in COLUMNAS if c not in tabla.columns]
    if faltan:
        raise ValueError(f"{TRABAJOS}: faltan columnas del esquema {faltan}")
    malos = sorted(set(tabla["tipo"]) - set(TIPOS))
    if malos:
        raise ValueError(f"{TRABAJOS}: tipos {malos} fuera de {TIPOS}")
    return tabla


def cargar(salida: Path = SALIDA, *, pistas: tuple[str, ...] = PISTAS, avisar: bool = True,
           variantes: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Tabla larga (una fila por trabajo) de los CSV de ``salida``.

    Lee los CSV sueltos por trabajo si existen (ejecución local) y, si no, el consolidado
    :data:`TRABAJOS` (clon limpio), que escribe :func:`escribir_tablas`.

    Valida el esquema pactado (columnas de :data:`COLUMNAS`, una fila, nombre de fichero
    coherente con ``pista``, ``base`` y ``variante``, ``tipo`` conocido) y falla si hay
    trabajos repetidos. Avisa (``warnings``) de los trabajos del diseño completo que faltan
    (o de ``variantes``, por pista, si se pasa), de variantes fuera del diseño y de si
    conviven varios ``step``. ``attrs["faltan"]`` lleva la lista completa y
    ``attrs["fuera_de_diseno"]`` las variantes ajenas.
    """
    salida = Path(salida)
    tabla = _tabla_trabajos(salida)
    if len(tabla):
        tabla = tabla[tabla["pista"].isin(pistas)].reset_index(drop=True)
    repetidos = tabla.duplicated(["pista", "base", "variante"], keep=False)
    if repetidos.any():
        raise ValueError(f"trabajos repetidos: {tabla.loc[repetidos, ['pista', 'base', 'variante']].values}")
    faltan = trabajos_faltantes(tabla, pistas, variantes)
    tabla.attrs["faltan"] = faltan
    ajenas = sorted({f"{p}_{v}" for p in pistas
                     for v in set(tabla.loc[tabla["pista"] == p, "variante"]) | set(variantes_lanzadas(p, salida) or [])
                     if v not in (variantes or {}).get(p, variantes_esperadas(p))})
    tabla.attrs["fuera_de_diseno"] = ajenas
    if avisar and ajenas:
        warnings.warn(f"ablaciones: variantes fuera del diseño {ajenas[:5]}", stacklevel=2)
    if avisar and faltan:
        warnings.warn(f"ablaciones: faltan {len(faltan)} trabajos (p. ej. {faltan[:5]})", stacklevel=2)
    pasos = sorted(tabla["step"].dropna().unique().tolist()) if len(tabla) else []
    if avisar and len(pasos) > 1:
        warnings.warn(f"ablaciones: conviven varios step {pasos}; el diseño usa uno solo (252)",
                      stacklevel=2)
    return tabla


def _scores(tabla: pd.DataFrame, pista: str, variante: str) -> pd.Series:
    """``score_deteccion`` por detector de una variante (solo los presentes y no nulos)."""
    t = tabla[(tabla["pista"] == pista) & (tabla["variante"] == variante)]
    return t.set_index("base")["score_deteccion"].astype(float).dropna()


def _pistas(tabla: pd.DataFrame) -> list[str]:
    return [p for p in PISTAS if p in set(tabla["pista"])]


# --------------------------------------------------------------------------- #
# Criterio (b)
# --------------------------------------------------------------------------- #
def medianas_azar(tabla: pd.DataFrame, pista: str) -> pd.DataFrame:
    """Por variante ``AZAR``: nº de detectores con score y mediana (NaN si incompleta)."""
    t = tabla[(tabla["pista"] == pista) & (tabla["tipo"] == "azar")]
    filas = []
    for variante, g in t.groupby("variante"):
        s = g.set_index("base")["score_deteccion"].astype(float).reindex(DETECTORES)
        completa = bool(s.notna().all())
        filas.append({"variante": variante, "n_detectores": int(s.notna().sum()), "completa": completa,
                      "mediana": float(s.median()) if completa else np.nan})
    return pd.DataFrame(filas, columns=["variante", "n_detectores", "completa", "mediana"])


def percentil_rango_medio(valor: float, muestra: np.ndarray) -> float:
    """Percentil (0–100) que ocupa ``valor`` en ``muestra``, con los empates a medias."""
    muestra = np.asarray(muestra, dtype=float)
    if not len(muestra) or np.isnan(valor):
        return np.nan
    return float(100 * ((muestra < valor).sum() + 0.5 * (muestra == valor).sum()) / len(muestra))


def _p90_cota(tabla: pd.DataFrame, pista: str, extremo: str) -> float:
    """p90 de las 100 medianas con los scores que faltan rellenos con el min o max del detector."""
    t = tabla[(tabla["pista"] == pista) & (tabla["tipo"] == "azar")]
    if t.empty:
        return np.nan
    m = t.pivot_table(index="variante", columns="base", values="score_deteccion", aggfunc="first")
    m = m.reindex(index=[f"AZAR-{i:03d}" for i in range(N_AZAR)], columns=list(DETECTORES))
    completas = m.dropna()
    if completas.empty:
        return np.nan
    relleno = completas.min() if extremo == "min" else completas.max()
    return float(np.quantile(m.fillna(relleno).median(axis=1).to_numpy(), 0.9))


def criterio_b(tabla: pd.DataFrame) -> pd.DataFrame:
    """Criterio (b) del ADR-008 §6 por pista, con las reglas de la errata §5.1.

    Columnas: ``n_ref_t`` (detectores de ``REF-T`` con score), ``mediana_ref_t`` (NaN si no
    están los 6), ``n_azar_completas`` / ``n_azar_incompletas`` / ``n_azar_ausentes`` (de las
    100), ``azar_incompletas``, ``p90_azar`` (sobre las completas), ``percentil_ref_t``,
    ``evaluable``, ``b_supera_p90`` (mediana ``REF-T`` > p90, estricto; False si no es
    evaluable), ``veredicto_b``, ``definitivo`` y, solo como sensibilidad,
    ``p90_cota_baja`` / ``p90_cota_alta``. Ver el docstring del módulo.
    """
    filas = []
    for pista in _pistas(tabla):
        ref = _scores(tabla, pista, "REF-T").reindex(DETECTORES)
        n_ref = int(ref.notna().sum())
        med_ref = float(ref.median()) if n_ref == len(DETECTORES) else np.nan
        az = medianas_azar(tabla, pista)
        comp = az.loc[az["completa"], "mediana"].to_numpy(dtype=float)
        p90 = float(np.quantile(comp, 0.9)) if len(comp) else np.nan
        evaluable = n_ref == len(DETECTORES) and len(comp) >= MIN_AZAR
        supera = bool(evaluable and med_ref > p90)
        if not evaluable:
            veredicto = "no evaluable (cuenta como no cumplido)"
        else:
            veredicto = "cumple" if supera else "no cumple"
        filas.append({
            "pista": pista, "n_ref_t": n_ref, "mediana_ref_t": med_ref,
            "n_azar_completas": len(comp),
            "n_azar_incompletas": int((~az["completa"]).sum()),
            "n_azar_ausentes": N_AZAR - len(az),
            "azar_incompletas": "|".join(az.loc[~az["completa"], "variante"]),
            "p90_azar": p90, "percentil_ref_t": percentil_rango_medio(med_ref, comp),
            "evaluable": evaluable, "b_supera_p90": supera, "veredicto_b": veredicto,
            "definitivo": evaluable,
            "p90_cota_baja": _p90_cota(tabla, pista, "min"),
            "p90_cota_alta": _p90_cota(tabla, pista, "max"),
        })
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# Quitar un bloque, media simple y validez del atajo
# --------------------------------------------------------------------------- #
def tabla_sin_bloque(tabla: pd.DataFrame) -> pd.DataFrame:
    """Cambio de ``score_deteccion`` al quitar cada bloque frente a ``REF-T``.

    Una fila por pista × bloque: ``delta_<D>`` = score(SIN-T<k>) − score(REF-T) por
    detector, ``mediana_delta`` (mediana de los deltas disponibles), ``delta_mediana``
    (mediana de los 6 scores sin el bloque − la de ``REF-T``; NaN si falta alguno),
    ``n_mejoran`` / ``n_detectores`` y ``efecto``: «estorba» si la mediana de los deltas es
    > 0 (quitarlo mejora), «ayuda» si es < 0 y «neutro» si es 0. Ordenada de más estorbo a
    más ayuda dentro de cada pista.
    """
    filas = []
    for pista in _pistas(tabla):
        ref = _scores(tabla, pista, "REF-T").reindex(DETECTORES)
        t = tabla[(tabla["pista"] == pista) & (tabla["tipo"] == "sin_bloque")]
        for variante, g in t.groupby("variante"):
            s = g.set_index("base")["score_deteccion"].astype(float).reindex(DETECTORES)
            d = s - ref
            med = float(d.median()) if d.notna().any() else np.nan
            completo = s.notna().all() and ref.notna().all()
            filas.append({
                "pista": pista, "variante": variante,
                "bloque": nombre_corto(g["bloque_quitado"].dropna().iloc[0])
                if g["bloque_quitado"].notna().any() else variante.removeprefix("SIN-"),
                **{f"delta_{k}": v for k, v in d.items()},
                "mediana_delta": med,
                "delta_mediana": float(s.median() - ref.median()) if completo else np.nan,
                "n_mejoran": int((d > 0).sum()), "n_detectores": int(d.notna().sum()),
                "efecto": "sin datos" if np.isnan(med) else
                ("estorba" if med > 0 else "ayuda" if med < 0 else "neutro"),
            })
    out = pd.DataFrame(filas)
    if out.empty:
        return out
    orden = {p: i for i, p in enumerate(PISTAS)}
    return (out.assign(_o=out["pista"].map(orden))
            .sort_values(["_o", "mediana_delta"], ascending=[True, False], na_position="last")
            .drop(columns="_o").reset_index(drop=True))


def tabla_media(tabla: pd.DataFrame) -> pd.DataFrame:
    """``MEDIA`` (media simple en vez de PCA) frente a ``REF-T`` por pista y detector.

    Añade por pista una fila ``base = "mediana"`` con las medianas de los 6 (solo si están
    los 6 en las dos variantes) y el número de detectores en que la media mejora.
    """
    filas = []
    for pista in _pistas(tabla):
        ref = _scores(tabla, pista, "REF-T").reindex(DETECTORES)
        med = _scores(tabla, pista, "MEDIA").reindex(DETECTORES)
        d = med - ref
        for det in DETECTORES:
            filas.append({"pista": pista, "base": det, "score_ref_t": ref[det], "score_media": med[det],
                          "delta": d[det], "media_mejora": bool(d[det] > 0) if pd.notna(d[det]) else None})
        completo = ref.notna().all() and med.notna().all()
        filas.append({"pista": pista, "base": "mediana",
                      "score_ref_t": float(ref.median()) if completo else np.nan,
                      "score_media": float(med.median()) if completo else np.nan,
                      "delta": float(med.median() - ref.median()) if completo else np.nan,
                      "media_mejora": f"{int((d > 0).sum())} de {int(d.notna().sum())}"})
    return pd.DataFrame(filas)


def validez_atajo(tabla: pd.DataFrame, oficial: Path | pd.DataFrame = OFICIAL) -> pd.DataFrame:
    """Signo de ``REF-T − REF-N`` (reajuste anual) frente al de T − N oficial (mensual).

    Una fila por pista × detector con los cuatro scores, los dos deltas, sus signos y
    ``coincide``. Un delta exactamente 0 tiene signo 0 y solo coincide con otro 0.
    ``attrs["coinciden"]``: «k de n» con los pares que tienen los dos deltas.
    """
    of = pd.read_csv(oficial) if not isinstance(oficial, pd.DataFrame) else oficial
    of = of.set_index(["pista", "base"])
    filas = []
    for pista in _pistas(tabla):
        n = _scores(tabla, pista, "REF-N")
        t = _scores(tabla, pista, "REF-T")
        for det in DETECTORES:
            sn, st = n.get(det, np.nan), t.get(det, np.nan)
            on = of["score_deteccion_N"].get((pista, det), np.nan)
            ot = of["score_deteccion_T"].get((pista, det), np.nan)
            da, do = st - sn, ot - on
            filas.append({
                "pista": pista, "base": det, "score_N_anual": sn, "score_T_anual": st,
                "score_N_oficial": on, "score_T_oficial": ot, "delta_anual": da, "delta_oficial": do,
                "signo_anual": np.sign(da), "signo_oficial": np.sign(do),
                "coincide": bool(np.sign(da) == np.sign(do)) if pd.notna(da) and pd.notna(do) else None,
            })
    out = pd.DataFrame(filas)
    if not out.empty:
        ok = out["coincide"].dropna()
        out.attrs["coinciden"] = f"{int(ok.astype(bool).sum())} de {len(ok)}"
    return out


# --------------------------------------------------------------------------- #
# Punto 6: MCS dentro de cada menú (paneles oficiales, reajuste mensual)
# --------------------------------------------------------------------------- #
def _senales_v2(pista: str, v2: Path = V2, ids: list[str] | None = None) -> pd.DataFrame:
    from regimenes.evaluacion import mcs

    paneles = Path(v2) / "panels"
    pan = mcs.cargar_paneles(pista, paneles, ids)
    return mcs.senales_crisis(pan, n_states=mcs.n_states_metricas(pista, paneles, list(pan)))


def senales_menu(pista: str, menu: str, *, v2: Path = V2, menus: Path = MENUS) -> pd.DataFrame:
    """Señales diarias de crisis de los detectores de un menú, sobre el OOS de v2.

    ``N``: los 12 de v2. ``N6``: los 6 multivariantes de v2 (el menú N de los mismos
    detectores que T). ``T``: los 6 del menú T oficial (``results/detectores/menus``, columnas
    ``<D>_T``). Falla si el índice no es el de v2.
    """
    from regimenes.evaluacion import mcs

    pista, menu = pista.upper(), menu.upper()
    if menu == "N":
        return _senales_v2(pista, v2)
    if menu == "N6":
        return _senales_v2(pista, v2, list(DETECTORES))
    if menu != "T":
        raise ValueError(f"menú desconocido: {menu!r} (N | N6 | T)")
    ids = [f"{d}_T" for d in DETECTORES]
    pan = mcs.cargar_paneles(pista, Path(menus), ids)
    sen = mcs.senales_crisis(pan, n_states=mcs.n_states_metricas(pista, Path(menus), ids))
    ref = _senales_v2(pista, v2, ["D02"]).index
    if not sen.index.equals(ref):
        raise ValueError(f"{pista}: los paneles del menú T no tienen el índice OOS de v2")
    return sen


def senales_baselines_vara(pista: str, *, v2: Path = V2) -> pd.DataFrame:
    """Los 5 baselines de la vara de medir, idénticos a los de ``mcs_diario.csv``.

    Misma construcción que el notebook 05 §10: ``AZAR_*`` con la tasa y la racha medianas de
    los 12 detectores de v2 de la pista (semilla por defecto) y ``VOL_P90`` sobre
    ``SP500_vol_z`` de ``data/processed``. Se usan los MISMOS en todos los menús: así los
    baselines son el mismo rival en N y en T y solo cambian los detectores.
    """
    from regimenes.evaluacion import mcs

    pista = pista.upper()
    sen = _senales_v2(pista, v2)
    F = sen.to_numpy()
    vol = pd.read_parquet(rutas.DATA_PROCESSED / f"pista{pista}_diaria.parquet",
                          columns=["SP500_vol_z"])["SP500_vol_z"]
    return mcs.senales_baselines(
        pista, pd.DatetimeIndex(sen.index), rate=float(np.median(F.mean(axis=0))),
        mean_run=float(np.median([mcs._racha_media(F[:, j]) for j in range(F.shape[1])])), vol=vol)


def mcs_por_menu(pista: str, menu: str = "T", *, v2: Path = V2, menus: Path = MENUS,
                 reutilizar_vara: bool = True, **kwargs) -> pd.DataFrame:
    """MCS dentro de un menú con los baselines (ADR-008 §5, punto 6).

    Detectores de :func:`senales_menu` + :func:`senales_baselines_vara`, pérdida y remuestreo
    definitivos (:data:`MCS_KW`: λ = 1, α = 0,10, ``T_max``, bootstrap estacionario de bloque
    :func:`~regimenes.evaluacion.mcs.longitud_bloque_crisis`, 5.000 réplicas de razón,
    semilla 0). Para ``menu="N"`` es exactamente el MCS de ``results/benchmark/vara/
    mcs_diario.csv`` (los 12 + 5 baselines): con ``reutilizar_vara`` se lee de ahí.
    Devuelve la tabla del MCS con ``pista``, ``bloque`` y ``menu``.
    """
    from regimenes.evaluacion import mcs
    from regimenes.evaluacion import ranking as rk

    pista, menu = pista.upper(), menu.upper()
    if menu == "N" and reutilizar_vara and VARA.exists():
        t = pd.read_csv(VARA, index_col="modelo")
        return t[t["pista"] == pista].assign(menu="N")
    sen = pd.concat([senales_menu(pista, menu, v2=v2, menus=menus), senales_baselines_vara(pista, v2=v2)],
                    axis=1)
    win = rk.track_crisis_windows(pista)
    bloque = mcs.longitud_bloque_crisis(pd.DatetimeIndex(sen.index), win)
    _, res = mcs.mcs_senales(sen, win, block_len=bloque, **{**MCS_KW, **kwargs})
    t = res.tabla.copy()
    t.index.name = "modelo"
    return t.assign(pista=pista, bloque=bloque, menu=menu)


# --------------------------------------------------------------------------- #
# Escritura
# --------------------------------------------------------------------------- #
def _csv(ruta: Path, df: pd.DataFrame, index: bool = False) -> Path:
    _write_atomic(ruta, lambda tmp: df.to_csv(tmp, index=index))
    return ruta


def escribir_tablas(salida: Path = SALIDA, *, mcs: bool = True) -> list[Path]:
    """Escribe en ``salida`` (atómico) lo que se pueda calcular con lo que hay.

    - Con ``mcs``: ``mcs_menu_T_{A,B}.csv`` y ``mcs_menu_N6_{A,B}.csv`` (paneles oficiales).
    - Si hay CSV sueltos por trabajo: el consolidado :data:`TRABAJOS` (el que se versiona).
    - Si hay trabajos (sueltos o consolidados): ``criterio_b.csv``, ``medianas_azar.csv``,
      ``sin_bloque.csv``, ``media.csv`` y ``validez_atajo.csv``.
    """
    salida = Path(salida)
    hechas = []
    if mcs:
        for pista in PISTAS:
            for menu in ("T", "N6"):
                hechas.append(_csv(salida / f"mcs_menu_{menu}_{pista}.csv", mcs_por_menu(pista, menu),
                                   index=True))
    if _rutas_trabajo(salida):
        hechas.append(_csv(salida / TRABAJOS, _tabla_trabajos(salida)))
    tabla = cargar(salida)
    if len(tabla):
        az = pd.concat([medianas_azar(tabla, p).assign(pista=p) for p in _pistas(tabla)], ignore_index=True)
        hechas += [_csv(salida / "criterio_b.csv", criterio_b(tabla)),
                   _csv(salida / "medianas_azar.csv", az),
                   _csv(salida / "sin_bloque.csv", tabla_sin_bloque(tabla)),
                   _csv(salida / "media.csv", tabla_media(tabla)),
                   _csv(salida / "validez_atajo.csv", validez_atajo(tabla))]
    return hechas


def restaurar_trabajos(salida: Path = SALIDA, *, sobrescribir: bool = False) -> list[Path]:
    """Re-escribe los CSV sueltos ``<P>_<D>_<VARIANTE>.csv`` a partir de :data:`TRABAJOS`.

    Son la marca de «trabajo hecho» de ``benchmark/ablaciones.py``: si faltan (clon nuevo, o git los borró al
    cambiar a una rama que aún los versionaba y volver), una re-ejecución repetiría los 1.338 trabajos
    (~78 h). Por defecto no toca los que ya existen.
    """
    salida = Path(salida)
    tabla = pd.read_csv(salida / TRABAJOS)
    escritas = []
    for i in range(len(tabla)):
        fila = tabla.iloc[[i]]
        ruta = salida / f"{fila['pista'].iloc[0]}_{fila['base'].iloc[0]}_{fila['variante'].iloc[0]}.csv"
        if ruta.exists() and not sobrescribir:
            continue
        escritas.append(_csv(ruta, fila))
    return escritas


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--salida", type=Path, default=SALIDA)
    parser.add_argument("--sin-mcs", action="store_true", help="no recalcula el MCS por menú")
    parser.add_argument("--restaurar-trabajos", action="store_true",
                        help="re-escribe los CSV por trabajo que falten a partir de trabajos.csv y termina")
    args = parser.parse_args(argv)
    if args.restaurar_trabajos:
        print(f"{len(restaurar_trabajos(args.salida))} CSV por trabajo restaurados en {args.salida}")
        return
    for ruta in escribir_tablas(args.salida, mcs=not args.sin_mcs):
        print(ruta)


if __name__ == "__main__":
    main()
