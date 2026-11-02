"""Ablaciones y controles del pool de termómetros (ADR-008 §5 punto 5 y §6 (b); errata §5.1).

Variantes por pista, todas con los seis multivariantes del menú T (D02, D03, D04, D08, D09,
D10) y **reajuste anual del detector** (``step=252``, errata del ADR-008 §5.1):

- ``REF-N`` · núcleo de v2 y ``REF-T`` · termómetros económicos oficiales, re-ejecutados con
  ese mismo ``step`` (las cifras oficiales del §8 usan el ``step`` de cada detector y no se
  tocan).
- ``SIN-<Tk>`` · menú T sin el termómetro ``Tk`` (uno por bloque: 7 en A, 10 en B). Solo se
  quita la columna; los demás termómetros son los oficiales.
- ``MEDIA`` · cada bloque económico se resume con la media simple de sus columnas en vez de la
  PC1. Es el modo «media» que ya tiene :mod:`regimenes.features.bloques` cuando la PC1 no
  llega al umbral, forzado en todos los reajustes (``umbral_var = inf``): mismas columnas,
  anclas, reajuste mensual causal y calentamiento; cada columna entra con el signo de su
  correlación, en la ventana de ajuste, con la ancla orientada de su bloque.
- ``AZAR-000`` … ``AZAR-099`` · reparto aleatorio de las MISMAS columnas del pool en el MISMO
  número de bloques con los MISMOS tamaños (permutación con ``numpy.random.default_rng(s)``,
  ``s`` = número de la variante; el bloque ``Zi`` tiene el tamaño del i-ésimo bloque de
  ``configs/bloques.yaml``) y termómetro por bloque con la misma PCA causal mensual, umbral y
  calentamiento. Ancla y signo de un bloque aleatorio: :func:`termometro_azar`.

Regla de signo de AZAR (sin etiquetas de crisis). Un bloque aleatorio no tiene ancla
económica, y D02 suma los termómetros con signo +1: un termómetro invertido lo hundiría. Por
eso cada columna conserva la orientación frente al estrés que tiene en su bloque ORIGINAL:

1. Signo de una columna ``c`` en el reajuste ``r``: si ``c`` es la ancla de su bloque
   original, su signo declarado; si no, ``signo(corr(c, ancla_original)) * signo_ancla``
   con las filas completas del par ``<= r`` (``>= min_obs``). Es la misma regla con la que
   ``bloques.py`` orienta las columnas en su modo media.
2. En el primer reajuste ``r0`` en que alguna columna del bloque es elegible y tiene signo
   determinable, se ajusta la PCA del bloque con los datos ``<= r0`` y la ancla del bloque
   es la columna con signo de mayor |carga| (la que mejor fija el signo de la PC1 en los
   reajustes siguientes). La columna ancla queda fija desde ``r0``; **su signo se re-estima
   en cada reajuste** ``r`` con la regla 1 y datos ``<= r`` (una ancla original conserva su
   signo declarado).
3. El termómetro sale del mismo bucle que :func:`~regimenes.features.bloques.termometro_bloque`
   (``_ajustar`` + ``_proyectar`` por reajuste) con el signo vigente de la ancla. Los días
   ``<= r0`` quedan en NaN (antes de ``r0`` el signo no se conocía), salvo en un bloque de
   una sola columna que sea ancla original (su signo no depende de datos).

Por qué se re-estima el signo (diagnóstico sin etiquetas, 02-11-2026): con el signo fijado en
``r0`` (1929–1966, a veces con una docena de observaciones mensuales) 12 de las 67 columnas no
ancla de B y 7 de A llegaban al OOS con el signo contrario al vigente frente a su ancla
original (p. ej. ``SP500_momentum``), y casi todas las semillas tenían alguna ancla invertida
en más de la mitad del OOS: D02, que suma con +1, quedaba hundido por construcción.

Así nada usa información posterior a cada fecha (prueba de prefijo en los tests). Descartadas:
fijar los signos con estadísticos de toda la muestra (no causal); fijarlos en ``r0`` (ver el
diagnóstico); una ancla al azar (pesa poco en la PC1 y el signo saltaría entre reajustes);
orientar con otra serie externa como la vol del S&P (añade una decisión que el pre-registro
no tomó y desorienta T2 o T9); y dejar el signo que devuelva ``eigh`` (penaliza a D02 de
forma artificial).

Reparto factible. Un reparto cuyo termómetro tenga algún NaN dentro del índice oficial (p. ej.
un bloque de una sola columna que empieza tarde, como ``UMCSENT_z`` en A) no se puede evaluar
sin cambiar el OOS. Se descarta y se saca la siguiente permutación del MISMO generador; el
número de intentos queda en ``variantes_<P>.json``. El reparto económico es factible, así que
el nulo es «reparto aleatorio factible».

Salida (``results/detectores/ablaciones/``):

- ``_entrada/<P>.parquet``: la entrada oficial del menú T (``menus_oficial.escribir_entrada``:
  columnas de N, termómetros económicos y ``SP500_ret`` sobre el índice oficial) más
  ``_entrada/<P>_<VARIANTE>.parquet`` con los termómetros de MEDIA y de cada AZAR. No se
  versionan (``.gitignore``: ``results/detectores/**/*.parquet``).
- ``variantes_<P>.json`` (versionable): composición exacta de cada variante (columnas por
  bloque, anclas y signos, semilla, intentos) y sha256 de cada parquet de entrada.
- ``<P>_<D>_<VARIANTE>.csv`` (marca de terminado) y su panel OOS ``.parquet``;
  ``ejecucion.log``; ``_errores/<P>_<D>_<VARIANTE>.json`` si un trabajo falla.

Uso (desde la raíz del repo)::

    python -m regimenes.benchmark.ablaciones preparar --pista A B --jobs 11
    python -m regimenes.benchmark.ablaciones ejecutar --pista A B --jobs 11 --horas 7.5
    python -m regimenes.benchmark.ablaciones estado --pista A B --jobs 11
    # subconjuntos (admite comodines)
    python -m regimenes.benchmark.ablaciones ejecutar --pista B --detector D02 --variantes "AZAR-00*"
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
import fnmatch
import gc
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from regimenes import rutas
from regimenes.benchmark import menus_oficial as mo
from regimenes.benchmark.cache import _write_atomic
from regimenes.benchmark.ejecucion import DEFAULT_TRAIN_DAYS, _thread_limits
from regimenes.benchmark.variantes import crudos_pista, evaluar_variante
from regimenes.detectores.menus import _con_columnas, variant_specs
from regimenes.detectores.registry import DetectorSpec
from regimenes.features import bloques as bl

SALIDA = rutas.RESULTS_DETECTORES / "ablaciones"
STEP = 252                                   # reajuste anual del detector (errata §5.1)
DETECTORES = mo.DETECTORES_T                 # D02, D03, D04, D08, D09, D10
N_AZAR = 100
MAX_INTENTOS = 1000                          # permutaciones por semilla hasta una factible
UMBRAL_SOLO_MEDIA = float("inf")             # var_pc1 <= 1 < inf: siempre el modo media

# Segundos por trabajo con step=252 medidos en el piloto del 02-11-2026 (11 procesos a la vez;
# media de N y T). Sirven para el orden LPT, el presupuesto --horas y la estimación de estado.
COSTE_PILOTO = {
    ("A", "D02"): 7.4, ("A", "D03"): 48.9, ("A", "D04"): 613.3, ("A", "D08"): 652.4,
    ("A", "D09"): 524.7, ("A", "D10"): 277.6,
    ("B", "D02"): 3.4, ("B", "D03"): 8.0, ("B", "D04"): 74.4, ("B", "D08"): 80.9,
    ("B", "D09"): 57.8, ("B", "D10"): 29.1,
}
COLUMNAS_EXTRA = ("base", "variante", "tipo", "bloque_quitado", "semilla", "step")

REGLA_MEDIA = ("termometro_bloque con umbral_var=inf: media simple de las columnas del bloque "
               "económico, cada una con el signo de su correlación (ventana de ajuste) con la "
               "ancla orientada; mismo reajuste mensual, min_obs, ventana y calentamiento")
REGLA_AZAR = ("permutación numpy.random.default_rng(semilla).permutation del pool en el orden de "
              "bloques.yaml, cortada con los tamaños de los bloques económicos; signo de cada "
              "columna = signo declarado si es ancla original, si no signo(corr con su ancla "
              "original, datos <= r)*signo_ancla; ancla del bloque = columna con signo de mayor "
              "|carga| en el primer reajuste r0 con alguna; la ancla queda fija y su signo se "
              "re-estima en cada reajuste r con datos <= r; días <= r0 en NaN; reparto con NaN "
              "en el índice oficial -> siguiente permutación del mismo generador")


# --------------------------------------------------------------------------- #
# Nombres y composición
# --------------------------------------------------------------------------- #
def nombre_corto(bloque: str) -> str:
    """``T1_renta_variable`` -> ``T1``."""
    return bloque.split("_")[0]


def nombres_variantes(bloques: Mapping[str, Any]) -> list[str]:
    """Las variantes de una pista en su orden canónico."""
    return (["REF-N", "REF-T"] + [f"SIN-{nombre_corto(b)}" for b in bloques] + ["MEDIA"]
            + [f"AZAR-{s:03d}" for s in range(N_AZAR)])


def tipo_de(variante: str) -> str:
    if variante in ("REF-N", "REF-T"):
        return "ref"
    if variante.startswith("SIN-"):
        return "sin_bloque"
    if variante == "MEDIA":
        return "media"
    if variante.startswith("AZAR-"):
        return "azar"
    raise ValueError(f"variante desconocida: {variante!r}")


def necesita_parquet(variante: str) -> bool:
    """MEDIA y AZAR tienen termómetros propios; el resto lee la entrada oficial."""
    return tipo_de(variante) in ("media", "azar")


def semilla_de(variante: str) -> int | None:
    return int(variante.split("-")[1]) if tipo_de(variante) == "azar" else None


def bloque_quitado(variante: str, bloques: Mapping[str, Any]) -> str | None:
    if tipo_de(variante) != "sin_bloque":
        return None
    corto = variante[len("SIN-"):]
    nombres = [b for b in bloques if nombre_corto(b) == corto]
    if len(nombres) != 1:
        raise ValueError(f"{variante}: bloque {corto!r} no identificado en {list(bloques)}")
    return nombres[0]


def columnas_sin(bloques: Mapping[str, Any], variante: str) -> list[str]:
    """Columnas del menú T sin el termómetro de ``variante`` (``SIN-<Tk>``)."""
    quitado = bloque_quitado(variante, bloques)
    return [b for b in bloques if b != quitado]


def pool_columnas(bloques: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Columnas del pool de la pista en el orden de ``bloques.yaml``."""
    return [c for b in bloques.values() for c in b["columnas"]]


def anclas_originales(bloques: Mapping[str, Mapping[str, Any]]) -> dict[str, tuple[str, int]]:
    """``{columna: (ancla de su bloque económico, signo de esa ancla)}``."""
    out = {}
    for b in bloques.values():
        col, signo = bl._ancla(b["ancla"])
        for c in b["columnas"]:
            out[c] = (col, signo)
    return out


def reparto_aleatorio(bloques: Mapping[str, Mapping[str, Any]], rng: np.random.Generator
                      ) -> dict[str, list[str]]:
    """Permuta el pool y lo corta con los tamaños de los bloques económicos (en su orden)."""
    pool = pool_columnas(bloques)
    perm = [pool[i] for i in rng.permutation(len(pool))]
    out, ini = {}, 0
    for i, b in enumerate(bloques.values()):
        k = len(b["columnas"])
        out[f"Z{i + 1:02d}"] = perm[ini:ini + k]
        ini += k
    return out


# --------------------------------------------------------------------------- #
# Termómetros de las variantes
# --------------------------------------------------------------------------- #
def _signo_columna(par: bl._Acumulados, propia: bool, signo_ancla: int, ini: int, fin: int,
                   min_obs: int) -> int | None:
    """Signo frente al estrés de una columna con los datos ``[ini, fin)`` (paso 1 de la regla).

    ``par`` acumula (columna, ancla original). None si no hay ``min_obs`` filas completas o
    alguna de las dos es constante."""
    if propia:
        return signo_ancla
    n, mu, cov = par.completas([0, 1], ini, fin)
    if n < min_obs or not np.all(np.diag(cov) > bl._TOL_VAR * (1 + mu ** 2)):
        return None
    s = int(np.sign(cov[0, 1]) * signo_ancla)
    return s if s != 0 else 1


def termometro_azar(panel: pd.DataFrame, columnas: Sequence[str],
                    anclas: Mapping[str, tuple[str, int]], *, umbral_var: float, min_obs: int,
                    ventana: int | None = None) -> tuple[pd.Series, dict[str, Any]]:
    """Termómetro causal de un bloque aleatorio con la regla de signo del docstring del módulo.

    ``anclas`` da, para cada columna, la ancla de su bloque económico y su signo
    (:func:`anclas_originales`). No recibe etiquetas de crisis. Devuelve la serie sobre
    ``panel.index`` y la ficha del bloque (ancla, signo, fecha de fijación ``r0``)."""
    cols = list(dict.fromkeys(columnas))
    idx = panel.index
    ficha: dict[str, Any] = {"columnas": cols, "ancla": None, "fecha_fijacion": None}
    X = panel[cols].to_numpy(dtype=float)
    acu = bl._Acumulados(X)
    pares: dict[str, bl._Acumulados] = {}
    ancla = signo = r0 = None
    for p in idx.get_indexer(bl.fechas_reajuste(idx)):
        fin = p + 1
        ini = 0 if ventana is None else max(0, fin - ventana)
        n_ok, var_ok = acu.por_columna(ini, fin)
        eleg = [c for j, c in enumerate(cols) if n_ok[j] >= min_obs and var_ok[j]]
        signos = {}
        for c in eleg:
            col_anc, s_anc = anclas[c]
            propia = c == col_anc
            if not propia and c not in pares:
                pares[c] = bl._Acumulados(panel[[c, col_anc]].to_numpy(dtype=float))
            s = _signo_columna(pares.get(c), propia, s_anc, ini, fin, min_obs)
            if s is not None:
                signos[c] = s
        if not signos:
            continue
        if len(cols) == 1:
            ancla, signo = cols[0], signos[cols[0]]
        else:
            # PCA del bloque en r0 con una ancla provisional (la de más datos): solo se usan
            # los módulos de las cargas, que no dependen de la ancla.
            prov = max(signos, key=lambda c: (n_ok[cols.index(c)], -cols.index(c)))
            aj = bl._ajustar(acu, ini, fin, cols.index(prov), 1, 0.0, min_obs)
            if aj is None:
                continue
            cargas = {cols[e]: abs(float(a)) for e, a in zip(aj["eleg"], aj["a"])}
            ancla = max((c for c in cols if c in signos and c in cargas),
                        key=lambda c: (cargas[c], -cols.index(c)))
            signo = signos[ancla]
        r0 = idx[p]
        break
    if ancla is None:
        return pd.Series(np.nan, index=idx), ficha
    col_anc, s_anc = anclas[ancla]
    propia = ancla == col_anc
    if len(cols) == 1 and propia:
        # ancla original sola: signo declarado, sin máscara (como un bloque económico de 1 columna)
        ficha.update({"ancla": {"columna": ancla, "signo_inicial": int(signo)},
                      "bloque_original_ancla": col_anc, "fecha_fijacion": None, "cambios_signo": []})
        return (signo * panel[ancla].astype(float)).rename(None), ficha
    # Bucle propio (el de bloques.termometro_bloque) con el signo de la ancla re-estimado en
    # cada reajuste r con datos <= r. Los reajustes < r0 no se usan: días <= r0 en NaN.
    out = np.full(len(idx), np.nan)
    k_anc = cols.index(ancla)
    pos = idx.get_indexer(bl.fechas_reajuste(idx))
    s_r, cambios = signo, []
    for j, p in enumerate(pos):
        if idx[p] < r0:
            continue
        ini = 0 if ventana is None else max(0, p + 1 - ventana)
        fin = pos[j + 1] if j + 1 < len(pos) else p          # aplica a (p, fin]
        nuevo = _signo_columna(pares.get(ancla), propia, s_anc, ini, p + 1, min_obs)
        if nuevo is not None and nuevo != s_r:
            s_r = nuevo
            cambios.append([str(idx[p].date()), int(s_r)])
        if len(cols) == 1:
            out[p + 1:fin + 1] = s_r * X[p + 1:fin + 1, 0]
            continue
        aj = bl._ajustar(acu, ini, p + 1, k_anc, s_r, umbral_var, min_obs)
        if aj is None:
            continue
        out[p + 1:fin + 1], _ = bl._proyectar(X[p + 1:fin + 1], aj)
    ficha.update({"ancla": {"columna": ancla, "signo_inicial": int(signo)},
                  "bloque_original_ancla": col_anc, "fecha_fijacion": str(pd.Timestamp(r0).date()),
                  "cambios_signo": cambios})
    return pd.Series(out, index=idx), ficha


def termometros_azar(panel: pd.DataFrame, bloques: Mapping[str, Mapping[str, Any]], semilla: int,
                     *, umbral_var: float, min_obs: int, ventana: int | None = None,
                     indice: pd.DatetimeIndex | None = None, max_intentos: int = MAX_INTENTOS
                     ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Termómetros de ``AZAR-<semilla>``: primer reparto factible del generador ``semilla``.

    Factible = ningún termómetro con NaN en ``indice`` (sin ``indice``, el primero vale)."""
    rng = np.random.default_rng(semilla)
    anclas = anclas_originales(bloques)
    for intento in range(1, max_intentos + 1):
        reparto = reparto_aleatorio(bloques, rng)
        series, fichas, factible = {}, {}, True
        for nombre, cols in reparto.items():
            serie, ficha = termometro_azar(panel, cols, anclas, umbral_var=umbral_var,
                                           min_obs=min_obs, ventana=ventana)
            if indice is not None and serie.reindex(indice).isna().any():
                factible = False
                break
            series[nombre], fichas[nombre] = serie, ficha
        if factible:
            return pd.DataFrame(series, index=panel.index), {
                "semilla": int(semilla), "intentos": intento, "bloques": fichas}
    raise RuntimeError(f"AZAR-{semilla:03d}: ningún reparto factible en {max_intentos} intentos")


def termometros_media(panel: pd.DataFrame, bloques: Mapping[str, Mapping[str, Any]], *,
                      min_obs: int, ventana: int | None = None) -> pd.DataFrame:
    """Termómetros de MEDIA: el modo media de ``bloques.py`` en todos los reajustes."""
    t, _ = bl.construir_termometros(panel, bloques, umbral_var=UMBRAL_SOLO_MEDIA,
                                    min_obs=min_obs, ventana=ventana)
    return t


def panel_pool_historia(track: str) -> pd.DataFrame:
    """Columnas del pool de la pista con toda la historia previa, como en
    :func:`regimenes.features.bloques.termometros_pista` (sin recortar a la ventana)."""
    from regimenes.features.paneles import construir_paneles

    track = track.upper()
    spec = yaml.safe_load(rutas.BENCHMARK_SPEC.read_text(encoding="utf-8"))[f"pista_{track}"]
    raw = crudos_pista(track)
    faltan = [n for n in spec["series_features"] if n not in raw]
    if faltan:
        raise FileNotFoundError(f"{track}: faltan crudos en data/raw: {faltan}")
    p = construir_paneles(raw, bl.INICIO_HISTORIA, spec["ventana_fin"])
    del raw
    mensual = p["mensual"][[c for c in p["mensual"].columns if not c.endswith("_edad_dias")]]
    panel = p["diaria"].join(mensual)
    pool = pool_columnas(bl.cargar_bloques(track))
    ausentes = [c for c in pool if c not in panel.columns]
    if ausentes:
        raise ValueError(f"{track}: columnas del pool ausentes del panel: {ausentes}")
    return panel[pool].copy()


# --------------------------------------------------------------------------- #
# Preparar
# --------------------------------------------------------------------------- #
def sha256(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def ruta_base(salida: Path, track: str) -> Path:
    return Path(salida) / "_entrada" / f"{track.upper()}.parquet"


def ruta_parquet_variante(salida: Path, track: str, variante: str) -> Path:
    return Path(salida) / "_entrada" / f"{track.upper()}_{variante}.parquet"


def ruta_composicion(salida: Path, track: str) -> Path:
    return Path(salida) / f"variantes_{track.upper()}.json"


def leer_composicion(salida: Path, track: str) -> dict[str, Any]:
    ruta = ruta_composicion(salida, track)
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def escribir_composicion(salida: Path, track: str, comp: Mapping[str, Any]) -> Path:
    ruta = ruta_composicion(salida, track)
    texto = json.dumps(comp, ensure_ascii=False, indent=1)
    _write_atomic(ruta, lambda tmp: Path(tmp).write_text(texto, encoding="utf-8"))
    return ruta


def _parametros() -> dict[str, Any]:
    p = bl.parametros_bloques()
    return {"umbral_var": p["umbral_var"], "min_obs": p["min_obs"], "ventana": p["ventana"],
            "reajuste_termometros": p["reajuste"], "historia_previa": p["historia_previa"]}


def ficha_simple(track: str, variante: str, bloques: Mapping[str, Mapping[str, Any]]
                 ) -> dict[str, Any]:
    """Ficha de las variantes que leen la entrada oficial (REF-N, REF-T, SIN-Tk)."""
    tipo = tipo_de(variante)
    ficha: dict[str, Any] = {"tipo": tipo, "archivo": f"{track}.parquet"}
    if variante == "REF-N":
        ficha["columnas_por_detector"] = {
            s.detector_id: list(s.features) for s in variant_specs(track, "N")
            if s.detector_id in DETECTORES}
    elif variante == "REF-T":
        ficha["columnas"] = list(bloques)
        ficha["bloques"] = {k: dict(v) for k, v in bloques.items()}
    else:
        ficha["bloque_quitado"] = bloque_quitado(variante, bloques)
        ficha["columnas"] = columnas_sin(bloques, variante)
    return ficha


_HISTORIA: dict[str, tuple[pd.DataFrame, pd.DatetimeIndex]] = {}


def _historia(track: str, salida: Path) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Panel del pool con historia previa e índice oficial (una vez por proceso)."""
    if track not in _HISTORIA:
        _, indice = mo.leer_entrada(ruta_base(salida, track))
        _HISTORIA[track] = (panel_pool_historia(track), indice)
    return _HISTORIA[track]


def construir_variante(track: str, variante: str, salida: Path) -> dict[str, Any]:
    """Termómetros de MEDIA o de un AZAR sobre el índice oficial; escribe su parquet.

    Devuelve la ficha para ``variantes_<P>.json`` (composición, sha256 y segundos)."""
    t0 = time.perf_counter()
    track = track.upper()
    panel, indice = _historia(track, salida)
    bloques = bl.cargar_bloques(track)
    p = _parametros()
    if variante == "MEDIA":
        t = termometros_media(panel, bloques, min_obs=p["min_obs"], ventana=p["ventana"])
        ficha: dict[str, Any] = {"tipo": "media", "regla": REGLA_MEDIA, "columnas": list(bloques),
                                 "bloques": {k: dict(v) for k, v in bloques.items()}}
    else:
        t, info = termometros_azar(panel, bloques, semilla_de(variante), umbral_var=p["umbral_var"],
                                   min_obs=p["min_obs"], ventana=p["ventana"], indice=indice)
        ficha = {"tipo": "azar", "columnas": list(info["bloques"]), **info}
    t = t.reindex(indice)
    if t.isna().any().any():
        raise ValueError(f"{track}/{variante}: NaN dentro del índice oficial "
                         f"{t.isna().sum()[lambda s: s > 0].to_dict()}")
    ruta = ruta_parquet_variante(salida, track, variante)
    _write_atomic(ruta, lambda tmp: t.to_parquet(tmp))
    ficha.update({"archivo": ruta.name, "sha256": sha256(ruta),
                  "segundos_preparar": round(time.perf_counter() - t0, 2)})
    return ficha


def _construir_seguro(args: tuple[str, str, str]) -> tuple[str, dict[str, Any]]:
    track, variante, salida = args
    try:
        return variante, construir_variante(track, variante, Path(salida))
    except Exception as exc:  # noqa: BLE001 - un fallo no debe tumbar al resto
        return variante, {"error": f"{type(exc).__name__}: {exc}"}


def comprobar_ref_t(track: str, salida: Path) -> None:
    """Los termómetros económicos recalculados aquí son, bit a bit, los de la entrada oficial
    (garantiza que MEDIA y AZAR salen del mismo panel y del mismo calentamiento)."""
    panel, indice = _historia(track, salida)
    base, _ = mo.leer_entrada(ruta_base(salida, track))
    p = _parametros()
    bloques = bl.cargar_bloques(track)
    t, _ = bl.construir_termometros(panel, bloques, umbral_var=p["umbral_var"],
                                    min_obs=p["min_obs"], ventana=p["ventana"])
    pd.testing.assert_frame_equal(t.reindex(indice), base[list(bloques)], check_exact=True,
                                  check_freq=False, check_names=False)


def _seleccionar(nombres: Sequence[str], patrones: Sequence[str] | None) -> list[str]:
    if not patrones:
        return list(nombres)
    return [n for n in nombres if any(fnmatch.fnmatchcase(n.upper(), p.upper()) for p in patrones)]


def huella_codigo() -> dict[str, str]:
    """sha256 del código y la configuración que fijan los termómetros de las variantes."""
    return {"ablaciones.py": sha256(Path(__file__)),
            "features/bloques.py": sha256(Path(bl.__file__)),
            "configs/bloques.yaml": sha256(bl.BLOQUES_YAML)}


def comprobar_huella(comp: Mapping[str, Any], salida: Path, track: str) -> None:
    """Falla si ``variantes_<P>.json`` se preparó con otro código (o sin huella)."""
    if not comp:
        return
    registrada, actual = comp.get("huella_codigo"), huella_codigo()
    if registrada != actual:
        distintos = sorted(k for k in actual if (registrada or {}).get(k) != actual[k])
        raise ValueError(
            f"{track}: el código cambió desde preparar ({distintos}). Borra en {Path(salida)}: "
            f"variantes_{track}.json, _entrada/{track}_*.parquet, {track}_D*_*.csv, "
            f"{track}_D*_*.parquet y _errores/{track}_*.json (deja _entrada/{track}.parquet); "
            "después vuelve a preparar y a ejecutar.")


def _cabecera(track: str, base: Path) -> dict[str, Any]:
    """Cabecera de ``variantes_<P>.json``; se reescribe en cada ``preparar``."""
    return {"pista": track, "step": STEP, "detectores": list(DETECTORES),
            "parametros_termometros": _parametros(), "regla_media": REGLA_MEDIA,
            "regla_azar": REGLA_AZAR, "huella_codigo": huella_codigo(),
            "entrada_base": {"archivo": base.name, "sha256": sha256(base)}}


def _construir_en_paralelo(args: list[tuple[str, str, str]], jobs: int, al_terminar,
                           rondas: int = 3) -> dict[str, str]:
    """Construye en procesos y llama a ``al_terminar(variante, ficha)`` por cada resultado.

    Si un hijo muere (``BrokenProcessPool``) se recrea el pool con lo que faltaba, hasta
    ``rondas`` veces; lo que siga faltando se devuelve como error."""
    faltan = list(args)
    for _ in range(rondas):
        if not faltan:
            break
        with ProcessPoolExecutor(max_workers=min(jobs, len(faltan)),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futuros = {pool.submit(_construir_seguro, a): a for a in faltan}
            try:
                for futuro in as_completed(futuros):
                    v, ficha = futuro.result()
                    faltan.remove(futuros[futuro])
                    al_terminar(v, ficha)
            except BrokenProcessPool:
                print(f"  pool roto; se reintenta con {len(faltan)} variantes", flush=True)
    return {a[1]: "BrokenProcessPool: el proceso murió en todas las rondas" for a in faltan}


def preparar(tracks: Sequence[str], variantes: Sequence[str] | None = None, *,
             salida: Path = SALIDA, jobs: int = 1) -> dict[str, dict[str, Any]]:
    """Entrada oficial y termómetros de cada variante; actualiza ``variantes_<P>.json``.

    Salta las variantes ya preparadas (parquet presente con el sha256 registrado). El JSON
    se reescribe tras cada variante (un corte no pierde lo hecho) y su cabecera en cada
    llamada. Se niega a seguir si el JSON es de otro código o de otra entrada base."""
    salida = Path(salida)
    out = {}
    for track in [t.upper() for t in tracks]:
        base = ruta_base(salida, track)
        comp = leer_composicion(salida, track)
        comprobar_huella(comp, salida, track)
        if not base.exists():
            if comp:
                raise ValueError(f"{track}: falta {base.name} pero existe {ruta_composicion(salida, track).name}"
                                 "; borra el JSON y las entradas y vuelve a preparar")
            mo.escribir_entrada(track, salida)
            gc.collect()
        if comp and comp.get("entrada_base", {}).get("sha256") != sha256(base):
            raise ValueError(f"{track}: {base.name} no es la registrada en {ruta_composicion(salida, track).name}"
                             "; borra las entradas y el JSON y vuelve a preparar")
        comprobar_ref_t(track, salida)
        bloques = bl.cargar_bloques(track)
        hechas = dict(comp.get("variantes", {}))
        comp = {**_cabecera(track, base), "variantes": hechas}
        elegidas = _seleccionar(nombres_variantes(bloques), variantes)
        for v in elegidas:
            if not necesita_parquet(v):
                hechas[v] = ficha_simple(track, v, bloques)

        def guardar() -> None:
            comp["variantes"] = {v: hechas[v] for v in nombres_variantes(bloques) if v in hechas}
            escribir_composicion(salida, track, comp)

        guardar()
        pendientes = [v for v in elegidas if necesita_parquet(v) and not (
            v in hechas and ruta_parquet_variante(salida, track, v).exists()
            and sha256(ruta_parquet_variante(salida, track, v)) == hechas[v].get("sha256"))]
        print(f"{track}: {len(pendientes)} variantes por construir", flush=True)
        t0 = time.perf_counter()
        errores: dict[str, str] = {}

        def al_terminar(v: str, ficha: dict[str, Any]) -> None:
            if "error" in ficha:
                errores[v] = ficha["error"]
                print(f"  {v}: ERROR {ficha['error']}", flush=True)
                return
            hechas[v] = ficha
            guardar()
            print(f"  {v}: {ficha['segundos_preparar']} s"
                  + (f", {ficha['intentos']} intento(s)" if "intentos" in ficha else ""), flush=True)

        args = [(track, v, str(salida)) for v in pendientes]
        try:
            if jobs <= 1 or len(args) <= 1:
                for a in args:
                    al_terminar(*_construir_seguro(a))
            else:
                errores.update(_construir_en_paralelo(args, jobs, al_terminar))
        finally:
            guardar()
        print(f"{track}: preparadas en {time.perf_counter() - t0:.1f} s de reloj", flush=True)
        out[track] = {"composicion": comp, "errores": errores}
    return out



# --------------------------------------------------------------------------- #
# Ejecutar
# --------------------------------------------------------------------------- #
def ruta_trabajo(salida: Path, track: str, detector: str, variante: str) -> Path:
    return Path(salida) / f"{track.upper()}_{detector.upper()}_{variante}.csv"


def ruta_error(salida: Path, track: str, detector: str, variante: str) -> Path:
    return Path(salida) / "_errores" / f"{track.upper()}_{detector.upper()}_{variante}.json"


def spec_variante(track: str, detector: str, variante: str, ficha: Mapping[str, Any]) -> DetectorSpec:
    """Spec del detector para la variante, con ``step=252``. REF-N y REF-T son las oficiales;
    el resto, la original con las columnas de la variante (``menus._con_columnas``)."""
    original = next(s for s in variant_specs(track, "N") if s.detector_id == detector)
    if variante == "REF-N":
        spec = original
    elif variante == "REF-T":
        spec = next(s for s in variant_specs(track, "T") if s.detector_id.split("-")[0] == detector)
    else:
        spec = _con_columnas(original, variante, tuple(ficha["columnas"]))
    return replace(spec, step=STEP)


def panel_trabajo(track: str, variante: str, salida: Path, comp: Mapping[str, Any]
                  ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Entrada del trabajo con verificación de sha256 frente a ``variantes_<P>.json``."""
    base = ruta_base(salida, track)
    if sha256(base) != comp["entrada_base"]["sha256"]:
        raise ValueError(f"{base.name}: sha256 distinto del registrado; vuelve a preparar")
    panel, indice = mo.leer_entrada(base)
    if not necesita_parquet(variante):
        return panel, indice
    ficha = comp["variantes"][variante]
    ruta = ruta_parquet_variante(salida, track, variante)
    if sha256(ruta) != ficha["sha256"]:
        raise ValueError(f"{ruta.name}: sha256 distinto del registrado; vuelve a preparar")
    t = pd.read_parquet(ruta)
    t.index = pd.DatetimeIndex(t.index)
    if not t.index.equals(indice):
        raise ValueError(f"{ruta.name}: su índice no es el oficial")
    return panel[["SP500_ret"]].join(t), indice


def ejecutar_trabajo(track: str, detector: str, variante: str, salida: Path) -> dict:
    """Un trabajo pista × detector × variante; panel OOS y, al final, el CSV (marca de
    terminado). No hace nada si el CSV ya existe."""
    track, detector = track.upper(), detector.upper()
    destino = ruta_trabajo(salida, track, detector, variante)
    if destino.exists():
        return {"trabajo": destino.stem, "estado": "ya_estaba"}
    comp = leer_composicion(salida, track)
    comprobar_huella(comp, salida, track)
    if variante not in comp.get("variantes", {}):
        raise KeyError(f"{variante} no está preparada en {ruta_composicion(salida, track).name}")
    ficha = comp["variantes"][variante]
    panel, indice = panel_trabajo(track, variante, Path(salida), comp)
    spec = spec_variante(track, detector, variante, ficha)
    metrics, wf = evaluar_variante(track, spec, panel, indice)
    del panel
    esperado = indice[DEFAULT_TRAIN_DAYS[track]:]
    if not pd.DatetimeIndex(wf.index).equals(esperado):
        raise ValueError(f"{destino.stem}: el índice OOS no es el oficial ({len(wf)} frente a {len(esperado)})")
    bloques = bl.cargar_bloques(track)
    extra = {"base": detector, "variante": variante, "tipo": tipo_de(variante),
             "bloque_quitado": bloque_quitado(variante, bloques), "semilla": semilla_de(variante),
             "step": STEP}
    for k in reversed(COLUMNAS_EXTRA):
        metrics.insert(0, k, extra[k])
    _write_atomic(destino.with_suffix(".parquet"), lambda tmp: wf.to_parquet(tmp))
    _write_atomic(destino, lambda tmp: metrics.to_csv(tmp, index=False))   # marca de terminado
    ruta_error(salida, track, detector, variante).unlink(missing_ok=True)
    return {"trabajo": destino.stem, "estado": "ok",
            "segundos": round(float(metrics.loc[0, "elapsed_seconds"]), 1),
            "score": float(metrics.loc[0, "score_deteccion"])}


def _trabajo_seguro(args: tuple[str, str, str, str]) -> dict:
    track, detector, variante, salida = args
    try:
        return ejecutar_trabajo(track, detector, variante, Path(salida))
    except Exception as exc:  # noqa: BLE001 - un fallo no debe tumbar al resto
        return {"trabajo": f"{track}_{detector}_{variante}", "estado": "error",
                "detalle": f"{type(exc).__name__}: {exc}"}


def coste(track: str, detector: str) -> float:
    """Segundos estimados de un trabajo (piloto; si falta, v2 / 12 por el reajuste anual)."""
    if (track, detector) in COSTE_PILOTO:
        return COSTE_PILOTO[(track, detector)]
    return mo.tiempos_v2().get((track, detector), 600.0) / 12


def ordenar_lpt(trabajos: Sequence[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Lentos primero; a igual coste, pista, detector y variante."""
    return sorted(trabajos, key=lambda t: (-coste(t[0], t[1]), t))


def _log(salida: Path, texto: str) -> None:
    linea = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {texto}"
    print(linea, flush=True)
    Path(salida).mkdir(parents=True, exist_ok=True)
    with open(Path(salida) / "ejecucion.log", "a", encoding="utf-8") as f:
        f.write(linea + "\n")


def trabajos(tracks: Sequence[str], detectores: Sequence[str], variantes: Sequence[str] | None,
             salida: Path) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """(trabajos preparados sin CSV, trabajos sin preparar) de la selección."""
    listos, sin_preparar = [], []
    for track in [t.upper() for t in tracks]:
        hechas = leer_composicion(salida, track).get("variantes", {})
        for v in _seleccionar(nombres_variantes(bl.cargar_bloques(track)), variantes):
            for d in detectores:
                if ruta_trabajo(salida, track, d, v).exists():
                    continue
                (listos if v in hechas else sin_preparar).append((track, d.upper(), v))
    return listos, sin_preparar


def _nuevo_pool(jobs: int, threads_per_job: int | None) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=max(1, int(jobs)), mp_context=multiprocessing.get_context("spawn"),
                               initializer=mo._inicializar_hijo, initargs=(threads_per_job,),
                               max_tasks_per_child=1)


MAX_ROTURAS = 3        # roturas del pool con un trabajo en vuelo antes de darlo por fallido


def ejecutar(tracks: Sequence[str], detectores: Sequence[str] = DETECTORES,
             variantes: Sequence[str] | None = None, *, salida: Path = SALIDA, jobs: int = 1,
             horas: float | None = None, threads_per_job: int | None = 1,
             trabajo=None) -> list[dict]:
    """Lanza los trabajos pendientes en orden LPT; solo arranca los que caben en ``horas``.

    Si un hijo muere (memoria, kill) el pool queda roto: los trabajos en vuelo vuelven a
    pendientes sin escribir ``_errores`` y el pool se recrea. Si solo había uno en vuelo, es
    el causante y se registra como error; un trabajo presente en ``MAX_ROTURAS`` roturas
    también. ``trabajo`` sustituye a :func:`_trabajo_seguro` (tests)."""
    salida = Path(salida)
    trabajo = trabajo or _trabajo_seguro
    for track in [t.upper() for t in tracks]:
        comprobar_huella(leer_composicion(salida, track), salida, track)
    pendientes, sin_preparar = trabajos(tracks, detectores, variantes, salida)
    pendientes = ordenar_lpt(pendientes)
    limite = float("inf") if horas is None else float(horas) * 3600
    _log(salida, f"ejecutar: {len(pendientes)} pendientes, {len(sin_preparar)} sin preparar, "
                 f"jobs={jobs}, horas={horas}, CPU estimada {sum(coste(t, d) for t, d, _ in pendientes) / 3600:.2f} h")
    estados: list[dict] = []
    if not pendientes:
        return estados
    roturas: dict[tuple[str, str, str], int] = {}

    def registrar(estado: dict, t: str, d: str, v: str) -> None:
        if estado["estado"] == "error":
            texto = json.dumps({**estado, "fecha": time.strftime("%Y-%m-%d %H:%M:%S")},
                               ensure_ascii=False, indent=1)
            _write_atomic(ruta_error(salida, t, d, v),
                          lambda tmp, x=texto: Path(tmp).write_text(x, encoding="utf-8"))
        estados.append(estado)
        _log(salida, str(estado))

    t0 = time.perf_counter()
    with _thread_limits(threads_per_job):
        pool = _nuevo_pool(min(int(jobs), len(pendientes)), threads_per_job)
        try:
            en_vuelo: dict = {}
            while pendientes or en_vuelo:
                roto: list[tuple[str, str, str]] = []
                while len(en_vuelo) < jobs and pendientes:
                    restante = limite - (time.perf_counter() - t0)
                    siguiente = next((t for t in pendientes if coste(t[0], t[1]) <= restante), None)
                    if siguiente is None:
                        break
                    try:
                        futuro = pool.submit(trabajo, (*siguiente, str(salida)))
                    except BrokenProcessPool:
                        roto.append(None)          # el pool ya estaba roto; siguiente sigue pendiente
                        break
                    pendientes.remove(siguiente)
                    en_vuelo[futuro] = siguiente
                if not en_vuelo and not roto:
                    break
                if not roto:
                    hechos, _ = wait(en_vuelo, return_when=FIRST_COMPLETED)
                    for futuro in hechos:
                        t, d, v = en_vuelo.pop(futuro)
                        try:
                            estado = futuro.result()
                        except BrokenProcessPool:
                            roto.append((t, d, v))
                            continue
                        except Exception as exc:  # noqa: BLE001 - un fallo no debe tumbar al resto
                            estado = {"trabajo": f"{t}_{d}_{v}", "estado": "error",
                                      "detalle": f"{type(exc).__name__}: {exc}"}
                        registrar(estado, t, d, v)
                if roto:
                    afectados = [a for a in roto if a is not None] + list(en_vuelo.values())
                    en_vuelo.clear()
                    pool.shutdown(wait=True, cancel_futures=True)
                    for a in afectados:
                        roturas[a] = roturas.get(a, 0) + 1
                        if len(afectados) == 1 or roturas[a] >= MAX_ROTURAS:
                            registrar({"trabajo": "_".join(a), "estado": "error",
                                       "detalle": f"BrokenProcessPool: el proceso murió ({roturas[a]} "
                                                  f"rotura(s), {len(afectados)} en vuelo)"}, *a)
                        else:
                            pendientes.append(a)
                    pendientes = ordenar_lpt(pendientes)
                    _log(salida, f"pool roto con {len(afectados)} trabajos en vuelo; se recrea")
                    pool = _nuevo_pool(min(int(jobs), max(1, len(pendientes))), threads_per_job)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    if pendientes:
        _log(salida, f"presupuesto agotado: {len(pendientes)} trabajos sin arrancar")
    return estados

# --------------------------------------------------------------------------- #
# Estado
# --------------------------------------------------------------------------- #
def comprobar_indices(salida: Path, track: str) -> list[str]:
    """Paneles OOS cuyo índice no coincide fecha a fecha con el de REF-N de la pista."""
    paneles = sorted(Path(salida).glob(f"{track}_D*_*.parquet"))
    refs = [p for p in paneles if p.stem.endswith("_REF-N")]
    if not refs:
        return []
    ref = pd.DatetimeIndex(pd.read_parquet(refs[0], columns=["state"]).index)
    return [p.stem for p in paneles
            if not pd.DatetimeIndex(pd.read_parquet(p, columns=["state"]).index).equals(ref)]


def estado(tracks: Sequence[str], detectores: Sequence[str] = DETECTORES,
           variantes: Sequence[str] | None = None, *, salida: Path = SALIDA, jobs: int = 11
           ) -> pd.DataFrame:
    """Hechos, pendientes, errores y horas restantes por pista (y paneles con otro índice)."""
    salida = Path(salida)
    filas = []
    for track in [t.upper() for t in tracks]:
        hechas = leer_composicion(salida, track).get("variantes", {})
        hechos = pendientes = errores = sin_prep = 0
        cpu = 0.0
        for v in _seleccionar(nombres_variantes(bl.cargar_bloques(track)), variantes):
            for d in detectores:
                if ruta_trabajo(salida, track, d, v).exists():
                    hechos += 1
                    continue
                pendientes += 1
                cpu += coste(track, d)
                errores += ruta_error(salida, track, d, v).exists()
                sin_prep += v not in hechas
        filas.append({"pista": track, "hechos": hechos, "pendientes": pendientes,
                      "errores": errores, "sin_preparar": sin_prep,
                      "horas_cpu_restantes": round(cpu / 3600, 2),
                      "horas_reloj_restantes": round(cpu / 3600 / max(1, jobs), 2),
                      "indice_distinto_de_REF_N": "|".join(comprobar_indices(salida, track))})
    return pd.DataFrame(filas)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("orden", choices=["preparar", "ejecutar", "estado"])
    parser.add_argument("--pista", nargs="+", default=["A", "B"], choices=["A", "B", "a", "b"])
    parser.add_argument("--detector", nargs="+", default=list(DETECTORES))
    parser.add_argument("--variantes", nargs="+", default=None,
                        help="nombres o comodines (REF-N, SIN-*, MEDIA, AZAR-00*...)")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--horas", type=float, default=7.5,
                        help="presupuesto de reloj de ejecutar (0 = sin límite)")
    parser.add_argument("--threads-per-job", type=int, default=1)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    args = parser.parse_args(argv)
    pistas = [p.upper() for p in args.pista]
    detectores = [d.upper() for d in args.detector]
    ajenos = sorted(set(detectores) - set(DETECTORES))
    if ajenos:
        parser.error(f"detectores fuera del experimento: {ajenos}")
    if args.orden == "preparar":
        res = preparar(pistas, args.variantes, salida=args.salida, jobs=args.jobs)
        errores = {t: r["errores"] for t, r in res.items() if r["errores"]}
        if errores:
            raise SystemExit(f"variantes con error: {errores}")
    elif args.orden == "ejecutar":
        estados = ejecutar(pistas, detectores, args.variantes, salida=args.salida, jobs=args.jobs,
                           horas=args.horas or None, threads_per_job=args.threads_per_job or None)
        errores = [e for e in estados if e["estado"] == "error"]
        print(estado(pistas, detectores, args.variantes, salida=args.salida,
                     jobs=args.jobs).to_string(index=False), flush=True)
        if errores:
            raise SystemExit(f"{len(errores)} trabajos con error (ver _errores/)")
    else:
        print(estado(pistas, detectores, args.variantes, salida=args.salida,
                     jobs=args.jobs).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
