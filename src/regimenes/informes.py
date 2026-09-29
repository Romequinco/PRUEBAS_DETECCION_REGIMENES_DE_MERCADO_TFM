"""Utilidades comunes de los notebooks de familia (05–11).

Extrae el código que se repetía, con pequeñas variantes, en cada notebook de familia
(``notebooks/05_familia_F1_reglas.ipynb`` … ``11_familia_F7_deep.ipynb``) para que
todos carguen resultados, dibujen y tabulen de la misma forma:

- **Configuración**: :func:`specs_familia`, :func:`tabla_configuracion`,
  :func:`carpeta_figuras`, :func:`comando_benchmark`.
- **Carga de resultados** (solo lectura de ``results/benchmark``):
  :func:`cargar_resultados_familia` devuelve un :class:`ResultadosFamilia` con las
  métricas verificadas por huella (``run_benchmark(cache_only=True)``), los paneles OOS
  y el ranking de detección ADR-003 (``ranking*.csv`` que escribe ``12_comparativa``),
  comprobando que el ranking corresponde a esas métricas. Si la caché no está vigente
  falla con el comando exacto a ejecutar.
- **Contexto de las pistas** (``data/processed`` y ``data/raw``, no dependen de
  ``results/``): :func:`cargar_contexto_pistas`, :func:`precio_sp500`.
- **Tablas**: :func:`tabla_cobertura`, :func:`tabla_trampas`,
  :func:`tabla_resumen_ranking`, :func:`matriz_jaccard`, :func:`retardo_confirmacion`.
- **Figuras** (estilo de casa de :mod:`regimenes.viz`): :func:`franjas_eventos`,
  :func:`dibujar_estados_sp500`, :func:`figura_estados_oos`, :func:`dibujar_cobertura`,
  :func:`figura_cobertura`, :func:`dibujar_heatmap`, :func:`dibujar_plano_ranking`,
  :func:`guardar_figura`.

Convenciones (comunes a los siete notebooks):

- El estado de crisis de un detector es ``n_states - 1`` (orden económico canónico).
- Figuras en ``results/detectores/<familia>/`` con nombres
  ``<Dxx>_<pista>_estados_oos.png``, ``<Dxx>_<pista>_cobertura_crisis.png``,
  ``<Dxx>_<pista>_<diagnostico>.png`` y ``<Fk>_comparativa_<tema>.png``.
- Franjas azules = crisis del catálogo de la pista ``[pico, suelo]``; grises = trampas.

Este módulo no entra en la huella de caché del benchmark (no cambia predicciones ni
métricas): modificarlo no invalida ``results/benchmark``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from regimenes import rutas
from regimenes.viz import figuras as _viz

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
#: Columnas del ranking ADR-003 que resume cada notebook de familia (en este orden).
COLUMNAS_RANKING = [
    "puesto_deteccion", "nivel_etiqueta", "score_deteccion", "det_event_recall",
    "det_n_detectados", "det_n_eventos", "det_precision", "det_lift_precision",
    "det_base_rate", "det_marked_rate", "det_mean_crisis_run", "mean_crisis_coverage",
    "false_alarm_rate", "mean_trap_activation", "switching_rate", "mean_regime_duration",
    "label_stability", "elapsed_seconds",
]

#: Columnas con las que se comprueba que el ranking corresponde a la caché cargada.
COLUMNAS_COHERENCIA = [
    "det_n_detectados", "det_n_eventos", "det_event_recall", "det_precision", "det_marked_rate",
]

#: Columnas de la tabla de cobertura por crisis (:func:`tabla_cobertura`).
COLUMNAS_COBERTURA = ["pico", "suelo", "cobertura", "ic_lo", "ic_hi", "detectada", "lead_lag_dias"]

#: Nombre de los estados canónicos por número de estados (0 = calma … K-1 = crisis).
ETIQUETAS_ESTADOS = {
    2: ["calma", "crisis"],
    3: ["calma", "corrección", "crisis"],
    4: ["calma", "leve", "corrección", "crisis"],
}

#: Atributos del constructor que se muestran como parámetros efectivos (si existen).
ATRIBUTOS_EFECTIVOS = (
    "n_states", "q_in", "q_out", "min_dwell", "cov_min_periods", "ridge", "jump_penalty",
)


def etiquetas_estados(n_estados: int) -> list[str]:
    """Nombres de los ``n_estados`` estados canónicos (0 = calma … K-1 = crisis)."""
    if n_estados in ETIQUETAS_ESTADOS:
        return list(ETIQUETAS_ESTADOS[n_estados])
    return ["calma"] + [f"intermedio {k}" for k in range(1, n_estados - 1)] + ["crisis"]


def ruta_relativa(ruta: Path | str) -> str:
    """Ruta relativa a la raíz del repositorio (POSIX) para imprimir; absoluta si está fuera."""
    ruta = Path(ruta)
    try:
        return ruta.resolve().relative_to(rutas.ROOT.resolve()).as_posix()
    except ValueError:
        return ruta.as_posix()


def opciones_pandas(max_columnas: int = 80, ancho: int = 220) -> None:
    """Opciones de visualización de pandas comunes a los notebooks de familia."""
    pd.set_option("display.max_columns", max_columnas)
    pd.set_option("display.width", ancho)


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
def carpeta_figuras(familia: str, *, crear: bool = True) -> Path:
    """``results/detectores/<familia>/`` (se crea si no existe)."""
    carpeta = rutas.RESULTS_DETECTORES / familia
    if crear:
        carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


def comando_benchmark(pistas: Iterable[str], detectores: Iterable[str]) -> str:
    """Comando de la CLI que recalcula la caché de una familia."""
    return (
        "python -m regimenes.benchmark --track " + " ".join(p.upper() for p in pistas)
        + " --detector " + " ".join(d.upper() for d in detectores)
    )


def specs_familia(detectores: Iterable[str], pistas: Iterable[str] = ("A", "B")) -> dict:
    """``{(pista, id): DetectorSpec}`` del registro para los detectores de una familia."""
    from regimenes.detectores import detector_specs

    ids = {d.upper() for d in detectores}
    return {
        (t.upper(), s.detector_id): s
        for t in pistas
        for s in detector_specs(t.upper())
        if s.detector_id in ids
    }


def tabla_configuracion(specs: Mapping, train_days: Mapping[str, int] | None = None) -> pd.DataFrame:
    """Tabla de configuración generada desde el registro (una fila por detector y pista).

    Columnas: detector, familia del registro, clase, features, parámetros declarados en el
    ``DetectorSpec``, parámetros por defecto del constructor que el registro NO fija (los
    que realmente usa el detector sin que aparezcan en el registro), atributos efectivos
    clave de la instancia (:data:`ATRIBUTOS_EFECTIVOS`), semilla, ``step``, train inicial,
    tipo de ventana, si es lento y la variante v2 declarada.
    """
    import inspect

    if train_days is None:
        from regimenes.benchmark.ejecucion import DEFAULT_TRAIN_DAYS as train_days
    filas = []
    for (t, d), s in sorted(specs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        por_defecto: dict = {}
        try:
            inst = s.factory()
            efectivos = {a: getattr(inst, a) for a in ATRIBUTOS_EFECTIVOS if hasattr(inst, a)}
            firma = inspect.signature(type(inst).__init__)
            por_defecto = {
                k: p.default for k, p in firma.parameters.items()
                if k not in ("self", "features", "random_state") and k not in s.params
                and p.default is not inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            }
        except ModuleNotFoundError as exc:  # extra opcional ([deep], [jump]) no instalado
            efectivos = {"no_instanciable": f"falta {exc.name}"}
        except (TypeError, ValueError):     # firma no inspeccionable
            pass
        declarados = {k: v for k, v in s.params.items() if k not in ("features", "random_state")}
        dias = int(train_days[t])
        filas.append({
            "id": d,
            "pista": t,
            "detector": s.label,
            "familia_registro": s.family,
            "clase": f"{s.module}.{s.class_name}",
            "n_features": len(s.features),
            "features": ", ".join(s.features),
            "params_declarados": ", ".join(f"{k}={v}" for k, v in declarados.items()) or "—",
            "params_por_defecto": ", ".join(f"{k}={v}" for k, v in por_defecto.items()) or "—",
            "params_efectivos": ", ".join(f"{k}={v}" for k, v in efectivos.items()) or "—",
            "semilla": s.params.get("random_state", "no consume azar"),
            "step": s.step,
            "train_inicial": f"{dias} ses. (~{dias / 252:.0f} años)",
            "ventana": "expanding" if s.expanding else "rolling",
            "lento": bool(s.slow),
            "variante_v2": s.variant or "—",
        })
    return pd.DataFrame(filas).set_index(["id", "pista"])


# --------------------------------------------------------------------------- #
# Ranking ADR-003
# --------------------------------------------------------------------------- #
def buscar_ranking(output_dir: Path | str = rutas.RESULTS_BENCHMARK) -> Path:
    """Ruta del ranking de detección: ``ranking.csv``, si no ``ranking_v2.csv``, si no el
    último ``ranking*.csv`` por nombre."""
    output_dir = Path(output_dir)
    preferidos = [output_dir / "ranking.csv", output_dir / "ranking_v2.csv"]
    candidatos = [p for p in preferidos if p.is_file()] or sorted(output_dir.glob("ranking*.csv"))
    if not candidatos:
        raise FileNotFoundError(
            f"No hay ranking*.csv en {ruta_relativa(output_dir)}. Ejecuta "
            "notebooks/12_comparativa.ipynb (escribe el ranking de detección ADR-003)."
        )
    return candidatos[0]


def leer_ranking(ruta: Path | str) -> pd.DataFrame:
    """Lee un ``ranking*.csv`` normalizando la pista a mayúsculas."""
    tabla = pd.read_csv(ruta)
    tabla["pista"] = tabla["pista"].astype(str).str.upper()
    return tabla


def claves_incoherentes(
    ranking: pd.DataFrame,
    metricas: pd.DataFrame,
    claves: Iterable[tuple[str, str]],
    columnas: Sequence[str] = COLUMNAS_COHERENCIA,
    atol: float = 1e-9,
) -> list[str]:
    """Combinaciones ``pista/id`` cuyo ranking no reproduce las métricas verificadas.

    ``ranking`` es plano (columnas ``pista`` e ``id``); ``metricas`` está indexado por
    ``(pista, id)``. Se comparan solo las columnas presentes en ambos.
    """
    rank = ranking.set_index(["pista", "id"])
    columnas = [c for c in columnas if c in rank.columns and c in metricas.columns]
    malas = []
    for clave in claves:
        etiqueta = f"{clave[0]}/{clave[1]}"
        if clave not in rank.index:
            malas.append(f"{etiqueta} (ausente del ranking)")
            continue
        if clave not in metricas.index:
            malas.append(f"{etiqueta} (sin métricas)")
            continue
        a = pd.to_numeric(rank.loc[clave, columnas], errors="coerce").to_numpy(float)
        b = pd.to_numeric(metricas.loc[clave, columnas], errors="coerce").to_numpy(float)
        if not np.allclose(a, b, atol=atol, rtol=0.0, equal_nan=True):
            malas.append(f"{etiqueta} (difiere)")
    return malas


def tabla_resumen_ranking(
    ranking: pd.DataFrame,
    claves: Iterable[tuple[str, str]],
    columnas: Sequence[str] = COLUMNAS_RANKING,
) -> pd.DataFrame:
    """Filas del ranking ADR-003 de ``claves`` con la columna ``de`` (detectores en la pista)."""
    rank = ranking.set_index(["pista", "id"])
    n_pista = ranking.groupby("pista")["id"].nunique()
    claves = list(claves)
    tabla = rank.loc[claves, [c for c in columnas if c in rank.columns]].copy()
    tabla.insert(1, "de", [int(n_pista[t]) for t, _ in tabla.index])
    return tabla


# --------------------------------------------------------------------------- #
# Carga de resultados de una familia
# --------------------------------------------------------------------------- #
def leer_panel(ruta: Path | str) -> pd.DataFrame:
    """Panel OOS (``state``, ``p_crisis``, ``fold``) con índice ``DatetimeIndex``."""
    panel = pd.read_parquet(ruta)
    panel.index = pd.to_datetime(panel.index)
    return panel


@dataclass
class ResultadosFamilia:
    """Resultados OOS verificados de los detectores de una familia.

    ``metricas`` está indexado por ``(pista, id)`` (una fila del CSV del benchmark por
    combinación), ``ranking`` es el ranking ADR-003 completo (todas las pistas y
    detectores, plano) y ``paneles`` es ``{(pista, id): panel OOS}``.
    """

    detectores: list[str]
    pistas: list[str]
    metricas: pd.DataFrame
    estado_cache: pd.DataFrame
    paneles: dict
    ranking: pd.DataFrame
    ruta_ranking: Path
    output_dir: Path
    comando: str
    incoherencias: list[str] = field(default_factory=list)

    @property
    def claves(self) -> list[tuple[str, str]]:
        return [(t, d) for t in self.pistas for d in self.detectores]

    @property
    def rank(self) -> pd.DataFrame:
        """Ranking indexado por ``(pista, id)``."""
        return self.ranking.set_index(["pista", "id"])

    @property
    def n_por_pista(self) -> dict[str, int]:
        """Número de detectores en el ranking de cada pista."""
        return self.ranking.groupby("pista")["id"].nunique().astype(int).to_dict()

    def fila(self, pista: str, det: str) -> pd.Series:
        """Métricas verificadas de una combinación."""
        return self.metricas.loc[(pista, det)]

    def fila_ranking(self, pista: str, det: str) -> pd.Series:
        """Fila del ranking ADR-003 de una combinación."""
        return self.rank.loc[(pista, det)]

    def puesto(self, pista: str, det: str) -> str:
        """``'<puesto> de <n>'`` en el ranking de detección de la pista."""
        r = self.fila_ranking(pista, det)
        return f"{int(r['puesto_deteccion'])} de {self.n_por_pista[pista]}"

    def n_estados(self, pista: str, det: str) -> int:
        return int(self.metricas.loc[(pista, det), "n_states"])

    def estado_crisis(self, pista: str, det: str) -> int:
        """Estado canónico de crisis (``n_states - 1``)."""
        return self.n_estados(pista, det) - 1

    def es_crisis(self, pista: str, det: str) -> pd.Series:
        """Serie booleana OOS: la sesión está en el estado de crisis."""
        return self.paneles[(pista, det)]["state"].eq(self.estado_crisis(pista, det))

    def resumen_ranking(self, detectores: Iterable[str] | None = None,
                        columnas: Sequence[str] = COLUMNAS_RANKING) -> pd.DataFrame:
        """Tabla del ranking ADR-003 de los detectores (por defecto, los de la familia)."""
        ids = list(detectores) if detectores is not None else self.detectores
        return tabla_resumen_ranking(self.ranking, [(t, d) for d in ids for t in self.pistas], columnas)


def verificar_estado_cache(estado_cache: pd.DataFrame, comando: str) -> None:
    """Falla con el comando a ejecutar si alguna combinación no tiene caché vigente."""
    if estado_cache is None or estado_cache.empty:
        raise RuntimeError(
            "La caché del benchmark no contiene ninguna de las combinaciones pedidas.\n"
            f"Ejecuta desde la raíz del repositorio:\n    {comando}\n"
            "(o pon EJECUTAR = True en la celda de §3) y vuelve a ejecutar el notebook."
        )
    malas = estado_cache[estado_cache["estado"] != "cache"]
    if len(malas):
        faltan = ", ".join(f"{r.pista}/{r.id}" for r in malas.itertuples())
        raise RuntimeError(
            f"La caché del benchmark NO está vigente para: {faltan}.\n"
            f"Ejecuta desde la raíz del repositorio:\n    {comando}\n"
            "(o pon EJECUTAR = True en la celda de §3) y vuelve a ejecutar el notebook."
        )


def ejecutar_familia(
    detectores: Iterable[str],
    pistas: Iterable[str] = ("A", "B"),
    *,
    output_dir: Path | str = rutas.RESULTS_BENCHMARK,
    forzar: bool = False,
) -> pd.DataFrame:
    """Gate previo + ``run_job`` por combinación (mismo camino que la CLI por detector).

    Solo escribe los archivos propios de cada combinación (``metrics/``, ``panels/`` y
    ``status/``); no toca ``manifest.json`` ni ``run_status.csv`` (consolidar después con
    ``python -m regimenes.benchmark --consolidate``).
    """
    from regimenes.benchmark.ejecucion import preflight, run_job

    pistas = [p.upper() for p in pistas]
    detectores = [d.upper() for d in detectores]
    specs = specs_familia(detectores, pistas)
    gate = pd.concat(
        [preflight(t, [specs[(t, d)] for d in detectores if (t, d) in specs]) for t in pistas],
        ignore_index=True,
    )
    if not gate[["datos_ok", "features_ok", "dependencia_ok"]].all().all():
        raise RuntimeError(
            "Gate previo en rojo (ver 04_protocolo_evaluacion):\n" + gate.to_string(index=False)
        )
    estados = [run_job(t, d, output_dir=Path(output_dir), force=forzar) for t in pistas for d in detectores]
    return pd.DataFrame(estados)


def cargar_resultados_familia(
    detectores: Iterable[str],
    pistas: Iterable[str] = ("A", "B"),
    *,
    output_dir: Path | str = rutas.RESULTS_BENCHMARK,
    exigir_ranking_coherente: bool = True,
) -> ResultadosFamilia:
    """Carga (solo lectura) métricas verificadas, paneles OOS y ranking ADR-003.

    1. ``run_benchmark(cache_only=True)``: solo acepta combinaciones cuya huella (código +
       datos + especificación + entorno) coincide con la actual y cuyo panel OOS existe.
       Si alguna no está vigente → ``RuntimeError`` con el comando
       ``python -m regimenes.benchmark --track … --detector …``.
       Se añaden ``mean_crisis_coverage`` y ``mean_trap_activation``
       (``ranking.add_comparison_metrics``).
    2. Paneles ``panels/<P>_<D>.parquet``; se comprueba que su largo es ``n_oos``.
    3. ``ranking*.csv`` (:func:`buscar_ranking`) y su coherencia con las métricas; si no
       corresponde (benchmark re-ejecutado sin regenerar ``12_comparativa``) y
       ``exigir_ranking_coherente`` → ``RuntimeError``.
    """
    from regimenes.benchmark.cache import _panel_path
    from regimenes.benchmark.ejecucion import run_benchmark

    pistas = [p.upper() for p in pistas]
    detectores = [d.upper() for d in detectores]
    output_dir = Path(output_dir)
    comando = comando_benchmark(pistas, detectores)

    metricas, estado = run_benchmark(
        tracks=pistas, detector_ids=detectores, output_dir=output_dir, cache_only=True
    )
    verificar_estado_cache(estado, comando)
    from regimenes.evaluacion.ranking import add_comparison_metrics

    metricas = add_comparison_metrics(metricas.assign(pista=metricas["pista"].astype(str).str.upper()))
    metricas = metricas.set_index(["pista", "id"]).sort_index()

    paneles = {}
    for t in pistas:
        for d in detectores:
            ruta = _panel_path(output_dir, t, d)
            if not ruta.is_file():
                raise FileNotFoundError(f"Falta el panel OOS {ruta_relativa(ruta)}. Ejecuta: {comando}")
            panel = leer_panel(ruta)
            n_oos = metricas.loc[(t, d)].get("n_oos", np.nan)
            if pd.notna(n_oos) and len(panel) != int(n_oos):
                raise RuntimeError(
                    f"{t}/{d}: el panel OOS ({len(panel)} sesiones) no cuadra con n_oos={int(n_oos)}. "
                    f"Ejecuta: {comando}"
                )
            paneles[(t, d)] = panel

    ruta_ranking = buscar_ranking(output_dir)
    ranking = leer_ranking(ruta_ranking)
    claves = [(t, d) for t in pistas for d in detectores]
    malas = claves_incoherentes(ranking, metricas, claves)
    if malas and exigir_ranking_coherente:
        raise RuntimeError(
            f"{ruta_ranking.name} no corresponde a las métricas verificadas de la familia "
            f"({', '.join(malas)}). Vuelve a ejecutar notebooks/12_comparativa.ipynb."
        )
    return ResultadosFamilia(
        detectores=detectores, pistas=pistas, metricas=metricas, estado_cache=estado,
        paneles=paneles, ranking=ranking, ruta_ranking=ruta_ranking, output_dir=output_dir,
        comando=comando, incoherencias=malas,
    )


# --------------------------------------------------------------------------- #
# Contexto de las pistas (datos procesados y precio del S&P 500)
# --------------------------------------------------------------------------- #
def precio_sp500() -> pd.Series:
    """Precio de cierre del S&P 500 (espina cruda de ``data/raw``) con índice de fechas."""
    from regimenes.benchmark.cache import _sp500_path

    precio = pd.read_parquet(_sp500_path())["SP500"].sort_index().astype(float)
    precio.index = pd.to_datetime(precio.index)
    return precio


@dataclass
class ContextoPistas:
    """Datos de apoyo de las pistas (no dependen de ``results/``).

    ``datos[t]`` es el panel procesado de la pista, ``indice[t]`` la rejilla común de los
    doce detectores, ``sp500`` el precio del S&P 500, ``crisis[t]`` las ventanas
    ``[pico, suelo]`` del juez y ``trampas`` las ventanas de falso positivo.
    """

    pistas: list[str]
    datos: dict
    indice: dict
    sp500: pd.Series
    crisis: dict
    trampas: dict

    def matriz(self, pista: str, features: Iterable[str]) -> pd.DataFrame:
        """``X`` que ve el benchmark: panel de la pista en la rejilla común, ``features``."""
        return self.datos[pista].loc[self.indice[pista], list(features)]

    def retornos(self, pista: str, index: pd.Index | None = None) -> pd.Series:
        """Retorno log diario del S&P 500 de la pista (``SP500_ret``)."""
        ret = self.datos[pista]["SP500_ret"]
        return ret.reindex(index) if index is not None else ret


def cargar_contexto_pistas(pistas: Iterable[str] = ("A", "B")) -> ContextoPistas:
    """Paneles procesados, rejilla común, S&P 500 y ventanas del juez de cada pista."""
    from regimenes.benchmark.cache import processed_available
    from regimenes.benchmark.ejecucion import common_track_index, load_track_panel
    from regimenes.evaluacion import ranking as rk

    pistas = [p.upper() for p in pistas]
    if not processed_available():
        raise FileNotFoundError(
            "Faltan los paneles procesados en data/processed. Ejecuta `python -m regimenes.datos` "
            "y después notebooks/03_preprocesado.ipynb."
        )
    datos = {t: load_track_panel(t) for t in pistas}
    return ContextoPistas(
        pistas=pistas,
        datos=datos,
        indice={t: common_track_index(t, datos[t]) for t in pistas},
        sp500=precio_sp500(),
        crisis={t: rk.track_crisis_windows(t) for t in pistas},
        trampas=rk.track_false_positive_windows(),
    )


# --------------------------------------------------------------------------- #
# Tablas
# --------------------------------------------------------------------------- #
def tabla_cobertura(fila: pd.Series, crisis: Mapping[str, tuple], *, solo_oos: bool = True) -> pd.DataFrame:
    """Cobertura OOS por crisis a partir de una fila de métricas del benchmark.

    Columnas (:data:`COLUMNAS_COBERTURA`): ``pico``, ``suelo``, ``cobertura`` (fracción de
    sesiones de ``[pico, suelo]`` marcadas como crisis), ``ic_lo``/``ic_hi`` (IC bootstrap
    por bloques), ``detectada`` (1.0/0.0/NaN: evento detectado según ADR-003, racha de al
    menos ``min_run`` sesiones) y ``lead_lag_dias`` (negativo = la señal llega antes del
    suelo). Con ``solo_oos`` se omiten las crisis sin cobertura (caen en el train).
    """
    filas = []
    for nombre, (pico, suelo) in crisis.items():
        cov = pd.to_numeric(fila.get(f"cov_{nombre}", np.nan), errors="coerce")
        if solo_oos and pd.isna(cov):
            continue
        filas.append({
            "crisis": nombre,
            "pico": str(pico),
            "suelo": str(suelo),
            "cobertura": float(cov),
            "ic_lo": float(pd.to_numeric(fila.get(f"cov_{nombre}_lo", np.nan), errors="coerce")),
            "ic_hi": float(pd.to_numeric(fila.get(f"cov_{nombre}_hi", np.nan), errors="coerce")),
            "detectada": float(pd.to_numeric(fila.get(f"det_ev_{nombre}", np.nan), errors="coerce")),
            "lead_lag_dias": float(pd.to_numeric(fila.get(f"leadlag_{nombre}", np.nan), errors="coerce")),
        })
    if not filas:
        return pd.DataFrame(columns=COLUMNAS_COBERTURA, index=pd.Index([], name="crisis"))
    return pd.DataFrame(filas).set_index("crisis")[COLUMNAS_COBERTURA]


def tabla_trampas(fila: pd.Series, trampas: Iterable[str] | Mapping | None = None) -> pd.Series:
    """Activación en las ventanas trampa (``fa_<trampa>``; fracción marcada, menor es mejor).

    Con ``trampas`` (nombres o dict de ventanas) se devuelven solo esas, en ese orden;
    si no, todas las columnas ``fa_*`` de la fila.
    """
    if trampas is None:
        nombres = [c.removeprefix("fa_") for c in fila.index if str(c).startswith("fa_")]
    else:
        nombres = list(trampas)
    valores = {n: float(pd.to_numeric(fila.get(f"fa_{n}", np.nan), errors="coerce")) for n in nombres}
    return pd.Series(valores, name="activacion_trampa", dtype=float)


def matriz_jaccard(flags: pd.DataFrame | Mapping[str, pd.Series]) -> pd.DataFrame:
    """Índice de Jaccard entre las sesiones marcadas por cada columna (1 = mismos días).

    Las series se alinean por índice (intersección); NaN si ninguna de las dos marca.
    """
    marco = pd.DataFrame(flags).dropna().astype(bool)
    cols = list(marco.columns)
    J = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            union = int((marco[a] | marco[b]).sum())
            J.loc[a, b] = (marco[a] & marco[b]).sum() / union if union else np.nan
    return J


def retardo_confirmacion(flags: pd.Series, crisis: Mapping[str, tuple]) -> pd.Series:
    """Sesiones OOS desde el pico de cada crisis hasta el primer día marcado como crisis.

    0 = ya estaba en crisis el día del pico; NaN = no marca ningún día de la ventana. Se
    omiten las crisis sin sesiones en ``flags``.
    """
    flags = flags.astype(bool)
    out = {}
    for nombre, (pico, suelo) in crisis.items():
        tramo = flags.loc[pd.Timestamp(pico):pd.Timestamp(suelo)]
        if tramo.empty:
            continue
        out[nombre] = float(np.argmax(tramo.to_numpy())) if tramo.any() else np.nan
    return pd.Series(out, name="retardo_confirmacion", dtype=float)


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def guardar_figura(fig, carpeta: Path | str, nombre: str, *, verbose: bool = True) -> Path:
    """Guarda ``fig`` como ``<carpeta>/<nombre>.png`` con el estilo de casa (dpi y bbox de
    ``viz.HOUSE_RC``). No cierra ni muestra la figura."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    nombre = nombre[:-4] if nombre.endswith(".png") else nombre
    ruta = carpeta / f"{nombre}.png"
    fig.savefig(ruta, dpi=_viz.HOUSE_RC["savefig.dpi"], bbox_inches="tight")
    if verbose:
        print("figura ->", ruta_relativa(ruta))
    return ruta


def franjas_eventos(ax, crisis: Mapping | None = None, trampas: Mapping | None = None, *,
                    etiquetas: bool = True, alpha_crisis: float = 0.10, alpha_trampas: float = 0.25) -> None:
    """Franjas verticales: crisis del catálogo ``[pico, suelo]`` (azul) y trampas (gris)."""
    for i, (a, b) in enumerate((crisis or {}).values()):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=_viz.C_LONG, alpha=alpha_crisis, lw=0,
                   label="crisis del catálogo [pico, suelo]" if etiquetas and i == 0 else None)
    for i, (a, b) in enumerate((trampas or {}).values()):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=_viz.C_FP, alpha=alpha_trampas, lw=0,
                   label="ventana trampa (no-crisis)" if etiquetas and i == 0 else None)


def dibujar_estados_sp500(
    ax,
    precio: pd.Series,
    estados: pd.Series,
    n_estados: int,
    *,
    crisis: Mapping | None = None,
    trampas: Mapping | None = None,
    todos_los_estados: bool = False,
    etiquetas: Sequence[str] | None = None,
    leyenda: bool = True,
):
    """S&P 500 (log) sobre las sesiones OOS con los estados del detector.

    Por defecto sombrea en rojo los tramos en el estado de crisis (``n_estados - 1``);
    con ``todos_los_estados`` colorea cada estado canónico (útil con ``K > 2``). Añade las
    franjas de crisis del catálogo y de trampas.
    """
    from matplotlib.patches import Patch

    estados = estados.astype(int)
    px = precio.reindex(estados.index).ffill()
    ax.plot(px.index, px.to_numpy(), color="black", lw=0.7, zorder=3)
    ax.set_yscale("log")
    crisis_state = int(n_estados) - 1
    etiquetas = list(etiquetas) if etiquetas is not None else etiquetas_estados(int(n_estados))
    handles = []
    if todos_los_estados and n_estados > 2:
        for s in range(int(n_estados)):
            color = _viz.regime_color(s, int(n_estados))
            _viz.shade_regime(ax, estados, s, color=color, alpha=0.30)
            handles.append(Patch(color=color, alpha=0.6, label=etiquetas[s]))
    else:
        _viz.shade_regime(ax, estados, crisis_state)
        handles.append(Patch(color=_viz.C_CRISIS, alpha=0.35, label="crisis marcada por el detector (OOS)"))
    franjas_eventos(ax, crisis, trampas, etiquetas=False)
    if crisis:
        handles.append(Patch(color=_viz.C_LONG, alpha=0.25, label="crisis del catálogo [pico, suelo]"))
    if trampas:
        handles.append(Patch(color=_viz.C_FP, alpha=0.4, label="ventana trampa (no-crisis)"))
    if len(estados):
        ax.set_xlim(estados.index[0], estados.index[-1])
    ax.set_ylabel("S&P 500 (log)")
    if leyenda:
        ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9,
                  ncol=min(len(handles), 3))
    return ax


def figura_estados_oos(
    precio: pd.Series,
    panel: pd.DataFrame,
    n_estados: int,
    *,
    crisis: Mapping | None = None,
    trampas: Mapping | None = None,
    titulo: str = "",
    diagnostico: Callable | None = None,
    titulo_diagnostico: str | None = None,
    todos_los_estados: bool = False,
    etiquetas: Sequence[str] | None = None,
    figsize: tuple[float, float] = (15, 7.5),
):
    """Figura canónica de §4: arriba estados OOS sobre el S&P 500; abajo el diagnóstico.

    ``diagnostico(ax)`` dibuja el diagnóstico propio del detector en el panel inferior; si
    es ``None`` se dibuja ``p_crisis`` OOS del panel (causal). Devuelve ``(fig, (ax1, ax2))``.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    dibujar_estados_sp500(ax1, precio, panel["state"], n_estados, crisis=crisis, trampas=trampas,
                          todos_los_estados=todos_los_estados, etiquetas=etiquetas)
    ax1.set_title(titulo)
    if diagnostico is None:
        p = panel["p_crisis"]
        ax2.plot(p.index, p.to_numpy(), color=_viz.C_CRISIS, lw=0.7, label="P(crisis) OOS (causal)")
        ax2.axhline(0.5, color="black", ls=":", lw=0.8, alpha=0.6)
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_ylabel("P(crisis)")
        ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)
    else:
        diagnostico(ax2)
    franjas_eventos(ax2, crisis, trampas, etiquetas=False)
    ax2.set_title(titulo_diagnostico or "P(crisis) OOS del panel (causal)", fontsize=10)
    fig.tight_layout()
    return fig, (ax1, ax2)


def dibujar_cobertura(
    ax,
    tabla: pd.DataFrame,
    trampas: pd.Series | Mapping | None = None,
    *,
    titulo: str = "",
    extra: pd.Series | None = None,
    etiqueta_extra: str = "",
    min_run: int | None = None,
):
    """Barras horizontales de cobertura por crisis (tabla de :func:`tabla_cobertura`).

    Rojo = evento detectado (ADR-003), gris oscuro = no detectado; barras de error = IC
    bootstrap; línea punteada en 0.5. ``trampas`` añade barras grises de activación en las
    ventanas trampa (menor es mejor). ``extra`` (misma indexación) se marca con rombos
    negros (p. ej. *corrección + crisis* en detectores con ``K > 2``).
    """
    tab = tabla.copy()
    nombres = list(tab.index)
    valores = tab["cobertura"].astype(float).tolist()
    colores = [_viz.C_CRISIS if d == 1 else _viz.C_NEG for d in tab["detectada"]]
    err_lo = (tab["cobertura"] - tab["ic_lo"]).clip(lower=0).fillna(0).tolist()
    err_hi = (tab["ic_hi"] - tab["cobertura"]).clip(lower=0).fillna(0).tolist()
    if trampas is not None and len(trampas):
        tr = pd.Series(trampas, dtype=float)
        nombres += [f"trampa: {k}" for k in tr.index]
        valores += tr.fillna(0).tolist()
        colores += [_viz.C_FP] * len(tr)
        err_lo += [0.0] * len(tr)
        err_hi += [0.0] * len(tr)
    y = np.arange(len(nombres))
    ax.barh(y, valores, color=colores, edgecolor="black", lw=0.4)
    if len(y):
        ax.errorbar(valores, y, xerr=[err_lo, err_hi], fmt="none", ecolor="black", lw=0.8, capsize=2)
    if extra is not None:
        ex = pd.Series(extra, dtype=float).reindex(tab.index)
        ok = ex.notna().to_numpy()
        ax.scatter(ex.to_numpy()[ok], np.arange(len(tab))[ok], marker="D", color="black", s=18,
                   zorder=4, label=etiqueta_extra or None)
        if etiqueta_extra:
            ax.legend(loc="lower right", fontsize=8)
    ax.set_yticks(y, nombres, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.axvline(0.5, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("fracción de sesiones de la ventana marcadas como crisis (OOS)")
    regla = f" (≥{min_run} sesiones seguidas)" if min_run else ""
    ax.set_title(titulo or f"cobertura por crisis: rojo = detectada{regla}; gris = trampas", fontsize=10)
    return ax


def figura_cobertura(
    tabla: pd.DataFrame,
    trampas: pd.Series | Mapping | None = None,
    *,
    titulo: str = "",
    extra: pd.Series | None = None,
    etiqueta_extra: str = "",
    min_run: int | None = None,
):
    """Figura de cobertura por crisis de UNA combinación detector × pista."""
    import matplotlib.pyplot as plt

    n = len(tabla) + (len(trampas) if trampas is not None else 0)
    fig, ax = plt.subplots(figsize=(12, 0.34 * n + 1.6))
    dibujar_cobertura(ax, tabla, trampas, titulo=titulo, extra=extra,
                      etiqueta_extra=etiqueta_extra, min_run=min_run)
    fig.tight_layout()
    return fig, ax


def dibujar_heatmap(ax, matriz: pd.DataFrame, *, cmap: str = "Blues", vmin: float | None = 0.0,
                    vmax: float | None = 1.0, fmt: str = "{:.2f}", titulo: str = "",
                    umbral_blanco: float = 0.6, fontsize: int = 7):
    """Mapa de calor anotado de un ``DataFrame`` (Jaccard, cobertura crisis × detector…).

    Las celdas NaN quedan en blanco y sin anotar; con ``vmin=None`` se usa una escala
    simétrica ``[-max|x|, max|x|]`` (útil para medias estandarizadas con ``cmap='RdBu_r'``).
    """
    valores = matriz.to_numpy(float)
    if vmin is None or vmax is None:
        lim = float(np.nanmax(np.abs(valores))) if np.isfinite(valores).any() else 1.0
        vmin, vmax = -lim, lim
    ax.imshow(np.ma.masked_invalid(valores), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(matriz.shape[1]), [str(c) for c in matriz.columns], fontsize=8)
    ax.set_yticks(range(matriz.shape[0]), [str(i) for i in matriz.index], fontsize=8)
    escala = max(abs(vmin), abs(vmax)) or 1.0
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            v = valores[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
                        color="white" if abs(v) > umbral_blanco * escala else "black")
    if titulo:
        ax.set_title(titulo, fontsize=10)
    return ax


def dibujar_plano_ranking(
    ax,
    ranking_pista: pd.DataFrame,
    resaltados: Iterable[str],
    *,
    beta: float = 1.0,
    etiqueta: str = "familia",
    isolineas: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    titulo: str = "",
):
    """Plano *recall por evento × precisión diaria* de una pista (ranking ADR-003).

    Todos los detectores de la pista en gris, ``resaltados`` en rojo con su puesto,
    iso-curvas de ``score_deteccion`` (F-beta) punteadas y la precisión del azar
    (tasa base) discontinua.
    """
    resaltados = list(resaltados)
    r_grid = np.linspace(0.01, 1, 400)
    b2 = float(beta) ** 2
    for f in isolineas:
        den = (1 + b2) * r_grid - b2 * f
        with np.errstate(divide="ignore", invalid="ignore"):
            p = f * r_grid / den
        m = (den > 0) & (p > 0) & (p <= 1)
        ax.plot(r_grid[m], p[m], color="grey", lw=0.5, ls=":")
        if m.any():
            ax.annotate(f"score={f:g}", (r_grid[m][-1], p[m][-1]), fontsize=7, color="grey")
    if "det_base_rate" in ranking_pista and len(ranking_pista):
        ax.axhline(float(ranking_pista["det_base_rate"].iloc[0]), color=_viz.C_FP, ls="--", lw=1,
                   label="precisión del azar (tasa base)")
    otros = ranking_pista[~ranking_pista["id"].isin(resaltados)]
    fam = ranking_pista[ranking_pista["id"].isin(resaltados)]
    ax.scatter(otros["det_event_recall"], otros["det_precision"], s=40, color=_viz.C_NA,
               edgecolor="grey", label="resto del banco")
    for _, r in otros.iterrows():
        ax.annotate(r["id"], (r["det_event_recall"], r["det_precision"]), fontsize=7, color="grey",
                    xytext=(3, 3), textcoords="offset points")
    ax.scatter(fam["det_event_recall"], fam["det_precision"], s=90, color=_viz.C_CRISIS,
               edgecolor="black", zorder=4, label=etiqueta)
    for _, r in fam.iterrows():
        ax.annotate(f"{r['id']} (#{int(r['puesto_deteccion'])})", (r["det_event_recall"], r["det_precision"]),
                    fontsize=9, fontweight="bold", xytext=(4, -10), textcoords="offset points")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("recall por evento (crisis detectadas / evaluables)")
    ax.set_ylabel("precisión diaria")
    if titulo:
        ax.set_title(titulo, fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    return ax


__all__ = [
    "COLUMNAS_RANKING", "COLUMNAS_COHERENCIA", "COLUMNAS_COBERTURA", "ETIQUETAS_ESTADOS",
    "ATRIBUTOS_EFECTIVOS", "etiquetas_estados", "ruta_relativa", "opciones_pandas",
    "carpeta_figuras", "comando_benchmark", "specs_familia", "tabla_configuracion",
    "buscar_ranking", "leer_ranking", "claves_incoherentes", "tabla_resumen_ranking",
    "leer_panel", "ResultadosFamilia", "verificar_estado_cache", "ejecutar_familia",
    "cargar_resultados_familia", "precio_sp500", "ContextoPistas", "cargar_contexto_pistas",
    "tabla_cobertura", "tabla_trampas", "matriz_jaccard", "retardo_confirmacion",
    "guardar_figura", "franjas_eventos", "dibujar_estados_sp500", "figura_estados_oos",
    "dibujar_cobertura", "figura_cobertura", "dibujar_heatmap", "dibujar_plano_ranking",
]
