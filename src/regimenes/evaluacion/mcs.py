"""
mcs.py — Model Confidence Set (Hansen, Lunde y Nason, 2011) para el ranking (A1).

Responde a «¿los primeros del ranking se distinguen de verdad?»: el MCS al nivel
``alpha`` es el conjunto de detectores que no se pueden descartar como peores que
el mejor, dada una matriz de pérdidas (filas = observaciones o bloques de
observaciones, columnas = modelos; menor es mejor).

- :func:`model_confidence_set`: implementación genérica (estadísticos ``T_max`` y
  ``T_R``, bootstrap por bloques circular o estacionario, eliminación secuencial
  y p-valores MCS).
- :func:`perdidas_diarias`: matriz **DEFINITIVA** día × detector desde los paneles
  walk-forward OOS (``results/benchmark/panels`` o cualquier carpeta con el mismo
  formato); :func:`replicas_perdida` la remuestrea por bloques de días recalculando
  coberturas y FPR en cada réplica; :func:`mcs_diario` lo monta todo con los
  baselines triviales y bootstrap estacionario de bloque medio
  :func:`longitud_bloque_crisis`.
- :func:`perdidas_por_crisis`: matriz **PRELIMINAR** crisis × detector desde los CSV
  versionados ``results/benchmark/metrics``; queda como análisis de sensibilidad.

Pérdida pre-registrada (ADR-008 §4): fallo por crisis + días de falsa alarma,
bootstrap por bloques de días, α = 0,10, con ``SIEMPRE_*``, ``AZAR_*`` y
``VOL_P90``. Las dos matrices tienen la MISMA pérdida media por detector
(1 − J de Youden con las crisis equiponderadas, ver :func:`perdidas_diarias`);
difieren en qué se remuestrea: días por bloques (definitiva) o crisis (preliminar,
bootstrap iid sobre crisis, con el término de falsa alarma constante).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

MCS_DEFAULTS: dict[str, float] = {
    "alpha": 0.10,   # decisión del usuario (6.9)
    "n_boot": 2000,
    "lam": 1.0,      # peso de las falsas alarmas en la pérdida por crisis
}


@dataclass
class MCSResult:
    """Resultado del MCS.

    ``tabla``: una fila por modelo con ``perdida_media``, ``p_mcs``,
    ``orden_eliminacion`` (1 = primero en salir; el último superviviente lleva
    ``n_modelos``) e ``incluido`` (``p_mcs >= alpha``), ordenada por ``p_mcs``
    descendente. ``supervivientes``: los modelos del MCS al nivel ``alpha``.
    """

    tabla: pd.DataFrame
    supervivientes: list[str]
    alpha: float
    estadistico: str
    block_len: int
    n_boot: int
    n_obs: int
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Bootstrap por bloques
# --------------------------------------------------------------------------- #
def bootstrap_indices(
    n: int, n_boot: int, *, block_len: int = 1, metodo: str = "estacionario",
    seed: int | None = 0,
) -> np.ndarray:
    """Índices de remuestreo ``(n_boot, n)`` por bloques, con envoltura circular.

    - ``circular``: bloques de longitud fija ``block_len`` (Politis y Romano, 1992).
    - ``estacionario``: longitudes geométricas de media ``block_len``
      (Politis y Romano, 1994; el que usan Hansen et al.).

    ``block_len = 1`` es el bootstrap iid en ambos casos.
    """
    if n < 1:
        raise ValueError("Hacen falta observaciones para el bootstrap.")
    block_len = max(1, int(block_len))
    rng = np.random.default_rng(seed)
    if block_len == 1:
        return rng.integers(0, n, size=(n_boot, n))
    if metodo == "circular":
        n_blocks = int(np.ceil(n / block_len))
        starts = rng.integers(0, n, size=(n_boot, n_blocks))
        idx = (starts[:, :, None] + np.arange(block_len)) % n
        return idx.reshape(n_boot, -1)[:, :n]
    if metodo != "estacionario":
        raise ValueError(f"metodo desconocido: {metodo!r} (circular | estacionario)")
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    nuevo = rng.random((n_boot, n)) < 1.0 / block_len
    saltos = rng.integers(0, n, size=(n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(nuevo[:, t], saltos[:, t], (idx[:, t - 1] + 1) % n)
    return idx


# Presupuesto de memoria del remuestreo (bytes). Con pérdidas diarias (≈ 14.000
# días × 17 modelos × 5.000 réplicas) ``L[idx]`` ocuparía ~10 GB; por encima del
# presupuesto las réplicas se generan por tandas.
_MEMORIA_BOOT = 64 * 1024 ** 2


def _medias_bootstrap(
    L: np.ndarray, n_boot: int, *, block_len: int, metodo: str, seed: int | None,
    memoria: int = _MEMORIA_BOOT,
) -> np.ndarray:
    """Medias bootstrap ``(n_boot, m)`` de las columnas de ``L`` ``(n, m)``.

    Si ``L[idx]`` cabe en ``memoria`` se hace de una vez (mismo resultado que la
    versión original con la misma ``seed``: el MCS preliminar no cambia). Si no,
    por tandas: cada tanda usa una semilla hija de ``SeedSequence(seed)`` y la
    media de la réplica b es ``recuento_b @ L / n`` (recuento = veces que sale cada
    observación), sin materializar ``(tanda, n, m)``. Reproducible con la misma
    ``seed`` y el mismo ``memoria``.
    """
    n, m = L.shape
    if n_boot * n * (m + 1) * 8 <= memoria:
        idx = bootstrap_indices(n, n_boot, block_len=block_len, metodo=metodo, seed=seed)
        return L[idx].mean(axis=1)
    out = np.empty((n_boot, m))
    for lo, hi, cuentas in _tandas_recuentos(n, n_boot, block_len=block_len, metodo=metodo,
                                             seed=seed, memoria=memoria):
        out[lo:hi] = cuentas @ L / n
    return out


def _tandas_recuentos(
    n: int, n_boot: int, *, block_len: int, metodo: str, seed: int | None,
    memoria: int = _MEMORIA_BOOT,
):
    """Genera ``(lo, hi, recuentos)`` por tandas de réplicas: ``recuentos[b, t]`` =
    veces que la réplica ``lo + b`` saca la observación t. Cada tanda usa una
    semilla hija de ``SeedSequence(seed)`` (reproducible con la misma ``seed`` y el
    mismo ``memoria``)."""
    tanda = max(1, int(memoria // (2 * 8 * n)))      # idx int64 + recuentos float64
    n_tandas = int(np.ceil(n_boot / tanda))
    for k, ss in enumerate(np.random.SeedSequence(seed).spawn(n_tandas)):
        lo, hi = k * tanda, min(n_boot, (k + 1) * tanda)
        idx = bootstrap_indices(n, hi - lo, block_len=block_len, metodo=metodo,
                                seed=int(ss.generate_state(1)[0]))
        cuentas = np.zeros((hi - lo, n))
        for b in range(hi - lo):
            cuentas[b] = np.bincount(idx[b], minlength=n)
        yield lo, hi, cuentas


def _tstat(diff: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """``diff / sd`` con varianza nula: 0 si la diferencia es 0, ±inf si no."""
    tiny = 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        t = diff / sd
        inf = np.where(np.abs(diff) <= tiny, 0.0, np.where(diff > 0, np.inf, -np.inf))
    return np.where(sd <= tiny, inf, t)


# --------------------------------------------------------------------------- #
# MCS
# --------------------------------------------------------------------------- #
def model_confidence_set(
    losses: pd.DataFrame | np.ndarray,
    *,
    alpha: float = MCS_DEFAULTS["alpha"],
    estadistico: str = "T_max",
    block_len: int = 1,
    metodo: str = "estacionario",
    n_boot: int = int(MCS_DEFAULTS["n_boot"]),
    seed: int | None = 0,
    replicas: np.ndarray | None = None,
) -> MCSResult:
    """Model Confidence Set de Hansen, Lunde y Nason (2011).

    ``losses``: filas = observaciones (o bloques), columnas = modelos; menor es
    mejor. Las filas con algún NaN se descartan (todas las comparaciones deben
    usar las mismas observaciones). Los índices bootstrap se generan una vez y se
    reutilizan en cada paso de la eliminación.

    Estadísticos (sobre el conjunto vivo M, con varianzas estimadas por bootstrap):

    - ``T_max`` = max_i d̄_i / sd(d̄_i), con d̄_i = media de L_i − media_{j∈M} L_j;
      se elimina argmax_i.
    - ``T_R`` = max_{i,j} |d̄_ij| / sd(d̄_ij); se elimina argmax_i max_j t_ij.

    p-valor del paso = P*(T* ≥ T) con T* recentrado; ``p_mcs`` = máximo acumulado
    de los p-valores hasta la eliminación del modelo (1 para el último). El MCS
    al nivel α son los modelos con ``p_mcs ≥ α``.

    ``replicas`` (opcional, ``(B, m)`` en el orden de las columnas): réplicas
    bootstrap ya calculadas de la pérdida de cada modelo, en lugar de las medias
    remuestreadas de ``losses``. Sirve para pérdidas que son una función no lineal
    de medias (p. ej. :func:`replicas_perdida`, que recalcula coberturas y FPR en
    cada réplica); el punto sigue siendo la media de ``losses``. Con ``replicas``
    se ignoran ``block_len``, ``metodo``, ``n_boot`` y ``seed``.
    """
    frame = pd.DataFrame(losses).astype(float)
    frame.columns = [str(c) for c in frame.columns]
    n_drop = int(frame.isna().any(axis=1).sum())
    frame = frame.dropna(how="any")
    if frame.shape[1] < 1 or frame.shape[0] < 2:
        raise ValueError("Hacen falta >= 2 observaciones completas y >= 1 modelo.")
    if estadistico not in {"T_max", "T_R"}:
        raise ValueError(f"estadistico desconocido: {estadistico!r} (T_max | T_R)")
    L = frame.to_numpy()
    n, m = L.shape
    names = list(frame.columns)
    mean = L.mean(axis=0)
    # Medias bootstrap de cada modelo (B, m): todas las d̄* son lineales en ellas.
    if replicas is None:
        boot = _medias_bootstrap(L, int(n_boot), block_len=block_len, metodo=metodo, seed=seed)
    else:
        boot = np.asarray(replicas, dtype=float)
        if n_drop or boot.shape[1] != m or not np.isfinite(boot).all():
            raise ValueError("replicas debe ser (B, m) finito y losses no puede tener NaN.")
        n_boot = boot.shape[0]

    alive = list(range(m))
    p_mcs = np.ones(m)
    p_step = np.full(m, np.nan)
    order = np.zeros(m, dtype=int)
    running = 0.0
    step = 0
    while len(alive) > 1:
        a = np.array(alive)
        if estadistico == "T_max":
            d = mean[a] - mean[a].mean()
            zeta = (boot[:, a] - boot[:, a].mean(axis=1, keepdims=True)) - d
            sd = np.sqrt((zeta ** 2).mean(axis=0))
            t = _tstat(d, sd)
            stat = t.max()
            stat_b = _tstat(zeta, sd).max(axis=1)
            worst = int(np.argmax(t))
        else:
            d = mean[a][:, None] - mean[a][None, :]
            zeta = (boot[:, a][:, :, None] - boot[:, a][:, None, :]) - d
            sd = np.sqrt((zeta ** 2).mean(axis=0))
            t = _tstat(d, sd)
            stat = np.abs(t).max()
            stat_b = np.abs(_tstat(zeta, sd)).reshape(len(zeta), -1).max(axis=1)
            worst = int(np.argmax(t.max(axis=1)))
        p = float(np.mean(stat_b >= stat))
        running = max(running, p)
        step += 1
        out = alive.pop(worst)
        p_step[out], p_mcs[out], order[out] = p, running, step
    order[alive[0]] = m
    tabla = pd.DataFrame({
        "perdida_media": mean,
        "p_mcs": p_mcs,
        "p_paso": p_step,
        "orden_eliminacion": order,
    }, index=pd.Index(names, name="modelo"))
    tabla["incluido"] = tabla["p_mcs"] >= alpha
    tabla = tabla.sort_values(["p_mcs", "orden_eliminacion"], ascending=False)
    return MCSResult(
        tabla=tabla,
        supervivientes=list(tabla.index[tabla["incluido"]]),
        alpha=float(alpha), estadistico=estadistico, block_len=int(block_len),
        n_boot=int(n_boot), n_obs=n,
        info={"filas_descartadas_nan": n_drop, "metodo": metodo, "seed": seed},
    )


# --------------------------------------------------------------------------- #
# Matriz de pérdidas PRELIMINAR (crisis × detector) desde los CSV versionados
# --------------------------------------------------------------------------- #
def tasa_dias_calma_marcados(metrics: pd.DataFrame) -> pd.Series:
    """Fracción de días OOS FUERA de las ventanas de crisis marcados como crisis.

    Se obtiene exacta desde las columnas ``det_*`` del CSV:
    ``det_marked_rate · (1 − det_precision) / (1 − det_base_rate)``. Es 0 si el
    detector nunca marca (``det_precision`` NaN). OJO: no es ``false_alarm_rate``
    de ``metricas.py``, que es la fracción de días MARCADOS que caen fuera
    (1 − precisión); esa no se anula al marcar menos y premia ``SIEMPRE_CRISIS``.
    """
    marked = pd.to_numeric(metrics["det_marked_rate"], errors="coerce")
    far = 1.0 - pd.to_numeric(metrics["det_precision"], errors="coerce")
    base = pd.to_numeric(metrics["det_base_rate"], errors="coerce")
    out = marked * far / (1.0 - base)
    return out.where(marked != 0, 0.0)  # nunca marca → 0; un NaN (dato ausente) sigue NaN


def perdidas_por_crisis(
    pista: str,
    *,
    lam: float = MCS_DEFAULTS["lam"],
    termino_fa: str = "calma",
    fallo: str = "cobertura",
    metrics: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    incluir_baselines: bool = False,
    oos_index: pd.DatetimeIndex | None = None,
    vol: pd.Series | None = None,
    benchmark: dict | None = None,
    seed: int = 20260929,
) -> pd.DataFrame:
    """Matriz de pérdidas crisis × detector (PRELIMINAR, sin paneles diarios).

    L_{c,d} = fallo_{c,d} + λ · FA_d, con

    - ``fallo="cobertura"``: 1 − ``cov_<c>`` (fracción de días de la ventana
      [pico, suelo] NO marcados); ``fallo="evento"``: 1 − ``det_ev_<c>``.
    - ``termino_fa="calma"`` (por defecto): fracción de días de calma marcados
      (:func:`tasa_dias_calma_marcados`). Con λ = 1 y ``fallo="cobertura"``,
      L = 1 − (TPR − FPR) = 1 − J de Youden por crisis: cualquier detector sin
      información (siempre calma, siempre crisis, azar) tiene pérdida esperada 1.
      ``termino_fa="far"``: ``false_alarm_rate`` literal de ``metricas.py``
      (NaN → 0).

    El término de falsa alarma es CONSTANTE por detector (no varía entre crisis
    ni entre réplicas bootstrap): solo desplaza la pérdida media y la incertidumbre
    viene solo de la cobertura. La versión definitiva, con pérdidas diarias y
    bootstrap por bloques de días, necesita ``results/benchmark/panels``.

    Solo entran las crisis evaluadas OOS en la pista (``cov`` NaN → fuera; se
    listan en ``attrs["crisis_excluidas"]``). ``incluir_baselines`` añade
    ``SIEMPRE_CALMA``, ``SIEMPRE_CRISIS``, ``AZAR_IID`` y ``AZAR_PERSISTENTE``
    (fila media de las simulaciones) de :func:`ranking.trivial_baselines`, con la
    tasa y persistencia medianas de la pista (como el notebook de comparativa) y
    ``VOL_P90`` si se pasa ``vol``. Sin paneles su índice OOS es el aproximado
    lunes–viernes de ``track_oos_index`` (``attrs["indice_oos_exacto"]``).
    """
    # Import diferido: ranking → benchmark → evaluacion (ver __init__).
    from regimenes import rutas
    from regimenes.benchmark.cache import load_metrics
    from regimenes.evaluacion import ranking as rk

    if termino_fa not in {"calma", "far"}:
        raise ValueError(f"termino_fa desconocido: {termino_fa!r} (calma | far)")
    if fallo not in {"cobertura", "evento"}:
        raise ValueError(f"fallo desconocido: {fallo!r} (cobertura | evento)")
    track = pista.upper()
    output_dir = Path(output_dir) if output_dir is not None else rutas.RESULTS_BENCHMARK
    if metrics is None:
        metrics = load_metrics(output_dir, require_current=False)
    part = metrics[metrics["pista"].astype(str).str.upper() == track].copy()
    if part.empty:
        raise KeyError(f"No hay métricas de la pista {track}.")
    exacto = None
    if incluir_baselines:
        if oos_index is None:
            oos_index, exacto = rk.track_oos_index(track, metrics, output_dir)
        else:
            exacto = True
        base = rk.trivial_baselines(
            track, pd.DatetimeIndex(oos_index),
            rate=float(part["det_marked_rate"].median()),
            mean_run=float(part["det_mean_crisis_run"].median()),
            benchmark=benchmark, seed=seed, vol=vol,
        )
        part = pd.concat([part, base], ignore_index=True)

    windows = rk.track_crisis_windows(track, benchmark)
    cov = part.set_index("id")[[f"cov_{c}" for c in windows]].astype(float)
    cov.columns = list(windows)
    if fallo == "cobertura":
        miss = 1.0 - cov
    else:
        ev_cols = [f"det_ev_{c}" for c in windows]
        miss = 1.0 - part.set_index("id")[ev_cols].astype(float).set_axis(list(windows), axis=1)
        miss = miss.where(cov.notna())
    if termino_fa == "calma":
        fa = tasa_dias_calma_marcados(part).set_axis(part["id"])
    else:
        fa = pd.to_numeric(part["false_alarm_rate"], errors="coerce").fillna(0.0).set_axis(part["id"])
    evaluadas = miss.notna().all(axis=0)
    out = (miss.loc[:, evaluadas] + float(lam) * fa.to_numpy()[:, None]).T
    out.index.name = "crisis"
    out.columns.name = "id"
    out.attrs.update({
        "pista": track, "lam": float(lam), "termino_fa": termino_fa, "fallo": fallo,
        "fa_por_detector": fa.to_dict(),
        "crisis_excluidas": list(evaluadas.index[~evaluadas]),
        "indice_oos_exacto": exacto,
        "preliminar": True,
    })
    return out


# --------------------------------------------------------------------------- #
# Versión DEFINITIVA (ADR-008 §4): pérdidas diarias desde los paneles walk-forward
# --------------------------------------------------------------------------- #
BASELINES_TRIVIALES: tuple[str, ...] = (
    "SIEMPRE_CRISIS", "SIEMPRE_CALMA", "AZAR_IID", "AZAR_PERSISTENTE", "VOL_P90",
)


def cargar_paneles(
    pista: str,
    panels_dir: Path | None = None,
    ids: tuple[str, ...] | list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Paneles OOS ``<pista>_<id>.parquet`` (``state``, ``p_crisis``) de una pista.

    ``panels_dir`` por defecto es ``results/benchmark/panels`` (v2); se puede
    apuntar a otra carpeta (p. ej. una reproducción) para repetir la vara de
    medir. Solo lee. Exige el mismo índice OOS en todos (el gate de 05).
    """
    from regimenes import rutas  # diferido, como en perdidas_por_crisis

    track = pista.upper()
    panels_dir = Path(panels_dir) if panels_dir is not None else rutas.RESULTS_BENCHMARK / "panels"
    if ids is None:
        paths = sorted(panels_dir.glob(f"{track}_*.parquet"))
    else:
        paths = [panels_dir / f"{track}_{i}.parquet" for i in ids]
    if not paths:
        raise FileNotFoundError(f"No hay paneles de la pista {track} en {panels_dir}.")
    out = {p.stem.split("_", 1)[1]: pd.read_parquet(p, columns=["state", "p_crisis"]) for p in paths}
    ref = next(iter(out.values())).index
    distintos = [k for k, v in out.items() if not v.index.equals(ref)]
    if distintos:
        raise ValueError(f"Pista {track}: índice OOS distinto en {distintos}.")
    return out


def senales_crisis(
    paneles: dict[str, pd.DataFrame],
    estados_crisis: dict[str, int] | None = None,
    *,
    n_states: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Señal binaria diaria ``state == estado de crisis`` (columnas = detectores).

    El estado de crisis es ``n_states − 1`` (``RegimeDetector.crisis_state``, el que
    usa ``run_one`` para las columnas ``det_*``). El panel no guarda ``n_states``:

    - ``n_states`` (recomendado; columna ``n_states`` del CSV de métricas, ver
      :func:`n_states_metricas`): estado de crisis = ``n_states − 1``. Es exacto
      aunque el detector NUNCA visite su estado de crisis en el OOS.
    - sin ``n_states``: se toma el máximo estado observado y se comprueba que sea el
      de mayor ``p_crisis`` media (el panel guarda ``p_crisis = proba[:, crisis_state]``).
      Si el panel solo tiene UN estado, falla: «nunca crisis» y «siempre crisis» no se
      distinguen (antes un detector que nunca marcaba se leía como ``SIEMPRE_CRISIS``).
      Tampoco detecta un estado de crisis no visitado con varios estados: por eso
      conviene pasar ``n_states``.
    - ``estados_crisis`` fija el estado a mano (prioridad máxima).
    """
    estados_crisis = estados_crisis or {}
    n_states = n_states or {}
    cols = {}
    for k, pan in paneles.items():
        if k in estados_crisis:
            cs = int(estados_crisis[k])
        elif k in n_states:
            cs = int(n_states[k]) - 1
            if int(pan["state"].max()) > cs:
                raise ValueError(f"{k}: estado {int(pan['state'].max())} > n_states − 1 = {cs}.")
        else:
            if pan["state"].nunique() < 2:
                raise ValueError(f"{k}: el panel solo tiene el estado {int(pan['state'].iloc[0])}; "
                                 "sin n_states no se sabe si es la crisis. Pasa n_states.")
            cs = int(pan["state"].max())
            medias = pan.groupby("state")["p_crisis"].mean()
            if int(medias.idxmax()) != cs:
                raise ValueError(f"{k}: el estado máximo ({cs}) no es el de mayor p_crisis "
                                 f"({int(medias.idxmax())}); pasa estados_crisis o n_states.")
        cols[k] = pan["state"].eq(cs)
    return pd.DataFrame(cols)


def n_states_metricas(
    pista: str, panels_dir: Path | None, ids: list[str] | tuple[str, ...],
) -> dict[str, int]:
    """``n_states`` de cada panel desde su CSV de métricas, si existe.

    Busca ``<panels_dir>/<P>_<id>.csv`` (formato de ``results/detectores/menus``) y
    ``<panels_dir>/../metrics/<P>_<id>.csv`` (formato del benchmark). Los ids sin CSV
    o sin columna ``n_states`` no aparecen en el resultado.
    """
    from regimenes import rutas  # diferido, como en perdidas_por_crisis

    track = pista.upper()
    panels_dir = Path(panels_dir) if panels_dir is not None else rutas.RESULTS_BENCHMARK / "panels"
    out = {}
    for k in ids:
        for ruta in (panels_dir / f"{track}_{k}.csv", panels_dir.parent / "metrics" / f"{track}_{k}.csv"):
            if ruta.exists():
                fila = pd.read_csv(ruta, nrows=1)
                if "n_states" in fila and pd.notna(fila.loc[0, "n_states"]):
                    out[k] = int(fila.loc[0, "n_states"])
                    break
    return out


def _racha_media(flags: np.ndarray) -> float:
    """Duración media de las rachas de True (0 si no hay)."""
    pasos = np.diff(np.concatenate(([0], np.asarray(flags, dtype=np.int8), [0])))
    largos = np.flatnonzero(pasos == -1) - np.flatnonzero(pasos == 1)
    return float(largos.mean()) if largos.size else 0.0


def senales_baselines(
    pista: str,
    oos_index: pd.DatetimeIndex,
    *,
    rate: float,
    mean_run: float,
    vol: pd.Series | None = None,
    vol_q: float = 0.9,
    vol_min_obs: int = 252,
    seed: int = 20260929,
) -> pd.DataFrame:
    """Señales diarias de los baselines triviales para el MCS con paneles.

    Mismas reglas que :func:`ranking.trivial_baselines` y la misma secuencia
    aleatoria (con la misma ``seed``, ``AZAR_IID`` es la misma realización y
    ``AZAR_PERSISTENTE`` es su primera simulación). Aquí hace falta UNA
    realización y no la media de 200: el MCS compara series de pérdidas con su
    propio ruido, y la media de simulaciones sería un competidor artificialmente
    suave. ``VOL_P90`` (si se pasa ``vol``): percentil 90 expansivo y causal
    (:func:`ranking.vol_percentile_flags`), con la historia previa al OOS.
    """
    from regimenes.evaluacion import ranking as rk

    n = len(oos_index)
    rng = np.random.default_rng(seed)
    out = {
        "SIEMPRE_CRISIS": np.ones(n, bool),
        "SIEMPRE_CALMA": np.zeros(n, bool),
        "AZAR_IID": rng.random(n) < rate,
        "AZAR_PERSISTENTE": rk._markov_flags(n, rate, mean_run, rng),
    }
    if vol is not None:
        out["VOL_P90"] = rk.vol_percentile_flags(
            vol, oos_index, q=vol_q, min_obs=vol_min_obs).to_numpy()
    return pd.DataFrame(out, index=pd.DatetimeIndex(oos_index))


def perdidas_diarias(
    senales: pd.DataFrame | dict[str, pd.DataFrame],
    windows: dict[str, tuple[str, str]],
    *,
    lam: float = MCS_DEFAULTS["lam"],
) -> pd.DataFrame:
    """Matriz de pérdidas DIARIAS día × detector (versión definitiva, ADR-008 §4).

    ``senales``: DataFrame booleano (True = el detector marca crisis ese día) o
    dict de paneles (se convierte con :func:`senales_crisis`). ``windows``: las
    ventanas [pico, suelo] de la pista; los demás días OOS son calma (también las
    trampas: marcarlas es falsa alarma).

    Con N días OOS, N_c días en la ventana c, K crisis evaluadas (N_c > 0) y
    N_calma días de calma, la pérdida del día t es

    - dentro de la ventana c: ``ℓ_t = N / (K·N_c) · (1 − señal_t)``   (fallo);
    - en calma:               ``ℓ_t = λ · N / N_calma · señal_t``      (falsa alarma).

    Los pesos dependen solo de las ventanas, no del detector, así que la pérdida
    es comparable entre detectores y su media vale exactamente

        L̄ = (1/K)·Σ_c (1 − cobertura_c) + λ · FPR,

    con FPR = fracción de días de calma marcados: **fallo por crisis** (cada crisis
    pesa 1/K, sea larga o corta, como en el remuestreo por crisis) **+ días de falsa
    alarma** (cada día de calma marcado pesa lo mismo, en tasa). Las dos partes
    están en la misma escala [0, 1] y con λ = 1 la suma es 1 − J de Youden con las
    crisis equiponderadas: siempre calma y siempre crisis dan 1 exacto y el azar 1
    en esperanza; un detector perfecto, 0. Es la misma pérdida media que
    :func:`perdidas_por_crisis` (``fallo="cobertura"``, ``termino_fa="calma"``);
    cambia la inferencia: aquí la incertidumbre viene de remuestrear DÍAS por
    bloques (las falsas alarmas también varían entre réplicas), no crisis; las
    réplicas de la versión definitiva las da :func:`replicas_perdida`.

    ``attrs``: λ, N, N_calma, K, días por crisis y crisis sin días OOS.
    """
    if isinstance(senales, dict):
        senales = senales_crisis(senales)
    flags = senales.fillna(False).astype(bool)
    index = pd.DatetimeIndex(flags.index)
    n = len(index)
    tramos = {}
    for c, (a, b) in windows.items():
        mask = (index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b))
        if mask.any():
            tramos[c] = mask
    if not tramos:
        raise ValueError("Ninguna ventana de crisis cae en el índice OOS.")
    k = len(tramos)
    peso = np.zeros(n)
    dentro = np.zeros(n, dtype=bool)
    for c, mask in tramos.items():
        if (dentro & mask).any():
            raise ValueError(f"Ventanas solapadas en {c}: la pérdida asume ventanas disjuntas.")
        dentro |= mask
        peso[mask] = n / (k * mask.sum())
    n_calma = int((~dentro).sum())
    if n_calma == 0:
        raise ValueError("No hay días de calma: el término de falsa alarma no está definido.")
    peso[~dentro] = float(lam) * n / n_calma
    F = flags.to_numpy(dtype=float)
    L = np.where(dentro[:, None], 1.0 - F, F) * peso[:, None]
    out = pd.DataFrame(L, index=index, columns=[str(c) for c in flags.columns])
    out.index.name = "fecha"
    out.columns.name = "id"
    out.attrs.update({
        "lam": float(lam), "n_dias": n, "n_calma": n_calma, "n_crisis": k,
        "dias_por_crisis": {c: int(m.sum()) for c, m in tramos.items()},
        "crisis_sin_dias_oos": [c for c in windows if c not in tramos],
        "preliminar": False,
    })
    return out


def descomponer_perdida(L: pd.DataFrame, windows: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Pérdida media de :func:`perdidas_diarias` partida en sus dos términos.

    ``fallo`` = (1/K)·Σ_c (1 − cobertura_c); ``falsa_alarma`` = λ·FPR;
    ``fpr`` = fracción de días de calma marcados; ``perdida_media`` = la suma
    (= media de cada columna de ``L``).
    """
    index = pd.DatetimeIndex(L.index)
    dentro = np.zeros(len(index), dtype=bool)
    for a, b in windows.values():
        dentro |= (index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b))
    n, lam = len(index), float(L.attrs.get("lam", 1.0))
    fallo = L[dentro].sum() / n
    fa = L[~dentro].sum() / n
    fpr = (L[~dentro] > 0).mean() if lam > 0 else pd.Series(np.nan, index=L.columns)
    return pd.DataFrame({"fallo": fallo, "falsa_alarma": fa, "fpr": fpr,
                         "perdida_media": fallo + fa})


def replicas_perdida(
    senales: pd.DataFrame,
    windows: dict[str, tuple[str, str]],
    *,
    lam: float = MCS_DEFAULTS["lam"],
    n_boot: int = int(MCS_DEFAULTS["n_boot"]),
    block_len: int = 1,
    metodo: str = "estacionario",
    seed: int | None = 0,
    memoria: int = _MEMORIA_BOOT,
) -> np.ndarray:
    """Réplicas ``(n_boot, m)`` de la pérdida de :func:`perdidas_diarias` con
    bootstrap por bloques de DÍAS, recalculando la pérdida en cada réplica.

    En cada réplica (días remuestreados por bloques, con su multiplicidad) se
    recalcula cada término sobre los días que han salido:
    ``cobertura*_c`` = fracción de días marcados entre los días sacados de la
    ventana c, ``FPR*`` = fracción de días marcados entre los días de calma sacados,
    y ``L* = media_c (1 − cobertura*_c) + λ · FPR*`` sobre las crisis presentes en
    la réplica. Es el bootstrap del estimador «de razón» de la pérdida.

    Por qué no basta con remuestrear la media de las pérdidas diarias con pesos
    fijos (``model_confidence_set(perdidas_diarias(...))``, que también es válido y
    es lo que hace ``arch.bootstrap.MCS``): con pesos fijos la media de una réplica
    depende de CUÁNTOS días de crisis y de calma le han tocado, no solo de lo bien
    que marca el detector, y eso infla la varianza de todas las diferencias (con
    pesos fijos, SIEMPRE_CALMA, de pérdida 1 en cualquier muestra, tendría
    desviación bootstrap ≈ 0,17 en A y 0,23 en B). Aquí un detector constante tiene
    pérdida exactamente 1 en todas las réplicas. Punto: la pérdida de toda la
    muestra (= media de :func:`perdidas_diarias`).

    Propiedad a declarar: una crisis sin ningún día en la réplica sale del promedio
    de esa réplica (K* < K), y una crisis sacada dos veces pesa 1/K*, no el doble.
    Con el bloque de la mediana de las ventanas, las crisis de 10–30 sesiones faltan
    en un 20–33 % de las réplicas (2026-11-02: en A solo el 7 % de las réplicas trae
    las 17 crisis, K* medio 14,7; en B el 25 %, K* medio 7,8). La incertidumbre
    entre crisis entra por las ausencias, no por las repeticiones.
    """
    flags = senales.fillna(False).astype(bool)
    index = pd.DatetimeIndex(flags.index)
    S = flags.to_numpy(dtype=float)
    n, m = S.shape
    grupos = []
    for a, b in windows.values():
        mask = (index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b))
        if mask.any():
            grupos.append(mask)
    if not grupos:
        raise ValueError("Ninguna ventana de crisis cae en el índice OOS.")
    calma = ~np.logical_or.reduce(grupos)
    out = np.empty((int(n_boot), m))
    for lo, hi, cuentas in _tandas_recuentos(n, int(n_boot), block_len=block_len, metodo=metodo,
                                             seed=seed, memoria=memoria):
        fallo = np.zeros((hi - lo, m))
        presentes = np.zeros((hi - lo, 1))
        for mask in grupos:
            c = cuentas[:, mask]
            dias = c.sum(axis=1, keepdims=True)
            hay = dias > 0
            cov = np.divide(c @ S[mask], dias, out=np.zeros((hi - lo, m)), where=hay)
            fallo += np.where(hay, 1.0 - cov, 0.0)
            presentes += hay
        c = cuentas[:, calma]
        dias_calma = c.sum(axis=1, keepdims=True)
        if (presentes == 0).any() or (dias_calma == 0).any():
            raise ValueError("Alguna réplica no tiene días de crisis o de calma; sube block_len o n.")
        out[lo:hi] = fallo / presentes + float(lam) * (c @ S[calma]) / dias_calma
    return out


def longitud_bloque_crisis(
    oos_index: pd.DatetimeIndex, windows: dict[str, tuple[str, str]]
) -> int:
    """Longitud media del bloque del bootstrap: la MEDIANA de sesiones OOS de las
    ventanas de crisis de la pista.

    Criterio declarado (ADR-008 §4) que no mira ni los detectores ni el ranking:
    solo las ventanas congeladas del juez. La dependencia de las pérdidas diarias
    viene sobre todo de los episodios (una crisis se cubre o se falla en bloque y
    las falsas alarmas llegan en rachas), así que el bloque debe poder contener
    una crisis típica entera; la mediana no se deja arrastrar por las crisis
    largas (punto com, estanflación).
    """
    index = pd.DatetimeIndex(oos_index)
    largos = [int(((index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b))).sum())
              for a, b in windows.values()]
    largos = [x for x in largos if x > 0]
    if not largos:
        raise ValueError("Ninguna ventana de crisis cae en el índice OOS.")
    return int(np.median(largos))


def mcs_senales(
    senales: pd.DataFrame,
    windows: dict[str, tuple[str, str]],
    *,
    lam: float = MCS_DEFAULTS["lam"],
    alpha: float = MCS_DEFAULTS["alpha"],
    block_len: int | None = None,
    estadistico: str = "T_max",
    metodo: str = "estacionario",
    n_boot: int = int(MCS_DEFAULTS["n_boot"]),
    seed: int | None = 0,
    remuestreo: str = "razon",
) -> tuple[pd.DataFrame, MCSResult]:
    """MCS sobre señales diarias ya construidas (detectores y baselines).

    ``remuestreo="razon"`` (definitivo): réplicas de :func:`replicas_perdida`.
    ``remuestreo="lineal"``: medias remuestreadas de :func:`perdidas_diarias` con
    pesos fijos (el MCS estándar sobre una matriz de pérdidas; comparable con
    ``arch.bootstrap.MCS``). Bloque: :func:`longitud_bloque_crisis` si no se pasa.
    Devuelve (pérdidas diarias, resultado).
    """
    if remuestreo not in {"razon", "lineal"}:
        raise ValueError(f"remuestreo desconocido: {remuestreo!r} (razon | lineal)")
    L = perdidas_diarias(senales, windows, lam=lam)
    bl = int(block_len) if block_len is not None else longitud_bloque_crisis(L.index, windows)
    replicas = None
    if remuestreo == "razon":
        replicas = replicas_perdida(senales[L.columns], windows, lam=lam, n_boot=n_boot,
                                    block_len=bl, metodo=metodo, seed=seed)
    res = model_confidence_set(L, alpha=alpha, estadistico=estadistico, block_len=bl,
                               metodo=metodo, n_boot=n_boot, seed=seed, replicas=replicas)
    res.info.update({"lam": float(lam), "n_crisis": L.attrs["n_crisis"], "remuestreo": remuestreo,
                     "metodo": metodo})
    return L, res


def mcs_diario(
    pista: str,
    *,
    panels_dir: Path | None = None,
    ids: tuple[str, ...] | list[str] | None = None,
    vol: pd.Series | None = None,
    incluir_baselines: bool = True,
    benchmark: dict | None = None,
    baseline_seed: int = 20260929,
    **kwargs,
) -> tuple[pd.DataFrame, MCSResult]:
    """MCS DEFINITIVO de una pista: pérdidas diarias + bootstrap por bloques de días.

    Lee los paneles de ``panels_dir`` (por defecto ``results/benchmark/panels``),
    construye las señales, añade los baselines triviales (``AZAR_*`` con la tasa de
    días marcados y la racha media medianas de los detectores de la pista, como en
    ``perdidas_por_crisis``; ``VOL_P90`` si se pasa ``vol``) y llama a
    :func:`mcs_senales` (``kwargs``: ``lam``, ``alpha``, ``block_len``,
    ``estadistico``, ``n_boot``, ``seed``, ``remuestreo``…). Por defecto, bootstrap
    estacionario de bloque medio :func:`longitud_bloque_crisis` y réplicas de razón.
    Devuelve (pérdidas diarias, resultado).
    """
    from regimenes.evaluacion import ranking as rk

    track = pista.upper()
    windows = rk.track_crisis_windows(track, benchmark)
    paneles = cargar_paneles(track, panels_dir, ids)
    senales = senales_crisis(paneles, n_states=n_states_metricas(track, panels_dir, list(paneles)))
    if incluir_baselines:
        F = senales.to_numpy()
        rate = float(np.median(F.mean(axis=0)))
        run = float(np.median([_racha_media(F[:, j]) for j in range(F.shape[1])]))
        base = senales_baselines(track, pd.DatetimeIndex(senales.index), rate=rate, mean_run=run,
                                 vol=vol, seed=baseline_seed)
        senales = pd.concat([senales, base], axis=1)
    L, res = mcs_senales(senales, windows, **kwargs)
    res.info.update({"pista": track,
                     "panels_dir": str(panels_dir) if panels_dir is not None else "v2 (por defecto)"})
    return L, res
