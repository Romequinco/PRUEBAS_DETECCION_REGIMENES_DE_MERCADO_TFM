"""
mcs.py — Model Confidence Set (Hansen, Lunde y Nason, 2011) para el ranking (A1).

Responde a «¿los primeros del ranking se distinguen de verdad?»: el MCS al nivel
``alpha`` es el conjunto de detectores que no se pueden descartar como peores que
el mejor, dada una matriz de pérdidas (filas = observaciones o bloques de
observaciones, columnas = modelos; menor es mejor).

- :func:`model_confidence_set`: implementación genérica (estadísticos ``T_max`` y
  ``T_R``, bootstrap por bloques circular o estacionario, eliminación secuencial
  y p-valores MCS).
- :func:`perdidas_por_crisis`: helper **PRELIMINAR** que construye la matriz
  crisis × detector desde los CSV versionados ``results/benchmark/metrics``.

Pérdida pre-registrada (6.9): fallo por crisis + días de falsa alarma, remuestreo
por bloques de días, α = 0,10. La versión definitiva necesita los paneles diarios
(``results/benchmark/panels``); sin ellos se remuestrea POR CRISIS (bootstrap iid
sobre crisis = bloques de longitud 1 en la dimensión crisis).
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
    idx = bootstrap_indices(n, int(n_boot), block_len=block_len, metodo=metodo, seed=seed)
    mean = L.mean(axis=0)
    # Medias bootstrap de cada modelo (B, m): todas las d̄* son lineales en ellas.
    boot = L[idx].mean(axis=1)

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
    return out.where(marked > 0, 0.0)


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
