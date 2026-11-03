"""Fusión online C1 (ADR-011 §4): media de probabilidades con pesos que aprenden de crisis terminadas.

Construcción (pre-registrada, ADR-011 §4.1–§4.4)
------------------------------------------------
* ``p̄_t = Σ_i w_i(m) · p_i,t`` sobre **todos** los candidatos de la pista (sin selección); crisis si
  ``p̄_t ≥ 0,50``, sin persistencia. Si un ``p_i,t`` fuera NaN, ese día el detector sale de la suma
  y los pesos se renormalizan entre los demás.
* **Calendario:** los pesos se calculan al cierre de la última sesión OOS de cada mes natural (τ) y
  rigen desde la sesión OOS siguiente (la primera del mes siguiente). En el primer mes, pesos
  iguales. Nunca se usa un dato del mismo día.
* **Etiquetas vistas en τ:** ``detectores.pool.supervisados.etiquetas_causales`` sobre el OOS
  recortado en τ (crisis con suelo ≤ τ; las sin terminar y las trampas, calma). Ventanas del juez
  (``ranking.track_crisis_windows``).
* **Pérdida en τ:** la de la vara (``evaluacion.mcs.perdidas_diarias``, 1 − J de Youden con crisis
  equiponderadas) sobre la **señal binaria** de cada detector en los días OOS ≤ τ, recalculada
  entera en cada τ. Su media es ``L_i(τ) = (1/K_τ)·Σ_c (1 − cobertura_i,c) + FPR_i``.
* **Pesos:** ``w_i(τ) = exp(−η·L_i(τ)) / Σ_j exp(−η·L_j(τ))``; iguales hasta la primera crisis
  terminada con días OOS. η = 0 es la media simple (M) y η = ∞ *follow the leader* (empates a
  partes iguales).

Este módulo tiene dos capas:

1. **Núcleo puro** (``calendario``, ``perdida_en_tau``, ``historia_perdidas``, ``pesos_desde_perdidas``,
   ``pesos_mensuales``, ``pesos_diarios``, ``combinar``, ``fusion_online``): funciones sobre
   DataFrames, sin E/S, que los tests ejercitan con datos sintéticos.
2. **Ejecución** (``ejecutar_pista``, ``ejecutar``, ``leer``, CLI): lee los paneles con
   ``regimenes.fusion.vara``, construye O, la rejilla de η y las sensibilidades del §6.3, los
   comparadores M, V y F* (§4.5), evalúa con la vara en todo el OOS, H1 y H2, hace el MCS-O y
   aplica el criterio del §6.2. Resultados en ``results/fusion/online/``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from regimenes import rutas
from regimenes.detectores.pool.supervisados import etiquetas_causales
from regimenes.evaluacion.mcs import perdidas_diarias


UMBRAL: float = 0.5
ETA_PRINCIPAL: float = 10.0
REJILLA_ETA: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, math.inf)
RETRASO_RECONOCIMIENTO: int = 63          # sesiones (§6.3)
MIN_DIAS_SENSIBILIDAD: int = 21           # días OOS de crisis terminadas (§4.2)
TOL_EMPATE: float = 1e-12                 # empates de pérdida para η = ∞ (ruido de coma flotante)
PISTAS: tuple[str, ...] = ("A", "B")

# Orden de prioridad para deduplicar señales idénticas en el MCS (§5.1).
ORDEN_MCS_COMPARADORES: tuple[str, ...] = ("F*", "V", "M", "O")


def directorio_resultados() -> Path:
    """``results/fusion/online/`` (``rutas.RESULTS_FUSION_ONLINE`` si ya existe)."""
    return Path(getattr(rutas, "RESULTS_FUSION_ONLINE", rutas.RESULTS_FUSION / "online"))


def directorio_fijas() -> Path:
    return Path(getattr(rutas, "RESULTS_FUSION_FIJAS", rutas.RESULTS_FUSION / "fijas"))


# --------------------------------------------------------------------------- #
# Variantes (principal y sensibilidades del §6.3)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Variante:
    """Cómo se ven las etiquetas y cuándo empiezan a aprender los pesos.

    ``retraso``: sesiones OOS de retraso de reconocimiento (una crisis cuenta en τ si su suelo es
    ≤ la sesión que está ``retraso`` posiciones antes de τ). ``min_dias``: días OOS de crisis
    terminadas necesarios para dejar los pesos iguales (1 = «primera crisis terminada con días
    OOS»). ``excluir_sin_terminar``: saca de la pérdida los días de crisis empezadas y sin suelo.
    ``binaria``: combina las señales binarias (0/1) en vez de ``p_crisis``.
    """

    nombre: str
    retraso: int = 0
    min_dias: int = 1
    excluir_sin_terminar: bool = False
    binaria: bool = False

    @property
    def clave_perdidas(self) -> tuple[int, bool]:
        # La pérdida solo depende de cómo se ven las etiquetas (no de η, de la entrada ni del
        # umbral de días), así que las variantes que comparten clave reutilizan la historia.
        return (self.retraso, self.excluir_sin_terminar)


PRINCIPAL = Variante("principal")
SENSIBILIDADES: tuple[Variante, ...] = (
    Variante("binaria", binaria=True),
    Variante("retraso63", retraso=RETRASO_RECONOCIMIENTO),
    Variante("min21", min_dias=MIN_DIAS_SENSIBILIDAD),
    Variante("excluir_sin_terminar", excluir_sin_terminar=True),
)


def id_senal(variante: Variante, eta: float) -> str:
    """Identificador de una señal online: ``ONLINE_<variante>_eta<η>`` (``etainf`` para ∞)."""
    e = "inf" if math.isinf(eta) else f"{eta:g}"
    return f"ONLINE_{variante.nombre}_eta{e}"


# --------------------------------------------------------------------------- #
# Núcleo puro · calendario
# --------------------------------------------------------------------------- #
def calendario(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Un renglón por mes natural OOS: τ (última sesión del mes) y ``vigencia`` (sesión siguiente).

    ``vigencia`` es NaT en el último mes (no hay sesión siguiente en el índice). ``pos_tau`` es la
    posición de τ en el índice (para el retraso de reconocimiento).
    """
    index = pd.DatetimeIndex(index)
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("El índice OOS debe ser temporal, único y ordenado.")
    periodo = index.to_period("M")
    ultima = ~pd.Series(periodo).duplicated(keep="last").to_numpy()
    pos = np.flatnonzero(ultima)
    vigencia = [index[p + 1] if p + 1 < len(index) else pd.NaT for p in pos]
    return pd.DataFrame(
        {"tau": index[pos], "vigencia": pd.DatetimeIndex(vigencia), "pos_tau": pos},
        index=pd.PeriodIndex(periodo[pos], name="mes"),
    )


# --------------------------------------------------------------------------- #
# Núcleo puro · pérdida en τ
# --------------------------------------------------------------------------- #
def _como_ventanas(crisis: Mapping[str, tuple] | Sequence[tuple]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    if isinstance(crisis, Mapping):
        items = crisis.items()
    else:
        items = ((f"c{i}", c) for i, c in enumerate(crisis))
    return {str(k): (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in items}


def perdida_en_tau(
    senales: pd.DataFrame,
    pos_tau: int,
    crisis: Mapping[str, tuple] | Sequence[tuple],
    trampas: Sequence[tuple] = (),
    *,
    retraso: int = 0,
    excluir_sin_terminar: bool = False,
) -> tuple[pd.Series | None, dict]:
    """``L_i(τ)`` de cada columna de ``senales`` (bool) con las etiquetas vistas en τ = ``index[pos_tau]``.

    Usa solo las filas ``≤ τ``. Devuelve ``(None, info)`` si no hay ninguna crisis terminada (y
    reconocida) con días OOS ≤ τ: entonces los pesos son iguales. ``info`` lleva K_τ, días de
    crisis y de calma y las crisis que cuentan.
    """
    idx = pd.DatetimeIndex(senales.index[: pos_tau + 1])
    tau = idx[-1]
    ventanas = _como_ventanas(crisis)
    # Retraso de reconocimiento (§6.3): solo cuentan las crisis cuyo suelo es ≤ la sesión que
    # está `retraso` posiciones antes de τ. Las demás no se pasan a etiquetas_causales y sus días
    # cuentan como calma (o se excluyen si son de crisis sin terminar y excluir_sin_terminar).
    if retraso:
        p = pos_tau - int(retraso)
        corte = idx[p] if p >= 0 else None
    else:
        corte = tau
    if retraso and excluir_sin_terminar:
        raise ValueError("El ADR-011 no combina el retraso de reconocimiento con excluir_sin_terminar.")
    reconocidas = {c: v for c, v in ventanas.items() if corte is not None and v[1] <= corte}
    # Sin retraso, etiquetas_causales decide sola con el corte τ (último día del índice); a la
    # variante excluir_sin_terminar hay que pasarle también las crisis sin suelo para que las
    # marque NaN. Con retraso solo se pasan las reconocidas: las demás cuentan como calma (§6.3).
    pasadas = list(ventanas.values()) if excluir_sin_terminar else list(reconocidas.values())
    y = etiquetas_causales(idx, pasadas, list(trampas), excluir_sin_terminar=excluir_sin_terminar)
    filas = y.notna().to_numpy()
    # Ventanas de la vara: las crisis reconocidas con días en las filas que entran en la pérdida.
    idx_f = idx[filas]
    win = {}
    for c, (a, b) in reconocidas.items():
        m = (idx_f >= a) & (idx_f <= b)
        if m.any():
            win[c] = (a, b)
    info = {"tau": tau, "K": len(win), "dias_crisis": int((y == 1).sum()),
            "dias_calma": int((y == 0).sum()), "dias_excluidos": int((~filas).sum()),
            "crisis": ",".join(win)}
    if not win:
        return None, info
    # Comprobación de coherencia: la etiqueta 1 de etiquetas_causales es exactamente la unión de
    # las ventanas que ve perdidas_diarias (las trampas no se solapan con ninguna crisis, §4.2).
    dentro = np.zeros(len(idx_f), dtype=bool)
    for a, b in win.values():
        dentro |= (idx_f >= a) & (idx_f <= b)
    if not np.array_equal(dentro, (y[filas] == 1).to_numpy()):
        raise ValueError(f"Etiquetas causales y ventanas de la vara no coinciden en τ={tau.date()}.")
    L = perdidas_diarias(senales.loc[idx_f].astype(bool), {c: tuple(map(str, v)) for c, v in win.items()})
    return L.mean(axis=0), info


def historia_perdidas(
    senales: pd.DataFrame,
    crisis: Mapping[str, tuple] | Sequence[tuple],
    trampas: Sequence[tuple] = (),
    *,
    retraso: int = 0,
    excluir_sin_terminar: bool = False,
    cal: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pérdidas ``L_i(τ)`` en cada τ del calendario (filas NaN si aún no hay crisis terminada).

    Devuelve ``(L, meta)``: ``L`` es τ × detector; ``meta`` lleva K, días de crisis/calma y la
    vigencia. Las variantes que comparten (retraso, excluir) comparten esta historia.
    """
    senales = senales.astype(bool)
    cal = calendario(senales.index) if cal is None else cal
    filas, metas = [], []
    for pos in cal["pos_tau"].to_numpy():
        L, info = perdida_en_tau(senales, int(pos), crisis, trampas, retraso=retraso,
                                 excluir_sin_terminar=excluir_sin_terminar)
        filas.append(L if L is not None else pd.Series(np.nan, index=senales.columns))
        metas.append(info)
    L = pd.DataFrame(filas, index=cal["tau"].to_numpy())
    L.index.name = "tau"
    meta = pd.DataFrame(metas).set_index("tau")
    meta["vigencia"] = cal["vigencia"].to_numpy()
    return L, meta


# --------------------------------------------------------------------------- #
# Núcleo puro · pesos
# --------------------------------------------------------------------------- #
def pesos_desde_perdidas(L: pd.Series, eta: float) -> pd.Series:
    """``exp(−η·L) / Σ exp(−η·L)``; η = ∞ reparte a partes iguales entre los de pérdida mínima."""
    L = pd.Series(L, dtype=float)
    if L.isna().any():
        raise ValueError("Pérdidas con NaN: los pesos no están definidos.")
    if math.isinf(eta):
        lideres = (L - L.min()).abs() <= TOL_EMPATE
        w = lideres.astype(float)
    else:
        # Restar el mínimo no cambia los pesos normalizados y evita desbordes con η grande.
        w = np.exp(-float(eta) * (L - L.min()))
    return w / w.sum()


def pesos_mensuales(
    L: pd.DataFrame, eta: float, aprendido: pd.Series | np.ndarray | None = None
) -> pd.DataFrame:
    """Pesos en cada τ: iguales donde no hay pérdida (o ``aprendido`` es False), Hedge si no."""
    n = L.shape[1]
    aprendido = L.notna().all(axis=1) if aprendido is None else pd.Series(np.asarray(aprendido, bool),
                                                                         index=L.index)
    W = pd.DataFrame(1.0 / n, index=L.index, columns=L.columns)
    for tau in L.index[aprendido.to_numpy()]:
        W.loc[tau] = pesos_desde_perdidas(L.loc[tau], eta).to_numpy()
    return W


def pesos_diarios(W: pd.DataFrame, vigencia: pd.Series | pd.DatetimeIndex, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Pesos vigentes cada día de ``index``: los de τ rigen desde ``vigencia(τ)``; antes, iguales.

    ``vigencia`` va alineada con las filas de ``W`` (NaT en el último τ, que no rige nunca).
    """
    index = pd.DatetimeIndex(index)
    vig = pd.DatetimeIndex(np.asarray(vigencia))
    ok = ~vig.isna()
    Wv = W.loc[ok].copy()
    Wv.index = vig[ok]
    D = Wv.reindex(index).ffill()
    return D.fillna(1.0 / W.shape[1])


def combinar(probs: pd.DataFrame, pesos: pd.DataFrame) -> pd.Series:
    """``p̄_t = Σ_i w_i·p_i,t``; con NaN en ``p_i,t`` el detector sale y se renormaliza (§4.1)."""
    P = probs.astype(float)
    W = pesos.reindex(index=P.index, columns=P.columns)
    valido = P.notna()
    num = (P.fillna(0.0) * W).sum(axis=1)
    den = W.where(valido, 0.0).sum(axis=1)
    return (num / den).where(den > 0)


def fusion_online(
    probs: pd.DataFrame,
    senales: pd.DataFrame,
    crisis: Mapping[str, tuple] | Sequence[tuple],
    trampas: Sequence[tuple] = (),
    *,
    eta: float = ETA_PRINCIPAL,
    variante: Variante = PRINCIPAL,
    historia: tuple[pd.DataFrame, pd.DataFrame] | None = None,
) -> dict[str, object]:
    """Fusión online completa de una variante y un η.

    ``probs``: ``p_crisis`` (día × detector); ``senales``: señal binaria (la que mide la pérdida).
    Con ``variante.binaria`` se combinan las señales binarias en vez de ``probs``. ``historia``
    reutiliza un ``historia_perdidas`` ya calculado con la misma ``clave_perdidas``.

    Devuelve ``{"pbar", "senal", "pesos" (por τ), "meta"}``.
    """
    if not probs.index.equals(senales.index) or list(probs.columns) != list(senales.columns):
        raise ValueError("probs y senales deben tener el mismo índice y las mismas columnas.")
    if historia is None:
        historia = historia_perdidas(senales, crisis, trampas, retraso=variante.retraso,
                                     excluir_sin_terminar=variante.excluir_sin_terminar)
    L, meta = historia
    aprendido = L.notna().all(axis=1) & (meta["dias_crisis"] >= variante.min_dias)
    W = pesos_mensuales(L, eta, aprendido)
    D = pesos_diarios(W, meta["vigencia"], probs.index)
    entrada = senales.astype(float) if variante.binaria else probs
    pbar = combinar(entrada, D)
    meta = meta.assign(aprendido=aprendido.to_numpy(), entropia_efectiva=entropia_efectiva(W))
    return {"pbar": pbar, "senal": pbar >= UMBRAL, "pesos": W, "meta": meta}


def entropia_efectiva(W: pd.DataFrame) -> pd.Series:
    """Número efectivo de detectores ``exp(H(w))`` en cada τ (n con pesos iguales, 1 con un líder)."""
    P = W.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
    return pd.Series(np.exp(H), index=W.index)


def voto_mayoria(senales: pd.DataFrame) -> pd.Series:
    """Votación por mayoría de todos los candidatos: al menos ⌈n/2⌉ de n (§3.2, = ``VOTO_mayoria``)."""
    n = senales.shape[1]
    return senales.astype(bool).sum(axis=1) >= math.ceil(n / 2)


# --------------------------------------------------------------------------- #
# Ejecución sobre los paneles reales (ADR-011 §4.5, §5, §6.2 y §6.3)
# --------------------------------------------------------------------------- #
def _trampas() -> list[tuple[str, str]]:
    from regimenes.evaluacion.ranking import track_false_positive_windows

    return list(track_false_positive_windows().values())


def senal_fstar(pista: str) -> tuple[str | None, pd.Series | None]:
    """``(id, señal)`` de F* (la mejor fija elegida en H1, §5.2), o ``(None, None)`` si no hay.

    La lee de lo que deja :mod:`regimenes.fusion.fijas` en ``results/fusion/fijas/``. Si esos
    resultados aún no existen, o si no hubo F* (ninguna fija de nivel 0 en H1), el MCS-O y los
    comparadores se quedan sin ella (§4.5)."""
    from regimenes.fusion import fijas

    p = pista.upper()
    elec = fijas.leer_tabla(f"eleccion_{p}", directorio_fijas())
    if elec is None or elec.empty or pd.isna(elec["f_star"].iloc[0]):
        return None, None
    fid = str(elec["f_star"].iloc[0])
    sen = fijas.leer_senales(p, None, directorio_fijas())
    if sen is None or fid not in sen:
        raise FileNotFoundError(f"F* = {fid} elegida, pero falta su señal en senales_{p}.parquet "
                                f"de {directorio_fijas()}.")
    # Coherencia con los paneles actuales: mismo índice OOS (un reindex convertiría los huecos en
    # True) y mismas señales de los individuales; si no, los resultados de las fijas son de otros
    # paneles y hay que volver a ejecutarlas antes que la online.
    from regimenes.fusion import vara

    actuales = vara.senales(p)
    if not pd.DatetimeIndex(sen.index).equals(actuales.index):
        raise ValueError(f"[{p}] senales_{p}.parquet de las fijas no tiene el índice OOS actual.")
    if not sen[list(actuales.columns)].astype(bool).equals(actuales):
        raise ValueError(f"[{p}] las señales de los individuales guardadas por las fijas no coinciden "
                         "con los paneles actuales: vuelve a ejecutar regimenes.fusion.fijas.")
    return fid, sen[fid].astype(bool)


def _orden_eta(tabla: pd.DataFrame) -> pd.DataFrame:
    """Orden del §5.2 sobre la rejilla de η; los empates se resuelven por el η menor (§4.4)."""
    from regimenes.evaluacion.ranking import DETECTION_DEFAULTS

    dec = int(DETECTION_DEFAULTS["score_decimals"])
    k = tabla.assign(_s=-tabla["score_deteccion"].astype(float).round(dec),
                     _w=tabla["switching_rate"].astype(float).fillna(np.inf))
    return tabla.loc[k.sort_values(["nivel", "_s", "_w", "eta"], kind="mergesort").index]


def ejecutar_pista(pista: str, *, n_boot: int | None = None, verbose: bool = True) -> dict[str, object]:
    """Calcula y guarda todo lo de la online en una pista. ``n_boot`` solo para pruebas rápidas."""
    from regimenes.fusion import vara

    p = vara._pista(pista)
    out_dir = directorio_resultados()
    out_dir.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    cand = vara.candidatos(p)
    sen = vara.senales(p)[cand]
    prob = vara.probabilidades(p)[cand]
    win = vara.ventanas(p)
    tr = _trampas()
    idx = pd.DatetimeIndex(sen.index)
    h1, h2 = vara.mitades(idx)
    cal = calendario(idx)
    log(f"[{p}] {len(cand)} candidatos, {len(idx)} sesiones, {len(cal)} meses OOS")

    # Historias de pérdida: una por (retraso, excluir); las variantes las comparten.
    historias: dict[tuple[int, bool], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for var in (PRINCIPAL, *SENSIBILIDADES):
        if var.clave_perdidas not in historias:
            historias[var.clave_perdidas] = historia_perdidas(
                sen, win, tr, retraso=var.retraso, excluir_sin_terminar=var.excluir_sin_terminar, cal=cal)
            log(f"[{p}] pérdidas {var.nombre}: listas")

    # Señales: la rejilla de η (principal) y las sensibilidades con η = 10 (la binaria también con
    # η = 0, que es VOTO_mayoria, §6.3).
    corridas: list[tuple[Variante, float]] = [(PRINCIPAL, e) for e in REJILLA_ETA]
    corridas += [(v, ETA_PRINCIPAL) for v in SENSIBILIDADES]
    corridas += [(SENSIBILIDADES[0], 0.0)]
    senales_on, pbar_on, pesos_largos, primeros = {}, {}, [], {}
    for var, eta in corridas:
        r = fusion_online(prob, sen, win, tr, eta=eta, variante=var, historia=historias[var.clave_perdidas])
        sid = id_senal(var, eta)
        senales_on[sid], pbar_on[sid] = r["senal"], r["pbar"]
        meta = r["meta"]
        primeros[sid] = meta.index[meta["aprendido"].to_numpy()].min()
        W = r["pesos"].copy()
        W.insert(0, "variante", var.nombre)
        W.insert(1, "eta", eta)
        W.insert(2, "vigencia", meta["vigencia"].to_numpy())
        W.insert(3, "aprendido", meta["aprendido"].to_numpy())
        W.insert(4, "K", meta["K"].to_numpy())
        W.insert(5, "entropia_efectiva", meta["entropia_efectiva"].to_numpy())
        pesos_largos.append(W.reset_index())
    pesos = pd.concat(pesos_largos, ignore_index=True)

    # Comparadores (§4.5).
    O = senales_on[id_senal(PRINCIPAL, ETA_PRINCIPAL)].rename("O")
    M = senales_on[id_senal(PRINCIPAL, 0.0)].rename("M")
    # V con la regla de las fijas (§3.2); voto_mayoria es la misma regla y aquí se comprueba.
    from regimenes.fusion import fijas

    V = fijas.votacion(sen, fijas.k_mayoria(len(cand))).rename("V")
    if not V.equals(voto_mayoria(sen).rename("V")):
        raise AssertionError(f"[{p}] voto_mayoria ≠ fijas.votacion con k = ⌈n/2⌉")
    fid, F = senal_fstar(p)
    comp = {"O": O, "M": M, "V": V}
    if F is not None:
        comp["F*"] = F.astype(bool).rename("F*")
    comparadores = pd.DataFrame(comp)
    online = pd.DataFrame(senales_on)

    # Comprobación del §3.2/§6.3: online η = 0 con entrada binaria = VOTO_mayoria.
    bin0 = online[id_senal(SENSIBILIDADES[0], 0.0)]
    if not np.array_equal(bin0.to_numpy(), V.to_numpy()):
        raise AssertionError(f"[{p}] online η=0 binaria ≠ VOTO_mayoria")

    # Evaluación con la vara: individuales, comparadores y todas las señales online.
    todas = pd.concat([sen, comparadores, online], axis=1)
    tramos = {"oos": None, "H1": h1, "H2": h2}
    ev = {t: vara.evaluar(todas, p, tramo) for t, tramo in tramos.items()}
    evaluacion = pd.concat([e.assign(tramo=t) for t, e in ev.items()]).reset_index()

    i_star = vara.mejor_individual(p)
    i_star_h1 = vara.mejor_individual(p, h1)
    log(f"[{p}] I* = {i_star} · I*_H1 = {i_star_h1} · F* = {fid}")

    # MCS-O: individuales, baselines, F*, V, M y O (este orden también es el de deduplicación).
    kw = {} if n_boot is None else {"n_boot": int(n_boot)}
    extra = [c for c in ORDEN_MCS_COMPARADORES if c in comparadores]
    prioridad = [*cand, *vara.BASELINES, *extra]
    mcs_sen = pd.concat([sen, comparadores[extra]], axis=1)
    mcs_t = {}
    for t in ("oos", "H2"):
        mcs_t[t] = vara.mcs(mcs_sen, p, tramo=tramos[t], prioridad=prioridad, **kw)
        mcs_t[t].to_csv(out_dir / f"mcs_O_{p}_{t}.csv")
        log(f"[{p}] MCS-O {t}: {int(mcs_t[t]['incluido'].sum())} dentro")

    # Criterio §6.2 (comparaciones sin redondeo; un empate exacto no cumple).
    def s(t, i):
        return float(ev[t].loc[i, "score_deteccion"])

    def n(t, i):
        return int(ev[t].loc[i, "nivel"])

    def dentro(t, i):
        return bool(mcs_t[t].loc[i, "incluido"])

    hay_f = "F*" in comparadores
    fila = {
        "pista": p, "I*": i_star, "I*_H1": i_star_h1, "F*": fid,
        "score_O": s("oos", "O"), "nivel_O": n("oos", "O"), "score_I*": s("oos", i_star),
        "score_M": s("oos", "M"), "score_V": s("oos", "V"),
        "score_F*": s("oos", "F*") if hay_f else None,
        "score_O_H2": s("H2", "O"), "nivel_O_H2": n("H2", "O"), "score_I*_H1_en_H2": s("H2", i_star_h1),
        "score_M_H2": s("H2", "M"), "score_V_H2": s("H2", "V"),
        "score_F*_H2": s("H2", "F*") if hay_f else None,
        "mcs_O_incluido": dentro("oos", "O"), "mcs_O_H2_incluido": dentro("H2", "O"),
        "mcs_M_incluido": dentro("oos", "M"), "mcs_V_incluido": dentro("oos", "V"),
        "mcs_Istar_incluido": dentro("oos", i_star),
        "mcs_F*_incluido": dentro("oos", "F*") if hay_f else None,
        "mcs_O_duplicado_de": mcs_t["oos"].loc["O", "duplicado_de"],
    }
    fila["i"] = bool(fila["nivel_O"] == 0 and fila["score_O"] > fila["score_I*"])
    fila["ii"] = fila["mcs_O_incluido"]
    fila["iii"] = bool(fila["nivel_O_H2"] == 0 and fila["score_O_H2"] > fila["score_I*_H1_en_H2"])
    fila["cumple"] = bool(fila["i"] and fila["ii"] and fila["iii"])
    fila["O_gt_M_oos"] = bool(fila["score_O"] > fila["score_M"])
    fila["O_gt_M_H2"] = bool(fila["score_O_H2"] > fila["score_M_H2"])
    fila["primer_tau_aprendido"] = str(primeros[id_senal(PRINCIPAL, ETA_PRINCIPAL)].date())

    # Control sin peso del η (§4.4): qué η de la rejilla gana en H1 y cómo rinde en H2.
    rej = []
    for e in REJILLA_ETA:
        sid = id_senal(PRINCIPAL, e)
        rej.append({"pista": p, "eta": e, "id": sid,
                    **{f"{c}_{t}": ev[t].loc[sid, c] for t in tramos
                       for c in ("nivel", "score_deteccion", "switching_rate")}})
    rej = pd.DataFrame(rej)
    orden_h1 = _orden_eta(rej.rename(columns={"nivel_H1": "nivel", "score_deteccion_H1": "score_deteccion",
                                              "switching_rate_H1": "switching_rate"}))
    rej["puesto_H1"] = rej.index.map({k: i + 1 for i, k in enumerate(orden_h1.index)})
    eta_h1 = float(rej.loc[orden_h1.index[0], "eta"])
    fila["eta_H1"] = eta_h1
    fila["score_eta_H1_en_H2"] = float(rej.loc[rej["eta"] == eta_h1, "score_deteccion_H2"].iloc[0])
    fila["eta_H1_gt_eta10_en_H2"] = bool(fila["score_eta_H1_en_H2"] > fila["score_O_H2"])

    # Sensibilidades del §6.3 (η = 10), en una línea por señal.
    fila["sensibilidades"] = {
        sid: {t: round(s(t, sid), 6) for t in tramos} for sid in online.columns
        if not sid.startswith("ONLINE_principal")
    }

    # Utilidad operativa (sin peso) de F*, O, M, V e I*.
    util_sen = pd.concat([comparadores, sen[[i_star]].rename(columns={i_star: f"I*={i_star}"})], axis=1)
    util = pd.concat([vara.utilidad(util_sen, p, tr_).assign(tramo=t) for t, tr_ in tramos.items()])

    # Guardado.
    pd.concat([comparadores, online], axis=1).to_parquet(out_dir / f"senales_{p}.parquet")
    pd.DataFrame(pbar_on).to_parquet(out_dir / f"pbar_{p}.parquet")
    pesos.to_parquet(out_dir / f"pesos_{p}.parquet")
    perd = []
    for (ret, exc), (L, meta) in historias.items():
        perd.append(L.assign(retraso=ret, excluir_sin_terminar=exc, K=meta["K"].to_numpy(),
                             dias_crisis=meta["dias_crisis"].to_numpy()).reset_index())
    pd.concat(perd, ignore_index=True).to_parquet(out_dir / f"perdidas_{p}.parquet")
    evaluacion.to_csv(out_dir / f"evaluacion_{p}.csv", index=False)
    rej.to_csv(out_dir / f"eta_{p}.csv", index=False)
    util.to_csv(out_dir / f"utilidad_{p}.csv")
    (out_dir / f"criterio_{p}.json").write_text(
        json.dumps(fila, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return fila


def veredicto(filas: Mapping[str, Mapping]) -> dict[str, object]:
    """Veredicto del §6.2 a partir de las filas de criterio de A y B (``aporta``/``indicio``/``no aporta``)."""
    cumple = {p: bool(f["cumple"]) for p, f in filas.items()}
    completas = set(filas) == set(PISTAS)
    n = sum(cumple.values())
    # Como fijas.veredicto: sin las dos pistas no hay veredicto (una pista que cumple no es aún
    # un «indicio», y una que no cumple no es un «no aporta»).
    if not completas:
        principal = "pendiente (faltan pistas)"
    else:
        principal = "aporta" if n == len(PISTAS) else "indicio" if n == 1 else "no aporta"
    aprenden = completas and all(bool(f["O_gt_M_oos"]) and bool(f["O_gt_M_H2"]) for f in filas.values())
    return {"cumple_por_pista": cumple, "veredicto": principal, "los_pesos_aprenden": aprenden,
            "pistas": sorted(filas), "F*": {p: f.get("F*") for p, f in filas.items()}}


def ejecutar(pistas: Iterable[str] = PISTAS, *, n_boot: int | None = None) -> dict[str, object]:
    """Ejecuta las pistas pedidas y rehace ``criterio.csv`` y ``resumen.json`` con las disponibles."""
    out_dir = directorio_resultados()
    for p in pistas:
        ejecutar_pista(p, n_boot=n_boot)
    filas = {p: leer_criterio_pista(p) for p in PISTAS if (out_dir / f"criterio_{p}.json").is_file()}
    pd.DataFrame([{k: v for k, v in f.items() if k != "sensibilidades"} for f in filas.values()]
                 ).to_csv(out_dir / "criterio.csv", index=False)
    res = veredicto(filas)
    res["n_boot"] = n_boot
    (out_dir / "resumen.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                          encoding="utf-8")
    return res


# --------------------------------------------------------------------------- #
# Lectura (para el notebook 17)
# --------------------------------------------------------------------------- #
_COLS_META_PESOS = ("tau", "variante", "eta", "vigencia", "aprendido", "K", "entropia_efectiva")


def _ruta(nombre: str) -> Path:
    ruta = directorio_resultados() / nombre
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe {ruta}: ejecuta `python -m regimenes.fusion.online ejecutar`.")
    return ruta


def leer_senales(pista: str) -> pd.DataFrame:
    """Señales O, M, V (y F*) y todas las online (bool), índice OOS."""
    return pd.read_parquet(_ruta(f"senales_{pista.upper()}.parquet"))


def leer_pbar(pista: str) -> pd.DataFrame:
    """``p̄`` de cada señal online."""
    return pd.read_parquet(_ruta(f"pbar_{pista.upper()}.parquet"))


def leer_pesos(pista: str, variante: str | None = "principal", eta: float | None = ETA_PRINCIPAL) -> pd.DataFrame:
    """Historia de pesos por τ en formato ancho (una columna por detector) con ``vigencia``,
    ``aprendido``, ``K`` y ``entropia_efectiva``; filtra variante y η si se dan."""
    W = pd.read_parquet(_ruta(f"pesos_{pista.upper()}.parquet"))
    if variante is not None:
        W = W[W["variante"] == variante]
    if eta is not None:
        W = W[W["eta"] == float(eta)]
    return W.reset_index(drop=True)


def leer_perdidas(pista: str) -> pd.DataFrame:
    """``L_i(τ)`` por (retraso, excluir_sin_terminar), con K y días de crisis vistos en τ."""
    return pd.read_parquet(_ruta(f"perdidas_{pista.upper()}.parquet"))


def leer_evaluacion(pista: str, tramo: str | None = None) -> pd.DataFrame:
    """Vara (``vara.evaluar``) de individuales, comparadores y online en ``oos``, ``H1`` y ``H2``."""
    t = pd.read_csv(_ruta(f"evaluacion_{pista.upper()}.csv"))
    return t if tramo is None else t[t["tramo"] == tramo].set_index("id")


def leer_mcs(pista: str, tramo: str = "oos") -> pd.DataFrame:
    """MCS-O de todo el OOS (``oos``, criterio ii) o de H2 (sin peso)."""
    return pd.read_csv(_ruta(f"mcs_O_{pista.upper()}_{tramo}.csv"), index_col=0)


def leer_eta(pista: str) -> pd.DataFrame:
    """Rejilla de η con nivel, score y switching por tramo y su puesto en H1 (§4.4)."""
    return pd.read_csv(_ruta(f"eta_{pista.upper()}.csv"))


def leer_utilidad(pista: str) -> pd.DataFrame:
    """Utilidad operativa (sin peso) de O, M, V, F* e I* por tramo."""
    return pd.read_csv(_ruta(f"utilidad_{pista.upper()}.csv"), index_col=0)


def leer_criterio_pista(pista: str) -> dict:
    """Criterio §6.2 de una pista, con el control de η y las sensibilidades."""
    return json.loads(_ruta(f"criterio_{pista.upper()}.json").read_text(encoding="utf-8"))


def leer_criterio() -> pd.DataFrame:
    return pd.read_csv(_ruta("criterio.csv"))


def leer_resumen() -> dict:
    return json.loads(_ruta("resumen.json").read_text(encoding="utf-8"))


def pesos_finales(pista: str, top: int | None = 5) -> pd.Series:
    """Pesos de O (η = 10) en el último τ, de mayor a menor."""
    W = leer_pesos(pista)
    ult = W.iloc[-1]
    cols = [c for c in W.columns if c not in _COLS_META_PESOS]
    s = ult[cols].astype(float).sort_values(ascending=False)
    return s if top is None else s.head(top)


def _imprimir() -> None:
    res = leer_resumen()
    print(f"\nVeredicto §6.2: {res['veredicto']} · cumple por pista {res['cumple_por_pista']}"
          f" · los pesos aprenden: {res['los_pesos_aprenden']} · F*: {res['F*']}")
    crit = leer_criterio()
    cols = ["pista", "I*", "I*_H1", "F*", "score_O", "score_M", "score_V", "score_F*", "score_I*",
            "score_O_H2", "score_M_H2", "score_V_H2", "score_F*_H2", "score_I*_H1_en_H2", "nivel_O",
            "nivel_O_H2", "i", "ii", "iii", "cumple", "mcs_O_incluido", "mcs_M_incluido", "mcs_V_incluido",
            "mcs_Istar_incluido", "mcs_O_H2_incluido", "primer_tau_aprendido", "eta_H1",
            "score_eta_H1_en_H2"]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(crit.set_index("pista")[[c for c in cols[1:] if c in crit.columns]].T.to_string())
        for p in crit["pista"]:
            print(f"\n[{p}] pesos finales de O (η = 10):")
            print(pesos_finales(p).round(3).to_string())
            print(f"[{p}] sensibilidades (score oos / H1 / H2):")
            for sid, v in leer_criterio_pista(p)["sensibilidades"].items():
                print(f"  {sid:38s} {v['oos']:.3f} / {v['H1']:.3f} / {v['H2']:.3f}")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m regimenes.fusion.online",
                                 description="Fusión online C1 (ADR-011 §4).")
    ap.add_argument("accion", choices=("ejecutar", "leer"))
    ap.add_argument("--pista", nargs="+", default=list(PISTAS), type=str.upper, choices=PISTAS)
    ap.add_argument("--n-boot", type=int, default=None, help="solo para pruebas rápidas del MCS")
    a = ap.parse_args(argv)
    if a.accion == "ejecutar":
        ejecutar(a.pista, n_boot=a.n_boot)
    _imprimir()


if __name__ == "__main__":
    main()
