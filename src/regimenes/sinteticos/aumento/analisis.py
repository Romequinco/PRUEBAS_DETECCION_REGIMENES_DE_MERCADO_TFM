"""Analisis pre-registrado del aumento (``aumento.contrastes``).

Medida: **exceso sobre el nulo** = ``score_deteccion`` - mediana del nulo de azar
persistente (senal de Markov independiente de las crisis con la tasa de marcado y
la racha media de la propia senal; misma ley que
``evaluacion.ranking.persistent_random_null``). ``Delta`` = exceso(aumentado) -
exceso(proporcion 0 = benchmark), sobre los mismos dias OOS.

Todo se calcula a partir de **estadisticos por segmento**: el tramo OOS se parte
en un segmento por crisis evaluada (la crisis con la calma que la rodea hasta el
punto medio con la vecina). Por segmento y senal: dias, dias marcados, aciertos
(marcados dentro de una ventana), inicios de racha y crisis detectada (racha de
``min_run`` dentro de la ventana, como ``detection_summary``), y si el primer y
el ultimo dia del segmento estan marcados. Precision y recall por evento son
sumas de esos estadisticos, asi que permutar o remuestrear segmentos es exacto
para el score. La racha media tambien es exacta: :func:`agregar` resta las
rachas que continuan de un segmento al siguiente (ultimo dia marcado y primer dia
del siguiente marcado), con los segmentos en el orden en que se concatenan.

La mediana del nulo depende de (tasa, racha media); se tabula una vez por pista
en una rejilla (``tabla_nulo``) y se interpola, de modo que el estadistico
observado y los replicados se calculan igual.

- p-valor (confirmatorio): permutacion pareada EXACTA intercambiando las dos
  senales dentro de cada segmento (2^S patrones), bilateral:
  ``p = #{|T*| >= |T_obs|} / 2^S``. El patron identidad y el intercambio total dan
  ``|T| = |T_obs|``, asi que el p minimo es ``2 / 2^S``.
- Intervalo (descriptivo): bootstrap pareado por segmentos, percentiles; los
  mismos replicados para todas las senales de la pista. La rejilla del nulo es la
  de la composicion completa (aproximacion declarada en los replicados).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from regimenes.sinteticos.aumento.config import DIR_RESULTADOS, config_aumento, config_completa, semilla

STATS = ("n_dias", "n_marcados", "n_aciertos", "n_inicios", "n_eventos", "n_detectados", "primero", "ultimo")


# --------------------------------------------------------------------------- segmentos
def ventanas_evaluadas(indice: pd.DatetimeIndex, ventanas: dict) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Ventanas con al menos un dia OOS, ordenadas por pico."""
    indice = pd.DatetimeIndex(indice)
    salida = {}
    for nombre, (a, b) in sorted(ventanas.items(), key=lambda kv: pd.Timestamp(kv[1][0])):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if ((indice >= a) & (indice <= b)).any():
            salida[nombre] = (a, b)
    return salida


def segmentos(indice: pd.DatetimeIndex, ventanas: dict) -> np.ndarray:
    """Segmento (0..S-1) de cada dia: frontera en el punto medio entre crisis consecutivas."""
    indice = pd.DatetimeIndex(indice)
    evaluadas = list(ventanas_evaluadas(indice, ventanas).values())
    if not evaluadas:
        raise ValueError("No hay crisis evaluadas en el tramo OOS.")
    pos = np.arange(len(indice))
    fronteras = []
    for (_, fin_i), (ini_j, _) in zip(evaluadas[:-1], evaluadas[1:]):
        ultimo = int(np.searchsorted(indice, fin_i, side="right")) - 1
        primero = int(np.searchsorted(indice, ini_j, side="left"))
        fronteras.append((ultimo + primero + 1) // 2)       # primer dia del segmento siguiente
    return np.searchsorted(np.asarray(fronteras), pos, side="right")


def _racha_max(v: np.ndarray) -> int:
    if not v.any():
        return 0
    d = np.diff(np.concatenate(([0], v.astype(np.int8), [0])))
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def estadisticos_segmento(flags: np.ndarray, indice: pd.DatetimeIndex, ventanas: dict, seg: np.ndarray,
                          min_run: int = 3) -> np.ndarray:
    """Matriz (S, 8) con ``STATS`` por segmento para una senal booleana."""
    flags = np.asarray(flags, dtype=bool)
    indice = pd.DatetimeIndex(indice)
    S = int(seg.max()) + 1
    dentro = np.zeros(len(flags), dtype=bool)
    out = np.zeros((S, len(STATS)), dtype=np.int64)
    eventos = ventanas_evaluadas(indice, ventanas)
    for a, b in eventos.values():
        m = (indice >= a) & (indice <= b)
        dentro |= m
        k = int(seg[np.flatnonzero(m)[0]])
        n_w = int(m.sum())
        out[k, 4] += 1
        out[k, 5] += int(_racha_max(flags[m]) >= min(int(min_run), n_w))
    prev = np.concatenate(([False], flags[:-1]))
    inicio_seg = np.concatenate(([True], seg[1:] != seg[:-1]))
    inicios = flags & (~prev | inicio_seg)
    out[:, 0] = np.bincount(seg, minlength=S)
    out[:, 1] = np.bincount(seg, weights=flags, minlength=S).astype(np.int64)
    out[:, 2] = np.bincount(seg, weights=flags & dentro, minlength=S).astype(np.int64)
    out[:, 3] = np.bincount(seg, weights=inicios, minlength=S).astype(np.int64)
    primeros = np.flatnonzero(inicio_seg)
    ultimos = np.r_[primeros[1:] - 1, len(flags) - 1]
    out[:, 6] = flags[primeros]
    out[:, 7] = flags[ultimos]
    return out


def agregar(st: np.ndarray) -> np.ndarray:
    """Suma estadisticos sobre el eje de segmentos (-2), con los inicios de racha exactos.

    ``st`` (..., S, 8) en el orden de concatenacion. Una racha que sigue del
    segmento k al k+1 se conto como inicio en los dos: se resta una vez.
    """
    st = np.asarray(st)
    total = st.sum(axis=-2).astype(float)
    sigue = (st[..., :-1, 7] > 0) & (st[..., 1:, 6] > 0)
    total[..., 3] -= sigue.sum(axis=-1)
    return total


def score_desde(st: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """F-beta(precision diaria, recall por evento) desde estadisticos agregados (..., 8)."""
    st = np.asarray(st, dtype=float)
    marc, acie, eve, det = st[..., 1], st[..., 2], st[..., 4], st[..., 5]
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(marc > 0, acie / marc, 0.0)
        rec = np.where(eve > 0, det / eve, 0.0)
        b2 = beta ** 2
        s = (1 + b2) * prec * rec / (b2 * prec + rec)
    return np.where((prec + rec) > 0, s, 0.0)


def tasa_y_racha(st: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    st = np.asarray(st, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        tasa = st[..., 1] / st[..., 0]
        racha = np.where(st[..., 3] > 0, st[..., 1] / st[..., 3], 1.0)
    return tasa, np.maximum(racha, 1.0)


# --------------------------------------------------------------------------- nulo
def _flags_markov(n: int, tasa: float, racha: float, rng: np.random.Generator) -> np.ndarray:
    """Misma ley que ``ranking._markov_flags`` (vectorizada)."""
    if tasa <= 0:
        return np.zeros(n, dtype=bool)
    if tasa >= 1:
        return np.ones(n, dtype=bool)
    r_c = max(float(racha), 1.0)
    r_k = max(r_c * (1 - tasa) / tasa, 1.0)
    estado = bool(rng.random() < tasa)
    m = int(2 * n / (r_c + r_k)) + 16
    longitudes: list[np.ndarray] = []
    total = 0
    while total < n:
        a = rng.geometric(1.0 / (r_c if estado else r_k), size=m)
        b = rng.geometric(1.0 / (r_k if estado else r_c), size=m)
        par = np.empty(2 * m, dtype=np.int64)
        par[0::2], par[1::2] = a, b
        longitudes.append(par)
        total += int(par.sum())
    L = np.concatenate(longitudes)
    estados = np.zeros(len(L), dtype=bool)
    estados[0::2] = estado
    estados[1::2] = not estado
    return np.repeat(estados, L)[:n]


def _score_rapido(flags: np.ndarray, dentro: np.ndarray, cortes: list[tuple[int, int]], min_run: int,
                  beta: float = 1.0) -> float:
    marc = int(flags.sum())
    if marc == 0:
        return 0.0
    prec = float((flags & dentro).sum()) / marc
    det = sum(_racha_max(flags[i:j]) >= min(min_run, j - i) for i, j in cortes)
    rec = det / len(cortes)
    b2 = beta ** 2
    return 0.0 if prec + rec == 0 else (1 + b2) * prec * rec / (b2 * prec + rec)


# Rejilla densa en tasas bajas (la mediana del nulo salta entre 0 y ~0,13 segun haya o no una crisis
# detectada por azar) y numeros aleatorios comunes en todas las celdas (misma semilla), para que el ruido
# de la mediana no cree escalones entre celdas vecinas.
TASAS = np.unique(np.round(np.r_[0.0, np.geomspace(0.001, 0.05, 15), np.linspace(0.05, 1.0, 39)], 5))
RACHAS = np.round(np.geomspace(1.0, 3000.0, 25), 4)


def tabla_nulo(indice: pd.DatetimeIndex, ventanas: dict, *, n_sims: int = 300, semilla_nulo: int = 42,
               min_run: int = 3, tasas=TASAS, rachas=RACHAS) -> pd.DataFrame:
    """Mediana del score del nulo persistente en la rejilla (tasa x racha)."""
    indice = pd.DatetimeIndex(indice)
    ev = ventanas_evaluadas(indice, ventanas)
    dentro = np.zeros(len(indice), dtype=bool)
    cortes = []
    for a, b in ev.values():
        m = (indice >= a) & (indice <= b)
        dentro |= m
        idx = np.flatnonzero(m)
        cortes.append((int(idx[0]), int(idx[-1]) + 1))
    filas = []
    for tasa in tasas:
        for racha in rachas:
            rng = np.random.default_rng(semilla_nulo)          # numeros aleatorios comunes
            s = [_score_rapido(_flags_markov(len(indice), tasa, racha, rng), dentro, cortes, min_run)
                 for _ in range(int(n_sims))]
            filas.append({"tasa": tasa, "racha": racha, "mediana": float(np.median(s))})
    return pd.DataFrame(filas)


@dataclass
class Nulo:
    """Interpolador bilineal de la mediana del nulo en (tasa, log racha)."""

    tabla: pd.DataFrame

    def __post_init__(self):
        from scipy.interpolate import RegularGridInterpolator

        t = np.sort(self.tabla["tasa"].unique())
        r = np.sort(self.tabla["racha"].unique())
        z = self.tabla.pivot(index="tasa", columns="racha", values="mediana").loc[t, r].to_numpy()
        self._f = RegularGridInterpolator((t, np.log(r)), z, bounds_error=False, fill_value=None)
        self._rmin, self._rmax = float(r.min()), float(r.max())

    def __call__(self, tasa, racha) -> np.ndarray:
        tasa = np.clip(np.asarray(tasa, dtype=float), 0.0, 1.0)
        racha = np.clip(np.asarray(racha, dtype=float), self._rmin, self._rmax)
        puntos = np.stack([tasa.ravel(), np.log(racha).ravel()], axis=-1)
        return self._f(puntos).reshape(tasa.shape)


def exceso(st: np.ndarray, nulo: Nulo, beta: float = 1.0) -> np.ndarray:
    """score - mediana del nulo con la tasa y racha de la propia senal (estadisticos agregados)."""
    tasa, racha = tasa_y_racha(st)
    return score_desde(st, beta) - nulo(tasa, racha)


# --------------------------------------------------------------------------- contrastes
def _patrones(S: int, desde: int, hasta: int) -> np.ndarray:
    k = np.arange(desde, hasta, dtype=np.int64)[:, None]
    return ((k >> np.arange(S, dtype=np.int64)) & 1).astype(bool)


def permutacion_exacta(st_a: np.ndarray, st_b: np.ndarray, nulo: Nulo, beta: float = 1.0,
                       bloque: int = 1 << 14) -> tuple[float, float]:
    """``(T_obs, p)`` del contraste exceso(a) - exceso(b) intercambiando senales por segmento."""
    st_a, st_b = np.asarray(st_a), np.asarray(st_b)
    S = st_a.shape[0]
    t_obs = float(exceso(agregar(st_a), nulo, beta) - exceso(agregar(st_b), nulo, beta))
    total, extremos = 1 << S, 0
    for desde in range(0, total, bloque):
        M = _patrones(S, desde, min(total, desde + bloque))[:, :, None]
        a = agregar(np.where(M, st_b[None], st_a[None]))
        b = agregar(np.where(M, st_a[None], st_b[None]))
        t = exceso(a, nulo, beta) - exceso(b, nulo, beta)
        extremos += int((np.abs(t) >= abs(t_obs) - 1e-12).sum())
    return t_obs, extremos / total


def replicas_bootstrap(S: int, n_boot: int, semilla_boot: int) -> np.ndarray:
    """Indices de segmentos remuestreados (n_boot, S), comunes a todas las senales."""
    return np.random.default_rng(semilla_boot).integers(0, S, size=(int(n_boot), S))


def bootstrap(st_list: list[np.ndarray], pesos: list[float], replicas: np.ndarray, nulo: Nulo,
              beta: float = 1.0) -> np.ndarray:
    """Replicados de ``sum_k pesos_k x exceso(senal_k)`` (combinacion lineal de excesos)."""
    total = np.zeros(len(replicas))
    for st, w in zip(st_list, pesos):
        total += w * exceso(agregar(np.asarray(st)[replicas]), nulo, beta)
    return total


def ic(valores: np.ndarray, alfa: float = 0.05) -> tuple[float, float]:
    return float(np.quantile(valores, alfa / 2)), float(np.quantile(valores, 1 - alfa / 2))


def holm(p: Iterable[float], alfa: float = 0.05) -> np.ndarray:
    """Rechazos de Holm (bool) en el orden de entrada."""
    p = np.asarray(list(p), dtype=float)
    orden = np.argsort(p)
    rechazo = np.zeros(len(p), dtype=bool)
    m = len(p)
    for k, i in enumerate(orden):
        if p[i] <= alfa / (m - k):
            rechazo[i] = True
        else:
            break
    return rechazo


# --------------------------------------------------------------------------- niveles ADR-003
def nivel_binario(flags: np.ndarray, base_rate: float, precision: float, lift_min: float | None = None,
                  min_mean_duration: float | None = None, min_crisis_run: float | None = None) -> int:
    """Nivel ADR-003 (0 elegible, 1 sin lift, 2 parpadeo, 3 degenerado) de una senal BINARIA.

    Umbrales de ``ranking.DETECTION_DEFAULTS``. Diferencia declarada: la duracion
    media de regimen se mide sobre la senal binaria crisis / no crisis (ADR-003 la
    mide sobre los K estados): es la unica definicion comun a la base y al voto.
    """
    from regimenes.evaluacion.ranking import DETECTION_DEFAULTS as D

    lift_min = D["lift_min"] if lift_min is None else lift_min
    min_mean_duration = D["min_mean_duration"] if min_mean_duration is None else min_mean_duration
    min_crisis_run = D["min_crisis_run"] if min_crisis_run is None else min_crisis_run
    flags = np.asarray(flags, dtype=bool)
    cambios = int((flags[1:] != flags[:-1]).sum())
    marc = flags.mean()
    if cambios == 0 or marc <= 0 or marc >= 1 or not np.isfinite(precision):
        return 3
    inicios = int((flags & ~np.concatenate(([False], flags[:-1]))).sum())
    duracion = len(flags) / (cambios + 1)
    racha = flags.sum() / inicios if inicios else 0.0
    if duracion < min_mean_duration or racha < min_crisis_run:
        return 2
    return 1 if not (precision / base_rate > lift_min) else 0


# --------------------------------------------------------------------------- consolidacion de senales
def _flags_de_panel(ruta: Path, n_states: int) -> pd.Series:
    panel = pd.read_parquet(ruta, columns=["state"])
    return panel["state"].eq(n_states - 1)


def consolidar_senales(pista: str, *, base: Path | None = None, base_res: Path | None = None) -> dict[str, pd.DataFrame]:
    """Senales (benchmark, por fuente, voto y suelo) y sus estadisticos por segmento.

    Escribe ``senales_<P>.csv`` y ``segmentos_<P>.csv`` (versionados) a partir de
    los paneles locales de los trabajos con cache VIGENTE (``consolidar_filas``).
    El voto (>= ``voto_min`` de los generadores) solo se forma cuando estan los
    cinco paneles; los votos incompletos se cuentan en ``attrs['votos_incompletos']``
    y se avisan (H1 trata un contraste ausente como p = 1).
    """
    from regimenes.benchmark.ejecucion import configure_evaluation
    from regimenes.evaluacion import metricas as met
    from regimenes.evaluacion.metricas import detection_summary
    from regimenes.sinteticos.aumento import ejecucion as ej

    cfg = config_aumento()
    pista = pista.upper()
    base_res = DIR_RESULTADOS if base_res is None else Path(base_res)
    configure_evaluation(pista)
    ventanas = dict(met.CRISIS_WINDOWS)
    filas = ej.consolidar_filas([pista], base_res)
    filas = filas[filas["pista"] == pista] if not filas.empty else filas
    senales: dict[str, tuple[dict, pd.Series]] = {}
    incompletos: list[str] = []
    indice = None
    for det in ej.detectores_pista(pista, cfg):
        ref = ej.panel_benchmark(pista, det)
        n_states = ej.spec_de(pista, det).factory().n_states
        f0 = ref["state"].eq(n_states - 1)
        indice = f0.index if indice is None else indice
        senales[f"{det}__base"] = ({"detector": det, "tipo": "base"}, f0)
        if filas.empty:
            continue
        for r in filas[filas["detector"] == det].itertuples():
            ruta = ej.ruta_panel(ej.Trabajo(pista, det, r.fuente, r.brazo, float(r.proporcion), int(r.rep),
                                            None if pd.isna(r.semilla_det) else int(r.semilla_det)), base)
            tipo = "suelo" if r.fuente == ej.SIN_FUENTE else ("control" if r.fuente == cfg["control"] else "fuente")
            meta = {"detector": det, "tipo": tipo, "fuente": r.fuente, "brazo": r.brazo,
                    "proporcion": float(r.proporcion), "rep": int(r.rep),
                    "semilla_det": None if pd.isna(r.semilla_det) else int(r.semilla_det)}
            senales[r.clave] = (meta, _flags_de_panel(ruta, n_states))
        # voto mayoritario por brazo, proporcion y repeticion
        gens = list(cfg["generadores"])
        for brazo in ("completas", "crisis"):
            for p in cfg["proporciones"]:
                for rep in range(int(cfg["repeticiones"][pista])):
                    claves = [ej.Trabajo(pista, det, g, brazo, float(p), rep).clave for g in gens]
                    if all(c in senales for c in claves):
                        votos = sum(senales[c][1].astype(int) for c in claves)
                        clave = f"{det}__voto__{brazo}__p{float(p):.2f}__r{rep}"
                        senales[clave] = ({"detector": det, "tipo": "voto", "fuente": "voto", "brazo": brazo,
                                           "proporcion": float(p), "rep": rep},
                                          votos >= int(cfg["contrastes"]["h1"]["voto_min"]))
                    else:
                        incompletos.append(f"{det}/{brazo}/p{float(p):.2f}/r{rep}")
    seg = segmentos(indice, ventanas)
    nombres_seg = list(ventanas_evaluadas(indice, ventanas))
    filas_s, filas_g = [], []
    for clave, (meta, flags) in senales.items():
        flags = flags.reindex(indice)
        if flags.isna().any():
            raise ValueError(f"La senal {clave} no cubre el indice OOS comun.")
        v = flags.to_numpy(dtype=bool)
        res = detection_summary(flags.astype(bool), ventanas, min_run=int(cfg["min_run"]))
        st = estadisticos_segmento(v, indice, ventanas, seg, int(cfg["min_run"]))
        filas_s.append({"senal": clave, **meta, "score": float(score_desde(agregar(st), float(cfg["beta"]))),
                        "precision": res["det_precision"], "recall": res["det_event_recall"],
                        "lift": res["det_lift_precision"], "tasa": res["det_marked_rate"],
                        "racha_media": res["det_mean_crisis_run"],
                        "nivel": nivel_binario(v, res["det_base_rate"], res["det_precision"])})
        for k in range(st.shape[0]):
            filas_g.append({"senal": clave, "segmento": k, "crisis": nombres_seg[k],
                            **dict(zip(STATS, st[k].tolist()))})
    tab_s, tab_g = pd.DataFrame(filas_s), pd.DataFrame(filas_g)
    base_res.mkdir(parents=True, exist_ok=True)
    tab_s.to_csv(base_res / f"senales_{pista}.csv", index=False)
    tab_g.to_csv(base_res / f"segmentos_{pista}.csv", index=False)
    asegurar_nulo(pista, indice, ventanas, base_res)
    if incompletos:
        print(f"[aviso] {pista}: {len(incompletos)} votos incompletos (faltan paneles): {incompletos[:6]}...",
              flush=True)
    tab_s.attrs["votos_incompletos"] = incompletos
    return {"senales": tab_s, "segmentos": tab_g}


def asegurar_nulo(pista: str, indice: pd.DatetimeIndex, ventanas: dict, base_res: Path | None = None) -> pd.DataFrame:
    """Rejilla del nulo de la pista, regenerada si cambia su huella (indice, ventanas, rejilla, sims)."""
    import hashlib
    import json

    cfg = config_aumento()
    base_res = DIR_RESULTADOS if base_res is None else Path(base_res)
    params = {"n_sims": int(cfg["contrastes"]["n_sims_nulo"]), "min_run": int(cfg["min_run"]),
              "semilla": semilla("nulo", pista, base=int(cfg["semilla"])), "tasas": TASAS.tolist(),
              "rachas": RACHAS.tolist(), "indice": [str(indice[0]), str(indice[-1]), len(indice)],
              "ventanas": {k: [str(a), str(b)] for k, (a, b) in sorted(ventanas.items())}}
    h = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    ruta = base_res / f"nulo_{pista}.csv"
    if ruta.exists():
        tabla = pd.read_csv(ruta)
        if "huella" in tabla and (tabla["huella"] == h).all():
            return tabla
    tabla = tabla_nulo(indice, ventanas, n_sims=params["n_sims"], semilla_nulo=params["semilla"],
                       min_run=params["min_run"])
    tabla["huella"] = h
    tabla.to_csv(ruta, index=False)
    return tabla


# --------------------------------------------------------------------------- hipotesis
@dataclass
class DatosPista:
    pista: str
    senales: pd.DataFrame
    st: dict[str, np.ndarray]          # senal -> (S, 8)
    nulo: Nulo
    replicas: np.ndarray
    nombres_segmento: list[str]

    @classmethod
    def cargar(cls, pista: str, base_res: Path | None = None) -> "DatosPista":
        cfg = config_aumento()
        pista = pista.upper()
        base_res = DIR_RESULTADOS if base_res is None else Path(base_res)
        senales = pd.read_csv(base_res / f"senales_{pista}.csv")
        seg = pd.read_csv(base_res / f"segmentos_{pista}.csv").sort_values(["senal", "segmento"])
        st = {k: g[list(STATS)].to_numpy() for k, g in seg.groupby("senal")}
        S = next(iter(st.values())).shape[0]
        rep = replicas_bootstrap(S, int(cfg["contrastes"]["n_boot"]), semilla("boot", pista, base=int(cfg["semilla"])))
        nulo = pd.read_csv(base_res / f"nulo_{pista}.csv")[["tasa", "racha", "mediana"]]
        nombres = seg.drop_duplicates("segmento").sort_values("segmento")["crisis"].tolist()
        return cls(pista, senales, st, Nulo(nulo), rep, nombres)

    def clave(self, det: str, *, tipo: str, fuente: str | None = None, brazo: str | None = None,
              p: float | None = None, rep: int = 0) -> str | None:
        s = self.senales
        m = (s["detector"] == det) & (s["tipo"] == tipo)
        if fuente is not None:
            m &= s["fuente"] == fuente
        if brazo is not None:
            m &= s["brazo"] == brazo
        if p is not None:
            m &= np.isclose(s["proporcion"].astype(float), float(p))
        if tipo != "base":
            m &= s["rep"].fillna(0).astype(int) == int(rep)
        hits = s.loc[m, "senal"].tolist()
        return hits[0] if hits else None

    def delta(self, a: str, b: str) -> float:
        return float(exceso(agregar(self.st[a]), self.nulo) - exceso(agregar(self.st[b]), self.nulo))

    def boot(self, pares: list[tuple[str, float]]) -> np.ndarray:
        return bootstrap([self.st[k] for k, _ in pares], [w for _, w in pares], self.replicas, self.nulo)


def h1(datos: dict[str, DatosPista], cfg: dict | None = None) -> pd.DataFrame:
    """Contraste principal por pista y detector (voto, completas, p = 1, repeticion 0).

    La familia de Holm es FIJA (todos los detectores evaluables de las pistas): un
    contraste que falta cuenta con p = 1 y veredicto "falta" (nunca encoge la
    familia). Condiciones de "mejora" (yaml): (a) Holm y Delta > 0; (b) Delta >=
    mejora_minima; (c) el nivel ADR-003 no empeora; (d) ningun generador por
    separado tiene el limite superior de su IC < 0; (e) si el detector esta en el
    suelo de ruido, Delta supera el mayor |Delta| entre semillas (frente a la base y
    entre las dos semillas); si falta su suelo, no hay "mejora".
    """
    from regimenes.sinteticos.aumento.config import detectores_pista

    cfg = config_aumento() if cfg is None else cfg
    c = cfg["contrastes"]
    suelo_cfg = cfg["suelo_ruido"]["detectores"]
    filas = []
    for pista, d in datos.items():
        for det in detectores_pista(pista, cfg):
            base = d.clave(det, tipo="base")
            voto = d.clave(det, tipo="voto", brazo=c["h1"]["brazo"], p=c["h1"]["proporcion"])
            fila = {"pista": pista, "detector": det}
            if voto is None or base is None:
                filas.append({**fila, "p": 1.0, "falta": True})
                continue
            t_obs, p = permutacion_exacta(d.st[voto], d.st[base], d.nulo)
            lo, hi = ic(d.boot([(voto, 1.0), (base, -1.0)]), c["alfa"])
            peor_gen = np.nan
            for g in cfg["generadores"]:
                k = d.clave(det, tipo="fuente", fuente=g, brazo=c["h1"]["brazo"], p=c["h1"]["proporcion"])
                if k is not None:
                    peor_gen = np.nanmin([peor_gen, ic(d.boot([(k, 1.0), (base, -1.0)]), c["alfa"])[1]])
            claves_suelo = d.senales.loc[(d.senales["detector"] == det) & (d.senales["tipo"] == "suelo"),
                                         "senal"].tolist()
            difs = [abs(d.delta(k, base)) for k in claves_suelo]
            difs += [abs(d.delta(a, b)) for i, a in enumerate(claves_suelo) for b in claves_suelo[i + 1:]]
            niv = d.senales.set_index("senal")["nivel"]
            filas.append({**fila, "delta": t_obs, "p": p, "ic_inf": lo, "ic_sup": hi,
                          "nivel_base": int(niv[base]), "nivel_aumento": int(niv[voto]),
                          "peor_ic_sup_generador": peor_gen,
                          "requiere_suelo": det in suelo_cfg.get(pista, []),
                          "suelo_ruido": max(difs) if difs else np.nan, "falta": False})
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla
    familia = c.get("familia_holm", "global")
    tabla["holm"] = False
    grupos = [tabla.index] if familia == "global" else [g.index for _, g in tabla.groupby("pista")]
    for idx in grupos:
        tabla.loc[idx, "holm"] = holm(tabla.loc[idx, "p"], c["alfa"])
    requiere = tabla["requiere_suelo"].astype("boolean").fillna(False).astype(bool)
    suelo_ok = ~requiere | (tabla["delta"] > tabla["suelo_ruido"])
    mejora = (tabla["holm"] & (tabla["delta"] > 0) & (tabla["delta"] >= c["mejora_minima"])
              & (tabla["nivel_aumento"] <= tabla["nivel_base"])
              & ~(tabla["peor_ic_sup_generador"] < 0)
              & suelo_ok)
    empeora = tabla["holm"] & (tabla["delta"] < 0)
    tabla["veredicto"] = np.select([tabla["falta"].astype(bool), mejora, empeora],
                                   ["falta", "mejora", "empeora"], "sin efecto detectable")
    return tabla


def h1_posterior(datos: dict[str, DatosPista], cfg: dict | None = None) -> pd.DataFrame:
    """EXPLORATORIO (sensibilidad declarada): Delta de H1 solo en los segmentos cuya crisis empieza
    despues de fin_train del 15 + margen. Antes de esa fecha, la familia y los parametros de los
    generadores se eligieron con datos posteriores al pliegue (fuga de diseno, no de deteccion)."""
    from regimenes.sinteticos.datos import ventanas_crisis

    cfg = config_aumento() if cfg is None else cfg
    total = config_completa()
    margen = pd.offsets.BDay(int(cfg["anclas"]["margen_reconocimiento"]))
    c = cfg["contrastes"]
    filas = []
    for pista, d in datos.items():
        corte = next(pd.Timestamp(total[k]["fin_train"]) for k in ("entrenamiento", "entrenamiento_secundario")
                     if total[k]["pista"].upper() == pista) + margen
        picos = {n: a for n, (a, _) in ventanas_crisis(pista).items()}
        sel = np.array([picos[n] > corte for n in d.nombres_segmento])
        if sel.sum() < 2:
            continue
        rep_ = replicas_bootstrap(int(sel.sum()), int(c["n_boot"]),
                                  semilla("boot_post", pista, base=int(cfg["semilla"])))
        for det in sorted(d.senales["detector"].unique()):
            base = d.clave(det, tipo="base")
            voto = d.clave(det, tipo="voto", brazo=c["h1"]["brazo"], p=c["h1"]["proporcion"])
            if voto is None:
                continue
            a, b = d.st[voto][sel], d.st[base][sel]
            delta = float(exceso(agregar(a), d.nulo) - exceso(agregar(b), d.nulo))
            boot = exceso(agregar(a[rep_]), d.nulo) - exceso(agregar(b[rep_]), d.nulo)
            lo, hi = ic(boot, c["alfa"])
            filas.append({"pista": pista, "detector": det, "desde": corte.date(), "n_segmentos": int(sel.sum()),
                          "delta": delta, "ic_inf": lo, "ic_sup": hi})
    return pd.DataFrame(filas)


def h2(datos: dict[str, DatosPista], cfg: dict | None = None) -> pd.DataFrame:
    """Pendiente de Delta(voto) frente a la proporcion (0 incluida), por brazo."""
    cfg = config_aumento() if cfg is None else cfg
    ps = np.array([0.0, *map(float, cfg["proporciones"])])
    x = ps - ps.mean()
    filas = []
    for pista, d in datos.items():
        for det in sorted(d.senales["detector"].unique()):
            base = d.clave(det, tipo="base")
            for brazo in ("completas", "crisis"):
                claves = [d.clave(det, tipo="voto", brazo=brazo, p=p) for p in ps[1:]]
                if any(k is None for k in claves):
                    continue
                # pendiente MCO = sum x_i Delta_i / sum x_i^2 (sum x_i = 0), con Delta_0 = 0
                pesos = [(k, xi / (x ** 2).sum()) for k, xi in zip(claves, x[1:])]
                pend = sum(w * d.delta(k, base) for k, w in pesos)
                rep = d.boot([*pesos, (base, -sum(w for _, w in pesos))])
                lo, hi = ic(rep, cfg["contrastes"]["alfa"])
                filas.append({"pista": pista, "detector": det, "brazo": brazo, "pendiente": pend,
                              "ic_inf": lo, "ic_sup": hi})
    return pd.DataFrame(filas)


def _diferencia(datos, cfg, a_fn, b_fn, etiqueta: str) -> pd.DataFrame:
    filas = []
    for pista, d in datos.items():
        for det in sorted(d.senales["detector"].unique()):
            a, b = a_fn(d, det), b_fn(d, det)
            if a is None or b is None:
                continue
            lo, hi = ic(d.boot([(a, 1.0), (b, -1.0)]), cfg["contrastes"]["alfa"])
            filas.append({"pista": pista, "detector": det, etiqueta: d.delta(a, b), "ic_inf": lo, "ic_sup": hi})
    return pd.DataFrame(filas)


def h3(datos: dict[str, DatosPista], cfg: dict | None = None) -> pd.DataFrame:
    """Delta(voto) - Delta(control) = exceso(voto) - exceso(control), completas, p = 1.

    Lectura secundaria (declarada): el voto promedia 5 ajustes y el control es uno
    solo, asi que se publica tambien la media de los 5 Delta(generador) - Delta(control),
    que compara ajustes individuales con ajustes individuales.
    """
    cfg = config_aumento() if cfg is None else cfg
    p = cfg["contrastes"]["h1"]["proporcion"]
    tabla = _diferencia(datos, cfg, lambda d, det: d.clave(det, tipo="voto", brazo="completas", p=p),
                        lambda d, det: d.clave(det, tipo="control", fuente=cfg["control"], brazo="completas", p=p),
                        "voto_menos_control")
    extra = []
    for pista, d in datos.items():
        for det in sorted(d.senales["detector"].unique()):
            ctrl = d.clave(det, tipo="control", fuente=cfg["control"], brazo="completas", p=p)
            gens = [d.clave(det, tipo="fuente", fuente=g, brazo="completas", p=p) for g in cfg["generadores"]]
            if ctrl is None or any(g is None for g in gens):
                continue
            pares = [(g, 1.0 / len(gens)) for g in gens] + [(ctrl, -1.0)]
            lo, hi = ic(d.boot(pares), cfg["contrastes"]["alfa"])
            extra.append({"pista": pista, "detector": det,
                          "media_gen_menos_control": float(np.mean([d.delta(g, ctrl) for g in gens])),
                          "media_ic_inf": lo, "media_ic_sup": hi})
    return tabla.merge(pd.DataFrame(extra), on=["pista", "detector"], how="left") if extra else tabla


def h5(datos: dict[str, DatosPista], cfg: dict | None = None) -> pd.DataFrame:
    """Delta(crisis) - Delta(completas) con p = 1 (estimando: menos el «efecto de anadir calma»)."""
    cfg = config_aumento() if cfg is None else cfg
    p = cfg["contrastes"]["h1"]["proporcion"]
    return _diferencia(datos, cfg, lambda d, det: d.clave(det, tipo="voto", brazo="crisis", p=p),
                       lambda d, det: d.clave(det, tipo="voto", brazo="completas", p=p), "crisis_menos_completas")


def h4(datos: dict[str, DatosPista], cfg: dict | None = None) -> pd.DataFrame:
    """Delta medio del grupo que aprende un estado menos el del grupo de umbral (por pista)."""
    cfg = config_aumento() if cfg is None else cfg
    c = cfg["contrastes"]
    filas = []
    for pista, d in datos.items():
        def pares(grupo):
            out = []
            dets = [x for x in c["grupos_h4"][grupo] if d.clave(x, tipo="voto", brazo="completas",
                                                                 p=c["h1"]["proporcion"]) is not None]
            for det in dets:
                v = d.clave(det, tipo="voto", brazo="completas", p=c["h1"]["proporcion"])
                out += [(v, 1.0 / len(dets)), (d.clave(det, tipo="base"), -1.0 / len(dets))]
            return out, dets
        pa, da = pares("aprende_estado")
        pu, du = pares("umbral")
        if not da or not du:
            continue
        est = sum(w * float(exceso(agregar(d.st[k]), d.nulo)) for k, w in pa) \
            - sum(w * float(exceso(agregar(d.st[k]), d.nulo)) for k, w in pu)
        lo, hi = ic(d.boot(pa + [(k, -w) for k, w in pu]), c["alfa"])
        filas.append({"pista": pista, "aprende_estado": ",".join(da), "umbral": ",".join(du),
                      "diferencia": est, "ic_inf": lo, "ic_sup": hi})
    return pd.DataFrame(filas)


def regla_19(t1: pd.DataFrame, t3: pd.DataFrame) -> pd.DataFrame:
    """El aumento se adopta solo si H1 = «mejora» y H3 tiene IC > 0 en esa pista."""
    if t1.empty:
        return t1
    out = t1[["pista", "detector", "veredicto"]].merge(t3[["pista", "detector", "ic_inf"]]
                                                       .rename(columns={"ic_inf": "h3_ic_inf"}),
                                                       on=["pista", "detector"], how="left")
    out["adoptar"] = out["veredicto"].eq("mejora") & out["h3_ic_inf"].gt(0)
    return out
