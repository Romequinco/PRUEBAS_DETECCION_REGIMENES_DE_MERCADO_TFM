"""Fidelidad: marginales, dependencia (hechos estilizados) y regimenes.

CONTRATO DE SALIDA (comun a todas las funciones de ``validacion``): tabla larga
con una fila por (regimen, columna, metrica) y columnas

    regimen     'calma' | 'crisis' | 'todos'
    columna     feature (o 'SP500_ret', 'persistentes', '—')
    tipo        'modelada' | 're-derivada' | '—'
    metrica     nombre corto estable (lo lee ``veredicto``)
    real        valor en el tramo real de entrenamiento
    sintetico   valor en lo sintetico (todas las trayectorias apiladas o
                mediana entre trayectorias, segun la metrica; lo dice el docstring)
    banda_inf, banda_sup   banda de referencia del real (bootstrap por bloques;
                en crisis por episodio). NaN si la metrica no lleva banda.
    cociente    sintetico / real cuando tiene sentido; NaN si no
    en_banda    bool: sintetico dentro de [banda_inf, banda_sup] (NaN si no hay banda)

Las autocorrelaciones se calculan DENTRO de cada regimen: solo con pares de
dias que pertenecen a la misma racha (pendiente heredado del notebook 15, donde
eran de la trayectoria entera).

Convenciones de este modulo (las cita el notebook 16)
-----------------------------------------------------
- ``en_banda`` es de tipo pandas ``boolean`` (``<NA>`` cuando la metrica no lleva
  banda). Con solo ``banda_sup`` (distancias) se lee ``sintetico <= banda_sup``.
  Excepcion: en ``curtosis_condicional_tray`` la banda es la del SINTETICO
  (valores por trayectoria) y ``en_banda`` dice si el REAL cae dentro.
- **Bootstrap del real** (``_remuestreos``), comun a todas las bandas y
  determinista con ``semilla``:

  * calma (regimen 0): bootstrap por bloques contiguos de ``largo_bloque``
    sesiones (63 por defecto, ~3 meses: conserva el agrupamiento de volatilidad
    hasta el retardo 21 con >= 42 pares por bloque) tomados DENTRO de las rachas
    de calma (un bloque nunca cruza a crisis; una racha mas corta que el bloque
    entra entera como un candidato), hasta igualar el numero de dias de calma
    del real; el ultimo bloque se recorta.
  * crisis (regimen 1): remuestreo por EPISODIO: se extraen con reemplazo tantas
    rachas de crisis completas como tiene el real (8 en la pista A, 4 en la B).
    El n efectivo de crisis es el numero de episodios, no el de dias; el tamano
    del remuestreo varia con las duraciones elegidas.
  * Cada bloque/episodio remuestreado es un segmento propio: los pares (t, t+k)
    de las ACF solo se forman dentro de un mismo segmento, igual que en el real.
  * Banda de un estadistico = percentiles 2,5 y 97,5 de sus ``n_boot`` valores.
  * Banda de una DISTANCIA (Wasserstein, KS, Frobenius): la referencia es la
    distancia entre el real y un remuestreo del propio real; ``banda_inf = 0``,
    ``banda_sup`` = p95 de esas distancias y la columna ``real`` lleva su
    mediana (nivel tipico de ruido). Es una referencia ESTRICTA: un remuestreo
    comparte dias con el real, asi que su distancia es menor que la de dos
    muestras independientes de la misma ley; el sintetico (100 trayectorias
    apiladas, casi sin ruido propio) debe quedar por debajo de ella.
- **Curtosis**: siempre de Pearson (NO exceso; la normal da 3), con momentos
  muestrales sin correccion de sesgo. Asimetria: tercer momento estandarizado.
- **Cocientes** (sintetico / real) solo para magnitudes de escala positivas:
  desviacion, amplitud de cola de los cuantiles (ver ``fidelidad_marginal``),
  ACF de |r| y r^2, curtosis condicional, pendiente log-log, persistencia,
  escalones, separacion de volatilidad, duraciones, P_kk y AUC. NaN para
  medias de features estandarizadas, asimetria, curtosis incondicional (depende
  de un solo dato; la cola la miden los cuantiles), ACF de r, apalancamiento y
  distancias. Un cociente tambien es NaN si ``|real|`` es menor que el minimo
  indicado en cada metrica (evita cocientes explosivos con denominador ~0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import ks_2samp, wasserstein_distance

from regimenes.sinteticos.datos import matriz_transicion
from regimenes.sinteticos.espacio import DERIVADAS
from regimenes.sinteticos.validacion._comun import id_episodio, separar_real, separar_sintetico, tramos

NOMBRES_REGIMEN = {0: "calma", 1: "crisis"}
COLUMNAS_SALIDA = ["regimen", "columna", "tipo", "metrica", "real", "sintetico",
                   "banda_inf", "banda_sup", "cociente", "en_banda"]
LARGO_BLOQUE = 63          # sesiones por bloque en el bootstrap de calma
LAMBDA_EWMA = 0.94         # RiskMetrics diario
QUEMADO_EWMA = 21          # sesiones iniciales sin residuo estandarizado
MAX_RETARDO_LOGLOG = 21    # retardos 1..21 para la pendiente log-log de ACF|r|
UMBRAL_PERSISTENCIA = 0.9  # acf1 del real (serie entera) por encima -> 'persistente'
UMBRAL_ESCALONES = 0.5     # fraccion de dias sin cambio del real por encima -> 'mensual'
TOPE_VIDA_MEDIA = 10_000.0  # sesiones: vida media de una columna con acf1 >= 1 (o mayor que esto)
MIN_TRAYECTORIAS_BANDA = 10  # trayectorias validas minimas para la banda por trayectoria
PASO_VENTANAS = 63         # sesiones entre inicios de ventanas reales (referencia por ventanas)
PREFIJOS_VENTANAS = ("acf_r_", "acf_abs_", "acf_sq_", "apalancamiento_")
METRICAS_VENTANAS = ("vida_media_persistentes",)
SIN_COLUMNA = "—"


# --------------------------------------------------------------------------- utilidades


def _nombre_regimen(k: int) -> str:
    return NOMBRES_REGIMEN.get(int(k), str(int(k)))


def _tipo(col: str) -> str:
    return "re-derivada" if col in DERIVADAS else "modelada"


def _cociente(sint: float, real: float, minimo: float = 0.0) -> float:
    if not (np.isfinite(sint) and np.isfinite(real)) or abs(real) <= minimo:
        return np.nan
    return float(sint / real)


def _fila(regimen, columna, tipo, metrica, real, sint, inf=np.nan, sup=np.nan, cociente=np.nan) -> dict:
    return {"regimen": regimen, "columna": columna, "tipo": tipo, "metrica": metrica,
            "real": float(real), "sintetico": float(sint), "banda_inf": float(inf),
            "banda_sup": float(sup), "cociente": float(cociente)}


def _tabla(filas: list[dict], **attrs) -> pd.DataFrame:
    tabla = pd.DataFrame(filas, columns=COLUMNAS_SALIDA[:-1])
    inf = tabla["banda_inf"].to_numpy(dtype=float)
    sup = tabla["banda_sup"].to_numpy(dtype=float)
    sint = tabla["sintetico"].to_numpy(dtype=float)
    sin_banda = np.isnan(inf) & np.isnan(sup)
    dentro = (np.where(np.isnan(inf), -np.inf, inf) <= sint) & (sint <= np.where(np.isnan(sup), np.inf, sup))
    en_banda = pd.array(dentro, dtype="boolean")
    en_banda[sin_banda] = pd.NA
    tabla["en_banda"] = en_banda
    tabla.attrs.update(attrs)
    return tabla[COLUMNAS_SALIDA]


def _banda(valores) -> tuple[float, float]:
    v = np.asarray(valores, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def _p95(valores) -> tuple[float, float]:
    """(mediana, p95) de distancias de referencia real frente a remuestreo."""
    v = np.asarray(valores, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan
    return float(np.median(v)), float(np.percentile(v, 95))


def _mediana(valores) -> float:
    """Mediana ignorando NaN (NaN si no queda ninguno)."""
    v = np.asarray(valores, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def _indices(segmentos: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Posiciones concatenadas de los segmentos ``[a, b)`` y su id de segmento."""
    if not segmentos:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    largos = np.array([b - a for a, b in segmentos], dtype=int)
    idx = np.concatenate([np.arange(a, b) for a, b in segmentos])
    seg = np.repeat(np.arange(len(segmentos)), largos)
    return idx, seg


def _segmentos_regimen(reg: np.ndarray, k: int) -> list[tuple[int, int]]:
    return [(a, b) for a, b, kk in tramos(reg) if kk == k]


def _remuestreo_bloques(rachas_k: list[tuple[int, int]], rng: np.random.Generator,
                        largo_bloque: int) -> list[tuple[int, int]]:
    """Bloques contiguos de ``largo_bloque`` dentro de las rachas hasta igualar sus dias."""
    objetivo = sum(b - a for a, b in rachas_k)
    ini, largo = [], []
    for a, b in rachas_k:
        if b - a >= largo_bloque:
            ini.append(np.arange(a, b - largo_bloque + 1))
            largo.append(np.full(b - a - largo_bloque + 1, largo_bloque))
        else:
            ini.append(np.array([a]))
            largo.append(np.array([b - a]))
    ini, largo = np.concatenate(ini), np.concatenate(largo)
    elegidos = []
    total = 0
    while total < objetivo:
        n_extra = int(np.ceil((objetivo - total) / largo_bloque)) + 1
        for j in rng.integers(0, ini.size, size=n_extra):
            resto = objetivo - total
            if resto <= 0:
                break
            dur = min(int(largo[j]), resto)
            elegidos.append((int(ini[j]), int(ini[j]) + dur))
            total += dur
    return elegidos


def _remuestreos(reg: np.ndarray, n_boot: int, semilla: int, largo_bloque: int = LARGO_BLOQUE,
                 por_episodio=(1,)) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    """``n_boot`` remuestreos del real por regimen: ``{k: [(idx, seg), ...]}``.

    Regimenes en ``por_episodio``: rachas completas con reemplazo (tantas como
    el real). Resto: bloques contiguos dentro de las rachas del regimen.
    """
    rng = np.random.default_rng(semilla)
    rachas = {int(k): _segmentos_regimen(reg, int(k)) for k in np.unique(reg)}
    salida: dict[int, list] = {k: [] for k in rachas}
    for _ in range(int(n_boot)):
        for k, rachas_k in rachas.items():
            if k in por_episodio:
                elegidas = rng.integers(0, len(rachas_k), size=len(rachas_k))
                segs = [rachas_k[j] for j in elegidas]
            else:
                segs = _remuestreo_bloques(rachas_k, rng, largo_bloque)
            salida[k].append(_indices(segs))
    return salida


def _remuestreos_serie(n: int, n_boot: int, semilla: int, largo_bloque: int = LARGO_BLOQUE):
    """Bootstrap por bloques de la serie entera (filas regimen='todos')."""
    rng = np.random.default_rng(semilla + 1)
    salida = []
    n_bloques = int(np.ceil(n / largo_bloque))
    for _ in range(int(n_boot)):
        ini = rng.integers(0, max(n - largo_bloque, 0) + 1, size=n_bloques)
        salida.append(_indices([(int(a), int(min(a + largo_bloque, n))) for a in ini]))
    return salida


def _real_y_sintetico(real, sintetico, regimen_real, columnas):
    X, reg = separar_real(real, regimen_real)
    trays = separar_sintetico(sintetico)
    columnas = list(X.columns) if columnas is None else list(columnas)
    for i, (T, _) in enumerate(trays):
        faltan = [c for c in columnas if c not in T.columns]
        if faltan:
            raise KeyError(f"La trayectoria {i} no tiene las columnas {faltan}.")
    faltan = [c for c in columnas if c not in X.columns]
    if faltan:
        raise KeyError(f"El real no tiene las columnas {faltan}.")
    return X[columnas].to_numpy(dtype=float), reg, [(T[columnas].to_numpy(dtype=float), r) for T, r in trays], columnas


def _apilar(trays) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trayectorias apiladas: matriz, regimen e id de racha unico global."""
    X = np.concatenate([T for T, _ in trays], axis=0)
    reg = np.concatenate([r for _, r in trays])
    seg, desplazamiento = [], 0
    for _, r in trays:
        cortes = np.r_[0, np.flatnonzero(np.diff(r) != 0) + 1]
        ids = np.zeros(r.size, dtype=int)
        ids[cortes[1:]] = 1
        ids = np.cumsum(ids) + desplazamiento
        seg.append(ids)
        desplazamiento = int(ids[-1]) + 1 if r.size else desplazamiento
    return X, reg, np.concatenate(seg)


# --------------------------------------------------------------------------- marginal


def _momentos(M: np.ndarray, cuantiles) -> dict[str, np.ndarray]:
    media = M.mean(axis=0)
    c = M - media
    m2 = (c ** 2).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        asim = (c ** 3).mean(axis=0) / m2 ** 1.5
        curt = (c ** 4).mean(axis=0) / m2 ** 2
    salida = {"media": media, "desviacion": M.std(axis=0, ddof=1), "asimetria": asim, "curtosis": curt}
    q = np.quantile(M, list(cuantiles) + [0.5], axis=0)
    for j, p in enumerate(cuantiles):
        salida[_nombre_cuantil(p)] = q[j]
    salida["_mediana"] = q[-1]
    return salida


def _nombre_cuantil(p: float) -> str:
    return f"q{int(round(100 * p)):02d}"


def _ks(a_ordenado: np.ndarray, b: np.ndarray) -> float:
    """Estadistico KS de dos muestras (``a`` ya ordenada)."""
    b = np.sort(b)
    todo = np.concatenate([a_ordenado, b])
    fa = np.searchsorted(a_ordenado, todo, side="right") / a_ordenado.size
    fb = np.searchsorted(b, todo, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def _distancias(R: np.ndarray, R_ord: np.ndarray, S: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = R.shape[1]
    w, ks = np.full(d, np.nan), np.full(d, np.nan)
    for j in range(d):
        if S.shape[0] == 0:
            continue
        if sd[j] > 0:
            w[j] = wasserstein_distance(R[:, j], S[:, j]) / sd[j]
        ks[j] = _ks(R_ord[:, j], S[:, j])
    return w, ks


def fidelidad_marginal(real: pd.DataFrame, sintetico: list[pd.DataFrame], *, regimen_real=None,
                       columnas: list[str] | None = None, cuantiles=(0.01, 0.05, 0.95, 0.99),
                       n_boot: int = 200, semilla: int = 42,
                       largo_bloque: int = LARGO_BLOQUE) -> pd.DataFrame:
    """Distribucion por feature y regimen: momentos, cuantiles de cola, Wasserstein y KS.

    Para cada regimen k (calma, crisis) y columna se comparan los dias reales con
    regimen k y los dias sinteticos con regimen k de TODAS las trayectorias
    apiladas (la cadena sintetica viene de la base comun; aqui solo importa la
    ley condicional).

    Metricas (``metrica``):

    - ``media``, ``desviacion`` (ddof=1), ``asimetria`` (m3 / m2^1.5) y
      ``curtosis`` (Pearson, m4 / m2^2: la normal da 3; NO es exceso).
    - ``q01``, ``q05``, ``q95``, ``q99`` (o los de ``cuantiles``): cuantiles
      empiricos (interpolacion lineal). Su ``cociente`` es la AMPLITUD DE COLA
      relativa ``(q_s - mediana_s) / (q_r - mediana_r)``, no ``q_s / q_r``: es
      invariante a la posicion (un cuantil cercano a 0, como el q99 del drawdown,
      haria explotar el cociente bruto) y mide si la cola es igual de ancha.
      NaN si ``|q_r - mediana_r| < 1e-12``.
    - ``wasserstein``: W1 entre real y sintetico dividida por la desviacion real
      de la columna en ese regimen (adimensional: "cuantas sd de real se desplaza
      la masa"). ``ks``: estadistico de Kolmogorov-Smirnov de dos muestras.
      Ambas son distancias: ``real`` = mediana de la distancia real frente a un
      remuestreo del real, ``banda_inf = 0``, ``banda_sup`` = su p95, cociente NaN.

    Bandas: bootstrap por bloques en calma y por episodio en crisis (ver el
    docstring del modulo), percentiles 2,5-97,5. Cocientes: ``desviacion``
    (sd_s / sd_r) y amplitudes de cola; NaN para media, asimetria y curtosis.

    Coste: ~2 x ``n_boot`` x columnas distancias W1/KS sobre el real.
    """
    X, reg, trays, columnas = _real_y_sintetico(real, sintetico, regimen_real, columnas)
    S_todo, reg_s, _ = _apilar(trays)
    cuantiles = tuple(cuantiles)
    boot = _remuestreos(reg, n_boot, semilla, largo_bloque)
    filas = []
    for k in sorted(boot):
        R = X[reg == k]
        S = S_todo[reg_s == k]
        R_ord = np.sort(R, axis=0)
        est_r = _momentos(R, cuantiles)
        est_s = _momentos(S, cuantiles) if len(S) > 1 else {m: np.full(len(columnas), np.nan) for m in est_r}
        sd_r = est_r["desviacion"]
        w_s, ks_s = _distancias(R, R_ord, S, sd_r)
        est_b = {m: [] for m in est_r}
        w_b, ks_b = [], []
        for idx, _ in boot[k]:
            B = X[idx]
            for m, v in _momentos(B, cuantiles).items():
                est_b[m].append(v)
            w, ks = _distancias(R, R_ord, B, sd_r)
            w_b.append(w)
            ks_b.append(ks)
        est_b = {m: np.vstack(v) for m, v in est_b.items()}
        w_b, ks_b = np.vstack(w_b), np.vstack(ks_b)
        nombre = _nombre_regimen(k)
        for j, col in enumerate(columnas):
            tipo = _tipo(col)
            for m in ["media", "desviacion", "asimetria", "curtosis"] + [_nombre_cuantil(p) for p in cuantiles]:
                inf, sup = _banda(est_b[m][:, j])
                r, s = est_r[m][j], est_s[m][j]
                if m == "desviacion":
                    coc = _cociente(s, r)
                elif m.startswith("q"):
                    coc = _cociente(s - est_s["_mediana"][j], r - est_r["_mediana"][j], 1e-12)
                else:
                    coc = np.nan
                filas.append(_fila(nombre, col, tipo, m, r, s, inf, sup, coc))
            for m, vs, vb in (("wasserstein", w_s, w_b), ("ks", ks_s, ks_b)):
                med, p95 = _p95(vb[:, j])
                filas.append(_fila(nombre, col, tipo, m, med, vs[j], 0.0, p95))
    return _tabla(filas, n_boot=int(n_boot), largo_bloque=int(largo_bloque),
                  curtosis="pearson (normal = 3)", sintetico="trayectorias apiladas")


# --------------------------------------------------------------------------- dependencia


def _corr_pares(x: np.ndarray, y: np.ndarray, valido: np.ndarray, k: int) -> np.ndarray | float:
    """Correlacion de Pearson de los pares (x_t, y_{t+k}) con ``t`` y ``t+k`` en la misma racha.

    ``valido`` = mascara ``seg[:-k] == seg[k:]``. Las medias y varianzas son las
    de los propios pares (estimador "por pares"). Admite matrices (por columna).
    """
    a = x[:-k][valido]
    b = y[k:][valido]
    if a.shape[0] < 3:
        return np.nan if a.ndim == 1 else np.full(a.shape[1], np.nan)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a * b).sum(axis=0) / np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))


def _z_ewma(r: np.ndarray, lam: float = LAMBDA_EWMA, quemado: int = QUEMADO_EWMA) -> np.ndarray:
    """Residuo estandarizado causal ``r_t / sigma_t`` con EWMA (RiskMetrics).

    ``sigma2_t = lam * sigma2_{t-1} + (1 - lam) * r_{t-1}^2`` (solo usa retornos
    PASADOS), arrancando en ``sigma2_0`` = media de ``r^2`` de las ``quemado``
    primeras sesiones; esas sesiones quedan NaN (calentamiento). No se reinicia
    en los cambios de racha: la volatilidad es continua entre regimenes.
    """
    r = np.asarray(r, dtype=float)
    z = np.full(r.size, np.nan)
    if r.size <= quemado:
        return z
    s0 = float(np.mean(r[:quemado] ** 2))
    y, _ = lfilter([1.0 - lam], [1.0, -lam], r[:-1] ** 2, zi=[lam * s0])
    sig2 = np.r_[s0, y]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sig2 > 0, r / np.sqrt(sig2), np.nan)
    z[:quemado] = np.nan
    return z


def _curtosis(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    if v.size < 30:
        return np.nan
    c = v - v.mean()
    m2 = (c ** 2).mean()
    return float((c ** 4).mean() / m2 ** 2) if m2 > 0 else np.nan


def _pendiente_loglog(acf: np.ndarray) -> float:
    k = np.arange(1, acf.size + 1)
    ok = np.isfinite(acf) & (acf > 0)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(k[ok]), np.log(acf[ok]), 1)[0])


def _vida_media(rho) -> np.ndarray:
    """Vida media en sesiones ``h = ln 0,5 / ln rho`` de un AR(1) con acf1 ``rho``.

    ``rho <= 0`` -> 0; ``rho >= 1`` (o ``h`` mayor que el tope) -> ``TOPE_VIDA_MEDIA``;
    NaN se conserva.
    """
    rho = np.asarray(rho, dtype=float)
    h = np.full(rho.shape, np.nan)
    ok = np.isfinite(rho)
    h[ok & (rho <= 0)] = 0.0
    h[ok & (rho >= 1)] = TOPE_VIDA_MEDIA
    medio = ok & (rho > 0) & (rho < 1)
    h[medio] = np.minimum(np.log(0.5) / np.log(rho[medio]), TOPE_VIDA_MEDIA)
    return h


def _estad_dependencia(r, z, P, E, seg, retardos) -> dict[str, float]:
    """Hechos estilizados de una muestra ya indexada (r, z, P, E alineados con ``seg``)."""
    out: dict[str, float] = {}
    max_k = max(MAX_RETARDO_LOGLOG, max(retardos))
    validos = {k: seg[:-k] == seg[k:] for k in range(1, max_k + 1)} if seg.size > max_k else {}
    a, s = np.abs(r), r * r
    acf_abs = np.array([_corr_pares(a, a, validos[k], k) if k in validos else np.nan
                        for k in range(1, MAX_RETARDO_LOGLOG + 1)])
    for k in retardos:
        v = validos.get(k)
        if v is None:
            out.update({f"acf_r_{k}": np.nan, f"acf_abs_{k}": np.nan, f"acf_sq_{k}": np.nan,
                        f"apalancamiento_{k}": np.nan})
            continue
        out[f"acf_r_{k}"] = float(_corr_pares(r, r, v, k))
        out[f"acf_abs_{k}"] = float(_corr_pares(a, a, v, k))
        out[f"acf_sq_{k}"] = float(_corr_pares(s, s, v, k))
        out[f"apalancamiento_{k}"] = float(_corr_pares(r, s, v, k))
    out["curtosis_condicional"] = _curtosis(z)
    out["pendiente_loglog_abs"] = _pendiente_loglog(acf_abs)
    if P.shape[1]:
        v = validos.get(1)
        if v is not None:
            rho = np.atleast_1d(_corr_pares(P, P, v, 1))
            out["acf1_persistentes"] = float(np.nanmean(rho))
            out["vida_media_persistentes"] = _mediana(_vida_media(rho))
        else:
            out["acf1_persistentes"] = out["vida_media_persistentes"] = np.nan
    if E.shape[1]:
        v = validos.get(1)
        if v is not None and v.sum():
            out["escalones"] = float(np.mean((E[1:][v] == E[:-1][v]).mean(axis=0)))
        else:
            out["escalones"] = np.nan
    return out


def _columnas_persistentes(X: pd.DataFrame, col_ret: str) -> tuple[list[str], list[str]]:
    """Columnas con acf1 > 0,9 (sin ``col_ret``) y con > 50 % de dias sin cambio, en el real."""
    pers, esc = [], []
    for col in X.columns:
        x = X[col].to_numpy(dtype=float)
        if x.size < 3 or np.std(x) == 0:
            continue
        if col != col_ret and np.corrcoef(x[:-1], x[1:])[0, 1] > UMBRAL_PERSISTENCIA:
            pers.append(col)
        if np.mean(np.diff(x) == 0) > UMBRAL_ESCALONES:
            esc.append(col)
    return pers, esc


def fidelidad_dependencia(real: pd.DataFrame, sintetico: list[pd.DataFrame], *, regimen_real=None,
                          col_ret: str = "SP500_ret", retardos=(1, 5, 21),
                          n_boot: int = 200, semilla: int = 42,
                          largo_bloque: int = LARGO_BLOQUE, agregacion: str = "mediana",
                          largo_referencia: int | None = None,
                          paso_ventanas: int = PASO_VENTANAS,
                          min_ventanas_no_solapadas: int = 3) -> pd.DataFrame:
    """Dependencia temporal dentro de regimen: ACF de r, |r| y r^2, apalancamiento,
    curtosis condicional, decaimiento log-log de la ACF de |r| y persistencia.

    Hechos estilizados de Cont (2001) calculados DENTRO de regimen: en el real y
    en cada trayectoria solo entran pares (t, t+k) de la misma racha.

    Metricas sobre ``col_ret`` (``columna = col_ret``, por regimen):

    - ``acf_r_k``, ``acf_abs_k``, ``acf_sq_k``: correlacion de Pearson de los
      pares (x_t, x_{t+k}) con x = r, |r|, r^2 (medias de los pares).
    - ``apalancamiento_k = corr(r_t, r^2_{t+k})`` (negativa en datos reales).
    - ``curtosis_condicional``: curtosis de Pearson (normal = 3) de
      ``z_t = r_t / sigma_t`` con ``sigma_t`` EWMA RiskMetrics (lambda = 0,94)
      sobre retornos PASADOS, calculada sobre la serie entera (real, o cada
      trayectoria) SIN reiniciar en los cambios de racha y descartando las 21
      primeras sesiones de cada serie; luego se agrupa por el regimen del dia t.
      Referencia: ruido normal de varianza constante da ~3,2 (no 3: sigma se
      estima con ~30 dias efectivos); un salto brusco de volatilidad al entrar
      en un regimen la sube porque la EWMA tarda en recogerlo.
    - ``pendiente_loglog_abs``: pendiente MCO de log ACF|r|(k) frente a log k,
      k = 1..21, solo con ACF > 0 (NaN si quedan < 3 puntos). Memoria larga
      ~ pendiente suave; decaimiento geometrico (GARCH) ~ mas empinada.

    Metricas de otras columnas (por regimen):

    - ``acf1_persistentes`` (``columna='persistentes'``): media de la acf1
      dentro de racha de las columnas que en el real (serie entera) tienen
      acf1 > 0,9, excluido ``col_ret``. Se sigue publicando, pero NO discrimina:
      con rho ~0,996 en el real, un generador con rho = 0,969 (bootstrap_regimen
      en A) da cociente 0,97 y pasa, aunque su vida media sea mas de diez veces
      menor (cifras en el comentario de ``validacion.reglas`` del yaml). De ahi
      la metrica siguiente.
    - ``vida_media_persistentes`` (``columna='persistentes'``): vida media en
      sesiones ``h = ln 0,5 / ln rho`` (lo que tarda en reducirse a la mitad un
      choque en un AR(1) de acf1 rho), con rho = acf1 dentro de racha de cada
      columna persistente; se resume con la MEDIANA entre columnas. ``rho <= 0``
      -> 0; ``rho >= 1`` -> tope de 10.000 sesiones (tambien si h lo supera).
      Real, sintetico (mediana entre trayectorias) y banda (bootstrap del real,
      igual que ``acf1_persistentes``, transformando cada remuestreo a vida
      media) como el resto; cociente ``h_sint / h_real``.
    - ``curtosis_condicional_tray`` (``columna=col_ret``): la pregunta se
      INVIERTE: es el real un valor tipico del generador? ``real`` = la
      ``curtosis_condicional`` del real; ``sintetico`` = mediana entre
      trayectorias; ``banda_inf``/``banda_sup`` = percentiles 2,5 y 97,5 de los
      valores POR TRAYECTORIA del sintetico (banda del SINTETICO, no del real);
      ``en_banda`` = el REAL dentro de esa banda; cociente NaN (decide solo
      ``en_banda``). Motivo: un estadistico de cola calculado por trayectoria
      tiene la mediana sesgada a la baja, y la banda del real en crisis (por
      episodio) es vacia o casi un punto. Se excluyen las trayectorias sin dias
      suficientes del regimen (< 30 residuos validos); con menos de 10
      trayectorias validas no hay banda (``en_banda`` = <NA>). Va siempre por
      trayectoria, sea cual sea ``agregacion``.
    - ``escalones`` (``columna='mensuales'``): fraccion media de pares
      consecutivos (misma racha) sin cambio en las columnas que en el real
      tienen > 50 % de dias sin cambio (series mensuales repetidas a diario).
      Las listas de columnas van en ``tabla.attrs``.

    Filas ``regimen='todos'``: ``acf_r_1`` y ``acf_abs_1`` sobre la serie ENTERA
    (pares que cruzan regimen incluidos), comparables con el notebook 15; su
    banda sale de un bootstrap por bloques de la serie entera.

    Sintetico: ``agregacion='mediana'`` (defecto) = mediana entre trayectorias del
    estadistico de cada una (las trayectorias sin dias del regimen no cuentan);
    ``'apilado'`` = estadistico sobre todas las trayectorias juntas (pares solo
    dentro de cada racha de cada trayectoria). Con la mediana el tamano de cada
    muestra (una trayectoria) es parecido al del real; con el apilado el valor
    es casi poblacional. Bandas: bootstrap del real dentro de regimen (bloques
    en calma, episodios en crisis), percentiles 2,5-97,5.

    Referencia por ventanas (``largo_referencia = L``; ``None`` = desactivada):
    la autocorrelacion estimada en muestras cortas esta sesgada a la baja, y las
    trayectorias (2.520 sesiones) son mas cortas que el real (~11.000 en A): el
    propio real troceado en ventanas de 2.520 da cociente de vida media ~0,80 en
    A. Con L dado, para ``acf_r_k``, ``acf_abs_k``, ``acf_sq_k``,
    ``apalancamiento_k`` y ``vida_media_persistentes`` (filas calma/crisis) el
    ``real`` y su banda NO salen del real entero y su bootstrap, sino de
    ventanas reales de L sesiones consecutivas, deslizantes con paso
    ``paso_ventanas`` (63 por defecto; solapadas si paso < L; la ultima ventana
    empieza en ``n - L`` aunque no caiga en el paso). En cada ventana la metrica
    se calcula igual que en una trayectoria: dentro de regimen, con pares de la
    misma racha y el regimen real de esa ventana (una racha cortada por el
    borde cuenta como racha). Una ventana sin dias del regimen (< 2) o con la
    metrica indefinida se excluye, igual que una trayectoria. ``real`` =
    mediana entre ventanas; banda = percentiles 2,5-97,5 entre ventanas;
    cociente = sint / ese real. Si el real tiene <= L filas hay una sola ventana
    (el real entero). Se compara asi mediana entre trayectorias con mediana
    entre ventanas reales del mismo largo. ``tabla.attrs``:
    ``referencia_ventanas`` (metricas afectadas), ``n_ventanas_reales``,
    ``ventanas_solapadas``, ``largo_referencia``, ``paso_ventanas``,
    ``referencia_ventanas_aplicada`` y ``motivo_referencia``.
    Solo se aplica si el real tiene al menos ``min_ventanas_no_solapadas`` (3)
    ventanas NO solapadas de L (``n_real // L >= 3``). Si no (pista B: 2.702
    filas frente a L = 2.520), unas pocas ventanas casi identicas darian una
    banda de un punto; se ignora ``largo_referencia`` y se usa la referencia
    normal (real entero + bootstrap), que ya es de largo comparable a L.

    Cocientes (sint / real): ``acf_abs_k`` y ``acf_sq_k`` (si |real| >= 0,05),
    ``curtosis_condicional``, ``pendiente_loglog_abs``, ``acf1_persistentes`` y
    ``escalones``; NaN para ``acf_r_k`` y ``apalancamiento_k`` (cerca de 0 y de
    signo variable).
    """
    if agregacion not in {"mediana", "apilado"}:
        raise ValueError("agregacion debe ser 'mediana' o 'apilado'.")
    retardos = tuple(int(k) for k in retardos)
    Xdf, reg = separar_real(real, regimen_real)
    if col_ret not in Xdf.columns:
        raise KeyError(f"El real no tiene la columna {col_ret!r}.")
    trays_df = separar_sintetico(sintetico)
    pers, esc = _columnas_persistentes(Xdf, col_ret)
    pers = [c for c in pers if all(c in T.columns for T, _ in trays_df)]
    esc = [c for c in esc if all(c in T.columns for T, _ in trays_df)]
    r_real = Xdf[col_ret].to_numpy(dtype=float)
    z_real = _z_ewma(r_real)
    P_real = Xdf[pers].to_numpy(dtype=float)
    E_real = Xdf[esc].to_numpy(dtype=float)
    seg_real = id_episodio(reg)

    trays = []
    for T, rr in trays_df:
        r = T[col_ret].to_numpy(dtype=float)
        trays.append((r, _z_ewma(r), T[pers].to_numpy(dtype=float), T[esc].to_numpy(dtype=float), rr,
                      id_episodio(rr)))

    def _sint(k: int | None) -> dict[str, float]:
        """Estadistico sintetico del regimen k (None = serie entera)."""
        piezas = []
        for r, z, P, E, rr, seg in trays:
            m = np.ones(rr.size, bool) if k is None else rr == k
            if m.sum() < 2:
                continue
            s = np.zeros(m.sum(), dtype=int) if k is None else seg[m]
            piezas.append((r[m], z[m], P[m], E[m], s))
        if not piezas:
            return {}
        if agregacion == "apilado":
            desp, segs = 0, []
            for p in piezas:
                segs.append(p[4] + desp)
                desp = int(segs[-1].max()) + 1
            return _estad_dependencia(np.concatenate([p[0] for p in piezas]),
                                      np.concatenate([p[1] for p in piezas]),
                                      np.concatenate([p[2] for p in piezas]),
                                      np.concatenate([p[3] for p in piezas]),
                                      np.concatenate(segs), retardos)
        valores = [_estad_dependencia(*p, retardos) for p in piezas]
        return {c: _mediana([v[c] for v in valores]) for c in valores[0]}

    def _col(metrica: str) -> tuple[str, str]:
        if metrica in {"acf1_persistentes", "vida_media_persistentes"}:
            return "persistentes", SIN_COLUMNA
        if metrica == "escalones":
            return "mensuales", SIN_COLUMNA
        return col_ret, _tipo(col_ret)

    def _coc(metrica: str, s: float, r: float) -> float:
        if metrica.startswith(("acf_abs_", "acf_sq_")):
            return _cociente(s, r, 0.05)
        if metrica in {"curtosis_condicional", "pendiente_loglog_abs", "acf1_persistentes", "escalones",
                       "vida_media_persistentes"}:
            return _cociente(s, r, 1e-12)
        return np.nan

    def _usa_ventanas(metrica: str) -> bool:
        return largo_referencia is not None and (metrica.startswith(PREFIJOS_VENTANAS)
                                                 or metrica in METRICAS_VENTANAS)

    pedido = largo_referencia
    motivo = None if pedido is None else "aplicada"
    if pedido is not None and reg.size // int(pedido) < int(min_ventanas_no_solapadas):
        motivo = (f"no aplicada: el real tiene {reg.size} filas, {reg.size // int(pedido)} ventanas no "
                  f"solapadas de {int(pedido)} < {int(min_ventanas_no_solapadas)}; referencia = real entero "
                  "+ bootstrap")
        largo_referencia = None

    inicios = np.empty(0, dtype=int)
    if largo_referencia is not None:
        L = int(largo_referencia)
        if L < 2 or int(paso_ventanas) < 1:
            raise ValueError("largo_referencia debe ser >= 2 y paso_ventanas >= 1.")
        n = reg.size
        inicios = np.array([0]) if n <= L else np.unique(np.r_[np.arange(0, n - L + 1, int(paso_ventanas)), n - L])

    def _ventanas(k: int) -> list[dict[str, float]]:
        """Estadisticos del regimen k en cada ventana real de L sesiones."""
        salida = []
        for a in inicios:
            sl = slice(int(a), int(a) + int(largo_referencia))
            rr = reg[sl]
            m = rr == k
            if m.sum() < 2:
                continue
            seg = id_episodio(rr)[m]
            salida.append(_estad_dependencia(r_real[sl][m], z_real[sl][m], P_real[sl][m], E_real[sl][m],
                                             seg, retardos))
        return salida

    boot = _remuestreos(reg, n_boot, semilla, largo_bloque)
    filas = []
    real_en_banda: dict[int, object] = {}  # filas cuyo en_banda se refiere al REAL
    for k in sorted(boot):
        m = reg == k
        est_r = _estad_dependencia(r_real[m], z_real[m], P_real[m], E_real[m], seg_real[m], retardos)
        est_s = _sint(k)
        est_b = [_estad_dependencia(r_real[idx], z_real[idx], P_real[idx], E_real[idx], seg, retardos)
                 for idx, seg in boot[k]]
        est_v = _ventanas(k) if largo_referencia is not None else []
        for metrica, r in est_r.items():
            s = est_s.get(metrica, np.nan)
            if _usa_ventanas(metrica):
                valores = [v[metrica] for v in est_v]
                r = _mediana(valores)
                inf, sup = _banda(valores)
            else:
                inf, sup = _banda([b[metrica] for b in est_b])
            columna, tipo = _col(metrica)
            filas.append(_fila(_nombre_regimen(k), columna, tipo, metrica, r, s, inf, sup, _coc(metrica, s, r)))
        # curtosis condicional por trayectoria: banda del sintetico, se pregunta por el real
        cc = np.array([_curtosis(t[1][t[4] == k]) for t in trays])
        cc = cc[np.isfinite(cc)]
        if cc.size >= MIN_TRAYECTORIAS_BANDA:
            inf, sup = np.percentile(cc, 2.5), np.percentile(cc, 97.5)
        else:
            inf, sup = np.nan, np.nan
        cc_real = est_r["curtosis_condicional"]
        filas.append(_fila(_nombre_regimen(k), col_ret, _tipo(col_ret), "curtosis_condicional_tray", cc_real,
                           _mediana(cc), inf, sup))
        real_en_banda[len(filas) - 1] = (pd.NA if np.isnan(inf) or not np.isfinite(cc_real)
                                         else bool(inf <= cc_real <= sup))

    # serie entera (mezcla regimenes, como el notebook 15)
    def _todos(r: np.ndarray, seg: np.ndarray) -> dict[str, float]:
        a = np.abs(r)
        v = seg[:-1] == seg[1:]
        return {"acf_r_1": float(_corr_pares(r, r, v, 1)), "acf_abs_1": float(_corr_pares(a, a, v, 1))}

    est_r = _todos(r_real, np.zeros(r_real.size, dtype=int))
    if agregacion == "apilado":
        segs = np.concatenate([np.full(t[0].size, i) for i, t in enumerate(trays)])
        est_s = _todos(np.concatenate([t[0] for t in trays]), segs)
    else:
        por_tray = [_todos(t[0], np.zeros(t[0].size, dtype=int)) for t in trays]
        est_s = {c: _mediana([v[c] for v in por_tray]) for c in est_r}
    boot_todos = [_todos(r_real[idx], seg) for idx, seg in _remuestreos_serie(r_real.size, n_boot, semilla,
                                                                               largo_bloque)]
    for metrica, r in est_r.items():
        inf, sup = _banda([b[metrica] for b in boot_todos])
        s = est_s[metrica]
        filas.append(_fila("todos", col_ret, _tipo(col_ret), metrica, r, s, inf, sup, _coc(metrica, s, r)))
    tabla = _tabla(filas, n_boot=int(n_boot), largo_bloque=int(largo_bloque), persistentes=pers,
                   mensuales=esc, agregacion=agregacion, lambda_ewma=LAMBDA_EWMA,
                   curtosis="pearson (normal = 3)", tope_vida_media=TOPE_VIDA_MEDIA,
                   largo_referencia=None if pedido is None else int(pedido),
                   referencia_ventanas_aplicada=largo_referencia is not None, motivo_referencia=motivo,
                   paso_ventanas=int(paso_ventanas) if largo_referencia is not None else None,
                   n_ventanas_reales=int(inicios.size),
                   ventanas_solapadas=bool(inicios.size > 1 and np.diff(inicios).min() < int(largo_referencia))
                   if largo_referencia is not None else False,
                   referencia_ventanas=sorted({f["metrica"] for f in filas
                                               if f["regimen"] != "todos" and _usa_ventanas(f["metrica"])}))
    for i, valor in real_en_banda.items():
        tabla.loc[i, "en_banda"] = valor
    return tabla


# --------------------------------------------------------------------------- correlaciones


def _corr(M: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.corrcoef(M, rowvar=False)


def _dist_corr(C1: np.ndarray, C2: np.ndarray) -> float:
    """``||C1 - C2||_F / sqrt(d (d - 1))``: RMS de la diferencia en los pares i < j."""
    iu = np.triu_indices(C1.shape[0], k=1)
    dif = (C1 - C2)[iu]
    dif = dif[np.isfinite(dif)]
    return float(np.sqrt(np.mean(dif ** 2))) if dif.size else np.nan


def distancia_correlaciones(real: pd.DataFrame, sintetico: list[pd.DataFrame], *, regimen_real=None,
                            columnas: list[str] | None = None, n_boot: int = 200,
                            semilla: int = 42, largo_bloque: int = LARGO_BLOQUE) -> pd.DataFrame:
    """Distancia de Frobenius entre correlaciones por regimen real y sintetica.

    - ``frobenius`` (por regimen): ``d(C_r, C_s) = ||C_r - C_s||_F / sqrt(d(d-1))``,
      es decir, la raiz del error cuadratico medio de las ``d(d-1)/2``
      correlaciones de Pearson (pares i < j; los pares NaN, p. ej. una columna
      constante, se omiten). ``C_s`` con todas las trayectorias apiladas.
    - ``salto_correlacion`` (``regimen='todos'``): la misma distancia entre los
      saltos ``J = C_crisis - C_calma`` real y sintetico (el hecho estilizado
      "las correlaciones cambian en crisis").

    Son distancias: ``real`` = mediana de la distancia del real a sus remuestreos
    (bloques en calma, episodios en crisis; para el salto se remuestrean ambos
    regimenes), ``banda_inf = 0``, ``banda_sup`` = p95, cociente NaN.
    """
    X, reg, trays, columnas = _real_y_sintetico(real, sintetico, regimen_real, columnas)
    S, reg_s, _ = _apilar(trays)
    boot = _remuestreos(reg, n_boot, semilla, largo_bloque)
    C_r, C_s, C_b = {}, {}, {}
    filas = []
    for k in sorted(boot):
        C_r[k] = _corr(X[reg == k])
        C_s[k] = _corr(S[reg_s == k]) if (reg_s == k).sum() > 2 else np.full_like(C_r[k], np.nan)
        C_b[k] = [_corr(X[idx]) for idx, _ in boot[k]]
        med, p95 = _p95([_dist_corr(C_r[k], C) for C in C_b[k]])
        filas.append(_fila(_nombre_regimen(k), SIN_COLUMNA, SIN_COLUMNA, "frobenius", med,
                           _dist_corr(C_r[k], C_s[k]), 0.0, p95))
    if 0 in C_r and 1 in C_r:
        J_r = C_r[1] - C_r[0]
        med, p95 = _p95([_dist_corr(J_r, c1 - c0) for c0, c1 in zip(C_b[0], C_b[1])])
        filas.append(_fila("todos", SIN_COLUMNA, SIN_COLUMNA, "salto_correlacion", med,
                           _dist_corr(J_r, C_s[1] - C_s[0]), 0.0, p95))
    return _tabla(filas, n_boot=int(n_boot), largo_bloque=int(largo_bloque), columnas=columnas)


# --------------------------------------------------------------------------- regimenes


def _duraciones(reg: np.ndarray) -> dict[int, np.ndarray]:
    t = tramos(reg)
    return {k: np.array([b - a for a, b, kk in t if kk == k], dtype=float) for k in (0, 1)}


def fidelidad_regimenes(real_regimes: pd.Series, sintetico: list[pd.DataFrame]) -> pd.DataFrame:
    """Duraciones de racha y matriz de transicion real frente a sintetica.

    OJO, es un CONTROL y no discrimina entre generadores: en este diseno la
    cadena de regimenes la simula la base comun (``GeneradorBase.sample`` con
    ``simular_regimenes`` y la matriz de transicion de train) y es la misma ley
    para todos los generadores; el generador no decide el regimen. Sirve para
    comprobar que la cadena simulada reproduce la del real (fraccion de crisis,
    duraciones) y para leer el resto de metricas sabiendo cuantos dias de cada
    regimen hay.

    Metricas por regimen (``columna='—'``, sin bandas):

    - ``fraccion_dias``: fraccion de dias del regimen (sintetico: apilado).
    - ``duracion_media``, ``duracion_mediana``, ``duracion_p90``: de las rachas
      (en sesiones). Sintetico: rachas de todas las trayectorias juntas. Se
      incluyen las rachas censuradas por los extremos (primera y ultima de cada
      serie), igual en real y sintetico; acortan algo las duraciones
      sinteticas porque 2.520 sesiones cortan rachas de calma largas.
    - ``P_kk``: probabilidad de permanencia; real con ``matriz_transicion``,
      sintetico con los conteos de transiciones dentro de cada trayectoria.
    - ``ks_duraciones`` y ``ks_duraciones_p``: estadistico y p-valor KS de dos
      muestras entre duraciones reales y sinteticas (``real`` = NaN). Con 8
      (A) o 4 (B) episodios de crisis el p-valor tiene muy poca potencia.

    Cociente sint / real en fraccion, duraciones y ``P_kk``.
    """
    reg = np.asarray(real_regimes.to_numpy() if isinstance(real_regimes, pd.Series) else real_regimes).astype(int)
    regs_s = [r for _, r in separar_sintetico(sintetico)]
    P_r = matriz_transicion(reg, n=2)
    conteo = np.zeros((2, 2))
    for r in regs_s:
        if r.size > 1:
            np.add.at(conteo, (r[:-1], r[1:]), 1.0)
    total = conteo.sum(axis=1, keepdims=True)
    P_s = np.divide(conteo, total, out=np.full_like(conteo, np.nan), where=total > 0)
    dur_r = _duraciones(reg)
    dur_s = {k: np.concatenate([_duraciones(r)[k] for r in regs_s]) for k in (0, 1)}
    todo_s = np.concatenate(regs_s)
    filas = []
    for k in (0, 1):
        if not dur_r[k].size:
            continue
        nombre = _nombre_regimen(k)
        dr, ds = dur_r[k], dur_s[k]
        vals = {
            "fraccion_dias": (np.mean(reg == k), np.mean(todo_s == k)),
            "duracion_media": (dr.mean(), ds.mean() if ds.size else np.nan),
            "duracion_mediana": (np.median(dr), np.median(ds) if ds.size else np.nan),
            "duracion_p90": (np.percentile(dr, 90), np.percentile(ds, 90) if ds.size else np.nan),
            "P_kk": (P_r[k, k], P_s[k, k]),
        }
        for m, (r, s) in vals.items():
            filas.append(_fila(nombre, SIN_COLUMNA, SIN_COLUMNA, m, r, s, cociente=_cociente(s, r, 1e-12)))
        if ds.size:
            ks = ks_2samp(dr, ds)
            filas.append(_fila(nombre, SIN_COLUMNA, SIN_COLUMNA, "ks_duraciones", np.nan, ks.statistic))
            filas.append(_fila(nombre, SIN_COLUMNA, SIN_COLUMNA, "ks_duraciones_p", np.nan, ks.pvalue))
    return _tabla(filas, nota="control: la cadena la simula la base comun, igual para todos los generadores")


# --------------------------------------------------------------------------- condicionamiento


def _logistica():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def _auc_oof(X: np.ndarray, y: np.ndarray, grupos: np.ndarray, n_pliegues: int) -> float:
    """AUC de las probabilidades fuera de pliegue (StratifiedGroupKFold) de una logistica estandarizada.

    ``StratifiedGroupKFold`` (sin barajar: determinista) mantiene cada grupo
    entero en un pliegue y reparte los grupos para que cada pliegue tenga dias
    de ambos regimenes; el numero de pliegues se limita al de grupos de la clase
    con menos grupos. AUC unico sobre todas las probabilidades fuera de pliegue.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    if np.unique(y).size < 2:
        return np.nan
    n_pl = int(min(n_pliegues, *(np.unique(grupos[y == c]).size for c in np.unique(y))))
    if n_pl < 2:
        return np.nan
    prob = np.empty(y.size, dtype=float)
    for tr, te in StratifiedGroupKFold(n_splits=n_pl).split(X, y, grupos):
        if np.unique(y[tr]).size < 2:
            prob[te] = y[tr].mean()
            continue
        prob[te] = _logistica().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, prob))


def _auc_ajuste(X: np.ndarray, y: np.ndarray) -> float:
    """AUC dentro de muestra de la misma logistica (ajuste y evaluacion sobre todo)."""
    from sklearn.metrics import roc_auc_score

    if np.unique(y).size < 2:
        return np.nan
    return float(roc_auc_score(y, _logistica().fit(X, y).predict_proba(X)[:, 1]))


def condicionamiento(real: pd.DataFrame, sintetico: list[pd.DataFrame], *, regimen_real=None,
                     col_ret: str = "SP500_ret", n_pliegues: int = 5, semilla: int = 42,
                     n_boot: int = 200, largo_bloque: int = LARGO_BLOQUE,
                     columnas: list[str] | None = None) -> pd.DataFrame:
    """Cuanta senal de regimen llevan las features: separacion de volatilidad,
    deriva en crisis y AUC de un clasificador que predice el regimen del dia.

    Diagnostica el condicionamiento debil visto en el notebook 15 (cGAN,
    difusion y cVAE en la pista B: separacion 1,10-1,59 frente a 2,21 real).

    - ``separacion_vol`` (``regimen='todos'``, ``columna=col_ret``):
      ``sd(r | crisis) / sd(r | calma)``. Sintetico apilado. Banda: bootstrap del
      real (bloques en calma, episodios en crisis), percentiles 2,5-97,5.
    - ``media_ret_crisis_pb`` (``regimen='crisis'``): media de ``col_ret`` en
      crisis en puntos basicos (x 1e4). Misma banda; cociente si |real| >= 0,5 pb.
    - ``auc_regimen`` (``regimen='todos'``, ``columna='—'``): AUC ROC de una
      regresion logistica (``StandardScaler`` + ``LogisticRegression`` L2, C=1)
      que predice el regimen del dia con las features de ese dia (todas las
      ``columnas``, por defecto las del real), con probabilidades fuera de
      pliegue de ``StratifiedGroupKFold(n_pliegues)``: en el real los grupos son
      las RACHAS (el modelo se evalua en episodios que no vio); en el sintetico,
      las TRAYECTORIAS apiladas. Es lineal en las features: la senal entra por
      niveles (vol_z, drawdown, spreads...), no por cambios de varianza de una
      feature de media constante (eso lo mide ``separacion_vol``).
      ASIMETRIA: los episodios reales son heterogeneos (en la pista A el AUC
      por pliegue va de ~0,25 a ~0,85) y los sinteticos salen de una sola ley
      condicional, asi que un generador fiel da un AUC MAYOR que el real (un
      AUC sintetico cercano a 0,5 si delata que el regimen apenas condiciona
      las features). Por eso NO lleva cociente: se publica como diagnostico.
    - ``auc_regimen_ajuste``: la misma logistica ajustada y evaluada dentro de
      muestra, en el real y en el sintetico apilado. Mide lo mismo en los dos
      lados (separabilidad lineal de la ley conjunta; con 10-15 coeficientes y
      miles de dias el optimismo dentro de muestra es despreciable) y es la que
      lleva cociente sint / real. Sin banda.
    """
    X, reg, trays, columnas = _real_y_sintetico(real, sintetico, regimen_real, columnas)
    if col_ret not in columnas:
        raise KeyError(f"{col_ret!r} no esta en las columnas comparadas.")
    j = columnas.index(col_ret)
    S, reg_s, _ = _apilar(trays)

    def _sep(r: np.ndarray, rg: np.ndarray) -> float:
        a, b = r[rg == 1], r[rg == 0]
        if a.size < 2 or b.size < 2:
            return np.nan
        return float(np.std(a, ddof=1) / np.std(b, ddof=1))

    def _media_pb(r: np.ndarray) -> float:
        return float(1e4 * r.mean()) if r.size else np.nan

    r_real, r_sint = X[:, j], S[:, j]
    boot = _remuestreos(reg, n_boot, semilla, largo_bloque)
    filas = []
    if 0 in boot and 1 in boot:
        sep_b = []
        for (i0, _), (i1, _) in zip(boot[0], boot[1]):
            a, b = r_real[i1], r_real[i0]
            sep_b.append(np.std(a, ddof=1) / np.std(b, ddof=1))
        sr, ss = _sep(r_real, reg), _sep(r_sint, reg_s)
        inf, sup = _banda(sep_b)
        filas.append(_fila("todos", col_ret, _tipo(col_ret), "separacion_vol", sr, ss, inf, sup,
                           _cociente(ss, sr, 1e-12)))
    if 1 in boot:
        mr, ms = _media_pb(r_real[reg == 1]), _media_pb(r_sint[reg_s == 1])
        inf, sup = _banda([_media_pb(r_real[idx]) for idx, _ in boot[1]])
        filas.append(_fila("crisis", col_ret, _tipo(col_ret), "media_ret_crisis_pb", mr, ms, inf, sup,
                           _cociente(ms, mr, 0.5)))
    grupos_r = id_episodio(reg)
    grupos_s = np.concatenate([np.full(len(r), i) for i, (_, r) in enumerate(trays)])
    auc_r = _auc_oof(X, reg, grupos_r, n_pliegues)
    auc_s = _auc_oof(S, reg_s, grupos_s, n_pliegues)
    filas.append(_fila("todos", SIN_COLUMNA, SIN_COLUMNA, "auc_regimen", auc_r, auc_s))
    aj_r, aj_s = _auc_ajuste(X, reg), _auc_ajuste(S, reg_s)
    filas.append(_fila("todos", SIN_COLUMNA, SIN_COLUMNA, "auc_regimen_ajuste", aj_r, aj_s,
                       cociente=_cociente(aj_s, aj_r, 1e-12)))
    return _tabla(filas, n_boot=int(n_boot), largo_bloque=int(largo_bloque), n_pliegues=int(n_pliegues),
                  clasificador="StandardScaler + LogisticRegression (L2, C=1)")
