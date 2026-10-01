"""Utilidad: TSTR (train on synthetic, test on real; Esteban et al. 2017) frente a TRTR.

Pregunta: un detector del benchmark ajustado SOLO con trayectorias sinteticas,
detecta las crisis reales igual de bien que ajustado con el real?

Protocolo (todo dentro del **tramo real de entrenamiento**, ``<= fin_train``;
lo posterior queda reservado a los notebooks 17-20):

- **TRTR** (referencia, :func:`referencia_trtr`): ``walk_forward`` del detector
  sobre el real de train con el protocolo del benchmark (``train_size`` =
  ``min_train`` = ``DEFAULT_TRAIN_DAYS`` de la pista, ``step`` y ``expanding``
  del spec, contexto ``'auto'``, orden economico re-fijado en cada fold con
  ``SP500_ret`` del train). Se evalua sobre sus dias OOS con
  ``detection_summary`` + ``detection_score`` (ADR-003). Es la misma para todos
  los generadores: calcularla una vez y pasarla con ``trtr=``.
- **TSTR**: por trayectoria, ``spec.factory()`` -> ``fit`` con las features del
  spec en la trayectoria -> ``label_states_economically`` con el ``SP500_ret``
  SINTETICO (igual que el benchmark hace con el real en cada fold) ->
  ``predict_online`` sobre TODO el real de train (parametros congelados) y
  evaluacion en los MISMOS dias OOS del TRTR (comparacion pareada).
- **Contexto / burn-in**: D06 y D07 guardan los retornos de ajuste
  (``_train_returns``) y los anteponen como burn-in a las fechas ANTERIORES a la
  primera fecha predicha. Las fechas sinteticas son posteriores a ``fin_train``,
  asi que ese burn-in queda vacio: la prediccion arranca en frio en el primer
  dia real y nunca mezcla dias sinteticos con reales. Se exige (``ValueError``)
  que todas las fechas sinteticas sean posteriores a la ultima real. El arranque
  en frio no llega a los dias evaluados: entre el primer dia real y el primer
  dia OOS hay ``train_size`` sesiones reales de propagacion del estado (el
  TRTR propaga 252 sesiones de contexto; el TSTR, toda la historia real previa,
  que es causal porque solo usa dias reales anteriores con parametros fijos).
- **Nulo**: ``persistent_random_null`` (senal de Markov independiente de las
  crisis) sobre los mismos dias OOS con la tasa de marcado y la racha media de
  crisis de la senal TSTR (medianas entre trayectorias): asi se pregunta si el
  TSTR supera al azar CON SU PROPIA agresividad, que es lo que exige el umbral
  del yaml. Se publica tambien el p95 con la tasa/racha del TRTR (columna
  ``real``) como referencia.

Salida: contrato largo de ``fidelidad`` mas la columna ``detector``. Filas
(``regimen='todos'``, ``columna='—'``, ``tipo='—'``):

- ``score_deteccion``, ``det_event_recall``, ``det_precision``: ``real`` = TRTR;
  ``sintetico`` = mediana TSTR entre trayectorias; ``banda_inf``/``banda_sup`` =
  p10/p90 TSTR entre trayectorias; ``cociente`` = TSTR/TRTR (NaN si TRTR = 0).
  La precision es NaN en una trayectoria cuyo detector no marca nunca; la
  mediana y la banda la omiten (el score la cuenta como 0, como en el benchmark).
- ``score_nulo_p95``: ``real`` = p95 del nulo con la tasa/racha del TRTR;
  ``sintetico`` = p95 con la tasa/racha TSTR (contra este se compara el TSTR).

``en_banda`` queda NaN: el veredicto aplica ``tstr_margen_trtr`` del yaml. En
``attrs['por_trayectoria']`` va la tabla por trayectoria y en
``attrs['dias_evaluados']`` el indice OOS evaluado. Fila informativa
``score_trtr_fijo``: score del detector ajustado UNA vez con todo el real de train
(parametros congelados, como el TSTR). Para unir varios detectores, :func:`unir_utilidad`.

**Aviso de lectura**: el generador se ajusto con todo el real ``<= fin_train``,
asi que el TSTR es dentro de muestra respecto al generador (vio los dias
evaluados y sus etiquetas). Con detectores no supervisados el TSTR mide si la
ley marginal sintetica permite calibrar el detector, no si el sintetico
transmite la senal de crisis: un control negativo (filas reales iid, regimen sin
informacion) aprueba la regla en la pista A. Por eso es informativo (ver
``validacion.niveles`` en ``configs/sinteticos.yaml``).

Los imports de ``regimenes.evaluacion``/``regimenes.detectores``/``regimenes.benchmark``
son perezosos (importar ``regimenes.sinteticos`` no arrastra el benchmark).
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from regimenes.sinteticos.datos import COL_REGIMEN, COL_RET

METRICAS = ("score_deteccion", "det_event_recall", "det_precision")
COLUMNAS = ["detector", "regimen", "columna", "tipo", "metrica", "real", "sintetico",
            "banda_inf", "banda_sup", "cociente", "en_banda"]


# --------------------------------------------------------------------------- auxiliares

def _spec(pista: str, detector_id: str):
    from regimenes.detectores.registry import detector_specs

    for spec in detector_specs(pista):
        if spec.detector_id == detector_id.upper():
            return spec
    raise KeyError(f"No existe el detector {detector_id} en la pista {pista}.")


def _preparar_real(real: pd.DataFrame, fin_train) -> pd.DataFrame:
    """Panel real sin ``regime``, ordenado y recortado a ``<= fin_train`` si se da."""
    panel = real.drop(columns=[COL_REGIMEN], errors="ignore").sort_index()
    if fin_train is not None:
        panel = panel.loc[panel.index <= pd.Timestamp(fin_train)]
    if COL_RET not in panel.columns:
        raise ValueError(f"El panel real necesita {COL_RET!r} para el orden economico de estados.")
    return panel


def _comprobar_features(panel: pd.DataFrame, spec, que: str) -> None:
    faltan = [c for c in spec.features if c not in panel.columns]
    if faltan:
        raise ValueError(
            f"{spec.detector_id}: faltan las features {faltan} en {que} "
            f"(el detector usa {list(spec.features)})."
        )


def _ventanas_en(index: pd.DatetimeIndex, ventanas: dict) -> dict:
    """Ventanas de crisis que tienen algun dia dentro de ``[index[0], index[-1]]``."""
    ini, fin = index[0], index[-1]
    return {n: (a, b) for n, (a, b) in ventanas.items()
            if pd.Timestamp(b) >= ini and pd.Timestamp(a) <= fin}


def _resumen(flags: pd.Series, ventanas: dict, min_run: int, beta: float) -> dict:
    from regimenes.evaluacion.metricas import detection_summary
    from regimenes.evaluacion.ranking import detection_score

    res = detection_summary(flags, ventanas, min_run=min_run)
    # np.float64: con floats de Python, precision = recall = 0 lanzaria ZeroDivisionError
    # (detection_score cuenta con errstate de numpy y devuelve 0 en ese caso).
    res["score_deteccion"] = float(detection_score(np.float64(res["det_precision"]),
                                                   np.float64(res["det_event_recall"]), beta))
    return res


def _p95_nulo(index, ventanas, tasa, racha, n_sims, semilla, min_run, beta) -> float:
    from regimenes.evaluacion.ranking import persistent_random_null

    if not np.isfinite(tasa):
        return np.nan
    racha = float(racha) if np.isfinite(racha) else 1.0
    nulo = persistent_random_null(index, ventanas, rate=float(tasa), mean_run=racha,
                                  n_sims=n_sims, seed=semilla, min_run=min_run, beta=beta)
    return float(np.quantile(nulo["score_deteccion"], 0.95))


# --------------------------------------------------------------------------- TRTR

def referencia_trtr(real: pd.DataFrame, detector_id: str, *, pista: str = "A",
                    fin_train=None, train_days: int | None = None,
                    ventanas: dict | None = None) -> dict:
    """Walk-forward del benchmark sobre el real de train (referencia TRTR).

    Reproduce ``benchmark.ejecucion.run_one`` sin escribir nada ni tocar el
    estado global del juez (``configure_evaluation`` no hace falta: aqui solo se
    usan ``walk_forward`` y ``detection_summary`` con ventanas explicitas).
    Diferencia con el benchmark: el indice es el del panel de train (nucleo +
    ``SP500_ret`` sin NaN), no el indice comun de los 12 detectores, y termina en
    ``fin_train``.

    Returns
    -------
    dict con ``detector``, ``pista``, ``oos_index``, ``flags`` (Serie bool OOS),
    ``resumen`` (``det_*`` + ``score_deteccion``), ``ventanas`` (las evaluables),
    ``train_days``, ``segundos``.
    """
    from regimenes.benchmark.ejecucion import DEFAULT_TRAIN_DAYS
    from regimenes.evaluacion.ranking import DETECTION_DEFAULTS, track_crisis_windows
    from regimenes.evaluacion.walk_forward import walk_forward

    pista = pista.upper()
    spec = _spec(pista, detector_id)
    panel = _preparar_real(real, fin_train)
    _comprobar_features(panel, spec, "el panel real")
    min_train = int(train_days or DEFAULT_TRAIN_DAYS[pista])
    X = panel[list(spec.features)]
    if len(X) <= min_train:
        raise ValueError(f"{pista}/{spec.detector_id}: {len(X)} filas <= train inicial {min_train}.")
    market_returns = panel[COL_RET]
    t0 = time.perf_counter()
    wf = walk_forward(spec.factory, X, market_returns=market_returns, train_size=min_train,
                      min_train=min_train, step=spec.step, expanding=spec.expanding)
    segundos = time.perf_counter() - t0
    crisis = spec.factory().crisis_state
    flags = wf["state"].eq(crisis)
    oos = pd.DatetimeIndex(flags.index)
    ventanas = _ventanas_en(oos, ventanas if ventanas is not None else track_crisis_windows(pista))
    min_run, beta = int(DETECTION_DEFAULTS["min_run"]), float(DETECTION_DEFAULTS["beta"])
    resumen = _resumen(flags, ventanas, min_run, beta)
    # TRTR fijo (informativo): un solo ajuste con TODO el real de train y parametros
    # congelados, como el TSTR. Solo difiere del TSTR en los datos de ajuste, asi que
    # separa "sintetico frente a real" de "ajuste fijo frente a reajuste walk-forward".
    flags_fijo = _senal_fija(spec, X, market_returns, X, oos)
    resumen_fijo = _resumen(flags_fijo, ventanas, min_run, beta)
    return {"detector": spec.detector_id, "pista": pista, "oos_index": oos, "flags": flags,
            "resumen": resumen, "resumen_fijo": resumen_fijo, "ventanas": ventanas,
            "train_days": min_train, "segundos": segundos}


# --------------------------------------------------------------------------- TSTR

def _senal_fija(spec, X_ajuste: pd.DataFrame, ret_ajuste: pd.Series, X_real: pd.DataFrame,
                oos: pd.DatetimeIndex) -> pd.Series:
    """Un ajuste con parametros congelados y la senal de crisis en los dias OOS reales."""
    det = spec.factory()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*label_states_economically.*")
        warnings.filterwarnings("ignore", message=r".*market_returns.*")
        det.fit(X_ajuste)
    det.label_states_economically(X_ajuste, market_returns=ret_ajuste)
    estados = pd.Series(np.asarray(det.predict_online(X_real)), index=X_real.index)
    return estados.reindex(oos).eq(det.crisis_state)


def _senal_tstr(spec, tray: pd.DataFrame, X_real: pd.DataFrame, oos: pd.DatetimeIndex) -> pd.Series:
    """Ajusta en una trayectoria y devuelve la senal de crisis en los dias OOS reales."""
    return _senal_fija(spec, tray[list(spec.features)], tray[COL_RET], X_real, oos)


def utilidad_tstr(real: pd.DataFrame, sintetico: list[pd.DataFrame], detector_id: str, *,
                  pista: str = "A", regimen_real=None, n_trayectorias: int = 20,
                  semilla: int = 42, trtr: dict | None = None, fin_train=None,
                  train_days: int | None = None, ventanas: dict | None = None,
                  n_sims_nulo: int = 300) -> pd.DataFrame:
    """Detector ajustado en cada trayectoria sintetica y evaluado en el real de train.

    Parameters
    ----------
    real:
        Panel real de entrenamiento (``datos.cargar_entrenamiento``); puede
        traer la columna ``regime`` (se ignora). Debe incluir ``SP500_ret`` y
        las features del detector.
    sintetico:
        Trayectorias (``persistencia.cargar_trayectorias``), todas con fechas
        posteriores a la ultima fecha real.
    detector_id:
        ``'D01'``, ``'D02'``... (``detectores.registry.detector_specs(pista)``).
    regimen_real:
        No se usa (las crisis las fijan las ventanas del juez); se acepta por
        uniformidad con el resto de ``validacion``.
    n_trayectorias:
        Trayectorias evaluadas, sorteadas sin reemplazo con ``semilla`` (todas
        si hay menos).
    trtr:
        Salida de :func:`referencia_trtr` ya calculada (la misma para todos los
        generadores). ``None`` = se calcula aqui.
    fin_train:
        Si se da, ``real`` se recorta a ``<= fin_train`` antes de nada.
    train_days, ventanas:
        Solo para pruebas: train inicial del walk-forward (defecto
        ``DEFAULT_TRAIN_DAYS``) y ventanas de crisis (defecto, las de la pista).
    n_sims_nulo:
        Simulaciones del nulo persistente.
    """
    from regimenes.evaluacion.ranking import DETECTION_DEFAULTS

    del regimen_real
    pista = pista.upper()
    spec = _spec(pista, detector_id)
    panel = _preparar_real(real, fin_train)
    _comprobar_features(panel, spec, "el panel real")
    if not len(sintetico):
        raise ValueError("No hay trayectorias sinteticas.")
    ultimo_real = panel.index.max()
    for i, tray in enumerate(sintetico):
        _comprobar_features(tray, spec, f"la trayectoria sintetica {i}")
        if COL_RET not in tray.columns:
            raise ValueError(f"La trayectoria {i} no tiene {COL_RET!r} (orden economico de estados).")
        if pd.DatetimeIndex(tray.index).min() <= ultimo_real:
            raise ValueError(
                f"La trayectoria {i} tiene fechas <= {ultimo_real.date()} (ultima real): el burn-in "
                "de D06/D07 mezclaria dias sinteticos con reales."
            )

    if trtr is None:
        trtr = referencia_trtr(panel, spec.detector_id, pista=pista, train_days=train_days,
                               ventanas=ventanas)
    elif trtr["detector"] != spec.detector_id or trtr["pista"] != pista:
        raise ValueError(f"`trtr` es de {trtr['pista']}/{trtr['detector']}, no de {pista}/{spec.detector_id}.")
    oos = trtr["oos_index"]
    if oos.max() > ultimo_real or not oos.isin(panel.index).all():
        raise ValueError("Los dias OOS de `trtr` no pertenecen al panel real de train.")
    ventanas_ev = trtr["ventanas"]
    min_run = int(DETECTION_DEFAULTS["min_run"])
    beta = float(DETECTION_DEFAULTS["beta"])

    rng = np.random.default_rng(semilla)
    n = min(int(n_trayectorias), len(sintetico))
    elegidas = np.sort(rng.choice(len(sintetico), size=n, replace=False))
    X_real = panel[list(spec.features)]
    t0 = time.perf_counter()
    filas = []
    for k in elegidas:
        flags = _senal_tstr(spec, sintetico[k], X_real, oos)
        res = _resumen(flags, ventanas_ev, min_run, beta)
        filas.append({"trayectoria": int(k), **{m: res[m] for m in METRICAS},
                      "det_marked_rate": res["det_marked_rate"],
                      "det_mean_crisis_run": res["det_mean_crisis_run"]})
    por_tray = pd.DataFrame(filas)
    segundos = time.perf_counter() - t0

    ref = trtr["resumen"]
    p95_real = _p95_nulo(oos, ventanas_ev, ref["det_marked_rate"], ref["det_mean_crisis_run"],
                         n_sims_nulo, semilla, min_run, beta)
    p95_sint = _p95_nulo(oos, ventanas_ev, float(np.nanmedian(por_tray["det_marked_rate"])),
                         float(np.nanmedian(por_tray["det_mean_crisis_run"]))
                         if por_tray["det_mean_crisis_run"].notna().any() else np.nan,
                         n_sims_nulo, semilla, min_run, beta)

    salida = []
    for m in METRICAS:
        v = por_tray[m].to_numpy(dtype=float)
        hay = np.isfinite(v).any()
        sint = float(np.nanmedian(v)) if hay else np.nan
        r = float(ref[m]) if ref[m] is not None else np.nan
        salida.append({
            "metrica": m, "real": r, "sintetico": sint,
            "banda_inf": float(np.nanpercentile(v, 10)) if hay else np.nan,
            "banda_sup": float(np.nanpercentile(v, 90)) if hay else np.nan,
            "cociente": sint / r if np.isfinite(r) and r > 0 and np.isfinite(sint) else np.nan,
        })
    salida.append({"metrica": "score_nulo_p95", "real": p95_real, "sintetico": p95_sint,
                   "banda_inf": np.nan, "banda_sup": np.nan, "cociente": np.nan})
    fijo = trtr.get("resumen_fijo")
    if fijo is not None:
        sf = float(fijo["score_deteccion"])
        sint_score = salida[METRICAS.index("score_deteccion")]["sintetico"]
        salida.append({"metrica": "score_trtr_fijo", "real": sf, "sintetico": sint_score,
                       "banda_inf": np.nan, "banda_sup": np.nan,
                       "cociente": sint_score / sf if np.isfinite(sf) and sf > 0 and np.isfinite(sint_score) else np.nan})
    out = pd.DataFrame(salida)
    out["detector"] = spec.detector_id
    out["regimen"] = "todos"
    out["columna"] = "—"
    out["tipo"] = "—"
    out["en_banda"] = pd.array([pd.NA] * len(out), dtype="boolean")
    out = out[COLUMNAS]
    out.attrs["por_trayectoria"] = por_tray
    out.attrs["dias_evaluados"] = oos
    out.attrs["segundos_tstr"] = segundos
    out.attrs["segundos_trtr"] = trtr.get("segundos", np.nan)
    return out


def unir_utilidad(tablas: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Une salidas de :func:`utilidad_tstr` (varios detectores) para ``veredicto``.

    ``pd.concat`` falla con los ``attrs`` (guardan DataFrames); aqui se vacian y
    se devuelve aparte la tabla por trayectoria con la columna ``detector``.
    """
    if not tablas:
        raise ValueError("No hay tablas de utilidad.")
    limpias, por_tray = [], []
    for t in tablas:
        det = str(t["detector"].iloc[0])
        p = t.attrs.get("por_trayectoria")
        if p is not None:
            por_tray.append(p.assign(detector=det))
        c = t.copy()
        c.attrs = {}
        limpias.append(c)
    unida = pd.concat(limpias, ignore_index=True)
    return unida, (pd.concat(por_tray, ignore_index=True) if por_tray else pd.DataFrame())
