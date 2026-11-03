"""``get_regimen(fecha)``: régimen de mercado publicado hasta ``fecha`` (PROVISIONAL).

Qué es
------
La tool de salida de la Fase 1 del TFM (ADR-005 §2.1): para una fecha devuelve
el estado de régimen, la probabilidad de crisis y los días que lleva en ese
estado, usando solo información publicada hasta esa fecha. Es **provisional**:
el detector final, su conjunto de estados y la regla congelada los fijará la
futura ADR-006 (notebook ``22_decision_final``) y los validará el pseudolive
(``23_pseudolive``). Mientras tanto el detector es intercambiable por
configuración (``configs/regimen.yaml``) o por parámetro.

Fuente de la señal
------------------
No se ajusta ni se recalcula nada: se lee el panel walk-forward OOS que el
benchmark v2 ya guardó para el detector, ``<directorio_benchmark>/panels/
{pista}_{detector}.parquet`` (misma convención que
``regimenes.benchmark.cache._panel_path``), con columnas ``state`` (etiqueta
canónica 0 = calma … ``n_states - 1`` = crisis), ``p_crisis`` y ``fold``. El
número de estados sale de ``metrics/{pista}_{detector}.csv`` (columna
``n_states``, igual que ``regimenes.fusion.crisis_state_from_metrics``), que
también aporta la huella de caché (``cache_fingerprint``). Los paneles se
cargan de forma perezosa y se guardan en memoria; la caché se invalida sola si
el fichero cambia (clave: ruta, tamaño y ``mtime``).

Por qué es causal
-----------------
``regimenes.evaluacion.walk_forward.walk_forward`` construye el panel así: en
cada fold ajusta un detector NUEVO solo con ``X.iloc[train_lo:t]`` (filas
anteriores al bloque), re-fija el orden económico de estados con los retornos
de ese mismo train, y predice el bloque ``[t, t + step)`` con
``predict_online`` (filtrado causal, no suavizado) y parámetros congelados. El
contexto de estado (ADR-003) es la cola del train, también anterior a ``t``.
Cada fecha aparece una sola vez, con la predicción del primer fold que la
cubrió; las re-predicciones no causales viven solo en
``.attrs['stability_panel']``, que no se guarda en el parquet ni se lee aquí.
Las features de entrada ya llevan los lags de publicación de ADR-003. Por
tanto la fila de la fecha ``t`` solo usa datos publicados hasta ``t``.
Ojo: esa fila **sí usa** ``X_t`` del propio día (la rentabilidad y los niveles
al cierre de ``t``), así que la respuesta para ``t`` es la que se conoce tras
el cierre de ``t``; quien decida durante la sesión ``t`` debe pedir ``t - 1``.

Sobre esa fila, esta tool solo añade cálculos que miran hacia atrás:
``dias_en_estado`` es la longitud de la racha que termina en ``t`` (incluido)
y la regla de pista depende de la primera fecha del OOS de B, que es pasado
para cualquier fecha a la que se aplica. Truncar el panel en ``t`` no cambia
ninguna respuesta para fechas ``<= t`` (lo comprueba ``tests/regimen``).

Contrato
--------
``get_regimen(fecha, *, pista=None, detector=None, directorio=None,
config=None) -> dict`` con las claves:

- ``fecha``: la pedida (ISO); ``fecha_dato``: la sesión usada (ISO).
- ``pista`` y ``detector`` efectivos.
- ``estado``: ``"calma"`` o ``"crisis"`` (provisional; ver abajo) y
  ``estado_detector``: la etiqueta canónica entera del panel.
- ``prob_crisis``: ``p_crisis`` del panel, o ``None`` si el detector no la da.
  Ojo: el estado sale de ``state``, no de umbralizar ``prob_crisis`` (los
  detectores con histéresis pueden seguir en crisis con probabilidad baja).
- ``dias_en_estado``: sesiones consecutivas (filas del panel, es decir, días
  hábiles de mercado) en el estado publicado, contando ``fecha_dato``; mínimo 1.
  ``dias_en_estado_censurado`` es ``True`` si la racha llega al inicio del panel
  (la cuenta real podría ser mayor). La racha no cruza de pista.
- ``estados_posibles``, ``provisional`` y ``fuente`` (ruta del panel, su SHA-256,
  la huella de caché del benchmark y la ruta de la configuración).
- ``contexto`` (opción 3 de CP2, 03-11-2026): segunda opinión y acuerdo del resto
  de detectores en ``fecha_dato``, o ``None`` si faltan sus ficheros o se pide
  ``contexto=False``. Claves:

  - ``prob_crisis_consenso``: probabilidad de crisis de la fusión online
    (ADR-011 §4, notebook 17), con pesos aprendidos solo con crisis terminadas;
    ``estado_consenso`` es ``crisis`` si ``>= umbral_consenso`` (0,5).
  - ``detectores_en_crisis``, ``detectores_total`` e ``ids_en_crisis``: cuántos
    candidatos no supervisados (15 en A, 17 en B) marcan crisis ese día.
  - ``confianza``: fracción de esos candidatos que dicen lo mismo que el
    principal (en crisis si el principal está en crisis; en calma si no). Es
    **acuerdo entre detectores**, no una probabilidad calibrada.
  - ``coincide_consenso`` y ``lectura``: ``crisis respaldada``, ``crisis sin
    respaldo``, ``calma con aviso del consenso`` o ``calma respaldada``, según
    coincidan el principal y el consenso.
  - ``fuente``: rutas y SHA-256 de los dos ficheros.

  Es causal por construcción: las señales son los paneles OOS de cada candidato
  y la fusión online solo usa pesos fijados con información anterior (lo
  comprueban ``tests/fusion``); aquí solo se lee la fila de ``fecha_dato``.

Fechas: fines de semana y festivos se resuelven al último día hábil del panel
``<= fecha``. Se lanza :class:`FechaFueraDeRango` si la fecha es anterior al
inicio del OOS de la pista, o posterior al último dato cuando entre ese último
dato y la fecha hay algún día laborable (lunes a viernes) del que aún no hay
dato; si solo median fines de semana se responde con el último dato.

Pista por fecha (si no se pasa ``pista``): la preferente (B) desde la primera
fecha de su panel OOS; antes, la de respaldo (A). Con ``pista`` explícita no se
aplica la regla y una fecha anterior a su OOS da error.

Estados provisionales: ``{calma, crisis}``. ``crisis`` es el estado canónico
``n_states - 1`` (``RegimeDetector.crisis_state``); el resto se publica como
``calma``. **Hueco documentado:** ``alerta`` (vigilancia de
``regimenes.fusion``) queda para la ADR-006; aquí no se fijan umbrales.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from regimenes import rutas

CONFIG_REGIMEN: Path = rutas.CONFIGS / "regimen.yaml"
ESTADOS_PROVISIONALES: tuple[str, ...] = ("calma", "crisis")
_CLAVES_CONFIG = ("directorio_benchmark", "detector_por_pista", "pista_preferente", "pista_respaldo")
# Columnas de fusiones en senales_{pista}.parquet de las fijas; el resto son los candidatos.
_PREFIJOS_FUSION = ("PAR_", "TRAS_", "CONF_", "VOTO_", "OR_", "AND_")
LECTURAS = {
    (True, True): "crisis respaldada",
    (True, False): "crisis sin respaldo",
    (False, True): "calma con aviso del consenso",
    (False, False): "calma respaldada",
}


class FechaFueraDeRango(ValueError):
    """La fecha pedida cae fuera del OOS disponible de la pista/detector."""


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
def cargar_config(config: Mapping[str, Any] | str | Path | None = None) -> dict:
    """Devuelve la configuración validada (por defecto ``configs/regimen.yaml``).

    Acepta un dict ya cargado (útil en tests) o la ruta de un yaml.
    """
    if config is None:
        config = CONFIG_REGIMEN
    if isinstance(config, (str, Path)):
        datos = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    else:
        datos = dict(config)
    faltan = [k for k in _CLAVES_CONFIG if k not in datos]
    if faltan:
        raise ValueError(f"Configuración de get_regimen incompleta: faltan {faltan}.")
    datos["detector_por_pista"] = {
        str(p).upper(): str(d).upper() for p, d in dict(datos["detector_por_pista"]).items()
    }
    datos["pista_preferente"] = str(datos["pista_preferente"]).upper()
    datos["pista_respaldo"] = str(datos["pista_respaldo"]).upper()
    estados = tuple(datos.get("estados", ESTADOS_PROVISIONALES))
    if estados != ESTADOS_PROVISIONALES:
        raise ValueError(
            f"Estados {estados} no soportados: la versión provisional solo publica "
            f"{ESTADOS_PROVISIONALES} (alerta queda para la ADR-006)."
        )
    datos["estados"] = estados
    return datos


def _ruta_config(config: Mapping[str, Any] | str | Path | None) -> str | None:
    if config is None:
        return _mostrar(CONFIG_REGIMEN)
    if isinstance(config, (str, Path)):
        return _mostrar(Path(config))
    return None  # dict en memoria


def _mostrar(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(rutas.ROOT).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


# --------------------------------------------------------------------------- #
# Paneles (carga perezosa + caché en memoria)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Panel:
    path: Path
    index: pd.DatetimeIndex
    state: np.ndarray            # etiqueta canónica entera
    p_crisis: np.ndarray | None  # None si el panel no la trae
    es_crisis: np.ndarray        # bool: state == crisis_state
    racha: np.ndarray            # longitud de la racha que termina en cada fila
    crisis_state: int
    sha256: str
    huella_cache: str | None


def _rachas(es_crisis: np.ndarray) -> np.ndarray:
    """Racha hacia atrás: ``r[i] = r[i-1] + 1`` si no cambia el estado, si no 1."""
    racha = np.ones(len(es_crisis), dtype=np.int64)
    for i in range(1, len(es_crisis)):
        if es_crisis[i] == es_crisis[i - 1]:
            racha[i] = racha[i - 1] + 1
    return racha


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _rutas_detector(directorio: Path, pista: str, detector: str) -> tuple[Path, Path]:
    # Misma convención que regimenes.benchmark.cache._panel_path / _metric_path.
    return (
        directorio / "panels" / f"{pista}_{detector}.parquet",
        directorio / "metrics" / f"{pista}_{detector}.csv",
    )


@lru_cache(maxsize=64)
def _cargar_panel_cacheado(
    panel_path: str, metric_path: str, _firma: tuple[int, int, int, int]
) -> _Panel:
    path = Path(panel_path)
    df = pd.read_parquet(path)
    if "state" not in df.columns:
        raise ValueError(f"{path}: falta la columna 'state'.")
    df.index = pd.to_datetime(df.index)
    if not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError(f"{path}: el índice debe ser temporal, único y ordenado.")
    if df["state"].isna().any():
        raise ValueError(f"{path}: 'state' contiene NaN.")

    metricas = pd.read_csv(metric_path)
    if len(metricas) != 1 or "n_states" not in metricas.columns:
        raise ValueError(f"{metric_path}: se esperaba una fila con 'n_states'.")
    crisis_state = int(metricas["n_states"].iloc[0]) - 1
    huella = None
    if "cache_fingerprint" in metricas.columns and pd.notna(metricas["cache_fingerprint"].iloc[0]):
        huella = str(metricas["cache_fingerprint"].iloc[0])

    state = df["state"].to_numpy(dtype=np.int64)
    p_crisis = df["p_crisis"].to_numpy(dtype=float) if "p_crisis" in df.columns else None
    es_crisis = state == crisis_state
    return _Panel(
        path=path,
        index=pd.DatetimeIndex(df.index).normalize(),
        state=state,
        p_crisis=p_crisis,
        es_crisis=es_crisis,
        racha=_rachas(es_crisis),
        crisis_state=crisis_state,
        sha256=_sha256(path),
        huella_cache=huella,
    )


def _cargar_panel(directorio: Path, pista: str, detector: str) -> _Panel:
    panel_path, metric_path = _rutas_detector(directorio, pista, detector)
    for p, que in ((panel_path, "panel walk-forward OOS"), (metric_path, "métricas del benchmark")):
        if not p.is_file():
            raise FileNotFoundError(
                f"No existe el {que} de {pista}/{detector}: {p}. "
                "Se genera con `python -m regimenes.benchmark --track A B`."
            )
    sp, sm = panel_path.stat(), metric_path.stat()
    firma = (sp.st_size, sp.st_mtime_ns, sm.st_size, sm.st_mtime_ns)
    return _cargar_panel_cacheado(str(panel_path.resolve()), str(metric_path.resolve()), firma)


def limpiar_cache() -> None:
    """Vacía la caché en memoria de paneles y contexto (p. ej. tras regenerar resultados)."""
    _cargar_panel_cacheado.cache_clear()
    _cargar_contexto_cacheado.cache_clear()


# --------------------------------------------------------------------------- #
# Contexto: consenso de detectores y fusión online (opción 3 de CP2)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Contexto:
    senales: pd.DataFrame   # bool, una columna por candidato
    prob: pd.Series         # probabilidad de crisis de la fusión online
    ruta_senales: Path
    ruta_prob: Path
    sha_senales: str
    sha_prob: str


@lru_cache(maxsize=8)
def _cargar_contexto_cacheado(
    ruta_senales: str, ruta_prob: str, columna: str, _firma: tuple[int, int, int, int]
) -> _Contexto:
    sen = pd.read_parquet(ruta_senales)
    sen = sen[[c for c in sen.columns if not str(c).startswith(_PREFIJOS_FUSION)]].astype(bool)
    sen.index = pd.DatetimeIndex(sen.index).normalize()
    pbar = pd.read_parquet(ruta_prob)
    if columna not in pbar.columns:
        raise ValueError(f"{ruta_prob}: falta la columna {columna!r} de la fusión online.")
    prob = pbar[columna].astype(float)
    prob.index = pd.DatetimeIndex(prob.index).normalize()
    return _Contexto(
        senales=sen,
        prob=prob,
        ruta_senales=Path(ruta_senales),
        ruta_prob=Path(ruta_prob),
        sha_senales=_sha256(Path(ruta_senales)),
        sha_prob=_sha256(Path(ruta_prob)),
    )


def _cargar_contexto(cfg: dict, pista: str) -> _Contexto | None:
    ctx = cfg.get("contexto")
    if not ctx:
        return None
    rutas_ctx = [
        rutas.ROOT / str(ctx[k]).format(pista=pista) for k in ("senales_detectores", "prob_consenso")
    ]
    if not all(r.is_file() for r in rutas_ctx):
        return None
    s1, s2 = (r.stat() for r in rutas_ctx)
    firma = (s1.st_size, s1.st_mtime_ns, s2.st_size, s2.st_mtime_ns)
    return _cargar_contexto_cacheado(
        str(rutas_ctx[0].resolve()),
        str(rutas_ctx[1].resolve()),
        str(ctx.get("columna_consenso", "ONLINE_principal_eta10")),
        firma,
    )


def _contexto(cfg: dict, pista: str, fecha_dato: pd.Timestamp, es_crisis: bool) -> dict | None:
    """Bloque ``contexto`` de la respuesta para la sesión ``fecha_dato`` (ver el módulo)."""
    ctx = _cargar_contexto(cfg, pista)
    if ctx is None or fecha_dato not in ctx.senales.index or fecha_dato not in ctx.prob.index:
        return None
    prob = float(ctx.prob.loc[fecha_dato])
    if not np.isfinite(prob):
        return None
    fila = ctx.senales.loc[fecha_dato]
    en_crisis = [str(c) for c in fila.index[fila.to_numpy()]]
    total = int(fila.size)
    coinciden = len(en_crisis) if es_crisis else total - len(en_crisis)
    consenso_crisis = bool(prob >= float(cfg["contexto"].get("umbral_consenso", 0.5)))
    return {
        "prob_crisis_consenso": prob,
        "estado_consenso": "crisis" if consenso_crisis else "calma",
        "detectores_en_crisis": len(en_crisis),
        "detectores_total": total,
        "ids_en_crisis": en_crisis,
        "confianza": coinciden / total if total else None,
        "coincide_consenso": consenso_crisis == es_crisis,
        "lectura": LECTURAS[(es_crisis, consenso_crisis)],
        "fuente": {
            "senales_detectores": _mostrar(ctx.ruta_senales),
            "sha256_senales": ctx.sha_senales,
            "prob_consenso": _mostrar(ctx.ruta_prob),
            "sha256_prob": ctx.sha_prob,
        },
    }


# --------------------------------------------------------------------------- #
# Resolución de fecha y pista
# --------------------------------------------------------------------------- #
def _como_fecha(fecha: Any) -> pd.Timestamp:
    ts = pd.Timestamp(fecha)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _posicion(panel: _Panel, ts: pd.Timestamp, pista: str, detector: str) -> int:
    """Fila del último día hábil del panel ``<= ts`` (con los controles de rango)."""
    primero, ultimo = panel.index[0], panel.index[-1]
    if ts < primero:
        raise FechaFueraDeRango(
            f"{ts.date()} es anterior al inicio del OOS de {pista}/{detector} "
            f"({primero.date()}): no hay predicción causal para esa fecha."
        )
    if ts > ultimo:
        laborables = pd.bdate_range(ultimo + pd.Timedelta(days=1), ts)
        if len(laborables):
            raise FechaFueraDeRango(
                f"{ts.date()} es posterior al último dato de {pista}/{detector} "
                f"({ultimo.date()}): faltan días laborables desde {laborables[0].date()}."
            )
    return int(panel.index.searchsorted(ts, side="right")) - 1


def _resolver_pista(
    ts: pd.Timestamp, cfg: dict, directorio: Path, detector: str | None
) -> str:
    pref, resp = cfg["pista_preferente"], cfg["pista_respaldo"]
    det_pref = detector or cfg["detector_por_pista"][pref]
    inicio_pref = _cargar_panel(directorio, pref, det_pref).index[0]
    return pref if ts >= inicio_pref else resp


# --------------------------------------------------------------------------- #
# Tool
# --------------------------------------------------------------------------- #
def get_regimen(
    fecha: Any,
    *,
    pista: str | None = None,
    detector: str | None = None,
    directorio: str | Path | None = None,
    config: Mapping[str, Any] | str | Path | None = None,
    contexto: bool = True,
) -> dict:
    """Régimen publicado hasta ``fecha`` según el panel walk-forward OOS del detector.

    Parameters
    ----------
    fecha : str | date | Timestamp
        Fecha consultada. Fines de semana/festivos → último día hábil ``<= fecha``.
    pista : {'A', 'B'} | None
        ``None`` aplica la regla por fecha (B desde el inicio de su OOS, A antes).
    detector : str | None
        Id del detector (p. ej. ``'D10'``); ``None`` usa el de la configuración
        para la pista resuelta.
    directorio : ruta | None
        Directorio del benchmark (con ``panels/`` y ``metrics/``); por defecto
        el de la configuración.
    config : dict | ruta | None
        Configuración; por defecto ``configs/regimen.yaml``.
    contexto : bool
        ``False`` omite el bloque de consenso (``contexto`` = ``None``).

    Returns
    -------
    dict con ``fecha``, ``fecha_dato``, ``pista``, ``detector``, ``estado``,
    ``estado_detector``, ``prob_crisis``, ``dias_en_estado``,
    ``dias_en_estado_censurado``, ``estados_posibles``, ``provisional``,
    ``fuente`` y ``contexto`` (ver el docstring del módulo).

    Raises
    ------
    FechaFueraDeRango
        Fecha anterior al OOS o posterior al último dato disponible.
    FileNotFoundError
        Si falta el panel o las métricas del detector pedido.
    """
    cfg = cargar_config(config)
    base = Path(directorio) if directorio is not None else rutas.ROOT / cfg["directorio_benchmark"]
    det = str(detector).upper() if detector is not None else None
    ts = _como_fecha(fecha)

    if pista is None:
        pista_ef = _resolver_pista(ts, cfg, base, det)
    else:
        pista_ef = str(pista).upper()
    if det is None:
        if pista_ef not in cfg["detector_por_pista"]:
            raise KeyError(f"La configuración no fija detector para la pista {pista_ef!r}.")
        det = cfg["detector_por_pista"][pista_ef]

    panel = _cargar_panel(base, pista_ef, det)
    i = _posicion(panel, ts, pista_ef, det)
    es_crisis = bool(panel.es_crisis[i])
    racha = int(panel.racha[i])
    prob = None
    if panel.p_crisis is not None and np.isfinite(panel.p_crisis[i]):
        prob = float(panel.p_crisis[i])

    return {
        "fecha": ts.date().isoformat(),
        "fecha_dato": panel.index[i].date().isoformat(),
        "pista": pista_ef,
        "detector": det,
        "estado": "crisis" if es_crisis else "calma",
        "estado_detector": int(panel.state[i]),
        "prob_crisis": prob,
        "dias_en_estado": racha,
        "dias_en_estado_censurado": racha == i + 1,
        "estados_posibles": list(cfg["estados"]),
        "provisional": bool(cfg.get("provisional", True)),
        "fuente": {
            "panel": _mostrar(panel.path),
            "sha256": panel.sha256,
            "huella_cache": panel.huella_cache,
            "config": _ruta_config(config),
        },
        "contexto": _contexto(cfg, pista_ef, panel.index[i], es_crisis) if contexto else None,
    }
