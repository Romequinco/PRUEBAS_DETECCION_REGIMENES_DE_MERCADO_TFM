"""Visibilidad: cuándo es público un dato y cuál es su ``fecha_publicacion``.

Reglas (cerradas):

- Momento de decisión: ``HORA_DECISION`` (16:30) en ``HUSO_DECISION`` (America/New_York, con DST)
  de cada día natural. Todos los días naturales son fechas de decisión posibles.
- Un CIERRE de mercado es visible a la hora de cierre local de la sesión + ``MARGEN_CIERRE``.
- Una PUBLICACIÓN es visible a su hora declarada (+ ``margen`` solo si se declara).
- ``fecha_publicacion`` = primera fecha de decisión cuyo momento es >= el instante de visibilidad
  (referencia vectorizada: ``contrato.primera_decision_vec``; las funciones de aquí deben coincidir
  con ella).

Las horas de cierre viven en ``config/mercados.yaml`` con vigencias (p. ej. Tokio cierra a las
15:30 desde el 2024-11-05). En los días de cierre anticipado se usa el cierre ordinario: es más
tarde que el real, luego conservador (nunca adelanta un dato). Las horas de los organismos que
publican (BLS, Fed, BCE...) viven en la sección ``publicadores`` de ``config/publicaciones.yaml``.

De ``contrato`` se importan las constantes de decisión y ``primera_decision_vec`` (la regla de
referencia, sin duplicarla); ``contrato`` solo importa este módulo de forma perezosa, dentro de
``validar_precios``: sin ciclos al importar.

Fechas con huso: aquí una sesión o un día de publicación con huso se toma como el día de SU reloj
local (la fecha de una sesión de Tokio es la de Tokio). Una fecha de DECISIÓN con huso, en cambio,
es un error (``timegate.fecha_decision``): su día solo tiene sentido en Nueva York.
"""
from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import yaml
from dateutil.easter import easter
from pandas.tseries.holiday import USFederalHolidayCalendar

from ingesta.mercados_series.contrato import (
    HORA_DECISION,
    HUSO_DECISION,
    METODOS_PUBLICACION,
    primera_decision_vec,
)

MARGEN_CIERRE = pd.Timedelta(minutes=30)
RUTA_MERCADOS = Path(__file__).resolve().parent / "config" / "mercados.yaml"
RUTA_PUBLICACIONES = Path(__file__).resolve().parent / "config" / "publicaciones.yaml"

FechaLike = date | str | pd.Timestamp

FIN_DEL_DIA = "24:00"
DIAS_PUBLICACION = ("mismo", "siguiente_habil", "lunes_siguiente")
CALENDARIOS_HABILES = ("semana", "us_federal", "target")

_CORTE_DECISION = pd.Timedelta(hours=HORA_DECISION.hour, minutes=HORA_DECISION.minute)
_PATRON_HORA = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
_CLAVES_MERCADO = {"nombre", "huso", "cierres", "fuente", "verificado"}
_CLAVES_MERCADO_OPC = {"notas"}
_CLAVES_PUBLICADOR = {"hora", "huso", "dia", "fuente", "verificado"}
_CLAVES_PUBLICADOR_OPC = {"habiles", "margen_min", "notas"}
_UTC = "datetime64[ns, UTC]"
_FECHA = "datetime64[ns]"

__all__ = [
    "HORA_DECISION", "HUSO_DECISION", "MARGEN_CIERRE", "RUTA_MERCADOS", "RUTA_PUBLICACIONES",
    "FIN_DEL_DIA", "DIAS_PUBLICACION", "CALENDARIOS_HABILES",
    "momento_decision", "primera_decision", "cargar_mercados", "hora_cierre", "disponible_cierre",
    "fecha_publicacion_cierre", "disponible_publicacion", "disponible_cierre_serie",
    "fecha_publicacion_cierre_serie", "cargar_publicaciones", "es_habil", "disponible_publicador",
]


# --- Utilidades ---------------------------------------------------------------------------------
def _dia(fecha: FechaLike) -> pd.Timestamp:
    """Día natural (naive, medianoche) de ``fecha``; con huso, el día de su reloj local."""
    t = pd.Timestamp(fecha)
    if pd.isna(t):
        raise ValueError("fecha nula")
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return t.normalize()


def _dias_vec(fechas: Any) -> pd.Series:
    """Versión vectorizada de ``_dia``: ``datetime64[ns]`` a medianoche; conserva índice y NaT."""
    s = fechas if isinstance(fechas, pd.Series) else pd.Series(fechas)
    s = pd.to_datetime(s)
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_localize(None)
    return s.dt.normalize().astype(_FECHA)


def _hora(valor: Any, donde: str) -> time | str:
    """``"HH:MM"`` → ``time``; ``"24:00"`` se conserva como texto (fin del día local)."""
    if isinstance(valor, time):
        return valor
    if isinstance(valor, int) and not isinstance(valor, bool):
        raise ValueError(f"{donde}: hora {valor!r} leída como entero; escríbela entre comillas (\"HH:MM\")")
    if not isinstance(valor, str):
        raise ValueError(f"{donde}: hora {valor!r} no es texto \"HH:MM\"")
    if valor == FIN_DEL_DIA:
        return FIN_DEL_DIA
    if not _PATRON_HORA.match(valor):
        raise ValueError(f"{donde}: hora {valor!r} no tiene formato HH:MM (00:00-23:59 o 24:00)")
    h, m = valor.split(":")
    return time(int(h), int(m))


def _desfase(hora: time | str) -> pd.Timedelta:
    """Desfase desde la medianoche local; ``24:00`` = un día."""
    if hora == FIN_DEL_DIA:
        return pd.Timedelta(days=1)
    return pd.Timedelta(hours=hora.hour, minutes=hora.minute)


def _fecha_cfg(valor: Any, donde: str) -> date:
    if isinstance(valor, datetime):
        return valor.date()
    if isinstance(valor, date):
        return valor
    if isinstance(valor, str):
        try:
            return date.fromisoformat(valor)
        except ValueError:
            pass
    raise ValueError(f"{donde}: fecha {valor!r} no tiene formato AAAA-MM-DD")


def _huso(valor: Any, donde: str) -> str:
    if not isinstance(valor, str) or not valor:
        raise ValueError(f"{donde}: huso {valor!r} no es un nombre IANA")
    try:
        ZoneInfo(valor)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"{donde}: huso {valor!r} no existe (zoneinfo)") from None
    return valor


def _localizar(local: Any, huso: str) -> Any:
    """Hora local naive (``Timestamp`` o ``DatetimeIndex``) → con huso; sin adivinar en el cambio de hora."""
    try:
        return local.tz_localize(huso, ambiguous="raise", nonexistent="raise")
    except Exception as e:  # pytz.InvalidTimeError (pandas 2) o ValueError (pandas 3)
        raise ValueError(f"hora local inexistente o ambigua en {huso} por el cambio de hora: {e}") from None


def _leer_yaml(ruta: Path) -> Any:
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _claves(entrada: Any, obligatorias: set[str], opcionales: set[str], donde: str) -> None:
    if not isinstance(entrada, dict):
        raise ValueError(f"{donde}: se esperaba un diccionario")
    faltan = sorted(obligatorias - set(entrada))
    sobran = sorted(set(entrada) - obligatorias - opcionales)
    if faltan:
        raise ValueError(f"{donde}: faltan claves {faltan}")
    if sobran:
        raise ValueError(f"{donde}: claves desconocidas {sobran}")


# --- Momento de decisión ------------------------------------------------------------------------
def momento_decision(fecha: FechaLike) -> pd.Timestamp:
    """Instante UTC de la decisión del día ``fecha``: 16:30 America/New_York (con DST).

    ``fecha`` es un día natural (se ignora cualquier hora). Ejemplos: 2024-01-15 → 21:30 UTC;
    2024-07-15 → 20:30 UTC. Devuelve ``pd.Timestamp`` con huso UTC.
    """
    local = _localizar(_dia(fecha) + _CORTE_DECISION, HUSO_DECISION)
    return local.tz_convert("UTC")


def primera_decision(instante_utc: pd.Timestamp) -> pd.Timestamp:
    """Primera fecha de decisión ``d`` con ``momento_decision(d) >= instante_utc``.

    Devuelve la fecha naive a medianoche (``pd.Timestamp``). Un instante sin huso es un error
    (``ValueError``). Debe coincidir con ``contrato.primera_decision_vec`` fila a fila. Un
    instante nulo devuelve ``NaT``.
    """
    t = pd.Timestamp(instante_utc)
    if pd.isna(t):
        return pd.NaT
    if t.tzinfo is None:
        raise ValueError(f"primera_decision necesita un instante con huso (UTC), no {t!r}")
    local = t.tz_convert(HUSO_DECISION).tz_localize(None)
    dia = local.normalize()
    return dia if local - dia <= _CORTE_DECISION else dia + pd.Timedelta(days=1)


# --- Mercados -----------------------------------------------------------------------------------
def _validar_mercados(crudo: Any, origen: str) -> dict[str, dict[str, Any]]:
    if not isinstance(crudo, dict) or not crudo:
        raise ValueError(f"{origen}: se esperaba un diccionario de mercados no vacío")
    out: dict[str, dict[str, Any]] = {}
    for clave, entrada in crudo.items():
        donde = f"{origen}: mercado {clave!r}"
        _claves(entrada, _CLAVES_MERCADO, _CLAVES_MERCADO_OPC, donde)
        if not isinstance(entrada["verificado"], bool):
            raise ValueError(f"{donde}: 'verificado' debe ser true o false")
        if not isinstance(entrada["fuente"], str) or not entrada["fuente"].strip():
            raise ValueError(f"{donde}: falta 'fuente'")
        huso = _huso(entrada["huso"], donde)
        tramos_crudos = entrada["cierres"]
        if not isinstance(tramos_crudos, list) or not tramos_crudos:
            raise ValueError(f"{donde}: 'cierres' debe ser una lista no vacía de tramos")
        tramos = []
        for i, tramo in enumerate(tramos_crudos):
            dt = f"{donde}, tramo {i}"
            _claves(tramo, {"desde", "hasta", "hora"}, set(), dt)
            desde = _fecha_cfg(tramo["desde"], dt)
            hasta = None if tramo["hasta"] is None else _fecha_cfg(tramo["hasta"], dt)
            if hasta is not None and hasta < desde:
                raise ValueError(f"{dt}: 'hasta' ({hasta}) anterior a 'desde' ({desde})")
            tramos.append({"desde": desde, "hasta": hasta, "hora": _hora(tramo["hora"], dt)})
        for i, (prev, sig) in enumerate(zip(tramos[:-1], tramos[1:], strict=True)):
            if prev["hasta"] is None:
                raise ValueError(f"{donde}: el tramo {i} es abierto (hasta: null) y no es el último")
            esperado = prev["hasta"] + timedelta(days=1)
            if sig["desde"] < esperado:
                raise ValueError(f"{donde}: los tramos {i} y {i + 1} se solapan (o no están ordenados)")
            if sig["desde"] > esperado:
                raise ValueError(f"{donde}: hueco entre los tramos {i} y {i + 1} "
                                 f"({esperado} a {sig['desde'] - timedelta(days=1)})")
        out[str(clave)] = {"nombre": entrada["nombre"], "huso": huso, "cierres": tramos,
                           "fuente": entrada["fuente"], "verificado": entrada["verificado"],
                           "notas": list(entrada.get("notas") or [])}
    return out


@lru_cache(maxsize=8)
def _mercados_cache(ruta: str, _mtime_ns: int) -> dict[str, dict[str, Any]]:
    return _validar_mercados(_leer_yaml(Path(ruta)), Path(ruta).name)


def cargar_mercados(ruta: Path | None = None) -> dict[str, dict[str, Any]]:
    """Lee ``config/mercados.yaml`` (o ``ruta``) y lo devuelve validado.

    Resultado: ``{clave: {nombre, huso, cierres: [{desde: date, hasta: date | None, hora: time
    o "24:00"}], fuente, verificado, notas}}``. Comprueba que el huso existe (``zoneinfo``), que
    cada ``hora`` es ``HH:MM`` (``24:00`` = fin del día local, p. ej. cripto en UTC) y que las
    vigencias de ``cierres`` no se solapan ni dejan huecos. Lanza ``ValueError`` con la clave
    culpable. Devuelve una copia (la lectura se cachea por ruta y fecha de modificación).
    """
    r = Path(ruta) if ruta is not None else RUTA_MERCADOS
    return copy.deepcopy(_mercados_cache(str(r.resolve()), r.stat().st_mtime_ns))


def _mercados(mercados: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """``mercados`` dado o el YAML por defecto (cacheado, solo lectura)."""
    if mercados is not None:
        return mercados
    return _mercados_cache(str(RUTA_MERCADOS), RUTA_MERCADOS.stat().st_mtime_ns)


def _info_mercado(mercados: dict[str, dict[str, Any]], mercado: str) -> dict[str, Any]:
    try:
        return mercados[mercado]
    except KeyError:
        raise KeyError(f"mercado desconocido {mercado!r}; definidos: {sorted(mercados)}") from None


def _tramos(info: dict[str, Any], mercado: str) -> list[tuple[date, date | None, time | str]]:
    """Tramos normalizados (tolera un ``mercados`` escrito a mano con textos)."""
    donde = f"mercado {mercado!r}"
    return [(_fecha_cfg(t["desde"], donde), None if t["hasta"] is None else _fecha_cfg(t["hasta"], donde),
             _hora(t["hora"], donde)) for t in info["cierres"]]


def hora_cierre(mercado: str, sesion: FechaLike,
                mercados: dict[str, dict[str, Any]] | None = None) -> tuple[time | str, str]:
    """Hora de cierre local y huso de ``mercado`` vigentes en ``sesion``: ``(hora, huso)``.

    Elige el tramo de ``cierres`` con ``desde <= sesion <= hasta`` (``hasta`` nulo = abierto).
    ``hora`` puede ser ``"24:00"`` (fin del día). ``KeyError`` si el mercado no existe;
    ``ValueError`` si ningún tramo cubre la sesión. ``mercados`` por defecto: ``cargar_mercados()``.
    """
    info = _info_mercado(_mercados(mercados), mercado)
    dia = _dia(sesion).date()
    for desde, hasta, hora in _tramos(info, mercado):
        if desde <= dia and (hasta is None or dia <= hasta):
            return hora, info["huso"]
    raise ValueError(f"ningún tramo de cierres de {mercado!r} cubre la sesión {dia}")


def disponible_cierre(sesion: FechaLike, mercado: str,
                      mercados: dict[str, dict[str, Any]] | None = None) -> pd.Timestamp:
    """Instante UTC en que el cierre de ``sesion`` es visible: cierre local + ``MARGEN_CIERRE``.

    La hora local se convierte con el huso del mercado en esa fecha (DST de cada huso, no el de
    Nueva York). Ejemplo: XLON 2024-03-15 cierra a las 16:30 GMT → visible a las 17:00 UTC.
    """
    hora, huso = hora_cierre(mercado, sesion, mercados)
    local = _localizar(_dia(sesion) + _desfase(hora), huso)
    return local.tz_convert("UTC") + MARGEN_CIERRE


def fecha_publicacion_cierre(sesion: FechaLike, mercado: str,
                             mercados: dict[str, dict[str, Any]] | None = None) -> pd.Timestamp:
    """``primera_decision(disponible_cierre(sesion, mercado))``.

    XNYS (16:00 + 30 min = 16:30 NY) es visible el mismo día; CBOE_IDX (16:15), ICE_DXY y CME
    (17:00 NY) y CRYPTO (vela diaria UTC) al día siguiente; Europa y Asia, el mismo día.
    """
    return primera_decision(disponible_cierre(sesion, mercado, mercados))


def _mercados_alineados(mercado: Any, dias: pd.Series) -> pd.Series:
    """``mercado`` escalar o alineado con ``dias`` → Series posicional (RangeIndex)."""
    n = len(dias)
    if isinstance(mercado, str):
        return pd.Series([mercado] * n, dtype=object)
    if isinstance(mercado, pd.Series):
        if not mercado.index.equals(dias.index):
            raise ValueError("'mercado' debe tener el mismo índice que 'sesiones'")
        valores = mercado.to_numpy(dtype=object)
    else:
        valores = pd.Series(list(mercado), dtype=object).to_numpy()
        if len(valores) != n:
            raise ValueError(f"'mercado' tiene {len(valores)} elementos y 'sesiones' {n}")
    return pd.Series(valores, dtype=object)


def disponible_cierre_serie(sesiones: pd.Series, mercado: str | pd.Series,
                            mercados: dict[str, dict[str, Any]] | None = None) -> pd.Series:
    """Versión vectorizada de ``disponible_cierre``; ``mercado`` escalar o alineado con ``sesiones``.

    Devuelve ``datetime64[ns, UTC]`` con el mismo índice que ``sesiones``; respeta vigencias y DST.
    Las sesiones nulas dan ``NaT``. Un ``mercado`` Series debe tener el mismo índice que
    ``sesiones``; una lista o array se alinea por posición.
    """
    cfg = _mercados(mercados)
    dias_orig = _dias_vec(sesiones)
    dias = dias_orig.reset_index(drop=True)
    claves = _mercados_alineados(mercado, dias_orig)
    validos = dias.notna()
    if (validos & claves.isna()).any():
        raise ValueError("hay sesiones con 'mercado' nulo")
    res = pd.Series(pd.NaT, index=dias.index, dtype=_UTC)
    for clave in pd.unique(claves[validos]):
        info = _info_mercado(cfg, clave)
        sel = validos & (claves == clave)
        cubierto = pd.Series(False, index=dias.index)
        for desde, hasta, hora in _tramos(info, clave):
            m = sel & (dias >= pd.Timestamp(desde))
            if hasta is not None:
                m &= dias <= pd.Timestamp(hasta)
            if not m.any():
                continue
            local = _localizar(pd.DatetimeIndex(dias[m] + _desfase(hora)), info["huso"])
            res.loc[m] = pd.Series(local.tz_convert("UTC") + MARGEN_CIERRE, index=dias.index[m])
            cubierto |= m
        sin_tramo = sel & ~cubierto
        if sin_tramo.any():
            ejemplo = dias[sin_tramo].iloc[0].date()
            raise ValueError(f"ningún tramo de cierres de {clave!r} cubre {int(sin_tramo.sum())} "
                             f"sesiones (p. ej. {ejemplo})")
    res.index = dias_orig.index
    return res


def fecha_publicacion_cierre_serie(sesiones: pd.Series, mercado: str | pd.Series,
                                   mercados: dict[str, dict[str, Any]] | None = None) -> pd.Series:
    """Versión vectorizada de ``fecha_publicacion_cierre`` (``datetime64[ns]`` a medianoche)."""
    return primera_decision_vec(disponible_cierre_serie(sesiones, mercado, mercados))


# --- Publicaciones ------------------------------------------------------------------------------
def disponible_publicacion(fecha_local: FechaLike, hora_local: time | str, huso: str,
                           margen: pd.Timedelta = pd.Timedelta(0)) -> pd.Timestamp:
    """Instante UTC de una publicación declarada a ``hora_local`` en ``huso`` el día ``fecha_local``.

    Sin margen salvo que se declare. Una hora inexistente o ambigua por cambio de hora es un
    ``ValueError`` (no se adivina). Ejemplo: BCE 2024-03-15 16:00 Europe/Berlin → 15:00 UTC.
    ``hora_local`` admite ``time``, ``"HH:MM"`` o ``"24:00"``.
    """
    hora = _hora(hora_local, "disponible_publicacion")
    local = _localizar(_dia(fecha_local) + _desfase(hora), _huso(huso, "disponible_publicacion"))
    return local.tz_convert("UTC") + pd.Timedelta(margen)


_CLAVES_SERIE = {"fuente", "metodo", "verificado"}
_CLAVES_SERIE_OPC = {"publicador", "hora_local", "huso", "margen_min", "lag", "mercado", "componentes",
                     "calendario", "notas", "revisable"}
_PATRON_LAG = re.compile(r"\d+[DM]")


def _validar_serie_cfg(entrada: Any, donde: str, publicadores: dict[str, Any]) -> dict[str, Any]:
    """Una entrada de ``series``: claves conocidas y los campos que exige su método."""
    _claves(entrada, _CLAVES_SERIE, _CLAVES_SERIE_OPC, donde)
    metodo = entrada["metodo"]
    if metodo not in METODOS_PUBLICACION:
        raise ValueError(f"{donde}: metodo {metodo!r} no es uno de {list(METODOS_PUBLICACION)}")
    if not isinstance(entrada["verificado"], bool):
        raise ValueError(f"{donde}: 'verificado' debe ser true o false")
    if "revisable" in entrada and not isinstance(entrada["revisable"], bool):
        raise ValueError(f"{donde}: 'revisable' debe ser true o false")
    if entrada.get("hora_local") is not None:
        _hora(entrada["hora_local"], donde)
        _huso(entrada.get("huso"), donde)
    pub = entrada.get("publicador")
    if pub is not None and pub not in publicadores:
        raise ValueError(f"{donde}: publicador {pub!r} no está en 'publicadores'")
    if metodo == "lag_fijo" and not (isinstance(entrada.get("lag"), str) and _PATRON_LAG.fullmatch(entrada["lag"])):
        raise ValueError(f"{donde}: lag_fijo necesita 'lag' como \"<n>D\" o \"<n>M\" (desde el fin del periodo)")
    if metodo == "cierre_mercado":
        mercado = entrada.get("mercado")
        if mercado not in _mercados(None):
            raise ValueError(f"{donde}: cierre_mercado necesita 'mercado' definido en mercados.yaml "
                             f"(encontrado {mercado!r})")
    if metodo == "derivada":
        comp = entrada.get("componentes")
        if not (isinstance(comp, list) and comp and all(isinstance(c, str) and c for c in comp)):
            raise ValueError(f"{donde}: derivada necesita 'componentes' (lista no vacía de ids)")
    return dict(entrada)


def _validar_cierres(crudo: Any, origen: str) -> dict[str, list[dict[str, Any]]]:
    """``cierres_extraordinarios``: ``{calendario: [{fecha, motivo}]}`` con calendarios conocidos."""
    if crudo is None:
        return {}
    if not isinstance(crudo, dict):
        raise ValueError(f"{origen}: 'cierres_extraordinarios' debe ser {{calendario: [{{fecha, motivo}}]}}")
    out: dict[str, list[dict[str, Any]]] = {}
    for calendario, lista in crudo.items():
        donde = f"{origen}: cierres_extraordinarios {calendario!r}"
        if calendario not in ("us_federal", "target"):
            raise ValueError(f"{donde}: calendario desconocido (us_federal o target)")
        if not isinstance(lista, list):
            raise ValueError(f"{donde}: se esperaba una lista")
        out[calendario] = []
        for i, e in enumerate(lista):
            _claves(e, {"fecha", "motivo"}, set(), f"{donde}, entrada {i}")
            out[calendario].append({"fecha": _fecha_cfg(e["fecha"], f"{donde}, entrada {i}"),
                                    "motivo": str(e["motivo"])})
    return out


def _validar_publicaciones(crudo: Any, origen: str) -> dict[str, Any]:
    if not isinstance(crudo, dict):
        raise ValueError(f"{origen}: se esperaba un diccionario")
    _claves(crudo, {"alias", "publicadores", "series"}, {"cierres_extraordinarios"}, origen)
    alias = crudo["alias"] or {}
    if not isinstance(alias, dict) or not all(isinstance(k, str) and isinstance(x, str)
                                              for k, x in alias.items()):
        raise ValueError(f"{origen}: 'alias' debe ser {{alias: id canónico}} con textos")
    publicadores: dict[str, dict[str, Any]] = {}
    for clave, entrada in (crudo["publicadores"] or {}).items():
        donde = f"{origen}: publicador {clave!r}"
        _claves(entrada, _CLAVES_PUBLICADOR, _CLAVES_PUBLICADOR_OPC, donde)
        if entrada["dia"] not in DIAS_PUBLICACION:
            raise ValueError(f"{donde}: dia {entrada['dia']!r} no es uno de {list(DIAS_PUBLICACION)}")
        habiles = entrada.get("habiles")
        if entrada["dia"] != "mismo" and habiles not in CALENDARIOS_HABILES:
            raise ValueError(f"{donde}: dia {entrada['dia']!r} necesita 'habiles' "
                             f"en {list(CALENDARIOS_HABILES)}")
        if habiles is not None and habiles not in CALENDARIOS_HABILES:
            raise ValueError(f"{donde}: habiles {habiles!r} no es uno de {list(CALENDARIOS_HABILES)}")
        margen = entrada.get("margen_min", 0)
        if not isinstance(margen, int) or isinstance(margen, bool) or margen < 0:
            raise ValueError(f"{donde}: 'margen_min' debe ser un entero >= 0")
        if not isinstance(entrada["verificado"], bool):
            raise ValueError(f"{donde}: 'verificado' debe ser true o false")
        if not isinstance(entrada["fuente"], str) or not entrada["fuente"].strip():
            raise ValueError(f"{donde}: falta 'fuente'")
        publicadores[str(clave)] = {
            "hora": _hora(entrada["hora"], donde), "huso": _huso(entrada["huso"], donde),
            "dia": entrada["dia"], "habiles": habiles, "margen_min": margen,
            "fuente": entrada["fuente"], "verificado": entrada["verificado"],
            "notas": list(entrada.get("notas") or [])}
    series = {str(clave): _validar_serie_cfg(entrada, f"{origen}: serie {clave!r}", publicadores)
              for clave, entrada in (crudo["series"] or {}).items()}
    cierres = _validar_cierres(crudo.get("cierres_extraordinarios"), origen)
    return {"alias": dict(alias), "publicadores": publicadores, "series": series,
            "cierres_extraordinarios": cierres}


@lru_cache(maxsize=8)
def _publicaciones_cache(ruta: str, _mtime_ns: int) -> dict[str, Any]:
    return _validar_publicaciones(_leer_yaml(Path(ruta)), Path(ruta).name)


def cargar_publicaciones(ruta: Path | None = None) -> dict[str, Any]:
    """Lee ``config/publicaciones.yaml`` (o ``ruta``) y lo devuelve validado.

    Resultado: ``{"alias": {...}, "publicadores": {clave: {hora, huso, dia, habiles, margen_min,
    fuente, verificado, notas}}, "series": {...}, "cierres_extraordinarios": {calendario:
    [{fecha: date, motivo}]}}``. Comprueba horas ``HH:MM``, husos, ``dia`` (``mismo``,
    ``siguiente_habil``, ``lunes_siguiente``), el calendario de hábiles cuando hace falta, las
    claves de cada serie y los campos que exige su método (``lag`` en ``lag_fijo``, ``mercado``
    de ``mercados.yaml`` en ``cierre_mercado``, ``componentes`` en ``derivada``), que su
    ``publicador`` exista y las fechas de los cierres extraordinarios. ``ValueError`` con la clave.
    """
    r = Path(ruta) if ruta is not None else RUTA_PUBLICACIONES
    return copy.deepcopy(_publicaciones_cache(str(r.resolve()), r.stat().st_mtime_ns))


@lru_cache(maxsize=256)
def _festivos(calendario: str, anio: int) -> frozenset[date]:
    if calendario == "semana":
        return frozenset()
    if calendario == "us_federal":
        dias = USFederalHolidayCalendar().holidays(f"{anio}-01-01", f"{anio}-12-31")
        return frozenset(d.date() for d in dias)
    if calendario == "target":
        pascua = easter(anio)
        return frozenset({date(anio, 1, 1), pascua - timedelta(days=2), pascua + timedelta(days=1),
                          date(anio, 5, 1), date(anio, 12, 25), date(anio, 12, 26)})
    raise ValueError(f"calendario de hábiles {calendario!r} no es uno de {list(CALENDARIOS_HABILES)}")


def _publicaciones_defecto() -> dict[str, Any]:
    """``config/publicaciones.yaml`` validado (cacheado, solo lectura)."""
    return _publicaciones_cache(str(RUTA_PUBLICACIONES), RUTA_PUBLICACIONES.stat().st_mtime_ns)


def _cierres_extraordinarios(calendario: str, publicaciones: dict[str, Any] | None) -> frozenset[date]:
    cfg = publicaciones if publicaciones is not None else _publicaciones_defecto()
    return frozenset(e["fecha"] for e in (cfg.get("cierres_extraordinarios") or {}).get(calendario, []))


def es_habil(fecha: FechaLike, calendario: str, festivos: Iterable[FechaLike] = (),
             publicaciones: dict[str, Any] | None = None) -> bool:
    """True si ``fecha`` es día hábil en ``calendario`` (``semana``, ``us_federal`` o ``target``).

    Hábil = lunes a viernes, fuera de los festivos del calendario, de los cierres extraordinarios
    de ese calendario en ``publicaciones`` (por defecto ``config/publicaciones.yaml``: lutos
    nacionales, cierres por decreto, huracanes...) y de ``festivos`` (extra).
    ``us_federal``: festivos federales de EE. UU. (``USFederalHolidayCalendar`` de pandas), con
    los de sábado trasladados al viernes y los de domingo al lunes. La Fed abre ese viernes, así
    que el calendario es algo más estricto que el suyo (conservador: retrasa, nunca adelanta).
    ``target``: 1-ene, Viernes Santo, Lunes de Pascua, 1-may, 25 y 26-dic.
    """
    d = _dia(fecha).date()
    if d.weekday() >= 5 or d in _festivos(calendario, d.year):
        return False
    if d in _cierres_extraordinarios(calendario, publicaciones):
        return False
    return d not in {_dia(f).date() for f in festivos}


def _siguiente_habil(d: date, calendario: str, festivos: Iterable[FechaLike],
                     publicaciones: dict[str, Any] | None = None) -> date:
    """Primer hábil >= ``d``."""
    extra = list(festivos)
    for _ in range(31):
        if es_habil(d, calendario, extra, publicaciones):
            return d
        d += timedelta(days=1)
    raise ValueError(f"sin día hábil en 31 días desde {d} ({calendario})")


def disponible_publicador(clave: str, fecha: FechaLike, publicaciones: dict[str, Any] | None = None,
                          festivos: Iterable[FechaLike] = ()) -> pd.Timestamp:
    """Instante UTC en que sale el dato de ``fecha`` según el publicador ``clave``.

    ``fecha`` es el día de publicación del calendario (``dia: mismo``) o el día del dato
    (``siguiente_habil``: primer hábil posterior; ``lunes_siguiente``: el lunes posterior o, si es
    festivo, el siguiente hábil). Los hábiles siguen el calendario ``habiles`` del publicador, los
    ``cierres_extraordinarios`` de ``publicaciones`` y ``festivos`` (extra).
    ``KeyError`` si el publicador no existe.
    """
    cfg = publicaciones if publicaciones is not None else _publicaciones_defecto()
    try:
        pub = cfg["publicadores"][clave]
    except KeyError:
        definidos = sorted(cfg["publicadores"])
        raise KeyError(f"publicador desconocido {clave!r}; definidos: {definidos}") from None
    d = _dia(fecha).date()
    if pub["dia"] == "siguiente_habil":
        d = _siguiente_habil(d + timedelta(days=1), pub["habiles"], festivos, cfg)
    elif pub["dia"] == "lunes_siguiente":
        d = _siguiente_habil(d + timedelta(days=7 - d.weekday()), pub["habiles"], festivos, cfg)
    return disponible_publicacion(d, pub["hora"], pub["huso"], pd.Timedelta(minutes=pub["margen_min"]))
