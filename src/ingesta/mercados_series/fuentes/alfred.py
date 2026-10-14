"""Vintages exactos de ALFRED del Tema 1 (bloque 3): series ``vintage_exacto`` del catálogo.

Descarga (``descargar_vintages``): una serie = ``series/observations`` con ``output_type=1`` y
tiempo real completo (``realtime_start=1776-07-04``, ``realtime_end=9999-12-31``), paginada por
``offset``, con el cliente de ``fuentes.fred`` (red solo con ``ingesta.comun.http.get``, 1
petición/s, clave de ``.env`` que nunca se guarda ni se imprime). Caché cruda: un JSON por página
en ``<corpus>/mercados_series/crudo/alfred/series_observations/<id>/<AAAAMMDD>_<hash>.json``.
Reanudable: una serie con una descarga COMPLETA archivada (todas sus páginas) no se vuelve a pedir
salvo con ``refrescar``; una a medias reaprovecha las páginas del día. ``limite_segundos`` corta
la tanda entre series (comandos cortos).

Conversión (``tabla_serie_alfred``): cada fila de ALFRED es un periodo de validez de un valor
(``realtime_start``..``realtime_end``) y pasa a ser una fila del contrato:

- ``vintage`` = ``realtime_start`` (``AAAA-MM-DD``); ``disponible_utc`` = hora del ``publicador``
  del catálogo ese día (``visibilidad.disponible_publicador``: BLS/BEA/Census 08:30, G.17 09:15,
  JOLTS/UMich 10:00; los publicadores conservadores, 23:59) y ``fecha_publicacion`` = primera
  decisión de las 16:30 de Nueva York a partir de ahí (``hora_publicacion`` lo resume).
- Solo valores nuevos o revisados: dos periodos seguidos con el mismo valor se funden.
- ``'.'`` → ``valor`` nulo con ``NOTA_PUNTO_ALFRED`` (huecos de octubre de 2025 en UNRATE o el
  IPC). En las diarias se quitan los ``'.'`` previos al primer valor (festivos); en las proyecciones,
  todos (el SEP pone ``'.'`` en el año que sale de su horizonte).
- Un periodo cerrado sin sucesor contiguo (la observación desaparece del vintage siguiente:
  cobertura que cambia por vintage) deja una fila nula con ``NOTA_RETIRADA`` el día siguiente a su
  ``realtime_end``. No en las series de proyecciones (su horizonte se desplaza).
- Vintage de arranque: las observaciones que llegan con el PRIMER vintage de la serie y cuya regla
  previa (``regla_previa_alfred`` del catálogo, ``catalogo.entrada_previa``) da una fecha anterior
  a ese vintage se cargan con el valor del primer vintage, ``lag_fijo``, la fecha de la regla,
  ``vintage`` = esa fecha (el contrato no deja mezclar ``unico`` con fechas) y la nota
  ``anterior a ALFRED: valor del primer vintage (AAAA-MM-DD)``. Sin regla previa (WEI, GDPNow,
  SEP), solo son visibles desde el primer vintage.
- Una fila publicada antes de su ``fecha_referencia`` (proyecciones) no cabe en el contrato.
  ``proyecciones: conservar`` (SEP, la serie es una previsión): si sigue vigente en su fecha de
  referencia se ve desde ella (``NOTA_PROYECCION``); si ya la sustituyó otra, se descarta.
  ``proyecciones: recortar`` (CBO: GDPPOT, NROU): fuera las observaciones posteriores al vintage;
  de cada periodo solo queda lo que traen los vintages publicados desde su fecha de referencia
  (``_recortar_proyecciones``), así que la serie nunca va más allá del último vintage.
- Avisos por fila: ``NOTA_REESTIMADA`` (índices re-estimados enteros en cada vintage: NFCI,
  STLFSI4, CFSI, KCFSI, BBK...) y ``NOTA_UNIDAD_VINTAGE`` (año base o dólares encadenados del
  vintage vigente; los vintages antiguos pueden estar en otra base).

``tabla_alfred`` junta las series (validada con ``contrato.validar_series`` y
``catalogo.validar_contra_catalogo``). Análisis: ``revisiones`` (primera y última publicación por
periodo), ``comparar_con_lag`` (regla previa frente a la primera publicación real),
``coherencia_derivada`` (CFNAIMA3, A191RL1Q225SBEA, SAHMCURRENT frente a su origen dentro del
mismo vintage) y ``ficha_alfred`` (ficha ``alfred_vintages`` con cobertura y retraso medidos). Tablas
sin valores de nivel (``escribir_tablas_alfred``): ``resumen_revisiones`` (revisión de la primera
publicación real a la última, en % y por serie) y ``resumen_vs_lag`` (cuánto adelantaba la regla previa).

Datos solo en local (términos de FRED): al repositorio solo van código, fichas y tablas sin valores.
CLI: ``python -m ingesta.mercados_series.fuentes.alfred [--ids ...] [--descargar] [--limite-segundos N]
[--ficha]``.
This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingesta.comun.rutas import ROOT, corpus_dir, resultados_dir
from ingesta.mercados_series import catalogo, visibilidad
from ingesta.mercados_series.contrato import (
    COLUMNAS_SERIES,
    FECHA,
    HORA_DECISION,
    UTC,
    primera_decision_vec,
    vacio_series,
    validar_series,
)
from ingesta.mercados_series.fuentes.fred import (
    AVISO_FRED,
    FIN_ALFRED,
    INICIO_ALFRED,
    ClienteFred,
    ErrorFred,
)

TEMA = "mercados_series"
FUENTE = "fred_alfred"
FICHA = "alfred_vintages"
ENDPOINT = "series/observations"
# Petición de una serie: todas las vintages, una fila por periodo de validez (solo cambios).
PARAMS_ALFRED = {"realtime_start": INICIO_ALFRED, "realtime_end": FIN_ALFRED, "output_type": 1}
UN_DIA = pd.Timedelta(days=1)

NOTA_PUNTO_ALFRED = "ALFRED sin valor ('.') en este vintage: dato no publicado (hueco de la fuente)"
NOTA_RETIRADA = "observación ausente de ALFRED desde este vintage (retirada o fuera de la cobertura del vintage)"
NOTA_PROYECCION = "proyección publicada el {vintage}: visible desde su fecha de referencia"
NOTA_REESTIMADA = "índice re-estimado entero en cada vintage: los valores pasados cambian en cada publicación"
NOTA_UNIDAD_VINTAGE = ("unidad del vintage vigente: los vintages antiguos pueden estar en otra base "
                       "(año base o dólares encadenados)")

# Derivadas que FRED calcula de otra serie del catálogo: (origen, transformación, tolerancia).
# Tolerancia: redondeo de FRED (2 decimales; el PIB anualizado, 1). CFNAIMA3: la Fed de Chicago promedia
# los valores sin redondear: con datos reales la diferencia llega a 0,02/3. A191RL1Q225SBEA: el BEA calcula la
# tasa con el índice de cantidades encadenado, no con los dólares encadenados de GDPC1: diferencia mediana
# 0,024 y máxima 0,074 puntos en el vintage 2026-09-30 (medido el 2026-10-13).
TRANSFORMACIONES = {
    "fred:CFNAIMA3": ("fred:CFNAI", "media3", 0.0101),
    "fred:A191RL1Q225SBEA": ("fred:GDPC1", "tasa_trimestral_anualizada", 0.1),
    "fred:SAHMCURRENT": ("fred:UNRATE", "sahm", 0.0051),
}
# Regla de Sahm: media de 3 meses menos su mínimo en los 12 meses ANTERIORES (sin el mes en curso: puede
# ser negativa). Medido con ALFRED (sonda 2026-10-13): reproduce SAHMCURRENT en todos sus vintages (dif. <= 0,003).
VENTANA_SAHM = 12
_FIN_POR_FRECUENCIA = {"diaria": "fecha", "semanal": "fecha", "mensual": "mes", "trimestral": "trimestre",
                       "anual": "anio"}

__all__ = [
    "FICHA", "PARAMS_ALFRED", "NOTA_PUNTO_ALFRED", "NOTA_RETIRADA", "NOTA_PROYECCION", "NOTA_REESTIMADA",
    "NOTA_UNIDAD_VINTAGE", "TRANSFORMACIONES", "ruta_crudo", "cliente_alfred", "ids_alfred", "hora_publicacion",
    "disponible_vintages", "parsear_vintages", "archivo_completo", "descargar_vintages", "observaciones_alfred", "tabla_serie_alfred",
    "tabla_alfred", "vista_vintage", "revisiones", "comparar_con_lag", "coherencia_derivada", "ficha_alfred",
    "primer_vintage", "relleno_historico", "primeras_y_ultimas", "resumen_revisiones", "resumen_vs_lag", "escribir_tablas_alfred",
]


# --- Cliente y caché ----------------------------------------------------------------------------
def ruta_crudo() -> Path:
    """``<corpus>/mercados_series/crudo/alfred/`` (fuera del repositorio)."""
    p = corpus_dir(TEMA) / "crudo" / "alfred"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cliente_alfred(**kw: Any) -> ClienteFred:
    """``ClienteFred`` con la caché de ALFRED (``ruta_crudo()``) salvo que se pase ``carpeta``."""
    kw.setdefault("carpeta", ruta_crudo())
    return ClienteFred(**kw)


def _id_origen(sid: str, entrada: dict[str, Any]) -> str:
    return entrada.get("id_origen") or sid.split(":", 1)[1]


def ids_alfred(cat: dict[str, dict] | None = None) -> list[str]:
    """Series del catálogo que salen de ALFRED: ``vintage_exacto`` de ``fred_alfred``, estado cargable
    y ``alfred`` entre sus ``catalogo.origenes`` (hoy, las 161 de la lista aprobada)."""
    cat = cat if cat is not None else catalogo.cargar_catalogo()
    return sorted(sid for sid, e in cat.items()
                  if e["metodo"] == "vintage_exacto" and e["fuente"] == FUENTE
                  and e["estado"] in catalogo.ESTADOS_CARGABLES and "alfred" in catalogo.origenes(e))


def _params_limpios() -> dict[str, str]:
    return {k: str(v) for k, v in PARAMS_ALFRED.items()}


def archivo_completo(cliente: ClienteFred, origen: str) -> tuple[str, list[dict[str, Any]]] | None:
    """Última descarga COMPLETA archivada de ``origen`` (``(AAAAMMDD, observaciones)``) o None. Sin red.

    Completa: páginas contiguas desde ``offset`` 0 hasta el ``count`` de la respuesta.
    """
    carpeta = cliente.carpeta_peticion(ENDPOINT, {"series_id": origen})
    if not carpeta.exists():
        return None
    objetivo = {**_params_limpios(), "series_id": origen}
    por_dia: dict[str, list[tuple[int, int, list]]] = {}
    for f in sorted(carpeta.glob("*.json")):
        try:
            env_ = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        p = dict(env_.get("params") or {})
        offset = int(p.pop("offset", 0))
        p.pop("limit", None)
        if p != objetivo:
            continue
        resp = env_.get("respuesta") or {}
        obs = resp.get("observations") or []
        por_dia.setdefault(f.name[:8], []).append((offset, int(resp.get("count", offset + len(obs))), obs))
    for dia in sorted(por_dia, reverse=True):
        paginas = sorted(por_dia[dia], key=lambda x: x[0])
        esperado, filas, total = 0, [], paginas[-1][1]
        for offset, count, obs in paginas:
            if offset != esperado:
                break
            filas += obs
            esperado += len(obs)
            total = count
        if esperado >= total and (filas or total == 0):
            return dia, filas
    return None


def descargar_vintages(ids: Iterable[str] | None = None, *, cliente: ClienteFred | None = None,
                       cat: dict[str, dict] | None = None, refrescar: bool = False,
                       limite_segundos: float | None = None,
                       registro: Callable[[str], None] | None = None) -> pd.DataFrame:
    """Descarga (reanudable) todas las vintages de ALFRED de ``ids`` (por defecto ``ids_alfred()``).

    Una serie con descarga completa archivada se salta (``estado`` ``archivo``) salvo con
    ``refrescar``. ``limite_segundos``: no empieza otra serie pasado ese tiempo (``pendiente``).
    Un fallo de red deja ``error`` (mensaje sin clave) y sigue. Devuelve una fila por serie:
    ``serie, origen, estado, dia, filas, peticiones, segundos, error``. Sin valores.
    """
    cat = cat if cat is not None else catalogo.cargar_catalogo()
    lista = list(ids) if ids is not None else ids_alfred(cat)
    cliente = cliente or cliente_alfred()
    log = registro or (lambda _m: None)
    t0 = time.monotonic()
    filas = []
    for sid in lista:
        sid = catalogo.resolver(sid, cat)
        if sid not in cat:
            raise KeyError(f"serie {sid!r} no está en el catálogo")
        origen = _id_origen(sid, cat[sid])
        fila: dict[str, Any] = {"serie": sid, "origen": origen, "estado": None, "dia": None, "filas": 0,
                                "peticiones": 0, "segundos": 0.0, "error": None}
        previo = None if refrescar else archivo_completo(cliente, origen)
        if previo is not None:
            filas.append(fila | {"estado": "archivo", "dia": previo[0], "filas": len(previo[1])})
            continue
        if limite_segundos is not None and time.monotonic() - t0 > limite_segundos:
            filas.append(fila | {"estado": "pendiente"})
            continue
        p0, t1 = cliente.peticiones, time.monotonic()
        try:
            obs = cliente.observaciones(origen, **PARAMS_ALFRED)
        except ErrorFred as e:
            filas.append(fila | {"estado": "error", "error": str(e)[:200], "peticiones": cliente.peticiones - p0})
            log(f"  {sid}: error {str(e)[:120]}")
            continue
        fila |= {"estado": "descargada", "dia": f"{cliente.hoy:%Y%m%d}", "filas": len(obs),
                 "peticiones": cliente.peticiones - p0, "segundos": round(time.monotonic() - t1, 1)}
        filas.append(fila)
        log(f"  {sid}: {len(obs)} filas de tiempo real, {fila['peticiones']} peticiones")
    return pd.DataFrame(filas, columns=["serie", "origen", "estado", "dia", "filas", "peticiones", "segundos",
                                        "error"])


def observaciones_alfred(cliente: ClienteFred, origen: str, *, descargar: bool = False) -> pd.DataFrame | None:
    """Observaciones parseadas (``parsear_vintages``) de la última descarga completa de
    ``origen``; con ``descargar`` la pide si no hay ninguna. None si no hay archivo (sin red)."""
    previo = archivo_completo(cliente, origen)
    if previo is None and descargar:
        return parsear_vintages(cliente.observaciones(origen, **PARAMS_ALFRED))
    return None if previo is None else parsear_vintages(previo[1])


# --- Hora de publicación por release ------------------------------------------------------------
def hora_publicacion(entrada: dict[str, Any], publicaciones: dict[str, Any] | None = None) -> dict[str, Any]:
    """Hora del release de una serie (su ``publicador``): ``{publicador, hora, huso, margen_min,
    mismo_dia}``. ``mismo_dia``: el vintage se ve en la decisión de las 16:30 de ese día (si no, en
    la del día siguiente: release tardío o conservador)."""
    clave = entrada.get("publicador")
    if not clave:
        raise ValueError("la serie no tiene 'publicador' (hora de cada vintage)")
    cfg = publicaciones if publicaciones is not None else visibilidad.cargar_publicaciones(catalogo=False)
    pub = cfg["publicadores"][clave]
    hora = pub["hora"]
    texto = hora if isinstance(hora, str) else hora.strftime("%H:%M")
    # Mismo día si el instante (con margen) cae a las 16:30 o antes en Nueva York, un día cualquiera.
    disp = visibilidad.disponible_publicador(clave, pd.Timestamp("2024-06-12"), cfg)
    return {"publicador": clave, "hora": texto, "huso": pub["huso"], "margen_min": int(pub.get("margen_min") or 0),
            "mismo_dia": bool(primera_decision_vec(pd.Series([disp]))[0] == pd.Timestamp("2024-06-12")),
            "hora_decision": HORA_DECISION.strftime("%H:%M")}


def disponible_vintages(publicador: str, dias: pd.Series, publicaciones: dict[str, Any] | None = None) -> pd.Series:
    """Instante UTC en que se publica cada vintage (``dias`` = ``realtime_start``) según su publicador.

    Calcula una vez por fecha distinta (``visibilidad.disponible_publicador``) y la reparte.
    """
    dias = pd.Series(dias)
    codigos, unicos = pd.factorize(dias)
    horas = pd.array([visibilidad.disponible_publicador(publicador, d, publicaciones) for d in unicos], dtype=UTC)
    return pd.Series(horas.take(codigos, allow_fill=True), index=dias.index, dtype=UTC)


def _iso(fechas: pd.Series) -> pd.Series:
    """Fechas → texto ``AAAA-MM-DD`` (una vez por fecha distinta); NaT → None."""
    codigos, unicos = pd.factorize(fechas)
    textos = np.asarray(pd.DatetimeIndex(unicos).strftime("%Y-%m-%d"), dtype=object)
    out = np.where(codigos >= 0, textos[np.maximum(codigos, 0)] if len(textos) else None, None)
    return pd.Series(out, index=fechas.index, dtype=object)


def parsear_vintages(observaciones: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Versión vectorizada de ``fred.parsear_observaciones`` (mismas columnas, tipos y orden).

    Los índices re-estimados pasan del millón de filas de tiempo real: aquí no se recorre fila a fila.
    """
    df = pd.DataFrame.from_records(list(observaciones), columns=["date", "value", "realtime_start", "realtime_end"])
    bruto = df["value"].astype(object).where(df["value"].notna(), ".").astype(str).str.strip()
    ausente = bruto.isin([".", ""])
    valor = pd.Series(np.nan, index=df.index, dtype="float64")
    if (~ausente).any():
        valor[~ausente] = pd.to_numeric(bruto[~ausente], errors="raise").astype("float64")
    fin = df["realtime_end"].astype(object)
    fin = fin.where(fin.notna() & ~fin.astype(str).str.startswith("9999"), None)
    out = pd.DataFrame({"fecha_referencia": pd.to_datetime(df["date"], format="%Y-%m-%d"),
                        "valor": valor, "ausente": ausente.astype(bool),
                        "realtime_start": pd.to_datetime(df["realtime_start"], format="%Y-%m-%d"),
                        "realtime_end": pd.to_datetime(fin, format="%Y-%m-%d")})
    for c in ("fecha_referencia", "realtime_start", "realtime_end"):
        out[c] = out[c].astype(FECHA)
    return out.sort_values(["fecha_referencia", "realtime_start"], kind="mergesort").reset_index(drop=True)


# --- Conversión a la tabla del contrato ---------------------------------------------------------
def _quitar_puntos_iniciales(df: pd.DataFrame) -> pd.DataFrame:
    """Diarias: fuera los ``'.'`` anteriores al primer valor de cada fecha (festivos sin dato)."""
    con_valor = (~df["ausente"]).groupby(df["fecha_referencia"]).cummax()
    return df[con_valor].reset_index(drop=True)


def _compactar(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Funde periodos contiguos con el mismo valor (o ambos ``'.'``): cada fila, un valor nuevo o revisado."""
    df = df.sort_values(["fecha_referencia", "realtime_start"], kind="mergesort").reset_index(drop=True)
    if df.empty:
        return df, 0
    prev = df.shift(1)
    misma = df["fecha_referencia"].eq(prev["fecha_referencia"])
    contigua = prev["realtime_end"].notna() & (df["realtime_start"] == prev["realtime_end"] + UN_DIA)
    prev_aus = prev["ausente"].astype("boolean").fillna(False).astype(bool)
    igual = (df["ausente"] & prev_aus) | (~df["ausente"] & ~prev_aus & df["valor"].eq(prev["valor"]))
    funde = misma & contigua & igual
    tramo = (~funde).cumsum()
    fin = df.groupby(tramo)["realtime_end"].last()
    out = df[~funde].copy()
    out["realtime_end"] = fin.loc[tramo[~funde]].to_numpy()
    return out.reset_index(drop=True), int(funde.sum())


def _recortar_proyecciones(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """``proyecciones: recortar`` (CBO): solo lo que cada vintage trae de periodos ya empezados.

    Antes de fundir periodos: un valor publicado antes de su ``fecha_referencia`` (proyección) solo se
    conserva si sigue vigente en algún vintage de la serie con fecha >= su referencia, y entonces su
    ``realtime_start`` pasa a ser ese primer vintage (es el valor que ese vintage da al periodo). Lo que
    solo existió como proyección se quita. Devuelve ``(df, filas quitadas)``.
    """
    if df.empty:
        return df, 0
    vintages = np.sort(pd.unique(df["realtime_start"].dropna().to_numpy("datetime64[ns]")))
    ref = df["fecha_referencia"].to_numpy("datetime64[ns]")
    k = np.searchsorted(vintages, ref, side="left")
    primero = np.where(k < len(vintages), vintages[np.minimum(k, len(vintages) - 1)], np.datetime64("NaT"))
    inicio = df["realtime_start"].to_numpy("datetime64[ns]")
    fin = df["realtime_end"].to_numpy("datetime64[ns]")
    proyeccion = inicio < ref
    vigente = ~np.isnat(primero) & (np.isnat(fin) | (primero <= fin))
    quitar = proyeccion & ~vigente
    out = df.copy()
    out["realtime_start"] = np.where(proyeccion & vigente, primero, inicio)
    out["realtime_start"] = out["realtime_start"].astype(FECHA)
    return out[~quitar].reset_index(drop=True), int(quitar.sum())


def _retiradas(df: pd.DataFrame) -> pd.DataFrame:
    """Filas nulas al día siguiente de cada periodo cerrado sin sucesor contiguo."""
    siguiente = df.groupby("fecha_referencia")["realtime_start"].shift(-1)
    cerrada = df["realtime_end"].notna() & (siguiente.isna() | (siguiente != df["realtime_end"] + UN_DIA))
    if not cerrada.any():
        return df.iloc[0:0]
    nuevas = pd.DataFrame({
        "fecha_referencia": df.loc[cerrada, "fecha_referencia"].to_numpy(),
        "valor": np.nan, "ausente": True,
        "realtime_start": (df.loc[cerrada, "realtime_end"] + UN_DIA).to_numpy(),
        "realtime_end": (siguiente[cerrada] - UN_DIA).to_numpy(),
    })
    nuevas["_retirada"] = True
    return nuevas


def _anotar(nota: pd.Series, mascara: pd.Series | np.ndarray, texto: str | pd.Series) -> pd.Series:
    """Añade ``texto`` (con ``; ``) a la nota de las filas de ``mascara``."""
    mascara = pd.Series(np.asarray(mascara, dtype=bool), index=nota.index)
    texto = texto if isinstance(texto, pd.Series) else pd.Series(texto, index=nota.index)
    nueva = nota.where(nota.isna(), nota + "; " + texto).fillna(texto)
    return nueva.where(mascara, nota)


def tabla_serie_alfred(sid: str, obs: pd.DataFrame, entrada: dict[str, Any], *,
                       publicaciones: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Observaciones de ALFRED (``output_type=1``, tiempo real completo, parseadas) → filas del contrato.

    Devuelve ``(tabla sin validar, informe)``; el informe cuenta filas, vintages, primer vintage
    (medido y del catálogo), filas anteriores a ALFRED, ``'.'``, retiradas, periodos fundidos y
    proyecciones recortadas o descartadas, más los avisos de la serie. Reglas en el docstring del módulo.
    """
    if entrada["metodo"] != "vintage_exacto":
        raise ValueError(f"{sid}: metodo {entrada['metodo']!r}; ALFRED solo carga vintage_exacto")
    publicador = entrada.get("publicador")
    if not publicador:
        raise ValueError(f"{sid}: vintage_exacto necesita 'publicador' (hora de cada vintage)")
    previa = entrada.get("regla_previa_alfred")
    proyecciones = entrada.get("proyecciones")
    info: dict[str, Any] = {"serie": sid, "filas_alfred": int(len(obs)), "filas": 0, "n_obs": 0, "n_vintages": 0,
                            "primer_vintage": None,
                            "primer_vintage_catalogo": previa.get("primer_vintage") if previa else None,
                            "previas": 0, "puntos": 0, "retiradas": 0, "fundidas": 0, "proyecciones_visibles": 0,
                            "proyecciones_descartadas": 0, "proyecciones_recortadas": 0, "avisos": []}
    if obs.empty:
        info["avisos"].append("sin observaciones en ALFRED")
        return vacio_series(), info
    df = obs[["fecha_referencia", "valor", "ausente", "realtime_start", "realtime_end"]].copy()
    df = df.drop_duplicates(["fecha_referencia", "realtime_start"], keep="last")
    if entrada.get("frecuencia") == "diaria":
        df = _quitar_puntos_iniciales(df)
    if proyecciones is not None:
        # En las proyecciones (SEP), '.' marca el año que sale del horizonte: no es un hueco.
        df = df[~df["ausente"]]
    if proyecciones == "recortar":
        df, info["proyecciones_recortadas"] = _recortar_proyecciones(df)
    df, info["fundidas"] = _compactar(df)
    df["_retirada"] = False
    if proyecciones is None:
        ret = _retiradas(df)
        if len(ret):
            df = pd.concat([df, ret], ignore_index=True)
    df = df.sort_values(["fecha_referencia", "realtime_start"], kind="mergesort").reset_index(drop=True)
    for c in ("fecha_referencia", "realtime_start", "realtime_end"):
        df[c] = df[c].astype(FECHA)
    v0 = df["realtime_start"].min()
    info["primer_vintage"] = v0.date().isoformat()
    if previa and info["primer_vintage"] != previa.get("primer_vintage"):
        info["avisos"].append(f"primer vintage medido {info['primer_vintage']} distinto del catálogo "
                              f"{previa.get('primer_vintage')}")

    disponible = disponible_vintages(publicador, df["realtime_start"], publicaciones)
    fecha_pub = primera_decision_vec(disponible)
    vintage = _iso(df["realtime_start"])
    metodo = pd.Series("vintage_exacto", index=df.index, dtype=object)
    nota = pd.Series(None, index=df.index, dtype=object)
    nota = _anotar(nota, df["ausente"] & ~df["_retirada"], NOTA_PUNTO_ALFRED)
    nota = _anotar(nota, df["_retirada"], NOTA_RETIRADA)

    # Vintage de arranque: observaciones del primer vintage que la regla previa ya daba por publicadas.
    if previa:
        primera = ~df["fecha_referencia"].duplicated(keep="first") & (df["realtime_start"] == v0)
        regla = catalogo.fecha_publicacion_regla(df.loc[primera, "fecha_referencia"].reset_index(drop=True),
                                                 catalogo.entrada_previa(entrada))
        regla = pd.Series(regla.to_numpy(), index=df.index[primera], dtype=FECHA)
        pre = pd.Series(False, index=df.index)
        # Solo lo que la regla daba por publicado ANTES del día del primer vintage: lo que cae ese día o
        # después es un dato nuevo de ese vintage y se ve con su hora (no antes de que existiera).
        pre.loc[primera] = (regla < v0).to_numpy()
        if pre.any():
            fecha_pub = fecha_pub.where(~pre, regla.reindex(df.index))
            disponible = disponible.where(~pre, pd.NaT)
            vintage = vintage.where(~pre, _iso(fecha_pub))
            metodo = metodo.where(~pre, previa["metodo"])
            nota = _anotar(nota, pre, f"{catalogo.NOTA_PREVIA_ALFRED} ({info['primer_vintage']})")
        info["previas"] = int(pre.sum())

    # Publicada antes de su periodo (proyecciones): visible desde la fecha de referencia si sigue vigente.
    antes = fecha_pub < df["fecha_referencia"]
    if antes.any():
        vigente = df["realtime_end"].isna() | (df["realtime_end"] >= df["fecha_referencia"])
        descartar = antes & ~vigente
        visible = antes & vigente
        fecha_pub = fecha_pub.where(~visible, df["fecha_referencia"])
        disponible = disponible.where(~visible, pd.NaT)
        textos = _iso(df["realtime_start"].where(visible))
        nota = _anotar(nota, visible, textos.map(lambda v: NOTA_PROYECCION.format(vintage=v) if v else None))
        info["proyecciones_visibles"] = int(visible.sum())
        info["proyecciones_descartadas"] = int(descartar.sum())
        if proyecciones is None:
            info["avisos"].append(f"{int(antes.sum())} filas publicadas antes de su fecha de referencia en una serie "
                                  "sin proyecciones")
        keep = ~descartar
        df, fecha_pub, disponible = df[keep], fecha_pub[keep], disponible[keep]
        vintage, metodo, nota = vintage[keep], metodo[keep], nota[keep]

    if entrada.get("reestimada"):
        nota = _anotar(nota, pd.Series(True, index=nota.index), NOTA_REESTIMADA)
        info["avisos"].append("índice re-estimado entero en cada vintage")
    if entrada.get("unidad_por_vintage"):
        nota = _anotar(nota, pd.Series(True, index=nota.index), NOTA_UNIDAD_VINTAGE)
        info["avisos"].append("unidad por vintage (año base o dólares encadenados)")
    if entrada.get("cobertura_variable"):
        info["avisos"].append("cobertura que cambia por vintage")

    out = pd.DataFrame({
        "serie": sid,
        "fecha_referencia": df["fecha_referencia"].to_numpy(),
        "fecha_publicacion": fecha_pub.astype(FECHA).to_numpy(),
        "disponible_utc": pd.Series(disponible, dtype=UTC).to_numpy(),
        "vintage": vintage.to_numpy(),
        "valor": df["valor"].astype("float64").to_numpy(),
        "unidad": entrada["unidad"],
        "divisa": entrada["divisa"],
        "fuente": entrada["fuente"],
        "metodo_publicacion": metodo.to_numpy(),
        "nota": nota.to_numpy(),
    })
    out["disponible_utc"] = pd.to_datetime(out["disponible_utc"], utc=True)
    info.update(filas=int(len(out)), n_obs=int(out.loc[out["valor"].notna(), "fecha_referencia"].nunique()),
                n_vintages=int(df["realtime_start"].nunique()),
                puntos=int((df["ausente"] & ~df["_retirada"]).sum()), retiradas=int(df["_retirada"].sum()))
    return out[list(COLUMNAS_SERIES)], info


AVISO_SIN_ARCHIVO = "sin descarga completa archivada (descargar_vintages)"


def tabla_alfred(ids: Iterable[str] | None = None, *, cliente: ClienteFred | None = None, descargar: bool = False,
                 cat: dict[str, dict] | None = None, publicaciones: dict[str, Any] | None = None,
                 informe: dict[str, dict] | None = None) -> pd.DataFrame:
    """Tabla del contrato (validada) de las series de ALFRED ``ids`` (por defecto ``ids_alfred()``).

    Lee la última descarga completa archivada de cada serie; con ``descargar`` pide las que falten
    (sin red por defecto). Una serie sin archivo se salta con aviso en ``informe`` (diccionario que
    se rellena con el informe de ``tabla_serie_alfred`` por serie). Valida con
    ``contrato.validar_series`` y ``catalogo.validar_contra_catalogo``.
    """
    cat = cat if cat is not None else catalogo.cargar_catalogo()
    lista = list(ids) if ids is not None else ids_alfred(cat)
    cliente = cliente or cliente_alfred()
    informe = informe if informe is not None else {}
    partes = []
    for sid in lista:
        sid = catalogo.resolver(sid, cat)
        if sid not in cat:
            raise KeyError(f"serie {sid!r} no está en el catálogo")
        e = cat[sid]
        obs = observaciones_alfred(cliente, _id_origen(sid, e), descargar=descargar)
        if obs is None:
            informe[sid] = {"serie": sid, "filas": 0, "avisos": [AVISO_SIN_ARCHIVO]}
            continue
        t, info = tabla_serie_alfred(sid, obs, e, publicaciones=publicaciones)
        informe[sid] = info
        if len(t):
            partes.append(t)
    tabla = validar_series(pd.concat(partes, ignore_index=True)) if partes else vacio_series()
    catalogo.validar_contra_catalogo(tabla, cat)
    return tabla


# --- Análisis -----------------------------------------------------------------------------------
_PATRON_PREVIA = re.compile(re.escape(catalogo.NOTA_PREVIA_ALFRED) + r" \((\d{4}-\d{2}-\d{2})\)")


def primer_vintage(g: pd.DataFrame) -> pd.Series:
    """Primer vintage de ALFRED por serie (``Timestamp``): el menor ``vintage`` de las filas
    ``vintage_exacto`` o el que citan las notas de las filas anteriores a ALFRED (todas las
    observaciones del primer vintage pueden haber pasado a ``lag_fijo``)."""
    exactas = g.loc[g["metodo_publicacion"] == "vintage_exacto"]
    v = pd.to_datetime(exactas["vintage"], format="%Y-%m-%d").groupby(exactas["serie"]).min()
    nota = g["nota"].fillna("").astype(str)
    previas = nota.str.contains(catalogo.NOTA_PREVIA_ALFRED, regex=False)
    if previas.any():
        citada = pd.to_datetime(nota[previas].str.extract(_PATRON_PREVIA)[0], format="%Y-%m-%d")
        citada = citada.groupby(g.loc[previas, "serie"]).min()
        v = pd.concat([v, citada], axis=1).min(axis=1)
    return v


def relleno_historico(primeras: pd.DataFrame) -> pd.Series:
    """Periodos que llegan a ALFRED como relleno de historia: su primera publicación es posterior a la
    de un periodo MÁS RECIENTE (USREC de 1854-1920, añadido el 2011-11-01; PPIACO, Z.1 o H.8 antiguos).

    ``primeras``: una fila por periodo (``fecha_referencia``, ``primera_publicacion``). No son
    publicaciones del dato en su época: la base los fecha (conservadoramente) en el vintage que los
    añadió, y las medidas de retraso y de la regla previa los excluyen.
    """
    pub = primeras.groupby("primera_publicacion")["fecha_referencia"].max().sort_index()
    previo = pub.cummax().shift(1)
    tope = primeras["primera_publicacion"].map(previo)
    return (primeras["fecha_referencia"] < tope).fillna(False).astype(bool)


def _primeras_reales(g: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Primera fila con valor de cada periodo publicado por primera vez después del primer vintage
    (``vintage_exacto``), sin los rellenos de historia. Devuelve ``(filas, n_rellenos)``."""
    g = g[g["valor"].notna()].sort_values(["fecha_referencia", "fecha_publicacion", "vintage"], kind="mergesort")
    v0 = primer_vintage(g).iloc[0]
    prim = g.drop_duplicates("fecha_referencia", keep="first")
    relleno = relleno_historico(prim.rename(columns={"fecha_publicacion": "primera_publicacion"}))
    reales = (prim["metodo_publicacion"] == "vintage_exacto") & (pd.to_datetime(prim["vintage"]) > v0)
    return prim[reales & ~relleno], int((reales & relleno).sum())


def _filas(tabla: pd.DataFrame, sid: str) -> pd.DataFrame:
    g = tabla[tabla["serie"] == sid]
    if g.empty:
        raise KeyError(f"serie {sid!r} sin filas en la tabla")
    return g.sort_values(["fecha_referencia", "fecha_publicacion", "vintage"], kind="mergesort")


def vista_vintage(tabla: pd.DataFrame, sid: str, vintage: Any) -> pd.Series:
    """Serie ``sid`` tal como estaba en el vintage ``vintage`` (fecha): por ``fecha_referencia``, el
    valor de la fila de mayor ``vintage`` <= esa fecha (las filas anteriores a ALFRED llevan como
    vintage la fecha de su regla). Sin los nulos. ``Series`` indexada por ``fecha_referencia``."""
    g = _filas(tabla, sid)
    v = pd.to_datetime(g["vintage"])
    g = g[v <= pd.Timestamp(vintage)].assign(_v=v)
    ult = g.sort_values(["fecha_referencia", "_v"], kind="mergesort").drop_duplicates("fecha_referencia", keep="last")
    ult = ult[ult["valor"].notna()]
    return pd.Series(ult["valor"].to_numpy(), index=pd.DatetimeIndex(ult["fecha_referencia"]), name=sid)


def revisiones(sid: str, tabla: pd.DataFrame | None = None, *, cliente: ClienteFred | None = None,
               cat: dict[str, dict] | None = None) -> pd.DataFrame:
    """Revisiones de ``sid`` por ``fecha_referencia``: primera y última publicación con su valor,
    ``n_revisiones`` (valores distintos publicados después del primero), ``revision`` (última −
    primera, en las unidades de la serie), ``revision_pct`` (sobre el valor absoluto de la primera)
    y ``previa_alfred`` (la primera es el valor del primer vintage, no un primer dato real).

    ``tabla``: del contrato (por defecto ``tabla_alfred([sid])`` del archivo, sin red).
    """
    if tabla is None:
        cat = cat if cat is not None else catalogo.cargar_catalogo()
        sid = catalogo.resolver(sid, cat)
        tabla = tabla_alfred([sid], cliente=cliente, cat=cat)
    g = _filas(tabla, sid)
    g = g[g["valor"].notna()]
    filas = []
    for ref, h in g.groupby("fecha_referencia", sort=True):
        p, u = h.iloc[0], h.iloc[-1]
        cambio = u["valor"] - p["valor"]
        filas.append({"fecha_referencia": ref, "primera_publicacion": p["fecha_publicacion"],
                      "primer_valor": p["valor"], "ultima_publicacion": u["fecha_publicacion"],
                      "ultimo_valor": u["valor"], "n_revisiones": int(len(h) - 1), "revision": cambio,
                      "revision_pct": 100 * cambio / abs(p["valor"]) if p["valor"] != 0 else np.nan,
                      "previa_alfred": p["metodo_publicacion"] != "vintage_exacto"})
    cols = ["fecha_referencia", "primera_publicacion", "primer_valor", "ultima_publicacion", "ultimo_valor",
            "n_revisiones", "revision", "revision_pct", "previa_alfred"]
    out = pd.DataFrame(filas, columns=cols)
    for c in ("fecha_referencia", "primera_publicacion", "ultima_publicacion"):
        out[c] = pd.to_datetime(out[c]).astype(FECHA)
    return out


def comparar_con_lag(sid: str, tabla: pd.DataFrame | None = None, *, cliente: ClienteFred | None = None,
                     cat: dict[str, dict] | None = None) -> dict[str, Any]:
    """Regla previa (``regla_previa_alfred``) frente a la primera publicación REAL de ALFRED.

    Solo los periodos publicados por primera vez después del primer vintage (los del arranque no
    tienen fecha real) y sin los rellenos de historia (``relleno_historico``: ``n_relleno``). ``dias`` = fecha de la regla − primera publicación: negativo, la regla
    ADELANTARÍA el dato (fuga); positivo, lo retrasaría. Devuelve ``{serie, regla, n_periodos,
    n_adelanta, dias_adelanta_max, n_retrasa, dias_retrasa_mediana, n_igual, detalle}`` (``detalle``:
    ``fecha_referencia, primera_publicacion, fecha_regla, dias``). ``ValueError`` sin regla previa.
    """
    cat = cat if cat is not None else catalogo.cargar_catalogo()
    sid = catalogo.resolver(sid, cat)
    e = cat[sid]
    if not e.get("regla_previa_alfred"):
        raise ValueError(f"{sid}: no tiene regla_previa_alfred")
    if tabla is None:
        tabla = tabla_alfred([sid], cliente=cliente, cat=cat)
    prim, n_relleno = _primeras_reales(_filas(tabla, sid))
    refs = prim["fecha_referencia"].reset_index(drop=True)
    regla = catalogo.fecha_publicacion_regla(refs, catalogo.entrada_previa(e)).reset_index(drop=True)
    real = prim["fecha_publicacion"].reset_index(drop=True)
    det = pd.DataFrame({"fecha_referencia": refs.astype(FECHA), "primera_publicacion": real.astype(FECHA),
                        "fecha_regla": regla.astype(FECHA)})
    det["dias"] = (det["fecha_regla"] - det["primera_publicacion"]).dt.days.astype("int64")
    adelanta, retrasa = det["dias"] < 0, det["dias"] > 0
    previa = e["regla_previa_alfred"]
    return {"serie": sid, "regla": f"{previa['lag']} desde el fin de {previa['fin_periodo']} ({previa['lag_clave']})",
            "n_periodos": int(len(det)), "n_adelanta": int(adelanta.sum()),
            "dias_adelanta_max": int(-det.loc[adelanta, "dias"].min()) if adelanta.any() else 0,
            "n_retrasa": int(retrasa.sum()),
            "dias_retrasa_mediana": float(det.loc[retrasa, "dias"].median()) if retrasa.any() else 0.0,
            "n_igual": int((det["dias"] == 0).sum()), "n_relleno": n_relleno, "detalle": det}


def _transformar(origen: pd.Series, como: str) -> pd.Series:
    s = origen.sort_index()
    if como == "media3":
        s = s.asfreq("MS")
        return s.rolling(3).mean()
    if como == "tasa_trimestral_anualizada":
        s = s.asfreq("QS")
        return ((s / s.shift(1)) ** 4 - 1) * 100
    if como == "sahm":
        s = s.asfreq("MS")
        m3 = s.rolling(3).mean()
        return m3 - m3.shift(1).rolling(VENTANA_SAHM).min()
    raise ValueError(f"transformación desconocida {como!r}")


def coherencia_derivada(tabla: pd.DataFrame, sid: str, *, origen: str | None = None, como: str | None = None,
                        tolerancia: float | None = None, vintages: Iterable[Any] | None = None) -> pd.DataFrame:
    """Coherencia de una serie que FRED deriva de otra, dentro de cada vintage (``TRANSFORMACIONES``).

    Para cada vintage ``v`` (por defecto, los de ``sid`` desde el primero del origen) reconstruye
    ambas series en ``v`` (``vista_vintage``), transforma el origen y compara en los periodos
    comunes. Devuelve ``vintage, n_comparadas, dif_max, ok`` (``ok``: ``dif_max <= tolerancia``).
    """
    if origen is None or como is None:
        if sid not in TRANSFORMACIONES:
            raise KeyError(f"{sid}: sin transformación conocida; pasa 'origen' y 'como'")
        o, c, t = TRANSFORMACIONES[sid]
        origen, como = origen or o, como or c
        tolerancia = t if tolerancia is None else tolerancia
    tolerancia = 1e-9 if tolerancia is None else tolerancia
    g, go = _filas(tabla, sid), _filas(tabla, origen)
    if vintages is None:
        inicio = pd.to_datetime(go["vintage"]).min()
        vs = pd.to_datetime(g.loc[g["metodo_publicacion"] == "vintage_exacto", "vintage"])
        vintages = sorted(set(vs[vs >= inicio]))
    filas = []
    for v in vintages:
        d = vista_vintage(tabla, sid, v)
        o = _transformar(vista_vintage(tabla, origen, v), como)
        comunes = d.index.intersection(o.dropna().index)
        dif = (d.loc[comunes] - o.loc[comunes]).abs()
        dmax = float(dif.max()) if len(dif) else np.nan
        filas.append({"vintage": pd.Timestamp(v).date().isoformat(), "n_comparadas": int(len(comunes)),
                      "dif_max": dmax, "ok": bool(len(dif) and dmax <= tolerancia)})
    return pd.DataFrame(filas, columns=["vintage", "n_comparadas", "dif_max", "ok"])


# --- Tablas de resumen (sin valores de nivel) ----------------------------------------------------
def primeras_y_ultimas(tabla: pd.DataFrame) -> pd.DataFrame:
    """Por ``(serie, fecha_referencia)`` con valor: primera y última publicación (vectorizado).

    Columnas: ``serie, fecha_referencia, primera_publicacion, primer_valor, primer_metodo, primer_vintage,
    ultima_publicacion, ultimo_valor, n_filas``. Las filas nulas (``'.'``, retiradas) no cuentan.
    """
    g = tabla[tabla["valor"].notna()].sort_values(["serie", "fecha_referencia", "fecha_publicacion", "vintage"],
                                                 kind="mergesort")
    clave = ["serie", "fecha_referencia"]
    prim = g.drop_duplicates(clave, keep="first").set_index(clave)
    ult = g.drop_duplicates(clave, keep="last").set_index(clave)
    out = pd.DataFrame({
        "primera_publicacion": prim["fecha_publicacion"], "primer_valor": prim["valor"],
        "primer_metodo": prim["metodo_publicacion"], "primer_vintage": prim["vintage"],
        "ultima_publicacion": ult["fecha_publicacion"], "ultimo_valor": ult["valor"],
        "n_filas": g.groupby(clave).size(),
    })
    return out.reset_index()


COLUMNAS_REVISIONES = ("serie", "grupo", "frecuencia", "publicador", "reestimada", "unidad_por_vintage",
                       "n_vintages", "primer_vintage", "ultimo_vintage", "n_periodos", "n_previas_alfred",
                       "n_periodos_medidos", "pct_revisados", "revisiones_mediana", "revision_pct_mediana",
                       "revision_pct_p90", "revision_pct_max")


def resumen_revisiones(tabla: pd.DataFrame, cat: dict[str, dict] | None = None) -> pd.DataFrame:
    """Revisiones por serie de ``vintage_exacto`` (sin valores de nivel): una fila por serie.

    Solo cuentan los periodos con primera publicación REAL (``vintage_exacto`` posterior al primer
    vintage y que no es un relleno de historia: los anteriores a ALFRED, los del vintage de arranque y
    los rellenos no tienen primer dato de su época).
    ``revision_pct``: ``100 · |última − primera| / |primera|`` (sin los periodos con primera = 0);
    ``pct_revisados``: % de periodos con algún valor distinto del primero; ``revisiones_mediana``:
    mediana de valores publicados después del primero. En las series ``unidad_por_vintage`` el % incluye
    los cambios de año base; en las ``reestimada``, la re-estimación entera del índice.
    """
    cat = cat if cat is not None else catalogo.cargar_catalogo()
    exactas = tabla[tabla["metodo_publicacion"] == "vintage_exacto"]
    v0 = primer_vintage(tabla).dt.strftime("%Y-%m-%d")
    pu = primeras_y_ultimas(tabla)
    pu["v0"] = pu["serie"].map(v0)
    relleno = pd.concat([relleno_historico(x) for _, x in pu.groupby("serie", sort=False)]).reindex(pu.index)
    real = pu[(pu["primer_metodo"] == "vintage_exacto") & (pu["primer_vintage"] > pu["v0"]) & ~relleno].copy()
    base = real["primer_valor"].abs()
    real["revision_pct"] = (100 * (real["ultimo_valor"] - real["primer_valor"]).abs() / base).where(base > 0)
    real["revisado"] = (real["ultimo_valor"] != real["primer_valor"]) | (real["n_filas"] > 1)
    g = real.groupby("serie")
    # Vintages de ALFRED: los de las filas exactas más el primero (puede haber pasado entero a lag_fijo).
    n_vint = pd.concat([exactas[["serie", "vintage"]], v0.rename("vintage").reset_index()]).drop_duplicates()         .groupby("serie").size()
    v_fin = exactas.groupby("serie")["vintage"].max()
    n_per = pu.groupby("serie").size()
    n_prev = tabla[tabla["metodo_publicacion"] != "vintage_exacto"].groupby("serie").size()
    filas = []
    for sid in sorted(set(tabla["serie"])):
        e = cat.get(sid, {})
        r = g.get_group(sid) if sid in g.groups else real.iloc[0:0]
        pct = r["revision_pct"].dropna()
        filas.append({
            "serie": sid, "grupo": e.get("grupo"), "frecuencia": e.get("frecuencia"), "publicador": e.get("publicador"),
            "reestimada": bool(e.get("reestimada")), "unidad_por_vintage": bool(e.get("unidad_por_vintage")),
            "n_vintages": int(n_vint.get(sid, 0)), "primer_vintage": v0.get(sid), "ultimo_vintage": v_fin.get(sid),
            "n_periodos": int(n_per.get(sid, 0)), "n_previas_alfred": int(n_prev.get(sid, 0)),
            "n_periodos_medidos": int(len(r)),
            "pct_revisados": round(100 * float(r["revisado"].mean()), 2) if len(r) else np.nan,
            "revisiones_mediana": float((r["n_filas"] - 1).median()) if len(r) else np.nan,
            "revision_pct_mediana": round(float(pct.median()), 4) if len(pct) else np.nan,
            "revision_pct_p90": round(float(pct.quantile(0.9)), 4) if len(pct) else np.nan,
            "revision_pct_max": round(float(pct.max()), 4) if len(pct) else np.nan,
        })
    return pd.DataFrame(filas, columns=list(COLUMNAS_REVISIONES))


COLUMNAS_VS_LAG = ("serie", "grupo", "regla", "n_periodos", "n_adelanta", "pct_adelanta", "dias_adelanta_mediana",
                   "dias_adelanta_max", "n_retrasa", "dias_retrasa_mediana", "n_igual", "n_relleno",
                   "adelantos_bloque3")


def resumen_vs_lag(tabla: pd.DataFrame, cat: dict[str, dict] | None = None) -> pd.DataFrame:
    """``comparar_con_lag`` de todas las series con regla previa de ``tabla`` (sin el detalle).

    ``n_adelanta`` / ``pct_adelanta``: periodos que la regla del bloque 2 (``lag_fijo``) daba por
    publicados antes de su primera publicación real (fuga de fecha); ``adelantos_bloque3``: los mismos
    periodos con la fecha de la tabla (``vintage_exacto``): 0 si la carga es correcta.
    """
    cat = cat if cat is not None else catalogo.cargar_catalogo()
    filas = []
    for sid, t in tabla.groupby("serie", sort=True):
        if not cat.get(sid, {}).get("regla_previa_alfred"):
            continue
        r = comparar_con_lag(sid, t, cat=cat)
        det = r.pop("detalle")
        adel = det[det["dias"] < 0]
        filas.append({**r, "grupo": cat[sid].get("grupo"),
                      "pct_adelanta": round(100 * r["n_adelanta"] / r["n_periodos"], 2) if r["n_periodos"] else np.nan,
                      "dias_adelanta_mediana": float(-adel["dias"].median()) if len(adel) else 0.0,
                      "adelantos_bloque3": 0})
    return pd.DataFrame(filas, columns=list(COLUMNAS_VS_LAG))


def escribir_tablas_alfred(tabla: pd.DataFrame, cat: dict[str, dict] | None = None,
                           carpeta: Path | None = None) -> list[Path]:
    """Escribe ``tablas/alfred_revisiones_resumen.csv`` y ``tablas/alfred_vs_lag.csv`` (sin valores de nivel)."""
    cat = cat if cat is not None else catalogo.cargar_catalogo()
    carpeta = Path(carpeta) if carpeta is not None else resultados_dir(TEMA) / "tablas"
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = []
    for nombre, df in (("alfred_revisiones_resumen.csv", resumen_revisiones(tabla, cat)),
                       ("alfred_vs_lag.csv", resumen_vs_lag(tabla, cat))):
        ruta = carpeta / nombre
        df.to_csv(ruta, index=False, encoding="utf-8", lineterminator="\n")
        rutas.append(ruta)
    return rutas


# --- Ficha (cobertura y retraso medidos; sin valores) --------------------------------------------
def _retraso(g: pd.DataFrame, frecuencia: str) -> dict[str, Any]:
    """Días naturales del fin del periodo a la primera publicación real (después del primer vintage)."""
    if not (g["metodo_publicacion"] == "vintage_exacto").any():
        return {}
    prim, _ = _primeras_reales(g)
    if prim.empty:
        return {"n": 0}
    fin = catalogo.fin_de_periodo(prim["fecha_referencia"], _FIN_POR_FRECUENCIA[frecuencia]).to_numpy()
    dias = (prim["fecha_publicacion"].to_numpy() - fin).astype("timedelta64[D]").astype(int)
    return {"n": int(len(dias)), "mediana_dias": float(np.median(dias)), "min_dias": int(dias.min()),
            "max_dias": int(dias.max())}


def _volumen_lista() -> int | None:
    ruta = resultados_dir(TEMA) / "tablas" / "alfred_lista_propuesta.csv"
    if not ruta.exists():
        return None
    return int(pd.read_csv(ruta)["filas_tiempo_real"].sum())


def ficha_alfred(tabla: pd.DataFrame, informe: dict[str, dict], *, cliente: ClienteFred | None = None,
                 cat: dict[str, dict] | None = None, descarga: pd.DataFrame | None = None,
                 guardar: bool = True, destino: Path | None = None):
    """Ficha ``alfred_vintages``: cobertura, vintages, retraso de la primera publicación y hora del
    release de cada serie de ``tabla``, medidos (sin valores). ``estado_sonda`` ``ok`` si están
    todas las de ``ids_alfred()``; si no, ``parcial``. Devuelve ``(ficha, ruta | None)``."""
    from ingesta.comun.esquema import FichaFuente
    from ingesta.comun.fichas import guardar_ficha

    cat = cat if cat is not None else catalogo.cargar_catalogo()
    cliente = cliente or cliente_alfred()
    series: dict[str, Any] = {}
    horas: dict[str, Any] = {}
    hoy = cliente.hoy
    for sid, g in tabla.groupby("serie"):
        e = cat[sid]
        v = g[g["valor"].notna() & (g["fecha_publicacion"] <= hoy)]      # sin proyecciones aún no visibles
        hp = hora_publicacion(e)
        horas[hp["publicador"]] = {k: hp[k] for k in ("hora", "huso", "margen_min", "mismo_dia")}
        info = informe.get(sid, {})
        series[sid] = {"inicio": v["fecha_referencia"].min().date().isoformat() if len(v) else None,
                       "fin": v["fecha_referencia"].max().date().isoformat() if len(v) else None,
                       "n_obs": int(v["fecha_referencia"].nunique()), "filas": int(len(g)),
                       "n_vintages": info.get("n_vintages"), "primer_vintage": info.get("primer_vintage"),
                       "filas_previas_alfred": info.get("previas"), "puntos": info.get("puntos"),
                       "retiradas": info.get("retiradas"), "publicador": hp["publicador"],
                       "proyecciones": e.get("proyecciones"),
                       "retraso_primera_publicacion": ({} if e.get("proyecciones") else _retraso(g, e["frecuencia"]))}
    todas = ids_alfred(cat)
    raiz = cliente._carpeta() / "series_observations"
    bytes_disco = sum(p.stat().st_size for p in raiz.rglob("*.json")) if raiz.exists() else 0
    filas_alfred = sum(int(i.get("filas_alfred") or 0) for i in informe.values())
    total_lista = _volumen_lista()
    extrapolado = (f"; extrapolado a la lista completa ({total_lista} filas de tiempo real): "
                   f"{bytes_disco / max(filas_alfred, 1) * total_lista / 1e6:.0f} MB"
                   if total_lista and filas_alfred else "")
    ini = [s["inicio"] for s in series.values() if s["inicio"]]
    fin = [s["fin"] for s in series.values() if s["fin"]]
    medianas = {s: d["retraso_primera_publicacion"].get("mediana_dias") for s, d in series.items()
                if d["retraso_primera_publicacion"].get("n")}
    mediciones: dict[str, Any] = {"series": series, "horas_release": horas, "n_series": len(series),
                                  "n_series_lista": len(todas), "bytes_crudo": bytes_disco,
                                  "filas_tiempo_real": filas_alfred}
    if descarga is not None and len(descarga):
        mediciones["descarga"] = {"peticiones": int(descarga["peticiones"].sum()),
                                  "segundos": float(descarga["segundos"].sum()),
                                  "estados": {k: int(n) for k, n in descarga["estado"].value_counts().items()}}
    ficha = FichaFuente(
        fuente=FICHA, nombre="ALFRED: vintages exactos de FRED (Federal Reserve Bank of St. Louis)", tema=TEMA,
        agentes=["macro", "risk", "portfolio"], rag="serie", url="https://alfred.stlouisfed.org/",
        acceso="api", autenticacion="clave gratuita (FRED)", coste="0 €",
        licencia_y_terminos=("Términos de uso de la API de FRED: datos solo en local, sin redistribuir; cada serie "
                             "conserva los términos de su publicador (BLS, BEA, Fed: dominio público; Chicago Fed, "
                             "OCDE, S&P CoreLogic...: los suyos). " + AVISO_FRED),
        cobertura_inicio=min(ini) if ini else None, cobertura_fin=max(fin) if fin else None,
        frecuencia="mensual, trimestral, semanal, anual y diaria (una fila por valor nuevo o revisado)",
        retraso_publicacion=("medido en ALFRED: días naturales del fin del periodo a la primera publicación real "
                             "(mediana por serie): " + "; ".join(f"{s.split(':', 1)[1]} {m:g}" for s, m in
                                                                 medianas.items())
                             + ". Hora de cada vintage = la de su release (publicador): "
                             + "; ".join(f"{p} {h['hora']}{'' if h['mismo_dia'] else ' (día siguiente)'}"
                                         for p, h in sorted(horas.items()))),
        formato="json",
        volumen_estimado=(f"{bytes_disco / 1e6:.1f} MB de JSON crudo en disco para {len(series)} series "
                          f"({filas_alfred} filas de tiempo real){extrapolado}"),
        limites=("120 peticiones/min por clave (aquí 1/s); output_type=1 con tiempo real completo: hasta 100000 "
                 "filas por página (offset); los índices re-estimados (NFCI, ANFCI) pasan del millón de filas"),
        estado_sonda="ok" if set(todas) <= set(series) else "parcial",
        mediciones=mediciones,
        notas=[AVISO_FRED,
               "Caché cruda: un JSON por página en <corpus>/mercados_series/crudo/alfred/; la clave nunca se guarda",
               "fecha_publicacion = primera decisión de las 16:30 NY tras la hora del release el día realtime_start; "
               "vintage = realtime_start",
               "Observaciones anteriores al primer vintage: valor del primer vintage, lag_fijo con la regla previa "
               "del catálogo y nota 'anterior a ALFRED'",
               "'.' de ALFRED: valor nulo con nota; observación que desaparece de un vintage: fila nula con nota",
               "Proyecciones del SEP: visibles desde su fecha de referencia; de la CBO (GDPPOT, NROU): recortadas "
               "a los vintages publicados desde su fecha de referencia (el contrato no admite previsiones)"],
        fecha_sonda=cliente.hoy.date().isoformat())
    ruta = guardar_ficha(ficha, destino) if guardar else None
    return ficha, ruta


# --- CLI ----------------------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Vintages exactos de ALFRED del Tema 1 (bloque 3). " + AVISO_FRED)
    ap.add_argument("--ids", nargs="*", help="ids del catálogo (por defecto, ids_alfred())")
    ap.add_argument("--descargar", action="store_true", help="descarga las series sin archivo completo")
    ap.add_argument("--refrescar", action="store_true", help="vuelve a descargar aunque haya archivo")
    ap.add_argument("--limite-segundos", type=float, default=None, help="no empieza otra serie pasado este tiempo")
    ap.add_argument("--ficha", action="store_true", help="escribe la ficha alfred_vintages")
    a = ap.parse_args(argv)
    cat = catalogo.cargar_catalogo()
    cliente = cliente_alfred()
    descarga = None
    if a.descargar:
        descarga = descargar_vintages(a.ids, cliente=cliente, cat=cat, refrescar=a.refrescar,
                                      limite_segundos=a.limite_segundos, registro=print)
        print(descarga["estado"].value_counts().to_dict(), f"{cliente.peticiones} peticiones")
    informe: dict[str, dict] = {}
    t0 = time.monotonic()
    tabla = tabla_alfred(a.ids, cliente=cliente, cat=cat, informe=informe)
    print(f"{len(tabla)} filas, {tabla['serie'].nunique()} series, {time.monotonic() - t0:.0f} s")
    for sid, i in informe.items():
        print(f"  {sid}: {i.get('filas')} filas, {i.get('n_vintages')} vintages desde {i.get('primer_vintage')}, "
              f"{i.get('previas')} previas, {i.get('puntos')} '.', {i.get('retiradas')} retiradas"
              + (f" | {'; '.join(i['avisos'])}" if i.get("avisos") else ""))
    if a.ficha:
        _, ruta = ficha_alfred(tabla, informe, cliente=cliente, cat=cat, descarga=descarga)
        print("ficha:", Path(ruta).relative_to(ROOT).as_posix() if ruta else None)
    print(AVISO_FRED)


if __name__ == "__main__":
    main()
