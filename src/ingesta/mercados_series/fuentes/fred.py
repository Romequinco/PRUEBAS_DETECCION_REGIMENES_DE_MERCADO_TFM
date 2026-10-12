"""Extractor de FRED/ALFRED del Tema 1 (bloque 2): cliente mínimo y sustitutos que no están en data/raw.

Cliente (``ClienteFred``):

- ``series/observations`` con ``realtime_start``/``realtime_end`` y ``vintage_dates`` (troceado en
  grupos de 500: la API admite 2000, pero esa URL la rechaza el CDN), paginado por ``offset``;
  ``series/vintagedates``, ``release/dates`` y ``series`` (metadatos).
- Red solo con ``ingesta.comun.http.get`` (User-Agent identificable, 1 petición/s por host: FRED
  admite 120/min). La clave ``FRED_API_KEY`` sale de ``.env``; nunca se guarda ni se imprime: los
  errores pasan por ``http.sin_clave`` y la caché guarda los parámetros sin ella.
- Caché cruda: un JSON por petición en ``<corpus>/mercados_series/crudo/fred/<endpoint>/<id>/``
  con nombre ``<AAAAMMDD>_<hash>.json`` (fecha de descarga en Nueva York). La misma petición el
  mismo día se sirve de la caché; cada día deja su fichero (archivo propio: las OAS de ICE BofA,
  capadas a 3 años, acumulan historia desde la primera descarga).

Conversión (``tabla_fred_sustitutos``): cada serie según su entrada del catálogo
(``config/catalogo_series.yaml``):

- ``calendario``: vintage ``unico``; ``disponible_utc`` = hora de su ``publicador``
  (``visibilidad.disponible_publicador``: H.15, H.10, Moody's, ICE en FRED, papel comercial...).
  Valores de la UNIÓN de todas las descargas archivadas (primer valor visto por fecha): la ventana
  móvil de las OAS no pierde filas.
- ``vintage_exacto`` (WEI): todas las vintages de ALFRED (``realtime_start`` = vintage) con la hora
  del publicador ese día.
- ``cierre_mercado`` y ``lag_fijo`` también se admiten (reutilizable por otros extractores).
- Valor ``'.'`` de FRED → ``valor`` nulo con ``NOTA_PUNTO``.

Datos solo en local (términos de FRED; ICE, Moody's: copyright). Al repositorio solo van código,
fichas y tablas de fechas, nunca valores.
CLI: ``python -m ingesta.mercados_series.fuentes.fred [--sin-red] [--fichas] [--escribir]``.
This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ingesta.comun import http
from ingesta.comun.rutas import ROOT, corpus_dir, env, resultados_dir
from ingesta.mercados_series import catalogo, visibilidad
from ingesta.mercados_series.contrato import (
    COLUMNAS_SERIES,
    FECHA,
    HUSO_DECISION,
    UTC,
    primera_decision_vec,
    vacio_series,
    validar_series,
)

API_FRED = "https://api.stlouisfed.org/fred"
TEMA = "mercados_series"
FUENTE = "fred_alfred"
AVISO_FRED = ("This product uses the FRED® API but is not endorsed or certified by the Federal Reserve "
              "Bank of St. Louis.")
MAX_VINTAGES = 2000          # vintage_dates por petición (límite de la API)
# Límite real más bajo: con 2000 fechas la URL ocupa ~26 KB y el CDN de FRED responde HTTP 400
# (medido 2026-10-12); con 600 (~8 KB) responde. Se trocea en grupos de 500 (~6,5 KB).
VINTAGES_POR_PETICION = 500
LIMITE_OBSERVACIONES = 100000  # observaciones por página (límite de la API)
LIMITE_FECHAS = 10000        # vintagedates y release/dates por página
INICIO_ALFRED = "1776-07-04"  # realtime_start mínimo de ALFRED
FIN_ALFRED = "9999-12-31"     # realtime_end abierto
NOTA_PUNTO = "FRED sin valor ('.') en esta fecha: hueco de la fuente (festivo o dato no publicado)"
NOTA_PUNTO_LARGO = "hueco permanente: FRED sin valor ('.') durante más de 31 días"
# Una diaria sin dato más tiempo que esto es un hueco declarado (igual que fuentes.fase1).
UMBRAL_HUECO_DIARIO = pd.Timedelta(days=31)
# Publicadores cuyas fechas se contrastan con la primera vintage de ALFRED (excepciones medidas): FRED
# es la vía pública de Moody's y de sus diferenciales calculados, y SOFR/EFFR/papel comercial tienen
# retrasos sueltos medidos. H.15 y H.10 diarios no: sus retrasos de ALFRED son de la carga de FRED.
PUBLICADORES_ALFRED = ("moodys", "fred_calculadas_dia_siguiente", "fred_tesoro_calculadas", "nyfed_sofr",
                       "nyfed_effr", "fed_cp")
# Columna 'origen' de config/excepciones_publicacion.csv: ALFRED output_type=4 + día de descarga.
ORIGEN_EXCEPCION = "alfred_ot4"
# Diarias que repiten el fechado de una mensual (USRECD = USREC en cada día del mes): se miden por la
# mensual (por ventanas, ALFRED agota el tiempo con ellas) y cada día hereda la fecha de su mes.
EQUIVALENTES_ALFRED = {"fred:USRECD": "fred:USREC"}
OAS_ICE = ("BAMLH0A0HYM2", "BAMLC0A0CM", "BAMLH0A0HYM2EY", "BAMLC0A0CMEY", "BAMLC0A4CBBB", "BAMLC0A1CAAA",
           "BAMLH0A1HYBB", "BAMLH0A3HYC")
METODOS_ADMITIDOS = ("calendario", "vintage_exacto", "cierre_mercado", "lag_fijo")
# Estados del catálogo que este extractor carga; pendiente_alfred solo con vintage_exacto (sus vintages).
ESTADOS_EXTRACTOR = (*catalogo.ESTADOS_CARGABLES, "pendiente_alfred")

Transporte = Callable[[str, dict[str, Any]], dict[str, Any]]

__all__ = [
    "API_FRED", "AVISO_FRED", "MAX_VINTAGES", "NOTA_PUNTO", "OAS_ICE", "ErrorFred", "ClienteFred",
    "ruta_crudo", "trozos", "parsear_observaciones", "ids_sustitutos", "tabla_serie", "tabla_fred_sustitutos",
    "medir_publicacion", "historial_vintage", "last_updated_ny", "fichas_fred", "ids_excepciones",
    "primeras_publicaciones", "medir_excepciones",
]


class ErrorFred(RuntimeError):
    """Fallo de una petición a FRED (mensaje siempre sin la clave)."""


def ruta_crudo() -> Path:
    """``<corpus>/mercados_series/crudo/fred/`` (fuera del repositorio)."""
    p = corpus_dir(TEMA) / "crudo" / "fred"
    p.mkdir(parents=True, exist_ok=True)
    return p


def trozos(elementos: Sequence[Any], n: int = VINTAGES_POR_PETICION) -> list[list[Any]]:
    """``elementos`` en grupos consecutivos de como mucho ``n`` (sin repetir ni perder ninguno).

    ``n`` entre 1 y ``MAX_VINTAGES`` (límite de la API); por defecto ``VINTAGES_POR_PETICION``.
    """
    if not 1 <= n <= MAX_VINTAGES:
        raise ValueError(f"el tamaño de trozo debe estar entre 1 y {MAX_VINTAGES}")
    lista = list(elementos)
    return [lista[i:i + n] for i in range(0, len(lista), n)]


def _hoy_ny() -> pd.Timestamp:
    return pd.Timestamp.now(tz=HUSO_DECISION).tz_localize(None).normalize()


def _transporte_http(url: str, params: dict[str, Any], *, reintentos: int = 3) -> dict[str, Any]:
    return http.get(url, params=params, por_segundo=1.0, reintentos=reintentos).json()


# --- Cliente ------------------------------------------------------------------------------------
class ClienteFred:
    """Cliente de FRED/ALFRED con caché cruda por petición.

    ``clave``: por defecto ``FRED_API_KEY`` de ``.env`` (se lee al primer uso, nunca se imprime).
    ``transporte``: función ``(url, params) -> dict`` (por defecto ``http.get(...).json()``; los
    tests pasan una sintética). ``carpeta``: caché (por defecto ``ruta_crudo()``); ``cache=False``
    no lee la caché (siempre descarga) pero sí guarda. ``hoy``: fecha de descarga para el nombre de
    los ficheros (por defecto, hoy en Nueva York). ``peticiones`` cuenta las que salen a la red.
    """

    def __init__(self, clave: str | None = None, *, transporte: Transporte | None = None,
                 carpeta: Path | None = None, cache: bool = True, hoy: Any = None):
        self._clave = clave
        self._transporte = transporte
        self.carpeta = Path(carpeta) if carpeta is not None else None
        self.cache = cache
        self.hoy = pd.Timestamp(hoy).normalize() if hoy is not None else _hoy_ny()
        self.peticiones = 0
        self.bytes = 0

    # -- infraestructura --
    def _carpeta(self) -> Path:
        if self.carpeta is None:
            self.carpeta = ruta_crudo()
        return self.carpeta

    def _api_key(self) -> str:
        clave = self._clave or env("FRED_API_KEY")
        if not clave:
            raise ErrorFred("falta FRED_API_KEY en .env o en el entorno")
        return clave

    @staticmethod
    def _limpios(params: dict[str, Any]) -> dict[str, str]:
        return {k: str(v) for k, v in sorted(params.items()) if k not in ("api_key", "file_type")}

    @staticmethod
    def huella(endpoint: str, params: dict[str, Any]) -> str:
        """Hash corto de la petición (endpoint + parámetros sin clave): nombre de su fichero en caché."""
        texto = json.dumps([endpoint, ClienteFred._limpios(params)], sort_keys=True)
        return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:12]

    def carpeta_peticion(self, endpoint: str, params: dict[str, Any]) -> Path:
        sujeto = params.get("series_id") or (f"release_{params['release_id']}" if "release_id" in params else "_")
        return self._carpeta() / endpoint.replace("/", "_") / str(sujeto)

    def ruta_cache(self, endpoint: str, params: dict[str, Any]) -> Path:
        return self.carpeta_peticion(endpoint, params) / f"{self.hoy:%Y%m%d}_{self.huella(endpoint, params)}.json"

    def peticion(self, endpoint: str, params: dict[str, Any] | None = None, *, reintentos: int = 3) -> dict[str, Any]:
        """Respuesta JSON de ``endpoint`` (caché del día o red). ``ErrorFred`` sin la clave si falla."""
        params = dict(params or {})
        ruta = self.ruta_cache(endpoint, params)
        if self.cache and ruta.exists():
            return json.loads(ruta.read_text(encoding="utf-8"))["respuesta"]
        consulta = {**params, "api_key": self._api_key(), "file_type": "json"}
        url = f"{API_FRED}/{endpoint}"
        try:
            if self._transporte is not None:
                datos = self._transporte(url, consulta)
            else:
                datos = _transporte_http(url, consulta, reintentos=reintentos)
        except Exception as e:  # el mensaje puede llevar la URL con la clave
            raise ErrorFred(http.sin_clave(f"FRED {endpoint} {self._limpios(params)}: {e}")) from None
        finally:
            consulta.pop("api_key", None)
        self.peticiones += 1
        envoltorio = {"endpoint": endpoint, "params": self._limpios(params),
                      "descargado_utc": pd.Timestamp.now(tz="UTC").isoformat(), "respuesta": datos}
        texto = json.dumps(envoltorio, ensure_ascii=False)
        self.bytes += len(texto)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        tmp = ruta.with_suffix(".json.tmp")
        tmp.write_text(texto, encoding="utf-8")
        os.replace(tmp, ruta)
        return datos

    def _paginado(self, endpoint: str, params: dict[str, Any], campo: str, limite: int,
                  reintentos: int = 3) -> list[dict[str, Any]]:
        filas: list[dict[str, Any]] = []
        offset = 0
        while True:
            d = self.peticion(endpoint, {**params, "limit": limite, "offset": offset}, reintentos=reintentos)
            lote = d.get(campo) or []
            filas += lote
            offset += len(lote)
            total = int(d.get("count", offset))
            if not lote or offset >= total:
                return filas

    # -- endpoints --
    def serie(self, series_id: str) -> dict[str, Any]:
        """Metadatos (``fred/series``): título, unidades, frecuencia, cobertura, ``last_updated``."""
        return (self.peticion("series", {"series_id": series_id}).get("seriess") or [{}])[0]

    def observaciones(self, series_id: str, *, observation_start: str | None = None,
                      observation_end: str | None = None, realtime_start: str | None = None,
                      realtime_end: str | None = None, vintage_dates: Sequence[str] | None = None,
                      output_type: int | None = None, reintentos: int = 3) -> list[dict[str, Any]]:
        """Observaciones (``output_type=1``: una fila por periodo de tiempo real), todas las páginas.

        ``vintage_dates`` se trocea en grupos de ``VINTAGES_POR_PETICION`` (una petición por grupo;
        nunca más de ``MAX_VINTAGES``) y las filas repetidas entre grupos se quitan. Sin realtime ni vintages: lo vigente hoy.
        ``output_type=4``: solo la primera publicación de cada observación (``realtime_start`` = su
        primera vintage).
        """
        base: dict[str, Any] = {"series_id": series_id}
        for k, v in (("observation_start", observation_start), ("observation_end", observation_end),
                     ("realtime_start", realtime_start), ("realtime_end", realtime_end),
                     ("output_type", output_type)):
            if v is not None:
                base[k] = v
        if vintage_dates is None:
            return self._paginado("series/observations", base, "observations", LIMITE_OBSERVACIONES, reintentos)
        filas: list[dict[str, Any]] = []
        vistas: set[tuple] = set()
        for grupo in trozos(sorted(set(vintage_dates))):
            params = {**base, "vintage_dates": ",".join(grupo)}
            for o in self._paginado("series/observations", params, "observations", LIMITE_OBSERVACIONES,
                                    reintentos):
                k = (o.get("date"), o.get("realtime_start"), o.get("realtime_end"), o.get("value"))
                if k not in vistas:
                    vistas.add(k)
                    filas.append(o)
        return filas

    def vintages(self, series_id: str, *, realtime_start: str = INICIO_ALFRED,
                 realtime_end: str = FIN_ALFRED) -> list[str]:
        """Fechas de vintage que ALFRED conserva de ``series_id`` (``fred/series/vintagedates``)."""
        params = {"series_id": series_id, "realtime_start": realtime_start, "realtime_end": realtime_end}
        return [str(x) for x in self._paginado("series/vintagedates", params, "vintage_dates", LIMITE_FECHAS)]

    def fechas_release(self, release_id: int, *, realtime_start: str, realtime_end: str) -> list[str]:
        """Fechas de publicación del release ``release_id`` (``fred/release/dates``)."""
        params = {"release_id": release_id, "realtime_start": realtime_start, "realtime_end": realtime_end,
                  "include_release_dates_with_no_data": "true", "sort_order": "asc"}
        return [d["date"] for d in self._paginado("release/dates", params, "release_dates", LIMITE_FECHAS)]

    # -- archivo --
    def archivadas(self, series_id: str, **params: Any) -> list[tuple[str, list[dict[str, Any]]]]:
        """``[(AAAAMMDD, observaciones)]`` de todas las descargas archivadas de esa petición, por fecha.

        Junta las páginas de cada día (mismos parámetros salvo ``offset``). Sin red.
        """
        base = {"series_id": series_id, **{k: v for k, v in params.items() if v is not None}}
        carpeta = self.carpeta_peticion("series/observations", base)
        if not carpeta.exists():
            return []
        objetivo = self._limpios(base)
        por_dia: dict[str, list[tuple[int, list]]] = {}
        for f in sorted(carpeta.glob("*.json")):
            env_ = json.loads(f.read_text(encoding="utf-8"))
            p = dict(env_["params"])
            offset = int(p.pop("offset", 0))
            p.pop("limit", None)
            if p != objetivo:
                continue
            por_dia.setdefault(f.name[:8], []).append((offset, env_["respuesta"].get("observations") or []))
        return [(dia, [o for _, lote in sorted(paginas, key=lambda x: x[0]) for o in lote])
                for dia, paginas in sorted(por_dia.items())]


# --- Parseo -------------------------------------------------------------------------------------
def _fecha_fred(texto: Any) -> pd.Timestamp:
    """Fecha de FRED; ``9999-12-31`` (realtime abierto) → NaT."""
    if texto is None or str(texto).startswith("9999"):
        return pd.NaT
    return pd.Timestamp(str(texto))


def parsear_observaciones(observaciones: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Observaciones JSON de FRED → ``fecha_referencia, valor, ausente, realtime_start, realtime_end``.

    ``value == '.'`` → ``valor`` nulo y ``ausente`` True. ``realtime_end`` 9999-12-31 → NaT (abierto).
    Fechas ``datetime64[ns]`` a medianoche; orden por fecha y ``realtime_start``.
    """
    filas = []
    for o in observaciones:
        bruto = str(o.get("value", ".")).strip()
        ausente = bruto in (".", "")
        filas.append({"fecha_referencia": pd.Timestamp(o["date"]),
                      "valor": float("nan") if ausente else float(bruto), "ausente": ausente,
                      "realtime_start": _fecha_fred(o.get("realtime_start")),
                      "realtime_end": _fecha_fred(o.get("realtime_end"))})
    df = pd.DataFrame(filas, columns=["fecha_referencia", "valor", "ausente", "realtime_start", "realtime_end"])
    for c in ("fecha_referencia", "realtime_start", "realtime_end"):
        df[c] = pd.to_datetime(df[c]).astype(FECHA)
    df["valor"] = df["valor"].astype("float64")
    df["ausente"] = df["ausente"].astype(bool)
    return df.sort_values(["fecha_referencia", "realtime_start"], kind="mergesort").reset_index(drop=True)


def unir_archivo(descargas: Sequence[tuple[str, list[dict[str, Any]]]]) -> pd.DataFrame:
    """Unión de varias descargas de lo vigente: por fecha, el PRIMER valor no nulo visto.

    Una fecha solo con ``'.'`` queda ausente. Columnas de ``parsear_observaciones`` más
    ``descarga`` (AAAAMMDD de la primera descarga que trajo ese valor). Así la ventana móvil de una
    serie capada (OAS de ICE BofA) no pierde las fechas que ya salieron de la ventana.
    """
    partes = []
    for dia, obs in descargas:
        p = parsear_observaciones(obs)
        p["descarga"] = dia
        partes.append(p)
    if not partes:
        return parsear_observaciones([]).assign(descarga=pd.Series(dtype=object))
    todo = pd.concat(partes, ignore_index=True)
    todo = todo.sort_values(["fecha_referencia", "ausente", "descarga"], kind="mergesort")
    return todo.drop_duplicates("fecha_referencia", keep="first").reset_index(drop=True)


# --- Catálogo → tabla del contrato --------------------------------------------------------------
def _id_origen(sid: str, entrada: dict[str, Any]) -> str:
    return entrada.get("id_origen") or sid.split(":", 1)[1]


def ids_sustitutos(cat: dict[str, dict] | None = None) -> list[str]:
    """Series del catálogo de fuente FRED que NO están en data/raw y este extractor carga.

    Fuente ``fred_alfred``, sin ``fichero_fase1``, método admitido y estado cargable
    (``pendiente_alfred`` solo con ``vintage_exacto``). Hoy: Moody's diarios, H.10 (DEXUSEU,
    DTWEXBGS, DTWEXAFEGS), financiación (TEDRATE, DCPF3M, CPFF), las OAS de ICE BofA y WEI.
    """
    cat = cat if cat is not None else catalogo.cargar_catalogo()
    out = []
    for sid, e in cat.items():
        if e["fuente"] != FUENTE or e["fichero_fase1"] or e["metodo"] not in METODOS_ADMITIDOS:
            continue
        if e["estado"] in catalogo.ESTADOS_CARGABLES or (e["estado"] == "pendiente_alfred"
                                                          and e["metodo"] == "vintage_exacto"):
            out.append(sid)
    return sorted(out)


_cache_publicador: dict[tuple[str, pd.Timestamp, int], pd.Timestamp] = {}


def _disponible_publicador_vec(publicador: str, fechas: pd.Series,
                               publicaciones: dict[str, Any] | None) -> pd.Series:
    """``visibilidad.disponible_publicador`` por fecha distinta (cacheado si es la config por defecto)."""
    res = {}
    # La clave lleva el mtime de publicaciones.yaml: editar el YAML invalida la caché (notebooks).
    version = visibilidad.RUTA_PUBLICACIONES.stat().st_mtime_ns
    for d in pd.unique(fechas.dropna()):
        d = pd.Timestamp(d)
        k = (publicador, d, version)
        if publicaciones is None and k in _cache_publicador:
            res[d] = _cache_publicador[k]
            continue
        t = visibilidad.disponible_publicador(publicador, d, publicaciones)
        res[d] = t
        if publicaciones is None:
            _cache_publicador[k] = t
    return pd.Series([res.get(pd.Timestamp(d), pd.NaT) if pd.notna(d) else pd.NaT for d in fechas],
                     index=fechas.index, dtype=UTC)


def _puntos_diarios(df: pd.DataFrame) -> pd.DataFrame:
    """Diarias: quita los ``'.'`` sueltos (festivos: un día sin dato no es un hueco) y deja UNA fila
    nula al inicio de cada racha sin valor de más de ``UMBRAL_HUECO_DIARIO`` (como ``fuentes.fase1``).

    Las rachas al principio o al final de la serie (antes del primer valor o tras el último) se
    quitan. Añade la columna auxiliar ``_largo``.
    """
    df = df.sort_values("fecha_referencia", kind="mergesort").reset_index(drop=True)
    con_valor = ~df["ausente"]
    if not con_valor.any():
        return df.iloc[0:0].assign(_largo=pd.Series(dtype=bool))
    fechas = df.loc[con_valor, "fecha_referencia"]
    previo = df["fecha_referencia"].where(con_valor).ffill()
    siguiente = df["fecha_referencia"].where(con_valor).bfill()
    largo = df["ausente"] & previo.notna() & siguiente.notna() & ((siguiente - previo) > UMBRAL_HUECO_DIARIO)
    primera_de_racha = largo & ~largo.shift(1, fill_value=False)
    dentro = df["fecha_referencia"].between(fechas.min(), fechas.max())
    conservar = (con_valor | primera_de_racha) & dentro
    return df[conservar].assign(_largo=primera_de_racha[conservar]).reset_index(drop=True)


def tabla_serie(sid: str, obs: pd.DataFrame, entrada: dict[str, Any], *,
                publicaciones: dict[str, Any] | None = None) -> pd.DataFrame:
    """Observaciones parseadas de ``sid`` → filas del contrato de series (sin validar).

    ``calendario`` y ``cierre_mercado``: vintage ``unico`` y ``disponible_utc`` por publicador o
    cierre; ``lag_fijo``: ``fecha_publicacion_regla`` (sin ``disponible_utc``; nota de revisables si
    procede); ``vintage_exacto``: una fila por vintage de ALFRED (``realtime_start``) con la hora del
    publicador ese día; los ``'.'`` anteriores al primer valor de una fecha (la observación aún no
    existía en esa vintage) se descartan. Diarias: los ``'.'`` sueltos (festivos) se descartan y
    cada racha sin valor de más de 31 días deja una fila nula con ``NOTA_PUNTO_LARGO``; en las demás
    frecuencias, cada ``'.'`` → valor nulo con ``NOTA_PUNTO``.
    """
    metodo = entrada["metodo"]
    if metodo not in METODOS_ADMITIDOS:
        raise ValueError(f"{sid}: metodo {metodo!r} no lo carga el extractor de FRED ({list(METODOS_ADMITIDOS)})")
    if obs.empty:
        return vacio_series()
    df = obs.copy().reset_index(drop=True)
    if metodo != "vintage_exacto" and entrada.get("frecuencia") == "diaria":
        df = _puntos_diarios(df)
    nota = pd.Series(None, index=df.index, dtype=object)
    nota[df["ausente"]] = NOTA_PUNTO
    nota[df["ausente"] & df.get("_largo", False)] = NOTA_PUNTO_LARGO
    df = df.drop(columns=["_largo"], errors="ignore")
    if metodo == "vintage_exacto":
        publicador = entrada.get("publicador")
        if not publicador:
            raise ValueError(f"{sid}: vintage_exacto necesita 'publicador' (hora de cada vintage)")
        primero = df.loc[~df["ausente"]].groupby("fecha_referencia")["realtime_start"].min()
        inicio = df["fecha_referencia"].map(primero)
        conservar = ~df["ausente"] | (inicio.notna() & (df["realtime_start"] >= inicio))
        df, nota = df[conservar].reset_index(drop=True), nota[conservar].reset_index(drop=True)
        vintage = df["realtime_start"].dt.strftime("%Y-%m-%d")
        disponible = _disponible_publicador_vec(publicador, df["realtime_start"], publicaciones)
        fecha_pub = primera_decision_vec(disponible)
    else:
        vintage = pd.Series("unico", index=df.index, dtype=object)
        if metodo == "calendario":
            disponible = _disponible_publicador_vec(entrada["publicador"], df["fecha_referencia"], publicaciones)
            fecha_pub = primera_decision_vec(disponible)
        elif metodo == "cierre_mercado":
            disponible = visibilidad.disponible_cierre_serie(df["fecha_referencia"], entrada["mercado"])
            fecha_pub = primera_decision_vec(disponible)
        else:  # lag_fijo
            disponible = pd.Series(pd.NaT, index=df.index, dtype=UTC)
            fecha_pub = catalogo.fecha_publicacion_regla(df["fecha_referencia"], entrada)
            if entrada.get("revisable"):
                nr = catalogo.nota_revisable(entrada)
                nota = nota.where(nota.isna(), nota + "; " + nr).fillna(nr)
    out = pd.DataFrame({
        "serie": sid,
        "fecha_referencia": df["fecha_referencia"].to_numpy(),
        "fecha_publicacion": pd.Series(fecha_pub).to_numpy(),
        "disponible_utc": pd.Series(disponible, dtype=UTC).to_numpy(),
        "vintage": vintage.to_numpy(),
        "valor": df["valor"].to_numpy(),
        "unidad": entrada["unidad"],
        "divisa": entrada["divisa"],
        "fuente": entrada["fuente"],
        "metodo_publicacion": metodo,
        "nota": nota.to_numpy(),
    })
    out["disponible_utc"] = pd.to_datetime(out["disponible_utc"], utc=True)
    return out[list(COLUMNAS_SERIES)]


def _observaciones_serie(cliente: ClienteFred, sid: str, entrada: dict[str, Any], descargar: bool) -> pd.DataFrame:
    origen = _id_origen(sid, entrada)
    if entrada["metodo"] == "vintage_exacto":
        rt = {"realtime_start": INICIO_ALFRED, "realtime_end": FIN_ALFRED}
        if descargar:
            return parsear_observaciones(cliente.observaciones(origen, **rt))
        archivo = cliente.archivadas(origen, **rt)
        return parsear_observaciones(archivo[-1][1] if archivo else [])
    if descargar:
        cliente.observaciones(origen)
    return unir_archivo(cliente.archivadas(origen))


def tabla_fred_sustitutos(ids: Iterable[str] | None = None, *, cliente: ClienteFred | None = None,
                          descargar: bool = True, cat: dict[str, dict] | None = None,
                          publicaciones: dict[str, Any] | None = None) -> pd.DataFrame:
    """Tabla del contrato (validada) de las series de FRED que no están en data/raw.

    ``ids``: por defecto ``ids_sustitutos()``. ``descargar=False`` solo usa el archivo crudo (sin
    red). Valida con ``contrato.validar_series`` y ``catalogo.validar_contra_catalogo``; una serie
    ``pendiente_alfred`` cargada con sus vintages de ALFRED cuenta como cargable (es lo que su
    estado pide). ``ErrorContrato`` si algo no cuadra; ``ErrorFred`` si falla la red.
    """
    cat = cat if cat is not None else catalogo.cargar_catalogo()
    lista = list(ids) if ids is not None else ids_sustitutos(cat)
    cliente = cliente or ClienteFred()
    partes = []
    for sid in lista:
        sid = catalogo.resolver(sid, cat)
        if sid not in cat:
            raise KeyError(f"serie {sid!r} no está en el catálogo")
        partes.append(tabla_serie(sid, _observaciones_serie(cliente, sid, cat[sid], descargar), cat[sid],
                                  publicaciones=publicaciones))
    partes = [p for p in partes if len(p)]
    tabla = validar_series(pd.concat(partes, ignore_index=True)) if partes else vacio_series()
    efectivo = {sid: (dict(e, estado="activa") if e["estado"] == "pendiente_alfred"
                      and e["metodo"] == "vintage_exacto" else e) for sid, e in cat.items()}
    catalogo.validar_contra_catalogo(tabla, efectivo)
    return tabla


# --- Excepciones de publicación medidas en ALFRED (solo fechas) ---------------------------------
def ids_excepciones(cat: dict[str, dict] | None = None) -> list[str]:
    """Series de FRED cuyas fechas de publicación se contrastan con la primera vintage de ALFRED.

    Estado cargable y método ``lag_fijo`` (macro, medias mensuales, índices semanales) o
    ``calendario`` con un publicador de ``PUBLICADORES_ALFRED`` (Moody's, diferenciales de FRED,
    SOFR, EFFR, papel comercial).
    """
    cat = cat if cat is not None else catalogo.cargar_catalogo()
    out = []
    for sid, e in cat.items():
        if not sid.startswith("fred:") or e["estado"] not in catalogo.ESTADOS_CARGABLES:
            continue
        if e["metodo"] == "lag_fijo" or (e["metodo"] == "calendario" and e.get("publicador") in PUBLICADORES_ALFRED):
            out.append(sid)
    return sorted(out)


def primeras_publicaciones(cliente: ClienteFred, series_id: str, *, diaria: bool = False,
                           margen_dias: int = 400, ventana_dias: int = 366,
                           retraso_max_dias: int = 365) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Primera vintage de ALFRED de cada observación publicada DESPUÉS de la primera vintage de la serie.

    Las observaciones anteriores a la primera vintage aparecen con ella (carga inicial de ALFRED:
    cota superior, no su fecha real) y se descartan. Pide ``series/vintagedates`` y
    ``series/observations`` con ``output_type=4`` desde ``margen_dias`` antes de la primera vintage.
    Diarias (``diaria=True``): una vintage por día hábil supera el límite de la API con todo el
    tiempo real, así que se pide por ventanas de ``ventana_dias`` de observaciones con tiempo real
    hasta ``retraso_max_dias`` después (un retraso mayor no se mediría). Devuelve
    ``(fecha_referencia, primera_publicacion)`` y ``{primera_vintage, n_vintages, n_cubiertas}``.
    Sin valores.
    """
    vint = cliente.vintages(series_id)
    vacio = pd.DataFrame({"fecha_referencia": pd.Series(dtype=FECHA), "primera_publicacion": pd.Series(dtype=FECHA)})
    if not vint:
        return vacio, {"primera_vintage": None, "n_vintages": 0, "n_cubiertas": 0}
    v0 = pd.Timestamp(vint[0])
    if len(vint) == 1:   # una sola vintage (serie retirada antes de ALFRED): nada publicado después
        return vacio, {"primera_vintage": v0.date().isoformat(), "n_vintages": 1, "n_cubiertas": 0}
    inicio = v0 - pd.Timedelta(days=margen_dias)
    if not diaria:
        rt = {"realtime_start": INICIO_ALFRED, "realtime_end": FIN_ALFRED, "output_type": 4, "reintentos": 2}
        try:
            obs = parsear_observaciones(cliente.observaciones(series_id, observation_start=inicio.date().isoformat(),
                                                              **rt))
        except ErrorFred:   # serie retirada antes de su primera vintage: observation_start fuera de rango
            obs = parsear_observaciones(cliente.observaciones(series_id, **rt))
    else:
        ultima = pd.Timestamp(vint[-1])
        partes = []
        a = inicio
        while a <= ultima:
            b = a + pd.Timedelta(days=ventana_dias - 1)
            fin_rt = min(b + pd.Timedelta(days=retraso_max_dias), cliente.hoy)
            partes += cliente.observaciones(
                series_id, observation_start=a.date().isoformat(), observation_end=b.date().isoformat(),
                realtime_start=max(a, v0).date().isoformat(), realtime_end=fin_rt.date().isoformat(),
                output_type=4, reintentos=2)
            a = b + pd.Timedelta(days=1)
        obs = parsear_observaciones(partes)
    obs = obs[~obs["ausente"]]
    primera = obs.groupby("fecha_referencia")["realtime_start"].min()
    primera = primera[primera > v0]
    out = pd.DataFrame({"fecha_referencia": primera.index.astype(FECHA),
                        "primera_publicacion": primera.to_numpy()}).astype(FECHA)
    return out, {"primera_vintage": v0.date().isoformat(), "n_vintages": len(vint), "n_cubiertas": int(len(out))}


def _fecha_regla(entrada: dict[str, Any], fechas: pd.Series) -> pd.Series:
    """Fecha de publicación por la regla del catálogo (lag_fijo o publicador) de ``fechas``."""
    if entrada["metodo"] == "lag_fijo":
        return catalogo.fecha_publicacion_regla(fechas, entrada).reset_index(drop=True)
    disp = _disponible_publicador_vec(entrada["publicador"], fechas.reset_index(drop=True), None)
    return primera_decision_vec(disp).astype(FECHA)


def _fecha_efectiva(entrada: dict[str, Any], dias: pd.Series) -> pd.Series:
    """Fecha de decisión de una publicación el día ``dias`` (a la hora del publicador si la tiene)."""
    dias = dias.reset_index(drop=True)
    if entrada["metodo"] == "lag_fijo" or not entrada.get("publicador"):
        return dias.astype(FECHA)
    pub = visibilidad.cargar_publicaciones(catalogo=False)["publicadores"][entrada["publicador"]]
    disp = pd.Series([visibilidad.disponible_publicacion(d, pub["hora"], pub["huso"],
                                                         pd.Timedelta(minutes=pub["margen_min"])) for d in dias],
                     dtype=UTC)
    return primera_decision_vec(disp).astype(FECHA)


def _equivalente(cliente: ClienteFred, sid: str, e: dict[str, Any], cat: dict[str, dict],
                 medidas_mes: dict[str, pd.DataFrame], origen: str) -> dict[str, Any]:
    """Excepciones de una diaria a partir de su mensual equivalente (``EQUIVALENTES_ALFRED``)."""
    base_id = EQUIVALENTES_ALFRED[sid]
    if base_id not in medidas_mes:
        prim, _ = primeras_publicaciones(cliente, _id_origen(base_id, cat[base_id]))
        medidas_mes[base_id] = pd.DataFrame({"fecha_referencia": prim["fecha_referencia"],
                                             "primera_publicacion": prim["primera_publicacion"],
                                             "regla": _fecha_regla(cat[base_id], prim["fecha_referencia"]).to_numpy()})
    m = medidas_mes[base_id]
    filas = []
    for ref, pub in zip(m["fecha_referencia"], m["primera_publicacion"]):
        dias = pd.Series(pd.date_range(ref, ref + pd.offsets.MonthEnd(0), freq="D"))
        regla = catalogo.fecha_publicacion_regla(dias, e)
        fuga = pd.Timestamp(pub) > regla
        if fuga.any():
            filas.append(pd.DataFrame({"serie": sid, "fecha_referencia": dias[fuga].to_numpy(),
                                       "fecha_publicacion": pub, "fecha_regla": regla[fuga].to_numpy(),
                                       "origen": f"{origen} (por {base_id})"}))
    exc = (pd.concat(filas, ignore_index=True) if filas
           else pd.DataFrame(columns=list(catalogo.COLUMNAS_EXCEPCIONES_DETALLE)))
    fila = {"serie": sid, "metodo": e["metodo"], "regla": f"{e['lag']} desde el fin de {e['fin_periodo']}",
            "primera_vintage": None, "n_cubiertas": int(len(m)), "n_fugas": int(exc["fecha_referencia"].dt.to_period("M")
                                                                             .nunique()) if len(exc) else 0,
            "equivalente": base_id}
    return {"exc": exc, "_fila": fila}


def medir_excepciones(cliente: ClienteFred, ids: Iterable[str] | None = None, *, cat: dict[str, dict] | None = None,
                      registro: Callable[[str], None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Contrasta la regla de publicación con la primera vintage de ALFRED, serie a serie.

    Devuelve ``(excepciones, medida)``. ``excepciones``: columnas de
    ``catalogo.COLUMNAS_EXCEPCIONES_DETALLE`` con las observaciones cuya primera publicación (a la hora del
    publicador) es POSTERIOR a la regla (fugas de la regla). ``medida``: por serie, periodos
    cubiertos por ALFRED, fugas, fuga máxima y holgura mediana (regla − publicación, días), primera
    vintage, o el error si la petición falló (la CLI es reanudable: la caché del día no repite).
    """
    cat = cat if cat is not None else catalogo.cargar_catalogo()
    log = registro or (lambda _m: None)
    lista = list(ids) if ids is not None else ids_excepciones(cat)
    excepciones, medida = [], []
    origen = f"{ORIGEN_EXCEPCION}_{cliente.hoy:%Y%m%d}"
    medidas_mes: dict[str, pd.DataFrame] = {}
    for sid in lista:
        e = cat[sid]
        if sid in EQUIVALENTES_ALFRED:
            fila_eq = _equivalente(cliente, sid, e, cat, medidas_mes, origen)
            medida.append(fila_eq.pop("_fila"))
            if len(fila_eq["exc"]):
                excepciones.append(fila_eq["exc"])
            continue
        fila: dict[str, Any] = {"serie": sid, "metodo": e["metodo"],
                                "regla": e.get("publicador") if e["metodo"] == "calendario" else
                                f"{e['lag']} desde el fin de {e['fin_periodo']}"}
        try:
            prim, info = primeras_publicaciones(cliente, _id_origen(sid, e), diaria=e["frecuencia"] == "diaria")
        except ErrorFred as err:
            medida.append(fila | {"error": str(err)[:200]})
            log(f"  {sid}: error {str(err)[:120]}")
            continue
        fila |= info
        if prim.empty:
            medida.append(fila | {"n_fugas": 0})
            continue
        regla = _fecha_regla(e, prim["fecha_referencia"])
        efectiva = _fecha_efectiva(e, prim["primera_publicacion"])
        dif = (efectiva - regla).dt.days
        fuga = dif > 0
        fila |= {"inicio_cubierto": prim["fecha_referencia"].min().date().isoformat(),
                 "fin_cubierto": prim["fecha_referencia"].max().date().isoformat(),
                 "n_fugas": int(fuga.sum()), "fuga_max_dias": int(dif.max()) if fuga.any() else 0,
                 "holgura_mediana_dias": float((-dif).median())}
        medida.append(fila)
        medidas_mes[sid] = pd.DataFrame({"fecha_referencia": prim["fecha_referencia"],
                                         "primera_publicacion": prim["primera_publicacion"], "regla": regla.to_numpy()})
        log(f"  {sid}: {info['n_cubiertas']} periodos cubiertos desde {info['primera_vintage']}, "
            f"{fila['n_fugas']} fugas (máx. {fila['fuga_max_dias']} d)")
        if fuga.any():
            excepciones.append(pd.DataFrame({
                "serie": sid, "fecha_referencia": prim.loc[fuga.to_numpy(), "fecha_referencia"].to_numpy(),
                "fecha_publicacion": prim.loc[fuga.to_numpy(), "primera_publicacion"].to_numpy(),
                "fecha_regla": regla[fuga].to_numpy(), "origen": origen}))
    cols = list(catalogo.COLUMNAS_EXCEPCIONES_DETALLE)
    exc = (pd.concat(excepciones, ignore_index=True)[cols] if excepciones
           else pd.DataFrame({c: pd.Series(dtype=object) for c in cols}))
    exc = exc.sort_values(["serie", "fecha_referencia"], kind="mergesort").reset_index(drop=True)
    return exc, pd.DataFrame(medida)


# --- Mediciones (solo fechas, nunca valores) ----------------------------------------------------
def medir_publicacion(cliente: ClienteFred, series_id: str, *, dias: int = 35,
                      hoy: Any = None) -> pd.DataFrame:
    """Primera vintage de ALFRED (``realtime_start``) de cada observación de los últimos ``dias``.

    Columnas: ``serie, fecha_referencia, primera_vintage, dias_naturales, dias_habiles_us,
    dia_semana``. Se piden las vintages desde ``hoy - dias - 7``; una observación cuya primera
    vintage es justo ese inicio ya existía antes (truncada) y se descarta. Sin valores.
    """
    hoy = pd.Timestamp(hoy).normalize() if hoy is not None else cliente.hoy
    rt = (hoy - pd.Timedelta(days=dias + 7)).date().isoformat()
    obs = parsear_observaciones(cliente.observaciones(
        series_id, observation_start=(hoy - pd.Timedelta(days=dias)).date().isoformat(),
        realtime_start=rt, realtime_end=FIN_ALFRED))
    obs = obs[~obs["ausente"]]
    primera = obs.groupby("fecha_referencia")["realtime_start"].min()
    primera = primera[primera > pd.Timestamp(rt)]
    filas = []
    for ref, pub in primera.items():
        habiles = sum(visibilidad.es_habil(d, "us_federal") for d in pd.date_range(ref + pd.Timedelta(days=1), pub))
        filas.append({"serie": series_id, "fecha_referencia": ref.date().isoformat(),
                      "primera_vintage": pub.date().isoformat(), "dias_naturales": int((pub - ref).days),
                      "dias_habiles_us": int(habiles), "dia_semana": pub.day_name()})
    return pd.DataFrame(filas, columns=["serie", "fecha_referencia", "primera_vintage", "dias_naturales",
                                        "dias_habiles_us", "dia_semana"])


def historial_vintage(cliente: ClienteFred, series_id: str, vintage: str) -> dict[str, Any]:
    """Cobertura de ``series_id`` tal como estaba en ``vintage`` (por ``vintage_dates`` y por realtime).

    Devuelve ``{metodo: {n_obs, inicio, fin} | {error}}``. Sin valores.
    """
    out: dict[str, Any] = {}
    for metodo, kw in (("vintage_dates", {"vintage_dates": [vintage]}),
                       ("realtime", {"realtime_start": vintage, "realtime_end": vintage})):
        try:
            obs = parsear_observaciones(cliente.observaciones(series_id, reintentos=1, **kw))
        except ErrorFred as e:
            out[metodo] = {"error": str(e)[:160]}
            continue
        obs = obs[~obs["ausente"]]
        out[metodo] = {"n_obs": int(len(obs)),
                       "inicio": obs["fecha_referencia"].min().date().isoformat() if len(obs) else None,
                       "fin": obs["fecha_referencia"].max().date().isoformat() if len(obs) else None}
    return out


def last_updated_ny(texto: str | None) -> str | None:
    """``last_updated`` de FRED (hora de Chicago con desfase, p. ej. ``2026-10-09 08:08:04-05``) en hora
    de Nueva York (``AAAA-MM-DD HH:MM``). None si falta o no se entiende."""
    if not texto:
        return None
    t = str(texto).strip()
    if len(t) >= 3 and t[-3] in "+-" and t[-2:].isdigit():
        t += ":00"                                   # '-05' → '-05:00' (ISO)
    try:
        ts = pd.Timestamp(t)
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is None:
        return None
    return ts.tz_convert(HUSO_DECISION).strftime("%Y-%m-%d %H:%M")


def _cobertura(tabla: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out = {}
    for sid, g in tabla.groupby("serie"):
        v = g[g["valor"].notna()]
        out[sid] = {"filas": int(len(g)), "n_obs": int(v["fecha_referencia"].nunique()),
                    "nulos_punto": int(g["valor"].isna().sum()),
                    "inicio": v["fecha_referencia"].min().date().isoformat() if len(v) else None,
                    "fin": v["fecha_referencia"].max().date().isoformat() if len(v) else None,
                    "vintages": int(g["vintage"].nunique())}
    return out


def _resumen_lags(lags: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {sid: {f"D+{k}": int(n) for k, n in g["dias_habiles_us"].value_counts().sort_index().items()}
            for sid, g in lags.groupby("serie")}


def fichas_fred(cliente: ClienteFred, tabla: pd.DataFrame, *, oas_vintage: str = "2026-03-31",
                oas_previos: Sequence[str] = ("2019-01-02",),
                series_lag: Sequence[str] = ("BAMLH0A0HYM2", "BAMLC0A0CM", "DBAA", "DCPF3M", "DEXUSEU", "WEI"),
                guardar: bool = True) -> dict[str, Any]:
    """Mide y escribe las fichas ``fred_alfred`` e ``ice_bofa_oas`` (y la tabla de retrasos medidos).

    Mide en ALFRED la primera vintage de las últimas observaciones (``medir_publicacion``), las
    vintages de WEI, la cobertura de cada serie de ``tabla`` y el historial de las OAS en
    ``oas_vintage``, en vintages anteriores a la ventana (``oas_previos``) y en su primer vintage
    conservado. Solo fechas y recuentos.
    """
    from ingesta.comun.esquema import FichaFuente
    from ingesta.comun.fichas import guardar_ficha

    t0 = time.monotonic()
    p0 = cliente.peticiones
    cobertura = _cobertura(tabla)
    lags = pd.concat([medir_publicacion(cliente, s) for s in series_lag], ignore_index=True)
    resumen_lags = _resumen_lags(lags)
    existen = {}
    for sid in sorted(set(tabla["serie"])):
        info = cliente.serie(sid.split(":", 1)[1])
        existen[sid] = {"observation_start": info.get("observation_start"),
                        "observation_end": info.get("observation_end"), "last_updated": info.get("last_updated"),
                        "last_updated_ny": last_updated_ny(info.get("last_updated"))}
    horas_oas = sorted({v["last_updated_ny"] for s, v in existen.items()
                        if s.split(":", 1)[1] in OAS_ICE and v["last_updated_ny"]})
    fin_oas = sorted({v["observation_end"] for s, v in existen.items()
                      if s.split(":", 1)[1] in OAS_ICE and v["observation_end"]})
    wei_v = cliente.vintages("WEI")
    wei = pd.Series(pd.to_datetime(wei_v))
    medicion_wei = {"n_vintages": len(wei_v), "primera": wei_v[0] if wei_v else None,
                    "ultima": wei_v[-1] if wei_v else None,
                    "dia_semana": {k: int(n) for k, n in wei.dt.day_name().value_counts().items()}}
    oas: dict[str, Any] = {}
    for s in ("BAMLH0A0HYM2", "BAMLC0A0CM"):
        vint = cliente.vintages(s)
        oas[s] = {"n_vintages": len(vint), "primer_vintage": vint[0] if vint else None,
                  f"vintage_{oas_vintage}": historial_vintage(cliente, s, oas_vintage)}
        for v in oas_previos:
            oas[s][f"vintage_{v}"] = historial_vintage(cliente, s, v)
        if vint:
            oas[s]["en_primer_vintage"] = historial_vintage(cliente, s, vint[0])
        # Todas las vintages en una llamada (primera publicación de cada observación).
        try:
            todo = parsear_observaciones(cliente.observaciones(s, realtime_start=INICIO_ALFRED, realtime_end=FIN_ALFRED,
                                                               output_type=4, reintentos=1))
            todo = todo[~todo["ausente"]]
            oas[s]["realtime_1776_9999_output_type_4"] = {
                "n_obs": int(len(todo)),
                "inicio": todo["fecha_referencia"].min().date().isoformat() if len(todo) else None}
        except ErrorFred as e:
            oas[s]["realtime_1776_9999_output_type_4"] = {"error": str(e)[:160]}
    mediciones_red = {"peticiones": cliente.peticiones - p0, "segundos": round(time.monotonic() - t0, 1)}
    hoy = cliente.hoy.date().isoformat()
    sids_fred = [s for s in cobertura if s.split(":", 1)[1] not in OAS_ICE]
    sids_oas = [s for s in cobertura if s.split(":", 1)[1] in OAS_ICE]
    # Volumen medido: tamaño del archivo crudo en disco (todas las descargas conservadas).
    raiz_crudo = cliente._carpeta()
    bytes_oas = sum(p.stat().st_size for s in OAS_ICE for p in (raiz_crudo / "series_observations" / s).glob("*.json"))
    bytes_todo = sum(p.stat().st_size for p in raiz_crudo.rglob("*.json"))
    dias_oas = sorted({p.name[:8] for s in OAS_ICE for p in (raiz_crudo / "series_observations" / s).glob("*.json")})
    carpeta = resultados_dir(TEMA) / "tablas"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_lags = carpeta / "publicacion_fred_sustitutos.csv"

    def _rango(sids: list[str]) -> tuple[str | None, str | None]:
        ini = [cobertura[s]["inicio"] for s in sids if cobertura[s]["inicio"]]
        fin = [cobertura[s]["fin"] for s in sids if cobertura[s]["fin"]]
        return (min(ini) if ini else None, max(fin) if fin else None)

    ini_f, fin_f = _rango(sids_fred)
    ficha_fred = FichaFuente(
        fuente=FUENTE, nombre="FRED / ALFRED (Federal Reserve Bank of St. Louis)", tema=TEMA,
        agentes=["macro", "risk", "portfolio"], rag="serie", url="https://fred.stlouisfed.org/docs/api/fred/",
        acceso="api", autenticacion="clave gratuita", coste="0 €",
        licencia_y_terminos=("Términos de uso de la API de FRED: datos solo en local, sin redistribuir; cada serie "
                             "conserva el copyright de su titular (ICE, Moody's, Cboe...). " + AVISO_FRED),
        cobertura_inicio=ini_f, cobertura_fin=fin_f, frecuencia="diaria y semanal (sustitutos del bloque 2)",
        retraso_publicacion=("medido en ALFRED (primera vintage de cada observación, días hábiles de EE. UU., "
                             "últimos 35 días): " + "; ".join(f"{s} {r}" for s, r in resumen_lags.items()
                                                              if s not in OAS_ICE)
                             + ". Hora (FRED last_updated en NY): "
                             + "; ".join(f"{s.split(':', 1)[1]} {v['last_updated_ny']}" for s, v in existen.items()
                                         if s.split(":", 1)[1] not in OAS_ICE and v["last_updated_ny"])
                             + ". OAS de ICE: ver la ficha ice_bofa_oas"),
        formato="json", volumen_estimado=(f"{(bytes_todo - bytes_oas) / 1e6:.1f} MB de JSON crudo en disco (todas las "
                                          f"peticiones archivadas, sin las OAS de ICE)"),
        limites=("120 peticiones/min por clave (aquí 1/s); series/observations hasta 100000 filas por página "
                 "(offset); vintage_dates hasta 2000 por petición según la API, pero con 2000 la URL (~26 KB) "
                 "da HTTP 400 en el CDN (medido): se trocea en grupos de 500; vintagedates y release/dates "
                 "hasta 10000 por página"),
        estado_sonda="ok",
        mediciones={"series": cobertura, "existen_en_fred": existen, "retrasos_habiles": resumen_lags,
                    "wei_vintages": medicion_wei, "red": mediciones_red,
                    "tabla_retrasos": ruta_lags.relative_to(ROOT).as_posix()},
        notas=[AVISO_FRED,
               "Caché cruda: un JSON por petición en <corpus>/mercados_series/crudo/fred/ (fecha de descarga en "
               "el nombre); la clave nunca se guarda",
               "Valores '.' de FRED: en las diarias, los sueltos (festivos) se descartan y cada racha de más de 31 "
               "días deja una fila nula con nota; en las demás frecuencias, valor nulo con nota",
               "Fechas de publicación de Moody's, diferenciales calculados por FRED, SOFR/EFFR/OBFR, papel comercial y "
               "la macro lag_fijo contrastadas con la primera vintage de ALFRED (output_type=4): "
               "tablas/publicacion_alfred_medida.csv y config/excepciones_publicacion.csv",
               "WEI se carga con todas sus vintages de ALFRED (vintage_exacto); el resto, vintage 'unico' "
               "(no revisables según el catálogo)"],
        fecha_sonda=hoy)
    ini_o, fin_o = _rango(sids_oas)
    ficha_oas = FichaFuente(
        fuente="ice_bofa_oas", nombre="ICE BofA: OAS y rendimientos de índices de crédito (vía FRED)", tema=TEMA,
        agentes=["risk", "macro"], rag="serie", url="https://fred.stlouisfed.org/series/BAMLH0A0HYM2",
        acceso="api", autenticacion="clave gratuita (FRED)", coste="0 €",
        licencia_y_terminos=("copyright ICE Data Indices, LLC; uso vía FRED solo en local, sin redistribuir. "
                             + AVISO_FRED),
        cobertura_inicio=ini_o, cobertura_fin=fin_o, frecuencia="diaria (días hábiles)",
        retraso_publicacion=(f"hábil siguiente hacia las 09:10 NY (FRED last_updated {', '.join(horas_oas)} con el "
                             f"dato de {', '.join(fin_oas)}): visible en la decisión de D+1. ALFRED fecha la "
                             "vintage el mismo día del dato (" + "; ".join(f"{s} {resumen_lags[s]}" for s in resumen_lags
                                                                          if s in OAS_ICE)
                             + "), antes de la carga real: realtime_start no sirve para medir la hora de ICE"),
        formato="json", volumen_estimado=(f"{bytes_oas / 1e6:.1f} MB de JSON crudo en disco ({len(sids_oas)} series, "
                                          f"{len(dias_oas)} día(s) de descarga archivados)"),
        limites=("FRED solo sirve desde el 2023-10-10 (unos 3 años); que sea una ventana móvil y no un corte fijo "
                 "está sin confirmar (se verá con la descarga siguiente). ALFRED no conserva el historial anterior: "
                 "archivo propio desde la primera descarga; para no perder historia basta una descarga dentro de "
                 "cada ventana, la diaria solo hace falta para tener el dato al día"),
        estado_sonda="parcial",
        mediciones={"series": {s: cobertura[s] for s in sids_oas}, "historial_alfred": oas,
                    "last_updated_ny": {s: existen[s]["last_updated_ny"] for s in sids_oas},
                    "primera_descarga_archivo": hoy},
        notas=["Capada: la historia propia empieza en la primera descarga del archivo; cada descarga diaria "
               "se conserva y tabla_fred_sustitutos une todas (primer valor visto por fecha)",
               f"Historial anterior: ALFRED en el vintage {oas_vintage} (por vintage_dates y por "
               "realtime_start/realtime_end) también empieza en el primer vintage conservado; los vintages "
               "anteriores dan error (ver mediciones.historial_alfred). Variante con todas las vintages en una "
               "llamada (realtime_start=1776-07-04, realtime_end=9999-12-31, output_type=4): empieza en "
               + ", ".join(f"{s} {v.get('realtime_1776_9999_output_type_4', {}).get('inicio')}" for s, v in oas.items())
               + ". No se puede recuperar por FRED",
               "Publicador ice_bofa_fred (09:15 NY del hábil siguiente, según last_updated): visible en la "
               "decisión de D+1",
               "Series archivadas: " + ", ".join(sorted(s.split(":", 1)[1] for s in sids_oas)),
               AVISO_FRED],
        fecha_sonda=hoy)
    rutas = {}
    if guardar:
        lags.to_csv(ruta_lags, index=False, encoding="utf-8")
        rutas = {"fred_alfred": guardar_ficha(ficha_fred), "ice_bofa_oas": guardar_ficha(ficha_oas),
                 "tabla_retrasos": ruta_lags}
    return {"fichas": (ficha_fred, ficha_oas), "lags": lags, "rutas": rutas}


# --- CLI ----------------------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Sustitutos de FRED del Tema 1 (bloque 2). " + AVISO_FRED)
    ap.add_argument("--ids", nargs="*", help="ids del catálogo (por defecto, ids_sustitutos())")
    ap.add_argument("--sin-red", action="store_true", help="solo el archivo crudo, sin descargar")
    ap.add_argument("--fichas", action="store_true", help="mide en ALFRED y escribe las fichas")
    ap.add_argument("--escribir", action="store_true", help="escribe la tabla en el almacén (un solo escritor)")
    a = ap.parse_args(argv)
    cliente = ClienteFred()
    t0 = time.monotonic()
    tabla = tabla_fred_sustitutos(a.ids, cliente=cliente, descargar=not a.sin_red)
    print(f"{len(tabla)} filas, {tabla['serie'].nunique()} series, {cliente.peticiones} peticiones, "
          f"{time.monotonic() - t0:.0f} s")
    for sid, c in _cobertura(tabla).items():
        print(f"  {sid}: {c['n_obs']} obs ({c['inicio']} a {c['fin']}), {c['nulos_punto']} '.', "
              f"{c['vintages']} vintages")
    if a.fichas:
        r = fichas_fred(cliente, tabla)
        print("fichas:", {k: str(v) for k, v in r["rutas"].items()})
    if a.escribir:
        from ingesta.mercados_series import almacen

        # Mismas fechas que la base: excepciones medidas en ALFRED (la vía completa es construir-base).
        tabla, _ = catalogo.aplicar_excepciones(tabla)
        print("almacén:", almacen.escribir_series(tabla))
    print(AVISO_FRED)


if __name__ == "__main__":
    main()
