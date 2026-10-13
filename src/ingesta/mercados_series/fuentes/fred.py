"""Extractor de FRED/ALFRED del Tema 1 (bloques 2 y 3): cliente, FRED como fuente viva y derivadas.

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

Fuente viva (bloque 3, ``tabla_fred_vivo``): todas las series del catálogo cuyo origen es FRED y que
NO van por ALFRED (``ids_fred_vivo``: ``catalogo.origenes`` con ``fred``; tipos diarios de H.15,
H.10, Moody's, NY Fed, papel comercial, cierres de Cboe/NASDAQ en FRED, mensuales de mercado...).
Se fechan con su regla (calendario, cierre o lag) y las excepciones medidas
(``catalogo.aplicar_excepciones``). Descarga reanudable (``descargar_fred_vivo``): una serie ya
descargada entera hoy no se pide otra vez; la primera descarga es completa y las siguientes
incrementales (desde ``SOLAPE_DIAS`` antes del último dato archivado); la tabla une todas las
descargas completas archivadas (``ClienteFred.descargas_vigentes``; las páginas de una descarga
interrumpida no cuentan hasta que se completa). ``data/raw`` queda para la Fase 1.

Derivadas (``tabla_derivadas``): diferencias ``A − B`` del catálogo (BAA10Y = DBAA − DGS10, T10YFF =
DGS10 − DFF, T5YFF = DGS5 − DFF, CPFF = DCPF3M − DFF) con ``contrato.fecha_publicacion_derivada``
(máxima de los componentes); ``medir_derivadas`` mide cuánto se separan de la serie que publica
FRED (su duplicada) y ``contrastar_fase1`` compara la fuente viva con ``data/raw``.

Datos solo en local (términos de FRED; ICE, Moody's: copyright). Al repositorio solo van código,
fichas y tablas de fechas, nunca valores.
CLI: ``python -m ingesta.mercados_series.fuentes.fred [--sin-red] [--fichas] [--escribir]`` (sustitutos
del bloque 2) y ``... --vivo [--ids ...] [--max-minutos N] [--completa] [--sin-red] [--fichas]``
(fuente viva por tandas: repetir la orden sigue donde se quedó).
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
    fecha_publicacion_derivada,
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
# Fuente viva (bloque 3): métodos de las series de FRED que no van por ALFRED.
METODOS_VIVO = ("calendario", "cierre_mercado", "lag_fijo")
# Descarga incremental: desde este margen antes del último dato archivado (un '.' que FRED rellena
# tarde, un dato de Moody's con retraso, la última mensual).
SOLAPE_DIAS = 62
# Parámetros que admite una descarga de lo vigente (sin realtime, vintage_dates ni output_type).
PARAMS_VIGENTE = frozenset({"series_id", "observation_start", "observation_end"})
# Derivada propia frente a la serie de FRED: valores distintos más allá del error de coma flotante.
TOLERANCIA_DERIVADA = 1e-6
SERIES_LAG_VIVO = ("DGS10", "DFF", "SOFR", "DBAA", "DCPF3M", "DEXUSEU", "VIXCLS", "BAMLH0A0HYM2", "BAMLC0A0CM")

Transporte = Callable[[str, dict[str, Any]], dict[str, Any]]

__all__ = [
    "API_FRED", "AVISO_FRED", "MAX_VINTAGES", "NOTA_PUNTO", "OAS_ICE", "ErrorFred", "ClienteFred",
    "ruta_crudo", "trozos", "parsear_observaciones", "ids_sustitutos", "tabla_serie", "tabla_fred_sustitutos",
    "medir_publicacion", "historial_vintage", "last_updated_ny", "fichas_fred", "ids_excepciones",
    "primeras_publicaciones", "medir_excepciones",
    "METODOS_VIVO", "SOLAPE_DIAS", "ids_fred_vivo", "descargar_vigente", "descargar_fred_vivo", "tabla_fred_vivo",
    "pares_derivadas", "tabla_derivadas", "medir_derivadas", "contrastar_fase1",
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

    def ficheros_vigentes(self, series_id: str) -> list[tuple[Path, dict[str, Any]]]:
        """``[(fichero, envoltorio)]`` de las páginas archivadas de lo vigente de ``series_id``.

        Lo vigente: ``series/observations`` sin ``realtime_*``, ``vintage_dates`` ni ``output_type``
        (solo ``observation_start``/``observation_end`` además del id). Sin red.
        """
        carpeta = self.carpeta_peticion("series/observations", {"series_id": series_id})
        out = []
        for f in sorted(carpeta.glob("*.json")) if carpeta.exists() else []:
            env_ = json.loads(f.read_text(encoding="utf-8"))
            p = {k: v for k, v in env_["params"].items() if k not in ("offset", "limit")}
            if p.get("series_id") == series_id and set(p) <= PARAMS_VIGENTE:
                out.append((f, env_))
        return out

    def descargas_vigentes(self, series_id: str) -> list[tuple[str, dict[str, str], list[dict[str, Any]]]]:
        """Descargas COMPLETAS archivadas de lo vigente: ``[(AAAAMMDD, params, observaciones)]`` por día.

        Junta las páginas de cada petición (mismo día y parámetros salvo ``offset``); una petición
        cuyas páginas no empiezan en 0, tienen saltos o no llegan al ``count`` de la API (descarga
        interrumpida) no se devuelve hasta que se completa. Completas e incrementales
        (``observation_start``). Sin red.
        """
        grupos: dict[tuple[str, str], dict[str, Any]] = {}
        for f, env_ in self.ficheros_vigentes(series_id):
            p = {k: v for k, v in env_["params"].items() if k not in ("offset", "limit")}
            g = grupos.setdefault((f.name[:8], json.dumps(p, sort_keys=True)), {"params": p, "paginas": {}, "count": 0})
            resp = env_["respuesta"]
            lote = resp.get("observations") or []
            g["paginas"][int(env_["params"].get("offset", 0))] = lote
            g["count"] = max(g["count"], int(resp.get("count", len(lote))))
        out = []
        for (dia, _), g in sorted(grupos.items()):
            n = 0
            for off, lote in sorted(g["paginas"].items()):
                if off != n:
                    break
                n += len(lote)
            if n >= g["count"]:
                out.append((dia, g["params"], [o for _, lote in sorted(g["paginas"].items()) for o in lote]))
        return out


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


def ids_fred_vivo(cat: dict[str, dict] | None = None) -> list[str]:
    """Series que FRED sirve como fuente viva y NO van por ALFRED (bloque 3).

    Estado cargable, ``fred`` entre sus ``catalogo.origenes`` (``fuente_viva: fred`` y método
    distinto de ``vintage_exacto``) y método de ``METODOS_VIVO``. Incluye las ``duplicada`` de FRED
    (BAA10Y, T10YFF, T5YFF, CPFF: contraste de las derivadas propias) y las retiradas que FRED
    aún sirve. Las ``vintage_exacto`` (macro revisable, WEI) las carga ALFRED.
    """
    cat = cat if cat is not None else catalogo.cargar_catalogo()
    return sorted(sid for sid, e in cat.items()
                  if e["estado"] in catalogo.ESTADOS_CARGABLES and e["metodo"] in METODOS_VIVO
                  and "fred" in catalogo.origenes(e))


def ids_sustitutos(cat: dict[str, dict] | None = None) -> list[str]:
    """Series de la fuente viva de FRED que NO están en data/raw (``ids_fred_vivo`` sin ``fichero_fase1``).

    Hoy: Moody's diarios, H.10 (DEXUSEU, DTWEXBGS, DTWEXAFEGS), financiación (TEDRATE, DCPF3M, CPFF)
    y las OAS de ICE BofA. Desde el bloque 3 WEI (``vintage_exacto``) ya no está: va por ALFRED.
    """
    cat = cat if cat is not None else catalogo.cargar_catalogo()
    return [sid for sid in ids_fred_vivo(cat) if not cat[sid]["fichero_fase1"]]


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
        descargar_vigente(cliente, origen)
    return unir_archivo([(dia, obs) for dia, _, obs in cliente.descargas_vigentes(origen)])


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


# --- Fuente viva (bloque 3) ---------------------------------------------------------------------
def _ultimo_dato(descargas: Sequence[tuple[str, dict[str, str], list[dict[str, Any]]]]) -> pd.Timestamp | None:
    fechas = [o["date"] for _, _, obs in descargas for o in obs if str(o.get("value", ".")).strip() not in (".", "")]
    return pd.Timestamp(max(fechas)) if fechas else None


def descargar_vigente(cliente: ClienteFred, series_id: str, *, completa: bool = False,
                      solape_dias: int = SOLAPE_DIAS) -> dict[str, Any]:
    """Descarga lo vigente de ``series_id`` si hace falta; devuelve qué hizo (sin valores).

    - ``cache``: ya hay hoy una descarga completa archivada (con ``completa=True``, una sin
      ``observation_start``); no sale ninguna petición. Con ``cliente.cache=False`` siempre se pide.
    - ``completa``: sin archivo previo (o ``completa=True``): toda la serie.
    - ``incremental``: desde ``solape_dias`` antes del último dato con valor archivado.

    Reanudable: una descarga paginada interrumpida deja sus páginas en la caché del día y, al
    repetirla, solo se piden las que faltan (los parámetros son los mismos porque la descarga
    incompleta no cuenta para el último dato). ``ErrorFred`` si falla la red.
    """
    previas = cliente.descargas_vigentes(series_id)
    hoy = f"{cliente.hoy:%Y%m%d}"
    de_hoy = [p for d, p, _ in previas if d == hoy and (not completa or set(p) == {"series_id"})]
    p0 = cliente.peticiones
    if cliente.cache and de_hoy:
        return {"serie": series_id, "modo": "cache", "desde": None, "n_obs": None, "peticiones": 0}
    ultimo = None if completa else _ultimo_dato([x for x in previas if x[0] != hoy] or previas)
    if ultimo is None:
        obs = cliente.observaciones(series_id)
        modo, desde = "completa", None
    else:
        desde = (ultimo - pd.Timedelta(days=solape_dias)).date().isoformat()
        obs = cliente.observaciones(series_id, observation_start=desde)
        modo = "incremental"
    return {"serie": series_id, "modo": modo, "desde": desde, "n_obs": len(obs), "peticiones": cliente.peticiones - p0}


def descargar_fred_vivo(ids: Iterable[str] | None = None, *, cliente: ClienteFred | None = None,
                        cat: dict[str, dict] | None = None, completa: bool = False,
                        max_segundos: float | None = None,
                        registro: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Descarga por tandas la fuente viva (``ids``: por defecto ``ids_fred_vivo``).

    Serie a serie con ``descargar_vigente``; un fallo de red queda en ``errores`` y se sigue. Con
    ``max_segundos``, al agotarse el tiempo las series que faltan quedan en ``pendientes`` (repetir
    la llamada sigue donde se quedó: lo de hoy sale de la caché). Devuelve ``{hechas, cache,
    errores, pendientes, peticiones, segundos, detalle}`` (solo recuentos y fechas).
    """
    cat = cat if cat is not None else catalogo.cargar_catalogo()
    lista = [catalogo.resolver(s, cat) for s in (ids if ids is not None else ids_fred_vivo(cat))]
    cliente = cliente or ClienteFred()
    log = registro or (lambda _m: None)
    t0, p0 = time.monotonic(), cliente.peticiones
    hechas, en_cache, pendientes, detalle = [], [], [], []
    errores: dict[str, str] = {}
    for sid in lista:
        if max_segundos is not None and time.monotonic() - t0 > max_segundos:
            pendientes.append(sid)
            continue
        try:
            r = descargar_vigente(cliente, _id_origen(sid, cat[sid]), completa=completa)
        except ErrorFred as e:
            errores[sid] = str(e)[:200]
            log(f"  {sid}: error {str(e)[:120]}")
            continue
        detalle.append({"id": sid, **r})
        (en_cache if r["modo"] == "cache" else hechas).append(sid)
        if r["modo"] != "cache":
            log(f"  {sid}: {r['modo']}" + (f" desde {r['desde']}" if r["desde"] else "")
                + f", {r['n_obs']} obs, {r['peticiones']} peticiones")
    return {"hechas": hechas, "cache": en_cache, "errores": errores, "pendientes": pendientes,
            "peticiones": cliente.peticiones - p0, "segundos": round(time.monotonic() - t0, 1), "detalle": detalle}


def tabla_fred_vivo(ids: Iterable[str] | None = None, sin_red: bool = False, *, cliente: ClienteFred | None = None,
                    cat: dict[str, dict] | None = None, publicaciones: dict[str, Any] | None = None,
                    excepciones: pd.DataFrame | None = None, estricto: bool = True,
                    registro: Callable[[str], None] | None = None) -> pd.DataFrame:
    """Tabla del contrato (validada) de la fuente viva de FRED, con las excepciones medidas aplicadas.

    ``ids``: por defecto ``ids_fred_vivo()``; una serie que no es de la fuente viva (``vintage_exacto``,
    otra fuente) es ``ValueError``. Sin ``sin_red``, primero ``descargar_fred_vivo`` (lo de hoy sale
    de la caché); con ``estricto`` un fallo de red es ``ErrorFred`` (sin la clave), y sin él la
    serie se carga con lo que haya archivado. Cada serie: unión de sus descargas completas
    archivadas (primer valor visto por fecha), ``tabla_serie`` con su regla y después
    ``catalogo.aplicar_excepciones`` (``excepciones``: por defecto ``config/excepciones_publicacion.csv``).
    Una serie sin nada archivado no aparece (la base la cuenta en ``sin_datos``).
    """
    cat = cat if cat is not None else catalogo.cargar_catalogo()
    vivas = set(ids_fred_vivo(cat))
    lista = [catalogo.resolver(s, cat) for s in (ids if ids is not None else sorted(vivas))]
    ajenas = [s for s in lista if s not in vivas]
    if ajenas:
        raise ValueError(f"no son de la fuente viva de FRED (ids_fred_vivo): {ajenas[:10]}")
    cliente = cliente or ClienteFred()
    log = registro or (lambda _m: None)
    if not sin_red:
        rep = descargar_fred_vivo(lista, cliente=cliente, cat=cat, registro=log)
        log(f"FRED vivo: {len(rep['hechas'])} descargadas, {len(rep['cache'])} de la caché, "
            f"{len(rep['errores'])} con error ({rep['peticiones']} peticiones, {rep['segundos']} s)")
        if rep["errores"] and estricto:
            sid, err = next(iter(rep["errores"].items()))
            raise ErrorFred(f"{len(rep['errores'])} series sin descargar ({sorted(rep['errores'])[:10]}); {sid}: {err}")
    tabla = tabla_fred_sustitutos(lista, cliente=cliente, descargar=False, cat=cat, publicaciones=publicaciones)
    tabla, n = catalogo.aplicar_excepciones(tabla, excepciones, publicaciones=publicaciones, catalogo=cat)
    log(f"FRED vivo: {tabla['serie'].nunique()} series, {len(tabla)} filas, {n} con excepción medida")
    return validar_series(tabla)


# --- Derivadas de diferencia (bloque 3) ---------------------------------------------------------
def _es_diferencia(entrada: dict[str, Any]) -> bool:
    comp = entrada.get("componentes") or []
    return (entrada["metodo"] == "derivada" and len(comp) == 2
            and str(entrada.get("id_origen", "")).replace(" ", "") == f"{comp[0]}-{comp[1]}")


def pares_derivadas(cat: dict[str, dict] | None = None) -> dict[str, str]:
    """``{derivada propia: serie de FRED que publica la misma diferencia}``.

    Las ``fred:`` en estado ``duplicada`` cuyo ``duplicado_de`` es una derivada ``A − B`` del catálogo:
    hoy ``deriv:DBAA-DGS10`` ↔ ``fred:BAA10Y``, ``deriv:DGS10-DFF`` ↔ ``fred:T10YFF``,
    ``deriv:DGS5-DFF`` ↔ ``fred:T5YFF`` y ``deriv:DCPF3M-DFF`` ↔ ``fred:CPFF``.
    """
    cat = cat if cat is not None else catalogo.cargar_catalogo()
    out = {}
    for sid, e in sorted(cat.items()):
        d = e.get("duplicado_de")
        if sid.startswith("fred:") and e["estado"] == "duplicada" and d in cat and _es_diferencia(cat[d]):
            out[d] = sid
    return out


def tabla_derivadas(tabla: pd.DataFrame, ids: Iterable[str] | None = None, *, cat: dict[str, dict] | None = None,
                    omitir_incompletas: bool = False) -> pd.DataFrame:
    """Filas del contrato (validadas) de las derivadas ``A − B`` calculadas con los componentes de ``tabla``.

    ``ids``: por defecto las de ``pares_derivadas``. Valor ``A − B`` en las fechas en que los dos
    componentes tienen valor; ``fecha_publicacion`` y ``disponible_utc`` con
    ``contrato.fecha_publicacion_derivada`` (la máxima de las primeras publicaciones de los
    componentes, ya con sus excepciones medidas). Igual que ``base.derivar_diferencias``.
    Componente sin datos: ``KeyError`` (o se omite la derivada con ``omitir_incompletas``);
    componentes con vintages que no son ``unico``: ``ValueError``.
    """
    cat = cat if cat is not None else catalogo.cargar_catalogo()
    lista = list(ids) if ids is not None else list(pares_derivadas(cat))
    partes = []
    for sid in lista:
        e = cat[sid]
        if not _es_diferencia(e):
            raise ValueError(f"{sid}: no es una derivada de diferencia 'A - B' del catálogo")
        a, b = e["componentes"]
        comp = tabla[tabla["serie"].isin([a, b]) & tabla["valor"].notna()]
        faltan = sorted({a, b} - set(comp["serie"]))
        if faltan:
            if omitir_incompletas:
                continue
            raise KeyError(f"{sid}: componentes sin datos {faltan}")
        if (comp["vintage"] != "unico").any():
            raise ValueError(f"{sid}: componentes con vintages (la derivada por vintage no se calcula aquí)")
        fechado = fecha_publicacion_derivada(comp, [a, b]).set_index("fecha_referencia")
        va = comp[comp["serie"] == a].set_index("fecha_referencia")["valor"].reindex(fechado.index)
        vb = comp[comp["serie"] == b].set_index("fecha_referencia")["valor"].reindex(fechado.index)
        partes.append(pd.DataFrame({
            "serie": sid, "fecha_referencia": fechado.index, "fecha_publicacion": fechado["fecha_publicacion"].to_numpy(),
            "disponible_utc": fechado["disponible_utc"].to_numpy(), "vintage": "unico", "valor": (va - vb).to_numpy(),
            "unidad": e["unidad"], "divisa": e["divisa"], "fuente": e["fuente"], "metodo_publicacion": e["metodo"],
            "nota": catalogo.nota_revisable(e) if e["revisable"] else None,
        })[list(COLUMNAS_SERIES)])
    out = validar_series(pd.concat(partes, ignore_index=True)) if partes else vacio_series()
    catalogo.validar_contra_catalogo(out, cat)
    return out


def _comparar(x: pd.DataFrame, y: pd.DataFrame) -> dict[str, Any]:
    """Compara dos series del contrato (vintage ``unico``) por fecha: recuentos y diferencia máxima."""
    cx = x[x["valor"].notna()].drop_duplicates("fecha_referencia").set_index("fecha_referencia")
    cy = y[y["valor"].notna()].drop_duplicates("fecha_referencia").set_index("fecha_referencia")
    comunes = cx.index.intersection(cy.index)
    dif = (cx.loc[comunes, "valor"] - cy.loc[comunes, "valor"]).abs()
    adelanto = (cy.loc[comunes, "fecha_publicacion"] - cx.loc[comunes, "fecha_publicacion"]).dt.days
    return {"fechas_comunes": int(len(comunes)), "valores_distintos": int((dif > TOLERANCIA_DERIVADA).sum()),
            "dif_max": round(float(dif.max()), 10) if len(dif) else None,
            "fecha_dif_max": dif.idxmax().date().isoformat() if len(dif) and dif.max() > TOLERANCIA_DERIVADA else None,
            "solo_x": int(len(cx.index.difference(cy.index))), "solo_y": int(len(cy.index.difference(cx.index))),
            "publicacion_distinta": int((adelanto != 0).sum()),
            "adelanto_mediano_dias": float(adelanto.median()) if len(adelanto) else None,
            "x_posterior": int((adelanto < 0).sum())}


def medir_derivadas(tabla: pd.DataFrame, derivadas: pd.DataFrame | None = None, *,
                    cat: dict[str, dict] | None = None) -> pd.DataFrame:
    """Contraste de cada derivada propia con la serie de FRED (``pares_derivadas``): solo recuentos.

    ``tabla``: con los componentes y la serie de FRED (fuente viva); ``derivadas``: por defecto
    ``tabla_derivadas(tabla)`` (las que tengan componentes). Columnas: ``derivada, serie_fred,
    fechas_comunes, valores_distintos`` (más de ``TOLERANCIA_DERIVADA``), ``dif_max``,
    ``fecha_dif_max``, ``solo_derivada``, ``solo_fred``, ``publicacion_distinta``,
    ``adelanto_mediano_dias`` (fecha de FRED − fecha de la derivada) y ``derivada_posterior`` (fechas
    en que la derivada se vería DESPUÉS que la de FRED).
    """
    cat = cat if cat is not None else catalogo.cargar_catalogo()
    pares = pares_derivadas(cat)
    if derivadas is None:
        derivadas = tabla_derivadas(tabla, cat=cat, omitir_incompletas=True)
    filas = []
    for d, f in pares.items():
        x, y = derivadas[derivadas["serie"] == d], tabla[tabla["serie"] == f]
        if x.empty or y.empty:
            continue
        m = _comparar(x, y)
        filas.append({"derivada": d, "serie_fred": f, "fechas_comunes": m["fechas_comunes"],
                      "valores_distintos": m["valores_distintos"], "dif_max": m["dif_max"],
                      "fecha_dif_max": m["fecha_dif_max"], "solo_derivada": m["solo_x"], "solo_fred": m["solo_y"],
                      "publicacion_distinta": m["publicacion_distinta"],
                      "adelanto_mediano_dias": m["adelanto_mediano_dias"], "derivada_posterior": m["x_posterior"]})
    return pd.DataFrame(filas, columns=["derivada", "serie_fred", "fechas_comunes", "valores_distintos", "dif_max",
                                        "fecha_dif_max", "solo_derivada", "solo_fred", "publicacion_distinta",
                                        "adelanto_mediano_dias", "derivada_posterior"])


def contrastar_fase1(vivo: pd.DataFrame, fase1: pd.DataFrame) -> pd.DataFrame:
    """Fuente viva de FRED frente a la copia de ``data/raw`` (``fuentes.fase1.tabla_fase1``), por serie.

    Solo metadatos: inicio y fin de cada una, fechas comunes, valores distintos (más de
    ``TOLERANCIA_DERIVADA``), diferencia máxima y su fecha, fechas solo en FRED o solo en ``data/raw``
    y fechas de publicación distintas en las comunes.
    """
    filas = []
    for sid in sorted(set(vivo["serie"]) & set(fase1["serie"])):
        x, y = vivo[vivo["serie"] == sid], fase1[fase1["serie"] == sid]
        m = _comparar(x, y)
        vx, vy = x[x["valor"].notna()]["fecha_referencia"], y[y["valor"].notna()]["fecha_referencia"]
        filas.append({"serie": sid, "inicio_fred": _iso(vx.min()), "fin_fred": _iso(vx.max()),
                      "inicio_fase1": _iso(vy.min()), "fin_fase1": _iso(vy.max()),
                      "fechas_comunes": m["fechas_comunes"], "valores_distintos": m["valores_distintos"],
                      "dif_max": m["dif_max"], "fecha_dif_max": m["fecha_dif_max"], "solo_fred": m["solo_x"],
                      "solo_fase1": m["solo_y"], "publicacion_distinta": m["publicacion_distinta"]})
    return pd.DataFrame(filas, columns=["serie", "inicio_fred", "fin_fred", "inicio_fase1", "fin_fase1", "fechas_comunes",
                                        "valores_distintos", "dif_max", "fecha_dif_max", "solo_fred", "solo_fase1",
                                        "publicacion_distinta"])


def _iso(valor: Any) -> str | None:
    return None if pd.isna(valor) else pd.Timestamp(valor).date().isoformat()


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


def _bytes_vigentes(cliente: ClienteFred, series_ids: Iterable[str]) -> tuple[int, list[str]]:
    """Bytes en disco de las descargas archivadas de lo vigente de ``series_ids`` y días de descarga."""
    total, dias = 0, set()
    for s in series_ids:
        for f, _ in cliente.ficheros_vigentes(s):
            total += f.stat().st_size
            dias.add(f.name[:8])
    return total, sorted(dias)


def fichas_fred(cliente: ClienteFred, tabla: pd.DataFrame, *, oas_vintage: str = "2026-03-31",
                oas_previos: Sequence[str] = ("2019-01-02",), series_lag: Sequence[str] = SERIES_LAG_VIVO,
                derivadas: pd.DataFrame | None = None, contraste_fase1: pd.DataFrame | None = None,
                descarga: dict[str, Any] | None = None, medir_oas: bool = True, cat: dict[str, dict] | None = None,
                guardar: bool = True) -> dict[str, Any]:
    """Mide y escribe las fichas ``fred_alfred`` (fuente viva) e ``ice_bofa_oas`` (y sus tablas).

    ``tabla``: la de ``tabla_fred_vivo`` (cobertura medida por serie). Mide en ALFRED la primera
    vintage de las últimas observaciones de ``series_lag`` (``medir_publicacion``), la hora de
    carga de cada serie (``fred/series`` ``last_updated`` en NY) y, con ``medir_oas``, el historial
    de las OAS en ``oas_vintage``, en vintages anteriores a la ventana (``oas_previos``) y en su
    primer vintage conservado. ``derivadas`` (``medir_derivadas``), ``contraste_fase1``
    (``contrastar_fase1``) y ``descarga`` (``descargar_fred_vivo``) se guardan en las mediciones y,
    con ``guardar``, en ``tablas/derivadas_fred_medida.csv`` y ``tablas/fred_vivo_contraste_fase1.csv``.
    Solo fechas y recuentos (la diferencia máxima de las derivadas es una medida, no un valor).
    """
    from ingesta.comun.esquema import FichaFuente
    from ingesta.comun.fichas import guardar_ficha

    cat = cat if cat is not None else catalogo.cargar_catalogo()
    t0 = time.monotonic()
    p0 = cliente.peticiones
    cobertura = _cobertura(tabla)
    lags = pd.concat([medir_publicacion(cliente, s) for s in series_lag], ignore_index=True)
    resumen_lags = _resumen_lags(lags)
    existen = {}
    for sid in sorted(set(tabla["serie"])):
        info = cliente.serie(_id_origen(sid, cat.get(sid, {})))
        existen[sid] = {"observation_start": info.get("observation_start"),
                        "observation_end": info.get("observation_end"), "last_updated": info.get("last_updated"),
                        "last_updated_ny": last_updated_ny(info.get("last_updated"))}
    es_oas = {s: s.split(":", 1)[1] in OAS_ICE for s in cobertura}
    horas_oas = sorted({v["last_updated_ny"] for s, v in existen.items() if es_oas.get(s) and v["last_updated_ny"]})
    fin_oas = sorted({v["observation_end"] for s, v in existen.items() if es_oas.get(s) and v["observation_end"]})
    oas: dict[str, Any] = {}
    for s in (("BAMLH0A0HYM2", "BAMLC0A0CM") if medir_oas else ()):
        vint = cliente.vintages(s)
        oas[s] = {"n_vintages": len(vint), "primer_vintage": vint[0] if vint else None,
                  f"vintage_{oas_vintage}": historial_vintage(cliente, s, oas_vintage)}
        for v in oas_previos:
            oas[s][f"vintage_{v}"] = historial_vintage(cliente, s, v)
        if vint:
            oas[s]["en_primer_vintage"] = historial_vintage(cliente, s, vint[0])
        try:   # todas las vintages en una llamada (primera publicación de cada observación)
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
    sids_fred = [s for s in cobertura if not es_oas[s]]
    sids_oas = [s for s in cobertura if es_oas[s]]
    bytes_fred, dias_fred = _bytes_vigentes(cliente, [_id_origen(s, cat.get(s, {})) for s in sids_fred])
    bytes_oas, dias_oas = _bytes_vigentes(cliente, [s.split(":", 1)[1] for s in sids_oas])
    carpeta = resultados_dir(TEMA) / "tablas"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_lags = carpeta / "publicacion_fred_sustitutos.csv"
    ruta_deriv = carpeta / "derivadas_fred_medida.csv"
    ruta_contraste = carpeta / "fred_vivo_contraste_fase1.csv"

    def _rango(sids: list[str]) -> tuple[str | None, str | None]:
        ini = [cobertura[s]["inicio"] for s in sids if cobertura[s]["inicio"]]
        fin = [cobertura[s]["fin"] for s in sids if cobertura[s]["fin"]]
        return (min(ini) if ini else None, max(fin) if fin else None)

    def _registros(df: pd.DataFrame | None) -> list[dict[str, Any]] | None:
        return None if df is None else json.loads(df.to_json(orient="records", force_ascii=False))

    frecuencias = {k: int(n) for k, n in pd.Series([cat[s]["frecuencia"] for s in sids_fred if s in cat],
                                                   dtype=object).value_counts().items()}
    metodos = {k: int(n) for k, n in pd.Series([cat[s]["metodo"] for s in sids_fred if s in cat],
                                               dtype=object).value_counts().items()}
    texto_deriv = ""
    if derivadas is not None and len(derivadas):
        texto_deriv = (". Derivadas propias frente a la serie de FRED (fechas comunes / distintas / dif. máx.): "
                       + "; ".join(f"{r.derivada} vs {r.serie_fred.split(':', 1)[1]} {r.fechas_comunes} / "
                                   f"{r.valores_distintos} / {r.dif_max}" for r in derivadas.itertuples(index=False)))
    ini_f, fin_f = _rango(sids_fred)
    ficha_fred = FichaFuente(
        fuente=FUENTE, nombre="FRED / ALFRED (Federal Reserve Bank of St. Louis)", tema=TEMA,
        agentes=["macro", "risk", "portfolio"], rag="serie", url="https://fred.stlouisfed.org/docs/api/fred/",
        acceso="api", autenticacion="clave gratuita", coste="0 €",
        licencia_y_terminos=("Términos de uso de la API de FRED: datos solo en local, sin redistribuir; cada serie "
                             "conserva el copyright de su titular (ICE, Moody's, Cboe...). " + AVISO_FRED),
        cobertura_inicio=ini_f, cobertura_fin=fin_f,
        frecuencia=("fuente viva (bloque 3), sin las OAS de ICE: "
                    + ", ".join(f"{k} {n}" for k, n in frecuencias.items()) + " series"),
        retraso_publicacion=("medido en ALFRED (primera vintage de cada observación, días hábiles de EE. UU., "
                             "últimos 35 días): " + "; ".join(f"{s} {r}" for s, r in resumen_lags.items()
                                                              if s not in OAS_ICE)
                             + ". Hora de carga en FRED (last_updated en NY): "
                             + "; ".join(f"{s.split(':', 1)[1]} {v['last_updated_ny']}" for s, v in existen.items()
                                         if not es_oas.get(s) and v["last_updated_ny"])
                             + ". Se fecha con la regla del catálogo (publicador, cierre o lag) y las excepciones "
                               "medidas; OAS de ICE: ver la ficha ice_bofa_oas" + texto_deriv),
        formato="json", volumen_estimado=(f"{bytes_fred / 1e6:.1f} MB de JSON crudo en disco (descargas de lo vigente "
                                          f"archivadas de {len(sids_fred)} series, {len(dias_fred)} día(s); sin las "
                                          f"OAS de ICE ni las vintages de ALFRED)"),
        limites=("120 peticiones/min por clave (aquí 1/s); series/observations hasta 100000 filas por página "
                 "(offset); vintage_dates hasta 2000 por petición según la API, pero con 2000 la URL (~26 KB) "
                 "da HTTP 400 en el CDN (medido): se trocea en grupos de 500; vintagedates y release/dates "
                 "hasta 10000 por página"),
        estado_sonda="ok",
        mediciones={"series": cobertura, "existen_en_fred": existen, "retrasos_habiles": resumen_lags,
                    "por_metodo": metodos, "por_frecuencia": frecuencias, "red": mediciones_red,
                    "descarga": ({k: v for k, v in descarga.items() if k != "detalle"} if descarga else None),
                    "derivadas": _registros(derivadas), "contraste_fase1": _registros(contraste_fase1),
                    "tabla_retrasos": ruta_lags.relative_to(ROOT).as_posix()},
        notas=[AVISO_FRED,
               "Fuente viva (bloque 3): las series de FRED que no van por ALFRED se descargan de la API y FRED pasa "
               "a ser su fuente; data/raw queda solo para la Fase 1 (y para lo que no está en FRED). Prioridad en la "
               "base: alfred (vintage_exacto) > fred vivo > fase1",
               "Descarga reanudable: la primera de cada serie es completa y las siguientes incrementales (desde "
               f"{SOLAPE_DIAS} días antes del último dato); la tabla une todas las descargas completas archivadas "
               "(primer valor visto por fecha). Caché cruda: un JSON por página en <corpus>/mercados_series/crudo/"
               "fred/ (fecha de descarga en el nombre); la clave nunca se guarda",
               "Valores '.' de FRED: en las diarias, los sueltos (festivos) se descartan y cada racha de más de 31 "
               "días deja una fila nula con nota; en las demás frecuencias, valor nulo con nota",
               "Fechas de publicación de Moody's, diferenciales calculados por FRED, SOFR/EFFR/OBFR, papel comercial y "
               "las mensuales lag_fijo contrastadas con la primera vintage de ALFRED (output_type=4): "
               "tablas/publicacion_alfred_medida.csv y config/excepciones_publicacion.csv (se aplican al cargar)",
               "BAA10Y, T10YFF, T5YFF y CPFF: la serie principal es la derivada propia de sus componentes (fecha = "
               "máxima de los componentes, D+1); la de FRED (D+2) queda como duplicada para contrastar: "
               "tablas/derivadas_fred_medida.csv",
               "Las series revisables de FRED (vintage_exacto) se cargan con sus vintages de ALFRED en "
               "fuentes.alfred (ficha alfred_vintages; caché propia en crudo/alfred/)"],
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
                                          f"{len(dias_oas)} día(s) de descarga archivados: {', '.join(dias_oas)})"),
        limites=("FRED solo sirve unos 3 años (ventana móvil). ALFRED no conserva el historial anterior: archivo "
                 "propio desde la primera descarga (2026-10-12); para no perder historia basta una descarga "
                 "(incremental) dentro de cada ventana, la diaria solo hace falta para tener el dato al día. El "
                 "archivo periódico está pendiente de programar"),
        estado_sonda="parcial",
        mediciones={"series": {s: cobertura[s] for s in sids_oas}, "historial_alfred": oas,
                    "last_updated_ny": {s: existen[s]["last_updated_ny"] for s in sids_oas},
                    "dias_archivados": dias_oas},
        notas=["Capada: la historia propia empieza en la primera descarga del archivo; cada descarga se conserva y "
               "tabla_fred_vivo une todas (primer valor visto por fecha); las siguientes a la primera son "
               "incrementales",
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
        if derivadas is not None:
            derivadas.to_csv(ruta_deriv, index=False, encoding="utf-8")
            rutas["derivadas"] = ruta_deriv
        if contraste_fase1 is not None:
            contraste_fase1.to_csv(ruta_contraste, index=False, encoding="utf-8")
            rutas["contraste_fase1"] = ruta_contraste
    return {"fichas": (ficha_fred, ficha_oas), "lags": lags, "rutas": rutas}


# --- CLI ----------------------------------------------------------------------------------------
def _vivo(a: argparse.Namespace, cliente: ClienteFred) -> int:
    """``--vivo``: descarga por tandas, tabla, derivadas y (``--fichas``) contraste con data/raw y fichas."""
    cat = catalogo.cargar_catalogo()
    rep = None
    if not a.sin_red:
        rep = descargar_fred_vivo(a.ids, cliente=cliente, cat=cat, completa=a.completa,
                                  max_segundos=a.max_minutos * 60 if a.max_minutos else None, registro=print)
        print(f"descarga: {len(rep['hechas'])} hechas, {len(rep['cache'])} ya de hoy, {len(rep['errores'])} con error, "
              f"{len(rep['pendientes'])} pendientes ({rep['peticiones']} peticiones, {rep['segundos']} s)")
        if rep["errores"]:
            print("errores (repetir la orden los reintenta):", rep["errores"])
            return 1
        if rep["pendientes"]:
            print(f"quedan {len(rep['pendientes'])} series: repite la orden para seguir (lo de hoy sale de la caché)")
            return 2
    tabla = tabla_fred_vivo(a.ids, sin_red=True, cliente=cliente, cat=cat, registro=print)
    for sid, c in _cobertura(tabla).items():
        print(f"  {sid}: {c['n_obs']} obs ({c['inicio']} a {c['fin']}), {c['nulos_punto']} nulos")
    deriv = medir_derivadas(tabla, cat=cat)
    if len(deriv):
        print("derivadas propias frente a FRED:")
        print(deriv.to_string(index=False))
    if a.fichas:
        from ingesta.mercados_series.fuentes import fase1

        sub = {s: cat[s] for s in set(tabla["serie"]) if cat[s]["fichero_fase1"]}
        contraste = contrastar_fase1(tabla, fase1.tabla_fase1(catalogo=sub)) if sub else None
        r = fichas_fred(cliente, tabla, derivadas=deriv, contraste_fase1=contraste, descarga=rep, cat=cat)
        print("fichas:", {k: str(v) for k, v in r["rutas"].items()})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FRED del Tema 1: sustitutos (bloque 2) y fuente viva (--vivo, "
                                             "bloque 3). " + AVISO_FRED)
    ap.add_argument("--ids", nargs="*", help="ids del catálogo (por defecto ids_sustitutos(), o ids_fred_vivo() con "
                                             "--vivo)")
    ap.add_argument("--sin-red", action="store_true", help="solo el archivo crudo, sin descargar")
    ap.add_argument("--fichas", action="store_true", help="mide en ALFRED y escribe las fichas")
    ap.add_argument("--escribir", action="store_true", help="escribe la tabla en el almacén (un solo escritor; no "
                                                            "con --vivo: la base la escribe construir-base)")
    ap.add_argument("--vivo", action="store_true", help="fuente viva: todas las series de FRED que no van por ALFRED")
    ap.add_argument("--max-minutos", type=float, default=None,
                    help="--vivo: corta la descarga tras N minutos (repetir la orden sigue donde se quedó)")
    ap.add_argument("--completa", action="store_true", help="--vivo: descarga completa aunque haya archivo previo")
    a = ap.parse_args(argv)
    if a.vivo and a.escribir:
        ap.error("--escribir no vale con --vivo: la base la escribe 'python -m ingesta.mercados_series construir-base'")
    cliente = ClienteFred()
    if a.vivo:
        codigo = _vivo(a, cliente)
        print(AVISO_FRED)
        return codigo
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
