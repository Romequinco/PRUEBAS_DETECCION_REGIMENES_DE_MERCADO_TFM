"""Fuentes académicas del Tema 1 (bloque 2): Ken French, Shiller, Goyal-Welch, OFR FSI y JST.

Ficheros (uno por fuente salvo French, que tiene cinco zip):

- ``ken_french``: Data Library de Kenneth R. French, zip con un CSV (cabecera «created using the
  AAAAMM CRSP database», bloques con cabecera ``,col1,col2...`` y pie de copyright). Se lee el
  PRIMER bloque (diario o mensual; en las industrias, el ponderado por valor). Retornos en %;
  ``-99.99`` y ``-999`` son faltantes.
- ``shiller``: ``ie_data.xls`` de shillerdata.com (hoja ``Data``): P, D, E, CPI, tipo largo
  (``Rate GS10``) y CAPE. Fechas ``AAAA.MM`` en coma flotante: ``1871.1`` es OCTUBRE (``1871.10``),
  no enero (bug de la Fase 1: 153 fechas repetidas en SHILLER_LONGRATE).
- ``goyal_welch``: hoja de predictores de Amit Goyal (Google Sheet exportada a xlsx, hoja
  ``Monthly``, ``yyyymm``); valores en tanto por uno.
- ``ofr_fsi``: ``fsi.csv`` oficial de la OFR (Financial Stress Index y subíndices, diario).
- ``jst_macrohistory``: Jordà-Schularick-Taylor Macrohistory R6 (panel anual, 18 países); fuera
  del catálogo: ``tabla_jst`` (validación, crisis de referencia).

Red solo con ``ingesta.comun.http.get`` (User-Agent identificable, 1 petición/s por host) tras
``robots_permite``. Un 403, un reto o una respuesta HTML no se esquivan: la descarga queda como
fallo en su ``.meta.json`` y la tabla usa el último crudo archivado (OFR: la copia de ``data/raw``
con su regla si no hay crudo). Crudos en ``<corpus>/mercados_series/crudo/<fuente>/`` con nombre
``<AAAAMMDD>_<fichero>`` (día de descarga en Nueva York) y su ``.meta.json`` (URL, estado,
cabeceras, bytes, sha256); la misma descarga el mismo día se sirve del archivo.

Ediciones: cada fichero archivado es una edición con ``fecha_edicion`` (``Last-Modified``; si no
hay, la fecha de modificación interna del xlsx o la descarga) y ``visto`` (última descarga con ese
contenido). La copia de ``data/raw`` cuenta como una edición descargada en el instante declarado en
``config/descargas_fase1.yaml``; en Ken French, si su último mes (la cabecera CRSP) coincide con el
de un crudo, es esa misma edición y toma su ``Last-Modified``. Etiqueta de la edición: el mes
(Nueva York) de ``fecha_edicion``, ``AAAA-MM``. ``archivar_ediciones`` copia cada edición en
``crudo/<fuente>/ediciones/<AAAA-MM>/`` (los crudos con su nombre y ``.meta.json``; la copia de
``data/raw`` como ``fase1_<fichero>.parquet``); los crudos de ese archivo también se leen.

Fechado (``metodo`` del catálogo: ``fichero``, o ``lag_fijo`` en la OFR):

- Ken French, Shiller y Goyal-Welch (``FUENTES_VINTAGE``): VINTAGE POR EDICIÓN. La primera edición
  da todos sus periodos con la regla (fin del periodo + ``lag`` de ``config/lags.yaml``,
  ``catalogo.fecha_publicacion_regla``) y ``vintage`` = su etiqueta. Cada edición posterior añade
  SOLO los periodos nuevos o con valor cambiado (``vintage`` = su etiqueta; ``fecha_publicacion`` =
  primera decisión tras su ``fecha_edicion``, nunca antes que la fila que revisa): la base crece
  sin perder valores y ``get_serie`` ve en cada fecha la edición vigente. Dos ediciones distintas en
  el mismo mes comparten etiqueta: la posterior sustituye a la anterior en los periodos que cambia
  (sus valores se ven desde la fecha de la posterior; antes, los de la edición previa).
- OFR: vintage ``unico`` con los valores de la última edición vista y
  ``fecha_publicacion = max(regla, primera_decision(t))``, siendo ``t`` la última vez que se vio
  una edición sin ese periodo (cota inferior PROBADA); revisable con ``catalogo.NOTA_REVISABLE_EDICION``.
- Periodo no cerrado en la fecha de edición (Shiller pone el mes en curso con el cierre del día 1):
  provisional, se descarta (como la barra intradía de yfinance).
- Notas: la primera edición de una revisable lleva ``NOTA_PRIMERA_EDICION`` + la causa (French
  recalcula con cada base CRSP; Shiller revisa E, D, CPI y E10); las filas posteriores dicen si el
  periodo es nuevo o qué edición revisan. Huecos dentro de una serie mensual: valor nulo con nota.

``tabla_academicos`` devuelve la tabla validada (``validar_series`` + ``validar_contra_catalogo``)
con las entradas del catálogo de estas fuentes. CLI:
``python -m ingesta.mercados_series.fuentes.academicos [--descargar] [--archivar] [--fichas] [--escribir]``.
Datos solo en local (copyright CRSP, S&P, ICE...; JST CC BY-NC-SA 4.0): al repositorio solo van
código, fichas y tablas de metadatos sin valores.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import warnings
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingesta.comun import http
from ingesta.comun.rutas import ROOT, corpus_dir, resultados_dir
from ingesta.mercados_series import catalogo as cat_mod
from ingesta.mercados_series.contrato import (
    COLUMNAS_SERIES,
    FECHA,
    HUSO_DECISION,
    UTC,
    primera_decision_vec,
    vacio_series,
    validar_series,
)

TEMA = "mercados_series"
VINTAGE = "unico"
RAIZ_FASE1 = ROOT / "data" / "raw"
FUENTES = ("ken_french", "shiller", "goyal_welch", "ofr_fsi", "jst_macrohistory")
FUENTES_CATALOGO = ("ken_french", "shiller", "goyal_welch", "ofr_fsi")
# Fuentes con vintage por edición (AAAA-MM); la OFR sigue con vintage 'unico' y la última edición.
FUENTES_VINTAGE = ("ken_french", "shiller", "goyal_welch")
CARPETA_EDICIONES = "ediciones"
PREFIJO_FASE1 = "fase1_"
# Dos valores son iguales entre ediciones si difieren menos que esto (ruido de coma flotante).
TOLERANCIA_ABS, TOLERANCIA_REL = 1e-9, 1e-12

URL_FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip"
FICHEROS_FRENCH = ("F-F_Research_Data_Factors_daily", "F-F_Research_Data_Factors",
                   "F-F_Research_Data_5_Factors_2x3_daily", "F-F_Momentum_Factor_daily",
                   "5_Industry_Portfolios_daily")
URL_SHILLER = ("https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/"
               "70fec4f5-727f-4e53-b5f1-179af109c5fa/ie_data.xls")
PAGINA_SHILLER = "https://shillerdata.com/"
URL_GOYAL_WELCH = "https://docs.google.com/spreadsheets/d/1qwpl2R_DNujpU5YUkk8lacP1tTeMb9iJ/export?format=xlsx"
URL_OFR = "https://www.financialresearch.gov/financial-stress-index/data/fsi.csv"
URL_JST = "https://www.macrohistory.net/app/download/9834512569/JSTdatasetR6.xlsx"

# fuente -> {clave del fichero: (nombre del crudo, url)}
FICHEROS: dict[str, dict[str, tuple[str, str]]] = {
    "ken_french": {n: (f"{n}_CSV.zip", URL_FRENCH.format(n)) for n in FICHEROS_FRENCH},
    "shiller": {"ie_data.xls": ("ie_data.xls", URL_SHILLER)},
    "goyal_welch": {"gw_predictores.xlsx": ("gw_predictores.xlsx", URL_GOYAL_WELCH)},
    "ofr_fsi": {"fsi.csv": ("fsi.csv", URL_OFR)},
    "jst_macrohistory": {"JSTdatasetR6.xlsx": ("JSTdatasetR6.xlsx", URL_JST)},
}
# Cabeceras HTTP que se archivan (sin cookies).
CABECERAS = ("Last-Modified", "ETag", "Content-Type", "Content-Length", "Content-Disposition", "Date")

FALTANTES_FRENCH = (-99.99, -999.0)
COLUMNAS_SHILLER = {"Long Interest Rate": "Rate GS10"}  # id_origen del catálogo -> cabecera real
NOTA_HUECO = "hueco permanente: la fuente no publica dato de este periodo"
# Si el catálogo la declara (para que get_serie la reconozca como nota de revisable), se usa la suya.
NOTA_PRIMERA_EDICION = getattr(cat_mod, "NOTA_PRIMERA_EDICION",
                               "valor de la primera edición archivada: posible fuga de revisiones anteriores a "
                               "ella (el archivo de ediciones empieza ahí)")
NOTA_FUENTE = {
    "ken_french": "Ken French recalcula toda la historia con cada base CRSP",
    "shiller": "Shiller revisa E, D, CPI y E10 entre ediciones",
    "goyal_welch": "Goyal-Welch reedita la hoja cada año",
    "ofr_fsi": "la OFR puede revisar el índice",
    "jst_macrohistory": "JST revisa entre ediciones (R5 -> R6)",
}
LICENCIAS = {
    "ken_french": ("Kenneth R. French Data Library: uso académico con cita; construida con datos de CRSP "
                   "(copyright CRSP / Center for Research in Security Prices): solo local, sin redistribuir"),
    "shiller": ("Robert J. Shiller, datos de 'Irrational Exuberance' (shillerdata.com): uso académico con cita; "
                "precios del S&P Composite con copyright S&P Dow Jones Indices: solo local, sin redistribuir"),
    "goyal_welch": ("Amit Goyal (Welch y Goyal 2008, actualizada): uso académico con cita; incluye datos CRSP, "
                    "S&P y Moody's con copyright: solo local, sin redistribuir"),
    "ofr_fsi": ("Office of Financial Research (Tesoro de EE. UU.): dominio público; insumos de mercado de "
                "terceros con copyright: solo local"),
    "jst_macrohistory": ("Jordà-Schularick-Taylor Macrohistory Database: Creative Commons BY-NC-SA 4.0 (uso no "
                         "comercial con cita; prohibido integrarla en servicios comerciales): solo local"),
}
_PATRON_CRSP = re.compile(r"created (?:by )?using the (\d{6}) CRSP", re.IGNORECASE)
_PATRON_FILA_FRENCH = re.compile(r"^\s*(\d{8}|\d{6})\s*,")
_PATRON_SHILLER = re.compile(r"\s*(\d{4})\.(\d{1,2})\s*")

__all__ = [
    "FUENTES", "FUENTES_VINTAGE", "FICHEROS", "NOTA_HUECO", "NOTA_PRIMERA_EDICION", "Edicion", "ruta_crudo",
    "descargar", "crudos", "ediciones", "etiqueta_edicion", "archivar_ediciones", "leer_french",
    "fechas_shiller", "leer_shiller", "leer_goyal_welch", "leer_ofr", "leer_jst", "leer_fichero", "localizar",
    "tabla_serie", "tabla_academicos", "resumen_academicos", "cambios_ediciones", "tabla_jst",
    "fichas_academicos",
]


# --- Crudos y descarga ---------------------------------------------------------------------------
def ruta_crudo(fuente: str, raiz: Path | None = None) -> Path:
    """``<corpus>/mercados_series/crudo/<fuente>/`` (fuera del repositorio)."""
    if fuente not in FICHEROS:
        raise ValueError(f"fuente {fuente!r} no es una de {list(FICHEROS)}")
    base = Path(raiz) if raiz is not None else corpus_dir(TEMA) / "crudo"
    p = base / fuente
    p.mkdir(parents=True, exist_ok=True)
    return p


def _hoy_ny() -> str:
    return pd.Timestamp.now(tz=HUSO_DECISION).strftime("%Y%m%d")


def _url_shiller_pagina() -> str | None:
    """Enlace vigente a ``ie_data.xls`` en shillerdata.com (cambia si Shiller reemplaza el fichero)."""
    if not http.robots_permite(PAGINA_SHILLER):
        return None
    html = http.get(PAGINA_SHILLER, timeout=60).text
    m = re.search(r"https?://[^\"'\s>]+ie_data\.xls", html)
    return m.group(0) if m else None


def _respuesta(url: str) -> tuple[Any, str | None]:
    """``(respuesta, error)``; el error es texto y la respuesta None si no se pudo descargar."""
    if not http.robots_permite(url):
        return None, "robots.txt no permite la descarga con nuestro User-Agent"
    try:
        r = http.get(url, timeout=90, reintentos=2)
    except Exception as e:  # noqa: BLE001 - se documenta en el .meta.json
        resp = getattr(e, "response", None)
        estado = getattr(resp, "status_code", None)
        return resp, f"{type(e).__name__}: {http.sin_clave(str(e))[:300]} (estado {estado}; no se esquiva)"
    if "html" in r.headers.get("Content-Type", "").lower():
        return r, "respuesta HTML en lugar del fichero (posible reto o bloqueo; no se esquiva)"
    return r, None


def descargar(fuente: str, *, claves: Iterable[str] | None = None, forzar: bool = False,
              raiz: Path | None = None) -> list[dict[str, Any]]:
    """Descarga los ficheros de ``fuente`` y devuelve su ``.meta.json`` (uno por fichero).

    El crudo de hoy (Nueva York) se reutiliza salvo ``forzar``. Un fallo (403, 404, HTML, robots) no
    lanza: queda en el meta con ``error`` y sin fichero.
    """
    carpeta = ruta_crudo(fuente, raiz)
    hoy = _hoy_ny()
    out = []
    for clave in (list(claves) if claves is not None else list(FICHEROS[fuente])):
        nombre, url = FICHEROS[fuente][clave]
        ruta = carpeta / f"{hoy}_{nombre}"
        ruta_meta = ruta.with_name(ruta.name + ".meta.json")
        if not forzar and ruta.exists() and ruta_meta.exists():
            out.append(json.loads(ruta_meta.read_text(encoding="utf-8")))
            continue
        r, error = _respuesta(url)
        if error is not None and fuente == "shiller" and getattr(r, "status_code", None) == 404:
            nueva = _url_shiller_pagina()
            if nueva and nueva != url:
                url = nueva
                r, error = _respuesta(url)
        meta: dict[str, Any] = {"fuente": fuente, "clave": clave, "url": url,
                                "descargado_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                                "estado": getattr(r, "status_code", None)}
        if r is not None:
            meta["cabeceras"] = {k: r.headers[k] for k in CABECERAS if k in r.headers}
            meta["url_final"] = http.sin_clave(r.url)
        if error is None:
            meta |= {"bytes": len(r.content), "sha256": hashlib.sha256(r.content).hexdigest()}
            ruta.write_bytes(r.content)
        else:
            meta["error"] = error
        ruta_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        out.append(meta)
    return out


def crudos(fuente: str, clave: str, raiz: Path | None = None) -> list[tuple[Path, dict[str, Any]]]:
    """Crudos de un fichero (con su meta), del más antiguo al más reciente por día de descarga.

    Incluye los del archivo de ediciones (``ediciones/<AAAA-MM>/``); una misma descarga en la
    carpeta y en el archivo sale dos veces (``ediciones`` las une por sha256).
    """
    nombre = FICHEROS[fuente][clave][0]
    carpeta = ruta_crudo(fuente, raiz)
    candidatos = [*carpeta.glob(f"[0-9]*_{nombre}"), *carpeta.glob(f"{CARPETA_EDICIONES}/*/[0-9]*_{nombre}")]
    out = []
    for p in sorted(candidatos, key=lambda p: (p.name, len(p.parts))):
        if p.name.endswith(".meta.json") or not re.fullmatch(rf"\d{{8}}_{re.escape(nombre)}", p.name):
            continue
        ruta_meta = p.with_name(p.name + ".meta.json")
        meta = json.loads(ruta_meta.read_text(encoding="utf-8")) if ruta_meta.exists() else {}
        out.append((p, meta))
    return out


# --- Ediciones ------------------------------------------------------------------------------------
@dataclass
class Edicion:
    """Un contenido de un fichero: cuándo se editó (cota superior) y hasta cuándo se vio."""

    origen: str                     # crudo | fase1
    ruta: Path
    fecha_edicion: pd.Timestamp     # UTC: Last-Modified, xlsx modificado o descarga
    fuente_fecha: str               # last_modified | xlsx_modificado | mtime | descarga
    visto: pd.Timestamp             # UTC: última descarga con este contenido (mtime en data/raw)
    sha256: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def etiqueta(self) -> str:
        """Vintage de la edición: ``AAAA-MM`` (mes de ``fecha_edicion`` en Nueva York)."""
        return etiqueta_edicion(self.fecha_edicion)


def etiqueta_edicion(t: pd.Timestamp) -> str:
    """``AAAA-MM`` del instante ``t`` (UTC) en Nueva York: el vintage de una edición."""
    return pd.Timestamp(t).tz_convert(HUSO_DECISION).strftime("%Y-%m")


def _utc(texto: Any) -> pd.Timestamp | None:
    if texto in (None, ""):
        return None
    t = pd.Timestamp(texto)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _xlsx_modificado(ruta: Path) -> pd.Timestamp | None:
    """``dcterms:modified`` de ``docProps/core.xml`` de un xlsx (UTC) o None."""
    try:
        with zipfile.ZipFile(ruta) as z:
            xml = z.read("docProps/core.xml").decode("utf-8", errors="replace")
    except (KeyError, zipfile.BadZipFile, OSError):
        return None
    m = re.search(r"<dcterms:modified[^>]*>([^<]+)</dcterms:modified>", xml)
    return _utc(m.group(1)) if m else None


def _mtime_utc(ruta: Path) -> pd.Timestamp:
    return pd.Timestamp(Path(ruta).stat().st_mtime, unit="s", tz="UTC")


def ediciones(fuente: str, clave: str, raiz: Path | None = None) -> list[Edicion]:
    """Ediciones distintas (por sha256) de los crudos archivados de un fichero, por ``fecha_edicion``.

    En Ken French, ``extra['crsp']`` es el ``AAAAMM`` de la cabecera (``created using the AAAAMM CRSP``).
    """
    por_sha: dict[str, Edicion] = {}
    for ruta, meta in crudos(fuente, clave, raiz):
        contenido = ruta.read_bytes()
        sha = meta.get("sha256") or hashlib.sha256(contenido).hexdigest()
        descarga = _utc(meta.get("descargado_utc")) or _mtime_utc(ruta)
        lm = _utc((meta.get("cabeceras") or {}).get("Last-Modified")
                  or (meta.get("cabeceras") or {}).get("last-modified"))
        fuente_fecha = "last_modified"
        if lm is None and ruta.suffix.lower() == ".xlsx":
            lm, fuente_fecha = _xlsx_modificado(ruta), "xlsx_modificado"
        if lm is None:
            lm, fuente_fecha = descarga, "descarga"
        lm = min(lm, descarga)
        e = por_sha.get(sha)
        if e is None:
            por_sha[sha] = Edicion("crudo", ruta, lm, fuente_fecha, descarga, sha)
        else:
            e.visto = max(e.visto, descarga)
            if lm < e.fecha_edicion:
                e.fecha_edicion, e.fuente_fecha = lm, fuente_fecha
            if descarga >= e.visto:
                e.ruta = ruta
    if fuente == "ken_french":
        for e in por_sha.values():
            try:
                e.extra["crsp"] = leer_french(e.ruta)[1]["crsp"]
            except (ValueError, zipfile.BadZipFile, OSError):
                e.extra["crsp"] = None
    return sorted(por_sha.values(), key=lambda e: e.fecha_edicion)


# --- Parsers --------------------------------------------------------------------------------------
def leer_french(fuente: Path | bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Primer bloque de un zip CSV de Ken French y su cabecera.

    Devuelve ``(datos, info)``: datos con índice de fechas (``AAAAMMDD`` diario o ``AAAAMM`` →
    día 1 del mes) y las columnas de la cabecera del bloque; ``info`` con ``crsp`` (``AAAAMM`` de la
    base CRSP), ``fichero_zip`` y ``creado_zip`` (fecha interna del zip, hora local del servidor).
    """
    datos = fuente if isinstance(fuente, bytes) else Path(fuente).read_bytes()
    with zipfile.ZipFile(io.BytesIO(datos)) as z:
        miembro = z.infolist()[0]
        texto = z.read(miembro).decode("latin-1")
    lineas = texto.splitlines()
    cabecera: list[str] | None = None
    fechas: list[str] = []
    filas: list[list[float]] = []
    for ln in lineas:
        m = _PATRON_FILA_FRENCH.match(ln)
        if m:
            if cabecera is None:
                raise ValueError("Ken French: fila de datos sin cabecera de bloque")
            valores = [float(p) for p in ln.split(",")[1:]]
            if len(valores) != len(cabecera):
                raise ValueError(f"Ken French: {len(valores)} valores con {len(cabecera)} columnas en {ln!r}")
            fechas.append(m.group(1))
            filas.append(valores)
        elif filas:
            break
        elif ln.lstrip().startswith(","):
            cabecera = [c.strip() for c in ln.split(",")[1:]]
    if not filas:
        raise ValueError("Ken French: bloque de datos vacío")
    formato = "%Y%m%d" if len(fechas[0]) == 8 else "%Y%m"
    df = pd.DataFrame(filas, index=pd.DatetimeIndex(pd.to_datetime(fechas, format=formato)), columns=cabecera)
    df = df.mask(df.isin(FALTANTES_FRENCH))
    m = _PATRON_CRSP.search(texto[:2000])
    info = {"crsp": m.group(1) if m else None, "fichero_zip": miembro.filename,
            "creado_zip": pd.Timestamp(*miembro.date_time).isoformat(),
            "copyright": next((ln.strip() for ln in reversed(lineas) if "copyright" in ln.lower()), None)}
    return df, info


def fechas_shiller(columna: Any) -> pd.DatetimeIndex:
    """Fechas ``AAAA.MM`` de Shiller → día 1 del mes (NaT si no es una fecha).

    En coma flotante octubre es ``1871.1``: el mes son los DOS decimales (``1871.1`` → ``1871.10``
    → octubre; ``1871.01`` → enero). En texto, un solo decimal también es decena (``'1871.1'`` →
    octubre). Leerlo como ``'1871-1'`` daba enero (bug de la Fase 1).
    """
    out = []
    for v in pd.Series(columna, dtype=object):
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            texto = f"{float(v):.2f}" if np.isfinite(v) else ""
        else:
            texto = str(v) if v is not None else ""
        m = _PATRON_SHILLER.fullmatch(texto)
        if not m:
            out.append(pd.NaT)
            continue
        mes_txt = m.group(2)
        mes = int(mes_txt) * (10 if len(mes_txt) == 1 else 1)
        out.append(pd.Timestamp(int(m.group(1)), mes, 1) if 1 <= mes <= 12 else pd.NaT)
    return pd.DatetimeIndex(out)


def _rejilla_shiller(ruta: Path) -> pd.DataFrame:
    if Path(ruta).suffix.lower() in (".csv", ".txt"):
        return pd.read_csv(ruta, header=None, dtype=str, keep_default_na=False)
    return pd.read_excel(ruta, sheet_name="Data", header=None)


def leer_shiller(fuente: Path | pd.DataFrame) -> pd.DataFrame:
    """Hoja ``Data`` de ``ie_data.xls`` (o la misma rejilla en CSV) → columnas numéricas por mes.

    Busca la fila de cabecera (primera celda ``Date``); se quedan las columnas con nombre único
    (``P``, ``D``, ``E``, ``CPI``, ``Rate GS10``, ``CAPE``...). Las filas sin fecha válida (notas al
    pie) se descartan; ``'NA'`` y vacíos son nulos. ``ValueError`` si hay meses repetidos.
    """
    g = fuente if isinstance(fuente, pd.DataFrame) else _rejilla_shiller(Path(fuente))
    primera = g.iloc[:, 0].astype(str).str.strip()
    filas = np.flatnonzero(primera.to_numpy() == "Date")
    if not len(filas):
        raise ValueError("Shiller: no se encuentra la fila de cabecera 'Date'")
    i = int(filas[0])
    nombres = [str(c).strip() if pd.notna(c) else "" for c in g.iloc[i]]
    cuerpo = g.iloc[i + 1:]
    fechas = fechas_shiller(cuerpo.iloc[:, 0])
    ok = ~pd.isna(fechas)
    cuenta = pd.Series(nombres).value_counts()
    cols = {}
    for j, n in enumerate(nombres[1:], start=1):
        if n and cuenta[n] == 1:
            cols[n] = pd.to_numeric(cuerpo.iloc[:, j].replace({"NA": None, "": None}), errors="coerce").to_numpy()
    df = pd.DataFrame(cols, index=cuerpo.index)[ok]
    df.index = fechas[ok]
    if df.index.has_duplicates:
        raise ValueError(f"Shiller: meses repetidos {df.index[df.index.duplicated()][:3].date.tolist()}")
    return df.sort_index()


def leer_goyal_welch(ruta: Path, hoja: str = "Monthly") -> pd.DataFrame:
    """Hoja de predictores de Goyal-Welch: índice ``yyyymm`` → día 1 del mes (o ``yyyy`` → 1 de enero)."""
    with warnings.catch_warnings():  # cabecera/pie de página que openpyxl no interpreta
        warnings.simplefilter("ignore", UserWarning)
        df = pd.read_excel(ruta, sheet_name=hoja)
    col = df.columns[0]
    txt = df[col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    formato = {"yyyymm": "%Y%m", "yyyy": "%Y"}.get(str(col).strip().lower())
    if formato is None:
        raise ValueError(f"Goyal-Welch: primera columna {col!r} no soportada (yyyymm o yyyy)")
    idx = pd.to_datetime(txt, format=formato, errors="coerce")
    df = df.drop(columns=[col]).set_index(pd.DatetimeIndex(idx))
    df = df[df.index.notna()].apply(pd.to_numeric, errors="coerce")
    if df.index.has_duplicates:
        raise ValueError("Goyal-Welch: fechas repetidas")
    return df.sort_index()


def leer_ofr(ruta: Path | pd.DataFrame) -> pd.DataFrame:
    """``fsi.csv`` de la OFR: índice de fechas (columna ``Date``) y una columna por índice."""
    df = ruta.copy() if isinstance(ruta, pd.DataFrame) else pd.read_csv(ruta)
    df.columns = [str(c).strip() for c in df.columns]
    if "Date" not in df.columns:
        raise ValueError(f"OFR: falta la columna Date ({list(df.columns)})")
    idx = pd.DatetimeIndex(pd.to_datetime(df.pop("Date"))).normalize()
    df = df.set_index(idx).apply(pd.to_numeric, errors="coerce")
    if df.index.has_duplicates:
        raise ValueError("OFR: fechas repetidas")
    return df.sort_index()


def leer_jst(ruta: Path, iso: str = "USA") -> pd.DataFrame:
    """Panel JST (primera hoja) de un país: índice = 1 de enero de cada año, columnas numéricas."""
    df = pd.read_excel(ruta, sheet_name=0)
    if "iso" not in df.columns or "year" not in df.columns:
        raise ValueError("JST: faltan las columnas iso y year")
    df = df[df["iso"].astype(str).str.upper() == iso.upper()]
    idx = pd.to_datetime(df["year"].astype(int).astype(str), format="%Y")
    out = df.drop(columns=["year"]).select_dtypes("number").set_index(pd.DatetimeIndex(idx))
    if out.index.has_duplicates:
        raise ValueError(f"JST: años repetidos para {iso}")
    return out.sort_index()


def leer_fichero(fuente: str, clave: str, ruta: Path, hoja: str | None = None) -> pd.DataFrame:
    """Tabla de un crudo según su fuente (columnas = nombres del fichero)."""
    if fuente == "ken_french":
        return leer_french(ruta)[0]
    if fuente == "shiller":
        return leer_shiller(ruta)
    if fuente == "goyal_welch":
        return leer_goyal_welch(ruta, hoja or "Monthly")
    if fuente == "ofr_fsi":
        return leer_ofr(ruta)
    if fuente == "jst_macrohistory":
        return leer_jst(ruta)
    raise ValueError(f"fuente {fuente!r} sin lector")


def localizar(entrada: dict[str, Any]) -> tuple[str, str, str | None]:
    """``(clave del fichero, columna, hoja)`` de una entrada del catálogo a partir de su ``id_origen``."""
    fuente, origen = entrada["fuente"], entrada["id_origen"]
    if fuente == "ken_french":
        clave, col = origen.split(":", 1)
        if clave not in FICHEROS["ken_french"]:
            raise ValueError(f"fichero de Ken French no registrado: {clave!r}")
        return clave, col, None
    if fuente == "shiller":
        m = re.search(r"\(col ([^)]+)\)", origen)
        if not m:
            raise ValueError(f"id_origen de Shiller sin '(col ...)': {origen!r}")
        return "ie_data.xls", COLUMNAS_SHILLER.get(m.group(1).strip(), m.group(1).strip()), None
    if fuente == "goyal_welch":
        m = re.search(r"\(hoja (\w+)\)", origen)
        return "gw_predictores.xlsx", origen.split(":", 1)[1], m.group(1) if m else "Monthly"
    if fuente == "ofr_fsi":
        return "fsi.csv", origen.split(":", 1)[1], None
    raise ValueError(f"fuente {fuente!r} no es académica")


# --- Fechado ----------------------------------------------------------------------------------------
_PERIODO = {"mensual": "M", "trimestral": "Q", "anual": "Y"}
_REJILLA = {"mensual": "MS", "trimestral": "QS", "anual": "YS"}


def _dia_ny(t: pd.Timestamp) -> pd.Timestamp:
    return t.tz_convert(HUSO_DECISION).tz_localize(None).normalize()


def _normalizar(s: pd.Series, frecuencia: str) -> pd.Series:
    idx = pd.DatetimeIndex(s.index).normalize()
    if frecuencia in _PERIODO:
        idx = idx.to_period(_PERIODO[frecuencia]).start_time.normalize()
    s = pd.Series(s.to_numpy(dtype=float), index=idx)
    if s.index.has_duplicates:
        raise ValueError(f"fechas repetidas tras normalizar a {frecuencia}")
    return s.sort_index()


def _cerrados(s: pd.Series, fin_periodo: str, edicion: pd.Timestamp) -> pd.Series:
    """Valores no nulos cuyo periodo había terminado antes del día (NY) de la edición."""
    s = s.dropna()
    fin = cat_mod.fin_de_periodo(pd.Series(s.index), fin_periodo)
    return s[(fin < _dia_ny(edicion)).to_numpy()]


@dataclass
class _Info:
    """Lo medido al construir una serie (resumen y fichas)."""

    sid: str
    fuente: str
    clave: str
    columna: str
    origen: str = "crudo"
    edicion: Edicion | None = None
    ediciones: int = 0
    provisionales: int = 0
    con_cota: int = 0
    n_huecos: int = 0
    revision: dict[str, Any] = field(default_factory=dict)
    ultimo_por_edicion: list[dict[str, Any]] = field(default_factory=list)
    cambios: list[dict[str, Any]] = field(default_factory=list)   # una entrada por edición posterior


def _leer_parquet_fase1(ruta: Path, col: str) -> pd.Series | None:
    df = pd.read_parquet(ruta)
    if col not in df.columns:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float), index=idx)
    return s[~s.index.duplicated(keep="first")]


def _copia_archivada(fuente: str, rel: str, raiz_crudo: Path | None) -> tuple[Path, dict[str, Any]] | None:
    """Copia más reciente de ``data/raw/<rel>`` en el archivo de ediciones (con su meta) o None."""
    carpeta = ruta_crudo(fuente, raiz_crudo) / CARPETA_EDICIONES
    for p in sorted(carpeta.glob(f"*/{PREFIJO_FASE1}{Path(rel).name}"), reverse=True):
        ruta_meta = p.with_name(p.name + ".meta.json")
        if ruta_meta.exists():
            meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
            if meta.get("ruta_fase1") == rel and meta.get("descargado_utc"):
                return p, meta
    return None


def _edicion_fase1(entrada: dict[str, Any], raiz_fase1: Path, raiz_crudo: Path | None = None,
                   crudas: Sequence[Edicion] = ()) -> tuple[Edicion, pd.Series] | None:
    """Copia de ``data/raw`` de una serie como edición, o None si no hay o no es fiable.

    Instante: el declarado en ``config/descargas_fase1.yaml`` (``fase1.instante_descarga``). Si
    ``data/raw`` ya no está, se usa su copia del archivo de ediciones. En Ken French el último mes
    de la copia es la cabecera CRSP de su edición: si coincide con el ``crsp`` de un crudo de
    ``crudas`` anterior a la descarga, es esa edición y toma su ``fecha_edicion`` (Last-Modified).
    """
    if not entrada["fichero_fase1"] or entrada.get("fase1_utilizable") is False:
        return None
    rel, col = entrada["fichero_fase1"][0].split(":", 1)
    ruta = Path(raiz_fase1) / rel
    if ruta.exists():
        from ingesta.mercados_series.fuentes.fase1 import instante_descarga

        t = instante_descarga(ruta, Path(raiz_fase1))
    else:
        copia = _copia_archivada(entrada["fuente"], rel, raiz_crudo)
        if copia is None:
            return None
        ruta, meta = copia
        t = _utc(meta["descargado_utc"])
    s = _leer_parquet_fase1(ruta, col)
    if s is None:
        return None
    validos = s.dropna()
    ultimo = validos.index.max().strftime("%Y%m") if len(validos) else None
    ed = Edicion("fase1", ruta, t, "descarga_fase1", t, extra={"ruta_fase1": rel, "ultimo_mes": ultimo})
    if entrada["fuente"] == "ken_french":
        ed.extra["crsp"] = ultimo
        misma = [c for c in crudas if ultimo and c.extra.get("crsp") == ultimo and c.fecha_edicion < t]
        if misma:
            c = min(misma, key=lambda c: c.fecha_edicion)
            ed.fecha_edicion, ed.fuente_fecha = c.fecha_edicion, f"crsp_{c.fuente_fecha}"
    return ed, s


def tabla_serie(sid: str, entrada: dict[str, Any], versiones: Sequence[tuple[Edicion, pd.Series]],
                info: _Info | None = None) -> pd.DataFrame:
    """Filas del contrato de una serie a partir de sus versiones ``(edición, valores)``.

    ``FUENTES_VINTAGE``: vintage por edición (``_tabla_ediciones``). OFR: vintage ``unico`` con la
    edición vista más recientemente (``_tabla_unica``).
    """
    if not versiones:
        raise ValueError(f"{sid}: sin versiones del fichero")
    info = info or _Info(sid, entrada["fuente"], "", "")
    if entrada["metodo"] not in cat_mod.METODOS_CON_LAG or entrada["lag"] is None:
        raise ValueError(f"{sid}: metodo {entrada['metodo']!r} sin lag: no se fecha como fichero")
    if entrada["fuente"] in FUENTES_VINTAGE:
        return _tabla_ediciones(sid, entrada, versiones, info)
    return _tabla_unica(sid, entrada, versiones, info)


def _filas(sid: str, entrada: dict[str, Any], fechas: Any, pub: Any, vintage: Any, valores: Any,
           notas: Any) -> pd.DataFrame:
    n = len(fechas)
    nota = pd.Series(notas, dtype=object) if n else pd.Series([], dtype=object)
    return pd.DataFrame({
        "serie": sid,
        "fecha_referencia": pd.Series(fechas, dtype=FECHA).to_numpy(),
        "fecha_publicacion": pd.Series(pub, dtype=FECHA).to_numpy(),
        "disponible_utc": pd.Series(pd.NaT, index=range(n), dtype=UTC).array,
        "vintage": pd.Series(vintage, index=range(n), dtype=object).to_numpy(),
        "valor": np.asarray(valores, dtype=float),
        "unidad": entrada["unidad"],
        "divisa": entrada["divisa"],
        "fuente": entrada["fuente"],
        "metodo_publicacion": entrada["metodo"],
        "nota": nota.where(nota.notna(), None).to_numpy(),
    }, index=range(n))


def _tabla_unica(sid: str, entrada: dict[str, Any], versiones: Sequence[tuple[Edicion, pd.Series]],
                 info: _Info) -> pd.DataFrame:
    """Vintage ``unico`` (OFR): valores de la edición vista más recientemente (``visto``).

    Sin los periodos no cerrados en su fecha de edición. ``fecha_publicacion = max(regla del
    catálogo, primera decisión tras la última vez que se vio una edición sin ese periodo)``.
    Mensuales: huecos internos con ``NOTA_HUECO``.
    """
    frecuencia, fin_periodo = entrada["frecuencia"], entrada["fin_periodo"]
    ordenadas = sorted(versiones, key=lambda v: (v[0].visto, v[0].fecha_edicion))
    cerradas = [(e, _cerrados(_normalizar(s, frecuencia), fin_periodo, e.fecha_edicion)) for e, s in ordenadas]
    ed, ultima = ordenadas[-1][0], cerradas[-1][1]
    todo = _normalizar(ordenadas[-1][1], frecuencia).dropna()
    info.edicion, info.ediciones = ed, len(ordenadas)
    info.provisionales = int(len(todo) - len(ultima))
    info.ultimo_por_edicion = [
        {"origen": e.origen, "fecha_edicion": e.fecha_edicion.isoformat(), "fuente_fecha": e.fuente_fecha,
         "visto": e.visto.isoformat(),
         "ultimo": c.index.max().date().isoformat() if len(c) else None} for e, c in cerradas]
    s = ultima
    if frecuencia in _REJILLA and len(s) >= 2:
        s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq=_REJILLA[frecuencia]))
    info.n_huecos = int(s.isna().sum())
    fechas = pd.Series(s.index, dtype=FECHA)
    regla = cat_mod.fecha_publicacion_regla(fechas, entrada)
    # Cota inferior probada: la última vez que se vio una edición que no tenía el periodo.
    visto_sin = pd.Series(pd.NaT, index=fechas.index, dtype=UTC)
    for e, c in cerradas:
        ult = c.index.max() if len(c) else pd.Timestamp.min
        sin = (fechas > ult).to_numpy() & ((visto_sin.isna()) | (visto_sin < e.visto)).to_numpy()
        visto_sin[sin] = e.visto
    cota = primera_decision_vec(visto_sin)
    pub = regla.where(cota.isna() | (regla >= cota), cota).astype(FECHA)
    info.con_cota = int((cota.notna() & (cota > regla)).sum())
    nota_base = f"{cat_mod.nota_revisable(entrada)}; {NOTA_FUENTE[entrada['fuente']]}" if entrada["revisable"] else None
    nulo = s.isna().to_numpy()
    hueco = f"{NOTA_HUECO}; {nota_base}" if nota_base else NOTA_HUECO
    nota = np.where(nulo, hueco, nota_base if nota_base else None)
    return _filas(sid, entrada, fechas, pub, VINTAGE, s.to_numpy(dtype=float), nota)


def _version(s: pd.Series, frecuencia: str, fin_periodo: str, edicion: pd.Timestamp) -> tuple[pd.Series, int]:
    """Periodos cerrados de una edición (con sus nulos, en la rejilla si es periódica) y provisionales."""
    s = _normalizar(s, frecuencia)
    fin = cat_mod.fin_de_periodo(pd.Series(s.index), fin_periodo)
    cerrado = (fin < _dia_ny(edicion)).to_numpy()
    provisionales = int((~cerrado & s.notna().to_numpy()).sum())
    s = s[cerrado]
    validos = s.dropna()
    if validos.empty:
        return validos, provisionales
    s = s.loc[validos.index[0]:validos.index[-1]]
    if frecuencia in _REJILLA and len(validos) >= 2:
        s = s.reindex(pd.date_range(validos.index[0], validos.index[-1], freq=_REJILLA[frecuencia]))
    return s, provisionales


def _iguales(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Igualdad entre ediciones: los dos nulos o a menos de la tolerancia (ruido de coma flotante)."""
    return (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=TOLERANCIA_REL, atol=TOLERANCIA_ABS)


def _sustituir(viejo: pd.Series, nuevo: pd.Series) -> pd.Series:
    """``viejo`` con los periodos de ``nuevo`` sustituidos o añadidos."""
    resto = viejo[~viejo.index.isin(nuevo.index)]
    return pd.concat([resto, nuevo]).sort_index() if len(resto) else nuevo.sort_index()


def _tabla_ediciones(sid: str, entrada: dict[str, Any], versiones: Sequence[tuple[Edicion, pd.Series]],
                     info: _Info) -> pd.DataFrame:
    """Vintage por edición: la primera con la regla; cada posterior, solo lo nuevo o cambiado.

    Orden de las ediciones: ``visto`` (la que se sirvió más tarde es la vigente), luego
    ``fecha_edicion``. Fecha efectiva de una edición: ``max(fecha_edicion, visto de la anterior,
    fecha efectiva de la anterior)`` (si la anterior aún se servía, esta no existía); su mes en NY es
    el vintage (coincide con ``Edicion.etiqueta`` salvo fechas contradictorias). La primera da todos
    sus periodos cerrados (diaria: sin faltantes; periódica: huecos internos con ``NOTA_HUECO``) con
    ``fecha_publicacion`` = regla del catálogo. Una posterior da los periodos que no estaban (o eran
    hueco), los de valor distinto (``_iguales``) y los que deja sin valor (fila nula con nota), con
    ``fecha_publicacion`` = primera decisión tras su fecha efectiva, nunca antes que la fila que
    sustituye. Misma etiqueta: gana la posterior.
    """
    frecuencia, fin_periodo = entrada["frecuencia"], entrada["fin_periodo"]
    rejilla = frecuencia in _REJILLA
    prep = []
    for e, s in versiones:
        v, prov = _version(s, frecuencia, fin_periodo, e.fecha_edicion)
        ult = v.dropna().index.max() if v.notna().any() else pd.Timestamp.min
        prep.append((e, v, prov, ult))
    prep.sort(key=lambda x: (x[0].visto, x[0].fecha_edicion, x[3]))
    efectivas: list[pd.Timestamp] = []
    for k, (e, *_r) in enumerate(prep):
        efectivas.append(e.fecha_edicion if k == 0 else max(e.fecha_edicion, prep[k - 1][0].visto, efectivas[-1]))
    etiquetas = [etiqueta_edicion(t) for t in efectivas]
    info.edicion, info.ediciones, info.provisionales = prep[-1][0], len(prep), prep[-1][2]
    info.ultimo_por_edicion = [
        {"origen": e.origen, "etiqueta": lab, "fecha_edicion": e.fecha_edicion.isoformat(),
         "fecha_efectiva": t.isoformat(), "fuente_fecha": e.fuente_fecha, "visto": e.visto.isoformat(),
         "ultimo": u.date().isoformat() if v.notna().any() else None}
        for (e, v, _, u), t, lab in zip(prep, efectivas, etiquetas, strict=True)]
    info.cambios = []
    valor = fecha = vint = None
    bloques = []
    for k, (e, s, _, _) in enumerate(prep):
        lab = etiquetas[k]
        dec = primera_decision_vec(pd.Series([efectivas[k]], dtype=UTC)).iloc[0]
        if valor is None:
            sel = s if rejilla else s.dropna()
            pub = cat_mod.fecha_publicacion_regla(pd.Series(sel.index, dtype=FECHA), entrada)
            base = f"{NOTA_PRIMERA_EDICION} ({lab}); {NOTA_FUENTE[entrada['fuente']]}" if entrada["revisable"] else None
            hueco = f"{NOTA_HUECO}; {base}" if base else NOTA_HUECO
            notas = np.where(sel.isna().to_numpy(), hueco, base if base else None)
            pub = pd.Series(pub.to_numpy(), index=sel.index, dtype=FECHA)
        else:
            previo = valor.reindex(s.index).to_numpy()
            actual = s.to_numpy(dtype=float)
            tiene = s.index.isin(valor.index)
            presente, prev_nulo = ~np.isnan(actual), np.isnan(previo)
            nuevo = ~tiene & (presente | rejilla)
            cambia = tiene & ~_iguales(actual, previo)
            revisado = cambia & presente & ~prev_nulo
            retirado = cambia & ~presente
            dif = np.abs(actual - previo)[revisado]
            info.cambios.append({
                "etiqueta": lab, "previa": etiquetas[k - 1], "origen": e.origen,
                "fecha_edicion": e.fecha_edicion.isoformat(), "fecha_efectiva": efectivas[k].isoformat(),
                "fuente_fecha": e.fuente_fecha,
                "comunes": int((tiene & presente & ~prev_nulo).sum()), "revisados": int(revisado.sum()),
                "nuevos": int(((nuevo | cambia) & presente & ~revisado).sum()), "retirados": int(retirado.sum()),
                "huecos_nuevos": int((nuevo & ~presente).sum()),
                "max_abs": float(dif.max()) if len(dif) else None})
            m = nuevo | cambia
            if not m.any():
                continue
            sel = s[m]
            previa = fecha.reindex(sel.index)
            pub = previa.where(previa > dec, dec).astype(FECHA)
            plab = vint.reindex(sel.index).to_numpy()
            notas = np.full(len(sel), f"edición {lab}: periodo nuevo", dtype=object)
            notas[(cambia & presente & prev_nulo)[m]] = f"edición {lab}: dato de un periodo que era hueco"
            notas[(nuevo & ~presente)[m]] = f"{NOTA_HUECO}; edición {lab}"
            for mascara, texto in ((revisado[m], "edición {e}: revisa el valor de la edición {p}"),
                                   (retirado[m], "edición {e}: la fuente ya no publica este periodo "
                                                 "(retira el valor de la edición {p})")):
                notas[mascara] = [texto.format(e=lab, p=p) for p in plab[mascara]]
        bloques.append(_filas(sid, entrada, sel.index, pub.to_numpy(), lab, sel.to_numpy(dtype=float), notas))
        val = pd.Series(sel.to_numpy(dtype=float), index=sel.index)
        etq = pd.Series(lab, index=sel.index, dtype=object)
        valor = val if valor is None else _sustituir(valor, val)
        fecha = pub if fecha is None else _sustituir(fecha, pub)
        vint = etq if vint is None else _sustituir(vint, etq)
    tabla = pd.concat(bloques, ignore_index=True).drop_duplicates(["fecha_referencia", "vintage"], keep="last")
    info.n_huecos = int(tabla["valor"].isna().sum())
    return tabla.sort_values(["fecha_referencia", "vintage"], kind="mergesort").reset_index(drop=True)


def _revision(fase1: pd.Series | None, nueva: pd.Series, frecuencia: str) -> dict[str, Any]:
    """Valores de ``data/raw`` que la edición nueva cambia (fechas comunes)."""
    if fase1 is None:
        return {}
    a, b = _normalizar(fase1, frecuencia).dropna(), _normalizar(nueva, frecuencia).dropna()
    comun = a.index.intersection(b.index)
    d = (a.loc[comun] - b.loc[comun]).abs()
    distintos = d > 1e-9
    return {"comunes": int(len(comun)), "distintos": int(distintos.sum()),
            "max_abs": float(d.max()) if len(d) else None,
            "mediana_abs": float(d[distintos].median()) if distintos.any() else 0.0}


def _entradas(cat: dict[str, dict], fuentes: Iterable[str]) -> list[str]:
    return [sid for sid in sorted(cat) if cat[sid]["fuente"] in set(fuentes)
            and cat[sid]["estado"] in cat_mod.ESTADOS_CARGABLES]


def _cargar(fuentes: Iterable[str] | None, catalogo: dict[str, dict] | None, raiz_crudo: Path | None,
            raiz_fase1: Path | None) -> tuple[pd.DataFrame, dict[str, _Info], dict[str, dict]]:
    cat = catalogo if catalogo is not None else cat_mod.cargar_catalogo()
    fuentes = tuple(fuentes) if fuentes is not None else FUENTES_CATALOGO
    raiz_fase1 = Path(raiz_fase1) if raiz_fase1 is not None else RAIZ_FASE1
    cache_ed: dict[tuple[str, str], list[Edicion]] = {}
    cache_df: dict[tuple[Path, str | None], pd.DataFrame] = {}
    tablas, infos = [], {}
    for sid in _entradas(cat, fuentes):
        e = cat[sid]
        clave, col, hoja = localizar(e)
        info = infos[sid] = _Info(sid, e["fuente"], clave, col)
        if (e["fuente"], clave) not in cache_ed:
            cache_ed[(e["fuente"], clave)] = ediciones(e["fuente"], clave, raiz_crudo)
        versiones: list[tuple[Edicion, pd.Series]] = []
        for ed in cache_ed[(e["fuente"], clave)]:
            k = (ed.ruta, hoja)
            if k not in cache_df:
                cache_df[k] = leer_fichero(e["fuente"], clave, ed.ruta, hoja)
            if col not in cache_df[k].columns:
                raise KeyError(f"{sid}: la columna {col!r} no está en {ed.ruta.name} ({list(cache_df[k].columns)})")
            versiones.append((ed, cache_df[k][col]))
        f1 = _edicion_fase1(e, raiz_fase1, raiz_crudo, cache_ed[(e["fuente"], clave)])
        if f1 is not None:
            if versiones:
                reciente = max(versiones, key=lambda v: v[0].visto)[1]
                info.revision = _revision(f1[1], reciente, e["frecuencia"])
            versiones.append(f1)
        if not versiones:
            info.origen = "sin_datos"
            continue
        if all(v[0].origen == "fase1" for v in versiones):
            info.origen = "fase1"
        tablas.append(tabla_serie(sid, e, versiones, info))
    if not tablas:
        return vacio_series(), infos, cat
    return pd.concat(tablas, ignore_index=True)[list(COLUMNAS_SERIES)], infos, cat


def tabla_academicos(fuentes: Iterable[str] | None = None, *, catalogo: dict[str, dict] | None = None,
                     raiz_crudo: Path | None = None, raiz_fase1: Path | None = None) -> pd.DataFrame:
    """Tabla larga validada de las series académicas del catálogo (estados cargables).

    ``fuentes``: subconjunto de ``ken_french``, ``shiller``, ``goyal_welch`` y ``ofr_fsi``. Ediciones:
    los crudos (carpeta y archivo ``ediciones/``) y la copia de ``data/raw``. French, Shiller y
    Goyal-Welch con vintage por edición ``AAAA-MM`` (la primera edición con la regla; las
    posteriores, solo lo nuevo o revisado, fechado con su edición); la OFR, vintage ``unico``. No
    toca la red (``descargar`` antes para tener la última edición).
    """
    tabla, _, cat = _cargar(fuentes, catalogo, raiz_crudo, raiz_fase1)
    out = validar_series(tabla)
    cat_mod.validar_contra_catalogo(out, cat)
    return out


# --- Archivo de ediciones ---------------------------------------------------------------------------
def _copiar(origen: Path, destino: Path, sha: str, meta: dict[str, Any]) -> str:
    """Copia ``origen`` y su meta en ``destino`` salvo que ya esté (mismo sha256): ``copiado`` | ``ya_estaba``."""
    ruta_meta = destino.with_name(destino.name + ".meta.json")
    if destino.exists():
        if hashlib.sha256(destino.read_bytes()).hexdigest() != sha:
            raise ValueError(f"{destino}: ya hay en el archivo un fichero distinto con ese nombre (no se sobrescribe)")
        if not ruta_meta.exists():
            ruta_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return "ya_estaba"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(Path(origen).read_bytes())
    ruta_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return "copiado"


def _meta_edicion(ed: Edicion) -> dict[str, Any]:
    return {"etiqueta": ed.etiqueta, "origen": ed.origen, "fecha_edicion": ed.fecha_edicion.isoformat(),
            "fuente_fecha": ed.fuente_fecha, "visto": ed.visto.isoformat(),
            **{k: v for k, v in ed.extra.items() if k in ("crsp", "ultimo_mes", "ruta_fase1")}}


def archivar_ediciones(fuentes: Iterable[str] = FUENTES_VINTAGE, *, catalogo: dict[str, dict] | None = None,
                       raiz_crudo: Path | None = None, raiz_fase1: Path | None = None) -> pd.DataFrame:
    """Copia cada edición en ``crudo/<fuente>/ediciones/<AAAA-MM>/`` sin descargar nada (idempotente).

    Crudos: ``<AAAAMMDD>_<fichero>`` con su ``.meta.json`` más ``edicion`` (etiqueta, fecha y su
    origen, visto; ``crsp`` en French). Copias de ``data/raw`` de las series del catálogo:
    ``fase1_<fichero>.parquet`` con un ``.meta.json`` (``ruta_fase1``, instante de descarga
    declarado, último mes). Devuelve una fila por fichero (solo metadatos): ``fuente, clave,
    etiqueta, origen, fecha_edicion, fuente_fecha, fichero, accion`` (``copiado`` | ``ya_estaba``).
    """
    cat = catalogo if catalogo is not None else cat_mod.cargar_catalogo()
    raiz_fase1 = Path(raiz_fase1) if raiz_fase1 is not None else RAIZ_FASE1
    filas = []
    for fuente in fuentes:
        if fuente not in FUENTES_VINTAGE:
            raise ValueError(f"{fuente!r} no tiene vintage por edición ({list(FUENTES_VINTAGE)})")
        carpeta = ruta_crudo(fuente, raiz_crudo) / CARPETA_EDICIONES
        crudas: dict[str, list[Edicion]] = {}
        for clave in FICHEROS[fuente]:
            crudas[clave] = ediciones(fuente, clave, raiz_crudo)
            for ed in crudas[clave]:
                ruta_meta = ed.ruta.with_name(ed.ruta.name + ".meta.json")
                meta = json.loads(ruta_meta.read_text(encoding="utf-8")) if ruta_meta.exists() else {}
                meta |= {"fuente": fuente, "clave": clave, "edicion": _meta_edicion(ed)}
                sha = ed.sha256 or hashlib.sha256(ed.ruta.read_bytes()).hexdigest()
                destino = carpeta / ed.etiqueta / ed.ruta.name
                filas.append({"fuente": fuente, "clave": clave, "etiqueta": ed.etiqueta, "origen": ed.origen,
                              "fecha_edicion": ed.fecha_edicion.isoformat(), "fuente_fecha": ed.fuente_fecha,
                              "fichero": destino.name, "accion": _copiar(ed.ruta, destino, sha, meta)})
        vistos: set[str] = set()
        for sid in _entradas(cat, [fuente]):
            e = cat[sid]
            if not e["fichero_fase1"] or e.get("fase1_utilizable") is False:
                continue
            rel = e["fichero_fase1"][0].split(":", 1)[0]
            if rel in vistos or not (raiz_fase1 / rel).exists():
                continue
            vistos.add(rel)
            clave = localizar(e)[0]
            f1 = _edicion_fase1(e, raiz_fase1, raiz_crudo, crudas.get(clave, ()))
            if f1 is None:
                continue
            ed = f1[0]
            contenido = ed.ruta.read_bytes()
            sha = hashlib.sha256(contenido).hexdigest()
            destino = carpeta / ed.etiqueta / f"{PREFIJO_FASE1}{ed.ruta.name}"
            meta = {"fuente": fuente, "clave": clave, "origen": "fase1", "ruta_fase1": rel,
                    "descargado_utc": ed.visto.isoformat(), "bytes": len(contenido), "sha256": sha,
                    "edicion": _meta_edicion(ed)}
            filas.append({"fuente": fuente, "clave": clave, "etiqueta": ed.etiqueta, "origen": "fase1",
                          "fecha_edicion": ed.fecha_edicion.isoformat(), "fuente_fecha": ed.fuente_fecha,
                          "fichero": destino.name, "accion": _copiar(ed.ruta, destino, sha, meta)})
    return pd.DataFrame(filas, columns=["fuente", "clave", "etiqueta", "origen", "fecha_edicion", "fuente_fecha",
                                        "fichero", "accion"])


def cambios_ediciones(fuentes: Iterable[str] | None = None, *, catalogo: dict[str, dict] | None = None,
                      raiz_crudo: Path | None = None, raiz_fase1: Path | None = None) -> pd.DataFrame:
    """Cuántos valores cambia cada edición respecto a la anterior, por serie (solo recuentos).

    Una fila por serie y edición posterior a la primera: ``id, fuente, previa, etiqueta, origen,
    fecha_edicion, fecha_efectiva, fuente_fecha, comunes, revisados, nuevos, retirados,
    huecos_nuevos, max_abs`` (``fecha_efectiva``: la que fecha sus filas, ver ``_tabla_ediciones``).
    """
    _, infos, _ = _cargar(fuentes if fuentes is not None else FUENTES_VINTAGE, catalogo, raiz_crudo, raiz_fase1)
    filas = [{"id": sid, "fuente": info.fuente, **c} for sid, info in sorted(infos.items()) for c in info.cambios]
    columnas = ["id", "fuente", "previa", "etiqueta", "origen", "fecha_edicion", "fecha_efectiva", "fuente_fecha",
                "comunes", "revisados", "nuevos", "retirados", "huecos_nuevos", "max_abs"]
    return pd.DataFrame(filas).reindex(columns=columnas)


_COLUMNAS_RESUMEN = ("id", "fuente", "fichero", "columna", "origen", "frecuencia", "metodo", "regla", "estado",
                     "n_filas", "n_obs", "n_huecos", "inicio", "fin", "ediciones", "fecha_edicion",
                     "fuente_fecha", "retraso_edicion_dias", "decision_edicion", "regla_ultimo",
                     "regla_vs_edicion_dias", "provisionales_descartados", "filas_con_cota",
                     "lag_mediano_dias", "lag_min_dias", "lag_max_dias", "fase1_comunes",
                     "fase1_distintos", "fase1_max_abs", "n_periodos", "n_vintages", "vintages",
                     "filas_revisadas", "filas_nuevas", "revision_max_abs")


def _resumen(tabla: pd.DataFrame, infos: dict[str, _Info], cat: dict[str, dict]) -> pd.DataFrame:
    grupos = dict(tuple(tabla.groupby("serie", sort=False))) if len(tabla) else {}
    filas = []
    for sid, info in sorted(infos.items()):
        e = cat[sid]
        fila: dict[str, Any] = {
            "id": sid, "fuente": e["fuente"], "fichero": info.clave, "columna": info.columna,
            "origen": info.origen, "frecuencia": e["frecuencia"], "metodo": e["metodo"],
            "regla": f"{e['lag']} desde el fin de {e['fin_periodo']}", "estado": e["estado"],
            "ediciones": info.ediciones, "provisionales_descartados": info.provisionales,
            "filas_con_cota": info.con_cota, "fase1_comunes": info.revision.get("comunes"),
            "fase1_distintos": info.revision.get("distintos"), "fase1_max_abs": info.revision.get("max_abs")}
        if info.cambios:
            difs = [c["max_abs"] for c in info.cambios if c["max_abs"] is not None]
            fila |= {"filas_revisadas": sum(c["revisados"] for c in info.cambios),
                     "filas_nuevas": sum(c["nuevos"] for c in info.cambios),
                     "revision_max_abs": max(difs) if difs else None}
        g = grupos.get(sid)
        if g is not None and len(g):
            obs = g[g["valor"].notna()]
            lag = (obs["fecha_publicacion"] - obs["fecha_referencia"]).dt.days
            fin = obs["fecha_referencia"].max()
            fila |= {"n_filas": len(g), "n_obs": len(obs), "n_huecos": int(g["valor"].isna().sum()),
                     "inicio": obs["fecha_referencia"].min().date().isoformat(), "fin": fin.date().isoformat(),
                     "lag_mediano_dias": float(lag.median()), "lag_min_dias": int(lag.min()),
                     "lag_max_dias": int(lag.max()), "n_periodos": int(g["fecha_referencia"].nunique()),
                     "n_vintages": int(g["vintage"].nunique()), "vintages": " ".join(sorted(set(g["vintage"])))}
            # Primera edición conocida (crudo o data/raw) que ya tenía el último periodo.
            con_ult = [(pd.Timestamp(u["fecha_edicion"]), u) for u in info.ultimo_por_edicion
                       if u["ultimo"] is not None and pd.Timestamp(u["ultimo"]) >= fin]
            # Una serie retirada (gw:M.csp acaba en 2002) no mide el retraso de la edición actual.
            if con_ult and e["estado"] != "retirada":
                t_ed, u = min(con_ult, key=lambda x: x[0])
                fin_p = cat_mod.fin_de_periodo(pd.Series([fin]), e["fin_periodo"]).iloc[0]
                dec = primera_decision_vec(pd.Series([t_ed], dtype=UTC)).iloc[0]
                regla_ult = cat_mod.fecha_publicacion_regla(pd.Series([fin]), e).iloc[0]
                fila |= {"fecha_edicion": t_ed.isoformat(), "fuente_fecha": u["fuente_fecha"],
                         "retraso_edicion_dias": int((_dia_ny(t_ed) - fin_p).days),
                         "decision_edicion": dec.date().isoformat(), "regla_ultimo": regla_ult.date().isoformat(),
                         "regla_vs_edicion_dias": int((regla_ult - dec).days)}
        filas.append(fila)
    out = pd.DataFrame(filas).reindex(columns=list(_COLUMNAS_RESUMEN))
    for c in ("n_filas", "n_obs", "n_huecos", "n_periodos", "n_vintages"):
        out[c] = out[c].fillna(0).astype(int)
    for c in ("filas_revisadas", "filas_nuevas"):
        out[c] = out[c].astype("Int64")
    return out


def resumen_academicos(fuentes: Iterable[str] | None = None, *, catalogo: dict[str, dict] | None = None,
                       raiz_crudo: Path | None = None, raiz_fase1: Path | None = None) -> pd.DataFrame:
    """Lo medido por serie: cobertura, ediciones, retraso de la edición, revisiones frente a ``data/raw``.

    ``retraso_edicion_dias``: días entre el fin del último periodo cerrado y el día (NY) de la
    edición. ``regla_vs_edicion_dias``: regla del último periodo menos la primera decisión tras la
    edición (negativo = la regla lo daría visible antes de que existiera el fichero, si esa edición
    fue la primera con ese periodo). ``fase1_distintos``: valores de ``data/raw`` cambiados.
    ``n_filas``/``n_obs`` cuentan filas (todas las vintages); ``n_periodos``, fechas de referencia;
    ``filas_revisadas``/``filas_nuevas``: filas de las ediciones posteriores a la primera.
    """
    tabla, infos, cat = _cargar(fuentes, catalogo, raiz_crudo, raiz_fase1)
    return _resumen(tabla, infos, cat)


# --- JST (fuera del catálogo) -----------------------------------------------------------------------
COLUMNAS_JST = {"crisisJST": "indicador 0/1 (crisis bancaria sistémica)", "stir": "% anual", "ltrate": "% anual",
                "cpi": "índice (2005 = 100)", "unemp": "%"}


def tabla_jst(columnas: Sequence[str] = tuple(COLUMNAS_JST), *, iso: str = "USA",
              raiz_crudo: Path | None = None) -> pd.DataFrame:
    """Series anuales de JST R6 de un país (ids ``jst:<ISO>.<columna>``), validadas con el contrato.

    Fuera del catálogo (validación: ``crisisJST`` es la verdad de crisis bancarias). Fecha de
    publicación = SALIDA DEL FICHERO: primera decisión tras la edición (Last-Modified) de la última
    descarga, para todos los años (la R6 se reeditó en 2024); metodo ``fichero``, vintage ``unico``.
    """
    eds = ediciones("jst_macrohistory", "JSTdatasetR6.xlsx", raiz_crudo)
    if not eds:
        return vacio_series()
    ed = eds[-1]
    df = leer_jst(ed.ruta, iso)
    pub = primera_decision_vec(pd.Series([ed.fecha_edicion], dtype=UTC)).iloc[0]
    nota = f"{cat_mod.NOTA_REVISABLE_EDICION}; {NOTA_FUENTE['jst_macrohistory']}"
    tablas = []
    for c in columnas:
        s = df[c].dropna()
        tablas.append(pd.DataFrame({
            "serie": f"jst:{iso}.{c}", "fecha_referencia": s.index, "fecha_publicacion": pub,
            "disponible_utc": pd.NaT, "vintage": VINTAGE, "valor": s.to_numpy(dtype=float),
            "unidad": COLUMNAS_JST.get(c, "según JST"), "divisa": None, "fuente": "jst_macrohistory",
            "metodo_publicacion": "fichero", "nota": nota}))
    return validar_series(pd.concat(tablas, ignore_index=True)[list(COLUMNAS_SERIES)])


# --- Fichas ------------------------------------------------------------------------------------------
_NOMBRES = {
    "ken_french": "Kenneth R. French Data Library (factores Fama-French, momentum, 5 industrias)",
    "shiller": "Robert J. Shiller: ie_data.xls (S&P Composite, dividendos, beneficios, CPI, tipo largo, CAPE)",
    "goyal_welch": "Goyal-Welch: predictores de la prima de riesgo (hoja mensual)",
    "ofr_fsi": "OFR Financial Stress Index (fsi.csv)",
    "jst_macrohistory": "Jordà-Schularick-Taylor Macrohistory Database R6",
}
_URLS = {
    "ken_french": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
    "shiller": PAGINA_SHILLER,
    "goyal_welch": "https://sites.google.com/view/agoyal145",
    "ofr_fsi": "https://www.financialresearch.gov/financial-stress-index/",
    "jst_macrohistory": "https://www.macrohistory.net/database/",
}
_FRECUENCIA = {"ken_french": "diaria y mensual; un fichero nuevo al mes", "shiller": "mensual; edición irregular",
               "goyal_welch": "mensual; edición anual", "ofr_fsi": "diaria; fichero actualizado cada día hábil",
               "jst_macrohistory": "anual; ediciones R1-R6 (R6 de 2022, reeditada en 2024)"}


def _ultima_meta(fuente: str, raiz: Path | None) -> list[dict[str, Any]]:
    out = []
    for clave in FICHEROS[fuente]:
        nombre = FICHEROS[fuente][clave][0]
        metas = sorted(m for m in ruta_crudo(fuente, raiz).glob(f"[0-9]*_{nombre}.meta.json")
                       if re.fullmatch(rf"\d{{8}}_{re.escape(nombre)}\.meta\.json", m.name))
        if metas:
            out.append(json.loads(metas[-1].read_text(encoding="utf-8")))
    return out


def _ediciones_medidas(fuente: str, raiz: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for clave in FICHEROS[fuente]:
        eds = ediciones(fuente, clave, raiz)
        if not eds:
            continue
        lista = []
        for e in eds:
            d: dict[str, Any] = {"etiqueta": e.etiqueta, "fecha_edicion": e.fecha_edicion.isoformat(),
                                 "fuente_fecha": e.fuente_fecha, "visto": e.visto.isoformat(),
                                 "bytes": e.ruta.stat().st_size}
            if fuente == "ken_french":
                _, info = leer_french(e.ruta)
                d |= {"crsp": info["crsp"], "creado_zip": info["creado_zip"]}
            if e.ruta.suffix.lower() == ".xlsx":
                xm = _xlsx_modificado(e.ruta)
                d["xlsx_modificado"] = xm.isoformat() if xm is not None else None
            lista.append(d)
        out[clave] = lista
    return out


def _archivo_medido(fuente: str, raiz: Path | None) -> dict[str, list[str]]:
    """Ficheros del archivo de ediciones por etiqueta (``ediciones/<AAAA-MM>/``), sin los .meta.json."""
    carpeta = ruta_crudo(fuente, raiz) / CARPETA_EDICIONES
    if not carpeta.exists():
        return {}
    return {d.name: sorted(p.name for p in d.iterdir() if not p.name.endswith(".meta.json"))
            for d in sorted(carpeta.iterdir()) if d.is_dir()}


def _py(v: Any) -> Any:
    """Escalar de numpy/pandas → tipo de Python (None si es nulo) para el JSON de la ficha."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    return v.item() if hasattr(v, "item") else v


def _ficha(fuente: str, res: pd.DataFrame, raiz: Path | None, extra: dict[str, Any]):
    from ingesta.comun.esquema import FichaFuente

    metas = _ultima_meta(fuente, raiz)
    errores = [m for m in metas if m.get("error")]
    r = res[(res["fuente"] == fuente) & (res["n_obs"] > 0)] if len(res) else res
    vigentes = r[r["estado"] != "retirada"] if len(r) else r
    eds = _ediciones_medidas(fuente, raiz)
    mediciones: dict[str, Any] = {"ediciones": eds, "descargas": [
        {k: m.get(k) for k in ("clave", "estado", "bytes", "descargado_utc", "error")} for m in metas]}
    if fuente in FUENTES_VINTAGE:
        mediciones["archivo_ediciones"] = _archivo_medido(fuente, raiz)
    if len(r):
        mediciones["series"] = {
            row.id: {k: _py(v) for k, v in row._asdict().items()
                     if k in ("n_obs", "n_huecos", "inicio", "fin", "retraso_edicion_dias", "regla_vs_edicion_dias",
                              "provisionales_descartados", "filas_con_cota", "fase1_comunes", "fase1_distintos",
                              "fase1_max_abs", "origen", "n_periodos", "n_vintages", "vintages",
                              "filas_revisadas", "filas_nuevas", "revision_max_abs")}
            for row in r.itertuples(index=False)}
    mediciones |= extra.get("mediciones", {})
    if len(r):
        ini, fin = r["inicio"].min(), r["fin"].max()
        ret = vigentes["retraso_edicion_dias"].dropna()
        regla = vigentes["regla_vs_edicion_dias"].dropna()
        reglas = " y ".join(f"{x} ({int((r['regla'] == x).sum())} series)" for x in sorted(set(r["regla"])))
        retraso = (f"medido con las ediciones archivadas (data/raw de la Fase 1 y descargas desde el bloque 2; "
                   f"las anteriores no están medidas): la primera edición conocida con el último periodo sale {int(ret.min())}-"
                   f"{int(ret.max())} días tras su fin (series vigentes); reglas del catálogo: {reglas} (regla - "
                   f"primera decisión tras esa edición: {int(regla.min())} a {int(regla.max())} días)"
                   ) if len(ret) else "sin edición medida"
    else:
        ini, fin = extra.get("cobertura", (None, None))
        retraso = extra.get("retraso", "no medido")
    volumen = sum(e["bytes"] for v in eds.values() for e in v[-1:])
    estado = "fallo" if metas and len(errores) == len(metas) else ("parcial" if errores else
                                                                     ("ok" if metas else "no_probada"))
    return FichaFuente(
        fuente=fuente, nombre=_NOMBRES[fuente], tema=TEMA,
        agentes=["macro", "risk"] if fuente in ("ofr_fsi", "jst_macrohistory") else ["equity", "macro", "portfolio"],
        rag="serie", url=_URLS[fuente], acceso="descarga", autenticacion="ninguna", coste="0 €",
        licencia_y_terminos=LICENCIAS[fuente], cobertura_inicio=ini, cobertura_fin=fin,
        frecuencia=_FRECUENCIA[fuente], retraso_publicacion=retraso,
        formato={"ken_french": "zip con CSV", "shiller": "xls", "goyal_welch": "xlsx (Google Sheet exportada)",
                 "ofr_fsi": "csv", "jst_macrohistory": "xlsx"}[fuente],
        volumen_estimado=f"{volumen / 1e6:.2f} MB por edición ({len(FICHEROS[fuente])} fichero(s))",
        limites=extra.get("limites", "1 petición/s por host; robots.txt permite la descarga"),
        estado_sonda=estado, mediciones=mediciones,
        notas=[f"Cargador: ingesta.mercados_series.fuentes.academicos (crudos en <corpus>/mercados_series/crudo/"
               f"{fuente}/, un fichero por día de descarga con su .meta.json)", *extra.get("notas", [])])


def _copias_fase1(fuente: str, cat: dict[str, dict], raiz_fase1: Path) -> dict[str, Any]:
    """Copias de ``data/raw`` de una fuente: día de descarga (mtime en NY), último dato y fechas repetidas."""
    out: dict[str, Any] = {}
    for sid in cat_mod.filtrar(cat, fuente=fuente):
        if not cat[sid]["fichero_fase1"]:
            continue
        rel, col = cat[sid]["fichero_fase1"][0].split(":", 1)
        ruta = Path(raiz_fase1) / rel
        if rel in out or not ruta.exists():
            continue
        df = pd.read_parquet(ruta)
        idx = pd.DatetimeIndex(pd.to_datetime(df.index))
        s = df[col] if col in df.columns else df.iloc[:, 0]
        out[rel] = {"descarga_ny": _mtime_utc(ruta).tz_convert(HUSO_DECISION).strftime("%Y-%m-%d %H:%M"),
                    "ultimo": idx[s.notna().to_numpy()].max().date().isoformat(),
                    "fechas_repetidas": int(idx.duplicated().sum())}
        if cat[sid]["frecuencia"] == "mensual":
            out[rel]["octubres"] = int((idx.month == 10).sum())
    return out


def fichas_academicos(res: pd.DataFrame, *, raiz_crudo: Path | None = None, catalogo: dict[str, dict] | None = None,
                      raiz_fase1: Path | None = None) -> list:
    """``FichaFuente`` de las cinco fuentes con lo medido en ``res`` (``resumen_academicos``) y los crudos.

    Añade las copias de ``data/raw`` (día de descarga, último dato, fechas repetidas) como ediciones
    antiguas medidas.
    """
    nota_ed = ("fecha_publicacion = max(regla del catálogo, primera decisión tras la última vez que se vio una "
               "edición sin ese periodo); periodos no cerrados en la fecha de edición: provisionales, fuera")
    nota_vintage = ("Vintage por edición AAAA-MM (mes NY de la fecha de edición): la primera edición archivada con "
                    "la regla del catálogo; cada edición posterior solo con los periodos nuevos o revisados, "
                    "fechados con su edición (Last-Modified, xlsx o descarga). Archivo en <corpus>/mercados_series/"
                    "crudo/<fuente>/ediciones/<AAAA-MM>/ (archivar_ediciones); periodos no cerrados en la fecha "
                    "de edición: provisionales, fuera")
    extra: dict[str, dict[str, Any]] = {
        "ken_french": {"notas": [nota_vintage, "Primer bloque de cada CSV (industrias: ponderadas por valor); -99.99 y "
                                 "-999 son faltantes; retornos en %",
                                 "La cabecera 'created using the AAAAMM CRSP database' da el último mes; el zip "
                                 "lleva la fecha interna de creación",
                                 "Revisión completa del momentum diario entre ediciones: data/raw (2026-07-20) y la "
                                 "edición del 2026-09-25 difieren en 14699 de 26152 días comunes (máx. 3,79 pp, "
                                 "correlación 0,994, sin desfase de fechas); el resto de ficheros, 0 diferencias "
                                 "salvo 1 de 0,01 en F3M. Con vintage por edición la base guarda las dos: "
                                 "data/raw (vintage 2026-07, primera) y las filas revisadas o nuevas de 2026-09",
                                 "data/raw de los diarios de 3 y 5 factores y de las 5 industrias (descarga del "
                                 "2026-09-29) tiene el mismo último mes (CRSP 202608) que el crudo del bloque 2: es "
                                 "la misma edición (Last-Modified 2026-09-25) y no cambia ningún valor",
                                 "Regla +45 días medida con una sola edición (agosto-2026, Last-Modified a 25 días); "
                                 "las ediciones históricas (p. ej. el cambio de CRSP a CIZ en 2023) no se pudieron "
                                 "medir: la Wayback Machine respondió 429. Pendiente con más calma"]},
        "shiller": {"notas": [nota_vintage, "Fechas AAAA.MM: 1871.1 es octubre (regresión del bug de la Fase 1: 153 "
                                       "octubres leídos como enero en SHILLER_LONGRATE)",
                              "La última fila es el mes en curso con el cierre del día 1 (nota al pie de la hoja): "
                              "se descarta; CPI de los últimos meses estimado por Shiller",
                              "El espejo de Yale (econ.yale.edu) está congelado en 2024-09: no se usa"]},
        "goyal_welch": {"notas": [nota_vintage, "Sin Last-Modified (Google): fecha de edición = dcterms:modified del xlsx",
                                  "Regla 7M1D (1 de agosto del año siguiente) sin ediciones antiguas medidas",
                                  "Series retiradas (gw:M.csp acaba en 2002): fuera de las métricas de retraso de "
                                  "la edición (no miden la edición actual)",
                                  "Content-Disposition da el nombre de la edición (PredictorData<AAAA>.xlsx)"],
                        "limites": "Google redirige a googleusercontent; 1 petición/s"},
        "ofr_fsi": {"notas": [nota_ed, "Descarga con el User-Agent identificable de ingesta.comun.http: HTTP 200 "
                                       "(sin 403); si la OFR bloquea, se usa la copia de data/raw con su regla",
                              "metodo lag_fijo (catálogo): 5D desde la fecha"]},
        "jst_macrohistory": {"notas": ["Fuera del catálogo: tabla_jst (ids jst:<ISO>.<col>), fecha = primera "
                                       "decisión tras la edición (Last-Modified) para todos los años",
                                       "La Fase 1 falló al descargarla; reintento OK"]},
    }
    try:
        eds = ediciones("jst_macrohistory", "JSTdatasetR6.xlsx", raiz_crudo)
    except ValueError:
        eds = []
    if eds:
        df = pd.read_excel(eds[-1].ruta, sheet_name=0)
        usa = df[df["iso"] == "USA"]
        extra["jst_macrohistory"] |= {
            "cobertura": (f"{int(df['year'].min())}-01-01", f"{int(df['year'].max())}-12-31"),
            "retraso": (f"R6 (datos hasta {int(df['year'].max())}) editada el "
                        f"{eds[-1].fecha_edicion.date().isoformat()} (Last-Modified); creada en 2022"),
            "mediciones": {"filas": int(len(df)), "paises": int(df["iso"].nunique()), "columnas": int(df.shape[1]),
                           "usa_anios": int(len(usa)),
                           "usa_crisis": [int(a) for a in usa.loc[usa["crisisJST"] == 1, "year"]]}}
    cat = catalogo if catalogo is not None else cat_mod.cargar_catalogo()
    raiz_fase1 = Path(raiz_fase1) if raiz_fase1 is not None else RAIZ_FASE1
    for f in FUENTES_CATALOGO:
        copias = _copias_fase1(f, cat, raiz_fase1)
        if copias:
            extra[f].setdefault("mediciones", {})["copias_data_raw"] = copias
    return [_ficha(f, res, raiz_crudo, extra[f]) for f in FUENTES]


# --- CLI ---------------------------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> None:
    from ingesta.comun.fichas import guardar_ficha

    p = argparse.ArgumentParser(description="Fuentes académicas -> tabla larga del contrato de series")
    p.add_argument("--descargar", action="store_true", help="descarga la edición de hoy (si no está archivada)")
    p.add_argument("--archivar", action="store_true",
                   help="copia las ediciones (crudos y data/raw) en crudo/<fuente>/ediciones/<AAAA-MM>/")
    p.add_argument("--fuentes", nargs="*", default=None, help=f"subconjunto de {list(FUENTES)}")
    p.add_argument("--fichas", action="store_true",
                   help="escribe fichas y tablas/academicos_resumen.csv y academicos_ediciones.csv")
    p.add_argument("--escribir", action="store_true", help="vuelca la tabla en el almacén del tema")
    p.add_argument("--reemplazar", action="store_true",
                   help="con --escribir: permite cambiar filas guardadas (revisiones del último vintage)")
    a = p.parse_args(argv)

    fuentes = a.fuentes or list(FUENTES)
    if a.descargar:
        for f in fuentes:
            for m in descargar(f):
                print(f"{f:16s} {m['clave']:40s} {m.get('estado')} {m.get('bytes')} {m.get('error') or ''}")
    if a.archivar:
        arch = archivar_ediciones([f for f in fuentes if f in FUENTES_VINTAGE])
        for r in arch.itertuples(index=False):
            print(f"{r.fuente:12s} {r.etiqueta} {r.origen:6s} {r.fichero:50s} {r.accion}")
    cat_f = [f for f in fuentes if f in FUENTES_CATALOGO]
    tabla, infos, cat = _cargar(cat_f, None, None, None)
    tabla = validar_series(tabla)
    cat_mod.validar_contra_catalogo(tabla, cat)
    res = _resumen(tabla, infos, cat)
    print(f"{int((res['n_obs'] > 0).sum())} series con datos de {len(res)}; {len(tabla)} filas")
    if a.fichas:
        carpeta = resultados_dir(TEMA) / "tablas"
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta_csv = carpeta / "academicos_resumen.csv"
        res.to_csv(ruta_csv, index=False, encoding="utf-8")
        print(f"resumen -> {ruta_csv}")
        cambios = pd.DataFrame([{"id": sid, "fuente": i.fuente, **c} for sid, i in sorted(infos.items())
                                for c in i.cambios])
        ruta_ed = carpeta / "academicos_ediciones.csv"
        cambios.to_csv(ruta_ed, index=False, encoding="utf-8")
        print(f"cambios entre ediciones -> {ruta_ed}")
        for ficha in fichas_academicos(res):
            if ficha.fuente in fuentes:
                print(f"ficha   -> {guardar_ficha(ficha)}")
    if a.escribir:
        from ingesta.mercados_series import almacen

        print(f"almacén -> {almacen.escribir_series(tabla, permitir_perdidas=a.reemplazar)}")


if __name__ == "__main__":
    main()
