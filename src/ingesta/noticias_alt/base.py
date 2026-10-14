"""Contrato interno del Tema 4: configuración, registro, tablas, documentos y series.

Cada módulo de fuente expone ``sondear(cfg: ConfigSonda) -> Resultado`` y solo usa lo de aquí y
``ingesta.comun``. ``sonda.py`` junta los ``Resultado`` en las tablas comunes del tema:

- ``registro_descargas``: una fila por petición (``COLUMNAS_REGISTRO``).
- ``cobertura``: una fila por fuente (``COLUMNAS_COBERTURA``), base del mapa por épocas.
- ``timegate``: ejemplos de fecha del hecho frente a fecha de publicación (``COLUMNAS_TIMEGATE``).
- tablas propias de cada módulo en ``Resultado.tablas`` (``<nombre>.csv`` en results).

Copyright: el texto de noticias solo va al corpus (``corpus(...)``). A ``results/`` y al repo solo
metadatos, cifras y titulares cortos.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from ingesta.comun.esquema import Documento, FichaFuente
from ingesta.comun.rutas import corpus_dir, resultados_dir
from ingesta.noticias_alt.muestra import DiaTestigo, dias_testigo

TEMA = "noticias_alt"
LIC_COPYRIGHT = "copyright del medio: texto solo en corpus local (uso académico); al repo solo metadatos"

COLUMNAS_REGISTRO = ["fuente", "item", "dia", "url", "ok", "bytes", "segundos", "error"]
COLUMNAS_COBERTURA = [
    "fuente", "familia", "tipo", "inicio", "fin", "hora", "idiomas", "lag_mediana_h", "destino",
    "backtest", "estado", "nota",
]
COLUMNAS_TIMEGATE = ["fuente", "item", "fecha_hecho", "fecha_publicacion", "campo_fecha", "lag_h"]
COLUMNAS_SERIE = [
    "serie", "fecha_referencia", "fecha_publicacion", "valor", "unidad", "fuente", "vintage",
    "disponible_utc", "metodo_publicacion", "nota",
]
SUFIJO_RAPIDO = "_rapido"      # salidas del modo rápido: nunca pisan las de la pasada completa
REVISABLE = "vintage_actual_revisable"   # metodo_publicacion de series guardadas con el vintage de hoy (revisadas)
TIPOS = ("texto", "titular", "serie", "evento", "instantanea", "ficha")
DESTINOS = ("cajon", "base", "serie", "base+cajon", "vivo", "ninguno")


@dataclass
class ConfigSonda:
    """Configuración común. ``modo_rapido`` recorta cada fuente a lo mínimo (pruebas y CI)."""

    dias: list[DiaTestigo] = field(default_factory=dias_testigo)
    modo_rapido: bool = False
    usar_cache: bool = True
    verbose: bool = True
    max_items: int | None = None          # tope por fuente y día (None = el de cada módulo)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def tope(self) -> int | None:
        return 3 if self.modo_rapido and self.max_items is None else self.max_items

    def log(self, *a: Any) -> None:
        if self.verbose:
            print(*a, flush=True)


@dataclass
class Resultado:
    """Lo que devuelve ``sondear`` de cada módulo."""

    modulo: str
    registro: list[dict] = field(default_factory=list)
    cobertura: list[dict] = field(default_factory=list)
    timegate: list[dict] = field(default_factory=list)
    tablas: dict[str, pd.DataFrame] = field(default_factory=dict)
    fichas: list[FichaFuente] = field(default_factory=list)
    series: list[str] = field(default_factory=list)       # nombres de parquet guardados en el corpus
    documentos: dict[str, int] = field(default_factory=dict)  # jsonl -> nº de documentos

    def registrar(self, fuente: str, item: str, url: str, ok: bool, *, dia: str = "", bytes: int = 0,
                  segundos: float = 0.0, error: str = "") -> None:
        self.registro.append(dict(fuente=fuente, item=item, dia=dia, url=url, ok=bool(ok), bytes=int(bytes),
                                  segundos=round(float(segundos), 2), error=str(error)[:300]))


# --------------------------------------------------------------------------- utilidades

def cronometro() -> float:
    return time.monotonic()


def segundos(t0: float) -> float:
    return round(time.monotonic() - t0, 2)


def corpus(*partes: str) -> Path:
    """Carpeta del corpus del tema (fuera del repo), creada si no existe."""
    p = corpus_dir(TEMA).joinpath(*partes)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ficha(**kw: Any) -> FichaFuente:
    """``FichaFuente`` con los valores por defecto del tema (se sobreescriben con ``kw``)."""
    base: dict[str, Any] = dict(tema=TEMA, acceso="api", autenticacion="ninguna", coste="0 €",
                                cobertura_inicio=None, cobertura_fin=None, mediciones={}, notas=[])
    base.update(kw)
    return FichaFuente(**base)


def estado(registro: Iterable[dict], fuente: str | None = None) -> str:
    """'fallo' si nada fue bien, 'ok' si al menos el 90 %, si no 'parcial' (como el Tema 3)."""
    filas = [r for r in registro if fuente is None or r["fuente"] == fuente]
    if not filas:
        return "no_probada"
    pct = sum(r["ok"] for r in filas) / len(filas)
    return "fallo" if pct == 0 else ("ok" if pct >= 0.9 else "parcial")


def lag_horas(hecho: datetime | date | None, publicado: datetime | date | None) -> float | None:
    """Horas entre el hecho y su publicación (None si falta alguna)."""
    if hecho is None or publicado is None:
        return None
    a, b = pd.Timestamp(hecho), pd.Timestamp(publicado)
    if (a.tzinfo is None) != (b.tzinfo is None):
        a, b = a.tz_localize(None) if a.tzinfo else a, b.tz_localize(None) if b.tzinfo else b
    return round((b - a).total_seconds() / 3600, 2)


def resumen_lag(lags: Iterable[float | None]) -> dict[str, float | None]:
    s = pd.Series([x for x in lags if x is not None], dtype=float)
    if s.empty:
        return dict(n=0, mediana_h=None, p10_h=None, p90_h=None)
    return dict(n=int(s.size), mediana_h=round(float(s.median()), 2), p10_h=round(float(s.quantile(.1)), 2),
                p90_h=round(float(s.quantile(.9)), 2))


# --------------------------------------------------------------------------- caché de crudos

def escribir_atomico(ruta: Path, datos: bytes) -> Path:
    """Escribe ``datos`` en ``ruta`` vía un ``.tmp`` propio + ``os.replace``: un corte no deja un crudo truncado.

    El ``.tmp`` lleva proceso e hilo (descargas en paralelo del mismo fichero). Si otro hilo ya dejó el fichero
    y Windows bloquea el reemplazo, se conserva el suyo.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_name(f"{ruta.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(datos)
    try:
        os.replace(tmp, ruta)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        if not ruta.exists():
            raise
    return ruta


def get_cacheado(res: Resultado, cfg: ConfigSonda, fuente: str, item: str, url: str, ruta: Path | None, *,
                 getter: Callable[..., Any], dia: str = "", max_edad_h: float | None = None,
                 validar: Callable[[bytes], Any] | None = None, registrar_fallo: bool = True,
                 marca_cache: str = " (caché)", guardar_meta: bool = False, **kw: Any) -> bytes | None:
    """GET con caché en ``ruta`` (escritura atómica); registra la petición. None si falla.

    - Reutiliza la caché si ``cfg.usar_cache``, no está vacía, no supera ``max_edad_h`` y pasa ``validar``
      (si no lo pasa, se borra y se vuelve a pedir: así un crudo corrupto no queda para siempre).
    - ``getter`` es el ``get`` del módulo que llama (lo parchean los tests).
    - ``validar(datos)`` debe lanzar si el contenido no sirve (zip truncado, JSON roto...): cuenta como fallo y
      no se guarda.
    - ``guardar_meta``: guarda al lado ``<ruta>.meta.json`` con ``Last-Modified`` (``leer_meta``); una caché sin
      ese fichero se vuelve a pedir.
    """
    sin_meta = guardar_meta and ruta is not None and not ruta_meta(ruta).exists()
    if ruta is not None and cfg.usar_cache and ruta.exists() and ruta.stat().st_size > 0 and not sin_meta:
        fresca = max_edad_h is None or (time.time() - ruta.stat().st_mtime) / 3600 <= max_edad_h
        if fresca:
            datos = ruta.read_bytes()
            try:
                if validar is not None:
                    validar(datos)
            except Exception:  # noqa: BLE001
                ruta.unlink(missing_ok=True)
                cfg.log(f"  [{fuente}] caché corrupta, se vuelve a bajar: {ruta.name}")
            else:
                res.registrar(fuente, f"{item}{marca_cache}", url, True, dia=dia, bytes=len(datos))
                return datos
    t0 = cronometro()
    try:
        r = getter(url, **kw)
        r.raise_for_status()
        datos = r.content
        if validar is not None:
            validar(datos)
    except Exception as e:  # noqa: BLE001
        if registrar_fallo:
            res.registrar(fuente, item, url, False, dia=dia, segundos=segundos(t0), error=f"{type(e).__name__}: {e}")
            cfg.log(f"  [{fuente}] fallo {item}: {type(e).__name__}: {str(e)[:120]}")
        return None
    res.registrar(fuente, item, url, True, dia=dia, bytes=len(datos), segundos=segundos(t0))
    if ruta is not None:
        escribir_atomico(ruta, datos)
        if guardar_meta:
            meta = dict(url=url, last_modified=(getattr(r, "headers", None) or {}).get("Last-Modified"),
                        bytes=len(datos), descargado_utc=datetime.now(UTC).isoformat(timespec="seconds"))
            escribir_atomico(ruta_meta(ruta), json.dumps(meta).encode("utf-8"))
    return datos


def ruta_meta(ruta: Path) -> Path:
    return ruta.with_name(ruta.name + ".meta.json")


def leer_meta(ruta: Path) -> dict[str, Any]:
    """Cabeceras guardadas por ``get_cacheado(..., guardar_meta=True)`` ({} si no hay)."""
    try:
        return json.loads(ruta_meta(ruta).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# --------------------------------------------------------------------------- documentos

def nombre_salida(nombre: str, rapido: bool = False) -> str:
    """Nombre de un jsonl/parquet del corpus: con sufijo ``_rapido`` en el modo rápido."""
    return nombre + SUFIJO_RAPIDO if rapido and not nombre.endswith(SUFIJO_RAPIDO) else nombre


def documento_dict(doc: Documento, texto: str | None = None) -> dict[str, Any]:
    """``Documento`` serializado + ``texto`` (solo para el corpus; pendiente de acordar en ``comun``)."""
    d = doc.to_dict()
    if texto is not None:
        d["texto"] = texto
    return d


def guardar_documentos(docs: Iterable[dict], nombre: str, *, rapido: bool = False) -> tuple[Path, int]:
    """Escribe ``<corpus>/noticias_alt/documentos/<nombre>.jsonl`` (atómico) y devuelve (ruta, n).

    ``rapido``: añade ``_rapido`` al nombre (``ruta.stem`` es el nombre final) para no pisar la pasada completa.
    """
    ruta = corpus("documentos") / f"{nombre_salida(nombre, rapido)}.jsonl"
    tmp = ruta.with_suffix(".jsonl.tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
            n += 1
    os.replace(tmp, ruta)
    return ruta, n


def leer_documentos(nombre: str) -> list[dict]:
    ruta = corpus("documentos") / f"{nombre}.jsonl"
    if not ruta.exists():
        return []
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------- series

def serie_larga(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza a tabla larga (``COLUMNAS_SERIE``) y valida el TimeGate básico.

    Obligatorias: serie, fecha_referencia, fecha_publicacion, valor, unidad, fuente. Las fechas son
    días (``datetime64``); ``fecha_publicacion >= fecha_referencia`` (sin previsiones).
    """
    falta = {"serie", "fecha_referencia", "fecha_publicacion", "valor", "unidad", "fuente"} - set(df.columns)
    if falta:
        raise ValueError(f"faltan columnas en la serie: {sorted(falta)}")
    out = df.copy()
    for c in COLUMNAS_SERIE:
        if c not in out.columns:
            out[c] = None
    for c in ("fecha_referencia", "fecha_publicacion"):
        out[c] = pd.to_datetime(out[c]).dt.tz_localize(None).dt.normalize()
    out["valor"] = pd.to_numeric(out["valor"], errors="coerce").astype(float)
    malas = out["fecha_publicacion"] < out["fecha_referencia"]
    if malas.any():
        raise ValueError(f"{int(malas.sum())} filas con fecha_publicacion < fecha_referencia")
    return out[COLUMNAS_SERIE].sort_values(["serie", "fecha_referencia"]).reset_index(drop=True)


def guardar_serie(df: pd.DataFrame, nombre: str, *, rapido: bool = False) -> Path:
    """Guarda ``<corpus>/noticias_alt/series/<nombre>.parquet`` (validada con ``serie_larga``).

    ``rapido``: añade ``_rapido`` al nombre (``ruta.stem`` es el nombre final) para no pisar la pasada completa.
    """
    ruta = corpus("series") / f"{nombre_salida(nombre, rapido)}.parquet"
    tmp = ruta.with_suffix(".parquet.tmp")
    serie_larga(df).to_parquet(tmp, index=False)
    os.replace(tmp, ruta)
    return ruta


def leer_serie(nombre: str) -> pd.DataFrame:
    ruta = corpus("series") / f"{nombre}.parquet"
    return pd.read_parquet(ruta) if ruta.exists() else pd.DataFrame(columns=COLUMNAS_SERIE)


def a_fecha_publicacion(serie: pd.DataFrame, fecha: date | str, *, excluir_revisables: bool = False) -> pd.DataFrame:
    """Vista TimeGate: filas publicadas hasta ``fecha`` (incluida).

    ``excluir_revisables``: quita las filas con ``metodo_publicacion == REVISABLE`` (valor del vintage de hoy,
    revisado después de su publicación: la fecha es correcta pero el valor no es el que se veía entonces).
    """
    out = serie[pd.to_datetime(serie["fecha_publicacion"]) <= pd.Timestamp(fecha)]
    if excluir_revisables:
        out = out[out["metodo_publicacion"] != REVISABLE]
    return out


# --------------------------------------------------------------------------- results

def guardar_tabla(df: pd.DataFrame, nombre: str, carpeta: Path | None = None) -> Path:
    """``results/ingesta/noticias_alt/<nombre>.csv`` (atómico)."""
    carpeta = carpeta or resultados_dir(TEMA)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{nombre}.csv"
    tmp = ruta.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, ruta)
    return ruta
