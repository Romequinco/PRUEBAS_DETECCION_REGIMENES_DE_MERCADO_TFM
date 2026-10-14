"""Common Crawl CC-NEWS (2016-08 →): noticias de todo el mundo con TEXTO completo, en WARC.

Es la única fuente abierta y gratuita con cuerpo de noticia, multilingüe y con fecha fiable para
el backtest desde finales de 2016. Cada mes tiene cientos de WARC de ~1 GB
(``crawl-data/CC-NEWS/<AAAA>/<MM>/warc.paths.gz``); cada registro es un miembro gzip
independiente, así que la sonda lee con ``Range`` unos MB del WARC del día testigo y descomprime
solo los miembros completos (sin bajar el GB).

TimeGate (``fecha_timegate``): ``WARC-Date`` (rastreo) es una cota SUPERIOR segura de cuándo
existía la página. La fecha de los metadatos del artículo (``article:published_time`` y
similares) se usa solo si es ≤ ``WARC-Date``; si es posterior, la página se reescribió después y
se queda la de rastreo (regla ``rastreo_meta_posterior``); sin metadato, también la de rastreo.

Copyright: el texto va solo al corpus (``documentos/ccnews_muestra.jsonl``); en ``results`` solo
metadatos y titulares (tabla ``ccnews_articulos``).
"""
from __future__ import annotations

import gzip
import re
import zlib
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from ingesta.comun.esquema import Documento, doc_id
from ingesta.comun.http import get
from ingesta.noticias_alt import base
from ingesta.noticias_alt.limpieza import (a_fecha, clave_url, deduplicar, dominio, embudo_duplicados, idioma,
                                           tokens_aprox)

FUENTE = "ccnews"
BASE = "https://data.commoncrawl.org/"
INICIO_CCNEWS = date(2016, 8, 1)
BYTES_DIA = 3_000_000            # lectura parcial por día testigo (modo completo)
BYTES_DIA_RAPIDO = 600_000
MIN_TOKENS = 100                 # por debajo es una portada/índice, no un artículo
LICENCIA = ("Common Crawl Terms of Use (uso libre de los datos de rastreo); el texto de cada artículo "
            "conserva el copyright de su medio: solo corpus local, nunca al repo")
_META_FECHA = ("article:published_time", "og:published_time", "datePublished", "datepublished",
               "article.published", "pubdate", "publishdate", "parsely-pub-date", "sailthru.date",
               "dc.date", "DC.date.issued", "date")
_RUTA_TS = re.compile(r"CC-NEWS-(\d{14})-\d+\.warc\.gz")


# --------------------------------------------------------------------------- listados de WARC

def url_paths(anio: int, mes: int) -> str:
    return f"{BASE}crawl-data/CC-NEWS/{anio}/{mes:02d}/warc.paths.gz"


def parse_paths(contenido: bytes) -> list[str]:
    """``warc.paths.gz`` (gzip o ya descomprimido) -> rutas ordenadas."""
    if contenido[:2] == b"\x1f\x8b":
        contenido = gzip.decompress(contenido)
    return sorted(l.strip() for l in contenido.decode("utf-8", "replace").splitlines() if l.strip())


def fecha_ruta(ruta: str) -> datetime | None:
    """Inicio de rastreo del WARC según su nombre (``CC-NEWS-20200316120000-00101.warc.gz``)."""
    m = _RUTA_TS.search(ruta)
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=UTC) if m else None


def elegir_warc(rutas: list[str], dia: date) -> tuple[str, bool]:
    """WARC del día (el central de los que empiezan ese día) o, si no hay, el más cercano.

    Devuelve ``(ruta, exacto)``; ``exacto=False`` significa que el mes no tiene WARC ese día.
    """
    con_fecha = [(r, fecha_ruta(r)) for r in rutas]
    con_fecha = [(r, f) for r, f in con_fecha if f is not None]
    if not con_fecha:
        raise ValueError("listado de WARC vacío o sin fechas en el nombre")
    del_dia = [r for r, f in con_fecha if f.date() == dia]
    if del_dia:
        return del_dia[len(del_dia) // 2], True
    objetivo = datetime(dia.year, dia.month, dia.day, 12, tzinfo=UTC)
    return min(con_fecha, key=lambda rf: abs(rf[1] - objetivo))[0], False


# --------------------------------------------------------------------------- WARC

def miembros_gzip(datos: bytes) -> list[bytes]:
    """Descomprime miembros gzip concatenados hasta el primero incompleto (trozo leído con Range)."""
    salida = []
    while datos:
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            bloque = d.decompress(datos)
        except zlib.error:
            break
        if not d.eof:
            break
        salida.append(bloque)
        datos = d.unused_data
    return salida


def _cabeceras(txt: bytes) -> dict[str, str]:
    h: dict[str, str] = {}
    for l in txt.decode("utf-8", "replace").split("\r\n"):
        if ": " in l:
            k, v = l.split(": ", 1)
            h[k.strip().lower()] = v.strip()
    return h


def registros_warc(miembros: list[bytes]) -> Iterator[dict[str, Any]]:
    """Registros ``response`` (uno por miembro gzip): URL, ``WARC-Date`` y cuerpo HTML."""
    for m in miembros:
        if not m.startswith(b"WARC/"):
            continue
        cab, _, resto = m.partition(b"\r\n\r\n")
        h = _cabeceras(cab)
        if h.get("warc-type") != "response":
            continue
        cab_http, _, cuerpo = resto.partition(b"\r\n\r\n")
        hh = _cabeceras(cab_http)
        if "gzip" in hh.get("content-encoding", "") and cuerpo[:2] == b"\x1f\x8b":
            try:
                cuerpo = gzip.decompress(cuerpo)
            except (OSError, EOFError):
                pass
        estado = cab_http.split(b" ", 2)[1:2]
        yield {"url": h.get("warc-target-uri", ""), "warc_date": a_fecha(h.get("warc-date")),
               "http_estado": int(estado[0]) if estado and estado[0].isdigit() else None,
               "html": cuerpo}


def extraer_articulo(html: bytes | str) -> dict[str, Any]:
    """Título, fecha de metadatos (y su campo), idioma declarado, medio y texto (``<p>`` > 40 car.)."""
    from bs4 import BeautifulSoup

    s = BeautifulSoup(html, "lxml")
    meta: dict[str, str] = {}
    for m in s.find_all("meta"):
        k = m.get("property") or m.get("name") or m.get("itemprop")
        if k and m.get("content") and k.strip() not in meta:
            meta[k.strip()] = m["content"].strip()
    fecha, campo = None, None
    for k in _META_FECHA:
        if k in meta and (f := a_fecha(meta[k])) is not None:
            fecha, campo = f, k
            break
    if fecha is None:
        t = s.find("time", attrs={"datetime": True})
        if t is not None and (f := a_fecha(t["datetime"])) is not None:
            fecha, campo = f, "time@datetime"
    titulo = meta.get("og:title") or (s.title.get_text(" ", strip=True) if s.title else "")
    lang = (s.html.get("lang") if s.html else None) or meta.get("og:locale") or ""
    for x in s(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        x.decompose()
    parrafos = [p.get_text(" ", strip=True) for p in s.find_all("p")]
    texto = "\n".join(p for p in parrafos if len(p) > 40)
    return {"titulo": " ".join(str(titulo).split())[:300], "fecha_meta": fecha, "campo_meta": campo,
            "lang_declarado": str(lang)[:2].lower(), "texto": texto, "medio": meta.get("og:site_name")}


def fecha_timegate(fecha_meta: datetime | None, warc_date: datetime | None) -> tuple[datetime | None, str]:
    """Fecha de publicación para el TimeGate y regla aplicada (ver docstring del módulo)."""
    if fecha_meta is not None and warc_date is not None and fecha_meta <= warc_date:
        return fecha_meta, "meta"
    if fecha_meta is not None and warc_date is not None:
        return warc_date, "rastreo_meta_posterior"
    if warc_date is not None:
        return warc_date, "rastreo_sin_meta"
    return fecha_meta, "meta_sin_rastreo"


def articulos(crudo: bytes) -> list[dict[str, Any]]:
    """Trozo de WARC (gzip parcial) -> artículos con fecha TimeGate, idioma, tokens y texto."""
    salida = []
    for reg in registros_warc(miembros_gzip(crudo)):
        if reg["http_estado"] not in (None, 200):
            continue
        a = extraer_articulo(reg["html"])
        pub, regla = fecha_timegate(a["fecha_meta"], reg["warc_date"])
        a.update(url=reg["url"], dominio=dominio(reg["url"]), warc_date=reg["warc_date"], fecha_publicacion=pub,
                 regla=regla, idioma=a["lang_declarado"] or idioma(a["texto"]), n_tokens=tokens_aprox(a["texto"]),
                 lag_h=base.lag_horas(a["fecha_meta"], reg["warc_date"]))
        salida.append(a)
    return salida


def documento(a: dict[str, Any], dia: str = "") -> dict[str, Any]:
    """Artículo -> ``Documento`` serializado (con ``texto``, solo para el corpus)."""
    pub = a["fecha_publicacion"]
    doc = Documento(doc_id=doc_id(clave_url(a["url"])), fuente=FUENTE, tipo_documento="noticia",
                    titulo=a["titulo"], fecha_publicacion=pub.date(), url=a["url"], agente="macro", rag="cajon",
                    institucion=a.get("medio") or a["dominio"], idioma=a["idioma"] or "und",
                    licencia=base.LIC_COPYRIGHT,
                    extra=dict(fecha_rastreo=a["warc_date"].isoformat() if a["warc_date"] else None,
                               hora_publicacion=pub.isoformat(), fecha_meta=a["fecha_meta"].isoformat()
                               if a["fecha_meta"] else None, regla_fecha=a["regla"], warc=a.get("warc"),
                               dia_testigo=dia))
    return base.documento_dict(doc, texto=a["texto"])


# --------------------------------------------------------------------------- red con caché

def _cache(nombre: str) -> Path:
    return base.corpus("crudo", FUENTE) / nombre


def _bajar(res: base.Resultado, cfg: base.ConfigSonda, item: str, url: str, cache: Path, *, dia: str = "",
           headers: dict | None = None, max_horas: float | None = None, timeout: int = 60) -> bytes:
    """GET con caché en el corpus (``max_horas``: caducidad, para listados del mes en curso)."""
    fresco = cache.exists() and (max_horas is None or
                                 (datetime.now().timestamp() - cache.stat().st_mtime) < max_horas * 3600)
    if cfg.usar_cache and fresco:
        datos = cache.read_bytes()
        res.registrar(FUENTE, item, url, True, dia=dia, bytes=len(datos))
        return datos
    t0 = base.cronometro()
    try:
        r = get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        res.registrar(FUENTE, item, url, False, dia=dia, segundos=base.segundos(t0), error=str(e))
        raise
    datos = r.content
    total = (r.headers or {}).get("content-range") or (r.headers or {}).get("Content-Range")
    base.escribir_atomico(cache, datos)
    if total:
        cache.with_suffix(cache.suffix + ".total").write_text(str(total).split("/")[-1], encoding="utf-8")
    res.registrar(FUENTE, item, url, True, dia=dia, bytes=len(datos), segundos=base.segundos(t0))
    return datos


def warc_paths(res: base.Resultado, cfg: base.ConfigSonda, anio: int, mes: int, hoy: date) -> list[str]:
    actual = (anio, mes) == (hoy.year, hoy.month)
    datos = _bajar(res, cfg, f"paths_{anio}{mes:02d}", url_paths(anio, mes),
                   _cache(f"warc_paths_{anio}{mes:02d}.gz"), max_horas=12 if actual else None)
    return parse_paths(datos)


def leer_trozo(res: base.Resultado, cfg: base.ConfigSonda, ruta: str, n_bytes: int, dia: str
               ) -> tuple[bytes, int | None]:
    """Primeros ``n_bytes`` de un WARC (Range) y su tamaño total (de ``Content-Range``)."""
    cache = _cache(f"{Path(ruta).name}.{n_bytes}.part")
    datos = _bajar(res, cfg, f"warc_{dia}", BASE + ruta, cache, dia=dia,
                   headers={"Range": f"bytes=0-{n_bytes - 1}"}, timeout=180)
    tot = cache.with_suffix(cache.suffix + ".total")
    total = tot.read_text(encoding="utf-8").strip() if tot.exists() else ""
    return datos, int(total) if total.isdigit() else None


# --------------------------------------------------------------------------- sonda

_COLS_ART = ["dia", "dominio", "medio", "idioma", "titulo", "url", "n_tokens", "con_texto", "fecha_meta",
             "campo_meta", "warc_date", "fecha_publicacion", "regla", "lag_h", "dia_exacto", "warc", "duplicado"]


def sondear(cfg: base.ConfigSonda) -> base.Resultado:
    """Un trozo de WARC por día testigo ≥ 2016-08: artículos, reglas de fecha, idiomas y volumen."""
    res = base.Resultado(FUENTE)
    hoy = next((d.fecha for d in cfg.dias if d.clave == "hoy"), date.today())
    n_bytes = BYTES_DIA_RAPIDO if cfg.modo_rapido else BYTES_DIA
    filas: list[dict[str, Any]] = []
    docs: list[dict[str, Any]] = []
    volumen: list[dict[str, Any]] = []
    notas: list[str] = []
    for d in cfg.dias:
        dia = d.fecha.isoformat()
        if d.fecha < INICIO_CCNEWS:
            notas.append(f"{d.clave} ({dia}): sin cobertura (CC-NEWS empieza en 2016-08)")
            volumen.append(dict(dia=dia, clave=d.clave, cobertura=False))
            continue
        try:
            rutas = warc_paths(res, cfg, d.fecha.year, d.fecha.month, hoy)
            ruta, exacto = elegir_warc(rutas, d.fecha)
            if not exacto:
                notas.append(f"{d.clave} ({dia}): el mes no tiene WARC de ese día; se usa {Path(ruta).name}")
            crudo, total = leer_trozo(res, cfg, ruta, n_bytes, d.clave)
            arts = articulos(crudo)
            for a in arts:
                a.update(dia=dia, dia_exacto=exacto, warc=Path(ruta).name, con_texto=a["n_tokens"] >= MIN_TOKENS)
            if cfg.tope:
                arts = [a for a in arts if a["con_texto"]][: cfg.tope] or arts[: cfg.tope]
            filas.extend(arts)
            docs.extend(documento(a, d.clave) for a in arts if a["con_texto"] and a["fecha_publicacion"])
            por_warc = round(len(arts) * total / max(len(crudo), 1)) if total and not cfg.tope else None
            volumen.append(dict(dia=dia, clave=d.clave, cobertura=True, dia_exacto=exacto, warc=Path(ruta).name,
                                warcs_mes=len(rutas), bytes_warc=total, bytes_leidos=len(crudo), respuestas=len(arts),
                                con_texto=sum(a["con_texto"] for a in arts),
                                respuestas_por_warc_est=por_warc,
                                respuestas_mes_est=por_warc * len(rutas) if por_warc else None,
                                gb_mes_est=round(total * len(rutas) / 1e9, 1) if total else None,
                                mes_completo=(d.fecha.year, d.fecha.month) != (hoy.year, hoy.month)))
            cfg.log(f"[ccnews] {d.clave}: {len(arts)} respuestas, {volumen[-1]['con_texto']} con texto")
        except Exception as e:  # noqa: BLE001
            notas.append(f"{d.clave} ({dia}): fallo {type(e).__name__}: {str(e)[:120]}")
            volumen.append(dict(dia=dia, clave=d.clave, cobertura=None))

    # Extremos medidos de la cobertura: primer WARC del archivo y último del mes en curso.
    inicio = fin = None
    try:
        inicio = fecha_ruta(warc_paths(res, cfg, INICIO_CCNEWS.year, INICIO_CCNEWS.month, hoy)[0])
    except Exception as e:  # noqa: BLE001
        notas.append(f"inicio no medido: {str(e)[:100]}")
    for k in (0, 1):
        mes = (hoy.replace(day=1) - timedelta(days=1)) if k else hoy
        try:
            fin = fecha_ruta(warc_paths(res, cfg, mes.year, mes.month, hoy)[-1])
            break
        except Exception as e:  # noqa: BLE001
            notas.append(f"listado de {mes:%Y-%m} no disponible: {str(e)[:80]}")

    # Duplicados y documentos al corpus.
    dedup = deduplicar(docs)
    vivos = {x["doc_id"] for x in dedup}
    if docs:
        ruta, n = base.guardar_documentos(dedup, "ccnews_muestra", rapido=cfg.modo_rapido)
        res.documentos[ruta.stem] = n
        res.tablas["ccnews_embudo"] = embudo_duplicados(docs)

    tabla = pd.DataFrame(filas)
    if not tabla.empty:
        tabla["duplicado"] = [a["con_texto"] and doc_id(clave_url(a["url"])) not in vivos for a in filas]
        for c in ("fecha_meta", "warc_date", "fecha_publicacion"):
            tabla[c] = tabla[c].map(lambda x: x.isoformat() if x is not None else None)
    tabla = tabla.reindex(columns=_COLS_ART)
    res.tablas["ccnews_articulos"] = tabla
    res.tablas["ccnews_volumen"] = pd.DataFrame(volumen)

    for a in [a for a in filas if a["con_texto"]][:40]:
        res.timegate.append(dict(fuente=FUENTE, item=a["dominio"],
                                 fecha_hecho=a["fecha_meta"].isoformat() if a["fecha_meta"] else None,
                                 fecha_publicacion=a["fecha_publicacion"].isoformat() if a["fecha_publicacion"] else None,
                                 campo_fecha=(f"{a['campo_meta']} (≤ WARC-Date)" if a["regla"] == "meta"
                                              else f"WARC-Date ({a['regla']})"),
                                 lag_h=a["lag_h"]))

    _cerrar(res, cfg, filas, volumen, inicio, fin, notas)
    return res


def _cerrar(res: base.Resultado, cfg: base.ConfigSonda, filas: list[dict], volumen: list[dict],
            inicio: datetime | None, fin: datetime | None, notas: list[str]) -> None:
    con_texto = [a for a in filas if a["con_texto"]]
    lags = base.resumen_lag(a["lag_h"] for a in con_texto if a["regla"] == "meta")
    reglas = Counter(a["regla"] for a in filas)
    idiomas = Counter(a["idioma"] or "und" for a in con_texto).most_common(8)
    con_hora = sum(1 for a in con_texto if a["fecha_meta"] is not None)
    vol = [v for v in volumen if v.get("respuestas_mes_est") and v.get("mes_completo")]
    gb_mes = max((v["gb_mes_est"] for v in vol if v.get("gb_mes_est")), default=None)
    art_mes = max((v["respuestas_mes_est"] for v in vol), default=None)
    est = base.estado(res.registro, FUENTE)
    if est == "ok" and any(v.get("cobertura") is None for v in volumen):
        est = "parcial"
    med = dict(dias=len([v for v in volumen if v.get("cobertura")]), respuestas=len(filas),
               con_texto=len(con_texto), tokens_mediana=int(pd.Series([a["n_tokens"] for a in con_texto]).median())
               if con_texto else None, reglas_fecha=dict(reglas), retraso_rastreo_h=lags,
               con_fecha_meta=con_hora, idiomas=idiomas, documentos=sum(res.documentos.values()),
               gb_mes_max_est=gb_mes, respuestas_mes_max_est=art_mes,
               bytes_descargados=sum(r["bytes"] for r in res.registro), modo_rapido=cfg.modo_rapido)
    ini_txt = inicio.date().isoformat() if inicio else None
    fin_txt = fin.date().isoformat() if fin else None
    res.cobertura.append(dict(
        fuente=FUENTE, familia="texto_historico", tipo="texto", inicio=ini_txt, fin=fin_txt, hora=con_hora > 0,
        idiomas=",".join(k for k, _ in idiomas), lag_mediana_h=lags["mediana_h"], destino="cajon", backtest=True,
        estado=est, nota=(f"{len(con_texto)} artículos con texto en {med['dias']} días; lag = retraso de rastreo "
                          f"(WARC-Date − metadato); ≈{gb_mes} GB/mes" + ("; " + "; ".join(notas[:3]) if notas else ""))))
    res.fichas.append(base.ficha(
        fuente=FUENTE, nombre="Common Crawl CC-NEWS (WARC, texto completo, mundial)", agentes=["macro", "equity", "risk"],
        rag="cajon", url="https://commoncrawl.org/blog/news-dataset-available", acceso="descarga",
        licencia_y_terminos=LICENCIA, cobertura_inicio=ini_txt, cobertura_fin=fin_txt,
        frecuencia="continua (un WARC cada pocas horas)",
        retraso_publicacion=(f"rastreo: mediana {lags['mediana_h']} h tras el metadato (p90 {lags['p90_h']} h)"
                             if lags["n"] else "no medido"),
        formato="warc.gz (HTTP crudo con HTML)",
        volumen_estimado=(f"≈{gb_mes} GB/mes y ≈{art_mes / 1e6:.1f} M respuestas/mes (máximo de los meses medidos)"
                          if gb_mes and art_mes else "no extrapolado (modo rápido o sin Content-Range)"),
        limites="sin clave; HTTP con Range sobre data.commoncrawl.org; WARC de ~1 GB: filtrar por dominio antes de parsear",
        estado_sonda=est, mediciones=med,
        notas=notas + ["Para el histórico 2016-2024/06 conviene la versión ya extraída stanford-oval/ccnews "
                       "(ver hf_datasets); WARC solo para meses posteriores.",
                       "Texto con copyright: solo en el corpus local; al repo, metadatos y titulares."]))
