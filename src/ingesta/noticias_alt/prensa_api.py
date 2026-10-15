"""APIs de prensa con clave gratuita: NYT Archive API y The Guardian Open Platform.

Decisión del usuario: NO hay claves. Las funciones quedan listas para cuando las haya
(``NYT_API_KEY`` y ``GUARDIAN_API_KEY`` en ``.env``); sin clave no se toca la red y las fichas
salen ``no_probada`` con lo documentado.

- NYT Archive API: todos los artículos de un mes desde **1851** (titular, abstract, lead paragraph,
  ``pub_date`` con hora, sección, palabras clave); SIN cuerpo. 10 peticiones/min y 4.000/día; una
  petición = un mes entero. TimeGate: ``pub_date``.
- Guardian Open Platform (clave *developer*): piezas desde **1999** con cuerpo (``show-fields=bodyText``);
  1 petición/s y 500/día; uso no comercial. TimeGate: ``webPublicationDate`` (o
  ``firstPublicationDate`` si es anterior: la primera vez que fue público).
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

import pandas as pd

from ingesta.comun.esquema import Documento, doc_id
from ingesta.comun.http import get
from ingesta.comun.rutas import env
from ingesta.noticias_alt import base
from ingesta.noticias_alt.limpieza import a_fecha, clave_url, limpiar_html

NYT_ARCHIVE = "https://api.nytimes.com/svc/archive/v1/{anio}/{mes}.json"
GUARDIAN = "https://content.guardianapis.com/search"

DOC = {
    "nyt": dict(nombre="NYT Archive API (titular, abstract y lead paragraph desde 1851)", inicio="1851-09-18",
                url="https://developer.nytimes.com/docs/archive-product/1/overview", tipo="titular",
                licencia="NYT APIs Terms of Use: no comercial, atribución, no almacenar más de lo necesario",
                limites="10 peticiones/min y 4.000/día; Archive = 1 petición por mes (barato)",
                formato="json: headline, abstract, lead_paragraph, pub_date, section_name, keywords (sin cuerpo)",
                volumen="≈5-8 k piezas/mes; ≈2-3 MB de json/mes; 1851-hoy ≈ 4-5 GB (documentado, no medido)",
                retraso="mismo día (pub_date con hora); el archivo del mes en curso se completa al cierre",
                variable="NYT_API_KEY",
                alta="cuenta en developer.nytimes.com -> crear app -> activar Archive API; NYT_API_KEY en .env"),
    "guardian": dict(nombre="The Guardian Open Platform (cuerpo completo desde 1999)", inicio="1999-01-01",
                     url="https://open-platform.theguardian.com/", tipo="texto",
                     licencia="Guardian Open Platform T&C (developer): no comercial; atribución; borrar lo retirado",
                     limites="developer key: 1 petición/s y 500/día (≈25 k piezas/día con page-size 50)",
                     formato="json: webTitle, webPublicationDate, sectionName, fields.bodyText",
                     volumen="≈2,5 M piezas; business+money ≈ 300 k; ≈2-4 GB de texto (documentado, no medido)",
                     retraso="minutos (webPublicationDate con hora)", variable="GUARDIAN_API_KEY",
                     alta="registro en open-platform.theguardian.com/access (developer key); GUARDIAN_API_KEY en .env"),
}


# --------------------------------------------------------------------------- clientes (listos para la clave)

def nyt_archivo(anio: int, mes: int, clave: str) -> list[dict[str, Any]]:
    """Todos los artículos de un mes del archivo del NYT (una petición, ~MB)."""
    r = get(NYT_ARCHIVE.format(anio=anio, mes=mes), params={"api-key": clave}, por_segundo=0.15, timeout=180)
    r.raise_for_status()
    return r.json()["response"]["docs"]


def guardian_buscar(desde: str, hasta: str, clave: str, q: str | None = None, seccion: str | None = "business",
                    pagina: int = 1, por_pagina: int = 50) -> dict[str, Any]:
    """Búsqueda del Guardian con cuerpo; devuelve el bloque ``response`` (total, pages, results)."""
    params: dict[str, Any] = {"from-date": desde, "to-date": hasta, "api-key": clave, "page": pagina,
                              "show-fields": "bodyText,headline,firstPublicationDate", "page-size": por_pagina,
                              "order-by": "oldest", "use-date": "published"}
    if q:
        params["q"] = q
    if seccion:
        params["section"] = seccion
    r = get(GUARDIAN, params=params, por_segundo=1.0, timeout=60)
    r.raise_for_status()
    return r.json()["response"]


# --------------------------------------------------------------------------- parseo -> Documento

def nyt_documento(d: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Pieza del Archive -> (``Documento`` con texto = abstract + lead, fila de metadatos). TimeGate: ``pub_date``."""
    pub = a_fecha(d.get("pub_date"))
    url = d.get("web_url") or ""
    if pub is None or not url:
        return None
    titulo = limpiar_html((d.get("headline") or {}).get("main") or "")
    texto = "\n".join(t for t in (limpiar_html(d.get("abstract")), limpiar_html(d.get("lead_paragraph"))) if t)
    texto = "\n".join(dict.fromkeys(texto.split("\n")))          # abstract y lead a veces coinciden
    doc = Documento(doc_id=doc_id(clave_url(url)), fuente="nyt", tipo_documento=d.get("document_type") or "article",
                    titulo=titulo, fecha_publicacion=pub.date(), url=url, agente="macro", rag="cajon",
                    institucion="The New York Times", idioma="en", seccion=d.get("section_name"),
                    entidades=[k.get("value") for k in d.get("keywords") or [] if k.get("value")][:15],
                    licencia=base.LIC_COPYRIGHT, extra=dict(hora_publicacion=pub.isoformat(), campo_fecha="pub_date",
                                                           palabras=d.get("word_count")))
    fila = dict(fuente="nyt", fecha_publicacion=pub.isoformat(), seccion=d.get("section_name"), titulo=titulo[:200],
                url=url, tipo=d.get("document_type"), palabras=d.get("word_count"), campo_fecha="pub_date")
    return base.documento_dict(doc, texto=texto), fila


def guardian_documento(r: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Resultado del Guardian -> (``Documento`` con cuerpo, fila). Fecha = la más antigua de web/first."""
    campos = r.get("fields") or {}
    web, first = a_fecha(r.get("webPublicationDate")), a_fecha(campos.get("firstPublicationDate"))
    candidatos = [(f, c) for f, c in ((web, "webPublicationDate"), (first, "firstPublicationDate")) if f]
    url = r.get("webUrl") or ""
    if not candidatos or not url:
        return None
    pub, campo = min(candidatos, key=lambda fc: fc[0])
    titulo = limpiar_html(campos.get("headline") or r.get("webTitle") or "")
    texto = limpiar_html(campos.get("bodyText") or "")
    doc = Documento(doc_id=doc_id(clave_url(url)), fuente="guardian", tipo_documento=r.get("type") or "article",
                    titulo=titulo, fecha_publicacion=pub.date(), url=url, agente="macro", rag="cajon",
                    institucion="The Guardian", idioma="en", seccion=r.get("sectionName"),
                    licencia=base.LIC_COPYRIGHT, extra=dict(hora_publicacion=pub.isoformat(), campo_fecha=campo))
    fila = dict(fuente="guardian", fecha_publicacion=pub.isoformat(), seccion=r.get("sectionName"),
                titulo=titulo[:200], url=url, tipo=r.get("type"), palabras=len(texto.split()), campo_fecha=campo)
    return base.documento_dict(doc, texto=texto), fila


# --------------------------------------------------------------------------- sonda

def _crudo(nombre: str) -> Any:
    return base.corpus("crudo", "prensa_api") / nombre


def _medir_nyt(res: base.Resultado, cfg: base.ConfigSonda, clave: str) -> tuple[list, list]:
    docs, filas, meses = [], [], {}
    for d in cfg.dias:
        mes = (d.fecha.year, d.fecha.month)
        try:
            if mes not in meses:
                cache = _crudo(f"nyt_{mes[0]}{mes[1]:02d}.json")
                url = NYT_ARCHIVE.format(anio=mes[0], mes=mes[1])
                if cfg.usar_cache and cache.exists():
                    meses[mes] = json.loads(cache.read_bytes())
                    res.registrar("nyt", f"archivo_{mes[0]}{mes[1]:02d}", url, True, dia=d.clave)
                else:
                    t0 = base.cronometro()
                    try:
                        meses[mes] = nyt_archivo(*mes, clave)
                    except Exception as e:  # noqa: BLE001
                        res.registrar("nyt", f"archivo_{mes[0]}{mes[1]:02d}", url, False, dia=d.clave,
                                      segundos=base.segundos(t0), error=str(e))
                        raise
                    datos = json.dumps(meses[mes]).encode()
                    base.escribir_atomico(cache, datos)
                    res.registrar("nyt", f"archivo_{mes[0]}{mes[1]:02d}", url, True, dia=d.clave, bytes=len(datos),
                                  segundos=base.segundos(t0))
            del_dia = [p for p in (nyt_documento(x) for x in meses[mes]) if p and p[1]["fecha_publicacion"][:10]
                       == d.fecha.isoformat()]
            for doc, fila in del_dia[: cfg.tope or None]:
                fila.update(dia=d.fecha.isoformat(), n_mes=len(meses[mes]))
                docs.append(doc)
                filas.append(fila)
        except Exception:  # noqa: BLE001
            continue
    return docs, filas


def _medir_guardian(res: base.Resultado, cfg: base.ConfigSonda, clave: str) -> tuple[list, list]:
    docs, filas = [], []
    for d in cfg.dias:
        dia = d.fecha.isoformat()
        cache = _crudo(f"guardian_{dia}.json")
        try:
            if cfg.usar_cache and cache.exists():
                resp = json.loads(cache.read_bytes())
                res.registrar("guardian", f"busqueda_{d.clave}", GUARDIAN, True, dia=d.clave)
            else:
                t0 = base.cronometro()
                try:
                    resp = guardian_buscar(dia, dia, clave, por_pagina=min(cfg.tope or 50, 50))
                except Exception as e:  # noqa: BLE001
                    res.registrar("guardian", f"busqueda_{d.clave}", GUARDIAN, False, dia=d.clave,
                                  segundos=base.segundos(t0), error=str(e))
                    continue
                datos = json.dumps(resp).encode()
                base.escribir_atomico(cache, datos)
                res.registrar("guardian", f"busqueda_{d.clave}", GUARDIAN, True, dia=d.clave, bytes=len(datos),
                              segundos=base.segundos(t0))
            for p in (guardian_documento(x) for x in resp.get("results") or []):
                if p:
                    p[1].update(dia=dia, total_dia=resp.get("total"))
                    docs.append(p[0])
                    filas.append(p[1])
        except Exception:  # noqa: BLE001
            continue
    return docs, filas


def sondear(cfg: base.ConfigSonda) -> base.Resultado:
    """Con clave: una petición por mes (NYT) o día (Guardian) testigo. Sin clave: ficha ``no_probada``."""
    res = base.Resultado("prensa_api")
    medidores = {"nyt": _medir_nyt, "guardian": _medir_guardian}
    for fuente, info in DOC.items():
        clave = env(info["variable"])
        docs: list = []
        filas: list = []
        if clave:
            docs, filas = medidores[fuente](res, cfg, clave)
            if docs:
                ruta, n = base.guardar_documentos(docs, f"{fuente}_muestra", rapido=cfg.modo_rapido)
                res.documentos[ruta.stem] = n
            if filas:
                res.tablas[f"{fuente}_articulos"] = pd.DataFrame(filas)
            for f in filas[:15]:
                res.timegate.append(dict(fuente=fuente, item=f["seccion"] or "", fecha_hecho=None,
                                         fecha_publicacion=f["fecha_publicacion"], campo_fecha=f["campo_fecha"],
                                         lag_h=None))
        _cerrar(res, fuente, info, bool(clave), filas)
    return res


def _cerrar(res: base.Resultado, fuente: str, info: dict[str, str], con_clave: bool, filas: list[dict]) -> None:
    est = base.estado(res.registro, fuente)
    if con_clave and filas:
        fechas = sorted(f["fecha_publicacion"] for f in filas)
        inicio, fin, origen = fechas[0][:10], fechas[-1][:10], "medidas en la muestra"
    else:
        inicio, fin, origen = info["inicio"], date.today().isoformat(), "documentadas (no medidas)"
    motivo = (f"sin clave ({info['variable']} vacía en .env): decisión del usuario, no se registra cuenta"
              if not con_clave else f"{len(filas)} piezas medidas en los días testigo")
    res.cobertura.append(dict(fuente=fuente, familia="texto_historico", tipo=info["tipo"], inicio=inicio, fin=fin,
                              hora=True, idiomas="en", lag_mediana_h=None, destino="cajon", backtest=True,
                              estado=est, nota=f"{motivo}; fechas {origen}"))
    res.fichas.append(base.ficha(
        fuente=fuente, nombre=info["nombre"], agentes=["macro", "equity", "risk"], rag="cajon", url=info["url"],
        acceso="api", autenticacion="clave gratuita", licencia_y_terminos=info["licencia"],
        cobertura_inicio=inicio, cobertura_fin=fin, frecuencia="continua", retraso_publicacion=info["retraso"],
        formato=info["formato"], volumen_estimado=info["volumen"], limites=info["limites"], estado_sonda=est,
        mediciones=dict(con_clave=con_clave, piezas=len(filas), fechas=origen,
                        peticiones=sum(1 for r in res.registro if r["fuente"] == fuente)),
        notas=[motivo, f"Cómo obtener la clave: {info['alta']}."]))
