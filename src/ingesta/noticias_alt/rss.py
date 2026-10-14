"""RSS en vivo de medios de todo el mundo, Bing News RSS (búsqueda) y Google News RSS.

Solo **en vivo** (``backtest=False``): un RSS no tiene histórico (los ítems caducan en horas o
días) y su fecha es la que declara el medio. Se guarda titular + extracto corto del feed + URL +
fecha; el cuerpo del artículo NO se descarga (copyright y términos de cada medio).

- ``FEEDS``: lista curada (id, país, idioma, medio, url), 36 feeds de 15 países, incluidos los que
  robots.txt prohíbe (para medirlo: se comprueba robots.txt y NO se descargan).
- Bing News RSS (``/news/search?format=rss``) para las ``CONSULTAS`` de ``muestra.py``; los enlaces
  vienen envueltos en ``apiclick.aspx?url=...`` y se desenvuelven.
- Google News RSS: robots.txt no permite ``/rss/search`` a un robot genérico -> fallo, sin descargar.

TimeGate: ``fecha_publicacion`` = ``pubDate`` del ítem (día UTC); si falta o es posterior a la
captura (reloj/zona mal declarados), la fecha de captura. Cada documento lleva ``fecha_captura``:
para un backtest sobre el archivo propio de capturas, la regla conservadora es la captura.

Caché: el XML crudo de cada captura va a ``<corpus>/crudo/rss/<AAAA-MM-DD>/``; con ``usar_cache``
se reutiliza la captura del mismo día. Repetida a diario, es el archivo propio de titulares.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import pandas as pd

from ingesta.comun.esquema import Documento, doc_id
from ingesta.comun.http import get, robots_permite
from ingesta.noticias_alt import base
from ingesta.noticias_alt.limpieza import clave_url, deduplicar, idioma, limpiar_html, parse_rss
from ingesta.noticias_alt.muestra import CONSULTAS

MODULO = "rss"
F_RSS = "rss_medios"
F_BING = "bing_news_rss"
F_GOOGLE = "google_news_rss"
BING_NEWS = "https://www.bing.com/news/search"
GOOGLE_NEWS = "https://news.google.com/rss/search"
MAX_EXTRACTO = 300            # caracteres del extracto que van al corpus (nunca a results)
HILOS = 8                     # hosts en paralelo (los feeds de un mismo host van en serie)

# (id, país ISO, idioma, medio, url). Primero los que se sabe abiertos (modo rápido = los primeros).
FEEDS: list[tuple[str, str, str, str, str]] = [
    ("bbc_business", "GB", "en", "BBC", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("guardian_business", "GB", "en", "The Guardian", "https://www.theguardian.com/business/rss"),
    ("nyt_economy", "US", "en", "New York Times", "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml"),
    ("nyt_business", "US", "en", "New York Times", "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml"),
    ("bloomberg_markets", "US", "en", "Bloomberg", "https://feeds.bloomberg.com/markets/news.rss"),
    ("bloomberg_economics", "US", "en", "Bloomberg", "https://feeds.bloomberg.com/economics/news.rss"),
    ("lemonde_eco", "FR", "fr", "Le Monde", "https://www.lemonde.fr/economie/rss_full.xml"),
    ("lesechos_marches", "FR", "fr", "Les Echos", "https://services.lesechos.fr/rss/les-echos-finance-marches.xml"),
    ("faz_finanzen", "DE", "de", "FAZ", "https://www.faz.net/rss/aktuell/finanzen/"),
    ("handelsblatt_finanzen", "DE", "de", "Handelsblatt", "https://www.handelsblatt.com/contentexport/feed/finanzen"),
    ("ilsole_finanza", "IT", "it", "Il Sole 24 Ore", "https://www.ilsole24ore.com/rss/finanza.xml"),
    ("expansion_mercados", "ES", "es", "Expansión", "https://e00-expansion.uecdn.es/rss/mercados.xml"),
    ("europapress_eco", "ES", "es", "Europa Press", "https://www.europapress.es/rss/rss.aspx?ch=00136"),
    ("elpais_economia", "ES", "es", "El País",
     "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/economia/portada"),
    ("eleconomista_mx", "MX", "es", "El Economista (MX)", "https://www.eleconomista.com.mx/rss/mercados"),
    ("infomoney_br", "BR", "pt", "InfoMoney", "https://www.infomoney.com.br/feed/"),
    ("nikkei_asia", "JP", "en", "Nikkei Asia", "https://asia.nikkei.com/rss/feed/nar"),
    ("japantimes", "JP", "en", "Japan Times", "https://www.japantimes.co.jp/feed/"),
    ("scmp_business", "HK", "en", "South China Morning Post", "https://www.scmp.com/rss/92/feed"),
    ("et_markets", "IN", "en", "Economic Times",
     "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("livemint_markets", "IN", "en", "Mint", "https://www.livemint.com/rss/markets"),
    ("aljazeera", "QA", "en", "Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    ("arabnews_business", "SA", "en", "Arab News", "https://www.arabnews.com/cat/3/rss.xml"),
    ("moscowtimes", "RU", "en", "Moscow Times", "https://www.themoscowtimes.com/rss/news"),
    ("allafrica_business", "ZA", "en", "AllAfrica", "https://allafrica.com/tools/headlines/rdf/business/headlines.rdf"),
    ("reuters_business", "GB", "en", "Reuters", "http://feeds.reuters.com/reuters/businessNews"),
    ("ft_home", "GB", "en", "Financial Times", "https://www.ft.com/rss/home"),
    ("ft_markets", "GB", "en", "Financial Times", "https://www.ft.com/markets?format=rss"),
    ("economist_finance", "GB", "en", "The Economist", "https://www.economist.com/finance-and-economics/rss.xml"),
    ("cnbc_top", "US", "en", "CNBC",
     "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    ("cnbc_finance", "US", "en", "CNBC",
     "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"),
    ("wsj_markets", "US", "en", "Wall Street Journal", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
    ("marketwatch_top", "US", "en", "MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("yahoo_finance", "US", "en", "Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    ("seekingalpha_currents", "US", "en", "Seeking Alpha", "https://seekingalpha.com/market_currents.xml"),
    ("investing_news", "US", "en", "Investing.com", "https://www.investing.com/rss/news_25.rss"),
]


# --------------------------------------------------------------------------- URLs

def url_bing_news(q: str, mercado: str = "en-US") -> str:
    return f"{BING_NEWS}?{urlencode({'q': q, 'format': 'rss', 'mkt': mercado})}"


def url_google_news(q: str, desde: str | None = None, hasta: str | None = None, hl: str = "en-US",
                    gl: str = "US") -> str:
    """Búsqueda de Google News RSS; ``after:``/``before:`` (AAAA-MM-DD) acotan por fecha (no documentado)."""
    q = q + (f" after:{desde}" if desde else "") + (f" before:{hasta}" if hasta else "")
    return f"{GOOGLE_NEWS}?{urlencode({'q': q, 'hl': hl, 'gl': gl, 'ceid': gl + ':' + hl.split('-')[0]})}"


def desenvolver_bing(url: str) -> str:
    """``bing.com/news/apiclick.aspx?...&url=<destino>`` -> ``<destino>`` (el resto, igual)."""
    p = urlsplit(url or "")
    if p.netloc.endswith("bing.com") and p.path.lower().endswith("apiclick.aspx"):
        destino = parse_qs(p.query).get("url")
        if destino and destino[0]:
            return destino[0]
    return url


def _slug(txt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", txt.lower()).strip("_")


# --------------------------------------------------------------------------- parseo y TimeGate

def _fuentes_item(contenido: bytes) -> list[str | None]:
    """``<News:Source>`` de cada ítem de Bing (con espacio de nombres: ``parse_rss`` no lo ve)."""
    import lxml.etree as ET

    raiz = ET.fromstring(contenido.lstrip(), parser=ET.XMLParser(recover=True, resolve_entities=False,
                                                                 no_network=True))
    salida: list[str | None] = []
    for it in raiz.iter():
        if isinstance(it.tag, str) and it.tag.split("}")[-1] in ("item", "entry"):
            src = next((("".join(h.itertext()).strip() or None) for h in it
                        if isinstance(h.tag, str) and h.tag.split("}")[-1].lower() == "source"), None)
            salida.append(src)
    return salida


def parsear(contenido: bytes, *, bing: bool = False) -> list[dict[str, Any]]:
    """Feed -> ítems de ``limpieza.parse_rss``; con ``bing``, URL desenvuelta y medio de ``News:Source``.

    Lanza ``ValueError`` si no es un feed (página HTML de error, consentimiento, CAPTCHA...).
    """
    items = parse_rss(contenido)
    if bing:
        fuentes = _fuentes_item(contenido)
        for it, src in zip(items, fuentes):
            it["url"] = desenvolver_bing(it["url"])
            it["medio"] = it.get("medio") or src
    return items


def fecha_publicacion(pub: datetime | None, captura: datetime) -> tuple[datetime, str]:
    """Regla TimeGate del RSS: ``pubDate`` si existe y no es posterior a la captura; si no, la captura.

    Devuelve (fecha, campo de origen). Un ``pubDate`` futuro es imposible (reloj o zona mal
    declarados): lo único seguro es que a la hora de la captura ya era público.
    """
    if pub is None:
        return captura, "captura (sin pubDate)"
    if pub > captura:
        return captura, "captura (pubDate futuro)"
    return pub, "pubDate"


def medir_items(items: list[dict[str, Any]], captura: datetime) -> dict[str, Any]:
    """n_items, profundidad (h entre el más antiguo y el más reciente), % con fecha, edad del más reciente."""
    fechas = sorted(i["fecha"] for i in items if i.get("fecha") and i["fecha"] <= captura)
    con_hora = sum(1 for i in items if i.get("fecha") and (i["fecha"].hour, i["fecha"].minute) != (0, 0))
    texto = " ".join(f"{i.get('titulo', '')}. {i.get('resumen', '')}" for i in items[:30])
    return dict(n_items=len(items),
                profundidad_h=round((fechas[-1] - fechas[0]).total_seconds() / 3600, 1) if len(fechas) > 1 else None,
                pct_con_fecha=round(100 * sum(1 for i in items if i.get("fecha")) / len(items), 1) if items else None,
                pct_con_hora=round(100 * con_hora / len(items), 1) if items else None,
                pct_con_resumen=round(100 * sum(1 for i in items if len(i.get("resumen") or "") > 40) / len(items), 1)
                if items else None,
                edad_min_h=round((captura - fechas[-1]).total_seconds() / 3600, 2) if fechas else None,
                mas_antiguo=fechas[0].isoformat() if fechas else None,
                mas_reciente=fechas[-1].isoformat() if fechas else None,
                idioma_detectado=idioma(texto))


def documentos(items: list[dict[str, Any]], *, fuente: str, medio: str, pais: str | None, lengua: str | None,
               captura: datetime, feed: str) -> list[dict[str, Any]]:
    """Ítems -> ``Documento`` tipo titular (titular + extracto corto) para el cajón, con su texto."""
    salida = []
    for it in items:
        if not it.get("titulo"):
            continue
        fecha, campo = fecha_publicacion(it.get("fecha"), captura)
        url = it.get("url") or ""
        extracto = limpiar_html(it.get("resumen"), MAX_EXTRACTO)
        doc = Documento(doc_id=doc_id(clave_url(url) if url else it["titulo"]), fuente=fuente,
                        tipo_documento="titular", titulo=it["titulo"], fecha_publicacion=fecha.date(), url=url,
                        agente="comun", rag="cajon", institucion=it.get("medio") or medio,
                        idioma=lengua or idioma(f"{it['titulo']}. {extracto}", "und"),
                        paises=[pais] if pais else [], licencia=base.LIC_COPYRIGHT,
                        extra=dict(fecha_captura=captura.isoformat(), hora_publicacion=fecha.isoformat(),
                                   campo_fecha=campo, feed=feed))
        salida.append(base.documento_dict(doc, texto=f"{it['titulo']}. {extracto}".strip()))
    return salida


# --------------------------------------------------------------------------- descarga (con caché del día)

def leer_feed(url: str, *, cache: Path | None = None, usar_cache: bool = True, timeout: int = 20
              ) -> tuple[bytes, datetime, bool]:
    """Bytes del feed, momento de captura (UTC) y si vino de la caché. Lanza ``PermissionError`` si
    robots.txt no lo permite (no se descarga) y lo que lance ``get`` si falla."""
    if not robots_permite(url):
        raise PermissionError("robots.txt")
    if cache is not None and usar_cache and cache.exists():
        return cache.read_bytes(), datetime.fromtimestamp(cache.stat().st_mtime, tz=UTC), True
    r = get(url, timeout=timeout, reintentos=1)
    r.raise_for_status()              # ``get`` ya lanza; por si se sustituye (tests)
    captura = datetime.now(UTC)
    if cache is not None:
        base.escribir_atomico(cache, r.content)
    return r.content, captura, False


def buscar_bing(q: str, *, mercado: str = "en-US", usar_cache: bool = False) -> list[dict[str, Any]]:
    """Titulares de Bing News para ``q`` (en vivo; para la tool ``buscar_noticias``)."""
    cache = base.corpus("crudo", "rss", datetime.now(UTC).date().isoformat()) / f"bing_{_slug(q)}.xml"
    contenido, _, _ = leer_feed(url_bing_news(q, mercado), cache=cache, usar_cache=usar_cache)
    return parsear(contenido, bing=True)


def _bajar_grupo(grupo: list[dict[str, Any]], carpeta: Path, usar_cache: bool) -> list[dict[str, Any]]:
    """Feeds de un mismo host, en serie. Devuelve una medición por feed (sin tocar ``Resultado``)."""
    salida = []
    for f in grupo:
        t0 = base.cronometro()
        m = dict(f, robots_ok=True, ok=False, error="", bytes=0, items=[], captura=None, cache=False)
        try:
            contenido, captura, de_cache = leer_feed(f["url"], cache=carpeta / f"{f['feed']}.xml",
                                                     usar_cache=usar_cache)
            m.update(bytes=len(contenido), captura=captura, cache=de_cache)
            m["items"] = parsear(contenido, bing=f["fuente"] == F_BING)
            m["ok"] = True
        except PermissionError:
            m.update(robots_ok=False, error="robots.txt")
        except Exception as e:  # noqa: BLE001
            m["error"] = f"{type(e).__name__}: {e}"
        m["segundos"] = base.segundos(t0)
        salida.append(m)
    return salida


def _por_host(feeds: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grupos: dict[str, list] = defaultdict(list)
    for f in feeds:
        grupos[urlsplit(f["url"]).netloc].append(f)
    return list(grupos.values())


# --------------------------------------------------------------------------- sonda

def _feeds_sonda(cfg: base.ConfigSonda) -> list[dict[str, Any]]:
    tope = cfg.tope
    feeds = [dict(fuente=F_RSS, feed=fid, pais=pais, idioma=lang, medio=medio, url=url, consulta="")
             for fid, pais, lang, medio, url in (FEEDS[:tope] if tope else FEEDS)]
    consultas = list(cfg.extra.get("consultas", CONSULTAS))
    for q in consultas[:tope] if tope else consultas:
        feeds.append(dict(fuente=F_BING, feed=f"bing_{_slug(q)}", pais="US", idioma="en", medio="Bing News",
                          url=url_bing_news(q), consulta=q))
    return feeds


def _primera_fecha(items_df: pd.DataFrame, fuente: str) -> str | None:
    if items_df.empty:
        return None
    f = pd.to_datetime(items_df.loc[items_df["fuente"] == fuente, "fecha_publicacion"], utc=True, errors="coerce")
    return None if f.dropna().empty else f.min().date().isoformat()


def _cobertura(res: base.Resultado, fuente: str, medidas: list[dict], items_df: pd.DataFrame, cfg: base.ConfigSonda,
               nota: str, destino: str = "vivo") -> dict[str, Any]:
    vivos = [m for m in medidas if m["ok"] and m["n_items"]]
    sub = items_df[items_df["fuente"] == fuente] if not items_df.empty else items_df
    fechas = pd.to_datetime(sub["fecha_publicacion"], utc=True, errors="coerce").dropna() if len(sub) else pd.Series()
    edades = [m["edad_min_h"] for m in vivos if m.get("edad_min_h") is not None]
    sin_cob = [d.clave for d in cfg.dias if d.clave != "hoy"]
    fila = dict(fuente=fuente, familia="vivo_redes", tipo="titular",
                inicio=fechas.min().date().isoformat() if len(fechas) else None,     # AAAA-MM-DD, como las demás
                fin=fechas.max().date().isoformat() if len(fechas) else None,
                hora=bool(vivos) and float(pd.Series([m["pct_con_hora"] or 0 for m in vivos]).median()) >= 50,
                idiomas=",".join(sorted({m["idioma"] for m in vivos})),
                lag_mediana_h=base.resumen_lag(edades)["mediana_h"], destino=destino, backtest=False,
                estado=base.estado(res.registro, fuente),
                nota=(f"{nota} Días testigo sin cobertura (solo en vivo): {', '.join(sin_cob)}." if sin_cob else nota))
    res.cobertura.append(fila)
    return fila


def _sondear_feeds(cfg: base.ConfigSonda, res: base.Resultado) -> None:
    hoy = datetime.now(UTC).date().isoformat()
    carpeta = base.corpus("crudo", MODULO, hoy)
    feeds = _feeds_sonda(cfg)
    with ThreadPoolExecutor(max_workers=HILOS) as ex:
        medidas = [m for grupo in ex.map(lambda g: _bajar_grupo(g, carpeta, cfg.usar_cache), _por_host(feeds))
                   for m in grupo]
    orden = {f["feed"]: i for i, f in enumerate(feeds)}
    medidas.sort(key=lambda m: orden[m["feed"]])

    filas_feed, filas_item, docs = [], [], {F_RSS: [], F_BING: []}
    for m in medidas:
        item = m["feed"] + (" (caché)" if m["cache"] else "")
        res.registrar(m["fuente"], item, m["url"], m["ok"], dia=hoy, bytes=m["bytes"], segundos=m["segundos"],
                      error=m["error"])
        captura = m["captura"] or datetime.now(UTC)
        med = medir_items(m["items"], captura) if m["ok"] else dict(n_items=None)
        m.update(med)
        estado = "ok" if m["ok"] and m["n_items"] else ("vacio" if m["ok"] else
                                                         ("robots" if not m["robots_ok"] else "error"))
        filas_feed.append(dict(fuente=m["fuente"], feed=m["feed"], medio=m["medio"], pais=m["pais"],
                               idioma=m["idioma"], consulta=m["consulta"], url=m["url"], robots_ok=m["robots_ok"],
                               estado=estado, n_items=m.get("n_items"), profundidad_h=m.get("profundidad_h"),
                               pct_con_fecha=m.get("pct_con_fecha"), pct_con_hora=m.get("pct_con_hora"),
                               pct_con_resumen=m.get("pct_con_resumen"), edad_min_h=m.get("edad_min_h"),
                               idioma_detectado=m.get("idioma_detectado"), bytes=m["bytes"],
                               segundos=m["segundos"], captura=m["captura"].isoformat() if m["captura"] else None,
                               error=m["error"][:200]))
        if not m["ok"]:
            cfg.log(f"  [{m['fuente']}] {m['feed']}: {m['error'][:100]}")
            continue
        for it in m["items"]:
            fecha, campo = fecha_publicacion(it.get("fecha"), captura)
            filas_item.append(dict(fuente=m["fuente"], feed=m["feed"], medio=it.get("medio") or m["medio"],
                                   pais=m["pais"], idioma=m["idioma"], titular=(it.get("titulo") or "")[:200],
                                   fecha_publicacion=fecha.isoformat(), campo_fecha=campo,
                                   lag_captura_h=base.lag_horas(fecha, captura)))
        docs[m["fuente"]] += documentos(m["items"], fuente=m["fuente"], medio=m["medio"], pais=m["pais"],
                                        lengua=m["idioma"] if m["fuente"] == F_RSS else None,
                                        captura=captura, feed=m["feed"])
        # TimeGate: el ítem más reciente de cada feed (retraso de la captura respecto a su pubDate)
        con_fecha = [i for i in m["items"] if i.get("fecha")]
        if con_fecha:
            ult = max(con_fecha, key=lambda i: i["fecha"])
            fecha, campo = fecha_publicacion(ult["fecha"], captura)
            res.timegate.append(dict(fuente=m["fuente"], item=f"{m['feed']}: {(ult.get('titulo') or '')[:70]}",
                                     fecha_hecho=ult["fecha"].isoformat(), fecha_publicacion=captura.isoformat(),
                                     campo_fecha=f"{campo} -> captura", lag_h=base.lag_horas(fecha, captura)))

    feeds_df = pd.DataFrame(filas_feed)
    items_df = pd.DataFrame(filas_item, columns=["fuente", "feed", "medio", "pais", "idioma", "titular",
                                                 "fecha_publicacion", "campo_fecha", "lag_captura_h"])
    res.tablas["rss_feeds"] = feeds_df
    res.tablas["rss_items"] = items_df

    for fuente, nombre in ((F_RSS, "rss_titulares"), (F_BING, "bing_news_titulares")):
        unicos = {d["doc_id"]: d for d in docs[fuente]}
        if unicos:
            ruta, n = base.guardar_documentos(unicos.values(), nombre, rapido=cfg.modo_rapido)
            res.documentos[ruta.stem] = n

    for fuente in (F_RSS, F_BING):
        sub = [m for m in medidas if m["fuente"] == fuente]
        if not sub:
            continue
        vivos = [m for m in sub if m["ok"] and m["n_items"]]
        bruto = [dict(titulo=r["titular"], url="", fecha=r["fecha_publicacion"], medio=r["medio"])
                 for r in filas_item if r["fuente"] == fuente]
        n_unicos = len(deduplicar(bruto)) if bruto else 0
        # titulares nuevos al día ≈ n_items / profundidad por feed (cada feed renueva su ventana)
        por_dia = sum(m["n_items"] / m["profundidad_h"] * 24 for m in vivos if m.get("profundidad_h"))
        prof = base.resumen_lag([m.get("profundidad_h") for m in vivos])
        edades = base.resumen_lag([m.get("edad_min_h") for m in vivos])
        mb = sum(m["bytes"] for m in sub) / 1e6
        robots_no = [m["feed"] for m in sub if not m["robots_ok"]]
        errores = {m["feed"]: m["error"][:120] for m in sub if m["robots_ok"] and not m["ok"]}
        mediciones = dict(feeds=len(sub), vivos=len(vivos), robots_prohibe=len(robots_no), errores=len(errores),
                          items=sum(m["n_items"] or 0 for m in vivos), titulares_unicos=n_unicos,
                          titulares_dia_estimados=round(por_dia), paises_vivos=sorted({m["pais"] for m in vivos}),
                          idiomas_vivos=dict(Counter(m["idioma"] for m in vivos)), profundidad_h=prof,
                          edad_item_mas_reciente_h=edades, mb_por_pasada=round(mb, 2),
                          feeds_robots_prohibe=robots_no, feeds_con_error=errores,
                          segundos=round(sum(m["segundos"] for m in sub), 1))
        nota = (f"{len(vivos)}/{len(sub)} vivos; {len(robots_no)} prohibidos por robots.txt; profundidad mediana "
                f"{prof['mediana_h']} h; lag = edad del ítem más reciente en la captura (cota del retraso "
                f"publicación->feed).")
        _cobertura(res, fuente, sub, items_df, cfg, nota)
        if fuente == F_RSS:
            res.fichas.append(base.ficha(
                fuente=F_RSS, nombre="RSS de medios financieros y generalistas (mundial, en vivo)",
                agentes=["macro", "equity", "risk"], rag="cajon", url="(lista FEEDS en noticias_alt/rss.py)",
                licencia_y_terminos="RSS para uso personal/no comercial según cada medio; al corpus titular + extracto "
                                    "+ enlace, nunca el cuerpo; a results solo titulares",
                cobertura_inicio=_primera_fecha(items_df, F_RSS),
                cobertura_fin=datetime.now(UTC).date().isoformat(), frecuencia="minutos",
                retraso_publicacion=f"minutos (ítem más reciente con {edades['mediana_h']} h de mediana en la captura)",
                formato="rss 2.0 / atom / rdf", volumen_estimado=(
                    f"{mediciones['items']} ítems por pasada ({n_unicos} únicos, {mb:.1f} MB de XML); ≈{round(por_dia)} "
                    f"titulares nuevos/día -> ≈{round(por_dia * 365 / 1000)} k/año si se captura a diario"),
                limites="sin clave; 1 petición/s por host; sin histórico (profundidad de horas a pocos días)",
                estado_sonda=base.estado(res.registro, F_RSS), mediciones=mediciones, notas=[
                    "Sin histórico: solo vale en vivo. Para backtest futuro, capturar a diario (la caché por día del "
                    "corpus ya es ese archivo) y usar fecha_captura como fecha de publicación conservadora.",
                    "Los feeds prohibidos por robots.txt se miden sin descargarse (Reuters cerró sus RSS públicos en 2020).",
                    "Destino: cajón dinámico del modo en vivo (titular + extracto corto)."]))
        else:
            res.fichas.append(base.ficha(
                fuente=F_BING, nombre="Bing News RSS (búsqueda en vivo)", agentes=["macro", "equity", "risk"],
                rag="cajon", url=BING_NEWS,
                licencia_y_terminos="Términos de Microsoft (el propio feed: solo para un agregador RSS personal, no "
                                    "comercial); la API oficial Bing News Search se retiró en 2025",
                cobertura_inicio=None, cobertura_fin=datetime.now(UTC).date().isoformat(), frecuencia="minutos",
                retraso_publicacion=f"minutos (ítem más reciente con {edades['mediana_h']} h de mediana)",
                formato="rss (título, extracto, enlace apiclick, News:Source)",
                volumen_estimado=f"≈{round(mediciones['items'] / max(len(vivos), 1))} ítems por consulta; sin paginación",
                limites="sin clave; sin histórico ni filtro de fechas; no documentado",
                estado_sonda=base.estado(res.registro, F_BING), mediciones=mediciones, notas=[
                    "Enlaces envueltos en bing.com/news/apiclick.aspx: se desenvuelven al parámetro url.",
                    "Útil en vivo para buscar_noticias(q) cuando no hay feed del medio; mismo papel que en Briefly."]))


def _sondear_google(cfg: base.ConfigSonda, res: base.Resultado) -> None:
    hoy = datetime.now(UTC).date().isoformat()
    url = url_google_news("stock market crash", "2020-03-15", "2020-03-17")
    t0 = base.cronometro()
    permitido = robots_permite(url)
    med: dict[str, Any] = {"robots_permite": permitido}
    if not permitido:
        res.registrar(F_GOOGLE, "busqueda crash 2020-03-16", url, False, dia=hoy, segundos=base.segundos(t0),
                      error="robots.txt")
    else:  # solo si robots.txt cambiase: se mide una consulta
        try:
            items = parsear(leer_feed(url, cache=base.corpus("crudo", MODULO, hoy) / "google_crash.xml",
                                      usar_cache=cfg.usar_cache)[0])
            med.update(items=len(items), dias=dict(Counter(i["fecha"].date().isoformat() for i in items if i["fecha"])))
            res.registrar(F_GOOGLE, "busqueda crash 2020-03-16", url, True, dia=hoy, segundos=base.segundos(t0))
        except Exception as e:  # noqa: BLE001
            res.registrar(F_GOOGLE, "busqueda crash 2020-03-16", url, False, dia=hoy, segundos=base.segundos(t0),
                          error=f"{type(e).__name__}: {e}")
    est = base.estado(res.registro, F_GOOGLE)
    res.tablas.setdefault("rss_feeds", pd.DataFrame())
    fila = pd.DataFrame([dict(fuente=F_GOOGLE, feed="google_news_busqueda", medio="Google News", pais="US",
                              idioma="en", consulta="stock market crash", url=url, robots_ok=permitido,
                              estado="robots" if not permitido else ("ok" if est == "ok" else "error"),
                              n_items=med.get("items"))])
    res.tablas["rss_feeds"] = pd.concat([res.tablas["rss_feeds"], fila.dropna(axis=1, how="all")], ignore_index=True)
    res.cobertura.append(dict(fuente=F_GOOGLE, familia="vivo_redes", tipo="titular", inicio=None, fin=None, hora=False,
                              idiomas="", lag_mediana_h=None, destino="ninguno" if not permitido else "vivo",
                              backtest=False, estado=est,
                              nota="robots.txt de news.google.com no permite /rss/search: no se descarga." if not permitido
                              else "robots.txt lo permite hoy: revisar antes de usarlo."))
    res.fichas.append(base.ficha(
        fuente=F_GOOGLE, nombre="Google News RSS (búsqueda; after:/before:)", agentes=["macro", "equity", "risk"],
        rag="cajon", url=GOOGLE_NEWS, licencia_y_terminos="Términos de Google (sin API oficial de noticias); "
                                                           "titulares de terceros",
        cobertura_inicio=None, cobertura_fin=None, frecuencia="minutos", retraso_publicacion="minutos (en vivo)",
        formato="rss (título, medio, fecha, enlace cifrado)", volumen_estimado="≤100 ítems por consulta; sin paginación",
        limites="robots.txt no permite /rss/search a robots genéricos; CAPTCHA si se abusa",
        estado_sonda=est, mediciones=med, notas=[
            "Comprobado con robots_permite: no se descarga el feed si robots.txt lo prohíbe.",
            "Medición previa (rama fase2/datos, una petición exploratoria): fechas históricas truncadas al día (36/37 "
            "ítems a las 07:00 GMT) -> no apto para el TimeGate del backtest aunque fuera accesible.",
            "Sustitutos: GDELT DOC (titular + URL + seendate) y los RSS directos de FEEDS."]))


def sondear(cfg: base.ConfigSonda) -> base.Resultado:
    """Sonda del vivo: FEEDS + Bing News (consultas de ``muestra``) + Google News (robots.txt)."""
    res = base.Resultado(MODULO)
    for nombre, paso in (("feeds", _sondear_feeds), ("google", _sondear_google)):
        try:
            cfg.log(f"[rss] {nombre}")
            paso(cfg, res)
        except Exception as e:  # noqa: BLE001
            res.registrar(F_RSS if nombre == "feeds" else F_GOOGLE, f"sonda {nombre}", "", False,
                          error=f"{type(e).__name__}: {e}")
    return res
