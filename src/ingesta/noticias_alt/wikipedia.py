"""Wikipedia: Portal:Current events (hechos del día, CC BY-SA) y pageviews (atención del público).

- **Current events**: una página por día con los hechos en viñetas, por categoría (Business and
  economy, Armed conflicts...), cada uno con enlace a su fuente. Texto propio de Wikipedia
  (CC BY-SA 4.0): puede ir al RAG BASE con atribución.
  TimeGate: la página se sigue editando días después; se usa la **revisión vigente a las 23:59:59
  UTC del día** (``rvstart`` + ``rvdir=older``), nunca la actual. Esa revisión es posterior a la hora de
  decisión (16:30 de Nueva York), así que ``fecha_publicacion`` = D+1 (día UTC cerrado, como las visitas).
  Hay páginas antiguas creadas años
  después (retroactivas): si no había revisión al cierre del día, ese día no tiene cobertura válida.
- **Pageviews** (REST de Wikimedia, desde 2015-07-01): visitas diarias por artículo -> serie
  alternativa para Risk/Equity; el día D se publica D+1.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import quote

import pandas as pd

from ingesta.comun.esquema import Documento, doc_id
from ingesta.comun.http import get, robots_permite  # noqa: F401  (robots: API documentada, no se rastrea)
from ingesta.noticias_alt import base
from ingesta.noticias_alt.limpieza import a_utc
from ingesta.noticias_alt.muestra import ARTICULOS_WIKI

MODULO = "wikipedia"
F_CE, F_PV = "wiki_current_events", "wiki_pageviews"

API = "https://en.wikipedia.org/w/api.php"
PAGEVIEWS = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
INICIO_PV = date(2015, 7, 1)
LICENCIA_CE = "CC BY-SA 4.0"
LICENCIA_CE_LARGA = "CC BY-SA 4.0 (texto de Wikipedia; atribución y misma licencia)"
LICENCIA_PV = "CC0 (datos de pageviews de Wikimedia)"
# Fechas para medir desde cuándo hay páginas diarias CONTEMPORÁNEAS (con revisión al cierre del día).
SONDEO_INICIO = (date(2004, 6, 1), date(2005, 6, 1), date(2006, 6, 1), date(2006, 12, 1),
                 date(2007, 3, 1), date(2007, 6, 1))

_ENLACE = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
_FUENTE = re.compile(r"\[(https?://\S+)\s+\(?((?:[^\[\]]|'')*?)\)?\]")
_CATEGORIA = re.compile(r"^(?:;\s*(.+?)|'''([^']+?)''')\s*$")


# --------------------------------------------------------------------------- utilidades

def titulo_portal(d: date) -> str:
    return f"Portal:Current events/{d.year} {d:%B} {d.day}"


def url_revision(revid: int | str) -> str:
    return f"https://en.wikipedia.org/w/index.php?oldid={revid}"


def _api(res: base.Resultado, cfg: base.ConfigSonda, fuente: str, item: str, params: dict[str, Any], *,
         cache: str | None, dia: str = "", url: str = API) -> dict[str, Any] | None:
    """GET JSON con caché en ``<corpus>/crudo/wikipedia`` (atómica; JSON validado); registra. None si falla."""
    ruta = base.corpus("crudo", MODULO) / cache if cache else None
    crudo = base.get_cacheado(res, cfg, fuente, item, url, ruta, getter=get, dia=dia, marca_cache="",
                              validar=json.loads, params=params or None, por_segundo=5.0, timeout=60)
    return None if crudo is None else json.loads(crudo)


def params_revision(titulo: str, *, al_cierre: date | None = None, primera: bool = False,
                    contenido: bool = True) -> dict[str, Any]:
    """Parámetros MediaWiki: revisión al cierre de un día (``al_cierre``), la primera o la actual."""
    p: dict[str, Any] = {"action": "query", "prop": "revisions", "titles": titulo, "rvlimit": 1,
                         "rvprop": "ids|timestamp|size" + ("|content" if contenido else ""), "format": "json",
                         "formatversion": 2}
    if contenido:
        p["rvslots"] = "main"
    if primera:
        p["rvdir"] = "newer"
    elif al_cierre is not None:
        p["rvdir"] = "older"
        p["rvstart"] = f"{al_cierre.isoformat()}T23:59:59Z"
    return p


def revision(j: dict[str, Any] | None) -> dict[str, Any] | None:
    """Respuesta de ``prop=revisions`` -> {revid, timestamp, size, wikitexto} (None si no existía)."""
    if not j:
        return None
    paginas = (j.get("query") or {}).get("pages") or []
    if not paginas or paginas[0].get("missing") or not paginas[0].get("revisions"):
        return None
    rev = paginas[0]["revisions"][0]
    texto = ((rev.get("slots") or {}).get("main") or {}).get("content")
    return dict(revid=rev["revid"], timestamp=rev["timestamp"], size=rev.get("size"), wikitexto=texto)


# --------------------------------------------------------------------------- parseo

def _limpiar(texto: str) -> str:
    t = _ENLACE.sub(r"\1", texto)
    t = re.sub(r"\{\{[^{}]*\}\}", "", t)               # plantillas sueltas
    t = re.sub(r"<ref[^>]*>.*?</ref>|<[^>]+>", "", t)
    return " ".join(t.replace("'''", "").replace("''", "").split())


def parse_current_events(wikitexto: str) -> list[dict[str, Any]]:
    """Viñetas hoja (con fuente) -> ``[{categoria, temas, texto, fuentes: [(url, medio)]}]``.

    Categoría: ``;Business and economy`` (formato 2017-2022) o ``'''Business and economy'''`` (reciente).
    Las viñetas padre sin fuente son temas (``[[Gaza war]]`` > ``[[Israeli occupation...]]``).
    """
    items: list[dict[str, Any]] = []
    categoria: str | None = None
    ruta: list[str] = []
    for linea in (wikitexto or "").splitlines():
        linea = linea.strip()
        mc = _CATEGORIA.match(linea)
        if mc:
            categoria = (mc.group(1) or mc.group(2)).strip()
            ruta = []
            continue
        m = re.match(r"^(\*+)\s*(.*)", linea)
        if not m:
            continue
        nivel, cuerpo = len(m.group(1)), m.group(2)
        fuentes = [(u, _limpiar(medio).strip("() ")) for u, medio in _FUENTE.findall(cuerpo)]
        texto = _limpiar(_FUENTE.sub("", cuerpo))
        ruta = ruta[: nivel - 1] + [texto]
        if fuentes and texto:
            items.append(dict(categoria=categoria, temas=[t for t in ruta[:-1] if t], texto=texto, fuentes=fuentes))
    return items


def hechos_a_documentos(items: list[dict[str, Any]], d: date, rev: dict[str, Any]) -> list[dict[str, Any]]:
    """Hechos de un día -> documentos del RAG base (texto CC BY-SA; fecha_publicacion = D+1: revisión de las 23:59 UTC)."""
    docs = []
    for i, it in enumerate(items):
        texto = it["texto"]
        titulo = texto if len(texto) <= 120 else texto[:119].rsplit(" ", 1)[0] + "…"
        doc = Documento(doc_id=doc_id(f"{F_CE}|{d.isoformat()}|{texto}"), fuente=F_CE,
                        tipo_documento="hecho_current_events", titulo=titulo, fecha_publicacion=d + timedelta(days=1),
                        url=url_revision(rev["revid"]), agente="macro", rag="base", fecha_referencia=d,
                        institucion="Wikipedia", idioma="en", seccion=it["categoria"], licencia=LICENCIA_CE,
                        version=str(rev["revid"]),
                        extra=dict(temas=it["temas"], fuentes=[u for u, _ in it["fuentes"]],
                                   medios=[m for _, m in it["fuentes"]], orden=i, ts_revision=rev["timestamp"],
                                   pagina=titulo_portal(d)))
        docs.append(base.documento_dict(doc, texto=texto))
    return docs


def url_pageviews(articulo: str, inicio: date, fin: date, proyecto: str = "en.wikipedia",
                  granularidad: str = "daily") -> str:
    return (f"{PAGEVIEWS}/{proyecto}/all-access/user/{quote(articulo, safe='')}/{granularidad}/"
            f"{inicio:%Y%m%d}/{fin:%Y%m%d}")


def pageviews_a_serie(j: dict[str, Any], articulo: str) -> pd.DataFrame:
    """Respuesta REST -> tabla larga ``wiki_pv:<art>``; publicación D+1."""
    hoy = date.today().isoformat()
    filas = []
    for it in j.get("items", []) or []:
        ref = datetime.strptime(str(it["timestamp"])[:8], "%Y%m%d").date()
        filas.append(dict(serie=f"wiki_pv:{articulo}", fecha_referencia=ref, fecha_publicacion=ref + timedelta(days=1),
                          valor=it.get("views"), unidad="visitas/día (user, all-access)", fuente=F_PV, vintage=hoy,
                          disponible_utc=f"{ref + timedelta(days=1)}T00:00:00Z", metodo_publicacion="día UTC cerrado -> D+1",
                          nota="Wikimedia REST pageviews per-article"))
    return pd.DataFrame(filas)


def picos(s: pd.DataFrame) -> dict[str, Any]:
    """Pico, mediana y su cociente de una serie de visitas."""
    v = s.set_index("fecha_referencia")["valor"].astype(float)
    med = float(v.median())
    return dict(fecha_pico=str(pd.Timestamp(v.idxmax()).date()), visitas_pico=int(v.max()), mediana=med,
                ratio_pico_mediana=round(float(v.max()) / med, 1) if med else None,
                inicio=str(pd.Timestamp(v.index.min()).date()), fin=str(pd.Timestamp(v.index.max()).date()),
                n_dias=int(v.size))


# --------------------------------------------------------------------------- sonda

def _sondear_ce(cfg: base.ConfigSonda, res: base.Resultado, acum: dict[str, Any]) -> None:
    hoy = date.today()
    filas, docs, lags = [], [], []
    for d in cfg.dias:
        dia = d.fecha.isoformat()
        titulo = titulo_portal(d.fecha)
        sufijo = f"_{hoy:%Y%m%d}" if d.fecha >= hoy - timedelta(days=7) else ""   # días vivos: caché diaria
        try:
            cierre = revision(_api(res, cfg, F_CE, "revision al cierre", params_revision(titulo, al_cierre=d.fecha),
                                   cache=f"ce_cierre_{d.fecha:%Y%m%d}{sufijo}.json", dia=dia))
            if cierre is None or not cierre["wikitexto"]:
                filas.append(dict(clave=d.clave, dia=dia, n_hechos=0, nota="sin cobertura (sin revisión al cierre)"))
                continue
            primera = revision(_api(res, cfg, F_CE, "primera revision", params_revision(titulo, primera=True, contenido=False),
                                    cache=f"ce_primera_{d.fecha:%Y%m%d}.json", dia=dia))
            actual = revision(_api(res, cfg, F_CE, "revision actual", params_revision(titulo),
                                   cache=f"ce_actual_{d.fecha:%Y%m%d}_{hoy:%Y%m%d}.json", dia=dia))
            items = parse_current_events(cierre["wikitexto"])
            n_actual = len(parse_current_events(actual["wikitexto"])) if actual and actual["wikitexto"] else None
            inicio_dia = datetime(d.fecha.year, d.fecha.month, d.fecha.day, tzinfo=UTC)
            ts = a_utc(cierre["timestamp"])
            ts0 = a_utc(primera["timestamp"]) if primera else None
            lag = base.lag_horas(inicio_dia, ts)
            lags.append(lag)
            filas.append(dict(
                clave=d.clave, dia=dia, n_hechos=len(items), n_categorias=len({i["categoria"] for i in items if i["categoria"]}),
                n_fuentes=sum(len(i["fuentes"]) for i in items),
                n_economia=sum(1 for i in items if (i["categoria"] or "").lower().startswith("business")),
                revid=cierre["revid"], ts_revision=cierre["timestamp"], lag_h=lag,
                ts_creacion=primera["timestamp"] if primera else None,
                lag_creacion_h=base.lag_horas(inicio_dia, ts0) if ts0 else None,
                n_hechos_actual=n_actual, revid_actual=actual["revid"] if actual else None,
                bytes_wikitexto=len(cierre["wikitexto"].encode("utf-8")), nota=""))
            lim = items if cfg.tope is None else items[: max(cfg.tope, 1)]
            docs += hechos_a_documentos(lim, d.fecha, cierre)
            acum["dias_ok"].append(dia)
            res.timegate.append(dict(fuente=F_CE, item=f"{d.clave}: {titulo}", fecha_hecho=dia,
                                     fecha_publicacion=cierre["timestamp"],
                                     campo_fecha="revisión vigente a las 23:59:59 UTC (rvstart, rvdir=older)", lag_h=lag))
        except Exception as e:  # noqa: BLE001
            res.registrar(F_CE, "parseo", API, False, dia=dia, error=f"{type(e).__name__}: {e}")
            filas.append(dict(clave=d.clave, dia=dia, n_hechos=None, nota=f"fallo: {str(e)[:80]}"))
    res.tablas["wiki_current_events"] = pd.DataFrame(filas)
    acum["lags_ce"] = lags
    if docs:
        ruta, n = base.guardar_documentos(docs, "wiki_current_events", rapido=cfg.modo_rapido)
        res.documentos[ruta.stem] = n
    # --- desde cuándo hay páginas diarias contemporáneas (revisión al cierre del propio día)
    inicio = []
    for f in (SONDEO_INICIO[-2:] if cfg.modo_rapido else SONDEO_INICIO):
        titulo = titulo_portal(f)
        cierre = revision(_api(res, cfg, F_CE, "sondeo inicio (cierre)",
                               params_revision(titulo, al_cierre=f, contenido=False), cache=f"ce_sondeo_cierre_{f:%Y%m%d}.json",
                               dia=f.isoformat()))
        primera = revision(_api(res, cfg, F_CE, "sondeo inicio (creación)",
                                params_revision(titulo, primera=True, contenido=False),
                                cache=f"ce_sondeo_primera_{f:%Y%m%d}.json", dia=f.isoformat()))
        ts0 = a_utc(primera["timestamp"]) if primera else None
        inicio.append(dict(fecha=f.isoformat(), existe_hoy=primera is not None, revision_al_cierre=cierre is not None,
                           ts_creacion=primera["timestamp"] if primera else None,
                           lag_creacion_dias=round((ts0 - datetime(f.year, f.month, f.day, tzinfo=UTC)).total_seconds()
                                                   / 86400, 1) if ts0 else None))
    res.tablas["wiki_current_events_inicio"] = pd.DataFrame(inicio)
    contemporaneas = [x["fecha"] for x in inicio if x["revision_al_cierre"]] + acum["dias_ok"]
    acum["inicio_ce"] = min(contemporaneas) if contemporaneas else None


def _sondear_pv(cfg: base.ConfigSonda, res: base.Resultado, acum: dict[str, Any]) -> None:
    hoy = date.today()
    arts = list(ARTICULOS_WIKI)[: cfg.tope] if cfg.tope else list(ARTICULOS_WIKI)
    series, filas_picos, filas_testigo = [], [], []
    for a in arts:
        url = url_pageviews(a, INICIO_PV, hoy)
        j = _api(res, cfg, F_PV, a, {}, cache=f"pv_{a}_{hoy:%Y%m%d}.json", url=url)
        if not j:
            continue
        try:
            s = pageviews_a_serie(j, a)
            if s.empty:
                continue
            series.append(s)
            filas_picos.append(dict(articulo=a, **picos(s)))
            v = s.set_index(pd.to_datetime(s["fecha_referencia"]))["valor"].astype(float)
            for d in cfg.dias:
                t = pd.Timestamp(d.fecha)
                if t not in v.index:
                    continue
                previa = v[(v.index < t) & (v.index >= t - pd.Timedelta(days=90))]
                med = float(previa.median()) if not previa.empty else None
                filas_testigo.append(dict(articulo=a, clave=d.clave, dia=d.fecha.isoformat(), visitas=int(v[t]),
                                          mediana_90d_previa=med, ratio=round(float(v[t]) / med, 2) if med else None))
        except Exception as e:  # noqa: BLE001
            res.registrar(F_PV, f"parseo {a}", url, False, error=f"{type(e).__name__}: {e}")
    res.tablas["wiki_pageviews_picos"] = pd.DataFrame(filas_picos)
    res.tablas["wiki_pageviews_testigo"] = pd.DataFrame(filas_testigo)
    if series:
        todo = pd.concat(series, ignore_index=True)
        res.series.append(base.guardar_serie(todo, "wiki_pageviews", rapido=cfg.modo_rapido).stem)
        acum["pv"] = todo
        u = todo.sort_values("fecha_referencia").iloc[-1]
        res.timegate.append(dict(fuente=F_PV, item=f"{u['serie']} {u['fecha_referencia']}",
                                 fecha_hecho=str(u["fecha_referencia"]), fecha_publicacion=str(u["fecha_publicacion"]),
                                 campo_fecha="timestamp (día UTC) -> D+1", lag_h=24.0))


def sondear(cfg: base.ConfigSonda | None = None) -> base.Resultado:
    cfg = cfg or base.ConfigSonda()
    res = base.Resultado(MODULO)
    acum: dict[str, Any] = {"dias_ok": []}
    t0 = base.cronometro()
    for paso, fn in (("current events", _sondear_ce), ("pageviews", _sondear_pv)):
        cfg.log(f"[wikipedia] {paso}…")
        try:
            fn(cfg, res, acum)
        except Exception as e:  # noqa: BLE001
            res.registrar(MODULO, paso, "", False, error=f"{type(e).__name__}: {e}")
    _fichas(cfg, res, acum)
    cfg.log(f"[wikipedia] hecho en {base.segundos(t0)} s; {sum(r['bytes'] for r in res.registro) / 1e6:.1f} MB")
    return res


# --------------------------------------------------------------------------- fichas y cobertura

def _fichas(cfg: base.ConfigSonda, res: base.Resultado, acum: dict[str, Any]) -> None:
    hoy = date.today()
    ce = res.tablas.get("wiki_current_events", pd.DataFrame())
    ok = ce[ce["n_hechos"].fillna(0) > 0] if "n_hechos" in ce else pd.DataFrame()
    ini = acum.get("inicio_ce")
    fin = max(acum["dias_ok"]) if acum.get("dias_ok") else None
    lag = base.resumen_lag(acum.get("lags_ce", []))
    hechos_dia = round(float(ok["n_hechos"].mean()), 1) if not ok.empty else None
    kb_dia = round(float(ok["bytes_wikitexto"].mean()) / 1e3, 1) if not ok.empty else None
    n_dias = (hoy - date.fromisoformat(ini)).days if ini else None
    inicio_tab = res.tablas.get("wiki_current_events_inicio", pd.DataFrame())
    retro = inicio_tab[inicio_tab["existe_hoy"] & ~inicio_tab["revision_al_cierre"]]["fecha"].tolist() \
        if not inicio_tab.empty else []
    cambios = ok[["dia", "n_hechos", "n_hechos_actual"]].to_dict("records") if not ok.empty else []
    extra_notas = []
    if not ok.empty and ok["n_hechos_actual"].notna().any():
        crec = (ok["n_hechos_actual"] / ok["n_hechos"] - 1).dropna()
        extra_notas.append(f"La versión actual tiene una mediana de {100 * crec.median():.0f} % más hechos que la del "
                           f"cierre del día (máx. {100 * crec.max():.0f} %): usar la actual sería look-ahead.")
    if not ok.empty and ok["lag_creacion_h"].notna().any() and ok["lag_creacion_h"].median() < 0:
        extra_notas.append(f"La página se crea antes del día (mediana {ok['lag_creacion_h'].median():.1f} h: plantilla "
                           "vacía de un bot); los hechos se añaden durante el día D.")
    med = dict(dias_con_cobertura=int(len(ok)), hechos_por_dia=hechos_dia, kb_wikitexto_por_dia=kb_dia,
               lag_revision_cierre_h=lag, documentos=res.documentos.get(base.nombre_salida(F_CE, cfg.modo_rapido), 0),
               hechos_cierre_vs_actual=cambios, paginas_retroactivas=retro,
               sin_cobertura=[d.clave for d in cfg.dias if d.fecha.isoformat() not in set(acum.get("dias_ok", []))])
    vol = (f"≈{n_dias:,} días × {hechos_dia} hechos ≈ {n_dias * hechos_dia / 1e3:.0f} mil hechos; "
           f"≈{n_dias * kb_dia / 1e3:.0f} MB de wikitexto") if n_dias and hechos_dia and kb_dia else "sin medir"
    res.fichas.append(base.ficha(
        fuente=F_CE, nombre="Wikipedia · Portal:Current events (hechos diarios del mundo)",
        agentes=["macro", "risk", "equity"], rag="base", url="https://en.wikipedia.org/wiki/Portal:Current_events",
        acceso="api", licencia_y_terminos=LICENCIA_CE_LARGA, cobertura_inicio=ini, cobertura_fin=fin,
        frecuencia="diaria", retraso_publicacion=f"revisión al cierre: mediana {lag['mediana_h']} h desde el inicio del "
        "día UTC cerrado: el día D entra en la decisión de D+1 (la revisión de las 23:59 UTC es posterior a las 16:30 NY)", formato="wikitexto (json MediaWiki API)", volumen_estimado=vol,
        limites="API MediaWiki: UA identificable, sin límite estricto en lectura; 1-3 peticiones por día de calendario",
        estado_sonda=base.estado(res.registro, F_CE), mediciones=med,
        notas=["TimeGate: revisión vigente a las 23:59:59 UTC del día (no la actual): las ediciones posteriores "
               "añaden o corrigen hechos (ver hechos_cierre_vs_actual).",
               f"Páginas creadas años después (retroactivas, sin revisión al cierre): {retro or 'ninguna en el sondeo'}.",
               "Formato: ';Categoría' (2017-2022) o \"'''Categoría'''\" (reciente); 2008 sin categorías.",
               "Texto propio de Wikipedia (CC BY-SA): apto para el RAG base y para fixtures con atribución.",
               *extra_notas]))
    res.cobertura.append(dict(
        fuente=F_CE, familia="gdelt_wiki", tipo="texto", inicio=ini, fin=fin, hora=False, idiomas="en",
        lag_mediana_h=lag["mediana_h"], destino="base", backtest=True, estado=base.estado(res.registro, F_CE),
        nota=f"{hechos_dia} hechos/día con fuente enlazada; revisión al cierre del día"))

    # ---- pageviews
    pv = acum.get("pv")
    pk = res.tablas.get("wiki_pageviews_picos", pd.DataFrame())
    ini_pv = str(pd.Timestamp(pv["fecha_referencia"].min()).date()) if pv is not None else None
    fin_pv = str(pd.Timestamp(pv["fecha_referencia"].max()).date()) if pv is not None else None
    cota = (round((datetime.now(UTC) - datetime.fromisoformat(fin_pv).replace(tzinfo=UTC)).total_seconds() / 3600, 1)
            if fin_pv else None)
    med_pv = dict(articulos=int(len(pk)), filas=int(len(pv)) if pv is not None else 0,
                  ultimo_dia=fin_pv, horas_desde_inicio_ultimo_dia=cota,
                  picos=pk.to_dict("records") if not pk.empty else [],
                  kb_por_articulo=round(sum(r["bytes"] for r in res.registro if r["fuente"] == F_PV)
                                        / max(len(pk), 1) / 1e3, 1))
    res.fichas.append(base.ficha(
        fuente=F_PV, nombre="Wikimedia pageviews (atención por artículo)", agentes=["risk", "equity"], rag="serie",
        url=PAGEVIEWS, acceso="api", licencia_y_terminos=LICENCIA_PV, cobertura_inicio=ini_pv, cobertura_fin=fin_pv,
        frecuencia="diaria", retraso_publicacion=f"último día disponible {fin_pv} (≤ {cota} h desde su inicio); regla D+1",
        formato="json", volumen_estimado=f"{med_pv['kb_por_articulo']} KB por artículo y serie completa",
        limites="≤100 peticiones/s con UA identificable; 'user' excluye bots declarados",
        estado_sonda=base.estado(res.registro, F_PV), mediciones=med_pv,
        notas=["Desde 2015-07-01 (antes, 'pagecounts' en volcados horarios de TB: solo si hace falta 2008).",
               "Valores absolutos y estables entre descargas (mejor que Google Trends para backtest).",
               "Serie 'wiki_pageviews' en tabla larga, serie='wiki_pv:<artículo>', fecha_publicacion = D+1."]))
    res.cobertura.append(dict(
        fuente=F_PV, familia="gdelt_wiki", tipo="serie", inicio=ini_pv, fin=fin_pv, hora=False, idiomas="en",
        lag_mediana_h=cota, destino="serie", backtest=True, estado=base.estado(res.registro, F_PV),
        nota=f"{len(pk)} artículos enteros; lag = cota medida (horas desde el inicio del último día publicado)"))
