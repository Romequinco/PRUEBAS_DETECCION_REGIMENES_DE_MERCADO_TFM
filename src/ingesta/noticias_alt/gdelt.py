"""GDELT: GKG 2.1 (tono, temas, titulares), Events 1.0/2.0 (fecha del hecho frente a captura) y DOC 2.0 API.

GDELT no distribuye el texto de las noticias: da URL, medio, hora de captura, temas, entidades y
**tono** (V2Tone). Sirve para (a) titulares con hora fiable por día (cajón), (b) series de tono y
atención (alternativo para Macro/Risk) y (c) la hora a la que una noticia se hizo pública.

TimeGate (medido en la sonda):

- Events: ``DATEADDED`` (captura), nunca ``SQLDATE`` (fecha del hecho, puede ser días o años antes).
  En los ficheros históricos (antes de 2013-04) ``DATEADDED`` es la fecha del reprocesado de 2013:
  no hay fecha de publicación real, solo sirven como serie de hechos fechados con retraso.
- GKG: ``DATE`` (captura, cada 15 min); ``<PAGE_PRECISEPUBTIMESTAMP>`` de Extras, cuando viene, es la
  hora de publicación del medio y da el retraso medio→GDELT.
- DOC API: ``seendate`` del artículo. El índice consultable va ≈4 días por detrás (las consultas de
  las últimas 24 h dan 0 artículos), así que el tono del día D calculado hoy incluye artículos
  indexados en D+1..D+4: la serie diaria de tono se publica en D+5 (``LAG_INDICE_DOC_DIAS`` + 1).
  Es un vintage único (el de hoy, recalculado), no lo que se veía en su día.
"""
from __future__ import annotations

import html
import io
import json
import re
import time
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, timedelta
from statistics import mean, median
from typing import Any, Iterable

import pandas as pd

from ingesta.comun.esquema import Documento, doc_id
from ingesta.comun.http import get, robots_permite  # noqa: F401  (robots: API documentada, no se rastrea)
from ingesta.noticias_alt import base
from ingesta.noticias_alt.limpieza import a_utc
from ingesta.noticias_alt.muestra import CONSULTAS, DiaTestigo

MODULO = "gdelt"
F_GKG, F_EV, F_DOC = "gdelt_gkg", "gdelt_eventos", "gdelt_doc"

BASE1 = "https://data.gdeltproject.org/events/"
BASE2 = "https://data.gdeltproject.org/gdeltv2/"
DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
LICENCIA = ("GDELT: uso libre, incluida reutilización y redistribución, citando a GDELT Project "
            "(https://www.gdeltproject.org/about.html#termsofuse); las URLs apuntan a contenido con copyright")

INICIO_V2 = datetime(2015, 2, 18, 23, 0, tzinfo=UTC)   # primer fichero GKG 2.1 / Events 2.0
INICIO_EV1_DIARIO = date(2013, 4, 1)                    # antes: mensuales (2006-2013) y anuales
INICIO_DOC = date(2017, 1, 1)                           # ventana de la DOC API
CONSULTA_TONO = "economy"
LAG_INDICE_DOC_DIAS = 4                                 # retraso medido del índice DOC (sonda 2026-10)
PUB_TONO_DIAS = LAG_INDICE_DOC_DIAS + 1                 # tono de D publicado en D+5
PREFIJOS_ECON = ("ECON_", "EPU_")

_TITULO = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)
_PUBTS = re.compile(r"<PAGE_PRECISEPUBTIMESTAMP>(\d{14})</PAGE_PRECISEPUBTIMESTAMP>")
_IDIOMAS = {"english": "en", "spanish": "es", "french": "fr", "german": "de", "italian": "it",
            "portuguese": "pt", "chinese": "zh", "japanese": "ja", "russian": "ru", "arabic": "ar"}


# --------------------------------------------------------------------------- URLs y descargas

def url_gkg(t: datetime) -> str:
    """Fichero GKG 2.1 de 15 minutos (``t`` redondeado al cuarto de hora anterior)."""
    return f"{BASE2}{t:%Y%m%d%H}{t.minute // 15 * 15:02d}00.gkg.csv.zip"


def url_eventos2(t: datetime) -> str:
    return f"{BASE2}{t:%Y%m%d%H}{t.minute // 15 * 15:02d}00.export.CSV.zip"


def url_eventos1(d: date) -> str:
    """Events 1.0: diario desde 2013-04-01; mensual 2006-01..2013-03; anual antes."""
    if d >= INICIO_EV1_DIARIO:
        return f"{BASE1}{d:%Y%m%d}.export.CSV.zip"
    if d >= date(2006, 1, 1):
        return f"{BASE1}{d:%Y%m}.zip"
    return f"{BASE1}{d:%Y}.zip"


def leer_zip(contenido: bytes) -> str:
    z = zipfile.ZipFile(io.BytesIO(contenido))
    return z.read(z.namelist()[0]).decode("utf-8", "replace")


def _zip_valido(datos: bytes) -> None:
    """Lanza si el zip está truncado o corrupto (no se cachea; una caché así se vuelve a bajar)."""
    with zipfile.ZipFile(io.BytesIO(datos)) as z:
        if z.testzip() is not None:
            raise zipfile.BadZipFile("CRC erróneo")


def bajar(res: base.Resultado, cfg: base.ConfigSonda, fuente: str, item: str, url: str, *, dia: str = "",
          cache: str | None = None, **kw: Any) -> bytes | None:
    """GET con caché en ``<corpus>/crudo/gdelt`` (escritura atómica; los zip se validan); registra. None si falla."""
    ruta = base.corpus("crudo", MODULO) / cache if cache else None
    kw.setdefault("timeout", 90)
    return base.get_cacheado(res, cfg, fuente, item, url, ruta, getter=get, dia=dia, marca_cache="",
                             validar=_zip_valido if url.lower().endswith(".zip") else None, **kw)


def horas_muestra(dia: date, n: int, ahora: datetime | None = None) -> list[datetime]:
    """``n`` instantes repartidos en el día (cada 24/n h desde las 00:00 UTC), solo los ya publicados
    (≥ 45 min antes de ``ahora``) y dentro de la cobertura 2.0."""
    ahora = ahora or datetime.now(UTC)
    paso = 24 / max(n, 1)
    ts = [datetime(dia.year, dia.month, dia.day, tzinfo=UTC) + timedelta(hours=int(i * paso)) for i in range(n)]
    return [t for t in ts if INICIO_V2 <= t <= ahora - timedelta(minutes=45)]


# --------------------------------------------------------------------------- parseo GKG

def parse_gkg(texto: str) -> list[dict[str, Any]]:
    """Filas GKG 2.1 -> dicts: fecha (captura UTC), medio, url, tono, temas, titulo, pub (hora del medio)."""
    filas = []
    for linea in texto.splitlines():
        c = linea.split("\t")
        if len(c) < 16:
            continue
        try:
            tono = float(c[15].split(",")[0]) if c[15] else None
        except ValueError:
            tono = None
        extras = c[26] if len(c) > 26 else ""
        m, p = _TITULO.search(extras), _PUBTS.search(extras)
        temas = sorted({t.split(",")[0] for t in c[7].split(";") if t}) if c[7] else []
        filas.append(dict(fecha=a_utc(c[1]), medio=c[3], url=c[4], tono=tono, temas=temas,
                          titulo=" ".join(html.unescape(m.group(1)).split()) if m and m.group(1).strip() else None,
                          pub=a_utc(p.group(1)) if p else None,
                          traduccion=(c[25] or None) if len(c) > 25 else None))
    return filas


def resumen_gkg(filas: list[dict[str, Any]]) -> dict[str, Any]:
    """Cifras de un fichero (o un día) GKG: n, medios, tono, % con titular y retraso medio→GDELT."""
    tonos = [f["tono"] for f in filas if f["tono"] is not None]
    lags = [base.lag_horas(f["pub"], f["fecha"]) for f in filas if f["pub"] and f["fecha"]]
    lags = [x for x in lags if x is not None and 0 <= x <= 24 * 30]   # fuera: pubts erróneos
    n = len(filas)
    return dict(n_articulos=n, n_medios=len({f["medio"] for f in filas}),
                tono_medio=round(mean(tonos), 4) if tonos else None,
                pct_con_titular=round(100 * sum(1 for f in filas if f["titulo"]) / n, 1) if n else None,
                pct_con_pubts=round(100 * sum(1 for f in filas if f["pub"]) / n, 1) if n else None,
                lag_pub_captura_mediana_h=round(median(lags), 2) if lags else None)


def temas_top(filas: Iterable[dict[str, Any]], n: int = 15, prefijos: tuple[str, ...] | None = None
              ) -> list[dict[str, Any]]:
    """Temas más frecuentes (artículos que los citan) con su tono medio; ``prefijos`` filtra (ECON_...)."""
    cuenta: Counter[str] = Counter()
    tonos: dict[str, list[float]] = defaultdict(list)
    total = 0
    for f in filas:
        total += 1
        for t in f["temas"]:
            if prefijos and not t.startswith(prefijos):
                continue
            cuenta[t] += 1
            if f["tono"] is not None:
                tonos[t].append(f["tono"])
    return [dict(tema=t, n=k, pct_articulos=round(100 * k / total, 2) if total else None,
                 tono_medio=round(mean(tonos[t]), 4) if tonos[t] else None)
            for t, k in cuenta.most_common(n)]


def tono_agregado(filas: Iterable[dict[str, Any]], tema: str | None = None) -> dict[str, dict[str, float]]:
    """Tono medio y nº de artículos por día UTC de captura (filtrable por tema GKG)."""
    por_dia: dict[str, list[float]] = defaultdict(list)
    for f in filas:
        if f.get("tono") is None or f.get("fecha") is None or (tema and tema not in f["temas"]):
            continue
        por_dia[f["fecha"].date().isoformat()].append(f["tono"])
    return {d: {"n": len(v), "tono": round(mean(v), 4)} for d, v in sorted(por_dia.items())}


def titulares_gkg(filas: Iterable[dict[str, Any]], dia: DiaTestigo | None = None) -> list[dict[str, Any]]:
    """Titulares del GKG como documentos del cajón (solo titular + metadatos; nunca cuerpo)."""
    docs = []
    for f in filas:
        if not f["titulo"] or not f["fecha"]:
            continue
        doc = Documento(doc_id=doc_id(f["url"] or f["titulo"]), fuente=F_GKG, tipo_documento="titular",
                        titulo=f["titulo"][:300], fecha_publicacion=f["fecha"].date(), url=f["url"],
                        agente="comun", rag="cajon", institucion=f["medio"], idioma="en", licencia=LICENCIA,
                        extra=dict(captura_utc=f["fecha"].isoformat(), tono=f["tono"],
                                   pub_medio_utc=f["pub"].isoformat() if f["pub"] else None,
                                   temas=f["temas"][:20], dia_testigo=dia.clave if dia else None))
        docs.append(base.documento_dict(doc))
    return docs


# --------------------------------------------------------------------------- parseo Events

def parse_eventos(texto: str) -> list[dict[str, Any]]:
    """Events 1.0 (57 col. histórico / 58 diario) o 2.0 (61 col.) -> sqldate, dateadded, url, tono.

    ``dateadded`` es la fecha de publicación TimeGate (captura); ``sqldate`` la del hecho.
    """
    filas = []
    for linea in texto.splitlines():
        c = linea.rstrip("\r").split("\t")
        if len(c) >= 61:
            version, da, url = "2.0", a_utc(c[59]), c[60] or None
        elif len(c) >= 57:
            version, da, url = ("1.0" if len(c) >= 58 else "1.0_historico"), a_utc(c[56]), (c[57] or None) if len(c) >= 58 else None
        else:
            continue
        sq = a_utc(c[1])
        if sq is None or da is None:
            continue
        try:
            tono = float(c[34]) if c[34] else None
        except ValueError:
            tono = None
        filas.append(dict(id=c[0], sqldate=sq, dateadded=da, url=url, tono=tono, codigo=c[26], version=version))
    return filas


def resumen_eventos(filas: list[dict[str, Any]]) -> dict[str, Any]:
    """% de eventos cuyo hecho (SQLDATE) es anterior al día de captura y retraso en días."""
    n = len(filas)
    if not n:
        return dict(n=0, pct_sqldate_anterior=None, lag_dias_mediana=None, lag_dias_p90=None, lag_dias_max=None,
                    pct_con_url=None)
    lag = pd.Series([(f["dateadded"].date() - f["sqldate"].date()).days for f in filas], dtype=float)
    return dict(n=n, pct_sqldate_anterior=round(100 * float((lag > 0).mean()), 2),
                lag_dias_mediana=float(lag.median()), lag_dias_p90=float(lag.quantile(.9)),
                lag_dias_max=float(lag.max()), pct_con_url=round(100 * sum(1 for f in filas if f["url"]) / n, 1))


def ejemplos_timegate_eventos(filas: list[dict[str, Any]], item: str) -> list[dict[str, Any]]:
    """Dos ejemplos (retraso típico y máximo) para la tabla TimeGate común."""
    if not filas:
        return []
    orden = sorted(filas, key=lambda f: f["dateadded"] - f["sqldate"])
    out = []
    for f in (orden[len(orden) // 2], orden[-1]):
        out.append(dict(fuente=F_EV, item=f"{item}:{f['id']}", fecha_hecho=f["sqldate"].date().isoformat(),
                        fecha_publicacion=f["dateadded"].isoformat(), campo_fecha="DATEADDED (no SQLDATE)",
                        lag_h=base.lag_horas(f["sqldate"], f["dateadded"])))
    return out


# --------------------------------------------------------------------------- DOC API

def _es_limite(texto: str) -> bool:
    return "limit requests" in texto.lower() or "too many" in texto.lower()


def doc_api(query: str, inicio: datetime, fin: datetime, modo: str = "artlist", max_registros: int = 75, *,
            intentos: int = 3, espera_429: float = 15.0) -> tuple[dict[str, Any], bytes]:
    """GDELT DOC 2.0 en JSON. Límite: 1 petición / 5 s por IP; ante 429 espera y reintenta."""
    params = {"query": query, "mode": modo, "format": "json", "startdatetime": f"{inicio:%Y%m%d%H%M%S}",
              "enddatetime": f"{fin:%Y%m%d%H%M%S}"}
    if modo == "artlist":
        params["maxrecords"] = str(max_registros)
    ultimo: Exception | None = None
    for k in range(intentos):
        try:
            r = get(DOC_API, params=params, por_segundo=0.2, reintentos=1, timeout=60)
            txt = r.content.decode("utf-8", "replace").strip()
            if _es_limite(txt):
                raise RuntimeError("429 (texto de límite de la DOC API)")
            if not txt:
                return {}, r.content      # sin resultados: la API responde vacío
            if not txt.startswith("{"):
                raise ValueError(txt[:160])
            return json.loads(txt), r.content
        except Exception as e:  # noqa: BLE001
            ultimo = e
            if "429" not in str(e) or k == intentos - 1:
                break
            time.sleep(espera_429 * (k + 1))
    assert ultimo is not None
    raise ultimo


def idioma_iso(nombre: str | None) -> str:
    return _IDIOMAS.get((nombre or "").strip().lower(), (nombre or "und").strip().lower() or "und")


def articulos_a_documentos(arts: Iterable[dict[str, Any]], consulta: str, dia: str | None = None) -> list[dict]:
    """Artículos de ``artlist`` -> documentos titular (cajón); fecha_publicacion = día de ``seendate``."""
    docs = []
    for a in arts:
        visto = a_utc(a.get("seendate"))
        if not visto or not a.get("title"):
            continue
        doc = Documento(doc_id=doc_id(a.get("url") or a["title"]), fuente=F_DOC, tipo_documento="titular",
                        titulo=" ".join(html.unescape(str(a["title"])).split())[:300], fecha_publicacion=visto.date(),
                        url=a.get("url", ""), agente="comun", rag="cajon", institucion=a.get("domain"),
                        idioma=idioma_iso(a.get("language")),
                        paises=[a["sourcecountry"]] if a.get("sourcecountry") else [], licencia=LICENCIA,
                        extra=dict(seendate_utc=visto.isoformat(), consulta=consulta, dia_testigo=dia))
        docs.append(base.documento_dict(doc))
    return docs


def tono_a_serie(j: dict[str, Any], serie: str, consulta: str) -> pd.DataFrame:
    """``timelinetone`` diario -> tabla larga; publicación D+5 (índice DOC con ≈4 días de retraso + margen)."""
    datos = ((j.get("timeline") or [{}])[0] or {}).get("data", [])
    filas = []
    hoy = date.today().isoformat()
    for p in datos:
        t = a_utc(p.get("date"))
        if t is None:
            continue
        ref = t.date()
        pub = ref + timedelta(days=PUB_TONO_DIAS)
        filas.append(dict(serie=serie, fecha_referencia=ref, fecha_publicacion=pub,
                          valor=p.get("value"), unidad="tono medio GDELT (-100..100)", fuente=F_DOC, vintage=hoy,
                          disponible_utc=f"{pub}T00:00:00Z",
                          metodo_publicacion=f"índice DOC con ~{LAG_INDICE_DOC_DIAS} d de retraso -> D+{PUB_TONO_DIAS}; "
                                             "vintage único",
                          nota=f"DOC API timelinetone, query={consulta!r}"))
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- sonda

def _sondear_gkg_eventos(cfg: base.ConfigSonda, res: base.Resultado, acum: dict[str, Any]) -> None:
    n_dia = 1 if cfg.modo_rapido else int(cfg.extra.get("gkg_por_dia", 6))
    filas_gkg, filas_temas, filas_ev, docs = [], [], [], []
    for d in cfg.dias:
        dia = d.fecha.isoformat()
        horas = horas_muestra(d.fecha, n_dia)
        # --- GKG
        if not horas:
            filas_gkg.append(dict(clave=d.clave, dia=dia, fichero_utc=None, nota="sin cobertura (GKG 2.1 desde 2015-02-18)"))
        filas_dia: list[dict] = []
        for t in horas:
            try:
                url = url_gkg(t)
                datos = bajar(res, cfg, F_GKG, f"gkg {t:%H%M}", url, dia=dia, cache=url.rsplit("/", 1)[1])
                if datos is None:
                    filas_gkg.append(dict(clave=d.clave, dia=dia, fichero_utc=t.isoformat(), nota="fallo de descarga"))
                    continue
                f = parse_gkg(leer_zip(datos))
                filas_dia += f
                filas_gkg.append(dict(clave=d.clave, dia=dia, fichero_utc=t.isoformat(), hora_utc=t.hour,
                                      bytes_zip=len(datos), **resumen_gkg(f), nota=""))
                docs += titulares_gkg(f if cfg.tope is None else f[: cfg.tope], d)
                acum["gkg_bytes"].append(len(datos))
                ej = next((x for x in f if x["pub"] and x["fecha"] and x["titulo"]), None)
                if ej and len([x for x in res.timegate if x["fuente"] == F_GKG]) < 6:
                    res.timegate.append(dict(fuente=F_GKG, item=ej["titulo"][:80], fecha_hecho=ej["pub"].isoformat(),
                                             fecha_publicacion=ej["fecha"].isoformat(),
                                             campo_fecha="DATE (captura); hecho = PAGE_PRECISEPUBTIMESTAMP",
                                             lag_h=base.lag_horas(ej["pub"], ej["fecha"])))
            except Exception as e:  # noqa: BLE001
                res.registrar(F_GKG, f"parseo gkg {t:%H%M}", url_gkg(t), False, dia=dia, error=repr(e))
        if filas_dia:
            for grupo, pref in (("todos", None), ("economia", PREFIJOS_ECON)):
                for x in temas_top(filas_dia, 15, pref):
                    filas_temas.append(dict(clave=d.clave, dia=dia, grupo=grupo, **x))
            acum["gkg_dias"].append(dia)
        # --- Events (1.0 antes de la cobertura 2.0; 2.0 en los mismos instantes que el GKG)
        try:
            if d.fecha < INICIO_V2.date():
                url = url_eventos1(d.fecha)
                cache = base.corpus("crudo", MODULO) / url.rsplit("/", 1)[1]
                if cfg.modo_rapido and not (cfg.usar_cache and cache.exists()):
                    continue                     # modo rápido: el zip histórico (≈33 MB) solo si ya está en caché
                datos = bajar(res, cfg, F_EV, "events1", url, dia=dia, cache=url.rsplit("/", 1)[1], timeout=180)
                fev = [x for x in parse_eventos(leer_zip(datos)) if x["sqldate"].date() == d.fecha] if datos else []
                version = fev[0]["version"] if fev else "1.0"
            else:
                fev, version = [], "2.0"
                for t in horas:
                    url = url_eventos2(t)
                    datos = bajar(res, cfg, F_EV, f"events2 {t:%H%M}", url, dia=dia, cache=url.rsplit("/", 1)[1])
                    if datos:
                        fev += parse_eventos(leer_zip(datos))
            if fev:
                filas_ev.append(dict(clave=d.clave, dia=dia, version=version, ficheros=1 if version != "2.0" else len(horas),
                                     **resumen_eventos(fev)))
                res.timegate.extend(ejemplos_timegate_eventos(fev, f"{d.clave} {version}"))
                acum["ev_dias"].append(dia)
                if version == "1.0_historico":
                    acum["ev_historico"] = resumen_eventos(fev)
                else:
                    acum["ev_lags_h"] += [base.lag_horas(f["sqldate"], f["dateadded"]) for f in fev]
        except Exception as e:  # noqa: BLE001
            res.registrar(F_EV, "parseo eventos", "", False, dia=dia, error=repr(e))
    res.tablas["gdelt_gkg_muestra"] = pd.DataFrame(filas_gkg)
    res.tablas["gdelt_temas"] = pd.DataFrame(filas_temas)
    g = pd.DataFrame(filas_gkg)
    res.tablas["gdelt_tono_intradia"] = (
        g.dropna(subset=["tono_medio"])[["clave", "dia", "hora_utc", "tono_medio", "n_articulos"]]
        .rename(columns={"n_articulos": "n"}).reset_index(drop=True)
        if "tono_medio" in g else pd.DataFrame(columns=["clave", "dia", "hora_utc", "tono_medio", "n"]))
    res.tablas["gdelt_eventos_lag"] = pd.DataFrame(filas_ev)
    if docs:
        ruta, n = base.guardar_documentos(_unicos(docs), "gdelt_gkg_titulares", rapido=cfg.modo_rapido)
        res.documentos[ruta.stem] = n


def _unicos(docs: Iterable[dict]) -> list[dict]:
    vistos: dict[str, dict] = {}
    for d in docs:
        vistos.setdefault(d["doc_id"], d)
    return list(vistos.values())


def _limites_v2(cfg: base.ConfigSonda, res: base.Resultado, acum: dict[str, Any]) -> None:
    """Fin medido (lastupdate.txt), inicio medido (primer fichero 2.0) y volumen de Events 1.0 (index)."""
    datos = bajar(res, cfg, F_GKG, "lastupdate", BASE2 + "lastupdate.txt", timeout=30)
    if datos:
        m = re.findall(r"/(\d{14})\.gkg\.csv\.zip", datos.decode("utf-8", "replace"))
        acum["fin_v2"] = a_utc(m[0]) if m else None
    primero = bajar(res, cfg, F_EV, "primer fichero 2.0", url_eventos2(INICIO_V2), cache="primero_export.zip")
    acum["inicio_v2"] = INICIO_V2 if primero else None
    idx = bajar(res, cfg, F_EV, "index events 1.0", BASE1 + "index.html", cache="events1_index.html", timeout=60)
    if idx:
        txt = idx.decode("utf-8", "replace")
        zips = sorted(re.findall(r'HREF="([^"]+\.zip)"', txt))
        tam = re.findall(r"\(([\d.]+)(MB|GB)\)", txt)
        acum["ev1_ficheros"] = len(zips)
        acum["ev1_primero"] = zips[0] if zips else None
        acum["ev1_gb"] = round(sum(float(v) / (1024 if u == "MB" else 1) for v, u in tam), 1)


def _doc_tono(cfg: base.ConfigSonda, res: base.Resultado, acum: dict[str, Any], espera: float) -> None:
    """Serie diaria de tono (timelinetone) desde 2017: va primero, es lo más valioso de la DOC API."""
    hoy = date.today()
    cache = base.corpus("crudo", MODULO) / f"doc_timelinetone_{CONSULTA_TONO}_{hoy:%Y%m%d}.json"
    t0 = base.cronometro()
    try:
        if cfg.usar_cache and cache.exists():
            crudo = cache.read_bytes()
            j = json.loads(crudo)
            res.registrar(F_DOC, "timelinetone", DOC_API, True, bytes=len(crudo))
        else:
            ini = datetime(hoy.year, 10, 1, tzinfo=UTC) if cfg.modo_rapido else datetime(2017, 1, 1, tzinfo=UTC)
            j, crudo = doc_api(CONSULTA_TONO, ini, datetime(hoy.year, hoy.month, hoy.day, tzinfo=UTC), "timelinetone",
                               espera_429=espera)
            base.escribir_atomico(cache, crudo)
            res.registrar(F_DOC, "timelinetone", DOC_API, True, bytes=len(crudo), segundos=base.segundos(t0))
        s = tono_a_serie(j, f"gdelt_tono:{CONSULTA_TONO}", CONSULTA_TONO)
        if not s.empty:
            res.series.append(base.guardar_serie(s, "gdelt_tono_economia", rapido=cfg.modo_rapido).stem)
            acum["tono"] = s
            u = s.iloc[-1]
            res.timegate.append(dict(fuente=F_DOC, item=f"tono {CONSULTA_TONO} {u['fecha_referencia']}",
                                     fecha_hecho=str(u["fecha_referencia"]), fecha_publicacion=str(u["fecha_publicacion"]),
                                     campo_fecha=f"date (día UTC) -> D+{PUB_TONO_DIAS} (retraso del índice DOC)",
                                     lag_h=24.0 * PUB_TONO_DIAS))
    except Exception as e:  # noqa: BLE001
        res.registrar(F_DOC, "timelinetone", DOC_API, False, segundos=base.segundos(t0), error=f"{type(e).__name__}: {e}")


def _sondear_doc(cfg: base.ConfigSonda, res: base.Resultado, acum: dict[str, Any]) -> None:
    """DOC API: tono diario y titulares (artlist) por consulta y día testigo.

    Con la IP compartida la API responde 429 a menudo: tras ``doc_max_fallos`` fallos seguidos se
    deja de pedir (cortacircuitos) y el resto se anota como omitido, sin peticiones.
    """
    espera = float(cfg.extra.get("espera_429", 10.0))
    n_q = 1 if cfg.modo_rapido else int(cfg.extra.get("doc_consultas", 3))
    consultas = list(CONSULTAS)[:n_q]
    max_reg = int(cfg.extra.get("doc_max_registros", 75))
    max_fallos = int(cfg.extra.get("doc_max_fallos", 3))
    _doc_tono(cfg, res, acum, espera)
    seguidos = 0 if any(r["ok"] for r in res.registro if r["fuente"] == F_DOC) else 1
    filas, docs = [], []
    ahora = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    dias = cfg.dias
    if cfg.modo_rapido:                          # modo rápido: solo el primer día con cobertura
        dias = [d for d in cfg.dias if d.fecha < INICIO_DOC] + [d for d in cfg.dias if d.fecha >= INICIO_DOC][:1]
    for d in dias:
        dia = d.fecha.isoformat()
        if d.fecha < INICIO_DOC:
            filas.append(dict(clave=d.clave, dia=dia, consulta=None, n=0, nota="sin cobertura (DOC API desde 2017)"))
            continue
        ini = datetime(d.fecha.year, d.fecha.month, d.fecha.day, tzinfo=UTC)
        fin = ini + timedelta(hours=23, minutes=59, seconds=59)
        if fin > ahora:                          # «hoy»: últimas 24 h cerradas
            ini, fin = ahora - timedelta(hours=24), ahora
        for q in consultas:
            consulta = f'"{q}" sourcelang:english' if " " in q else f"{q} sourcelang:english"
            slug = re.sub(r"\W+", "_", q)
            cache = base.corpus("crudo", MODULO) / f"doc_artlist_{slug}_{ini:%Y%m%d%H}_{fin:%Y%m%d%H}.json"
            en_cache = cfg.usar_cache and cache.exists()
            if not en_cache and seguidos >= max_fallos:
                filas.append(dict(clave=d.clave, dia=dia, consulta=q, n=None,
                                  nota=f"omitida: {seguidos} fallos seguidos (429)"))
                continue
            t0 = base.cronometro()
            try:
                if en_cache:
                    crudo = cache.read_bytes()
                    j = json.loads(crudo) if crudo.strip() else {}
                    res.registrar(F_DOC, f"artlist {q}", DOC_API, True, dia=dia, bytes=len(crudo))
                else:
                    j, crudo = doc_api(consulta, ini, fin, "artlist", max_reg, intentos=2, espera_429=espera)
                    base.escribir_atomico(cache, crudo)
                    res.registrar(F_DOC, f"artlist {q}", DOC_API, True, dia=dia, bytes=len(crudo),
                                  segundos=base.segundos(t0))
                seguidos = 0
            except Exception as e:  # noqa: BLE001
                seguidos += 1
                res.registrar(F_DOC, f"artlist {q}", DOC_API, False, dia=dia, segundos=base.segundos(t0),
                              error=f"{type(e).__name__}: {e}")
                filas.append(dict(clave=d.clave, dia=dia, consulta=q, n=None, nota=f"fallo: {str(e)[:80]}"))
                continue
            arts = j.get("articles", []) or []
            vistos = [a_utc(a.get("seendate")) for a in arts]
            vistos = [v for v in vistos if v]
            filas.append(dict(clave=d.clave, dia=dia, consulta=q, n=len(arts),
                              n_dominios=len({a.get("domain") for a in arts}),
                              n_idiomas=len({a.get("language") for a in arts}),
                              n_paises=len({a.get("sourcecountry") for a in arts}),
                              seen_min=min(vistos).isoformat() if vistos else None,
                              seen_max=max(vistos).isoformat() if vistos else None, nota=""))
            docs += articulos_a_documentos(arts, q, d.clave)
            acum["doc_dias"] += [v.date().isoformat() for v in vistos]
            if vistos and not any(x["fuente"] == F_DOC and x["item"].startswith(d.clave) for x in res.timegate):
                a = next(a for a in arts if a_utc(a.get("seendate")))
                res.timegate.append(dict(fuente=F_DOC, item=f"{d.clave}: {str(a.get('title', ''))[:70]}",
                                         fecha_hecho=None, fecha_publicacion=a_utc(a.get("seendate")).isoformat(),
                                         campo_fecha="seendate (vista por GDELT)", lag_h=None))
    res.tablas["gdelt_doc_muestra"] = pd.DataFrame(filas)
    if docs:
        ruta, n = base.guardar_documentos(_unicos(docs), "gdelt_doc_titulares", rapido=cfg.modo_rapido)
        res.documentos[ruta.stem] = n



def sondear(cfg: base.ConfigSonda | None = None) -> base.Resultado:
    cfg = cfg or base.ConfigSonda()
    res = base.Resultado(MODULO)
    acum: dict[str, Any] = defaultdict(list)
    t0 = base.cronometro()
    for paso, fn in (("límites", _limites_v2), ("gkg+eventos", _sondear_gkg_eventos), ("doc", _sondear_doc)):
        cfg.log(f"[gdelt] {paso}…")
        try:
            fn(cfg, res, acum)
        except Exception as e:  # noqa: BLE001
            res.registrar(MODULO, paso, "", False, error=f"{type(e).__name__}: {e}")
    _fichas(cfg, res, acum)
    cfg.log(f"[gdelt] hecho en {base.segundos(t0)} s; {sum(r['bytes'] for r in res.registro) / 1e6:.1f} MB")
    return res


# --------------------------------------------------------------------------- fichas y cobertura

def _mb(res: base.Resultado, fuente: str) -> float:
    return round(sum(r["bytes"] for r in res.registro if r["fuente"] == fuente) / 1e6, 2)


def _fichas(cfg: base.ConfigSonda, res: base.Resultado, acum: dict[str, Any]) -> None:
    hoy = date.today()
    fin_v2 = acum.get("fin_v2")
    fin_txt = fin_v2.date().isoformat() if isinstance(fin_v2, datetime) else None
    ini_v2 = acum.get("inicio_v2")
    ini_txt = ini_v2.date().isoformat() if isinstance(ini_v2, datetime) else None
    sin_cob = lambda dias: [d.clave for d in cfg.dias if d.fecha.isoformat() not in set(dias)]  # noqa: E731

    # ---- GKG
    g = res.tablas.get("gdelt_gkg_muestra", pd.DataFrame())
    ok = g.dropna(subset=["n_articulos"]) if "n_articulos" in g else pd.DataFrame()
    mb_fich = round(mean(acum["gkg_bytes"]) / 1e6, 2) if acum["gkg_bytes"] else None
    dias_v2 = (hoy - INICIO_V2.date()).days
    lag_gkg = round(float(ok["lag_pub_captura_mediana_h"].median()), 2) if not ok.empty else None
    med_gkg = dict(ficheros=int(len(ok)), mb_zip_por_fichero=mb_fich, mb_descargados=_mb(res, F_GKG),
                   articulos_por_fichero=round(float(ok["n_articulos"].mean()), 1) if not ok.empty else None,
                   medios_por_fichero=round(float(ok["n_medios"].mean()), 1) if not ok.empty else None,
                   pct_con_titular=round(float(ok["pct_con_titular"].mean()), 1) if not ok.empty else None,
                   pct_con_pubts=round(float(ok["pct_con_pubts"].mean()), 1) if not ok.empty else None,
                   lag_pub_captura_mediana_h=lag_gkg, ultimo_fichero=fin_v2.isoformat() if fin_v2 else None,
                   tono_por_dia={r["dia"]: round(float(r["tono_medio"]), 3)
                                 for r in ok.groupby("dia", as_index=False)["tono_medio"].mean().to_dict("records")}
                   if not ok.empty else {})
    sin_titular = sorted(ok.loc[ok["pct_con_titular"] == 0, "dia"].unique().tolist()) if not ok.empty else []
    med_gkg["dias_sin_titular"] = sin_titular
    vol_gkg = (f"≈{mb_fich * 96 / 1000:.2f} GB zip/día medidos -> ≈{mb_fich * 96 * dias_v2 / 1e6:.1f} TB 2015-hoy "
               "(sin contar *.translation.gkg)") if mb_fich else "sin medir"
    res.fichas.append(base.ficha(
        fuente=F_GKG, nombre="GDELT Global Knowledge Graph 2.1 (tono, temas, entidades, titulares)",
        agentes=["macro", "risk", "equity"], rag="cajon", url="https://www.gdeltproject.org/data.html#rawdatafiles",
        acceso="descarga", licencia_y_terminos=LICENCIA, cobertura_inicio=ini_txt, cobertura_fin=fin_txt,
        frecuencia="15 min", retraso_publicacion=f"captura ≈ {lag_gkg} h después de la hora del medio (mediana, "
        "PAGE_PRECISEPUBTIMESTAMP); fichero publicado ≈15 min tras la captura", formato="csv tabulado en zip",
        volumen_estimado=vol_gkg, limites="sin límite documentado; un zip por cuarto de hora (descarga masiva = TB)",
        estado_sonda=base.estado(res.registro, F_GKG), mediciones=med_gkg,
        notas=[f"Sin cobertura en la muestra: {sin_cob(acum['gkg_dias']) or 'ninguno'}.",
               f"{med_gkg['pct_con_titular']} % de filas con <PAGE_TITLE>: titular sin descargar la página"
               + (f" (ninguno en {sin_titular}: el campo aún no existía)." if sin_titular else "."),
               "Ficheros con huecos: algunos cuartos de hora traen muy pocas filas (caídas de GDELT).",
               "V2Tone: tono medio, % positivo, % negativo, polaridad, densidad, autorreferencia, nº palabras.",
               "Feed principal solo en inglés; el traducido (≈65 idiomas) va en *.translation.gkg.csv.zip."]))
    res.cobertura.append(dict(
        fuente=F_GKG, familia="gdelt_wiki", tipo="titular", inicio=ini_txt, fin=fin_txt, hora=True, idiomas="en",
        lag_mediana_h=lag_gkg, destino="cajon", backtest=True, estado=base.estado(res.registro, F_GKG),
        nota=f"{med_gkg['articulos_por_fichero']} art./15 min, {mb_fich} MB/fichero; titulares + tono desde 2015-02"))

    # ---- Events
    ev = res.tablas.get("gdelt_eventos_lag", pd.DataFrame())
    hist = acum.get("ev_historico") or {}
    lag_ev = base.resumen_lag(acum["ev_lags_h"])
    ini_ev = None
    if acum.get("ev1_primero"):
        anio = re.match(r"\d{4}", acum["ev1_primero"])
        ini_ev = f"{anio.group(0)}-01-01" if anio else None
    med_ev = dict(ficheros_events1=acum.get("ev1_ficheros"), primero_events1=acum.get("ev1_primero"),
                  volumen_events1_gb=acum.get("ev1_gb"), mb_descargados=_mb(res, F_EV), lag_2_0_h=lag_ev,
                  historico_lehman=hist, por_dia=ev.to_dict("records") if not ev.empty else [])
    notas_ev = ["TimeGate: DATEADDED (captura), nunca SQLDATE (fecha del hecho)."]
    if hist:
        notas_ev.append(f"Histórico (antes de 2013-04): DATEADDED = reprocesado de 2013 (mediana {hist['lag_dias_mediana']:.0f} "
                        "días tras el hecho) y sin SOURCEURL: solo serie de hechos, sin fecha de publicación real.")
    res.fichas.append(base.ficha(
        fuente=F_EV, nombre="GDELT Events 1.0 (1979-) y 2.0 (2015-)", agentes=["macro", "risk"], rag="serie",
        url="https://www.gdeltproject.org/data.html", acceso="descarga", licencia_y_terminos=LICENCIA,
        cobertura_inicio=ini_ev, cobertura_fin=fin_txt, frecuencia="15 min (2.0) / diaria (1.0 desde 2013-04)",
        retraso_publicacion=f"DATEADDED − SQLDATE: mediana {lag_ev['mediana_h']} h, p90 {lag_ev['p90_h']} h (2.0)",
        formato="csv tabulado en zip", volumen_estimado=f"Events 1.0 completo ≈ {acum.get('ev1_gb')} GB zip; 2.0 ≈ 0,1 MB/15 min",
        limites="sin límite documentado; descarga por fichero", estado_sonda=base.estado(res.registro, F_EV),
        mediciones=med_ev, notas=notas_ev + ["Sin texto ni titular: códigos CAMEO, actores, tono, Goldstein y URL (desde 2013-04)."]))
    res.cobertura.append(dict(
        fuente=F_EV, familia="gdelt_wiki", tipo="evento", inicio=ini_ev, fin=fin_txt, hora=True, idiomas="en",
        lag_mediana_h=lag_ev["mediana_h"], destino="serie", backtest=True, estado=base.estado(res.registro, F_EV),
        nota=(f"{100 - ev.loc[ev['version'] == '2.0', 'pct_sqldate_anterior'].mean():.1f} % capturados el día del hecho "
              "(SQLDATE sin hora: lag desde las 00:00); antes de 2013-04 la captura es el reprocesado de 2013"
              if not ev.empty and (ev["version"] == "2.0").any() else "sin muestra 2.0")))

    # ---- DOC
    dm = res.tablas.get("gdelt_doc_muestra", pd.DataFrame())
    tono = acum.get("tono")
    ini_doc = str(pd.Timestamp(tono["fecha_referencia"].min()).date()) if tono is not None else (
        min(acum["doc_dias"]) if acum["doc_dias"] else None)
    fin_doc = str(pd.Timestamp(tono["fecha_referencia"].max()).date()) if tono is not None else (
        max(acum["doc_dias"]) if acum["doc_dias"] else None)
    okd = dm.dropna(subset=["n"]) if "n" in dm else pd.DataFrame()
    okd = okd[okd["consulta"].notna()] if not okd.empty else okd
    retraso_dias = (hoy - date.fromisoformat(fin_doc)).days if fin_doc else None
    hoy_vacio = bool(((dm.get("clave") == "hoy") & (dm.get("n") == 0)).any()) if "clave" in dm else False
    med_doc = dict(retraso_indice_dias=retraso_dias, hoy_sin_resultados=hoy_vacio, peticiones=int(sum(1 for r in res.registro if r["fuente"] == F_DOC)),
                   errores=int(sum(1 for r in res.registro if r["fuente"] == F_DOC and not r["ok"])),
                   articulos_por_consulta=round(float(okd["n"].mean()), 1) if not okd.empty else None,
                   titulares=res.documentos.get(base.nombre_salida("gdelt_doc_titulares", cfg.modo_rapido), 0),
                   puntos_tono=int(len(tono)) if tono is not None else 0, mb_descargados=_mb(res, F_DOC),
                   omitidas_por_429=int(dm["nota"].astype(str).str.startswith("omitida").sum()) if "nota" in dm else 0)
    res.fichas.append(base.ficha(
        fuente=F_DOC, nombre="GDELT DOC 2.0 API (búsqueda de titulares y timelines de tono)",
        agentes=["macro", "equity", "risk"], rag="cajon", url=DOC_API, acceso="api", licencia_y_terminos=LICENCIA,
        cobertura_inicio=ini_doc, cobertura_fin=fin_doc, frecuencia="15 min (artlist) / diaria (tono)",
        retraso_publicacion=(f"seendate ≈ minutos tras la publicación; índice consultable ≈ {retraso_dias} días "
                             f"después; tono D+{PUB_TONO_DIAS}"),
        formato="json (url, título, dominio, idioma, país, seendate)",
        volumen_estimado="máx. 250 artículos por consulta y ventana; tono diario ≈ 3.500 puntos 2017-hoy",
        limites="1 petición / 5 s por IP (429 si se excede o la IP es compartida); máx. 250 por consulta",
        estado_sonda=base.estado(res.registro, F_DOC), mediciones=med_doc,
        notas=[f"Último dato del índice: {fin_doc} ({retraso_dias} días antes de la sonda)"
               + ("; las consultas de «hoy» no devuelven nada: para vivo, mejor el GKG (≈15 min)." if hoy_vacio else "."),
               "Sin cuerpo: solo titular + URL + seendate; el texto se busca en CC-NEWS por URL.",
               f"Sin cobertura en la muestra: {[d.clave for d in cfg.dias if d.fecha < INICIO_DOC]} (antes de 2017).",
               f"Serie 'gdelt_tono_economia': query={CONSULTA_TONO!r}, fecha_publicacion = D+{PUB_TONO_DIAS} (índice "
               f"DOC ≈{LAG_INDICE_DOC_DIAS} d por detrás + 1 d de margen); vintage único de hoy, recalculado."]))
    res.cobertura.append(dict(
        fuente=F_DOC, familia="gdelt_wiki", tipo="titular", inicio=ini_doc, fin=fin_doc, hora=True,
        idiomas="en (sourcelang:english; la API cubre ≈65 idiomas)", lag_mediana_h=None, destino="cajon", backtest=True,
        estado=base.estado(res.registro, F_DOC),
        nota=f"{med_doc['articulos_por_consulta']} titulares/consulta-día; tono diario {med_doc['puntos_tono']} puntos"))
