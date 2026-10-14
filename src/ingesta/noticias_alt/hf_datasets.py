"""Datasets de noticias con texto en Hugging Face: SOLO metadatos (no se descarga ningún dataset).

Se miden con la API del Hub (``/api/datasets/<id>?blobs=true``: licencia, *gated*, ficheros y bytes)
y con el *dataset viewer* (``datasets-server``: filas por config, primeras filas, estadísticas). Las
primeras filas traen texto: se guardan solo crudas en el corpus; a ``results`` van metadatos.

Ninguno se acepta a ciegas: licencia, procedencia del texto (muchos son *scrapes* de medios con
copyright) y sesgo de cobertura se anotan en la ficha. Los *gated* (hay que aceptar condiciones en
la web con cuenta) NO se tocan: solo sus metadatos públicos, y quedan en estado ``parcial``.

TimeGate: ``stanford-oval/ccnews`` trae ``published_date`` (día, declarado por el medio) y
``crawl_date`` (rastreo): misma regla que ``ccnews`` (la publicada solo si es ≤ rastreo).
FNSPID trae ``Date`` con hora UTC.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ingesta.comun.http import get
from ingesta.noticias_alt import base
from ingesta.noticias_alt.limpieza import a_fecha, dominio, tokens_aprox

HUB = "https://huggingface.co"
VIEWER = "https://datasets-server.huggingface.co"


@dataclass(frozen=True)
class DatasetHF:
    fuente: str
    dataset: str
    nombre: str
    inicio_doc: str            # cobertura documentada (paper / tarjeta); se sustituye si se mide
    fin_doc: str
    tipo: str = "texto"
    agentes: tuple[str, ...] = ("macro", "equity", "risk")
    backtest: bool = True
    hora: bool = False
    licencia_doc: str = ""
    viewer: bool = True        # intentar datasets-server (los gated devuelven 401)
    notas: tuple[str, ...] = field(default_factory=tuple)


DATASETS: tuple[DatasetHF, ...] = (
    DatasetHF("hf_fnspid", "Zihan1004/FNSPID", "FNSPID (Dong et al. 2024): noticias de empresas del S&P 500",
              "1999-01-01", "2023-12-31", agentes=("equity",), hora=True,
              licencia_doc="sin licencia en el Hub (paper: CC BY-NC 4.0); texto de Nasdaq.com/Benzinga… con copyright",
              notas=("15,7 M filas según el paper (el viewer no calcula el tamaño).",
                     "nasdaq_exteral_data.csv trae el cuerpo (Article) en la parte reciente; All_external.csv es "
                     "sobre todo titular + URL.",
                     "Sesgo: EE. UU. y tickers del S&P 500; poco texto antes de 2009 según el paper.")),
    DatasetHF("hf_ccnews", "stanford-oval/ccnews", "CC-NEWS limpio y deduplicado (stanford-oval/ccnews)",
              "2016-08-24", "2024-06-30",
              licencia_doc="Common Crawl Terms of Use; texto con copyright de cada medio: solo local",
              notas=("Configs por AÑO DE RASTREO (no de publicación): usar published_date ≤ crawl_date.",
                     "published_date es solo el día: el lag hasta crawl_date se mide desde las 00:00 UTC (cota superior).",
                     "Mejor relación coste/beneficio para 2016-2024: texto ya extraído, idioma y sitename; "
                     "filtrar por medio e idioma fichero a fichero (pyarrow por row group).")),
    DatasetHF("hf_bloomberg_2006_2013", "KrossKinetic/Bloomberg_Financial_News",
              "Bloomberg financial news 2006-2013 (copia en HF del corpus de Ding et al. 2014)",
              "2006-10-20", "2013-11-26", viewer=True,
              licencia_doc="declara apache-2.0 pero el texto es de Bloomberg (copyright): licencia NO fiable",
              notas=("Cubre 2008 con texto completo, que CC-NEWS no cubre: candidato para el backtest de "
                     "Lehman con advertencia legal explícita.",)),
    DatasetHF("hf_reuters_cnbc_2018", "ashraq/financial-news-articles",
              "Financial news articles (Reuters/CNBC, ene-may 2018)", "2018-01-01", "2018-05-31",
              backtest=False, licencia_doc="sin licencia declarada; texto de Reuters/CNBC con copyright",
              notas=("Sin columna de fecha: hay que sacarla de la URL o del texto.",
                     "Ventana de 5 meses: banco de pruebas del troceado, no corpus de backtest.")),
    DatasetHF("hf_financial_news_multisource", "Brianferrell787/financial-news-multisource",
              "Financial News Multisource (24 datasets agregados, 1990-2025)", "1990-01-01", "2025-11-29",
              viewer=False, licencia_doc="'other': investigación/no comercial, hereda términos de cada fuente; GATED",
              notas=("Gated: hay que aceptar condiciones con cuenta HF (no se hace automáticamente); "
                     "medidos solo metadatos públicos.",
                     "Agrega FNSPID, Reuters/Bloomberg antiguos, CC-NEWS…: atajo útil, pero con licencias y fechas "
                     "heterogéneas; validar subconjunto a subconjunto.")),
)


# --------------------------------------------------------------------------- red con caché

def _bajar_json(res: base.Resultado, cfg: base.ConfigSonda, fuente: str, item: str, url: str,
                params: dict | None = None) -> dict[str, Any]:
    clave = item.replace("/", "_")
    cache = base.corpus("crudo", "hf_datasets") / f"{fuente}__{clave}.json"
    if cfg.usar_cache and cache.exists():
        datos = cache.read_bytes()
        res.registrar(fuente, item, url, True, bytes=len(datos))
        return json.loads(datos)
    t0 = base.cronometro()
    try:
        r = get(url, params=params, timeout=90, reintentos=2)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        res.registrar(fuente, item, url, False, segundos=base.segundos(t0), error=str(e))
        raise
    base.escribir_atomico(cache, r.content)
    res.registrar(fuente, item, url, True, bytes=len(r.content), segundos=base.segundos(t0))
    return json.loads(r.content)


def info_hub(j: dict[str, Any]) -> dict[str, Any]:
    """Respuesta de ``/api/datasets/<id>?blobs=true`` -> metadatos útiles."""
    card = j.get("cardData") or {}
    sib = j.get("siblings") or []
    lic = card.get("license")
    configs = [c.get("config_name") for c in card.get("configs") or [] if isinstance(c, dict)]
    return dict(gated=j.get("gated") or False, licencia=",".join(lic) if isinstance(lic, list) else lic,
                ficheros=len(sib), bytes=sum(s.get("size") or 0 for s in sib), descargas_mes=j.get("downloads"),
                modificado=j.get("lastModified"), configs_tarjeta=[c for c in configs if c],
                extensiones=dict(Counter(s["rfilename"].rsplit(".", 1)[-1] for s in sib if "." in s["rfilename"])))


def info_size(j: dict[str, Any]) -> dict[str, Any]:
    """Respuesta de ``/size`` -> filas totales y por config (None si el viewer no lo calculó)."""
    tam = j.get("size") or {}
    ds = tam.get("dataset") or {}
    filas = ds.get("estimated_num_rows") or ds.get("num_rows") or None
    por_config = {c["config"]: c.get("estimated_num_rows") or c.get("num_rows")
                  for c in tam.get("configs") or [] if c.get("config") != "default"}
    return dict(filas=filas, por_config={k: v for k, v in por_config.items() if v},
                parquet_bytes=ds.get("num_bytes_parquet_files") or None)


def filas_primeras(j: dict[str, Any]) -> list[dict[str, Any]]:
    """Respuesta de ``/first-rows`` o ``/rows`` -> lista de filas."""
    return [r.get("row", {}) for r in j.get("rows") or []]


def idiomas_estadisticas(j: dict[str, Any], columna: str = "language") -> dict[str, int]:
    """Frecuencias de idioma de ``/statistics`` (sobre la parte del dataset que el viewer procesó)."""
    for s in j.get("statistics") or []:
        if s.get("column_name") == columna:
            return dict((s.get("column_statistics") or {}).get("frequencies") or {})
    return {}


def fecha_timegate_ccnews(published: Any, crawl: Any) -> tuple[Any, str, float | None]:
    """``published_date`` (día) si es ≤ ``crawl_date``; si no, la de rastreo. Devuelve (fecha, campo, lag_h)."""
    p, c = a_fecha(published), a_fecha(crawl)
    lag = base.lag_horas(p, c)
    if p is not None and (c is None or p <= c):
        return p, "published_date (≤ crawl_date)", lag
    return c, "crawl_date (published_date posterior)" if p else "crawl_date (sin published_date)", lag


# --------------------------------------------------------------------------- sonda

def _medir(res: base.Resultado, cfg: base.ConfigSonda, ds: DatasetHF) -> dict[str, Any]:
    fila: dict[str, Any] = dict(fuente=ds.fuente, dataset=ds.dataset, inicio=ds.inicio_doc, fin=ds.fin_doc,
                                fechas="documentadas", filas=None, gb=None, licencia=None, gated=None, ficheros=None,
                                descargas_mes=None, modificado=None, viewer=False, columnas="", tokens_mediana=None,
                                dominios_top="")
    hub = info_hub(_bajar_json(res, cfg, ds.fuente, "hub", f"{HUB}/api/datasets/{ds.dataset}", {"blobs": "true"}))
    fila.update(gb=round(hub["bytes"] / 1e9, 2), licencia=hub["licencia"], gated=hub["gated"],
                ficheros=hub["ficheros"], descargas_mes=hub["descargas_mes"], modificado=hub["modificado"])
    fila["_hub"] = hub
    if not ds.viewer or hub["gated"]:
        return fila
    q = {"dataset": ds.dataset}
    try:
        tam = info_size(_bajar_json(res, cfg, ds.fuente, "size", f"{VIEWER}/size", q))
        fila["filas"] = tam["filas"]
        fila["_por_config"] = tam["por_config"]
        anios = sorted(k for k in tam["por_config"] if k.isdigit())
        if anios:   # configs por año (ccnews): extremos medidos del año
            fila.update(inicio=max(ds.inicio_doc, f"{anios[0]}-01-01"), fin=min(ds.fin_doc, f"{anios[-1]}-12-31"),
                        fechas="medidas (configs por año) y acotadas por la tarjeta")
    except Exception as e:  # noqa: BLE001
        fila["_error_size"] = str(e)[:120]
    config = "2020" if ds.fuente == "hf_ccnews" else "default"
    try:
        filas = filas_primeras(_bajar_json(res, cfg, ds.fuente, f"first-rows_{config}", f"{VIEWER}/first-rows",
                                           dict(q, config=config, split="train")))
        if cfg.tope:
            filas = filas[: max(cfg.tope, 10)]
        fila["viewer"] = True
        fila["_filas"] = filas
        if filas:
            fila["columnas"] = ",".join(filas[0].keys())
            col_txt = next((c for c in ("plain_text", "Article", "text") if c in filas[0]), None)
            if col_txt:
                ntok = [tokens_aprox(str(f.get(col_txt) or "")) for f in filas]
                ntok = [n for n in ntok if n > 5]
                fila["tokens_mediana"] = int(pd.Series(ntok).median()) if ntok else None
            col_url = next((c for c in ("requested_url", "Url", "url") if c in filas[0]), None)
            if col_url:
                fila["dominios_top"] = ",".join(k for k, _ in Counter(dominio(f.get(col_url)) for f in filas)
                                                .most_common(5))
    except Exception as e:  # noqa: BLE001
        fila["_error_rows"] = str(e)[:120]
    if ds.fuente == "hf_ccnews":
        try:
            st = _bajar_json(res, cfg, ds.fuente, "statistics_2020", f"{VIEWER}/statistics",
                             dict(q, config="2020", split="train"))
            fila["_idiomas"] = idiomas_estadisticas(st)
        except Exception as e:  # noqa: BLE001
            fila["_error_stats"] = str(e)[:120]
    return fila


def sondear(cfg: base.ConfigSonda) -> base.Resultado:
    res = base.Resultado("hf_datasets")
    medidas: list[dict[str, Any]] = []
    for ds in DATASETS:
        try:
            medidas.append(_medir(res, cfg, ds))
            cfg.log(f"[hf] {ds.dataset}: {medidas[-1]['gb']} GB, filas={medidas[-1]['filas']}")
        except Exception as e:  # noqa: BLE001
            medidas.append(dict(fuente=ds.fuente, dataset=ds.dataset, _error=f"{type(e).__name__}: {str(e)[:150]}"))

    por_anio, idiomas_tab = [], []
    for ds, m in zip(DATASETS, medidas):
        for k, v in (m.get("_por_config") or {}).items():
            if k.isdigit():
                por_anio.append(dict(dataset=ds.dataset, fuente=ds.fuente, anio=int(k), filas=int(v),
                                     base="config (año de rastreo)"))
        tot = sum((m.get("_idiomas") or {}).values())
        for k, v in sorted((m.get("_idiomas") or {}).items(), key=lambda kv: -kv[1]):
            idiomas_tab.append(dict(dataset=ds.dataset, config="2020", idioma=k, filas=int(v),
                                    pct=round(100 * v / tot, 2)))
        _timegate(res, ds, m.get("_filas") or [])
        _cerrar_fuente(res, ds, m)

    tabla = pd.DataFrame([{k: v for k, v in m.items() if not k.startswith("_")} for m in medidas])
    res.tablas["hf_datasets"] = tabla
    res.tablas["hf_filas_anio"] = pd.DataFrame(por_anio, columns=["dataset", "fuente", "anio", "filas", "base"])
    if idiomas_tab:
        res.tablas["hf_ccnews_idiomas"] = pd.DataFrame(idiomas_tab)
    return res


def _timegate(res: base.Resultado, ds: DatasetHF, filas: list[dict[str, Any]]) -> None:
    for f in filas[: 15 if ds.fuente == "hf_ccnews" else 5]:
        if ds.fuente == "hf_ccnews":
            pub, campo, lag = fecha_timegate_ccnews(f.get("published_date"), f.get("crawl_date"))
            hecho = a_fecha(f.get("published_date"))
            item = f.get("sitename") or dominio(f.get("requested_url"))
        elif ds.fuente == "hf_fnspid":
            pub, campo, lag, hecho = a_fecha(f.get("Date")), "Date (UTC, con hora)", None, None
            item = f.get("Stock_symbol") or ""
        else:
            continue
        res.timegate.append(dict(fuente=ds.fuente, item=str(item)[:60],
                                 fecha_hecho=hecho.isoformat() if hecho else None,
                                 fecha_publicacion=pub.isoformat() if pub else None, campo_fecha=campo, lag_h=lag))


def _cerrar_fuente(res: base.Resultado, ds: DatasetHF, m: dict[str, Any]) -> None:
    est = base.estado(res.registro, ds.fuente)
    motivo = []
    if m.get("gated"):
        motivo.append("gated: solo metadatos públicos")
    if est == "ok" and m.get("filas") is None:
        motivo.append("el viewer no da filas: tamaño en filas documentado, no medido")
    if est == "ok" and motivo:
        est = "parcial"
    lags = base.resumen_lag(t["lag_h"] for t in res.timegate if t["fuente"] == ds.fuente)
    idiomas = [k for k, _ in sorted((m.get("_idiomas") or {}).items(), key=lambda kv: -kv[1])[:8]]
    if not idiomas:
        idiomas = ["en"]
    hub = m.get("_hub") or {}
    med = dict(hub=hub, filas=m.get("filas"), filas_por_config=m.get("_por_config"), fechas=m.get("fechas"),
               columnas=m.get("columnas"), tokens_mediana=m.get("tokens_mediana"), dominios_top=m.get("dominios_top"),
               lag_published_a_crawl_h=lags if lags["n"] else None,
               errores={k: v for k, v in m.items() if k.startswith("_error")})
    nota = (f"{m.get('gb')} GB, filas={m.get('filas')}, fechas {m.get('fechas', 'no medidas')}"
            + (f"; {'; '.join(motivo)}" if motivo else "") + (f"; error: {m['_error']}" if m.get("_error") else ""))
    res.cobertura.append(dict(fuente=ds.fuente, familia="texto_historico", tipo=ds.tipo, inicio=m.get("inicio"),
                              fin=m.get("fin"), hora=ds.hora, idiomas=",".join(idiomas),
                              lag_mediana_h=lags["mediana_h"], destino="cajon", backtest=ds.backtest, estado=est,
                              nota=nota))
    gated = bool(m.get("gated"))
    res.fichas.append(base.ficha(
        fuente=ds.fuente, nombre=ds.nombre, agentes=list(ds.agentes), rag="cajon", url=f"{HUB}/datasets/{ds.dataset}",
        acceso="descarga", autenticacion="cuenta HF + aceptar condiciones" if gated else "ninguna",
        licencia_y_terminos=f"declarada en el Hub: {m.get('licencia')}; {ds.licencia_doc}",
        cobertura_inicio=m.get("inicio"), cobertura_fin=m.get("fin"), frecuencia="estático (instantánea)",
        retraso_publicacion=(f"rastreo: mediana {lags['mediana_h']} h tras published_date" if lags["n"]
                             else "n/a (histórico cerrado)"),
        formato=",".join(hub.get("extensiones", {}).keys()) or "desconocido",
        volumen_estimado=f"{m.get('gb')} GB en {m.get('ficheros')} ficheros; filas={m.get('filas') or 'no medidas'}",
        limites="descarga HTTP del Hub (sin clave si no es gated); viewer: 100 filas por petición",
        estado_sonda=est, mediciones=med, notas=list(ds.notas) + motivo))
