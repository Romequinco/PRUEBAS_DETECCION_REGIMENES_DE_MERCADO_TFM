"""Redes sociales y señales alternativas en vivo: Truth Social (Trump), FINRA short volume y las cerradas.

Lo abierto y con fecha fiable:

- **Truth Social de Donald Trump**: archivo público comunitario en JSON (GitHub). Cada post trae
  ``created_at`` exacto (UTC): es a la vez el hecho y su publicación. Serie de posts por día.
- **FINRA Reg SHO short sale volume**: un fichero diario consolidado (``CNMSshvol<AAAAMMDD>.txt``,
  ``Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market``) que se publica el mismo día
  ≈18:00 ET, tras el cierre: en la serie, ``fecha_publicacion`` = D+1 hábil. Muestra: un fichero por
  mes + los días testigo -> ratio short/total del mercado.

Cerradas o bloqueadas (ficha documentada, sin rodeos): Reddit (OAuth), X (de pago), Google
Trends (sin API pública; reescala 0-100 en cada descarga -> riesgo de look-ahead), AAII y CNN
Fear & Greed (anti-bot: se comprueba con UNA petición educada y se registra).

Caché: crudos en ``<corpus>/crudo/social/`` (Truth) y ``<corpus>/crudo/finra/`` (con la cabecera
``Last-Modified`` al lado, que mide el retraso real de publicación).
"""
from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ingesta.comun.esquema import Documento, doc_id
from ingesta.comun.http import get, robots_permite
from ingesta.noticias_alt import base
from ingesta.noticias_alt.limpieza import a_fecha, limpiar_html

MODULO = "social"
F_TRUTH = "truth_social_trump"
F_FINRA = "finra_short_volume"
F_REDDIT = "reddit"
F_X = "x_twitter"
F_TRENDS = "google_trends"
F_AAII = "aaii_sentiment"
F_CNN = "cnn_fear_greed"

TRUTH_ARCHIVO = "https://raw.githubusercontent.com/stiles/trump-truth-social-archive/main/data/truth_archive.json"
FINRA = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d:%Y%m%d}.txt"
FINRA_DESDE = date(2019, 1, 1)          # muestra mensual (el CDN llega algo antes: se mide la frontera)
AAII = "https://www.aaii.com/files/surveys/sentiment.xls"
CNN = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
REDDIT = "https://www.reddit.com/r/wallstreetbets/new.json"
TRENDS = "https://trends.google.com/trends/api/explore"
SIN_COB = "sin cobertura"               # prefijo de error que no cuenta como fallo de la fuente
ORIGINAL_MAX_H = 72                     # Last-Modified más tardío = fichero re-subido (no mide el lag)


def _http_status(e: Exception) -> int | None:
    """Código HTTP de un error de ``get`` (re-lanza sin ``response``: se saca del mensaje)."""
    resp = getattr(e, "response", None)
    if resp is not None and getattr(resp, "status_code", None):
        return int(resp.status_code)
    m = re.match(r"\s*(\d{3})\b", str(e))
    return int(m.group(1)) if m else None


def _estado(registro: list[dict], fuente: str) -> str:
    """``base.estado`` sin contar las peticiones fuera de cobertura (días sin sesión, antes del CDN)."""
    return base.estado([r for r in registro if not r["error"].startswith(SIN_COB)], fuente)


# =========================================================================== Truth Social

def leer_truth(*, usar_cache: bool = True, cache: Path | None = None) -> tuple[list[dict[str, Any]], dict]:
    """Posts del archivo (lista de dicts) e info de la descarga. Lanza ``PermissionError`` si robots.txt
    no lo permite y lo que lance ``get``."""
    cache = cache or base.corpus("crudo", MODULO) / "truth_archive.json"
    if usar_cache and cache.exists():
        datos = cache.read_bytes()
        info = dict(cache=True, bytes=len(datos), captura=datetime.fromtimestamp(cache.stat().st_mtime, tz=UTC))
    else:
        if not robots_permite(TRUTH_ARCHIVO):
            raise PermissionError("robots.txt")
        r = get(TRUTH_ARCHIVO, timeout=180, reintentos=2)
        r.raise_for_status()          # ``get`` ya lanza; por si se sustituye (tests)
        datos = r.content
        base.escribir_atomico(cache, datos)
        info = dict(cache=False, bytes=len(datos), captura=datetime.now(UTC))
    posts = json.loads(datos)
    if not isinstance(posts, list):
        raise ValueError("el archivo no es una lista de posts")
    return posts, info


def tabla_posts(posts: list[dict[str, Any]]) -> pd.DataFrame:
    """Posts -> DataFrame (id, created_at UTC, url, vacio). Sin texto: el texto solo va al corpus."""
    df = pd.DataFrame([dict(id=str(p.get("id", "")), created_at=p.get("created_at"), url=p.get("url", ""),
                            vacio=not limpiar_html(p.get("content"))) for p in posts])
    if df.empty:
        return pd.DataFrame(columns=["id", "created_at", "url", "vacio"])
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    return df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)


def posts_por_dia(df: pd.DataFrame) -> pd.Series:
    """Posts por día UTC, con ceros en los días sin posts entre el primero y el último."""
    if df.empty:
        return pd.Series(dtype=int)
    n = df.groupby(df["created_at"].dt.tz_localize(None).dt.normalize()).size()
    return n.reindex(pd.date_range(n.index.min(), n.index.max(), freq="D"), fill_value=0)


def serie_truth(por_dia: pd.Series) -> pd.DataFrame:
    """Serie larga ``truth_posts_dia``: el recuento del día UTC D solo se cierra a las 24:00 UTC (después del
    cierre de NY), así que se publica en D+1 (``disponible_utc`` = D+1 00:00 UTC)."""
    siguiente = por_dia.index + pd.Timedelta(days=1)
    return pd.DataFrame(dict(serie="truth_posts_dia", fecha_referencia=por_dia.index, fecha_publicacion=siguiente,
                             valor=por_dia.values.astype(float), unidad="posts/día", fuente=F_TRUTH,
                             disponible_utc=[f"{d.date()}T00:00:00Z" for d in siguiente],
                             metodo_publicacion="día UTC cerrado -> D+1",
                             nota="días sin posts = 0 (o hueco del archivo)"))


def _sondear_truth(cfg: base.ConfigSonda, res: base.Resultado) -> None:
    t0 = base.cronometro()
    try:
        posts, info = leer_truth(usar_cache=cfg.usar_cache)
    except Exception as e:  # noqa: BLE001
        res.registrar(F_TRUTH, "truth_archive.json", TRUTH_ARCHIVO, False, segundos=base.segundos(t0),
                      error="robots.txt" if isinstance(e, PermissionError) else f"{type(e).__name__}: {e}")
        posts, info = [], {}
    else:
        res.registrar(F_TRUTH, "truth_archive.json" + (" (caché)" if info["cache"] else ""), TRUTH_ARCHIVO, True,
                      bytes=info["bytes"], segundos=base.segundos(t0))
    df = tabla_posts(posts)
    por_dia = posts_por_dia(df)
    captura = info.get("captura") or datetime.now(UTC)
    if not por_dia.empty:
        res.series.append(base.guardar_serie(serie_truth(por_dia), "truth_posts_dia", rapido=cfg.modo_rapido).stem)
    ini = df["created_at"].min() if not df.empty else None
    fin = df["created_at"].max() if not df.empty else None
    retraso_archivo_h = base.lag_horas(fin, captura) if fin is not None else None
    hueco = int((por_dia == 0).astype(int).groupby((por_dia != 0).cumsum()).sum().max()) if not por_dia.empty else 0
    res.tablas["truth_resumen"] = pd.DataFrame([dict(
        inicio=ini.isoformat() if ini is not None else None, fin=fin.isoformat() if fin is not None else None,
        n_posts=len(df), n_posts_vacios=int(df["vacio"].sum()) if not df.empty else 0,
        dias=len(por_dia), dias_con_posts=int((por_dia > 0).sum()),
        posts_dia_media=round(float(por_dia.mean()), 2) if not por_dia.empty else None,
        posts_dia_p90=float(por_dia.quantile(.9)) if not por_dia.empty else None,
        max_dias_sin_posts=hueco, ultimo=fin.isoformat() if fin is not None else None,
        captura=captura.isoformat(), horas_desde_ultimo=retraso_archivo_h, bytes=info.get("bytes", 0))])

    # días testigo: posts de ese día (UTC); los textos solo al corpus
    filas, docs = [], []
    por_id = {str(p.get("id")): p for p in posts}
    for d in cfg.dias:
        sel = df[df["created_at"].dt.date == d.fecha] if not df.empty else df
        cubre = ini is not None and ini.date() <= d.fecha <= fin.date()
        filas.append(dict(dia=d.clave, fecha=d.fecha.isoformat(), cobertura=cubre, n_posts=len(sel)))
        if not cubre:
            continue
        for _, fila in sel.head(cfg.tope or len(sel)).iterrows():
            p = por_id.get(fila["id"], {})
            texto = limpiar_html(p.get("content"))
            if not texto:
                continue
            creado = fila["created_at"].to_pydatetime()
            doc = Documento(doc_id=doc_id(f"truth:{fila['id']}"), fuente=F_TRUTH, tipo_documento="post",
                            titulo=texto[:120], fecha_publicacion=creado.date(), url=fila["url"], agente="comun",
                            rag="cajon", institucion="Truth Social (@realDonaldTrump)", idioma="en", paises=["US"],
                            licencia="post público de un cargo político; archivo comunitario (MIT); solo corpus local",
                            extra=dict(hora_publicacion=creado.isoformat(), campo_fecha="created_at", dia_testigo=d.clave))
            docs.append(base.documento_dict(doc, texto=texto))
            if len([t for t in res.timegate if t["fuente"] == F_TRUTH]) < 5:
                res.timegate.append(dict(fuente=F_TRUTH, item=f"post {fila['id']} ({d.clave})",
                                         fecha_hecho=creado.isoformat(), fecha_publicacion=creado.isoformat(),
                                         campo_fecha="created_at", lag_h=0.0))
    res.tablas["truth_dias_testigo"] = pd.DataFrame(filas)
    if fin is not None:
        res.timegate.append(dict(fuente=F_TRUTH, item="archivo comunitario: último post frente a la captura",
                                 fecha_hecho=fin.isoformat(), fecha_publicacion=captura.isoformat(),
                                 campo_fecha="created_at -> captura del archivo", lag_h=retraso_archivo_h))
    if docs:
        ruta, n = base.guardar_documentos(docs, "truth_posts_testigo", rapido=cfg.modo_rapido)
        res.documentos[ruta.stem] = n

    sin_cob = [f["dia"] for f in filas if not f["cobertura"]]
    est = base.estado(res.registro, F_TRUTH)
    viejo = retraso_archivo_h is not None and retraso_archivo_h > 24 * 7
    res.cobertura.append(dict(
        fuente=F_TRUTH, familia="vivo_redes", tipo="serie", inicio=ini.date().isoformat() if ini is not None else None,
        fin=fin.date().isoformat() if fin is not None else None, hora=True, idiomas="en", lag_mediana_h=0.0,
        destino="cajon", backtest=bool(len(df)), estado=est,
        nota=(f"{len(df)} posts; serie truth_posts_dia (posts/día UTC, publicada en D+1); textos de los días testigo al corpus. "
              + (f"Archivo sin actualizar: último post hace {retraso_archivo_h / 24:.0f} días. " if viejo else "")
              + (f"Días testigo sin cobertura: {', '.join(sin_cob)}." if sin_cob else ""))))
    res.fichas.append(base.ficha(
        fuente=F_TRUTH, nombre="Truth Social de Donald Trump (archivo público comunitario)", agentes=["macro", "risk"],
        rag="cajon", url="https://github.com/stiles/trump-truth-social-archive", acceso="descarga",
        licencia_y_terminos="archivo comunitario (código MIT); posts públicos de un cargo político; términos de Truth "
                            "Social. Al repo solo recuentos; el texto solo al corpus",
        cobertura_inicio=ini.date().isoformat() if ini is not None else None,
        cobertura_fin=fin.date().isoformat() if fin is not None else None, frecuencia="por evento (posts)",
        retraso_publicacion=("0 h: created_at es la publicación; el archivo se refresca cada pocas horas"
                             + (f" (hoy lleva {retraso_archivo_h / 24:.0f} días sin actualizarse)" if viejo else "")),
        formato="json (id, created_at, content, url, media, recuentos)",
        volumen_estimado=f"{len(df)} posts, {info.get('bytes', 0) / 1e6:.1f} MB (fichero único)",
        limites="sin clave; un fichero; robots.txt de raw.githubusercontent.com no existe (404 = permitido)",
        estado_sonda=est, mediciones=dict(
            posts=len(df), posts_vacios=int(df["vacio"].sum()) if not df.empty else 0,
            posts_dia_media=float(por_dia.mean()) if not por_dia.empty else None,
            max_dias_sin_posts=hueco, horas_desde_ultimo=retraso_archivo_h, bytes=info.get("bytes", 0),
            segundos=base.segundos(t0)),
        notas=["Marca temporal exacta: sirve para atribuir saltos de mercado o de Polymarket (aranceles 2025).",
               "Serie diaria en UTC: un post de las 17:00 ET cae en el mismo día UTC que el cierre; para decisiones "
               "al cierre, filtrar por hora (created_at) antes de agregar.",
               "Posts vacíos = solo imagen, vídeo o repost."]
        + (["El archivo comunitario ha dejado de actualizarse: para el vivo haría falta otra vía (no hay API pública)."]
           if viejo else [])))


# =========================================================================== FINRA short volume

class SinFichero(Exception):
    """El CDN no tiene fichero para ese día (403/404: día sin sesión, futuro o antes del inicio)."""


def url_finra(d: date) -> str:
    return FINRA.format(d=d)


def parse_finra(texto: str) -> pd.DataFrame:
    """Fichero diario -> DataFrame (Date, Symbol, ShortVolume, ShortExemptVolume, TotalVolume, Market).

    Ignora la última línea (recuento de filas) y cualquier línea mal formada.
    """
    lineas = [l for l in texto.splitlines() if l.strip()]
    if not lineas or not lineas[0].startswith("Date|"):
        raise ValueError("no es un fichero Reg SHO (sin cabecera Date|...)")
    cab = lineas[0].split("|")
    filas = [l.split("|") for l in lineas[1:]]
    df = pd.DataFrame([f for f in filas if len(f) == len(cab)], columns=cab)
    for c in ("ShortVolume", "ShortExemptVolume", "TotalVolume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["ShortVolume", "TotalVolume"])


def ratio_mercado(df: pd.DataFrame) -> dict[str, Any]:
    """Ratio agregado ShortVolume/TotalVolume del día (todos los valores)."""
    sv, tv = float(df["ShortVolume"].sum()), float(df["TotalVolume"].sum())
    return dict(n_tickers=int(df["Symbol"].nunique()), short_volume=sv, total_volume=tv,
                short_ratio=round(sv / tv, 5) if tv else None)


def descargar_finra(d: date, *, usar_cache: bool = True) -> tuple[bytes, dict[str, Any]]:
    """Bytes del fichero de ``d`` e info (``last_modified``, ``cache``). Lanza ``SinFichero`` si el CDN
    responde 403/404 (S3 da 403 a cualquier clave inexistente) y lo que lance ``get`` si falla."""
    carpeta = base.corpus("crudo", "finra")
    ruta, meta = carpeta / f"CNMSshvol{d:%Y%m%d}.txt", carpeta / f"CNMSshvol{d:%Y%m%d}.meta.json"
    faltan = carpeta / "_sin_fichero.json"
    vistos = set(json.loads(faltan.read_text(encoding="utf-8"))) if faltan.exists() else set()
    if usar_cache and ruta.exists():
        info = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        return ruta.read_bytes(), dict(info, cache=True)
    if usar_cache and d.isoformat() in vistos:
        raise SinFichero("403 (caché de ausencias)")
    try:
        r = get(url_finra(d), timeout=60, reintentos=1)
        r.raise_for_status()
    except Exception as e:
        if _http_status(e) in (403, 404):
            if d < date.today() - timedelta(days=3):          # hoy o ayer pueden aparecer más tarde
                faltan.write_text(json.dumps(sorted(vistos | {d.isoformat()})), encoding="utf-8")
            raise SinFichero(f"{_http_status(e)}") from None
        raise
    info = dict(last_modified=r.headers.get("Last-Modified"), bytes=len(r.content))
    base.escribir_atomico(ruta, r.content)
    meta.write_text(json.dumps(info), encoding="utf-8")
    return r.content, dict(info, cache=False)


def _hora_et(d: date, hora: int) -> pd.Timestamp:
    """``d`` a la ``hora`` de Nueva York, en UTC (respeta el horario de verano)."""
    return pd.Timestamp(datetime(d.year, d.month, d.day, hora), tz="America/New_York").tz_convert("UTC")


def _lag_finra(d: date, last_modified: str | None) -> tuple[str, str | None, float | None]:
    """Cierre del día (16:00 ET, en UTC), Last-Modified (UTC) y horas entre ambos."""
    cierre = _hora_et(d, 16)
    lm = a_fecha(last_modified) if last_modified else None
    return cierre.isoformat(), lm.isoformat() if lm else None, base.lag_horas(cierre.to_pydatetime(), lm)


def _meses(cfg: base.ConfigSonda, hoy: date) -> list[date]:
    meses = [m.date() for m in pd.date_range(FINRA_DESDE, hoy, freq="MS")]
    if cfg.modo_rapido and len(meses) > 3:
        meses = [meses[0], meses[len(meses) // 2], meses[-1]]
    return meses


def _candidatos(mes: date, hoy: date, n: int = 4) -> list[date]:
    """Primeros ``n`` días laborables del mes (el primero con fichero es el de la muestra)."""
    return [d.date() for d in pd.bdate_range(mes, periods=n) if d.date() <= hoy and d.month == mes.month]


class _Finra:
    """Estado de la sonda FINRA: intentos, filas de la muestra y TimeGate."""

    def __init__(self, cfg: base.ConfigSonda, res: base.Resultado):
        self.cfg, self.res, self.filas = cfg, res, []

    def probar(self, d: date, origen: str, *, motivo_ausencia: str) -> dict | None:
        """Descarga y mide ``d``. Si no hay fichero, registra ``sin cobertura`` (o error si ``motivo_ausencia``
        es vacío). Devuelve la fila de la muestra o None."""
        t0 = base.cronometro()
        item = f"{d.isoformat()} ({origen})"
        try:
            datos, info = descargar_finra(d, usar_cache=self.cfg.usar_cache)
            m = ratio_mercado(parse_finra(datos.decode("utf-8", errors="replace")))
        except SinFichero as e:
            error = f"{SIN_COB}: {e} ({motivo_ausencia})" if motivo_ausencia else f"SinFichero: {e}"
            self.res.registrar(F_FINRA, item, url_finra(d), False, dia=d.isoformat(),
                               segundos=base.segundos(t0), error=error)
            return None
        except Exception as e:  # noqa: BLE001
            self.res.registrar(F_FINRA, item, url_finra(d), False, dia=d.isoformat(),
                               segundos=base.segundos(t0), error=f"{type(e).__name__}: {e}")
            return None
        self.res.registrar(F_FINRA, item + (" (caché)" if info.get("cache") else ""), url_finra(d), True,
                           dia=d.isoformat(), bytes=info.get("bytes", len(datos)), segundos=base.segundos(t0))
        cierre, lm, lag = _lag_finra(d, info.get("last_modified"))
        fila = dict(fecha=d.isoformat(), origen=origen, disponible=True, **m, bytes=info.get("bytes", len(datos)),
                    last_modified=lm, lag_publicacion_h=lag,
                    re_subido=lag is not None and lag > ORIGINAL_MAX_H)
        self.filas.append(fila)
        if lag is not None:
            self.res.timegate.append(dict(fuente=F_FINRA, item=f"CNMSshvol{d:%Y%m%d}.txt", fecha_hecho=cierre,
                                          fecha_publicacion=lm, campo_fecha="Last-Modified (cabecera HTTP)", lag_h=lag))
        return fila

    def ausente(self, d: date, origen: str) -> None:
        self.filas.append(dict(fecha=d.isoformat(), origen=origen, disponible=False))

    def mes(self, mes: date, hoy: date, origen: str = "mensual") -> dict | None:
        """Primer día con fichero del mes; los anteriores fallidos son días sin sesión."""
        cands = _candidatos(mes, hoy)
        for i, d in enumerate(cands):
            motivo = ("aún no publicado (≈18:00 ET)" if d >= hoy - timedelta(days=1)
                      else "" if i == len(cands) - 1 else "día sin sesión")
            fila = self.probar(d, origen, motivo_ausencia=motivo)
            if fila:
                return fila
        return None


def _sondear_finra(cfg: base.ConfigSonda, res: base.Resultado) -> None:
    t0 = base.cronometro()
    hoy = date.today()
    fx = _Finra(cfg, res)
    for mes in _meses(cfg, hoy):
        fx.mes(mes, hoy)

    # frontera hacia atrás (solo en modo completo): meses previos a FINRA_DESDE hasta el primero sin fichero
    frontera: dict[str, Any] = {}
    if not cfg.modo_rapido:
        mes = (pd.Timestamp(FINRA_DESDE) - pd.offsets.MonthBegin(1)).date()
        primero_ok = None
        while mes.year >= 2009:
            fila = None
            for d in _candidatos(mes, hoy):
                fila = fx.probar(d, "mensual", motivo_ausencia="antes del inicio del CDN o día sin sesión")
                if fila:
                    break
            if not fila:
                break
            primero_ok = date.fromisoformat(fila["fecha"])
            mes = (pd.Timestamp(mes) - pd.offsets.MonthBegin(1)).date()
        if primero_ok:   # bisección por días laborables entre el mes vacío y el primer día con fichero
            dias = [d.date() for d in pd.bdate_range(mes, primero_ok)][:-1]
            lo, hi = 0, len(dias)   # invariante: dias[hi:] tienen fichero (o son el límite)
            while lo < hi:
                mid = (lo + hi) // 2
                if fx.probar(dias[mid], "frontera", motivo_ausencia="antes del inicio del CDN"):
                    hi = mid
                else:
                    fx.ausente(dias[mid], "frontera")
                    lo = mid + 1
            frontera = dict(primer_dia=(dias[hi] if hi < len(dias) else primero_ok).isoformat(),
                            ultimo_mes_sin_fichero=mes.isoformat()[:7])

    # último día hábil ya publicado (mide hasta dónde llega el CDN hacia delante)
    ayer = (pd.Timestamp(hoy) - pd.offsets.BDay(1)).date()
    if not any(f["fecha"] == ayer.isoformat() for f in fx.filas):
        fx.probar(ayer, "ultimo", motivo_ausencia="festivo o aún no publicado")

    # días testigo
    for d in cfg.dias:
        if d.fecha.weekday() >= 5:
            fx.ausente(d.fecha, f"testigo:{d.clave}")
            continue
        motivo = "aún no publicado (≈18:00 ET)" if d.fecha >= hoy - timedelta(days=1) else "fuera del CDN"
        if not fx.probar(d.fecha, f"testigo:{d.clave}", motivo_ausencia=motivo):
            fx.ausente(d.fecha, f"testigo:{d.clave}")

    muestra = pd.DataFrame(fx.filas, columns=[
        "fecha", "origen", "disponible", "n_tickers", "short_volume", "total_volume", "short_ratio", "bytes",
        "last_modified", "lag_publicacion_h", "re_subido"])
    muestra = muestra.sort_values(["fecha", "origen"]).reset_index(drop=True)
    res.tablas["finra_short_muestra"] = muestra
    ok = muestra[muestra["disponible"] == True]  # noqa: E712
    mensual = ok[ok["origen"] == "mensual"].drop_duplicates("fecha")
    if not mensual.empty:
        a_las_18 = [_hora_et(date.fromisoformat(f), 18).isoformat() for f in mensual["fecha"]]
        serie = pd.DataFrame(dict(
            serie="finra_short_ratio_mercado", fecha_referencia=pd.to_datetime(mensual["fecha"]).values,
            fecha_publicacion=(pd.to_datetime(mensual["fecha"]) + pd.offsets.BDay(1)).values,
            valor=mensual["short_ratio"].values,
            unidad="ratio (0-1)", fuente=F_FINRA,
            disponible_utc=[lm if isinstance(lm, str) and not re else c  # re-subido: la hora documentada
                            for lm, re, c in zip(mensual["last_modified"], mensual["re_subido"].astype(bool), a_las_18)],
            metodo_publicacion="D+1 hábil (fichero ≈18:00 ET, tras el cierre; Last-Modified en disponible_utc)",
            nota="primer día con fichero de cada mes; ShortVolume/TotalVolume agregado"))
        res.series.append(base.guardar_serie(serie, "finra_short_ratio_mercado", rapido=cfg.modo_rapido).stem)

    originales = ok[~ok["re_subido"].astype(bool)]["lag_publicacion_h"].dropna().tolist() if not ok.empty else []
    lag = base.resumen_lag(originales)
    ini = ok["fecha"].min() if not ok.empty else None
    fin = ok["fecha"].max() if not ok.empty else None
    est = _estado(res.registro, F_FINRA)
    sin_cob = [r["origen"].split(":")[1] for r in fx.filas if not r["disponible"] and r["origen"].startswith("testigo")]
    mb_dia = ok["bytes"].median() / 1e6 if not ok.empty else 0
    n_dias = len(pd.bdate_range(ini, hoy)) if ini else 0
    res.cobertura.append(dict(
        fuente=F_FINRA, familia="indices_ews", tipo="serie", inicio=ini, fin=fin, hora=False, idiomas="",
        lag_mediana_h=lag["mediana_h"], destino="serie", backtest=bool(ini), estado=est,
        nota=(f"{len(ok)} ficheros; ratio short/total medio {ok['short_ratio'].mean():.3f}; " if not ok.empty else "")
        + (f"primer día del CDN {frontera['primer_dia']}; " if frontera else "")
        + "lag = cierre 16:00 ET -> Last-Modified (solo ficheros no re-subidos)."
        + (f" Días testigo sin cobertura: {', '.join(sin_cob)}." if sin_cob else "")))
    res.fichas.append(base.ficha(
        fuente=F_FINRA, nombre="FINRA Reg SHO · volumen diario en corto (fichero consolidado CNMS)",
        agentes=["equity", "risk"], rag="serie",
        url="https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files",
        acceso="descarga", licencia_y_terminos="FINRA Data: uso libre no comercial con atribución",
        cobertura_inicio=frontera.get("primer_dia") or ini, cobertura_fin=fin, frecuencia="diaria",
        retraso_publicacion=(f"mismo día: mediana {lag['mediana_h']} h tras el cierre (Last-Modified, n={lag['n']}); "
                             "usable en D+1 hábil" if lag["n"] else "mismo día ≈18:00 ET (documentado); usable en D+1"),
        formato="txt separado por | (Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market)",
        volumen_estimado=f"≈{mb_dia:.2f} MB/día x {n_dias} días hábiles ≈ {mb_dia * n_dias / 1000:.1f} GB desde {ini}",
        limites="sin clave; 403 para cualquier día sin fichero (S3)", estado_sonda=est,
        mediciones=dict(ficheros=len(ok), peticiones=len([r for r in res.registro if r["fuente"] == F_FINRA]),
                        frontera=frontera, lag_publicacion_h=lag,
                        re_subidos=int(ok["re_subido"].astype(bool).sum()) if not ok.empty else 0,
                        short_ratio_min=float(ok["short_ratio"].min()) if not ok.empty else None,
                        short_ratio_max=float(ok["short_ratio"].max()) if not ok.empty else None,
                        tickers_mediana=float(ok["n_tickers"].median()) if not ok.empty else None,
                        segundos=base.segundos(t0)),
        notas=["Solo volumen en corto del día en los TRF/ADF/ORF de FINRA (no el interés corto ni todo el mercado); "
               "el ratio short/total es presión vendedora relativa.",
               "robots.txt de cdn.finra.org no existe (S3 responde 403 a toda clave inexistente, también a robots.txt): "
               "no hay rastreo, es el enlace de descarga oficial que publica FINRA.",
               "Antes de la frontera del CDN (2018) hay que ir a la API de FINRA (cuenta gratuita) o al catálogo; "
               "los días testigo de 2008 y 2016 quedan sin cobertura por esta vía.",
               "Los ficheros antiguos fueron re-subidos (Last-Modified de 2021): su cabecera no mide el lag original."]))


# =========================================================================== cerradas o bloqueadas

def _peticion_educada(res: base.Resultado, fuente: str, url: str) -> dict[str, Any]:
    """UNA petición (robots.txt primero) para documentar el bloqueo; no se insiste ni se esquiva."""
    t0 = base.cronometro()
    if not robots_permite(url):
        # robots.txt prohíbe la ruta, o el propio /robots.txt responde 401/403 (anti-bot): no se pide el fichero
        res.registrar(fuente, "comprobacion", url, False, segundos=base.segundos(t0),
                      error="robots.txt (prohíbe la ruta o responde 401/403: anti-bot)")
        return dict(robots_permite=False, error="robots.txt prohíbe o bloquea (401/403)")
    try:
        r = get(url, timeout=30, reintentos=1)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        res.registrar(fuente, "comprobacion", url, False, segundos=base.segundos(t0), error=f"{type(e).__name__}: {e}")
        return dict(robots_permite=True, http=_http_status(e), error=f"{type(e).__name__}: {str(e)[:120]}")
    res.registrar(fuente, "comprobacion", url, True, bytes=len(r.content), segundos=base.segundos(t0))
    return dict(robots_permite=True, http=200, bytes=len(r.content))


def _fila_cerrada(res: base.Resultado, fuente: str, familia: str, tipo: str, estado: str, nota: str) -> None:
    res.cobertura.append(dict(fuente=fuente, familia=familia, tipo=tipo, inicio=None, fin=None, hora=False,
                              idiomas="en" if familia == "vivo_redes" else "", lag_mediana_h=None, destino="ninguno",
                              backtest=False, estado=estado, nota=nota))


def _sondear_cerradas(cfg: base.ConfigSonda, res: base.Resultado) -> None:
    # Reddit: solo robots.txt (sin OAuth no se consulta la API)
    rob = robots_permite(REDDIT)
    _fila_cerrada(res, F_REDDIT, "vivo_redes", "texto", "no_probada",
                  f"OAuth obligatorio (Data API 2023); robots.txt {'permite' if rob else 'prohíbe'} el .json público.")
    res.fichas.append(base.ficha(
        fuente=F_REDDIT, nombre="Reddit (r/wallstreetbets, r/investing...)", agentes=["equity", "risk"], rag="cajon",
        url="https://www.reddit.com/dev/api", autenticacion="clave gratuita (OAuth, no comercial ≤100 req/min) / pago",
        licencia_y_terminos="Reddit Data API Terms (2023): prohíbe entrenar modelos sin acuerdo; el histórico solo "
                            "en dumps comunitarios (Arctic Shift, Academic Torrents) de licencia dudosa",
        cobertura_inicio=None, cobertura_fin=None, frecuencia="continua", retraso_publicacion="tiempo real",
        formato="json", volumen_estimado="WSB 2019-2024 en dumps ≈ decenas de GB", limites="OAuth; 100 req/min",
        estado_sonda="no_probada", mediciones=dict(robots_permite_json=rob),
        notas=["No probada: sin OAuth no se consulta. Cómo: app «script» en reddit.com/prefs/apps (gratis, uso personal).",
               "Valor: menciones de tickers (meme stocks 2021). Riesgo legal medio-alto para el histórico."]))

    _fila_cerrada(res, F_X, "vivo_redes", "texto", "no_probada", "API de pago (0,005 USD por post leído).")
    res.fichas.append(base.ficha(
        fuente=F_X, nombre="X (Twitter) API", agentes=["equity", "risk"], rag="cajon", url="https://docs.x.com/x-api",
        autenticacion="pago", coste="pago por uso: 0,005 USD por post leído (tope 3 M lecturas/mes); 100 k posts ≈ "
                                    "500 USD; histórico masivo solo Enterprise (consultado 2026-10-06)",
        licencia_y_terminos="X Developer Agreement: prohíbe redistribuir y entrenar modelos", cobertura_inicio=None,
        cobertura_fin=None, frecuencia="continua", retraso_publicacion="tiempo real", formato="json",
        volumen_estimado="n/a", limites="según plan", estado_sonda="no_probada",
        notas=["No probada: de pago. Sustitutos: Truth Social (Trump), GDELT, visitas de Wikipedia, Polymarket."]))

    rob_t = robots_permite(TRENDS)
    _fila_cerrada(res, F_TRENDS, "vivo_redes", "serie", "no_probada",
                  "Sin API pública; reescala 0-100 en cada descarga (look-ahead si se baja entera).")
    res.fichas.append(base.ficha(
        fuente=F_TRENDS, nombre="Google Trends", agentes=["macro", "equity"], rag="serie",
        url="https://trends.google.com", acceso="manual", licencia_y_terminos="Términos de Google; API alfa oficial "
                                                                               "solo por solicitud",
        cobertura_inicio=None, cobertura_fin=None, frecuencia="semanal (diaria en ventanas < 9 meses)",
        retraso_publicacion="≈1-3 días", formato="csv manual / json interno",
        volumen_estimado="pequeño (series de 0-100)", limites="429 frecuentes; endpoints internos (pytrends) no oficiales",
        estado_sonda="no_probada", mediciones=dict(robots_permite_api=rob_t),
        notas=["No probada: sin API pública; los endpoints internos (pytrends) no son oficiales y dan 429.",
               "Riesgo de look-ahead: cada consulta se reescala a 0-100 sobre la ventana pedida y la muestra cambia "
               "entre descargas; bajar la serie entera hoy «mira el futuro». Solución: ventanas que terminan en la "
               "fecha de decisión, o visitas de Wikipedia (absolutas)."]))

    med = _peticion_educada(res, F_AAII, AAII)
    est = base.estado(res.registro, F_AAII)
    _fila_cerrada(res, F_AAII, "indices_ews", "serie", est,
                  f"Comprobación educada: {med.get('error') or med.get('http')}. No se esquiva el bloqueo.")
    res.fichas.append(base.ficha(
        fuente=F_AAII, nombre="AAII Investor Sentiment Survey (semanal, desde 1987)", agentes=["equity", "risk"],
        rag="serie", url="https://www.aaii.com/sentimentsurvey", acceso="descarga",
        autenticacion="socio (descarga)", licencia_y_terminos="AAII: datos para socios; redistribución no permitida",
        cobertura_inicio=None, cobertura_fin=None, frecuencia="semanal (jueves)", retraso_publicacion="mismo día",
        formato="xls", volumen_estimado="< 1 MB", limites="403 al robot (protección anti-bot / socios)",
        estado_sonda=est, mediciones=med,
        notas=["Bloqueado al robot: no se esquiva. Opciones: descarga manual por un socio o Nasdaq Data Link (pago).",
               "Documentado: serie semanal desde 1987-07-24.",
               "Alternativa abierta: NAAIM Exposure Index (csv público semanal)."]))

    med = _peticion_educada(res, F_CNN, CNN)
    est = base.estado(res.registro, F_CNN)
    _fila_cerrada(res, F_CNN, "indices_ews", "serie", est,
                  f"Comprobación educada: {med.get('error') or med.get('http')}. Mejor reconstruirlo con datos propios.")
    res.fichas.append(base.ficha(
        fuente=F_CNN, nombre="CNN Fear & Greed Index", agentes=["equity", "risk"], rag="serie",
        url="https://edition.cnn.com/markets/fear-and-greed", acceso="api",
        licencia_y_terminos="endpoint interno de CNN sin términos de uso de datos", cobertura_inicio=None,
        cobertura_fin=None, frecuencia="diaria", retraso_publicacion="intradía", formato="json",
        volumen_estimado="pequeño", limites="anti-bot (418 «You're a bot») sin cabeceras de navegador",
        estado_sonda=est, mediciones=med,
        notas=["No se suplantan cabeceras de navegador.",
               "Es una combinación de 7 series que ya hay en el repo (VIX, put/call, momentum, máximos-mínimos, spreads, "
               "refugio, amplitud): mejor RECONSTRUIRLO con datos propios y su propia fecha de publicación."]))


# =========================================================================== sonda

def sondear(cfg: base.ConfigSonda) -> base.Resultado:
    """Truth Social + FINRA short volume + fichas de las redes cerradas o bloqueadas."""
    res = base.Resultado(MODULO)
    for nombre, fuente, paso in (("truth", F_TRUTH, _sondear_truth), ("finra", F_FINRA, _sondear_finra),
                                 ("cerradas", F_AAII, _sondear_cerradas)):
        try:
            cfg.log(f"[social] {nombre}")
            paso(cfg, res)
        except Exception as e:  # noqa: BLE001
            res.registrar(fuente, f"sonda {nombre}", "", False, error=f"{type(e).__name__}: {e}")
            cfg.log(f"  [social] {nombre} falló: {type(e).__name__}: {e}")
    return res
