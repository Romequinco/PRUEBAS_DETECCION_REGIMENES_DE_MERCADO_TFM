"""Reglas de limpieza del Tema 4: fechas, HTML, RSS, duplicados, sindicación e idioma.

Funciones puras (sin red) que usan todas las sondas de texto y que reutilizarán las tools
``buscar_noticias`` y el llenado del cajón:

- ``a_fecha`` / ``a_utc`` / ``fecha_rss``: fechas heterogéneas (epoch, ISO 8601, RFC 822, ``AAAAMMDDhhmmss``) -> UTC.
- ``limpiar_html`` y ``parse_rss`` (RSS 2.0 / Atom sin ``feedparser``).
- Duplicados en tres etapas: URL canónica (``clave_url``), titular casi igual (``similares``, Jaccard
  ≥ 0,75) y cuerpo casi igual (``simhash`` de 64 bits, Hamming ≤ 3). ``deduplicar`` conserva la
  versión MÁS ANTIGUA (la que respeta el TimeGate) y anota los otros medios en
  ``extra['sindicada_en']``; ``embudo_duplicados`` cuenta lo que queda tras cada etapa.
- ``idioma``: detector ligero por palabras vacías (en, es, fr, de, it, pt); ``und`` si duda.
- ``tokens_aprox`` y ``trocear``: tamaño y troceado corto (misma convención que la Fase 2).

Los documentos son diccionarios: valen las entradas de ``parse_rss`` (``titulo, url, fecha, medio,
resumen``) y los ``Documento`` serializados (``titulo, url, fecha_publicacion, institucion, texto``).
"""
from __future__ import annotations

import email.utils
import hashlib
import html as _html
import re
from datetime import UTC, date, datetime
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd

UMBRAL_JACCARD = 0.75
UMBRAL_HAMMING = 3
MIN_PALABRAS_SIMHASH = 30      # por debajo, el simhash de un cuerpo corto da falsos positivos

# --------------------------------------------------------------------------- fechas


def fecha_rss(txt: str | None) -> datetime | None:
    """Fecha RFC 822 de RSS (``Mon, 16 Mar 2020 07:00:00 GMT``) -> UTC."""
    if not txt:
        return None
    try:
        d = email.utils.parsedate_to_datetime(str(txt).strip())
    except (TypeError, ValueError, IndexError):
        return None
    if d is None:
        return None
    return d.astimezone(UTC) if d.tzinfo else d.replace(tzinfo=UTC)


def a_fecha(valor: Any) -> datetime | None:
    """epoch (s/ms), ISO 8601, RFC 822, ``AAAAMMDD[hhmmss]``, ``date`` o ``datetime`` -> ``datetime`` UTC.

    Sin zona se asume UTC. Devuelve None si no se entiende (nunca lanza).
    """
    if valor is None or valor == "" or (isinstance(valor, float) and valor != valor):
        return None
    try:
        if isinstance(valor, pd.Timestamp):
            valor = valor.to_pydatetime()
        if isinstance(valor, datetime):
            return valor.astimezone(UTC) if valor.tzinfo else valor.replace(tzinfo=UTC)
        if isinstance(valor, date):
            return datetime(valor.year, valor.month, valor.day, tzinfo=UTC)
        if isinstance(valor, (int, float)):
            s = valor / 1000 if valor > 1e11 else valor
            return datetime.fromtimestamp(s, tz=UTC)
        txt = str(valor).strip()
        if re.fullmatch(r"\d{14}", txt):
            return datetime.strptime(txt, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        if re.fullmatch(r"\d{8}", txt):
            return datetime.strptime(txt, "%Y%m%d").replace(tzinfo=UTC)
        if re.fullmatch(r"\d{9,13}(\.\d+)?", txt):
            return a_fecha(float(txt))
        if re.match(r"^\d{4}-\d{2}-\d{2}", txt):
            txt = re.sub(r"\s*(UTC|GMT|Z)$", "+00:00", txt)
            txt = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", txt)
            d = datetime.fromisoformat(txt)
            return d.astimezone(UTC) if d.tzinfo else d.replace(tzinfo=UTC)
        return fecha_rss(txt)
    except (ValueError, OverflowError, OSError, TypeError):
        return None


_COMPACTA = re.compile(r"(\d{8})(?:T?(\d{6})Z?)?")


def a_utc(valor: Any) -> datetime | None:
    """``a_fecha`` + formatos compactos de GDELT (``AAAAMMDDThhmmssZ``) y ISO raros (``+00``, nanosegundos).

    Única conversión a UTC de las sondas (GDELT, Wikipedia, mercados de predicción). Nunca lanza.
    """
    if isinstance(valor, str):
        m = _COMPACTA.fullmatch(valor.strip())
        if m:
            try:
                return datetime.strptime(m.group(1) + (m.group(2) or ""),
                                         "%Y%m%d%H%M%S" if m.group(2) else "%Y%m%d").replace(tzinfo=UTC)
            except ValueError:
                return None
    d = a_fecha(valor)
    if d is not None or not isinstance(valor, str) or not re.match(r"^\d{4}-\d{2}-\d{2}", valor.strip()):
        return d
    try:
        ts = pd.Timestamp(valor.strip())
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return ts.floor("us").to_pydatetime()
    except (ValueError, TypeError, OverflowError):
        return None


# --------------------------------------------------------------------------- HTML y RSS

_TAG = re.compile(r"<[^>]+>")


def limpiar_html(texto: str | None, max_chars: int | None = None) -> str:
    """Quita etiquetas y entidades, normaliza espacios y recorta (por palabra) si se pide."""
    if not texto:
        return ""
    t = _html.unescape(_TAG.sub(" ", _html.unescape(str(texto))))
    t = " ".join(t.replace("\xa0", " ").split())
    if max_chars and len(t) > max_chars:
        t = t[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return t


def _hijo(el: Any, *nombres: str) -> str | None:
    for n in nombres:
        for h in el:
            tag = h.tag.split("}")[-1] if isinstance(h.tag, str) else ""
            if tag == n:
                if n == "link" and h.get("href"):
                    return h.get("href")
                txt = "".join(h.itertext()).strip()
                if txt:
                    return txt
    return None


def parse_rss(contenido: bytes | str) -> list[dict[str, Any]]:
    """RSS 2.0, RDF o Atom -> ``[{titulo, url, fecha (datetime UTC|None), fecha_txt, medio, resumen}]``.

    Lanza ``ValueError`` si no es un feed (p. ej. una página HTML de error o de consentimiento).
    """
    import lxml.etree as ET

    if isinstance(contenido, str):
        contenido = contenido.encode("utf-8")
    try:
        raiz = ET.fromstring(contenido.lstrip(), parser=ET.XMLParser(recover=True, resolve_entities=False,
                                                                     no_network=True))
    except ET.XMLSyntaxError as e:
        raise ValueError(f"no es XML: {e}") from e
    if raiz is None:
        raise ValueError("no es XML")
    tag = raiz.tag.split("}")[-1].lower() if isinstance(raiz.tag, str) else ""
    if tag not in ("rss", "feed", "rdf"):
        raise ValueError(f"no es RSS/Atom (raíz <{tag}>)")
    items = [e for e in raiz.iter() if isinstance(e.tag, str) and e.tag.split("}")[-1] in ("item", "entry")]
    salida = []
    for it in items:
        fecha_txt = _hijo(it, "pubDate", "published", "updated", "date")
        medio = _hijo(it, "source")
        titulo = limpiar_html(_hijo(it, "title"))
        if medio and titulo.endswith(f" - {medio}"):          # Google News añade « - Medio»
            titulo = titulo[: -len(medio) - 3].rstrip()
        salida.append(dict(titulo=titulo, url=(_hijo(it, "link", "guid") or "").strip(),
                           fecha=fecha_rss(fecha_txt) or a_fecha(fecha_txt), fecha_txt=fecha_txt, medio=medio,
                           resumen=limpiar_html(_hijo(it, "description", "summary", "encoded", "content"))))
    return salida


# --------------------------------------------------------------------------- claves de duplicado

_SEGUIMIENTO = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_id", "ocid",
                "fbclid", "gclid", "cmpid", "smid", "ref", "ref_src", "src", "rss", "feed", "mc_cid", "mc_eid",
                "guccounter", "taid", "mod", "cid", "yptr", "soc_src", "soc_trk", "partner", "via"}


def dominio(url: str | None) -> str:
    """Host sin ``www.`` ni ``m.`` (el «medio» cuando no hay otro dato)."""
    if not url:
        return ""
    host = urlsplit(url.strip()).netloc.lower().split("@")[-1].split(":")[0]
    for pref in ("www.", "m.", "amp."):
        host = host.removeprefix(pref)
    return host


def clave_url(url: str) -> str:
    """URL canónica: sin esquema, ``www.``, fragmento, barra final ni parámetros de seguimiento."""
    p = urlsplit(url.strip())
    q = urlencode(sorted((k, v) for k, v in parse_qsl(p.query) if k.lower() not in _SEGUIMIENTO
                         and not k.lower().startswith("utm_")))
    return urlunsplit(("", dominio(url), p.path.rstrip("/"), q, ""))


_PAL = re.compile(r"\w+", re.UNICODE)


def tokens_titulo(titulo: str) -> frozenset[str]:
    return frozenset(w for w in _PAL.findall((titulo or "").lower()) if len(w) > 2)


def clave_titulo(titulo: str) -> str:
    return " ".join(sorted(tokens_titulo(titulo)))


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens_titulo(a), tokens_titulo(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def similares(a: str, b: str, umbral: float = UMBRAL_JACCARD) -> bool:
    """Jaccard de palabras del titular ≥ ``umbral``: la misma noticia sindicada en otro medio."""
    return jaccard(a, b) >= umbral


def simhash(texto: str, bits: int = 64) -> int:
    """SimHash de palabras (cuerpos casi idénticos: notas de agencia republicadas)."""
    v = [0] * bits
    for w in _PAL.findall((texto or "").lower()):
        h = int.from_bytes(hashlib.md5(w.encode("utf-8")).digest()[:8], "big")
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    return sum(1 << i for i in range(bits) if v[i] > 0)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# --------------------------------------------------------------------------- deduplicar

ETAPAS = ("url", "jaccard", "simhash")


def _fecha_doc(d: dict[str, Any]) -> datetime | None:
    extra = d.get("extra") or {}
    return a_fecha(d.get("fecha") or extra.get("hora_publicacion") or d.get("fecha_publicacion"))


def _medio_doc(d: dict[str, Any]) -> str:
    return d.get("medio") or d.get("institucion") or dominio(d.get("url"))


def _texto_doc(d: dict[str, Any]) -> str:
    return d.get("texto") or d.get("resumen") or ""


def _dedup(docs: Iterable[dict[str, Any]], etapas: Iterable[str], umbral: float, max_hamming: int
           ) -> list[dict[str, Any]]:
    etapas = set(etapas)
    lejos = datetime.max.replace(tzinfo=UTC)
    orden = sorted((dict(d, extra=dict(d.get("extra") or {})) for d in docs),
                   key=lambda d: (_fecha_doc(d) is None, _fecha_doc(d) or lejos))
    por_url: dict[str, dict] = {}
    salida: list[dict[str, Any]] = []
    huellas: list[tuple[int, dict]] = []

    def _anotar(original: dict, copia: dict) -> None:
        medio = _medio_doc(copia)
        lista = original["extra"].setdefault("sindicada_en", [])
        if medio and medio != _medio_doc(original) and medio not in lista:
            lista.append(medio)

    for d in orden:
        if "url" in etapas and d.get("url"):
            k = clave_url(d["url"])
            if k in por_url:
                _anotar(por_url[k], d)
                continue
        if "jaccard" in etapas:
            dup = next((s for s in salida if similares(s.get("titulo", ""), d.get("titulo", ""), umbral)), None)
            if dup is not None:
                _anotar(dup, d)
                continue
        h = None
        if "simhash" in etapas and len(_PAL.findall(_texto_doc(d))) >= MIN_PALABRAS_SIMHASH:
            h = simhash(_texto_doc(d))
            dup = next((s for hs, s in huellas if hamming(hs, h) <= max_hamming), None)
            if dup is not None:
                _anotar(dup, d)
                continue
        if d.get("url"):
            por_url[clave_url(d["url"])] = d
        if h is not None:
            huellas.append((h, d))
        salida.append(d)
    return salida


def deduplicar(docs: Iterable[dict[str, Any]], umbral: float = UMBRAL_JACCARD,
               max_hamming: int = UMBRAL_HAMMING) -> list[dict[str, Any]]:
    """Quita duplicados exactos (URL canónica), sindicados (titular, Jaccard) y cuerpos casi iguales
    (simhash). Conserva la versión con fecha MÁS ANTIGUA (sin fecha, al final) y anota en
    ``extra['sindicada_en']`` los otros medios que la repiten. No modifica los diccionarios de entrada."""
    return _dedup(docs, ETAPAS, umbral, max_hamming)


def embudo_duplicados(docs: Iterable[dict[str, Any]], umbral: float = UMBRAL_JACCARD,
                      max_hamming: int = UMBRAL_HAMMING) -> pd.DataFrame:
    """Embudo acumulado: documentos que quedan en bruto, tras URL, tras titular y tras simhash.

    Columnas: ``etapa, orden, n, eliminados, pct_restante`` (para pintar un embudo).
    """
    docs = list(docs)
    filas = [dict(etapa="bruto", orden=0, n=len(docs))]
    for i, nombre in enumerate(ETAPAS, start=1):
        filas.append(dict(etapa=f"tras_{nombre}", orden=i, n=len(_dedup(docs, ETAPAS[:i], umbral, max_hamming))))
    df = pd.DataFrame(filas)
    df["eliminados"] = (df["n"].shift(1) - df["n"]).fillna(0).astype(int)
    df["pct_restante"] = (100 * df["n"] / max(len(docs), 1)).round(1)
    return df


# --------------------------------------------------------------------------- idioma y tamaño

_VACIAS = {
    "en": set("the of and to in is that for on with as was by at from it be are this have has said".split()),
    "es": set("de la que el en los del se las por un para con una su al es lo como más pero sus".split()),
    "fr": set("de la le et les des en du un une est que pour dans sur pas par au qui avec".split()),
    "de": set("der die und in den von zu das mit sich des auf für ist im dem nicht ein eine".split()),
    "it": set("di il la che e in un per del della non sono una con le si da gli dei".split()),
    "pt": set("de que o a do da em um para com não uma os no se na por mais as dos".split()),
}


def idioma(texto: str | None, defecto: str = "und") -> str:
    """Idioma por recuento de palabras vacías (basta para filtrar y etiquetar; ``defecto`` si duda)."""
    palabras = _PAL.findall((texto or "").lower()[:4000])
    if len(palabras) < 5:
        return defecto
    puntos = {k: sum(1 for w in palabras if w in v) for k, v in _VACIAS.items()}
    mejor = max(puntos, key=lambda k: puntos[k])
    return mejor if puntos[mejor] >= max(2, 0.05 * len(palabras)) else defecto


def tokens_aprox(texto: str | None) -> int:
    """Tokens ≈ palabras × 1,3 (misma convención que el resto de la Fase 2)."""
    return round(len(_PAL.findall(texto or "")) * 1.3)


_FRASE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÁÉÍÓÚÑ0-9\"“'(])")


def trocear(texto: str, max_tokens: int = 180, min_tokens: int = 60, titulo: str | None = None) -> list[str]:
    """Trozos de ``min_tokens``-``max_tokens`` por frases (solape de una frase); el titular delante."""
    frases = [f.strip() for p in re.split(r"\n+", texto or "") for f in _FRASE.split(p) if f.strip()]
    trozos: list[list[str]] = []
    actual: list[str] = []
    for f in frases:
        if actual and tokens_aprox(" ".join(actual + [f])) > max_tokens:
            trozos.append(actual)
            actual = actual[-1:] if tokens_aprox(" ".join(actual[-1:] + [f])) <= max_tokens else []
        actual.append(f)
    if actual:
        if trozos and tokens_aprox(" ".join(actual)) < min_tokens:
            trozos[-1] = trozos[-1] + [f for f in actual if f not in trozos[-1]]
        else:
            trozos.append(actual)
    pref = f"{titulo.strip()}. " if titulo else ""
    return [pref + " ".join(t) for t in trozos]
