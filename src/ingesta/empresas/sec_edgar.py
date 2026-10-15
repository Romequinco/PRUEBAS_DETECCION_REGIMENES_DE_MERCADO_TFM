"""SEC EDGAR: lista de filings de una empresa, TimeGate y descarga de documentos.

Fuente: API pública de la SEC, gratis y sin clave. Exige un User-Agent con contacto
(``SEC_EDGAR_USER_AGENT`` en ``.env``) y permite hasta 10 peticiones/s; aquí se usan 5.

Piezas:

- ``tabla_filings(cik)``: todos los filings de la empresa (no solo los ~1.000 recientes: también
  las páginas antiguas de ``filings.files``), con una ``fecha_publicacion`` prudente.
- ``verificar_aceptacion(df)``: lee la hora oficial de aceptación de cada filing (cabecera
  ``.hdr.sgml``, cacheada) y fija su ``fecha_publicacion`` definitiva.
- ``fecha_publicacion(...)``: el **TimeGate** de este tema (ver abajo).
- ``descargar_documento(fila)``: guarda el documento principal comprimido en el corpus.
- ``exhibit_resultados(fila)``: localiza el anexo 99.1 (nota de prensa) de un 8-K de resultados.

TimeGate
--------
Un filing se considera conocido el día ``filingDate``, salvo que la SEC lo aceptara a partir de
las 16:00 (hora de Nueva York, cierre del mercado) de ese mismo día: entonces el mercado no pudo
usarlo hasta la sesión siguiente y ``fecha_publicacion`` pasa al siguiente día hábil.
(Lo aceptado después de las 17:30 ya trae ``filingDate`` del día siguiente por norma de la SEC.)

La hora oficial de aceptación (Nueva York) está en la cabecera ``<acc>.hdr.sgml``
(``ACCEPTANCE-DATETIME``). La de ``acceptanceDateTime`` de la API **no es fiable**: en la prueba
real (10-K de Apple del 31-oct-2025) la cabecera dice 06:01:26 y la API 14:01:26Z, que no es ni la
hora de Nueva York ni UTC. Por eso:

- hasta verificarla, cada filing recibe la fecha prudente (día hábil siguiente a ``filingDate``:
  retrasar un día nunca mira el futuro);
- ``verificar_aceptacion`` lee la cabecera de los filings que se van a usar y aplica la regla
  de las 16:00. La API se guarda en ``aceptacion_api`` solo para medir su desfase.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import zlib
from html.parser import HTMLParser
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay, DateOffset

from dateutil.relativedelta import MO

from ingesta.comun.http import get
from ingesta.comun.rutas import corpus_dir
from ingesta.empresas.universo import cik10

BASE_DATA = "https://data.sec.gov"
BASE_WWW = "https://www.sec.gov"
POR_SEGUNDO = 5.0               # la SEC permite 10/s; se va a la mitad
CIERRE_MERCADO = time(16, 0)    # hora de Nueva York

# Formularios que interesan y a qué familia pertenecen (los 10-K405/10-KSB son 10-K antiguos).
FAMILIA_FORM = {
    "10-K": "10-K", "10-K405": "10-K", "10-KSB": "10-K", "10-KT": "10-K",
    "10-K/A": "10-K", "10-K405/A": "10-K", "10-KSB/A": "10-K",
    "10-Q": "10-Q", "10-QSB": "10-Q", "10-Q/A": "10-Q",
    "20-F": "20-F", "20-F/A": "20-F",
    "8-K": "8-K", "8-K/A": "8-K",
}

# Columnas de las listas de filings de la API (``filings.recent`` y las páginas antiguas).
_CAMPOS = {
    "accessionNumber": "accession", "filingDate": "filing_date", "reportDate": "report_date",
    "acceptanceDateTime": "aceptacion_raw", "form": "form", "items": "items",
    "primaryDocument": "documento", "primaryDocDescription": "descripcion", "size": "bytes_filing",
    "isXBRL": "is_xbrl", "isInlineXBRL": "is_inline_xbrl",
}


# Cierres extraordinarios de la NYSE desde que existe EDGAR (1993).
_CIERRES_EXTRA = [
    ("Funeral de Nixon", 1994, 4, 27), ("11-S", 2001, 9, 11), ("11-S", 2001, 9, 12), ("11-S", 2001, 9, 13),
    ("11-S", 2001, 9, 14), ("Funeral de Reagan", 2004, 6, 11), ("Funeral de Ford", 2007, 1, 2),
    ("Huracán Sandy", 2012, 10, 29), ("Huracán Sandy", 2012, 10, 30), ("Funeral de Bush padre", 2018, 12, 5),
    ("Funeral de Carter", 2025, 1, 9),
]


class _CalendarioNYSE(AbstractHolidayCalendar):
    """Festivos de la bolsa de Nueva York desde 1993, con sus cierres extraordinarios.

    Año nuevo en sábado no se recupera el viernes anterior (la NYSE abre el 31 de diciembre);
    Martin Luther King cierra desde 1998.
    """

    rules = [
        Holiday("Año nuevo", month=1, day=1, observance=sunday_to_monday),
        Holiday("Martin Luther King", month=1, day=1, offset=DateOffset(weekday=MO(3)), start_date="1998-01-01"),
        USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independencia", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Navidad", month=12, day=25, observance=nearest_workday),
        *[Holiday(f"{n} {y}", year=y, month=m, day=d) for n, y, m, d in _CIERRES_EXTRA],
    ]


DIA_HABIL = CustomBusinessDay(calendar=_CalendarioNYSE())


# --------------------------------------------------------------------------- TimeGate

def parsear_aceptacion(valor: str | None) -> datetime | None:
    """``2024-11-01T06:01:36.000Z`` -> ``datetime(2024, 11, 1, 6, 1, 36)`` (tal cual; la zona de la API no es fiable)."""
    if not valor or not isinstance(valor, str):
        return None
    try:
        return datetime.fromisoformat(valor.strip().rstrip("Z").split(".")[0])
    except ValueError:
        return None


def siguiente_dia_habil(dia: date) -> date:
    return (pd.Timestamp(dia) + DIA_HABIL).date()


def fecha_publicacion(filing_date: date, aceptacion: datetime | None) -> date:
    """Día en que el mercado pudo usar el filing, con ``aceptacion`` = hora oficial (Nueva York).

    Sin hora oficial se devuelve la fecha prudente (día hábil siguiente a ``filing_date``).
    """
    if aceptacion is None or pd.isna(aceptacion):
        return siguiente_dia_habil(filing_date)
    if aceptacion.date() == filing_date and aceptacion.time() >= CIERRE_MERCADO:
        return siguiente_dia_habil(filing_date)
    return filing_date


# --------------------------------------------------------------------------- caché

def _escribir_atomico(ruta: Path, contenido: bytes) -> None:
    """Escribe en un temporal y lo renombra: una ejecución interrumpida no deja ficheros a medias."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_name(ruta.name + ".tmp")
    tmp.write_bytes(contenido)
    os.replace(tmp, ruta)


def _leer_gz(ruta: Path) -> bytes | None:
    """Contenido de una caché gzip, o ``None`` si no existe o está corrupta (se volverá a descargar)."""
    if not ruta.exists():
        return None
    try:
        return gzip.decompress(ruta.read_bytes())
    except (OSError, EOFError, zlib.error):
        return None


def _cache_json(ruta: Path, url: str, usar_cache: bool) -> dict:
    if usar_cache and (crudo := _leer_gz(ruta)) is not None:
        try:
            return json.loads(crudo)
        except ValueError:
            pass
    datos = get(url, por_segundo=POR_SEGUNDO).json()
    _escribir_atomico(ruta, gzip.compress(json.dumps(datos).encode("utf-8")))
    return datos


# --------------------------------------------------------------------------- lista de filings


def descargar_submissions(cik: int | str, usar_cache: bool = True) -> tuple[dict, list[dict]]:
    """JSON principal de ``submissions`` y sus páginas antiguas (``filings.files``)."""
    carpeta = corpus_dir("empresas") / "sec" / "submissions"
    principal = _cache_json(carpeta / f"CIK{cik10(cik)}.json.gz",
                            f"{BASE_DATA}/submissions/CIK{cik10(cik)}.json", usar_cache)
    paginas = [
        _cache_json(carpeta / f"{f['name']}.gz", f"{BASE_DATA}/submissions/{f['name']}", usar_cache)
        for f in principal.get("filings", {}).get("files", [])
    ]
    return principal, paginas


def submissions_a_tabla(principal: dict, paginas: list[dict] | None = None) -> pd.DataFrame:
    """Une ``filings.recent`` y las páginas antiguas en una tabla con ``fecha_publicacion``."""
    bloques = [principal.get("filings", {}).get("recent", {})] + list(paginas or [])
    partes = []
    for b in bloques:
        if not b or not b.get("accessionNumber"):
            continue
        n = len(b["accessionNumber"])
        partes.append(pd.DataFrame({nuevo: b.get(viejo, [None] * n) for viejo, nuevo in _CAMPOS.items()}))
    if not partes:
        return pd.DataFrame(columns=["cik", "entidad", *_CAMPOS.values()])
    df = pd.concat(partes, ignore_index=True).drop_duplicates("accession")

    cik = int(principal["cik"])
    df.insert(0, "cik", cik)
    df.insert(1, "entidad", principal.get("name"))
    df["filing_date"] = pd.to_datetime(df["filing_date"]).dt.date
    df["report_date"] = pd.to_datetime(df["report_date"].replace("", None), errors="coerce").dt.date
    # dtype object: sin hora queda None, no NaT. La hora de la API solo se guarda para medir su desfase.
    df["aceptacion_api"] = pd.Series([parsear_aceptacion(v) for v in df["aceptacion_raw"]], index=df.index, dtype=object)
    df["aceptacion"] = pd.Series([None] * len(df), index=df.index, dtype=object)   # oficial: verificar_aceptacion
    df["aceptacion_verificada"] = False
    df["fecha_publicacion"] = [siguiente_dia_habil(f) for f in df["filing_date"]]  # prudente hasta verificar
    df["familia"] = df["form"].map(FAMILIA_FORM)
    df["enmienda"] = df["form"].str.endswith("/A")
    df["items"] = df["items"].fillna("")
    df["documento"] = df["documento"].fillna("")
    acc = df["accession"].str.replace("-", "", regex=False)
    df["url_carpeta"] = f"{BASE_WWW}/Archives/edgar/data/{cik}/" + acc
    # Sin documento principal (filings muy antiguos): el envío completo en texto.
    df["url_documento"] = [
        f"{c}/{d}" if d else f"{c}/{a}.txt" for c, d, a in zip(df["url_carpeta"], df["documento"], df["accession"])
    ]
    # Días entre el cierre del periodo (reportDate) y la presentación: el retraso de publicación.
    df["retraso_dias"] = (pd.to_datetime(df["filing_date"]) - pd.to_datetime(df["report_date"])).dt.days
    return df.sort_values("filing_date").reset_index(drop=True)


def tabla_filings(cik: int | str, usar_cache: bool = True) -> pd.DataFrame:
    """Todos los filings de una empresa con su ``fecha_publicacion`` (TimeGate)."""
    principal, paginas = descargar_submissions(cik, usar_cache)
    return submissions_a_tabla(principal, paginas)


def es_resultados(items: str) -> bool:
    """8-K de resultados: Item 2.02 (desde agosto de 2004) o su antecesor Item 12 (2003-2004)."""
    codigos = {c.strip() for c in str(items).split(",")}
    return "2.02" in codigos or "12" in codigos


def filtrar(df: pd.DataFrame, familias=("10-K", "10-Q"), resultados_8k: bool = True,
            enmiendas: bool = False, desde: date | None = None) -> pd.DataFrame:
    """Filings que interesan: 10-K / 10-Q (y 20-F si se pide) + 8-K de resultados."""
    m = df["familia"].isin(familias)
    if resultados_8k:
        m |= (df["familia"] == "8-K") & df["items"].map(es_resultados)
    if not enmiendas:
        m &= ~df["enmienda"]
    if desde is not None:
        m &= df["filing_date"] >= desde
    return df.loc[m].reset_index(drop=True)


# --------------------------------------------------------------------------- descargas

def ruta_cruda(fila: pd.Series, nombre: str | None = None) -> Path:
    """``<corpus>/empresas/sec/crudo/<cik>/<año>/<accession>__<documento>.gz``."""
    nombre = nombre or fila["url_documento"].rsplit("/", 1)[-1]
    return (corpus_dir("empresas") / "sec" / "crudo" / str(fila["cik"]) / str(fila["filing_date"].year)
            / f"{fila['accession']}__{nombre}.gz")


def descargar_documento(fila: pd.Series, url: str | None = None, usar_cache: bool = True) -> tuple[bytes, Path]:
    """Descarga (o lee de la caché) un documento del filing; lo guarda comprimido con gzip."""
    url = url or fila["url_documento"]
    ruta = ruta_cruda(fila, url.rsplit("/", 1)[-1])
    if usar_cache and (crudo := _leer_gz(ruta)) is not None:
        return crudo, ruta
    contenido = get(url, por_segundo=POR_SEGUNDO, timeout=60).content
    _escribir_atomico(ruta, gzip.compress(contenido))
    return contenido, ruta


# "ex99-1.htm", "ex991.htm", "exhibit991.htm", "a8-kexhibit991.htm", "exh99_1.htm", "d123dex991.htm"...
_EX99 = re.compile(r"ex(?:hibit|h)?[-_ ]?99", re.IGNORECASE)


def elegir_exhibit_99(nombres: list[str]) -> str | None:
    """Nombre del anexo 99.1 entre los ficheros de la carpeta del filing (``None`` si no hay).

    Prefiere 99.1 frente a otros 99.x y HTML frente a texto.
    """
    candidatos = [n for n in nombres if _EX99.search(n) and n.lower().endswith((".htm", ".html", ".txt"))]
    if not candidatos:
        return None

    def prioridad(n: str) -> tuple[int, int]:
        b = n.lower()
        es_991 = bool(re.search(r"99[-_.]?0?1(?!\d)", b))
        return (0 if es_991 else 1, 0 if b.endswith((".htm", ".html")) else 1)

    return sorted(candidatos, key=prioridad)[0]


class _FilasIndice(HTMLParser):
    """Filas de la tabla de documentos de ``<acc>-index.htm``: (textos de las celdas, enlaces)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.filas: list[tuple[list[str], list[str]]] = []
        self._celdas: list[str] | None = None
        self._enlaces: list[str] = []
        self._celda: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._celdas, self._enlaces = [], []
        elif tag in ("td", "th") and self._celdas is not None:
            self._celda = []
        elif tag == "a" and self._celdas is not None:
            href = dict(attrs).get("href") or ""
            if "/Archives/edgar/data/" in href:
                self._enlaces.append(href)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._celda is not None and self._celdas is not None:
            self._celdas.append(" ".join("".join(self._celda).split()))
            self._celda = None
        elif tag == "tr" and self._celdas is not None:
            self.filas.append((self._celdas, self._enlaces))
            self._celdas = None

    def handle_data(self, data):
        if self._celda is not None:
            self._celda.append(data)


def documentos_del_indice(html: str) -> list[tuple[str, str]]:
    """``[(nombre_fichero, tipo)]`` de la página ``<acc>-index.htm`` (tipo = ``EX-99.1``, ``8-K``...)."""
    p = _FilasIndice()
    p.feed(html)
    salida = []
    for celdas, enlaces in p.filas:
        if not enlaces:
            continue
        nombre = enlaces[0].rsplit("/", 1)[-1].split("?")[0]
        tipo = next((c for c in celdas if re.fullmatch(r"(EX-[0-9][\w.\-]*|8-K[\w/]*|10-[KQ][\w/]*|GRAPHIC|XML|ZIP|JSON)", c)), "")
        salida.append((nombre, tipo))
    return salida


def elegir_anexo_99(documentos: list[tuple[str, str]]) -> str | None:
    """Anexo de resultados: por tipo (``EX-99.1`` primero, luego cualquier ``EX-99.x``) y, si no hay tipos, por nombre."""
    html = [(n, t) for n, t in documentos if n.lower().endswith((".htm", ".html", ".txt"))]
    for criterio in (lambda t: t.upper() == "EX-99.1", lambda t: t.upper().startswith("EX-99")):
        for n, t in html:
            if criterio(t):
                return n
    return elegir_exhibit_99([n for n, _ in html])


def exhibit_resultados(fila: pd.Series, usar_cache: bool = True) -> tuple[str | None, list[str]]:
    """URL de la nota de prensa (anexo 99) de un 8-K de resultados y la lista de documentos del filing.

    Se lee la página ``<acc>-index.htm``, que da el **tipo** de cada documento (``EX-99.1``): el
    nombre del fichero no siempre lo dice. Si no hay anexo 99, la URL es ``None``.
    """
    ruta = corpus_dir("empresas") / "sec" / "indices" / str(fila["cik"]) / f"{fila['accession']}-index.htm.gz"
    html = _leer_gz(ruta) if usar_cache else None
    if html is None:
        html = get(f"{fila['url_carpeta']}/{fila['accession']}-index.htm", por_segundo=POR_SEGUNDO).content
        _escribir_atomico(ruta, gzip.compress(html))
    documentos = documentos_del_indice(html.decode("utf-8", errors="replace"))
    elegido = elegir_anexo_99(documentos)
    return (f"{fila['url_carpeta']}/{elegido}" if elegido else None), [f"{n} ({t or '?'})" for n, t in documentos]


_HDR = re.compile(r"<ACCEPTANCE-DATETIME>\s*(\d{14})")


def aceptacion_desde_hdr(texto: str) -> datetime | None:
    m = _HDR.search(texto)
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else None


def aceptacion_hdr(fila: pd.Series) -> datetime | None:
    """Hora oficial de aceptación (Nueva York) según la cabecera SGML del filing."""
    url = f"{fila['url_carpeta']}/{fila['accession']}.hdr.sgml"
    return aceptacion_desde_hdr(get(url, por_segundo=POR_SEGUNDO).text)


def verificar_aceptacion(df: pd.DataFrame, usar_cache: bool = True) -> pd.DataFrame:
    """Copia de ``df`` con la hora oficial (cabecera ``.hdr.sgml``) y la ``fecha_publicacion`` definitiva.

    Una petición pequeña por filing, cacheada en ``<corpus>/empresas/sec/aceptacion/<cik>.json``.
    Si la cabecera no trae hora o no se puede leer, el filing conserva la fecha prudente.
    """
    df = df.copy()
    for cik, idx in df.groupby("cik").groups.items():
        ruta = corpus_dir("empresas") / "sec" / "aceptacion" / f"{cik}.json"
        cache = json.loads(ruta.read_text(encoding="utf-8")) if (usar_cache and ruta.exists()) else {}
        cambios = False
        for i in idx:
            fila = df.loc[i]
            acc = fila["accession"]
            if acc not in cache:
                try:
                    hora = aceptacion_hdr(fila)
                except Exception:  # noqa: BLE001 — sin cabecera: se queda la fecha prudente (no se cachea)
                    continue
                cache[acc] = hora.isoformat() if hora else None
                cambios = True
            if cache[acc]:
                hora = datetime.fromisoformat(cache[acc])
                df.at[i, "aceptacion"] = hora
                df.at[i, "aceptacion_verificada"] = True
                df.at[i, "fecha_publicacion"] = fecha_publicacion(fila["filing_date"], hora)
        if cambios:
            _escribir_atomico(ruta, json.dumps(cache).encode("utf-8"))
    return df
