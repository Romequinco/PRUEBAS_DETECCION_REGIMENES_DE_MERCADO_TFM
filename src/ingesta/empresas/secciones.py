"""Texto limpio y secciones de 10-K y 10-Q, convertidas en ``Documento`` del contrato común.

Un 10-K se organiza en *Items* fijados por la SEC (1 negocio, 1A riesgos, 7 MD&A, 7A riesgo de
mercado, 8 estados financieros...). Un 10-Q tiene Parte I (financiera: Item 2 MD&A, Item 3
riesgo de mercado) y Parte II (otra información: Item 1A riesgos).

Cómo se localiza cada sección
-----------------------------
1. HTML (o SGML de los filings antiguos) -> texto con un salto de línea por bloque.
2. Se buscan los encabezados ``Item N`` a principio de línea seguidos de un título (no de "of
   this report", que es una referencia cruzada).
3. Encabezados consecutivos del mismo Item (cabeceras de página) se funden en uno.
4. Cada Item aparece varias veces (índice, encabezado real). Se toma, para cada Item, el **tramo
   más largo** hasta el siguiente encabezado: el índice da tramos de una línea y el cuerpo real
   da el tramo largo.
5. Si una sección es muy corta y dice "incorporated by reference", se marca ``por_referencia``:
   muchos 10-K anteriores a ~2008 remiten el MD&A al informe anual (anexo 13). Medir cuántos
   pasa es parte de la sonda.
"""
from __future__ import annotations

import re
from datetime import date
from html.parser import HTMLParser

import pandas as pd

from ingesta.comun.esquema import Documento, doc_id

NOMBRES_10K = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity", "6": "Selected Financial Data (reserved)",
    "7": "Management's Discussion and Analysis", "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data", "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information", "9C": "Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors and Executive Officers", "11": "Executive Compensation", "12": "Security Ownership",
    "13": "Certain Relationships", "14": "Principal Accountant Fees", "15": "Exhibits", "16": "Form 10-K Summary",
}
NOMBRES_10Q = {
    "I-1": "Financial Statements", "I-2": "Management's Discussion and Analysis",
    "I-3": "Quantitative and Qualitative Disclosures About Market Risk", "I-4": "Controls and Procedures",
    "II-1": "Legal Proceedings", "II-1A": "Risk Factors", "II-2": "Unregistered Sales of Equity Securities",
    "II-5": "Other Information", "II-6": "Exhibits",
}
# Secciones que van al RAG base (el resto del filing queda en el cajón).
SECCIONES_BASE_10K = ("1", "1A", "1C", "7", "7A", "8")
SECCIONES_BASE_10Q = ("I-2", "I-3", "II-1A")

MIN_CARACTERES = 300            # por debajo, la "sección" es un índice o un "None."
_POR_REFERENCIA = re.compile(r"incorporated\s+(?:herein\s+)?by\s+reference", re.IGNORECASE)

# Encabezado de Item a principio de línea: "Item 1A.", "ITEM 7 —", "Item 1(A)", "Items 1 and 2.", "PART II, Item 5".
# Tras el número solo se acepta un título (mayúscula), comillas, fin de línea o "and"/"&": así una
# referencia cruzada que cae a principio de línea ("Item 8 of this Form 10-K") no cuenta como encabezado.
_ITEM = re.compile(
    r"^[ \t]*(?:part[ \t]+(?P<parte>iv|iii|ii|i)\b[ \t.,:\-–—]*)?"
    r"items?[ \t]*(?P<num>1[0-6]|[1-9])[ \t]*(?:\([ \t]*(?P<l1>[a-c])[ \t]*\)|(?P<l2>[a-c]))?(?![0-9a-z])"
    r"(?=[ \t]*[.:\-–—)]?[ \t]*(?:$|(?-i:[A-Z])|[\"“'(]|and\b|&))",
    re.IGNORECASE | re.MULTILINE,
)
_PARTE = re.compile(r"^[ \t]*part[ \t]+(?P<parte>iv|iii|ii|i)(?![a-z])", re.IGNORECASE | re.MULTILINE)
_BLOQUES = frozenset({"p", "div", "br", "tr", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6", "title", "center", "pre"})
_OCULTOS = frozenset({"script", "style", "ix:header"})   # ix:header = metadatos inline XBRL ocultos
# Filings de texto de los 90/2000: no son HTML, pero llevan marcas SGML (<TABLE>, <S>, <C>, <PAGE>...).
_ES_HTML = re.compile(r"<(html|body|p|div|font|span|br)\b", re.IGNORECASE)
_MARCAS_SGML = re.compile(r"</?(?:table|caption|s|c|page|fn|f\d+|pre|r\d+|description|type|sequence|filename|text)\b[^>\n]*>",
                          re.IGNORECASE)


class _TextoHTML(HTMLParser):
    """HTML -> texto con la librería estándar (sin dependencias): salto de línea por bloque."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.partes: list[str] = []
        self._oculto = 0
        self._pre = 0

    def handle_starttag(self, tag, attrs):
        if tag in _OCULTOS:
            self._oculto += 1
        elif tag in _BLOQUES:
            self.partes.append("\n")
        if tag == "pre":
            self._pre += 1

    def handle_startendtag(self, tag, attrs):
        if tag in _BLOQUES:
            self.partes.append("\n")

    def handle_endtag(self, tag):
        if tag == "pre":
            self._pre = max(0, self._pre - 1)
        if tag in _OCULTOS:
            self._oculto = max(0, self._oculto - 1)
        elif tag in _BLOQUES:
            self.partes.append("\n")
        elif tag in ("td", "th"):
            self.partes.append(" ")

    def handle_data(self, data):
        if self._oculto:
            return
        # Los saltos de línea del código fuente HTML no son saltos de párrafo: fuera de <pre> se
        # colapsan, para que "...in\nItem 8 of this report" no empiece una línea con "Item 8".
        self.partes.append(data if self._pre else re.sub(r"\s+", " ", data))


# --------------------------------------------------------------------------- texto limpio

def _documento_principal_sgml(texto: str) -> str:
    """De un envío completo ``.txt`` (SGML), el ``<TEXT>`` del primer ``<DOCUMENT>``."""
    m = re.search(r"<TEXT>(.*?)</TEXT>", texto, re.IGNORECASE | re.DOTALL)
    return m.group(1) if m else texto


def html_a_texto(contenido: bytes | str) -> str:
    """HTML o SGML de EDGAR -> texto plano con un salto de línea por bloque."""
    if isinstance(contenido, bytes):
        contenido = contenido.decode("utf-8", errors="replace")
    if re.search(r"<DOCUMENT>", contenido, re.IGNORECASE):   # envío completo (la cabecera puede ser larga)
        contenido = _documento_principal_sgml(contenido)
    if not _ES_HTML.search(contenido[:200000]):
        # Texto plano con marcas SGML: se quitan las marcas y se conservan los saltos de línea
        # (si pasara por el parser HTML, los saltos se colapsarían y no habría encabezados).
        texto = _MARCAS_SGML.sub(" ", contenido)
    else:
        parser = _TextoHTML()
        parser.feed(contenido)
        parser.close()
        texto = "".join(parser.partes)
    texto = texto.replace("\xa0", " ").replace("​", "")
    lineas = (re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in texto.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lineas)).strip()


# --------------------------------------------------------------------------- secciones

def _encabezados(texto: str, por_partes: bool) -> list[tuple[int, str]]:
    """(posición, clave) de cada encabezado de Item, en orden."""
    partes = [(m.start(), m.group("parte").upper()) for m in _PARTE.finditer(texto)] if por_partes else []
    salida: list[tuple[int, str]] = []
    for m in _ITEM.finditer(texto):
        num = m.group("num") + (m.group("l1") or m.group("l2") or "").upper()
        if por_partes:
            parte = (m.group("parte") or "").upper()
            if not parte:
                previas = [p for pos, p in partes if pos <= m.start()]
                parte = previas[-1] if previas else "I"
            clave = f"{parte}-{num}"
        else:
            clave = num
        # Cabeceras de página que repiten el Item ("PART II · Item 7" en cada página): se funden con
        # el encabezado anterior para no trocear la sección.
        if salida and salida[-1][1] == clave:
            continue
        salida.append((m.start(), clave))
    return salida


def extraer_secciones(texto: str, tipo: str = "10-K") -> dict[str, str]:
    """``{clave: texto}`` de cada Item encontrado (claves ``"1A"`` en 10-K, ``"II-1A"`` en 10-Q).

    Para cada clave se queda el tramo más largo entre su encabezado y el siguiente.
    Se descartan las de menos de ``MIN_CARACTERES`` salvo que remitan a otro documento
    (``es_por_referencia``): esas se devuelven para poder contarlas, pero no generan ``Documento``.
    """
    encabezados = _encabezados(texto, por_partes=tipo.upper().startswith("10-Q"))
    mejores: dict[str, str] = {}
    for i, (pos, clave) in enumerate(encabezados):
        fin = encabezados[i + 1][0] if i + 1 < len(encabezados) else len(texto)
        tramo = texto[pos:fin].strip()
        if len(tramo) > len(mejores.get(clave, "")):
            mejores[clave] = tramo
    return {k: v for k, v in mejores.items() if len(v) >= MIN_CARACTERES or es_por_referencia(v)}


def es_por_referencia(texto: str) -> bool:
    """Sección corta que solo remite a otro documento (típico del MD&A antiguo en el anexo 13)."""
    return len(texto) < 3000 and bool(_POR_REFERENCIA.search(texto))


# --------------------------------------------------------------------------- Documento

def _a_fecha(valor) -> date | None:
    return None if valor is None or pd.isna(valor) else valor


def _iso(valor) -> str | None:
    """ISO de una fecha/hora, o ``None`` si falta (``None`` o ``NaT``)."""
    return None if valor is None or pd.isna(valor) else pd.Timestamp(valor).isoformat()


def documentos_de_filing(fila: pd.Series, secciones: dict[str, str], ticker: str) -> list[dict]:
    """Una fila jsonl por sección: metadatos del contrato común (``Documento``) + ``texto``.

    ``Documento`` no tiene campo de texto; hasta que se acuerde en ``comun``, el texto va en la
    clave ``texto`` de la misma línea. Las secciones que solo remiten a otro documento no se emiten.
    """
    familia = fila["familia"]
    nombres = NOMBRES_10Q if familia == "10-Q" else NOMBRES_10K
    base = SECCIONES_BASE_10Q if familia == "10-Q" else SECCIONES_BASE_10K
    filas = []
    for clave, texto in secciones.items():
        if es_por_referencia(texto):
            continue
        etiqueta = f"Part {clave.split('-')[0]} Item {clave.split('-')[1]}" if "-" in clave else f"Item {clave}"
        doc = Documento(
            doc_id=doc_id(f"{fila['accession']}|{clave}"),
            fuente=f"sec_{familia.lower().replace('-', '')}",
            tipo_documento=fila["form"],
            titulo=f"{fila['entidad']} · {fila['form']} {fila['report_date']} · {etiqueta} {nombres.get(clave, '')}".strip(),
            fecha_publicacion=fila["fecha_publicacion"],
            url=fila["url_documento"],
            agente="equity",
            rag="base" if (clave in base and not fila["enmienda"]) else "cajon",
            fecha_referencia=_a_fecha(fila["report_date"]),
            institucion=fila["entidad"],
            tickers=[ticker],
            paises=["US"],
            seccion=etiqueta,
            licencia="dominio público (SEC EDGAR, documento oficial)",
            version="enmienda" if fila["enmienda"] else None,
            extra={
                "cik": int(fila["cik"]),
                "accession": fila["accession"],
                "filing_date": fila["filing_date"].isoformat(),
                "aceptacion": _iso(fila["aceptacion"]),
                "clave_seccion": clave,
                "n_caracteres": len(texto),
                "n_palabras": len(texto.split()),
            },
        )
        filas.append({**doc.to_dict(), "texto": texto})
    return filas


def documento_completo(fila: pd.Series, texto: str, ticker: str) -> dict:
    """Fila jsonl con el filing entero, para 10-K / 10-Q sin Items reconocibles (al cajón).

    Algunas empresas (p. ej. GE) presentan el 10-K como informe anual con un índice de referencias
    cruzadas al final, sin encabezados "Item N" en el cuerpo: así no se pierde el texto.
    """
    familia = fila["familia"]
    doc = Documento(
        doc_id=doc_id(f"{fila['accession']}|completo"),
        fuente=f"sec_{familia.lower().replace('-', '')}",
        tipo_documento=fila["form"],
        titulo=f"{fila['entidad']} · {fila['form']} {fila['report_date']} · documento completo",
        fecha_publicacion=fila["fecha_publicacion"],
        url=fila["url_documento"],
        agente="equity",
        rag="cajon",
        fecha_referencia=_a_fecha(fila["report_date"]),
        institucion=fila["entidad"],
        tickers=[ticker],
        paises=["US"],
        seccion="Documento completo",
        licencia="dominio público (SEC EDGAR, documento oficial)",
        version="enmienda" if fila["enmienda"] else None,
        extra={"cik": int(fila["cik"]), "accession": fila["accession"], "aceptacion": _iso(fila["aceptacion"]),
               "clave_seccion": "completo", "n_caracteres": len(texto), "n_palabras": len(texto.split())},
    )
    return {**doc.to_dict(), "texto": texto}


def documento_resultados(fila: pd.Series, texto: str, url: str, ticker: str) -> dict:
    """Fila jsonl de un 8-K de resultados (nota de prensa del anexo 99). Va al cajón del agente."""
    doc = Documento(
        doc_id=doc_id(f"{fila['accession']}|ex99"),
        fuente="sec_8k_resultados",
        tipo_documento=fila["form"],
        titulo=f"{fila['entidad']} · resultados ({fila['filing_date']})",
        fecha_publicacion=fila["fecha_publicacion"],
        url=url,
        agente="equity",
        rag="cajon",
        fecha_referencia=_a_fecha(fila["report_date"]),
        institucion=fila["entidad"],
        tickers=[ticker],
        paises=["US"],
        seccion="Exhibit 99",
        licencia="dominio público (SEC EDGAR, documento oficial)",
        extra={"cik": int(fila["cik"]), "accession": fila["accession"], "items": fila["items"],
               "aceptacion": _iso(fila["aceptacion"]),
               "n_palabras": len(texto.split())},
    )
    return {**doc.to_dict(), "texto": texto}
