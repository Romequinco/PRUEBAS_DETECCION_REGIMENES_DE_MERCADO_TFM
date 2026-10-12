"""Esquemas comunes de la Fase 2.

``Documento``: metadatos que lleva TODO documento o trozo indexable, en cualquier RAG (base o
cajón) de cualquier agente. ``fecha_publicacion`` es obligatoria: es la que usa el TimeGate
(nada con ``fecha_publicacion > fecha`` de decisión sale de una tool).

``FichaFuente``: lo que se sabe y se ha medido de una fuente. Es la fila de la tabla de fuentes
del plan de datos.

Contrato común: todos los temas producen ``Documento`` (texto) o series en tabla larga con
``fecha_publicacion``. Cambiar este fichero afecta a todos: solo con un PR pequeño a ``main``.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal

Agente = Literal["macro", "equity", "risk", "portfolio", "cio", "comun"]
Rag = Literal["base", "cajon", "serie", "ninguno"]


def doc_id(contenido: bytes | str) -> str:
    """Identificador por contenido (sha256 truncado): el mismo texto se guarda una vez."""
    if isinstance(contenido, str):
        contenido = contenido.encode("utf-8")
    return hashlib.sha256(contenido).hexdigest()[:24]


@dataclass
class Documento:
    """Metadatos comunes de un documento (o trozo) indexable."""

    doc_id: str
    fuente: str                      # p. ej. "fed_fomc_statement", "sec_10k"
    tipo_documento: str              # p. ej. "statement", "minutes", "10-K", "paper"
    titulo: str
    fecha_publicacion: date          # TimeGate (granularidad: día)
    url: str
    agente: Agente
    rag: Rag
    fecha_referencia: date | None = None   # periodo al que se refiere (reunión, ejercicio fiscal...)
    institucion: str | None = None
    idioma: str = "en"
    tickers: list[str] = field(default_factory=list)
    entidades: list[str] = field(default_factory=list)
    paises: list[str] = field(default_factory=list)
    seccion: str | None = None       # p. ej. "Item 1A", "§2"
    licencia: str | None = None      # p. ej. "dominio público (US Gov)", "copyright: solo local"
    atemporal: bool = False          # manuales sin dependencia temporal (exentos del TimeGate)
    version: str | None = None       # vintage / revisión / enmienda
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("fecha_publicacion", "fecha_referencia"):
            if d[k] is not None:
                d[k] = d[k].isoformat()
        return d


@dataclass
class FichaFuente:
    """Ficha de una fuente: qué es, cómo se consigue y qué se ha medido."""

    fuente: str                      # id corto, nombre del JSON
    nombre: str                      # nombre legible
    tema: str                        # mercados_series | bancos_riesgo | empresas | noticias_alt
    agentes: list[str]
    rag: str                         # base | cajon | serie | base+cajon
    url: str
    acceso: str                      # api | descarga | scraping | libreria | manual
    autenticacion: str               # ninguna | clave gratuita | pago
    coste: str                       # "0 €" o precio
    licencia_y_terminos: str
    cobertura_inicio: str | None     # fecha real medida (ISO) o None si no se pudo medir
    cobertura_fin: str | None
    frecuencia: str                  # diaria | mensual | por evento | ...
    retraso_publicacion: str         # lag típico, medido o documentado
    formato: str                     # html | pdf | json | xbrl | csv | parquet ...
    volumen_estimado: str            # GB / nº de documentos del corpus completo (extrapolado)
    limites: str                     # rate limits, paginación, bloqueos
    estado_sonda: Literal["ok", "parcial", "fallo", "no_probada"]
    mediciones: dict[str, Any] = field(default_factory=dict)  # n_docs, bytes, segundos, errores...
    notas: list[str] = field(default_factory=list)
    fecha_sonda: str = field(default_factory=lambda: date.today().isoformat())
