"""Fundamentales XBRL de la SEC (``companyfacts``) como tabla larga *point-in-time*.

Desde 2009 (grandes empresas) los estados financieros se presentan también en XBRL: cada cifra
lleva concepto (``us-gaap:NetIncomeLoss``), periodo, unidad y el filing en que se publicó.

``companyfacts`` devuelve **cada cifra tantas veces como se ha publicado**: el beneficio de 2019
aparece en el 10-K de 2019 y otra vez, como comparativo (y a veces corregido), en los de 2020 y
2021. Por eso la tabla se guarda entera, con ``fecha_publicacion`` por fila, y la vista "lo que se
sabía el día X" se calcula al consultar (``vista_a_fecha``). Así no se mira el futuro aunque la
empresa corrija cifras después.

Contrato de series (``docs/fase2``): ``serie, fecha_referencia, fecha_publicacion, valor, unidad,
fuente`` + columnas propias del tema.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from ingesta.comun.rutas import corpus_dir
from ingesta.empresas.sec_edgar import BASE_DATA, _cache_json, siguiente_dia_habil
from ingesta.empresas.universo import cik10

FUENTE = "sec_companyfacts"

# Conceptos para resumir la sonda (la tabla guarda todos). Varios nombres = la empresa usó
# uno u otro según la época (p. ej. los ingresos cambiaron de etiqueta con ASC 606 en 2018).
CONCEPTOS_CLAVE = {
    "ingresos": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "beneficio_neto": ["NetIncomeLoss"],
    "beneficio_operativo": ["OperatingIncomeLoss"],
    "bpa_diluido": ["EarningsPerShareDiluted"],
    "activo_total": ["Assets"],
    "pasivo_total": ["Liabilities"],
    "patrimonio": ["StockholdersEquity"],
    "caja": ["CashAndCashEquivalentsAtCarryingValue"],
    "deuda_largo_plazo": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "flujo_operativo": ["NetCashProvidedByUsedInOperatingActivities"],
}

COLUMNAS = ["serie", "fecha_referencia", "fecha_publicacion", "valor", "unidad", "fuente",
            "cik", "ticker", "taxonomia", "concepto", "fecha_inicio", "form", "fy", "fp", "accession", "frame"]


def descargar_companyfacts(cik: int | str, usar_cache: bool = True) -> dict:
    """JSON ``companyfacts`` de la empresa (caché comprimida en el corpus)."""
    ruta = corpus_dir("empresas") / "sec" / "companyfacts" / f"CIK{cik10(cik)}.json.gz"
    return _cache_json(ruta, f"{BASE_DATA}/api/xbrl/companyfacts/CIK{cik10(cik)}.json", usar_cache)


def companyfacts_a_tabla(cf: dict, ticker: str, filings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla larga con una fila por (concepto, periodo, filing).

    ``fecha_publicacion`` sale de la tabla de ``filings`` de la empresa (TimeGate con hora de
    aceptación) para ese ``accession``. Si no se pasa o el ``accession`` no está, se usa, por
    prudencia, el día hábil siguiente a ``filed``: sin la hora no se sabe si se aceptó tras el
    cierre, y retrasar un día nunca mira el futuro.
    """
    filas = []
    cik = int(cf.get("cik", 0))
    for taxonomia, conceptos in cf.get("facts", {}).items():
        for concepto, info in conceptos.items():
            for unidad, hechos in info.get("units", {}).items():
                for h in hechos:
                    filas.append((taxonomia, concepto, unidad, h.get("start"), h.get("end"), h.get("val"),
                                  h.get("accn"), h.get("fy"), h.get("fp"), h.get("form"), h.get("filed"),
                                  h.get("frame")))
    df = pd.DataFrame(filas, columns=["taxonomia", "concepto", "unidad", "fecha_inicio", "fecha_referencia",
                                      "valor", "accession", "fy", "fp", "form", "filed", "frame"])
    if df.empty:
        return pd.DataFrame(columns=COLUMNAS)
    for c in ("fecha_inicio", "fecha_referencia", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    mapa = {} if filings is None or filings.empty else dict(zip(filings["accession"], filings["fecha_publicacion"]))
    prudente = {f: siguiente_dia_habil(f) for f in df["filed"].dropna().unique()}
    df["fecha_publicacion"] = [mapa.get(a, prudente.get(f)) for a, f in zip(df["accession"], df["filed"])]
    df["cik"] = cik
    df["ticker"] = ticker
    df["serie"] = ticker + "." + df["taxonomia"] + ":" + df["concepto"]
    df["fuente"] = FUENTE
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df[COLUMNAS].sort_values(["serie", "fecha_referencia", "fecha_publicacion"]).reset_index(drop=True)


def vista_a_fecha(tabla: pd.DataFrame, fecha: date) -> pd.DataFrame:
    """Lo que se sabía el día ``fecha``: para cada (serie, unidad, periodo), la última cifra publicada hasta ese día."""
    visible = tabla.loc[tabla["fecha_publicacion"] <= fecha]
    clave = ["serie", "unidad", "fecha_inicio", "fecha_referencia"]
    return (visible.sort_values(["fecha_publicacion", "accession"], kind="stable")
            .drop_duplicates(clave, keep="last")
            .reset_index(drop=True))


def reexpresiones(tabla: pd.DataFrame) -> pd.DataFrame:
    """Periodos cuya cifra cambió entre publicaciones (correcciones / reclasificaciones)."""
    clave = ["serie", "unidad", "fecha_inicio", "fecha_referencia"]
    g = tabla.dropna(subset=["valor"]).groupby(clave, dropna=False)["valor"]
    resumen = g.agg(n_publicaciones="size", n_valores="nunique", primero="first", ultimo="last").reset_index()
    return resumen.loc[resumen["n_valores"] > 1].reset_index(drop=True)


def resumen_conceptos(tabla: pd.DataFrame) -> pd.DataFrame:
    """Para cada concepto clave: desde cuándo hay dato y cuántas filas (cobertura por empresa)."""
    filas = []
    for nombre, conceptos in CONCEPTOS_CLAVE.items():
        sub = tabla.loc[tabla["concepto"].isin(conceptos)]
        filas.append({
            "concepto": nombre,
            "etiquetas_usadas": ", ".join(sorted(sub["concepto"].unique())),
            "primer_periodo": sub["fecha_referencia"].min() if not sub.empty else None,
            "primera_publicacion": sub["fecha_publicacion"].min() if not sub.empty else None,
            "n_filas": len(sub),
        })
    return pd.DataFrame(filas)
