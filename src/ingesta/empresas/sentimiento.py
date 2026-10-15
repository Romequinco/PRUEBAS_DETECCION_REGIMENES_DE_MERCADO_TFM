"""Sentimiento del lenguaje financiero con el diccionario Loughran-McDonald (LM) y ``risk_delta``.

LM es el diccionario estándar para textos financieros (Loughran y McDonald, 2011): listas de
palabras negativas, positivas, de incertidumbre, litigiosas, etc., construidas sobre 10-K. Los
diccionarios generales fallan aquí: "liability" o "tax" no son negativas en un balance.

El diccionario se descarga a mano (gratis para uso académico) desde
https://sraf.nd.edu/loughranmcdonald-master-dictionary/ y se deja en
``<corpus>/empresas/lm/`` (``Loughran-McDonald_MasterDictionary_*.csv``). No se versiona.

En el CSV cada categoría es una columna con el año en que se añadió la palabra (0 = no
pertenece; año negativo = retirada).

``risk_delta``: cuánto cambia la sección de riesgos (Item 1A) de un año al siguiente. Las
empresas copian casi literal su Item 1A; un cambio grande suele anunciar novedades (Cohen,
Malloy y Nguyen, 2020, *Lazy Prices*). Se mide como 1 − similitud coseno de las frecuencias de
palabras entre los dos textos.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from ingesta.comun.rutas import corpus_dir

CATEGORIAS = ("Negative", "Positive", "Uncertainty", "Litigious", "Strong_Modal", "Weak_Modal", "Constraining")
_PALABRA = re.compile(r"[A-Za-z]+")


def ruta_diccionario() -> Path | None:
    """CSV de LM en ``<corpus>/empresas/lm/`` (el más reciente si hay varios) o ``None``."""
    candidatos = sorted((corpus_dir("empresas") / "lm").glob("Loughran-McDonald_MasterDictionary*.csv"))
    return candidatos[-1] if candidatos else None


def cargar_lm(ruta: Path | None = None) -> dict[str, frozenset[str]]:
    """``{categoría: palabras en MAYÚSCULAS}``. Lanza ``FileNotFoundError`` si no está el CSV."""
    ruta = ruta or ruta_diccionario()
    if ruta is None or not Path(ruta).exists():
        raise FileNotFoundError(
            "Falta el diccionario Loughran-McDonald: descárgalo de "
            "https://sraf.nd.edu/loughranmcdonald-master-dictionary/ y déjalo en "
            f"{corpus_dir('empresas') / 'lm'}"
        )
    df = pd.read_csv(ruta, usecols=lambda c: c in ("Word", *CATEGORIAS), keep_default_na=False)
    palabras = df["Word"].astype(str).str.upper()
    return {c: frozenset(palabras[pd.to_numeric(df[c], errors="coerce").fillna(0) > 0]) for c in CATEGORIAS if c in df}


def tokenizar(texto: str) -> list[str]:
    return [p.upper() for p in _PALABRA.findall(texto)]


def puntuar(texto: str, lm: dict[str, frozenset[str]]) -> dict[str, float]:
    """Proporción de palabras de cada categoría + ``tono`` = (pos − neg) / (pos + neg)."""
    tokens = tokenizar(texto)
    n = len(tokens)
    cuenta = Counter(tokens)
    salida: dict[str, float] = {"n_palabras": n}
    for cat, vocab in lm.items():
        k = sum(v for w, v in cuenta.items() if w in vocab)
        salida[f"lm_{cat.lower()}"] = k / n if n else 0.0
    pos, neg = salida.get("lm_positive", 0.0), salida.get("lm_negative", 0.0)
    salida["lm_tono"] = (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0
    return salida


def similitud_coseno(a: str, b: str) -> float:
    ca, cb = Counter(tokenizar(a)), Counter(tokenizar(b))
    if not ca or not cb:
        return 0.0
    producto = sum(v * cb.get(w, 0) for w, v in ca.items())
    return producto / (math.sqrt(sum(v * v for v in ca.values())) * math.sqrt(sum(v * v for v in cb.values())))


def risk_delta(actual: str, anterior: str | None) -> dict[str, float | None]:
    """Cambio de la sección respecto a la del filing anterior (``None`` si no hay anterior)."""
    if not anterior:
        return {"similitud_anterior": None, "risk_delta": None, "cambio_longitud": None}
    s = similitud_coseno(actual, anterior)
    la, lb = len(actual.split()), len(anterior.split())
    return {"similitud_anterior": s, "risk_delta": 1.0 - s, "cambio_longitud": (la - lb) / lb if lb else None}
