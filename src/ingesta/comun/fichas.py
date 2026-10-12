"""Fichas de fuente: ``results/ingesta/<tema>/fichas/<fuente>.json``.

Una ficha resume lo que se sabe (y lo que se ha medido) de una fuente. Es la fila de la tabla de
fuentes del plan de datos.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from ingesta.comun.esquema import FichaFuente
from ingesta.comun.rutas import TEMAS, resultados_dir


def guardar_ficha(ficha: FichaFuente, destino: Path | None = None) -> Path:
    """Escribe la ficha en la carpeta de su tema (atómico: tmp + replace) y devuelve su ruta."""
    carpeta = destino or resultados_dir(ficha.tema) / "fichas"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{ficha.fuente}.json"
    tmp = ruta.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(ficha), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, ruta)
    return ruta


def cargar_fichas(tema: str | None = None, carpeta: Path | None = None) -> list[dict]:
    """Fichas de un tema (o de todos si ``tema`` es None)."""
    if carpeta is not None:
        carpetas = [carpeta]
    else:
        temas = [tema] if tema else list(TEMAS)
        carpetas = [resultados_dir(t) / "fichas" for t in temas]
    return [json.loads(p.read_text(encoding="utf-8")) for c in carpetas if c.exists()
            for p in sorted(c.glob("*.json"))]
