"""Enlaces relativos de la documentacion.

Recorre la documentacion versionada y comprueba que cada enlace Markdown
relativo apunta a un fichero o carpeta que existe y, si lleva ancla
(``#seccion``) hacia un ``.md``, que ese encabezado (o un ``<a id>``) existe
con la normalizacion de GitHub.

Que se escanea:

- ``docs/**/*.md`` salvo las carpetas solo locales (``CARPETAS_LOCALES``);
- ``README.md`` de la raiz;
- ``src/**/README.md``;
- celdas *markdown* de ``notebooks/**/*.ipynb`` (no las salidas ni el codigo).

Que no se comprueba: URLs absolutas (``http``, ``mailto``...), enlaces dentro
de bloques o fragmentos de codigo y anclas hacia ficheros que no son ``.md`` ni
``.ipynb`` (en un notebook, las anclas se buscan en sus celdas markdown).

Sin red y sin dependencias: solo lee ficheros. Las rutas se tratan con
``pathlib`` y ``/``, asi que vale igual en Windows que en la CI de Linux.
"""
from __future__ import annotations

import json
import re
import unicodedata
import unittest
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parents[1]

# Carpetas de docs/ que solo existen en local: no se escanean porque en la CI
# no estan. docs/tfm/internos esta en el .gitignore de la raiz.
CARPETAS_LOCALES = ("docs/tfm/internos",)

# Enlaces rotos aceptados, como (fichero_origen, destino_tal_cual). Cada
# entrada debe llevar su porque. Hoy esta vacia.
#
# Referencias a rutas antiguas en ficheros que este test no escanea (son texto
# en comentarios o docstrings, no enlaces Markdown, asi que no necesitan
# entrada en LISTA_BLANCA):
#
# - Se dejan a proposito porque entran en la huella de cache del benchmark
#   (regimenes.benchmark.cache hashea los .py por bytes): cambiar un byte
#   obliga a re-ejecutarlo.
#     src/regimenes/evaluacion/metricas.py:46
#         docs/memory/01_data_and_eda.md -> solo en el tag capa1-final
#     src/regimenes/detectores/__init__.py:5
#         docs/revisiones/registro_eliminaciones.md -> solo en la historia git
# - No afectan a la huella (benchmark_spec.yaml se hashea ya parseado, sin
#   comentarios, y catalog.yaml no entra): se pueden corregir sin re-ejecutar.
#     configs/benchmark_spec.yaml:8
#         docs/datos/EDA_v2.md -> hoy docs/fase1/eda.md
#     configs/catalog.yaml:3
#         docs/datos/SOTA_datos.md -> hoy docs/estado_del_arte/fuentes_datos.md
LISTA_BLANCA: frozenset[tuple[str, str]] = frozenset()

_ESQUEMA = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_CODIGO_EN_LINEA = re.compile(r"(`+)(?:(?!\1).)+?\1")
# [texto](destino "titulo") e ![alt](destino); admite un nivel de [] anidados.
_ENLACE = re.compile(
    r"!?\[(?:[^\[\]]|\[[^\[\]]*\])*\]\(\s*(<[^>]*>|[^)\s]+)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)"
)
_DEFINICION = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(<[^>]*>|\S+)")
_HTML = re.compile(r"<(?:a|img)\b[^>]*?\b(?:href|src)\s*=\s*\"([^\"]+)\"", re.IGNORECASE)
_ID_HTML = re.compile(r"<a\b[^>]*?\b(?:id|name)\s*=\s*\"([^\"]+)\"", re.IGNORECASE)
_ENCABEZADO = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")


def _lineas(texto: str, *, quitar_codigo_en_linea: bool = True) -> list[tuple[int, str]]:
    """Lineas numeradas fuera de bloques de codigo (``` o ~~~).

    Con ``quitar_codigo_en_linea`` se borran ademas los fragmentos `entre comillas
    invertidas` (para buscar enlaces); sin el, se conservan (cuentan en el ancla).
    """
    lineas, dentro, marca = [], False, ""
    for n, linea in enumerate(texto.splitlines(), 1):
        m = _FENCE.match(linea)
        if m:
            if not dentro:
                dentro, marca = True, m.group(1)[0]
                continue
            if m.group(1)[0] == marca:
                dentro = False
                continue
        if dentro:
            continue
        lineas.append((n, _CODIGO_EN_LINEA.sub("", linea) if quitar_codigo_en_linea else linea))
    return lineas


def _slug(titulo: str) -> str:
    """Ancla de un encabezado como la genera GitHub."""
    t = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", titulo)  # enlaces -> su texto
    t = re.sub(r"<[^>]+>", "", t)  # etiquetas HTML
    t = t.strip().lower()
    salida = []
    for c in t:
        if c == " ":
            salida.append("-")
        elif c in "-_" or unicodedata.category(c)[0] in "LNM":
            salida.append(c)
    return "".join(salida)


def _anclas_de_texto(texto: str) -> set[str]:
    anclas: set[str] = set()
    vistos: dict[str, int] = {}
    for _, linea in _lineas(texto, quitar_codigo_en_linea=False):
        m = _ENCABEZADO.match(linea)
        if m:
            base = _slug(m.group(2))
            k = vistos.get(base, 0)
            anclas.add(base if k == 0 else f"{base}-{k}")
            vistos[base] = k + 1
        anclas.update(a for a in _ID_HTML.findall(linea))
    return anclas


@lru_cache(maxsize=None)
def _anclas(ruta: Path) -> frozenset[str]:
    if ruta.suffix == ".ipynb":
        return frozenset(_anclas_de_texto("\n".join(_celdas_markdown(ruta))))
    return frozenset(_anclas_de_texto(ruta.read_text(encoding="utf-8")))


def _celdas_markdown(ruta: Path) -> list[str]:
    nb = json.loads(ruta.read_text(encoding="utf-8"))
    celdas = []
    for c in nb.get("cells", []):
        if c.get("cell_type") == "markdown":
            src = c.get("source", "")
            celdas.append("".join(src) if isinstance(src, list) else src)
    return celdas


def _es_local(ruta: Path) -> bool:
    rel = ruta.relative_to(REPO).as_posix()
    return any(rel == c or rel.startswith(c + "/") for c in CARPETAS_LOCALES)


def _ficheros() -> list[Path]:
    md = [p for p in (REPO / "docs").rglob("*.md") if not _es_local(p)]
    md.append(REPO / "README.md")
    md += list((REPO / "src").rglob("README.md"))
    nbs = [p for p in (REPO / "notebooks").rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts]
    return sorted(set(md)) + sorted(nbs)


def _destinos(texto: str) -> list[tuple[int, str]]:
    salida = []
    for n, linea in _lineas(texto):
        for m in _ENLACE.finditer(linea):
            salida.append((n, m.group(1)))
        m = _DEFINICION.match(linea)
        if m:
            salida.append((n, m.group(1)))
        for m in _HTML.finditer(linea):
            salida.append((n, m.group(1)))
    return salida


def _comprobar(origen: Path, destino: str) -> str | None:
    """Devuelve el motivo de la rotura o ``None`` si el enlace es valido."""
    d = destino.strip("<>").strip()
    if not d or _ESQUEMA.match(d):
        return None
    ruta_txt, _, ancla = d.partition("#")
    ruta_txt = unquote(ruta_txt.split("?")[0])
    if not ruta_txt:
        objetivo = origen
    elif ruta_txt.startswith("/"):
        objetivo = REPO / ruta_txt.lstrip("/")
    else:
        objetivo = origen.parent / ruta_txt
    try:
        objetivo = objetivo.resolve()
        objetivo.relative_to(REPO)
    except ValueError:
        return "sale del repositorio"
    if not objetivo.exists():
        return "no existe"
    if _es_local(objetivo):
        return "apunta a una carpeta solo local"
    if ancla and objetivo.is_file() and objetivo.suffix in (".md", ".ipynb"):
        if unquote(ancla).lower() not in {a.lower() for a in _anclas(objetivo)}:
            return f"no existe el ancla #{ancla}"
    return None


def enlaces_rotos() -> list[str]:
    rotos = []
    for f in _ficheros():
        rel = f.relative_to(REPO).as_posix()
        if f.suffix == ".ipynb":
            pares = []
            for i, celda in enumerate(_celdas_markdown(f)):
                pares += [(f"celda md {i}, l.{n}", d) for n, d in _destinos(celda)]
        else:
            pares = [(f"l.{n}", d) for n, d in _destinos(f.read_text(encoding="utf-8"))]
        for donde, destino in pares:
            if (rel, destino) in LISTA_BLANCA:
                continue
            motivo = _comprobar(f, destino)
            if motivo:
                rotos.append(f"{rel} ({donde}): {destino} -> {motivo}")
    return rotos


class EnlacesDocsTests(unittest.TestCase):
    def test_slug_como_github(self) -> None:
        self.assertEqual(_slug("1. Teoría"), "1-teoría")
        self.assertEqual(_slug("Pipeline 00–20"), "pipeline-0020")
        self.assertEqual(
            _slug("Términos del ranking (columnas de `ranking_v2.csv` / `metrics_master_v2.csv`)"),
            "términos-del-ranking-columnas-de-ranking_v2csv--metrics_master_v2csv",
        )

    def test_no_hay_enlaces_rotos(self) -> None:
        rotos = enlaces_rotos()
        self.assertEqual(rotos, [], "Enlaces rotos:\n" + "\n".join(rotos))

    def test_lista_blanca_sin_entradas_caducadas(self) -> None:
        # Cada excepcion debe seguir siendo necesaria: si el enlace ya resuelve, sobra.
        for origen, destino in LISTA_BLANCA:
            with self.subTest(origen=origen, destino=destino):
                self.assertIsNotNone(_comprobar(REPO / origen, destino))


if __name__ == "__main__":
    for r in enlaces_rotos():
        print(r)
