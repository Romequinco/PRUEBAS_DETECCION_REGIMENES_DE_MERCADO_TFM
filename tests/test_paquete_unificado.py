"""Guardas del paquete unificado ``regimenes`` (ADR-004).

Sustituye a ``test_clean_sys_path_removes_legacy_root_temporarily`` (eliminado con
fila en docs/revisiones/registro_eliminaciones.md): con el paquete instalado ya no
existe manipulacion de ``sys.path`` que limpiar, asi que se verifica justamente su
ausencia en el codigo del paquete y en ``factory()``.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import regimenes
from regimenes.detectores.registry import detector_specs

OPCIONALES = {"torch", "jumpmodels"}  # extras [deep] y [jump] (no se instalan en CI)
PKG_DIR = Path(regimenes.__file__).resolve().parent
PROHIBIDO = re.compile(
    r"^\s*(from\s+src[\s.]|import\s+src\b|from\s+detectors\b)"
    r"|sys\.path|capa1_exploracion",
    re.MULTILINE,
)


class PaqueteUnificadoTests(unittest.TestCase):
    def test_package_source_has_no_legacy_imports_or_sys_path(self) -> None:
        hits = []
        for py in sorted(PKG_DIR.rglob("*.py")):
            for m in PROHIBIDO.finditer(py.read_text(encoding="utf-8")):
                hits.append(f"{py.relative_to(PKG_DIR)}: {m.group(0).strip()}")
        self.assertEqual(hits, [])

    def test_factory_does_not_touch_sys_path(self) -> None:
        before = list(sys.path)
        for spec in detector_specs("A") + detector_specs("B"):
            try:
                spec.factory()
            except ModuleNotFoundError as exc:
                # CI sin extras [deep]/[jump]: D12 importa torch al construirse.
                if exc.name not in OPCIONALES:
                    raise
        self.assertEqual(sys.path, before)


if __name__ == "__main__":
    unittest.main()
