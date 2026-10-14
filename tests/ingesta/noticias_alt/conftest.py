"""Fixtures comunes del Tema 4 (sin red).

``sin_red`` aísla corpus y results en ``tmp_path``. ``red_falsa(modulo, rutas)`` sustituye ``get`` del
módulo por un falso que sirve ficheros de ``fixtures/`` según un trozo de la URL y falla ante una
URL no prevista (así ningún test toca la red por accidente).
"""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class RespFalsa:
    def __init__(self, contenido: bytes, status_code: int = 200, headers: dict | None = None):
        self.content = contenido
        self.status_code = status_code
        self.headers = headers or {}

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self):
        import json

        return json.loads(self.content)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}", response=self)  # type: ignore[arg-type]


@pytest.fixture
def sin_red(tmp_path, monkeypatch):
    """Corpus y results en tmp; devuelve ``tmp_path``."""
    import ingesta.comun.rutas as rutas

    monkeypatch.setenv("INGESTA_CORPUS_DIR", str(tmp_path / "corpus"))
    monkeypatch.setattr(rutas, "ENV_FILE", tmp_path / "no_existe.env")
    monkeypatch.setattr(rutas, "RESULTS_INGESTA", tmp_path / "results")
    return tmp_path


@pytest.fixture
def red_falsa(monkeypatch):
    """``red_falsa(modulo, {trozo_url: fichero_o_bytes})`` parchea ``modulo.get``."""

    def _instalar(modulo, rutas: dict[str, str | bytes | tuple[int, bytes]]):
        def _get(url, **kw):
            params = kw.get("params") or {}
            completa = url + ("?" + "&".join(f"{k}={v}" for k, v in params.items()) if params else "")
            for trozo, destino in rutas.items():
                if trozo in completa:
                    if isinstance(destino, tuple):
                        return RespFalsa(destino[1], status_code=destino[0])
                    if isinstance(destino, bytes):
                        return RespFalsa(destino)
                    return RespFalsa((FIXTURES / destino).read_bytes())
            raise AssertionError(f"URL no prevista en el test: {completa}")

        monkeypatch.setattr(modulo, "get", _get)
        return _get

    return _instalar
