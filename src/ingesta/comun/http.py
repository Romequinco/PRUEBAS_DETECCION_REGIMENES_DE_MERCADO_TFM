"""Cliente HTTP educado para las sondas.

- User-Agent identificable (SEC EDGAR lo exige con contacto: ``SEC_EDGAR_USER_AGENT``).
- Límite de ritmo por host (por defecto 1 petición/s; SEC permite hasta 10/s).
- Reintentos con espera creciente.
- ``robots_permite``: consulta robots.txt antes de rastrear un sitio sin API.

No se esquivan CAPTCHAs, muros de pago, logins ni protecciones anti-bot: si una fuente las
pone, la sonda la marca como ``fallo`` y se documenta.
"""
from __future__ import annotations

import re
import time
import urllib.robotparser
from functools import lru_cache
from urllib.parse import urlparse

import requests

from ingesta.comun.rutas import env

UA_DEFECTO = "TFM-MIAX-regimenes (investigacion academica)"
_ultimo: dict[str, float] = {}
_CLAVE = re.compile(r"((?:api_key|apikey|api-key|key|token)=)[^&\s'\"]+", re.IGNORECASE)


def sin_clave(texto: str) -> str:
    """Enmascara claves de API en URLs o mensajes (``api_key=***``)."""
    return _CLAVE.sub(lambda m: m.group(1) + "***", texto)


def user_agent() -> str:
    return env("SEC_EDGAR_USER_AGENT", UA_DEFECTO) or UA_DEFECTO


def _esperar(host: str, por_segundo: float) -> None:
    minimo = 1.0 / por_segundo
    t = _ultimo.get(host)
    if t is not None:
        falta = minimo - (time.monotonic() - t)
        if falta > 0:
            time.sleep(falta)
    _ultimo[host] = time.monotonic()


def get(url: str, *, params: dict | None = None, por_segundo: float = 1.0, reintentos: int = 3,
        timeout: int = 30, headers: dict | None = None) -> requests.Response:
    """GET con User-Agent, ritmo por host y reintentos. Lanza si el estado final no es 2xx."""
    host = urlparse(url).netloc
    h = {"User-Agent": user_agent(), "Accept-Encoding": "gzip, deflate"}
    if headers:
        h.update(headers)
    ultimo_error: Exception | None = None
    for k in range(reintentos):
        _esperar(host, por_segundo)
        try:
            r = requests.get(url, params=params, headers=h, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} en {url}", response=r)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            ultimo_error = e
            time.sleep(2.0 * (k + 1))
    assert ultimo_error is not None
    # El mensaje original lleva la URL completa: se re-lanza sin claves en claro.
    raise type(ultimo_error)(sin_clave(str(ultimo_error))) from None


@lru_cache(maxsize=64)
def _robots(base: str) -> urllib.robotparser.RobotFileParser:
    """Lee robots.txt con NUESTRO User-Agent (el de urllib lo rechazan algunos sitios con 403 y
    ``RobotFileParser.read`` lo interpretaría como «todo prohibido»).

    Semántica estándar: 401/403 → todo prohibido; otro error o 404 → todo permitido.
    """
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(base + "/robots.txt")
    try:
        r = requests.get(base + "/robots.txt", headers={"User-Agent": user_agent()}, timeout=20)
    except requests.RequestException:
        rp.allow_all = True
        return rp
    if r.status_code in (401, 403):
        rp.disallow_all = True
    elif r.status_code >= 400:
        rp.allow_all = True
    else:
        rp.parse(r.text.splitlines())
    return rp


def robots_permite(url: str) -> bool:
    """True si robots.txt del host permite rastrear ``url`` con nuestro User-Agent."""
    p = urlparse(url)
    return _robots(f"{p.scheme}://{p.netloc}").can_fetch(user_agent(), url)
