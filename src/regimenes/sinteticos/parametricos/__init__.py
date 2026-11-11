"""Generadores parametricos: un fichero por generador, registro perezoso.

Modulos esperados (ver ``regimenes.sinteticos.registry.CATALOGO``): ``jitter``,
``bootstrap_regimen``, ``gaussiano_regimen``, ``var_regimen``, ``garch_regimen``
y ``rbig``. Este ``__init__`` no importa ninguno a proposito: se cargan bajo
demanda con ``registry.crear(nombre)``, de modo que un generador ausente o roto
no impide usar los demas.
"""

__all__: list[str] = []
