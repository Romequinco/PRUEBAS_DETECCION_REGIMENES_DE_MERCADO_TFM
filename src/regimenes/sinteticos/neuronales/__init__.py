"""Generadores neuronales (extra ``[deep]``): un fichero por generador, registro perezoso.

Modulos esperados (ver ``regimenes.sinteticos.registry.CATALOGO``):
``flow_matching``, ``difusion``, ``cvae`` y ``cgan``; ``_torch`` reune el import
perezoso de torch, la siembra y el bucle de entrenamiento comun. Este
``__init__`` no importa torch ni ningun generador: se cargan bajo demanda con
``registry.crear(nombre)``, asi el paquete funciona sin torch instalado.
"""

__all__: list[str] = []
