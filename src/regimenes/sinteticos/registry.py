"""Registro de generadores sinteticos con catalogo perezoso.

Por que es perezoso: los generadores neuronales necesitan torch (extra
``[deep]``) y cada generador vive en su propio fichero. Importar
``regimenes.sinteticos`` no debe importar torch ni fallar porque un generador
aun no exista. ``CATALOGO`` declara nombre -> (modulo, familia) y ``crear``
importa el modulo bajo demanda; el modulo se registra al importarse con el
decorador ``@registrar``.

Uso::

    from regimenes.sinteticos import registry
    gen = registry.crear("bootstrap_regimen", longitud_media=21)
    registry.disponibles()      # nombres cuyo modulo existe en disco
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

from regimenes.sinteticos.base import Generador

_PAQUETE = "regimenes.sinteticos"

# nombre -> (modulo dentro de regimenes.sinteticos, familia)
CATALOGO: dict[str, tuple[str, str]] = {
    "jitter": ("parametricos.jitter", "parametricos"),
    "bootstrap_regimen": ("parametricos.bootstrap_regimen", "parametricos"),
    "gaussiano_regimen": ("parametricos.gaussiano_regimen", "parametricos"),
    "var_regimen": ("parametricos.var_regimen", "parametricos"),
    "garch_regimen": ("parametricos.garch_regimen", "parametricos"),
    "rbig": ("parametricos.rbig", "parametricos"),
    "flow_matching": ("neuronales.flow_matching", "neuronales"),
    "difusion": ("neuronales.difusion", "neuronales"),
    "cvae": ("neuronales.cvae", "neuronales"),
    "cgan": ("neuronales.cgan", "neuronales"),
}

GENERADORES: dict[str, type[Generador]] = {}


def _clave(cls: type[Generador]) -> str:
    nombre = getattr(cls, "nombre", None)
    if isinstance(nombre, str) and nombre:
        return nombre
    try:
        nombre = cls().name
    except Exception as exc:  # pragma: no cover - mensaje de ayuda
        raise TypeError(
            f"{cls.__name__}: define el atributo de clase `nombre` para poder registrarlo."
        ) from exc
    return str(nombre)


def registrar(cls: type[Generador]) -> type[Generador]:
    """Decorador: registra ``cls`` bajo ``cls.nombre`` (o su ``name``).

    Registrar dos clases distintas con el mismo nombre es un error; volver a
    registrar la misma clase (recarga del modulo) sustituye la entrada.
    """
    if not (isinstance(cls, type) and issubclass(cls, Generador)):
        raise TypeError("Solo se registran subclases de Generador.")
    clave = _clave(cls)
    previo = GENERADORES.get(clave)
    if previo is not None and (previo.__module__, previo.__qualname__) != (cls.__module__, cls.__qualname__):
        raise ValueError(
            f"Ya hay un generador {clave!r} registrado ({previo.__module__}.{previo.__qualname__})."
        )
    GENERADORES[clave] = cls
    return cls


def modulo_de(nombre: str) -> str:
    """Ruta completa del modulo que implementa ``nombre`` segun el catalogo."""
    return f"{_PAQUETE}.{CATALOGO[nombre][0]}"


def familia_de(nombre: str) -> str:
    """Familia (``parametricos`` | ``neuronales``) de ``nombre`` segun el catalogo."""
    return CATALOGO[nombre][1]


def existe_modulo(nombre: str) -> bool:
    """True si el fichero del generador ``nombre`` existe (no lo importa)."""
    try:
        return find_spec(modulo_de(nombre)) is not None
    except ModuleNotFoundError:
        return False


def disponibles() -> list[str]:
    """Nombres utilizables: los ya registrados mas los del catalogo con modulo en disco."""
    return sorted(set(GENERADORES) | {n for n in CATALOGO if existe_modulo(n)})


def clase(nombre: str) -> type[Generador]:
    """Clase del generador ``nombre``, importando su modulo si hace falta.

    Raises
    ------
    KeyError
        Nombre desconocido (ni registrado ni en el catalogo).
    ModuleNotFoundError
        El generador esta en el catalogo pero su fichero aun no existe.
    ImportError
        Falta una dependencia opcional (torch: ``pip install -e .[deep]``).
    """
    if nombre in GENERADORES:
        return GENERADORES[nombre]
    if nombre not in CATALOGO:
        raise KeyError(
            f"Generador desconocido {nombre!r}. Catalogo: {sorted(CATALOGO)}; "
            f"registrados: {sorted(GENERADORES)}."
        )
    modulo = modulo_de(nombre)
    try:
        import_module(modulo)
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == modulo or modulo.startswith(exc.name + ".")):
            raise ModuleNotFoundError(
                f"El generador {nombre!r} aun no esta implementado: falta el modulo {modulo} "
                f"(src/{modulo.replace('.', '/')}.py).",
                name=exc.name,
            ) from exc
        if exc.name and exc.name.split(".")[0] == "torch":
            raise ImportError(
                f"El generador {nombre!r} necesita torch: instala el extra con `pip install -e .[deep]`."
            ) from exc
        raise
    if nombre not in GENERADORES:
        raise RuntimeError(
            f"El modulo {modulo} no registro ningun generador {nombre!r}: "
            f"decora la clase con @registrar y define nombre = {nombre!r}."
        )
    return GENERADORES[nombre]


def crear(nombre: str, **params) -> Generador:
    """Instancia el generador ``nombre`` con ``params`` (ver ``clase`` para los errores)."""
    return clase(nombre)(**params)
