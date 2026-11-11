"""Configuracion pre-registrada del aumento, rutas y semillas estables."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import yaml

from regimenes import rutas

DIR_DATOS: Path = rutas.DATA_SINTETICOS / "aumento"          # gitignored: generadores, reservorios, paneles
DIR_RESULTADOS: Path = rutas.RESULTS_SINTETICOS / "aumento"  # versionado: CSV/JSON pequenos


def config_completa(config: dict | None = None) -> dict[str, Any]:
    """``configs/sinteticos.yaml`` entero (o ``config``)."""
    if config is None:
        config = yaml.safe_load(rutas.SINTETICOS_CONFIG.read_text(encoding="utf-8"))
    return config


def config_aumento(config: dict | None = None) -> dict[str, Any]:
    """Seccion ``aumento`` de ``configs/sinteticos.yaml`` (o de ``config``)."""
    return dict(config_completa(config)["aumento"])


def fuentes(cfg: dict | None = None) -> list[str]:
    """Generadores del aumento mas el control, en el orden del yaml."""
    cfg = config_aumento() if cfg is None else cfg
    return [*cfg["generadores"], cfg["control"]]


def detectores_pista(pista: str, cfg: dict | None = None) -> list[str]:
    """Detectores evaluables en la pista (los del registro menos ``excluir``)."""
    from regimenes.detectores.registry import detector_specs

    cfg = config_aumento() if cfg is None else cfg
    pista = pista.upper()
    fuera = set(cfg.get("excluir", {}).get(pista, []))
    return [s.detector_id for s in detector_specs(pista) if s.detector_id not in fuera]


def semilla(*partes: Any, base: int = 42) -> int:
    """Semilla estable (independiente de PYTHONHASHSEED) a partir de ``partes``."""
    texto = "|".join(str(p) for p in (base, *partes))
    return int(hashlib.sha256(texto.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)


def hash_rutas(rutas_: list[Path]) -> str:
    """Huella del texto de ficheros (normaliza finales de linea; orden por ruta)."""
    h = hashlib.sha256()
    for ruta in sorted(Path(r) for r in rutas_):
        h.update(ruta.name.encode())
        h.update(ruta.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def hash_ficheros(*nombres: str) -> str:
    """Huella del texto de modulos de este paquete."""
    return hash_rutas([Path(__file__).parent / n for n in nombres])


def huella_generador(pista: str, fuente: str, tau, cfg_total: dict | None = None) -> str:
    """Huella de un generador por ancla: todo lo que fija su ajuste, su validacion y sus muestras.

    Entra la configuracion del generador, de la validacion, del muestreo y de las
    anclas, y el codigo de ``sinteticos`` (base comun, familia del generador,
    validacion) y de ``anclas.py``/``muestras.py``. Si cambia, el ancla se vuelve a
    ajustar y sus reservorios se regeneran desde cero.
    """
    import pandas as pd

    from regimenes.sinteticos.registry import familia_de

    cfg_total = config_completa(cfg_total)
    raiz = Path(__file__).parents[1]
    codigo = [*raiz.glob("*.py"), *(raiz / familia_de(fuente)).glob("*.py"), *(raiz / "validacion").glob("*.py"),
              Path(__file__).parent / "anclas.py", Path(__file__).parent / "muestras.py"]
    aum = config_aumento(cfg_total)
    contenido = {
        "pista": pista.upper(), "fuente": fuente, "tau": pd.Timestamp(tau).date().isoformat(),
        "generador": cfg_total["generadores"][fuente], "validacion": cfg_total["validacion"],
        "muestreo": cfg_total["muestreo"], "entrenamiento": [cfg_total["entrenamiento"],
                                                            cfg_total["entrenamiento_secundario"]],
        "aumento": {k: aum[k] for k in ("semilla", "anclas", "brazos", "trayectoria_largo", "control")},
        "codigo": hash_rutas(codigo),
    }
    return hashlib.sha256(json.dumps(contenido, sort_keys=True, default=str).encode()).hexdigest()[:16]


@contextmanager
def un_hilo() -> Iterator[None]:
    """Un solo hilo BLAS/OpenMP (y torch) DENTRO del proceso, como el benchmark.

    Las variables de entorno solo actuan antes de cargar numpy; aqui se limita en
    caliente con ``threadpoolctl`` (y ``torch.set_num_threads`` si torch ya esta
    cargado), asi que vale tambien en el proceso padre y en el notebook.
    """
    import sys

    # threadpoolctl solo limita las librerias YA cargadas: scipy y sklearn traen su propio OpenBLAS y
    # OpenMP y los detectores los importan dentro de fit, asi que se cargan antes de limitar.
    import scipy.linalg  # noqa: F401
    import sklearn.mixture  # noqa: F401
    from threadpoolctl import threadpool_limits

    previo = None
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        previo = torch.get_num_threads()
        torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            yield
    finally:
        if previo is not None:
            sys.modules["torch"].set_num_threads(previo)
