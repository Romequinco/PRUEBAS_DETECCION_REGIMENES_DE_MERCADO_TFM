"""Ayudas minimas de torch compartidas por los generadores neuronales.

Por que existe: torch es un extra opcional (``[deep]``); CI y ``make
test-rapido`` corren sin el. Este modulo concentra el import perezoso (con un
mensaje de instalacion claro), la siembra y el bucle de entrenamiento basico
para que los cuatro generadores neuronales sean reproducibles de la misma
manera y no repitan fontaneria. NO define ninguna red: cada generador aporta
la suya en su fichero.

Reproducibilidad: ``sembrar`` fija la semilla global de torch; el orden de los
lotes sale de un ``numpy.random.Generator`` (``lotes``), no del estado global.
En ``_sample`` usa ``generador_torch(rng)`` para derivar de ``rng`` el generador
de torch, de modo que la ``random_state`` de ``sample`` controle todo el ruido.

Hilos: con varios hilos intra-op las sumas de torch cambian de orden y el
resultado deja de ser identico bit a bit entre maquinas. ``GeneradorNeuronal``
(la base de los cuatro generadores neuronales) envuelve ``fit`` y ``sample`` en
``hilos_torch(1)``: fija un hilo mientras dura la llamada y RESTAURA el valor
previo al salir, tambien si hay excepcion. Es el unico sitio donde se tocan los
hilos: ``sembrar`` ya no los cambia y nada deja ``torch.set_num_threads(1)``
como efecto global permanente del proceso.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import pandas as pd

from regimenes.sinteticos.comun import GeneradorBase

MENSAJE_TORCH = (
    "Los generadores neuronales necesitan torch (extra opcional): "
    "instalalo con `pip install -e .[deep]`."
)


def importar_torch():
    """Devuelve el modulo ``torch`` o levanta ``ImportError`` con instrucciones."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ImportError(MENSAJE_TORCH) from exc
    return torch


@contextmanager
def hilos_torch(n: int = 1) -> Iterator[None]:
    """Fija ``n`` hilos intra-op de torch dentro del bloque y restaura el valor previo al salir.

    ``n=1``: resultados identicos bit a bit con cualquier numero de nucleos.
    """
    torch = importar_torch()
    previos = torch.get_num_threads()
    torch.set_num_threads(max(int(n), 1))
    try:
        yield
    finally:
        torch.set_num_threads(previos)


def sembrar(seed: int | None):
    """Fija la semilla global de torch y devuelve un ``torch.Generator`` sembrado.

    Llamar al principio de ``_fit`` con ``self.random_state``, antes de crear la
    red: la inicializacion de pesos usa el estado global de torch. No toca los
    hilos (eso lo hace ``GeneradorNeuronal`` con ``hilos_torch``).
    """
    torch = importar_torch()
    semilla = 0 if seed is None else int(seed)
    torch.manual_seed(semilla)
    return torch.Generator().manual_seed(semilla)


class GeneradorNeuronal(GeneradorBase):
    """Base de los generadores neuronales: ``fit`` y ``sample`` corren con un hilo de torch.

    Unico punto donde se fijan los hilos (ajuste y muestreo de flow matching,
    difusion, CVAE y CGAN); el valor previo del proceso se restaura al salir.
    """

    familia = "neuronales"
    HILOS_TORCH = 1

    def fit(self, train: pd.DataFrame, regimes: pd.Series | None = None) -> "GeneradorNeuronal":
        with hilos_torch(self.HILOS_TORCH):
            return super().fit(train, regimes)

    def sample(
        self,
        n_paths: int,
        length: int,
        regimes: np.ndarray | pd.Series | dict | None = None,
        *,
        random_state: int | None = None,
    ) -> list[pd.DataFrame]:
        with hilos_torch(self.HILOS_TORCH):
            return super().sample(n_paths, length, regimes, random_state=random_state)


def generador_torch(rng: np.random.Generator):
    """``torch.Generator`` (CPU) sembrado a partir de un ``numpy.random.Generator``.

    Uso en ``_sample``: ``g = generador_torch(rng); torch.randn(..., generator=g)``.
    """
    torch = importar_torch()
    return torch.Generator().manual_seed(int(rng.integers(0, 2**63 - 1)))


def a_tensor(x: Any):
    """Array -> tensor ``float32`` en CPU."""
    torch = importar_torch()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def a_numpy(t: Any) -> np.ndarray:
    """Tensor -> array ``float64`` (sin gradiente)."""
    return t.detach().cpu().numpy().astype(np.float64)


def lotes(n: int, tam_lote: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    """Indices de una epoca: permutacion de ``range(n)`` troceada en lotes."""
    orden = rng.permutation(int(n))
    for ini in range(0, int(n), int(tam_lote)):
        yield orden[ini:ini + int(tam_lote)]


def entrenar(
    perdida_lote: Callable[[np.ndarray], Any],
    parametros: Iterable[Any],
    n: int,
    *,
    epocas: int,
    tam_lote: int = 256,
    lr: float = 1e-3,
    rng: np.random.Generator,
    registrar: Callable[..., None] | None = None,
    clip: float | None = 1.0,
    weight_decay: float = 0.0,
) -> list[float]:
    """Bucle Adam generico de una sola perdida (flow matching, difusion, CVAE).

    Parameters
    ----------
    perdida_lote:
        ``f(indices: np.ndarray) -> tensor escalar`` con la perdida del lote.
    parametros:
        Parametros a optimizar (``red.parameters()``).
    n:
        Numero de ejemplos de entrenamiento.
    registrar:
        Normalmente ``self.registrar``; recibe ``epoca=..., perdida=...`` al
        final de cada epoca (media de los lotes).
    clip:
        Norma maxima del gradiente (``None`` desactiva el recorte).

    Returns
    -------
    list[float]
        Perdida media por epoca. Levanta ``FloatingPointError`` si diverge.

    Las GAN (dos optimizadores alternos) escriben su propio bucle con ``lotes``.
    """
    torch = importar_torch()
    parametros = [p for p in parametros if p.requires_grad]
    optim = torch.optim.Adam(parametros, lr=lr, weight_decay=weight_decay)
    curva: list[float] = []
    for epoca in range(int(epocas)):
        suma, n_lotes = 0.0, 0
        for idx in lotes(n, tam_lote, rng):
            optim.zero_grad()
            perdida = perdida_lote(idx)
            perdida.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(parametros, clip)
            optim.step()
            suma += float(perdida.detach())
            n_lotes += 1
        media = suma / max(n_lotes, 1)
        if not np.isfinite(media):
            raise FloatingPointError(f"La perdida diverge en la epoca {epoca}.")
        curva.append(media)
        if registrar is not None:
            registrar(epoca=epoca, perdida=media)
    return curva
