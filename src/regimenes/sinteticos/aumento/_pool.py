"""Ejecucion paralela robusta: un hijo que muere no tumba la sesion."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import multiprocessing
from typing import Any, Callable, Iterable

from regimenes.sinteticos.aumento.config import un_hilo

MAX_INTENTOS = 2


def nuevo_pool(n_jobs: int) -> ProcessPoolExecutor:
    """Pool ``spawn`` con un hilo BLAS por hijo (variables fijadas antes de crear los hijos)."""
    from regimenes.benchmark.ejecucion import _thread_limits

    with _thread_limits(1):
        return ProcessPoolExecutor(max_workers=int(n_jobs), mp_context=multiprocessing.get_context("spawn"))


def mapa(funcion: Callable[[Any], dict], lista: Iterable[Any], n_jobs: int,
         al_terminar: Callable[[Any, dict], None]) -> None:
    """Llama a ``funcion(a)`` para cada ``a`` y entrega cada resultado a ``al_terminar(a, r)``.

    ``funcion`` debe capturar sus propias excepciones. Si el pool se rompe (un
    hijo muere: fallo nativo, memoria), lo que no termino se reintenta en un pool
    nuevo hasta ``MAX_INTENTOS`` veces; despues se entrega como error.
    """
    pendientes = [(a, 0) for a in lista]
    if int(n_jobs) <= 1:
        with un_hilo():
            for a, _ in pendientes:
                al_terminar(a, funcion(a))
        return
    while pendientes:
        reintentar = []
        with nuevo_pool(n_jobs) as pool:
            futuros = {pool.submit(funcion, a): (a, k) for a, k in pendientes}
            for fut in as_completed(futuros):
                a, k = futuros[fut]
                try:
                    al_terminar(a, fut.result())
                except BrokenProcessPool as exc:
                    if k + 1 < MAX_INTENTOS:
                        reintentar.append((a, k + 1))
                    else:
                        al_terminar(a, {"estado": f"error: proceso hijo caido ({exc})"})
        pendientes = reintentar
