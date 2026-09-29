"""``python -m regimenes.benchmark``: ver :mod:`regimenes.benchmark.cli`.

Las funciones que viajan a los procesos hijos (``--jobs``) viven en
``regimenes.benchmark.ejecucion``, así que se serializan por su nombre de módulo
real y no como ``__main__.*``.
"""

from regimenes.benchmark.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
