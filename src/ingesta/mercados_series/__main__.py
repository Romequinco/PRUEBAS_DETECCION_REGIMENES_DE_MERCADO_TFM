"""CLI del Tema 1: ``python -m ingesta.mercados_series <orden>``.

- ``construir-base [--sin-red] [--reemplazar] [--purgar] [--sin-escribir]``: une fase1 + FRED +
  académicos (``base.construir_base``), escribe ``<corpus>/mercados_series/series/base_series.parquet``
  y ``results/ingesta/mercados_series/tablas/cobertura_series.csv``. ``--sin-red``: solo crudos ya
  descargados y ``data/raw``. Las revisiones de valores de las revisables con vintage ``unico``
  (ediciones académicas) se admiten siempre; ``--reemplazar`` permite cambiar cualquier fila
  guardada (correcciones propias) y ``--purgar`` quitar las series guardadas que ya no se
  construyen. ``--sin-escribir``: no escribe nada (ni el parquet ni la cobertura).
- ``resumen [--csv]``: recuentos de la base guardada (y, con ``--csv``, rehace la tabla de cobertura).
- ``excepciones-alfred [--ids ...] [--sin-red]``: contrasta las reglas de publicación de las series de
  FRED con la primera vintage de ALFRED (``fuentes.fred.medir_excepciones``) y escribe
  ``config/excepciones_publicacion.csv`` y ``tablas/publicacion_alfred_medida.csv`` (solo fechas).
  Reanudable: lo descargado hoy se sirve de la caché.

This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence


def _imprimir(d: dict) -> None:
    from ingesta.mercados_series.base import _np_a_py

    print(json.dumps(_np_a_py(d), indent=2, ensure_ascii=False, default=str))


def _construir(a: argparse.Namespace) -> int:
    from ingesta.mercados_series import base

    resumen, tabla = base.construir_base(sin_red=a.sin_red, escribir=not a.sin_escribir,
                                         permitir_perdidas=a.reemplazar, purgar=a.purgar, registro=print)
    if a.sin_escribir:
        print("ensayo (--sin-escribir): no se escribe ni la base ni la cobertura")
    else:
        print(f"cobertura -> {base.escribir_cobertura(base.cobertura(tabla))}")
        print(f"anomalías -> {base.escribir_anomalias(resumen.detalle_anomalias)}")
    _imprimir({k: v for k, v in resumen.como_dict().items() if k != "detalle_anomalias"})
    print(base.AVISO_FRED)
    return 0


def _resumen(a: argparse.Namespace) -> int:
    from ingesta.mercados_series import almacen, base

    ruta = almacen.ruta_series()
    if not ruta.exists():
        print(f"no hay base en {ruta}: python -m ingesta.mercados_series construir-base --sin-red", file=sys.stderr)
        return 1
    tabla = almacen.leer_series()
    print(f"base: {ruta}")
    _imprimir(base.resumen_base(tabla))
    if a.csv:
        print(f"cobertura -> {base.escribir_cobertura(base.cobertura(tabla))}")
    return 0


def _excepciones(a: argparse.Namespace) -> int:
    import pandas as pd

    from ingesta.comun.rutas import resultados_dir
    from ingesta.mercados_series import catalogo
    from ingesta.mercados_series.base import AVISO_FRED
    from ingesta.mercados_series.fuentes import fred

    cliente = fred.ClienteFred()
    if a.sin_red:
        cliente._transporte = _sin_red
    exc, medida = fred.medir_excepciones(cliente, a.ids or None, registro=print)
    exc = exc[list(catalogo.COLUMNAS_EXCEPCIONES)]
    if a.ids:   # actualización parcial: conserva las excepciones del resto de series
        previas = catalogo.cargar_excepciones()
        exc = pd.concat([previas[~previas["serie"].isin(a.ids)], exc], ignore_index=True)
    exc = exc.copy()
    for c in ("fecha_referencia", "fecha_publicacion"):
        exc[c] = pd.to_datetime(exc[c]).dt.strftime("%Y-%m-%d")
    exc = exc.sort_values(["serie", "fecha_referencia"], kind="mergesort")
    exc.to_csv(catalogo.RUTA_EXCEPCIONES, index=False, encoding="utf-8", lineterminator="\n")
    carpeta = resultados_dir("mercados_series") / "tablas"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_medida = carpeta / "publicacion_alfred_medida.csv"
    medida["origen"] = f"{fred.ORIGEN_EXCEPCION}_{cliente.hoy:%Y%m%d}"
    if a.ids and ruta_medida.exists():
        previa = pd.read_csv(ruta_medida)
        medida = pd.concat([previa[~previa["serie"].isin(a.ids)], medida], ignore_index=True)
    medida.sort_values("serie").to_csv(ruta_medida, index=False, encoding="utf-8", lineterminator="\n")
    errores = int(medida["error"].notna().sum()) if "error" in medida else 0
    print(f"{len(exc)} excepciones -> {catalogo.RUTA_EXCEPCIONES}\nmedida -> {ruta_medida}; "
          f"{cliente.peticiones} peticiones; {errores} series con error (repetir la orden las reintenta)")
    print(AVISO_FRED)
    return 1 if errores else 0


def _sin_red(url: str, params: dict) -> dict:
    raise RuntimeError("sin red (--sin-red): falta en la caché de hoy")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m ingesta.mercados_series",
                                description="Tema 1 · base de series (bloque 2)")
    sub = p.add_subparsers(dest="orden", required=True)
    c = sub.add_parser("construir-base", help="une fase1 + FRED + académicos y escribe la base")
    c.add_argument("--sin-red", action="store_true", help="solo crudos ya descargados y data/raw")
    c.add_argument("--reemplazar", action="store_true",
                   help="permite cambiar filas guardadas (revisiones del último vintage, errores propios)")
    c.add_argument("--purgar", action="store_true",
                   help="quita de la base guardada las series que ya no se construyen (las enumera)")
    c.add_argument("--sin-escribir", action="store_true",
                   help="construye y valida sin escribir nada (ni el parquet ni la cobertura)")
    c.set_defaults(func=_construir)
    r = sub.add_parser("resumen", help="recuentos de la base guardada")
    r.add_argument("--csv", action="store_true", help="rehace tablas/cobertura_series.csv")
    r.set_defaults(func=_resumen)
    x = sub.add_parser("excepciones-alfred", help="fechas de publicación medidas en ALFRED (excepciones)")
    x.add_argument("--ids", nargs="*", help="ids del catálogo (por defecto fuentes.fred.ids_excepciones())")
    x.add_argument("--sin-red", action="store_true", help="solo lo descargado hoy (caché)")
    x.set_defaults(func=_excepciones)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
