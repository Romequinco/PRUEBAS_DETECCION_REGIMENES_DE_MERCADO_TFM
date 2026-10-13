"""CLI del Tema 1: ``python -m ingesta.mercados_series <orden>``.

- ``descargar-alfred [--ids ...] [--limite-segundos N] [--refrescar]``: una tanda (reanudable) de
  vintages de ALFRED (``fuentes.alfred.descargar_vintages``); las series con error se reintentan una
  vez. Sale con 2 si quedan series pendientes (repetir la orden sigue donde se quedó) y con 1 si
  alguna sigue con error. Cada tanda deja una fila en ``<corpus>/mercados_series/logs/descargas.csv``.
- ``descargar-fred-vivo [--ids ...] [--max-minutos N] [--completa]``: ídem con la fuente viva de FRED
  (``fuentes.fred.descargar_fred_vivo``: completa la primera vez, incremental después).
- ``construir-base [--sin-red] [--desde-cero] [--reemplazar] [--purgar] [--sin-escribir]``: une ALFRED +
  FRED vivo + académicos + fase1 (``base.construir_base``), escribe
  ``<corpus>/mercados_series/series/base_series.parquet`` y
  ``results/ingesta/mercados_series/tablas/cobertura_series.csv``. ``--sin-red``: solo crudos ya
  descargados y ``data/raw``. ``--desde-cero``: copia la base guardada a
  ``series/respaldo_bloque2.parquet`` (si no existe ya) y la sustituye entera (cambio de reglas o
  fuentes). Las revisiones de valores de las revisables con vintage ``unico`` se admiten siempre;
  ``--reemplazar`` permite cambiar cualquier fila guardada (correcciones propias) y ``--purgar``
  quitar las series guardadas que ya no se construyen. ``--sin-escribir``: no escribe nada.
- ``informe-alfred``: desde la base guardada, ``tablas/alfred_revisiones_resumen.csv`` (revisiones por
  serie, en %, sin niveles) y ``tablas/alfred_vs_lag.csv`` (adelantos de la regla previa frente a la
  primera publicación real).
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
from pathlib import Path


def _imprimir(d: dict) -> None:
    from ingesta.mercados_series.base import _np_a_py

    print(json.dumps(_np_a_py(d), indent=2, ensure_ascii=False, default=str))


def _registrar_descarga(fila: dict) -> Path:
    """Añade una fila (tanda de descarga, sin valores) a ``<corpus>/mercados_series/logs/descargas.csv``."""
    import pandas as pd

    from ingesta.mercados_series import almacen

    carpeta = almacen.corpus_dir(almacen.TEMA) / "logs"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "descargas.csv"
    pd.DataFrame([fila]).to_csv(ruta, mode="a", header=not ruta.exists(), index=False, encoding="utf-8")
    return ruta


def _descargar_alfred(a: argparse.Namespace) -> int:
    import time

    import pandas as pd

    from ingesta.mercados_series import catalogo
    from ingesta.mercados_series.fuentes import alfred

    cat = catalogo.cargar_catalogo()
    cliente = alfred.cliente_alfred()
    t0 = time.monotonic()
    rep = alfred.descargar_vintages(a.ids or None, cliente=cliente, cat=cat, refrescar=a.refrescar,
                                    limite_segundos=a.limite_segundos, registro=print)
    fallidas = rep.loc[rep["estado"] == "error", "serie"].tolist()
    reintentadas = []
    if fallidas and (a.limite_segundos is None or time.monotonic() - t0 < a.limite_segundos):
        print(f"reintento (una vez) de {len(fallidas)} series con error: {fallidas}")
        otra = alfred.descargar_vintages(fallidas, cliente=cliente, cat=cat, registro=print)
        rep = pd.concat([rep[rep["estado"] != "error"], otra], ignore_index=True)
        reintentadas = fallidas
    estados = {k: int(v) for k, v in rep["estado"].value_counts().items()}
    errores = rep.loc[rep["estado"] == "error", ["serie", "error"]].to_dict("records")
    pendientes = rep.loc[rep["estado"] == "pendiente", "serie"].tolist()
    fila = {"fecha": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "orden": "descargar-alfred",
            "peticiones": cliente.peticiones, "segundos": round(time.monotonic() - t0, 1),
            "descargadas": estados.get("descargada", 0), "archivo": estados.get("archivo", 0),
            "pendientes": len(pendientes), "errores": len(errores), "reintentadas": len(reintentadas),
            "series_error": " ".join(e["serie"] for e in errores)}
    print(f"tanda ALFRED: {estados}; {cliente.peticiones} peticiones en {fila['segundos']} s; log -> "
          f"{_registrar_descarga(fila)}")
    if errores:
        print("series con error tras el reintento:", errores)
        return 1
    if pendientes:
        print(f"quedan {len(pendientes)} series: repite la orden para seguir")
        return 2
    return 0


def _descargar_fred_vivo(a: argparse.Namespace) -> int:
    import time

    import pandas as pd

    from ingesta.mercados_series import catalogo
    from ingesta.mercados_series.fuentes import fred

    cat = catalogo.cargar_catalogo()
    cliente = fred.ClienteFred()
    t0 = time.monotonic()
    rep = fred.descargar_fred_vivo(a.ids or None, cliente=cliente, cat=cat, completa=a.completa,
                                   max_segundos=a.max_minutos * 60 if a.max_minutos else None, registro=print)
    reintentadas = sorted(rep["errores"])
    if reintentadas:
        print(f"reintento (una vez) de {len(reintentadas)} series con error: {reintentadas}")
        otra = fred.descargar_fred_vivo(reintentadas, cliente=cliente, cat=cat, completa=a.completa, registro=print)
        rep["errores"] = otra["errores"]
        rep["hechas"] += otra["hechas"]
    fila = {"fecha": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "orden": "descargar-fred-vivo",
            "peticiones": cliente.peticiones, "segundos": round(time.monotonic() - t0, 1),
            "descargadas": len(rep["hechas"]), "archivo": len(rep["cache"]), "pendientes": len(rep["pendientes"]),
            "errores": len(rep["errores"]), "reintentadas": len(reintentadas),
            "series_error": " ".join(sorted(rep["errores"]))}
    print(f"tanda FRED vivo: {len(rep['hechas'])} descargadas, {len(rep['cache'])} ya de hoy, "
          f"{len(rep['pendientes'])} pendientes, {len(rep['errores'])} con error; {cliente.peticiones} peticiones "
          f"en {fila['segundos']} s; log -> {_registrar_descarga(fila)}")
    if rep["errores"]:
        print("series con error tras el reintento:", rep["errores"])
        return 1
    if rep["pendientes"]:
        print(f"quedan {len(rep['pendientes'])} series: repite la orden para seguir")
        return 2
    return 0


def _informe_alfred(a: argparse.Namespace) -> int:
    from ingesta.mercados_series import almacen, catalogo
    from ingesta.mercados_series.base import AVISO_FRED
    from ingesta.mercados_series.fuentes import alfred

    ruta = almacen.ruta_series()
    if not ruta.exists():
        print(f"no hay base en {ruta}: python -m ingesta.mercados_series construir-base", file=sys.stderr)
        return 1
    cat = catalogo.cargar_catalogo()
    ids = alfred.ids_alfred(cat)
    tabla = almacen.leer_series(ids)
    for r in alfred.escribir_tablas_alfred(tabla, cat):
        print("->", r)
    print(AVISO_FRED)
    return 0


def _construir(a: argparse.Namespace) -> int:
    from ingesta.mercados_series import base

    resumen, tabla = base.construir_base(sin_red=a.sin_red, escribir=not a.sin_escribir,
                                         permitir_perdidas=a.reemplazar, purgar=a.purgar,
                                         desde_cero=a.desde_cero, registro=print)
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
                                description="Tema 1 · base de series (bloques 2 y 3)")
    sub = p.add_subparsers(dest="orden", required=True)
    da = sub.add_parser("descargar-alfred", help="una tanda de vintages de ALFRED (reanudable)")
    da.add_argument("--ids", nargs="*", help="ids del catálogo (por defecto fuentes.alfred.ids_alfred())")
    da.add_argument("--limite-segundos", type=float, default=480.0,
                    help="no empieza otra serie pasado este tiempo (por defecto 480)")
    da.add_argument("--refrescar", action="store_true", help="vuelve a descargar aunque haya archivo completo")
    da.set_defaults(func=_descargar_alfred)
    dv = sub.add_parser("descargar-fred-vivo", help="una tanda de la fuente viva de FRED (reanudable)")
    dv.add_argument("--ids", nargs="*", help="ids del catálogo (por defecto fuentes.fred.ids_fred_vivo())")
    dv.add_argument("--max-minutos", type=float, default=8.0, help="corta entre series tras N minutos (8)")
    dv.add_argument("--completa", action="store_true", help="descarga completa aunque haya archivo previo")
    dv.set_defaults(func=_descargar_fred_vivo)
    c = sub.add_parser("construir-base", help="une ALFRED + FRED vivo + académicos + fase1 y escribe la base")
    c.add_argument("--sin-red", action="store_true", help="solo crudos ya descargados y data/raw")
    c.add_argument("--desde-cero", action="store_true",
                   help="copia la base guardada a series/respaldo_bloque2.parquet y la sustituye entera")
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
    ia = sub.add_parser("informe-alfred", help="tablas de revisiones y de adelantos de la regla previa (ALFRED)")
    ia.set_defaults(func=_informe_alfred)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
