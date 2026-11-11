"""CLI del aumento (notebook 21).

Uso (desde la raiz del repo)::

    # 1. anclas (ajuste + validacion por ancla) y reservorios de trayectorias
    python -m regimenes.sinteticos.aumento preparar --pistas A B --n-jobs 10
    # 2. piloto de coste (solo tiempos) y prueba bit a bit con p = 0
    python -m regimenes.sinteticos.aumento piloto --pistas A B --n-jobs 10
    # 3. una sesion de calculo con presupuesto (reanudable; repetir hasta que no quede nada)
    python -m regimenes.sinteticos.aumento ejecutar --horas 7.5 --n-jobs 10
    # estado: trabajos pendientes y horas estimadas por lote
    python -m regimenes.sinteticos.aumento estado
    # 4. senales, estadisticos por segmento y nulo (para el notebook)
    python -m regimenes.sinteticos.aumento consolidar --pistas A B
"""

from __future__ import annotations

import argparse

from regimenes.sinteticos.aumento.config import config_aumento


def main(argv=None) -> None:
    cfg = config_aumento()
    parser = argparse.ArgumentParser(prog="python -m regimenes.sinteticos.aumento")
    parser.add_argument("orden", choices=["preparar", "piloto", "ejecutar", "estado", "consolidar"])
    parser.add_argument("--pistas", nargs="+", default=list(cfg["pistas"]))
    parser.add_argument("--n-jobs", type=int, default=int(cfg["ejecucion"]["n_jobs"]))
    parser.add_argument("--horas", type=float, default=float(cfg["ejecucion"]["horas"]))
    args = parser.parse_args(argv)
    pistas = [p.upper() for p in args.pistas]

    if args.orden == "preparar":
        from regimenes.sinteticos.aumento.anclas import preparar_anclas
        from regimenes.sinteticos.aumento.muestras import preparar_reservorios

        registros = preparar_anclas(pistas, n_jobs=args.n_jobs)
        for pista, tabla in registros.items():
            resumen = tabla.groupby("fuente")["valida"].agg(["sum", "count"])
            print(f"[anclas {pista}] validas por fuente:\n{resumen.to_string()}", flush=True)
        estados = preparar_reservorios(pistas, n_jobs=args.n_jobs)
        errores = estados[estados["estado"] != "ok"] if not estados.empty else estados
        print(f"[reservorios] {len(estados)} reservorios, {len(errores)} con error", flush=True)
        if len(errores):
            print(errores.to_string(), flush=True)
    elif args.orden == "piloto":
        from regimenes.sinteticos.aumento.ejecucion import ejecutar_piloto

        tabla = ejecutar_piloto(pistas, n_jobs=args.n_jobs)
        print(tabla.to_string(), flush=True)
    elif args.orden == "ejecutar":
        from regimenes.sinteticos.aumento.ejecucion import ejecutar

        ejecutar(horas=args.horas, n_jobs=args.n_jobs, pistas=pistas)
    elif args.orden == "estado":
        from regimenes.sinteticos.aumento.ejecucion import pendientes

        tabla = pendientes(pistas)
        if tabla.empty:
            print("[aumento] no queda ningun trabajo pendiente", flush=True)
            return
        resumen = tabla.groupby(["lote", "pista"]).agg(trabajos=("clave", "size"),
                                                         horas_cpu=("estimacion_s", lambda s: s.sum() / 3600))
        resumen["horas_reloj"] = resumen["horas_cpu"] / args.n_jobs
        print(resumen.round(2).to_string(), flush=True)
        print(f"[aumento] total: {len(tabla)} trabajos, ~{resumen['horas_reloj'].sum():.1f} h de reloj "
              f"con {args.n_jobs} procesos", flush=True)
    elif args.orden == "consolidar":
        from regimenes.sinteticos.aumento.analisis import consolidar_senales

        for pista in pistas:
            salida = consolidar_senales(pista)
            print(f"[consolidar {pista}] {len(salida['senales'])} senales", flush=True)


if __name__ == "__main__":
    main()
