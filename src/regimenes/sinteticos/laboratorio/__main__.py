"""CLI del laboratorio: ``python -m regimenes.sinteticos.laboratorio``.

- ``--piloto``: 1 trayectoria de la primera celda por pista y detector; escribe
  ``piloto_coste.csv`` (solo tiempos) y propone ``n_paths`` para el presupuesto.
- sin ``--piloto``: genera las trayectorias que falten y ejecuta la rejilla completa
  con ``--n-paths`` trayectorias por celda (reanudable).
"""

from __future__ import annotations

import argparse
import math

import pandas as pd

from regimenes.sinteticos.laboratorio import escenarios as esc
from regimenes.sinteticos.laboratorio import ejecucion as ej


def proponer_n_paths(coste: pd.DataFrame, cfg: dict) -> int:
    """Mayor n_paths que cabe en ``presupuesto_horas`` con ``n_jobs`` procesos.

    Coste estimado = sum_pista(n_celdas x sum_detector segundos_piloto) x n_paths / n_jobs,
    con un 20 % de margen por desequilibrio de carga.
    """
    por_tray = sum(len(esc.celdas(p, cfg)) * coste.loc[coste.pista == p, "segundos"].sum()
                   for p in coste.pista.unique())
    presupuesto = float(cfg["presupuesto_horas"]) * 3600 * int(cfg["n_jobs"]) / 1.2
    return max(int(cfg["n_paths_min"]), math.floor(presupuesto / por_tray))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pistas", nargs="+", default=None)
    ap.add_argument("--n-paths", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--piloto", action="store_true")
    a = ap.parse_args(argv)
    cfg = esc.config_laboratorio()
    pistas = [p.upper() for p in (a.pistas or cfg["pistas"])]
    n_jobs = a.n_jobs or int(cfg["n_jobs"])
    ej.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    if a.piloto:
        ej.preparar_trayectorias(pistas, 1, cfg)
        estados = ej.ejecutar_lote(ej.trabajos(pistas, 1, cfg, celdas_por_pista=1), n_jobs=n_jobs)
        coste = estados[["pista", "id", "segundos", "estado"]]
        coste.to_csv(ej.DIR_RESULTADOS / "piloto_coste.csv", index=False)
        print(coste.to_string(index=False))
        print("n_paths propuesto:", proponer_n_paths(coste, cfg))
        return
    n_paths = a.n_paths or cfg.get("n_paths")
    if not n_paths:
        raise SystemExit("Fija laboratorio.n_paths en el yaml (tras el piloto) o pasa --n-paths.")
    print("trayectorias nuevas:", ej.preparar_trayectorias(pistas, n_paths, cfg), flush=True)
    estados = ej.ejecutar_lote(ej.trabajos(pistas, n_paths, cfg), n_jobs=n_jobs)
    columnas = [c for c in ("pista", "celda", "path_id", "id", "estado", "segundos", "detalle") if c in estados]
    estados[columnas].to_csv(ej.DIR_RESULTADOS / "estado_ejecucion.csv", index=False)
    print(estados["estado"].value_counts().to_string())
    ej.consolidar()


if __name__ == "__main__":
    main()
