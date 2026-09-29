"""CLI de la capa de datos: ``python -m regimenes.datos [--offline|--force --only A,B]``.
"""
from regimenes.datos.descarga import download_all

if __name__ == "__main__":  # pragma: no cover
    import sys

    args = sys.argv[1:]
    force = "--force" in args
    offline = "--offline" in args
    only = None
    if "--only" in args:
        k = args.index("--only")
        if k + 1 >= len(args):
            raise SystemExit("--only necesita una lista separada por comas")
        only = [x.strip() for x in args[k + 1].split(",") if x.strip()]
        if not force:
            print("AVISO: --only sin --force no re-descarga lo que ya esta en disco")
    rep = download_all(force=force, only=only, offline=offline)
    ok = (rep["status"].isin(["OK", "CACHE"])).sum()
    print(f"\n=== RESUMEN: {ok}/{len(rep)} series disponibles ===")
    print(rep["status"].value_counts().to_string())
