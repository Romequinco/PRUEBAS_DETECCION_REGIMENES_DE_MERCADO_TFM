"""Ejecuta la sonda y guarda salidas reales; los fallos de fuentes son resultados."""

import argparse

import nbformat
from nbclient import NotebookClient

from ingesta.comun.rutas import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', default='python3', help='Kernel Jupyter del entorno de la sonda.')
    args = parser.parse_args()
    path = ROOT / 'notebooks/fase2/bancos_riesgo/01_fuentes_prioridad_gratuita.ipynb'
    notebook = nbformat.read(path, as_version=4)

    def progress(cell, cell_index, **kwargs):
        print(f'Celda {cell_index}: {cell.source.splitlines()[0][:100]}', flush=True)

    client = NotebookClient(notebook, timeout=1200, kernel_name=args.kernel,
                            resources={'metadata': {'path': str(path.parent)}},
                            on_cell_start=progress)
    client.execute()
    nbformat.write(notebook, path)
    print('Notebook ejecutado y guardado.', flush=True)


if __name__ == '__main__':
    main()
