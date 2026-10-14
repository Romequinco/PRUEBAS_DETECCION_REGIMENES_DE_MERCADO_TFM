"""Ejecuta la sonda y guarda salidas reales; los fallos de fuentes son resultados."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / 'notebooks/01_fuentes_prioridad_gratuita.ipynb'
    notebook = nbformat.read(path, as_version=4)

    def progress(cell, cell_index, **kwargs):
        print(f'Celda {cell_index}: {cell.source.splitlines()[0][:100]}', flush=True)

    client = NotebookClient(notebook, timeout=1200, kernel_name='bancos-centrales-tfm',
                            resources={'metadata': {'path': str(root / 'notebooks')}},
                            on_cell_start=progress)
    client.execute()
    nbformat.write(notebook, path)
    print('Notebook ejecutado y guardado.', flush=True)


if __name__ == '__main__':
    main()
