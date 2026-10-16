"""Sincroniza Markdown y HTML del notebook; no ejecuta consultas de red."""

import nbformat
from nbconvert import HTMLExporter

from ingesta.comun.rutas import ROOT, corpus_dir


def main():
    path = ROOT / 'notebooks/fase2/bancos_riesgo/01_fuentes_prioridad_gratuita.ipynb'
    directory = corpus_dir('bancos_riesgo') / 'exports/notebooks'
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / path.stem
    notebook = nbformat.read(path, as_version=4)
    markdown = []
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            markdown.append(cell.source)
        elif cell.cell_type == 'code':
            markdown.append('```python\n' + cell.source + '\n```')
    output.with_suffix('.md').write_text('\n\n'.join(markdown) + '\n', encoding='utf-8')
    html, _ = HTMLExporter().from_notebook_node(notebook)
    output.with_suffix('.html').write_text(html, encoding='utf-8')
    print('Exportados:', output.with_suffix('.md'), output.with_suffix('.html'))


if __name__ == '__main__':
    main()
