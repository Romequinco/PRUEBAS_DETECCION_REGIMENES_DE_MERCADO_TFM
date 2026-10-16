"""Sincroniza Markdown y HTML del notebook; no ejecuta consultas de red."""

from pathlib import Path

import nbformat
from nbconvert import HTMLExporter


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / 'notebooks/01_fuentes_prioridad_gratuita.ipynb'
    notebook = nbformat.read(path, as_version=4)
    markdown = []
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            markdown.append(cell.source)
        elif cell.cell_type == 'code':
            markdown.append('```python\n' + cell.source + '\n```')
    path.with_suffix('.md').write_text('\n\n'.join(markdown) + '\n', encoding='utf-8')
    html, _ = HTMLExporter().from_notebook_node(notebook)
    path.with_suffix('.html').write_text(html, encoding='utf-8')
    print('Exportados:', path.with_suffix('.md'), path.with_suffix('.html'))


if __name__ == '__main__':
    main()
