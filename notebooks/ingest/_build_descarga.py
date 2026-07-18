"""
_build_descarga.py — Regenera y EJECUTA notebooks/ingest/00_descarga.ipynb.

Patron builder-por-notebook (heredado de Capa 1): la fuente de verdad reproducible
es este script; el .ipynb es su salida ejecutada. La descarga usa cache (force=False),
asi que re-ejecutar es barato si data/raw/ ya esta poblado.

Uso:  python notebooks/ingest/_build_descarga.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "notebooks" / "ingest" / "00_descarga.ipynb"

cells: list = []
md = lambda s: cells.append(new_markdown_cell(s))
code = lambda s: cells.append(new_code_cell(s))

# --------------------------------------------------------------------------- #
md(r"""# FASE 2 — Descarga de datos (capa v2)

> **Notebook de descarga**, dirigido por `data/catalog.yaml`. Baja el universo de datos
> **máximo y gratis** para la detección de regímenes, **sin imputar** (cada serie arranca
> en su fecha real), y deja los crudos en `data/raw/<fuente>/<serie>.parquet` +
> procedencia (`provenance.json`) + cobertura (`coverage_report.csv`).

## Decisiones (echadas a tierra)

- **Dos pistas** (ADR-001): **A** = espina histórica profunda (S&P500 + vol, ~1927/1950+,
  para tener 10+ crisis); **B** = panel rico multi-activo (crédito, curva, vol, FX,
  commodities, liquidez, macro, ~1990/2007+, para atacar el punto ciego de 2013).
- **Validación externa**: índices de estrés ya hechos (OFR FSI + subíndices, NFCI) y
  labels NBER — *ground truth laxo* de régimen.
- **Fuentes (todo gratis, verificadas una a una en FASE 2A)**:
  | Fuente | Series | Nota |
  |---|---|---|
  | **FRED** (API con key en `.env`) | ~99 | curva completa, crédito Moody's, macro, estrés |
  | **yfinance** | ~39 | índices, VIX/MOVE/VVIX, ETFs, FX, commodities |
  | **OFR** (CSV público) | 5 | Financial Stress Index + subíndices (diario 2000+) |
  | **académico** | ~9 | Ken French (factores diarios 1926+), Shiller (mensual 1871+) |
  | **github/datahub** | 5 | CSV sin auth (fallbacks) |
  | ~~stooq~~ | 2 | **bloqueado** por challenge JS (proof-of-work) → no descargable |
- **Hallazgo clave de crédito**: FRED capó las OAS de ICE BofA (`BAML*`) a ventana rodante
  de 3 años (licencia ICE, abril-2026). Sustituto histórico honesto: spreads **Moody's
  Baa-Aaa / Baa-10Y** (diarios desde 1986, sin restricción) + proxies ETF (HYG/LQD/IEF).
""")

code(r"""import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')

# ROOT = raíz del repo (donde vive data/catalog.yaml)
ROOT = Path.cwd()
while not (ROOT / 'data' / 'catalog.yaml').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from src import viz
viz.use_house_style()
from src.ingest import download_all, load_catalog
print('ROOT:', ROOT)""")

md("## 1. El catálogo de datos declarado")

code(r"""cat = load_catalog()
print('meta:', {k: cat['meta'][k] for k in ('version','estado','actualizado') if k in cat['meta']})
resumen = []
for pista in ['pista_A','pista_B','validacion_externa']:
    series = (cat.get(pista) or {}).get('series', [])
    fuentes = pd.Series([s['fuente'] for s in series]).value_counts().to_dict()
    resumen.append({'pista': pista, 'n_series': len(series), 'fuentes': fuentes})
print('crisis_catalog:', len(cat['crisis_catalog']['eventos']), 'eventos')
pd.DataFrame(resumen)""")

md(r"""## 2. Descarga

La descarga la ejecuta el módulo reproducible **`python -m src.ingest.download`**
(dirigido por el catálogo, cacheado, resiliente por serie). Aquí cargamos el
`coverage_report.csv` que produjo para documentar y visualizar el resultado. Para
re-descargar desde cero: `download_all(force=True)`.""")

code(r"""rep = pd.read_csv(ROOT / 'data' / 'raw' / 'coverage_report.csv')
print('Estado de la descarga:')
print(rep['status'].value_counts().to_string())
ok = rep['status'].isin(['OK','CACHE']).sum()
print(f'\n>>> {ok}/{len(rep)} series disponibles en disco')""")

code(r"""# Detalle por fuente y estado
tab = rep.pivot_table(index='fuente', columns='status', values='nombre',
                      aggfunc='count', fill_value=0)
tab['total'] = tab.sum(axis=1)
tab""")

code(r"""# Series que fallaron (transparencia total)
err = rep[rep['status']=='ERROR']
if len(err):
    print(f'{len(err)} series con ERROR:')
    for _, r in err.iterrows():
        print("  - {:26} ({}) {}".format(r['nombre'], r['fuente'], str(r.get('error',''))[:80]))
else:
    print('Sin errores.')""")

md("## 3. Cobertura temporal (¿cuánta historia tiene cada serie?)")

code(r"""ok_rep = rep[rep['status'].isin(['OK','CACHE'])].copy()
ok_rep['inicio'] = pd.to_datetime(ok_rep['inicio'])
ok_rep['fin'] = pd.to_datetime(ok_rep['fin'])
ok_rep = ok_rep.sort_values(['pista','inicio'])

colores = {'A':'#2c7fb8', 'B':'#d95f0e', 'validacion':'#31a354'}
fig, ax = plt.subplots(figsize=(13, max(6, len(ok_rep)*0.16)))
for i, (_, r) in enumerate(ok_rep.iterrows()):
    ax.barh(i, (r['fin']-r['inicio']).days, left=r['inicio'],
            color=colores.get(r['pista'], '#888'), height=0.7)
ax.set_yticks(range(len(ok_rep)))
ax.set_yticklabels(ok_rep['nombre'], fontsize=5)
ax.set_ylim(-1, len(ok_rep))
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=p) for p,c in colores.items()], loc='lower right')
ax.set_title('Cobertura temporal por serie (color = pista)')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout(); plt.show()""")

code(r"""# Histograma de profundidad histórica (año de inicio)
fig, axes = plt.subplots(1, 2, figsize=(13,4))
ok_rep['anio_inicio'] = ok_rep['inicio'].dt.year
axes[0].hist(ok_rep['anio_inicio'], bins=range(1920,2030,5), color='#2c7fb8', edgecolor='white')
axes[0].set_title('Año de inicio de las series'); axes[0].set_xlabel('año')
cnt = ok_rep['fuente'].value_counts()
axes[1].bar(cnt.index, cnt.values, color='#d95f0e')
axes[1].set_title('Series disponibles por fuente')
plt.tight_layout(); plt.show()
print('Serie más profunda:', ok_rep.loc[ok_rep['inicio'].idxmin(),'nombre'],
      '->', ok_rep['inicio'].min().date())""")

md("## 4. Catálogo de crisis (ataca el n≈4 de la Capa 1)")

code(r"""ev = pd.DataFrame(cat['crisis_catalog']['eventos'])
ev['peak'] = pd.to_datetime(ev['peak']); ev['trough'] = pd.to_datetime(ev['trough'])
ev = ev.sort_values('peak')
print(f'{len(ev)} eventos de crisis catalogados ({ev.peak.dt.year.min()}-{ev.peak.dt.year.max()})')
fig, ax = plt.subplots(figsize=(13,5))
for _, r in ev.iterrows():
    ax.barh(0, (r['trough']-r['peak']).days, left=r['peak'],
            height=abs(r['depth'])*100, align='edge',
            color='#c0392b', alpha=0.7)
    ax.text(r['peak'], abs(r['depth'])*100, r['name'], rotation=90,
            fontsize=5, va='bottom', ha='left')
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_ylabel('profundidad drawdown (%)'); ax.set_title('Cronología de crisis del S&P500 (peak→trough)')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout(); plt.show()
ev[['name','peak','trough','depth','tipo','nber']]""")

md(r"""## 5. Cierre

- Los crudos quedan en `data/raw/<fuente>/<serie>.parquet` (gitignored), con
  `provenance.json` (checksum + procedencia por serie) y `coverage_report.csv` versionados.
- **Siguiente**: FASE 3 (EDA + análisis) sobre estas series → definir las features causales
  v2 y **congelar el benchmark por pista**.
- Todo lo no descargable (stooq, algún académico) queda **declarado** en el coverage, no
  omitido en silencio.
""")

nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
ep = ExecutePreprocessor(timeout=1200, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": str(NB.parent)}})
nbformat.write(nb, NB)
print("OK ->", NB)
