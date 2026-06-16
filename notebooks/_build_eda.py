"""Ensambla y EJECUTA notebooks/00_eda.ipynb con outputs y figuras reales.

Uso:  python notebooks/_build_eda.py
Lee data/raw/raw_panel.parquet (descargado por src/data_loader). Guarda figuras
en results/ y el notebook ejecutado en notebooks/00_eda.ipynb.
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("md", src))


def code(src: str) -> None:
    CELLS.append(("code", src))


# --------------------------------------------------------------------------- #
md(
    "# 00 — EDA: datos y features causales\n\n"
    "Banco de pruebas de detección de regímenes (capa TFM). Este notebook ejecuta el "
    "EDA de la **FASE 1**:\n"
    "- Cobertura y fechas de inicio por serie (set ampliado, **sin imputar**).\n"
    "- Ventana común resultante y periodos faltantes.\n"
    "- Estadísticos, **fat tails** (kurtosis), correlaciones.\n"
    "- Suelos de drawdown del S&P 500 (que alimentan `DRAWDOWN_TROUGHS`).\n"
    "- Construcción de **features causales** + test de no look-ahead.\n\n"
    "Resumen en `docs/memory/01_data_and_eda.md`."
)

code(
    "%matplotlib inline\n"
    "import sys\n"
    "from pathlib import Path\n"
    "import numpy as np, pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "from scipy import stats\n"
    "\n"
    "# Localizar raíz del repo (contiene src/)\n"
    "ROOT = Path.cwd()\n"
    "while not (ROOT / 'src').exists() and ROOT != ROOT.parent:\n"
    "    ROOT = ROOT.parent\n"
    "sys.path.insert(0, str(ROOT))\n"
    "RESULTS = ROOT / 'results'; RESULTS.mkdir(exist_ok=True)\n"
    "sns.set_theme(style='whitegrid', context='notebook')\n"
    "from src import data_loader as dl, features as ft, evaluation as ev\n"
    "\n"
    "raw = pd.read_parquet(ROOT / 'data' / 'raw' / 'raw_panel.parquet')\n"
    "print('Panel crudo:', raw.shape, '| union de fechas', raw.index.min().date(), '->', raw.index.max().date())\n"
    "raw.tail(3)"
)

md("## 1. Cobertura y fechas de inicio por serie (sin imputar)\n\n"
   "Cada serie arranca en su fecha real. La última fila es la **ventana común** "
   "(intersección de todas las series, sin NaN).")
code(
    "cov = dl.coverage_report(raw, save=False)\n"
    "cov"
)

md("## 2. Procedencia de los datos (yfinance + fallbacks documentados)\n\n"
   "FRED (`fredgraph.csv`) resultó **inaccesible** en este entorno (timeouts). "
   "Sustituciones documentadas, sin inventar ni omitir en silencio:\n"
   "- **DXY** → yfinance `DX-Y.NYB` (índice dólar ICE clásico).\n"
   "- **Curva 10Y-3M** → proxy real `^TNX − ^IRX` (FRED `T10Y3M` no respondió).\n"
   "- **HY OAS** → omitido (respuesta FRED truncada); crédito cubierto por HYG y spread HYG−IEF.")
code(
    "import json\n"
    "prov = json.loads((ROOT / 'data' / 'raw' / 'provenance.json').read_text(encoding='utf-8'))\n"
    "pd.Series(prov, name='fuente').to_frame()"
)

md("## 3. Periodos faltantes dentro del rango de cada serie\n\n"
   "NaN dentro del propio rango de cada serie (festivos desalineados entre fuentes, "
   "cierres). No se imputan; los detectores trabajan sobre la ventana común con `dropna`.")
code(
    "rng = pd.date_range(raw.index.min(), raw.index.max(), freq='B')\n"
    "miss = {}\n"
    "for c in raw.columns:\n"
    "    s = raw[c]\n"
    "    first = s.first_valid_index()\n"
    "    within = s.loc[first:]\n"
    "    miss[c] = int(within.isna().sum())\n"
    "pd.Series(miss, name='NaN_dentro_de_rango').sort_values(ascending=False).to_frame()"
)

md("## 4. Estadísticos descriptivos de los retornos diarios\n\n"
   "Retornos log de precios; niveles (VIX, MOVE, yield slope) se describen aparte.")
code(
    "px_cols = ['SP500','TLT','IEF','HYG','GOLD','DXY']\n"
    "rets = np.log(raw[px_cols] / raw[px_cols].shift(1)).dropna(how='all')\n"
    "desc = rets.describe().T[['mean','std','min','max']]\n"
    "desc['ann_vol'] = rets.std() * np.sqrt(252)\n"
    "desc.round(5)"
)

md("## 5. Fat tails: skew y kurtosis (exceso)\n\n"
   "Hallazgo metodológico de la tarea previa: los retornos son **leptocúrticos** "
   "(colas gordas), lo que el supuesto gaussiano del HMM subestima. Kurtosis de "
   "exceso ≫ 0 y test de normalidad Jarque-Bera lo confirman.")
code(
    "tail = pd.DataFrame({\n"
    "    'skew': rets.skew(),\n"
    "    'excess_kurtosis': rets.kurtosis(),  # Fisher: 0 = normal\n"
    "})\n"
    "jb = {c: stats.jarque_bera(rets[c].dropna()) for c in rets.columns}\n"
    "tail['jarque_bera_stat'] = {c: jb[c][0] for c in rets.columns}\n"
    "tail['jb_p_value'] = {c: jb[c][1] for c in rets.columns}\n"
    "tail.round(3)"
)
code(
    "fig, axes = plt.subplots(2, 3, figsize=(15, 8))\n"
    "for ax, c in zip(axes.ravel(), px_cols):\n"
    "    r = rets[c].dropna()\n"
    "    ax.hist(r, bins=120, density=True, alpha=0.6, color='steelblue')\n"
    "    x = np.linspace(r.min(), r.max(), 200)\n"
    "    ax.plot(x, stats.norm.pdf(x, r.mean(), r.std()), 'r-', lw=1.2, label='Normal')\n"
    "    ax.set_title(f'{c}  (exc.kurt={r.kurtosis():.1f})')\n"
    "    ax.set_yscale('log'); ax.legend(fontsize=8)\n"
    "fig.suptitle('Distribución de retornos vs Normal (escala log): colas gordas', y=1.02)\n"
    "fig.tight_layout(); fig.savefig(RESULTS / 'eda_fat_tails.png', dpi=110, bbox_inches='tight')\n"
    "plt.show()"
)

md("## 6. Correlaciones entre retornos\n\n"
   "Estructura de dependencia incondicional. La dependencia condicional al régimen "
   "(que sube en crisis) es justo lo que los detectores deben capturar.")
code(
    "corr = rets.corr()\n"
    "fig, ax = plt.subplots(figsize=(7, 6))\n"
    "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, ax=ax)\n"
    "ax.set_title('Correlación de retornos diarios (muestra completa)')\n"
    "fig.tight_layout(); fig.savefig(RESULTS / 'eda_corr.png', dpi=110, bbox_inches='tight')\n"
    "plt.show()"
)

md("## 7. S&P 500, drawdown y ventanas de crisis / falsos positivos\n\n"
   "Ventanas conocidas (de `evaluation.py`): crisis = 2008/2011/2020/2022; "
   "trampas (no-crisis) = taper 2013 y Q4 2018. Todas caen dentro de la ventana "
   "común de datos (≥ 2007-04-11).")
code(
    "spx = raw['SP500'].dropna()\n"
    "dd = spx / spx.expanding().max() - 1.0\n"
    "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)\n"
    "ax1.plot(spx.index, spx, color='black', lw=0.8); ax1.set_yscale('log'); ax1.set_ylabel('S&P 500 (log)')\n"
    "ax2.fill_between(dd.index, dd, 0, color='firebrick', alpha=0.5); ax2.set_ylabel('Drawdown')\n"
    "for name, (a, b) in ev.CRISIS_WINDOWS.items():\n"
    "    for ax in (ax1, ax2): ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='red', alpha=0.12)\n"
    "for name, (a, b) in ev.FALSE_POSITIVE_WINDOWS.items():\n"
    "    for ax in (ax1, ax2): ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='orange', alpha=0.18)\n"
    "ax1.set_title('S&P 500 con ventanas de CRISIS (rojo) y FALSOS POSITIVOS (naranja)')\n"
    "fig.tight_layout(); fig.savefig(RESULTS / 'eda_sp500_drawdown.png', dpi=110, bbox_inches='tight')\n"
    "plt.show()"
)

md("## 8. Suelos de drawdown CALCULADOS (alimentan `DRAWDOWN_TROUGHS`)\n\n"
   "Se calculan desde la serie real del S&P 500 (mínimo del drawdown en cada "
   "episodio), no a mano. Estos valores están cableados en `evaluation.DRAWDOWN_TROUGHS`.")
code(
    "episodes = {\n"
    "    'DotCom_2002':   ('2000-01-01','2003-06-30'),\n"
    "    'GFC_2008':      ('2007-10-01','2009-12-31'),\n"
    "    'EuroDebt_2011': ('2011-04-01','2012-06-30'),\n"
    "    'COVID_2020':    ('2020-01-01','2020-06-30'),\n"
    "    'Inflation_2022':('2022-01-01','2023-06-30'),\n"
    "}\n"
    "rows = []\n"
    "for name, (a, b) in episodes.items():\n"
    "    seg = dd.loc[a:b]\n"
    "    rows.append({'episodio': name, 'trough': seg.idxmin().date().isoformat(),\n"
    "                 'max_drawdown': round(float(seg.min()), 3),\n"
    "                 'en_ventana_comun': seg.idxmin() >= pd.Timestamp('2007-04-11')})\n"
    "troughs_df = pd.DataFrame(rows)\n"
    "print('DRAWDOWN_TROUGHS en evaluation.py:', ev.DRAWDOWN_TROUGHS)\n"
    "troughs_df"
)

md("## 9. Correlación rolling S&P 500 / Treasuries (Gulko 2002)\n\n"
   "La correlación equity/bonos **cambia de signo entre regímenes**: suele ser "
   "negativa en calma (diversificación) y puede romperse en crisis. Es una feature "
   "del detector (`corr_spx_bond`).")
code(
    "spx_ret = np.log(raw['SP500']/raw['SP500'].shift(1))\n"
    "tlt_ret = np.log(raw['TLT']/raw['TLT'].shift(1))\n"
    "rcorr = spx_ret.rolling(60, min_periods=30).corr(tlt_ret).dropna()\n"
    "fig, ax = plt.subplots(figsize=(15, 4))\n"
    "ax.plot(rcorr.index, rcorr, color='purple', lw=0.8); ax.axhline(0, color='k', lw=0.6)\n"
    "for name, (a, b) in ev.CRISIS_WINDOWS.items():\n"
    "    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='red', alpha=0.12)\n"
    "ax.set_title('Correlación rolling 60d S&P 500 / TLT (ventanas de crisis en rojo)')\n"
    "fig.tight_layout(); fig.savefig(RESULTS / 'eda_rolling_corr.png', dpi=110, bbox_inches='tight')\n"
    "plt.show()"
)

md("## 10. Features causales + verificación de no look-ahead\n\n"
   "`features.build_features` produce 15 features causales. `assert_causal` trunca "
   "la entrada y comprueba que el pasado no cambia al añadir futuro: "
   "`max_abs_diff` debe ser 0 para todas.")
code(
    "feats = ft.build_features(raw, save=True)\n"
    "print('Features:', feats.shape, '| ventana efectiva:', feats.index.min().date(), '->', feats.index.max().date())\n"
    "feats.describe().T[['mean','std','min','max']].round(3)"
)
code(
    "causal = ft.assert_causal(raw, cut='2015-01-01')\n"
    "assert causal['causal_ok'].all(), 'HAY FEATURES NO CAUSALES'\n"
    "print('TODAS las features son causales (max_abs_diff == 0):')\n"
    "causal"
)

md("---\n**Conclusión FASE 1:** datos descargados sin imputar, ventana común "
   "2007-04-11 → 2026-06 (gobernada por HYG), 15 features causales verificadas, "
   "troughs de drawdown calculados. Detalle en `docs/memory/01_data_and_eda.md`.")


# --------------------------------------------------------------------------- #
def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(s) if t == "md" else nbf.v4.new_code_cell(s)
        for t, s in CELLS
    ]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    print("Ejecutando notebook...")
    ep.preprocess(nb, {"metadata": {"path": str(ROOT / "notebooks")}})
    out = ROOT / "notebooks" / "00_eda.ipynb"
    nbf.write(nb, out)
    print("Notebook ejecutado y guardado en", out)


if __name__ == "__main__":
    main()
