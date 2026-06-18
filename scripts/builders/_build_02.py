"""Construye y EJECUTA notebooks/02_rule_composite_riskoff.ipynb (D2).

Patrón idéntico al de la Tanda 1: nbformat para montar las celdas +
ExecutePreprocessor (kernel python3) para ejecutarlo. Tras ejecutar, comprueba 0
errores leyendo los outputs. Figuras -> results/.
"""
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[2]
NB_PATH = ROOT / "notebooks" / "02_rule_composite_riskoff.ipynb"

cells = []

cells.append(new_markdown_cell(
    "# 02 — D2 `rule_composite_riskoff` (Reglas / Umbrales, voto multivariante)\n\n"
    "Regla **compuesta** y causal: agrega 4 señales de estrés ya causales de "
    "`features.parquet` en un **score de risk-off** (ALTO = estrés) y lo umbraliza "
    "con **histéresis** (τ_in/τ_out) + **dwell-time** mínimo, igual que D1 pero "
    "sobre un VOTO multivariante en vez de un único nivel de VIX. 2 estados: "
    "0=calma, 1=crisis (varias señales risk-off simultáneas).\n\n"
    "**Señales y orientación** (signo con que entran al score, ALTO=estrés):\n"
    "- `VIX_level_z` (+): miedo equity (bloom2009).\n"
    "- `credit_spread_z` (−): ret(HYG)−ret(IEF); el deterioro de crédito lo vuelve "
    "NEGATIVO (Gilchrist-Zakrajšek 2012).\n"
    "- `yield_slope_z` (−): pendiente 10Y−3M; curva baja/invertida = estrés "
    "(Estrella-Mishkin 1998). Signo INVERTIDO.\n"
    "- `SP500_drawdown` (−): drawdown corriente ∈[−1,0] (kritzman2012).\n\n"
    "**Ventana**: crédito (HYG) y curva existen en `features.parquet` desde "
    "2007-07. Con `train_size≈8 años` el primer bloque OOS cae hacia 2015, así que "
    "**2008 y 2011 quedan en el train inicial → cobertura NaN out-of-sample** "
    "(correcto y declarado): D2 solo se evalúa OOS sobre 2020 y 2022.\n\n"
    "**Hipótesis CHECKPOINT 2 (D2)**: *captará estrés multivariante "
    "equity+crédito+curva en 2008/2011/2020/2022; fallará por calibración de pesos "
    "sensible*. Verificamos al final."
))

cells.append(new_code_cell(
    "%matplotlib inline\n"
    "import sys\n"
    "from pathlib import Path\n"
    "import numpy as np, pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "from matplotlib.patches import Patch\n\n"
    "ROOT = Path.cwd()\n"
    "while not (ROOT / 'src').exists() and ROOT != ROOT.parent:\n"
    "    ROOT = ROOT.parent\n"
    "sys.path.insert(0, str(ROOT))\n"
    "RESULTS = ROOT / 'results'; RESULTS.mkdir(exist_ok=True)\n"
    "from src import evaluation as ev\n"
    "from detectors.rule_composite_riskoff import RuleCompositeRiskoff\n\n"
    "feats = pd.read_parquet(ROOT / 'data' / 'processed' / 'features.parquet')\n"
    "raw = pd.read_parquet(ROOT / 'data' / 'raw' / 'raw_panel.parquet')\n"
    "print('features:', feats.shape, '|', feats.index.min().date(), '->', feats.index.max().date())"
))

cells.append(new_markdown_cell(
    "## 1. Señales del voto + retorno del S&P 500 (market_returns)\n\n"
    "`X` = las 4 señales causales del score. `market_returns` = **retorno log del "
    "S&P 500** (`np.log(raw['SP500']/raw['SP500'].shift(1))`) reindexado a `X`. Lo "
    "pasaremos SIEMPRE a `walk_forward` y a `evaluate` para que el núcleo re-fije el "
    "orden económico (0=calma..n−1=crisis) y NO haya warning de fallback."
))

cells.append(new_code_cell(
    "SIGNALS = ['VIX_level_z', 'credit_spread_z', 'yield_slope_z', 'SP500_drawdown']\n"
    "X = feats[SIGNALS].dropna().copy()\n"
    "X.index = pd.to_datetime(X.index); X = X.sort_index()\n"
    "market_returns = np.log(raw['SP500'] / raw['SP500'].shift(1)).rename('SP500_ret')\n"
    "market_returns = market_returns.reindex(X.index)\n"
    "print('X (desde HYG/curva):', X.shape, '|', X.index.min().date(), '->', X.index.max().date())\n"
    "print('Orientación estrés (ALTO=risk-off):', RuleCompositeRiskoff().signs)\n"
    "X.head(3)"
))

cells.append(new_markdown_cell(
    "**Dirección de estrés empírica** de cada señal: media por ventana. Confirma los "
    "signos (VIX alto, credit_spread_z bajo/negativo, yield_slope_z bajo, drawdown "
    "negativo = estrés). Nótese que la curva se EMPINÓ en 2008/2011 (Fed recortando) "
    "— es señal adelantada, no contemporánea: fuente de fricción de pesos."
))

cells.append(new_code_cell(
    "wins = {**ev.CRISIS_WINDOWS, 'calm_2017': ('2017-01-01','2017-12-31')}\n"
    "tbl = {w: {c: round(float(feats.loc[a:b, c].mean()), 2) for c in SIGNALS}\n"
    "       for w, (a, b) in wins.items()}\n"
    "pd.DataFrame(tbl).T"
))

cells.append(new_markdown_cell(
    "## 2. El detector D2 y su score compuesto causal\n\n"
    "`RuleCompositeRiskoff(q_in=0.90, q_out=0.70, min_dwell=5)`: cada señal se "
    "orienta (signo·valor), se re-estandariza con μ/σ **del train** (causal) y se "
    "promedia → **score de risk-off**. τ_in/τ_out = percentiles 90/70 del score en "
    "el train. Ajuste sobre todo el histórico SOLO para inspección (la evaluación "
    "honesta es el walk-forward de la sección 3)."
))

cells.append(new_code_cell(
    "det0 = RuleCompositeRiskoff(q_in=0.90, q_out=0.70, min_dwell=5).fit(X)\n"
    "print(f'τ_in  (q=0.90) = {det0._tau_in:.3f}  (unidades de score)')\n"
    "print(f'τ_out (q=0.70) = {det0._tau_out:.3f}')\n"
    "print('pesos:', det0.weights)\n"
    "print('bibliografía:', det0.bibliography)\n"
    "# Re-fijar orden económico con market_returns (lo que hace walk_forward por fold).\n"
    "det0.label_states_economically(X, market_returns=market_returns)\n"
    "print('canonical order =', det0._canonical_order, '| crisis_state =', det0.crisis_state)\n"
    "states_is = pd.Series(det0.predict(X), index=X.index)\n"
    "byst = {int(s): round(float(market_returns[states_is == s].mean()), 5) for s in [0, 1]}\n"
    "print('retorno medio S&P por estado:', byst, '-> 1=crisis (peor) =', byst[1] < byst[0])\n"
    "assert det0.crisis_state == 1 and byst[1] < byst[0], 'crisis canónico != voto risk-off'\n"
    "print('% días crisis in-sample:', round(float((states_is == det0.crisis_state).mean()), 3))"
))

cells.append(new_markdown_cell(
    "## 3. Walk-forward causal (2008/2011 en el train inicial → NaN OOS)\n\n"
    "`ev.walk_forward` reentrena el detector en ventanas crecientes y predice el "
    "siguiente bloque sin ver el futuro, **pasando `market_returns`** para re-fijar "
    "el orden de estados por fold. Como `X` arranca en 2007 y `train_size≈8 años`, "
    "el panel OOS empieza hacia 2015: **2008 y 2011 NO son evaluables OOS** (caen en "
    "el train), solo 2020 y 2022."
))

cells.append(new_code_cell(
    "factory = lambda: RuleCompositeRiskoff(q_in=0.90, q_out=0.70, min_dwell=5)\n"
    "panel = ev.walk_forward(factory, X, market_returns=market_returns,\n"
    "                        train_size=252*8, step=21, expanding=True)\n"
    "print('Panel OOS:', panel.shape, '|', panel.index.min().date(), '->', panel.index.max().date())\n"
    "panel.head(3)"
))

cells.append(new_markdown_cell(
    "## 4. Evaluación estandarizada y fila de métricas (23 columnas)\n\n"
    "`ev.evaluate` con `market_returns` calcula cobertura por crisis, falsas alarmas "
    "(global y en trampas), lead/lag, switching/persistencia y estabilidad. Se vuelca "
    "a `results/metrics_02_rule_composite_riskoff.csv`."
))

cells.append(new_code_cell(
    "res = ev.evaluate(det0, panel, market_returns=market_returns, X_full=X)\n"
    "row = ev.results_table([res])\n"
    "out_csv = RESULTS / 'metrics_02_rule_composite_riskoff.csv'\n"
    "row.to_csv(out_csv, index=False)\n"
    "print('ventana_eval:', res.extra['ventana_eval'])\n"
    "print('retorno medio S&P por estado:', {k: round(v,5) for k,v in res.extra['mean_return_by_state'].items()})\n"
    "print('columnas:', row.shape[1], '| guardado:', out_csv.name)\n"
    "row.T"
))

cells.append(new_markdown_cell(
    "## 5. Visualización — S&P 500 coloreado por régimen (OOS)\n\n"
    "S&P 500 (log) con los días OOS clasificados como **crisis** sombreados; bandas "
    "de crisis conocidas (rojo) y trampas 2013/2018 (naranja). La zona pre-2015 es "
    "el train inicial (sin etiqueta OOS)."
))

cells.append(new_code_cell(
    "spx = raw['SP500'].reindex(panel.index)\n"
    "is_crisis = (panel['state'] == det0.crisis_state)\n"
    "fig, ax = plt.subplots(figsize=(15, 5))\n"
    "ax.plot(spx.index, spx, color='black', lw=0.7, zorder=3)\n"
    "ax.set_yscale('log'); ax.set_ylabel('S&P 500 (log)')\n"
    "ymin, ymax = ax.get_ylim()\n"
    "ax.fill_between(panel.index, ymin, ymax, where=is_crisis.values, color='crimson',\n"
    "                alpha=0.25, step='mid', zorder=1)\n"
    "ax.set_ylim(ymin, ymax)\n"
    "for a, b in ev.CRISIS_WINDOWS.values():\n"
    "    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='red', alpha=0.10, zorder=0)\n"
    "for a, b in ev.FALSE_POSITIVE_WINDOWS.values():\n"
    "    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='orange', alpha=0.18, zorder=0)\n"
    "handles = [Patch(color='crimson', alpha=0.25, label='D2 crisis (OOS)'),\n"
    "           Patch(color='red', alpha=0.10, label='Ventanas crisis'),\n"
    "           Patch(color='orange', alpha=0.18, label='Trampas 2013/2018')]\n"
    "ax.legend(handles=handles, loc='upper left', fontsize=8)\n"
    "ax.set_title('D2 rule_composite_riskoff — S&P 500 coloreado por régimen (out-of-sample)')\n"
    "fig.tight_layout(); fig.savefig(RESULTS / 'd2_regime_sp500.png', dpi=110, bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(new_markdown_cell(
    "## 6. Score compuesto vs umbral + contribución de cada señal + timeline\n\n"
    "Arriba: score de risk-off OOS con τ_in/τ_out (histéresis). Medio: z orientada de "
    "cada señal (ALTO=estrés) para ver QUÉ aporta el voto en cada episodio. Abajo: "
    "timeline de régimen (banda roja = crisis)."
))

cells.append(new_code_cell(
    "score = pd.Series(det0.composite_score(X), index=X.index).reindex(panel.index)\n"
    "# z orientadas por señal (con μ/σ del fit de inspección)\n"
    "ori = det0._oriented(X.reindex(panel.index))\n"
    "zsig = {f: (ori[f] - det0._mu[f]) / (det0._sigma[f] if det0._sigma[f] > 0 else 1.0)\n"
    "        for f in det0.features}\n"
    "fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 9), sharex=True,\n"
    "                                    gridspec_kw={'height_ratios': [3, 3, 1]})\n"
    "ax1.plot(score.index, score, color='purple', lw=0.8, label='score risk-off')\n"
    "ax1.axhline(det0._tau_in, color='crimson', ls='--', lw=1, label=f'τ_in={det0._tau_in:.2f}')\n"
    "ax1.axhline(det0._tau_out, color='darkorange', ls='--', lw=1, label=f'τ_out={det0._tau_out:.2f}')\n"
    "ax1.fill_between(panel.index, score.min(), score.max(), where=is_crisis.values,\n"
    "                 color='crimson', alpha=0.12, step='mid')\n"
    "ax1.set_ylabel('score'); ax1.legend(loc='upper left', fontsize=8)\n"
    "ax1.set_title('Score compuesto vs umbral (histéresis τ_in/τ_out)')\n"
    "for f, c in zip(det0.features, ['steelblue','seagreen','goldenrod','firebrick']):\n"
    "    ax2.plot(panel.index, zsig[f], lw=0.6, color=c, label=f)\n"
    "ax2.axhline(0, color='gray', lw=0.5)\n"
    "ax2.set_ylabel('z orientada (ALTO=estrés)'); ax2.legend(loc='upper left', fontsize=7, ncol=2)\n"
    "ax2.set_title('Contribución de cada señal al voto')\n"
    "ax3.fill_between(panel.index, 0, 1, where=is_crisis.values, color='crimson', alpha=0.6, step='mid')\n"
    "ax3.set_yticks([]); ax3.set_ylabel('régimen'); ax3.set_title('Timeline (rojo = crisis)')\n"
    "fig.tight_layout(); fig.savefig(RESULTS / 'd2_score_timeline.png', dpi=110, bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(new_markdown_cell(
    "## 7. Verificación explícita contra crisis y trampas\n\n"
    "Cobertura (% días crisis) en cada ventana conocida y en cada trampa. 2008/2011 "
    "salen NaN (en el train inicial). Para 2020/2022: alto=bueno. Para trampas "
    "2013/2018: bajo=bueno."
))

cells.append(new_code_cell(
    "states_oos = panel['state']\n"
    "cov = ev.crisis_coverage(states_oos, det0.crisis_state)\n"
    "fa  = ev.false_alarm_in_windows(states_oos, det0.crisis_state)\n"
    "print('=== COBERTURA EN CRISIS (alto = bueno; NaN = en train inicial) ===')\n"
    "for k, v in cov.items():\n"
    "    tag = 'sin OOS (train)' if v != v else ('OK' if v >= 0.5 else 'BAJA')\n"
    "    print(f'  {k:16s}: ' + ('  NaN' if v != v else f'{v:6.2%}') + f'  [{tag}]')\n"
    "print('\\n=== ACTIVACIÓN EN TRAMPAS (bajo = bueno) ===')\n"
    "for k, v in fa.items():\n"
    "    print(f'  {k:16s}: ' + ('  NaN (train)' if v != v else f'{v:6.2%}'))\n"
    "print(f'\\nfalse_alarm_rate global: {res.false_alarm_rate:.2%}')\n"
    "print(f'switching_rate: {res.switching_rate:.4f}  |  duración media régimen: {res.mean_regime_duration:.1f} días')\n"
    "print(f'label_stability: {res.label_stability:.3f}')"
))

cells.append(new_markdown_cell(
    "## 8. Comparación con D1 (VIX-solo) y veredicto sobre la hipótesis del CP2\n\n"
    "Carga la fila de D1 y compara cobertura/falsas alarmas/persistencia. La pregunta "
    "clave: ¿el voto compuesto capta estrés multivariante (crédito + drawdown + curva) "
    "que el VIX solo no ve, sobre todo en el bear market de tipos de 2022?"
))

cells.append(new_code_cell(
    "d1_csv = RESULTS / 'metrics_01_rule_vix_threshold.csv'\n"
    "cmp_cols = ['detector','ventana_eval','cov_COVID_2020','cov_Inflation_2022',\n"
    "            'fa_Selloff_Q4_2018','false_alarm_rate','switching_rate','mean_regime_duration']\n"
    "rows = [row]\n"
    "if d1_csv.exists():\n"
    "    rows.insert(0, pd.read_csv(d1_csv))\n"
    "cmp = pd.concat(rows, ignore_index=True)[cmp_cols]\n"
    "print('Comparación D1 (VIX-solo) vs D2 (voto compuesto):')\n"
    "import IPython.display as disp\n"
    "disp.display(cmp.set_index('detector').T)\n"
    "d1_infl = float(pd.read_csv(d1_csv)['cov_Inflation_2022'].iloc[0]) if d1_csv.exists() else float('nan')\n"
    "d2_infl = float(row['cov_Inflation_2022'].iloc[0])\n"
    "print(f'\\nInflation 2022: D1(VIX)={d1_infl:.1%}  ->  D2(compuesto)={d2_infl:.1%}  '\n"
    "      f'(mejora={d2_infl - d1_infl:+.1%})')\n"
    "print('Hipótesis CP2: captará estrés multivariante 2008/2011/2020/2022; fallará por\\n'\n"
    "      'calibración de pesos sensible. Discusión completa en\\n'\n"
    "      'docs/memory/detectors/02_rule_composite_riskoff.md')"
))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})

ep = ExecutePreprocessor(timeout=1200, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": str(ROOT / "notebooks")}})
nbf.write(nb, NB_PATH)

n_err = sum(1 for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error")
print(f"[exec] {NB_PATH.name}: errores={n_err}")
if n_err:
    for c in nb.cells:
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                print("  ", o.get("ename"), o.get("evalue"))
