"""
_build_05.py — Construye y EJECUTA notebooks/05_markov_switching_var.ipynb (D5).

Crea el notebook celda a celda (patrón de la Tanda 1), lo ejecuta con
ExecutePreprocessor y lo escribe ya ejecutado. Reporta el nº de errores (debe ser 0)
y vuelca las figuras a results/. Ejecutar desde la raíz del repo:

    python notebooks/_build_05.py
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "05_markov_switching_var.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md(r"""# 05 — D5 `markov_switching_var` (Markov-Switching econométrico · familia F4)

**Baseline econométrico interpretable** en la tradición de Hamilton (1989): un
Markov-Switching model sobre el **retorno del S&P 500** cuyo régimen latente (cadena de
Markov de 2 estados) gobierna a la vez la **media** y la **varianza**. El régimen de
**alta varianza** (media baja/negativa) es la *crisis/estrés*; el de baja varianza, la
*calma*. `statsmodels.MarkovRegression(switching_variance=True, trend='c')`.

**Por qué D5 es univariante y por qué importa el etiquetado robusto.** D5 modela SOLO
el retorno del propio índice (no la correlación cross-asset, a diferencia del HMM
multivariante D4). Como el régimen separa por **varianza**, no por el signo del retorno,
es exactamente el caso donde un etiquetado por "primera columna" podría INVERTIR
crisis/calma → pasamos `market_returns` (retorno log del S&P 500) **explícito** a
`walk_forward` y `evaluate`, y verificamos que crisis = alta varianza.

**Causalidad — probabilidades FILTRADAS, no smoothed.** `MarkovRegression` da
`filtered_marginal_probabilities` P(S_t | y≤t) (causales) y
`smoothed_marginal_probabilities` P(S_t | y₁..T) (look-ahead). Para la evaluación online
se usan FILTRADAS. En walk-forward, con los parámetros congelados del train, se corre un
**forward filter gaussiano univariante propio** (extraído μ_k, σ²_k, matriz de
transición) sobre `burn-in de train + bloque`, devolviendo solo el bloque: cero
look-ahead intra-bloque. Las smoothed se usan solo IN-SAMPLE, marcadas como NO causales.

**Ventana LARGA.** El retorno del S&P 500 existe desde **1985**; con train inicial de
~8 años, **2008 y 2011 SÍ son evaluables out-of-sample** (como D1), a diferencia de D4.

**Hipótesis CHECKPOINT 2 (D5):** *baseline econométrico interpretable; capta calma/estrés;
punto ciego en crisis rápidas; univariante no ve correlación cross-asset; gaussiano
insuficiente para colas.* Verificamos al final."""))

cells.append(code(r"""%matplotlib inline
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path.cwd()
while not (ROOT / 'src').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / 'results'; RESULTS.mkdir(exist_ok=True)
from src import evaluation as ev
from detectors.markov_switching_var import MarkovSwitchingVar

raw = pd.read_parquet(ROOT / 'data' / 'raw' / 'raw_panel.parquet')
print('Panel crudo:', raw.shape, '|', raw.index.min().date(), '->', raw.index.max().date())
print('S&P 500 primer dato válido:', raw['SP500'].first_valid_index().date())"""))

cells.append(md(r"""## 1. Endog: retorno log del S&P 500 (histórico largo desde 1985)

`SP500_ret` = retorno log del S&P 500. Es a la vez (a) el **endog** del Markov-Switching,
(b) el `market_returns` para el **etiquetado económico** robusto (0=calma..1=crisis) y
(c) la base de la `p_crisis`. El retorno es causal por construcción (diferencia de
precios; en t solo usa precios ≤ t). La columna se llama `SP500_ret`, reconocida por el
núcleo en `_RETURN_COLS`, de modo que el etiquetado económico NO cae al fallback
peligroso que "PUEDE INVERTIR" crisis/calma."""))

cells.append(code(r"""spx_ret = np.log(raw['SP500'] / raw['SP500'].shift(1)).rename('SP500_ret')
X = pd.DataFrame({'SP500_ret': spx_ret}).dropna()
X.index = pd.to_datetime(X.index); X = X.sort_index()
mr = X['SP500_ret']   # market_returns explícito (retorno log S&P 500)
print('X (histórico largo):', X.shape, '|', X.index.min().date(), '->', X.index.max().date())
print('retorno diario: media=%.4f  std=%.4f  kurtosis=%.1f' % (mr.mean(), mr.std(), mr.kurtosis()))
X.head(3)"""))

cells.append(md(r"""## 2. Ajuste in-sample, parámetros por régimen y verificación crisis = ALTA varianza

Ajuste sobre TODO el histórico solo para inspeccionar los parámetros (la evaluación
honesta es el walk-forward de la sección 4). Imprimimos media y **varianza por estado en
orden canónico** y verificamos que el estado de crisis (canónico `n-1`) es el de **mayor
varianza** (no invertido)."""))

cells.append(code(r"""with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    det = MarkovSwitchingVar(n_states=2).fit(X)

print('crisis_state canónico =', det.crisis_state, '| n_states =', det.n_states)
print('bibliografía:', det.bibliography)
means_c = det.means_canonical(); vars_c = det.variances_canonical()
print('\n--- Parámetros por estado (orden canónico 0=calma .. n-1=crisis) ---')
for i in range(det.n_states):
    tag = 'CRISIS' if i == det.crisis_state else 'calma '
    print(f'  estado {i} [{tag}]: media={means_c[i]:+.4f}   varianza={vars_c[i]:.4f}')
crisis_is_high_var = np.isclose(vars_c[det.crisis_state], vars_c.max())
print('\nVERIFICACIÓN crisis = ALTA varianza:', crisis_is_high_var)
assert crisis_is_high_var, 'FALLO: el estado crisis NO es el de mayor varianza (invertido)'
# Validación del filtrado propio: debe coincidir con statsmodels in-sample (sin burn-in).
diff = np.max(np.abs(det.predict_proba(X) - det.insample_proba('filtered')))
print('max|forward-filter propio - statsmodels filtered| =', f'{diff:.2e}', '(≈0 ⇒ OK)')"""))

cells.append(md(r"""## 3. Selección de k por AIC/BIC (k=2 vs k=3)

Ajustamos también k=3 y comparamos AIC/BIC. Reportamos qué k prefiere cada criterio. El
detector evaluado es k=2 (baseline interpretable); k=3 se reporta como referencia."""))

cells.append(code(r"""rows = []
fits = {}
for k in (2, 3):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        d = MarkovSwitchingVar(n_states=k, search_reps=20).fit(X)
    fits[k] = d
    rows.append({'k': k, 'logL': d.score(X), 'n_params': d.n_parameters(),
                 'AIC': d.aic(X), 'BIC': d.bic(X)})
sel = pd.DataFrame(rows).set_index('k')
print(sel.round(1))
print('\nk* por AIC =', int(sel['AIC'].idxmin()), ' | k* por BIC =', int(sel['BIC'].idxmin()))
print('\nVarianzas por estado (canónico) para cada k:')
for k in (2, 3):
    print(f'  k={k}:', np.round(fits[k].variances_canonical(), 4))"""))

cells.append(md(r"""## 4. Walk-forward CAUSAL (cubre 2008 y 2011 OOS)

`ev.walk_forward` reentrena el MS en ventanas crecientes y predice el siguiente bloque de
21 días usando solo el pasado. La predicción del bloque usa el **forward filter
univariante propio** con parámetros congelados y burn-in de train ⇒ probabilidades
FILTRADAS causales, sin look-ahead intra-bloque. `market_returns` re-fija el orden
económico en cada fold."""))

cells.append(code(r"""factory = lambda: MarkovSwitchingVar(n_states=2)
# step=63 (refit trimestral): el MS reestima ~131 veces en vez de ~394 (con
# ventana expanding los folds tardíos ajustan sobre 10k+ obs, ~10s cada uno).
# Los regímenes de varianza son persistentes, así que un refit trimestral es
# adecuado y mantiene la ventana larga (2008/2011 evaluables OOS).
panel = ev.walk_forward(factory, X, market_returns=mr,
                        train_size=252*8, step=63, expanding=True)
print('Panel OOS:', panel.shape, '|', panel.index.min().date(), '->', panel.index.max().date())
panel.head(3)"""))

cells.append(md(r"""## 5. Evaluación estandarizada y fila de métricas (23 columnas)

`ev.evaluate` con `market_returns` (validación económica: retorno medio por estado) y
`X_full` (logL/AIC/BIC). Guardamos la fila en `results/metrics_05_markov_switching_var.csv`."""))

cells.append(code(r"""res = ev.evaluate(det, panel, market_returns=mr.reindex(panel.index), X_full=X)
row = ev.results_table([res])
out_csv = RESULTS / 'metrics_05_markov_switching_var.csv'
row.to_csv(out_csv, index=False)
print('ventana_eval:', res.extra['ventana_eval'])
print('retorno medio por estado canónico:', {k: round(v, 5) for k, v in res.extra['mean_return_by_state'].items()})
print('Guardado:', out_csv, '| columnas =', row.shape[1])
row.T"""))

cells.append(md(r"""## 6. S&P 500 coloreado por régimen (out-of-sample)

Días OOS clasificados como **crisis** sombreados en rojo; bandas de crisis conocidas
(rojo claro) y trampas 2013/2018 (naranja)."""))

cells.append(code(r"""spx = raw['SP500'].reindex(panel.index)
is_crisis = (panel['state'] == det.crisis_state)
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(spx.index, spx, color='black', lw=0.7, zorder=3)
ax.set_yscale('log'); ax.set_ylabel('S&P 500 (log)')
ymin, ymax = ax.get_ylim()
ax.fill_between(panel.index, ymin, ymax, where=is_crisis.values, color='crimson',
                alpha=0.25, step='mid', zorder=1)
ax.set_ylim(ymin, ymax)
for a, b in ev.CRISIS_WINDOWS.values():
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='red', alpha=0.10, zorder=0)
for a, b in ev.FALSE_POSITIVE_WINDOWS.values():
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='orange', alpha=0.18, zorder=0)
handles = [Patch(color='crimson', alpha=0.25, label='D5 crisis (OOS, alta varianza)'),
           Patch(color='red', alpha=0.10, label='Ventanas crisis'),
           Patch(color='orange', alpha=0.18, label='Trampas 2013/2018')]
ax.legend(handles=handles, loc='upper left', fontsize=8)
ax.set_title('D5 markov_switching_var — S&P 500 coloreado por régimen (out-of-sample)')
fig.tight_layout(); fig.savefig(RESULTS / 'd5_msvar_sp500_regimes.png', dpi=110, bbox_inches='tight')
plt.show()"""))

cells.append(md(r"""## 7. Probabilidad FILTRADA de crisis (causal, OOS)

`p_crisis` out-of-sample = P(régimen de alta varianza | y≤t), filtrada (causal). Banda
inferior = timeline de régimen (rojo = crisis)."""))

cells.append(code(r"""fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})
ax1.plot(panel.index, panel['p_crisis'].values, color='crimson', lw=0.8)
ax1.fill_between(panel.index, 0, panel['p_crisis'].values, color='crimson', alpha=0.20, step='mid')
ax1.axhline(0.5, color='grey', ls='--', lw=0.8)
ax1.set_ylabel('p_crisis (filtrada)'); ax1.set_ylim(0, 1)
ax1.set_title('D5 — Probabilidad FILTRADA de crisis (causal, out-of-sample)')
for a, b in ev.CRISIS_WINDOWS.values():
    ax1.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='red', alpha=0.10)
for a, b in ev.FALSE_POSITIVE_WINDOWS.values():
    ax1.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='orange', alpha=0.18)
ax2.fill_between(panel.index, 0, 1, where=is_crisis.values, color='crimson', alpha=0.6, step='mid')
ax2.set_yticks([]); ax2.set_ylabel('régimen'); ax2.set_title('Timeline (rojo = crisis)')
fig.tight_layout(); fig.savefig(RESULTS / 'd5_msvar_crisis_proba.png', dpi=110, bbox_inches='tight')
plt.show()"""))

cells.append(md(r"""## 8. Filtrada vs smoothed IN-SAMPLE (efecto del look-ahead)

Comparación, sobre el ajuste de toda la muestra, de la probabilidad de crisis
**FILTRADA** (causal, P(S_t|y≤t)) vs **SMOOTHED** (NO causal, P(S_t|y₁..T), usa todo el
futuro). La smoothed es más nítida/anticipada porque mira el futuro: ilustra el sesgo de
look-ahead que la evaluación causal evita. **No comparable** con la versión walk-forward."""))

cells.append(code(r"""cs = det.crisis_state
p_filt = det.insample_proba('filtered')[:, cs]
p_smooth = det.insample_proba('smoothed')[:, cs]
idx = X.index
fig, ax = plt.subplots(figsize=(15, 4.5))
ax.plot(idx, p_smooth, color='steelblue', lw=0.7, label='smoothed (NO causal, look-ahead)')
ax.plot(idx, p_filt, color='crimson', lw=0.7, alpha=0.8, label='filtered (causal)')
ax.set_ylabel('p_crisis in-sample'); ax.set_ylim(0, 1)
for a, b in ev.CRISIS_WINDOWS.values():
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='red', alpha=0.08)
ax.legend(loc='upper left', fontsize=8)
ax.set_title('D5 — IN-SAMPLE: probabilidad de crisis FILTRADA (causal) vs SMOOTHED (look-ahead)')
fig.tight_layout(); fig.savefig(RESULTS / 'd5_msvar_filtered_vs_smoothed.png', dpi=110, bbox_inches='tight')
plt.show()
print('correlación filtered/smoothed:', round(float(np.corrcoef(p_filt, p_smooth)[0, 1]), 3))
print('media |smoothed - filtered|:', round(float(np.abs(p_smooth - p_filt).mean()), 4))"""))

cells.append(md(r"""## 9. Matriz de transición (persistencia de regímenes)

Matriz fila-estocástica P(S_t=j | S_{t-1}=i) en orden canónico. La diagonal alta da
persistencia (calma y crisis se "pegan"); fuera de diagonal = probabilidad de
conmutación diaria."""))

cells.append(code(r"""A = det.transition_canonical()
labels = ['calma', 'crisis']
fig, ax = plt.subplots(figsize=(4.6, 4))
im = ax.imshow(A, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
ax.set_xlabel('a estado (t)'); ax.set_ylabel('desde estado (t-1)')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{A[i, j]:.3f}', ha='center', va='center',
                color='white' if A[i, j] > 0.5 else 'black', fontsize=11)
ax.set_title('D5 — Matriz de transición (canónica)')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig(RESULTS / 'd5_msvar_transition.png', dpi=110, bbox_inches='tight')
plt.show()
dur_calma = 1.0 / (1.0 - A[0, 0]); dur_crisis = 1.0 / (1.0 - A[1, 1])
print(f'Persistencia esperada: calma ≈ {dur_calma:.0f} días | crisis ≈ {dur_crisis:.0f} días')"""))

cells.append(md(r"""## 10. Verificación explícita contra crisis y trampas

Cobertura (% días crisis) en cada ventana de crisis (alto = bueno) y en cada trampa
2013/2018 (bajo = bueno). Con histórico largo, **2008 y 2011 son OOS**."""))

cells.append(code(r"""states_oos = panel['state']
cov = ev.crisis_coverage(states_oos, det.crisis_state)
fa  = ev.false_alarm_in_windows(states_oos, det.crisis_state)
print('=== COBERTURA EN CRISIS (alto = bueno) ===')
for k, v in cov.items():
    flag = 'sin OOS' if v != v else ('OK' if v >= 0.5 else 'BAJA')
    print(f'  {k:16s}: ' + ('  NaN' if v != v else f'{v:6.1%}') + f'  [{flag}]')
print('\n=== ACTIVACIÓN EN TRAMPAS (bajo = bueno) ===')
for k, v in fa.items():
    print(f'  {k:16s}: ' + ('  NaN' if v != v else f'{v:6.1%}'))
print(f'\nfalse_alarm_rate global: {res.false_alarm_rate:.2%}')
print(f'switching_rate: {res.switching_rate:.4f} | duración media: {res.mean_regime_duration:.1f} días'
      f' | label_stability: {res.label_stability:.3f}')
print(f'logL={res.log_likelihood:.1f}  AIC={res.aic:.1f}  BIC={res.bic:.1f}')"""))

cells.append(md(r"""## 11. Conclusión y contraste con la hipótesis del CHECKPOINT 2

Hipótesis CP2 (D5): *baseline econométrico interpretable; capta calma/estrés; punto ciego
en crisis rápidas; univariante no ve correlación cross-asset; gaussiano insuficiente para
colas.* Los números de arriba (cobertura por crisis incl. **2008/2011 OOS**, activación
en trampas 2013/2018, AIC/BIC, persistencia, flickering) confirman o matizan la
hipótesis. Detalle en `docs/memory/detectors/05_markov_switching_var.md`."""))

cells.append(code(r"""print('crisis = ALTA varianza confirmado:', bool(np.isclose(vars_c[det.crisis_state], vars_c.max())))
crisis_evaluables = {k: v for k, v in cov.items() if v == v}
print('crisis evaluables OOS:', list(crisis_evaluables.keys()))
print('2008 OOS cubierta:', '2008' in ' '.join(k for k, v in cov.items() if v == v),
      '-> GFC_2008 cov =', round(cov['GFC_2008'], 3) if cov['GFC_2008'] == cov['GFC_2008'] else 'NaN')
print('¿capta trampas 2013/2018? act:',
      {k: (round(v, 3) if v == v else None) for k, v in fa.items()})"""))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}

ep = ExecutePreprocessor(timeout=2400, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": str(ROOT / "notebooks")}})
nbf.write(nb, NB_PATH)

n_err = sum(1 for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error")
print(f"[build_05] escrito {NB_PATH}")
print(f"[build_05] celdas={len(nb.cells)}  errores={n_err}")
if n_err:
    for c in nb.cells:
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                print("ERROR:", o.get("ename"), o.get("evalue"))
