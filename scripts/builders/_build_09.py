# -*- coding: utf-8 -*-
"""Constructor del notebook 09_jump_model.ipynb (D9, FASE 3 Tanda 4 - EXPLORATORIA).

Crea las celdas, ejecuta con nbconvert (0 errores esperados) y guarda el .ipynb con
las figuras inline. Vuelca figuras a results/ y la fila de metricas (23 columnas) a
results/metrics_09_jump_model.csv, y refresca results/metrics_master.csv.

STEP elegido = 21 (refit ~mensual), igual que D6/D10 para comparacion JUSTA con D3.
Coste estimado: fit ~2.6s @2000 filas -> ~6.6s @4000 filas (lineal). Train de 252*8=2016
filas, OOS hasta 4665 -> ~127 folds, fit medio ~5s -> ~11-13 min de walk-forward, por
debajo del guardarrail de 15 min (timeout nbconvert = 2400s, amplio margen). No se sube
el step para no perder resolucion OOS frente al rival D3 (que usa el protocolo estandar).

Uso:  python scripts/builders/_build_09.py
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[2]
NB_PATH = ROOT / "notebooks" / "09_jump_model.ipynb"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ------------------------------------------------------------------ #
md(r"""# 09 — D9 `jump_model`: Statistical Jump Model (Nystrup et al., 2020)

Familia **CLUSTERING con regularización temporal** — Nystrup, Lindström & Madsen (2020),
*"Learning Hidden Markov Models with Persistent States by Penalizing Jumps"*.

Un **Statistical Jump Model (SJM)** es un clustering de estados (como k-means / GMM) PERO
con una **penalización de salto** $\lambda$ que castiga cambiar de estado entre $t$ y $t+1$.
El ajuste minimiza, por descenso por coordenadas (DP sobre la secuencia + recálculo de
centroides):

$$ \min_{\{s_t\},\,\{\mu_k\}}\ \sum_t \lVert x_t - \mu_{s_t}\rVert^2 \;+\; \lambda \sum_t \mathbf{1}[\,s_t \neq s_{t-1}\,]. $$

$\lambda$ introduce una **histéresis APRENDIDA** → persistencia, online, anti-flickering.

## El rival honesto de D3
D9 usa **exactamente las mismas 15 features causales que D3** (`clustering_gmm`) y la misma
evaluación walk-forward. La única diferencia es el término $\lambda\sum 1[s_t\neq s_{t-1}]$.
Por eso D9 aísla **cuánto aporta la persistencia temporal** sobre el clustering puro:

> **Hipótesis D9:** *frente a D3 (GMM estático, sin término temporal: switching≈0.126,
> duración media≈7.9 d → flickering alto), D9 debe dar MENOS switching y MAYOR duración
> media SIN perder cobertura de crisis, gracias a $\lambda$.* Es también el rival clásico
> de D12 (autoencoder).

## Vía de implementación
Librería **`jumpmodels`** (Nystrup et al.), coordinate-descent + DP estándar, con métodos
**causales** `predict_online` / `predict_proba_online` (la etiqueta de la fila $i$ usa solo
filas $<i$). Las features se reescalan con un `StandardScaler` ajustado **solo con el train**
(causal) porque el SJM mide distancias euclídeas y 3 de las 15 features (corr, drawdown,
momentum) tienen escala ~0.1–0.3 y quedarían infra-ponderadas.

**Estados:** $K=2$ (bull/bear clásico de Nystrup; crisis = estado de alta vol). $\lambda=50$
sobre features estandarizadas (persistencia ~mensual sin congelar la señal).""")

# ------------------------------------------------------------------ #
co(r"""%matplotlib inline
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
warnings.filterwarnings('ignore')
ROOT = Path.cwd()
while not (ROOT / 'src').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / 'results'; RESULTS.mkdir(exist_ok=True)
from src import evaluation as ev
from detectors.jump_model import JumpModel

# Mismas 15 features causales que D3 (rival directo), ventana 2007+.
X = pd.read_parquet(ROOT / 'data' / 'processed' / 'features.parquet')
X.index = pd.to_datetime(X.index); X = X.sort_index()
FEATURES = list(X.columns)
raw = pd.read_parquet(ROOT / 'data' / 'raw' / 'raw_panel.parquet')
mkt = np.log(raw['SP500'] / raw['SP500'].shift(1)).reindex(X.index)
mkt.name = 'SP500_ret'                              # market_returns (S&P 500 log-ret)
sp500 = raw['SP500'].reindex(X.index)
N_STATES = 2; JUMP_PENALTY = 50.0
print('X (15 features causales, =D3):', X.shape, '|', X.index.min().date(), '->', X.index.max().date())
print('Features:', FEATURES)
print(f'n_states={N_STATES}  jump_penalty(lambda)={JUMP_PENALTY}')""")

# ------------------------------------------------------------------ #
md(r"""## 1. Ajuste in-sample: estados del jump model y orientación económica

Ajuste sobre toda la muestra (solo para inspeccionar; la clasificación comparable es la
**causal** de §3). El orden económico (0=calma · 1=crisis) lo fija `label_states_economically`
con el retorno del S&P 500 (vol-primario, Arreglo 4).""")

co(r"""det_is = JumpModel(n_states=N_STATES, jump_penalty=JUMP_PENALTY).fit(X)
det_is.label_states_economically(X, market_returns=mkt)   # orden económico con S&P 500
states_is = pd.Series(det_is.predict(X), index=X.index, name='state')

print('Retorno/vol del S&P 500 por estado canónico (in-sample):')
for s in sorted(states_is.unique()):
    r = mkt[states_is == s]
    print(f'  estado {s}: n={len(r):5d}  mean_ret={r.mean():+.5f}  vol_ret={r.std():.5f}')

# VERIFICACION: crisis canonico = ALTA vol (no invertido)
r_cri = mkt[states_is == det_is.crisis_state]; r_cal = mkt[states_is == 0]
assert r_cri.std() > r_cal.std(), 'INVERTIDO: crisis deberia ser ALTA vol'
print(f'\nOK -> crisis (estado {det_is.crisis_state}) = ALTA vol ({r_cri.std():.4f} > {r_cal.std():.4f}) '
      f'y menor retorno. No invertido (in-sample).')
print('switching in-sample =', round(ev.switching_rate(states_is), 4),
      '| duracion media =', round(ev.mean_regime_duration(states_is), 1), 'd')""")

# ------------------------------------------------------------------ #
md(r"""## 2. Verificación de CAUSALIDAD de `predict_online`

`predict_online` asigna la fila $i$ usando **solo filas $<i$** del bloque. Test: añadir días
FUTUROS al bloque NO debe cambiar las etiquetas online de los días del bloque.""")

co(r"""det_c = JumpModel(n_states=N_STATES, jump_penalty=JUMP_PENALTY).fit(X.loc[:'2015-12-31'])
block = X.loc['2016-01-01':'2016-06-30']
lab_block = np.asarray(det_c.predict_online(block))
lab_long  = np.asarray(det_c.predict_online(X.loc['2016-01-01':'2017-12-31']))[:len(block)]
ndiff = int((lab_block != lab_long).sum())
print(f'dias del bloque cuya etiqueta online cambia al anadir futuro = {ndiff} / {len(block)}')
assert ndiff == 0, 'predict_online NO es causal: mira el futuro del bloque'
print('causal_ok = True -> predict_online usa solo el pasado del bloque (sin look-ahead).')""")

# ------------------------------------------------------------------ #
md(r"""## 3. Versión CAUSAL walk-forward (la comparable)

`ev.walk_forward` reentrena (re-fit expanding + StandardScaler con solo el train, λ congelada)
en ventanas de train inicial **8 años** y predice el bloque de `step=21` días con
`predict_online`. **Se pasa `market_returns=mkt`** para re-fijar el orden económico de forma
robusta (la señal es clustering multivariante, no retorno crudo).""")

co(r"""TRAIN_SIZE = 252 * 8   # ~8 anios: 2008/2011/2020/2022 OOS; 2013/2018 (trampas) OOS
STEP = 21              # refit ~mensual, igual que D6/D10 (comparacion justa con D3)
panel = ev.walk_forward(lambda: JumpModel(n_states=N_STATES, jump_penalty=JUMP_PENALTY), X,
                        market_returns=mkt, train_size=TRAIN_SIZE, step=STEP, expanding=True)
print('OOS:', panel.index.min().date(), '->', panel.index.max().date(), '| n_oos =', len(panel))
states_c = panel['state']; p_c = panel['p_crisis']

res = ev.evaluate(det_is, panel, market_returns=mkt, X_full=X)
res.detector_name = 'jump_model'
print('\nventana_eval:', res.extra['ventana_eval'])
print('\nCobertura de crisis (CAUSAL OOS):')
for k, v in res.crisis_coverage.items(): print(f'  {k:16s}: {v:6.1%}')
print('Falsas alarmas en ventanas TRAMPA (CAUSAL OOS):')
for k, v in res.false_alarm_in_fp.items(): print(f'  {k:16s}: {v:6.1%}')
print(f'\nfalse_alarm_rate = {res.false_alarm_rate:.3f} | switching = {res.switching_rate:.4f} '
      f'| dur media = {res.mean_regime_duration:.1f} d | label_stability = {res.label_stability:.3f}')

# VERIFICACION CRITICA en WALK-FORWARD: crisis = alta vol de retornos reales (no invertido)
print('\nVERIFICACION ORIENTACION en WALK-FORWARD (retornos reales por estado canonico):')
for s in sorted(states_c.unique()):
    r = mkt.reindex(states_c.index)[states_c == s]
    print(f'  estado {s}: n={len(r):5d}  mean_ret={r.mean():+.5f}  vol_ret={r.std():.5f}')
r_cri = mkt.reindex(states_c.index)[states_c == det_is.crisis_state]
r_cal = mkt.reindex(states_c.index)[states_c == 0]
assert r_cri.std() > r_cal.std(), 'INVERTIDO en walk-forward: crisis deberia ser ALTA vol'
print(f'OK -> crisis = ALTA vol de retornos ({r_cri.std():.4f} > {r_cal.std():.4f}) '
      f'y menor retorno medio. No invertido.')""")

# ------------------------------------------------------------------ #
md(r"""## 4. Persistencia: D9 (jump model) vs D3 (GMM estático)

El contraste central de este notebook. D3 (`clustering_gmm_k3`) en `metrics_master`:
switching≈**0.126**, duración media≈**7.9 d** (flickering alto, sin término temporal). D9
debe mejorar ambos gracias a $\lambda$.""")

co(r"""cmp_rows = [{'detector': 'D9 jump_model (lambda=%g)' % JUMP_PENALTY,
             'switching_rate': res.switching_rate,
             'mean_regime_duration': res.mean_regime_duration}]
mpath = RESULTS / 'metrics_master.csv'
if mpath.exists():
    m = pd.read_csv(mpath)
    for d in ['clustering_gmm_k3', 'clustering_gmm_k2', 'turbulence_mahalanobis']:
        r = m[m['detector'] == d]
        if len(r):
            cmp_rows.append({'detector': 'D3/otros: ' + d,
                             'switching_rate': float(r['switching_rate'].iloc[0]),
                             'mean_regime_duration': float(r['mean_regime_duration'].iloc[0])})
cmp = pd.DataFrame(cmp_rows).set_index('detector')
display(cmp.style.format({'switching_rate': '{:.4f}', 'mean_regime_duration': '{:.1f}'}))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
a1.barh(cmp.index, cmp['switching_rate'].values, color=['#27ae60', '#c0392b', '#e67e22', '#7f8c8d'][:len(cmp)])
a1.set_title('switching_rate (menor = mas persistente)'); a1.invert_yaxis()
a2.barh(cmp.index, cmp['mean_regime_duration'].values, color=['#27ae60', '#c0392b', '#e67e22', '#7f8c8d'][:len(cmp)])
a2.set_title('duracion media de regimen (dias)'); a2.invert_yaxis()
fig.tight_layout(); fig.savefig(RESULTS / 'd09_persistence_vs_d3.png', dpi=110, bbox_inches='tight'); plt.show()
print('Lectura: si D9 (verde) tiene switching mas bajo y duracion mas alta que D3 (rojo),')
print('la penalizacion de salto lambda ha anadido persistencia sin la dinamica de Markov de un HMM.')""")

# ------------------------------------------------------------------ #
md(r"""## 5. S&P 500 coloreado por régimen (CAUSAL OOS)

Sombreado rojo = días clasificados **crisis** por el walk-forward causal.""")

co(r"""def shade_regime(ax, states, crisis_state, color='red', alpha=0.25):
    v = (states == crisis_state).astype(int).values; idx = states.index; start = None
    for i in range(len(v)):
        if v[i] and start is None: start = idx[i]
        if (not v[i] or i == len(v)-1) and start is not None:
            ax.axvspan(start, idx[i], color=color, alpha=alpha); start = None

fig, ax = plt.subplots(figsize=(15, 5))
px = sp500.reindex(states_c.index)
ax.plot(px.index, px.values, color='black', lw=0.7)
ax.set_yscale('log')
shade_regime(ax, states_c, det_is.crisis_state, color='red', alpha=0.22)
for a, b in ev.CRISIS_WINDOWS.values():
    ax.axvline(pd.Timestamp(a), color='darkred', ls='--', lw=0.6); ax.axvline(pd.Timestamp(b), color='darkred', ls=':', lw=0.6)
for a, b in ev.FALSE_POSITIVE_WINDOWS.values():
    ax.axvline(pd.Timestamp(a), color='darkorange', ls='--', lw=0.6)
ax.set_title('S&P 500 (log) coloreado por regimen CAUSAL OOS — D9 jump_model\n(rojo = estado crisis del jump model; naranja punteado = trampas 2013/2018)')
ax.legend(handles=[Patch(color='red', alpha=0.22, label='crisis (alta vol, estado SJM)')], loc='upper left')
ax.margins(x=0.01); fig.tight_layout()
fig.savefig(RESULTS / 'd09_sp500_regimes.png', dpi=110, bbox_inches='tight'); plt.show()""")

# ------------------------------------------------------------------ #
md(r"""## 6. Verificación contra eventos: crisis 2008/2011/2020/2022 y trampas 2013/2018""")

co(r"""rows = []
for k in ev.CRISIS_WINDOWS:
    rows.append({'ventana': k, 'tipo': 'crisis', 'cobertura_OOS': res.crisis_coverage.get(k, float('nan'))})
for k in ev.FALSE_POSITIVE_WINDOWS:
    rows.append({'ventana': k, 'tipo': 'trampa', 'cobertura_OOS': res.false_alarm_in_fp.get(k, float('nan'))})
cmpw = pd.DataFrame(rows).set_index('ventana')
display(cmpw.style.format({'cobertura_OOS': '{:.1%}'}))

fig, ax = plt.subplots(figsize=(11, 4.2))
colors = ['#c0392b' if t == 'crisis' else '#e67e22' for t in cmpw['tipo']]
ax.bar(cmpw.index, cmpw['cobertura_OOS'].values, color=colors)
ax.axhline(0.5, color='grey', ls='--', lw=0.8)
ax.set_ylabel('% dias en estado crisis (OOS)')
ax.set_title('D9 jump_model — cobertura por ventana (rojo=crisis, naranja=trampa)')
ax.set_xticklabels(cmpw.index, rotation=30, ha='right'); ax.set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig(RESULTS / 'd09_coverage.png', dpi=110, bbox_inches='tight'); plt.show()""")

# ------------------------------------------------------------------ #
md(r"""## 7. Timeline de régimen y duraciones (flickering)

Timeline causal OOS + histograma de duraciones. La penalización de salto $\lambda$ debe dar
episodios largos (poco flickering) frente a D3.""")

co(r"""fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 4.5), gridspec_kw={'height_ratios': [1, 2.2]})
ax1.plot(p_c.index, p_c.values, color='#2980b9', lw=0.7); ax1.set_ylabel('P(crisis)')
for a, b in ev.CRISIS_WINDOWS.values(): ax1.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='red', alpha=0.12)
for a, b in ev.FALSE_POSITIVE_WINDOWS.values(): ax1.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='orange', alpha=0.15)
ax1.set_title('P(crisis) (one-hot causal del SJM discreto) — CAUSAL OOS'); ax1.margins(x=0.01)
ax2.imshow(states_c.values.reshape(1, -1), aspect='auto', cmap='RdYlGn_r',
           extent=[0, len(states_c), 0, 1]); ax2.set_yticks([])
tk = np.linspace(0, len(states_c)-1, 8).astype(int)
ax2.set_xticks(tk); ax2.set_xticklabels([states_c.index[i].year for i in tk])
ax2.set_title('Timeline de regimen (verde=calma, rojo=crisis)')
fig.tight_layout(); fig.savefig(RESULTS / 'd09_timeline.png', dpi=110, bbox_inches='tight'); plt.show()

def episode_durations(states):
    v = states.values; out = {s: [] for s in range(N_STATES)}; run = 1
    for i in range(1, len(v)):
        if v[i] == v[i-1]: run += 1
        else: out[int(v[i-1])].append(run); run = 1
    out[int(v[-1])].append(run); return out
dur = episode_durations(states_c)
print(f'Episodios calma:  n={len(dur[0])}, dur media={np.mean(dur[0]) if dur[0] else float("nan"):.1f} d')
print(f'Episodios crisis: n={len(dur[N_STATES-1])}, dur media={np.mean(dur[N_STATES-1]) if dur[N_STATES-1] else float("nan"):.1f} d')
print(f'switching_rate={res.switching_rate:.4f}  dur media global={res.mean_regime_duration:.1f} d')""")

# ------------------------------------------------------------------ #
md(r"""## 8. Volcado de métricas a results/ (esquema 23 columnas)""")

co(r"""tbl = ev.results_table([res])
assert tbl.shape[1] == 23, f'esperaba 23 columnas, hay {tbl.shape[1]}'
tbl.to_csv(RESULTS / 'metrics_09_jump_model.csv', index=False)
print('Guardado results/metrics_09_jump_model.csv  (1 fila,', tbl.shape[1], 'columnas)')

master_path = RESULTS / 'metrics_master.csv'
if master_path.exists():
    master = pd.read_csv(master_path)
    master = master[master['detector'] != 'jump_model']
    master = pd.concat([master, tbl], ignore_index=True)
else:
    master = tbl.copy()
master.to_csv(master_path, index=False)
print('master actualizado:', master.shape)
display(tbl.T)""")

# ------------------------------------------------------------------ #
md(r"""## 9. Conclusión D9 — ¿añade persistencia el jump model sobre D3?

**Hipótesis D9:** *el Statistical Jump Model, con su penalización de salto $\lambda$, debe dar
MENOS flickering (menor switching, mayor duración) que D3 (GMM estático) sobre las MISMAS 15
features, sin perder cobertura de crisis.*

Veredicto (con los números de arriba):
- **Clustering con histéresis aprendida:** $K=2$, $\lambda=50$; crisis = estado de **alta vol**
  (verificado en walk-forward con S&P 500, **no invertido**, sin fallback).
- **Causal:** `predict_online` / `predict_proba_online` usan solo el pasado del bloque
  (verificado: añadir futuro no cambia la etiqueta) + StandardScaler train-only por fold.
- **Persistencia vs D3 (§4):** comparar switching_rate y duración media de D9 frente a D3
  (0.126 / 7.9 d). El delta es la aportación NETA de $\lambda$ sobre el clustering puro.
- **vs D12 (autoencoder):** ambos son detectores no-HMM sobre las features causales; D9 es el
  baseline de clustering+persistencia contra el que se mide el aprendizaje de representación
  del AE.

(El veredicto numérico definitivo lo cierra el ORQUESTADOR en
`docs/memory/detectors/09_jump_model.md` tras el build completo.)""")

# ------------------------------------------------------------------ #
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
print('Ejecutando notebook (walk-forward ~11-13 min con step=21)...')
ep = ExecutePreprocessor(timeout=2400, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': str(ROOT / 'notebooks')}})
nbf.write(nb, NB_PATH)
n_err = sum(1 for c in nb.cells for o in c.get('outputs', []) if o.get('output_type') == 'error')
print(f'OK -> {NB_PATH.name}  | errores = {n_err}')
