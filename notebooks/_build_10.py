# -*- coding: utf-8 -*-
"""Constructor del notebook 10_turbulence_mahalanobis.ipynb (D10, FASE 3 Tanda 3).

Crea las celdas, ejecuta con nbconvert (0 errores esperados) y guarda el .ipynb con
las figuras inline. Vuelca figuras a results/ y la fila de métricas (23 columnas) a
results/metrics_10_turbulence_mahalanobis.csv, y refresca results/metrics_master.csv.

Uso:  python notebooks/_build_10.py
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "10_turbulence_mahalanobis.ipynb"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ------------------------------------------------------------------ #
md(r"""# 10 — D10 `turbulence_mahalanobis`: índice de turbulencia (Mahalanobis expanding causal)

Familia **F1 (Multivariante)** — Kritzman, Page & Turkington (2012); Gulko (2002).
**Índice de turbulencia financiera** = distancia de Mahalanobis del vector de mercado
del día respecto a su distribución histórica,

$$ d_t = (x_t - \mu)^\top \Sigma^{-1} (x_t - \mu), $$

con $\mu$ y $\Sigma$ estimadas de forma **CAUSAL EXPANDING** (solo datos $< t$, nunca la
propia $x_t$ ni el futuro). $d_t$ es grande cuando el vector de mercado es **raro** respecto
a su covarianza histórica: magnitudes extremas **o** un **patrón de co-movimiento atípico**
(colapso / inversión de correlaciones). Esa segunda parte — la geometría de $\Sigma$ — es lo
que las reglas **univariantes** (D1 VIX, D6 GARCH-equity) **no ven**.

El régimen binario (0=calma, 1=crisis) se obtiene **umbralizando** $d_t$ con un percentil del
train (`q_in`) + **histéresis** (`q_out`) + **dwell** (mismo autómata causal de D1/D6).

## Hipótesis CP2 (la que este notebook pone a prueba)
> *"D10 capta el **colapso de correlaciones multivariante** que las reglas univariantes no
> ven. En particular DEBERÍA captar **2013 (taper tantrum)** — el agujero que D4 (HMM
> gaussiano) y D6 (GARCH equity) NO tapan — porque el taper reordenó conjuntamente
> equity/tipos/divisa/curva aunque la vol del equity fuera modesta."*

**Ventana — 2013 OOS (contraste clave).** Vector de **cambios causales desde 1990**:
`SP500_ret`, `VIX_change`, `DXY_change`, `yield_slope_chg`. Sin HYG/oro (restringirían a
2007 y mandarían 2013 al train). Con train de ~8 años → OOS desde ~1998 → **2008/2011/2020/2022
OOS** (4 crisis) y **2013/2018 OOS** (trampas).""")

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
from detectors.turbulence_mahalanobis import TurbulenceMahalanobis

raw = pd.read_parquet(ROOT / 'data' / 'raw' / 'raw_panel.parquet')
# Vector multivariante de CAMBIOS CRUDOS (la Sigma^-1 de Mahalanobis estandariza/decorrela).
spx_ret   = np.log(raw['SP500'] / raw['SP500'].shift(1)).rename('SP500_ret')
vix_chg   = raw['VIX'].diff().rename('VIX_change')
dxy_chg   = np.log(raw['DXY'] / raw['DXY'].shift(1)).rename('DXY_change')
slope_chg = raw['YIELD_10Y_3M'].diff().rename('yield_slope_chg')
X = pd.concat([spx_ret, vix_chg, dxy_chg, slope_chg], axis=1).dropna()
X.index = pd.to_datetime(X.index); X = X.sort_index()
FEATURES = list(X.columns)
mkt = X['SP500_ret']                              # market_returns (S&P 500 log-ret)
sp500 = raw['SP500'].reindex(X.index)
print('X (multivariante, histórico largo):', X.shape, '|', X.index.min().date(), '->', X.index.max().date())
print('Features Mahalanobis:', FEATURES)
print('2013 debe quedar OOS (train ~8 anios desde 1990) -> se verifica en la seccion 3.')""")

# ------------------------------------------------------------------ #
md(r"""## 1. Ajuste in-sample: turbulencia de Mahalanobis y umbral

Ajuste sobre toda la muestra (solo para inspeccionar la señal; la clasificación comparable es
la **causal** de §3). El orden económico (0=calma · 1=crisis) lo fija
`label_states_economically` con el retorno del S&P 500.""")

co(r"""det_is = TurbulenceMahalanobis(features=FEATURES).fit(X)
det_is.label_states_economically(X, market_returns=mkt)   # orden económico con S&P 500
d_is = det_is.turbulence(X)
states_is = pd.Series(det_is.predict(X), index=X.index, name='state')
print(f'tau_in  (p{int(det_is.q_in*100)} de d en train)  = {det_is._tau_in:.3f}')
print(f'tau_out (p{int(det_is.q_out*100)} de d en train)  = {det_is._tau_out:.3f}')
print(f'turbulencia in-sample: media={d_is.mean():.2f}  max={d_is.max():.1f}  (dim={len(FEATURES)})')

# VERIFICACION: crisis canonico = ALTA turbulencia (no invertido)
d_cri = d_is[states_is.values == det_is.crisis_state].mean()
d_cal = d_is[states_is.values == 0].mean()
print(f'\nturbulencia media CRISIS={d_cri:.2f}  vs  CALMA={d_cal:.2f}')
assert d_cri > d_cal, 'INVERTIDO: crisis deberia ser ALTA turbulencia'
print(f'OK -> crisis (estado canonico {det_is.crisis_state}) = ALTA turbulencia. No invertido (in-sample).')""")

# ------------------------------------------------------------------ #
md(r"""## 2. Verificación de CAUSALIDAD de la turbulencia

La covarianza Mahalanobis es **expanding causal**: $d_t$ usa solo filas $< t$ (más burn-in del
train). Test: ocultar el futuro NO debe cambiar $d_t$ del bloque.""")

co(r"""det_c = TurbulenceMahalanobis(features=FEATURES).fit(X.loc[:'2007-12-31'])
block   = X.loc['2008-01-01':'2008-12-31']
d_block = det_c.turbulence(block)
d_plus  = det_c.turbulence(X.loc['2008-01-01':'2010-12-31']).loc[d_block.index]
maxdiff = float((d_block - d_plus).abs().max())
print(f'max |d_bloque(ver futuro) - d_bloque(ocultar futuro)| = {maxdiff:.2e}')
assert maxdiff < 1e-9, 'La turbulencia del bloque NO es causal'
print('causal_ok = True  -> d_t usa solo filas <= t (burn-in del train propaga la covarianza expanding)')""")

# ------------------------------------------------------------------ #
md(r"""## 3. Versión CAUSAL walk-forward (la comparable)

`ev.walk_forward` reentrena (recalcula el umbral; la covarianza es expanding) en ventanas
**expanding** (train inicial **8 años**) y predice el bloque de `step=21` días. **Se pasa
`market_returns=mkt`** para re-fijar el orden económico de forma robusta (la señal es
turbulencia, no retorno → sin esto el etiquetado podría invertirse).""")

co(r"""TRAIN_SIZE = 252 * 8   # ~8 anios: 2008/2011/2013/2020/2022 caen OOS (desde 1990)
STEP = 21
panel = ev.walk_forward(lambda: TurbulenceMahalanobis(features=FEATURES), X, market_returns=mkt,
                        train_size=TRAIN_SIZE, step=STEP, expanding=True)
print('OOS:', panel.index.min().date(), '->', panel.index.max().date(), '| n_oos =', len(panel))
assert panel.index.min() < pd.Timestamp('2013-05-01'), '2013 NO quedo OOS!'
print('CONFIRMADO: 2013 (taper) cae OOS ->', panel.index.min().date(), '< 2013-05-01')
states_c = panel['state']; p_c = panel['p_crisis']

res = ev.evaluate(det_is, panel, market_returns=mkt, X_full=X)
res.detector_name = 'turbulence_mahalanobis'
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
print(f'OK -> crisis = ALTA vol de retornos ({r_cri.std():.4f} > {r_cal.std():.4f}) y menor retorno medio. No invertido.')""")

# ------------------------------------------------------------------ #
md(r"""## 4. Serie de turbulencia con umbral y ventanas de crisis

Índice de turbulencia de Mahalanobis (in-sample) con el umbral $\tau_{in}$. Bandas rojas =
ventanas de crisis conocidas; naranjas = trampas (2013/2018).""")

co(r"""fig, ax = plt.subplots(figsize=(15, 4.5))
ax.plot(d_is.index, d_is.values, color='#34495e', lw=0.6, label='turbulencia $d_t$ (Mahalanobis)')
ax.axhline(det_is._tau_in, color='#c0392b', ls='--', lw=1.0, label=f'tau_in (p{int(det_is.q_in*100)})')
ax.axhline(det_is._tau_out, color='#e67e22', ls=':', lw=1.0, label=f'tau_out (p{int(det_is.q_out*100)})')
for a, b in ev.CRISIS_WINDOWS.values(): ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='red', alpha=0.13)
for a, b in ev.FALSE_POSITIVE_WINDOWS.values(): ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='orange', alpha=0.18)
ax.set_title('D10 turbulence_mahalanobis — indice de turbulencia (Mahalanobis expanding causal) con umbral\n(rojo=crisis conocidas, naranja=trampas 2013/2018)')
ax.set_ylabel('turbulencia $d_t$'); ax.legend(loc='upper right', fontsize=9); ax.margins(x=0.01)
ax.set_ylim(0, np.nanpercentile(d_is.values, 99.5))
fig.tight_layout(); fig.savefig(RESULTS / 'd10_turbulence_threshold.png', dpi=110, bbox_inches='tight'); plt.show()""")

# ------------------------------------------------------------------ #
md(r"""## 5. S&P 500 coloreado por régimen (CAUSAL OOS)

Sombreado rojo = días clasificados **crisis** por el walk-forward causal. Cubre 2008/2011
(OOS) además de 2013 (la trampa-clave).""")

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
ax.set_title('S&P 500 (log) coloreado por regimen CAUSAL OOS — D10 turbulence_mahalanobis\n(rojo = turbulencia alta; naranja punteado = trampas 2013/2018)')
ax.legend(handles=[Patch(color='red', alpha=0.22, label='crisis (alta turbulencia)')], loc='upper left')
ax.margins(x=0.01); fig.tight_layout()
fig.savefig(RESULTS / 'd10_sp500_regimes.png', dpi=110, bbox_inches='tight'); plt.show()""")

# ------------------------------------------------------------------ #
md(r"""## 6. Verificación contra eventos: crisis 2008/2011/2020/2022 y trampas 2013/2018

La pregunta clave de CP2: ¿capta D10 el **colapso de correlaciones** de **2013 (taper)** que
D4 (HMM gaussiano) y D6 (GARCH equity) NO tapan?""")

co(r"""rows = []
for k in ev.CRISIS_WINDOWS:
    rows.append({'ventana': k, 'tipo': 'crisis', 'cobertura_OOS': res.crisis_coverage.get(k, float('nan'))})
for k in ev.FALSE_POSITIVE_WINDOWS:
    rows.append({'ventana': k, 'tipo': 'trampa', 'cobertura_OOS': res.false_alarm_in_fp.get(k, float('nan'))})
cmp = pd.DataFrame(rows).set_index('ventana')
display(cmp.style.format({'cobertura_OOS': '{:.1%}'}))

fig, ax = plt.subplots(figsize=(11, 4.2))
colors = ['#c0392b' if t == 'crisis' else '#e67e22' for t in cmp['tipo']]
ax.bar(cmp.index, cmp['cobertura_OOS'].values, color=colors)
ax.axhline(0.5, color='grey', ls='--', lw=0.8)
ax.set_ylabel('% dias en estado crisis (OOS)')
ax.set_title('D10 — cobertura por ventana (rojo=crisis, naranja=trampa)')
ax.set_xticklabels(cmp.index, rotation=30, ha='right'); ax.set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig(RESULTS / 'd10_coverage.png', dpi=110, bbox_inches='tight'); plt.show()

t2013 = res.false_alarm_in_fp['TaperTantrum_2013']
t2018 = res.false_alarm_in_fp['Selloff_Q4_2018']
print('Lectura de la hipotesis CP2 (colapso de correlaciones 2013):')
print(f'  TaperTantrum_2013: {t2013:.1%}  -> {"CAPTADO (colapso de correlaciones multivariante)" if t2013>0.3 else "apenas"}')
print(f'  Selloff_Q4_2018:   {t2018:.1%}')
print('  Recordatorio: D6 (GARCH equity) marco solo ~11% en 2013; D4 (HMM gaussiano) tampoco lo veia.')""")

# ------------------------------------------------------------------ #
md(r"""## 7. Timeline de régimen y duraciones (flickering)

Timeline causal OOS + histograma de duraciones. La histéresis + dwell debe dar episodios
largos (poco flickering).""")

co(r"""fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 4.5), gridspec_kw={'height_ratios': [1, 2.2]})
ax1.plot(p_c.index, p_c.values, color='#2980b9', lw=0.7); ax1.set_ylabel('P(crisis)')
for a, b in ev.CRISIS_WINDOWS.values(): ax1.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='red', alpha=0.12)
for a, b in ev.FALSE_POSITIVE_WINDOWS.values(): ax1.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='orange', alpha=0.15)
ax1.set_title('P(crisis) blanda (sigmoide de la turbulencia) — CAUSAL OOS'); ax1.margins(x=0.01)
ax2.imshow(states_c.values.reshape(1, -1), aspect='auto', cmap='RdYlGn_r',
           extent=[0, len(states_c), 0, 1]); ax2.set_yticks([])
tk = np.linspace(0, len(states_c)-1, 8).astype(int)
ax2.set_xticks(tk); ax2.set_xticklabels([states_c.index[i].year for i in tk])
ax2.set_title('Timeline de regimen (verde=calma, rojo=crisis)')
fig.tight_layout(); fig.savefig(RESULTS / 'd10_timeline.png', dpi=110, bbox_inches='tight'); plt.show()

def episode_durations(states):
    v = states.values; out = {0: [], 1: []}; run = 1
    for i in range(1, len(v)):
        if v[i] == v[i-1]: run += 1
        else: out[int(v[i-1])].append(run); run = 1
    out[int(v[-1])].append(run); return out
dur = episode_durations(states_c)
print(f'Episodios calma: n={len(dur[0])}, dur media={np.mean(dur[0]):.1f} d')
print(f'Episodios crisis: n={len(dur[1])}, dur media={np.mean(dur[1]) if dur[1] else float("nan"):.1f} d')
print(f'switching_rate={res.switching_rate:.4f}  dur media global={res.mean_regime_duration:.1f} d')""")

# ------------------------------------------------------------------ #
md(r"""## 8. Volcado de métricas a results/ (esquema 23 columnas)""")

co(r"""tbl = ev.results_table([res])
assert tbl.shape[1] == 23, f'esperaba 23 columnas, hay {tbl.shape[1]}'
tbl.to_csv(RESULTS / 'metrics_10_turbulence_mahalanobis.csv', index=False)
print('Guardado results/metrics_10_turbulence_mahalanobis.csv  (1 fila,', tbl.shape[1], 'columnas)')

master_path = RESULTS / 'metrics_master.csv'
if master_path.exists():
    master = pd.read_csv(master_path)
    master = master[master['detector'] != 'turbulence_mahalanobis']
    master = pd.concat([master, tbl], ignore_index=True)
else:
    master = tbl.copy()
master.to_csv(master_path, index=False)
print('master actualizado:', master.shape)
display(tbl.T)""")

# ------------------------------------------------------------------ #
md(r"""## 9. Conclusión D10 — ¿se cumple la hipótesis CP2?

**Hipótesis CP2:** *D10 capta el colapso de correlaciones multivariante que las reglas
univariantes no ven; en particular DEBERÍA captar 2013 (taper), el agujero de D4/D6.*

Veredicto (con los números de arriba):
- **Estados por umbral de turbulencia:** 2 estados vía τ_in/τ_out + dwell sobre la distancia
  de Mahalanobis; crisis = **alta turbulencia** (verificado en walk-forward, no invertido).
- **Multivariante y causal:** μ y Σ expanding causal (verificado: ocultar el futuro no cambia
  $d_t$). Histórico largo desde 1990 → 2008/2011/2020/2022 OOS y **2013/2018 OOS**.
- **2013 (taper tantrum):** ver el % de arriba — es el contraste clave frente a D6 (~11 %) y
  D4. Si D10 lo eleva claramente, confirma que ve el **colapso de correlaciones** (equity +
  tipos + divisa + curva moviéndose de forma atípica) que un detector univariante de equity
  no percibe.
- **Persistencia / flickering:** ver switching_rate y duración media (histéresis+dwell).

(El veredicto numérico definitivo queda en `docs/memory/detectors/10_turbulence_mahalanobis.md`.)""")

# ------------------------------------------------------------------ #
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
print('Ejecutando notebook (walk-forward ~1-2 min)...')
ep = ExecutePreprocessor(timeout=2400, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': str(ROOT / 'notebooks')}})
nbf.write(nb, NB_PATH)
n_err = sum(1 for c in nb.cells for o in c.get('outputs', []) if o.get('output_type') == 'error')
print(f'OK -> {NB_PATH.name}  | errores = {n_err}')
