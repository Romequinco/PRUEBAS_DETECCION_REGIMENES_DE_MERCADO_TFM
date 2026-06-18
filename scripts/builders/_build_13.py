# -*- coding: utf-8 -*-
"""
_build_13.py — Construye y EJECUTA notebooks/13_comparison.ipynb (FASE 4, síntesis).

Replica el patrón de _build_08.py: crea el notebook celda a celda con nbformat, lo
ejecuta con ExecutePreprocessor(timeout=3600, kernel python3) desde notebooks/, escribe
el .ipynb ya ejecutado, cuenta errores (debe dar 0) e imprime `celdas=… errores=N`.

La FASE 4 sintetiza los 12 detectores (D1–D12) ya evaluados walk-forward causal:
  · construye la tabla maestra FINAL `results/metrics_master_final.csv` (crisis estricta
    + estrés agregado + vio_2008_oos + clase);
  · recomputa el panel walk-forward de los 3 multi-estado (D3 k=3, D8 k=4, D12 k=3) para
    derivar la máscara de ESTRÉS AGREGADO (severidad alta = unión de los 2 estados más
    severos); los 8 binarios tienen estrés=crisis por definición (copia directa);
  · produce el ranking POR EJE y las figuras de cruce de ejes `results/fase4_*.png`.

NO modifica results/metrics_master.csv (crea el _final.csv aparte). No toca data/.

Uso:  python scripts/builders/_build_13.py
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[2]
NB_PATH = ROOT / "notebooks" / "13_comparison.ipynb"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ------------------------------------------------------------------ #
md(r"""# 13 — FASE 4: síntesis comparativa de los 12 detectores de régimen

Esta es la **síntesis honesta** del banco de pruebas. No buscamos "el mejor detector": cada
familia gana en un **eje** distinto y la comparación debe ser **justa** (controlando por la
ventana de evaluación) y **causal** (todo viene del walk-forward de la FASE 3, sin look-ahead).

## Qué hace este notebook
1. **Tabla maestra FINAL** `results/metrics_master_final.csv`: añade a la tabla de crisis
   estricta dos bloques nuevos — **estrés agregado** y los metadatos de **equidad de ventana**
   (`vio_2008_oos`) y **clase** ({baseline, avanzado, exploratorio-negativo}).
2. **Estrés agregado** para los 3 multi-estado (D3, D8, D12): recomputamos su panel
   walk-forward con su config EXACTA y medimos la severidad alta = unión de los 2 estados más
   severos. Los 8 binarios tienen estrés=crisis (copia directa, sin recomputar; D5 tarda ~33
   min).
3. **Ranking por eje** (no un único número) + **figuras de cruce de ejes** `results/fase4_*.png`.

## Principios no negociables
- **Causalidad**: todo sale de las etiquetas walk-forward de la FASE 3.
- **Equidad de ventana**: un detector de ventana corta (2012+/2015+) **NO vio la GFC 2008 OOS**
  → su `cov_GFC_2008` es NaN y **nunca** se penaliza por lo que no pudo ver. Se separa por grupo.
- **Honestidad con los negativos**: D11 (`msgarch_regime`, degeneración GFC=0%) y D12
  (`deep_ae_regime`, el AE empeora a la PCA) son **resultados negativos**; se marcan como tales
  (validan la parsimonia), no se esconden ni se maquillan.
- **2013 (taper) se mantiene como TRAMPA** (false-positive window). El hallazgo es que la
  *taxonomía* de crisis importa, no que 2013 sea ruido reclasificable.""")

# ------------------------------------------------------------------ #
co(r"""%matplotlib inline
import sys, warnings, time, logging
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
warnings.filterwarnings('ignore')
logging.getLogger('hmmlearn').setLevel(logging.CRITICAL)

ROOT = Path.cwd()
while not (ROOT / 'src').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / 'results'; RESULTS.mkdir(exist_ok=True)
from src import evaluation as ev

master = pd.read_csv(ROOT / 'results' / 'metrics_master.csv')
print('master cargado:', master.shape, '| detectores:', master.shape[0])

# Features y retornos de mercado (S&P 500) para recomputar los multi-estado.
feats = pd.read_parquet(ROOT / 'data' / 'processed' / 'features.parquet')
feats.index = pd.to_datetime(feats.index); feats = feats.sort_index()
raw = pd.read_parquet(ROOT / 'data' / 'raw' / 'raw_panel.parquet')
mkt_full = np.log(raw['SP500'] / raw['SP500'].shift(1))
print('features:', feats.shape, '| ventana', feats.index.min().date(), '->', feats.index.max().date())

CRISIS = ev.CRISIS_WINDOWS; FP = ev.FALSE_POSITIVE_WINDOWS; TROUGH = ev.DRAWDOWN_TROUGHS
print('crisis windows:', list(CRISIS)); print('trampas (FP):', list(FP))""")

# ------------------------------------------------------------------ #
md(r"""## 1. La métrica de ESTRÉS AGREGADO (decisión del proyecto)

La tabla de la FASE 3 mide **crisis estricta** = día con `state == n_states-1` (el estado más
severo). Para los detectores **multi-estado** esto solo cuenta la **cola extrema** y los compara
injustamente con los **binarios** (donde el único estado de riesgo ya engloba todo el estrés).
Para una comparación **justa** definimos además la máscara de **estrés agregado** (severidad alta):

| K del detector | máscara de ESTRÉS | significado |
|---|---|---|
| `n_states == 2` (binarios) | `state == 1` | **IDÉNTICO a crisis estricta** |
| `n_states >= 3` | `state >= n_states-2` | **unión de los DOS estados más severos** (corrección + crisis) |

Corte explícito por detector:
- **Binarios** (D1, D2, D4, D5, D6, D7, D10, D11): estrés = crisis → copia directa de `cov_*`,
  `fa_*` y `false_alarm_rate` (NO se recomputan; D5 tarda ~33 min).
- **D8** (k=4): estrés = {corrección(estado 2), crisis(estado 3)}.
- **D3** y **D12** (k=3): estrés = {corrección(estado 1), crisis(estado 2)}.

Bajo la máscara de estrés calculamos `cov_estres_*` (cobertura por ventana de crisis),
`fa_estres_*` (activación en las trampas 2013/2018) y `false_alarm_rate_estres` (tasa global).

**Lectura honesta (los dos lados):** ampliar a "corrección" **sube la cobertura** de las
crisis lentas (p. ej. Inflación 2022, que el HMM-t clasifica como corrección, no cola extrema)
**pero puede subir también la activación en las trampas** 2013/2018. Mostramos ambos lados.""")

# ------------------------------------------------------------------ #
co(r"""# Helpers: cobertura / falsa-alarma a partir de una MÁSCARA booleana (no de crisis_state)
def cov_from_mask(mask: pd.Series, windows) -> dict:
    out = {}
    for name, (a, b) in windows.items():
        seg = mask.loc[(mask.index >= pd.Timestamp(a)) & (mask.index <= pd.Timestamp(b))]
        out[name] = float(seg.mean()) if len(seg) else float('nan')
    return out

def far_from_mask(mask: pd.Series, crisis_windows) -> float:
    is_c = mask.values.astype(bool)
    n = int(is_c.sum())
    if n == 0:
        return float('nan')
    inwin = ev._in_any_window(mask.index, crisis_windows)
    return int((is_c & ~inwin).sum()) / n

def metrics_from_states(states: pd.Series, which: str, n_states: int):
    '''which="crisis" -> state==n-1 ; which="estres" -> state>=n-2 (=crisis si n==2).'''
    if which == 'crisis':
        mask = (states == n_states - 1)
    else:
        mask = (states >= n_states - 2)
    cov = cov_from_mask(mask, CRISIS)
    fa = cov_from_mask(mask, FP)
    far = far_from_mask(mask, CRISIS)
    return cov, fa, far

print('helpers listos')""")

# ------------------------------------------------------------------ #
md(r"""## 2. Recompute walk-forward de los 3 MULTI-ESTADO (D3, D8, D12) — config EXACTA

Recomputamos SOLO los multi-estado para derivar la máscara de estrés sobre `panel['state']`.
Config idéntica a sus builds de la FASE 3:
- **D3 `clustering_gmm_k3`**: 15 features, `train_size=252*8`, `step=21`, expanding. Rápido (~30 s).
- **D8 `hmm_tstudent_4s`**: 7 features puente, `n_init=3`, `t_n_iter=25`, `train_size=252*5`,
  `step=126`, expanding. EM-t caro (~5 min, paciencia).
- **D12 `deep_ae_regime`**: 15 features, AE(hidden=8, latent=2, epochs=40), `train_size=252*8`,
  `step=21`. Torch (~1–2 min).

**SANITY CHECK**: la crisis ESTRICTA recomputada (`state==n-1`) debe coincidir (±0.01) con los
`cov_*`/`fa_*` del master. Si no, se investiga antes de seguir.""")

co(r"""from detectors.clustering_gmm import ClusteringGMM
from detectors.hmm_tstudent import HMMTStudent
from detectors.hmm_gaussian_2s import BRIDGE_FEATURES
from detectors.deep_ae_regime import DeepAERegime

multi = {}   # nombre_master -> dict(n_states, cov_estricta, fa_estricta, far_estricta, cov_estres, ...)

def recompute(name, factory, X, n_states, train_size, step):
    t0 = time.time()
    mkt = mkt_full.reindex(X.index)
    panel = ev.walk_forward(factory, X, market_returns=mkt,
                            train_size=train_size, step=step, expanding=True)
    st = panel['state']
    cov_c, fa_c, far_c = metrics_from_states(st, 'crisis', n_states)
    cov_s, fa_s, far_s = metrics_from_states(st, 'estres', n_states)
    print(f'{name}: walk-forward {time.time()-t0:5.1f}s | OOS {st.index.min().date()}->{st.index.max().date()} | n={len(st)}')
    return dict(n_states=n_states, panel=panel,
                cov_c=cov_c, fa_c=fa_c, far_c=far_c,
                cov_s=cov_s, fa_s=fa_s, far_s=far_s)

# D3
Xall = feats.copy()
multi['clustering_gmm_k3'] = recompute(
    'clustering_gmm_k3', lambda: ClusteringGMM(n_states=3), Xall, 3, 252*8, 21)
# D8
Xb = feats[BRIDGE_FEATURES].copy()
multi['hmm_tstudent_4s'] = recompute(
    'hmm_tstudent_4s', lambda: HMMTStudent(n_states=4, n_init=3, t_n_iter=25), Xb, 4, 252*5, 126)
# D12
multi['deep_ae_regime'] = recompute(
    'deep_ae_regime',
    lambda: DeepAERegime(n_states=3, latent_dim=2, hidden=8, epochs=40,
                         weight_decay=1e-3, dropout=0.10, gmm_n_init=3),
    Xall, 3, 252*8, 21)
print('\nRecompute de los 3 multi-estado COMPLETO.')""")

# ------------------------------------------------------------------ #
md(r"""### 2b. SANITY CHECK — crisis estricta recomputada vs master (±0.01)""")

co(r"""def get_master(name, col):
    v = master.loc[master['detector'] == name, col]
    return float(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else float('nan')

def approx(a, b, tol=0.01):
    if (a != a) and (b != b):   # ambos NaN
        return True
    if (a != a) or (b != b):
        return False
    return abs(a - b) <= tol

print('=== SANITY: crisis estricta recomputada (state==n-1) vs metrics_master.csv ===')
all_ok = True
for name, d in multi.items():
    print(f'\n{name} (k={d["n_states"]})')
    for w in CRISIS:
        rc, mc = d['cov_c'][w], get_master(name, f'cov_{w}')
        ok = approx(rc, mc); all_ok &= ok
        print(f'  cov_{w:16s}: recomputado={rc:7.4f}  master={mc:7.4f}  {"OK" if ok else "<<< DESVIA"}')
    for w in FP:
        rf, mf = d['fa_c'][w], get_master(name, f'fa_{w}')
        ok = approx(rf, mf); all_ok &= ok
        print(f'  fa_{w:17s}: recomputado={rf:7.4f}  master={mf:7.4f}  {"OK" if ok else "<<< DESVIA"}')
print('\nSANITY GLOBAL:', 'TODO COINCIDE (±0.01) -> estrés derivado del mismo panel' if all_ok
      else 'HAY DESVIACIONES -> revisar (semilla/config); D12 puede driftar ~poco por torch')""")

# ------------------------------------------------------------------ #
md(r"""## 3. Tabla maestra FINAL `results/metrics_master_final.csv`

Una fila por detector (los 12). Bloques de columnas:
- **Identidad / equidad**: `detector, n_states, clase, ventana_eval, oos_start, oos_end, n_oos`,
  y `vio_2008_oos` (True si `oos_start` < 2008-09 → evaluó la GFC out-of-sample). La comparación
  agrupa por esta bandera: los de ventana corta (2012+/2015+) NO vieron 2008 OOS.
- **Crisis estricta** (del master): `cov_*`, `fa_*`, `false_alarm_rate`.
- **Estrés agregado** (nuevo): `cov_estres_*`, `fa_estres_*`, `false_alarm_rate_estres`.
- **Dinámica**: `switching_rate, mean_regime_duration, label_stability`.
- **Ajuste**: `log_likelihood, aic, bic` (solo modelos generativos).
- **Lead/lag**: `leadlag_*` (negativo = anticipa el suelo del drawdown).
- `clase` ∈ {baseline, avanzado, **exploratorio-negativo**}: D11 y D12 son negativos
  explícitos. `nota` documenta el corte de estrés del detector.

Filas ordenadas por `vio_2008_oos` (primero ventana larga que vio 2008 OOS) y, dentro, por
cobertura media disponible — para que la lectura **agrupe por ventana**.""")

co(r"""CLASE = {
    'rule_vix_threshold': 'baseline', 'rule_composite_riskoff': 'baseline',
    'clustering_gmm_k3': 'baseline', 'hmm_gaussian_2s': 'baseline',
    'markov_switching_var_2s': 'avanzado', 'garch_t_vol': 'avanzado',
    'changepoint_online': 'avanzado', 'hmm_tstudent_4s': 'avanzado',
    'turbulence_mahalanobis': 'avanzado', 'jump_model': 'avanzado',
    'msgarch_regime': 'exploratorio-negativo', 'deep_ae_regime': 'exploratorio-negativo',
}
# Coste computacional cualitativo, DOCUMENTADO desde las notas de build (INDEX.md):
#   alto  = refit caro (D5 MS-VAR ~33 min; D8 HMM-t ~5 min; D11 MS-GARCH desde cero)
#   medio = HMM/GMM/AE/jump con refit por fold moderado
#   bajo  = reglas, CUSUM, turbulencia (cálculo cerrado por día)
COSTE = {
    'rule_vix_threshold': 'bajo', 'rule_composite_riskoff': 'bajo',
    'clustering_gmm_k3': 'medio', 'hmm_gaussian_2s': 'medio',
    'markov_switching_var_2s': 'alto', 'garch_t_vol': 'medio',
    'changepoint_online': 'bajo', 'hmm_tstudent_4s': 'alto',
    'turbulence_mahalanobis': 'bajo', 'jump_model': 'medio',
    'msgarch_regime': 'alto', 'deep_ae_regime': 'medio',
}
def nota_estres(name, n_states):
    if n_states == 2:
        return 'binario: estres = crisis (state==1)'
    if name == 'hmm_tstudent_4s':
        return 'k=4: estres = {correccion(2), crisis(3)}'
    return 'k=3: estres = {correccion(1), crisis(2)}'

rows = []
for _, r in master.iterrows():
    name = r['detector']; n = int(r['n_states'])
    row = dict(r)   # arranca con TODAS las columnas del master (crisis estricta + dinamica + fit + leadlag)
    row['clase'] = CLASE.get(name, '?')
    row['coste'] = COSTE.get(name, '?')
    row['vio_2008_oos'] = pd.Timestamp(r['oos_start']) < pd.Timestamp('2008-09-01')
    row['nota'] = nota_estres(name, n)
    if n == 2:   # binario: estres = crisis (copia directa, NO recomputar)
        for w in CRISIS: row[f'cov_estres_{w}'] = r[f'cov_{w}']
        for w in FP:     row[f'fa_estres_{w}']  = r[f'fa_{w}']
        row['false_alarm_rate_estres'] = r['false_alarm_rate']
    else:        # multi-estado: estres recomputado
        d = multi[name]
        for w in CRISIS: row[f'cov_estres_{w}'] = d['cov_s'][w]
        for w in FP:     row[f'fa_estres_{w}']  = d['fa_s'][w]
        row['false_alarm_rate_estres'] = d['far_s']
    rows.append(row)

final = pd.DataFrame(rows)

# Orden de columnas (bien visible: ventana e identidad primero).
id_cols   = ['detector', 'n_states', 'clase', 'coste', 'vio_2008_oos',
             'ventana_eval', 'oos_start', 'oos_end', 'n_oos']
cov_cols  = [f'cov_{w}' for w in CRISIS]
fa_cols   = [f'fa_{w}' for w in FP]
cove_cols = [f'cov_estres_{w}' for w in CRISIS]
fae_cols  = [f'fa_estres_{w}' for w in FP]
dyn_cols  = ['switching_rate', 'mean_regime_duration', 'label_stability',
             'false_alarm_rate', 'false_alarm_rate_estres']
fit_cols  = ['log_likelihood', 'aic', 'bic']
ll_cols   = [f'leadlag_{w}' for w in TROUGH]
order = id_cols + cov_cols + fa_cols + cove_cols + fae_cols + dyn_cols + fit_cols + ll_cols + ['nota']
final = final[order]

# cobertura media disponible (crisis estricta) para ordenar dentro de cada grupo de ventana.
final['_cov_mean'] = final[cov_cols].mean(axis=1, skipna=True)
final = final.sort_values(['vio_2008_oos', '_cov_mean'], ascending=[False, False]).drop(columns='_cov_mean').reset_index(drop=True)

out = RESULTS / 'metrics_master_final.csv'
final.to_csv(out, index=False)
print('Guardado:', out, '| shape =', final.shape)
final[id_cols]""")

co(r"""# Vista compacta de la tabla final (estricta vs estres) para inspeccion
view = final[['detector', 'n_states', 'vio_2008_oos', 'clase',
              'cov_GFC_2008', 'cov_COVID_2020', 'cov_Inflation_2022',
              'cov_estres_COVID_2020', 'cov_estres_Inflation_2022',
              'fa_Selloff_Q4_2018', 'fa_estres_Selloff_Q4_2018',
              'switching_rate', 'mean_regime_duration', 'bic']].copy()
display(view.round(3))""")

# ------------------------------------------------------------------ #
md(r"""## 4. Ranking POR EJE (no un único número)

Seis ejes, cada uno con su lógica. **Clave de equidad**: la cobertura de sistémicas grandes
separa los que vieron 2008 OOS de los que no (NaN donde no estaba en su ventana, **sin
penalizar**). Construimos una tabla de rango por detector y eje (1 = mejor).

1. **Cobertura de sistémicas grandes** = media(GFC 2008, COVID 2020) donde el detector las vio
   OOS. Alto = mejor.
2. **Especificidad / no disparar en trampas** = 1 − media(fa_2013, fa_2018). En versión crisis
   estricta Y estrés. Alto = mejor.
3. **Flickering**: `switching_rate` (bajo = mejor) y `mean_regime_duration` (alto = mejor).
4. **Lead/lag sostenido**: media de `leadlag_*` (negativo = anticipa). Más negativo = mejor.
5. **Ajuste estadístico**: `BIC` SOLO en modelos generativos (HMM/GMM/MS/GARCH). Reglas, CUSUM y
   turbulencia **no tienen BIC comparable** → NaN. Menor = mejor. (Aviso: el BIC solo es
   estrictamente comparable entre modelos sobre las MISMAS features/ventana, p. ej. D4 vs D8.)
6. **Coste computacional**: escala cualitativa {bajo, medio, alto} documentada desde las notas
   de build del INDEX (no inventada con falsa precisión). bajo = mejor.""")

co(r"""ax = final.set_index('detector').copy()
# Ejes numericos (mayor=mejor salvo donde se indique)
ax['eje1_cob_sistemica'] = ax[['cov_GFC_2008', 'cov_COVID_2020']].mean(axis=1, skipna=True)
ax['eje2_especif_estricta'] = 1 - ax[['fa_TaperTantrum_2013', 'fa_Selloff_Q4_2018']].mean(axis=1, skipna=True)
ax['eje2_especif_estres']   = 1 - ax[['fa_estres_TaperTantrum_2013', 'fa_estres_Selloff_Q4_2018']].mean(axis=1, skipna=True)
ax['eje3_persistencia'] = ax['mean_regime_duration']
ax['eje3_switching']    = ax['switching_rate']               # bajo mejor
ax['eje4_leadlag']      = ax[[f'leadlag_{w}' for w in TROUGH]].mean(axis=1, skipna=True)  # negativo mejor
ax['eje5_bic']          = ax['bic']                          # bajo mejor
_coste_num = {'bajo': 1, 'medio': 2, 'alto': 3}
ax['eje6_coste_num']    = ax['coste'].map(_coste_num)        # bajo mejor

# Ranks (1 = mejor). ascending del valor: True si menor es mejor.
rank_spec = {
    'Cob.sistemica':   ('eje1_cob_sistemica', False),
    'Especif.estricta':('eje2_especif_estricta', False),
    'Especif.estres':  ('eje2_especif_estres', False),
    'Persistencia':    ('eje3_persistencia', False),
    'Anti-flicker':    ('eje3_switching', True),
    'Lead/lag':        ('eje4_leadlag', True),
    'BIC':             ('eje5_bic', True),
    'Coste':           ('eje6_coste_num', True),
}
ranks = pd.DataFrame(index=ax.index)
for label, (col, asc) in rank_spec.items():
    ranks[label] = ax[col].rank(ascending=asc, method='min')
print('=== VALORES POR EJE ===')
display(ax[['eje1_cob_sistemica','eje2_especif_estricta','eje2_especif_estres','eje3_persistencia',
            'eje3_switching','eje4_leadlag','eje5_bic','coste']].round(3))
print('\n=== RANKS POR EJE (1 = mejor; NaN = no aplica / fuera de ventana) ===')
display(ranks.round(0))""")

co(r"""# Ganador por eje (mejor rank) — con la salvedad de equidad de ventana en cobertura.
print('=== GANADOR POR EJE ===')
for label in rank_spec:
    col = rank_spec[label][0]; asc = rank_spec[label][1]
    s = ax[col].dropna()
    if not len(s):
        continue
    best = s.idxmin() if asc else s.idxmax()
    print(f'  {label:18s}: {best:24s} ({s.loc[best]:.3f})')

# Cobertura: separar por grupo de ventana (justo)
print('\n=== Cobertura sistemica POR GRUPO DE VENTANA (equidad) ===')
for grp, sub in ax.groupby('vio_2008_oos'):
    s = sub['eje1_cob_sistemica'].dropna().sort_values(ascending=False)
    tag = 'VIO 2008 OOS (ventana larga)' if grp else 'NO vio 2008 (ventana corta 2012+/2015+)'
    print(f'\n  [{tag}]')
    for d, v in s.items():
        print(f'    {d:24s}: {v:.3f}')""")

# ------------------------------------------------------------------ #
md(r"""## 5. Figuras de cruce de ejes (`results/fase4_*.png`)

No tablas planas: **planos** que cruzan ejes para ver los trade-offs. Color por grupo de
ventana (vio 2008 o no); D11 y D12 marcados como **negativos**.""")

co(r"""# Estilo comun
def is_neg(name): return CLASE.get(name) == 'exploratorio-negativo'
def color_for(row):
    if is_neg(row.name): return '#7f7f7f'        # negativos en gris
    return '#1f77b4' if row['vio_2008_oos'] else '#ff7f0e'
SHORT = {  # nombres cortos para etiquetar
    'rule_vix_threshold':'D1 vix','rule_composite_riskoff':'D2 comp','clustering_gmm_k3':'D3 gmm',
    'hmm_gaussian_2s':'D4 hmm','markov_switching_var_2s':'D5 msvar','garch_t_vol':'D6 garch',
    'changepoint_online':'D7 cusum','hmm_tstudent_4s':'D8 hmm-t','turbulence_mahalanobis':'D10 turb',
    'jump_model':'D9 jump','msgarch_regime':'D11 msg(-)','deep_ae_regime':'D12 ae(-)'}
af = ax.reset_index()
print('paleta lista')""")

co(r"""# FIG 1: plano SENSIBILIDAD (cobertura sistemica) vs ESPECIFICIDAD-trampas
fig, ax1 = plt.subplots(figsize=(10.5, 7))
for _, row in ax.iterrows():
    x = 1 - (1 - row['eje2_especif_estricta'])   # = especificidad estricta
    y = row['eje1_cob_sistemica']
    if y != y:   # sin cobertura sistemica evaluable -> lo situamos en el borde inferior anotado
        continue
    c = color_for(row); mk = 'X' if is_neg(row.name) else 'o'
    ax1.scatter(row['eje2_especif_estricta'], y, s=170, c=c, marker=mk,
                edgecolor='black', linewidth=0.8, zorder=3)
    ax1.annotate(SHORT[row.name], (row['eje2_especif_estricta'], y),
                 textcoords='offset points', xytext=(7, 5), fontsize=9)
ax1.set_xlabel('Especificidad = 1 - media(fa_2013, fa_2018)  (alto = no dispara en trampas)')
ax1.set_ylabel('Cobertura sistemica = media(GFC 2008, COVID 2020)  (alto = sensible)')
ax1.set_title('FASE 4 - Plano sensibilidad <-> especificidad (crisis estricta)\n'
              'azul = vio 2008 OOS | naranja = ventana corta | gris X = negativo (D11, D12)')
ax1.grid(alpha=0.3)
ax1.legend(handles=[Patch(color='#1f77b4', label='vio 2008 OOS'),
                    Patch(color='#ff7f0e', label='ventana corta (no vio 2008)'),
                    Patch(color='#7f7f7f', label='negativo (D11/D12)')], loc='lower left')
fig.tight_layout(); fig.savefig(RESULTS / 'fase4_sensibilidad_especificidad.png', dpi=120, bbox_inches='tight')
plt.show()""")

co(r"""# FIG 2: plano PERSISTENCIA (anti-flickering) vs SENSIBILIDAD (cobertura)
fig, ax2 = plt.subplots(figsize=(10.5, 7))
for _, row in ax.iterrows():
    y = row['eje1_cob_sistemica']
    yv = y if y == y else np.nanmean([row['cov_COVID_2020']])  # fallback a COVID si no vio sistemicas grandes
    c = color_for(row); mk = 'X' if is_neg(row.name) else 'o'
    ax2.scatter(row['mean_regime_duration'], yv, s=170, c=c, marker=mk,
                edgecolor='black', linewidth=0.8, zorder=3)
    ax2.annotate(SHORT[row.name], (row['mean_regime_duration'], yv),
                 textcoords='offset points', xytext=(7, 5), fontsize=9)
ax2.set_xscale('log')
ax2.set_xlabel('Persistencia = duracion media de regimen (dias, log)  (alto = no flickea)')
ax2.set_ylabel('Sensibilidad (cobertura sistemica; COVID si no vio 2008)')
ax2.set_title('FASE 4 - Plano flickering <-> sensibilidad\n'
              'D3/D12 (gmm/ae) flickean; D7 cusum y reglas son persistentes')
ax2.grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig(RESULTS / 'fase4_persistencia_sensibilidad.png', dpi=120, bbox_inches='tight')
plt.show()""")

co(r"""# FIG 3: BIC (solo modelos generativos). Menor = mejor.
bic_s = ax['bic'].dropna().sort_values()
fig, ax3 = plt.subplots(figsize=(10, 5))
cols = ['#7f7f7f' if is_neg(n) else '#4c72b0' for n in bic_s.index]
ax3.bar([SHORT[n] for n in bic_s.index], bic_s.values, color=cols, edgecolor='black', linewidth=0.6)
ax3.set_ylabel('BIC (menor = mejor ajuste)')
ax3.set_title('FASE 4 - BIC de modelos generativos (gris = negativo)\n'
              'Aviso: BIC solo estrictamente comparable sobre las MISMAS features (D4 vs D8)')
for i, (n, v) in enumerate(bic_s.items()):
    ax3.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=8, rotation=0)
plt.xticks(rotation=30, ha='right')
fig.tight_layout(); fig.savefig(RESULTS / 'fase4_bic.png', dpi=120, bbox_inches='tight')
plt.show()""")

co(r"""# FIG 4: lead/lag por evento (negativo = anticipa el suelo del drawdown)
events = list(TROUGH)
det_order = [n for n in ax.index]
M = np.array([[ax.loc[n, f'leadlag_{w}'] for w in events] for n in det_order], dtype=float)
fig, ax4 = plt.subplots(figsize=(11, 6.5))
im = ax4.imshow(M, aspect='auto', cmap='RdBu', vmin=-260, vmax=260)
ax4.set_xticks(range(len(events))); ax4.set_xticklabels(events, rotation=20, ha='right')
ax4.set_yticks(range(len(det_order))); ax4.set_yticklabels([SHORT[n] for n in det_order])
for i in range(len(det_order)):
    for j in range(len(events)):
        v = M[i, j]
        if v == v:
            ax4.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=8,
                     color='white' if abs(v) > 150 else 'black')
ax4.set_title('FASE 4 - Lead/lag por evento (dias; AZUL negativo = ANTICIPA el suelo, ROJO = retrasa)')
fig.colorbar(im, ax=ax4, fraction=0.025, pad=0.02, label='dias (- anticipa)')
fig.tight_layout(); fig.savefig(RESULTS / 'fase4_leadlag.png', dpi=120, bbox_inches='tight')
plt.show()""")

co(r"""# FIG 5: heatmap de RANKS por eje (detectores x ejes; 1 = mejor)
R = ranks.copy()
fig, ax5 = plt.subplots(figsize=(11, 7.5))
im = ax5.imshow(R.values, aspect='auto', cmap='RdYlGn_r', vmin=1, vmax=12)
ax5.set_xticks(range(R.shape[1])); ax5.set_xticklabels(R.columns, rotation=25, ha='right')
ax5.set_yticks(range(R.shape[0])); ax5.set_yticklabels([SHORT[n] for n in R.index])
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        v = R.values[i, j]
        if v == v:
            ax5.text(j, i, f'{int(v)}', ha='center', va='center', fontsize=8,
                     color='white' if (v <= 3 or v >= 10) else 'black')
ax5.set_title('FASE 4 - Heatmap de RANKS por eje (verde=1=mejor, rojo=peor; blanco=no aplica)')
fig.colorbar(im, ax=ax5, fraction=0.025, pad=0.02, label='rank (1 = mejor)')
fig.tight_layout(); fig.savefig(RESULTS / 'fase4_rank_heatmap.png', dpi=120, bbox_inches='tight')
plt.show()""")

co(r"""# FIG 6: crisis ESTRICTA vs ESTRES para los 3 multi-estado (corazon del ajuste de equidad)
multi_names = ['clustering_gmm_k3', 'hmm_tstudent_4s', 'deep_ae_regime']
wins = ['cov_COVID_2020', 'cov_Inflation_2022']   # ventanas OOS comunes a los 3
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for axx, name in zip(axes, multi_names):
    r = final[final['detector'] == name].iloc[0]
    estr = [r[w] for w in wins]
    estres = [r[w.replace('cov_', 'cov_estres_')] for w in wins]
    xx = np.arange(len(wins)); w = 0.38
    b1 = axx.bar(xx - w/2, estr, w, label='crisis estricta (cola extrema)', color='#c44e52')
    b2 = axx.bar(xx + w/2, estres, w, label='estres agregado (correccion+crisis)', color='#dd8452')
    axx.set_xticks(xx); axx.set_xticklabels(['COVID 2020', 'Inflation 2022'])
    neg = ' (NEGATIVO)' if is_neg(name) else ''
    axx.set_title(SHORT[name] + neg); axx.set_ylim(0, 1.05); axx.grid(alpha=0.3, axis='y')
    for b in list(b1) + list(b2):
        axx.text(b.get_x() + b.get_width()/2, b.get_height(), f'{b.get_height():.2f}',
                 ha='center', va='bottom', fontsize=8)
axes[0].set_ylabel('cobertura'); axes[0].legend(loc='upper left', fontsize=8)
fig.suptitle('FASE 4 - Multi-estado: crisis ESTRICTA vs ESTRES AGREGADO (ajuste de equidad con los binarios)', y=1.02)
fig.tight_layout(); fig.savefig(RESULTS / 'fase4_estres_vs_estricta.png', dpi=120, bbox_inches='tight')
plt.show()

# fa en trampas: el otro lado (ampliar a correccion puede subir la activacion en 2013/2018)
print('=== Multi-estado: activacion en TRAMPAS estricta vs estres (el otro lado, honesto) ===')
for name in multi_names:
    r = final[final['detector'] == name].iloc[0]
    print(f'\n{SHORT[name]}:')
    for w in FP:
        print(f'  {w:16s}: estricta={r[f"fa_{w}"]:.3f}  ->  estres={r[f"fa_estres_{w}"]:.3f}')""")

# ------------------------------------------------------------------ #
md(r"""## 6. RESUMEN ESTRUCTURADO (para redactar las conclusiones)

Ranking por eje, números estrés vs crisis estricta de los multi-estado, y qué familia gana en
qué eje. Es el insumo de las conclusiones de la FASE 4.""")

co(r"""print('='*78)
print('RESUMEN ESTRUCTURADO - FASE 4')
print('='*78)

def top(col, asc, k=3):
    s = ax[col].dropna()
    s = s.sort_values(ascending=asc)
    return [(SHORT[n], round(float(v), 3)) for n, v in list(s.items())[:k]]

print('\n--- RANKING POR EJE (top-3; 1 = mejor) ---')
print('1. Cobertura sistemica (GFC+COVID, solo quien la vio OOS):')
for grp, sub in ax.groupby('vio_2008_oos'):
    tag = 'VIO 2008 OOS' if grp else 'ventana corta (no vio 2008)'
    s = sub['eje1_cob_sistemica'].dropna().sort_values(ascending=False)
    print(f'     [{tag}] ' + ', '.join(f'{SHORT[n]}={v:.2f}' for n, v in s.items()))
print('2. Especificidad estricta (1-fa trampas):', top('eje2_especif_estricta', False))
print('   Especificidad estres                 :', top('eje2_especif_estres', False))
print('3. Persistencia (dur media, dias)        :', top('eje3_persistencia', False))
print('   Anti-flicker (switching bajo)         :', top('eje3_switching', True))
print('4. Lead/lag (mas negativo = anticipa)    :', top('eje4_leadlag', True))
print('5. BIC (menor = mejor; solo generativos) :', top('eje5_bic', True))
print('6. Coste (bajo = mejor)                  :', top('eje6_coste_num', True))

print('\n--- ESTRES AGREGADO vs CRISIS ESTRICTA (multi-estado D3, D8, D12) ---')
for name in ['clustering_gmm_k3', 'hmm_tstudent_4s', 'deep_ae_regime']:
    r = final[final['detector'] == name].iloc[0]
    print(f'\n {SHORT[name]} (k={int(r["n_states"])})  {r["nota"]}')
    for w in ['COVID_2020', 'Inflation_2022']:
        print(f'   cov_{w:14s}: estricta={r[f"cov_{w}"]:.3f}  ->  estres={r[f"cov_estres_{w}"]:.3f}'
              f'   (delta={r[f"cov_estres_{w}"]-r[f"cov_{w}"]:+.3f})')
    for w in ['TaperTantrum_2013', 'Selloff_Q4_2018']:
        ce, ee = r[f'fa_{w}'], r[f'fa_estres_{w}']
        print(f'   fa_{w:15s}: estricta={ce:.3f}  ->  estres={ee:.3f}   (delta={ee-ce:+.3f})')
    print(f'   false_alarm_rate: estricta={r["false_alarm_rate"]:.3f} -> estres={r["false_alarm_rate_estres"]:.3f}')

print('\n--- QUE FAMILIA GANA EN QUE EJE ---')
def winner(col, asc):
    s = ax[col].dropna(); return SHORT[s.idxmin() if asc else s.idxmax()]
print(f'  Cobertura sistemica (ventana larga) : {winner("eje1_cob_sistemica", False)}  '
      '(reglas/vol/MS-VAR: familias F1/F4/F5)')
print(f'  Especificidad (no trampas)          : {winner("eje2_especif_estricta", False)}  (CUSUM F6 / reglas F1)')
print(f'  Persistencia / anti-flicker         : {winner("eje3_persistencia", False)}  (change-point F6)')
print(f'  Lead/lag (anticipacion)             : {winner("eje4_leadlag", True)}  (change-point F6)')
print(f'  Ajuste BIC (generativos)            : {winner("eje5_bic", True)}  (GARCH-t F5 / HMM-t F3)')
print(f'  Coste (mas barato)                  : {winner("eje6_coste_num", True)}  (reglas F1 / CUSUM F6)')

print('\n--- RESULTADOS NEGATIVOS (validan parsimonia) ---')
for name in ['msgarch_regime', 'deep_ae_regime']:
    r = final[final['detector'] == name].iloc[0]
    print(f'  {SHORT[name]:10s}: clase={r["clase"]}, cov_GFC={r["cov_GFC_2008"]}, '
          f'switching={r["switching_rate"]:.3f}, far={r["false_alarm_rate"]:.3f}')
print('\nEntregables: results/metrics_master_final.csv + results/fase4_*.png')
print('='*78)""")

# ------------------------------------------------------------------ #
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}

print("Ejecutando notebook 13_comparison (recompute D8 ~5 min)...")
ep = ExecutePreprocessor(timeout=3600, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": str(ROOT / "notebooks")}})
nbf.write(nb, NB_PATH)

n_err = sum(1 for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error")
print(f"[build_13] escrito {NB_PATH}")
print(f"[build_13] celdas={len(nb.cells)}  errores={n_err}")
if n_err:
    for c in nb.cells:
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                print("ERROR:", o.get("ename"), o.get("evalue"))
                for ln in o.get("traceback", []):
                    print(ln)
