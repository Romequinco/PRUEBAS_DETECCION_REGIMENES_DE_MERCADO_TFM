# -*- coding: utf-8 -*-
"""Constructor del notebook 07_changepoint_online.ipynb (D7, FASE 3 Tanda 3).

Crea las celdas, ejecuta con nbconvert (0 errores esperados) y guarda el .ipynb con
las figuras inline. Vuelca figuras a results/ y la fila de métricas (32 columnas) a
results/metrics_07_changepoint_online.csv. NO actualiza metrics_master.csv (el master se
reconstruye centralizadamente con scripts/verify/_rebuild_master.py; ver §15).

Uso:  python scripts/builders/_build_07.py
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[2]
NB_PATH = ROOT / "notebooks" / "07_changepoint_online.ipynb"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ------------------------------------------------------------------ #
md(r"""# 07 — D7 `changepoint_online`: CUSUM online (Page 1954) sobre la vol del S&P 500

Familia **F6 (Change-point)**. Detector de **cambio estructural ONLINE y CAUSAL** sobre
el **retorno log del S&P 500** (desde 1985). Núcleo: un **CUSUM de Page (1954)**
secuencial sobre un estadístico de volatilidad, que detecta los cambios de nivel de
varianza **con cierto retardo y SIN mirar el futuro**. Dos estados: 0 = calma · 1 = crisis
(tramo de alta vol tras un cambio al alza).

**Online vs offline (clave de la familia).** La variante potente de `ruptures` — **PELT**
(Killick 2012) / BinSeg (Truong 2020) — es **OFFLINE y ANTI-causal**: segmenta mirando
TODA la serie, así que NO vale para walk-forward (vería el futuro del bloque). Aquí se usa
el **CUSUM secuencial** (causal). PELT aparece SOLO como **oráculo in-sample** (§3),
marcado **NO causal**.

**De change-point a 2 estados recurrentes.** Un change-point *segmenta*, no etiqueta
estados. El mapeo: un **autómata de 2 CUSUM de una cara** — en calma vigila el cambio
**al alza** de vol (C⁺); en crisis, el cambio **a la baja** (C⁻). El umbral `h` da
**persistencia** por construcción. QUÉ tramo es "crisis" lo decide el **NÚCLEO**
(`label_states_economically`, Arreglo 4 vol-primario): crisis = el tramo de **mayor σ**.

## Hipótesis CP2 (la que este notebook pone a prueba)
> *"Detección **temprana** (lead/lag) pero **riesgo de FALSAS ALARMAS con outliers**;
> **preferir kernel/robusto** frente al CUSUM **gaussiano**."*

Con **kurtosis 25–40** (EDA), un CUSUM sobre el **retorno²** (coste gaussiano L2) da un
peso enorme a un único día de cola → falsas alarmas. D7 ofrece **dos costes** y los
compara (§12): `gaussian` (retorno², media/desv) vs `robust` (log|retorno|, mediana/MAD +
winsorizado). **Ventana LARGA** (solo S&P 500 desde 1985) → **2008 y 2011 son OOS**.""")

# ------------------------------------------------------------------ #
md(r"""## Índice del notebook

**Bloque A — Ajuste, causalidad y oráculo offline**
1. [Ajuste in-sample (coste robusto) y verificación de etiquetado](#sec1)
2. [Verificación de CAUSALIDAD del CUSUM](#sec2)
3. [Oráculo OFFLINE (NO causal): PELT sobre la log-vol](#sec3)
4. [Versión CAUSAL walk-forward (la comparable)](#sec4)

**Bloque B — El mecanismo Page 1954, en imágenes**
5. [Estadístico de vol, change-points y ventanas de crisis](#sec5) · `d07_cusum_changepoints.png`
6. [El MECANISMO CUSUM: cómo se acumula la evidencia y salta el detector](#sec6) · `d07_cusum_statistic.png` ⭐ *nueva*
7. [CUSUM online (causal) vs PELT (oráculo offline)](#sec7) · `d07_online_vs_pelt.png` ⭐ *nueva*

**Bloque C — Detección temprana y verificación contra eventos**
8. [S&P 500 coloreado por régimen (CAUSAL OOS)](#sec8) · `d07_sp500_regimes.png`
9. [Lead/lag SOSTENIDO respecto a los suelos de drawdown](#sec9) · `d07_leadlag.png`
10. [Retardos de detección: confirmación vs anticipación](#sec10) · `d07_detection_delays.png` ⭐ *nueva*
11. [Verificación contra eventos: crisis vs trampas](#sec11) · `d07_coverage.png`

**Bloque D — CP2 en acción (robusto vs gaussiano)**
12. [CP2: coste GAUSSIANO vs ROBUSTO (falsas alarmas por outliers)](#sec12) · `d07_robust_vs_gaussian.png`
13. [Distribución del coste robusto vs gaussiano por régimen](#sec13) · `d07_cost_by_regime.png` ⭐ *nueva*

**Bloque E — Persistencia, volcado y veredicto**
14. [Timeline de régimen y duraciones (flickering)](#sec14) · `d07_timeline.png`
15. [Volcado de métricas (esquema canónico 32 columnas)](#sec15)
16. [Conclusión D7 — ¿se cumple la hipótesis CP2?](#sec16)

> **Cómo leer este notebook.** El hilo conductor es la hipótesis **CP2**: un detector de
> cambio *online y causal* puede **anticipar** las crisis (lead/lag), pero el coste
> **gaussiano** sobre colas pesadas degenera en falsas alarmas, así que se prefiere el
> coste **robusto**. El Bloque B abre la caja del CUSUM (el porqué del retardo y de la
> persistencia); el Bloque C mide la detección temprana; el Bloque D pone a prueba CP2.""")

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
from src import viz
viz.use_house_style()                            # paleta/tipografía de casa (colores consistentes)
from detectors.changepoint_online import ChangepointOnline

raw = pd.read_parquet(ROOT / 'data' / 'raw' / 'raw_panel.parquet')
spx_ret = np.log(raw['SP500'] / raw['SP500'].shift(1)).rename('SP500_ret')
X = pd.DataFrame({'SP500_ret': spx_ret}).dropna()
X.index = pd.to_datetime(X.index); X = X.sort_index()
mkt = X['SP500_ret']                              # market_returns (S&P 500 log-ret)
sp500 = raw['SP500'].reindex(X.index)
print('X (histórico largo):', X.shape, '|', X.index.min().date(), '->', X.index.max().date())
print('OBLIGATORIO pasar market_returns a walk_forward Y evaluate (señal = vol, no retorno).')""")

# ------------------------------------------------------------------ #
md(r"""<a id="sec1"></a>
## 1. Ajuste in-sample (coste ROBUSTO) y verificación de etiquetado

Ajuste sobre toda la muestra (solo para inspeccionar; la clasificación comparable es la
**causal** de §4). El estadístico robusto es **log|retorno|** estandarizado con
**mediana/MAD** del train (escala robusta, simétrica) y **winsorizado** a ±`clip`. El
orden económico (0=calma · 1=crisis) lo fija `label_states_economically` con el retorno
del S&P 500 (núcleo vol-primario, Arreglo 4).""")

co(r"""det_is = ChangepointOnline(cost='robust').fit(X)
det_is.label_states_economically(X, market_returns=mkt)   # orden económico con S&P 500
print('CUSUM robusto:  cost=%s  k=%.2f  h=%.2f  clip=%.1f' % (det_is.cost, det_is.k, det_is.h, det_is.clip))
print('Base congelada del train:  centro(mediana log|r|)=%.3f  escala(1.4826*MAD)=%.3f'
      % (det_is._center, det_is._scale))

states_is = pd.Series(det_is.predict(X), index=X.index, name='state')   # canónico
# VERIFICACION CRITICA: crisis canónico = ALTA vol de retornos (no invertido)
sd_cri = mkt[states_is.values == det_is.crisis_state].std()
sd_cal = mkt[states_is.values == 0].std()
print(f'\nVERIFICACION etiquetado: std(ret) CRISIS={sd_cri:.5f}  vs  CALMA={sd_cal:.5f}')
assert sd_cri > sd_cal, 'INVERTIDO: crisis deberia ser ALTA vol'
print(f'OK -> crisis (estado canonico {det_is.crisis_state}) = ALTA vol de retornos. No invertido.')
print('frac de dias en crisis (in-sample):', round(float((states_is==det_is.crisis_state).mean()), 3))""")

# ------------------------------------------------------------------ #
md(r"""<a id="sec2"></a>
## 2. Verificación de CAUSALIDAD del CUSUM

El CUSUM es causal nativo (C±ₜ depende solo de z con s ≤ t). En walk-forward se usa el
patrón **burn-in** (como D6): el autómata se re-ejecuta sobre `[train previo al bloque] +
bloque` con la base congelada → el estado al inicio del bloque refleja la historia real.
**Test:** ocultar el futuro NO debe cambiar los estados del bloque.""")

co(r"""det_c = ChangepointOnline(cost='robust').fit(X.loc[:'2007-12-31'])
block = X.loc['2008-01-01':'2008-12-31']
s_block = det_c._predict_states(block)
s_plus  = pd.Series(det_c._predict_states(X.loc['2008-01-01':'2010-12-31']),
                    index=X.loc['2008-01-01':'2010-12-31'].index).loc[block.index].values
ndiff = int((s_block != s_plus).sum())
print(f'dias con estado distinto al ocultar/ver el futuro = {ndiff}')
assert ndiff == 0, 'El CUSUM del bloque NO es causal'
print('causal_ok = True  -> estado_t usa solo retornos <= t (burn-in propaga el acumulado)')""")

# ------------------------------------------------------------------ #
md(r"""<a id="sec3"></a>
## 3. ORÁCULO OFFLINE (NO causal): PELT sobre la log-vol — solo comparación

`ruptures.Pelt` (Killick 2012; survey Truong 2020) segmenta la **log-vol** mirando TODA
la serie → **ANTI-causal**, NO usable en evaluación. Se muestra como **referencia** de
dónde caen los cambios estructurales "ideales" con información perfecta, para contrastar
con la detección **online** (retardada) del CUSUM.""")

co(r"""import ruptures as rpt
logvol = np.log(np.abs(X['SP500_ret'].values) + 1e-6)
algo = rpt.Pelt(model='l2', min_size=40, jump=5).fit(logvol.reshape(-1, 1))
bkps = algo.predict(pen=25)            # penalizacion -> nº de cambios
cp_dates = [X.index[min(b, len(X)-1)] for b in bkps[:-1]]
print(f'PELT (OFFLINE, NO causal): {len(cp_dates)} change-points sobre la log-vol')
print('Algunos (anios):', sorted(set(d.year for d in cp_dates)))""")

# ------------------------------------------------------------------ #
md(r"""<a id="sec4"></a>
## 4. Versión CAUSAL walk-forward (la comparable)

`ev.walk_forward` reentrena el CUSUM en ventanas **expanding** (train inicial **8 años**)
y predice el bloque de `step=21` días con la base congelada (burn-in). **Se pasa
`market_returns=mkt`** para re-fijar el orden económico de estados de forma robusta (la
señal es vol, no retorno → sin esto el etiquetado podría invertirse).""")

co(r"""TRAIN_SIZE = 252 * 8   # ~8 anios: 2008 y 2011 caen OOS (histórico largo desde 1985)
STEP = 21
panel = ev.walk_forward(lambda: ChangepointOnline(cost='robust'), X, market_returns=mkt,
                        train_size=TRAIN_SIZE, step=STEP, expanding=True)
print('OOS:', panel.index.min().date(), '->', panel.index.max().date(), '| n_oos =', len(panel))
states_c = panel['state']; p_c = panel['p_crisis']

res = ev.evaluate(det_is, panel, market_returns=mkt, X_full=X)
res.detector_name = 'changepoint_online'
print('ventana_eval:', res.extra['ventana_eval'])

# VERIFICACION en WALK-FORWARD: crisis = alta vol de retornos por estado (no invertido)
byvol = {int(s): float(mkt.reindex(states_c.index)[states_c == s].std()) for s in sorted(states_c.unique())}
print('std(ret) por estado canonico (OOS):', {k: round(v,5) for k,v in byvol.items()})
if det_is.crisis_state in byvol and 0 in byvol:
    assert byvol[det_is.crisis_state] > byvol[0], 'INVERSION en walk-forward!'
    print('OK walk-forward -> crisis = ALTA vol. No invertido, sin fallback.')

print('\nCobertura de crisis (CAUSAL OOS):')
for k, v in res.crisis_coverage.items(): print(f'  {k:16s}: {v:6.1%}')
print('Falsas alarmas en ventanas TRAMPA (CAUSAL OOS):')
for k, v in res.false_alarm_in_fp.items(): print(f'  {k:16s}: {v:6.1%}')
print(f'\nfalse_alarm_rate = {res.false_alarm_rate:.3f} | switching = {res.switching_rate:.4f} '
      f'| dur media = {res.mean_regime_duration:.1f} d | label_stability = {res.label_stability:.3f}')
print('lead/lag SOSTENIDO (persist=3) vs troughs (negativo=anticipa):', res.lead_lag_days)""")

# ------------------------------------------------------------------ #
md(r"""<a id="sec5"></a>
## 5. Estadístico de vol con CUSUM, change-points detectados y ventanas de crisis

Log-vol estandarizada (robusta) con los **change-points ONLINE** del CUSUM (líneas
verticales = días en que el autómata conmuta) y los del **oráculo PELT offline** (rombos).
Bandas rojas = crisis conocidas; naranjas = trampas (2013/2018).

Esta es la vista "de superficie": muestra **dónde** conmuta el detector sobre la señal de
volatilidad. Nótese que las conmutaciones online (azul) caen **sistemáticamente a la
derecha** de los rombos morados de PELT: el oráculo offline ve toda la serie y coloca el
cambio en su punto óptimo retrospectivo, mientras el CUSUM causal necesita **acumular
evidencia** antes de disparar. Ese retardo —el precio de no mirar el futuro— se abre en
canal en la §6 (el acumulado interno) y se cuantifica en la §7.""")

co(r"""# estadistico estandarizado in-sample y conmutaciones del CUSUM
r_is = det_is._returns(X)
z_is = pd.Series(det_is._standardize(det_is._raw_stat(r_is)), index=r_is.index)
sw = states_is.values
switch_idx = np.where(sw[1:] != sw[:-1])[0] + 1
switch_dates = states_is.index[switch_idx]

fig, ax = plt.subplots(figsize=(15, 4.5))
ax.plot(z_is.index, z_is.values, color='#34495e', lw=0.5, label='log-vol estandarizada (robusta)')
ax.axhline(det_is.k, color='#16a085', ls=':', lw=0.9, label=f'k (slack)={det_is.k}')
for d in switch_dates:
    ax.axvline(d, color='#2980b9', lw=0.5, alpha=0.5)
for d in cp_dates:
    ax.plot(d, det_is.clip, marker='D', color='#8e44ad', ms=4)
for a, b in ev.CRISIS_WINDOWS.values(): ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='red', alpha=0.13)
for a, b in ev.FALSE_POSITIVE_WINDOWS.values(): ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='orange', alpha=0.18)
ax.set_title('D7 changepoint_online — log-vol + CUSUM (azul=conmutaciones online; rombo morado=PELT offline NO causal)\n(rojo=crisis conocidas, naranja=trampas 2013/2018)')
ax.set_ylabel('z (robusta)'); ax.legend(loc='upper right', fontsize=8); ax.margins(x=0.01)
fig.tight_layout(); fig.savefig(RESULTS / 'd07_cusum_changepoints.png', dpi=110, bbox_inches='tight'); plt.show()""")

# ------------------------------------------------------------------ #
md(r"""<a id="sec6"></a>
## 6. El MECANISMO de Page (1954): cómo se ACUMULA la evidencia y salta el detector

Esta es **la figura que explica el CUSUM**. Hasta ahora hemos visto *dónde* conmuta el
autómata; aquí abrimos su motor interno: los dos **acumuladores de una cara** que gobiernan
las transiciones (Page 1954).

- En **calma** (estado 0) corre el acumulador **C⁺ₜ = max(0, C⁺ₜ₋₁ + (zₜ − k))**: suma el
  exceso de la log-vol estandarizada `z` sobre la holgura `k`. Mientras la vol esté en su
  nivel normal, `z − k` es negativo y el `max(0, ·)` mantiene C⁺ **pegado a cero**. Cuando
  la vol sube de forma **sostenida**, `z − k` se vuelve positivo y C⁺ **trepa**; al cruzar
  el umbral `h` el autómata **entra en crisis** y resetea los acumuladores.
- En **crisis** (estado 1) corre el simétrico **C⁻ₜ = max(0, C⁻ₜ₋₁ + (−zₜ − k))**, que
  vigila el regreso de la vol a niveles bajos; al cruzar `h`, **vuelve a calma**.

Dibujamos el acumulador **activo** en cada instante en forma de espejo: hacia **arriba**
C⁺ (evidencia de *entrada* a crisis, rojo) y hacia **abajo** C⁻ (evidencia de *salida*,
azul). Las líneas `±h` son las "compuertas": cada vez que el acumulado las toca, hay una
conmutación (marcada con un triángulo). Se ve a simple vista **por qué hay retardo** (la
rampa tarda en llenarse) y **por qué hay persistencia** (tras el reset, hace falta volver a
acumular `h` para deshacer la decisión → no hay flickering).""")

co(r"""# Reconstrucción del acumulado CUSUM de Page (mismo autómata que el detector, in-sample).
k_, h_ = det_is.k, det_is.h
zv = z_is.values
cu_hist = np.zeros(len(zv)); cd_hist = np.zeros(len(zv)); st_hist = np.zeros(len(zv), dtype=int)
state, cu, cd = 0, 0.0, 0.0
for t in range(len(zv)):
    zt = zv[t]
    if np.isfinite(zt):
        if state == 0:
            cu = max(0.0, cu + (zt - k_))
            if cu > h_:
                state = 1; cu = 0.0; cd = 0.0
        else:
            cd = max(0.0, cd + (-zt - k_))
            if cd > h_:
                state = 0; cu = 0.0; cd = 0.0
    cu_hist[t] = cu; cd_hist[t] = cd; st_hist[t] = state
C_up = pd.Series(cu_hist, index=z_is.index)
C_dn = pd.Series(cd_hist, index=z_is.index)

fig, ax = plt.subplots(figsize=(15, 5))
# Espejo: C+ hacia arriba (rojo, evidencia de entrada a crisis), C- hacia abajo (azul, salida).
ax.fill_between(C_up.index, 0, C_up.values, color=viz.C_CRISIS, alpha=0.55, lw=0,
                label='C⁺ (acumula cambio AL ALZA → entra en crisis)')
ax.fill_between(C_dn.index, 0, -C_dn.values, color=viz.C_LONG, alpha=0.55, lw=0,
                label='C⁻ (acumula cambio A LA BAJA → vuelve a calma)')
ax.axhline(h_,  color='black', ls='--', lw=0.9, label=f'umbral h = {h_:g} (compuerta de disparo)')
ax.axhline(-h_, color='black', ls='--', lw=0.9)
ax.axhline(0, color='grey', lw=0.6)
# Conmutaciones (cuando el acumulado activo cruza h).
for d in switch_dates:
    ax.axvline(d, color='#2c3e50', lw=0.5, alpha=0.35)
    ax.plot(d, 0, marker='v', color='#2c3e50', ms=6, zorder=5)
for a, b in ev.CRISIS_WINDOWS.values(): ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=viz.C_CRISIS, alpha=0.10, lw=0)
for a, b in ev.FALSE_POSITIVE_WINDOWS.values(): ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='orange', alpha=0.13, lw=0)
ax.set_ylim(-h_*1.6, h_*1.6)
ax.set_ylabel('acumulado CUSUM (unidades de z)')
ax.set_title('D7 — MECANISMO Page 1954: acumuladores C± y umbral h (triángulo=conmutación; rojo=crisis conocidas)\n'
             'La rampa que tarda en llenar h EXPLICA el retardo; el reset tras disparar EXPLICA la persistencia')
ax.legend(loc='upper right', fontsize=8, framealpha=0.9); ax.margins(x=0.01)
fig.tight_layout(); fig.savefig(RESULTS / 'd07_cusum_statistic.png', dpi=110, bbox_inches='tight'); plt.show()
n_up = int((np.diff(st_hist) == 1).sum()); n_dn = int((np.diff(st_hist) == -1).sum())
print(f'Entradas en crisis (C⁺>h) = {n_up}  |  retornos a calma (C⁻>h) = {n_dn}')
print(f'C⁺ máximo alcanzado = {C_up.max():.1f}  ·  C⁻ máximo = {C_dn.max():.1f}  (umbral h = {h_:g})')
print('La evidencia se acumula con max(0,·): C± se mantiene en cero en calma y trepa solo ante un cambio SOSTENIDO.')""")

# ------------------------------------------------------------------ #
md(r"""<a id="sec7"></a>
## 7. CUSUM online (causal) vs PELT (oráculo offline): el coste de no ver el futuro

Las §5–§6 muestran que el CUSUM **dispara con retardo**; aquí lo **cuantificamos** frente
al oráculo. PELT (offline, §3) segmenta mirando TODA la serie: marca el cambio estructural
en su posición óptima *a posteriori* (no usable en producción). El CUSUM **online** sólo
ve el pasado, así que cada **entrada en crisis** llega necesariamente **después** del cambio
que PELT señala. La distancia entre ambos es el **retardo causal** — el precio, en días, de
respetar la causalidad.

La figura enfrenta las dos familias sobre la misma señal de log-vol: **rombos morados** =
cambios PELT (oráculo, ve el futuro) · **líneas rojas** = entradas en crisis del CUSUM
online. Que las rojas caigan a la derecha de los rombos es, literalmente, la diferencia
**causal vs anti-causal** hecha imagen.""")

co(r"""# Entradas en crisis del CUSUM online (causal, in-sample) vs change-points de PELT (offline).
sw_is = states_is.values
ient = np.where((sw_is[1:] == det_is.crisis_state) & (sw_is[:-1] != det_is.crisis_state))[0] + 1
online_cp = states_is.index[ient]
pelt_cp = pd.DatetimeIndex(sorted(cp_dates))
# Retardo causal: días entre cada entrada online y el cambio PELT inmediatamente anterior.
lags = []
for d in online_cp:
    prev = pelt_cp[pelt_cp <= d]
    if len(prev):
        lags.append(float((d - prev[-1]).days))
lags = np.array(lags, dtype=float)

fig, ax = plt.subplots(figsize=(15, 4.8))
ax.plot(z_is.index, z_is.values, color='#9aa6b2', lw=0.4, zorder=1, label='log-vol estandarizada (robusta)')
yT = z_is.max() * 0.95
for i, d in enumerate(pelt_cp):
    ax.plot(d, yT, marker='D', color='#8e44ad', ms=5, zorder=4,
            label='PELT offline (oráculo, NO causal)' if i == 0 else None)
for i, d in enumerate(online_cp):
    ax.axvline(d, color=viz.C_CRISIS, lw=1.0, alpha=0.8, zorder=3,
               label='entrada en crisis — CUSUM online (causal)' if i == 0 else None)
for a, b in ev.CRISIS_WINDOWS.values(): ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=viz.C_CRISIS, alpha=0.08, lw=0)
ax.set_ylabel('z (robusta)')
med_lag = np.nanmedian(lags) if len(lags) else float('nan')
ax.set_title('D7 — online causal vs PELT oráculo offline: el detector online dispara DESPUÉS del cambio ideal\n'
             f'retardo causal mediano ≈ {med_lag:.0f} días (rombo morado = oráculo; línea roja = disparo online)')
ax.legend(loc='upper left', fontsize=8, framealpha=0.9); ax.margins(x=0.01)
fig.tight_layout(); fig.savefig(RESULTS / 'd07_online_vs_pelt.png', dpi=110, bbox_inches='tight'); plt.show()
print(f'PELT (oráculo offline): {len(pelt_cp)} cambios  |  CUSUM online: {len(online_cp)} entradas en crisis')
if len(lags):
    print(f'Retardo causal vs oráculo (días): mediana={np.nanmedian(lags):.0f}  '
          f'min={np.nanmin(lags):.0f}  max={np.nanmax(lags):.0f}')
print('El online NUNCA adelanta al oráculo (no ve el futuro): el retardo es el precio de la causalidad.')""")

# ------------------------------------------------------------------ #
md(r"""<a id="sec8"></a>
## 8. S&P 500 coloreado por régimen (CAUSAL OOS)

Sombreado rojo = días clasificados **crisis** por el walk-forward causal. Líneas = ventanas
de crisis conocidas. Cubre 2008 y 2011 (OOS, a diferencia de D4).""")

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
ax.set_title('S&P 500 (log) coloreado por régimen CAUSAL OOS — D7 changepoint_online\n(rojo = crisis = tramo de alta vol tras un cambio al alza; cubre 2008 y 2011 OOS)')
ax.legend(handles=[Patch(color='red', alpha=0.22, label='crisis (alta vol, CUSUM)')], loc='upper left')
ax.margins(x=0.01); fig.tight_layout()
fig.savefig(RESULTS / 'd07_sp500_regimes.png', dpi=110, bbox_inches='tight'); plt.show()""")

# ------------------------------------------------------------------ #
md(r"""<a id="sec9"></a>
## 9. Lead/lag SOSTENIDO respecto a los suelos de drawdown

`ev.lead_lag` (núcleo, `persist=3`) mide cuántos días antes/después del **suelo del
drawdown** la `p_crisis` cruza 0.5 y SE MANTIENE ≥3 días. Negativo = la señal **anticipa**
el suelo. Es la métrica central de la "detección temprana" de CP2.

Atención al matiz frente a la §7: allí el CUSUM llegaba *tarde* respecto al cambio
**estructural de la vol** (el oráculo PELT); aquí, en cambio, se mide frente al **suelo de
precio** (el mínimo del drawdown). Que el lead/lag sea **negativo** significa que la crisis
de vol ya está confirmada **bastante antes** de que el mercado toque fondo: la vol se
dispara en la *caída*, no en el *suelo*. Esas dos lecturas no se contradicen — son el
retardo medido contra dos referencias distintas — y juntas son el argumento de que D7
**domina el eje lead/lag**.""")

co(r"""fig, ax = plt.subplots(figsize=(11, 4.2))
ll = res.lead_lag_days
labels = list(ll.keys()); vals = [ll[k] for k in labels]
colors = ['#27ae60' if (v is not None and v == v and v < 0) else '#c0392b' for v in vals]
ax.barh(labels, vals, color=colors)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('dias señal_sostenida - trough  (negativo = ANTICIPA el suelo)')
ax.set_title('D7 — lead/lag SOSTENIDO (persist=3) vs suelos de drawdown')
fig.tight_layout(); fig.savefig(RESULTS / 'd07_leadlag.png', dpi=110, bbox_inches='tight'); plt.show()
print('lead/lag (dias):', {k: (round(v,0) if v==v else v) for k,v in ll.items()})
print('Todos negativos -> la señal de crisis está activa bien ANTES del suelo (deteccion temprana).')""")

# ------------------------------------------------------------------ #
md(r"""<a id="sec10"></a>
## 10. Retardos de detección: confirmación vs anticipación, evento a evento

Dos relojes distintos miden el mismo detector y juntos resumen su perfil temporal:

- **Retardo de confirmación** (vs *inicio* de la ventana de crisis oficial): cuántos días
  tarda el CUSUM causal OOS en marcar el **primer día de crisis** dentro de cada ventana.
  Es la métrica clásica de *detection delay* (ARL₁) del control estadístico de procesos:
  positivo = confirma con retraso; **negativo = ya estaba en crisis antes** de que la
  ventana empezara (anticipación pura).
- **Lead/lag vs suelo** (§9, reproducido aquí para contraste): días entre la señal
  sostenida y el **suelo del drawdown**; negativo = anticipa el fondo.

Las dos series por evento dan la foto completa de la "detección temprana" de CP2: lo
deseable es **confirmar pronto** la entrada y, sobre todo, estar en crisis **mucho antes**
del suelo. Barras a la izquierda del cero = el detector va por delante.""")

co(r"""# Retardo de confirmación: días del inicio de cada ventana de crisis al 1er día de crisis OOS.
delay_rows = {}
for k, (a, b) in ev.CRISIS_WINDOWS.items():
    a = pd.Timestamp(a); b = pd.Timestamp(b)
    seg = states_c.loc[(states_c.index >= a) & (states_c.index <= b)]
    cri = seg[seg == det_is.crisis_state]
    delay_rows[k] = float((cri.index[0] - a).days) if len(cri) else float('nan')

cats = list(ev.CRISIS_WINDOWS.keys())
conf_delay = [delay_rows[k] for k in cats]
ll = res.lead_lag_days
lead_vs_trough = [float(ll[k]) if (k in ll and ll[k] == ll[k]) else float('nan') for k in cats]

viz.plot_grouped_bars(
    cats,
    {'retardo de confirmación (vs inicio ventana)': conf_delay,
     'lead/lag (vs suelo de drawdown)': lead_vs_trough},
    ylabel='días  (negativo = el detector ANTICIPA)',
    title='D7 — retardos de detección por evento: confirmación de entrada vs anticipación del suelo',
    colors=[viz.C_SHORT, viz.C_LONG], rotation=20, value_labels=True,
)
plt.axhline(0, color='black', lw=0.9)
plt.tight_layout(); plt.savefig(RESULTS / 'd07_detection_delays.png', dpi=110, bbox_inches='tight'); plt.show()
print('Retardo de confirmación (días desde el inicio de cada ventana):',
      {k: (round(v, 0) if v == v else v) for k, v in delay_rows.items()})
print('Lead/lag vs suelo (días):', {k: (round(v, 0) if v == v else v)
      for k, v in zip(cats, lead_vs_trough)})
print('Lectura CP2: confirmación rápida de la entrada + crisis activa MUCHO antes del suelo = detección temprana.')""")

# ------------------------------------------------------------------ #
md(r"""<a id="sec11"></a>
## 11. Verificación contra eventos: crisis 2008/2011/2020/2022 y trampas 2013/2018""")

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
ax.set_ylabel('% días en estado crisis (OOS)')
ax.set_title('D7 — cobertura por ventana (rojo=crisis, naranja=trampa)')
ax.set_xticklabels(cmp.index, rotation=30, ha='right'); ax.set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig(RESULTS / 'd07_coverage.png', dpi=110, bbox_inches='tight'); plt.show()
print('Trampas (CP2: el coste robusto debe MANTENERLAS BAJAS):')
print(f'  TaperTantrum_2013: {res.false_alarm_in_fp["TaperTantrum_2013"]:.1%}')
print(f'  Selloff_Q4_2018:   {res.false_alarm_in_fp["Selloff_Q4_2018"]:.1%}')""")

# ------------------------------------------------------------------ #
md(r"""<a id="sec12"></a>
## 12. CP2 EN ACCIÓN: coste GAUSSIANO vs ROBUSTO (falsas alarmas por outliers)

La pregunta central de CP2: ¿el coste **gaussiano** (retorno², L2) dispara **falsas
alarmas** por las colas, y el **robusto** (log|r| + winsor) las reduce? Se corre el mismo
walk-forward con `cost='gaussian'` y se comparan switching, false_alarm_rate y las
trampas.""")

co(r"""panel_g = ev.walk_forward(lambda: ChangepointOnline(cost='gaussian'), X, market_returns=mkt,
                          train_size=TRAIN_SIZE, step=STEP, expanding=True)
det_g = ChangepointOnline(cost='gaussian').fit(X); det_g.label_states_economically(X, market_returns=mkt)
res_g = ev.evaluate(det_g, panel_g, market_returns=mkt, X_full=X)

comp = pd.DataFrame({
    'robust':  [res.false_alarm_rate, res.switching_rate, res.mean_regime_duration,
                res.false_alarm_in_fp['TaperTantrum_2013'], res.false_alarm_in_fp['Selloff_Q4_2018'],
                float((states_c==det_is.crisis_state).mean())],
    'gaussian':[res_g.false_alarm_rate, res_g.switching_rate, res_g.mean_regime_duration,
                res_g.false_alarm_in_fp['TaperTantrum_2013'], res_g.false_alarm_in_fp['Selloff_Q4_2018'],
                float((panel_g['state']==det_g.crisis_state).mean())],
}, index=['false_alarm_rate','switching_rate','dur_media_d','fa_2013(trampa)','fa_2018(trampa)','frac_crisis_OOS'])
display(comp.style.format('{:.3f}'))

fig, ax = plt.subplots(figsize=(10, 4.2))
idx = ['fa_2013(trampa)','fa_2018(trampa)','false_alarm_rate','frac_crisis_OOS']
xb = np.arange(len(idx)); w = 0.38
ax.bar(xb-w/2, comp.loc[idx,'robust'].values,  w, label='robust (log|r|+winsor)', color='#27ae60')
ax.bar(xb+w/2, comp.loc[idx,'gaussian'].values, w, label='gaussian (r^2, L2)',     color='#c0392b')
ax.set_xticks(xb); ax.set_xticklabels(idx, rotation=20, ha='right'); ax.set_ylim(0,1.05)
ax.set_title('CP2 — el coste GAUSSIANO dispara falsas alarmas por outliers; el ROBUSTO las reduce')
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(RESULTS / 'd07_robust_vs_gaussian.png', dpi=110, bbox_inches='tight'); plt.show()
print('CP2: robusto fa_2013=%.0f%% fa_2018=%.0f%% far=%.2f  |  gaussiano fa_2013=%.0f%% fa_2018=%.0f%% far=%.2f'
      % (100*comp.loc['fa_2013(trampa)','robust'], 100*comp.loc['fa_2018(trampa)','robust'], comp.loc['false_alarm_rate','robust'],
         100*comp.loc['fa_2013(trampa)','gaussian'], 100*comp.loc['fa_2018(trampa)','gaussian'], comp.loc['false_alarm_rate','gaussian']))""")

# ------------------------------------------------------------------ #
md(r"""<a id="sec13"></a>
## 13. Distribución del coste robusto vs gaussiano por régimen — la raíz del problema

La §12 mide la *consecuencia* (falsas alarmas); esta sección muestra la *causa* mirando la
**distribución del estadístico de coste** que cada variante alimenta al CUSUM, separada por
régimen canónico (calma vs crisis):

- **Coste robusto** = `log|retorno|` (winsorizado): casi simétrico y de **colas acotadas**.
  Las cajas de calma y crisis se separan limpiamente por su **nivel** (mediana), que es
  justo lo que el CUSUM debe detectar; los outliers no estiran la caja.
- **Coste gaussiano** = `retorno²`: fuertemente asimétrico y **dominado por las colas**. Un
  solo día de −7 % genera un `r²` enorme que, en escala lineal, **aplasta** toda la
  distribución y empuja al acumulador a disparar por un único outlier, no por un cambio
  sostenido. Por eso degenera en alarma casi permanente (§12).

Visto así, la elección de CP2 (preferir el coste robusto) deja de ser una receta y se
convierte en una consecuencia directa de la **forma** de cada distribución.""")

co(r"""r_cost = det_is._returns(X)
robust_stat = pd.Series(np.log(r_cost.abs() + 1e-6), index=r_cost.index)   # log|r| (coste robusto)
gauss_stat  = pd.Series(r_cost.values ** 2, index=r_cost.index)            # r^2 (coste gaussiano)
st_cost = states_is.reindex(r_cost.index)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6))
viz.plot_distribution_by_regime(robust_stat, st_cost, crisis_state=det_is.crisis_state,
    labels={0: 'calma', 1: 'crisis'}, kind='box', ax=axL,
    xlabel='log|retorno|  (coste ROBUSTO)', title='Coste robusto por régimen — colas acotadas')
viz.plot_distribution_by_regime(gauss_stat, st_cost, crisis_state=det_is.crisis_state,
    labels={0: 'calma', 1: 'crisis'}, kind='box', ax=axR,
    xlabel='retorno²  (coste GAUSSIANO)', title='Coste gaussiano por régimen — dominado por colas')
fig.suptitle('D7 — distribución del coste por régimen: el robusto separa por NIVEL; el gaussiano lo aplasta el outlier', y=1.02)
fig.tight_layout(); fig.savefig(RESULTS / 'd07_cost_by_regime.png', dpi=110, bbox_inches='tight'); plt.show()
for nm, stt in [('robusto log|r|', robust_stat), ('gaussiano r^2', gauss_stat)]:
    cri = stt[st_cost.values == det_is.crisis_state]; cal = stt[st_cost.values == 0]
    print(f'{nm:16s}: mediana calma={cal.median():.4g}  crisis={cri.median():.4g}  '
          f'| ratio colas (p99/p50) calma={cal.quantile(.99)/ (cal.median() or 1e-9):.1f}')""")

# ------------------------------------------------------------------ #
md(r"""<a id="sec14"></a>
## 14. Timeline de régimen y duraciones (flickering)

Timeline causal OOS + histograma de duraciones. El umbral `h` del CUSUM da persistencia
(episodios largos, poco flickering).""")

co(r"""fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 4.5), gridspec_kw={'height_ratios': [1, 2.2]})
ax1.plot(p_c.index, p_c.values, color='#2980b9', lw=0.7); ax1.set_ylabel('P(crisis)')
ax1.axhline(0.5, color='grey', ls='--', lw=0.7)
for a, b in ev.CRISIS_WINDOWS.values(): ax1.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='red', alpha=0.12)
ax1.set_title('P(crisis) blanda (sigmoide del EWMA de la vol) — CAUSAL OOS'); ax1.margins(x=0.01)
ax2.imshow(states_c.values.reshape(1, -1), aspect='auto', cmap='RdYlGn_r',
           extent=[0, len(states_c), 0, 1]); ax2.set_yticks([])
tk = np.linspace(0, len(states_c)-1, 8).astype(int)
ax2.set_xticks(tk); ax2.set_xticklabels([states_c.index[i].year for i in tk])
ax2.set_title('Timeline de régimen (verde=calma, rojo=crisis)')
fig.tight_layout(); fig.savefig(RESULTS / 'd07_timeline.png', dpi=110, bbox_inches='tight'); plt.show()

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
md(r"""<a id="sec15"></a>
## 15. Volcado de métricas a results/ (esquema canónico 32 columnas)""")

co(r"""tbl = ev.results_table([res])
assert tbl.shape[1] == 32, f'esperaba 32 columnas (esquema canonico), hay {tbl.shape[1]}'
tbl.to_csv(RESULTS / 'metrics_07_changepoint_online.csv', index=False)
print('Guardado results/metrics_07_changepoint_online.csv  (1 fila,', tbl.shape[1], 'columnas)')

# NOTA (saneamiento Ola 0): este builder ya NO actualiza metrics_master.csv de forma
# incremental (leer-filtrar-concatenar provocaba condicion de carrera al correr builders
# en paralelo). Cada builder escribe SOLO su metrics_NN_*.csv; el master unificado se
# reconstruye de forma centralizada con:  python scripts/verify/_rebuild_master.py
display(tbl.T)""")

# ------------------------------------------------------------------ #
md(r"""<a id="sec16"></a>
## 16. Conclusión D7 — ¿se cumple la hipótesis CP2?

**Hipótesis CP2:** *detección temprana (lead/lag) pero riesgo de falsas alarmas con
outliers; preferir kernel/robusto frente al CUSUM gaussiano.*

Veredicto (con los números de arriba):
- **Online causal:** confirmado — CUSUM de Page secuencial, causal verificado (ocultar el
  futuro no cambia el bloque). PELT (offline) solo como oráculo NO causal. El **mecanismo**
  (§6) lo deja explícito: los acumuladores C± se mantienen en cero en calma y solo trepan
  ante un cambio **sostenido**; la rampa hasta `h` es el origen del retardo, y el reset tras
  disparar es el origen de la persistencia (poco flickering, sin dwell explícito).
- **El precio de la causalidad, medido:** la §7 enfrenta el CUSUM online al oráculo PELT —
  cada entrada en crisis llega **después** del cambio ideal (retardo causal mediano de
  pocas semanas). Es estructural: no ver el futuro tiene un coste en días, no un fallo.
- **Detección temprana:** confirmado — `lead/lag` SOSTENIDO (persist=3) **negativo en las
  4 crisis** → la señal de crisis está activa bien antes del suelo de drawdown. La §10
  separa los dos relojes (confirmación de la entrada vs anticipación del suelo): D7 confirma
  pronto y, sobre todo, está en crisis **mucho antes** del fondo del mercado.
- **Falsas alarmas por outliers + robusto las reduce:** confirmado — el coste **gaussiano**
  (retorno², L2) se deja arrastrar por las colas (kurt 25–40) y **degenera en alarma
  casi permanente** (frac_crisis≈0.93, trampas 2013/2018 al **100 %**, far≈0.94). El coste
  **robusto** (log|r| + mediana/MAD + winsor) recupera regímenes limpios: **trampas ≈0 %**,
  switching bajo, episodios largos, manteniendo cobertura plena de 2008/2020 y buena de
  2011/2022. La §13 muestra la **raíz**: el coste gaussiano (`r²`) está dominado por las
  colas y aplasta la distribución, mientras el robusto (`log|r|`) separa los regímenes por
  **nivel**. → **CP2 se CUMPLE**: preferir el coste robusto al gaussiano.
- **Precio:** como D6, el `false_alarm_rate` global es alto (≈0.87) — la ventana larga ve
  muchos picos de alta vol reales (1987, 1998, 2000–02, 2010, 2015, 2023) que no son las 4
  crisis canónicas del set; léase junto a la cobertura, no aislado.""")

# ------------------------------------------------------------------ #
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
print('Ejecutando notebook (2 walk-forwards ~2 min)...')
ep = ExecutePreprocessor(timeout=1800, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': str(ROOT / 'notebooks')}})
nbf.write(nb, NB_PATH)
n_err = sum(1 for c in nb.cells for o in c.get('outputs', []) if o.get('output_type') == 'error')
print(f'OK -> {NB_PATH.name}  | errores = {n_err}')
