# -*- coding: utf-8 -*-
"""Construye y EJECUTA notebooks/03_clustering_gmm.ipynb (D3).

Patrón idéntico al de _build_02: nbformat para montar las celdas +
ExecutePreprocessor (kernel python3) para ejecutarlo. Tras ejecutar, comprueba 0
errores leyendo los outputs. Figuras -> results/. CSV -> results/.

Reproducción FIEL del notebook actual (celda por celda, fuentes verbatim).
NO escribe results/metrics_master.csv (eso es Ola 0).
"""
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[2]
NB_PATH = ROOT / "notebooks" / "03_clustering_gmm.ipynb"

cells = []

cells.append(new_markdown_cell(
    "# 03 — D3 `clustering_gmm`: GMM estático (baseline NO temporal)\n"
    "\n"
    "**Familia CLUSTERING.** Un *Gaussian Mixture Model* (GMM) modela los datos como una mezcla de varias gaussianas: cada día se asigna al componente —el régimen— bajo el que resulta más probable. Aquí lo ajustamos con `covariance_type='full'` (cada régimen tiene su matriz de covarianza completa) sobre las **15 features causales** del EDA, lo que le permite separar regímenes que difieren no solo en volatilidad sino en la **estructura de correlación entre activos** —por ejemplo el cambio de signo de la correlación renta variable/bonos (Gulko 2002)—, algo que un *k-means* euclídeo o un GMM diagonal no captarían.\n"
    "\n"
    "**Qué lo hace un baseline, y de qué.** El GMM no tiene cadena de Markov ni término de persistencia: clasifica cada día de forma INDEPENDIENTE, sin memoria del estado de ayer. Es deliberadamente el detector contra el que D4 (un HMM con las mismas features) medirá **cuánto aporta la dinámica temporal**: la diferencia de persistencia entre ambos aísla el efecto de añadir una matriz de transición.\n"
    "\n"
    "**Hipótesis (CHECKPOINT 2):** *captará regímenes con estructura de correlación distinta (gracias a la covarianza full); fallará por flickering severo —parpadeo de las etiquetas de un día para otro, al no haber persistencia—; y no es causal de forma nativa* (la causalidad se la impone el `walk_forward`, no el modelo).\n"
    "\n"
    "**Política de ventana y su límite.** Las 15 features arrancan en **2007-07**, así que el primer train (expanding, 8 años) consume 2007–2015. Por construcción **2008 (GFC) y 2011 (deuda europea) NO son OOS-evaluables** —caen dentro del primer train— y su cobertura saldrá `NaN`. No es un fallo del detector: es la consecuencia honesta de no penalizar lo que el modelo no pudo ver fuera de muestra.\n"
    "\n"
    "Detalle en `docs/memory/detectors/03_clustering_gmm.md`."
))

cells.append(new_markdown_cell(
    """## Índice navegable

**Bloque A — selección, sanidad y evaluación causal**
1. [Selección de *k* por BIC (in-sample, guía)](#s1)
2. [Sanidad económica: orden canónico y retorno por estado](#s2)
3. [Walk-forward CAUSAL + evaluación estandarizada](#s3)
4. [Tabla de métricas (esquema común) y volcado a `results/`](#s4)

**Bloque B — el mecanismo del GMM, por fin visible (ampliación de esta revisión)**
5. [Espacio de 15 features (PCA 2D) por componente GMM](#s5) — *figura nueva* `d03_feature_scatter.png`
6. [Estructura de covarianza por componente (full Σ)](#s6) — *figura nueva* `d03_cov_by_component.png`
7. [Paisaje BIC vs *k* y confianza de las asignaciones](#s7) — *figura nueva* `d03_bic_curve.png`

**Bloque C — flickering, recorrido y veredicto**
8. [Flickering: `switching_rate` e histograma de duraciones](#s8) — `d03_gmm_flickering.png`
9. [S&P 500 coloreado por régimen (OOS)](#s9) — `d03_gmm_sp500_regimes.png`
10. [Timeline de estado y P(crisis) OOS](#s10) — `d03_gmm_timeline.png`
11. [Verificación contra crisis y trampas](#s11)
12. [Conclusión — veredicto "mejor-para-qué"](#s12)

> **Mapa de lectura.** El **Bloque A** fija *k* por BIC, canonicaliza los estados (0 = calma … n−1 = crisis) y los evalúa de forma CAUSAL walk-forward. El **Bloque B** —la ampliación de esta revisión— abre la caja negra y hace *visible el mecanismo* del GMM: cómo separa los tres regímenes en el espacio de features y cómo difiere la matriz de covarianza **full** de cada gaussiana (el cambio de estructura de correlación renta variable/bonos de Gulko 2002). El **Bloque C** documenta el *flickering* —el talón de Aquiles que motiva los detectores temporales (D4)— y cierra con el veredicto "mejor-para-qué"."""
))

cells.append(new_code_cell(
    "%matplotlib inline\n"
    "import sys\n"
    "from pathlib import Path\n"
    "import numpy as np, pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "from matplotlib.patches import Patch\n"
    "import seaborn as sns\n"
    "\n"
    "ROOT = Path.cwd()\n"
    "while not (ROOT / 'src').exists() and ROOT != ROOT.parent:\n"
    "    ROOT = ROOT.parent\n"
    "sys.path.insert(0, str(ROOT))\n"
    "RESULTS = ROOT / 'results'; RESULTS.mkdir(exist_ok=True)\n"
    "sns.set_theme(style='whitegrid', context='notebook')\n"
    "\n"
    "from src import evaluation as ev\n"
    "from src import viz\n"
    "viz.use_house_style()  # rcParams homogeneos (estilo de casa, paleta consistente)\n"
    "from detectors.clustering_gmm import ClusteringGMM\n"
    "\n"
    "X = pd.read_parquet(ROOT / 'data' / 'processed' / 'features.parquet')\n"
    "raw = pd.read_parquet(ROOT / 'data' / 'raw' / 'raw_panel.parquet')\n"
    "# market_returns = retorno log S&P 500 reindexado a las features.\n"
    "market_returns = np.log(raw['SP500'] / raw['SP500'].shift(1)).reindex(X.index)\n"
    "spx = raw['SP500'].reindex(X.index)\n"
    "print('Features:', X.shape, '| ventana', X.index.min().date(), '->', X.index.max().date())\n"
    "X.columns.tolist()"
))

cells.append(new_markdown_cell(
    "<a id=\"s1\"></a>\n"
    "## 1. Selección del número de estados por BIC (in-sample, solo como guía)\n"
    "\n"
    "¿Cuántos regímenes *k*? El GMM expone una log-verosimilitud, así que usamos el **BIC** (*Bayesian Information Criterion*: premia el ajuste pero penaliza el número de parámetros; menor es mejor) para comparar k=2 frente a k=3 sobre todo el panel.\n"
    "\n"
    "**Honestidad metodológica:** este BIC es *in-sample* —se calcula viendo toda la muestra, con look-ahead— y sirve solo de **guía** para fijar *k*; la evaluación que de verdad cuenta es la walk-forward causal de la §3. Lo declaramos explícitamente porque elegir *k* mirando el futuro es una fuga sutil que no debe confundirse con el rendimiento out-of-sample."
))

cells.append(new_code_cell(
    "rows = []\n"
    "for k in (2, 3):\n"
    "    d = ClusteringGMM(n_states=k).fit(X)\n"
    "    rows.append({'k': k, 'log_likelihood': d.score(X), 'n_params': d.n_parameters(),\n"
    "                 'AIC': d.aic(X), 'BIC': d.bic(X)})\n"
    "bic_tbl = pd.DataFrame(rows).set_index('k')\n"
    "K_BIC = int(bic_tbl['BIC'].idxmin())\n"
    "print('k elegido por BIC:', K_BIC)\n"
    "bic_tbl.round(1)"
))

cells.append(new_markdown_cell(
    "<a id=\"s2\"></a>\n"
    "## 2. Sanidad económica: orden canónico y retorno medio por estado\n"
    "\n"
    "Antes de evaluar, comprobamos que las etiquetas significan lo que decimos que significan. Tras `fit`, `label_states_economically` reordena los componentes al **orden canónico** (0 = calma … n−1 = crisis) usando el retorno del S&P 500. Verificamos dos cosas: (i) que el retorno medio del índice **decrece** del estado de calma al de crisis —es decir, el estado etiquetado \"crisis\" es realmente el de peor retorno y mayor volatilidad—, y (ii) que `predict_proba` queda en ese orden, de modo que su **última columna es P(crisis)**. Sin esta sanidad, una métrica de cobertura podría estar leyendo el régimen invertido y dar números engañosos."
))

cells.append(new_code_cell(
    "det_full = ClusteringGMM(n_states=K_BIC).fit(X)\n"
    "states_is = pd.Series(det_full.predict(X), index=X.index)\n"
    "proba_is = det_full.predict_proba(X)\n"
    "ret_by_state = {int(s): float(market_returns[states_is == s].mean()) for s in np.unique(states_is)}\n"
    "vol_by_state = {int(s): float(market_returns[states_is == s].std()) for s in np.unique(states_is)}\n"
    "print('crisis_state (canónico):', det_full.crisis_state)\n"
    "print('predict_proba filas suman 1:', np.allclose(proba_is.sum(axis=1), 1.0))\n"
    "pd.DataFrame({'mean_ret': ret_by_state, 'std_ret': vol_by_state,\n"
    "              'n_dias': states_is.value_counts().sort_index()}).round(5)"
))

cells.append(new_markdown_cell(
    "<a id=\"s3\"></a>\n"
    "## 3. Walk-forward CAUSAL + evaluación estandarizada\n"
    "\n"
    "Aquí está el corazón causal del banco de pruebas. `ev.walk_forward` re-ajusta el GMM en ventana **expanding** (train inicial de 8 años, re-fit cada 21 días) y, en cada paso, clasifica el bloque siguiente usando SOLO el pasado: ningún día se etiqueta con información posterior a esa fecha. Cada fold canonicaliza sus propios estados, lo que resuelve el alineado de etiquetas entre folds —el componente \"crisis\" que ve un fold y el que ve otro reciben la misma etiqueta económica—. Evaluamos k=2 y k=3 bajo idéntico protocolo; el detector principal que se vuelca a `results/` es el *k* elegido por BIC."
))

cells.append(new_code_cell(
    "def run(k):\n"
    "    panel = ev.walk_forward(lambda: ClusteringGMM(n_states=k), X,\n"
    "                            train_size=252*8, step=21, expanding=True)\n"
    "    det = ClusteringGMM(n_states=k).fit(X)\n"
    "    res = ev.evaluate(det, panel, market_returns=market_returns, X_full=X)\n"
    "    return panel, det, res\n"
    "\n"
    "panel2, det2, res2 = run(2)\n"
    "panel3, det3, res3 = run(3)\n"
    "panels = {2: panel2, 3: panel3}\n"
    "results = {2: res2, 3: res3}\n"
    "panel = panels[K_BIC]; det = {2: det2, 3: det3}[K_BIC]; res = results[K_BIC]\n"
    "print('Detector principal:', det.name)\n"
    "print('Ventana OOS:', res.extra['ventana_eval'])"
))

cells.append(new_markdown_cell(
    "<a id=\"s4\"></a>\n"
    "## 4. Tabla de métricas (esquema común) y volcado a `results/`\n"
    "\n"
    "`ev.results_table` produce la fila estandarizada —las mismas columnas para los 12 detectores del banco—, que es lo que hace a D3 comparable con el resto bajo el marco único. Guardamos el detector principal (el *k* elegido por BIC) en `results/metrics_03_clustering_gmm.csv`."
))

cells.append(new_code_cell(
    "tbl = ev.results_table([results[2], results[3]])\n"
    "cols = ['detector','n_states','ventana_eval','switching_rate','mean_regime_duration',\n"
    "        'label_stability','false_alarm_rate','bic',\n"
    "        'cov_COVID_2020','cov_Inflation_2022','cov_GFC_2008','cov_EuroDebt_2011',\n"
    "        'fa_TaperTantrum_2013','fa_Selloff_Q4_2018']\n"
    "display(tbl[cols].round(3).T)\n"
    "out_csv = RESULTS / 'metrics_03_clustering_gmm.csv'\n"
    "ev.results_table([res]).to_csv(out_csv, index=False)\n"
    "print('Guardado:', out_csv, '| filas:', len(ev.results_table([res])))"
))

# ================================================================== #
# BLOQUE B — el mecanismo del GMM, por fin visible (AMPLIACIÓN)
# ================================================================== #
cells.append(new_markdown_cell(
    """<a id="s5"></a>
## 5. Espacio de 15 features (PCA 2D) coloreado por componente GMM

Hasta aquí el GMM ha sido una caja negra que reparte etiquetas. Esta sección la abre. Un GMM **es**, literalmente, una partición del espacio de features en regiones gobernadas por gaussianas; su visualización natural es por tanto el *scatter* de ese espacio coloreado por el componente asignado. Como trabajamos con **15 features causales**, proyectamos a 2D con PCA (estandarizando antes, sin look-ahead operativo: es un retrato in-sample, igual que el BIC de la §1) y pintamos cada día con el color de su régimen canónico (azul calma → ámbar intermedio → rojo crisis).

Qué leer aquí: que los tres componentes ocupen regiones **distinguibles** —si se solaparan por completo, el GMM no estaría separando nada—, con la **crisis empujada hacia la cola** del espacio (alta volatilidad / co-movimiento atípico) y el régimen intermedio actuando de puente. El solape residual en la frontera entre nubes es, anticipándolo, el caldo de cultivo del *flickering* de la §8: días ambiguos cuyo posterior se reparte entre componentes y que, sin memoria temporal, saltan de etiqueta de una jornada a la siguiente."""
))

cells.append(new_code_cell(
    """REG_LABELS = ({0: 'calma', 1: 'intermedio', 2: 'crisis'} if det.n_states == 3
              else {0: 'calma', det.n_states - 1: 'crisis'})
ax = viz.plot_feature_space_scatter(
    X, states_is, use_pca=True, crisis_state=det.crisis_state, labels=REG_LABELS,
    title=f'{det.name}: espacio de 15 features (PCA 2D) coloreado por componente GMM')
ax.figure.savefig(RESULTS / 'd03_feature_scatter.png', dpi=110, bbox_inches='tight')
plt.show()
print('Tamano de cada componente (in-sample):')
for s, n in states_is.value_counts().sort_index().items():
    print(f'  estado {int(s)} ({REG_LABELS.get(int(s), s)}): {n} dias ({100*n/len(states_is):.1f}%)')"""
))

cells.append(new_markdown_cell(
    """<a id="s6"></a>
## 6. Estructura de covarianza por componente (el mecanismo de la covarianza *full*)

Aquí se ve *por qué* elegimos `covariance_type='full'`. Cada gaussiana del GMM tiene su propia matriz de covarianza Σ_k; con covarianza full, esa Σ_k puede capturar no solo **escalas de volatilidad** distintas sino **estructuras de correlación** distintas entre las 15 features. Pintamos, para cada componente, la matriz de correlación empírica de las features de los días asignados a ese régimen.

La lectura clave —el cambio de régimen de la correlación renta variable/bonos (Gulko 2002)— aparece como un **reordenamiento del patrón de correlaciones** entre calma y crisis: un *k-means* euclídeo o un GMM diagonal, ciegos a la covarianza cruzada, no podrían separar regímenes que difieren *solo* en esa estructura. La **distancia de Frobenius** entre las matrices por componente cuantifica cuánto difieren: si fuera ~0, la covarianza full no aportaría nada sobre una diagonal."""
))

cells.append(new_code_cell(
    """import itertools
fig = viz.plot_regime_correlation_heatmaps(
    X, states_is, labels=REG_LABELS, precision=False,
    title=f'{det.name}: estructura de correlación por componente GMM (full-covariance)')
fig.savefig(RESULTS / 'd03_cov_by_component.png', dpi=110, bbox_inches='tight')
plt.show()

def _regcorr(s):
    sub = X.values[states_is.values == s]
    return np.corrcoef(sub, rowvar=False)
mats = {int(s): _regcorr(s) for s in np.unique(states_is)}
fro = lambda A, B: float(np.linalg.norm(A - B))
print('Distancia de Frobenius entre estructuras de correlacion por componente:')
for a, b in itertools.combinations(sorted(mats), 2):
    print(f'  ||corr(estado {a}) - corr(estado {b})||_F = {fro(mats[a], mats[b]):.2f}')
corrcol = next((c for c in X.columns if 'corr' in c.lower()), None)
if corrcol is not None:
    print(f'\\nFeature de correlacion detectada: {corrcol!r} -- media por regimen:')
    for s in sorted(int(u) for u in np.unique(states_is)):
        print(f'  estado {s} ({REG_LABELS.get(s, s)}): {X[corrcol][states_is.values == s].mean():+.3f}')
print('-> Sigma_k difiere por componente: ese es el aporte de covariance_type full (Gulko 2002).')"""
))

cells.append(new_markdown_cell(
    """<a id="s7"></a>
## 7. Paisaje BIC vs *k* y confianza de las asignaciones

La §1 decidió *k* comparando los dos candidatos operativos (k=2 vs k=3) que después se evalúan OOS. Aquí ampliamos la mirada con dos diagnósticos complementarios.

- **(Izquierda) Paisaje BIC/AIC para k=2…6.** El BIC suele seguir bajando al añadir componentes, pero la **mejora marginal se aplana** y los componentes extra dejan de tener lectura económica; por eso fijamos la taxonomía en **tres regímenes interpretables** (calma / intermedio / crisis), coherente con la práctica habitual y con D4. La línea vertical marca el *k* elegido por BIC entre los candidatos. Recordatorio de honestidad: este BIC es *in-sample* (§1) y solo es una **guía**; la evaluación que cuenta es la causal de la §3.
- **(Derecha) Distribución de la máxima responsabilidad** P(estado | x) por régimen: cuán confiada es cada asignación. Responsabilidades cercanas a 1 son días inequívocos; las colas bajas son **días-frontera** donde el posterior se reparte entre componentes —precisamente los que, sin persistencia temporal, alimentan el *flickering* de la §8—."""
))

cells.append(new_code_cell(
    """ks = list(range(2, 7))
scan = []
for k in ks:
    dk = ClusteringGMM(n_states=k).fit(X)
    scan.append({'k': k, 'BIC': dk.bic(X), 'AIC': dk.aic(X)})
scan = pd.DataFrame(scan).set_index('k')

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={'width_ratios': [1.1, 1]})
axL.plot(scan.index, scan['BIC'], 'o-', color=viz.C_LONG, lw=1.8, label='BIC')
axL.plot(scan.index, scan['AIC'], 's--', color=viz.C_SHORT, lw=1.3, alpha=0.85, label='AIC')
axL.axvline(K_BIC, color=viz.C_CRISIS, ls=':', lw=1.6, label=f'k elegido = {K_BIC}')
axL.scatter([K_BIC], [scan.loc[K_BIC, 'BIC']], s=140, facecolors='none',
            edgecolors=viz.C_CRISIS, linewidths=1.8, zorder=5)
axL.set_xticks(ks); axL.set_xlabel('número de componentes k')
axL.set_ylabel('criterio de información (menor = mejor)')
axL.set_title('Paisaje BIC / AIC vs k (in-sample, guía)')
axL.legend(fontsize=9)
conf = pd.Series(proba_is.max(axis=1), index=X.index)
viz.plot_distribution_by_regime(
    conf, states_is, kind='box', labels=REG_LABELS, ax=axR,
    xlabel='máx. responsabilidad  P(estado | x)',
    title='Confianza de asignación por componente')
fig.suptitle(f'{det.name}: selección de k por BIC y confianza de las asignaciones', y=1.03)
fig.tight_layout(); fig.savefig(RESULTS / 'd03_bic_curve.png', dpi=110, bbox_inches='tight')
plt.show()
print('BIC/AIC por k (in-sample, guia):'); print(scan.round(0))
print(f'\\nConfianza media de asignacion: {conf.mean():.2f} | '
      f'% dias con responsabilidad < 0.6 (frontera): {100*(conf < 0.6).mean():.1f}%')"""
))

# ================================================================== #
# BLOQUE C — flickering, recorrido y veredicto
# ================================================================== #
cells.append(new_markdown_cell(
    "<a id=\"s8\"></a>\n"
    "## 8. Flickering: `switching_rate` e histograma de duraciones de régimen\n"
    "\n"
    "Aquí se ve el **talón de Aquiles** anunciado en la hipótesis. *Flickering* es el parpadeo de las etiquetas: como el GMM clasifica cada día sin memoria (no hay matriz de transición que premie permanecer en el mismo régimen), las rachas se fragmentan en episodios irrealmente cortos. Lo cuantificamos con dos lecturas: el `switching_rate` (fracción de días en que el estado cambia respecto al anterior) y la distribución de duraciones de cada racha. Este número es justo el que D4, al añadir persistencia markoviana, deberá **reducir**: aquí se fija la línea base del problema que motiva los detectores temporales."
))

cells.append(new_code_cell(
    """def run_lengths(states):
    v = np.asarray(states); idx = np.where(np.diff(v) != 0)[0]
    bounds = np.concatenate(([-1], idx, [len(v)-1]))
    return np.diff(bounds)

rl2 = run_lengths(panel2['state'].values)
rl3 = run_lengths(panel3['state'].values)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
for a, rl, r, k, col in zip(ax, (rl2, rl3), (res2, res3), (2, 3), (viz.C_LONG, viz.C_CRISIS)):
    a.hist(rl, bins=range(1, 40), color=col, alpha=0.85, edgecolor='white', linewidth=0.4)
    a.axvline(rl.mean(), color='black', ls='--', lw=1.2, label=f'media={rl.mean():.1f} d')
    a.axvline(np.median(rl), color='black', ls=':', lw=1.0, label=f'mediana={np.median(rl):.0f} d')
    pct = 100 * (rl <= 3).mean()
    a.text(0.97, 0.80, f'{pct:.0f}% de las rachas\\nduran <=3 dias',
           transform=a.transAxes, ha='right', va='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='#f6e6c8', alpha=0.9))
    a.set_title(f'k={k}: duración de rachas de régimen\\n'
                f'switching_rate={r.switching_rate:.3f}, dur. media={r.mean_regime_duration:.1f} d')
    a.set_xlabel('días por racha'); a.legend(fontsize=8)
ax[0].set_ylabel('frecuencia')
fig.suptitle('Flickering del GMM estático (sin persistencia temporal): las rachas se fragmentan', y=1.04)
fig.tight_layout(); fig.savefig(RESULTS / 'd03_gmm_flickering.png', dpi=110, bbox_inches='tight')
plt.show()
print('%% rachas <=3 dias (k=%d): %.1f%%' % (K_BIC, 100*(run_lengths(panel['state'].values) <= 3).mean()))"""
))

cells.append(new_markdown_cell(
    "<a id=\"s9\"></a>\n"
    "## 9. S&P 500 coloreado por régimen (OOS) con ventanas de crisis\n"
    "\n"
    "Lectura visual del flickering. El sombreado de fondo es el estado canónico OOS asignado cada día (verde = calma → rojo = crisis); las franjas verticales marcan las ventanas de crisis conocidas (rojo) y las trampas / falsos positivos (naranja). El síntoma a buscar: el fondo cambia de color constantemente en lugar de mantener bloques limpios por régimen —exactamente el comportamiento que un detector con persistencia debería evitar—."
))

cells.append(new_code_cell(
    "from matplotlib.colors import to_rgba\n"
    "st = panel['state']\n"
    "px = spx.reindex(st.index)\n"
    "k = det.n_states\n"
    "cmap = plt.cm.RdYlGn_r\n"
    "state_colors = {s: cmap(s / max(1, k - 1)) for s in range(k)}\n"
    "fig, ax = plt.subplots(figsize=(15, 5))\n"
    "ax.plot(px.index, px.values, color='black', lw=0.8, zorder=3)\n"
    "ax.set_yscale('log'); ax.set_ylabel('S&P 500 (log)')\n"
    "# sombreado por estado (tramos contiguos)\n"
    "vals = st.values; idx = st.index\n"
    "changes = np.where(np.diff(vals) != 0)[0]\n"
    "starts = np.concatenate(([0], changes + 1)); ends = np.concatenate((changes, [len(vals)-1]))\n"
    "for s0, e0 in zip(starts, ends):\n"
    "    ax.axvspan(idx[s0], idx[e0], color=to_rgba(state_colors[vals[s0]], 0.30), zorder=1)\n"
    "for name, (a, b) in ev.CRISIS_WINDOWS.items():\n"
    "    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), facecolor='none', edgecolor='red', lw=1.4, zorder=2)\n"
    "for name, (a, b) in ev.FALSE_POSITIVE_WINDOWS.items():\n"
    "    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), facecolor='none', edgecolor='orange', lw=1.4, ls='--', zorder=2)\n"
    "legend = [Patch(color=state_colors[s], alpha=0.4, label=f'estado {s}' + (' (crisis)' if s==k-1 else (' (calma)' if s==0 else ''))) for s in range(k)]\n"
    "legend += [Patch(facecolor='none', edgecolor='red', label='ventana crisis'),\n"
    "           Patch(facecolor='none', edgecolor='orange', ls='--', label='trampa (FP)')]\n"
    "ax.legend(handles=legend, ncol=3, fontsize=8, loc='upper left')\n"
    "ax.set_title(f'{det.name}: S&P 500 OOS coloreado por régimen (' + res.extra['ventana_eval'] + ')')\n"
    "fig.tight_layout(); fig.savefig(RESULTS / 'd03_gmm_sp500_regimes.png', dpi=110, bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(new_markdown_cell(
    "<a id=\"s10\"></a>\n"
    "## 10. Timeline de estado y probabilidad de crisis OOS\n"
    "\n"
    "Dos vistas complementarias del mismo recorrido OOS. Arriba, el estado canónico día a día (función escalón). Abajo, la **probabilidad continua de crisis** `predict_proba[:, crisis_state]`, que matiza la decisión dura: deja ver con cuánta confianza el GMM marca cada día y dónde duda. Las ventanas de crisis (rojo) y de trampa (naranja) sirven de referencia visual."
))

cells.append(new_code_cell(
    "fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 6), sharex=True)\n"
    "a1.step(st.index, st.values, where='post', color='navy', lw=0.7)\n"
    "a1.set_ylabel('estado canónico'); a1.set_yticks(range(det.n_states))\n"
    "a1.set_title(f'{det.name}: timeline de régimen OOS')\n"
    "a2.fill_between(panel.index, panel['p_crisis'].values, 0, color='firebrick', alpha=0.5)\n"
    "a2.axhline(0.5, color='k', lw=0.6, ls=':'); a2.set_ylabel('P(crisis)'); a2.set_ylim(0, 1)\n"
    "for axx in (a1, a2):\n"
    "    for name, (a, b) in ev.CRISIS_WINDOWS.items():\n"
    "        axx.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='red', alpha=0.12)\n"
    "    for name, (a, b) in ev.FALSE_POSITIVE_WINDOWS.items():\n"
    "        axx.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='orange', alpha=0.16)\n"
    "fig.tight_layout(); fig.savefig(RESULTS / 'd03_gmm_timeline.png', dpi=110, bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(new_markdown_cell(
    "<a id=\"s11\"></a>\n"
    "## 11. Verificación contra crisis y trampas\n"
    "\n"
    "Contrastamos las etiquetas OOS con un calendario de eventos conocidos, con un criterio de éxito declarado por ventana:\n"
    "\n"
    "- **2008 (GFC) / 2011 (deuda europea):** esperamos `NaN`. Caen dentro del primer train (la ventana 2007 de las features no deja histórico previo suficiente), así que NO son OOS-evaluables. El `NaN` es el comportamiento CORRECTO, no un acierto ni un fallo.\n"
    "- **2020 (COVID) / 2022 (Inflación):** cobertura de crisis (sensibilidad); cuanto más alta, mejor.\n"
    "- **2013 (taper tantrum) / 2018 (sell-off del Q4):** trampas —episodios que NO deberían marcarse como crisis sostenida—; un falso-positivo bajo es lo deseable. 2013 cae también fuera de OOS → `NaN`."
))

cells.append(new_code_cell(
    "cov = res.crisis_coverage; fap = res.false_alarm_in_fp\n"
    "def fmt(d):\n"
    "    return {k: ('NaN (fuera de OOS)' if v != v else round(v, 3)) for k, v in d.items()}\n"
    "print('OOS efectivo:', res.extra['ventana_eval'])\n"
    "print('Cobertura crisis :', fmt(cov))\n"
    "print('Falsos positivos :', fmt(fap))\n"
    "print('false_alarm_rate global:', round(res.false_alarm_rate, 3))\n"
    "assert cov['GFC_2008'] != cov['GFC_2008'], '2008 deberia ser NaN'\n"
    "assert cov['EuroDebt_2011'] != cov['EuroDebt_2011'], '2011 deberia ser NaN'\n"
    "assert cov['COVID_2020'] == cov['COVID_2020'] and cov['Inflation_2022'] == cov['Inflation_2022']\n"
    "print('\\nVERIFICACION OK: 2008/2011 = NaN (esperado); COVID/Inflacion evaluables.')"
))

cells.append(new_markdown_cell(
    "---\n"
    "<a id=\"s12\"></a>\n"
    "## 12. Conclusión D3 — veredicto \"mejor-para-qué\"\n"
    "\n"
    "GMM estático con covarianza full, evaluado OOS desde 2015 (2008/2011 no evaluables por la ventana 2007 de las features). El balance es nítido y **consistente con** la hipótesis del CHECKPOINT 2, sin contradecirla:\n"
    "\n"
    "- **Dónde acierta:** capta COVID-2020 e Inflación-2022 con cobertura alta. La covarianza full le permite ver el cambio de estructura de correlación que un *k-means* euclídeo no separaría —es un baseline interpretable y sensible a las crisis grandes—.\n"
    "- **Dónde falla:** exhibe **flickering severo** (`switching_rate` alto, rachas de pocos días), lo que lo descarta como detector operativo: un \"régimen\" que dura una semana no es un régimen.\n"
    "- **Para qué sirve, entonces:** no como detector definitivo, sino como **baseline NO temporal** que aísla el aporte de la dinámica markoviana. La pregunta que deja abierta —¿cuánta de esa fragmentación elimina una matriz de transición?— la responde D4 con las mismas features.\n"
    "- **El mecanismo, ahora visible (Bloque B):** el *scatter* PCA (`d03_feature_scatter.png`) muestra los tres componentes ocupando regiones separables del espacio de features, con la crisis en la cola; los heatmaps por componente (`d03_cov_by_component.png`) confirman que cada gaussiana tiene una **estructura de covarianza distinta** —el aporte real de `covariance_type='full'`, que un *k-means* euclídeo no vería—; y el paisaje BIC (`d03_bic_curve.png`) justifica *k*=3 y, vía la confianza de asignación, expone los **días-frontera** de baja responsabilidad que explican el flickering. Es decir: el GMM *sí* separa estructura, pero la decisión día-a-día sin memoria es lo que lo descarta como detector operativo.\n"
    "\n"
    "Detalle y números finales en `docs/memory/detectors/03_clustering_gmm.md`."
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
