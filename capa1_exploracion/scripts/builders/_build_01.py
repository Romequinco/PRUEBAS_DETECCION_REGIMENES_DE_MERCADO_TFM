"""Construye y EJECUTA notebooks/01_rule_vix_threshold.ipynb (D1).

Patrón idéntico al de _build_02.py: nbformat para montar las celdas +
ExecutePreprocessor (kernel python3) para ejecutarlo. Tras ejecutar, comprueba 0
errores leyendo los outputs. Figuras -> results/.

D1 usa histórico largo (VIX desde 1990) pero es barato de ejecutar.
"""
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[2]
NB_PATH = ROOT / "notebooks" / "01_rule_vix_threshold.ipynb"

cells = []

cells.append(new_markdown_cell(
    "# 01 — D1 `rule_vix_threshold` (Reglas / Umbrales)\n\n"
    "Primer detector del banco y **baseline** deliberadamente simple: si los "
    "modelos complejos no le ganan a una regla sobre el VIX, habrá que "
    "preguntarse para qué sirven. Es una regla reactiva y **causal nativa** sobre "
    "el NIVEL del VIX (índice de volatilidad implícita del S&P 500, el "
    "\"termómetro del miedo\" del equity) con dos mecanismos anti-ruido:\n\n"
    "- **histéresis**: una banda muerta entre el umbral de **entrada** τ_in y el "
    "de **salida** τ_out (< τ_in), para no entrar y salir de crisis con cada "
    "oscilación —igual que un termostato que enciende y apaga a temperaturas "
    "distintas—.\n"
    "- **dwell-time**: permanencia mínima en crisis antes de poder salir.\n\n"
    "Juntos matan el *flickering* (parpadeo) de un umbral simple. 2 estados: "
    "0=calma, 1=crisis.\n\n"
    "**Política de ventana**: el VIX existe desde **1990**, así que la feature se "
    "construye desde el panel crudo largo (no desde `features.parquet`, atado a "
    "2007 por HYG). Así el walk-forward cubre **2008 y 2011** out-of-sample "
    "—precisamente las dos crisis que D2 no podrá juzgar—.\n\n"
    "**Hipótesis CHECKPOINT 2 (D1)**: *capta las 4 crisis y, por reactividad, "
    "probablemente 2013 y 2018 que el HMM falló; su talón de Aquiles son los "
    "falsos positivos ante picos efímeros si NO lleva histéresis*. La "
    "contrastamos al final, sin adelantar veredicto."
))

cells.append(new_markdown_cell(
    "### Índice\n\n"
    "Hoja de ruta del notebook —del dato crudo al veredicto— para navegar las "
    "secciones sin perder el hilo argumental:\n\n"
    "1. **Feature causal de VIX** (histórico largo) + retorno del S&P 500.\n"
    "2. **El detector D1 y sus umbrales causales** (τ_in / τ_out, dwell-time).\n"
    "3. **Walk-forward causal** (cubre 2008 y 2011 OOS).\n"
    "4. **Evaluación estandarizada** y fila de métricas (CSV `metrics_01_*`).\n"
    "5. **S&P 500 coloreado por régimen** — primera lectura visual.\n"
    "6. **Señal de crisis y timeline** — el detector por dentro.\n"
    "7. **Distribución del nivel de VIX por régimen** — *la figura estrella*: "
    "separación calma vs crisis.\n"
    "8. **Panel de histéresis temporal** — el mecanismo de banda muerta + "
    "dwell-time, ampliado sobre la GFC.\n"
    "9. **Duración de rachas** — persistencia de los episodios (anti-flickering).\n"
    "10. **Verificación explícita** contra las 4 crisis y las 2 trampas.\n"
    "11. **Tabla de cobertura** por ventana (crisis vs trampa).\n"
    "12. **Conclusión** y contraste con la hipótesis del CHECKPOINT 2.\n\n"
    "Las figuras se guardan en `results/` con prefijo `d1_` y se embeben en el "
    "informe LaTeX de la Capa 1."
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
    "from src import features as ft, evaluation as ev, viz\n"
    "from detectors.rule_vix_threshold import RuleVixThreshold\n\n"
    "viz.use_house_style()   # paleta y estilo homogéneos con el resto del banco\n"
    "raw = pd.read_parquet(ROOT / 'data' / 'raw' / 'raw_panel.parquet')\n"
    "print('Panel crudo:', raw.shape, '|', raw.index.min().date(), '->', raw.index.max().date())\n"
    "print('VIX primer dato válido:', raw['VIX'].first_valid_index().date())"
))

cells.append(new_markdown_cell(
    "## 1. Feature causal de VIX (histórico largo) + retorno del S&P 500\n\n"
    "Construimos la única entrada del detector y el retorno auxiliar que ordena "
    "los estados.\n\n"
    "- `VIX_level_z`: nivel del VIX z-scoreado de forma **causal** (expanding, "
    "`features.causal_zscore`): en *t* solo usa media/std de datos ≤ *t*, nunca la "
    "muestra completa.\n"
    "- `SP500_ret`: retorno log del S&P 500, reindexado a la feature. Sirve para "
    "(a) el **etiquetado económico** de estados —deducir cuál de los 2 estados es "
    "\"crisis\" por su peor retorno realizado, en vez de imponerlo a mano— y (b) "
    "como `market_returns` del evaluador.\n\n"
    "El panel arranca cuando `VIX_level_z` tiene suficientes observaciones "
    "(`min_periods`), es decir ~1990."
))

cells.append(new_code_cell(
    "vix = raw['VIX'].dropna()                       # desde 1990\n"
    "vix_z = ft.causal_zscore(vix.rename('VIX_level'))\n"
    "spx_ret = np.log(raw['SP500'] / raw['SP500'].shift(1)).rename('SP500_ret')\n\n"
    "X = pd.DataFrame({\n"
    "    'VIX_level_z': vix_z,\n"
    "    'SP500_ret': spx_ret.reindex(vix_z.index),\n"
    "}).dropna()\n"
    "X.index = pd.to_datetime(X.index); X = X.sort_index()\n"
    "print('X (histórico largo):', X.shape, '|', X.index.min().date(), '->', X.index.max().date())\n"
    "X.head(3)"
))

cells.append(new_markdown_cell(
    "**Verificación de causalidad** de la feature, no como formalismo sino como "
    "salvaguarda contra el look-ahead: truncar la entrada en una fecha y "
    "recomputar el z-score debe dar exactamente los mismos valores que "
    "computarlo sobre la serie completa y recortar. Si el pasado no cambia al "
    "añadir futuro, la feature es causal."
))

cells.append(new_code_cell(
    "cut = '2015-01-01'\n"
    "full = ft.causal_zscore(vix.rename('VIX_level'))\n"
    "trunc = ft.causal_zscore(vix.loc[:cut].rename('VIX_level'))\n"
    "idx = trunc.index.intersection(full.index); idx = idx[idx <= pd.Timestamp(cut)]\n"
    "max_diff = float((full.loc[idx] - trunc.loc[idx]).abs().max())\n"
    "print(f'max_abs_diff hasta {cut}: {max_diff:.2e}  ->  causal_ok = {max_diff < 1e-9}')\n"
    "assert max_diff < 1e-9, 'La feature VIX_level_z NO es causal'"
))

cells.append(new_markdown_cell(
    "## 2. El detector D1 y sus umbrales causales\n\n"
    "El corazón del detector son dos umbrales que NUNCA se fijan a ojo ni usando "
    "toda la muestra, sino como **percentiles del train** (causal):\n\n"
    "`RuleVixThreshold(q_in=0.90, q_out=0.70, min_dwell=5)`:\n"
    "- **τ_in** = percentil 90 del VIX z en el train → umbral de ENTRADA.\n"
    "- **τ_out** = percentil 70 (< τ_in) → umbral de SALIDA (la banda muerta de "
    "la histéresis).\n"
    "- **min_dwell** = 5 días mínimos en crisis antes de poder salir.\n\n"
    "Aquí se ajusta sobre TODO el histórico **solo para inspeccionar** los "
    "umbrales; la evaluación honesta es el walk-forward de la sección 3."
))

cells.append(new_code_cell(
    "det0 = RuleVixThreshold(q_in=0.90, q_out=0.70, min_dwell=5).fit(X)\n"
    "print(f'τ_in  (q=0.90) = {det0._tau_in:.3f}  (en unidades de VIX z)')\n"
    "print(f'τ_out (q=0.70) = {det0._tau_out:.3f}')\n"
    "print('crisis_state canónico =', det0.crisis_state, '| n_states =', det0.n_states)\n"
    "print('bibliografía:', det0.bibliography)\n"
    "states_is = pd.Series(det0.predict(X), index=X.index)\n"
    "print('% días crisis in-sample:', round(float((states_is == det0.crisis_state).mean()), 3))"
))

cells.append(new_markdown_cell(
    "## 3. Walk-forward causal (cubre 2008 y 2011 OOS)\n\n"
    "Esta es la evaluación honesta. El **walk-forward** reentrena el detector en "
    "ventanas crecientes y predice solo el siguiente bloque que aún no ha visto, "
    "avanzando en el tiempo: cada predicción es *out-of-sample* (OOS) y nunca usa "
    "datos posteriores a la fecha que clasifica. Con histórico desde 1990 y "
    "`train_size` de ~8 años, el primer bloque OOS cae a finales de los 90 y el "
    "panel OOS abarca **1998–2026**, incluyendo las 4 crisis y las 2 trampas."
))

cells.append(new_code_cell(
    "factory = lambda: RuleVixThreshold(q_in=0.90, q_out=0.70, min_dwell=5)\n"
    "panel = ev.walk_forward(factory, X, train_size=252*8, step=21, expanding=True)\n"
    "print('Panel OOS:', panel.shape, '|', panel.index.min().date(), '->', panel.index.max().date())\n"
    "panel.head(3)"
))

cells.append(new_markdown_cell(
    "## 4. Evaluación estandarizada y fila de métricas\n\n"
    "`ev.evaluate` resume el panel OOS en una fila comparable entre todos los "
    "detectores del banco: **cobertura** por crisis (% de días de cada ventana "
    "marcados crisis), **falsas alarmas** (global y dentro de las trampas), "
    "**lead/lag** vs el suelo del drawdown (signo negativo = la señal precede al "
    "fondo), **switching rate** y persistencia (duración media de régimen), y "
    "**estabilidad** de las etiquetas al reentrenar. `market_returns` = retorno "
    "log del S&P 500 reindexado al panel OOS."
))

cells.append(new_code_cell(
    "mr = X['SP500_ret'].reindex(panel.index)\n"
    "res = ev.evaluate(det0, panel, market_returns=mr, X_full=X)\n"
    "row = ev.results_table([res])\n"
    "out_csv = RESULTS / 'metrics_01_rule_vix_threshold.csv'\n"
    "row.to_csv(out_csv, index=False)\n"
    "print('ventana_eval:', res.extra['ventana_eval'])\n"
    "print('Guardado:', out_csv)\n"
    "row.T"
))

cells.append(new_markdown_cell(
    "## 5. Visualización — S&P 500 coloreado por régimen\n\n"
    "Primera lectura visual: ¿coinciden los tramos que D1 marca como crisis con "
    "las caídas reales? S&P 500 (log) con los días OOS clasificados como "
    "**crisis** sombreados en rojo; bandas de las ventanas de crisis conocidas "
    "(rojo claro) y de los falsos positivos / trampas (naranja)."
))

cells.append(new_code_cell(
    "spx = raw['SP500'].reindex(panel.index)\n"
    "is_crisis = (panel['state'] == det0.crisis_state)\n"
    "fig, ax = plt.subplots(figsize=(15, 5))\n"
    "ax.plot(spx.index, spx, color='black', lw=0.7, zorder=3)\n"
    "ax.set_yscale('log'); ax.set_ylabel('S&P 500 (log)')\n"
    "# Sombrear días OOS en crisis\n"
    "ymin, ymax = ax.get_ylim()\n"
    "ax.fill_between(panel.index, ymin, ymax, where=is_crisis.values, color='crimson',\n"
    "                alpha=0.25, step='mid', zorder=1, label='D1: crisis (OOS)')\n"
    "ax.set_ylim(ymin, ymax)\n"
    "for a, b in ev.CRISIS_WINDOWS.values():\n"
    "    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='red', alpha=0.10, zorder=0)\n"
    "for a, b in ev.FALSE_POSITIVE_WINDOWS.values():\n"
    "    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='orange', alpha=0.18, zorder=0)\n"
    "handles = [Patch(color='crimson', alpha=0.25, label='D1 crisis (OOS)'),\n"
    "           Patch(color='red', alpha=0.10, label='Ventanas crisis'),\n"
    "           Patch(color='orange', alpha=0.18, label='Trampas 2013/2018')]\n"
    "ax.legend(handles=handles, loc='upper left', fontsize=8)\n"
    "ax.set_title('D1 rule_vix_threshold — S&P 500 coloreado por régimen (out-of-sample)')\n"
    "fig.tight_layout(); fig.savefig(RESULTS / 'd1_regime_sp500.png', dpi=110, bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(new_markdown_cell(
    "## 6. Probabilidad / serie de crisis y timeline de regímenes\n\n"
    "Para ver el detector \"por dentro\": cómo la señal cruza la histéresis. "
    "Arriba: `p_crisis` OOS —una regla dura emite 0/1, no una probabilidad suave— "
    "junto al VIX z y sus umbrales τ_in/τ_out. Abajo: timeline de régimen (banda "
    "continua roja = crisis), que evidencia si los episodios son largos y "
    "estables o parpadean."
))

cells.append(new_code_cell(
    "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6), sharex=True,\n"
    "                               gridspec_kw={'height_ratios': [3, 1]})\n"
    "vz = X['VIX_level_z'].reindex(panel.index)\n"
    "ax1.plot(vz.index, vz, color='steelblue', lw=0.7, label='VIX_level_z')\n"
    "ax1.axhline(det0._tau_in, color='crimson', ls='--', lw=1, label=f'τ_in={det0._tau_in:.2f}')\n"
    "ax1.axhline(det0._tau_out, color='darkorange', ls='--', lw=1, label=f'τ_out={det0._tau_out:.2f}')\n"
    "ax1.fill_between(panel.index, 0, panel['p_crisis'].values * vz.max(),\n"
    "                 color='crimson', alpha=0.12, step='mid', label='p_crisis (escalada)')\n"
    "ax1.set_ylabel('VIX z / señal'); ax1.legend(loc='upper left', fontsize=8)\n"
    "ax1.set_title('Señal de crisis vs VIX z e histéresis (τ_in / τ_out)')\n"
    "ax2.fill_between(panel.index, 0, 1, where=is_crisis.values, color='crimson', alpha=0.6, step='mid')\n"
    "ax2.set_yticks([]); ax2.set_ylabel('régimen'); ax2.set_title('Timeline (rojo = crisis)')\n"
    "fig.tight_layout(); fig.savefig(RESULTS / 'd1_prob_timeline.png', dpi=110, bbox_inches='tight')\n"
    "plt.show()"
))

# ------------------------------------------------------------------ #
# 7. Distribución del nivel de VIX por régimen (FIGURA ESTRELLA)
# ------------------------------------------------------------------ #
cells.append(new_markdown_cell(
    "## 7. Distribución del nivel de VIX por régimen (la separación, de un vistazo)\n\n"
    "Las secciones anteriores muestran *cuándo* D1 marca crisis; esta muestra "
    "**por qué puede hacerlo**. Si el nivel del VIX es realmente un termómetro del "
    "miedo, su distribución condicionada al régimen debería estar **claramente "
    "desplazada**: masa baja y compacta en calma, cola alta y dispersa en crisis. "
    "Es la evidencia más directa de que la *única* feature del detector tiene poder "
    "discriminante —y, en el fondo, la justificación de todo el baseline—.\n\n"
    "El violín separa la densidad del `VIX_level_z` (z-score causal) en los dos "
    "estados OOS. Superponemos τ_in y τ_out: la banda muerta de la histéresis cae "
    "justo en la **zona de solape** entre las dos distribuciones, que es "
    "exactamente donde un umbral simple parpadearía. Una separación amplia con "
    "solape estrecho es la firma de un cribado limpio; un solape ancho avisaría de "
    "que el VIX por sí solo no basta (preludio de por qué hará falta D2)."
))

cells.append(new_code_cell(
    "vix_oos = X['VIX_level_z'].reindex(panel.index)\n"
    "states_oos = panel['state'].astype(int)\n"
    "m = vix_oos.notna() & states_oos.notna()\n"
    "fig, ax = plt.subplots(figsize=(8.5, 5))\n"
    "viz.plot_distribution_by_regime(\n"
    "    vix_oos[m], states_oos[m], crisis_state=det0.crisis_state,\n"
    "    labels={0: 'calma', 1: 'crisis'}, kind='violin', ax=ax,\n"
    "    xlabel='VIX nivel (z-score causal)',\n"
    "    title='D1 — Distribución del nivel de VIX por régimen (OOS)')\n"
    "ax.axhline(det0._tau_in, color='crimson', ls='--', lw=1.1, label=f'τ_in={det0._tau_in:.2f}')\n"
    "ax.axhline(det0._tau_out, color='darkorange', ls='--', lw=1.1, label=f'τ_out={det0._tau_out:.2f}')\n"
    "ax.axhspan(det0._tau_out, det0._tau_in, color='gold', alpha=0.12)\n"
    "ax.legend(loc='upper left', fontsize=8)\n"
    "fig.tight_layout(); fig.savefig(RESULTS / 'd1_vix_dist_by_regime.png', dpi=110, bbox_inches='tight')\n"
    "plt.show()\n"
    "# Resumen numérico de la separación (mediana por régimen)\n"
    "med_calma = float(np.nanmedian(vix_oos[m][states_oos[m] == 0]))\n"
    "med_crisis = float(np.nanmedian(vix_oos[m][states_oos[m] == det0.crisis_state]))\n"
    "print(f'mediana VIX z  calma={med_calma:.2f}  crisis={med_crisis:.2f}  '\n"
    "      f'(salto={med_crisis - med_calma:.2f} desv. típicas)')"
))

# ------------------------------------------------------------------ #
# 8. Panel de histéresis temporal (el MECANISMO, ampliado sobre la GFC)
# ------------------------------------------------------------------ #
cells.append(new_markdown_cell(
    "## 8. Panel de histéresis temporal — el mecanismo, en detalle\n\n"
    "La sección 6 mostró la señal a lo largo de tres décadas; aquí hacemos "
    "**zoom sobre la Gran Crisis Financiera (2007–2009)** para ver el autómata "
    "funcionando día a día. Es el lugar donde la histéresis se gana el sueldo:\n\n"
    "- El VIX z (azul) sube y cruza **τ_in** (rojo) → el detector **entra** en "
    "crisis (sombreado rojo).\n"
    "- Mientras oscila dentro de la **banda muerta** (franja dorada entre τ_out y "
    "τ_in), el régimen **no cambia**: un umbral simple parpadearía aquí; D1 no.\n"
    "- Solo cuando el VIX baja de **τ_out** (naranja) *y* se ha cumplido el "
    "`min_dwell`, el detector **sale** a calma.\n\n"
    "Esa asimetría entre entrada y salida —subir por una puerta alta y bajar por "
    "otra más baja— es lo que produce episodios largos y limpios en vez de una "
    "metralleta de señales. La figura aísla este mecanismo en su propio panel, "
    "complementando la vista panorámica de la sección 6."
))

cells.append(new_code_cell(
    "# Ventana de zoom: la GFC. Si por la cobertura OOS no estuviese disponible,\n"
    "# se recurre al tramo completo (guarda defensiva, no debería activarse).\n"
    "win = slice('2007-06-01', '2009-12-31')\n"
    "vz = X['VIX_level_z'].reindex(panel.index)\n"
    "seg = vz.loc[win].dropna()\n"
    "if len(seg) < 30:\n"
    "    seg = vz.dropna()\n"
    "seg_state = (panel['state'].reindex(seg.index) == det0.crisis_state)\n"
    "fig, ax = plt.subplots(figsize=(14, 5))\n"
    "ax.plot(seg.index, seg.values, color='steelblue', lw=1.0, zorder=3, label='VIX nivel (z)')\n"
    "ax.axhspan(det0._tau_out, det0._tau_in, color='gold', alpha=0.20, zorder=0,\n"
    "           label='banda muerta (τ_out–τ_in)')\n"
    "ax.axhline(det0._tau_in, color='crimson', ls='--', lw=1.2, label=f'τ_in={det0._tau_in:.2f} (ENTRADA)')\n"
    "ax.axhline(det0._tau_out, color='darkorange', ls='--', lw=1.2, label=f'τ_out={det0._tau_out:.2f} (SALIDA)')\n"
    "ymin, ymax = ax.get_ylim()\n"
    "ax.fill_between(seg.index, ymin, ymax, where=seg_state.values, color='crimson',\n"
    "                alpha=0.12, step='mid', zorder=1, label='régimen crisis (D1)')\n"
    "ax.set_ylim(ymin, ymax)\n"
    "ax.set_ylabel('VIX nivel (z-score causal)')\n"
    "ax.set_title('D1 — Mecanismo de histéresis durante la GFC (2007–2009): banda muerta + dwell-time')\n"
    "ax.legend(loc='upper left', fontsize=8, ncol=2)\n"
    "fig.tight_layout(); fig.savefig(RESULTS / 'd1_hysteresis_panel.png', dpi=110, bbox_inches='tight')\n"
    "plt.show()"
))

# ------------------------------------------------------------------ #
# 9. Duración de rachas (persistencia / anti-flickering)
# ------------------------------------------------------------------ #
cells.append(new_markdown_cell(
    "## 9. Duración de las rachas — ¿episodios o parpadeo?\n\n"
    "El argumento de venta de la histéresis + dwell-time es que produce regímenes "
    "**persistentes**. Lo cuantificamos con el histograma de duraciones de cada "
    "racha consecutiva, separado por estado. La hipótesis a refutar es la del "
    "*flickering*: si el detector parpadease, la masa se concentraría en rachas de "
    "1–3 días. Lo que esperamos ver en D1 es lo contrario —una cola de episodios "
    "de crisis de decenas de días, coherente con la duración media de régimen "
    "(≈75 días) que reporta la métrica—. Las rachas de crisis cortas serían la "
    "huella dactilar de los falsos positivos efímeros que la histéresis pretende "
    "matar."
))

cells.append(new_code_cell(
    "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
    "viz.plot_duration_histogram(\n"
    "    panel['state'].astype(int), n_states=det0.n_states, ax=ax,\n"
    "    title='D1 — Duración de episodios por régimen (OOS): persistencia vs flickering')\n"
    "fig.tight_layout(); fig.savefig(RESULTS / 'd1_durations.png', dpi=110, bbox_inches='tight')\n"
    "plt.show()\n"
    "# Estadísticos de persistencia por estado\n"
    "dur = viz.episode_durations(panel['state'].astype(int), det0.n_states)\n"
    "for k, lab in {0: 'calma', det0.crisis_state: 'crisis'}.items():\n"
    "    d = dur.get(k, [])\n"
    "    if d:\n"
    "        print(f'{lab:7s}: n_rachas={len(d):3d}  mediana={np.median(d):5.1f}d  '\n"
    "              f'máx={max(d):4d}d  rachas≤3d={sum(1 for x in d if x <= 3)}')"
))

cells.append(new_markdown_cell(
    "## 10. Verificación explícita contra crisis y trampas\n\n"
    "Bajamos del resumen a la prueba episodio por episodio. **Cobertura** (% de "
    "días marcados crisis) en cada ventana de crisis conocida y activación en "
    "cada trampa (taper tantrum 2013, sell-off Q4 2018). Para crisis: cuanto más "
    "alta, mejor (sensibilidad). Para trampas: cuanto más baja, mejor "
    "(**especificidad** = no encenderse cuando no toca)."
))

cells.append(new_code_cell(
    "states_oos = panel['state']\n"
    "cov = ev.crisis_coverage(states_oos, det0.crisis_state)\n"
    "fa  = ev.false_alarm_in_windows(states_oos, det0.crisis_state)\n"
    "print('=== COBERTURA EN CRISIS (alto = bueno) ===')\n"
    "for k, v in cov.items():\n"
    "    flag = 'sin OOS' if (v != v) else ('OK' if v >= 0.5 else 'BAJA')\n"
    "    print(f'  {k:16s}: {v:.2%}  [{flag}]' if v == v else f'  {k:16s}:   NaN  [{flag}]')\n"
    "print('\\n=== ACTIVACIÓN EN TRAMPAS (bajo = bueno; el HMM gaussiano falla aquí) ===')\n"
    "for k, v in fa.items():\n"
    "    print(f'  {k:16s}: {v:.2%}' if v == v else f'  {k:16s}:   NaN')\n"
    "print(f'\\nfalse_alarm_rate global (fuera de crisis): {res.false_alarm_rate:.2%}')\n"
    "print(f'switching_rate: {res.switching_rate:.4f}  |  duración media régimen: {res.mean_regime_duration:.1f} días')"
))

cells.append(new_code_cell(
    "capto_2013 = (fa.get('TaperTantrum_2013', float('nan')) or 0) > 0.05\n"
    "capto_2018 = (fa.get('Selloff_Q4_2018', float('nan')) or 0) > 0.05\n"
    "crisis_ok = {k: (v is not None and v == v and v >= 0.5) for k, v in cov.items()}\n"
    "print('Crisis cubiertas (≥50% días crisis):')\n"
    "for k, ok in crisis_ok.items():\n"
    "    print(f'   {k:16s}: {\"SI\" if ok else (\"sin OOS\" if cov[k] != cov[k] else \"parcial\")}')\n"
    "print(f'\\n¿Se activó en 2013 (taper)? {\"SI\" if capto_2013 else \"no\"}  '\n"
    "      f'(act={fa.get(\"TaperTantrum_2013\", float(\"nan\")):.1%})')\n"
    "print(f'¿Se activó en 2018 (Q4)?     {\"SI\" if capto_2018 else \"no\"}  '\n"
    "      f'(act={fa.get(\"Selloff_Q4_2018\", float(\"nan\")):.1%})')\n"
    "print('\\nNota: en 2013/2018 \"activarse\" es señal DESEABLE de reactividad (fueron '\n"
    "      'episodios reales de estrés) pero cuenta como falso positivo en el marco '\n"
    "      'estricto, que solo considera crisis sistémicas a 2008/2011/2020/2022.')"
))

# ------------------------------------------------------------------ #
# 11. Tabla-figura de cobertura por ventana (crisis vs trampa)
# ------------------------------------------------------------------ #
cells.append(new_markdown_cell(
    "## 11. Tabla de cobertura por ventana — el veredicto numérico, ordenado\n\n"
    "Para cerrar la verificación, condensamos los dos diccionarios anteriores "
    "(`crisis_coverage` y `false_alarm_in_windows`) en una **única tabla-figura** "
    "embebible en el informe LaTeX. Cada fila es una ventana etiquetada por su "
    "**tipo**: en las de tipo *crisis* interesa una cobertura **alta** "
    "(sensibilidad); en las de tipo *trampa* —los episodios de estrés real pero no "
    "sistémico de 2013 y 2018— interesa una activación **baja** (especificidad "
    "bajo el marco estricto). Leerlas juntas evita la falacia de celebrar la "
    "sensibilidad sin mirar el coste en falsos positivos."
))

cells.append(new_code_cell(
    "rows = []\n"
    "for k, v in cov.items():\n"
    "    rows.append({'ventana': k, 'tipo': 'crisis',\n"
    "                 'cobertura/activación': v, 'lectura': 'alto = bueno (sensibilidad)'})\n"
    "for k, v in fa.items():\n"
    "    rows.append({'ventana': k, 'tipo': 'trampa',\n"
    "                 'cobertura/activación': v, 'lectura': 'bajo = bueno (especificidad)'})\n"
    "tab = pd.DataFrame(rows).set_index('ventana')\n"
    "fig = viz.render_table_figure(\n"
    "    tab, title='D1 — Cobertura por crisis y activación en trampas (OOS)',\n"
    "    highlight_cols=['cobertura/activación'],\n"
    "    fmt={'cobertura/activación': '{:.1%}'})\n"
    "fig.savefig(RESULTS / 'd1_coverage_table.png', dpi=110, bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(new_markdown_cell(
    "## 12. Conclusión y contraste con la hipótesis del CP2\n\n"
    "La hipótesis del CHECKPOINT 2 para D1 era: *capta las 4 crisis y "
    "probablemente 2013/2018 que el HMM falló; falla en falsos positivos sin "
    "histéresis*. Los números OOS de arriba la **cumplen en lo esencial, con "
    "matices cuantificados** —que no la \"confirman\" en sentido fuerte: n≈4 "
    "crisis, sin tests de significancia, lead/lag censurado por la ventana—:\n\n"
    "- **Crisis**: 3 de 4 captadas con fuerza (GFC 2008 ≈93.8%, COVID 2020 "
    "90.0%, EuroDebt 2011 ≈63.5%); la 4ª, **Inflación 2022 (≈34.9%)**, queda "
    "infra-detectada. Es **consistente con** el límite real de un detector "
    "univariante de VIX: 2022 fue un bear market lento de tipos con VIX moderado, "
    "no un shock de miedo. No es un fallo de implementación, sino el hueco que "
    "motiva D2.\n"
    "- **Trampas y falsos positivos**: con histéresis NO se dispara de forma "
    "sostenida (2013 ≈0.0%, 2018 ≈6.3%), y `switching_rate≈0.013` con "
    "**duración media de régimen ≈75 días** describen episodios largos, no "
    "parpadeo. Esto **no contradice** la tesis de que la banda muerta + "
    "dwell-time son necesarios: justamente los suprimen.\n"
    "- El `false_alarm_rate` global elevado (≈0.70) debe leerse con cautela: las "
    "ventanas oficiales de crisis son estrechas y hay estrés real no catalogado "
    "(LTCM, DotCom, 2015-16, SVB) que el marco cuenta como falsa alarma sin serlo "
    "del todo.\n\n"
    "**Mejor-para-qué**: D1 queda como el baseline de *miedo limpio y "
    "persistente* —fuerte y de entrada temprana (lead/lag negativo en los 4 "
    "eventos) cuando la crisis trae pico de volatilidad; débil ante estrés lento "
    "sin VIX—. Detalle y discusión en "
    "`docs/memory/detectors/01_rule_vix_threshold.md`."
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
