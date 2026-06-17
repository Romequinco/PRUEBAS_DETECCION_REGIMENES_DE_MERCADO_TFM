# D9 — `jump_model` (FASE 3, Tanda 4 · exploratoria) · Familia F2↔F3 (clustering con persistencia)

> Statistical Jump Model (Nystrup et al.): clustering de estados con **penalización
> de salto** λ → histéresis "aprendida", persistencia, online. Rival honesto de D3
> (GMM sin persistencia) y de D12 (AE). Código: `detectors/jump_model.py` · Notebook:
> `notebooks/09_jump_model.ipynb` · Métricas: `results/metrics_09_jump_model.csv`.

## Implementado

**Modelo.** Librería `jumpmodels` v0.1.1 (`JumpModel`, SJM discreto), `n_states=2`,
**jump_penalty λ=50**. El SJM minimiza distancia intra-cluster + λ·(nº de saltos de
estado) → penaliza cambiar de régimen, induciendo persistencia sin la matriz de
transición de un HMM. `StandardScaler` ajustado **solo con el train** dentro de `fit`
(causal, re-fit por fold) porque 3 de las 15 features (corr, drawdown, momentum) tienen
escala distinta a los z. Mismas 15 features causales y ventana 2007+ que **D3** (es su
rival directo). Bibliografía: `hmm_nystrup2020, hmm_nystrup2017, clust_munnix2012`.

**Causalidad.** `jumpmodels` expone `predict_online` causal (la etiqueta de la fila i
usa solo filas < i). Verificado en el notebook: **0/120 etiquetas del bloque cambian
al añadir futuro**. market_returns a walk_forward Y evaluate.

**Orientación (Arreglo 4).** Verificada en walk-forward: crisis (estado 1) = vol 0.0218
vs calma 0.0089, retorno negativo → **no invertido**, sin warning de fallback.

## Descubierto

**ventana_eval: 2015-09 → 2026-06 (n=2649).** 15 features de 2007 → 2008/2011 en el
train (NaN OOS), como D3.

**La hipótesis anti-flickering se cumple ROTUNDAMENTE.** Comparado con su rival D3
(GMM clustering sin persistencia):

| | switching_rate | duración media | cov COVID | cov Inflation |
|---|---:|---:|---:|---:|
| D3 clustering_gmm_k3 | 0.126 | 7.9 d | 0.96 | 0.87 |
| **D9 jump_model (λ=50)** | **0.005** | **176.6 d** | 0.72 | 0.17 |

La penalización de salto reduce el flickering **~24×** (0.126→0.005) y multiplica la
duración de los episodios ×22 — exactamente lo que prometía Nystrup: histéresis
aprendida sin dinámica de Markov explícita.

**Pero hay un coste claro de sensibilidad.** Con λ=50 (alto), D9 se vuelve tan
persistente que **pierde cobertura de las crisis más lentas/menos extremas**: COVID
72% (vs 96% de D3) e **Inflación 2022 solo 17%** (vs 87% de D3). El bear market lento
de 2022 no genera un salto suficientemente nítido como para vencer la penalización, así
que D9 se queda en "calma" gran parte de él. fa_2018 = 0% (no se dispara en la trampa),
false_alarm_rate 0.62, label_stability ~1.0.

**Lectura honesta del trade-off.** D9 NO domina a D3: cambia muchísimo flickering por
cobertura. Es el extremo "ultra-persistente" del eje clustering. Su λ es un mando
directo sobre ese trade-off (λ bajo → se parece a D3; λ alto → ultra-persistente y
ciego a estrés suave). Para el TFM es valioso como demostración de que la persistencia
se puede imponer explícitamente, y como punto de comparación frente al GMM (sin
persistencia) y al AE (D12).

## Hipótesis del CHECKPOINT 2 para D9 — veredicto

> *"Histéresis aprendida: estados persistentes, online, menos flickering, drawdowns
> suaves; rival honesto del AE+clustering en muestra pequeña."*

**Se cumple en lo esencial, con matiz.** Persistencia/online/menos flickering: SÍ,
rotundo (switching 0.005). "Drawdowns suaves": a costa de **perder cobertura de crisis
lentas** (Inflación 17%) — la persistencia fuerte es un arma de doble filo. Como rival
del AE (D12): D9 es mucho más limpio (switching 0.005 vs 0.287 del AE) — el jump model
gana de calle al deep como forma de imponer persistencia con pocos datos.

## Fricción con el núcleo

Ninguna; no se tocó `src/`. El Arreglo 4 (vol-primario) orientó D9 sin parche
(candidato a inversión por separar en features; no la hubo). Observación menor (ya
conocida): `walk_forward` tiene `min_train=252*5` que domina sobre `train_size`
pequeños — irrelevante aquí (train=252×8).
