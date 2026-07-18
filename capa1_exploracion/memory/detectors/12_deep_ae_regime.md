# D12 — `deep_ae_regime` (FASE 3, Tanda 4 · exploratoria) · Familia F7 (redes neuronales)

> Autoencoder ligero → GMM sobre el latente, como **contraste ablativo honesto** frente
> al baseline lineal PCA→GMM. Objetivo: ¿aporta la no linealidad del AE algo con ~4
> crisis? Código: `detectors/deep_ae_regime.py` (`DeepAERegime` + `PCAGMMBaseline`) ·
> Notebook: `notebooks/12_deep_ae_regime.ipynb` · Métricas:
> `results/metrics_12_deep_ae_regime.csv` (2 filas: AE y baseline PCA).

## Implementado

**Modelo.** Autoencoder denso PyTorch (15→8→2→8→15, ReLU + dropout 0.10 + weight-decay
1e-3, 40 épocas, Adam, semillas fijas) que comprime las 15 features causales a un
**latente 2D**; encima, `GaussianMixture` full con **K=3** (mismo esquema que D3 pero no
lineal). Estandarización **causal por fold** (μ/σ del train). Se expone también
`reconstruction_error()` como score de anomalía (complemento, no principal).
Bibliografía: `nn_kingma2014, nn_akioyamen2021, nn_bucci2021, lopezdeprado2018`.

**Baseline ablativo.** Clase hermana `PCAGMMBaseline` (PCA 2D → GMM K=3) en el mismo
fichero, pasa por el MISMO `walk_forward` → comparación limpia AE-no-lineal vs
PCA-lineal con idéntica dim latente, K y ventana.

**Causalidad (con nota — se corrigió una aserción).** El encoder es puntual (eval() +
no_grad + scaler congelado del train). La verificación inicial del notebook usaba
`assert maxdiff_latente < 1e-9`, que **fallaba por ruido de coma flotante de torch**
(BLAS sobre tensores de distinto tamaño en float32 da ~2.4e-7), NO por look-ahead. Se
corrigió la aserción al criterio REAL de causalidad: **0/247 estados del bloque cambian
al ocultar el futuro** (y maxdiff latente < 1e-4, tolerancia FP). Confirmado causal.
market_returns a walk_forward Y evaluate.

**Orientación (Arreglo 4).** Verificada en walk-forward: crisis (estado 2) vol 0.0275
vs calma 0.0070; PCA igual (crisis vol 0.0524 vs 0.0066). No invertido, sin fallback.

## Descubierto — RESULTADO NEGATIVO (esperado y válido)

**ventana_eval: 2015-09 → 2026-06 (n=2649)** (15 features de 2007 → 2008/2011 NaN OOS).

**Contraste ablativo AE→GMM vs PCA→GMM (mismo K, dim, ventana):**

| | cov COVID | cov Inflation | fa 2018 | false_alarm_rate | switching | dur |
|---|---:|---:|---:|---:|---:|---:|
| **D12 AE→GMM** | 0.54 | 0.10 | 0.15 | 0.60 | **0.287** | 3.5 d |
| PCA→GMM (lineal) | 0.62 | 0.005 | 0.00 | 0.14 | 0.091 | 10.9 d |

**El AE NO mejora al baseline lineal — al contrario, lo empeora.** La no linealidad
**añade flickering** (switching 0.287 vs 0.091, ~3×) **y falsas alarmas** (far 0.60 vs
0.14) **sin ganar cobertura** (COVID 0.54 vs 0.62). Es el resultado que anticipaba el
CHECKPOINT 2: con ~4 crisis reales no hay señal suficiente para que la capacidad extra
del AE generalice; solo ajusta ruido y produce un latente más inestable que la PCA.

**Valor del hallazgo (negativo pero útil).** La conclusión metodológica es limpia: en
este problema y con estos datos, **un reductor lineal (PCA) es preferible a un AE** como
front-end de clustering de regímenes. El deep learning solo se justificaría con muchos
más datos o features de alta frecuencia (intradía), como ya señaló el estado del arte.
No se sobre-optimizó para "ganar": el resultado es la comparación honesta.

## Hipótesis del CHECKPOINT 2 para D12 — veredicto

> *"Contraste ablativo honesto; con ~4 crisis el resultado negativo es aceptable; el
> deep solo se justificaría con más datos."*

**Se cumple exactamente.** El AE no aporta sobre PCA (resultado negativo), lo cual es
una contribución válida: documenta que la complejidad deep no está justificada aquí. La
comparación se hizo limpia (mismo pipeline) y sin sobreajustar para inflar el AE.

## Fricción con el núcleo

Ninguna en `src/`. Única incidencia: la aserción de causalidad del notebook era
demasiado estricta (`<1e-9`) para float32 de torch; se relajó a tolerancia FP + chequeo
de igualdad de estados (el criterio correcto). El detector es causal; no se tocó el
núcleo. Aprendizaje transversal: **las verificaciones de causalidad de detectores deep
deben comprobar invariancia de ESTADOS/probabilidades, no igualdad exacta del latente
en float** (el FP de torch da ~1e-7 inocuo).
