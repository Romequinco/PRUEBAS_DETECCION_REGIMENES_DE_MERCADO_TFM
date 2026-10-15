# Estado del arte v1: lista de detectores propuesta

> **Documento histórico.** Secciones 5 y 6 del documento maestro del estado del arte de la
> Capa 1 · etapa 2 (v1; original en el tag `capa1-final`), separadas de
> [`../estado_del_arte/detectores.md`](../estado_del_arte/detectores.md), que conserva la
> síntesis por familias, los solapes y las candidatas. El texto es el de entonces: las
> expectativas («captará», «fallará»), las librerías previstas y las prioridades son la
> **propuesta**, no lo implementado. Lo que se construyó de verdad, y en qué se apartó de esta
> propuesta (D7 sin BOCPD, D8 sin pomegranate, D11 sin R, D13 añadido como ablación), está en
> [§5 de `detectores.md`](../estado_del_arte/detectores.md#5-qué-se-implementó) y en las fichas
> de familia de [`../fase1/`](../fase1/README.md). Las claves entre corchetes son de
> [`../references.bib`](../references.bib). Las curtosis «25–40» que se citan son las del EDA v1
> (retornos diarios en la ventana común 2007-04-11 → 2026-06-12); el EDA actual está en
> [`../fase1/eda.md`](../fase1/eda.md).

## 1. Propuesta de lista definitiva de detectores para la etapa 3

Ordenada de **baseline a avanzado**. Cada detector lleva: nombre propuesto (estilo
`detectors/`), familia, por qué entra (literatura + EDA), qué se espera que
**capte** y qué se espera que **falle**, librería y **prioridad**
(imprescindible / recomendado / opcional-exploratorio). Numerados para
identificarlos por número.

> **BASELINES IMPRESCINDIBLES: D1 (regla sobre VIX) y D4 (HMM gaussiano puente).**

---

**D1 — `rule_vix_threshold`** · Familia F1 (Reglas/Umbrales) · **IMPRESCINDIBLE
(baseline)**
- *Por qué entra*: baseline reactivo, robusto a colas y causal nativo; contrasta
  cuánto pierde realmente el HMM por la mala especificación gaussiana
  [reglas_bloom2009]. Encaja 1:1 con `VIX_level_z` (causal, `max_abs_diff=0.0`).
- *Captará*: las 4 crisis y, por reactividad, probablemente **2013 y 2018** que el
  HMM falló.
- *Fallará*: falsos positivos ante picos efímeros si no lleva histéresis; ciego al
  cambio de signo de correlación (univariante).
- *Detalle*: umbral sobre `VIX_level_z` con **histéresis** (banda muerta τ_in/τ_out)
  + dwell-time mínimo. Librería: **pandas/NumPy**.

**D2 — `rule_composite_riskoff`** · Familia F1 · **RECOMENDADO**
- *Por qué entra*: una regla compuesta (voto de varios umbrales) capta interacciones
  que un umbral simple no ve, incluido el crédito [reglas_gilchristzakrajsek2012] y
  la pendiente de curva como señal líder [estrellamishkin1998].
- *Captará*: estrés multivariante (equity + crédito + curva); 2008/2011/2020/2022.
- *Fallará*: calibración de pesos/umbrales sensible; sigue siendo dura (0/1) salvo
  construir score.
- *Detalle*: voto/score sobre `VIX_level_z` + `credit_spread_z` + `yield_slope_z` +
  `SP500_drawdown`. Librería: **pandas/NumPy**.

**D3 — `clustering_gmm`** · Familia F2 (Clustering) · **RECOMENDADO (baseline no
temporal)**
- *Por qué entra*: aísla limpiamente el valor de la dinámica temporal — mismo input
  instantáneo que el HMM, sin matriz de transición [clust_twosigma2021regime]. GMM
  con covarianza plena capta el régimen de correlación que cambia de signo
  [gulko2002].
- *Captará*: regímenes con estructura de correlación distinta (diversificación vs
  "todo cae junto").
- *Fallará*: **flickering severo** (sin persistencia); no causal nativo.
- *Detalle*: `GaussianMixture` cov. plena, k=2–3 por BIC/silhouette, **wrapper
  causal** (re-fit expanding + asignación por probabilidad + alineado de etiquetas
  húngaro). Librería: **scikit-learn**.

**D4 — `hmm_gaussian_2s`** · Familia F3 (HMM) · **IMPRESCINDIBLE (baseline
puente)**
- *Por qué entra*: reproduce la tarea previa (misma ventana, `GaussianHMM full`,
  2 estados) para medir honestamente cuánto aportan las mejoras [hamilton1989].
- *Captará*: 2008 y 2020 (crisis grandes y persistentes).
- *Fallará*: **2013 y 2018** (correcciones rápidas); el gaussiano subestima las
  colas (kurt 25–40). Etiquetado claramente como **NO causal** en su versión
  in-sample; resultados evaluables solo con **filtrado forward / re-fit expanding**.
- *Detalle*: `GaussianHMM(n_components=2, covariance_type='full')`. Librería:
  **hmmlearn**.

**D5 — `markov_switching_var`** · Familia F4 (Markov-Switching) · **RECOMENDADO**
- *Por qué entra*: baseline econométrico **interpretable** con probabilidades
  filtradas causales nativas; switching de varianza ataca los clústeres de
  volatilidad [hamilton1989; ms_guidolin2011; ms_kim1994].
- *Captará*: regímenes calma/estrés sobre el retorno del S&P 500 con duración y
  matriz de transición legibles.
- *Fallará*: mismo punto ciego que el HMM en crisis rápidas (entrada gradual);
  univariante → no ve la correlación cross-asset; gaussiano insuficiente para las
  colas.
- *Detalle*: `MarkovRegression`/`MarkovAutoregression` sobre `SP500_ret`,
  `switching_variance=True`, K=2–3, **probabilidades filtradas**, re-fit expanding.
  Librería: **statsmodels**.

**D6 — `garch_t_vol`** · Familia F5 (Volatilidad) · **RECOMENDADO (baseline fuerte)**
- *Por qué entra*: **causal por construcción**, reacciona el mismo día a un shock y
  GARCH-t resuelve las colas residuales [vol_bollerslev1986; vol_bollerslev1987];
  la asimetría (GJR/EGARCH) replica el efecto apalancamiento de los drawdowns de
  equity [vol_glosten1993].
- *Captará*: las 4 crisis y, por reactividad de la varianza, **probablemente 2013 y
  2018**.
- *Fallará*: no da estados "tipados" (solo intensidad de vol → calma/estrés);
  univariante sobre equity.
- *Detalle*: `GJR-GARCH-t(1,1)` sobre `SP500_ret`, umbral sobre `σ_t` (percentil
  expanding) con histéresis. Librería: **arch** (Sheppard).

**D7 — `changepoint_online`** · Familia F6 (Change-point) · **RECOMENDADO**
- *Por qué entra*: detección temprana — la familia natural para la **métrica
  lead/lag** respecto al suelo del drawdown [cp_page1954; cp_adamsmackay2007].
- *Captará*: el **giro** de entrada en crisis con poco retardo (CUSUM/BOCPD).
- *Fallará*: falsas alarmas ante outliers leptocúrticos si se usa coste gaussiano
  (preferir **kernel/robusto**); requiere post-etiquetado económico causal de cada
  segmento.
- *Detalle*: CUSUM/ICSS + **BOCPD** online sobre features z-scoreadas; PELT (kernel
  `rbf`) de `ruptures` solo como oráculo offline re-aplicado en ventana
  [cp_killick2012; cp_truong2020]. Librería: **ruptures** +
  **bayesian_changepoint_detection**.

**D8 — `hmm_tstudent`** · Familia F3 (HMM avanzado) · **RECOMENDADO**
- *Por qué entra*: mejora más directa sobre el baseline gaussiano — las emisiones t
  absorben las colas y **aumentan la persistencia** (menos flickering)
  [hmm_bulla2011]; K=3–4 separa "corrección" de "crisis sistémica"
  [guidolintimmermann2007].
- *Captará*: 2008/2020 y, gracias a más estados + colas pesadas, **potencialmente
  2013/2018**.
- *Fallará*: más parámetros → riesgo de sobreajuste con pocas observaciones por
  estado; coste de walk-forward + anclaje de etiquetas.
- *Detalle*: HMM con emisiones **t-Student** (o GMM-HMM como fallback), cov. plena,
  K por BIC/AIC, **filtrado forward** causal. Librería: **pomegranate** (t) o
  **hmmlearn** (GMM-HMM).

**D9 — `jump_model`** · Familia F2↔F3 (clustering con persistencia / jump models) ·
**RECOMENDADO**
- *Por qué entra*: puente exacto entre clustering estático y dinámica temporal —
  impone persistencia penalizando saltos (histéresis "aprendida"), pensado para
  detección **online** y contra el flickering [hmm_nystrup2020].
- *Captará*: regímenes persistentes con drawdowns más suaves; rival honesto del
  AE+clustering en muestra pequeña.
- *Fallará*: depende de la penalización de salto (otro hiperparámetro a calibrar);
  madurez de librería en Python.
- *Detalle*: Statistical Jump Model (clustering + jump penalty). Librería:
  **jumpmodels** (o implementación propia según [hmm_nystrup2020]).

**D10 — `turbulence_mahalanobis`** · Familia F1 multivariante (Kritzman) ·
**RECOMENDADO (opcional)**
- *Por qué entra*: índice de turbulencia multivariante umbralizable; capta el
  **colapso de correlaciones** que las reglas univariantes no ven, manteniéndose
  barato y causal [kritzman2012; gulko2002].
- *Captará*: episodios de estrés sistémico multivariante (ruptura de la estructura
  de covarianza).
- *Fallará*: sensible a la ventana de la covarianza; un único umbral sigue siendo
  duro.
- *Detalle*: distancia de Mahalanobis sobre features con **covarianza expanding**
  (causal), umbral por percentil + histéresis. Librería: **NumPy/SciPy**.

**D11 — `msgarch_regime`** · Familia F5 (RS-GARCH avanzado) · **OPCIONAL-EXPLORATORIO**
- *Por qué entra*: la "extensión natural" — heteroscedasticidad GARCH **dentro** de
  cada régimen + probabilidad filtrada de régimen, mejor para las colas que el MS
  gaussiano [vol_gray1996; vol_haasmittnikpaolella2004; vol_marcucci2005].
- *Captará*: regímenes de volatilidad con colas dependientes del estado (ν por
  régimen).
- *Fallará*: **coste alto y fragilidad** (path dependence, óptimos locales,
  label-switching); sin equivalente maduro en Python puro (R MSGARCH o
  implementación propia) [vol_ardia2019].
- *Detalle*: MS-GARCH 2–3 estados, prob. filtrada. Librería: **rpy2 + MSGARCH (R)**
  o implementación propia (formulación Haas–Mittnik–Paolella).

**D12 — `deep_ae_regime`** · Familia F7 (Redes neuronales) · **OPCIONAL-EXPLORATORIO**
- *Por qué entra*: contraste ablativo honesto — comprobar si la no linealidad aporta
  algo sobre PCA+GMM; resultado **negativo aceptable** como contribución
  [nn_akioyamen2021; lopezdeprado2018].
- *Captará* (en teoría): no linealidades y, vía AE-anomalía, "rareza" instantánea
  (COVID-2020).
- *Fallará* (lo esperable): **overfitting** con ~4 crisis y ventana corta;
  flickering del latente; baja interpretabilidad; poca reproducibilidad. Con
  disclaimer explícito de muestra pequeña [nn_bucci2021].
- *Detalle*: **un solo** representante ligero — AE pequeño → GMM híbrido, **o**
  AE-anomalía como stress index — regularización fuerte, normalización ajustada
  por fold. Librería: **PyTorch + scikit-learn** (o PyOD para AE-anomalía).

---

### Resumen de prioridades

| Prioridad | Detectores |
|---|---|
| **Imprescindible (baseline)** | D1 `rule_vix_threshold`, D4 `hmm_gaussian_2s` |
| **Recomendado** | D2 `rule_composite_riskoff`, D3 `clustering_gmm`, D5 `markov_switching_var`, D6 `garch_t_vol`, D7 `changepoint_online`, D8 `hmm_tstudent`, D9 `jump_model`, D10 `turbulence_mahalanobis` |
| **Opcional-exploratorio** | D11 `msgarch_regime`, D12 `deep_ae_regime` |

Cobertura de familias: **las 7 familias quedan representadas** (F1: D1, D2, D10;
F2: D3, D9; F3: D4, D8; F4: D5; F5: D6, D11; F6: D7; F7: D12). Ninguna familia se
excluye; el deep (F7) entra solo como exploratorio por escasez de datos, como
exige la honestidad metodológica del EDA.

---

## 2. Cierre

Esta lista de **12 detectores** fue la propuesta de la etapa 2 de la Capa 1; se aprobó y se implementó en la etapa 3 (hoy `src/regimenes/detectores/`, fichas por detector en [`../fase1/`](../fase1/README.md)). Después se añadió D13 (`hsmm_tstudent`, ablación HSMM frente a D8), fuera del benchmark. Las decisiones que quedaban abiertas entonces (estrategia de walk-forward ante el inicio tardío de datos y set de features) se resolvieron en [ADR-001](../decisions/ADR-001-rebase-datos.md) y [ADR-002](../decisions/ADR-002-ajuste-ventanas.md).
