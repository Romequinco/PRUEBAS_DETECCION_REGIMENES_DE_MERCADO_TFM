# Resumen de la Tarea — Detección de Regímenes de Mercado y Stress Testing

> Documento de síntesis de lo **hecho, descubierto y usado** en la tarea
> `Tarea_riesgos.ipynb` (Sistema de Stress Testing Multi-Activo).
> Énfasis en la **detección de regímenes de estado/mercado**, como base para
> diseñar un detector nuevo y más robusto.

---

## 1. Contexto general de la tarea

La práctica implementa un pipeline completo de gestión de riesgo de una cartera
multi-activo en 6 fases encadenadas:

| Fase | Contenido | Variables clave |
|------|-----------|-----------------|
| **0** | Setup y datos (2006–2026), 5 indicadores + 18 activos, retornos log | `data_market`, `returns_market`, `returns_portfolio` |
| **1** | **Detección de regímenes (HMM)** ← núcleo de este resumen | `hmm_model`, `states_df`, `state_calm`, `state_crisis` |
| **2** | Análisis marginal del riesgo por régimen | `marginal_stats`, `returns_with_regime` |
| **3** | Dependencia y diversificación (cópulas) por régimen | `corr_calm`, `corr_crisis`, `copula_models` |
| **4** | Motor de simulación Monte Carlo (`PortfolioSimulator`) | `simulator`, `simulation_results` |
| **5** | Escenarios de estrés (2022, 2008, pérdida de confianza crediticia USA) | `scenarioN_sim_metrics` |

La detección de regímenes (Fase 1) es la **columna vertebral**: alimenta el
análisis marginal, las correlaciones condicionales y el régimen de partida del
motor Monte Carlo.

---

## 2. Datos usados para detectar el régimen

- **Fuente**: Yahoo Finance (`yfinance`), 2006–2026, frecuencia diaria.
- **Política de no imputación**: cada serie arranca en su fecha real; no se
  rellenan NaN artificialmente.
- **Indicadores de estado de mercado** (5): `^GSPC` (S&P 500), `^VIX`,
  `TLT` (Treasuries largo), `IEF` (Treasuries medio), `HYG` (high yield).
- **Decisión clave de ventana**: el HMM **arranca en `2007-04-11`** (fecha de
  inicio de HYG) para que el riesgo de crédito participe en la detección desde
  el primer día. Tras el `rolling` y `dropna`, la muestra efectiva es
  **2007-05-08 → 2026-02-10 = 4.721 observaciones**.

---

## 3. Cómo se detectó el régimen — metodología (Fase 1)

### 3.1. Features de entrada al HMM (7 variables)

No se usaron retornos crudos sino **features transformadas y estandarizadas**:

1. `^GSPC_zscore` — retorno estandarizado del S&P 500
2. `TLT_zscore` — retorno estandarizado de Treasuries largos
3. `IEF_zscore` — retorno estandarizado de Treasuries medios
4. `HYG_zscore` — retorno estandarizado de high yield
5. `SP500_volatility_20d` — volatilidad móvil 20 días, anualizada (×√252), estandarizada
6. `credit_spread` — proxy de spread de crédito = retorno(HYG) − retorno(IEF), estandarizado
7. `VIX_level` — **nivel absoluto** del VIX (no su retorno), estandarizado

**Justificación de cada elección** (relevante para el nuevo detector):
- Estandarizar evita que el VIX (niveles 10–80) domine sobre los retornos (±5 %).
- La volatilidad rolling capta el "pulso" del mercado mejor que el retorno puntual.
- El spread HYG−IEF es un proxy de riesgo crediticio más robusto que retornos sueltos.
- El **nivel** de VIX (no su variación) es el indicador directo de "miedo".

### 3.2. Modelo

- **Gaussian HMM de 2 estados** (`hmmlearn.hmm.GaussianHMM`).
- `covariance_type='full'` → permite correlaciones entre las 7 features.
- `n_iter=1000`, `tol=1e-4`.
- **10 inicializaciones aleatorias** (seeds 42–51) y se elige la de mayor
  log-likelihood → mitiga óptimos locales.
- Decodificación de la secuencia de estados con **algoritmo de Viterbi**
  (`model.predict`).

### 3.3. Etiquetado de estados (Calma vs Crisis)

El HMM asigna etiquetas arbitrarias (0/1). Se identifican post-hoc por criterio
económico: **Crisis = estado con mayor volatilidad del S&P 500 Y mayor VIX
medio** (con fallback solo a volatilidad si el VIX no es concluyente).

### 3.4. Validación

- **Matriz de transición** → persistencia y duración esperada de cada régimen.
- **Distribución estacionaria** (eigenvector izquierdo de P con autovalor 1).
- **Coherencia histórica**: % de días clasificados como Crisis dentro de 6
  ventanas de crisis conocidas. Verificación crítica obligatoria: 2008 y 2020.
- 3 visualizaciones: S&P 500 coloreado por régimen, timeline de regímenes,
  histogramas de duración de episodios.

---

## 4. Resultados obtenidos (lo descubierto)

### 4.1. Caracterización de los dos regímenes

| Métrica | Calma | Crisis | Ratio Crisis/Calma |
|---|---|---|---|
| Vol S&P 500 (anual) | 11,75 % | 34,16 % | **2,91×** |
| Retorno S&P 500 (anual) | +18,48 % | −22,35 % | −1,21× |
| VIX medio | 16,30 | 30,60 | 1,88× |
| Retorno HYG (anual) | +8,19 % | −5,11 % | −0,62× |
| Duración media episodio | 49 días | 17 días | 0,34× |
| % del tiempo total | 74,7 % | 25,3 % |  — |

### 4.2. Dinámica de transición

```
P(Calma → Calma)  = 0,9794      P(Calma → Crisis)  = 0,0206
P(Crisis → Calma) = 0,0605      P(Crisis → Crisis) = 0,9395
```
- Duración esperada: **Calma ≈ 48,5 días**, **Crisis ≈ 16,5 días**.
- Distribución estacionaria de largo plazo: **74,6 % Calma / 25,4 % Crisis**.
- Ambos regímenes son **persistentes** (diagonal alta); la entrada en crisis es
  rara (2,1 %/día) pero la salida también es lenta (6,1 %/día).
- Log-likelihood del mejor modelo: **3.822,73**; las 10 inicializaciones
  convergieron al **mismo óptimo** → solución estable.

### 4.3. Validación histórica

| Evento | % días en Crisis | Veredicto |
|---|---|---|
| Lehman (2008) | **98,6 %** | ✓ alta coherencia |
| Deuda Europea (2011) | 67,2 % | ✓ alta coherencia |
| COVID-19 (2020) | **92,3 %** | ✓ alta coherencia |
| Inflación (2022) | 78,9 % | ✓ alta coherencia |
| Taper Tantrum (2013) | 10,9 % | ✗ **no detectado** |
| Sell-off Q4 2018 | 20,6 % | ✗ **no detectado** |

El modelo acierta de pleno las **crisis sistémicas grandes** (2008, 2020, 2022,
2011) pero **se pierde las correcciones rápidas / de menor magnitud** (2013,
2018). Este es el hallazgo más importante de cara a robustez.

---

## 5. Cómo se usó el régimen aguas abajo

- **Fase 2**: estadísticas marginales (media, vol, skew, kurtosis) calculadas
  *condicionadas al régimen* → la vol se amplifica 2–4× en crisis.
- **Fase 3**: matrices de correlación y cópulas Gaussianas **separadas por
  régimen** (`corr_calm`, `corr_crisis`) → las correlaciones suben
  sistemáticamente en crisis (falla la diversificación) y aumenta la
  dependencia en colas.
- **Fase 4**: el `PortfolioSimulator` integra HMM + marginales + cópulas;
  simula 10.000 trayectorias × 126 días arrancando del régimen actual y
  conmutando según la matriz de transición.
- **Fase 5**: los escenarios de estrés fijan/condicionan el régimen para
  estresar VaR 99 % y CVaR 99 %.

---

## 6. Stack técnico

- `numpy`, `pandas` — cálculo y manejo de series.
- `yfinance` — descarga de datos.
- `hmmlearn` (`GaussianHMM`) — detección de regímenes.
- `scipy.stats` (`t`, `norm`, `chi2`) — distribuciones del simulador.
- `copulas` — cópulas multivariantes.
- `matplotlib`, `seaborn` — visualización.
- Semilla global **42** → reproducibilidad.

---

## 7. Limitaciones del detector actual (puntos de mejora)

Diagnóstico para diseñar el **nuevo detector robusto**:

1. **Solo 2 estados.** Colapsa "corrección normal", "crisis sistémica" y
   "estanflación" en un único estado Crisis. No distingue *tipos* de estrés.
2. **Ciego a crisis rápidas/medianas** (Taper Tantrum 2013, Q4 2018): el HMM
   gaussiano prioriza episodios largos y de alta varianza.
3. **Etiquetado por umbral post-hoc** (vol + VIX). Frágil si aparece un régimen
   intermedio o si VIX y vol divergen.
4. **Look-ahead / estabilidad temporal**: el modelo se entrena sobre **toda** la
   muestra (in-sample). Los regímenes históricos pueden cambiar si se reentrena;
   no hay validación *walk-forward* ni detección *online* causal.
5. **Supuesto gaussiano** de las emisiones: subestima colas gordas; los retornos
   reales son leptocúrticos.
6. **Features estandarizadas con media/desviación de toda la muestra** → fuga de
   información del futuro en el z-score (otro look-ahead sutil).
7. **`P(estado 0 inicial)=1`**: la inicialización es degenerada; conviene
   revisar `startprob_`.
8. **Sin medida de incertidumbre** en la clasificación diaria (se usa Viterbi
   "duro", no las probabilidades posteriores suaves `predict_proba`).

---

## 8. Ideas para un detector de regímenes nuevo y robusto

Pruebas sugeridas a partir de lo aprendido:

- **Más estados / selección de K**: probar HMM de 3–4 estados
  (calma · corrección · crisis · estanflación) y elegir K por BIC/AIC o
  log-likelihood penalizada.
- **Emisiones no gaussianas**: HMM con mixturas (GMM-HMM) o emisiones t-Student
  para capturar colas; o **Markov-Switching** (statsmodels) sobre retornos.
- **Validación walk-forward / out-of-sample**: reentrenar en ventana móvil y
  evaluar la clasificación causal (sin ver el futuro); medir estabilidad de
  etiquetas.
- **Probabilidades suaves**: usar `predict_proba` (filtrado/forward-backward)
  para tener una probabilidad de crisis continua en lugar de binaria, y umbral
  calibrado.
- **Z-scores causales**: estandarizar con media/desv *expanding* o rolling, no
  con estadísticos de toda la muestra.
- **Ampliar el set de features**: pendiente de la curva (10y−3m), MOVE index,
  drawdown, momentum, dispersión de correlaciones, breadth, spreads de crédito
  reales (OAS), liquidez.
- **Benchmark contra métodos no-HMM**: reglas sobre VIX, modelos de cambio
  estructural, clustering (k-means/GMM sobre features), o detección de cambios
  (CUSUM, bayesian change-point) y comparar con eventos 2013/2018.
- **Métricas de evaluación objetivas**: además del % en crisis por ventana,
  medir lead/lag respecto a drawdowns, tasa de falsas alarmas, persistencia y
  frecuencia de conmutación (evitar "flickering").

---

## 9. Inventario de artefactos de la tarea

- `Tarea_riesgos.ipynb` — notebook completo (140 celdas, Fases 0–5).
- `README.md` — descripción de fases y hallazgos.
- `docs/Practica_Gestion_Riesgos (1).pdf` — enunciado / especificación.
- `docs/pruebas.html`, `informe_ejecutivo_stress_testing (15).html` — informes.
- `charts/chart_01..24.png` — 24 figuras exportadas.
- Celdas de la Fase 1: 32–54 (preparación de features, estimación HMM, Viterbi,
  etiquetado, matriz de transición, validación, visualizaciones, resumen).

---

*Generado el 2026-06-16 a partir del contenido de `Tarea_riesgos.ipynb`.*
