# 02 — Tablas y figuras (extracción cuantitativa para el PDF ejecutivo)

> **PROCEDENCIA DE LOS NÚMEROS.** Todos los valores numéricos de este documento provienen
> de `results/metrics_master_final.csv` (12 detectores × 34 columnas), copiados **VERBATIM**,
> salvo el BIC de D4 (`hmm_gaussian_2s`), que además se corrobora en
> `results/metrics_04_hmm_gaussian_2s.csv` (coincide exactamente: 35379.407741).
> No se ha recalculado ni estimado nada. Donde el CSV dice `NaN` se mantiene `NaN`
> (= métrica no aplicable: la ventana OOS del detector no cubre ese evento, o la trampa cae
> fuera de su ventana). Redondeos indicados en cada tabla; los valores de máxima precisión
> (BIC, AIC, log-lik, tasas) se citan tal cual para evitar derivas del redactor.

## Mapa detector → etiqueta corta → familia → ventana

| Etiqueta | detector (CSV) | n_states | clase | coste | vio_2008_oos | ventana_eval | n_oos |
|---|---|---|---|---|---|---|---|
| D6 | garch_t_vol | 2 | avanzado | medio | True | 1993-03-23→2026-06-12 | 8278 |
| D5 | markov_switching_var_2s | 2 | avanzado | alto | True | 1993-03-23→2026-06-12 | 8278 |
| D7 | changepoint_online | 2 | avanzado | bajo | True | 1993-03-23→2026-06-12 | 8278 |
| D1 | rule_vix_threshold | 2 | baseline | bajo | True | 1998-06-23→2026-06-12 | 6994 |
| D10 | turbulence_mahalanobis | 2 | avanzado | bajo | True | 1998-06-02→2026-06-12 | 6987 |
| D11 | msgarch_regime | 2 | exploratorio-negativo | alto | True | 1991-03-04→2026-06-12 | 8782 |
| D3 | clustering_gmm_k3 | 3 | baseline | medio | False | 2015-09-15→2026-06-12 | 2649 |
| D4 | hmm_gaussian_2s | 2 | baseline | medio | False | 2012-07-20→2026-06-12 | 3405 |
| D2 | rule_composite_riskoff | 2 | baseline | bajo | False | 2015-09-15→2026-06-12 | 2649 |
| D8 | hmm_tstudent_4s | 4 | avanzado | alto | False | 2012-07-20→2026-06-12 | 3405 |
| D9 | jump_model | 2 | avanzado | medio | False | 2015-09-15→2026-06-12 | 2649 |
| D12 | deep_ae_regime | 3 | exploratorio-negativo | medio | False | 2015-09-15→2026-06-12 | 2649 |

> **Aviso de comparabilidad transversal.** Las dos mitades de la tabla NO son directamente
> comparables: los 6 de arriba (`vio_2008_oos=True`) evalúan sobre una ventana larga que
> incluye la GFC 2008; los 6 de abajo (`vio_2008_oos=False`) arrancan en 2012/2015 y nunca
> vieron 2008 fuera de muestra. Cualquier ranking de cobertura que mezcle ambos grupos es
> engañoso (ver nota sobre `fase4_rank_heatmap.png`).

---

## Tabla 1 — Maestra resumida

Orden = orden del CSV (primero `vio_2008_oos=True`, luego por cobertura sistémica
decreciente; después los de ventana corta). Tasas a 4 decimales, duración a 2, BIC entero.

| Etiqueta | detector | clase | ventana_eval | vio_2008_oos | switching_rate | mean_regime_duration | false_alarm_rate | bic |
|---|---|---|---|---|---|---|---|---|
| D6 | garch_t_vol | avanzado | 1993→2026 (n=8278) | True | 0.0141 | 70.15 | 0.8451 | 26626.56 |
| D5 | markov_switching_var_2s | avanzado | 1993→2026 (n=8278) | True | 0.0557 | 17.92 | 0.7744 | 28023.77 |
| D7 | changepoint_online | avanzado | 1993→2026 (n=8278) | True | 0.0022 | 435.68 | 0.8674 | NaN |
| D1 | rule_vix_threshold | baseline | 1998→2026 (n=6994) | True | 0.0132 | 75.20 | 0.6974 | NaN |
| D10 | turbulence_mahalanobis | avanzado | 1998→2026 (n=6987) | True | 0.0873 | 11.44 | 0.8152 | NaN |
| D11 | msgarch_regime | exploratorio-negativo | 1991→2026 (n=8782) | True | 0.0312 | 31.93 | 0.9494 | 26823.31 |
| D3 | clustering_gmm_k3 | baseline | 2015→2026 (n=2649) | False | 0.1261 | 7.91 | 0.4945 | 63016.25 |
| D4 | hmm_gaussian_2s | baseline | 2012→2026 (n=3405) | False | 0.1004 | 9.93 | 0.7314 | 35379.41 |
| D2 | rule_composite_riskoff | baseline | 2015→2026 (n=2649) | False | 0.0385 | 25.72 | 0.7245 | NaN |
| D8 | hmm_tstudent_4s | avanzado | 2012→2026 (n=3405) | False | 0.0520 | 19.13 | 0.5189 | 24415.89 |
| D9 | jump_model | avanzado | 2015→2026 (n=2649) | False | 0.0053 | 176.60 | 0.6243 | NaN |
| D12 | deep_ae_regime | exploratorio-negativo | 2015→2026 (n=2649) | False | 0.2873 | 3.48 | 0.6033 | NaN |

*Nota:* `false_alarm_rate` aquí es la tasa global (no la de trampas concretas, que está en la Tabla 4).

---

## Tabla 2 — Cobertura de crisis (definición ESTRICTA: cola extrema / state==crisis)

Valores a 3 decimales (proporción de días de cada ventana de crisis con el detector "encendido").

**Grupo A — ventana larga (`vio_2008_oos = True`), única que tiene GFC 2008 y EuroDebt 2011 OOS:**

| Etiqueta | detector | cov_GFC_2008 | cov_EuroDebt_2011 | cov_COVID_2020 | cov_Inflation_2022 |
|---|---|---|---|---|---|
| D6 | garch_t_vol | 1.000 | 0.741 | 0.940 | 0.804 |
| D5 | markov_switching_var_2s | 0.993 | 0.741 | 0.960 | 0.737 |
| D7 | changepoint_online | 1.000 | 0.671 | 0.840 | 0.766 |
| D1 | rule_vix_threshold | 0.938 | 0.635 | 0.900 | 0.349 |
| D10 | turbulence_mahalanobis | 0.822 | 0.482 | 0.760 | 0.431 |
| D11 | msgarch_regime (neg.) | 0.000 | 0.118 | 0.200 | 0.359 |

**Grupo B — ventana corta (`vio_2008_oos = False`), GFC y EuroDebt = `NaN` (fuera de muestra):**

| Etiqueta | detector | cov_GFC_2008 | cov_EuroDebt_2011 | cov_COVID_2020 | cov_Inflation_2022 |
|---|---|---|---|---|---|
| D3 | clustering_gmm_k3 | NaN | NaN | 0.960 | 0.865 |
| D4 | hmm_gaussian_2s | NaN | NaN | 0.960 | 0.861 |
| D2 | rule_composite_riskoff | NaN | NaN | 0.840 | 0.538 |
| D8 | hmm_tstudent_4s | NaN | NaN | 0.660 | 0.332 |
| D9 | jump_model | NaN | NaN | 0.720 | 0.168 |
| D12 | deep_ae_regime (neg.) | NaN | NaN | 0.540 | 0.101 |

---

## Tabla 3 — Cobertura de crisis (ESTRÉS AGREGADO: corrección + crisis)

> En **detectores binarios** (D6, D5, D7, D1, D10, D11, D4, D2, D9) el estrés agregado
> **coincide exactamente con la crisis estricta** (sólo hay un estado de riesgo,
> `state==1`); por eso `cov_estres_* == cov_*`. Sólo difieren los **multi-estado**:
> D3 (k=3), D8 (k=4) y D12 (k=3), donde estrés = {corrección + crisis}.

| Etiqueta | detector | cov_estres_GFC | cov_estres_EuroDebt | cov_estres_COVID | cov_estres_Inflation | ¿difiere de estricta? |
|---|---|---|---|---|---|---|
| D6 | garch_t_vol | 1.000 | 0.741 | 0.940 | 0.804 | no (binario) |
| D5 | markov_switching_var_2s | 0.993 | 0.741 | 0.960 | 0.737 | no (binario) |
| D7 | changepoint_online | 1.000 | 0.671 | 0.840 | 0.766 | no (binario) |
| D1 | rule_vix_threshold | 0.938 | 0.635 | 0.900 | 0.349 | no (binario) |
| D10 | turbulence_mahalanobis | 0.822 | 0.482 | 0.760 | 0.431 | no (binario) |
| D11 | msgarch_regime (neg.) | 0.000 | 0.118 | 0.200 | 0.359 | no (binario) |
| D3 | clustering_gmm_k3 | NaN | NaN | 0.960 | 0.865 | **sí** (k=3) |
| D4 | hmm_gaussian_2s | NaN | NaN | 0.960 | 0.861 | no (binario) |
| D2 | rule_composite_riskoff | NaN | NaN | 0.840 | 0.538 | no (binario) |
| D8 | hmm_tstudent_4s | NaN | NaN | **0.960** | **0.904** | **sí** (k=4) |
| D9 | jump_model | NaN | NaN | 0.720 | 0.168 | no (binario) |
| D12 | deep_ae_regime (neg.) | NaN | NaN | 0.960 | 0.788 | **sí** (k=3) |

*Contrastes que cambian con estrés (estricta → estrés):*
- **D8 COVID: 0.660 → 0.960** ; **D8 Inflation: 0.332 → 0.904**.
- **D3 COVID: 0.960 → 0.960** (igual) ; **D3 Inflation: 0.865 → 0.865** (igual en cobertura).
- **D12 COVID: 0.540 → 0.960** ; **D12 Inflation: 0.101 → 0.788**.

---

## Tabla 4 — Especificidad / trampas (falsos positivos en no-crisis)

Tasa de "encendido" en dos ventanas-trampa (TaperTantrum 2013, Sell-off Q4 2018).
Más bajo = mejor (no dispara en falso). 3 decimales. `NaN` = la trampa cae fuera de la ventana OOS.

| Etiqueta | detector | fa_Taper_2013 | fa_Selloff_2018 | fa_estres_Taper_2013 | fa_estres_Selloff_2018 |
|---|---|---|---|---|---|
| D6 | garch_t_vol | 0.113 | 0.873 | 0.113 | 0.873 |
| D5 | markov_switching_var_2s | 0.038 | 0.810 | 0.038 | 0.810 |
| D7 | changepoint_online | 0.000 | 0.000 | 0.000 | 0.000 |
| D1 | rule_vix_threshold | 0.000 | 0.063 | 0.000 | 0.063 |
| D10 | turbulence_mahalanobis | 0.123 | 0.302 | 0.123 | 0.302 |
| D11 | msgarch_regime (neg.) | 0.000 | 0.937 | 0.000 | 0.937 |
| D3 | clustering_gmm_k3 | NaN | 0.000 | NaN | **0.729** |
| D4 | hmm_gaussian_2s | 0.250 | 0.458 | 0.250 | 0.458 |
| D2 | rule_composite_riskoff | NaN | 0.424 | NaN | 0.424 |
| D8 | hmm_tstudent_4s | 0.000 | **0.034** | 0.000 | **0.814** |
| D9 | jump_model | NaN | 0.000 | NaN | 0.000 |
| D12 | deep_ae_regime (neg.) | NaN | 0.153 | NaN | 0.186 |

*Trade-off clave del estrés agregado:* ampliar a "corrección+crisis" mejora cobertura pero
**dispara los falsos positivos en multi-estado**: D8 Sell-off 2018 pasa de **0.034 → 0.814**;
D3 de **0.000 → 0.729**; D12 de 0.153 → 0.186. Los binarios no cambian.

---

## Tabla 5 — Bondad de ajuste (sólo modelos generativos, no-`NaN`)

Sólo 5 detectores reportan verosimilitud. **El BIC sólo es comparable entre modelos sobre
las MISMAS features** → el contraste limpio es **D4 vs D8** (ambos HMM, mismo feature set).
Valores VERBATIM del master final; BIC de D4 corroborado en `metrics_04_hmm_gaussian_2s.csv`.

| Etiqueta | detector | log_likelihood | aic | bic |
|---|---|---|---|---|
| D8 | hmm_tstudent_4s | −11536.339896 | 23390.679792 | **24415.886847** |
| D6 | garch_t_vol | −13285.563254 | 26583.126507 | 26626.562406 |
| D11 | msgarch_regime (neg.) | −13365.456001 | 26750.912002 | 26823.305167 |
| D5 | markov_switching_var_2s | −13984.167114 | 27980.334228 | 28023.770127 |
| D4 | hmm_gaussian_2s | −17381.357597 | 34908.715193 | **35379.407741** |
| D3 | clustering_gmm_k3 | −29788.986979 | 60391.973957 | 63016.246104 |

**Contraste D4 vs D8 (BIC, menor = mejor ajuste):**
- D4 `hmm_gaussian_2s`: BIC = **35379.407741** (≈ 35379)
- D8 `hmm_tstudent_4s`: BIC = **24415.886847** (≈ 24416)
- **ΔBIC (D4 − D8) = 10963.520894 ≈ 10964** → el HMM t-Student de 4 estados (colas gordas)
  ajusta dramáticamente mejor que el HMM gaussiano de 2 estados. Diferencia de BIC > 10
  ya se considera evidencia "muy fuerte"; aquí es ~11000.

---

## Tabla 6 — Lead/lag por evento (días respecto al suelo del evento)

Negativo = el detector **anticipa** el suelo (se enciende antes); más negativo = más antelación.
`NaN` = evento fuera de la ventana OOS del detector.

| Etiqueta | detector | leadlag_GFC_2008 | leadlag_EuroDebt_2011 | leadlag_COVID_2020 | leadlag_Inflation_2022 |
|---|---|---|---|---|---|
| D6 | garch_t_vol | −252 | −42 | −158 | −217 |
| D5 | markov_switching_var_2s | −252 | −43 | −159 | −220 |
| D7 | changepoint_online | −252 | −252 | −204 | −252 |
| D1 | rule_vix_threshold | −251 | −39 | −17 | −179 |
| D10 | turbulence_mahalanobis | −250 | −39 | −15 | −220 |
| D11 | msgarch_regime (neg.) | NaN | −39 | −35 | −220 |
| D3 | clustering_gmm_k3 | NaN | NaN | −20 | −219 |
| D4 | hmm_gaussian_2s | NaN | NaN | −151 | −219 |
| D2 | rule_composite_riskoff | NaN | NaN | −251 | −219 |
| D8 | hmm_tstudent_4s | NaN | NaN | −150 | −172 |
| D9 | jump_model | NaN | NaN | −8 | −80 |
| D12 | deep_ae_regime (neg.) | NaN | NaN | −4 | NaN |

> **NOTA CRÍTICA sobre lead/lag.** Los valores en torno a **−252 (y −251, −250, −220, −219)
> están CENSURADOS por el lookback** de la ventana de evaluación del evento: no significan
> "anticipó exactamente 252 días", sino "ya estaba encendido al inicio de la ventana de
> medición" (saturación en el borde). Interpretar estos extremos como antelación literal en
> días sería un error; sólo los valores claramente interiores (p.ej. D9 COVID −8, D10/D1
> COVID −15/−17, D3 COVID −20) reflejan antelación real medible.

---

## Verificación de las 3 cifras clave (confirmadas contra el CSV, sin alterar)

1. **ΔBIC D4 − D8.** D4 `hmm_gaussian_2s` BIC = 35379.407741 ; D8 `hmm_tstudent_4s`
   BIC = 24415.886847 → **ΔBIC = 10963.520894 ≈ 10964** (redactor puede citar "≈35379 vs ≈24416,
   ΔBIC≈10963"). **CONFIRMADO.**
2. **Salto COVID estricta → estrés en D8.** `cov_COVID_2020` = 0.66 → `cov_estres_COVID_2020`
   = 0.96. **CONFIRMADO (0.66 → 0.96).**
3. **Subida de fa_2018 de D8 con estrés.** `fa_Selloff_Q4_2018` = 0.033898 → `fa_estres_Selloff_Q4_2018`
   = 0.813559 → **0.034 → 0.81. CONFIRMADO.**

---

## Inventario de figuras (`results/*.png`)

### Grupo EDA (`eda_*`)
- **`eda_corr.png`** — Matriz/heatmap de correlación entre activos del universo (SP500, TLT, IEF, HYG, GOLD, DXY): estructura de correlaciones estática.
- **`eda_rolling_corr.png`** — Correlación móvil (rolling) entre activos a lo largo del tiempo: muestra cómo se disparan las correlaciones en crisis (pérdida de diversificación).
- **`eda_fat_tails.png`** — Distribución de retornos vs Normal en escala log para 6 activos, con exceso de curtosis anotado (SP500 25.6, HYG 39.6, GOLD 6.8, TLT 3.4, IEF 2.5, DXY 2.2): evidencia de colas gordas que justifica modelos t-Student.
- **`eda_sp500_drawdown.png`** — S&P 500 (log) y su drawdown, con sombreado de las ventanas de CRISIS (rojo) y de los FALSOS POSITIVOS/trampas (naranja): define visualmente el ground-truth de eventos.

### Grupo Fase 4 — comparativa final (`fase4_*`)
- **`fase4_bic.png`** — Barras de BIC de los 6 modelos generativos (D8≈24416, D6≈26627, D11≈26823, D5≈28024, D4≈35379, D3≈63016; D11 en gris=negativo). Aviso en el título: BIC sólo estrictamente comparable sobre las MISMAS features (D4 vs D8).
- **`fase4_estres_vs_estricta.png`** — Barras pareadas crisis estricta vs estrés agregado para los 3 multi-estado (D3, D8, D12) en COVID e Inflation; ilustra el salto de D8 (0.66→0.96 COVID, 0.33→0.90 Inflation) y D12 (0.54→0.96, 0.10→0.79).
- **`fase4_leadlag.png`** — Heatmap lead/lag (días) por detector × evento (azul=anticipa, rojo=retrasa); reproduce la Tabla 6, con los extremos −252 saturados.
- **`fase4_persistencia_sensibilidad.png`** — Scatter persistencia (duración media de régimen, eje log) vs sensibilidad; separa los que "flickean" (D3/D12) de los persistentes (D7 cusum, reglas).
- **`fase4_sensibilidad_especificidad.png`** — Scatter especificidad (1 − media fa_2013/fa_2018) vs sensibilidad (media cov GFC/COVID); azul=ventana larga (vio 2008), naranja=ventana corta, X gris=negativos (D11, D12). Plano de trade-off principal.
- **`fase4_rank_heatmap.png`** — Heatmap de RANKS (1=mejor) por eje (Cob.sistémica, Especif.estricta/estrés, Persistencia, Anti-flicker, Lead/lag, BIC, Coste) para los 12 detectores.
  > **AVISO (revisión académica): esta figura MEZCLA VENTANAS en su columna de cobertura
  > ("Cob.sistémica"):** rankea juntos detectores de ventana larga (que vieron GFC 2008) y de
  > ventana corta (que no), de modo que el ranking de cobertura no es comparable entre grupos.
  > Usar con cautela / no citar el ranking de cobertura de esta figura sin la salvedad.

### Grupos por detector individual (referencia, no para el cuerpo del PDF)
- **`d1_*`** (rule_vix_threshold): `d1_prob_timeline`, `d1_regime_sp500`.
- **`d2_*`** (rule_composite_riskoff): `d2_regime_sp500`, `d2_score_timeline`.
- **`d03_*`** (clustering_gmm_k3): `d03_gmm_flickering`, `d03_gmm_sp500_regimes`, `d03_gmm_timeline`.
- **`d5_*`** (markov_switching_var): `d5_msvar_crisis_proba`, `d5_msvar_filtered_vs_smoothed`, `d5_msvar_sp500_regimes`, `d5_msvar_transition`.
- **`d06_*`** (garch_t_vol): `d06_coverage`, `d06_sigma_threshold`, `d06_sp500_regimes`, `d06_timeline`.
- **`d07_*`** (changepoint_online): `d07_coverage`, `d07_cusum_changepoints`, `d07_leadlag`, `d07_robust_vs_gaussian`, `d07_sp500_regimes`, `d07_timeline`.
- **`d8_*`** (hmm_tstudent_4s): `d8_tstudent_bic_vs_d4`, `d8_tstudent_crisis_proba`, `d8_tstudent_sp500_regimes`, `d8_tstudent_transition`.
- **`d09_*`** (jump_model): `d09_coverage`, `d09_persistence_vs_d3`, `d09_sp500_regimes`, `d09_timeline`.
- **`d10_*`** (turbulence_mahalanobis): `d10_coverage`, `d10_sp500_regimes`, `d10_timeline`, `d10_turbulence_threshold`.
- **`d11_*`** (msgarch_regime, negativo): `d11_coverage`, `d11_prob_timeline`, `d11_sp500_regimes`.
- **`d12_*`** (deep_ae_regime, negativo): `d12_ablative_compare`, `d12_latent_recon`, `d12_sp500_regimes`, `d12_timeline`.
- **`metrics_04_*`** (HMM gaussiano D4, fase intermedia): `metrics_04_coverage_compare`, `metrics_04_crisis_proba`, `metrics_04_durations`, `metrics_04_price_by_regime`, `metrics_04_transition_timeline`.

---

## Cifras del CSV potencialmente sospechosas (reportadas, NO alteradas)

1. **Lead/lag saturados en −252/−251/−250/−220/−219** (Tabla 6 y `fase4_leadlag.png`): son
   valores censurados al lookback del evento, no antelaciones literales. Riesgo de
   sobreinterpretación; ya señalado con nota.
2. **`fase4_rank_heatmap.png` mezcla ventanas** en la columna de cobertura (confirmado por
   revisión académica): no es un error del CSV sino de presentación, pero afecta a cualquier
   ranking que se cite de esa figura.
3. **`switching_rate` no es 1/`mean_regime_duration`** exacto en varios detectores
   (p.ej. D7: switching 0.0022 vs 1/435.68 ≈ 0.0023; D6: 0.0141 vs 1/70.15 ≈ 0.0143).
   Las pequeñas discrepancias sugieren que se midieron sobre poblaciones/definiciones
   ligeramente distintas (transiciones brutas vs duración media de tramos). No se ha tocado;
   citar cada métrica por separado, no derivar una de la otra.
4. **D11 (msgarch) tiene `cov_GFC_2008 = 0.000`** pese a ser ventana larga (1991→2026):
   coherente con su clasificación "exploratorio-negativo" (no detecta la GFC), pero conviene
   no confundir el 0.000 con `NaN`: aquí SÍ vio 2008 y falló, no es ausencia de dato.
5. **`false_alarm_rate` global muy alto en varios (D6 0.845, D7 0.867, D11 0.949)** frente a
   sus bajas `fa_*` en las trampas concretas: la tasa global parece medirse sobre todo el
   periodo "no-crisis" (incluye régimen de alta vol prolongada), no sólo las dos ventanas-trampa.
   Definiciones distintas; citarlas como métricas separadas.
