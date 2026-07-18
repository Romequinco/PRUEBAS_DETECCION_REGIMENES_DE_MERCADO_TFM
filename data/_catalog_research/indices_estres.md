# Categoría: Índices de estrés / régimen (VALIDACIÓN) — estado del arte de datos (verificado)

Investigador: agente **Índices de estrés (validación)**. Foco: **OFR Financial Stress Index**,
**NFCI/ANFCI** (Chicago Fed), **STLFSI4** (St. Louis Fed), **KCFSI** (Kansas City Fed),
**CFSI** (Cleveland Fed, histórico) y **VIX** como termómetro laxo. Estas series son
**ground truth laxo de régimen** (`rol=validation`): índices de estrés ya computados por bancos
centrales / OFR que sirven para *etiquetar* y *validar* los regímenes que detecte el modelo, no como
features de entrada. Se complementan con **fechas de crisis** (indicadores de recesión NBER y
probabilidades de recesión) que son el ground truth de eventos.

Todas las verificaciones se hicieron **de verdad** el **2026-07-18**: descarga real vía FRED API
(clave `FRED_API_KEY` del `.env`, sin imprimirla) y descarga directa del CSV de OFR
(financialresearch.gov). Reporto la fecha de inicio y el último dato **observados**.

---

## Resumen ejecutivo de lo verificado

**La joya para validación es el OFR FSI.** No está en FRED (probé `OFRFSI` → HTTP 400). Se baja
directo del CSV público de OFR: **diario, 6.715 filas válidas, desde 2000-01-03 hasta 2026-07-15,
vivo** (lag ~1 día hábil). Además viene **descompuesto**: 5 subíndices por **categoría** (Credit,
Equity valuation, Safe assets, Funding, Volatility) y 3 por **región** (United States, Other advanced
economies, Emerging markets) en el mismo fichero. Es el índice de estrés más granular y de mayor
frecuencia disponible gratis.

**El de historia más profunda es el NFCI (Chicago Fed)**: semanal desde **1971-01-08** (2.897 obs),
vivo. Viene con su versión ajustada por el ciclo macro (**ANFCI**) y 4 subíndices
(**Risk, Credit, Leverage, Nonfinancial Leverage**), todos con el mismo arranque de 1971. Es la mejor
etiqueta de estrés que cubre casi toda la Pista B e incluso parte del solape con la Pista A profunda.

**STLFSI4** (St. Louis Fed, versión actual) es semanal desde **1993-12-31**, vivo. **OJO**: las
versiones anteriores están **descontinuadas** y NO deben mezclarse ciegamente porque cambia la cesta
de componentes: `STLFSI` (v1, fin 2020-03-13), `STLFSI2` (fin 2022-01-07, salida por transición LIBOR),
`STLFSI3` (fin 2022-10-28). Para histórico continuo usar **STLFSI4** (recalculado hacia atrás hasta 1993).

**KCFSI** (Kansas City Fed) es **mensual** desde **1990-02-01**, vivo (437 obs). Baja frecuencia pero
cubre 1990+ y es un ground truth clásico de estrés en la literatura.

**CFSI (Cleveland Financial Stress Index)** está **descontinuado** (último 2016-05-05) pero tiene una
propiedad única: **diario y desde 1991-09-25** (8.990 obs). Como ground truth *histórico* diario
1991-2016 es valioso; no sirve para tiempo real. Lo marco `rol=fallback`.

**Ground truth de crisis (fechas):** el canónico es el **indicador de recesión NBER**: `USRECD`
(diario, desde 1854, 62.674 obs, vivo) y `USREC`/`USRECM` (mensual). Contiene **11 recesiones desde 1950**
(1953, 1957, 1960, 1970, 1973, 1980, 1981, 1990, 2001, 2008, 2020) → cubre la promesa de "10+ crisis"
de la Pista A. Complemento con probabilidades de recesión: `RECPROUSM156N` (smoothed, Chauvet-Piger,
mensual desde 1967) y la **regla de Sahm** (`SAHMREALTIME` desde 1959, `SAHMCURRENT` desde 1949).

**VIX** (`VIXCLS`, FRED, diario 1990+) lo incluyo aquí sólo como **referencia de validación laxa**
(umbral/percentil de VIX ≈ etiqueta risk-off). Su tratamiento completo como *feature* está en el
catálogo de **volatilidad**; aquí no lo dupolico salvo por su rol de ground truth.

**No verificado / lo que no encontré:**
- OFR FSI **no está en FRED** (400 en `OFRFSI`); sólo por CSV directo de OFR. El CSV funciona con
  User-Agent de navegador; sin UA algunos paths dan 403 (el path bueno `.../data/fsi.csv` sí responde).
- El **Anxious Index** del Philly Fed (prob. de recesión de la Survey of Professional Forecasters,
  trimestral) es descargable (Excel) pero **no lo verifiqué** en esta pasada; lo dejo como pista.
- Las versiones viejas de STLFSI (v1/v2/v3) existen pero están congeladas; útiles sólo para auditoría
  de metodología, no para features vivas.

---

## Detalle por serie (evidencia)

| serie | fuente | inicio verificado | fin/estado | freq | rol |
|---|---|---|---|---|---|
| OFR FSI (+8 subíndices) | OFR CSV directo | 2000-01-03 | vivo (upd 2026-07-15) | diaria | validation ★ |
| NFCI | FRED NFCI | 1971-01-08 | vivo (upd 2026-07-15) | semanal | validation ★ |
| ANFCI | FRED ANFCI | 1971-01-08 | vivo | semanal | validation |
| NFCI Risk/Credit/Leverage/NonfinLev | FRED | 1971-01-08 | vivo | semanal | validation |
| STLFSI4 | FRED STLFSI4 | 1993-12-31 | vivo (upd 2026-07-15) | semanal | validation |
| KCFSI | FRED KCFSI | 1990-02-01 | vivo (upd 2026-07-05) | mensual | validation |
| CFSI (Cleveland) | FRED CFSI | 1991-09-25 | descontinuado 2016-05-05 | diaria | fallback |
| USRECD (NBER diario) | FRED USRECD | 1854-12-01 | vivo (upd 2026-07-06) | diaria | validation ★ |
| USREC (NBER mensual) | FRED USREC | 1854-12-01 | vivo (upd 2026-07-01) | mensual | validation |
| RECPROUSM156N (prob. recesión) | FRED | 1967-06-01 | vivo | mensual | validation |
| SAHMREALTIME (regla Sahm) | FRED | 1959-12-01 | vivo | mensual | validation |
| VIXCLS (referencia laxa) | FRED VIXCLS | 1990-01-02 | vivo | diaria | validation |

Notas metodológicas para la capa de validación:
- **Convención de signo**: en NFCI/ANFCI/STLFSI4/KCFSI/OFR-FSI, **valores positivos = estrés / condiciones
  más apretadas** y negativos = condiciones laxas (media 0). Al etiquetar régimen, umbral típico:
  `>0` estrés moderado, `>+1σ`/`>+2` estrés agudo (crisis).
- **Frecuencias mezcladas**: OFR-FSI, CFSI, USRECD y VIX son diarios; NFCI/ANFCI/STLFSI4 semanales
  (miércoles/viernes); KCFSI, USREC, Sahm, RECPRO mensuales. Para validar contra features diarias hay que
  **reindexar con forward-fill causal** (nunca interpolar hacia atrás).
- **Ground truth de eventos**: USRECD marca los intervalos NBER (pico→valle). Para "crisis" en sentido
  de mercado (que a veces preceden a la recesión), combinar con umbral alto de OFR-FSI/NFCI.
- **No usar como feature**: estas series ya son "la respuesta"; usarlas de input sería fuga de información.

```yaml
series_indices_estres:
  # ============ OFR FINANCIAL STRESS INDEX (la joya: diario, 2000+, con subíndices) ============
  - nombre_interno: OFR_FSI
    descripcion: "OFR Financial Stress Index (headline). Índice de estrés global construido de 33 variables de mercado, media 0. Diario, con descomposición por categoría (Credit/Equity/Safe assets/Funding/Volatility) y región (US/Other advanced/EM) en el mismo CSV."
    fuente: ofr
    id: "fsi.csv:OFR FSI"
    auth: none
    inicio_verificado: "2000-01-03"
    granularidad: diaria
    pista: validacion
    rol: validation
    relevancia_regimen: "Ground truth de estrés más granular y de mayor frecuencia libre; positivo=estrés. Etiqueta risk-off/crisis 2000+ (dotcom, GFC, 2011, 2015, 2018, 2020, 2022, SVB)."
    verificado: true
    evidencia: "GET financialresearch.gov/.../data/fsi.csv (UA navegador) -> 6715 filas validas, first 2000-01-03 OFR_FSI=2.14, last 2026-07-15 OFR_FSI=-2.615. Cols: Date,OFR FSI,Credit,Equity valuation,Safe assets,Funding,Volatility,United States,Other advanced economies,Emerging markets. No esta en FRED (OFRFSI -> HTTP 400)."
    url: "https://www.financialresearch.gov/financial-stress-index/data/fsi.csv"

  - nombre_interno: OFR_FSI_CREDIT
    descripcion: "Subíndice de contribución de CRÉDITO al OFR FSI (columna 'Credit' del mismo CSV)."
    fuente: ofr
    id: "fsi.csv:Credit"
    auth: none
    inicio_verificado: "2000-01-03"
    granularidad: diaria
    pista: validacion
    rol: validation
    relevancia_regimen: "Descompone cuánto del estrés viene de crédito (spreads, CDS). Valida regímenes de estrés crediticio."
    verificado: true
    evidencia: "Misma descarga OFR fsi.csv, columna 'Credit'. first 2000-01-03=0.54, last 2026-07-15=-1.169. 6715 filas."
    url: "https://www.financialresearch.gov/financial-stress-index/data/fsi.csv"

  - nombre_interno: OFR_FSI_FUNDING
    descripcion: "Subíndice de contribución de FUNDING/liquidez al OFR FSI (columna 'Funding')."
    fuente: ofr
    id: "fsi.csv:Funding"
    auth: none
    inicio_verificado: "2000-01-03"
    granularidad: diaria
    pista: validacion
    rol: validation
    relevancia_regimen: "Estrés de financiación (repo, TED, FX swaps). Discrimina crisis de liquidez (2008, mar-2020) de otras."
    verificado: true
    evidencia: "Misma descarga OFR fsi.csv, columna 'Funding'. first 2000-01-03=0.472, last 2026-07-15=-0.082. 6715 filas."
    url: "https://www.financialresearch.gov/financial-stress-index/data/fsi.csv"

  - nombre_interno: OFR_FSI_VOLATILITY
    descripcion: "Subíndice de contribución de VOLATILIDAD al OFR FSI (columna 'Volatility')."
    fuente: ofr
    id: "fsi.csv:Volatility"
    auth: none
    inicio_verificado: "2000-01-03"
    granularidad: diaria
    pista: validacion
    rol: validation
    relevancia_regimen: "Parte del estrés atribuible a vol de mercado; cruza con VIX/MOVE del catálogo de volatilidad."
    verificado: true
    evidencia: "Misma descarga OFR fsi.csv, columna 'Volatility'. first 2000-01-03=0.509, last 2026-07-15=-0.454. 6715 filas."
    url: "https://www.financialresearch.gov/financial-stress-index/data/fsi.csv"

  - nombre_interno: OFR_FSI_US
    descripcion: "Contribución regional de ESTADOS UNIDOS al OFR FSI (columna 'United States')."
    fuente: ofr
    id: "fsi.csv:United States"
    auth: none
    inicio_verificado: "2000-01-03"
    granularidad: diaria
    pista: validacion
    rol: validation
    relevancia_regimen: "Aísla el estrés doméstico US del global; útil si el modelo se centra en S&P500."
    verificado: true
    evidencia: "Misma descarga OFR fsi.csv, columna 'United States'. first 2000-01-03=1.769, last 2026-07-15=-1.327. 6715 filas."
    url: "https://www.financialresearch.gov/financial-stress-index/data/fsi.csv"

  # ============ CHICAGO FED NFCI (historia más profunda: semanal 1971+) ============
  - nombre_interno: NFCI
    descripcion: "Chicago Fed National Financial Conditions Index. 105 indicadores de dinero, deuda y equity. Media 0, positivo=condiciones apretadas/estrés."
    fuente: fred
    id: "NFCI"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-08"
    granularidad: semanal
    pista: validacion
    rol: validation
    relevancia_regimen: "Ground truth de estrés de mayor historia (1971+); cubre casi toda la Pista B y solapa con crisis de Pista A (1973, 1980, 1987, 1990)."
    verificado: true
    evidencia: "FRED NFCI -> 2897 obs semanales, first 1971-01-08=0.598, last 2026-07-10=-0.538 (upd 2026-07-15)."
    url: "https://fred.stlouisfed.org/series/NFCI"

  - nombre_interno: ANFCI
    descripcion: "Chicago Fed Adjusted NFCI: NFCI ortogonalizado respecto a las condiciones económicas (aísla estrés financiero 'puro' del ciclo)."
    fuente: fred
    id: "ANFCI"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-08"
    granularidad: semanal
    pista: validacion
    rol: validation
    relevancia_regimen: "Etiqueta estrés financiero descontando el estado macro; mejor para separar régimen financiero de recesión."
    verificado: true
    evidencia: "FRED ANFCI -> 2897 obs, first 1971-01-08=0.587, last 2026-07-10=-0.535 (upd 2026-07-15)."
    url: "https://fred.stlouisfed.org/series/ANFCI"

  - nombre_interno: NFCI_RISK
    descripcion: "Subíndice de RIESGO del NFCI (volatilidad y funding risk)."
    fuente: fred
    id: "NFCIRISK"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-08"
    granularidad: semanal
    pista: validacion
    rol: validation
    relevancia_regimen: "Componente de riesgo/vol del estrés; el más correlacionado con VIX."
    verificado: true
    evidencia: "FRED NFCIRISK -> 2897 obs, first 1971-01-08=0.626, last 2026-07-10=-0.619 (upd 2026-07-15)."
    url: "https://fred.stlouisfed.org/series/NFCIRISK"

  - nombre_interno: NFCI_CREDIT
    descripcion: "Subíndice de CRÉDITO del NFCI (condiciones de crédito)."
    fuente: fred
    id: "NFCICREDIT"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-08"
    granularidad: semanal
    pista: validacion
    rol: validation
    relevancia_regimen: "Estrés crediticio de historia larga (1971+); valida regímenes de crédito."
    verificado: true
    evidencia: "FRED NFCICREDIT -> 2897 obs, first 1971-01-08=-1.105, last 2026-07-10=-0.044 (upd 2026-07-15)."
    url: "https://fred.stlouisfed.org/series/NFCICREDIT"

  - nombre_interno: NFCI_LEVERAGE
    descripcion: "Subíndice de APALANCAMIENTO del NFCI (leverage de deuda y equity)."
    fuente: fred
    id: "NFCILEVERAGE"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-08"
    granularidad: semanal
    pista: validacion
    rol: validation
    relevancia_regimen: "Leverage como pre-condición de fragilidad; sube antes de crisis (2007)."
    verificado: true
    evidencia: "FRED NFCILEVERAGE -> 2897 obs, first 1971-01-08=-1.019, last 2026-07-10=0.327 (upd 2026-07-15)."
    url: "https://fred.stlouisfed.org/series/NFCILEVERAGE"

  # ============ ST. LOUIS FED FSI (versión actual v4) ============
  - nombre_interno: STLFSI4
    descripcion: "St. Louis Fed Financial Stress Index, versión 4 (actual). 18 series de mercado, media 0, positivo=estrés."
    fuente: fred
    id: "STLFSI4"
    auth: FRED_API_KEY
    inicio_verificado: "1993-12-31"
    granularidad: semanal
    pista: validacion
    rol: validation
    relevancia_regimen: "Ground truth de estrés semanal 1993+; captura LTCM 1998, GFC 2008, 2011, 2020, 2022. Usar v4 (no v1/v2/v3, descontinuadas y con cesta distinta)."
    verificado: true
    evidencia: "FRED STLFSI4 -> 1698 obs, first 1993-12-31=-0.291, last 2026-07-10=-0.882 (upd 2026-07-15). Versiones viejas verificadas descontinuadas: STLFSI(v1) fin 2020-03-13, STLFSI2 fin 2022-01-07, STLFSI3 fin 2022-10-28."
    url: "https://fred.stlouisfed.org/series/STLFSI4"

  # ============ KANSAS CITY FED FSI ============
  - nombre_interno: KCFSI
    descripcion: "Kansas City Financial Stress Index. 11 variables (spreads, correlaciones, vol). Mensual, media 0, positivo=estrés."
    fuente: fred
    id: "KCFSI"
    auth: FRED_API_KEY
    inicio_verificado: "1990-02-01"
    granularidad: mensual
    pista: validacion
    rol: validation
    relevancia_regimen: "Ground truth clásico de la literatura de estrés; mensual desde 1990. Baja frecuencia -> forward-fill causal para alinear con features diarias."
    verificado: true
    evidencia: "FRED KCFSI -> 437 obs mensuales, first 1990-02-01=0.372, last 2026-06-01=-0.763 (upd 2026-07-05)."
    url: "https://fred.stlouisfed.org/series/KCFSI"

  # ============ CLEVELAND FSI (histórico diario, DESCONTINUADO) ============
  - nombre_interno: CFSI
    descripcion: "Cleveland Financial Stress Index. Índice diario de estrés (mercados de crédito, equity, FX, funding, interbancario). DESCONTINUADO en 2016."
    fuente: fred
    id: "CFSI"
    auth: FRED_API_KEY
    inicio_verificado: "1991-09-25"
    granularidad: diaria
    pista: validacion
    rol: fallback
    relevancia_regimen: "Único FSI DIARIO con historia larga (1991-2016); ground truth diario para validar el periodo pre-OFR (que arranca en 2000). No sirve en tiempo real (congelado)."
    verificado: true
    evidencia: "FRED CFSI -> 8990 obs diarias, first 1991-09-25=-1.05, last 2016-05-05=1.47 (DESCONTINUADO, upd 2016-05-06)."
    url: "https://fred.stlouisfed.org/series/CFSI"

  # ============ GROUND TRUTH DE CRISIS: FECHAS NBER + PROBABILIDADES ============
  - nombre_interno: NBER_RECESSION_DAILY
    descripcion: "Indicador de recesión NBER, versión diaria (1=recesión pico-valle, 0=expansión). Ground truth canónico de fechas de crisis económica."
    fuente: fred
    id: "USRECD"
    auth: FRED_API_KEY
    inicio_verificado: "1854-12-01"
    granularidad: diaria
    pista: validacion
    rol: validation
    relevancia_regimen: "Etiqueta binaria de recesión más usada; 11 recesiones desde 1950 (1953,57,60,70,73,80,81,90,2001,2008,2020) -> cubre 10+ crisis de Pista A. Diario, alineable con S&P500."
    verificado: true
    evidencia: "FRED USRECD -> 62674 obs, first 1854-12-01=1, last 2026-07-05=0 (upd 2026-07-06). Transiciones 0->1 desde 1950 = 11 recesiones."
    url: "https://fred.stlouisfed.org/series/USRECD"

  - nombre_interno: NBER_RECESSION_MONTHLY
    descripcion: "Indicador de recesión NBER mensual (mid-month, 1/0)."
    fuente: fred
    id: "USREC"
    auth: FRED_API_KEY
    inicio_verificado: "1854-12-01"
    granularidad: mensual
    pista: validacion
    rol: validation
    relevancia_regimen: "Versión mensual del ground truth NBER; conveniente para features mensuales/macro."
    verificado: true
    evidencia: "FRED USREC -> 2059 obs, first 1854-12-01=1, last 2026-06-01=0 (upd 2026-07-01)."
    url: "https://fred.stlouisfed.org/series/USREC"

  - nombre_interno: RECESSION_PROB_SMOOTHED
    descripcion: "Smoothed U.S. Recession Probabilities (modelo dinámico-factor Markov-switching de Chauvet-Piger). Probabilidad [0,1] mensual."
    fuente: fred
    id: "RECPROUSM156N"
    auth: FRED_API_KEY
    inicio_verificado: "1967-06-01"
    granularidad: mensual
    pista: validacion
    rol: validation
    relevancia_regimen: "Ground truth PROBABILÍSTICO (no binario) de régimen recesivo; ideal para comparar con salidas de un HMM/Markov-switching propio."
    verificado: true
    evidencia: "FRED RECPROUSM156N -> 708 obs, first 1967-06-01=0.92, last 2026-05-01=0.54 (upd 2026-07-01)."
    url: "https://fred.stlouisfed.org/series/RECPROUSM156N"

  - nombre_interno: SAHM_RULE_REALTIME
    descripcion: "Real-time Sahm Rule Recession Indicator (basado en desempleo). Dispara cuando la media 3m del paro sube >=0.5pp sobre su mínimo de 12m."
    fuente: fred
    id: "SAHMREALTIME"
    auth: FRED_API_KEY
    inicio_verificado: "1959-12-01"
    granularidad: mensual
    pista: validacion
    rol: validation
    relevancia_regimen: "Indicador de inicio de recesión de baja latencia; ground truth complementario a NBER (que se publica con retraso)."
    verificado: true
    evidencia: "FRED SAHMREALTIME -> 798 obs, first 1959-12-01=0.77, last 2026-06-01=0.07 (upd 2026-07-02). Variante SAHMCURRENT (revisada) verificada 1949-03-01+."
    url: "https://fred.stlouisfed.org/series/SAHMREALTIME"

  # ============ VIX como referencia laxa (feature completa en catálogo volatilidad) ============
  - nombre_interno: VIX_VALIDATION_REF
    descripcion: "CBOE VIX (FRED VIXCLS) usado aquí SÓLO como ground truth laxo: umbral/percentil de VIX ~ etiqueta risk-off. Su ficha completa como feature está en el catálogo de volatilidad."
    fuente: fred
    id: "VIXCLS"
    auth: FRED_API_KEY
    inicio_verificado: "1990-01-02"
    granularidad: diaria
    pista: validacion
    rol: validation
    relevancia_regimen: "Etiqueta binaria simple (VIX>20 estrés, >30 crisis); referencia diaria 1990+ para sanity-check de los FSI. No usar simultáneamente como feature Y como label (fuga)."
    verificado: true
    evidencia: "FRED VIXCLS -> 9231 obs, first 1990-01-02=17.24, last 2026-07-16=16.73 (upd 2026-07-17). Coincide con catálogo volatilidad."
    url: "https://fred.stlouisfed.org/series/VIXCLS"

  # ============ FALLBACK / NO VERIFICADO ============
  - nombre_interno: PHILLY_ANXIOUS_INDEX
    descripcion: "Anxious Index del Philadelphia Fed: probabilidad media de que el PIB caiga el próximo trimestre (Survey of Professional Forecasters). Trimestral desde 1968-Q4."
    fuente: academico
    id: "spf-anxious-index"
    auth: none
    inicio_verificado: null
    granularidad: mensual
    pista: validacion
    rol: fallback
    relevancia_regimen: "Ground truth de sentimiento de recesión (expectativas de forecasters); complementaria a NBER. Trimestral (baja frecuencia)."
    verificado: false
    evidencia: "NO verificado en esta pasada. Descargable como Excel desde philadelphiafed.org/surveys-and-data/real-time-data-research/anxious-index (no probado end-to-end aqui)."
    url: "https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/anxious-index"
```

---

## Recomendación de priorización para validación

1. **Etiqueta primaria de estrés (diaria, 2000+):** `OFR_FSI` + sus subíndices Credit/Funding/Volatility.
   Es la de mayor frecuencia y granularidad; ideal para etiquetar día a día contra features diarias.
2. **Etiqueta de historia profunda (semanal, 1971+):** `NFCI` (+`ANFCI` para separar estrés del ciclo).
   Cubre casi toda la Pista B y solapa con crisis de la Pista A; el mejor puente temporal.
3. **Ground truth de eventos de crisis:** `USRECD` (diario, 1854+, 11 recesiones desde 1950) como
   etiqueta binaria; `RECPROUSM156N` como versión probabilística para comparar con un Markov-switching/HMM.
4. **Complementos:** `STLFSI4` (semanal 1993+) y `KCFSI` (mensual 1990+) como segundo/tercer voto de estrés;
   `CFSI` sólo para rellenar el hueco diario 1991-2000 (pre-OFR), asumiendo que está congelado en 2016.
5. **Reglas de uso:** (a) positivo=estrés en todos los FSI; (b) reindexar por forward-fill **causal** al
   pasar de semanal/mensual a diario; (c) **nunca** usar un índice de estrés a la vez como feature de
   entrada y como label de validación (fuga de información); (d) para "crisis de mercado" que preceden a la
   recesión oficial, combinar umbral alto de OFR-FSI/NFCI con las ventanas NBER.

## Fuentes y métodos de verificación
- **FRED API** (`series` + `series/observations`, JSON), clave `FRED_API_KEY` del `.env` (no impresa).
  Verificadas 18 IDs; reportado n, primer/último dato y `last_updated`.
- **OFR** (financialresearch.gov): descarga directa del CSV `fsi.csv` con User-Agent de navegador
  (sin UA algunos paths dan 403). No está en FRED (`OFRFSI` → HTTP 400).
- Fecha de verificación: **2026-07-18**.
