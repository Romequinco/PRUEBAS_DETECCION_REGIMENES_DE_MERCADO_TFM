# Categoría: Crédito — estado del arte de datos (verificado)

Investigador: agente **Crédito**. Foco encargado: HY OAS, IG OAS, CCC OAS, familia HYG/LQD/JNK,
TED spread y el spread HYG−IEF, tirando de **FRED con API key** para las OAS (que era *el hueco
histórico de v1*). El crédito es, junto con la volatilidad, la feature más discriminante de régimen:
los spreads de crédito se ensanchan en risk-off de forma persistente y con memoria larga.

Todas las verificaciones se hicieron **de verdad** el 2026-07-18: FRED API con `FRED_API_KEY` del
`.env` (sin imprimir la clave), yfinance `period='max'` y pruebas reales a Stooq/Nasdaq. Reporto la
fecha de inicio **observada**, no la de marketing.

---

## ⚠️ Hallazgo central: el "hueco histórico" de las OAS es REAL y se agravó en abril 2026

Las series ICE BofA OAS de FRED (todas las `BAML...`) **ya NO sirven su historia completa**. FRED las
limitó a una **ventana rodante de 3 años** por licencia de ICE Data Indices. Verificado tres veces:

- `BAMLH0A0HYM2` (HY OAS): `observation_start = 2023-07-18` en la metadata (= hoy − 3 años exactos),
  787 obs válidas, `first 2023-07-18:3.90 → last 2026-07-16:2.71`, viva (upd 2026-07-17).
- **Forzar `observation_start=1996-01-01` en la API NO devuelve nada más**: siguen 787 filas desde
  2023-07-18. La API está capada en origen, no es un problema de parámetros.
- **El mismo patrón en TODAS las OAS** (IG, CCC, BB, B, BBB, AAA, EM, Euro HY): todas arrancan
  2023-07-18. En cambio las series **Moody's** (`BAA`, `DBAA`, `BAA10Y`) sí devuelven historia
  completa desde 1919/1986 con mi mismo código → la restricción es **específica de ICE BofA**, no un
  bug mío.
- Confirmado por fuente: *"Starting in April 2026, this series will only include 3 years of
  observations. For more data, go to the source."* (nota oficial en la página FRED de `BAMLH0A0HYM2`).

**Implicación para el TFM:** con fuentes gratis y vivas **no se puede reconstruir la historia larga de
las OAS por rating** (2008, 2011, 2015-16, 2020 quedan fuera de la ventana). Esto afecta sobre todo a
la **CCC OAS**, que es el mejor termómetro de estrés de crédito y del que **ya no hay histórico
profundo gratis**. Dos vías honestas para tapar el hueco:

1. **Sustituto de spine profundo (recomendado, verificado):** el **spread Baa−Aaa de Moody's**
   (`DBAA−DAAA`, diario desde 1986) y **`BAA10Y`** (Baa − Treasury 10y, diario desde 1986), más
   `BAA`/`AAA` **mensuales desde 1919**. Es crédito grado de inversión (no HY), pero es *el* índice de
   riesgo de crédito con memoria de +100 años y captura todas las crisis. Sanity check real:
   pico Baa−Aaa **3.50 el 2008-12-03**, 1.99 en 2020-04, 1.50 en 2016-01, 1.46 en 2011-10 (coherente).
   `BAA10Y` pico **6.16 en 2008** y 4.31 en 2020.
2. **Proxy de mercado para HY (verificado):** el par **HYG vs IEF** (o HYG/LQD). Rebasando HYG/IEF a
   100 en 2007: mínimo **56.0 el 2008-12-11** y **76.2 el 2020-03-23** → captura 2008 y COVID con
   nitidez. Cubre desde 2007 (HY) y 2002 (IG con LQD). Es total-return, no OAS, pero se mueve inverso
   al spread y es libre + vivo.
3. **(No verificado) Snapshot estático pre-2026:** empalmar un CSV archivado de las OAS
   (repos GitHub de market-data / datasets de "credit spreads recession risk") con la ventana viva de
   3 años de FRED. Es la única forma de recuperar 1996–2023 por rating, pero es **una foto estática, no
   viva**, y no la pude descargar/verificar en este entorno. Marcada `verificado=false`.

---

## Resumen ejecutivo de lo verificado

**Spine de crédito profundo (Pista A, 1919/1986+, todo FRED y vivo):** `BAA`/`AAA` mensual 1919,
`DBAA`/`DAAA` diario 1986, `BAA10Y`/`AAA10Y` diario 1986, `BAAFFM` mensual 1954. El spread **Baa−Aaa**
y **Baa−10y** son el núcleo honesto de riesgo de crédito con historia de crisis larga.

**Panel de crédito moderno (Pista B):**
- OAS por rating (FRED, **solo 3 años**): HY, IG, CCC, BB, B, BBB, AAA/AA/A, EM, Euro HY. Útiles como
  features **vivas** (nivel y momentum a 3 años) pero **sin profundidad histórica**.
- Proxies ETF (yfinance, historia completa y viva): **HYG** (2007-04-11), **JNK** (2007-12-04),
  **LQD** (2002-07-30), **IEF** (2002-07-30, la pata Treasury del spread), **EMB** (2007-12),
  **BKLN** (loans, 2011), **AGG** (2003), **VCIT**/**SHYG**/**HYD** (fallbacks).
- Spread derivado **HYG−IEF** (o HYG/LQD): proxy de estrés HY, 2007+.

**Funding / TED:** `TEDRATE` completo **1986-01-02 → 2022-01-21** pero **DESCONTINUADO** (fin de LIBOR).
No hay continuación gratis limpia post-2022 (el sustituto natural, spreads SOFR-OIS, cae en la
categoría *liquidez*). Sirve como historia de estrés de financiación 1986-2022 (picos 1987, 1998,
2008).

**Validación (crédito-específica):** `NFCICREDIT` (Chicago Fed, subíndice de crédito, **semanal desde
1971**, +50 años, vivo) y `STLFSI4` (St. Louis Fed Financial Stress Index, semanal 1993+, vivo). Son
índices de estrés ya hechos → ground-truth laxo para el detector.

**Fuentes que NO funcionan hoy (probadas):**
- **Stooq**: bloqueado por challenge JavaScript de proof-of-work (igual que reportó el agente de
  volatilidad). `stooq.com/q/d/l/?s=hyg.us` devuelve el script de verificación, no el CSV. Fallback
  no usable sin navegador.
- **Nasdaq Data Link** legacy BofA database (`ML/HYOAS`, `ML/CCCOAS`): **HTTP 403** (deprecado/
  bloqueado). No recupera la historia OAS.
- **fredgraph.csv** (host web `fred.stlouisfed.org`): read-timeout repetido desde este entorno (el
  host **API** `api.stlouisfed.org` sí funciona). Y aunque respondiera, sirve la misma ventana capada
  de 3 años.

---

## Detalle por serie (evidencia)

| serie | fuente | inicio verificado | fin/estado | nota |
|---|---|---|---|---|
| Baa yield | FRED BAA | 1919-01-01 (M) | vivo | spine crédito profundo |
| Aaa yield | FRED AAA | 1919-01-01 (M) | vivo | spine |
| Baa yield diario | FRED DBAA | 1986-01-02 | vivo | spine diario |
| Aaa yield diario | FRED DAAA | 1983-01-03 | vivo | spine diario |
| **Baa−Aaa spread** | calc DBAA−DAAA | 1986-01-02 | vivo | **núcleo crédito, pico 3.50 en 2008** |
| Baa−10y | FRED BAA10Y | 1986-01-02 | vivo | spread crédito vs Treasury, pico 6.16 en 2008 |
| Aaa−10y | FRED AAA10Y | 1983-01-03 | vivo | spread IG alto |
| Baa−FedFunds | FRED BAAFFM | 1954-07-01 (M) | vivo | crédito vs política monetaria |
| HY OAS | FRED BAMLH0A0HYM2 | 2023-07-18 | vivo, **solo 3y** | core HY, histórico perdido |
| IG OAS | FRED BAMLC0A0CM | 2023-07-18 | vivo, **solo 3y** | core IG |
| CCC OAS | FRED BAMLH0A3HYC | 2023-07-18 | vivo, **solo 3y** | mejor señal estrés, sin histórico |
| BB OAS | FRED BAMLH0A1HYBB | 2023-07-18 | vivo, 3y | HY alta calidad |
| B OAS | FRED BAMLH0A2HYB | 2023-07-18 | vivo, 3y | HY media |
| BBB OAS | FRED BAMLC0A4CBBB | 2023-07-18 | vivo, 3y | IG más cíclico (frontera HY) |
| A/AA/AAA OAS | FRED BAMLC0A3CA/…2CAA/…1CAAA | 2023-07-18 | vivo, 3y | buckets IG |
| EM Corp OAS | FRED BAMLEMCBPIOAS | 2023-07-18 | vivo, 3y | crédito emergente |
| Euro HY OAS | FRED BAMLHE00EHYIOAS | 2023-07-18 | vivo, 3y | crédito europeo |
| HY yield | FRED BAMLH0A0HYM2EY | 2023-07-18 | vivo, 3y | nivel de yield HY |
| IG yield | FRED BAMLC0A0CMEY | 2023-07-18 | vivo, 3y | nivel de yield IG |
| HYG | yfinance HYG | 2007-04-11 | vivo | ETF HY, proxy estrés |
| JNK | yfinance JNK | 2007-12-04 | vivo | ETF HY alternativo |
| LQD | yfinance LQD | 2002-07-30 | vivo | ETF IG |
| IEF | yfinance IEF | 2002-07-30 | vivo | Treasury 7-10y (pata del spread) |
| **HYG−IEF** | calc yfinance | 2007-04-11 | vivo | **proxy estrés HY, mín 56 en 2008** |
| EMB | yfinance EMB | 2007-12-19 | vivo | ETF deuda EM |
| BKLN | yfinance BKLN | 2011-03-03 | vivo | leveraged loans (crédito flotante) |
| AGG | yfinance AGG | 2003-09-29 | vivo | agregado bonos US |
| VCIT/SHYG/HYD | yfinance | 2009/2013/2009 | vivo | fallbacks IG/HY/muni-HY |
| TED spread | FRED TEDRATE | 1986-01-02 | **descont. 2022-01-21** | funding stress, sin sucesor gratis |
| NFCI Credit | FRED NFCICREDIT | 1971-01-08 (W) | vivo | validación crédito, +50y |
| StLouis FSI | FRED STLFSI4 | 1993-12-31 (W) | vivo | validación estrés |

**Features derivadas recomendadas (causales/expanding):**
- **Baa−Aaa** y **Baa−10y**: nivel + Δ (memoria larga de régimen de crédito, 1986+).
- **HYG/IEF** y **HYG/LQD**: ratios de total-return; caídas persistentes = risk-off de crédito.
- **CCC−BB OAS** (compresión/dispersión de calidad dentro de HY): potente **pero solo 3 años**.
- **HY OAS − IG OAS** (prima de riesgo HY): solo 3 años vía FRED; proxy largo = HYG vs LQD.

```yaml
series_credito:
  # ============ SPINE PROFUNDO (FRED, historia completa, vivo) ============
  - nombre_interno: MOODYS_BAA_AAA_SPREAD
    descripcion: "Spread Baa−Aaa de Moody's (calc. DBAA−DAAA). Núcleo honesto de riesgo de crédito con memoria larga; sustituto de las OAS para historia profunda."
    fuente: fred
    id: "DBAA,DAAA"
    auth: FRED_API_KEY
    inicio_verificado: "1986-01-02"
    granularidad: diaria
    pista: ambas
    rol: spine
    relevancia_regimen: "Spread de crédito IG con +40 años diarios; se ensancha en risk-off persistente. Discriminador de régimen de crédito núcleo."
    verificado: true
    evidencia: "FRED DBAA (10170 obs, 1986-01-02:11.38) − DAAA -> spread diario 10170 pts; pico 3.50 el 2008-12-03, 1.99 en 2020-04, 1.46 en 2011-10. Coherente."
    url: "https://fred.stlouisfed.org/series/DBAA"

  - nombre_interno: BAA10Y
    descripcion: "Moody's Baa Corporate Bond menos Treasury 10y. Spread de crédito vs tipo libre de riesgo."
    fuente: fred
    id: "BAA10Y"
    auth: FRED_API_KEY
    inicio_verificado: "1986-01-02"
    granularidad: diaria
    pista: ambas
    rol: spine
    relevancia_regimen: "Riesgo de crédito ajustado por nivel de tipos; pico 6.16 en 2008 y 4.31 en 2020. Clásico predictor de recesión."
    verificado: true
    evidencia: "FRED BAA10Y -> 10134 obs, first 1986-01-02:2.34 last 2026-07-16:1.59 (upd 2026-07-17). Max 2008=6.16, 2020=4.31."
    url: "https://fred.stlouisfed.org/series/BAA10Y"

  - nombre_interno: MOODYS_BAA
    descripcion: "Moody's Seasoned Baa Corporate Bond Yield (mensual). Rama de yield IG-bajo con historia desde 1919."
    fuente: fred
    id: "BAA"
    auth: FRED_API_KEY
    inicio_verificado: "1919-01-01"
    granularidad: mensual
    pista: A
    rol: spine
    relevancia_regimen: "+100 años; con AAA da el spread Baa−Aaa mensual que cubre la Gran Depresión (1930s) y todas las crisis modernas."
    verificado: true
    evidencia: "FRED BAA -> 1290 obs mensuales, first 1919-01-01:7.12 last 2026-06-01:6.0 (upd 2026-07-01)."
    url: "https://fred.stlouisfed.org/series/BAA"

  - nombre_interno: MOODYS_AAA
    descripcion: "Moody's Seasoned Aaa Corporate Bond Yield (mensual). Pata de alta calidad del spread de crédito profundo."
    fuente: fred
    id: "AAA"
    auth: FRED_API_KEY
    inicio_verificado: "1919-01-01"
    granularidad: mensual
    pista: A
    rol: spine
    relevancia_regimen: "Referencia IG-alto; Baa−Aaa mensual desde 1919 es el spread de crédito con historia más larga disponible."
    verificado: true
    evidencia: "FRED AAA -> 1290 obs mensuales, first 1919-01-01:5.35 last 2026-06-01:5.52 (upd 2026-07-01)."
    url: "https://fred.stlouisfed.org/series/AAA"

  - nombre_interno: DBAA
    descripcion: "Moody's Baa yield diario. Componente del spread diario y feature de nivel de yield crédito."
    fuente: fred
    id: "DBAA"
    auth: FRED_API_KEY
    inicio_verificado: "1986-01-02"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Nivel de yield de crédito IG-bajo, diario; base del spread Baa−Aaa/Baa−10y."
    verificado: true
    evidencia: "FRED DBAA -> 10170 obs, first 1986-01-02:11.38 last 2026-07-16:6.16 (upd 2026-07-17)."
    url: "https://fred.stlouisfed.org/series/DBAA"

  - nombre_interno: DAAA
    descripcion: "Moody's Aaa yield diario. Pata alta calidad del spread diario."
    fuente: fred
    id: "DAAA"
    auth: FRED_API_KEY
    inicio_verificado: "1983-01-03"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Referencia IG-alto diaria; spread frente a DBAA = riesgo de crédito puro."
    verificado: true
    evidencia: "FRED DAAA -> 10929 obs, first 1983-01-03:11.77 last 2026-07-16:5.73 (upd 2026-07-17)."
    url: "https://fred.stlouisfed.org/series/DAAA"

  - nombre_interno: BAAFFM
    descripcion: "Moody's Baa menos Fed Funds Rate (mensual). Spread crédito vs política monetaria."
    fuente: fred
    id: "BAAFFM"
    auth: FRED_API_KEY
    inicio_verificado: "1954-07-01"
    granularidad: mensual
    pista: A
    rol: enricher
    relevancia_regimen: "Estrés de crédito relativo al coste de financiación a un día; útil para régimen tardío de ciclo."
    verificado: true
    evidencia: "FRED BAAFFM -> 864 obs, first 1954-07-01:2.70 last 2026-06-01:2.37 (upd 2026-07-01)."
    url: "https://fred.stlouisfed.org/series/BAAFFM"

  # ============ OAS POR RATING (FRED, VIVO PERO SOLO 3 AÑOS) ============
  - nombre_interno: HY_OAS
    descripcion: "ICE BofA US High Yield Index OAS. El spread HY de referencia. OJO: FRED solo sirve ventana rodante de 3 años (restricción ICE, abril 2026)."
    fuente: fred
    id: "BAMLH0A0HYM2"
    auth: FRED_API_KEY
    inicio_verificado: "2023-07-18"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Termómetro de estrés HY núcleo; nivel y momentum. Feature viva potente pero SIN histórico de crisis (solo 3y)."
    verificado: true
    evidencia: "FRED BAMLH0A0HYM2 -> 787 obs, observation_start=2023-07-18 (=hoy-3y), first 2023-07-18:3.90 last 2026-07-16:2.71. Forzar observation_start=1996 NO amplía. Nota FRED: 'only 3 years'."
    url: "https://fred.stlouisfed.org/series/BAMLH0A0HYM2"

  - nombre_interno: IG_OAS
    descripcion: "ICE BofA US Corporate Index OAS (grado de inversión). Solo 3 años en FRED."
    fuente: fred
    id: "BAMLC0A0CM"
    auth: FRED_API_KEY
    inicio_verificado: "2023-07-18"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Spread IG de referencia; HY_OAS−IG_OAS = prima de riesgo por calidad. Sin histórico profundo (usar Baa−Aaa)."
    verificado: true
    evidencia: "FRED BAMLC0A0CM -> 786 obs, start 2023-07-18:1.28 last 2026-07-16:0.78. Mismo cap de 3 años."
    url: "https://fred.stlouisfed.org/series/BAMLC0A0CM"

  - nombre_interno: CCC_OAS
    descripcion: "ICE BofA CCC & Lower US HY Index OAS. La señal de estrés de crédito más sensible. Solo 3 años en FRED."
    fuente: fred
    id: "BAMLH0A3HYC"
    auth: FRED_API_KEY
    inicio_verificado: "2023-07-18"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "El bucket más basura: explota primero en crisis de crédito. CRÍTICO pero SIN histórico profundo gratis (mayor pérdida del cap de 3 años)."
    verificado: true
    evidencia: "FRED BAMLH0A3HYC -> 787 obs, start 2023-07-18:9.22 last 2026-07-16:9.70. Cap 3 años."
    url: "https://fred.stlouisfed.org/series/BAMLH0A3HYC"

  - nombre_interno: BB_OAS
    descripcion: "ICE BofA BB US HY Index OAS (HY de mayor calidad). Solo 3 años en FRED."
    fuente: fred
    id: "BAMLH0A1HYBB"
    auth: FRED_API_KEY
    inicio_verificado: "2023-07-18"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "CCC−BB = dispersión de calidad dentro de HY, señal fina de estrés. Solo 3y."
    verificado: true
    evidencia: "FRED BAMLH0A1HYBB -> 787 obs, start 2023-07-18:2.50 last 2026-07-16:1.61."
    url: "https://fred.stlouisfed.org/series/BAMLH0A1HYBB"

  - nombre_interno: B_OAS
    descripcion: "ICE BofA Single-B US HY Index OAS. Solo 3 años en FRED."
    fuente: fred
    id: "BAMLH0A2HYB"
    auth: FRED_API_KEY
    inicio_verificado: "2023-07-18"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Bucket medio de HY; completa la curva de crédito HY (BB→B→CCC). Solo 3y."
    verificado: true
    evidencia: "FRED BAMLH0A2HYB -> 787 obs, start 2023-07-18:4.13 last 2026-07-16:2.90."
    url: "https://fred.stlouisfed.org/series/BAMLH0A2HYB"

  - nombre_interno: BBB_OAS
    descripcion: "ICE BofA BBB US Corporate Index OAS (IG más bajo, frontera con HY). Solo 3 años en FRED."
    fuente: fred
    id: "BAMLC0A4CBBB"
    auth: FRED_API_KEY
    inicio_verificado: "2023-07-18"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "El IG más cíclico; riesgo de 'fallen angels'. BBB−BB cruza la frontera IG/HY. Solo 3y."
    verificado: true
    evidencia: "FRED BAMLC0A4CBBB -> 787 obs, start 2023-07-18:1.58 last 2026-07-16:0.96."
    url: "https://fred.stlouisfed.org/series/BAMLC0A4CBBB"

  - nombre_interno: A_OAS
    descripcion: "ICE BofA Single-A US Corporate Index OAS. Solo 3 años en FRED."
    fuente: fred
    id: "BAMLC0A3CA"
    auth: FRED_API_KEY
    inicio_verificado: "2023-07-18"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Bucket IG medio; granularidad de la curva de crédito IG. Solo 3y."
    verificado: true
    evidencia: "FRED BAMLC0A3CA -> 787 obs, start 2023-07-18:1.09 last 2026-07-16:0.65."
    url: "https://fred.stlouisfed.org/series/BAMLC0A3CA"

  - nombre_interno: EM_CORP_OAS
    descripcion: "ICE BofA Emerging Markets Corporate Plus Index OAS. Crédito emergente. Solo 3 años en FRED."
    fuente: fred
    id: "BAMLEMCBPIOAS"
    auth: FRED_API_KEY
    inicio_verificado: "2023-07-18"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Estrés de crédito EM; suele liderar en shocks de dólar/riesgo global. Solo 3y (proxy largo = EMB ETF)."
    verificado: true
    evidencia: "FRED BAMLEMCBPIOAS -> 787 obs, start 2023-07-18:2.68 last 2026-07-16:1.47."
    url: "https://fred.stlouisfed.org/series/BAMLEMCBPIOAS"

  - nombre_interno: EURO_HY_OAS
    descripcion: "ICE BofA Euro High Yield Index OAS. Crédito HY europeo. Solo 3 años en FRED."
    fuente: fred
    id: "BAMLHE00EHYIOAS"
    auth: FRED_API_KEY
    inicio_verificado: "2023-07-18"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "HY europeo; divergencia US−Euro HY marca estrés regional. Solo 3y."
    verificado: true
    evidencia: "FRED BAMLHE00EHYIOAS -> 787 obs, start 2023-07-18:4.45 last 2026-07-16:2.49."
    url: "https://fred.stlouisfed.org/series/BAMLHE00EHYIOAS"

  - nombre_interno: HY_YIELD
    descripcion: "ICE BofA US High Yield Index Effective Yield (nivel de yield HY, no spread). Solo 3 años en FRED."
    fuente: fred
    id: "BAMLH0A0HYM2EY"
    auth: FRED_API_KEY
    inicio_verificado: "2023-07-18"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Yield absoluto HY (spread + tipo base); complementa el OAS. Solo 3y."
    verificado: true
    evidencia: "FRED BAMLH0A0HYM2EY -> 787 obs, start 2023-07-18:8.08 last 2026-07-16:6.96."
    url: "https://fred.stlouisfed.org/series/BAMLH0A0HYM2EY"

  # ============ PROXIES ETF (yfinance, historia completa, vivo) ============
  - nombre_interno: HYG
    descripcion: "iShares iBoxx $ High Yield Corporate Bond ETF. Proxy de mercado del crédito HY, total-return."
    fuente: yfinance
    id: "HYG"
    auth: none
    inicio_verificado: "2007-04-11"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Cubre 2008/2020 (donde la OAS de FRED ya no llega). Caídas persistentes = risk-off de crédito. Base del spread HYG−IEF."
    verificado: true
    evidencia: "yf.download('HYG','max') -> 4848 filas desde 2007-04-11 hasta 2026-07-17 (auto_adjust)."
    url: "https://finance.yahoo.com/quote/HYG"

  - nombre_interno: JNK
    descripcion: "SPDR Bloomberg High Yield Bond ETF. Proxy HY alternativo/redundante a HYG."
    fuente: yfinance
    id: "JNK"
    auth: none
    inicio_verificado: "2007-12-04"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Redundancia de fuente para HY; corrobora HYG en episodios de estrés."
    verificado: true
    evidencia: "yf.download('JNK','max') -> 4683 filas desde 2007-12-04 hasta 2026-07-17."
    url: "https://finance.yahoo.com/quote/JNK"

  - nombre_interno: LQD
    descripcion: "iShares iBoxx $ Investment Grade Corporate Bond ETF. Proxy de mercado del crédito IG."
    fuente: yfinance
    id: "LQD"
    auth: none
    inicio_verificado: "2002-07-30"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Proxy IG con historia desde 2002 (cubre 2008); HYG/LQD = prima de calidad HY vs IG. Base de spread largo."
    verificado: true
    evidencia: "yf.download('LQD','max') -> 6030 filas desde 2002-07-30 hasta 2026-07-17."
    url: "https://finance.yahoo.com/quote/LQD"

  - nombre_interno: IEF
    descripcion: "iShares 7-10 Year Treasury Bond ETF. Pata Treasury para el spread de crédito HYG−IEF."
    fuente: yfinance
    id: "IEF"
    auth: none
    inicio_verificado: "2002-07-30"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Benchmark libre de riesgo de duración similar a HYG; HYG−IEF aísla el componente de crédito puro."
    verificado: true
    evidencia: "yf.download('IEF','max') -> 6030 filas desde 2002-07-30 hasta 2026-07-17."
    url: "https://finance.yahoo.com/quote/IEF"

  - nombre_interno: HYG_IEF_SPREAD
    descripcion: "Spread de crédito HY proxy = comportamiento relativo HYG vs IEF (ratio rebased o exceso de retorno). Sustituye a la HY OAS histórica."
    fuente: yfinance
    id: "HYG,IEF"
    auth: none
    inicio_verificado: "2007-04-11"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Proxy de estrés HY con historia real de crisis. Ratio HYG/IEF rebased-100: mín 56.0 el 2008-12-11 y 76.2 el 2020-03-23 (captura 2008 y COVID)."
    verificado: true
    evidencia: "Calc de yf HYG/IEF desde 2007-04-11: mín 2008=56.0 (2008-12-11), 2020=76.2 (2020-03-23), 2022=103.7. Inverso al spread OAS."
    url: "https://finance.yahoo.com/quote/HYG"

  - nombre_interno: EMB
    descripcion: "iShares JP Morgan USD Emerging Markets Bond ETF. Proxy de mercado de crédito EM."
    fuente: yfinance
    id: "EMB"
    auth: none
    inicio_verificado: "2007-12-19"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Proxy largo de EM_CORP_OAS (que solo tiene 3y en FRED); estrés de crédito emergente 2008+."
    verificado: true
    evidencia: "yf.download('EMB','max') -> 4672 filas desde 2007-12-19 hasta 2026-07-17."
    url: "https://finance.yahoo.com/quote/EMB"

  - nombre_interno: BKLN
    descripcion: "Invesco Senior Loan ETF (leveraged loans / crédito flotante)."
    fuente: yfinance
    id: "BKLN"
    auth: none
    inicio_verificado: "2011-03-03"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Crédito a tipo flotante; BKLN vs HYG separa riesgo de crédito de riesgo de duración."
    verificado: true
    evidencia: "yf.download('BKLN','max') -> 3866 filas desde 2011-03-03 hasta 2026-07-17."
    url: "https://finance.yahoo.com/quote/BKLN"

  - nombre_interno: AGG
    descripcion: "iShares Core US Aggregate Bond ETF. Agregado de renta fija US (IG + Treasuries + MBS)."
    fuente: yfinance
    id: "AGG"
    auth: none
    inicio_verificado: "2003-09-29"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Referencia amplia de bonos; contexto de retorno total de renta fija por régimen."
    verificado: true
    evidencia: "yf.download('AGG','max') -> 5736 filas desde 2003-09-29 hasta 2026-07-17."
    url: "https://finance.yahoo.com/quote/AGG"

  # ============ FUNDING / TED ============
  - nombre_interno: TED_SPREAD
    descripcion: "TED Spread (3M LIBOR − 3M T-Bill). Estrés de financiación bancaria. DESCONTINUADO (fin de LIBOR)."
    fuente: fred
    id: "TEDRATE"
    auth: FRED_API_KEY
    inicio_verificado: "1986-01-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Estrés de financiación/interbancario 1986-2022 (picos 1987, 1998-LTCM, 2008=lehman, 2020). SIN sucesor gratis limpio tras el fin de LIBOR (sustituto SOFR-OIS = categoría liquidez)."
    verificado: true
    evidencia: "FRED TEDRATE -> 8853 obs, first 1986-01-02:0.90 last 2022-01-21:0.09. Metadata: 'TED Spread (DISCONTINUED)', last_updated 2022-01-28."
    url: "https://fred.stlouisfed.org/series/TEDRATE"

  # ============ VALIDACIÓN (índices de estrés de crédito ya hechos) ============
  - nombre_interno: NFCI_CREDIT
    descripcion: "Chicago Fed National Financial Conditions Credit Subindex. Índice de estrés de crédito ya construido."
    fuente: fred
    id: "NFCICREDIT"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-08"
    granularidad: semanal
    pista: validacion
    rol: validation
    relevancia_regimen: "Ground-truth laxo: subíndice de condiciones de crédito con +50 años; >0 = crédito tenso. Valida los regímenes del detector."
    verificado: true
    evidencia: "FRED NFCICREDIT -> 2897 obs semanales, first 1971-01-08:-1.105 last 2026-07-10:-0.044 (upd 2026-07-15)."
    url: "https://fred.stlouisfed.org/series/NFCICREDIT"

  - nombre_interno: STLFSI4
    descripcion: "St. Louis Fed Financial Stress Index (v4). Índice compuesto de estrés financiero (incluye spreads de crédito)."
    fuente: fred
    id: "STLFSI4"
    auth: FRED_API_KEY
    inicio_verificado: "1993-12-31"
    granularidad: semanal
    pista: validacion
    rol: validation
    relevancia_regimen: "Ground-truth de estrés compuesto (18 series, muchas de crédito); >0 = estrés por encima de media."
    verificado: true
    evidencia: "FRED STLFSI4 -> 1698 obs semanales, first 1993-12-31:-0.291 last 2026-07-10:-0.882 (upd 2026-07-15)."
    url: "https://fred.stlouisfed.org/series/STLFSI4"

  # ============ FALLBACKS NO VERIFICADOS ============
  - nombre_interno: OAS_FULL_HISTORY_SNAPSHOT
    descripcion: "Reconstrucción de la historia larga de las OAS por rating (1996-2023) desde un CSV archivado pre-abril-2026, empalmado con la ventana viva de 3 años de FRED."
    fuente: github
    id: "credit-spreads-recession-risk / archived FRED CSV"
    auth: none
    inicio_verificado: null
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Única forma de recuperar OAS por rating (incl. CCC) para 2008/2011/2015/2020. PERO es foto estática, no viva, y no verificada aquí."
    verificado: false
    evidencia: "No descargable/verificable en este entorno. Existen datasets tipo eco3min.fr 'credit spreads recession risk' con BAMLH0A0HYM2 histórico, pero son snapshots de terceros sin garantía de fidelidad/actualización."
    url: "https://eco3min.fr/en/credit-spreads-recession-risk-dataset/"

  - nombre_interno: HYG_STOOQ
    descripcion: "HYG vía Stooq CSV como fallback de yfinance."
    fuente: stooq
    id: "hyg.us"
    auth: none
    inicio_verificado: null
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Redundancia de fuente para los ETFs de crédito si cae yfinance."
    verificado: false
    evidencia: "BLOQUEADO: stooq.com/q/d/l/?s=hyg.us&i=d devuelve challenge JavaScript de proof-of-work (SHA-256), no el CSV. No usable sin navegador."
    url: "https://stooq.com/q/d/l/?s=hyg.us&i=d"

  - nombre_interno: OAS_NASDAQ_ML
    descripcion: "OAS BofA histórico vía Nasdaq Data Link legacy database ML/*."
    fuente: kaggle
    id: "ML/HYOAS, ML/CCCOAS"
    auth: none
    inicio_verificado: null
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Habría dado OAS 1996+ gratis; deprecado."
    verificado: false
    evidencia: "HTTP 403 en data.nasdaq.com/api/v3/datasets/ML/HYOAS.csv (database deprecado/bloqueado)."
    url: "https://data.nasdaq.com/data/ML"
```

---

## Recomendación de priorización para el pipeline

1. **Spine de crédito profundo (imprescindible, cubre todas las crisis):** `MOODYS_BAA_AAA_SPREAD`
   (DBAA−DAAA, 1986+) + `BAA10Y` (1986+) + `BAA`/`AAA` mensual (1919+, Pista A). Todo FRED, vivo,
   sin restricción de licencia. Es el sustituto honesto de las OAS para historia larga.
2. **Proxies de HY con historia real (Pista B, cubren 2008/2020):** `HYG`/`IEF` y el spread
   `HYG_IEF_SPREAD` (2007+), `LQD` (2002+), `EMB` (2007+). Son la vía libre para tener estrés HY en
   crisis pasadas, ya que la OAS HY de FRED ya no lo da.
3. **OAS por rating como features VIVAS (Pista B, solo 3 años):** `HY_OAS`, `IG_OAS`, `CCC_OAS`,
   `BB_OAS`, `B_OAS`, `BBB_OAS`, `EM_CORP_OAS`, `EURO_HY_OAS`. Granularidad por rating imbatible para
   el régimen **actual**, pero **no aportan histórico**. Diseñar el pipeline para no depender de su
   profundidad. La dispersión `CCC−BB` es la mejor señal fina… pero solo desde 2023.
4. **Funding:** `TED_SPREAD` como historia 1986-2022 (descontinuado; sin continuación gratis).
5. **Validación:** `NFCI_CREDIT` (1971+) y `STLFSI4` (1993+) como ground-truth de estrés de crédito.
6. **Deuda técnica honesta:** si el TFM necesita CCC/HY OAS por rating en crisis pre-2023, hay que
   **archivar un snapshot pre-abril-2026** (`OAS_FULL_HISTORY_SNAPSHOT`, no verificado) y empalmarlo
   con la ventana viva. Documentarlo como limitación de reproducibilidad. **Stooq y Nasdaq ML no son
   viables hoy** (bloqueados/deprecados).

**Fuentes web consultadas:** [Nota oficial FRED BAMLH0A0HYM2 (restricción 3 años, abril 2026)](https://fred.stlouisfed.org/series/BAMLH0A0HYM2) ·
[FRED BAMLC0A0CM (IG OAS)](https://fred.stlouisfed.org/series/BAMLC0A0CM) ·
[Dataset terceros credit-spreads-recession-risk](https://eco3min.fr/en/credit-spreads-recession-risk-dataset/)
