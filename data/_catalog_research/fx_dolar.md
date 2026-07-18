# Categoría: FX y dólar — estado del arte de datos (verificado)

Investigador: agente **FX y dólar**. Foco: índice dólar (DXY / broad dollar), pares mayores
(EURUSD, USDJPY, GBP, CHF), FX de refugio (JPY, CHF) vs FX de riesgo (AUD, NZD, EM), y
cesta EM (MXN, ZAR, BRL, TRY, KRW, INR, CNY). El dólar es el **eje del risk-off global**: en
crisis el USD se aprecia (flight-to-dollar), el yen y el franco se aprecian, y las divisas
emergentes y de commodities se hunden. La forma de la cesta (broad vs advanced vs EM) y la
dispersión entre pares separan regímenes de risk-on / risk-off / stress.

Todas las verificaciones se hicieron **de verdad** el 2026-07-18: descarga real contra la
**FRED API** con `FRED_API_KEY` del `.env` (sin imprimir la clave) usando el endpoint
`fred/series/observations`, y contra **yfinance `period='max'`** para Yahoo. Reporto la fecha
de inicio **observada** en la respuesta, no la de marketing. Conteo de obs = filas con valor
válido (descarto los `.` de festivos).

---

## Resumen ejecutivo de lo verificado

**El DXY real (ICE) está disponible GRATIS, diario y con máximo histórico profundo.**
`DX-Y.NYB` en Yahoo devuelve **14.102 filas desde 1971-01-04** hasta hoy (2026-07-17=100.75),
serie **única y continua, viva**. Es el índice dólar de 6 divisas (EUR 57,6% / JPY / GBP / CAD /
SEK / CHF). **252 filas en 2013 → sin punto ciego.** Es el mejor candidato a **spine de FX**:
un solo símbolo cubre 55 años y todas las crisis. (Nota: el DXY oficial arranca en 1973-03 con
base 100; Yahoo backfillea a 1971 — tratar el tramo 1971-1973 con cautela.)

**Broad dollar profundo vía FRED (trade-weighted, 1973):** `DTWEXM` (Major Currencies,
**diaria 1973-01-02 → 2019-12-31**, 11.834 obs) es la cesta amplia de divisas mayores, pero
está **DISCONTINUADA en 2019**. Su continuación viva es `DTWEXAFEGS` (Advanced Foreign
Economies, diaria **2006-01-02 → viva**). **Empalme recomendado: DTWEXM (1973-2019) + DTWEXAFEGS
(2006-live) con solape 2006-2019 para calibrar el nivel.** Para la cesta EM: `DTWEXO` (Other
Important Trading Partners, 1995-2019, discontinuada) → `DTWEXEMEGS` (EM Economies, 2006-live).
La broad total viva es `DTWEXBGS` (2006-live).

**Dólar efectivo MÁS profundo de todo FRED (BIS REER, 1964):** `RNUSBIS` (Real Narrow Effective
Exchange Rate US, **mensual desde 1964-01**, 749 obs) y su nominal `NNUSBIS` (1964). Es la serie
de dólar más larga que verifiqué — cubre Bretton Woods, el Nixon shock (1971), los años 70. Ideal
como **dólar de baja frecuencia para la Pista A**. Broad BIS `RBUSBIS`/`NBUSBIS` desde 1994.

**Pares mayores vía FRED (diarios, profundos 1971):** `DEXJPUS` (USDJPY), `DEXUSUK` (GBPUSD),
`DEXSZUS` (USDCHF), `DEXCAUS` (USDCAD), `DEXUSAL` (AUDUSD), `DEXUSNZ` (NZDUSD), `DEXSDUS`/
`DEXNOUS`/`DEXDNUS` (SEK/NOK/DKK) — **todos desde 1971-01-04, ~13.900 obs, vivos, 251 obs/2013**.
El euro `DEXUSEU` sólo desde **1999-01-04** (nacimiento del euro; no hay marco alemán diario en
FRED — `DEXGEUS` y demás legacy euro-zone dan HTTP 400, NO existen). El USD como par se lee vía
la broad dollar para el tramo pre-1999.

**Cesta EM vía FRED (diaria):** `DEXMXUS` (USDMXN, **1993-11**, el barómetro EM de alta beta),
`DEXBZUS` (USDBRL, 1995), `DEXSFUS` (USDZAR, 1980, otro high-beta risk), `DEXINUS` (USDINR,
**1973**), `DEXKOUS` (USDKRW, 1981, proxy de riesgo asiático), `DEXCHUS` (USDCNY, 1981, gestionado),
`DEXTHUS`/`DEXSIUS`/`DEXTAUS`/`DEXHKUS`/`DEXMAUS` (Asia, 1981-1983). **Turquía y Rusia NO están en
FRED H.10** → los cubro por yfinance: `USDTRY=X` (2005) y `USDRUB=X` (2003), verificados.

**yfinance para pares FX = fallback poco profundo.** Los pares en Yahoo (`EURUSD=X`, `JPY=X`,
`USDMXN=X`, …) arrancan sólo en **2003-2006** (historia FX corta de Yahoo). Para pares, **FRED
gana claramente** (1971-1993 vs 2003). Los dejo como fallback/redundancia. La excepción es el
propio **DXY (`DX-Y.NYB`, 1971)** donde Yahoo es la mejor y casi única fuente gratis.

**Stooq BLOQUEADO** (igual que reportaron los agentes de Volatilidad y Tipos): `stooq.com/q/d/l/`
devuelve un HTML `noscript`/challenge JS (796 bytes, sin CSV) para `dxy`, `eurusd`, `usdjpy`,
`usdmxn`. No usable sin navegador. Lo dejo listado como fallback `verificado:false`.

**Límites honestos:** (a) EURUSD diario sólo desde 1999 — pre-euro no hay par diario gratis
verificado (usar broad dollar); (b) DXY 1971-1973 es backfill de Yahoo (el índice nació 1973);
(c) `DX=F` (futuro del índice dólar) da 404 en Yahoo, NO disponible; (d) Turquía/Rusia sólo por
yfinance (2003-2005), sin versión FRED profunda; (e) las series broad discontinuadas (DTWEXM,
DTWEXB, DTWEXO) **paran en 2019-12-31** y requieren empalme con las *GS (2006+) para estar vivas.

---

## Detalle por serie (evidencia)

| serie | fuente | inicio verificado | fin/estado | nota |
|---|---|---|---|---|
| DX-Y.NYB | yfinance | 1971-01-04 | vivo (2026-07-17=100.75) | **DXY ICE, spine FX, 14102 filas, 252/2013** |
| DTWEXM | FRED | 1973-01-02 | **discontinuada 2019-12-31** | broad major, 11834 obs, empalmar con DTWEXAFEGS |
| DTWEXAFEGS | FRED | 2006-01-02 | vivo | advanced economies (continúa DTWEXM) |
| DTWEXBGS | FRED | 2006-01-02 | vivo | broad dollar total vivo |
| DTWEXEMEGS | FRED | 2006-01-02 | vivo | EM economies (dólar vs EM) |
| DTWEXB | FRED | 1995-01-04 | **discont. 2019-12-31** | broad viejo, fallback |
| DTWEXO | FRED | 1995-01-04 | **discont. 2019-12-31** | other partners (EM), empalmar DTWEXEMEGS |
| RNUSBIS | FRED (BIS) | 1964-01-01 | vivo (mensual) | **dólar más profundo, real narrow REER** |
| NNUSBIS | FRED (BIS) | 1964-01-01 | vivo (mensual) | nominal narrow REER 1964 |
| RBUSBIS | FRED (BIS) | 1994-01-01 | vivo (mensual) | real broad REER |
| NBUSBIS | FRED (BIS) | 1994-01-01 | vivo (mensual) | nominal broad REER |
| TWEXMMTH | FRED | 1973-01-01 | **discont. 2019-12-01** | broad major mensual |
| TWEXAFEGSMTH | FRED | 2006-01-01 | vivo (mensual) | advanced mensual vivo |
| TWEXEMEGSMTH | FRED | 2006-01-01 | vivo (mensual) | EM mensual vivo |
| DEXUSEU | FRED | 1999-01-04 | vivo | EURUSD (USD por EUR), 6901 obs |
| DEXJPUS | FRED | 1971-01-04 | vivo | USDJPY (JPY por USD), refugio |
| DEXSZUS | FRED | 1971-01-04 | vivo | USDCHF (CHF por USD), refugio |
| DEXUSUK | FRED | 1971-01-04 | vivo | GBPUSD (USD por GBP) |
| DEXCAUS | FRED | 1971-01-04 | vivo | USDCAD, commodity FX |
| DEXUSAL | FRED | 1971-01-04 | vivo | AUDUSD, risk/commodity FX |
| DEXUSNZ | FRED | 1971-01-04 | vivo | NZDUSD, risk FX |
| DEXSDUS | FRED | 1971-01-04 | vivo | USDSEK |
| DEXNOUS | FRED | 1971-01-04 | vivo | USDNOK, petro-FX |
| DEXMXUS | FRED | 1993-11-08 | vivo | **USDMXN, barómetro EM alta beta** |
| DEXBZUS | FRED | 1995-01-02 | vivo | USDBRL, EM |
| DEXSFUS | FRED | 1980-01-02 | vivo | USDZAR, EM high-beta risk |
| DEXINUS | FRED | 1973-01-02 | vivo | USDINR, EM (profundo) |
| DEXKOUS | FRED | 1981-04-13 | vivo | USDKRW, proxy riesgo asiático |
| DEXCHUS | FRED | 1981-01-02 | vivo | USDCNY, gestionado |
| USDTRY=X | yfinance | 2005-01-03 | vivo | Turquía (no en FRED) |
| USDRUB=X | yfinance | 2003-12-01 | vivo | Rusia (no en FRED) |
| EURUSD=X … | yfinance | 2003-2006 | vivo | pares fallback poco profundos |
| dxy/eurusd/… | stooq | — | **BLOQUEADO** | noscript/challenge, no CSV |

```yaml
series_fx_dolar:
  # ================= SPINE FX: DXY (ICE) =================
  - nombre_interno: DXY
    descripcion: "ICE US Dollar Index (DXY): cesta fija de 6 divisas (EUR 57.6%, JPY, GBP, CAD, SEK, CHF). Índice dólar de referencia del mercado."
    fuente: yfinance
    id: "DX-Y.NYB"
    auth: none
    inicio_verificado: "1971-01-04"
    granularidad: diaria
    pista: ambas
    rol: spine
    relevancia_regimen: "Nivel del dólar = eje risk-off global; USD sube en crisis (flight-to-dollar). Serie única, diaria, 55 años, viva. Mejor spine de FX gratis."
    verificado: true
    evidencia: "yf.download('DX-Y.NYB','max') -> 14102 filas desde 1971-01-04, last 2026-07-17=100.75. 252 filas en 2013 (sin hueco). OJO: DXY oficial nace 1973-03; 1971-73 es backfill de Yahoo."
    url: "https://finance.yahoo.com/quote/DX-Y.NYB"

  # ================= BROAD DOLLAR (FRED, trade-weighted) =================
  - nombre_interno: DTWEXM
    descripcion: "Nominal Major Currencies US Dollar Index (trade-weighted, diaria). Cesta amplia de divisas mayores. DISCONTINUADA 2019."
    fuente: fred
    id: "DTWEXM"
    auth: FRED_API_KEY
    inicio_verificado: "1973-01-02"
    granularidad: diaria
    pista: ambas
    rol: spine
    relevancia_regimen: "Dólar broad-major diario desde 1973 (Bretton Woods post-Nixon). Cubre todas las crisis. Empalmar con DTWEXAFEGS para tramo vivo."
    verificado: true
    evidencia: "FRED DTWEXM -> 11834 obs, first 1973-01-02=108.22 last 2019-12-31=90.82. DISCONTINUADA (para en 2019). 249 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DTWEXM"

  - nombre_interno: DTWEXAFEGS
    descripcion: "Nominal Advanced Foreign Economies US Dollar Index (trade-weighted, diaria). Continuación viva de DTWEXM."
    fuente: fred
    id: "DTWEXAFEGS"
    auth: FRED_API_KEY
    inicio_verificado: "2006-01-02"
    granularidad: diaria
    pista: ambas
    rol: core
    relevancia_regimen: "Dólar frente a economías avanzadas; empalma con DTWEXM (solape 2006-2019) para spine broad-major vivo hasta hoy."
    verificado: true
    evidencia: "FRED DTWEXAFEGS -> 5144 obs, first 2006-01-02=101.79 last 2026-07-10=113.69. 249 obs en 2013. Viva."
    url: "https://fred.stlouisfed.org/series/DTWEXAFEGS"

  - nombre_interno: DTWEXBGS
    descripcion: "Nominal Broad US Dollar Index (trade-weighted, diaria). Cesta total (avanzadas + EM)."
    fuente: fred
    id: "DTWEXBGS"
    auth: FRED_API_KEY
    inicio_verificado: "2006-01-02"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Dólar broad total; medida oficial Fed del dólar ponderado por comercio. Nivel de estrés/refugio del USD."
    verificado: true
    evidencia: "FRED DTWEXBGS -> 5144 obs, first 2006-01-02=101.42 last 2026-07-10=120.50. 249 obs en 2013. Viva."
    url: "https://fred.stlouisfed.org/series/DTWEXBGS"

  - nombre_interno: DTWEXEMEGS
    descripcion: "Nominal Emerging Market Economies US Dollar Index (trade-weighted, diaria). Dólar frente a EM."
    fuente: fred
    id: "DTWEXEMEGS"
    auth: FRED_API_KEY
    inicio_verificado: "2006-01-02"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Dólar vs cesta EM = termómetro directo de risk-off en emergentes (sube fuerte en 2008, 2015, 2020, 2022). Empalmar con DTWEXO para tramo 1995-2006."
    verificado: true
    evidencia: "FRED DTWEXEMEGS -> 5144 obs, first 2006-01-02=100.94 last 2026-07-10=129.16. 249 obs en 2013. Viva."
    url: "https://fred.stlouisfed.org/series/DTWEXEMEGS"

  - nombre_interno: DTWEXO
    descripcion: "Nominal Other Important Trading Partners US Dollar Index (trade-weighted, diaria). Precursor EM. DISCONTINUADA 2019."
    fuente: fred
    id: "DTWEXO"
    auth: FRED_API_KEY
    inicio_verificado: "1995-01-04"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Dólar vs socios EM 1995-2019; extiende DTWEXEMEGS hacia atrás (crisis asiática 1997, Rusia 1998, Argentina 2001). Empalmar con DTWEXEMEGS."
    verificado: true
    evidencia: "FRED DTWEXO -> 6326 obs, first 1995-01-04=89.67 last 2019-12-31=169.09. DISCONTINUADA. 249 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DTWEXO"

  - nombre_interno: DTWEXB
    descripcion: "Nominal Broad US Dollar Index (versión antigua, diaria). DISCONTINUADA 2019, reemplazada por DTWEXBGS."
    fuente: fred
    id: "DTWEXB"
    auth: FRED_API_KEY
    inicio_verificado: "1995-01-04"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Broad dollar 1995-2019; fallback / extensión de DTWEXBGS hacia 1995. Empalmar con DTWEXBGS."
    verificado: true
    evidencia: "FRED DTWEXB -> 6328 obs, first 1995-01-04=94.35 last 2019-12-31=128.01. DISCONTINUADA."
    url: "https://fred.stlouisfed.org/series/DTWEXB"

  # ================= DÓLAR EFECTIVO PROFUNDO (BIS REER, mensual) =================
  - nombre_interno: RNUSBIS
    descripcion: "Real Narrow Effective Exchange Rate for United States (BIS, mensual). El dólar efectivo real más profundo de FRED."
    fuente: fred
    id: "RNUSBIS"
    auth: FRED_API_KEY
    inicio_verificado: "1964-01-01"
    granularidad: mensual
    pista: A
    rol: spine
    relevancia_regimen: "Dólar real desde 1964: cubre Bretton Woods, Nixon shock 1971, años 70. Espina de FX de baja frecuencia para la Pista A (60 años)."
    verificado: true
    evidencia: "FRED RNUSBIS -> 749 obs mensuales, first 1964-01-01=115.86 last 2026-05-01=102.93. Viva."
    url: "https://fred.stlouisfed.org/series/RNUSBIS"

  - nombre_interno: NNUSBIS
    descripcion: "Nominal Narrow Effective Exchange Rate for United States (BIS, mensual). Versión nominal desde 1964."
    fuente: fred
    id: "NNUSBIS"
    auth: FRED_API_KEY
    inicio_verificado: "1964-01-01"
    granularidad: mensual
    pista: A
    rol: enricher
    relevancia_regimen: "Dólar nominal efectivo profundo; par nominal/real (RNUSBIS) separa componente de inflación relativa. 1964+."
    verificado: true
    evidencia: "FRED NNUSBIS -> 749 obs mensuales, first 1964-01-01=118.38 last 2026-05-01=105.57."
    url: "https://fred.stlouisfed.org/series/NNUSBIS"

  - nombre_interno: RBUSBIS
    descripcion: "Real Broad Effective Exchange Rate for United States (BIS, mensual). Dólar real broad."
    fuente: fred
    id: "RBUSBIS"
    auth: FRED_API_KEY
    inicio_verificado: "1994-01-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Dólar real broad (cesta amplia); nivel de competitividad/valoración del USD. 1994+."
    verificado: true
    evidencia: "FRED RBUSBIS -> 389 obs mensuales, first 1994-01-01=88.66 last 2026-05-01=107.26."
    url: "https://fred.stlouisfed.org/series/RBUSBIS"

  - nombre_interno: TWEXMMTH
    descripcion: "Trade Weighted US Dollar Index: Major Currencies (mensual). Versión mensual de DTWEXM. DISCONTINUADA 2019."
    fuente: fred
    id: "TWEXMMTH"
    auth: FRED_API_KEY
    inicio_verificado: "1973-01-01"
    granularidad: mensual
    pista: A
    rol: enricher
    relevancia_regimen: "Dólar broad-major mensual desde 1973 para modelos de baja frecuencia (Pista A). Empalmar con TWEXAFEGSMTH."
    verificado: true
    evidencia: "FRED TWEXMMTH -> 564 obs mensuales, first 1973-01-01=108.19 last 2019-12-01=91.87. DISCONTINUADA."
    url: "https://fred.stlouisfed.org/series/TWEXMMTH"

  - nombre_interno: TWEXAFEGSMTH
    descripcion: "Nominal Advanced Foreign Economies US Dollar Index (mensual). Continuación viva mensual."
    fuente: fred
    id: "TWEXAFEGSMTH"
    auth: FRED_API_KEY
    inicio_verificado: "2006-01-01"
    granularidad: mensual
    pista: A
    rol: fallback
    relevancia_regimen: "Dólar advanced mensual vivo; empalma con TWEXMMTH para la espina mensual larga."
    verificado: true
    evidencia: "FRED TWEXAFEGSMTH -> 246 obs, first 2006-01-01=100.00 last 2026-06-01=113.09. Viva."
    url: "https://fred.stlouisfed.org/series/TWEXAFEGSMTH"

  - nombre_interno: TWEXEMEGSMTH
    descripcion: "Nominal Emerging Market Economies US Dollar Index (mensual)."
    fuente: fred
    id: "TWEXEMEGSMTH"
    auth: FRED_API_KEY
    inicio_verificado: "2006-01-01"
    granularidad: mensual
    pista: B
    rol: fallback
    relevancia_regimen: "Dólar vs EM mensual vivo; versión low-freq de DTWEXEMEGS para Pista A/B."
    verificado: true
    evidencia: "FRED TWEXEMEGSMTH -> 246 obs, first 2006-01-01=100.00 last 2026-06-01=128.94. Viva."
    url: "https://fred.stlouisfed.org/series/TWEXEMEGSMTH"

  # ================= PARES MAYORES (FRED, diarios, refugio vs riesgo) =================
  - nombre_interno: DEXUSEU
    descripcion: "US Dollars per Euro (EURUSD, diaria). Par FX más negociado del mundo."
    fuente: fred
    id: "DEXUSEU"
    auth: FRED_API_KEY
    inicio_verificado: "1999-01-04"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "EURUSD = eje de la cesta DXY (57.6%). Su caída = fortaleza USD / risk-off. Sólo desde 1999 (nacimiento euro); pre-1999 usar broad dollar."
    verificado: true
    evidencia: "FRED DEXUSEU -> 6901 obs, first 1999-01-04=1.1812 last 2026-07-10=1.1438. 251 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DEXUSEU"

  - nombre_interno: DEXJPUS
    descripcion: "Japanese Yen per US Dollar (USDJPY, diaria). Divisa refugio."
    fuente: fred
    id: "DEXJPUS"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-04"
    granularidad: diaria
    pista: ambas
    rol: core
    relevancia_regimen: "El yen es REFUGIO: USDJPY cae (yen se aprecia) en risk-off / unwind de carry. Diaria desde 1971. Señal de régimen de primer orden."
    verificado: true
    evidencia: "FRED DEXJPUS -> 13916 obs, first 1971-01-04=357.73 last 2026-07-10=161.31. 251 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DEXJPUS"

  - nombre_interno: DEXSZUS
    descripcion: "Swiss Francs per US Dollar (USDCHF, diaria). Divisa refugio."
    fuente: fred
    id: "DEXSZUS"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-04"
    granularidad: diaria
    pista: ambas
    rol: core
    relevancia_regimen: "El franco suizo es REFUGIO clásico: se aprecia en estrés europeo/global. Diaria desde 1971. Complementa al yen."
    verificado: true
    evidencia: "FRED DEXSZUS -> 13922 obs, first 1971-01-04=4.3180 last 2026-07-10=0.8063. 251 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DEXSZUS"

  - nombre_interno: DEXUSUK
    descripcion: "US Dollars per British Pound (GBPUSD, diaria)."
    fuente: fred
    id: "DEXUSUK"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-04"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Cable; segunda mayor pata del DXY tras el euro. Sensible a eventos idiosincráticos (Brexit) y risk-off global."
    verificado: true
    evidencia: "FRED DEXUSUK -> 13922 obs, first 1971-01-04=2.3938 last 2026-07-10=1.3421. 251 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DEXUSUK"

  - nombre_interno: DEXCAUS
    descripcion: "Canadian Dollars per US Dollar (USDCAD, diaria). FX de commodities."
    fuente: fred
    id: "DEXCAUS"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-04"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "CAD ligado al petróleo/commodities; se debilita en shocks de crecimiento. Componente del DXY. Diaria 1971."
    verificado: true
    evidencia: "FRED DEXCAUS -> 13928 obs, first 1971-01-04=1.0109 last 2026-07-10=1.4132. 251 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DEXCAUS"

  - nombre_interno: DEXUSAL
    descripcion: "US Dollars per Australian Dollar (AUDUSD, diaria). FX de riesgo/commodities."
    fuente: fred
    id: "DEXUSAL"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-04"
    granularidad: diaria
    pista: ambas
    rol: core
    relevancia_regimen: "AUD = divisa PROCÍCLICA de riesgo (commodities, carry, exposición China). Cae fuerte en risk-off. AUDJPY es el termómetro carry clásico. Diaria 1971."
    verificado: true
    evidencia: "FRED DEXUSAL -> 13915 obs, first 1971-01-04=1.1127 last 2026-07-10=0.6959. 251 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DEXUSAL"

  - nombre_interno: DEXUSNZ
    descripcion: "US Dollars per New Zealand Dollar (NZDUSD, diaria). FX de riesgo/carry."
    fuente: fred
    id: "DEXUSNZ"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-04"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "NZD = alta beta de riesgo/carry, muy correlacionada con AUD. Cae en risk-off. Diaria 1971."
    verificado: true
    evidencia: "FRED DEXUSNZ -> 13906 obs, first 1971-01-04=1.1138 last 2026-07-10=0.5774. 251 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DEXUSNZ"

  - nombre_interno: DEXNOUS
    descripcion: "Norwegian Kroner per US Dollar (USDNOK, diaria). Petro-FX."
    fuente: fred
    id: "DEXNOUS"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-04"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "NOK ligada al petróleo; se debilita en shocks de crecimiento/energía. Diaria 1971."
    verificado: true
    evidencia: "FRED DEXNOUS -> 13921 obs, first 1971-01-04=7.1359 last 2026-07-10=9.7625. 251 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DEXNOUS"

  - nombre_interno: DEXSDUS
    descripcion: "Swedish Kronor per US Dollar (USDSEK, diaria). Componente del DXY."
    fuente: fred
    id: "DEXSDUS"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-04"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "SEK = divisa cíclica europea, pata del DXY. Se debilita en risk-off europeo. Diaria 1971."
    verificado: true
    evidencia: "FRED DEXSDUS -> 13921 obs, first 1971-01-04=5.1643 last 2026-07-10=9.6382. 251 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DEXSDUS"

  # ================= CESTA EM (FRED, diaria) =================
  - nombre_interno: DEXMXUS
    descripcion: "Mexican Pesos per US Dollar (USDMXN, diaria). Barómetro EM de alta beta."
    fuente: fred
    id: "DEXMXUS"
    auth: FRED_API_KEY
    inicio_verificado: "1993-11-08"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "MXN = la divisa EM más líquida y usada como PROXY de risk sentiment (se vende 24h como hedge). Salta en risk-off (Tequila 1994, 2008, 2016, 2020). Diaria 1993."
    verificado: true
    evidencia: "FRED DEXMXUS -> 8190 obs, first 1993-11-08=3.152 last 2026-07-10=17.4732. 251 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DEXMXUS"

  - nombre_interno: DEXSFUS
    descripcion: "South African Rand per US Dollar (USDZAR, diaria). EM high-beta."
    fuente: fred
    id: "DEXSFUS"
    auth: FRED_API_KEY
    inicio_verificado: "1980-01-02"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "ZAR = otra divisa EM de altísima beta de riesgo (commodities + carry). Muy sensible a risk-off global. Diaria desde 1980."
    verificado: true
    evidencia: "FRED DEXSFUS -> 11665 obs, first 1980-01-02=0.8271 last 2026-07-10=16.2746. 251 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DEXSFUS"

  - nombre_interno: DEXBZUS
    descripcion: "Brazilian Reais per US Dollar (USDBRL, diaria). EM."
    fuente: fred
    id: "DEXBZUS"
    auth: FRED_API_KEY
    inicio_verificado: "1995-01-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "BRL = EM latinoamericano de alto carry; se hunde en risk-off (2002, 2008, 2015, 2020). Diaria 1995."
    verificado: true
    evidencia: "FRED DEXBZUS -> 7905 obs, first 1995-01-02=0.8440 last 2026-07-10=5.1046. 251 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DEXBZUS"

  - nombre_interno: DEXKOUS
    descripcion: "South Korean Won per US Dollar (USDKRW, diaria). Proxy de riesgo asiático."
    fuente: fred
    id: "DEXKOUS"
    auth: FRED_API_KEY
    inicio_verificado: "1981-04-13"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "KRW = proxy de riesgo/ciclo exportador asiático; se depreció fuerte en 1997 (crisis asiática) y 2008. Diaria 1981."
    verificado: true
    evidencia: "FRED DEXKOUS -> 11308 obs, first 1981-04-13=675.40 last 2026-07-10=1501.06. 251 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DEXKOUS"

  - nombre_interno: DEXINUS
    descripcion: "Indian Rupees per US Dollar (USDINR, diaria). EM (profundo, 1973)."
    fuente: fred
    id: "DEXINUS"
    auth: FRED_API_KEY
    inicio_verificado: "1973-01-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "INR = EM asiático deficitario, sensible a taper/energía (taper tantrum 2013). Historia diaria profunda desde 1973."
    verificado: true
    evidencia: "FRED DEXINUS -> 13414 obs, first 1973-01-02=8.02 last 2026-07-10=95.33. 251 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DEXINUS"

  - nombre_interno: DEXCHUS
    descripcion: "Chinese Yuan Renminbi per US Dollar (USDCNY, diaria). Divisa gestionada."
    fuente: fred
    id: "DEXCHUS"
    auth: FRED_API_KEY
    inicio_verificado: "1981-01-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "CNY gestionado (poca varianza intradía), pero sus devaluaciones administradas (ago-2015, 2019) MARCAN regímenes de estrés global. Diaria 1981."
    verificado: true
    evidencia: "FRED DEXCHUS -> 11362 obs, first 1981-01-02=1.5341 last 2026-07-10=6.7766. 251 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DEXCHUS"

  - nombre_interno: USDTRY_YF
    descripcion: "USD/TRY (lira turca) vía Yahoo. EM de crisis; Turquía no está en FRED H.10."
    fuente: yfinance
    id: "USDTRY=X"
    auth: none
    inicio_verificado: "2005-01-03"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "TRY = divisa EM de crisis recurrente (2018, 2021-2023); marca estrés idiosincrático y contagio EM. Única fuente gratis (no en FRED)."
    verificado: true
    evidencia: "yf.download('USDTRY=X','max') -> 5600 filas desde 2005-01-03, last 2026-07-17=47.14. 260 filas en 2013."
    url: "https://finance.yahoo.com/quote/USDTRY=X"

  - nombre_interno: USDRUB_YF
    descripcion: "USD/RUB (rublo ruso) vía Yahoo. EM; Rusia no está en FRED H.10."
    fuente: yfinance
    id: "USDRUB=X"
    auth: none
    inicio_verificado: "2003-12-01"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "RUB = petro-EM con shocks geopolíticos (2014, 2022, con hueco de convertibilidad post-2022). Sólo yfinance; usar con cuidado tras 2022."
    verificado: true
    evidencia: "yf.download('USDRUB=X','max') -> 5763 filas desde 2003-12-01, last 2026-07-18=78.13. 260 filas en 2013."
    url: "https://finance.yahoo.com/quote/USDRUB=X"

  # ================= FALLBACKS yfinance (pares poco profundos) =================
  - nombre_interno: EURUSD_YF
    descripcion: "EURUSD vía Yahoo (fallback de DEXUSEU, historia más corta)."
    fuente: yfinance
    id: "EURUSD=X"
    auth: none
    inicio_verificado: "2003-12-01"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Redundancia de fuente para EURUSD si cae FRED; Yahoo sólo desde 2003 (FRED DEXUSEU es mejor, 1999)."
    verificado: true
    evidencia: "yf.download('EURUSD=X','max') -> 5871 filas desde 2003-12-01, last 2026-07-17=1.1446. 260 filas en 2013."
    url: "https://finance.yahoo.com/quote/EURUSD=X"

  - nombre_interno: USDJPY_YF
    descripcion: "USDJPY vía Yahoo (fallback de DEXJPUS)."
    fuente: yfinance
    id: "JPY=X"
    auth: none
    inicio_verificado: "1996-10-30"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Redundancia del yen refugio; Yahoo desde 1996 (FRED DEXJPUS es mejor, 1971)."
    verificado: true
    evidencia: "yf.download('JPY=X','max') -> 7705 filas desde 1996-10-30, last 2026-07-17=162.35. 260 filas en 2013."
    url: "https://finance.yahoo.com/quote/JPY=X"

  - nombre_interno: USDMXN_YF
    descripcion: "USDMXN vía Yahoo (fallback de DEXMXUS)."
    fuente: yfinance
    id: "USDMXN=X"
    auth: none
    inicio_verificado: "2003-12-01"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Redundancia del barómetro EM (peso mexicano); Yahoo desde 2003 (FRED DEXMXUS es mejor, 1993)."
    verificado: true
    evidencia: "yf.download('USDMXN=X','max') -> 5892 filas desde 2003-12-01, last 2026-07-17=17.53. 261 filas en 2013."
    url: "https://finance.yahoo.com/quote/USDMXN=X"

  # ================= NO USABLE / BLOQUEADO =================
  - nombre_interno: DXY_STOOQ
    descripcion: "DXY / pares FX vía Stooq CSV como fallback terciario."
    fuente: stooq
    id: "dxy"
    auth: none
    inicio_verificado: null
    granularidad: diaria
    pista: ambas
    rol: fallback
    relevancia_regimen: "Redundancia adicional si caen FRED y yfinance."
    verificado: false
    evidencia: "BLOQUEADO: stooq.com/q/d/l/?s=dxy (y eurusd/usdjpy/usdmxn) devuelve HTML noscript/challenge JS (796 bytes), no CSV. No usable sin navegador (confirmado por agentes de Volatilidad y Tipos)."
    url: "https://stooq.com/q/d/l/?s=dxy&i=d"

  - nombre_interno: DX_FUT_YF
    descripcion: "US Dollar Index futures (ICE) vía Yahoo."
    fuente: yfinance
    id: "DX=F"
    auth: none
    inicio_verificado: null
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Futuro del DXY; sería alternativa al índice cash DX-Y.NYB."
    verificado: false
    evidencia: "NO DISPONIBLE: yf.download('DX=F') -> HTTP 404 'Quote not found for symbol: DX=F'. Usar DX-Y.NYB (cash) en su lugar."
    url: "https://finance.yahoo.com/quote/DX=F"
```

---

## Recomendación de priorización para el pipeline

1. **Spine de FX (imprescindible, ambas pistas):** `DX-Y.NYB` (DXY ICE, yfinance) — un solo
   símbolo, diario, 1971→vivo, sin hueco 2013. Es el nivel del dólar de referencia.
2. **Broad dollar profundo + vivo (empalme):** `DTWEXM` (1973-2019) **+** `DTWEXAFEGS`
   (2006→vivo) con solape 2006-2019 para el broad-major; y `DTWEXBGS` (broad total vivo). Para EM:
   `DTWEXO` (1995-2019) **+** `DTWEXEMEGS` (2006→vivo).
3. **Dólar de baja frecuencia (Pista A, deep):** `RNUSBIS`/`NNUSBIS` (BIS narrow REER **mensual
   1964**) — el dólar más profundo verificado; empalma conceptualmente con DXY/DTWEXM.
4. **Refugio vs riesgo (features de régimen de primer orden, FRED diario 1971):**
   refugio = `DEXJPUS` (USDJPY), `DEXSZUS` (USDCHF); riesgo = `DEXUSAL` (AUDUSD), `DEXUSNZ`
   (NZDUSD), `DEXCAUS`/`DEXNOUS` (petro-FX). Feature clave: **AUDJPY = DEXUSAL / (1/DEXJPUS)**
   (termómetro carry / risk-on-off).
5. **Cesta EM (risk-off directo):** `DEXMXUS` (USDMXN, barómetro), `DEXSFUS` (USDZAR),
   `DEXBZUS` (USDBRL), `DEXKOUS` (USDKRW), `DEXINUS` (USDINR), `DEXCHUS` (USDCNY) + `USDTRY=X`
   y `USDRUB=X` (yfinance, no están en FRED). El índice `DTWEXEMEGS` resume la cesta.
6. **Euro:** `DEXUSEU` (EURUSD, FRED, 1999). Pre-1999 no hay par diario gratis verificado — el
   régimen del dólar pre-euro se lee vía la broad dollar (DTWEXM 1973) y RNUSBIS (1964).
7. **Features derivadas sugeridas (capa causal/expanding):** retorno del USD (broad),
   dispersión entre pares (std de retornos FX), spread refugio−riesgo (JPY+CHF vs AUD+EM),
   beta EM al dólar (DTWEXEMEGS vs DTWEXBGS), y régimen de carry (AUDJPY, NZDJPY).
8. **Punto ciego 2013 — cubierto:** todas las series FX diarias tienen 249-261 obs en 2013
   (taper tantrum limpio: MXN, INR, ZAR, BRL se depreciaron con fuerza — señal EM nítida).
9. **Fallbacks:** yfinance pares (`EURUSD=X`, `JPY=X`, `USDMXN=X`, 2003-2006) como redundancia
   de fuente 1:1 pero poco profundos. **Stooq NO usable** (bloqueado). `DX=F` NO disponible (404).
```
