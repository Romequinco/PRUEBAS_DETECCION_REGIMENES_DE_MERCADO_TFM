# Categoría: Commodities — estado del arte de datos (verificado)

Investigador: agente **Commodities**. Foco encargado: **oro** (GLD, futuros GC=F, FRED), **petróleo**
(CL=F/WTI, Brent, FRED `DCOILWTICO`), **cobre** (HG=F), **índices** de materias primas (S&P GSCI,
Bloomberg Commodity vía ETF) y el **ratio oro/cobre**. Los commodities son una feature de régimen
distinta y complementaria a crédito/vol: son **procíclicos** (el índice amplio se hunde en recesión y
risk-off: 2008, 2015-16, 2020), el **oro** es el activo refugio por excelencia (sube en estrés
monetario y risk-off), y el **ratio oro/cobre** (*"Dr. Copper" vs safe-haven*) es un termómetro clásico
de *growth-scare* / risk-off que anticipa cambios de régimen.

Todas las verificaciones se hicieron **de verdad** el 2026-07-18: FRED API con `FRED_API_KEY` del
`.env` (sin imprimir la clave), yfinance `period='max'` (auto_adjust) y pruebas reales a Stooq. Reporto
la fecha de inicio **observada**, no la de marketing.

---

## ⚠️ Hallazgo central nº1: el precio DIARIO del oro desapareció de FRED (misma licencia ICE que rompió las OAS)

La serie histórica de referencia del oro en FRED, **`GOLDPMGBD228NLBM`** (London Bullion Market Gold
Fixing PM, diaria desde 1968), **ya no existe**:

- `GET /series/observations?series_id=GOLDPMGBD228NLBM` → **HTTP 400 Bad Request** (igual la AM
  `GOLDAMGBD228NLBM`).
- Búsquedas FRED de *"Gold Fixing"*, *"London Bullion Market"* y *"gold london fixing"* → **0
  resultados**. La serie fue retirada, no es un fallo de parámetros.
- Es **el mismo patrón de licencia ICE Benchmark Administration** que el agente de *crédito* documentó
  para las OAS BofA: FRED perdió el derecho a redistribuir el fixing LBMA (administrado por ICE).

**Implicación:** con fuentes gratis y vivas **no hay precio diario de oro en FRED**. Lo que queda en
FRED del oro son índices de precio de import/export mensuales (`IQ12260` 1984+, `IR14270` 1992+) —
proxies pobres, mensuales, no el spot. **El oro diario hay que sacarlo de yfinance**: futuros `GC=F`
(desde 2000-08-30) y el ETF `GLD` (2004-11-18). Cubren 2008/2011/2015/2020 con nitidez, así que para
Pista B es suficiente; el hueco es sólo la **profundidad pre-2000** del oro (que tapan parcialmente los
índices amplios de commodities, ver hallazgo nº2).

## ⚠️ Hallazgo central nº2: los índices amplios de commodities NO están en FRED, pero Yahoo los sirve con historia profunda

FRED **no tiene** S&P GSCI ni Bloomberg Commodity Index (búsquedas "GSCI" y "Bloomberg commodity
index" → 0 resultados). Lo que FRED sí tiene son sus **propios índices IMF/World Bank** de precios
globales (mensuales, 1992+): `PALLFNFINDEXM` (all commodities), `PNRGINDEXM` (energía), `PMETAINDEXM`
(metales), `PINDUINDEXM` (industriales). Útiles pero **mensuales y sólo desde 1992**.

La joya está en **yfinance como tickers de índice**, verificados con valores coherentes en crisis:

- **`^SPGSCI`** (S&P GSCI, *energy-heavy*): **10 717 días desde 1984-01-03**. Sanity real: pico ~489
  (jun-2007) → **349 en dic-2008** (colapso del petróleo) → 658 (jun-2014) → **300 en ene-2016** →
  **255 en el suelo COVID (mar-2020)** → 709 (jun-2022). Comportamiento de régimen impecable.
- **`^BCOM`** (Bloomberg Commodity, diversificado): **8 918 días desde 1991-01-02**. Sanity: 170
  (2007) → 117 (dic-2008) → **77 (ene-2016)** → **61 (suelo COVID)** → 127 (2026). Correcto.

Advertencia de robustez: son *tickers de índice* de Yahoo (no ETFs), pueden tener algún día de retraso
(`^BCOM` cerró en 2026-07-10 en mi descarga) o discontinuarse sin aviso. **Fallback diario fiable = los
ETFs**: `GSG` (iShares GSCI, 2006), `DBC` (DB Commodity, 2006), `DJP` (iPath Bloomberg, 2006).

## ⚠️ Hallazgo central nº3: el petróleo WTI cotizó NEGATIVO (-36.98) el 2020-04-20

Verificado en `DCOILWTICO`: `2020-04-17: 18.31 → 2020-04-20: -36.98 → 2020-04-21: 8.91`. **El precio
del petróleo puede ser negativo**, así que **los log-returns rompen** ahí. El pipeline debe: (a) usar
retornos simples o diferencias en el crudo, o (b) apoyarse en el **futuro front-month / ETF** (`CL=F`,
`USO`) y en los **índices** (`^SPGSCI`) que no tienen ese artefacto de vencimiento. Es un caso de test
obligado para la robustez del detector de régimen.

## Fuentes que NO funcionan hoy (probadas)

- **Stooq**: bloqueado por challenge JavaScript (igual que reportaron crédito y volatilidad).
  `stooq.com/q/d/l/?s=xauusd&i=d`, `gc.f`, `cl.f`, `hg.f`, `gld.us` → devuelven el HTML `<noscript>`,
  no el CSV. **No usable sin navegador.** No hay fallback Stooq para commodities en este entorno.
- **FRED gold daily** (`GOLDPMGBD228NLBM` / `GOLDAMGBD228NLBM`): HTTP 400, retiradas (licencia ICE).
- **`JJC`** (iPath Copper ETN): descargable pero **delistado**, datos terminan 2023-07-21. No vivo.

---

## Resumen ejecutivo de lo verificado

**Spine profundo mensual (Pista A, FRED, todo vivo):**
- `PPIACO` — PPI All Commodities, **mensual desde 1913-01** (1362 obs). El termómetro de precios de
  materias primas más profundo que existe gratis; cubre la Gran Depresión y todas las crisis.
- `WTISPLC` — WTI spot mensual **desde 1946-01** (966 obs). Petróleo con memoria de 80 años.
- `WPU0561` (PPI combustibles, 1947+), `WPU102301` (PPI metales/cobre, 1957+).

**Spine diario profundo (Pista A/B, yfinance índices):**
- `^SPGSCI` (1984+, energy-heavy) y `^BCOM` (1991+, diversificado). Los únicos **índices amplios
  diarios** con crisis pre-2000 verificados.

**Núcleo diario Pista B (2000+, cubren 2008/2015/2020):**
- Oro: `GC=F` (futuros, 2000-08-30), `GLD` (ETF, 2004-11-18), `IAU` (2005).
- Petróleo: `DCOILWTICO` (FRED, diario 1986+), `DCOILBRENTEU` (Brent, 1987+), `CL=F`/`BZ=F`, `USO`.
- Cobre: `HG=F` (futuros, 2000-08-30), `CPER` (ETF, 2011), `PCOPPUSDM` (FRED mensual 1992+).
- **`GOLD_COPPER_RATIO`** (`GC=F/HG=F`, 2000+): derivada estrella de régimen. Picos **633 (dic-2008)**
  y **707 (suelo COVID mar-2020)**, suelo **162 en el boom de 2006**. Sube en risk-off / growth-scare.

**Enrichers:** plata (`SI=F` 2000, `SLV` 2006), platino (`PL=F` 1997), paladio (`PA=F` 1998), gas
natural (`NG=F` 2000, `DHHNGSP` FRED diario 1997+, `UNG`), mineras de oro (`GDX` 2006), ETFs amplios
(`GSG`/`DBC`/`DJP`/`GCC`), índices IMF de FRED (`PNRGINDEXM`/`PMETAINDEXM`/`PINDUINDEXM`, 1992+).

**Validación / estrés (commodity-específico):** `GVZCLS` — CBOE **Gold** ETF Volatility Index (GVZ),
diario **2008-06-03+**, vivo. El "VIX del oro"; picos = estrés en el refugio. (Análogo del petróleo
`OVXCLS` existe pero **no lo verifiqué** aquí → marcado `verificado=false`.)

---

## Detalle por serie (evidencia)

| serie | fuente | inicio verificado | fin/estado | nota |
|---|---|---|---|---|
| PPI All Commodities | FRED PPIACO | 1913-01-01 (M) | vivo | **spine más profundo**, cubre 1930s |
| WTI spot mensual | FRED WTISPLC | 1946-01-01 (M) | vivo | petróleo 80 años |
| PPI combustibles | FRED WPU0561 | 1947-01-01 (M) | vivo | energía profunda |
| PPI metales (cobre) | FRED WPU102301 | 1957-01-01 (M) | vivo | metales profundo |
| **S&P GSCI** | yfinance ^SPGSCI | 1984-01-03 | vivo | **índice amplio diario, 1984** |
| **Bloomberg Commodity** | yfinance ^BCOM | 1991-01-02 | vivo (~1d lag) | índice diversificado diario |
| WTI diario | FRED DCOILWTICO | 1986-01-02 | vivo | **-36.98 el 2020-04-20** |
| Brent diario | FRED DCOILBRENTEU | 1987-05-20 | vivo | crudo europeo |
| Oro futuros | yfinance GC=F | 2000-08-30 | vivo | oro diario más profundo (libre) |
| Oro ETF | yfinance GLD | 2004-11-18 | vivo | refugio, total-return |
| Oro ETF | yfinance IAU | 2005-01-28 | vivo | fallback de GLD |
| Cobre futuros | yfinance HG=F | 2000-08-30 | vivo | "Dr. Copper" diario |
| Cobre ETF | yfinance CPER | 2011-11-15 | vivo | fallback cobre |
| Cobre global | FRED PCOPPUSDM | 1992-01-01 (M) | vivo | cobre mensual $/t |
| **Ratio oro/cobre** | calc GC=F/HG=F | 2000-08-30 | vivo | **633 en 2008, 707 en COVID** |
| Plata futuros | yfinance SI=F | 2000-08-30 | vivo | metal híbrido |
| Plata ETF | yfinance SLV | 2006-04-28 | vivo | fallback plata |
| Platino futuros | yfinance PL=F | 1997-10-29 | vivo | PGM, ciclo industrial |
| Paladio futuros | yfinance PA=F | 1998-09-28 | vivo | PGM/autos |
| Gas natural futuros | yfinance NG=F | 2000-08-30 | vivo | energía idiosincrática |
| Henry Hub gas | FRED DHHNGSP | 1997-01-07 | vivo | gas spot diario |
| Oil ETF | yfinance USO | 2006-04-10 | vivo | proxy WTI sin negativo |
| NatGas ETF | yfinance UNG | 2007-04-18 | vivo | proxy gas |
| GSCI ETF | yfinance GSG | 2006-07-21 | vivo | fallback fiable de ^SPGSCI |
| DB Commodity ETF | yfinance DBC | 2006-02-06 | vivo | índice amplio ETF |
| Bloomberg Cmdty ETN | yfinance DJP | 2006-10-30 | vivo | fallback fiable de ^BCOM |
| Cont. Commodity ETF | yfinance GCC | 2008-01-24 | vivo | índice equal-weight |
| Gold Miners ETF | yfinance GDX | 2006-05-22 | vivo | beta apalancada al oro |
| IMF energía | FRED PNRGINDEXM | 1992-01-01 (M) | vivo | índice energía mensual |
| IMF metales | FRED PMETAINDEXM | 1992-01-01 (M) | vivo | índice metales mensual |
| IMF industriales | FRED PINDUINDEXM | 1992-01-01 (M) | vivo | inputs industriales |
| IMF all commodities | FRED PALLFNFINDEXM | 1992-01-01 (M) | vivo | índice global amplio |
| IMF WTI mensual | FRED POILWTIUSDM | 1992-01-01 (M) | vivo | WTI mensual homogéneo |
| Export price oro | FRED IQ12260 | 1984-12-01 (M) | vivo | **proxy pobre** de oro |
| Gold VIX (GVZ) | FRED GVZCLS | 2008-06-03 | vivo | validación estrés oro |
| Gold Fixing PM | FRED GOLDPMGBD228NLBM | — | **RETIRADA (400)** | licencia ICE, ya no existe |
| Oil VIX (OVX) | FRED OVXCLS | — | no probada | análogo petróleo, sin verificar |
| Copper ETN | yfinance JJC | 2018→2023 | **delistado** | no vivo |

**Features derivadas recomendadas (causales/expanding):**
- **`GOLD_COPPER_RATIO`** (nivel + Δ): risk-off / growth-scare. La feature de régimen más limpia aquí.
- **Retornos y vol realizada de `^SPGSCI`/`^BCOM`**: drawdowns persistentes = régimen de recesión.
- **Oro vs índice amplio** (`GLD`/`DBC` o `GC=F`/`^SPGSCI`): refugio relativo, sube en risk-off.
- **Brent−WTI** (`DCOILBRENTEU−DCOILWTICO`): estrés logístico/geopolítico de crudo.
- **`GVZCLS`** (nivel): estrés específico del oro, complementa al VIX de equity.

```yaml
series_commodities:
  # ============ SPINE PROFUNDO MENSUAL (FRED, historia completa, vivo) ============
  - nombre_interno: PPI_ALL_COMMODITIES
    descripcion: "Producer Price Index by Commodity: All Commodities. El índice de precios de materias primas más profundo disponible gratis (mensual desde 1913)."
    fuente: fred
    id: "PPIACO"
    auth: FRED_API_KEY
    inicio_verificado: "1913-01-01"
    granularidad: mensual
    pista: A
    rol: spine
    relevancia_regimen: "+110 años de precios de commodities; cubre Gran Depresión, estanflación 70s, todas las crisis modernas. Deflacion/inflacion de materias primas por régimen."
    verificado: true
    evidencia: "FRED PPIACO -> 1362 obs mensuales, first 1913-01-01:12.1 last 2026-06-01:286.827 (upd 2026-07-15, Index 1982=100)."
    url: "https://fred.stlouisfed.org/series/PPIACO"

  - nombre_interno: WTI_SPOT_MONTHLY
    descripcion: "Spot Crude Oil Price: West Texas Intermediate (WTI), mensual. Serie construida por la Fed de St. Louis para extender la historia del petróleo."
    fuente: fred
    id: "WTISPLC"
    auth: FRED_API_KEY
    inicio_verificado: "1946-01-01"
    granularidad: mensual
    pista: A
    rol: spine
    relevancia_regimen: "Petróleo con memoria de 80 años; shocks del crudo (1973, 1979, 1990, 2008, 2020) marcan regímenes de inflación/recesión."
    verificado: true
    evidencia: "FRED WTISPLC -> 966 obs, first 1946-01-01:1.17 last 2026-06-01:84.81 (upd 2026-07-08, $/barril)."
    url: "https://fred.stlouisfed.org/series/WTISPLC"

  - nombre_interno: PPI_FUELS
    descripcion: "Producer Price Index by Commodity: Fuels and Related Products. PPI de energía, mensual desde 1947."
    fuente: fred
    id: "WPU0561"
    auth: FRED_API_KEY
    inicio_verificado: "1947-01-01"
    granularidad: mensual
    pista: A
    rol: enricher
    relevancia_regimen: "Componente energético del PPI con historia larga; regímenes de shock de oferta energética."
    verificado: true
    evidencia: "FRED WPU0561 -> 954 obs, first 1947-01-01:7.3 last 2026-06-01:263.364 (upd 2026-07-15)."
    url: "https://fred.stlouisfed.org/series/WPU0561"

  - nombre_interno: PPI_METALS
    descripcion: "Producer Price Index by Commodity: Metals and Metal Products. PPI de metales (incl. cobre), mensual desde 1957."
    fuente: fred
    id: "WPU102301"
    auth: FRED_API_KEY
    inicio_verificado: "1957-01-01"
    granularidad: mensual
    pista: A
    rol: enricher
    relevancia_regimen: "Metales industriales con historia larga; proxy profundo del ciclo industrial (Dr. Copper) donde no llega HG=F."
    verificado: true
    evidencia: "FRED WPU102301 -> 834 obs, first 1957-01-01:58.7 last 2026-06-01:953.169 (upd 2026-07-15)."
    url: "https://fred.stlouisfed.org/series/WPU102301"

  # ============ SPINE DIARIO PROFUNDO: INDICES AMPLIOS (yfinance) ============
  - nombre_interno: SPGSCI
    descripcion: "S&P GSCI (Goldman Sachs Commodity Index), ponderado por produccion (energy-heavy). Ticker de indice de Yahoo, diario desde 1984."
    fuente: yfinance
    id: "^SPGSCI"
    auth: none
    inicio_verificado: "1984-01-03"
    granularidad: diaria
    pista: ambas
    rol: spine
    relevancia_regimen: "Indice amplio de commodities con mas historia diaria gratis. Procíclico: 489(2007)->349(dic-2008)->300(2016)->255(COVID). Discriminador de recesion/risk-off."
    verificado: true
    evidencia: "yf.download('^SPGSCI','max') -> 10717 filas 1984-01-03 -> 2026-07-17. Sanity crisis verificado (dic-2008=349, mar-2020=255)."
    url: "https://finance.yahoo.com/quote/%5ESPGSCI"

  - nombre_interno: BCOM
    descripcion: "Bloomberg Commodity Index, diversificado con caps por sector. Ticker de indice de Yahoo, diario desde 1991."
    fuente: yfinance
    id: "^BCOM"
    auth: none
    inicio_verificado: "1991-01-02"
    granularidad: diaria
    pista: ambas
    rol: spine
    relevancia_regimen: "Indice amplio menos sesgado a energia que GSCI. 170(2007)->117(dic-2008)->77(2016)->61(COVID). Complemento diversificado del GSCI."
    verificado: true
    evidencia: "yf.download('^BCOM','max') -> 8918 filas 1991-01-02 -> 2026-07-10 (a veces ~1 dia de lag). Sanity crisis verificado."
    url: "https://finance.yahoo.com/quote/%5EBCOM"

  # ============ PETROLEO (FRED diario + futuros) ============
  - nombre_interno: WTI_DAILY
    descripcion: "Crude Oil Prices: West Texas Intermediate (WTI) - Cushing. Precio spot diario de referencia US."
    fuente: fred
    id: "DCOILWTICO"
    auth: FRED_API_KEY
    inicio_verificado: "1986-01-02"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Petroleo diario 1986+; shocks de crudo = regimenes de inflacion/recesion. CAVEAT: cotizo -36.98 el 2020-04-20 (log-returns rompen, usar retorno simple)."
    verificado: true
    evidencia: "FRED DCOILWTICO -> 10200 obs, first 1986-01-02:25.56 last 2026-07-13:79.2. Verificado 2020-04-20:-36.98 (2020-04-17:18.31->2020-04-21:8.91)."
    url: "https://fred.stlouisfed.org/series/DCOILWTICO"

  - nombre_interno: BRENT_DAILY
    descripcion: "Crude Oil Prices: Brent - Europe. Precio spot diario del crudo de referencia global."
    fuente: fred
    id: "DCOILBRENTEU"
    auth: FRED_API_KEY
    inicio_verificado: "1987-05-20"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Crudo global; el spread Brent-WTI aisla estres logistico/geopolitico vs desbalance US. No sufrio el negativo de 2020."
    verificado: true
    evidencia: "FRED DCOILBRENTEU -> 9932 obs, first 1987-05-20:18.63 last 2026-07-13:81.62 (upd 2026-07-15)."
    url: "https://fred.stlouisfed.org/series/DCOILBRENTEU"

  - nombre_interno: WTI_FUT
    descripcion: "Futuro continuo WTI COMEX/NYMEX (front-month). Alternativa a DCOILWTICO con OHLC."
    fuente: yfinance
    id: "CL=F"
    auth: none
    inicio_verificado: "2000-08-23"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Redundancia de fuente para WTI diario; retornos de futuro para vol realizada del crudo."
    verificado: true
    evidencia: "yf.download('CL=F','max') -> 6502 filas 2000-08-23 -> 2026-07-17."
    url: "https://finance.yahoo.com/quote/CL=F"

  - nombre_interno: OIL_ETF_USO
    descripcion: "United States Oil Fund. ETF sobre futuros WTI; proxy sin precio negativo (aunque con roll)."
    fuente: yfinance
    id: "USO"
    auth: none
    inicio_verificado: "2006-04-10"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Proxy investible del petroleo; total-return sin el artefacto de -37 de 2020."
    verificado: true
    evidencia: "yf.download('USO','max') -> 5099 filas 2006-04-10 -> 2026-07-17."
    url: "https://finance.yahoo.com/quote/USO"

  # ============ ORO (yfinance; FRED diario ya NO existe) ============
  - nombre_interno: GOLD_FUT
    descripcion: "Futuro continuo del oro COMEX (front-month, $/oz). La serie de oro diaria mas profunda disponible gratis y viva."
    fuente: yfinance
    id: "GC=F"
    auth: none
    inicio_verificado: "2000-08-30"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Activo refugio nucleo; sube en risk-off y estres monetario. Cubre 2008/2011/2015/2020. Componente del ratio oro/cobre."
    verificado: true
    evidencia: "yf.download('GC=F','max') -> 6493 filas 2000-08-30 -> 2026-07-17."
    url: "https://finance.yahoo.com/quote/GC=F"

  - nombre_interno: GOLD_ETF_GLD
    descripcion: "SPDR Gold Shares. ETF fisico de oro, total-return, el mas liquido."
    fuente: yfinance
    id: "GLD"
    auth: none
    inicio_verificado: "2004-11-18"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Proxy investible del oro; refugio. GLD vs indice amplio (DBC) = oro relativo, sube en risk-off."
    verificado: true
    evidencia: "yf.download('GLD','max') -> 5448 filas 2004-11-18 -> 2026-07-17."
    url: "https://finance.yahoo.com/quote/GLD"

  - nombre_interno: GOLD_ETF_IAU
    descripcion: "iShares Gold Trust. ETF fisico de oro; fallback de GLD."
    fuente: yfinance
    id: "IAU"
    auth: none
    inicio_verificado: "2005-01-28"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Redundancia de fuente del oro por si falla GLD/GC=F."
    verificado: true
    evidencia: "yf.download('IAU','max') -> 5400 filas 2005-01-28 -> 2026-07-17."
    url: "https://finance.yahoo.com/quote/IAU"

  # ============ COBRE ============
  - nombre_interno: COPPER_FUT
    descripcion: "Futuro continuo del cobre COMEX (HG, $/lb). 'Dr. Copper', termometro del ciclo industrial global."
    fuente: yfinance
    id: "HG=F"
    auth: none
    inicio_verificado: "2000-08-30"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Metal mas ciclico; cae fuerte en desaceleracion. Denominador del ratio oro/cobre. Cubre 2008/2015/2020."
    verificado: true
    evidencia: "yf.download('HG=F','max') -> 6498 filas 2000-08-30 -> 2026-07-17."
    url: "https://finance.yahoo.com/quote/HG=F"

  - nombre_interno: COPPER_ETF_CPER
    descripcion: "United States Copper Index Fund. ETF sobre futuros de cobre; fallback investible de HG=F."
    fuente: yfinance
    id: "CPER"
    auth: none
    inicio_verificado: "2011-11-15"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Redundancia de fuente para el cobre desde 2011."
    verificado: true
    evidencia: "yf.download('CPER','max') -> 3687 filas 2011-11-15 -> 2026-07-17."
    url: "https://finance.yahoo.com/quote/CPER"

  - nombre_interno: COPPER_GLOBAL_MONTHLY
    descripcion: "Global price of Copper (IMF), $/tonelada metrica, mensual desde 1992."
    fuente: fred
    id: "PCOPPUSDM"
    auth: FRED_API_KEY
    inicio_verificado: "1992-01-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Cobre mensual homogeneo IMF; ancla de nivel para el ciclo industrial. Fallback mensual de HG=F."
    verificado: true
    evidencia: "FRED PCOPPUSDM -> 414 obs, first 1992-01-01:2150.58 last 2026-06-01:13552.04 (upd 2026-07-13, $/t)."
    url: "https://fred.stlouisfed.org/series/PCOPPUSDM"

  # ============ DERIVADA ESTRELLA: RATIO ORO/COBRE ============
  - nombre_interno: GOLD_COPPER_RATIO
    descripcion: "Ratio oro/cobre (GC=F / HG=F). Safe-haven vs 'Dr. Copper': termometro clasico de risk-off / growth-scare."
    fuente: yfinance
    id: "GC=F,HG=F"
    auth: none
    inicio_verificado: "2000-08-30"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Sube cuando el oro (refugio) supera al cobre (crecimiento) = risk-off/recesion. Picos 633 (dic-2008) y 707 (suelo COVID mar-2020); suelo 162 en el boom 2006. Feature de regimen limpia."
    verificado: true
    evidencia: "Calc yf GC=F/HG=F -> 6492 pts desde 2000-08-30. Verificado: dic-2008=633, mar-2020=707, ene-2016=541, 2006-05=162 (min). max 902 el 2026-02-23."
    url: "https://finance.yahoo.com/quote/GC=F"

  # ============ ENRICHERS: OTROS METALES / ENERGIA ============
  - nombre_interno: SILVER_FUT
    descripcion: "Futuro continuo de la plata COMEX (SI, $/oz). Metal hibrido monetario/industrial."
    fuente: yfinance
    id: "SI=F"
    auth: none
    inicio_verificado: "2000-08-30"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Oro/plata ratio = apetito de riesgo dentro de los metales preciosos; plata mas beta que oro."
    verificado: true
    evidencia: "yf.download('SI=F','max') -> 6495 filas 2000-08-30 -> 2026-07-17."
    url: "https://finance.yahoo.com/quote/SI=F"

  - nombre_interno: SILVER_ETF_SLV
    descripcion: "iShares Silver Trust. ETF fisico de plata; fallback de SI=F."
    fuente: yfinance
    id: "SLV"
    auth: none
    inicio_verificado: "2006-04-28"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Redundancia de fuente para la plata."
    verificado: true
    evidencia: "yf.download('SLV','max') -> 5086 filas 2006-04-28 -> 2026-07-17."
    url: "https://finance.yahoo.com/quote/SLV"

  - nombre_interno: PLATINUM_FUT
    descripcion: "Futuro continuo del platino NYMEX (PL). PGM, muy ligado al ciclo industrial/automocion."
    fuente: yfinance
    id: "PL=F"
    auth: none
    inicio_verificado: "1997-10-29"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Metal precioso industrial; platino/oro ratio marca preferencia industrial vs refugio. Historia diaria desde 1997."
    verificado: true
    evidencia: "yf.download('PL=F','max') -> 6521 filas 1997-10-29 -> 2026-07-17."
    url: "https://finance.yahoo.com/quote/PL=F"

  - nombre_interno: PALLADIUM_FUT
    descripcion: "Futuro continuo del paladio NYMEX (PA). PGM de automocion, muy ciclico."
    fuente: yfinance
    id: "PA=F"
    auth: none
    inicio_verificado: "1998-09-28"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Metal industrial de nicho; complementa la lectura del ciclo manufacturero."
    verificado: true
    evidencia: "yf.download('PA=F','max') -> 6532 filas 1998-09-28 -> 2026-07-17."
    url: "https://finance.yahoo.com/quote/PA=F"

  - nombre_interno: NATGAS_FUT
    descripcion: "Futuro continuo del gas natural Henry Hub NYMEX (NG). Energia idiosincratica/estacional."
    fuente: yfinance
    id: "NG=F"
    auth: none
    inicio_verificado: "2000-08-30"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Energia con dinamica propia (clima, almacenamiento); util para separar shocks de energia de shocks macro amplios."
    verificado: true
    evidencia: "yf.download('NG=F','max') -> 6499 filas 2000-08-30 -> 2026-07-17."
    url: "https://finance.yahoo.com/quote/NG=F"

  - nombre_interno: NATGAS_HENRYHUB
    descripcion: "Henry Hub Natural Gas Spot Price (FRED/EIA), diario desde 1997."
    fuente: fred
    id: "DHHNGSP"
    auth: FRED_API_KEY
    inicio_verificado: "1997-01-07"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Spot de gas diario, fuente oficial EIA; alternativa a NG=F sin roll de futuros."
    verificado: true
    evidencia: "FRED DHHNGSP -> 7410 obs, first 1997-01-07:3.82 last 2026-07-13:2.83 (upd 2026-07-15, $/MMBtu)."
    url: "https://fred.stlouisfed.org/series/DHHNGSP"

  - nombre_interno: GOLD_MINERS_GDX
    descripcion: "VanEck Gold Miners ETF. Beta apalancada al precio del oro (equity de mineras)."
    fuente: yfinance
    id: "GDX"
    auth: none
    inicio_verificado: "2006-05-22"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "GDX vs GLD = apetito de riesgo dentro del complejo oro (equity vs metal fisico); amplifica los giros de regimen."
    verificado: true
    evidencia: "yf.download('GDX','max') -> 5070 filas 2006-05-22 -> 2026-07-17."
    url: "https://finance.yahoo.com/quote/GDX"

  # ============ INDICES AMPLIOS ETF (fallbacks fiables de ^SPGSCI/^BCOM) ============
  - nombre_interno: GSCI_ETF_GSG
    descripcion: "iShares S&P GSCI Commodity-Indexed Trust. Fallback investible y FIABLE del ticker ^SPGSCI."
    fuente: yfinance
    id: "GSG"
    auth: none
    inicio_verificado: "2006-07-21"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Robustez: si Yahoo retira el ticker de indice ^SPGSCI, GSG lo replica desde 2006."
    verificado: true
    evidencia: "yf.download('GSG','max') -> 5028 filas 2006-07-21 -> 2026-07-17."
    url: "https://finance.yahoo.com/quote/GSG"

  - nombre_interno: DBC_ETF
    descripcion: "Invesco DB Commodity Index Tracking Fund. Indice amplio diversificado (14 commodities)."
    fuente: yfinance
    id: "DBC"
    auth: none
    inicio_verificado: "2006-02-06"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Indice amplio investible; benchmark de commodities como clase de activo para el regimen."
    verificado: true
    evidencia: "yf.download('DBC','max') -> 5143 filas 2006-02-06 -> 2026-07-17."
    url: "https://finance.yahoo.com/quote/DBC"

  - nombre_interno: BCOM_ETN_DJP
    descripcion: "iPath Bloomberg Commodity Index Total Return ETN. Fallback investible FIABLE del ticker ^BCOM."
    fuente: yfinance
    id: "DJP"
    auth: none
    inicio_verificado: "2006-10-30"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Robustez: replica el Bloomberg Commodity si Yahoo retira ^BCOM. Desde 2006."
    verificado: true
    evidencia: "yf.download('DJP','max') -> 4958 filas 2006-10-30 -> 2026-07-17."
    url: "https://finance.yahoo.com/quote/DJP"

  # ============ INDICES IMF/WB MENSUALES (FRED, Pista B) ============
  - nombre_interno: IMF_ENERGY_INDEX
    descripcion: "Global price of Energy index (IMF), mensual base 2016=100 desde 1992."
    fuente: fred
    id: "PNRGINDEXM"
    auth: FRED_API_KEY
    inicio_verificado: "1992-01-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Subindice energia global; separa el bloque energetico del resto de commodities en el régimen."
    verificado: true
    evidencia: "FRED PNRGINDEXM -> 414 obs, first 1992-01-01:44.45 last 2026-06-01:198.80 (upd 2026-07-13)."
    url: "https://fred.stlouisfed.org/series/PNRGINDEXM"

  - nombre_interno: IMF_METALS_INDEX
    descripcion: "Global price of Metal index (IMF), mensual base 2016=100 desde 1992."
    fuente: fred
    id: "PMETAINDEXM"
    auth: FRED_API_KEY
    inicio_verificado: "1992-01-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Subindice metales industriales global; lectura del ciclo manufacturero mundial."
    verificado: true
    evidencia: "FRED PMETAINDEXM -> 414 obs, first 1992-01-01:46.95 last 2026-06-01:230.49 (upd 2026-07-13)."
    url: "https://fred.stlouisfed.org/series/PMETAINDEXM"

  - nombre_interno: IMF_ALLCOMMODITIES_INDEX
    descripcion: "Global Price Index of All Commodities (IMF), mensual base 2016=100 desde 1992."
    fuente: fred
    id: "PALLFNFINDEXM"
    auth: FRED_API_KEY
    inicio_verificado: "1992-01-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Indice global amplio mensual; version FRED del concepto GSCI/BCOM para cross-check macro."
    verificado: true
    evidencia: "FRED PALLFNFINDEXM -> 414 obs, first 1992-01-01:48.05 last 2026-06-01:194.85 (upd 2026-07-13)."
    url: "https://fred.stlouisfed.org/series/PALLFNFINDEXM"

  # ============ VALIDACION / ESTRES (commodity-especifico) ============
  - nombre_interno: GOLD_VIX_GVZ
    descripcion: "CBOE Gold ETF Volatility Index (GVZ). Volatilidad implicita del oro (el 'VIX del oro'), diario desde 2008."
    fuente: fred
    id: "GVZCLS"
    auth: FRED_API_KEY
    inicio_verificado: "2008-06-03"
    granularidad: diaria
    pista: validacion
    rol: validation
    relevancia_regimen: "Ground-truth laxo de estres en el refugio: picos de GVZ = incertidumbre sobre el oro/regimen monetario. Complementa VIX de equity."
    verificado: true
    evidencia: "FRED search/meta GVZCLS -> 2008-06-03 -> 2026-07-16 diario (pop=70, CBOE Gold ETF Volatility Index)."
    url: "https://fred.stlouisfed.org/series/GVZCLS"

  # ============ NO VERIFICADOS / RETIRADOS / FALLBACKS ============
  - nombre_interno: GOLD_LBMA_FIXING_FRED
    descripcion: "Gold Fixing Price 3:00 PM (London) LBMA en USD. La serie de oro diaria historica de FRED (desde 1968)."
    fuente: fred
    id: "GOLDPMGBD228NLBM"
    auth: FRED_API_KEY
    inicio_verificado: null
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Habria dado oro diario 1968+ (mas profundo que GC=F). RETIRADA de FRED por licencia ICE/LBMA (mismo problema que las OAS)."
    verificado: false
    evidencia: "HTTP 400 Bad Request en observations; busquedas 'Gold Fixing'/'London Bullion' en FRED -> 0 resultados. Serie eliminada, no accesible."
    url: "https://fred.stlouisfed.org/series/GOLDPMGBD228NLBM"

  - nombre_interno: OIL_VIX_OVX
    descripcion: "CBOE Crude Oil ETF Volatility Index (OVX). Analogo del GVZ para el petroleo."
    fuente: fred
    id: "OVXCLS"
    auth: FRED_API_KEY
    inicio_verificado: null
    granularidad: diaria
    pista: validacion
    rol: validation
    relevancia_regimen: "Estres implicito del crudo (util: picos brutales en 2020). Muy probable que exista en FRED, pero NO lo verifique en este entorno."
    verificado: false
    evidencia: "No probada. El analogo GVZCLS si esta verificado; OVXCLS queda como candidata a confirmar."
    url: "https://fred.stlouisfed.org/series/OVXCLS"

  - nombre_interno: GOLD_STOOQ
    descripcion: "Oro/plata/futuros via Stooq CSV como fallback de yfinance."
    fuente: stooq
    id: "xauusd, gc.f, cl.f, hg.f"
    auth: none
    inicio_verificado: null
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Redundancia de fuente si cae yfinance."
    verificado: false
    evidencia: "BLOQUEADO: stooq.com/q/d/l/?s=xauusd&i=d (y gc.f/cl.f/hg.f/gld.us) devuelve HTML con <noscript> (challenge JS), no CSV. No usable sin navegador."
    url: "https://stooq.com/q/d/l/?s=xauusd&i=d"

  - nombre_interno: COPPER_ETN_JJC
    descripcion: "iPath Series B Bloomberg Copper ETN. Fallback de cobre."
    fuente: yfinance
    id: "JJC"
    auth: none
    inicio_verificado: "2018-01-17"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Descargable pero DELISTADO (no vivo); datos solo 2018-2023. Usar CPER/HG=F en su lugar."
    verificado: false
    evidencia: "yf.download('JJC','max') -> 1387 filas 2018-01-17 -> 2023-07-21 (delistado, no actualiza)."
    url: "https://finance.yahoo.com/quote/JJC"
```

---

## Recomendación de priorización para el pipeline

1. **Spine diario amplio (imprescindible, cubre crisis pre-2000):** `^SPGSCI` (1984+) y `^BCOM`
   (1991+) como índices de commodities diarios, con `GSG`/`DJP` como fallbacks ETF fiables. Es la
   columna vertebral procíclica de la categoría.
2. **Núcleo oro/petróleo/cobre diario (Pista B, 2000+):** `GC=F` (oro), `DCOILWTICO`+`DCOILBRENTEU`
   (petróleo), `HG=F` (cobre). Cubren 2008/2011/2015/2020. **Manejar el -36.98 de WTI (2020-04-20)**:
   retornos simples o apoyarse en `USO`/`^SPGSCI`.
3. **Feature derivada estrella:** `GOLD_COPPER_RATIO` (`GC=F/HG=F`). El indicador de risk-off /
   growth-scare más limpio de esta categoría (633 en 2008, 707 en COVID). Añadir también oro/plata y
   Brent−WTI como features de segundo orden.
4. **Spine profundo mensual (Pista A):** `PPIACO` (1913+), `WTISPLC` (1946+), `WPU102301` (metales
   1957+). Dan memoria de crisis pre-1984 (estanflación 70s, etc.) que ningún diario gratis alcanza.
5. **Validación:** `GVZCLS` (Gold VIX, 2008+) como estrés específico del refugio; confirmar `OVXCLS`
   (Oil VIX) para el análogo del crudo.
6. **Deuda técnica honesta:**
   - **No hay oro diario en FRED** (`GOLDPMGBD228NLBM` retirada por licencia ICE/LBMA, igual que las
     OAS de crédito). El oro diario **depende de yfinance** (`GC=F` 2000, `GLD` 2004): sin fallback
     libre profundo pre-2000. Si el TFM necesita oro diario pre-2000 habría que archivar un CSV LBMA
     externo (no verificado aquí).
   - **Stooq no es viable** en este entorno (challenge JS). yfinance es la única fuente diaria de
     futuros/ETFs de commodities disponible; conviene cachear las descargas por si Yahoo capa.
   - Los tickers de índice `^SPGSCI`/`^BCOM` pueden discontinuarse en Yahoo sin aviso → mantener
     `GSG`/`DJP` como respaldo ya cableado.

**Fuentes web/API consultadas:** FRED API (`api.stlouisfed.org`, key del `.env`) para series y
búsquedas · yfinance 0.2.66 (`period='max'`, auto_adjust) · Stooq (probado y bloqueado).
