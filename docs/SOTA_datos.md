# Estado del arte de datos — Detección de regímenes de mercado

**Proyecto:** banco de pruebas de detección de regímenes (Capa 1 de un TFM MIAX).
**Fecha de verificación de todo lo que sigue:** 2026-07-18.
**Método:** cada serie se **descargó de verdad** (FRED API con `FRED_API_KEY` del `.env`, `yfinance period='max'`, CSV de OFR, ficheros académicos y repos GitHub). Se reporta la fecha de inicio **observada**, no la de marketing, y el estado (vivo / descontinuado / retirado).

Este documento es el estado del arte **de los datos** (no de los detectores). Explica, por categoría, qué existe gratis para detección de regímenes, qué elegimos y por qué, desde cuándo (histórico real), de qué fuente gratuita, y cómo se reparte entre las dos pistas del proyecto.

---

## 1. Marco: las dos pistas, los roles y las fuentes

### 1.1. Las dos pistas (decisión ADR-001)

| Pista | Nombre | Ventana objetivo | Filosofía |
|---|---|---|---|
| **A** | Espina histórica **profunda** | ~1871/1926/1950 → hoy | **Pocas** features, máxima historia, **maximizar el nº de crisis** (Gran Depresión, 1937, estanflación 70s, 1987, dotcom, GFC, COVID…). Series espina mensuales/diarias muy largas. |
| **B** | Panel **rico** multi-activo | banco 2003+ (vol-of-vol 2007) → hoy | **Muchas** features cross-asset (vol, crédito, curva, FX, commodities, liquidez, macro). Granularidad y diversidad para atacar regímenes modernos (incluido el punto ciego de 2013). |

Muchas series sirven a **ambas** (p.ej. VIX desde 1990, DGS10 desde 1962): son profundas y a la vez parte del panel rico.

### 1.2. Roles de cada serie

- **spine** (espina): columna vertebral de máxima profundidad de su categoría; ancla de Pista A.
- **core** (núcleo): feature discriminante primaria del régimen.
- **enricher** (enriquecedor): feature secundaria que añade granularidad.
- **fallback** (respaldo): redundancia de fuente o sustituto si cae la principal.
- **validation** (validación): *ground truth* laxo; **no se usa como feature** (sería fuga de información).

### 1.3. Las fuentes gratuitas que funcionan (verificadas)

| Fuente | Auth | Qué aporta | Estado |
|---|---|---|---|
| **FRED** (`api.stlouisfed.org`) | `FRED_API_KEY` | Tipos, curva, crédito Moody's, macro, FX (H.10), funding, índices de estrés, NBER | Vivo, limpio, API |
| **yfinance** (Yahoo) | ninguna | VIX y familia CBOE, MOVE/VVIX, estructura VIX, ETFs, futuros de commodities, DXY, `^GSPC` diario 1927+, pares FX poco profundos | Vivo |
| **Académico** (Ken French, Shiller, Goyal-Welch, JST) | ninguna | Espina profunda de Pista A: Mkt-RF diario 1926+, S&P500+CAPE mensual 1871+, predictores 1871+, crisis internacionales 1870+ | Vivo (ficheros/Sheets) |
| **OFR** (`financialresearch.gov`, CSV) | ninguna | Financial Stress Index diario 2000+ con subíndices (Credit/Funding/Volatility…) | Vivo |
| **GitHub datahub.io** (PDDL) | ninguna | Mirrors sin auth: S&P500+CAPE 1871+, VIX diario 1990+, oro mensual 1833+, constituyentes point-in-time | Vivo |

### 1.4. Política de datos (dura, del `catalog.yaml`)

- **NUNCA imputar**: cada serie arranca en su fecha real; los huecos quedan `NaN`.
- **Causalidad**: toda feature derivada será z-score *expanding/rolling* (nunca de muestra completa).
- **Reproducibilidad**: `raw` con checksum + provenance; descargas versionadas.

---

## 2. Volatilidad

**Qué existe.** La volatilidad es la feature más discriminante de régimen, así que importa a **ambas pistas**. Pero **no hay volatilidad implícita antes de 1986**: el VIX arranca en 1990 y el VXO (metodología antigua, S&P100) en 1986. Para ir más atrás la única opción honesta es **volatilidad realizada** del S&P500.

**Qué elegimos y por qué.**
- **Espina profunda (Pista A):** `REALIZED_VOL_SP500`, una vol realizada rolling-21d anualizada calculada de `^GSPC` (yfinance), que existe desde **1928**. Encima, el **VXO** (FRED `VXOCLS`, 1986) empalma casi limpio con el VIX (media VXO−VIX = +0.27 en 7.990 días comunes) y cubre el crash de 1987 que el VIX no alcanza.
- **Núcleo (ambas):** el **VIX** (FRED `VIXCLS`, 1990) y el **MOVE** (yfinance `^MOVE`, 2002), que es el "VIX de los bonos" y no está en FRED. El MOVE actualiza con ~1 semana de retraso — a tener en cuenta para features vivas.
- **Panel rico (Pista B):** la **estructura temporal** del VIX (`VIX9D` 2011, `VIX3M` 2006, `VIX6M` 2008, todas yfinance) — la pendiente `VIX/VIX3M` es de las mejores señales de cambio de régimen; **VVIX** (vol-of-vol, 2007), **SKEW** (cola, 1990) y la familia cross-asset del CBOE en FRED: **VXN** (Nasdaq, 2001), **VXD** (DJIA, 1997 — FRED gana a yfinance que empieza en 2005), **RVX** (Russell 2000, 2004 — **solo** en FRED), **OVX** (petróleo, 2007), **GVZ** (oro, 2008), **VXEEM** (EM, 2011).

**Reparto A/B.** Pista A: RV realizada (1928) + VXO (1986). Pista B: todo el panel de implícitas modernas. VIX y SKEW sirven a ambas.

---

## 3. Crédito

**Qué existe — y el hallazgo que condiciona la categoría.** Las series ICE BofA **OAS por rating** de FRED (todas las `BAML…`) **dejaron de servir su historia completa**: desde abril-2026 FRED las limita a una **ventana rodante de 3 años** por licencia de ICE. Verificado: `BAMLH0A0HYM2` (HY OAS) arranca en 2023-07-18 (= hoy − 3 años) y forzar `observation_start=1996` no devuelve más. Es decir: **con fuentes gratis ya no se puede reconstruir la historia larga de las OAS por rating** (2008/2011/2015/2020 quedan fuera). La mayor pérdida es la **CCC OAS**, el mejor termómetro de estrés de crédito.

**Qué elegimos y por qué.**
- **Espina profunda (Pista A/ambas):** el **spread Baa−Aaa de Moody's** (`DBAA−DAAA`, diario desde 1986) y **`BAA10Y`** (Baa − Treasury 10y, diario 1986), más `BAA`/`AAA` **mensuales desde 1919**. Es grado de inversión, no HY, pero es el índice de riesgo de crédito con memoria de +100 años que **sí sobrevive** (Moody's no está capado como ICE). Sanity check: Baa−Aaa pico 3.50 en dic-2008, 1.99 en 2020, 1.46 en 2011. Es el **sustituto honesto de las OAS** para historia larga.
- **Proxies de HY con historia real (Pista B):** el par **HYG vs IEF** (spread `HYG−IEF`, 2007) y `HYG/LQD`. Cubren 2008 y COVID con nitidez (HYG/IEF rebased-100: mín 56 en dic-2008, 76 en mar-2020) justo donde la OAS de FRED ya no llega. Más `LQD` (IG, 2002), `EMB` (EM, 2007), `BKLN` (loans, 2011), `AGG` (2003).
- **OAS por rating como features VIVAS (Pista B, solo 3 años):** `HY_OAS`, `IG_OAS`, `CCC_OAS`, `BB_OAS`, `B_OAS`, `BBB_OAS`, `EM_CORP_OAS`, `EURO_HY_OAS`. Granularidad por rating imbatible para el régimen **actual**, pero sin histórico de crisis. El pipeline debe estar diseñado para no depender de su profundidad.

**Reparto A/B.** Pista A: Baa−Aaa y Baa−10y (1986) + Baa/Aaa mensual (1919). Pista B: proxies ETF (2002/2007) + OAS por rating vivas (2023).

---

## 4. Tipos y curva

**Qué existe.** La curva de Treasuries es una feature de régimen de primer orden (inversión → recesión; nivel/pendiente separan ciclos de política). Está **casi toda en FRED, limpia y muy profunda**, y sirve para **ambas pistas**. Comprobación relevante: la curva diaria es **densa y sin agujeros a través de 2013** (250 obs/año; único festivo 2013-10-14), así que puede usarse como ancla para tapar huecos de 2013 de otros paneles.

**Qué elegimos y por qué.**
- **Curva núcleo diaria (ambas, máximo histórico):** el vector `DGS3MO`(1981), `DGS2`(1976), `DGS5`(1962), `DGS10`(1962, spine), `DGS30`(1977) + `DGS1/DGS3/DGS7`. Todo constant-maturity, FRED, vivo. `DGS10` es la espina de la curva.
- **Pendientes (señal de régimen fuerte):** `T10Y2Y` (1976) y `T10Y3M` (1982, el predictor de recesión canónico de Estrella-Mishkin); `T10YFF`/`T5YFF` (curva vs política, 1962).
- **Overnight / funding (Pista B):** `DFF` (Fed Funds effective diario, spine, 1954), `EFFR` (2000), `SOFR` (repo, 2018), `OBFR` (2016), `IORB` (2021), `DPCREDIT` (2003).
- **Tramo corto profundo (Pista A):** las letras en secundario van mucho más atrás que el CMT — `DTB3` (3M diario, **1954**, spine corto), `DTB6` (1958), y la mensual `TB3MS` (3M, **1934**, la serie de tipos más profunda de FRED). Curva mensual profunda: `GS10/GS5/GS1` (**1953**). Pendiente profunda pre-1976 = `GS10 − TB3MS`.
- **Reales / breakevens (régimen inflación vs crecimiento):** TIPS `DFII10`/`DFII5` (2003), `DFII30` (2010), breakevens `T10YIE`/`T5YIE` (2003) y el forward `T5YIFR` (5y5y, 2003).

**Ojo a huecos estructurales:** `DGS20` tiene hueco 1987–1993 y `DGS30` hueco 2002-02→2006-02 (bono 30y descontinuado) — imputar/excluir esos nodos en esos tramos.

**Fallbacks.** yfinance `^TNX`(10y)/`^FVX`(5y)/`^TYX`(30y)/`^IRX`(13-wk) redundan a FRED 1:1 con historia igual de larga (1960-1962). Hoy `^TNX` cotiza el rendimiento directo (no ×10).

**Reparto A/B.** Pista A: `TB3MS`(1934), `GS10/GS5/GS1`(1953), `DFF`/`DTB3`(1954), `FEDFUNDS`. Pista B: curva diaria completa, funding moderno, reales/breakevens. `DGS10`, `DGS2`, `T10Y2Y`, `T10Y3M` sirven a ambas.

---

## 5. Liquidez y funding

**Qué existe — y qué murió.** El estrés de funding *lidera* los episodios sistémicos (se tensa antes de que caiga el equity). El **TED spread clásico está muerto**: `TEDRATE` (LIBOR−T-bill) es limpio 1986→2022-01-21 pero **descontinuado** por el fin de LIBOR, y FRED **ya no sirve el USD LIBOR diario** (`USD3MTD156N` y familia → HTTP 400). **Un LIBOR-OIS diario con historia larga no está disponible gratis hoy.**

**Qué elegimos y por qué.**
- **Espina de funding profunda (Pista A, empalmable):** `EURODOLLAR_TBILL_SPREAD` (`DED3−DTB3`, el TED *original* pre-LIBOR sobre Eurodólares, diario **1971-2016**; picos 1974=6.85, 1987=2.83, 1998=1.53, 2008=5.76) → empalma con `TEDRATE` (1986-2022) → y con `PAPER_BILL_SPREAD` (1997→vivo). Es **una serie continua de estrés de funding 1971→hoy**. Textura pre-1971: `NBER_NY_COMMERCIAL_PAPER` (papel comercial NY, mensual **1857**, pico 24% en el Pánico de 1857) y `TBILL3M_MINUS_FEDFUNDS` (1954).
- **Sucesores vivos del TED (Pista B, núcleo):** `PAPER_BILL_SPREAD` (`DCPF3M−DTB3`, papel comercial financiero − T-bill, 1997, vivo; pico 3.73 en 2008) y `CP_FFR_SPREAD` (`CPFF`, 1997, vivo). Más `ABCP_BILL_SPREAD` (2001, señal temprana estilo agosto-2007).
- **Repo moderno (Pista B):** `SOFR` (2018) y `SOFR_EFFR_SPREAD` (proxy libre de SOFR-OIS; pico +295 bp el 2019-09-17, la crisis de repo). `TGCR_REPO` (2018), `ON_RRP_VOLUME` (glut de colateral, pico 2.55 B$ en dic-2022).

**Reparto A/B.** Pista A: espina empalmada 1857/1971→hoy. Pista B: paper-bill, CP-FFR, ABCP, SOFR-OIS, repo.

*(La validación de esta categoría — OFR FSI y su subíndice Funding, NFCI/KCFSI — está en la sección 11.)*

---

## 6. Equity y amplitud (breadth)

**Qué existe.** La espina histórica de equity es **sólida y muy profunda** (ver también sección 10, académicos). El hueco real está en la **amplitud clásica**: la línea advance-decline, el % de valores sobre su media de 200d, nuevos máximos/mínimos y McClellan **no están gratis** como serie descargable (Yahoo `^ADD`/`^ADVN`/`^S5FI` → vacío/404; FRED sin series de advance-decline). Hay que **proxiarla** o **computarla** desde constituyentes.

**Qué elegimos y por qué.**
- **Espina diaria (Pista A):** `^GSPC` (S&P500 precio, **1927**, la mejor espina diaria gratis) + `SP500_TR` (total return, 1988). El retorno de referencia limpio lo da Ken French (sección 10).
- **Índices amplios / total market:** `NYSE_COMP` (`^NYA`, **1965**), `NASDAQ_COMP` (`^IXIC`, 1971), `NASDAQ100` (FRED, 1986), `WILSHIRE5000` (`^W5000`, 1989 — Yahoo lo sirve aunque FRED lo retiró), `SP100` (`^OEX`, 1982).
- **Small/large y estilo:** `RUSSELL2000` (`^RUT`, 1987), `RUSSELL1000`/`RUSSELL3000`, `SP_MIDCAP400` (**1981**), `SP_SMALLCAP600` (1989). En profundidad, los factores **SMB** (tamaño) y **HML** (value) de Ken French desde **1926** (sección 10). ETFs de estilo `IWM`/`IWD`/`IWF` (2000).
- **Sectores GICS:** los 9 SPDR originales (XLK/XLF/XLE/XLV/XLI/XLY/XLP/XLU/XLB) desde **1998**, `XLRE` (2015), `XLC` (2018). Para profundidad histórica, los **portfolios de 5/10/49 industrias de Ken French desde 1926**.
- **Amplitud operativa (proxies verificados):** `RSP/SPY` (equal vs cap-weight, concentración, 2003 — el mejor proxy de breadth gratis), `SMB` (small/large), `SPHB/SPLV` (high-beta vs low-vol, risk-on/off, 2011), `MTUM` (momentum, 2013). La amplitud "pura" se **computa** desde los 11 SPDR o desde constituyentes point-in-time (`fja05680/sp500`, 1996+).

**Reparto A/B.** Pista A: `^GSPC`(1927), SMB/HML/industrias French(1926), MidCap400(1981), OEX(1982). Pista B: sectores SPDR, Russell/estilo ETFs, proxies de breadth.

---

## 7. FX y dólar

**Qué existe.** El dólar es el eje del risk-off global (en crisis el USD se aprecia; JPY/CHF refugio; EM/commodity FX se hunden). Está **muy bien cubierto y profundo** en FRED (H.10) y yfinance. Todas las series FX diarias tienen 249-261 obs en 2013 (taper tantrum limpio, sin punto ciego).

**Qué elegimos y por qué.**
- **Espina de FX (ambas):** `DXY` (ICE, yfinance `DX-Y.NYB`, **1971**→vivo, un solo símbolo cubre 55 años). El DXY oficial nació en 1973; el tramo 1971-73 es backfill de Yahoo (tratar con cautela).
- **Broad dollar profundo + vivo (empalme):** `DTWEXM` (broad-major, diario **1973**-2019, **descontinuada**) **+** `DTWEXAFEGS` (2006→vivo) con solape; y `DTWEXBGS` (broad total vivo). Para EM: `DTWEXO` (1995-2019) **+** `DTWEXEMEGS` (2006→vivo).
- **Dólar de baja frecuencia (Pista A, deep):** `RNUSBIS` (BIS real narrow REER, mensual **1964**) — el dólar más profundo verificado; cubre Bretton Woods y el Nixon shock. Más `NNUSBIS` (nominal, 1964).
- **Refugio vs riesgo (features de régimen, FRED diario 1971):** refugio = `DEXJPUS` (USDJPY), `DEXSZUS` (USDCHF); riesgo = `DEXUSAL` (AUDUSD), `DEXUSNZ` (NZDUSD), petro-FX `DEXCAUS`/`DEXNOUS`. Feature clave: **AUDJPY** (termómetro carry / risk-on-off).
- **Cesta EM (risk-off directo):** `DEXMXUS` (USDMXN, barómetro EM de alta beta, 1993), `DEXSFUS` (USDZAR, 1980), `DEXBZUS` (USDBRL, 1995), `DEXKOUS` (USDKRW, 1981), `DEXINUS` (USDINR, 1973), `DEXCHUS` (USDCNY, 1981). Turquía y Rusia **no están en FRED**: `USDTRY=X` (2005) y `USDRUB=X` (2003) por yfinance.
- **Euro:** `DEXUSEU` (EURUSD, FRED, **solo 1999**, nacimiento del euro). Pre-1999 no hay par diario gratis; el régimen del dólar pre-euro se lee vía broad dollar (1973) y RNUSBIS (1964).

**Reparto A/B.** Pista A: DXY(1971), RNUSBIS(1964), pares mayores diarios 1971 (JPY/CHF/AUD). Pista B: cesta EM, euro, índices broad *GS.

---

## 8. Commodities

**Qué existe — y dos huecos de licencia.** Los commodities son procíclicos (el índice amplio se hunde en recesión) y el oro es refugio; el **ratio oro/cobre** es un termómetro clásico de growth-scare. Dos hallazgos: (1) **el precio diario del oro desapareció de FRED** (`GOLDPMGBD228NLBM`, LBMA fixing, retirada por licencia ICE, igual que las OAS) → el oro diario **hay que sacarlo de yfinance**; (2) los **índices amplios (GSCI, Bloomberg Commodity) no están en FRED** pero Yahoo los sirve como tickers de índice con historia profunda. Caso de test obligado: **el WTI cotizó −36.98 el 2020-04-20** (los log-returns rompen; usar retornos simples o el ETF/índice).

**Qué elegimos y por qué.**
- **Espina diaria amplia (ambas, cubre crisis pre-2000):** `^SPGSCI` (S&P GSCI, energy-heavy, **1984**) y `^BCOM` (Bloomberg Commodity, diversificado, **1991**). Sanity de crisis verificado (GSCI: 349 en dic-2008, 255 en el suelo COVID). Fallbacks ETF fiables por si Yahoo retira el ticker: `GSG`, `DBC`, `DJP` (2006).
- **Espina profunda mensual (Pista A):** `PPIACO` (PPI All Commodities, **1913**, cubre la Gran Depresión), `WTISPLC` (WTI spot mensual, **1946**), `WPU102301` (PPI metales, 1957).
- **Núcleo diario (Pista B, 2000+):** oro `GC=F` (2000) y `GLD` (2004); petróleo `DCOILWTICO` (FRED, 1986) + `DCOILBRENTEU` (Brent, 1987); cobre `HG=F` (2000) + `PCOPPUSDM` (FRED mensual, 1992).
- **Feature derivada estrella:** `GOLD_COPPER_RATIO` (`GC=F/HG=F`, 2000) — el indicador de risk-off más limpio de la categoría (633 en dic-2008, 707 en el suelo COVID).
- **Enrichers:** plata `SI=F` (2000), platino `PL=F` (1997), paladio `PA=F` (1998), gas `NG=F`/`DHHNGSP` (1997), mineras `GDX` (2006), índices IMF de FRED (`PNRGINDEXM`/`PMETAINDEXM`, 1992). Validación de estrés del refugio: `GVZCLS` (Gold VIX, 2008).

**Reparto A/B.** Pista A: `PPIACO`(1913), `WTISPLC`(1946), índices amplios diarios (1984/1991). Pista B: oro/petróleo/cobre diario 2000+, ratio oro/cobre, enrichers.

---

## 9. Macro

**Qué existe — y qué no está gratis.** El macro es contexto de régimen de primer orden (marca el ciclo real: expansión/desaceleración/recesión/recuperación). Casi todo vive en FRED, largo y vivo. Dos avisos: (1) **el PMI/ISM no está en FRED ni gratis limpio** (ISM y S&P Global son propietarios) → se proxya con las **encuestas manufactureras de los Fed regionales**; (2) para backtest sin look-ahead importa la distinción **`SAHMREALTIME`** (datos tal como se publicaron, sin revisión → la correcta) vs `SAHMCURRENT` (revisada, con look-ahead leve).

**Qué elegimos y por qué.**
- **Espina macro profunda (Pista A):** `INDPRO` (producción industrial, **1919**, el termómetro macro núcleo; YoY −15% en 2009, −17% en 2020), `UNRATE` (paro, 1948), `PAYEMS`/`MANEMP` (empleo, **1939**), `CPIAUCSL`/`CPILFESL` (inflación, 1947/1957), `CFNAI`/`CFNAIMA3` (actividad, 1967).
- **Señales de recesión de una cifra (features directas):** `SAHMREALTIME ≥ 0.50` (flag de recesión en tiempo real, sin look-ahead), `CFNAIMA3 < −0.70` (umbral oficial Chicago Fed).
- **Alta frecuencia dentro del macro (Pista B, puentes semanales):** `ICSA`/`IC4WSA`/`CCSA` (claims, 1967, semanal; pico 6.14M en abril-2020), `WEI` (nowcast semanal, 2008), y `NFCI` (condiciones financieras, 1971 — validación).
- **Régimen inflacionario (clave para correlación acción-bono):** `CPI/CPI core` YoY, `PCEPILFE` (la medida de la Fed, 1959), `MICH`/`T10YIE` (expectativas). Distingue risk-off deflacionario (2008, 2020) de inflacionario (2022).
- **PMI proxy (libre):** `PMI_PROXY_PHILLY` (Philadelphia Fed, difusión manufacturera, **1968**, el de más historia) + Empire (2001) + Dallas (2004).
- **Cíclicos de contexto:** `HOUST`/`PERMIT` (vivienda, líderes), `RSAFS` (ventas minoristas), `NEWORDER` (capex), `TOTALSA` (vehículos), `UMCSENT` (sentimiento), `GEPUCURRENT` (incertidumbre política), `GDPC1` (PIB trimestral).

**Caveat de vintages.** FRED sirve datos **revisados**; para point-in-time estricto haría falta ALFRED. `SAHMREALTIME`, `WEI` y las encuestas de difusión son lo más "as-of". Documentado como limitación. Alinear frecuencias mixtas a un calendario común con **forward-fill causal** y respetar el lag de publicación.

**Reparto A/B.** Pista A: `INDPRO`(1919), `PAYEMS`(1939), `UNRATE`(1948), `CPI`(1947), `CFNAI`(1967). Pista B: claims semanales, WEI, PMI-proxy, inflación de mercado, cíclicos.

---

## 10. Históricos profundos académicos (la espina de Pista A)

**Qué existe.** Cuatro fuentes académicas de coste cero y máximo histórico que son **la espina de la Pista A** (maximizar nº de crisis) y, en un caso, ground truth de crisis.

**Qué elegimos y por qué.**
- **Ken French / Dartmouth (la joya diaria):** `F-F_Research_Data_Factors_daily` da el exceso de retorno del mercado US (**Mkt-RF**) **diario desde 1926-07-01** (+ SMB, HML, RF). Compuesto `(1+Mkt-RF+RF)` reconstruye un índice de retorno total diario **más profundo y limpio que `^GSPC`** (que arranca en 1927-12), sin huecos de splits. También `F-F_Momentum_Factor_daily` (UMD, 1926; los *momentum crashes* marcan giros de régimen) y `5_Factors` (RMW/CMA, 1963).
- **Shiller (S&P500 + CAPE mensual, 1871):** valoración secular. Fuente canónica (única con TR CAPE), pero **la copia de Yale va con retraso** (acaba ~2023-09). Para frescura se usa el **mirror GitHub `datasets/s-and-p-500`** (limpio, actualizado a 2026-06, con PE10=CAPE) como primario programático.
- **Goyal-Welch (predictores del equity premium, mensual 1871):** de aquí salen gratis y desde el s.XIX el **term spread** (`lty−tbl`), el **default spread** (`BAA−AAA`), el dividend/earnings yield y `svar` (varianza realizada mensual). Vive en un Google Sheet (el host institucional antiguo está muerto).
- **Jordà-Schularick-Taylor Macrohistory R6 (anual, 1870, 18 países):** no es feature diaria, pero su columna **`crisisJST`** es la lista experta más citada de crisis bancarias sistémicas → **ground truth de crisis** (sección 11). Trae además retornos totales equity/bonos/vivienda, crédito, tipos y deuda/PIB.

**Reparto A/B.** Todo es **Pista A** (espina profunda). `crisisJST` es validación.

---

## 11. Validación (ground truth laxo): índices de estrés + fechas de crisis

Estas series son **la respuesta**, no features (usarlas de input = fuga). Dos familias: índices de estrés ya computados y fechas de crisis/recesión.

**Índices de estrés (qué elegimos).**
- **OFR Financial Stress Index** (la joya): **no está en FRED**, se baja del CSV público de OFR. **Diario 2000+**, vivo, y viene **descompuesto** en subíndices por categoría (**Credit, Funding, Volatility**, Equity valuation, Safe assets) y región (US, Other advanced, EM). El más granular y de mayor frecuencia gratis. Pico global 29.32 el 2008-10-10.
- **NFCI / ANFCI** (Chicago Fed): la mayor historia, **semanal desde 1971**, con subíndices Risk/Credit/Leverage. `ANFCI` aísla el estrés financiero del ciclo. Cubre casi toda la Pista B y solapa con crisis de Pista A.
- **STLFSI4** (St. Louis, semanal 1993) y **KCFSI** (Kansas City, mensual 1990) como segundo/tercer voto. **CFSI** (Cleveland, diario 1991-2016, **descontinuado**) solo para rellenar el hueco diario 1991-2000 (pre-OFR).

**Fechas de crisis / recesión (qué elegimos).**
- **NBER**: `USRECD` (diario, **1854**) y `USREC` (mensual). **12 recesiones desde 1948.** Ojo metodológico: el `1` empieza el periodo **siguiente** al pico NBER (2008-01 = pico dic-2007) → retroceder un periodo para marcar inicio.
- **Probabilidades de recesión:** `RECPROUSM156N` (Chauvet-Piger, Markov-switching, 1967 — es literalmente un régimen-switching benchmark), `SAHMREALTIME` (1959), `JHDUSRGDPBR` (Hamilton, 1967).
- **`CRISIS_WINDOWS_SP500` (aportación propia):** las recesiones NBER no bastan — se pierden las crisis financieras **sin** recesión (1987, 1998 LTCM, 2010 flash crash, 2011 downgrade, 2015-16, 2018, 2022, 2023 SVB). Derivamos del cierre diario de `^GSPC` una lista de **~20 ventanas peak→trough desde 1950** (y hasta 103 con zigzag), cubriendo de sobra la promesa de "10+ crisis". Matiz honesto: **SVB (mar-2023) cae sub-umbral** en el índice (−7.8%); es crisis de crédito/banca, no drawdown del S&P500 → capturarla requiere estrés bancario (OFR-FSI Funding, spreads), no el nivel del índice.
- **`crisisJST`** (JST R6, 1870, 18 países): ground truth internacional de crisis bancarias sistémicas.

**Reglas de uso.** Positivo = estrés en todos los FSI. Reindexar semanal/mensual a diario con **forward-fill causal** (nunca interpolar hacia atrás). Doble ground truth (NBER ∪ CRISIS_WINDOWS) = recall alto; intersección = precisión.

---

## 12. Tabla resumen de todas las series elegidas

Series elegidas por categoría, con rol y pista. Fuente: `FRED` (auth FRED_API_KEY), `yf` (yfinance, sin auth), `acad` (académico, sin auth), `OFR` (CSV, sin auth), `GH` (GitHub datahub, sin auth). Inicio = fecha **verificada** observada. `M`=mensual, `W`=semanal, `Q`=trimestral, `D`=diaria (por defecto). Estado especial entre paréntesis.

### Volatilidad
| Serie | Fuente / id | Inicio | Pista | Rol |
|---|---|---|---|---|
| REALIZED_VOL_SP500 | yf `^GSPC` (calc) | 1928-01 | A | spine |
| VXO | FRED `VXOCLS` | 1986-01 (fin 2021) | A | core |
| VIX | FRED `VIXCLS` | 1990-01 | ambas | core |
| MOVE | yf `^MOVE` | 2002-11 | B | core |
| SKEW | yf `^SKEW` | 1990-01 | ambas | enricher |
| VIX3M | yf `^VIX3M` | 2006-07 | B | enricher |
| VIX6M | yf `^VIX6M` | 2008-01 | B | enricher |
| VIX9D | yf `^VIX9D` | 2011-01 | B | enricher |
| VVIX | yf `^VVIX` | 2007-01 | B | enricher |
| VXN | FRED `VXNCLS` | 2001-02 | B | enricher |
| VXD | FRED `VXDCLS` | 1997-10 | B | enricher |
| RVX | FRED `RVXCLS` | 2004-01 | B | enricher |
| OVX | FRED `OVXCLS` | 2007-05 | B | enricher |
| GVZ | FRED `GVZCLS` | 2008-06 | B | enricher/valid |
| VXEEM | FRED `VXEEMCLS` | 2011-03 | B | enricher |

### Crédito
| Serie | Fuente / id | Inicio | Pista | Rol |
|---|---|---|---|---|
| MOODYS_BAA_AAA_SPREAD | FRED `DBAA−DAAA` | 1986-01 | ambas | spine |
| BAA10Y | FRED `BAA10Y` | 1986-01 | ambas | spine |
| MOODYS_BAA / _AAA | FRED `BAA`/`AAA` | 1919-01 (M) | A | spine |
| DAAA | FRED `DAAA` | 1983-01 | B | core |
| BAAFFM | FRED `BAAFFM` | 1954-07 (M) | A | enricher |
| HYG | yf `HYG` | 2007-04 | B | core |
| LQD | yf `LQD` | 2002-07 | B | core |
| IEF | yf `IEF` | 2002-07 | B | core |
| HYG_IEF_SPREAD | yf `HYG,IEF` (calc) | 2007-04 | B | core |
| EMB | yf `EMB` | 2007-12 | B | enricher |
| BKLN | yf `BKLN` | 2011-03 | B | enricher |
| AGG | yf `AGG` | 2003-09 | B | fallback |
| HY_OAS | FRED `BAMLH0A0HYM2` | 2023-07 (cap 3a) | B | core |
| IG_OAS | FRED `BAMLC0A0CM` | 2023-07 (cap 3a) | B | core |
| CCC_OAS | FRED `BAMLH0A3HYC` | 2023-07 (cap 3a) | B | core |
| BB_OAS / B_OAS | FRED `BAMLH0A1HYBB`/`…A2HYB` | 2023-07 (cap 3a) | B | enricher |
| BBB_OAS | FRED `BAMLC0A4CBBB` | 2023-07 (cap 3a) | B | enricher |
| EM_CORP_OAS | FRED `BAMLEMCBPIOAS` | 2023-07 (cap 3a) | B | enricher |
| EURO_HY_OAS | FRED `BAMLHE00EHYIOAS` | 2023-07 (cap 3a) | B | enricher |

### Tipos y curva
| Serie | Fuente / id | Inicio | Pista | Rol |
|---|---|---|---|---|
| DGS10 | FRED `DGS10` | 1962-01 | ambas | spine |
| DGS5 | FRED `DGS5` | 1962-01 | ambas | core |
| DGS2 | FRED `DGS2` | 1976-06 | ambas | core |
| DGS30 | FRED `DGS30` | 1977-02 (hueco 02-06) | ambas | core |
| DGS3MO | FRED `DGS3MO` | 1981-09 | ambas | core |
| DGS1 / DGS3 / DGS7 | FRED | 1962/1962/1969 | B | enricher |
| DGS6MO / DGS1MO | FRED | 1981/2001 | B | enricher |
| T10Y2Y | FRED `T10Y2Y` | 1976-06 | ambas | core |
| T10Y3M | FRED `T10Y3M` | 1982-01 | ambas | core |
| T10YFF / T5YFF | FRED | 1962-01 | ambas/B | core/enricher |
| DFF | FRED `DFF` | 1954-07 | ambas | spine |
| FEDFUNDS | FRED `FEDFUNDS` | 1954-07 (M) | A | enricher |
| DTB3 | FRED `DTB3` | 1954-01 | ambas | spine |
| DTB6 / DTB1YR | FRED | 1958/1959 | B | enricher |
| TB3MS | FRED `TB3MS` | 1934-01 (M) | A | spine |
| GS10 / GS5 / GS1 | FRED | 1953-04 (M) | A | spine/enricher |
| EFFR | FRED `EFFR` | 2000-07 | B | enricher |
| SOFR | FRED `SOFR` | 2018-04 | B | core |
| OBFR / IORB / DPCREDIT | FRED | 2016/2021/2003 | B | fallback |
| DFII10 / DFII5 | FRED | 2003-01 | B | core/enricher |
| DFII30 | FRED `DFII30` | 2010-02 | B | fallback |
| T10YIE / T5YIE | FRED | 2003-01 | B | core/enricher |
| T5YIFR | FRED `T5YIFR` | 2003-01 | B | enricher |
| ^TNX/^FVX/^TYX/^IRX | yf | 1960-1962 | ambas | fallback |

### Liquidez y funding
| Serie | Fuente / id | Inicio | Pista | Rol |
|---|---|---|---|---|
| NBER_NY_COMMERCIAL_PAPER | FRED `M13002US35620M156NNBR` | 1857-01 (M) | A | spine |
| EURODOLLAR_TBILL_SPREAD | FRED `DED3−DTB3` | 1971-01 (fin 2016) | ambas | spine |
| TBILL3M_MINUS_FEDFUNDS | FRED `TB3SMFFM` | 1954-07 (M) | A | enricher |
| TED_SPREAD | FRED `TEDRATE` | 1986-01 (fin 2022) | B | core |
| PAPER_BILL_SPREAD | FRED `DCPF3M−DTB3` | 1997-01 | B | core |
| CP_FFR_SPREAD | FRED `CPFF` | 1997-01 | B | core |
| ABCP_BILL_SPREAD | FRED `RIFSPPAAAD90NB−DTB3` | 2001-01 | B | enricher |
| SOFR_EFFR_SPREAD | FRED `SOFR−EFFR` | 2018-04 | B | core |
| TGCR_REPO | FRED `TGCRRATE` | 2018-04 | B | enricher |
| ON_RRP_VOLUME | FRED `RRPONTSYD` | 2003-02 | B | enricher |

### Equity y amplitud
| Serie | Fuente / id | Inicio | Pista | Rol |
|---|---|---|---|---|
| SP500 | yf `^GSPC` | 1927-12 | A | spine |
| SP500_TR | yf `^SP500TR` | 1988-01 | A | core |
| NYSE_COMP | yf `^NYA` | 1965-12 | A | enricher |
| SP100 | yf `^OEX` | 1982-08 | A | enricher |
| SP_MIDCAP400 | yf `^SP400` | 1981-01 | A | enricher |
| NASDAQ_COMP | yf `^IXIC` | 1971-02 | A | core |
| NASDAQ100 | FRED `NASDAQ100` | 1986-01 | A | core |
| WILSHIRE5000 | yf `^W5000` | 1989-01 | B | enricher |
| RUSSELL2000 | yf `^RUT` | 1987-09 | A | core |
| RUSSELL1000 / 3000 | yf `^RUI`/`^RUA` | 1992/1987 | B | enricher |
| SP_SMALLCAP600 | yf `^SP600` | 1989-01 | B | enricher |
| IWM / IWD / IWF | yf | 2000-05 | B | enricher |
| SPDR_SECTORS_9 | yf `XLK…XLB` | 1998-12 | B | core |
| XLRE / XLC | yf | 2015/2018 | B | enricher |
| SP500_EW (RSP) | yf `RSP` | 2003-05 | B | core |
| SPHB / SPLV | yf | 2011-05 | B | enricher |
| MTUM | yf `MTUM` | 2013-04 | B | enricher |

### FX y dólar
| Serie | Fuente / id | Inicio | Pista | Rol |
|---|---|---|---|---|
| DXY | yf `DX-Y.NYB` | 1971-01 | ambas | spine |
| DTWEXM | FRED `DTWEXM` | 1973-01 (fin 2019) | ambas | spine |
| DTWEXAFEGS | FRED `DTWEXAFEGS` | 2006-01 | ambas | core |
| DTWEXBGS | FRED `DTWEXBGS` | 2006-01 | B | core |
| DTWEXEMEGS | FRED `DTWEXEMEGS` | 2006-01 | B | core |
| DTWEXO | FRED `DTWEXO` | 1995-01 (fin 2019) | B | enricher |
| RNUSBIS | FRED `RNUSBIS` | 1964-01 (M) | A | spine |
| NNUSBIS | FRED `NNUSBIS` | 1964-01 (M) | A | enricher |
| DEXJPUS (USDJPY) | FRED `DEXJPUS` | 1971-01 | ambas | core |
| DEXSZUS (USDCHF) | FRED `DEXSZUS` | 1971-01 | ambas | core |
| DEXUSAL (AUDUSD) | FRED `DEXUSAL` | 1971-01 | ambas | core |
| DEXUSNZ (NZDUSD) | FRED `DEXUSNZ` | 1971-01 | B | enricher |
| DEXUSUK / DEXCAUS / DEXNOUS / DEXSDUS | FRED | 1971-01 | B | enricher |
| DEXUSEU (EURUSD) | FRED `DEXUSEU` | 1999-01 | B | core |
| DEXMXUS (USDMXN) | FRED `DEXMXUS` | 1993-11 | B | core |
| DEXSFUS (USDZAR) | FRED `DEXSFUS` | 1980-01 | B | core |
| DEXBZUS / DEXKOUS / DEXINUS / DEXCHUS | FRED | 1995/1981/1973/1981 | B | enricher |
| USDTRY / USDRUB | yf `USDTRY=X`/`USDRUB=X` | 2005/2003 | B | enricher/fallback |

### Commodities
| Serie | Fuente / id | Inicio | Pista | Rol |
|---|---|---|---|---|
| SPGSCI | yf `^SPGSCI` | 1984-01 | ambas | spine |
| BCOM | yf `^BCOM` | 1991-01 | ambas | spine |
| PPI_ALL_COMMODITIES | FRED `PPIACO` | 1913-01 (M) | A | spine |
| WTI_SPOT_MONTHLY | FRED `WTISPLC` | 1946-01 (M) | A | spine |
| PPI_METALS | FRED `WPU102301` | 1957-01 (M) | A | enricher |
| WTI_DAILY | FRED `DCOILWTICO` | 1986-01 | B | core |
| BRENT_DAILY | FRED `DCOILBRENTEU` | 1987-05 | B | core |
| GOLD_FUT | yf `GC=F` | 2000-08 | B | core |
| GOLD_ETF_GLD | yf `GLD` | 2004-11 | B | core |
| COPPER_FUT | yf `HG=F` | 2000-08 | B | core |
| GOLD_COPPER_RATIO | yf `GC=F/HG=F` (calc) | 2000-08 | B | core |
| COPPER_GLOBAL_MONTHLY | FRED `PCOPPUSDM` | 1992-01 (M) | B | enricher |
| SILVER_FUT / PLATINUM / PALLADIUM | yf `SI=F`/`PL=F`/`PA=F` | 2000/1997/1998 | B | enricher |
| NATGAS_FUT / HENRYHUB | yf `NG=F` / FRED `DHHNGSP` | 2000/1997 | B | enricher |
| GOLD_MINERS_GDX | yf `GDX` | 2006-05 | B | enricher |
| GSG / DBC / DJP | yf | 2006 | B | fallback |
| IMF energy/metals/all | FRED `PNRGINDEXM`/`PMETAINDEXM`/`PALLFNFINDEXM` | 1992-01 (M) | B | enricher |

### Macro
| Serie | Fuente / id | Inicio | Pista | Rol |
|---|---|---|---|---|
| INDPRO | FRED `INDPRO` | 1919-01 (M) | ambas | spine |
| PAYEMS / MANEMP | FRED | 1939-01 (M) | ambas/A | core/enricher |
| UNRATE | FRED `UNRATE` | 1948-01 (M) | ambas | core |
| U6RATE | FRED `U6RATE` | 1994-01 (M) | B | enricher |
| CPIAUCSL / CPILFESL | FRED | 1947/1957 (M) | ambas | core |
| PCEPI / PCEPILFE | FRED | 1959-01 (M) | B | enricher |
| CFNAI / CFNAIMA3 | FRED | 1967 (M) | ambas | core |
| SAHMREALTIME | FRED `SAHMREALTIME` | 1959-12 (M) | ambas | core |
| ICSA / IC4WSA / CCSA | FRED | 1967 (W) | ambas/B | core/enricher |
| WEI | FRED `WEI` | 2008-01 (W) | B | enricher |
| PMI_PROXY_PHILLY | FRED `GACDFSA066MSFRBPHI` | 1968-05 (M) | ambas | core |
| PMI_PROXY_EMPIRE / _DALLAS | FRED | 2001/2004 (M) | B | enricher |
| MICH / UMCSENT / GEPUCURRENT | FRED | 1978/1952/1997 (M) | B | enricher |
| HOUST / PERMIT / RSAFS / NEWORDER / TOTALSA | FRED | 1959-1992 (M) | B | enricher |
| GDPC1 | FRED `GDPC1` | 1947-01 (Q) | A | enricher |

### Históricos profundos académicos (espina Pista A)
| Serie | Fuente / id | Inicio | Pista | Rol |
|---|---|---|---|---|
| FF_FACTORS_3_DAILY (Mkt-RF/SMB/HML/RF) | acad Ken French | 1926-07 | A | spine |
| FF_MOM_DAILY (UMD) | acad Ken French | 1926-11 | A | enricher |
| FF_FACTORS_5_DAILY (RMW/CMA) | acad Ken French | 1963-07 | A | enricher |
| FF_5_INDUSTRY | acad Ken French | 1926-07 | A | enricher |
| SHILLER_SP500_CAPE | GH `datasets/s-and-p-500` | 1871-01 (M) | A | spine |
| GW_PREDICTORS_MONTHLY | acad Goyal-Welch | 1871-01 (M) | A | core |
| JST_MACROHISTORY_R6 | acad macrohistory.net | 1870 (anual) | ambas | enricher |

### Validación (ground truth, NO feature)
| Serie | Fuente / id | Inicio | Pista | Rol |
|---|---|---|---|---|
| OFR_FSI (+ Credit/Funding/Volatility/US) | OFR CSV | 2000-01 | validación | validation |
| NFCI / ANFCI (+ subíndices) | FRED | 1971-01 (W) | validación | validation |
| STLFSI4 | FRED `STLFSI4` | 1993-12 (W) | validación | validation |
| KCFSI | FRED `KCFSI` | 1990-02 (M) | validación | validation |
| CFSI | FRED `CFSI` | 1991-09 (fin 2016) | validación | fallback |
| USRECD / USREC (NBER) | FRED | 1854-12 | validación | validation |
| RECPROUSM156N | FRED `RECPROUSM156N` | 1967-06 (M) | validación | validation |
| JHDUSRGDPBR (Hamilton) | FRED | 1967-10 (Q) | validación | validation |
| CRISIS_WINDOWS_SP500 | yf `^GSPC` (derivada) | 1927-12 | validación | validation |
| crisisJST | acad JST R6 | 1870 (anual) | validación | validation |

**Reparto por pista (aproximado, series elegidas):** Pista A ≈ 35-40 series (espina profunda), Pista B ≈ 90-100 series (panel rico), ~15 sirven a ambas, ~14 de validación. La Pista A prioriza profundidad (muchas series arrancan en 1913-1971); la Pista B prioriza granularidad cross-asset (banco 2003+).

---

## 13. Descartados y por qué

### 13.1. Series concretas descartadas o degradadas

| Serie / dato | Estado | Por qué se descarta o degrada |
|---|---|---|
| **PMI / ISM Manufacturing** (`NAPM`, `NAPMPI`…) | No en FRED (HTTP 400) | ISM retiró sus series por licencia; S&P Global PMI es propietario. **No hay PMI US gratis con histórico.** Se sustituye por Fed regionales (Philly/Empire/Dallas). |
| **OAS por rating, historia larga** (`BAML…` 1996-2023) | Capado a 3 años (abril-2026) | Restricción de licencia ICE en FRED. Irrecuperable gratis. Se usa Baa−Aaa (Moody's) como spine profundo y HYG/IEF como proxy HY de crisis. |
| **Oro diario en FRED** (`GOLDPMGBD228NLBM`, LBMA fixing) | Retirada (HTTP 400) | Misma licencia ICE/LBMA que rompió las OAS. Sin fallback libre profundo pre-2000. El oro diario depende de yfinance (`GC=F` 2000, `GLD` 2004). |
| **USD LIBOR diario** (`USD3MTD156N`…) | Retirado (HTTP 400) | Fin de LIBOR + licencia ICE. Un **LIBOR-OIS diario largo no existe gratis**. Sustituido por paper-bill / CP-FFR / SOFR-OIS. |
| **TED spread** (`TEDRATE`) | Descontinuado 2022-01 | Fin de LIBOR. Se conserva como historia 1986-2022 y se empalma con paper-bill (1997→vivo). |
| **Amplitud clásica** (advance-decline, % sobre MA200, McClellan) | No disponible gratis | Yahoo `^ADD`/`^ADVN`/`^S5FI` → vacío/404; FRED no la tiene. Se proxya (RSP/SPY, SMB, SPHB/SPLV) o se **computa** desde constituyentes. |
| **Treasury liquidity / bid-ask index** (Bloomberg) | Propietario | Sin fuente libre viva. Proxies: OFR-FSI (Safe assets), MOVE. La *noise measure* de Hu-Pan-Wang es académica y estática. |
| **Conference Board LEI** (`USSLIND`) | Descontinuada en FRED (2020-02) | El Conference Board la retiró; congelada, no viva. |
| **OECD CLI US** (`USALOLITONOSTSAM`) | Rancia (última 2024-01) | Retraso de +2 años → no viable como serie viva; solo histórico. |
| **Wilshire 5000 en FRED** (`WILL5000IND`) | Retirada ("series does not exist") | Se usa `^W5000` de Yahoo (1989+) en su lugar. |
| **FRED `SP500` / `DJIA`** | Solo 10 años (licencia S&P) | Inservibles para historia; se usa Yahoo `^GSPC` (1927) y factores French. |
| **Dow Jones diario pre-1992** | No gratis-verificado | yfinance `^DJI` arranca en 1992; Stooq (que llega a 1896) está bloqueado. Poca ganancia sobre `^GSPC` (1927). |
| **EURUSD pre-1999** | No hay par diario gratis | El euro nació en 1999 (`DEXUSEU`); pre-euro se lee vía broad dollar (1973) y RNUSBIS (1964). |
| **`DX=F`** (futuro DXY) | 404 en Yahoo | Se usa el cash `DX-Y.NYB`. |
| **`JJC`** (Copper ETN) | Delistado (fin 2023-07) | No vivo; se usa `HG=F`/`CPER`. |
| **EVZ / VXTYN** (vol FX-euro / vol 10y) | Descontinuadas (2025 / 2020) | Solo como fallback; MOVE cubre la vol de bonos. |
| **STLFSI / STLFSI2 / STLFSI3** | Descontinuadas | Cambia la cesta de componentes; usar solo **STLFSI4** (recalculado hacia atrás a 1993). |

### 13.2. Fuentes enteras descartadas (probadas y bloqueadas)

| Fuente | Estado | Por qué |
|---|---|---|
| **Stooq** (CSV per-ticker y bulk) | Bloqueado | Challenge JavaScript proof-of-work (SHA-256); resuelto el PoW pero el CSV devuelve `Access denied`; el bulk (`d_us_txt.zip`) da HTTP 401. **No usable sin navegador.** Confirmado por todos los agentes. Sustituto: yfinance/FRED. |
| **Nasdaq Data Link** (ex-Quandl) | Tapiado | Anónimo → Akamai 403 (requiere API key). El clásico free `WIKI/WIKIP` está **congelado en 2018-03**. La DB legacy `ML/*` (OAS BofA) da HTTP 403. |
| **Tiingo** | Requiere token | Anónimo → 401. Free tier útil (30+ años EOD) pero exige registro; no verificable end-to-end aquí. |
| **Kaggle** | No scriptable | Páginas SPA sin metadatos server-side; descarga requiere `kaggle.json`. Los datasets famosos (`borismarjanovic`, `^GSPC 1927-2025`) son **re-exports congelados** de `^GSPC`/`SPY`/Stooq → la copia viva (yfinance) es mejor salvo que se quiera un snapshot reproducible fijo. |
| **fredgraph.csv** (host web FRED) | Read-timeout | El host **API** (`api.stlouisfed.org`) sí funciona; y aunque respondiera, serviría la misma ventana capada de las OAS. |

---

## 14. Síntesis y deuda técnica honesta

**Lo que tenemos, sólido:** una **espina de Pista A** muy profunda y verificada (equity/vol/tipos/crédito/macro/commodities/FX con memoria de 1913-1971, y hasta 1857/1870/1871 en funding, JST y Shiller), y un **panel de Pista B** rico y vivo (vol implícita cross-asset, curva completa, proxies de crédito, funding moderno, cesta FX/EM, commodities) con validación de estrés de alta calidad (OFR-FSI diario, NFCI semanal 1971, NBER 1854).

**Las limitaciones que hay que documentar en el TFM:**
1. **OAS por rating sin histórico** (cap ICE de 3 años). Si el detector necesita CCC/HY OAS en crisis pre-2023, hay que **archivar un snapshot pre-abril-2026** y empalmarlo — foto estática, no viva, no verificada. Alternativa preferida: features de crédito basadas en Moody's (Baa−Aaa) y HYG/IEF.
2. **Oro diario y LIBOR-OIS diario: irrecuperables gratis** (licencia ICE). Oro → yfinance; LIBOR-OIS → sustituido por paper-bill/CP-FFR/SOFR-OIS.
3. **PMI/ISM no gratis** → proxy Fed regionales (documentar que no es idéntico).
4. **Amplitud "pura" y liquidez de Treasuries: no hay serie libre** → computar breadth desde constituyentes; usar OFR-FSI/MOVE como proxy de liquidez.
5. **Vintages/revisiones macro:** FRED sirve datos revisados; para point-in-time estricto haría falta ALFRED. `SAHMREALTIME`/`WEI`/encuestas de difusión son lo más "as-of".
6. **Huecos estructurales conocidos:** `DGS20` (1987-93), `DGS30` (2002-06), series broad *DTWEX* que paran en 2019 (empalmar con *GS). Se respeta la política de **no imputar** (huecos = `NaN`).
7. **Stooq y Kaggle no son fuentes vivas** en este entorno; toda descarga viva se apoya en FRED + yfinance + académicos + OFR + GitHub. Conviene **cachear** las descargas de yfinance por si Yahoo capa un ticker.

Todo lo listado como *elegido* está **verificado con descarga real el 2026-07-18**. Lo no verificado (snapshots OAS, Stooq, LIBOR diario, PMI ISM) queda fuera del conjunto elegido y documentado aquí como descartado o como deuda técnica.
