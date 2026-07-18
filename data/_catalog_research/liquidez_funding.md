# Categoría: Liquidez y funding — estado del arte de datos (verificado)

Investigador: agente **Liquidez y funding**. Foco encargado: TED spread, SOFR-OIS, LIBOR-OIS
histórico, financial conditions, repo, commercial paper spread y Treasury bid-ask/liquidity index.
El estrés de *funding* (interbancario, repo, papel comercial) es la señal que **lidera** los
episodios sistémicos: se tensa antes de que caiga el equity y con memoria muy corta (spikes agudos),
lo que la hace complementaria a crédito/volatilidad para separar "corrección" de "crisis de liquidez".

Todas las verificaciones se hicieron **de verdad** el 2026-07-18: FRED API con `FRED_API_KEY` del
`.env` (sin imprimir la clave), descarga real de observaciones y cálculo de spreads, y descarga real
del CSV de la **OFR**. Reporto la fecha de inicio y los picos de crisis **observados**, no los de
marketing.

---

## Hallazgos centrales (honestos)

**1. El TED spread clásico está MUERTO y no tiene sucesor gratis directo.**
`TEDRATE` (LIBOR 3M − T-bill 3M) es completo y limpio **1986-01-02 → 2022-01-21**, pico verificado
**4.58 el 2008-10-10** (Lehman), pero está **DESCONTINUADO** por el fin de LIBOR. FRED **ya no sirve
las series diarias de USD LIBOR** (`USD3MTD156N`, `USD1MTD156N`, `USDONTD156N`… todas devuelven
**HTTP 400**, retiradas por licencia ICE). Solo sobrevive `LIOR3M` (3M LIBOR **trimestral**,
1970→2016) y `LIOR3MUKM` (mensual, →2017): inservibles para un LIBOR-OIS diario. **Conclusión: un
LIBOR-OIS diario con historia larga NO está disponible gratis hoy.**

**2. El hueco de TED se tapa bien con dos sustitutos verificados y VIVOS:**
   - **Paper-bill spread** = `DCPF3M − DTB3` (papel comercial financiero AA 90d − T-bill 3M), diario
     **1997+, vivo**. Pico **3.73 el 2008-10-15**, **2.15 en 2007-08-20** (congelación ABCP), **2.57
     en 2020-03-25**. Es el heredero natural del TED (mismo espíritu: coste de fondeo privado − libre
     de riesgo) y sigue actualizándose.
   - **CP−FFR** = `CPFF` (papel comercial 3M − Fed Funds), diario **1997+, vivo** ya calculado por
     FRED. Pico **2.91 el 2008-10-15**, mínimo −1.55 en 1998. Equivalente "OIS-like" (FFR ≈ OIS).

**3. La historia PROFUNDA de funding se reconstruye por tramos (spine, Pista A):**
   - **Eurodollar−T-bill** = `DED3 − DTB3` (Eurodólar 3M London − T-bill 3M). Es *el TED original*
     (pre-LIBOR el TED se medía sobre Eurodólares). Diario **1971→2016**. Picos verificados: **1974
     = 6.85** (Franklin National), **1982 = 4.43**, **1987 = 2.83** (Black Monday), **1998 = 1.53**
     (LTCM), **2008 = 5.76**. Solapa con `TEDRATE` en 1986-2016 → **empalmable** para un spine de
     funding 1971→2022.
   - **NBER NY Commercial Paper** `M13002US35620M156NNBR`: tipo de papel comercial de Nueva York,
     **mensual 1857-01-01 → 1971-12-01**. Pico **24.0% en 1857-10** (Pánico de 1857) y captura
     1873/1907/1929-33. Nivel (no spread), pero da textura de coste de fondeo con **+160 años**.
   - **`TB3SMFFM`** (T-bill 3M − Fed Funds, mensual **1954+, vivo**): proxy de escasez de bills /
     flight-to-quality; se vuelve muy negativo en pánicos (bills caros vs fondeo).

**4. SOFR / repo cubren el estrés de funding MODERNO (Pista B, 2018+, vivos).**
`SOFR` (tasa repo garantizada) y su spread contra Fed Funds (`SOFR − EFFR`, proxy de **SOFR-OIS**)
capturan el **repo spike de septiembre 2019** con nitidez: SOFR **saltó a 5.25% el 2019-09-17** con
EFFR ~2.30, y el spread **SOFR−EFFR = +295 bp ese día** (idéntico en `TGCR−EFFR`). Es el episodio de
iliquidez de colateral por excelencia. Contrapartida: solo hay **2018+**, no cubre 2008.

**5. Índices de condiciones financieras = validación de oro + features de funding.**
La **OFR Financial Stress Index** (US Office of Financial Research) descarga como CSV público, **diaria
2000-01-03 → hoy, viva**, y trae una **columna "Funding" dedicada** además de la global. Pico global
**29.32 el 2008-10-10** y Funding **9.58 el 2008-10-10**. Junto con `NFCI`/`ANFCI` (Chicago Fed,
semanal **1971+**) y `KCFSI` (mensual **1990+**) forman el ground-truth laxo de estrés.

**6. Treasury bid-ask / liquidity index: hueco real.** El índice de liquidez de Treasuries de
referencia (Bloomberg US Govt Securities Liquidity Index) es **propietario y NO gratis**. No hay serie
FRED equivalente. Proxies honestos: la propia OFR FSI, el **MOVE** (vol de bonos, categoría
*volatilidad*), o la *noise measure* de Hu-Pan-Wang (académica, **estática**, no viva). Lo marco como
`verificado=false` / gap documentado.

---

## Resumen de lo verificado (evidencia real, 2026-07-18)

| serie | fuente | inicio verif. | fin/estado | pico de estrés verificado |
|---|---|---|---|---|
| TED spread `TEDRATE` | FRED | 1986-01-02 | **descont. 2022-01-21** | 4.58 @ 2008-10-10 |
| Eurodollar−Tbill `DED3−DTB3` | FRED calc | 1971-01-04 | 2016-10-07 | 6.85 @ 1974 · 5.76 @ 2008 |
| Eurodollar 3M `DED3` | FRED | 1971-01-04 | 2016-10-07 | nivel 22.0 @ 1980 |
| NBER NY Commercial Paper | FRED (NBER) | 1857-01-01 | 1971-12-01 (M) | 24.0 @ 1857-10 |
| T-bill3M−FFR `TB3SMFFM` | FRED | 1954-07-01 | vivo (M) | flight-to-quality |
| Paper-bill `DCPF3M−DTB3` | FRED calc | 1997-01-02 | **vivo** | 3.73 @ 2008-10-15 |
| CP−FFR `CPFF` | FRED | 1997-01-02 | **vivo** | 2.91 @ 2008-10-15 |
| ABCP−Tbill `RIFSPPAAAD90NB−DTB3` | FRED calc | 2001-01-02 | **vivo** | 2.78 @ 2007-08-20 (ABCP freeze) |
| CP fin 3M `DCPF3M` | FRED | 1997-01-02 | vivo | nivel |
| CP nofin 3M `DCPN3M` | FRED | 1997-01-02 | vivo | nivel |
| ABCP 3M `RIFSPPAAAD90NB` | FRED | 2001-01-02 | vivo | nivel |
| SOFR | FRED | 2018-04-03 | vivo | 5.25 @ 2019-09-17 |
| SOFR−EFFR (SOFR-OIS proxy) | FRED calc | 2018-04-03 | vivo | +295 bp @ 2019-09-17 |
| Tri-Party GC repo `TGCRRATE` | FRED | 2018-04-02 | vivo | +295 bp vs EFFR @ 2019-09 |
| Overnight Bank Funding `OBFR` | FRED | 2016-03-01 | vivo | — |
| EFFR / DFF (Fed Funds ≈ OIS) | FRED | 2000 / 1954 | vivo | pata OIS de los spreads |
| ON RRP volumen `RRPONTSYD` | FRED | 2003-02-07 | vivo | 2553.7 \$B @ 2022-12-30 (glut) |
| IORB | FRED | 2021-07-29 | vivo | suelo de tipos |
| LIBOR 3M `LIOR3M` | FRED | 1970-01-01 | 2016-10-01 (Q) | trimestral, muerto |
| LIBOR USD diario | FRED | — | **HTTP 400 (retirado)** | no disponible |
| NFCI | FRED | 1971-01-08 | vivo (W) | 5.21 @ 1974-07 |
| ANFCI | FRED | 1971-01-08 | vivo (W) | 5.20 @ 1974-07 |
| NFCI Leverage/Risk | FRED | 1971-01-08 | vivo (W) | subíndices |
| KCFSI | FRED | 1990-02-01 | vivo (M) | 5.82 @ 2008-11 |
| **OFR FSI (global + Funding)** | OFR CSV | 2000-01-03 | **vivo (D)** | 29.32 / Funding 9.58 @ 2008-10-10 |
| STLFSI4 | FRED | 1993-12-31 | vivo (W) | (cross-check con crédito) |
| Treasury liquidity index | Bloomberg | — | **no gratis** | gap documentado |

**Fuentes que NO funcionan / no aplican (probadas):**
- **USD LIBOR diario en FRED**: `USD3MTD156N` y familia → **HTTP 400**, retiradas por licencia ICE.
- **Broad General Collateral Rate (BGCR)**: **no está en FRED** (`BGCRRATE` → 400); solo el
  tri-party `TGCRRATE`. Para repo GC uso TGCR (equivalente práctico).
- **Stooq**: no aporta aquí (las series de funding son tipos/índices Fed, no cotizaciones); además el
  agente de crédito/volatilidad confirmó que Stooq está bloqueado por challenge JS.
- **Treasury bid-ask index gratis**: no existe fuente libre y viva.

---

```yaml
series_liquidez_funding:
  # ============ SPINE PROFUNDO DE FUNDING (Pista A) ============
  - nombre_interno: NBER_NY_COMMERCIAL_PAPER
    descripcion: "Tipo de papel comercial de Nueva York (NBER Macrohistory). Coste de fondeo privado con historia de +160 años; captura pánicos bancarios clásicos."
    fuente: fred
    id: "M13002US35620M156NNBR"
    auth: FRED_API_KEY
    inicio_verificado: "1857-01-01"
    granularidad: mensual
    pista: A
    rol: spine
    relevancia_regimen: "Nivel de coste de fondeo mercantil desde 1857; picos en pánicos (1857, 1873, 1907, 1929-33). Textura de estrés de liquidez pre-moderna para el spine."
    verificado: true
    evidencia: "FRED obs -> 1380 obs mensuales, 1857-01-01:8.81 -> 1971-12-01:4.74, MAX 24.0 @ 1857-10 (Panico de 1857)."
    url: "https://fred.stlouisfed.org/series/M13002US35620M156NNBR"

  - nombre_interno: EURODOLLAR_TBILL_SPREAD
    descripcion: "Spread Eurodolar 3M (London) menos T-bill 3M = el TED spread ORIGINAL (pre-LIBOR). Estres de funding interbancario 1971-2016."
    fuente: fred
    id: "DED3,DTB3"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-04"
    granularidad: diaria
    pista: ambas
    rol: spine
    relevancia_regimen: "TED original diario 1971-2016; extiende la historia de funding 15 anos antes de TEDRATE. Picos 1974=6.85, 1982=4.43, 1987=2.83, 1998=1.53(LTCM), 2008=5.76. Empalmable con TEDRATE (solapa 1986-2016)."
    verificado: true
    evidencia: "Calc FRED DED3-DTB3 -> 11370 obs, 1971-01-04->2016-10-07, MAX 6.85 @ 1974-07-16; picos por crisis verificados."
    url: "https://fred.stlouisfed.org/series/DED3"

  - nombre_interno: EURODOLLAR_3M
    descripcion: "Tasa de deposito Eurodolar 3M (London). Pata privada del TED original; nivel de coste de fondeo en dolares offshore."
    fuente: fred
    id: "DED3"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-04"
    granularidad: diaria
    pista: A
    rol: enricher
    relevancia_regimen: "Nivel de fondeo interbancario 1971-2016; con DTB3 da el TED original. Descontinuado 2016 (fin de la serie H.15 de Eurodolares)."
    verificado: true
    evidencia: "FRED obs -> 11652 obs, 1971-01-04:6.5 -> 2016-10-07:0.93, MAX nivel 22.0 @ 1980-12-12 (era Volcker)."
    url: "https://fred.stlouisfed.org/series/DED3"

  - nombre_interno: TBILL3M_MINUS_FEDFUNDS
    descripcion: "T-bill 3M menos Fed Funds Rate (mensual). Escasez de bills / flight-to-quality; se vuelve negativo cuando el colateral seguro escasea."
    fuente: fred
    id: "TB3SMFFM"
    auth: FRED_API_KEY
    inicio_verificado: "1954-07-01"
    granularidad: mensual
    pista: A
    rol: enricher
    relevancia_regimen: "Proxy de demanda de seguridad/liquidez con historia de +70 anos; muy negativo en pánicos (bills escasos vs fondeo overnight)."
    verificado: true
    evidencia: "FRED metadata -> 1954-07-01 -> 2026-06-01, vivo (upd 2026-07-01)."
    url: "https://fred.stlouisfed.org/series/TB3SMFFM"

  # ============ FUNDING SPREADS MODERNOS (Pista B, TED y sucesores) ============
  - nombre_interno: TED_SPREAD
    descripcion: "TED Spread clasico (LIBOR 3M - T-bill 3M). Estres de financiacion bancaria. DESCONTINUADO por fin de LIBOR."
    fuente: fred
    id: "TEDRATE"
    auth: FRED_API_KEY
    inicio_verificado: "1986-01-02"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "El termometro de funding de referencia 1986-2022; picos 1987, 1998-LTCM, 2008-Lehman(4.58), 2020-COVID. Sin sucesor gratis directo (usar PAPER_BILL_SPREAD / CP_FFR vivos)."
    verificado: true
    evidencia: "FRED obs -> 8853 obs, 1986-01-02:0.90 -> 2022-01-21:0.09, MAX 4.58 @ 2008-10-10. Metadata: 'TED Spread (DISCONTINUED)'."
    url: "https://fred.stlouisfed.org/series/TEDRATE"

  - nombre_interno: PAPER_BILL_SPREAD
    descripcion: "Paper-bill spread = papel comercial financiero AA 90d menos T-bill 3M. Sucesor VIVO del TED (coste de fondeo privado - libre de riesgo)."
    fuente: fred
    id: "DCPF3M,DTB3"
    auth: FRED_API_KEY
    inicio_verificado: "1997-01-02"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Heredero directo del TED, VIVO desde 1997. Picos 3.73 @ 2008-10-15, 2.15 @ 2007-08-20 (ABCP freeze), 2.57 @ 2020-03-25 (COVID). Feature de funding viva post-LIBOR."
    verificado: true
    evidencia: "Calc FRED DCPF3M-DTB3 -> 6710 obs, 1997-01-02 -> 2026-07-16, MAX 3.73 @ 2008-10-15; 2007=2.15, 2020=2.57."
    url: "https://fred.stlouisfed.org/series/DCPF3M"

  - nombre_interno: CP_FFR_SPREAD
    descripcion: "Papel comercial 3M menos Fed Funds Rate (ya calculado por FRED). Spread de funding tipo 'CP-OIS' (FFR ~ OIS overnight)."
    fuente: fred
    id: "CPFF"
    auth: FRED_API_KEY
    inicio_verificado: "1997-01-02"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Estres del mercado de papel comercial vs coste overnight; VIVO 1997+. Pico 2.91 @ 2008-10-15 (freeze CP tras Lehman/Reserve Primary). Complementa PAPER_BILL."
    verificado: true
    evidencia: "FRED obs -> 6720 obs, 1997-01-02:-0.44 -> 2026-07-16:0.29, MAX 2.91 @ 2008-10-15, MIN -1.55 @ 1998-06-30."
    url: "https://fred.stlouisfed.org/series/CPFF"

  - nombre_interno: ABCP_BILL_SPREAD
    descripcion: "Papel comercial respaldado por activos (ABCP) 90d AA menos T-bill 3M. Termometro fino de la crisis de financiacion estructurada."
    fuente: fred
    id: "RIFSPPAAAD90NB,DTB3"
    auth: FRED_API_KEY
    inicio_verificado: "2001-01-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "El ABCP fue el epicentro del inicio de la GFC. Pico 4.39 @ 2008-09-26 y 2.78 @ 2007-08-20 (congelacion ABCP, arranque de la crisis). Senal temprana de estres de liquidez."
    verificado: true
    evidencia: "Calc FRED RIFSPPAAAD90NB-DTB3 -> 6354 obs, MAX 4.39 @ 2008-09-26; Aug2007 max 2.78 @ 2007-08-20."
    url: "https://fred.stlouisfed.org/series/RIFSPPAAAD90NB"

  - nombre_interno: CP_FINANCIAL_3M
    descripcion: "Tasa de papel comercial financiero AA a 90 dias (nivel). Pata privada de los spreads de funding."
    fuente: fred
    id: "DCPF3M"
    auth: FRED_API_KEY
    inicio_verificado: "1997-01-02"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Nivel de coste de fondeo bancario a 3M; base de PAPER_BILL_SPREAD. Vivo."
    verificado: true
    evidencia: "FRED metadata -> 1997-01-02 -> 2026-07-16, vivo (upd 2026-07-17)."
    url: "https://fred.stlouisfed.org/series/DCPF3M"

  - nombre_interno: CP_NONFINANCIAL_3M
    descripcion: "Tasa de papel comercial NO financiero AA a 90 dias (nivel). Fondeo corporativo real."
    fuente: fred
    id: "DCPN3M"
    auth: FRED_API_KEY
    inicio_verificado: "1997-01-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Financiero-nofinanciero = prima de riesgo bancario dentro del CP. Vivo (upd 2026-06-29, ligero lag)."
    verificado: true
    evidencia: "FRED metadata -> 1997-01-02 -> 2026-06-26, vivo."
    url: "https://fred.stlouisfed.org/series/DCPN3M"

  - nombre_interno: ABCP_3M
    descripcion: "Tasa de papel comercial respaldado por activos (ABCP) 90d AA (nivel)."
    fuente: fred
    id: "RIFSPPAAAD90NB"
    auth: FRED_API_KEY
    inicio_verificado: "2001-01-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Nivel de fondeo de vehiculos estructurados; base de ABCP_BILL_SPREAD. Vivo."
    verificado: true
    evidencia: "FRED obs -> 6359 obs, 2001-01-02:6.18 -> 2026-07-16:3.90, vivo."
    url: "https://fred.stlouisfed.org/series/RIFSPPAAAD90NB"

  # ============ SOFR / REPO (Pista B, funding moderno vivo) ============
  - nombre_interno: SOFR
    descripcion: "Secured Overnight Financing Rate. Tasa repo garantizada de referencia; spikes = estres de liquidez de repo."
    fuente: fred
    id: "SOFR"
    auth: FRED_API_KEY
    inicio_verificado: "2018-04-03"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Iliquidez del mercado repo. Salto a 5.25% el 2019-09-17 (repo crisis) con EFFR ~2.30; minimo 0.01 en 2020-03 (ZIRP COVID). No cubre 2008 (empieza 2018)."
    verificado: true
    evidencia: "FRED obs -> 2069 obs, 2018-04-03:1.83 -> 2026-07-16:3.62; nivel 5.25 @ 2019-09-17 (repo spike)."
    url: "https://fred.stlouisfed.org/series/SOFR"

  - nombre_interno: SOFR_EFFR_SPREAD
    descripcion: "Spread SOFR menos EFFR = proxy libre de SOFR-OIS (repo garantizado - fondeo overnight sin garantia). Estres de colateral/repo."
    fuente: fred
    id: "SOFR,EFFR"
    auth: FRED_API_KEY
    inicio_verificado: "2018-04-03"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "SOFR-OIS es EL indicador de estres de repo. Pico +295 bp el 2019-09-17 (septiembre 2019 repo crisis). Feature viva de iliquidez de funding garantizado."
    verificado: true
    evidencia: "Calc FRED (SOFR-EFFR)*100 -> 2069 obs, MAX 295 bp @ 2019-09-17 (identico en TGCR-EFFR)."
    url: "https://fred.stlouisfed.org/series/SOFR"

  - nombre_interno: TGCR_REPO
    descripcion: "Tri-Party General Collateral Rate. Tasa repo GC tri-party (NY Fed). Alternativa/redundancia a SOFR."
    fuente: fred
    id: "TGCRRATE"
    auth: FRED_API_KEY
    inicio_verificado: "2018-04-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Repo GC; TGCR-EFFR replica SOFR-OIS (+295 bp @ 2019-09-17). Broad GC (BGCR) NO esta en FRED (400), TGCR es el sustituto."
    verificado: true
    evidencia: "FRED obs -> 2070 obs, 2018-04-02:1.77 -> 2026-07-16:3.59; TGCR-EFFR MAX 295 bp @ 2019-09-17."
    url: "https://fred.stlouisfed.org/series/TGCRRATE"

  - nombre_interno: OBFR
    descripcion: "Overnight Bank Funding Rate. Fondeo bancario overnight sin garantia (fed funds + eurodolar domestico)."
    fuente: fred
    id: "OBFR"
    auth: FRED_API_KEY
    inicio_verificado: "2016-03-01"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Fondeo overnight no garantizado; OBFR-SOFR = prima garantia vs no-garantia (senal de estres bancario). Vivo."
    verificado: true
    evidencia: "FRED metadata -> 2016-03-01 -> 2026-07-16, vivo (upd 2026-07-17)."
    url: "https://fred.stlouisfed.org/series/OBFR"

  - nombre_interno: EFFR
    descripcion: "Effective Federal Funds Rate (diaria). Pata OIS/overnight de los spreads de funding."
    fuente: fred
    id: "EFFR"
    auth: FRED_API_KEY
    inicio_verificado: "2000-07-03"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "FFR efectivo ~ tasa OIS overnight; base de SOFR-OIS y CP-OIS. Para historia mas larga usar DFF (1954+)."
    verificado: true
    evidencia: "FRED metadata -> 2000-07-03 -> 2026-07-16, vivo."
    url: "https://fred.stlouisfed.org/series/EFFR"

  - nombre_interno: DFF
    descripcion: "Federal Funds Effective Rate (diaria, historia larga). Pata OIS overnight con historia desde 1954."
    fuente: fred
    id: "DFF"
    auth: FRED_API_KEY
    inicio_verificado: "1954-07-01"
    granularidad: diaria
    pista: ambas
    rol: enricher
    relevancia_regimen: "OIS overnight proxy con +70 anos; permite construir spreads de funding contra fondeo overnight en historia profunda."
    verificado: true
    evidencia: "FRED metadata -> 1954-07-01 -> 2026-07-16, vivo."
    url: "https://fred.stlouisfed.org/series/DFF"

  - nombre_interno: ON_RRP_VOLUME
    descripcion: "Overnight Reverse Repo (ON RRP) volumen usado en la facility de la Fed. Indicador de exceso de liquidez / escasez de colateral."
    fuente: fred
    id: "RRPONTSYD"
    auth: FRED_API_KEY
    inicio_verificado: "2003-02-07"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Regimen de liquidez (signo opuesto al estres): volumen enorme = glut de reservas/escasez de colateral. Pico 2553.7 B USD el 2022-12-30 (regimen de sobre-liquidez post-COVID)."
    verificado: true
    evidencia: "FRED obs -> 3285 obs, 2003-02-07 -> 2026-07-17, MAX 2553.716 B USD @ 2022-12-30."
    url: "https://fred.stlouisfed.org/series/RRPONTSYD"

  - nombre_interno: IORB
    descripcion: "Interest Rate on Reserve Balances. Suelo administrado del corredor de tipos; referencia para spreads de fondeo."
    fuente: fred
    id: "IORB"
    auth: FRED_API_KEY
    inicio_verificado: "2021-07-29"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Suelo de politica; SOFR-IORB y EFFR-IORB miden presion de fondeo dentro del corredor. Corto historico (2021+); antes IOER (2008-2021)."
    verificado: true
    evidencia: "FRED metadata -> 2021-07-29 -> 2026-07-20, vivo."
    url: "https://fred.stlouisfed.org/series/IORB"

  # ============ LIBOR (mayormente muerto en FRED) ============
  - nombre_interno: LIBOR_3M_QUARTERLY
    descripcion: "3-month USD LIBOR (trimestral, serie superviviente). Unica LIBOR con historia en FRED tras la retirada de las diarias."
    fuente: fred
    id: "LIOR3M"
    auth: FRED_API_KEY
    inicio_verificado: "1970-01-01"
    granularidad: mensual
    pista: A
    rol: fallback
    relevancia_regimen: "LIBOR 3M trimestral 1970-2016 (marketing dice mensual/trimestral). Demasiado baja frecuencia y muerta en 2016; solo referencia historica, NO util para LIBOR-OIS diario."
    verificado: true
    evidencia: "FRED search/metadata -> LIOR3M 1970-01-01 -> 2016-10-01, frecuencia Q. Las diarias (USD3MTD156N) devuelven HTTP 400."
    url: "https://fred.stlouisfed.org/series/LIOR3M"

  # ============ FINANCIAL CONDITIONS / VALIDACION ============
  - nombre_interno: OFR_FSI
    descripcion: "OFR Financial Stress Index (US Office of Financial Research). Indice de estres global diario con subindices por categoria (incluye Funding)."
    fuente: academico
    id: "OFR FSI (CSV publico)"
    auth: none
    inicio_verificado: "2000-01-03"
    granularidad: diaria
    pista: validacion
    rol: validation
    relevancia_regimen: "Ground-truth de estres DIARIO 2000+ (mejor granularidad que NFCI semanal). Pico global 29.32 @ 2008-10-10. Descargable como CSV publico, vivo. Valida los regimenes del detector."
    verificado: true
    evidencia: "GET financialresearch.gov/.../fsi.csv (511 KB) -> 6715 obs, 2000-01-03 -> 2026-07-15, cols [OFR FSI, Credit, Equity valuation, Safe assets, Funding, Volatility, US, OtherAdv, EM]. MAX 29.32 @ 2008-10-10."
    url: "https://www.financialresearch.gov/financial-stress-index/"

  - nombre_interno: OFR_FSI_FUNDING
    descripcion: "Subindice Funding de la OFR FSI. Componente de estres de financiacion ya construido (repo, CP, FX swap, etc.)."
    fuente: academico
    id: "OFR FSI -> columna 'Funding'"
    auth: none
    inicio_verificado: "2000-01-03"
    granularidad: diaria
    pista: validacion
    rol: validation
    relevancia_regimen: "EXACTAMENTE mi categoria, ya agregada por la OFR: estres de funding diario 2000+. Pico 9.58 @ 2008-10-10; 2011=2.11 (crisis euro), 2020=0.96. Ground-truth especifico de liquidez/funding."
    verificado: true
    evidencia: "Misma CSV OFR, columna 'Funding' -> 6715 obs, MAX 9.582 @ 2008-10-10; 2011 max 2.113 @ 2011-11-28, 2020 max 0.956 @ 2020-03-30."
    url: "https://www.financialresearch.gov/financial-stress-index/"

  - nombre_interno: NFCI
    descripcion: "Chicago Fed National Financial Conditions Index. Indice compuesto de condiciones financieras (105 indicadores, muchos de funding/liquidez)."
    fuente: fred
    id: "NFCI"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-08"
    granularidad: semanal
    pista: validacion
    rol: validation
    relevancia_regimen: "Ground-truth laxo con +50 anos; >0 = condiciones mas tensas que la media. Pico 5.21 @ 1974-07 (crisis 1974). Incluye subindices leverage/risk/credit."
    verificado: true
    evidencia: "FRED obs -> 2897 obs semanales, 1971-01-08:0.598 -> 2026-07-10:-0.538, MAX 5.212 @ 1974-07-19."
    url: "https://fred.stlouisfed.org/series/NFCI"

  - nombre_interno: ANFCI
    descripcion: "Chicago Fed ADJUSTED NFCI. NFCI ortogonalizado respecto a las condiciones economicas (estres financiero 'puro')."
    fuente: fred
    id: "ANFCI"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-08"
    granularidad: semanal
    pista: validacion
    rol: validation
    relevancia_regimen: "Estres financiero neto de ciclo economico; mejor para aislar shocks de liquidez de la actividad real. Pico 5.20 @ 1974-07."
    verificado: true
    evidencia: "FRED obs -> 2897 obs semanales, 1971-01-08:0.587 -> 2026-07-10:-0.535, MAX 5.196 @ 1974-07-19, MIN -1.32 @ 1977-02."
    url: "https://fred.stlouisfed.org/series/ANFCI"

  - nombre_interno: NFCI_LEVERAGE
    descripcion: "Chicago Fed NFCI Leverage subindex. Componente de apalancamiento (deuda y equity) de las condiciones financieras."
    fuente: fred
    id: "NFCILEVERAGE"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-08"
    granularidad: semanal
    pista: validacion
    rol: enricher
    relevancia_regimen: "Apalancamiento del sistema; suele adelantar fragilidad (build-up pre-crisis). Semanal 1971+, vivo."
    verificado: true
    evidencia: "FRED metadata -> 1971-01-08 -> 2026-07-10, vivo (upd 2026-07-15)."
    url: "https://fred.stlouisfed.org/series/NFCILEVERAGE"

  - nombre_interno: KCFSI
    descripcion: "Kansas City Financial Stress Index (mensual). Indice de estres financiero de 11 variables (incluye spreads de funding e interbancarios)."
    fuente: fred
    id: "KCFSI"
    auth: FRED_API_KEY
    inicio_verificado: "1990-02-01"
    granularidad: mensual
    pista: validacion
    rol: validation
    relevancia_regimen: "Ground-truth mensual 1990+; >0 = estres por encima de media. Pico 5.82 @ 2008-11 (post-Lehman). Complementa NFCI/OFR con otra metodologia."
    verificado: true
    evidencia: "FRED obs -> 437 obs mensuales, 1990-02-01:0.37 -> 2026-06-01:-0.76, MAX 5.824 @ 2008-11-01."
    url: "https://fred.stlouisfed.org/series/KCFSI"

  - nombre_interno: STLFSI4
    descripcion: "St. Louis Fed Financial Stress Index v4 (semanal). Indice compuesto de 18 series (varias de funding/interbancario)."
    fuente: fred
    id: "STLFSI4"
    auth: FRED_API_KEY
    inicio_verificado: "1993-12-31"
    granularidad: semanal
    pista: validacion
    rol: validation
    relevancia_regimen: "Ground-truth de estres compuesto; versiones previas (STLFSI/2/3) DESCONTINUADAS, v4 es la viva. Cross-check con el agente de credito (misma serie)."
    verificado: true
    evidencia: "Verificado por agente credito: FRED STLFSI4 -> 1698 obs semanales, 1993-12-31 -> 2026-07-10, vivo. STLFSI/2/3 confirmadas descontinuadas (2020/2022/2022)."
    url: "https://fred.stlouisfed.org/series/STLFSI4"

  # ============ GAPS / NO VERIFICADO ============
  - nombre_interno: LIBOR_USD_DAILY
    descripcion: "USD LIBOR diario (overnight/1M/3M) para un LIBOR-OIS diario historico."
    fuente: fred
    id: "USD3MTD156N / USDONTD156N / USD1MTD156N"
    auth: FRED_API_KEY
    inicio_verificado: null
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Daria el LIBOR-OIS diario clasico (mejor termometro interbancario 2001-2021). NO DISPONIBLE: retirado de FRED."
    verificado: false
    evidencia: "HTTP 400 en todas las USD*D156N (retiradas por licencia ICE tras el fin de LIBOR). Solo sobrevive LIOR3M trimestral -> 2016. Sin fuente gratis viva para LIBOR-OIS diario."
    url: "https://fred.stlouisfed.org/series/USD3MTD156N"

  - nombre_interno: TREASURY_LIQUIDITY_INDEX
    descripcion: "Indice de liquidez del mercado de Treasuries (bid-ask / market depth), tipo Bloomberg US Govt Securities Liquidity Index."
    fuente: academico
    id: "Bloomberg USGGSPRD / Hu-Pan-Wang noise measure"
    auth: none
    inicio_verificado: null
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Liquidez del activo mas seguro; se deteriora en crisis (2008, marzo-2020 'dash for cash'). PROPIETARIO / no gratis. Proxies: OFR FSI (Safe assets), MOVE (cat. volatilidad), noise measure academico (estatico)."
    verificado: false
    evidencia: "Bloomberg Liquidity Index no es gratis; no hay serie FRED equivalente. La noise measure de Hu-Pan-Wang existe como dataset academico pero es estatica, no viva. Marcado como gap honesto."
    url: "https://www.mit.edu/~junpan/"
```

---

## Recomendación de priorización para el pipeline

1. **Spine de funding profundo (Pista A, empalmable):** `EURODOLLAR_TBILL_SPREAD` (DED3−DTB3, TED
   original 1971-2016) empalmado con `TED_SPREAD` (TEDRATE 1986-2022) y luego `PAPER_BILL_SPREAD`
   (1997→vivo) = **una serie continua de estres de funding 1971→hoy**. Añadir `NBER_NY_COMMERCIAL_PAPER`
   (1857+) y `TBILL3M_MINUS_FEDFUNDS` (1954+) para textura de crisis pre-1971.
2. **Funding spreads vivos (Pista B, features nucleo):** `PAPER_BILL_SPREAD` y `CP_FFR_SPREAD` (los
   sucesores vivos del TED), `ABCP_BILL_SPREAD` (senal temprana estilo agosto-2007), y `SOFR_EFFR_SPREAD`
   (SOFR-OIS, el estres de repo moderno con el spike de sep-2019). Estos cuatro son el corazon de la
   categoria.
3. **Repo / liquidez de reservas:** `SOFR`, `TGCR_REPO`, `OBFR`, `ON_RRP_VOLUME` (regimen de
   glut/escasez de colateral). Solo 2016/2018+, no cubren 2008.
4. **Validacion (ground-truth de estres):** `OFR_FSI` + `OFR_FSI_FUNDING` (diario 2000+, con
   subindice de funding especifico — lo mejor para MI categoria), `NFCI`/`ANFCI` (1971+ semanal),
   `KCFSI` (1990+ mensual), `STLFSI4` (1993+). Son indices ya construidos = ground-truth laxo.
5. **Deuda tecnica honesta:**
   - **LIBOR-OIS diario clasico: irrecuperable gratis** (LIBOR retirado de FRED). Sustituirlo por
     `PAPER_BILL_SPREAD` / `CP_FFR_SPREAD` (pre/post-2022) y `SOFR_EFFR_SPREAD` (repo).
   - **Treasury bid-ask/liquidity index: no hay fuente gratis viva.** Usar OFR FSI ("Safe assets") o
     el MOVE (categoria volatilidad) como proxy; la noise measure academica es estatica.
   - **Broad GC repo (BGCR) no esta en FRED**; usar `TGCR_REPO` como equivalente tri-party.

**Fuentes web/datos consultados y verificados hoy:**
[FRED API (metadata + observations)](https://fred.stlouisfed.org/) ·
[OFR Financial Stress Index CSV](https://www.financialresearch.gov/financial-stress-index/) ·
[NBER Macrohistory via FRED (commercial paper 1857+)](https://fred.stlouisfed.org/series/M13002US35620M156NNBR)
