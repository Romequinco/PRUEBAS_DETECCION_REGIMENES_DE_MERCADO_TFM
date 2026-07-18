# Categoría: Tipos y curva — estado del arte de datos (verificado)

Investigador: agente **Tipos y curva**. Foco: curva completa de Treasuries (constant maturity
DGS3MO / DGS2 / DGS5 / DGS10 / DGS30 y nodos intermedios), pendientes/spreads (T10Y2Y, T10Y3M,
T10YFF), tasas overnight de política (DFF, EFFR, OBFR, **SOFR**), y el **máximo histórico**
(la curva diaria arranca en **1962-01-02**; el tramo corto mensual llega a **1934** vía T-bill).
La curva es una feature de régimen de primer orden: **inversión → recesión** y **pendiente/nivel**
separan ciclos de política monetaria. Sirve para **ambas pistas** (espina profunda mensual + panel
diario rico).

Todas las verificaciones se hicieron **de verdad** el 2026-07-18: descarga real contra la **FRED
API** con `FRED_API_KEY` del `.env` (sin imprimir la clave), y contra **yfinance `period='max'`**
para los proxies de Yahoo. Reporto la fecha de inicio **observada** en la propia respuesta de la
API, no la de marketing. Conteos y primeros/últimos valores salen del endpoint
`fred/series/observations`.

---

## Resumen ejecutivo de lo verificado

**El punto ciego 2013 NO está en la curva — la curva es la que lo tapa.** Comprobé explícitamente
la cobertura de 2013 en **todas** las series diarias: cada una tiene **250 observaciones válidas en
2013** y **22 días hábiles válidos en octubre 2013**. El único hueco de octubre es **2013-10-14
(Columbus Day, festivo del mercado de bonos SIFMA)**, que es correcto que falte. Es decir: **el
cierre del gobierno / crisis del techo de deuda de octubre 2013 NO deja hueco** en las constant
maturity. Verificado día a día:
- `DGS10` 2013-10-01..18 poblado todos los hábiles (2.66 → 2.60), sólo `.` el 2013-10-14.
- `DTB3` en ese tramo muestra el **estrés real del mercado de letras** por miedo a default técnico:
  saltó **0.02 → 0.14 (2013-10-15)** y volvió a 0.04. Eso es señal de régimen micro, no ruido.

Y el **taper tantrum** (el evento de régimen de tipos de 2013) queda capturado limpio:
`DGS10` 1.66% (2013-05-01) → 2.98% (2013-09-05) → 3.04% (2013-12-31); la pendiente
`T10Y2Y` se empinó **1.46 → 2.46** y `T10Y3M` **1.60 → 2.96** entre mayo y septiembre 2013.
**Conclusión operativa:** la curva diaria de FRED es densa y sin agujeros a través de 2013, así que
puede usarse como **ancla para rellenar cualquier hueco de 2013** que tengan otros paneles (Pista B).

**Curva núcleo (diaria, FRED, máximo histórico):** El tramo largo constant maturity arranca todo el
mismo día **1962-01-02**: `DGS1`, `DGS3`, `DGS5`, `DGS10`, `DGS20` (16.119 obs; DGS20 con hueco
1987–1993). `DGS7` desde 1969-07-01. `DGS2` desde **1976-06-01**. Tramo corto CMT (`DGS3MO`,
`DGS6MO`) desde **1981-09-01**; `DGS1MO` desde 2001-07-31. `DGS30` desde **1977-02-15** con el
**hueco de descontinuación 2002-02 → 2006-02**.

**Tramo corto profundo (mejor que DGS3MO para pre-1981):** las letras a descuento en secundario van
mucho más atrás — `DTB3` (3M) **diaria desde 1954-01-04** (18.125 obs), `DTB6` desde 1958-12,
`DTB1YR` desde 1959-07. Y la mensual `TB3MS` (3M) llega a **1934-01**, la serie de tipos más
profunda de todo FRED. Para el overnight de política, `DFF` (Fed Funds effective **diaria**) desde
**1954-07-01** (26.314 obs) es la espina corta continua.

**Pendientes / spreads (señal de recesión):** `T10Y2Y` (1976-06), `T10Y3M` (1982-01, el predictor
canónico de recesión), `T10YFF` y `T5YFF` (ambos desde **1962-01-02**, curva vs política). Para
pendiente profunda pre-1976 se **computa** `GS10 − TB3MS` (mensual, desde 1953/1934).

**Overnight / funding:** `DFF` (1954, spine), `EFFR` (2000-07), `OBFR` (2016-03), **`SOFR`
(2018-04-03)**, `IORB` (2021-07). SOFR sólo existe desde 2018; para estrés de repo/funding la
feature útil es `SOFR − IORB` o `SOFR − EFFR` (pico repo sep-2019). `DPCREDIT` (discount window)
desde 2003.

**Reales / breakevens (régimen inflación vs crecimiento):** TIPS `DFII5`/`DFII10` desde
**2003-01-02**, `DFII30` desde 2010-02; breakevens `T10YIE`/`T5YIE` desde 2003-01, y el forward
`T5YIFR` (5y5y, ancla de expectativas de inflación) desde 2003-01.

**Fallbacks y hallazgos de fuente:**
- **yfinance (Yahoo) es fallback sólido y con historia igual de larga**: `^TNX` (10Y) 16.122 filas
  desde **1962-01-02**, `^TYX` (30Y) desde 1977-02-15, `^FVX` (5Y) desde 1962-01-02, y `^IRX`
  (13-week T-bill) desde **1960-01-04** (¡2 años más que DGS3MO!). Ojo: hoy Yahoo cotiza `^TNX`
  como el **rendimiento directo** (último 4.541 = 4.54%), ya no ×10. Redundan a FRED 1:1.
- **Stooq está BLOQUEADO** (igual que reportó el agente de Volatilidad): `stooq.com/q/d/l/?s=...`
  devuelve un HTML `noscript` / challenge JS, no CSV. Lo dejo como fallback **no verificado**.
- **No en FRED (fuera de alcance libre):** el **term premium ACM** (modelo del NY Fed) no está como
  serie FRED estándar; el **MOVE** (vol de bonos) lo cubre el agente de Volatilidad (yfinance ^MOVE).

**No verificado / límites honestos:** para tipos **pre-1934** (curva larga s. XIX–XX) la vía es
académica (Shiller long rate mensual desde 1871; Jorda-Schularick-Taylor macrohistory anual 1870+;
NBER); **no lo descargué en esta pasada**, lo dejo listado como `academico / verificado:false`.
Dentro de FRED, la profundidad honesta verificada es 1934 (corto, mensual) / 1953 (curva, mensual) /
1962 (curva, diaria).

---

## Detalle por serie (evidencia)

| serie | fuente | inicio verificado | fin/estado | 2013 | nota |
|---|---|---|---|---|---|
| DGS10 | FRED DGS10 | 1962-01-02 | vivo (upd 2026-07-17) | 250 obs, ok | core largo, spine curva |
| DGS5 | FRED DGS5 | 1962-01-02 | vivo | 250 | core |
| DGS3 | FRED DGS3 | 1962-01-02 | vivo | 250 | nodo intermedio |
| DGS1 | FRED DGS1 | 1962-01-02 | vivo | 250 | tramo 1y |
| DGS2 | FRED DGS2 | 1976-06-01 | vivo | 250 | core corto (pata de T10Y2Y) |
| DGS7 | FRED DGS7 | 1969-07-01 | vivo | 250 | nodo intermedio |
| DGS20 | FRED DGS20 | 1962-01-02 | vivo | 250 | **hueco 1987–1993** |
| DGS30 | FRED DGS30 | 1977-02-15 | vivo | 250 | **hueco 2002-02→2006-02** |
| DGS6MO | FRED DGS6MO | 1981-09-01 | vivo | 250 | CMT 6m |
| DGS3MO | FRED DGS3MO | 1981-09-01 | vivo | 250 | CMT 3m (pata de T10Y3M) |
| DGS1MO | FRED DGS1MO | 2001-07-31 | vivo | 250 | CMT 1m |
| T10Y2Y | FRED T10Y2Y | 1976-06-01 | vivo (upd 2026-07-17) | 250 | pendiente 10y-2y |
| T10Y3M | FRED T10Y3M | 1982-01-04 | vivo | 250 | pendiente 10y-3m (recesión) |
| T10YFF | FRED T10YFF | 1962-01-02 | vivo | 250 | curva vs política, largo |
| T5YFF | FRED T5YFF | 1962-01-02 | vivo | 250 | curva vs política |
| DFF | FRED DFF | 1954-07-01 | vivo | 365 | Fed Funds effective diario, spine overnight |
| FEDFUNDS | FRED FEDFUNDS | 1954-07-01 | vivo (mensual) | 12 | política mensual profunda |
| EFFR | FRED EFFR | 2000-07-03 | vivo | 251 | effective FFR (NY Fed) |
| OBFR | FRED OBFR | 2016-03-01 | vivo | 0 | overnight bank funding |
| SOFR | FRED SOFR | 2018-04-03 | vivo | 0 | repo overnight (post-LIBOR) |
| IORB | FRED IORB | 2021-07-29 | vivo | 0 | interés reservas (techo corridor) |
| DPCREDIT | FRED DPCREDIT | 2003-01-09 | vivo | 261 | discount window primary |
| DTB3 | FRED DTB3 | 1954-01-04 | vivo | 250 | 3M T-bill diario, spine corto profundo |
| DTB6 | FRED DTB6 | 1958-12-09 | vivo | 250 | 6M T-bill diario |
| DTB1YR | FRED DTB1YR | 1959-07-15 | vivo | 250 | 1Y T-bill diario |
| TB3MS | FRED TB3MS | 1934-01-01 | vivo (mensual) | 12 | **tipo más profundo de FRED** |
| GS10 | FRED GS10 | 1953-04-01 | vivo (mensual) | 12 | curva mensual profunda |
| GS5 | FRED GS5 | 1953-04-01 | vivo (mensual) | 12 | curva mensual profunda |
| GS1 | FRED GS1 | 1953-04-01 | vivo (mensual) | 12 | curva mensual profunda |
| IRLTLT01USM156N | FRED (OECD) | 1953-04-01 | vivo (mensual) | 12 | long rate 10y (= GS10) |
| DFII10 | FRED DFII10 | 2003-01-02 | vivo | 250 | TIPS 10y real |
| DFII5 | FRED DFII5 | 2003-01-02 | vivo | 250 | TIPS 5y real |
| DFII30 | FRED DFII30 | 2010-02-22 | vivo | 250 | TIPS 30y real |
| T10YIE | FRED T10YIE | 2003-01-02 | vivo | 250 | breakeven 10y |
| T5YIE | FRED T5YIE | 2003-01-02 | vivo | 250 | breakeven 5y |
| T5YIFR | FRED T5YIFR | 2003-01-02 | vivo | 250 | 5y5y forward inflación |
| ^TNX/^FVX/^TYX/^IRX | yfinance | 1962/1960 | vivo | ok | fallback Yahoo (redunda a FRED) |

Notas de features derivadas (capa de features, causales/expanding):
- **Nivel** = `DGS10` (o el vector completo 3M-2-5-10-30 como forma de curva).
- **Pendiente** = `T10Y2Y`, `T10Y3M` (nativas) o `DGS10 − DGS2`, `DGS30 − DGS5` (bull/bear steepen).
  Pendiente profunda pre-1976 = `GS10 − TB3MS` (mensual, desde 1934/1953).
- **Curvatura (butterfly)** = `2·DGS5 − DGS2 − DGS10` (cambia de signo en giros de ciclo).
- **Inversión (dummy de recesión)** = `T10Y3M < 0` (predictor canónico, desde 1982).
- **Real vs nominal** = `DFII10` (real), `T10YIE`/`T5YIFR` (inflación esperada) → separa régimen
  crecimiento vs inflación.
- **Estrés de funding** = `SOFR − IORB`, `SOFR − EFFR` (repo, pico sep-2019), `DTB3 − DFF`
  (miedo default, pico oct-2013).

```yaml
series_tipos_curva:
  # ================= CURVA NÚCLEO DIARIA (FRED, máximo histórico) =================
  - nombre_interno: DGS10
    descripcion: "US Treasury 10y constant maturity (diaria). Nivel largo de referencia; espina de la curva."
    fuente: fred
    id: "DGS10"
    auth: FRED_API_KEY
    inicio_verificado: "1962-01-02"
    granularidad: diaria
    pista: ambas
    rol: spine
    relevancia_regimen: "Nivel de tipos largos; motor de valoraciones, duración y ciclo. Máximo histórico diario 1962."
    verificado: true
    evidencia: "FRED DGS10 -> 16119 obs validas, first 1962-01-02:4.06 last 2026-07-16:4.57 (upd 2026-07-17). 250 obs en 2013, 22 en oct-2013 (solo falta 10-14 Columbus Day). yfinance ^TNX confirma 16122 filas mismo inicio."
    url: "https://fred.stlouisfed.org/series/DGS10"

  - nombre_interno: DGS5
    descripcion: "US Treasury 5y constant maturity (diaria). Nodo medio de la curva."
    fuente: fred
    id: "DGS5"
    auth: FRED_API_KEY
    inicio_verificado: "1962-01-02"
    granularidad: diaria
    pista: ambas
    rol: core
    relevancia_regimen: "Belly de la curva; clave para curvatura (butterfly) y sensibilidad al ciclo de política."
    verificado: true
    evidencia: "FRED DGS5 -> 16119 obs, first 1962-01-02:3.88 last 2026-07-16:4.28. 250 obs en 2013. yfinance ^FVX confirma 16122 filas desde 1962-01-02."
    url: "https://fred.stlouisfed.org/series/DGS5"

  - nombre_interno: DGS3
    descripcion: "US Treasury 3y constant maturity (diaria)."
    fuente: fred
    id: "DGS3"
    auth: FRED_API_KEY
    inicio_verificado: "1962-01-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Nodo intermedio corto; forma fina de la curva front-end."
    verificado: true
    evidencia: "FRED DGS3 -> 16119 obs, first 1962-01-02:3.70 last 2026-07-16:4.2. 250 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DGS3"

  - nombre_interno: DGS1
    descripcion: "US Treasury 1y constant maturity (diaria)."
    fuente: fred
    id: "DGS1"
    auth: FRED_API_KEY
    inicio_verificado: "1962-01-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Tramo 1y; sensible a expectativas de tipos a 12 meses (ancla de política)."
    verificado: true
    evidencia: "FRED DGS1 -> 16119 obs, first 1962-01-02:3.22 last 2026-07-16:3.99. 250 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DGS1"

  - nombre_interno: DGS2
    descripcion: "US Treasury 2y constant maturity (diaria). Pata corta de la pendiente 10y-2y."
    fuente: fred
    id: "DGS2"
    auth: FRED_API_KEY
    inicio_verificado: "1976-06-01"
    granularidad: diaria
    pista: ambas
    rol: core
    relevancia_regimen: "Refleja expectativas de política a 2 años; T10Y2Y = predictor de recesión clasico."
    verificado: true
    evidencia: "FRED DGS2 -> 12527 obs, first 1976-06-01:7.260 last 2026-07-16:4.16. 250 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DGS2"

  - nombre_interno: DGS7
    descripcion: "US Treasury 7y constant maturity (diaria)."
    fuente: fred
    id: "DGS7"
    auth: FRED_API_KEY
    inicio_verificado: "1969-07-01"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Nodo intermedio largo; completa la forma de la curva."
    verificado: true
    evidencia: "FRED DGS7 -> 14249 obs, first 1969-07-01:6.88 last 2026-07-16:4.41. 250 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DGS7"

  - nombre_interno: DGS20
    descripcion: "US Treasury 20y constant maturity (diaria). OJO: hueco de descontinuación 1987-1993."
    fuente: fred
    id: "DGS20"
    auth: FRED_API_KEY
    inicio_verificado: "1962-01-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Extremo largo de la curva; usar con cuidado por el hueco 1987-1993 (imputar/omitir en ese tramo)."
    verificado: true
    evidencia: "FRED DGS20 -> 14430 obs (vs 16119 de DGS10: ~1689 dias menos = hueco 1987-1993), first 1962-01-02:4.07 last 2026-07-16:5.09. 250 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DGS20"

  - nombre_interno: DGS30
    descripcion: "US Treasury 30y constant maturity (diaria). OJO: hueco 2002-02 -> 2006-02 (bono 30y descontinuado)."
    fuente: fred
    id: "DGS30"
    auth: FRED_API_KEY
    inicio_verificado: "1977-02-15"
    granularidad: diaria
    pista: ambas
    rol: core
    relevancia_regimen: "Long bond; pendiente 30y-5y y term premium. Manejar el hueco 2002-2006 explícitamente."
    verificado: true
    evidencia: "FRED DGS30 -> 12349 obs, first 1977-02-15:7.70 last 2026-07-16:5.09. 250 obs en 2013. yfinance ^TYX confirma 12380 filas mismo inicio 1977-02-15."
    url: "https://fred.stlouisfed.org/series/DGS30"

  - nombre_interno: DGS6MO
    descripcion: "US Treasury 6-month constant maturity (diaria)."
    fuente: fred
    id: "DGS6MO"
    auth: FRED_API_KEY
    inicio_verificado: "1981-09-01"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Tramo muy corto CMT; expectativas de política inmediatas. Para pre-1981 usar DTB6."
    verificado: true
    evidencia: "FRED DGS6MO -> 11217 obs, first 1981-09-01:17.17 last 2026-07-16:3.94. 250 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DGS6MO"

  - nombre_interno: DGS3MO
    descripcion: "US Treasury 3-month constant maturity (diaria). Pata corta de la pendiente 10y-3m."
    fuente: fred
    id: "DGS3MO"
    auth: FRED_API_KEY
    inicio_verificado: "1981-09-01"
    granularidad: diaria
    pista: ambas
    rol: core
    relevancia_regimen: "T10Y3M (10y-3m) es el mejor predictor de recesion. Para pre-1981 usar DTB3 (1954)."
    verificado: true
    evidencia: "FRED DGS3MO -> 11217 obs, first 1981-09-01:17.01 last 2026-07-16:3.84. 250 obs en 2013, 22 en oct-2013."
    url: "https://fred.stlouisfed.org/series/DGS3MO"

  - nombre_interno: DGS1MO
    descripcion: "US Treasury 1-month constant maturity (diaria). Extremo ultra-corto CMT."
    fuente: fred
    id: "DGS1MO"
    auth: FRED_API_KEY
    inicio_verificado: "2001-07-31"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Sensible a estres de bills / techo de deuda (spikes en 2011, 2013, 2023)."
    verificado: true
    evidencia: "FRED DGS1MO -> 6241 obs, first 2001-07-31:3.67 last 2026-07-16:3.76. 250 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DGS1MO"

  # ================= PENDIENTES / SPREADS =================
  - nombre_interno: T10Y2Y
    descripcion: "Pendiente 10y-2y (10-Year minus 2-Year CMT). Curva de referencia."
    fuente: fred
    id: "T10Y2Y"
    auth: FRED_API_KEY
    inicio_verificado: "1976-06-01"
    granularidad: diaria
    pista: ambas
    rol: core
    relevancia_regimen: "Inversion (<0) precede recesiones; pendiente separa fases de ciclo. Empino en taper tantrum 2013 (1.46->2.46)."
    verificado: true
    evidencia: "FRED T10Y2Y -> 12528 obs, first 1976-06-01:0.68 last 2026-07-17:0.37. 250 obs en 2013; 2013-05-01=1.46 vs 2013-09-05=2.46."
    url: "https://fred.stlouisfed.org/series/T10Y2Y"

  - nombre_interno: T10Y3M
    descripcion: "Pendiente 10y-3m (10-Year minus 3-Month CMT). El predictor de recesion canonico (Estrella-Mishkin)."
    fuente: fred
    id: "T10Y3M"
    auth: FRED_API_KEY
    inicio_verificado: "1982-01-04"
    granularidad: diaria
    pista: ambas
    rol: core
    relevancia_regimen: "Mejor senal de recesion a 12m; su inversion marca cambio de regimen. Empino fuerte en 2013 (1.60->2.96)."
    verificado: true
    evidencia: "FRED T10Y3M -> 11137 obs, first 1982-01-04:2.32 last 2026-07-17:0.7. 250 obs en 2013, 22 en oct-2013."
    url: "https://fred.stlouisfed.org/series/T10Y3M"

  - nombre_interno: T10YFF
    descripcion: "10-Year CMT menos Federal Funds Rate. Curva vs politica (largo)."
    fuente: fred
    id: "T10YFF"
    auth: FRED_API_KEY
    inicio_verificado: "1962-01-02"
    granularidad: diaria
    pista: ambas
    rol: core
    relevancia_regimen: "Postura de politica relativa al largo; negativo = politica restrictiva vs mercado. Historia diaria 1962."
    verificado: true
    evidencia: "FRED T10YFF -> 16117 obs, first 1962-01-02:1.31 last 2026-07-16:0.94. 250 obs en 2013."
    url: "https://fred.stlouisfed.org/series/T10YFF"

  - nombre_interno: T5YFF
    descripcion: "5-Year CMT menos Federal Funds Rate. Curva vs politica (medio)."
    fuente: fred
    id: "T5YFF"
    auth: FRED_API_KEY
    inicio_verificado: "1962-01-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Complementa T10YFF; forma del front-end frente a la tasa de politica. Historia diaria 1962."
    verificado: true
    evidencia: "FRED T5YFF -> 16117 obs, first 1962-01-02:1.13 last 2026-07-16:0.65. 250 obs en 2013."
    url: "https://fred.stlouisfed.org/series/T5YFF"

  # ================= OVERNIGHT / POLITICA / FUNDING =================
  - nombre_interno: DFF
    descripcion: "Federal Funds Effective Rate (diaria). Tasa de politica overnight; espina corta continua."
    fuente: fred
    id: "DFF"
    auth: FRED_API_KEY
    inicio_verificado: "1954-07-01"
    granularidad: diaria
    pista: ambas
    rol: spine
    relevancia_regimen: "Nivel de politica monetaria; ciclos de subida/bajada definen regimenes macro. Diaria desde 1954 (26314 obs)."
    verificado: true
    evidencia: "FRED DFF -> 26314 obs, first 1954-07-01:1.13 last 2026-07-16:3.63. 365 filas-calendario en 2013 (rellena findes con ultimo valor)."
    url: "https://fred.stlouisfed.org/series/DFF"

  - nombre_interno: FEDFUNDS
    descripcion: "Federal Funds Effective Rate (mensual). Version mensual de politica para la espina profunda."
    fuente: fred
    id: "FEDFUNDS"
    auth: FRED_API_KEY
    inicio_verificado: "1954-07-01"
    granularidad: mensual
    pista: A
    rol: enricher
    relevancia_regimen: "Serie de politica mensual limpia para modelos de baja frecuencia (Pista A)."
    verificado: true
    evidencia: "FRED FEDFUNDS -> 864 obs mensuales, first 1954-07-01:0.80 last 2026-06-01:3.63."
    url: "https://fred.stlouisfed.org/series/FEDFUNDS"

  - nombre_interno: EFFR
    descripcion: "Effective Federal Funds Rate (NY Fed, diaria). Version moderna con volumen/percentiles."
    fuente: fred
    id: "EFFR"
    auth: FRED_API_KEY
    inicio_verificado: "2000-07-03"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Base para spreads de funding (SOFR-EFFR); complementa DFF con la serie oficial NY Fed."
    verificado: true
    evidencia: "FRED EFFR -> 6537 obs, first 2000-07-03:7.03 last 2026-07-16:3.63. 251 obs en 2013."
    url: "https://fred.stlouisfed.org/series/EFFR"

  - nombre_interno: OBFR
    descripcion: "Overnight Bank Funding Rate (fed funds + eurodollar). Funding no colateralizado."
    fuente: fred
    id: "OBFR"
    auth: FRED_API_KEY
    inicio_verificado: "2016-03-01"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Estres de funding bancario unsecured; complementa SOFR (secured). Solo desde 2016."
    verificado: true
    evidencia: "FRED OBFR -> 2607 obs, first 2016-03-01:0.37 last 2026-07-16:3.62. 0 obs en 2013 (empieza 2016)."
    url: "https://fred.stlouisfed.org/series/OBFR"

  - nombre_interno: SOFR
    descripcion: "Secured Overnight Financing Rate. Tasa de repo overnight, sustituto de LIBOR."
    fuente: fred
    id: "SOFR"
    auth: FRED_API_KEY
    inicio_verificado: "2018-04-03"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Estres de repo/liquidez colateralizada; SOFR-IORB marca dislocaciones (pico repo sep-2019). Solo desde 2018."
    verificado: true
    evidencia: "FRED SOFR -> 2069 obs, first 2018-04-03:1.83 last 2026-07-16:3.62. 0 obs en 2013 (no existia)."
    url: "https://fred.stlouisfed.org/series/SOFR"

  - nombre_interno: IORB
    descripcion: "Interest Rate on Reserve Balances. Techo administrado del corridor de politica."
    fuente: fred
    id: "IORB"
    auth: FRED_API_KEY
    inicio_verificado: "2021-07-29"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Referencia del corridor; SOFR-IORB / EFFR-IORB miden presion de liquidez. Reemplaza IOER (pre-2021)."
    verificado: true
    evidencia: "FRED IORB -> 1818 obs, first 2021-07-29:0.15 last 2026-07-20:3.65."
    url: "https://fred.stlouisfed.org/series/IORB"

  - nombre_interno: DPCREDIT
    descripcion: "Discount Window Primary Credit Rate (diaria). Techo penalizador de la Fed."
    fuente: fred
    id: "DPCREDIT"
    auth: FRED_API_KEY
    inicio_verificado: "2003-01-09"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Ancla superior del corridor; cambios senalan estres (recortes de emergencia 2008, 2020)."
    verificado: true
    evidencia: "FRED DPCREDIT -> 6136 obs, first 2003-01-09:2.25 last 2026-07-16:3.75. 261 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DPCREDIT"

  # ================= TRAMO CORTO PROFUNDO (T-BILLS, pre-1981) =================
  - nombre_interno: DTB3
    descripcion: "3-Month Treasury Bill secondary market (discount basis, diaria). Spine corto profundo."
    fuente: fred
    id: "DTB3"
    auth: FRED_API_KEY
    inicio_verificado: "1954-01-04"
    granularidad: diaria
    pista: ambas
    rol: spine
    relevancia_regimen: "Tramo corto diario desde 1954 (mucho antes que DGS3MO 1981); base para pendiente profunda GS10-TB3. En oct-2013 marca el estres del techo de deuda (0.02->0.14)."
    verificado: true
    evidencia: "FRED DTB3 -> 18125 obs, first 1954-01-04:1.330 last 2026-07-16:3.7. 250 obs en 2013; spike 2013-10-15=0.14 (miedo default)."
    url: "https://fred.stlouisfed.org/series/DTB3"

  - nombre_interno: DTB6
    descripcion: "6-Month Treasury Bill secondary market (discount basis, diaria)."
    fuente: fred
    id: "DTB6"
    auth: FRED_API_KEY
    inicio_verificado: "1958-12-09"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Tramo 6m diario profundo (pre-1981); completa el front-end historico."
    verificado: true
    evidencia: "FRED DTB6 -> 16885 obs, first 1958-12-09:3.09 last 2026-07-16:3.78. 250 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DTB6"

  - nombre_interno: DTB1YR
    descripcion: "1-Year Treasury Bill secondary market (discount basis, diaria)."
    fuente: fred
    id: "DTB1YR"
    auth: FRED_API_KEY
    inicio_verificado: "1959-07-15"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Tramo 1y diario profundo; alternativa a DGS1 con historia comparable."
    verificado: true
    evidencia: "FRED DTB1YR -> 15049 obs, first 1959-07-15:4.52 last 2026-07-16:3.82. 250 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DTB1YR"

  - nombre_interno: TB3MS
    descripcion: "3-Month Treasury Bill secondary market (mensual, discount basis). La serie de tipos mas profunda de FRED."
    fuente: fred
    id: "TB3MS"
    auth: FRED_API_KEY
    inicio_verificado: "1934-01-01"
    granularidad: mensual
    pista: A
    rol: spine
    relevancia_regimen: "Tramo corto mensual desde 1934: cubre la Depresion, guerra, todos los ciclos. Base de pendiente profunda GS10-TB3MS."
    verificado: true
    evidencia: "FRED TB3MS -> 1110 obs mensuales, first 1934-01-01:0.72 last 2026-06-01:3.66."
    url: "https://fred.stlouisfed.org/series/TB3MS"

  # ================= CURVA MENSUAL PROFUNDA (1953) =================
  - nombre_interno: GS10
    descripcion: "10y Treasury constant maturity (mensual). Curva larga mensual profunda para Pista A."
    fuente: fred
    id: "GS10"
    auth: FRED_API_KEY
    inicio_verificado: "1953-04-01"
    granularidad: mensual
    pista: A
    rol: spine
    relevancia_regimen: "Nivel largo mensual desde 1953; extiende DGS10 (1962) ~9 anos hacia atras para modelos de baja frecuencia."
    verificado: true
    evidencia: "FRED GS10 -> 879 obs mensuales, first 1953-04-01:2.83 last 2026-06-01:4.47. Coincide con IRLTLT01USM156N (OECD)."
    url: "https://fred.stlouisfed.org/series/GS10"

  - nombre_interno: GS5
    descripcion: "5y Treasury constant maturity (mensual)."
    fuente: fred
    id: "GS5"
    auth: FRED_API_KEY
    inicio_verificado: "1953-04-01"
    granularidad: mensual
    pista: A
    rol: enricher
    relevancia_regimen: "Belly mensual profundo; curvatura historica desde 1953."
    verificado: true
    evidencia: "FRED GS5 -> 879 obs mensuales, first 1953-04-01:2.62 last 2026-06-01:4.21."
    url: "https://fred.stlouisfed.org/series/GS5"

  - nombre_interno: GS1
    descripcion: "1y Treasury constant maturity (mensual)."
    fuente: fred
    id: "GS1"
    auth: FRED_API_KEY
    inicio_verificado: "1953-04-01"
    granularidad: mensual
    pista: A
    rol: enricher
    relevancia_regimen: "Front-end mensual profundo; pata corta de pendientes historicas (GS10-GS1)."
    verificado: true
    evidencia: "FRED GS1 -> 879 obs mensuales, first 1953-04-01:2.36 last 2026-06-01:3.91."
    url: "https://fred.stlouisfed.org/series/GS1"

  - nombre_interno: IRLTLT01USM156N
    descripcion: "OECD Long-Term Government Bond Yield 10y US (mensual). Espejo de GS10, homogeneo cross-country."
    fuente: fred
    id: "IRLTLT01USM156N"
    auth: FRED_API_KEY
    inicio_verificado: "1953-04-01"
    granularidad: mensual
    pista: A
    rol: fallback
    relevancia_regimen: "Redundancia de GS10 con definicion OECD (util si se compara con otros paises)."
    verificado: true
    evidencia: "FRED IRLTLT01USM156N -> 879 obs, first 1953-04-01:2.83 last 2026-06-01:4.47 (identica a GS10)."
    url: "https://fred.stlouisfed.org/series/IRLTLT01USM156N"

  # ================= REALES / BREAKEVENS (regimen inflacion vs crecimiento) =================
  - nombre_interno: DFII10
    descripcion: "10y TIPS real yield (constant maturity, diaria). Tipo real largo."
    fuente: fred
    id: "DFII10"
    auth: FRED_API_KEY
    inicio_verificado: "2003-01-02"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Tipo real = coste real del dinero; separa shocks de crecimiento (real) de inflacion (breakeven)."
    verificado: true
    evidencia: "FRED DFII10 -> 5888 obs, first 2003-01-02:2.43 last 2026-07-16:2.35. 250 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DFII10"

  - nombre_interno: DFII5
    descripcion: "5y TIPS real yield (constant maturity, diaria)."
    fuente: fred
    id: "DFII5"
    auth: FRED_API_KEY
    inicio_verificado: "2003-01-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Tipo real medio; pendiente real DFII10-DFII5."
    verificado: true
    evidencia: "FRED DFII5 -> 5888 obs, first 2003-01-02:1.75 last 2026-07-16:2.04. 250 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DFII5"

  - nombre_interno: DFII30
    descripcion: "30y TIPS real yield (constant maturity, diaria)."
    fuente: fred
    id: "DFII30"
    auth: FRED_API_KEY
    inicio_verificado: "2010-02-22"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Extremo largo real; term premium real. Solo desde 2010."
    verificado: true
    evidencia: "FRED DFII30 -> 4103 obs, first 2010-02-22:2.22 last 2026-07-16:2.91. 250 obs en 2013."
    url: "https://fred.stlouisfed.org/series/DFII30"

  - nombre_interno: T10YIE
    descripcion: "10-Year Breakeven Inflation Rate (nominal DGS10 - real DFII10, diaria)."
    fuente: fred
    id: "T10YIE"
    auth: FRED_API_KEY
    inicio_verificado: "2003-01-02"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Inflacion esperada de mercado; su colapso marca deflacion/deleveraging (2008, 2020)."
    verificado: true
    evidencia: "FRED T10YIE -> 5889 obs, first 2003-01-02:1.64 last 2026-07-17:2.24. 250 obs en 2013."
    url: "https://fred.stlouisfed.org/series/T10YIE"

  - nombre_interno: T5YIE
    descripcion: "5-Year Breakeven Inflation Rate (diaria)."
    fuente: fred
    id: "T5YIE"
    auth: FRED_API_KEY
    inicio_verificado: "2003-01-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Inflacion esperada corta; mas sensible a shocks de energia/ciclo."
    verificado: true
    evidencia: "FRED T5YIE -> 5889 obs, first 2003-01-02:1.30 last 2026-07-17:2.27. 250 obs en 2013."
    url: "https://fred.stlouisfed.org/series/T5YIE"

  - nombre_interno: T5YIFR
    descripcion: "5-Year, 5-Year Forward Inflation Expectation Rate (diaria). Ancla de expectativas de largo plazo."
    fuente: fred
    id: "T5YIFR"
    auth: FRED_API_KEY
    inicio_verificado: "2003-01-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Expectativas de inflacion desancladas (subida) o deflacion (caida) = cambio de regimen macro."
    verificado: true
    evidencia: "FRED T5YIFR -> 5889 obs, first 2003-01-02:1.98 last 2026-07-17:2.21. 250 obs en 2013."
    url: "https://fred.stlouisfed.org/series/T5YIFR"

  # ================= FALLBACKS (yfinance / stooq) =================
  - nombre_interno: TNX_YF
    descripcion: "CBOE 10y Treasury yield (Yahoo). Fallback diario de DGS10, misma historia (1962)."
    fuente: yfinance
    id: "^TNX"
    auth: none
    inicio_verificado: "1962-01-02"
    granularidad: diaria
    pista: ambas
    rol: fallback
    relevancia_regimen: "Redundancia de fuente para el 10y si cae FRED; hoy cotiza el rendimiento directo (no x10)."
    verificado: true
    evidencia: "yf.download('^TNX','max') -> 16122 filas desde 1962-01-02, ultimo 2026-07-17=4.541. Coincide con DGS10."
    url: "https://finance.yahoo.com/quote/%5ETNX"

  - nombre_interno: IRX_YF
    descripcion: "CBOE 13-week T-bill yield (Yahoo). Fallback corto con historia mas larga que DGS3MO."
    fuente: yfinance
    id: "^IRX"
    auth: none
    inicio_verificado: "1960-01-04"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Tramo corto diario desde 1960 (2 anos antes que DGS3MO 1981); proxy de DTB3."
    verificado: true
    evidencia: "yf.download('^IRX','max') -> 16619 filas desde 1960-01-04, ultimo 2026-07-17=3.707."
    url: "https://finance.yahoo.com/quote/%5EIRX"

  - nombre_interno: TYX_YF
    descripcion: "CBOE 30y Treasury yield (Yahoo). Fallback de DGS30."
    fuente: yfinance
    id: "^TYX"
    auth: none
    inicio_verificado: "1977-02-15"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Redundancia del long bond; mismo inicio que DGS30 (1977-02-15)."
    verificado: true
    evidencia: "yf.download('^TYX','max') -> 12380 filas desde 1977-02-15, ultimo 2026-07-17=5.064."
    url: "https://finance.yahoo.com/quote/%5ETYX"

  - nombre_interno: FVX_YF
    descripcion: "CBOE 5y Treasury yield (Yahoo). Fallback de DGS5."
    fuente: yfinance
    id: "^FVX"
    auth: none
    inicio_verificado: "1962-01-02"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Redundancia del belly; mismo inicio que DGS5 (1962-01-02)."
    verificado: true
    evidencia: "yf.download('^FVX','max') -> 16122 filas desde 1962-01-02, ultimo 2026-07-17=4.273."
    url: "https://finance.yahoo.com/quote/%5EFVX"

  - nombre_interno: TNX_STOOQ
    descripcion: "10y yield via Stooq CSV como fallback terciario."
    fuente: stooq
    id: "10usy.b"
    auth: none
    inicio_verificado: null
    granularidad: diaria
    pista: ambas
    rol: fallback
    relevancia_regimen: "Redundancia adicional si caen FRED y yfinance."
    verificado: false
    evidencia: "BLOQUEADO: stooq.com/q/d/l/?s=10usy.b devuelve HTML noscript/challenge JS, no CSV. No usable sin navegador (igual que reporto el agente de Volatilidad)."
    url: "https://stooq.com/q/d/l/?s=10usy.b&i=d"

  # ================= NO VERIFICADO / DEEP HISTORY ACADEMICO =================
  - nombre_interno: SHILLER_LONGRATE
    descripcion: "Long-term interest rate (10y gov bond) mensual desde 1871, dataset de Robert Shiller."
    fuente: academico
    id: "ie_data.xls (col Long Interest Rate GS10)"
    auth: none
    inicio_verificado: null
    granularidad: mensual
    pista: A
    rol: fallback
    relevancia_regimen: "Extiende el 10y a 1871 para la espina historica profunda; empalma con GS10 (1953) y DGS10 (1962). NO descargado en esta pasada."
    verificado: false
    evidencia: "No verificado aqui (fichero Excel pesado). Bien documentado y estable; empalma limpio con GS10. Coordinar con el agente de espina/S&P500 que ya usa el dataset Shiller."
    url: "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
```

---

## Recomendación de priorización para el pipeline

1. **Curva núcleo diaria (imprescindible, ambas pistas):** `DGS3MO, DGS2, DGS5, DGS10, DGS30` +
   `DGS1` como vector de forma de curva. Descargar por **FRED** (limpio, API, vivo). `^TNX/^FVX/^TYX/^IRX`
   de yfinance como **fallback 1:1** (misma historia, 1960-1962).
2. **Pendientes (señal de régimen fuerte):** `T10Y2Y`, `T10Y3M` (nativas FRED). Derivar además
   curvatura `2·DGS5 − DGS2 − DGS10` y la dummy de inversión `T10Y3M<0`.
3. **Espina profunda (Pista A):** `DFF` (overnight diario 1954) + `DTB3` (3M diario 1954) para el
   tramo corto; `GS10/GS5/GS1` (mensual 1953) y `TB3MS` (3M mensual **1934**) para la curva de baja
   frecuencia. Pendiente profunda = `GS10 − TB3MS`. Opción de extender a 1871 con Shiller (academico,
   sin verificar aquí — coordinar con el agente de espina).
4. **Overnight/funding moderno (Pista B):** `SOFR` (2018), `EFFR` (2000), `IORB` (2021), `OBFR` (2016).
   Features de estrés: `SOFR − IORB`, `DTB3 − DFF` (miedo default, pico oct-2013).
5. **Reales/breakevens (régimen inflación):** `DFII10`, `T10YIE`, `T5YIFR` (todas 2003+).
6. **Punto ciego 2013 — resuelto:** la curva diaria es densa y sin agujeros a través de 2013
   (250 obs/año, 22 en oct-2013; único festivo 2013-10-14). Úsese como ancla para tapar cualquier
   hueco de 2013 en otros paneles. Manejar explícitamente los **huecos estructurales** que sí existen:
   `DGS20` 1987–1993 y `DGS30` 2002-02→2006-02 (imputar o excluir esos nodos en esos tramos).
