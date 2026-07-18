# Categoría: Fechas de crisis / recesiones — GROUND TRUTH (verificado)

Investigador: agente **Fechas de crisis / recesiones**. Rol: `validation`. Objetivo: fijar el
**ground truth de eventos** con el que se validará la detección de regímenes — es decir, (1) las
**recesiones NBER** (definición macro oficial) y probabilidades de recesión, y (2) una lista curada de
**CRISIS_WINDOWS de mercado** (peak→trough del S&P500) que **maximice el nº de eventos**, incluyendo las
crisis *puramente financieras* que NO fueron recesión (1987, 1998 LTCM, 2010 flash crash, 2011 downgrade,
2015-16, 2018, 2022, 2023 SVB). Estas etiquetas **no se usan como features** (sería fuga de información):
son la "respuesta" contra la que se mide el detector.

Todas las verificaciones se hicieron **de verdad** el **2026-07-18**:
- **NBER + probabilidades de recesión**: FRED API (clave `FRED_API_KEY` del `.env`, no impresa). Extraje
  los **intervalos pico→valle reales** de las series 0/1, no me fié de la doc.
- **CRISIS_WINDOWS**: las **derivé yo** descargando el **S&P500 diario real** (`yfinance ^GSPC`, 24.752
  filas 1927-12-30→2026-07-17) y computando drawdowns con dos algoritmos (peak-to-recovery y zigzag de
  swings). Reporto fechas y profundidades **observadas** en los datos, no de marketing.

> Solape declarado: el agente hermano `indices_estres.md` ya verificó USRECD/USREC/RECPRO/SAHM y el
> agente `historico_profundo.md` verificó el `crisisJST` de Jordà-Schularick-Taylor. Aquí **re-verifico
> de forma independiente** las series de recesión que son núcleo de mi mandato y **aporto lo que falta**:
> la lista dateada de CRISIS_WINDOWS de mercado derivada del propio S&P500.

---

## Resumen ejecutivo

**1) Recesiones NBER (ground truth macro oficial).** `USREC` (mensual) y `USRECD` (diario) arrancan en
1854 y están vivas. Extraje **12 recesiones desde 1948** (11 desde 1953) con sus intervalos exactos.
OJO metodológico crítico: la serie marca **1 desde el mes/día SIGUIENTE al pico** hasta el valle. Así,
`USREC=1` en 2008-01 significa que el **pico NBER real fue diciembre-2007**. Para etiquetar "inicio de
crisis" hay que **retroceder un periodo** el flanco de subida.

**2) Probabilidades / indicadores de recesión (ground truth probabilístico).** Verifiqué 4 series FRED
que dan una etiqueta *blanda* (0-1) ideal para comparar contra un HMM/Markov-switching propio:
`RECPROUSM156N` (Chauvet-Piger smoothed, 1967+), `SAHMREALTIME` (1959+), `SAHMCURRENT` (1949+) y
`JHDUSRGDPBR` (indicador de recesión basado en PIB de Hamilton, trimestral 1967+).

**3) CRISIS_WINDOWS de mercado (la aportación propia).** Las recesiones NBER **no bastan**: se pierden
las crisis financieras sin recesión (1987, 1998, 2010, 2011, 2015-16, 2018, 2022, 2025). Por eso derivé
del S&P500 diario una lista de **ventanas peak→trough**. Con umbral **≥10%** obtengo **~20 eventos desde
1950** (y 57 con el zigzag más granular), cubriendo de sobra la promesa de "10+ crisis" de la Pista A.

**4) Un aviso de honestidad importante sobre 2 métodos de drawdown:**
- *Peak-to-recovery* (episodio = desde un máximo histórico hasta que se recupera ese máximo): da los
  **grandes bear markets limpios** (GFC completa −56.8%, dotcom −49.1%), pero **enmascara** las crisis
  secundarias que ocurren estando aún bajo el agua de un pico mayor. Enmascara **2011** (dentro del
  periodo 2007→2013 sin nuevos máximos) y **1937-38** (dentro de 1929→1954).
- *Zigzag de swings* (usa **picos locales**): recupera 2010, 2011, 1937-38 y maximiza el nº de eventos.
  Uso este para las ventanas secundarias.

**5) SVB (marzo 2023) — matiz honesto.** SVB es una crisis bancaria/crédito de libro, pero el **drawdown
del índice amplio fue solo −7.8%** (pico 2023-02-02 → valle 2023-03-13). NO cruza el umbral del 10%. La
mayor caída de 2023 (−10.3%) fue en **octubre** (susto de tipos), no en SVB. La incluyo como ventana
**flagged "credit-event / sub-threshold"**: para capturarla como régimen hay que mirar spreads/estrés
bancario (KBW banks, OFR-FSI Funding), no el nivel del S&P500.

---

## Recesiones NBER — intervalos verificados (FRED)

`USREC` mensual (obs 1854-12→2026-06, upd 2026-07-01) y `USRECD` diario (→2026-07-05, upd 2026-07-06).
**12 recesiones desde 1948** (flanco de subida = mes siguiente al pico NBER):

| # | USREC (pico+1 → valle) | Pico NBER real | Nombre / causa | ¿bear equity asociado? |
|---|---|---|---|---|
| 1 | 1948-12 → 1949-10 | nov-1948 | post-guerra | −20.6% (1948-49) |
| 2 | 1953-08 → 1954-05 | jul-1953 | post-Corea | −14.8% |
| 3 | 1957-09 → 1958-04 | ago-1957 | recesión 1957-58 | −21.5% |
| 4 | 1960-05 → 1961-02 | abr-1960 | recesión 1960-61 | −14.0% |
| 5 | 1970-01 → 1970-11 | dic-1969 | recesión 1969-70 | −36.1% |
| 6 | 1973-12 → 1975-03 | nov-1973 | shock petróleo/estanflación | −48.2% |
| 7 | 1980-02 → 1980-07 | ene-1980 | shock Volcker I | −17.1% |
| 8 | 1981-08 → 1982-11 | jul-1981 | Volcker II (doble-dip) | −27.1% |
| 9 | 1990-08 → 1991-03 | jul-1990 | Guerra del Golfo / S&L | −19.9% |
| 10 | 2001-04 → 2001-11 | mar-2001 | pinchazo dotcom | (parte de −49.1%) |
| 11 | 2008-01 → 2009-06 | dic-2007 | Gran Crisis Financiera | −56.8% |
| 12 | 2020-03 → 2020-04 | feb-2020 | COVID-19 | −33.9% |

Evidencia: `USREC` → 2059 obs, 35 recesiones totales, 12 desde 1948. `USRECD` da los mismos flancos con
resolución diaria (62.674 obs). Recesiones **con equity leve** (1948, 1953, 1960) vs **crisis de mercado
sin recesión** (1987, 1998, 2010, 2011, 2015-16, 2018, 2022, 2025): por eso el ground truth de mercado
(abajo) es complementario, no redundante, con NBER.

---

## CRISIS_WINDOWS del S&P500 — derivadas y verificadas (`yfinance ^GSPC`)

Ventanas **peak→trough** del cierre del S&P500. `nber`=¿solapa recesión NBER? `tipo`: crash (caída
vertical), bear (≥20%), correction (10-20%), vol-spike / credit-event (flagged). Profundidad y fechas
**observadas** en los datos (2026-07-18). Método: peak-to-recovery para los grandes bears; zigzag de
picos locales para 2010/2011/1937-38.

### Desde 1950 (foco Pista A) — 20 eventos ≥ ~8%

| peak_date | trough_date | depth | nber | tipo | nombre |
|---|---|---|---|---|---|
| 1956-08-02 | 1957-10-22 | −21.5% | sí | bear | recesión 1957-58 |
| 1961-12-12 | 1962-06-26 | −28.0% | no | bear | "Kennedy Slide" 1962 |
| 1966-02-09 | 1966-10-07 | −22.2% | no | bear | credit crunch 1966 |
| 1968-11-29 | 1970-05-26 | −36.1% | sí | bear | bear 1969-70 |
| 1973-01-11 | 1974-10-03 | −48.2% | sí | bear | shock petróleo / estanflación |
| 1980-11-28 | 1982-08-12 | −27.1% | sí | bear | Volcker / doble-dip 1980-82 |
| 1987-08-25 | 1987-12-04 | −33.5% | **no** | crash | **Black Monday** (19-oct-1987 = −20.5% en 1 día) |
| 1990-07-16 | 1990-10-11 | −19.9% | sí | correction | Guerra del Golfo / S&L |
| 1998-07-17 | 1998-08-31 | −19.3% | **no** | correction | **LTCM / default Rusia** |
| 2000-03-24 | 2002-10-09 | −49.1% | sí | bear | pinchazo dotcom |
| 2007-10-09 | 2009-03-09 | −56.8% | sí | bear | **Gran Crisis Financiera** |
| 2010-04-23 | 2010-07-02 | −16.0% | **no** | correction | **Flash Crash** (6-may-2010) / Euro I |
| 2011-04-29 | 2011-10-03 | −19.4% | **no** | correction | **rebaja rating US / crisis euro II** |
| 2015-05-21 | 2016-02-11 | −14.2% | **no** | correction | devaluación yuan / crash petróleo |
| 2018-01-26 | 2018-02-08 | −10.2% | **no** | vol-spike | **"Volmageddon"** (colapso XIV/VIX) |
| 2018-09-20 | 2018-12-24 | −19.8% | **no** | correction | Q4-2018 / endurecimiento Fed |
| 2020-02-19 | 2020-03-23 | −33.9% | sí | crash | **COVID-19** (16-mar = −12.0% en 1 día) |
| 2022-01-03 | 2022-10-12 | −25.4% | **no** | bear | inflación / subidas Fed |
| 2023-02-02 | 2023-03-13 | −7.8% | **no** | credit-event* | **SVB / Signature** (*sub-umbral en índice*) |
| 2025-02-19 | 2025-04-08 | −18.9% | **no** | correction | selloff aranceles 2025 |

\* SVB: el índice solo cayó −7.8%; es crisis por estrés bancario, no por drawdown del S&P500. Flagged.

### Pre-1950 (profundidad histórica, `^GSPC` diario 1927+ y Shiller mensual 1871+)

| peak_date | trough_date | depth | fuente | nombre |
|---|---|---|---|---|
| 1929-09-16 | 1932-06-01 | −86.2% | GSPC diario | Gran Crash / Depresión |
| 1937-03-10 | 1938-03-31 | −54.5% | GSPC diario | recesión 1937-38 |
| 1919/1921 | | −37.4% | Shiller mensual | depresión 1920-21 (mensual) |
| 1906-09 | 1907-11 | −37.7% | Shiller mensual | Pánico de 1907 |
| 1881-1896 | | −42.1% | Shiller mensual | Larga Depresión (1893) |
| 1872-1877 | | −47.3% | Shiller mensual | pánico de 1873 |

Notas de reproducibilidad:
- Con **umbral ≥10% peak-to-recovery** salen **26 episodios** (1927+). Con **zigzag de swings ≥10%**
  salen **103** (57 desde 1950) — útil si se quiere el máximo de eventos, incluidas correcciones menores.
- La GFC como *un solo* bear = pico 2007-10-09 → valle 2009-03-09 = **−56.8%** (recuperó 2013-03-28).
  El zigzag la parte en varias piernas (−18.6%, −37.0%, −25.2%, −27.6%); ambas vistas son válidas.
- **Umbrales recomendados para CRISIS_WINDOWS**: `bear` si depth ≤ −20%, `correction` si −20% < depth
  ≤ −10%. Para banda de crisis extendida, tomar `[peak_date, recovery_date]`; para "núcleo de estrés",
  `[peak_date, trough_date]`.

---

## Series (formato catálogo)

```yaml
series_crisis_labels:
  # ================= GROUND TRUTH MACRO: RECESIONES NBER =================
  - nombre_interno: NBER_RECESSION_MONTHLY
    descripcion: "Indicador de recesión NBER, mensual (1=del mes siguiente al pico hasta el valle, 0=expansión). Etiqueta binaria macro oficial de recesión US."
    fuente: fred
    id: "USREC"
    auth: FRED_API_KEY
    inicio_verificado: "1854-12-01"
    granularidad: mensual
    pista: validacion
    rol: validation
    relevancia_regimen: "Ground truth canónico de recesión. 12 recesiones desde 1948. OJO: el 1 empieza el mes SIGUIENTE al pico NBER (2008-01=pico dic-2007) -> retroceder 1 periodo para marcar inicio."
    verificado: true
    evidencia: "FRED USREC -> 2059 obs, first 1854-12-01=1, last 2026-06-01=0 (upd 2026-07-01). Extraidos 35 intervalos pico->valle, 12 desde 1948 (1948-12->1949-10 ... 2020-03->2020-04)."
    url: "https://fred.stlouisfed.org/series/USREC"

  - nombre_interno: NBER_RECESSION_DAILY
    descripcion: "Indicador de recesión NBER, diario 7-day (1/0). Versión diaria de USREC, alineable con precios del S&P500."
    fuente: fred
    id: "USRECD"
    auth: FRED_API_KEY
    inicio_verificado: "1854-12-01"
    granularidad: diaria
    pista: validacion
    rol: validation
    relevancia_regimen: "Ground truth binario diario para casar con features/precios diarios. Mismos 12 intervalos que USREC con resolución de día."
    verificado: true
    evidencia: "FRED USRECD -> 62674 obs, first 1854-12-01=1, last 2026-07-05=0 (upd 2026-07-06). 12 intervalos desde 1948 con fechas fin de mes (p.ej. 2008-01-01->2009-06-30)."
    url: "https://fred.stlouisfed.org/series/USRECD"

  # ================= GROUND TRUTH PROBABILÍSTICO DE RECESIÓN =================
  - nombre_interno: RECESSION_PROB_SMOOTHED
    descripcion: "Smoothed U.S. Recession Probabilities (Chauvet-Piger, modelo dinámico-factor Markov-switching). Probabilidad [0,1] mensual de estar en recesión."
    fuente: fred
    id: "RECPROUSM156N"
    auth: FRED_API_KEY
    inicio_verificado: "1967-06-01"
    granularidad: mensual
    pista: validacion
    rol: validation
    relevancia_regimen: "Etiqueta BLANDA (0-1) ideal para comparar contra la prob. de régimen de un HMM/Markov-switching propio (mismo tipo de salida). Complementa el 0/1 de NBER."
    verificado: true
    evidencia: "FRED RECPROUSM156N -> 708 obs, first 1967-06-01=0.92, last 2026-05-01=0.54 (upd 2026-07-01)."
    url: "https://fred.stlouisfed.org/series/RECPROUSM156N"

  - nombre_interno: SAHM_RULE_REALTIME
    descripcion: "Real-time Sahm Rule Recession Indicator (basado en desempleo, datos vintage). Dispara cuando la media 3m del paro sube >=0.5pp sobre su mínimo de 12m."
    fuente: fred
    id: "SAHMREALTIME"
    auth: FRED_API_KEY
    inicio_verificado: "1959-12-01"
    granularidad: mensual
    pista: validacion
    rol: validation
    relevancia_regimen: "Indicador de INICIO de recesión de baja latencia (no revisado a posteriori); ground truth de tiempo real complementario al NBER (que se publica con retraso de meses)."
    verificado: true
    evidencia: "FRED SAHMREALTIME -> 798 obs, first 1959-12-01=0.77, last 2026-06-01=0.07 (upd 2026-07-02)."
    url: "https://fred.stlouisfed.org/series/SAHMREALTIME"

  - nombre_interno: SAHM_RULE_CURRENT
    descripcion: "Sahm Rule Recession Indicator (versión revisada con datos actuales, no vintage). Mismo umbral 0.5pp."
    fuente: fred
    id: "SAHMCURRENT"
    auth: FRED_API_KEY
    inicio_verificado: "1949-03-01"
    granularidad: mensual
    pista: validacion
    rol: validation
    relevancia_regimen: "Versión con historia más larga (1949+) del indicador Sahm; útil para backtest histórico del ground truth de recesión. Usa datos revisados (no simula tiempo real)."
    verificado: true
    evidencia: "FRED SAHMCURRENT -> 928 obs, first 1949-03-01=1.10, last 2026-06-01=0.07 (upd 2026-07-02)."
    url: "https://fred.stlouisfed.org/series/SAHMCURRENT"

  - nombre_interno: GDP_RECESSION_HAMILTON
    descripcion: "Dates of U.S. recessions inferidas por el indicador de recesión basado en PIB de James Hamilton. Probabilidad trimestral [0,1]."
    fuente: fred
    id: "JHDUSRGDPBR"
    auth: FRED_API_KEY
    inicio_verificado: "1967-10-01"
    granularidad: trimestral
    pista: validacion
    rol: validation
    relevancia_regimen: "Segundo/tercer voto de recesión basado en PIB (no en la datación por comité NBER); triangula el ground truth macro. Trimestral -> baja frecuencia."
    verificado: true
    evidencia: "FRED JHDUSRGDPBR -> 233 obs trimestrales, first 1967-10-01=0.00, last 2025-10-01=0.00 (upd 2026-04-30)."
    url: "https://fred.stlouisfed.org/series/JHDUSRGDPBR"

  # ================= ARTEFACTO DERIVADO: CRISIS_WINDOWS DE MERCADO =================
  - nombre_interno: CRISIS_WINDOWS_SP500
    descripcion: "Lista curada de ventanas peak->trough del S&P500 (crisis de mercado), derivada por mí del cierre diario de ^GSPC. ~20 eventos desde 1950, incluye crisis financieras SIN recesión (1987, 1998, 2010, 2011, 2015-16, 2018, 2022, 2025). Es el ground truth de EVENTOS de mercado."
    fuente: yfinance
    id: "^GSPC -> drawdown windows (peak-to-recovery + zigzag)"
    auth: none
    inicio_verificado: "1927-12-30"
    granularidad: diaria
    pista: validacion
    rol: validation
    relevancia_regimen: "Ground truth de crisis de MERCADO (drawdown), complementario a NBER (crisis MACRO). Maximiza n de eventos: 26 episodios peak-to-recovery >=10% o 103 swings zigzag >=10% (57 desde 1950). Ver tablas de este documento para fechas exactas."
    verificado: true
    evidencia: "yf.download('^GSPC','max') -> 24752 filas 1927-12-30..2026-07-17. Drawdowns computados: 26 episodios peak-to-recovery >=10%; zigzag 10% -> 103 down-legs (57 desde 1950). Spot-checks OK: 1987-10-19=-20.5% 1dia; COVID 2020-03-16=-12.0%; SVB pico-valle solo -7.8%; 1937-38 -54.5%; GFC -56.8%."
    url: "https://finance.yahoo.com/quote/%5EGSPC"

  # ================= FUENTES DE PRECIO PARA DERIVAR LOS DRAWDOWNS =================
  - nombre_interno: SP500_DAILY_DRAWDOWN_SRC
    descripcion: "S&P500 cierre diario (yfinance ^GSPC), fuente primaria para derivar las CRISIS_WINDOWS de drawdown. (Serie de precio también cubierta por catálogos historico_profundo/volatilidad; aquí es el insumo del ground truth de eventos)."
    fuente: yfinance
    id: "^GSPC"
    auth: none
    inicio_verificado: "1927-12-30"
    granularidad: diaria
    pista: validacion
    rol: validation
    relevancia_regimen: "Precio diario más profundo y fiable gratis (1927+) para datar picos/valles de crisis con resolución de día. Stooq (^spx) hoy bloqueado por gate JS; yfinance es el primario robusto."
    verificado: true
    evidencia: "yf.download('^GSPC','max') -> 24752 filas, first 1927-12-30=17.66, last 2026-07-17=7457.69."
    url: "https://finance.yahoo.com/quote/%5EGSPC"

  - nombre_interno: SP500_MONTHLY_DEEP_SRC
    descripcion: "S&P500 mensual desde 1871 (mirror GitHub datasets/s-and-p-500, derivado de Shiller). Fuente para datar crisis PRE-1927 (1873, 1893, 1907, 1920-21) que ^GSPC no alcanza."
    fuente: github
    id: "datasets/s-and-p-500 -> data/data.csv (col SP500)"
    auth: none
    inicio_verificado: "1871-01-01"
    granularidad: mensual
    pista: validacion
    rol: fallback
    relevancia_regimen: "Extiende el ground truth de drawdowns a los pánicos del s.XIX/principios s.XX (mensual). Complementa ^GSPC para profundidad histórica máxima."
    verificado: true
    evidencia: "pd.read_csv raw.githubusercontent -> 1866 filas, 1871-01-01=4.44 .. 2026-06-01=7450.03. Drawdowns >=20%: 13 episodios (1873 -47.3%, 1893 -42.1%, 1907 -37.7%, 1929 -84.8%...)."
    url: "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"

  # ================= CROSS-REF (verificado por agentes hermanos) =================
  - nombre_interno: JST_CRISIS_GROUNDTRUTH
    descripcion: "Columna crisisJST del dataset Jordà-Schularick-Taylor R6: dummy experto de crisis bancaria sistémica país-año (18 países, 1870-2020). Ground truth internacional de crisis financieras."
    fuente: academico
    id: "JSTdatasetR6.xlsx -> columna crisisJST"
    auth: none
    inicio_verificado: "1870"
    granularidad: anual
    pista: validacion
    rol: validation
    relevancia_regimen: "Lista experta más citada de crisis financieras sistémicas; ground truth laxo internacional para validar detección de regímenes de crisis a largo plazo. Anual -> mapear años-crisis a fechas."
    verificado: true
    evidencia: "CROSS-REF historico_profundo.md (agente hermano): JSTdatasetR6.xlsx 2718 filas, 1870-2020, 18 paises, col crisisJST verificada al parsear el xlsx. NO re-descargado aqui; confirmado leyendo su evidencia."
    url: "https://www.macrohistory.net/database/"

  # ================= NO VERIFICADO / FALLBACK =================
  - nombre_interno: SP500_STOOQ_FALLBACK
    descripcion: "S&P500 diario vía Stooq (^spx) como fuente de precio alternativa para derivar drawdowns."
    fuente: stooq
    id: "^spx"
    auth: none
    inicio_verificado: null
    granularidad: diaria
    pista: validacion
    rol: fallback
    relevancia_regimen: "Alternativa a yfinance si este falla; pero HOY el endpoint CSV de Stooq devuelve una pagina HTML con gate JS (anti-bot), no CSV. No usable programáticamente en esta pasada."
    verificado: false
    evidencia: "GET stooq.com/q/d/l/?s=^spx&i=d (UA navegador) -> HTTP 200 pero body es '<!DOCTYPE html>...<noscript>' (gate JS), NO el CSV Date,Open,High,Low,Close. Fallo confirmado para ^spx/^gspc/spx."
    url: "https://stooq.com/q/d/l/?s=%5Espx&i=d"
```

---

## CRISIS_WINDOWS en formato máquina (para el pipeline)

Listado dateado y verificado (mismas fechas que las tablas de arriba), directo para construir la máscara
de eventos. `depth` = drawdown peak→trough del cierre S&P500; `nber`=solapa recesión NBER.

```yaml
crisis_windows_sp500:
  # since 1950 (foco Pista A)
  - {name: "recesion_1957_58",     peak: "1956-08-02", trough: "1957-10-22", depth: -0.215, nber: true,  tipo: bear}
  - {name: "kennedy_slide_1962",   peak: "1961-12-12", trough: "1962-06-26", depth: -0.280, nber: false, tipo: bear}
  - {name: "credit_crunch_1966",   peak: "1966-02-09", trough: "1966-10-07", depth: -0.222, nber: false, tipo: bear}
  - {name: "bear_1969_70",         peak: "1968-11-29", trough: "1970-05-26", depth: -0.361, nber: true,  tipo: bear}
  - {name: "oil_stagflation_1973", peak: "1973-01-11", trough: "1974-10-03", depth: -0.482, nber: true,  tipo: bear}
  - {name: "volcker_1980_82",      peak: "1980-11-28", trough: "1982-08-12", depth: -0.271, nber: true,  tipo: bear}
  - {name: "black_monday_1987",    peak: "1987-08-25", trough: "1987-12-04", depth: -0.335, nber: false, tipo: crash}
  - {name: "gulf_war_sl_1990",     peak: "1990-07-16", trough: "1990-10-11", depth: -0.199, nber: true,  tipo: correction}
  - {name: "ltcm_russia_1998",     peak: "1998-07-17", trough: "1998-08-31", depth: -0.193, nber: false, tipo: correction}
  - {name: "dotcom_2000_02",       peak: "2000-03-24", trough: "2002-10-09", depth: -0.491, nber: true,  tipo: bear}
  - {name: "gfc_2007_09",          peak: "2007-10-09", trough: "2009-03-09", depth: -0.568, nber: true,  tipo: bear}
  - {name: "flash_crash_euro1_2010", peak: "2010-04-23", trough: "2010-07-02", depth: -0.160, nber: false, tipo: correction}
  - {name: "us_downgrade_euro2_2011", peak: "2011-04-29", trough: "2011-10-03", depth: -0.194, nber: false, tipo: correction}
  - {name: "china_oil_2015_16",    peak: "2015-05-21", trough: "2016-02-11", depth: -0.142, nber: false, tipo: correction}
  - {name: "volmageddon_2018q1",   peak: "2018-01-26", trough: "2018-02-08", depth: -0.102, nber: false, tipo: vol_spike}
  - {name: "fed_tightening_2018q4",peak: "2018-09-20", trough: "2018-12-24", depth: -0.198, nber: false, tipo: correction}
  - {name: "covid_2020",           peak: "2020-02-19", trough: "2020-03-23", depth: -0.339, nber: true,  tipo: crash}
  - {name: "inflation_bear_2022",  peak: "2022-01-03", trough: "2022-10-12", depth: -0.254, nber: false, tipo: bear}
  - {name: "svb_banking_2023",     peak: "2023-02-02", trough: "2023-03-13", depth: -0.078, nber: false, tipo: credit_event, flag: sub_threshold_index}
  - {name: "tariff_selloff_2025",  peak: "2025-02-19", trough: "2025-04-08", depth: -0.189, nber: false, tipo: correction}
  # pre-1950 (profundidad; 1929/1937 diarios, resto mensual Shiller)
  - {name: "great_crash_depresion_1929", peak: "1929-09-16", trough: "1932-06-01", depth: -0.862, nber: true, tipo: crash}
  - {name: "recesion_1937_38",     peak: "1937-03-10", trough: "1938-03-31", depth: -0.545, nber: true,  tipo: bear}
```

---

## Recomendación de uso para validación

1. **Etiqueta macro (recesión):** `USRECD` (diario 1854+) como máscara binaria; recordar retroceder un
   periodo el flanco de subida (el 1 empieza el mes/día siguiente al pico NBER). `RECPROUSM156N` como
   versión probabilística para comparar contra la prob. de régimen de un HMM.
2. **Etiqueta de mercado (crisis de drawdown):** `CRISIS_WINDOWS_SP500` (tabla dateada arriba). Es la que
   maximiza el nº de eventos y captura las crisis financieras sin recesión — clave para que el detector
   no se limite a replicar NBER.
3. **Doble ground truth = mejor validación.** Un régimen "crisis" ideal debería activarse tanto en
   ventanas NBER como en CRISIS_WINDOWS; la unión de ambos da recall alto, la intersección da precisión.
4. **SVB y crisis "solo-crédito":** no esperar señal en el drawdown del S&P500; validar contra estrés
   bancario/crédito (OFR-FSI Funding, KBW banks, spreads) del catálogo `credito`/`indices_estres`.
5. **No usar como features.** Estas etiquetas son la respuesta; usarlas de input = fuga de información.
6. **Frecuencias mixtas:** USRECD/GSPC diarios; USREC/RECPRO/SAHM mensuales; JHDUSRGDPBR/JST trimestral/
   anual. Reindexar a diario con **forward-fill causal** (nunca interpolar hacia atrás).

## Huecos / dudas honestas

- **Doble contabilidad peak-to-recovery vs zigzag.** El método peak-to-recovery enmascara 2011 y 1937-38
  (crisis "bajo el agua" de un pico mayor); el zigzag las recupera pero fragmenta los grandes bears (GFC
  en 4 piernas). Publiqué **ambas vistas**; las ventanas de la tabla combinan lo mejor de cada una, pero
  la elección de umbral (10% vs 20%) cambia el nº de eventos (26 vs ~9 peak-to-recovery; 103 vs ~30 zigzag).
- **SVB (2023) queda sub-umbral** en el índice amplio (−7.8%). Es un evento de crédito/banca, no un crash
  del S&P500. La incluyo flagged; capturarla como régimen requiere features de estrés bancario.
- **NBER se publica con retraso** (la datación oficial de un pico llega meses/años después). Para tiempo
  real usar `SAHMREALTIME`/`RECPROUSM156N` como proxies de baja latencia, no `USREC` (que se revisa).
- **Stooq (^spx) hoy inaccesible** por gate JS anti-bot; no pude verificar su CSV. yfinance ^GSPC es el
  primario robusto verificado; si en el futuro se necesita redundancia, probar Stooq con navegador headless.
- **Fechas de crisis = convención, no verdad absoluta.** El "peak" y "trough" dependen de umbral y de si
  se usa cierre o intradía (uso cierre). Las fechas NBER son de comité; las de drawdown son mecánicas.
  Trátese como **ground truth LAXO** (para recall/precision aproximados), no como etiqueta exacta.
- **JST crisisJST NO re-descargado aquí** (lo verificó el agente hermano `historico_profundo.md`); lo
  incluyo como cross-ref honesto. Es anual y acaba en 2020.

## Fuentes y métodos de verificación
- **FRED API** (`series` + `series/observations`, JSON), clave `FRED_API_KEY` del `.env` (no impresa).
  Verificadas 6 series (USREC, USRECD, RECPROUSM156N, SAHMREALTIME, SAHMCURRENT, JHDUSRGDPBR) con n,
  primer/último dato, `last_updated` e **intervalos pico→valle extraídos**.
- **yfinance** `^GSPC` period='max' (24.752 filas 1927-2026); drawdowns derivados con dos algoritmos
  (peak-to-recovery y zigzag de swings) + spot-checks de ventanas conocidas.
- **GitHub** mirror Shiller `datasets/s-and-p-500` (mensual 1871+) para pánicos pre-1927.
- **Stooq** ^spx: intento fallido (gate JS), documentado como fallback no verificado.
- Fecha de verificación: **2026-07-18**.
