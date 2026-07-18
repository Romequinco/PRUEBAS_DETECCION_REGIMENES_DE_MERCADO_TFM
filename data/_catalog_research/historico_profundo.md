# Categoría: Históricos académicos profundos (verificado)

Investigador: agente **Histórico Profundo**. Foco: las cuatro fuentes académicas clásicas de máximo
histórico y coste cero — **Shiller** (S&P500 + CAPE mensual desde 1871), **Ken French / Data Library
de Dartmouth** (factores diarios Mkt-RF desde 1926), **Goyal-Welch** (equity premium y predictores
mensuales desde 1871) y **Jòrda-Schularick-Taylor Macrohistory** (macro + crisis anual desde 1870,
18 países). Son la **espina profunda de la Pista A** (maximizar número de crisis) y, en el caso de
JST, una fuente directa de **ground truth de crisis** para validación.

Todas las verificaciones se hicieron **de verdad** el 2026-07-18: descarga real vía `requests`,
descompresión de los .zip de Ken French, parseo de los .xls/.xlsx y conteo de filas + fecha de inicio
**observada** (no la de marketing). No se usó `FRED_API_KEY` aquí (estas fuentes son ficheros de
webs académicas / GitHub / Google Sheets, sin auth). Reporto el nº de filas y primera/última obs reales.

---

## Resumen ejecutivo de lo verificado

**La joya de la Pista A es Ken French.** El fichero `F-F_Research_Data_Factors_daily` (Dartmouth)
da el **exceso de retorno del mercado US (Mkt-RF) DIARIO desde 1926-07-01** — 26.253 filas
verificadas hasta 2026-05-29. Trae también SMB, HML y **RF (risk-free diario)** en el mismo CSV.
Compuesto (Mkt-RF + RF) reconstruye un índice de retorno total del mercado US diario desde 1926, que
es **~1.5 años más profundo y más limpio** que `^GSPC` de yfinance (1927-12) y sin huecos de splits.
Es la serie diaria gratis con más historia que existe: cubre 1929, 1937, 1987, etc.

**El histórico mensual profundo tiene tres fuentes que se solapan y se refuerzan:**
- **Shiller `ie_data.xls`** (Yale): S&P500 + dividendos + earnings + CPI + GS10 + **CAPE** + **TR CAPE**
  mensual desde **1871-01**. 1.834 filas. OJO: la copia de Yale que descargué **acaba en 2023-09**
  (parece ir con retraso); es la fuente canónica académica (única con TR CAPE) pero para *frescura*
  conviene el mirror de GitHub.
- **Mirror GitHub `datasets/s-and-p-500`** (derivado de Shiller): SP500 + Dividend + Earnings + CPI +
  Long rate + **PE10 (=CAPE)** mensual desde **1871-01-01 hasta 2026-06-01** (1.866 filas). CSV limpio,
  parseable de una línea, **actualizado**. Es el mejor primario programático para la espina mensual
  S&P500+CAPE.
- **Goyal-Welch** (Google Sheet oficial de Amit Goyal): mensual **1871-01 → 2025-12** (1.860 filas)
  con Index, D12 (div 12m), E12 (earnings 12m), b/m, **tbl** (T-bill), **AAA/BAA** (corp yields),
  **lty** (long yield), **ntis** (net equity issuance), Rfree, **infl**, **ltr/corpr** (bond returns).
  De aquí salen *gratis y mensuales desde el s.XIX* el **term spread (lty−tbl)**, el **default spread
  (BAA−AAA)** y el dividend/earnings yield — todos features de régimen de primer orden.

**JST Macrohistory (R6)** es anual (1870-2020, 18 países, 2.718 filas, 59 columnas) → no sirve como
feature diaria, pero es **oro para validación**: la columna **`crisisJST`** es una lista experta de
**crisis bancarias sistémicas** país-año, el ground-truth de crisis más citado en la literatura.
Además trae retornos totales de equity/bonos/vivienda, crédito (`tloans`), tipos corto/largo y
deuda/PIB — contexto macro de máximo histórico para etiquetar regímenes de largo plazo.

**Notas de acceso / fricciones:**
- El antiguo host de Goyal (`www.hec.unil.ch/agoyal/...`) está **muerto** (ConnectTimeout). La fuente
  viva es su Google Sheet; se baja como xlsx con `.../export?format=xlsx` (verificado, 3 hojas
  Monthly/Quarterly/Annual).
- Ken French también es accesible con `pandas_datareader` (`web.DataReader('F-F_Research_Data_Factors_daily','famafrench')`),
  pero el **.zip directo de Dartmouth es más robusto** y da el histórico completo de una.
- Todas estas fuentes son **auth=none**; ninguna necesita la FRED key.

---

## Detalle por fuente (evidencia)

### 1. Ken French Data Library (Dartmouth) — espina diaria Pista A
- `F-F_Research_Data_Factors_daily_CSV.zip` → CSV con `Date(YYYYMMDD),Mkt-RF,SMB,HML,RF`.
  **26.253 filas, primera 1926-07-01, última 2026-05-29.** (`0.09,-0.25,-0.27` la primera fila).
- `F-F_Momentum_Factor_daily_CSV.zip` → WML/UMD diario, 26.152 filas, **1926-11-03 → 2026-05-29**.
- `F-F_Research_Data_5_Factors_2x3_daily_CSV.zip` → +RMW,CMA, 15.833 filas, **1963-07-01 → 2026-05-29**.
- `F-F_Research_Data_Factors_CSV.zip` → 3 factores **mensual**, 1.199 filas, **1926-07 → 2026-05**.
- Interpretación régimen: Mkt-RF = drawdowns/vol del mercado desde 1926; rotación SMB/HML y **momentum
  crashes** (UMD muy negativo) son marcadores clásicos de cambio de régimen (2009-03, 2020-03…).

### 2. Robert Shiller `ie_data.xls` (Yale) — S&P500 + CAPE mensual desde 1871
- Hoja `Data`, 1.834 filas, Date fraccional **1871.01 → 2023.09** (la copia de Yale va con retraso).
- Columnas clave: `P` (precio S&P), `D`, `E`, `CPI`, `Rate GS10`, `CAPE`, `TR CAPE` (única fuente con
  el CAPE de retorno total). Perfecta para valoración/régimen secular; **usar mirror GitHub para lo reciente**.

### 3. Mirror GitHub `datasets/s-and-p-500` — Shiller limpio y actualizado
- `data/data.csv`, 1.866 filas, **1871-01-01 → 2026-06-01**. Columnas: `SP500, Dividend, Earnings,
  Consumer Price Index, Long Interest Rate, Real Price, Real Dividend, Real Earnings, PE10`.
- `PE10` = CAPE. CSV plano, ideal como **primario mensual programático** del S&P500 + CAPE.

### 4. Goyal-Welch (Amit Goyal, Google Sheet) — predictores del equity premium desde 1871
- Export xlsx del Sheet `1qwpl2R_DNujpU5YUkk8lacP1tTeMb9iJ`, 3 hojas.
  - **Monthly**: 1.860 filas, `yyyymm` **187101 → 202512**. Cols: Index, D12, E12, b/m, tbl, AAA, BAA,
    lty, ntis, Rfree, infl, ltr, corpr, svar.
  - **Quarterly**: 620 filas, **1871Q1 → 2025Q4**, añade `cay` (consumption-wealth ratio).
  - **Annual**: 155 filas, **1871 → 2025**, añade `eqis`.
- De aquí: term spread `lty−tbl`, default spread `BAA−AAA`, dividend yield `D12/Index`, **svar**
  (varianza realizada mensual del stock market) — todos mensuales y profundísimos.

### 5. Jòrda-Schularick-Taylor Macrohistory Database R6 — crisis + macro anual desde 1870
- `JSTdatasetR6.xlsx` (macrohistory.net), 2.718 filas, 59 cols, **year 1870 → 2020**, **18 países**
  (Australia, Bélgica, Canadá, Dinamarca, Finlandia, Francia, Alemania, Irlanda, Italia, Japón…).
- Columna **`crisisJST`** = dummy de crisis bancaria sistémica (ground truth). Además `eq_tr, bond_tr,
  housing_tr, tloans, stir, ltrate, debtgdp, cpi, xrusd`. Anual → validación/etiquetado, no feature diaria.

---

```yaml
- nombre_interno: FF_FACTORS_3_DAILY
  descripcion: "Fama-French 3 factores DIARIOS (Mkt-RF, SMB, HML) + RF risk-free diario, mercado US (CRSP value-weighted)."
  fuente: academico
  id: "F-F_Research_Data_Factors_daily"
  auth: none
  inicio_verificado: "1926-07-01"
  granularidad: diaria
  pista: A
  rol: spine
  relevancia_regimen: "Espina diaria de máxima profundidad: exceso de retorno del mercado US desde 1926; drawdowns y vol de todas las crisis (1929, 1937, 1987, 2008, 2020). Compuesto (Mkt-RF+RF) da índice total-return diario más limpio y profundo que ^GSPC."
  verificado: true
  evidencia: "requests+unzip -> F-F_Research_Data_Factors_daily.csv: 26253 filas, primera 19260701 (0.09,-0.25,-0.27), última 20260529"
  url: "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"

- nombre_interno: FF_MOM_DAILY
  descripcion: "Fama-French momentum factor (WML/UMD) DIARIO, mercado US."
  fuente: academico
  id: "F-F_Momentum_Factor_daily"
  auth: none
  inicio_verificado: "1926-11-03"
  granularidad: diaria
  pista: A
  rol: enricher
  relevancia_regimen: "Momentum crashes (UMD fuertemente negativo) son marcadores clásicos de giro de régimen (2009-03, 2020-03). Complementa Mkt-RF."
  verificado: true
  evidencia: "requests+unzip -> F-F_Momentum_Factor_daily.csv: 26152 filas, primera 19261103, última 20260529"
  url: "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

- nombre_interno: FF_FACTORS_5_DAILY
  descripcion: "Fama-French 5 factores DIARIOS (Mkt-RF, SMB, HML, RMW, CMA) + RF, mercado US."
  fuente: academico
  id: "F-F_Research_Data_5_Factors_2x3_daily"
  auth: none
  inicio_verificado: "1963-07-01"
  granularidad: diaria
  pista: A
  rol: enricher
  relevancia_regimen: "Rotación de estilos (value/quality/investment) informa risk-on/off; RMW y CMA distinguen regímenes de flight-to-quality. Menos profundo (1963) que el 3-factor."
  verificado: true
  evidencia: "requests+unzip -> F-F_Research_Data_5_Factors_2x3_daily.csv: 15833 filas, primera 19630701, última 20260529"
  url: "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"

- nombre_interno: FF_FACTORS_3_MONTHLY
  descripcion: "Fama-French 3 factores MENSUALES (Mkt-RF, SMB, HML) + RF, mercado US."
  fuente: academico
  id: "F-F_Research_Data_Factors"
  auth: none
  inicio_verificado: "1926-07"
  granularidad: mensual
  pista: A
  rol: fallback
  relevancia_regimen: "Version mensual del mismo panel factorial; util como remuestreo/fallback mensual del spine diario y para alinear con features mensuales (Shiller/Goyal-Welch)."
  verificado: true
  evidencia: "requests+unzip -> F-F_Research_Data_Factors.csv: 1199 filas mensuales, primera 192607, última 202605"
  url: "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"

- nombre_interno: SHILLER_SP500_CAPE_GH
  descripcion: "S&P500 mensual + CAPE (PE10) + Dividend/Earnings/CPI/Long-rate. Mirror GitHub limpio y ACTUALIZADO de la serie de Shiller."
  fuente: github
  id: "datasets/s-and-p-500 -> data/data.csv"
  auth: none
  inicio_verificado: "1871-01-01"
  granularidad: mensual
  pista: A
  rol: spine
  relevancia_regimen: "Espina mensual profunda: nivel S&P500 desde 1871 + CAPE (valoración) para régimen secular (burbujas/reversiones). CSV plano y vivo hasta 2026-06, mejor primario programático que el xls de Yale."
  verificado: true
  evidencia: "pd.read_csv raw.githubusercontent -> shape (1866,10), cols incl PE10; primera 1871-01-01, última 2026-06-01"
  url: "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"

- nombre_interno: SHILLER_IE_XLS
  descripcion: "Robert Shiller ie_data.xls (Yale): S&P500, Dividend, Earnings, CPI, GS10, CAPE y TR CAPE mensual. Fuente académica canónica."
  fuente: academico
  id: "ie_data.xls (hoja Data)"
  auth: none
  inicio_verificado: "1871-01"
  granularidad: mensual
  pista: A
  rol: core
  relevancia_regimen: "Fuente canónica del CAPE; UNICA con TR CAPE (total-return CAPE). Valoracion secular para régimen de largo plazo. OJO: la copia de Yale descargada acaba en 2023-09 (va con retraso); usar mirror GitHub para frescura."
  verificado: true
  evidencia: "pd.read_excel(header=7) hoja Data: 1834 filas, Date fraccional 1871.01 -> 2023.09; cols incl CAPE y 'TR CAPE'"
  url: "http://www.econ.yale.edu/~shiller/data/ie_data.xls"

- nombre_interno: GW_PREDICTORS_MONTHLY
  descripcion: "Goyal-Welch predictores del equity premium MENSUALES: Index, D12, E12, b/m, tbl, AAA, BAA, lty, ntis, Rfree, infl, ltr, corpr, svar."
  fuente: academico
  id: "GoogleSheet 1qwpl2R_DNujpU5YUkk8lacP1tTeMb9iJ (hoja Monthly)"
  auth: none
  inicio_verificado: "1871-01"
  granularidad: mensual
  pista: A
  rol: core
  relevancia_regimen: "De aquí salen mensuales y desde 1871: term spread (lty-tbl), default/credit spread (BAA-AAA), dividend yield (D12/Index), earnings yield y svar (varianza realizada mensual). Features de régimen de primer orden con máximo histórico."
  verificado: true
  evidencia: "requests -> export?format=xlsx, hoja Monthly: shape (1860,18), yyyymm 187101 -> 202512"
  url: "https://docs.google.com/spreadsheets/d/1qwpl2R_DNujpU5YUkk8lacP1tTeMb9iJ/export?format=xlsx"

- nombre_interno: GW_PREDICTORS_QUARTERLY
  descripcion: "Goyal-Welch predictores TRIMESTRALES; añade cay (consumption-wealth ratio) a los mensuales."
  fuente: academico
  id: "GoogleSheet 1qwpl2R_DNujpU5YUkk8lacP1tTeMb9iJ (hoja Quarterly)"
  auth: none
  inicio_verificado: "1871Q1"
  granularidad: trimestral
  pista: A
  rol: enricher
  relevancia_regimen: "cay (desviación consumo-riqueza) es predictor macro-financiero de régimen; solo disponible trimestral. Complemento de baja frecuencia."
  verificado: true
  evidencia: "requests -> export?format=xlsx, hoja Quarterly: shape (620,22), yyyyq 18711 -> 20254, incluye 'cay'"
  url: "https://docs.google.com/spreadsheets/d/1qwpl2R_DNujpU5YUkk8lacP1tTeMb9iJ/export?format=xlsx"

- nombre_interno: JST_MACROHISTORY_R6
  descripcion: "Jorda-Schularick-Taylor Macrohistory Database R6: panel anual 18 países, 59 vars (equity/bond/housing total return, crédito, tipos, deuda/PIB, CPI, FX)."
  fuente: academico
  id: "JSTdatasetR6.xlsx"
  auth: none
  inicio_verificado: "1870"
  granularidad: anual
  pista: ambas
  rol: enricher
  relevancia_regimen: "Contexto macro de máximo histórico (1870+) multi-país: retornos totales equity/bonos/vivienda, crédito bancario (tloans), tipos corto/largo, deuda/PIB. Anual -> etiquetado de régimen secular, no feature diaria."
  verificado: true
  evidencia: "requests+pd.read_excel: shape (2718,59), year 1870->2020, 18 países; cols incl eq_tr, bond_tr, housing_tr, tloans, stir, ltrate, crisisJST"
  url: "https://www.macrohistory.net/app/download/9834512569/JSTdatasetR6.xlsx"

- nombre_interno: JST_CRISIS_GROUNDTRUTH
  descripcion: "Columna crisisJST del dataset JST R6: dummy experto de crisis bancaria sistémica país-año, 1870-2020, 18 países. Ground truth de crisis."
  fuente: academico
  id: "JSTdatasetR6.xlsx -> columna crisisJST"
  auth: none
  inicio_verificado: "1870"
  granularidad: anual
  pista: validacion
  rol: validation
  relevancia_regimen: "Lista experta más citada de crisis financieras sistémicas; ground truth laxo para validar detección de regímenes de crisis a largo plazo (internacional). Anual, requiere mapear a fechas."
  verificado: true
  evidencia: "columna 'crisisJST' presente en JSTdatasetR6.xlsx (2718 filas, 1870-2020, 18 países) verificada al parsear el xlsx"
  url: "https://www.macrohistory.net/database/"
```

---

## Huecos / dudas honestas
- **Shiller xls va con retraso** (última obs 2023-09 en la copia de Yale al descargar hoy). Mitigado
  con el mirror GitHub (`datasets/s-and-p-500`, actual a 2026-06) y con Goyal-Welch Monthly (a 2025-12),
  que son fuentes vivas de la misma serie subyacente.
- **JST es anual** y acaba en 2020 (R6). Es ground-truth/contexto secular, NO una feature diaria; hay
  que mapear los años-crisis a fechas para casarlo con la Pista A. Puede salir R7 en el futuro.
- **Goyal-Welch depende de un Google Sheet** (host de Amit Goyal); el antiguo host institucional
  (hec.unil.ch) está caído. El endpoint `export?format=xlsx` funciona hoy pero es menos estable que un
  .zip académico; conviene cachear el xlsx localmente al bajarlo.
- **Ken French Mkt-RF es un retorno (exceso), no un índice de precio**: para un "S&P500 diario 1926+"
  hay que componer (1+Mkt-RF+RF) a índice; es un proxy CRSP value-weighted, no el S&P500 literal
  (que como índice diario solo arranca ~1927 en yfinance). Para régimen (vol/drawdown) es equivalente
  y superior en profundidad/limpieza.
- No dupliqué series puramente-FRED (BAA/AAA Moody's, GS10, etc.): las cubre el agente `macro`/`credito`.
  Aquí AAA/BAA/lty/tbl vienen *dentro* de Goyal-Welch como paquete mensual profundo autocontenido.
