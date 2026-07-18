# Catálogo de datos: Equity + Breadth

**Categoría:** Equity + breadth (S&P500 máximo histórico, sectores GICS, amplitud, small/large, Nasdaq, Russell).
**Fecha de verificación:** 2026-07-18.
**Entorno:** Python 3.13.7, yfinance 0.2.66, pandas 2.3.3, FRED_API_KEY presente y funcional.

Todo lo marcado `verificado: true` se ha **descargado de verdad** en esta sesión y se reporta la
fecha de inicio y el nº de filas REALES observadas, no las de marketing.

---

## Resumen ejecutivo

**La columna vertebral histórica (Pista A) es sólida y muy profunda:**

- **S&P500 diario desde 1927-12-30** vía yfinance `^GSPC` (24.752 filas). Cubre el crash del 29,
  la Gran Depresión y todas las crisis modernas. Es el mejor spine diario gratuito.
- **S&P500 mensual desde 1871-01** vía Shiller (`ie_data.xls`), con precio, dividendo, earnings y CPI.
  Permite un ancla mensual de 150+ años (incluye pánicos de 1873, 1893, 1907).
- **Factor de mercado diario desde 1926-07-01** vía Ken French (`Mkt-RF`, 26.253 filas). Es la serie
  diaria de renta variable US más profunda y limpia que existe gratis, y trae **SMB (small-minus-big)
  y HML (value)** en el mismo fichero → resuelve "small/large" y "value/growth" con 100 años de historia.

**Volatilidad de equity (muy relevante para régimen):** VIX diario desde 1990 (`^VIX` / FRED `VIXCLS`),
y **VXO desde 1986** (`^VXO` / FRED `VXOCLS`, la metodología antigua; dejó de actualizarse en 2021-09).
Empalmando VXO(1986-1990)+VIX(1990→) se obtiene un índice de miedo continuo desde 1986. Además `^SKEW`
(riesgo de cola, 1990+) y `^VXN` (Nasdaq, 2001+).

**Sectores GICS:** los 9 SPDR originales (XLK/XLF/XLE/XLV/XLI/XLY/XLP/XLU/XLB) desde **1998-12-22**;
XLRE desde 2015, XLC desde 2018. Para profundidad histórica de sectores, **Ken French 5/10/49 industry
portfolios diarios desde 1926** (verificado el de 5 industrias: 52.506 filas, 1926-2026).

**Small/large y estilo:** Russell 2000 `^RUT` (1987), Russell 3000 `^RUA` (1987), Russell 1000 `^RUI`
(1992), S&P MidCap 400 `^SP400` (1981, sorprendentemente profundo), S&P SmallCap 600 `^SP600` (1989),
más SMB/HML de French (1926). ETFs de estilo IWM/IWD/IWF y SPYG/SPYV desde 2000.

**Índices amplios / total market:** NYSE Composite `^NYA` desde **1965** (15.235 filas), Wilshire 5000
`^W5000` desde 1989 (Yahoo lo sigue sirviendo aunque FRED retiró las series Wilshire), Nasdaq Composite
`^IXIC`/`NASDAQCOM` desde 1971, Nasdaq 100 desde 1985/86.

**Amplitud (breadth) real — HUECO parcial, ver abajo:** las series de amplitud "clásicas" (línea
advance-decline, % de valores sobre su media de 200d, nuevos máximos/mínimos, McClellan) **NO están
disponibles gratis como serie descargable** ni en Yahoo (`^ADD`, `^ADVN`, `^S5FI`… todos vacíos/404)
ni en FRED (búsqueda "advance decline" / "stocks above moving average" → nada). Se cubren con **proxies
verificados**: equal-weight vs cap-weight (`RSP`/`SPY`, desde 2003), small-vs-large (SMB, IWM/SPY),
high-beta vs low-vol (`SPHB`/`SPLV`, 2011) y, si se necesita amplitud "pura", **calcularla** a partir
de los constituyentes o de la cesta de 11 sectores SPDR (fuera de este catálogo, pero anotado).

**Ground truth de régimen (validación, equity-relevante):** NBER recession `USREC` en FRED,
mensual desde **1854** (2.059 obs). Es el marcador de régimen recesión/expansión canónico.

---

## Detalle de verificación por fuente

### yfinance (funciona perfectamente en este entorno)
Todos los `^GSPC, ^IXIC, ^NDX, ^RUT, ^DJI, ^VIX, ^NYA, ^W5000, ^SP400, ^SP600, ^RUA, ^RUI, ^OEX,
^VXO, ^VXN, ^OVX, ^GVZ, ^VVIX, ^SKEW, ^SP500TR` y los ETFs `XLK…XLC, RSP, SPY, QQQ, IWM, IWD, IWF,
SPYG, SPYV, SPHB, SPLV, MTUM` descargaron con `period='max'`. Fechas de inicio reales en el YAML.
Los tickers de amplitud (`^ADD, ^ADDN, ^DECN, ^ADVN, ^UPD, ^GSPCEW`) devolvieron **vacío/404**.

### FRED (API key verificada, funciona)
- `SP500` y `DJIA`: **solo 10 años** (2016→2026) por licencia S&P → NO sirven para historia; usar Yahoo.
- `NASDAQCOM` (1971), `NASDAQ100` (1986), `VIXCLS` (1990), `VXOCLS` (1986-2021), `VXNCLS` (2001),
  `USREC` (1854): todos completos y verificados.
- **Wilshire retirado de FRED:** `WILL5000IND/PR/…` → "series does not exist"; búsqueda "Wilshire 5000"
  devuelve 0 resultados. Usar `^W5000` de Yahoo en su lugar.
- Sin series de breadth (advance-decline / % sobre media) en FRED.

### Académicos (deep history, verificados descargando y parseando)
- **Shiller** `ie_data.xls` (1.6 MB, hoja *Data*): mensual **1871.01 → 2024.09** en esta copia alojada.
  Ojo: la copia del blob va **~22 meses retrasada** hoy; se actualiza de forma irregular. Tratar como
  ancla mensual histórica y cruzar frescura con Yahoo para el tramo reciente.
- **Ken French Data Library** (Dartmouth): `F-F_Research_Data_Factors_daily` (Mkt-RF/SMB/HML/RF,
  1926-07-01→2026-05-29, 26.253 filas) y `5_Industry_Portfolios_daily` (Cnsmr/Manuf/HiTec/Hlth/Other,
  1926→2026, 52.506 filas). Descarga directa por ZIP, sin auth. Actualización viva mensual.

### Stooq — NO verificable desde este entorno (honestidad)
El endpoint CSV `stooq.com/q/d/l/?s=...` devuelve una **página HTML de verificación JavaScript
anti-bot** (796 bytes, "This site requires JavaScript to verify your browser") para `^spx`, `^dji`,
`spy.us`, etc. No he podido confirmar NINGUNA serie concreta de Stooq en esta sesión. Stooq sí suele
tener `^dji` con más historia que Yahoo (que solo da Dow desde 1992), pero queda **sin verificar**.

### Goyal-Welch — no verificado aquí
La URL que probé dio 404 (el fichero está en Google Drive con enlace cambiante). Aportaría S&P500
mensual con dividendos desde 1926, pero Shiller + Ken French ya cubren esa profundidad, así que baja
prioridad. Marcado `verificado: false`.

---

## Recomendación de armado

- **Spine diario Pista A:** `^GSPC` (1927) como precio + `FF_MKT_DAILY` (1926) como retorno de
  referencia; volatilidad realizada calculable desde 1927 con los retornos de `^GSPC`.
- **Ancla mensual larga:** Shiller (1871) para contexto secular / valoración (CAPE).
- **Spine de miedo:** empalme `VXO`(1986-1990) + `VIX`(1990→); `SKEW` como enricher de cola.
- **Breadth operativo (proxies):** `RSP/SPY` (concentración), `SMB` (small/large), `SPHB/SPLV`
  (risk-on/off). Breadth "pura" → computar desde los 11 SPDR o constituyentes.
- **Validación:** `USREC` (NBER) como etiqueta laxa de régimen recesión/expansión.

---

```yaml
# ================= SPINE HISTÓRICO PROFUNDO (Pista A) =================
- nombre_interno: SP500
  descripcion: "S&P 500 índice de precio, la espina diaria más profunda gratis"
  fuente: yfinance
  id: "^GSPC"
  auth: none
  inicio_verificado: "1927-12-30"
  granularidad: diaria
  pista: A
  rol: spine
  relevancia_regimen: "Nivel/retorno del mercado US; base para drawdowns, vol realizada y estados bull/bear"
  verificado: true
  evidencia: "yf.download('^GSPC','max') -> 24752 filas, 1927-12-30 a 2026-07-17"
  url: "https://finance.yahoo.com/quote/%5EGSPC"

- nombre_interno: SP500_SHILLER
  descripcion: "S&P 500 mensual (P, D, E, CPI, CAPE) de Robert Shiller desde 1871"
  fuente: academico
  id: "ie_data.xls"
  auth: none
  inicio_verificado: "1871-01"
  granularidad: mensual
  pista: A
  rol: spine
  relevancia_regimen: "Ancla secular 150+ años; valoración (CAPE) y contexto de largos ciclos/regímenes"
  verificado: true
  evidencia: "read_excel hoja Data, skiprows=7 -> 1845 filas mensuales 1871.01 a 2024.09 (copia alojada va ~22m retrasada)"
  url: "https://shillerdata.com/  (blob: img1.wsimg.com/.../ie_data.xls)"

- nombre_interno: FF_MKT_DAILY
  descripcion: "Factor de mercado US (Mkt-RF) + RF diario de Ken French, la serie diaria de equity más larga"
  fuente: academico
  id: "F-F_Research_Data_Factors_daily"
  auth: none
  inicio_verificado: "1926-07-01"
  granularidad: diaria
  pista: A
  rol: spine
  relevancia_regimen: "Retorno de mercado limpio desde 1926; base de vol/drawdown y de todos los factores"
  verificado: true
  evidencia: "unzip CSV -> 26253 filas 1926-07-01 a 2026-05-29 (cols Mkt-RF,SMB,HML,RF)"
  url: "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"

- nombre_interno: SP500_TR
  descripcion: "S&P 500 Total Return (con dividendos reinvertidos)"
  fuente: yfinance
  id: "^SP500TR"
  auth: none
  inicio_verificado: "1988-01-04"
  granularidad: diaria
  pista: A
  rol: core
  relevancia_regimen: "Retorno total real del inversor; mejor que precio para regímenes de rendimiento"
  verificado: true
  evidencia: "yf.download('^SP500TR','max') -> 9707 filas desde 1988-01-04"
  url: "https://finance.yahoo.com/quote/%5ESP500TR"

# ================= VOLATILIDAD DE EQUITY (régimen de miedo) =================
- nombre_interno: VIX
  descripcion: "CBOE VIX, volatilidad implícita 30d del S&P 500"
  fuente: yfinance
  id: "^VIX"
  auth: none
  inicio_verificado: "1990-01-02"
  granularidad: diaria
  pista: ambas
  rol: core
  relevancia_regimen: "Termómetro de estrés/miedo; clave para separar régimen calmado vs crisis"
  verificado: true
  evidencia: "yf.download('^VIX','max') -> 9203 filas desde 1990-01-02; FRED VIXCLS igual (9231 obs)"
  url: "https://finance.yahoo.com/quote/%5EVIX"

- nombre_interno: VXO
  descripcion: "CBOE VXO (VIX metodología antigua, S&P100), extiende el miedo hasta 1986"
  fuente: yfinance
  id: "^VXO"
  auth: none
  inicio_verificado: "1986-01-02"
  granularidad: diaria
  pista: A
  rol: fallback
  relevancia_regimen: "Vol implícita pre-1990; empalmar con VIX para spine de miedo desde 1986 (incl. crash 1987)"
  verificado: true
  evidencia: "yf.download('^VXO','max') -> 8998 filas 1986-01-02 a 2021-09-23 (descontinuado); FRED VXOCLS igual"
  url: "https://fred.stlouisfed.org/series/VXOCLS"

- nombre_interno: SKEW
  descripcion: "CBOE SKEW, riesgo de cola (probabilidad implícita de movimientos extremos)"
  fuente: yfinance
  id: "^SKEW"
  auth: none
  inicio_verificado: "1990-01-02"
  granularidad: diaria
  pista: ambas
  rol: enricher
  relevancia_regimen: "Complementa al VIX: detecta miedo a colas incluso con vol baja"
  verificado: true
  evidencia: "yf.download('^SKEW','max') -> 9128 filas desde 1990-01-02"
  url: "https://finance.yahoo.com/quote/%5ESKEW"

- nombre_interno: VXN
  descripcion: "CBOE VXN, volatilidad implícita del Nasdaq 100"
  fuente: yfinance
  id: "^VXN"
  auth: none
  inicio_verificado: "2001-01-23"
  granularidad: diaria
  pista: B
  rol: enricher
  relevancia_regimen: "Estrés específico de tecnología/growth; diverge del VIX en regímenes sectoriales"
  verificado: true
  evidencia: "yf.download('^VXN','max') -> 6408 filas desde 2001-01-23; FRED VXNCLS desde 2001-02"
  url: "https://finance.yahoo.com/quote/%5EVXN"

- nombre_interno: VVIX
  descripcion: "CBOE VVIX, vol de la vol (incertidumbre sobre el propio VIX)"
  fuente: yfinance
  id: "^VVIX"
  auth: none
  inicio_verificado: "2007-01-03"
  granularidad: diaria
  pista: B
  rol: enricher
  relevancia_regimen: "Picos de VVIX anticipan cambios de régimen de volatilidad"
  verificado: true
  evidencia: "yf.download('^VVIX','max') -> 4906 filas desde 2007-01-03"
  url: "https://finance.yahoo.com/quote/%5EVVIX"

# ================= ÍNDICES AMPLIOS / TOTAL MARKET =================
- nombre_interno: NASDAQ_COMP
  descripcion: "Nasdaq Composite"
  fuente: yfinance
  id: "^IXIC"
  auth: none
  inicio_verificado: "1971-02-05"
  granularidad: diaria
  pista: A
  rol: core
  relevancia_regimen: "Sesgo tech/growth; su divergencia con S&P marca regímenes de rotación"
  verificado: true
  evidencia: "yf.download('^IXIC','max') -> 13978 filas desde 1971-02-05; FRED NASDAQCOM idéntico (13979 obs)"
  url: "https://finance.yahoo.com/quote/%5EIXIC"

- nombre_interno: NASDAQ100
  descripcion: "Nasdaq 100 (100 mayores no financieras)"
  fuente: fred
  id: "NASDAQ100"
  auth: FRED_API_KEY
  inicio_verificado: "1986-01-02"
  granularidad: diaria
  pista: A
  rol: core
  relevancia_regimen: "Mega-cap tech; concentración y liderazgo growth por régimen"
  verificado: true
  evidencia: "FRED NASDAQ100 -> 10215 obs desde 1986-01-02; Yahoo ^NDX desde 1985-10-01 (10277 filas)"
  url: "https://fred.stlouisfed.org/series/NASDAQ100"

- nombre_interno: NYSE_COMP
  descripcion: "NYSE Composite (todas las acciones del NYSE)"
  fuente: yfinance
  id: "^NYA"
  auth: none
  inicio_verificado: "1965-12-31"
  granularidad: diaria
  pista: A
  rol: enricher
  relevancia_regimen: "Mercado amplio (no solo mega-caps); proxy de amplitud a nivel índice desde 1965"
  verificado: true
  evidencia: "yf.download('^NYA','max') -> 15235 filas desde 1965-12-31"
  url: "https://finance.yahoo.com/quote/%5ENYA"

- nombre_interno: WILSHIRE5000
  descripcion: "Wilshire 5000 Total Market (Yahoo lo sigue sirviendo; FRED lo retiró)"
  fuente: yfinance
  id: "^W5000"
  auth: none
  inicio_verificado: "1989-01-03"
  granularidad: diaria
  pista: B
  rol: enricher
  relevancia_regimen: "Mercado total US all-cap; su ratio vs S&P mide participación/amplitud"
  verificado: true
  evidencia: "yf.download('^W5000','max') -> 9437 filas desde 1989-01-03. FRED WILL5000* -> 'series does not exist' (retirado)"
  url: "https://finance.yahoo.com/quote/%5EW5000"

- nombre_interno: DJIA
  descripcion: "Dow Jones Industrial Average"
  fuente: yfinance
  id: "^DJI"
  auth: none
  inicio_verificado: "1992-01-02"
  granularidad: diaria
  pista: B
  rol: fallback
  relevancia_regimen: "Blue-chips; poca aportación extra sobre S&P. Yahoo solo da desde 1992 (Stooq tendría más, sin verificar)"
  verificado: true
  evidencia: "yf.download('^DJI','max') -> 8696 filas desde 1992-01-02 (historia corta en Yahoo)"
  url: "https://finance.yahoo.com/quote/%5EDJI"

- nombre_interno: SP100
  descripcion: "S&P 100 (mayores blue-chips del S&P 500)"
  fuente: yfinance
  id: "^OEX"
  auth: none
  inicio_verificado: "1982-08-02"
  granularidad: diaria
  pista: A
  rol: enricher
  relevancia_regimen: "Mega-cap vs S&P amplio: concentración/liderazgo desde 1982"
  verificado: true
  evidencia: "yf.download('^OEX','max') -> 11078 filas desde 1982-08-02"
  url: "https://finance.yahoo.com/quote/%5EOEX"

# ================= SMALL / LARGE / ESTILO =================
- nombre_interno: RUSSELL2000
  descripcion: "Russell 2000 small-cap"
  fuente: yfinance
  id: "^RUT"
  auth: none
  inicio_verificado: "1987-09-10"
  granularidad: diaria
  pista: A
  rol: core
  relevancia_regimen: "Small-caps lideran en risk-on y caen primero en risk-off; clave para régimen de riesgo"
  verificado: true
  evidencia: "yf.download('^RUT','max') -> 9786 filas desde 1987-09-10"
  url: "https://finance.yahoo.com/quote/%5ERUT"

- nombre_interno: RUSSELL3000
  descripcion: "Russell 3000 (98% del mercado investable US)"
  fuente: yfinance
  id: "^RUA"
  auth: none
  inicio_verificado: "1987-09-10"
  granularidad: diaria
  pista: B
  rol: enricher
  relevancia_regimen: "Mercado amplio all-cap; base para ratio small/large con ^RUT"
  verificado: true
  evidencia: "yf.download('^RUA','max') -> 9757 filas desde 1987-09-10"
  url: "https://finance.yahoo.com/quote/%5ERUA"

- nombre_interno: RUSSELL1000
  descripcion: "Russell 1000 large-cap"
  fuente: yfinance
  id: "^RUI"
  auth: none
  inicio_verificado: "1992-12-10"
  granularidad: diaria
  pista: B
  rol: enricher
  relevancia_regimen: "Large-cap; numerador natural del spread small/large (^RUT/^RUI)"
  verificado: true
  evidencia: "yf.download('^RUI','max') -> 8456 filas desde 1992-12-10"
  url: "https://finance.yahoo.com/quote/%5ERUI"

- nombre_interno: SP_MIDCAP400
  descripcion: "S&P MidCap 400"
  fuente: yfinance
  id: "^SP400"
  auth: none
  inicio_verificado: "1981-01-02"
  granularidad: diaria
  pista: A
  rol: enricher
  relevancia_regimen: "Mid-caps con historia profunda (1981); eslabón entre small y large por régimen"
  verificado: true
  evidencia: "yf.download('^SP400','max') -> 11477 filas desde 1981-01-02"
  url: "https://finance.yahoo.com/quote/%5ESP400"

- nombre_interno: SP_SMALLCAP600
  descripcion: "S&P SmallCap 600 (small-cap con filtro de calidad)"
  fuente: yfinance
  id: "^SP600"
  auth: none
  inicio_verificado: "1989-01-03"
  granularidad: diaria
  pista: B
  rol: enricher
  relevancia_regimen: "Alternativa al Russell 2000 con filtro de rentabilidad; small-cap risk gauge"
  verificado: true
  evidencia: "yf.download('^SP600','max') -> 9454 filas desde 1989-01-03"
  url: "https://finance.yahoo.com/quote/%5ESP600"

- nombre_interno: FF_SMB_DAILY
  descripcion: "Factor SMB (small-minus-big) diario de Ken French desde 1926"
  fuente: academico
  id: "F-F_Research_Data_Factors_daily:SMB"
  auth: none
  inicio_verificado: "1926-07-01"
  granularidad: diaria
  pista: A
  rol: spine
  relevancia_regimen: "Spread small/large con 100 años de historia; su signo define régimen de tamaño"
  verificado: true
  evidencia: "misma descarga que FF_MKT_DAILY; columna SMB, 26253 filas 1926-2026"
  url: "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"

- nombre_interno: FF_HML_DAILY
  descripcion: "Factor HML (value-minus-growth) diario de Ken French desde 1926"
  fuente: academico
  id: "F-F_Research_Data_Factors_daily:HML"
  auth: none
  inicio_verificado: "1926-07-01"
  granularidad: diaria
  pista: A
  rol: enricher
  relevancia_regimen: "Value vs growth con historia máxima; rotación de estilo es marcador de régimen"
  verificado: true
  evidencia: "misma descarga que FF_MKT_DAILY; columna HML, 26253 filas 1926-2026"
  url: "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"

- nombre_interno: IWM_SMALL
  descripcion: "iShares Russell 2000 ETF (small-cap tradeable)"
  fuente: yfinance
  id: "IWM"
  auth: none
  inicio_verificado: "2000-05-26"
  granularidad: diaria
  pista: B
  rol: fallback
  relevancia_regimen: "Proxy tradeable de small-caps; IWM/SPY es proxy de amplitud/risk-appetite"
  verificado: true
  evidencia: "yf.download('IWM','max') -> 6573 filas desde 2000-05-26"
  url: "https://finance.yahoo.com/quote/IWM"

- nombre_interno: IWD_VALUE
  descripcion: "iShares Russell 1000 Value ETF"
  fuente: yfinance
  id: "IWD"
  auth: none
  inicio_verificado: "2000-05-26"
  granularidad: diaria
  pista: B
  rol: enricher
  relevancia_regimen: "Value large-cap; par con IWF para rotación value/growth"
  verificado: true
  evidencia: "yf.download('IWD','max') -> 6573 filas desde 2000-05-26"
  url: "https://finance.yahoo.com/quote/IWD"

- nombre_interno: IWF_GROWTH
  descripcion: "iShares Russell 1000 Growth ETF"
  fuente: yfinance
  id: "IWF"
  auth: none
  inicio_verificado: "2000-05-26"
  granularidad: diaria
  pista: B
  rol: enricher
  relevancia_regimen: "Growth large-cap; IWF/IWD = ratio de estilo por régimen"
  verificado: true
  evidencia: "yf.download('IWF','max') -> 6573 filas desde 2000-05-26"
  url: "https://finance.yahoo.com/quote/IWF"

# ================= SECTORES GICS =================
- nombre_interno: SPDR_SECTORS_9
  descripcion: "9 SPDR Select Sector originales: XLK,XLF,XLE,XLV,XLI,XLY,XLP,XLU,XLB"
  fuente: yfinance
  id: "XLK,XLF,XLE,XLV,XLI,XLY,XLP,XLU,XLB"
  auth: none
  inicio_verificado: "1998-12-22"
  granularidad: diaria
  pista: B
  rol: core
  relevancia_regimen: "Rotación sectorial (cíclicos vs defensivos, p.ej. XLY/XLP) es firma directa de régimen"
  verificado: true
  evidencia: "yf.download de cada uno 'max' -> 6933 filas cada uno desde 1998-12-22"
  url: "https://www.sectorspdrs.com/"

- nombre_interno: XLRE_REALESTATE
  descripcion: "SPDR Real Estate Select Sector (11º sector GICS, spin-off 2015)"
  fuente: yfinance
  id: "XLRE"
  auth: none
  inicio_verificado: "2015-10-08"
  granularidad: diaria
  pista: B
  rol: enricher
  relevancia_regimen: "Sensible a tipos; régimen inmobiliario/tipos"
  verificado: true
  evidencia: "yf.download('XLRE','max') -> 2708 filas desde 2015-10-08"
  url: "https://finance.yahoo.com/quote/XLRE"

- nombre_interno: XLC_COMMSERVICES
  descripcion: "SPDR Communication Services Select Sector (reclasificación GICS 2018)"
  fuente: yfinance
  id: "XLC"
  auth: none
  inicio_verificado: "2018-06-19"
  granularidad: diaria
  pista: B
  rol: enricher
  relevancia_regimen: "Agrupa mega-cap internet/media; liderazgo growth moderno"
  verificado: true
  evidencia: "yf.download('XLC','max') -> 2030 filas desde 2018-06-19"
  url: "https://finance.yahoo.com/quote/XLC"

- nombre_interno: FF_5_INDUSTRY
  descripcion: "Ken French 5 Industry Portfolios diarios (Cnsmr,Manuf,HiTec,Hlth,Other) desde 1926"
  fuente: academico
  id: "5_Industry_Portfolios_daily"
  auth: none
  inicio_verificado: "1926-07-01"
  granularidad: diaria
  pista: A
  rol: enricher
  relevancia_regimen: "Rotación sectorial con 100 años; permite estudiar sectores en crisis pre-1998 (donde no hay SPDR). También existen versiones 10/12/49 industrias"
  verificado: true
  evidencia: "unzip CSV -> 52506 filas 1926-07-01 a 2026-05-29, cabecera Cnsmr,Manuf,HiTec,Hlth,Other"
  url: "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"

# ================= AMPLITUD / CONCENTRACIÓN / RIESGO (proxies) =================
- nombre_interno: SP500_EW
  descripcion: "Invesco S&P 500 Equal Weight (RSP): proxy de amplitud/concentración"
  fuente: yfinance
  id: "RSP"
  auth: none
  inicio_verificado: "2003-05-01"
  granularidad: diaria
  pista: B
  rol: core
  relevancia_regimen: "RSP/SPY (equal vs cap-weight) mide si el rally es amplio o concentrado en pocas mega-caps: el mejor proxy de breadth gratis"
  verificado: true
  evidencia: "yf.download('RSP','max') -> 5840 filas desde 2003-05-01"
  url: "https://finance.yahoo.com/quote/RSP"

- nombre_interno: SP500_CAPWEIGHT_ETF
  descripcion: "SPDR S&P 500 ETF (SPY), denominador para el ratio de amplitud RSP/SPY"
  fuente: yfinance
  id: "SPY"
  auth: none
  inicio_verificado: "1993-01-29"
  granularidad: diaria
  pista: B
  rol: fallback
  relevancia_regimen: "Cap-weight tradeable; par de RSP para concentración y de IWM para small/large"
  verificado: true
  evidencia: "yf.download('SPY','max') -> 8423 filas desde 1993-01-29"
  url: "https://finance.yahoo.com/quote/SPY"

- nombre_interno: SPHB_HIGHBETA
  descripcion: "Invesco S&P 500 High Beta ETF"
  fuente: yfinance
  id: "SPHB"
  auth: none
  inicio_verificado: "2011-05-05"
  granularidad: diaria
  pista: B
  rol: enricher
  relevancia_regimen: "SPHB/SPLV (high-beta vs low-vol) es proxy directo de risk-on/risk-off"
  verificado: true
  evidencia: "yf.download('SPHB','max') -> 3822 filas desde 2011-05-05"
  url: "https://finance.yahoo.com/quote/SPHB"

- nombre_interno: SPLV_LOWVOL
  descripcion: "Invesco S&P 500 Low Volatility ETF"
  fuente: yfinance
  id: "SPLV"
  auth: none
  inicio_verificado: "2011-05-05"
  granularidad: diaria
  pista: B
  rol: enricher
  relevancia_regimen: "Par defensivo de SPHB; liderazgo low-vol marca régimen de aversión al riesgo"
  verificado: true
  evidencia: "yf.download('SPLV','max') -> 3822 filas desde 2011-05-05"
  url: "https://finance.yahoo.com/quote/SPLV"

- nombre_interno: MTUM_MOMENTUM
  descripcion: "iShares MSCI USA Momentum Factor ETF"
  fuente: yfinance
  id: "MTUM"
  auth: none
  inicio_verificado: "2013-04-18"
  granularidad: diaria
  pista: B
  rol: enricher
  relevancia_regimen: "El factor momentum sufre 'crashes' en giros de régimen; señal de transición"
  verificado: true
  evidencia: "yf.download('MTUM','max') -> 3332 filas desde 2013-04-18"
  url: "https://finance.yahoo.com/quote/MTUM"

# ================= VALIDACIÓN (ground truth de régimen) =================
- nombre_interno: NBER_RECESSION
  descripcion: "NBER US Recession Indicator (1=recesión) desde 1854"
  fuente: fred
  id: "USREC"
  auth: FRED_API_KEY
  inicio_verificado: "1854-12-01"
  granularidad: mensual
  pista: validacion
  rol: validation
  relevancia_regimen: "Etiqueta oficial recesión/expansión; ground truth laxo para validar regímenes detectados"
  verificado: true
  evidencia: "FRED USREC -> 2059 obs desde 1854-12-01 hasta 2026-06-01"
  url: "https://fred.stlouisfed.org/series/USREC"

# ================= HUECOS / NO VERIFICADO (honestidad) =================
- nombre_interno: BREADTH_ADLINE_PCTMA
  descripcion: "Amplitud clásica: línea advance-decline, % sobre media 50/200d, nuevos máx/mín, McClellan"
  fuente: yfinance
  id: "^ADD,^ADVN,^DECN,^S5FI,^S5TH (todos vacíos)"
  auth: none
  inicio_verificado: null
  granularidad: diaria
  pista: B
  rol: enricher
  relevancia_regimen: "Sería la amplitud 'pura' ideal; su deterioro anticipa cambios de régimen"
  verificado: false
  evidencia: "^ADD/^ADVN/^DECN/^ADDN/^UPD -> 404/vacío en Yahoo; FRED sin series de advance-decline ni %-sobre-media. NO disponible gratis como serie lista; se debe COMPUTAR desde constituyentes o proxiar con RSP/SPY y SMB"
  url: "https://finance.yahoo.com/"

- nombre_interno: DOW_STOOQ_DEEP
  descripcion: "Dow Jones histórico largo (pre-1992) que Stooq suele ofrecer"
  fuente: stooq
  id: "^dji"
  auth: none
  inicio_verificado: null
  granularidad: diaria
  pista: A
  rol: fallback
  relevancia_regimen: "Extendería el Dow antes de 1992; pero Yahoo ya da S&P500 desde 1927, poca ganancia marginal"
  verificado: false
  evidencia: "stooq.com/q/d/l/?s=^dji devuelve pagina HTML de verificacion JavaScript anti-bot (796 bytes) en este entorno; NO se pudo confirmar la serie"
  url: "https://stooq.com/q/d/l/?s=%5Edji&i=d"

- nombre_interno: SP500_GOYALWELCH
  descripcion: "S&P500 mensual con dividendos (Goyal-Welch predictor dataset) desde 1926"
  fuente: academico
  id: "PredictorData"
  auth: none
  inicio_verificado: null
  granularidad: mensual
  pista: A
  rol: fallback
  relevancia_regimen: "Cross-check de Shiller para retorno total mensual; baja prioridad (Shiller+French ya cubren)"
  verificado: false
  evidencia: "URL de export probada dio 404 (fichero en Google Drive con enlace cambiante); no verificado en esta sesion"
  url: "https://sites.google.com/view/agoyal145/home"
```
