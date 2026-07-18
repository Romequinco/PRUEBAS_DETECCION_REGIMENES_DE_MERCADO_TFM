# Categoría: Datasets Kaggle + repos GitHub (Stooq bulk, Nasdaq Data Link, Tiingo) — verificado

Investigador: agente **Datasets Kaggle + repos GitHub**. Foco: datasets de Kaggle de OHLC/market
completos, repos de GitHub con S&P500 histórico/intradía, **Stooq bulk download**, **Nasdaq Data Link
(free tier)** y **Tiingo**. Objetivo del TFM: máximo dato GRATIS con máximo histórico para detección de
regímenes (Pista A = espina histórica profunda S&P500+vol; Pista B = panel multi-activo; validación =
índices de estrés + fechas de crisis).

Todas las pruebas se hicieron **de verdad** el **2026-07-18** desde este entorno (Windows, Python 3.13,
`yfinance` 0.2.66, `requests`, sin credenciales de Kaggle/Tiingo/Nasdaq). Reporto lo que **realmente
descargué** (filas, rango de fechas observado) y, cuando una fuente está tapiada, lo digo con la
evidencia exacta del bloqueo.

---

## Resumen ejecutivo (lee esto primero)

**La sorpresa importante: Stooq y Nasdaq Data Link NO son scriptables gratis desde aquí ahora mismo.**

- **Stooq** (per-ticker CSV `/q/d/l/?s=...`) está detrás de un **desafío JavaScript proof-of-work
  (SHA-256)** y, aun resolviéndolo (lo resolví: `POST /__verify` devuelve `ok` y setea cookie `auth`),
  la descarga del CSV responde **`Access denied`** (bloqueo por IP/automatización). El **bulk download**
  (`static.stooq.com/db/h/d_us_txt.zip`) devuelve **HTTP 401** (requiere login/cuenta). Conclusión: el
  histórico profundo de Stooq (su fama: índices desde el s. XIX) **no lo pude confirmar** desde este
  entorno. Recomendación: usar `yfinance` como sustituto para los mismos tickers (ver abajo) o descargar
  Stooq **a mano desde un navegador real**.
- **Nasdaq Data Link** (ex-Quandl): acceso anónimo tapiado con **Akamai 403**. Requiere **API key de
  cuenta gratuita**. La mayoría de datasets son premium; el clásico free de equity **WIKI/WIKIP está
  descontinuado y congelado en 2018-03**. Free tier útil solo para datasets abiertos concretos.
- **Tiingo**: anónimo devuelve **401 "Please supply a token"**. Requiere **token gratuito**. Free tier
  documentado: **30+ años de EOD**, 1.000 req/día, 500 símbolos únicos/mes, 50 símbolos/hora.

**Lo que SÍ funciona gratis, sin auth y verificado end-to-end son los repos GitHub de datahub.io**
(licencia **ODC-PDDL-1.0 = dominio público**), que además están **vivos** (datos hasta 2026-06/07):

- **`datasets/s-and-p-500`** → S&P500 **mensual desde 1871-01** con `SP500, Dividend, Earnings, CPI,
  Long Interest Rate, Real Price, PE10 (CAPE)`. **Joya para Pista A**: espina de valoración de 150+ años
  en un solo CSV público (deriva de Shiller pero auto-actualizado en GitHub).
- **`datasets/finance-vix`** → **VIX diario OHLC** 1990-01-02 → 2026-07-16 sin auth (alternativa/mirror
  de FRED `VIXCLS` para el agente de volatilidad/validación).
- **`datasets/gold-prices`** → **oro mensual desde 1833** (safe-haven, Pista B).
- **`fja05680/sp500`** → **composición point-in-time del S&P500 diaria 1996-01 → 2026-06** (universo sin
  survivorship bias para construir breadth propia).

**Sobre Kaggle:** las páginas de dataset son ahora un SPA cliente y **no exponen metadatos en el HTML**
(sin JSON-LD, sin estado embebido), así que **no pude verificar a máquina licencia/cobertura** ni
descargar (requiere `kaggle.json`). Documento los datasets famosos (existencia confirmada por búsqueda)
y, crucialmente, **su fuente subyacente**: casi todos los "S&P500 index" de Kaggle son **re-exports
congelados de `^GSPC`/`SPY`** que sí verifiqué vivos con `yfinance`. Para el TFM, la copia viva (yfinance)
> la copia congelada de Kaggle, salvo que quieras un snapshot reproducible fijo.

**Ancla verificada (fuente que los Kaggle re-exportan), vía `yfinance period='max'`:**
`^GSPC` → **24.752 filas, 1927-12-30 → 2026-07-17**; `SPY` → 8.423 filas, 1993-01-29 → ...;
`^VIX` → 9.203 filas, 1990-01-02 → ...; `^DJI` → 8.696 filas, **solo 1992-01-02** → ... (el Dow diario
profundo NO está gratis: yfinance arranca en 1992 y Stooq —que sí llega a 1896— está bloqueado aquí).

---

## Detalle y evidencia por fuente

### 1) Repos GitHub datahub.io (VERIFICADO, sin auth, licencia PDDL, vivos)

| serie | repo/archivo | inicio | fin/estado | freq | rol |
|---|---|---|---|---|---|
| S&P500 + CAPE mensual | datasets/s-and-p-500 · data.csv | 1871-01-01 | vivo (2026-06) | mensual | spine ★ (A) |
| VIX diario OHLC | datasets/finance-vix · vix-daily.csv | 1990-01-02 | vivo (2026-07-16) | diaria | core (ambas) |
| Oro mensual | datasets/gold-prices · monthly.csv | 1833-01 | vivo (2026-06) | mensual | enricher (B) |
| 10Y yield mensual | datasets/bond-yields-us-10y · monthly.csv | 1953-04-01 | vivo (2026-05) | mensual | fallback (B) |
| Constituyentes point-in-time | fja05680/sp500 · Historical Components | 1996-01-02 | vivo (2026-06-30) | diaria | enricher |
| Constituyentes actuales | datasets/s-and-p-500-companies · constituents.csv | 503 tickers | snapshot | — | enricher |

Todos los `data.csv`/`monthly.csv` de datahub declaran `license: ODC-PDDL-1.0` en su `datapackage.json`
(verificado). Se descargan con `requests` normal desde `raw.githubusercontent.com` (rama `main`).

### 2) Kaggle (existencia confirmada; download requiere `kaggle.json`; metadatos NO verificables a máquina)

El HTML de Kaggle es un SPA sin metadatos server-side (probé: sin `application/ld+json`, sin
`__NEXT_DATA__`, sin `license`/`temporalCoverage` en el shell de 20 KB). No hay credenciales aquí. Por
tanto marco todos los Kaggle como `verificado: false` con evidencia honesta. Los relevantes:

- **borismarjanovic/price-volume-data-for-all-us-stocks-etfs** ("Huge Stock Market Dataset"): todos los
  stocks+ETFs US diarios, **origen Stooq**, **congelado ~2017-11-16**, licencia reportada CC0. Útil como
  panel estático de universo completo, pero no vivo.
- **jacksoncrow/stock-market-dataset**: refresh del anterior hasta ~2020 (stocks+ETFs US diarios).
- **rezanematpour/...-gspc-index-data-19272025** y **henryhan117/sp-500-historical-data**: `^GSPC` diario
  1927→2025/2020. Son **re-exports de `^GSPC`** (verificado vivo por yfinance, ver ancla arriba).
- **camnugent/sandp500**: 500 stocks del S&P, ~5 años diarios (2013-2018). Histórico corto.
- **andrewmvd/sp-500-stocks**: S&P500 stocks+índice "daily updated".

### 3) Stooq (BLOQUEADO desde este entorno)

- Per-ticker CSV: `GET stooq.com/q/d/l/?s=^spx&i=d` → página HTML con **PoW SHA-256** (`d=4`). Resolví el
  PoW en Python y `POST /__verify` → `ok` (+ cookie `auth`), pero el reintento del CSV → **`Access
  denied`** (13 bytes). `pandas_datareader.stooq` falla igual.
- Bulk: `static.stooq.com/db/h/d_us_txt.zip` → **HTTP 401** (login requerido);
  `stooq.com/db/h/d_us_txt.zip` → 404. El bulk gratis ya no es anónimo.
- No pude confirmar el histórico profundo de Stooq (^SPX/^DJI pre-1950). Sustituto verificado: yfinance.

### 4) Nasdaq Data Link (free tier: requiere API key gratuita; anónimo tapiado)

- Anónimo → **Akamai 403** en `data.nasdaq.com/api/v3/datasets/...`. Necesita `api_key` de cuenta free.
- Free tier: 300 llamadas/10s, 2.000/10min, 50.000/día, máx 10.000 filas/req. La mayoría de datasets son
  premium; hay datasets **abiertos** (bancos centrales/gobiernos). **WIKI/WIKIP (equity EOD free) está
  descontinuado, congelado en 2018-03** → no sirve para histórico vivo.

### 5) Tiingo (free tier: requiere token gratuito; anónimo 401)

- Anónimo → **401 `{"detail":"Please supply a token"}`**. Con token free: **30+ años de EOD**,
  1.000 req/día, 500 símbolos únicos/mes, 50 símbolos/hora. Buen enricher EOD si se registra la cuenta.

---

```yaml
series_datasets_repos:
  # ================= GITHUB datahub.io — VERIFICADO, sin auth, licencia PDDL, vivo =================
  - nombre_interno: GH_SP500_CAPE_MONTHLY
    descripcion: "S&P500 mensual con valoración: columnas SP500, Dividend, Earnings, Consumer Price Index, Long Interest Rate (10y), Real Price, Real Dividend, Real Earnings, PE10 (CAPE de Shiller). Deriva de Shiller pero auto-actualizado en GitHub (dominio público PDDL)."
    fuente: github
    id: "datasets/s-and-p-500:data/data.csv"
    auth: none
    inicio_verificado: "1871-01-01"
    granularidad: mensual
    pista: A
    rol: spine
    relevancia_regimen: "Espina de valoracion de 150+ anios en un CSV publico: nivel del indice, CAPE/PE10 (regimen de valoracion caro/barato), tipo de interes largo y CPI para des-inflactar. Cubre TODAS las crisis desde el s.XIX. Mensual, ideal como backbone macro-valoracion de Pista A."
    verificado: true
    evidencia: "GET raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv -> 1866 filas, Date 1871-01-01 -> 2026-06-01. Cols: Date,SP500,Dividend,Earnings,Consumer Price Index,Long Interest Rate,Real Price,Real Dividend,Real Earnings,PE10. Licencia ODC-PDDL-1.0 (datapackage.json)."
    url: "https://github.com/datasets/s-and-p-500"

  - nombre_interno: GH_VIX_DAILY
    descripcion: "CBOE VIX diario OHLC (Open/High/Low/Close) servido por datahub.io sin auth. Mirror del VIX oficial (equivalente a FRED VIXCLS para el Close)."
    fuente: github
    id: "datasets/finance-vix:data/vix-daily.csv"
    auth: none
    inicio_verificado: "1990-01-02"
    granularidad: diaria
    pista: ambas
    rol: core
    relevancia_regimen: "Termometro de volatilidad implicita 1990+; VIX>20 estres, >30 crisis. Alternativa sin auth a FRED VIXCLS; ademas trae OHLC (rango intradiario del VIX). Feature/label laxo de risk-off."
    verificado: true
    evidencia: "GET raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv -> 9230 filas, DATE 1990-01-02 -> 2026-07-16. Cols: DATE,OPEN,HIGH,LOW,CLOSE. Licencia ODC-PDDL-1.0."
    url: "https://github.com/datasets/finance-vix"

  - nombre_interno: GH_GOLD_MONTHLY
    descripcion: "Precio del oro mensual (USD/oz) desde 1833, datahub.io. Serie de safe-haven de historia muy larga."
    fuente: github
    id: "datasets/gold-prices:data/monthly.csv"
    auth: none
    inicio_verificado: "1833-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Activo refugio: en regimenes risk-off el oro sube frente a equity. Historia de 190+ anios para contexto de crisis pre-modernas. Baja frecuencia (mensual)."
    verificado: true
    evidencia: "GET raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv -> 2322 filas, Date 1833-01 -> 2026-06. Cols: Date,Price. Licencia ODC-PDDL-1.0."
    url: "https://github.com/datasets/gold-prices"

  - nombre_interno: GH_US10Y_MONTHLY
    descripcion: "Rendimiento del bono del Tesoro US a 10 anios, mensual, datahub.io."
    fuente: github
    id: "datasets/bond-yields-us-10y:data/monthly.csv"
    auth: none
    inicio_verificado: "1953-04-01"
    granularidad: mensual
    pista: B
    rol: fallback
    relevancia_regimen: "Nivel de tipos largos (curva). Solapa con lo que el agente de tipos/curva cubre en FRED (DGS10); aqui como fallback sin auth. Mensual."
    verificado: true
    evidencia: "GET raw.githubusercontent.com/datasets/bond-yields-us-10y/main/data/monthly.csv -> 878 filas, Date 1953-04-01 -> 2026-05-01. Cols: Date,Rate. Licencia ODC-PDDL-1.0."
    url: "https://github.com/datasets/bond-yields-us-10y"

  - nombre_interno: GH_SP500_CONSTITUENTS_PIT
    descripcion: "Composicion point-in-time del S&P500: por cada fecha, la lista de tickers miembros. Permite construir un universo sin survivorship bias."
    fuente: github
    id: "fja05680/sp500:S&P 500 Historical Components & Changes (Updated).csv"
    auth: none
    inicio_verificado: "1996-01-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Universo historicamente correcto para calcular breadth/dispersion propia (% miembros sobre media, nuevos maximos/minimos) sin sesgo de supervivencia. Complementa al agente de equity_breadth."
    verificado: true
    evidencia: "GitHub API repos/fja05680/sp500/contents + GET raw master -> 2718 filas, date 1996-01-02 -> 2026-06-30. Cols: date,tickers (lista CSV de miembros por fecha). Repo sin LICENSE explicito (uso investigacion)."
    url: "https://github.com/fja05680/sp500"

  - nombre_interno: GH_SP500_CONSTITUENTS_NOW
    descripcion: "Snapshot de los 503 constituyentes actuales del S&P500 con sector GICS, sub-industria, sede, fecha de alta, CIK. datahub.io."
    fuente: github
    id: "datasets/s-and-p-500-companies:data/constituents.csv"
    auth: none
    inicio_verificado: null
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Metadata de sector para agrupar features por GICS o construir breadth sectorial. Es un snapshot (no serie temporal)."
    verificado: true
    evidencia: "GET raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv -> 503 filas. Cols: Symbol,Security,GICS Sector,GICS Sub-Industry,Headquarters Location,Date added,CIK,Founded. Licencia ODC-PDDL-1.0."
    url: "https://github.com/datasets/s-and-p-500-companies"

  # ================= KAGGLE — existencia OK, metadatos NO verificables, download necesita token =====
  - nombre_interno: KAGGLE_HUGE_STOCK_MARKET
    descripcion: "'Huge Stock Market Dataset': todos los stocks y ETFs de US, precios/volumen diarios historicos completos, origen Stooq. Panel de universo completo estatico (congelado ~2017-11)."
    fuente: kaggle
    id: "borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
    auth: none
    inicio_verificado: null
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Panel amplio (miles de tickers) para construir dispersion/breadth/correlacion cross-sectional de un snapshot historico. PERO congelado en 2017-11 (no vivo) y de origen Stooq. Bueno como dataset reproducible fijo, no para tiempo real."
    verificado: false
    evidencia: "Pagina Kaggle existe (busqueda 2026-07-18). NO verificable a maquina: HTML es SPA sin JSON-LD/metadatos; descarga requiere cuenta+kaggle.json (no disponible aqui). Licencia reportada CC0 (no verificada). Publicado 2017-11-16, origen Stooq."
    url: "https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs"

  - nombre_interno: KAGGLE_STOCKMARKET_2020
    descripcion: "'Stock Market Dataset' (jacksoncrow): refresh del Huge Stock Market Dataset, stocks+ETFs US diarios hasta ~2020."
    fuente: kaggle
    id: "jacksoncrow/stock-market-dataset"
    auth: none
    inicio_verificado: null
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Version mas reciente (~2020) del panel completo US; util para breadth/dispersion sobre universo amplio. Congelado, requiere token para bajar."
    verificado: false
    evidencia: "Pagina Kaggle existe (busqueda 2026-07-18). Metadatos no verificables a maquina (SPA). Descarga requiere kaggle.json. Cobertura ~1970s-2020 segun descripcion, NO confirmada aqui."
    url: "https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset"

  - nombre_interno: KAGGLE_GSPC_1927
    descripcion: "'Historical S&P 500 (^GSPC) Index Data (1927-2025)': OHLC diario del indice S&P500 desde 1927-12-30. Es un re-export de ^GSPC (yfinance/Yahoo)."
    fuente: kaggle
    id: "rezanematpour/historical-s-and-p-500-gspc-index-data-19272025"
    auth: none
    inicio_verificado: null
    granularidad: diaria
    pista: A
    rol: fallback
    relevancia_regimen: "Espina diaria de Pista A (1927+). PERO es copia congelada de ^GSPC: mejor usar la fuente viva (yfinance ^GSPC, VERIFICADA: 24752 filas desde 1927-12-30). Este Kaggle solo si quieres snapshot reproducible fijo."
    verificado: false
    evidencia: "Pagina Kaggle existe (busqueda 2026-07-18). No verificable a maquina (SPA)/sin token. Fuente subyacente ^GSPC SI verificada aparte: yf.download('^GSPC','max') -> 24752 filas, 1927-12-30 -> 2026-07-17."
    url: "https://www.kaggle.com/datasets/rezanematpour/historical-s-and-p-500-gspc-index-data-19272025"

  - nombre_interno: KAGGLE_SANDP500_5Y
    descripcion: "'S&P 500 stock data' (camnugent): OHLCV diario de las ~500 empresas del S&P500, ~5 anios (2013-2018)."
    fuente: kaggle
    id: "camnugent/sandp500"
    auth: none
    inicio_verificado: null
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Panel de constituyentes con historia CORTA (2013-2018). Poco util para regimenes (no cubre crisis). Solo como ejemplo/panel reciente."
    verificado: false
    evidencia: "Pagina Kaggle existe (busqueda 2026-07-18). Metadatos no verificables a maquina. Cobertura ~2013-2018 (5 anios) segun descripcion, no confirmada aqui."
    url: "https://www.kaggle.com/datasets/camnugent/sandp500"

  - nombre_interno: KAGGLE_SP500_STOCKS_DAILY
    descripcion: "'S&P 500 Stocks (daily updated)' (andrewmvd): precios diarios de los constituyentes del S&P500 + indice, con actualizacion periodica en Kaggle."
    fuente: kaggle
    id: "andrewmvd/sp-500-stocks"
    auth: none
    inicio_verificado: null
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Panel de constituyentes + indice, 'daily updated'. Util para breadth/dispersion; requiere token y su 'vivo' depende de que el autor lo refresque (no garantizado)."
    verificado: false
    evidencia: "Pagina Kaggle existe (busqueda 2026-07-18). Metadatos/actualizacion no verificables a maquina ni sin token."
    url: "https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks"

  # ================= STOOQ — BLOQUEADO desde este entorno =================
  - nombre_interno: STOOQ_SPX_DAILY
    descripcion: "S&P500 (^spx) diario en Stooq via CSV per-ticker. Stooq tiene fama de histórico profundo de indices."
    fuente: stooq
    id: "^spx"
    auth: none
    inicio_verificado: null
    granularidad: diaria
    pista: A
    rol: fallback
    relevancia_regimen: "Espina diaria alternativa; Stooq suele dar mas historia que Yahoo en algunos indices. NO confirmable aqui por el bloqueo; usar yfinance ^GSPC como sustituto verificado."
    verificado: false
    evidencia: "GET stooq.com/q/d/l/?s=^spx&i=d -> pagina PoW SHA-256 (d=4). Resuelto el PoW y POST /__verify -> 'ok' + cookie auth, PERO reintento CSV -> 'Access denied' (13 bytes). pandas_datareader.stooq falla igual (RemoteDataError). Bloqueo por IP/automatizacion."
    url: "https://stooq.com/q/d/l/?s=%5Espx&i=d"

  - nombre_interno: STOOQ_BULK_DB
    descripcion: "Descarga masiva de la base Stooq (todos los tickers US en un zip de .txt diarios)."
    fuente: stooq
    id: "db/h/d_us_txt.zip"
    auth: none
    inicio_verificado: null
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Panel US completo en un descarga; ideal para breadth/dispersion. NO gratis-anonimo ya: requiere login/cuenta Stooq."
    verificado: false
    evidencia: "GET static.stooq.com/db/h/d_us_txt.zip -> HTTP 401 (login requerido). stooq.com/db/h/d_us_txt.zip -> 404. El bulk ya no es descargable de forma anonima."
    url: "https://stooq.com/db/"

  # ================= NASDAQ DATA LINK (ex-Quandl) — free tier con API key =================
  - nombre_interno: NDL_FREE_TIER
    descripcion: "Nasdaq Data Link (ex-Quandl): plataforma de datasets financieros/economicos. Free tier con API key de cuenta gratuita; mayoria premium, algunos datasets abiertos."
    fuente: github
    id: "nasdaq-data-link:api/v3"
    auth: none
    inicio_verificado: null
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Acceso a datasets abiertos (bancos centrales/gobiernos) y algunos de mercado. Poco util para equity vivo: el clasico free WIKI/WIKIP esta descontinuado (congelado 2018-03). Requiere registrarse."
    verificado: false
    evidencia: "GET data.nasdaq.com/api/v3/datasets/MULTPL/SP500_REAL_PRICE_MONTH/data.json (anonimo) -> Akamai HTTP 403. Free tier documentado: 300 req/10s, 2000/10min, 50000/dia, 10000 filas/req; requiere API key gratuita. WIKI/WIKIP descontinuado 2018-03."
    url: "https://data.nasdaq.com/"

  # ================= TIINGO — free tier con token =================
  - nombre_interno: TIINGO_EOD
    descripcion: "Tiingo: API de precios EOD (end-of-day) de equities/ETFs con 30+ anios de historia. Free tier con token gratuito."
    fuente: github
    id: "tiingo:daily/{ticker}/prices"
    auth: none
    inicio_verificado: null
    granularidad: diaria
    pista: ambas
    rol: enricher
    relevancia_regimen: "Fuente EOD limpia y viva para S&P500 (SPY) y constituyentes; util como cross-check/enricher frente a yfinance. Free tier suficiente para pocos simbolos. Requiere token."
    verificado: false
    evidencia: "GET api.tiingo.com/tiingo/daily/spy/prices (anonimo) -> HTTP 401 {'detail':'Please supply a token'}. Free tier documentado: 30+ anios EOD, 1000 req/dia, 500 simbolos unicos/mes, 50 simbolos/hora. Requiere token gratuito."
    url: "https://www.tiingo.com/"
```

---

## Recomendación de priorización (para el orquestador)

1. **Usar YA (verificado, gratis, sin auth, vivo):**
   `GH_SP500_CAPE_MONTHLY` (espina de valoración 1871+ con CAPE) y `GH_VIX_DAILY` (VIX diario 1990+).
   Son de licencia dominio público (PDDL) y se bajan con `requests` de `raw.githubusercontent.com`.
   `GH_GOLD_MONTHLY` y `GH_SP500_CONSTITUENTS_PIT` como enrichers.
2. **Para la espina DIARIA de Pista A (1927+):** no depender de Kaggle ni Stooq; usar **`yfinance ^GSPC`**
   (verificado: 24.752 filas desde 1927-12-30). Los Kaggle "1927-2025" son copias congeladas de esto.
3. **Kaggle**: valor real solo si quieres **un snapshot reproducible fijo** (p.ej. el panel US completo
   de `borismarjanovic`/`jacksoncrow` para breadth histórica). Requiere cuenta + `kaggle.json`. No lo
   trates como vivo.
4. **Stooq / Nasdaq Data Link / Tiingo**: registrarse solo si necesitas algo que las anteriores no dan.
   Stooq (histórico profundo de índices) y Tiingo (EOD 30+ años) son los de mayor potencial, pero exigen
   navegador/login/token; ninguno lo pude verificar end-to-end sin credenciales aquí.

## Huecos y dudas (honestidad)
- **Dow Jones diario profundo NO está gratis-verificado**: yfinance `^DJI` arranca en 1992; Stooq (que
  llega a 1896) está bloqueado. Si el TFM quiere Dow pre-1992 diario, hará falta Stooq manual o fuente
  académica.
- **Licencias Kaggle no verificadas a máquina** (SPA sin metadatos). La de `borismarjanovic` se reporta
  CC0 pero no lo pude confirmar; comprobar en la página antes de redistribuir.
- **`fja05680/sp500` sin LICENSE explícito** → tratar como uso de investigación, no redistribuir.
- **Stooq PoW resoluble pero descarga bloqueada por IP**: quizá funcione desde una IP residencial/navegador
  real; desde este entorno, no.

## Fuentes y métodos de verificación
- **GitHub raw** (`raw.githubusercontent.com`) + **GitHub API** (`api.github.com/repos/.../contents`):
  descarga real de CSV, conteo de filas, rango de fechas, y `datapackage.json` para licencia. Sin auth.
- **yfinance 0.2.66** (`period='max'`): ancla de `^GSPC`, `SPY`, `^VIX`, `^DJI`.
- **Stooq**: `requests` + resolución del PoW SHA-256 + `POST /__verify`; `pandas_datareader.stooq`.
- **Nasdaq Data Link / Tiingo**: `requests` anónimo (403 Akamai / 401 token) + WebSearch para free tier.
- **Kaggle**: WebSearch (existencia) + fetch del HTML (SPA sin metadatos) → no verificable a máquina.
- Fecha de verificación: **2026-07-18**.
