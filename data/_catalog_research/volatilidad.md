# Categoría: Volatilidad — estado del arte de datos (verificado)

Investigador: agente **Volatilidad**. Foco: VIX, VXO (proxy pre-1990), MOVE (vol de bonos),
VVIX (vol-of-vol), estructura temporal del VIX (VIX9D / VIX3M / VIX6M), realized vol y familia
de índices de vol cross-asset del CBOE. La volatilidad es la feature más discriminante para
detección de régimen, así que es clave para **ambas pistas**.

Todas las verificaciones se hicieron **de verdad** el 2026-07-18 (descarga real vía yfinance
`period='max'` y FRED API con `FRED_API_KEY` del `.env`, sin imprimir la clave). Reporto la
fecha de inicio **observada**, no la de marketing.

---

## Resumen ejecutivo de lo verificado

**Espina de vol para Pista A (~1928+):** No existe volatilidad implícita antes de 1986. La
espina honesta de volatilidad profunda es **realized vol computada del S&P 500**: `^GSPC` en
yfinance arranca en **1927-12-30** (24.752 filas verificadas), así que una RV rolling 21d
anualizada existe desde **1928-01-31**. Sanity-check: el pico de RV21 en 2008-Q4 sale 85.4 (correcto).
Encima de eso, la implícita más antigua es el **VXO (S&P 100)** desde **1986-01-02**, y el **VIX**
desde **1990-01-02**. El solape VXO/VIX (7.990 días comunes) da media(VXO−VIX)=+0.27, o sea el
VXO es un empalme casi limpio para extender el VIX 4 años hacia atrás (1986→1990).

**Panel de vol para Pista B (~1990/2002+):** VIX (core), **MOVE** (vol de bonos, ^MOVE desde
2002-11-12, crítico y complementario al VIX porque captura estrés de tipos), **VVIX** (vol-of-vol,
2007+), **estructura temporal** VIX9D/VIX3M/VIX6M (señal de backwardation = estrés) y la familia
cross-asset del CBOE (VXN, VXD, RVX, OVX, GVZ, EVZ, VXEEM, VXEWZ).

**Hallazgos de fuente relevantes:**
- **FRED gana a yfinance** en dos series: `VXDCLS` (DJIA vol) empieza **1997-10-07** en FRED vs
  2005-11 en yfinance; y `RVXCLS` (Russell 2000 vol) **sólo funciona en FRED** (2004-01-02) —
  yfinance `^RVX` devuelve vacío.
- **yfinance gana a FRED** en la estructura temporal: `^VIX9D` (2011), `^VIX3M` (2006-07, ~1.4 años
  antes que FRED `VXVCLS` 2007-12) y `^VIX6M` (2008) **no están en FRED**. VVIX y MOVE tampoco
  están en FRED (son propietarios / no cargados), sólo yfinance.
- **Stooq está bloqueado** ahora mismo: el endpoint CSV (`stooq.com/q/d/l/?s=^vix`) devuelve un
  challenge JavaScript de proof-of-work (probado con curl y con pandas_datareader → RemoteDataError).
  Lo dejo como fallback **no verificado**; hoy no es utilizable sin navegador.
- Varias series del CBOE están **descontinuadas** pero siguen siendo válidas como historia:
  VXO (fin 2021-09-23), EVZ (FX euro, fin 2025-03-11), VXTYN (10y Treasury, fin 2020-05-15),
  y VXFXI/VXSLV/VXGDX (fin 2022-02-11).

**No verificado / lo que no encontré:** no hay volatilidad **implícita** libre y fiable antes de
1986 (VXO es el suelo); para pre-1986 la única opción honesta es realized vol. No encontré una
serie MOVE en FRED (propietaria ICE BofA); ^MOVE de yfinance actualiza con ~1 semana de retraso
respecto al VIX (último dato 2026-07-10 vs VIX 2026-07-17), a tener en cuenta para features vivas.

---

## Detalle por serie (evidencia)

| serie | fuente elegida | inicio verificado | fin/estado | nota |
|---|---|---|---|---|
| VIX | FRED VIXCLS | 1990-01-02 | vivo (upd 2026-07-17) | core, ambas pistas |
| VXO | FRED VXOCLS | 1986-01-02 | descontinuado 2021-09-23 | extiende VIX pre-1990 |
| Realized vol S&P500 | calc. de yfinance ^GSPC | 1928-01-31 | vivo | espina profunda Pista A |
| MOVE | yfinance ^MOVE | 2002-11-12 | vivo (upd ~2026-07-10) | vol de bonos, crítico |
| VVIX | yfinance ^VVIX | 2007-01-03 | vivo | vol-of-vol |
| VIX9D | yfinance ^VIX9D | 2011-01-03 | vivo | term structure corto |
| VIX3M | yfinance ^VIX3M | 2006-07-17 | vivo | term structure (ex-VXV) |
| VIX6M | yfinance ^VIX6M | 2008-01-02 | vivo | term structure largo |
| SKEW | yfinance ^SKEW | 1990-01-02 | vivo | riesgo de cola |
| VXN | FRED VXNCLS | 2001-02-02 | vivo | NASDAQ100 vol |
| VXD | FRED VXDCLS | 1997-10-07 | vivo | DJIA vol (FRED>yf) |
| RVX | FRED RVXCLS | 2004-01-02 | vivo | Russell2000 vol (sólo FRED) |
| OVX | FRED OVXCLS | 2007-05-10 | vivo | vol petróleo |
| GVZ | FRED GVZCLS | 2008-06-03 | vivo | vol oro |
| EVZ | FRED EVZCLS | 2007-11-01 | descontinuado 2025-03-11 | vol EUR/USD |
| VXEEM | FRED VXEEMCLS | 2011-03-16 | vivo | vol EM ETF |
| VXEWZ | FRED VXEWZCLS | 2011-03-16 | vivo | vol Brasil ETF |
| VXTYN | FRED VXTYN | 2003-01-02 | descontinuado 2020-05-15 | vol 10y Treasury (fallback MOVE) |

Notas de features derivadas (para la capa de features, causales/expanding):
- **Pendiente term structure** = `VIX3M − VIX` o ratio `VIX/VIX3M` (>1 backwardation = estrés). Desde 2006-07.
- **VIX9D/VIX** = estrés de muy corto plazo (event risk). Desde 2011.
- **VIX − RV21** = varianza risk premium (compresión/expansión de prima). Desde 1990.
- **VVIX/VIX** = convexidad del riesgo. Desde 2007.
- **MOVE/VIX** = estrés bonos vs equity (rotación de régimen). Desde 2002.

```yaml
series_volatilidad:
  - nombre_interno: VIX
    descripcion: "CBOE Volatility Index (implícita S&P500 30d). Termómetro de miedo, feature de vol núcleo."
    fuente: fred
    id: "VIXCLS"
    auth: FRED_API_KEY
    inicio_verificado: "1990-01-02"
    granularidad: diaria
    pista: ambas
    rol: core
    relevancia_regimen: "Discriminador primario risk-on/risk-off; niveles y saltos definen crisis."
    verificado: true
    evidencia: "FRED VIXCLS -> 9231 obs validas, first 1990-01-02:17.24 last 2026-07-16:16.73 (upd 2026-07-17). yfinance ^VIX confirma 9203 filas mismo rango."
    url: "https://fred.stlouisfed.org/series/VIXCLS"

  - nombre_interno: VXO
    descripcion: "CBOE S&P100 Volatility Index (metodologia VIX vieja). Implícita más antigua disponible; extiende el VIX pre-1990."
    fuente: fred
    id: "VXOCLS"
    auth: FRED_API_KEY
    inicio_verificado: "1986-01-02"
    granularidad: diaria
    pista: A
    rol: core
    relevancia_regimen: "Cubre el crash de 1987 (Black Monday) que el VIX no alcanza; empalme casi limpio con VIX."
    verificado: true
    evidencia: "FRED VXOCLS -> 9002 obs, first 1986-01-02:18.07 last 2021-09-23:17.87 (DESCONTINUADO). Solape con VIX: 7990 dias comunes, media(VXO-VIX)=+0.27. yfinance ^VXO confirma 8998 filas."
    url: "https://fred.stlouisfed.org/series/VXOCLS"

  - nombre_interno: REALIZED_VOL_SP500
    descripcion: "Volatilidad realizada del S&P500 (rolling std de log-returns, anualizada). Calculada de ^GSPC. Espina de vol profunda para Pista A donde no hay implícita."
    fuente: yfinance
    id: "^GSPC"
    auth: none
    inicio_verificado: "1928-01-31"
    granularidad: diaria
    pista: A
    rol: spine
    relevancia_regimen: "Unica vol disponible pre-1986; captura ~10+ crisis (1929, 1937, 1987, 2000, 2008...). Base de variance risk premium vs VIX."
    verificado: true
    evidencia: "yf.download('^GSPC','max') -> 24752 filas desde 1927-12-30. RV21d anualizada existe desde 1928-01-31; pico 2008-Q4 = 85.4 (coherente)."
    url: "https://finance.yahoo.com/quote/%5EGSPC"

  - nombre_interno: MOVE
    descripcion: "ICE BofA MOVE Index: volatilidad implícita de opciones sobre Treasuries (curva 2/5/10/30y). El 'VIX de los bonos'."
    fuente: yfinance
    id: "^MOVE"
    auth: none
    inicio_verificado: "2002-11-12"
    granularidad: diaria
    pista: B
    rol: core
    relevancia_regimen: "Estrés de tipos/duración, complementa al VIX; picos en 2008, 2020, 2022-2023 (SVB/gilts). No esta en FRED."
    verificado: true
    evidencia: "yf.download('^MOVE','max') -> 5849 filas desde 2002-11-12 (primer valor 120.2), ultimo 2026-07-10:69.6. OJO: actualiza ~1 semana tarde vs VIX."
    url: "https://finance.yahoo.com/quote/%5EMOVE"

  - nombre_interno: VVIX
    descripcion: "CBOE VVIX: volatilidad de la volatilidad (implícita de opciones sobre VIX)."
    fuente: yfinance
    id: "^VVIX"
    auth: none
    inicio_verificado: "2007-01-03"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Convexidad/tail risk; VVIX/VIX marca estados de miedo al miedo. No esta en FRED."
    verificado: true
    evidencia: "yf.download('^VVIX','max') -> 4906 filas desde 2007-01-03, vivo hasta 2026-07-17."
    url: "https://finance.yahoo.com/quote/%5EVVIX"

  - nombre_interno: VIX9D
    descripcion: "CBOE VIX de 9 dias (implícita ultra-corto plazo)."
    fuente: yfinance
    id: "^VIX9D"
    auth: none
    inicio_verificado: "2011-01-03"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Extremo corto de la estructura temporal; VIX9D/VIX detecta event risk inminente. No esta en FRED."
    verificado: true
    evidencia: "yf.download('^VIX9D','max') -> 3902 filas desde 2011-01-03, ultimo 2026-07-10."
    url: "https://finance.yahoo.com/quote/%5EVIX9D"

  - nombre_interno: VIX3M
    descripcion: "CBOE VIX de 3 meses (antes VXV). Nodo medio de la estructura temporal del VIX."
    fuente: yfinance
    id: "^VIX3M"
    auth: none
    inicio_verificado: "2006-07-17"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "VIX/VIX3M (pendiente) es señal de régimen fuerte: >1 backwardation = estrés agudo. Alternativa FRED VXVCLS (desde 2007-12-04, ~1.4y menos)."
    verificado: true
    evidencia: "yf.download('^VIX3M','max') -> 5027 filas desde 2006-07-17. FRED VXVCLS confirma serie equivalente: 4682 obs, first 2007-12-04:24.65 last 2026-07-16:19.5."
    url: "https://finance.yahoo.com/quote/%5EVIX3M"

  - nombre_interno: VIX6M
    descripcion: "CBOE VIX de 6 meses. Extremo largo de la estructura temporal del VIX."
    fuente: yfinance
    id: "^VIX6M"
    auth: none
    inicio_verificado: "2008-01-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Ancla el nivel base de vol esperada; forma completa de la curva VIX9D->VIX->VIX3M->VIX6M. No esta en FRED."
    verificado: true
    evidencia: "yf.download('^VIX6M','max') -> 4659 filas desde 2008-01-02, ultimo 2026-07-10."
    url: "https://finance.yahoo.com/quote/%5EVIX6M"

  - nombre_interno: SKEW
    descripcion: "CBOE SKEW Index: precio del riesgo de cola izquierda (tail risk) del S&P500."
    fuente: yfinance
    id: "^SKEW"
    auth: none
    inicio_verificado: "1990-01-02"
    granularidad: diaria
    pista: ambas
    rol: enricher
    relevancia_regimen: "Complementa al VIX: mide asimetria/miedo a crash; historia larga (1990)."
    verificado: true
    evidencia: "yf.download('^SKEW','max') -> 9128 filas desde 1990-01-02, vivo hasta 2026-07-17."
    url: "https://finance.yahoo.com/quote/%5ESKEW"

  - nombre_interno: VXN
    descripcion: "CBOE NASDAQ-100 Volatility Index. Vol implícita del segmento tech/growth."
    fuente: fred
    id: "VXNCLS"
    auth: FRED_API_KEY
    inicio_verificado: "2001-02-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Dispersion de vol sectorial (tech vs broad); VXN-VIX marca estres growth-especifico."
    verificado: true
    evidencia: "FRED VXNCLS -> 6400 obs, first 2001-02-02:54.89 last 2026-07-16:27.34. yfinance ^VXN confirma 6408 filas."
    url: "https://fred.stlouisfed.org/series/VXNCLS"

  - nombre_interno: VXD
    descripcion: "CBOE DJIA Volatility Index. Vol implícita de large-cap value/industrial."
    fuente: fred
    id: "VXDCLS"
    auth: FRED_API_KEY
    inicio_verificado: "1997-10-07"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Historia mas larga que VXN; FRED empieza 1997 (vs 2005 en yfinance ^VXD), cubre dotcom y 1998."
    verificado: true
    evidencia: "FRED VXDCLS -> 7239 obs, first 1997-10-07:21.48 last 2026-07-16:13.32. yfinance ^VXD solo desde 2005-11-22 (FRED gana)."
    url: "https://fred.stlouisfed.org/series/VXDCLS"

  - nombre_interno: RVX
    descripcion: "CBOE Russell 2000 Volatility Index. Vol implícita de small-caps."
    fuente: fred
    id: "RVXCLS"
    auth: FRED_API_KEY
    inicio_verificado: "2004-01-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Estres small-cap (sensibles a ciclo/credito); RVX-VIX = prima de riesgo tamaño. SOLO en FRED (yfinance ^RVX devuelve vacio)."
    verificado: true
    evidencia: "FRED RVXCLS -> 5667 obs, first 2004-01-02:23.04 last 2026-07-16:20.64. yfinance ^RVX -> EMPTY (delisted)."
    url: "https://fred.stlouisfed.org/series/RVXCLS"

  - nombre_interno: OVX
    descripcion: "CBOE Crude Oil ETF Volatility Index (implícita sobre USO)."
    fuente: fred
    id: "OVXCLS"
    auth: FRED_API_KEY
    inicio_verificado: "2007-05-10"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Vol de commodities/energia; shock 2020 (petroleo negativo) y shocks geopoliticos."
    verificado: true
    evidencia: "FRED OVXCLS metadata 2007-05-10 -> 2026-07-16 (upd 2026-07-17). yfinance ^OVX confirma 4827 filas desde 2007-05-10."
    url: "https://fred.stlouisfed.org/series/OVXCLS"

  - nombre_interno: GVZ
    descripcion: "CBOE Gold ETF Volatility Index (implícita sobre GLD)."
    fuente: fred
    id: "GVZCLS"
    auth: FRED_API_KEY
    inicio_verificado: "2008-06-03"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Vol del activo refugio; GVZ sube en flight-to-safety desordenado."
    verificado: true
    evidencia: "FRED GVZCLS metadata 2008-06-03 -> 2026-07-16 (upd 2026-07-17). yfinance ^GVZ confirma 4559 filas desde 2008-06-03."
    url: "https://fred.stlouisfed.org/series/GVZCLS"

  - nombre_interno: EVZ
    descripcion: "CBOE EuroCurrency ETF Volatility Index (implícita FX EUR/USD sobre FXE)."
    fuente: fred
    id: "EVZCLS"
    auth: FRED_API_KEY
    inicio_verificado: "2007-11-01"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Vol de FX; estres de divisas (crisis euro 2011-12). DESCONTINUADO 2025-03-11."
    verificado: true
    evidencia: "FRED EVZCLS metadata 2007-11-01 -> 2025-03-11 (DESCONTINUADO, upd 2025-03-12)."
    url: "https://fred.stlouisfed.org/series/EVZCLS"

  - nombre_interno: VXEEM
    descripcion: "CBOE Emerging Markets ETF Volatility Index (implícita sobre EEM)."
    fuente: fred
    id: "VXEEMCLS"
    auth: FRED_API_KEY
    inicio_verificado: "2011-03-16"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Estres de mercados emergentes; VXEEM-VIX = prima de riesgo EM (taper tantrum 2013)."
    verificado: true
    evidencia: "FRED VXEEMCLS metadata 2011-03-16 -> 2026-07-16 (upd 2026-07-17)."
    url: "https://fred.stlouisfed.org/series/VXEEMCLS"

  - nombre_interno: VXEWZ
    descripcion: "CBOE Brazil ETF Volatility Index (implícita sobre EWZ)."
    fuente: fred
    id: "VXEWZCLS"
    auth: FRED_API_KEY
    inicio_verificado: "2011-03-16"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Proxy de estres EM/LatAm y commodities; complementa VXEEM."
    verificado: true
    evidencia: "FRED VXEWZCLS metadata 2011-03-16 -> 2026-07-16 (upd 2026-07-17), vivo."
    url: "https://fred.stlouisfed.org/series/VXEWZCLS"

  - nombre_interno: VXTYN
    descripcion: "CBOE/CBOT 10-Year Treasury Note Volatility Index. Alternativa a MOVE (vol de bonos) basada en futuros."
    fuente: fred
    id: "VXTYN"
    auth: FRED_API_KEY
    inicio_verificado: "2003-01-02"
    granularidad: diaria
    pista: B
    rol: fallback
    relevancia_regimen: "Vol de tipos 10y; peor que MOVE (mas corta y descontinuada 2020-05-15) pero libre en FRED."
    verificado: true
    evidencia: "FRED VXTYN metadata 2003-01-02 -> 2020-05-15 (DESCONTINUADO, upd 2020-06-17)."
    url: "https://fred.stlouisfed.org/series/VXTYN"

  # --- FALLBACK NO VERIFICADO ---
  - nombre_interno: VIX_STOOQ
    descripcion: "VIX vía Stooq CSV como fallback de yfinance/FRED."
    fuente: stooq
    id: "^VIX"
    auth: none
    inicio_verificado: null
    granularidad: diaria
    pista: ambas
    rol: fallback
    relevancia_regimen: "Redundancia de fuente para el VIX si cae yfinance/FRED."
    verificado: false
    evidencia: "BLOQUEADO: stooq.com/q/d/l/?s=^vix devuelve challenge JS proof-of-work (curl) y pandas_datareader -> RemoteDataError. No usable sin navegador ahora mismo."
    url: "https://stooq.com/q/d/l/?s=^vix&i=d"
```

---

## Recomendación de priorización para el pipeline

1. **Imprescindibles (core):** VIX (FRED VIXCLS) + VXO (FRED VXOCLS, empalme pre-1990) +
   REALIZED_VOL_SP500 (de ^GSPC, espina Pista A 1928+) + MOVE (^MOVE, vol de bonos).
2. **Estructura temporal (muy informativa, Pista B):** VIX9D, VIX3M, VIX6M (yfinance) + VVIX.
   La pendiente VIX/VIX3M es de las mejores señales de cambio de régimen.
3. **Cross-asset (enrichers Pista B):** SKEW, VXN, VXD (FRED, 1997+), RVX (sólo FRED),
   OVX, GVZ, VXEEM. EVZ/VXTYN sólo como fallback (descontinuados).
4. **Descargar VIX/VXO/term-structure preferentemente por FRED** (limpio, API, vivo) y usar
   yfinance para lo que FRED no tiene (MOVE, VVIX, VIX9D/3M/6M, SKEW). Stooq NO fiable hoy.
