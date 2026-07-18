# Categoría: Macro — estado del arte de datos (verificado)

Investigador: agente **Macro**. Foco encargado: **UNRATE** (desempleo), **SAHMREALTIME** (regla de
Sahm en tiempo real), **PMI/ISM**, **CFNAI** (Chicago Fed), **INDPRO** (producción industrial),
**ICSA** (peticiones iniciales de paro) y **CPIAUCSL** (inflación). Series mensuales/semanales, no
diarias, pero **contexto de régimen** de primer orden: marcan el ciclo económico real que subyace a
los regímenes de mercado (expansión / desaceleración / recesión / recuperación).

Todas las verificaciones se hicieron **de verdad** el 2026-07-18 contra la **FRED API** con
`FRED_API_KEY` del `.env` (sin imprimir la clave). Reporto la fecha de inicio **observada** en la API,
la última observación, la fecha de actualización y, para las series clave, un **sanity check en crisis
conocidas**. Casi todo el universo macro útil vive en FRED y es gratis, largo y vivo.

---

## ⚠️ Hallazgo central 1: el PMI/ISM **NO está en FRED** (ni gratis en ningún sitio limpio)

Probé de tres formas y el resultado es inequívoco:

- **IDs directos** `NAPM`, `NAPMPI`, `NAPMNOI`, `NAPMEI` (los códigos históricos del ISM Manufacturing
  PMI y sus subíndices) → **HTTP 400** en la API. Ya no existen.
- **Búsqueda FRED** `search_text=PMI` → **cero resultados**. `search_text=ISM Manufacturing PMI` →
  **cero**. `search_text=Purchasing Managers Index` → devuelve **basura no relacionada** (un índice
  hipotecario "AD&Co US Mortgage High Yield"), no el PMI.
- Motivo conocido: el **ISM (Institute for Supply Management)** retiró sus series de FRED por licencia;
  el **PMI de S&P Global/Markit** también es propietario. **No hay PMI manufacturero/servicios de EE.UU.
  gratis, vivo y con histórico** en fuentes abiertas fiables.

**Sustituto honesto y gratis (verificado):** las **encuestas manufactureras de los Fed regionales**,
que son índices de difusión de la misma familia que el PMI (>0/50 = expansión) y **sí están en FRED**:
- **Philadelphia Fed** (`GACDFSA066MSFRBPHI`): general activity, **mensual desde 1968-05**, vivo. Es el
  mejor: +55 años, cubre todas las recesiones modernas. El más parecido a un "PMI con historia".
- **Empire State / NY Fed** (`GACDISA066MSFRBNY`): 2001-07+, vivo.
- **Dallas Fed** (`BACTSAMFRBDAL`): 2004-06+, vivo.

Un promedio de los Fed regionales (o solo el Philly, por profundidad) es el proxy libre razonable del
pulso manufacturero tipo-PMI. **No es idéntico al ISM PMI**, pero correlaciona alto y es reproducible.

## ⚠️ Hallazgo central 2: Sahm **en tiempo real** vs **revisado** (evitar look-ahead)

Hay dos series de la regla de Sahm y **importan para el TFM**:
- **`SAHMREALTIME`** (Real-time Sahm Rule): usa los datos de paro **tal como se publicaron** (sin
  revisiones posteriores). **Sin sesgo de look-ahead** → es la correcta para un detector causal.
- **`SAHMCURRENT`** (Sahm Rule, current): recalculada con la serie de paro **ya revisada**. Tiene
  look-ahead leve (incorpora revisiones que no existían en tiempo real). Útil como referencia, no para
  backtest honesto.

Regla: se dispara la señal de recesión cuando la media móvil 3m del paro sube **≥ 0.50 pp** sobre su
mínimo de los 12 meses previos. Sanity check real de `SAHMREALTIME`: **1.27 en 2008-09**, **3.90 en
2009-06**, **9.50 en 2020-06** (máximo histórico, COVID). Dispara en todas las recesiones. Correcto.

**Caveat general de revisiones (vintages):** casi todo el macro (paro, INDPRO, payrolls, GDP) se
**revisa** tras la primera publicación. FRED sirve por defecto la **serie revisada**. Para un backtest
sin look-ahead estricto haría falta **ALFRED / vintages** (as-of). Está fuera de alcance aquí, pero se
documenta como limitación: `SAHMREALTIME`, `WEI` y las encuestas de difusión (que apenas se revisan)
son las opciones más "point-in-time" del panel.

---

## Resumen ejecutivo de lo verificado

**Espina macro profunda (todo FRED, vivo):**
- **`INDPRO`** producción industrial **mensual desde 1919** (1290 obs) — la serie de actividad real con
  más historia; YoY −15% en 2009, −17% en 2020, −9% en 1975. El termómetro macro núcleo.
- **`UNRATE`** paro **desde 1948**, `PAYEMS`/`MANEMP`/`AWHMAN` **desde 1939**, `CPIAUCSL` inflación
  **desde 1947**, `CFNAI`/`CFNAIMA3` actividad **desde 1967**.
- **`USREC`** (indicador de recesión NBER) **desde 1854** — el ground-truth de ciclo más largo que
  existe. Validación pura.

**Foco encargado — todo verificado y vivo:**
`UNRATE` (1948), `SAHMREALTIME` (1959), `SAHMCURRENT` (1949), `CFNAI` (1967), `CFNAIMA3` (1967),
`INDPRO` (1919), `ICSA` (1967, **semanal**), `CPIAUCSL` (1947). Los ocho responden y actualizan.

**Alta frecuencia dentro del macro (semanal, valiosa para régimen):**
- **`ICSA`** peticiones iniciales de paro, **semanal desde 1967**; pico histórico **6.14M el 2020-04-04**
  (de 208k a 6M en 3 semanas). `IC4WSA` (MA-4) suaviza, `CCSA` (continuadas) confirma persistencia.
- **`WEI`** Weekly Economic Index (Lewis-Mertens-Stock), **semanal desde 2008**, vivo. Nowcast semanal
  de actividad; buen puente entre lo diario de mercado y lo mensual macro.
- **`NFCI`** condiciones financieras semanal 1971 (validación).

**Inflación:** `CPIAUCSL` (headline SA 1947), `CPILFESL` (core 1957), `PCEPI`/`PCEPILFE` (deflactor PCE
1959, la medida preferida de la Fed), `MICH` (expectativas Michigan 1978), `T10YIE` (breakeven 10y de
mercado, diaria 2003 — solapa con curva/mercados).

**PMI (sin ISM gratis):** proxy = Fed regionales `GACDFSA066MSFRBPHI` (Philly, 1968), `GACDISA066MSFRBNY`
(Empire, 2001), `BACTSAMFRBDAL` (Dallas, 2004). Ver hallazgo 1.

**Validación (índices/indicadores de ciclo ya hechos, ground-truth laxo):**
`USREC` (NBER, 1854), `RECPROUSM156N` (probabilidad de recesión suavizada Chauvet-Piger, 1967),
`JHDUSRGDPBR` (Hamilton, 1967), `NFCI` (condiciones financieras Chicago Fed, 1971).

**Series que NO sirven hoy (probadas):**
- **PMI/ISM**: no está en FRED (hallazgo 1). `verificado=false`.
- **`USSLIND`** (Conference Board Leading Index): **DESCONTINUADA en FRED**, última obs **2020-02**
  (upd 2020-04). El Conference Board la retiró. No usable como serie viva.
- **`USALOLITONOSTSAM`** (OECD CLI para EE.UU.): responde pero **rancia**, última obs **2024-01**
  (aunque metadata upd 2025-11). Retraso de +2 años → no viva. Fallback, no core.

---

## Detalle por serie (evidencia)

| serie | id FRED | inicio verif. | frec | último / estado | rol |
|---|---|---|---|---|---|
| Paro | UNRATE | 1948-01-01 | M | 4.2 (2026-06), vivo | core labor |
| Sahm real-time | SAHMREALTIME | 1959-12-01 | M | 0.07 (2026-06), vivo | core señal recesión |
| Sahm revisada | SAHMCURRENT | 1949-03-01 | M | 0.07 (2026-06), vivo | fallback (look-ahead) |
| Paro U6 | U6RATE | 1994-01-01 | M | 7.9 (2026-06), vivo | enricher |
| Claims iniciales | ICSA | 1967-01-07 | **W** | 208k (2026-07-11), vivo | core alta-frec |
| Claims MA-4 | IC4WSA | 1967-01-28 | **W** | 214k, vivo | enricher |
| Claims continuadas | CCSA | 1967-01-07 | **W** | 1.805M, vivo | enricher |
| Nóminas no agrícolas | PAYEMS | 1939-01-01 | M | 158984k, vivo | core labor |
| Empleo manufacturero | MANEMP | 1939-01-01 | M | vivo | enricher |
| Chicago Fed NAI | CFNAI | 1967-03-01 | M | −0.1 (2026-05), vivo | core actividad |
| CFNAI MA-3 | CFNAIMA3 | 1967-05-01 | M | −0.03, vivo | **core (umbral −0.7)** |
| CFNAI diffusion | CFNAIDIFF | 1967-05-01 | M | vivo | enricher |
| Producción industrial | INDPRO | **1919-01-01** | M | 102.6 (2026-06), vivo | **spine macro** |
| IP manufacturera | IPMAN | 1972-01-01 | M | vivo | enricher |
| Utilización capacidad | TCU | 1967-01-01 | M | 76.1, vivo | enricher |
| Coincident activity US | USPHCI | 1979-01-01 | M | vivo | enricher |
| CPI headline SA | CPIAUCSL | 1947-01-01 | M | 332.6 (2026-06), vivo | core inflación |
| CPI core SA | CPILFESL | 1957-01-01 | M | 336.1, vivo | core inflación |
| PCE deflactor | PCEPI | 1959-01-01 | M | vivo | enricher |
| PCE core | PCEPILFE | 1959-01-01 | M | vivo | enricher (medida Fed) |
| Exp. inflación Michigan | MICH | 1978-01-01 | M | 4.8, vivo | enricher |
| Breakeven 10y | T10YIE | 2003-01-02 | D | 2.24, vivo | enricher (solapa curva) |
| PMI proxy Philly | GACDFSA066MSFRBPHI | 1968-05-01 | M | vivo | **proxy PMI (mejor)** |
| PMI proxy Empire NY | GACDISA066MSFRBNY | 2001-07-01 | M | 15.6, vivo | proxy PMI |
| PMI proxy Dallas | BACTSAMFRBDAL | 2004-06-01 | M | vivo | proxy PMI |
| Sentimiento Michigan | UMCSENT | 1952-11-01 | M | 44.8, vivo | enricher |
| Weekly Economic Index | WEI | 2008-01-05 | **W** | 2.61, vivo | enricher alta-frec |
| Global EPU | GEPUCURRENT | 1997-01-01 | M | vivo | enricher incertidumbre |
| Viviendas iniciadas | HOUST | 1959-01-01 | M | 1427, vivo | enricher cíclico |
| Permisos construcción | PERMIT | 1960-01-01 | M | 1367, vivo | enricher líder |
| Ventas minoristas | RSAFS | 1992-01-01 | M | vivo | enricher |
| Nuevos pedidos capex | NEWORDER | 1992-02-01 | M | vivo | enricher |
| Ventas vehículos | TOTALSA | 1976-01-01 | M | 16.95, vivo | enricher cíclico |
| PIB real | GDPC1 | 1947-01-01 | Q | vivo | contexto |
| PIB real % (QoQ ann.) | A191RL1Q225SBEA | 1947-04-01 | Q | 2.1, vivo | contexto |
| **Recesión NBER** | USREC | **1854-12-01** | M | 0, vivo | **validación ground-truth** |
| Prob. recesión suav. | RECPROUSM156N | 1967-06-01 | M | 0.54, vivo | validación |
| Recesión Hamilton | JHDUSRGDPBR | 1967-10-01 | Q | 0, vivo | validación |
| Cond. financieras | NFCI | 1971-01-08 | **W** | −0.538, vivo | validación (solapa crédito) |
| Leading Index CB | USSLIND | 1982-01-01 | M | **descont. 2020-02** | fallback muerto |
| OECD CLI US | USALOLITONOSTSAM | 1955-01-01 | M | **rancia 2024-01** | fallback |
| **PMI / ISM** | (no existe) | — | — | **no en FRED** | no disponible |

**Features derivadas recomendadas (causales/expanding, alineadas mensual):**
- **INDPRO YoY** y **INDPRO 3m-ann.**: caída persistente = recesión (spine de actividad, 1919+).
- **`SAHMREALTIME` ≥ 0.50** como **flag binario de recesión en tiempo real** (mejor que fechar NBER,
  que llega con años de retraso).
- **`CFNAIMA3` < −0.70**: umbral oficial de "recesión en curso" del Chicago Fed.
- **ICSA / IC4WSA**: Δ y nivel vs mínimo de 12m — se disparan antes que el paro (alta frecuencia).
- **CPI/PCE YoY** y **core YoY**: régimen inflacionario (importa para correlación acción-bono).
- **PMI-proxy (Philly)** cruzando 0: pulso manufacturero expansión/contracción.

```yaml
series_macro:
  # ============ FOCO: LABOR / RECESIÓN ============
  - nombre_interno: UNRATE
    descripcion: "Tasa de paro civil de EE.UU. (U-3), ajustada estacionalmente. Núcleo del ciclo laboral."
    fuente: fred
    id: "UNRATE"
    auth: FRED_API_KEY
    inicio_verificado: "1948-01-01"
    granularidad: mensual
    pista: ambas
    rol: core
    relevancia_regimen: "Sube de forma persistente en recesión; base de la regla de Sahm. Discriminador de régimen de ciclo laboral con +75 años."
    verificado: true
    evidencia: "FRED UNRATE -> 941 obs mensuales, first 1948-01-01:3.4 last 2026-06-01:4.2 (upd 2026-07-02)."
    url: "https://fred.stlouisfed.org/series/UNRATE"

  - nombre_interno: SAHMREALTIME
    descripcion: "Real-time Sahm Rule Recession Indicator. MA-3 del paro menos su mínimo de 12m, con datos tal como se publicaron (sin revisiones = sin look-ahead)."
    fuente: fred
    id: "SAHMREALTIME"
    auth: FRED_API_KEY
    inicio_verificado: "1959-12-01"
    granularidad: mensual
    pista: ambas
    rol: core
    relevancia_regimen: "Señal binaria de recesión en tiempo real (dispara en >=0.50). LA versión honesta para backtest causal. Pico 1.27 (2008-09), 3.90 (2009-06), 9.50 (2020-06)."
    verificado: true
    evidencia: "FRED SAHMREALTIME -> 798 obs, first 1959-12-01:0.77 last 2026-06-01:0.07 (upd 2026-07-02). Max 9.5 en 2020-06; 1.27 en 2008-09. Dispara en todas las recesiones."
    url: "https://fred.stlouisfed.org/series/SAHMREALTIME"

  - nombre_interno: SAHMCURRENT
    descripcion: "Sahm Rule Recession Indicator (current). Igual que SAHMREALTIME pero con la serie de paro ya revisada (leve look-ahead)."
    fuente: fred
    id: "SAHMCURRENT"
    auth: FRED_API_KEY
    inicio_verificado: "1949-03-01"
    granularidad: mensual
    pista: ambas
    rol: fallback
    relevancia_regimen: "Referencia de la regla de Sahm con datos revisados; NO usar para backtest sin look-ahead (usar SAHMREALTIME)."
    verificado: true
    evidencia: "FRED SAHMCURRENT -> 928 obs, first 1949-03-01:1.10 last 2026-06-01:0.07 (upd 2026-07-02)."
    url: "https://fred.stlouisfed.org/series/SAHMCURRENT"

  - nombre_interno: U6RATE
    descripcion: "Tasa de infrautilización laboral U-6 (incluye subempleo y desanimados)."
    fuente: fred
    id: "U6RATE"
    auth: FRED_API_KEY
    inicio_verificado: "1994-01-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Medida amplia de debilidad laboral; U6-U3 = holgura oculta, sube antes en desaceleraciones."
    verificado: true
    evidencia: "FRED U6RATE -> 389 obs, first 1994-01-01:11.7 last 2026-06-01:7.9 (upd 2026-07-02)."
    url: "https://fred.stlouisfed.org/series/U6RATE"

  - nombre_interno: ICSA
    descripcion: "Initial Claims (peticiones iniciales de subsidio de paro), ajustadas estacionalmente. SEMANAL."
    fuente: fred
    id: "ICSA"
    auth: FRED_API_KEY
    inicio_verificado: "1967-01-07"
    granularidad: semanal
    pista: ambas
    rol: core
    relevancia_regimen: "Indicador laboral de ALTA FRECUENCIA; se dispara antes que el paro. Pico histórico 6.14M el 2020-04-04 (COVID). Feature de estrés de ciclo temprano."
    verificado: true
    evidencia: "FRED ICSA -> 3106 obs semanales, first 1967-01-07:208000 last 2026-07-11:208000 (upd 2026-07-16). Max 6137000 el 2020-04-04; salto 208k->2.9M->5.9M->6.1M en marzo-2020."
    url: "https://fred.stlouisfed.org/series/ICSA"

  - nombre_interno: IC4WSA
    descripcion: "Initial Claims, media móvil de 4 semanas, ajustada estacionalmente. Suaviza el ruido semanal de ICSA."
    fuente: fred
    id: "IC4WSA"
    auth: FRED_API_KEY
    inicio_verificado: "1967-01-28"
    granularidad: semanal
    pista: B
    rol: enricher
    relevancia_regimen: "Versión suavizada de las claims; tendencia laboral de alta frecuencia sin el ruido semana a semana."
    verificado: true
    evidencia: "FRED IC4WSA -> 3103 obs, first 1967-01-28:209000 last 2026-07-11:214250 (upd 2026-07-16)."
    url: "https://fred.stlouisfed.org/series/IC4WSA"

  - nombre_interno: CCSA
    descripcion: "Continued Claims (personas cobrando paro de forma continuada), ajustadas estacionalmente. SEMANAL."
    fuente: fred
    id: "CCSA"
    auth: FRED_API_KEY
    inicio_verificado: "1967-01-07"
    granularidad: semanal
    pista: B
    rol: enricher
    relevancia_regimen: "Persistencia del desempleo: si las continuadas suben, el paro no encuentra recolocación (recesión en marcha)."
    verificado: true
    evidencia: "FRED CCSA -> 3105 obs, first 1967-01-07:1134000 last 2026-07-04:1805000 (upd 2026-07-16)."
    url: "https://fred.stlouisfed.org/series/CCSA"

  - nombre_interno: PAYEMS
    descripcion: "Nóminas no agrícolas (Total Nonfarm Payrolls), ajustadas estacionalmente. Empleo total EE.UU."
    fuente: fred
    id: "PAYEMS"
    auth: FRED_API_KEY
    inicio_verificado: "1939-01-01"
    granularidad: mensual
    pista: ambas
    rol: core
    relevancia_regimen: "El dato laboral más seguido; ΔPAYEMS<0 sostenido = recesión. +85 años de historia."
    verificado: true
    evidencia: "FRED PAYEMS -> 1050 obs, first 1939-01-01:29923 last 2026-06-01:158984 (upd 2026-07-02)."
    url: "https://fred.stlouisfed.org/series/PAYEMS"

  - nombre_interno: MANEMP
    descripcion: "Empleo en manufacturas, ajustado estacionalmente. Rama cíclica del empleo desde 1939."
    fuente: fred
    id: "MANEMP"
    auth: FRED_API_KEY
    inicio_verificado: "1939-01-01"
    granularidad: mensual
    pista: A
    rol: enricher
    relevancia_regimen: "Empleo del sector más cíclico; gira antes que el empleo total en los puntos de inflexión del ciclo."
    verificado: true
    evidencia: "FRED MANEMP -> 1050 obs, first 1939-01-01:9077 last 2026-06-01:12598 (upd 2026-07-02)."
    url: "https://fred.stlouisfed.org/series/MANEMP"

  # ============ FOCO: ACTIVIDAD REAL ============
  - nombre_interno: INDPRO
    descripcion: "Índice de Producción Industrial (manufacturas + minería + utilities), ajustado estacionalmente. La serie de actividad real con más historia en FRED."
    fuente: fred
    id: "INDPRO"
    auth: FRED_API_KEY
    inicio_verificado: "1919-01-01"
    granularidad: mensual
    pista: ambas
    rol: spine
    relevancia_regimen: "Termómetro macro NÚCLEO con +100 años. INDPRO YoY: -15% (2009), -17% (2020), -9% (1975), -6% (1982), -5% (2001). Cubre todas las recesiones incluida la Gran Depresión."
    verificado: true
    evidencia: "FRED INDPRO -> 1290 obs mensuales, first 1919-01-01:4.8739 last 2026-06-01:102.6395 (upd 2026-07-17). YoY verificado deeply negativo en cada recesión."
    url: "https://fred.stlouisfed.org/series/INDPRO"

  - nombre_interno: CFNAI
    descripcion: "Chicago Fed National Activity Index. Componente principal de 85 indicadores mensuales; 0 = crecimiento tendencial."
    fuente: fred
    id: "CFNAI"
    auth: FRED_API_KEY
    inicio_verificado: "1967-03-01"
    granularidad: mensual
    pista: ambas
    rol: core
    relevancia_regimen: "Índice compuesto de actividad; <0 = por debajo de tendencia. Resumen macro de una cifra muy usado para régimen de ciclo."
    verificado: true
    evidencia: "FRED CFNAI -> 711 obs, first 1967-03-01:-0.35 last 2026-05-01:-0.1 (upd 2026-06-26)."
    url: "https://fred.stlouisfed.org/series/CFNAI"

  - nombre_interno: CFNAIMA3
    descripcion: "CFNAI, media móvil de 3 meses. La versión con el umbral oficial de recesión."
    fuente: fred
    id: "CFNAIMA3"
    auth: FRED_API_KEY
    inicio_verificado: "1967-05-01"
    granularidad: mensual
    pista: ambas
    rol: core
    relevancia_regimen: "Umbral oficial Chicago Fed: CFNAIMA3 < -0.70 = recesión en curso; > +0.70 desde recesión = recuperación. Min -7.54 (2020-04), -2.02 (2008-11). Regla de régimen directa."
    verificado: true
    evidencia: "FRED CFNAIMA3 -> 709 obs, first 1967-05-01:-0.3 last 2026-05-01:-0.03 (upd 2026-06-26). Min -7.54 en 2020-04, -2.02 en 2008-11 (ambos < -0.7)."
    url: "https://fred.stlouisfed.org/series/CFNAIMA3"

  - nombre_interno: CFNAIDIFF
    descripcion: "CFNAI Diffusion Index (media móvil 3m del índice de difusión de los 85 subindicadores)."
    fuente: fred
    id: "CFNAIDIFF"
    auth: FRED_API_KEY
    inicio_verificado: "1967-05-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Amplitud de la debilidad/fortaleza: cuántos indicadores contribuyen negativamente. Confirma si la señal de CFNAI es generalizada."
    verificado: true
    evidencia: "FRED CFNAIDIFF -> 709 obs, first 1967-05-01:-0.17 last 2026-05-01:-0.01 (upd 2026-06-26)."
    url: "https://fred.stlouisfed.org/series/CFNAIDIFF"

  - nombre_interno: IPMAN
    descripcion: "Producción Industrial: solo Manufacturas (NAICS). Núcleo cíclico de INDPRO sin utilities/minería."
    fuente: fred
    id: "IPMAN"
    auth: FRED_API_KEY
    inicio_verificado: "1972-01-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Actividad manufacturera pura; más cíclica que el INDPRO total. Complementa el PMI-proxy."
    verificado: true
    evidencia: "FRED IPMAN -> 654 obs, first 1972-01-01:36.04 last 2026-06-01:98.70 (upd 2026-07-17)."
    url: "https://fred.stlouisfed.org/series/IPMAN"

  - nombre_interno: TCU
    descripcion: "Utilización de la Capacidad: índice total. % de capacidad productiva en uso."
    fuente: fred
    id: "TCU"
    auth: FRED_API_KEY
    inicio_verificado: "1967-01-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Holgura industrial; cae bruscamente en recesión y presiona inflación cuando está alta. Señal de fase del ciclo."
    verificado: true
    evidencia: "FRED TCU -> 714 obs, first 1967-01-01:89.39 last 2026-06-01:76.09 (upd 2026-07-17)."
    url: "https://fred.stlouisfed.org/series/TCU"

  - nombre_interno: USPHCI
    descripcion: "US Coincident Activity Index (Philadelphia Fed). Índice coincidente de actividad económica nacional."
    fuente: fred
    id: "USPHCI"
    auth: FRED_API_KEY
    inicio_verificado: "1979-01-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Resumen coincidente del estado de la economía (empleo, horas, ventas, paro). Alternativa/confirmación de CFNAI."
    verificado: true
    evidencia: "FRED USPHCI -> 569 obs, first 1979-01-01:44.41 last 2026-05-01:149.17 (upd 2026-06-26)."
    url: "https://fred.stlouisfed.org/series/USPHCI"

  # ============ FOCO: PMI (proxy Fed regionales, ISM no está gratis) ============
  - nombre_interno: PMI_PROXY_PHILLY
    descripcion: "Philadelphia Fed Manufacturing Business Outlook — Current General Activity (índice de difusión). Mejor proxy libre del PMI manufacturero por profundidad histórica."
    fuente: fred
    id: "GACDFSA066MSFRBPHI"
    auth: FRED_API_KEY
    inicio_verificado: "1968-05-01"
    granularidad: mensual
    pista: ambas
    rol: core
    relevancia_regimen: "Difusión manufacturera >0 = expansión, <0 = contracción (análogo al 50 del PMI). +55 años, cubre todas las recesiones. Sustituto del ISM PMI, que NO está en FRED."
    verificado: true
    evidencia: "FRED GACDFSA066MSFRBPHI -> mensual first 1968-05-01 last 2026-07-01 (upd 2026-07-16). ID correcto (GACDFSA066MSFRBPHIL da HTTP 400)."
    url: "https://fred.stlouisfed.org/series/GACDFSA066MSFRBPHI"

  - nombre_interno: PMI_PROXY_EMPIRE
    descripcion: "Empire State (NY Fed) Manufacturing Survey — Current General Business Conditions (difusión). Proxy PMI regional."
    fuente: fred
    id: "GACDISA066MSFRBNY"
    auth: FRED_API_KEY
    inicio_verificado: "2001-07-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Pulso manufacturero regional de alta visibilidad (primer dato del mes). Promedio con Philly/Dallas = proxy PMI nacional."
    verificado: true
    evidencia: "FRED GACDISA066MSFRBNY -> 301 obs, first 2001-07-01:-13.30 last 2026-07-01:15.6 (upd 2026-07-15)."
    url: "https://fred.stlouisfed.org/series/GACDISA066MSFRBNY"

  - nombre_interno: PMI_PROXY_DALLAS
    descripcion: "Dallas Fed Manufacturing Survey — Current General Business Activity (difusión). Proxy PMI regional."
    fuente: fred
    id: "BACTSAMFRBDAL"
    auth: FRED_API_KEY
    inicio_verificado: "2004-06-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Actividad manufacturera regional (Texas/energía); complementa Philly y Empire en el proxy PMI compuesto."
    verificado: true
    evidencia: "FRED BACTSAMFRBDAL -> 265 obs, first 2004-06-01:47.6 last 2026-06-01:0.0 (upd 2026-06-29)."
    url: "https://fred.stlouisfed.org/series/BACTSAMFRBDAL"

  # ============ FOCO: INFLACIÓN ============
  - nombre_interno: CPIAUCSL
    descripcion: "CPI para todos los consumidores urbanos, todos los ítems, ajustado estacionalmente. Nivel de precios headline."
    fuente: fred
    id: "CPIAUCSL"
    auth: FRED_API_KEY
    inicio_verificado: "1947-01-01"
    granularidad: mensual
    pista: ambas
    rol: core
    relevancia_regimen: "Régimen inflacionario (YoY): determina la correlación acción-bono y el tipo de risk-off. +75 años cubre la gran inflación de los 70-80."
    verificado: true
    evidencia: "FRED CPIAUCSL -> 953 obs, first 1947-01-01:21.48 last 2026-06-01:332.568 (upd 2026-07-14)."
    url: "https://fred.stlouisfed.org/series/CPIAUCSL"

  - nombre_interno: CPILFESL
    descripcion: "CPI core (todos los ítems menos alimentos y energía), ajustado estacionalmente."
    fuente: fred
    id: "CPILFESL"
    auth: FRED_API_KEY
    inicio_verificado: "1957-01-01"
    granularidad: mensual
    pista: ambas
    rol: core
    relevancia_regimen: "Inflación subyacente/persistente; filtra el ruido de energía. Núcleo del régimen inflacionario para la reacción de la Fed."
    verificado: true
    evidencia: "FRED CPILFESL -> 833 obs, first 1957-01-01:28.5 last 2026-06-01:336.065 (upd 2026-07-14)."
    url: "https://fred.stlouisfed.org/series/CPILFESL"

  - nombre_interno: PCEPI
    descripcion: "PCE Price Index (deflactor del gasto en consumo personal), headline. Medida de inflación preferida de la Fed."
    fuente: fred
    id: "PCEPI"
    auth: FRED_API_KEY
    inicio_verificado: "1959-01-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Inflación de referencia para el objetivo del 2% de la Fed; canasta más amplia que el CPI. Régimen de política monetaria."
    verificado: true
    evidencia: "FRED PCEPI -> 809 obs, first 1959-01-01:15.164 last 2026-05-01:131.527 (upd 2026-06-25)."
    url: "https://fred.stlouisfed.org/series/PCEPI"

  - nombre_interno: PCEPILFE
    descripcion: "PCE core (deflactor PCE menos alimentos y energía). LA medida que mira la Fed para decidir tipos."
    fuente: fred
    id: "PCEPILFE"
    auth: FRED_API_KEY
    inicio_verificado: "1959-01-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Inflación subyacente PCE; determinante directo del régimen de tipos de la Fed y por tanto del entorno de mercado."
    verificado: true
    evidencia: "FRED PCEPILFE -> 809 obs, first 1959-01-01:15.501 last 2026-05-01:130.082 (upd 2026-06-25)."
    url: "https://fred.stlouisfed.org/series/PCEPILFE"

  - nombre_interno: MICH
    descripcion: "Expectativas de inflación a 1 año, encuesta Universidad de Michigan."
    fuente: fred
    id: "MICH"
    auth: FRED_API_KEY
    inicio_verificado: "1978-01-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Expectativas de inflación del consumidor; su desanclaje marca regímenes de estrés inflacionario."
    verificado: true
    evidencia: "FRED MICH -> 581 obs, first 1978-01-01:5.2 last 2026-05-01:4.8 (upd 2026-06-26)."
    url: "https://fred.stlouisfed.org/series/MICH"

  - nombre_interno: T10YIE
    descripcion: "10-Year Breakeven Inflation Rate (expectativas de inflación implícitas en TIPS vs nominal). DIARIA. Solapa con curva/mercados."
    fuente: fred
    id: "T10YIE"
    auth: FRED_API_KEY
    inicio_verificado: "2003-01-02"
    granularidad: diaria
    pista: B
    rol: enricher
    relevancia_regimen: "Expectativas de inflación de mercado, alta frecuencia. Caídas bruscas = shock deflacionario/risk-off (2008, 2020). NOTA: solapa con el agente de tipos/curva."
    verificado: true
    evidencia: "FRED T10YIE -> 5889 obs diarias, first 2003-01-02:1.64 last 2026-07-17:2.24 (upd 2026-07-17)."
    url: "https://fred.stlouisfed.org/series/T10YIE"

  # ============ SENTIMIENTO / ALTA FRECUENCIA / INCERTIDUMBRE ============
  - nombre_interno: UMCSENT
    descripcion: "Índice de Sentimiento del Consumidor de la Universidad de Michigan."
    fuente: fred
    id: "UMCSENT"
    auth: FRED_API_KEY
    inicio_verificado: "1952-11-01"
    granularidad: mensual
    pista: ambas
    rol: enricher
    relevancia_regimen: "Confianza del consumidor; se hunde en recesiones y crisis. Componente blando líder del ciclo."
    verificado: true
    evidencia: "FRED UMCSENT -> 673 obs (con hueco pre-1978), first 1952-11-01:86.2 last 2026-05-01:44.8 (upd 2026-06-26)."
    url: "https://fred.stlouisfed.org/series/UMCSENT"

  - nombre_interno: WEI
    descripcion: "Weekly Economic Index (Lewis-Mertens-Stock, NY Fed). Nowcast de actividad económica de frecuencia SEMANAL."
    fuente: fred
    id: "WEI"
    auth: FRED_API_KEY
    inicio_verificado: "2008-01-05"
    granularidad: semanal
    pista: B
    rol: enricher
    relevancia_regimen: "Puente de alta frecuencia entre lo diario de mercado y lo mensual macro; escala en % de crecimiento anual del PIB. Se desplomó a -11 en abril-2020."
    verificado: true
    evidencia: "FRED WEI -> 967 obs semanales, first 2008-01-05:1.95 last 2026-07-11:2.61 (upd 2026-07-16)."
    url: "https://fred.stlouisfed.org/series/WEI"

  - nombre_interno: GEPUCURRENT
    descripcion: "Global Economic Policy Uncertainty Index (Baker-Bloom-Davis), current-price GDP weighted."
    fuente: fred
    id: "GEPUCURRENT"
    auth: FRED_API_KEY
    inicio_verificado: "1997-01-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Incertidumbre de política económica; picos coinciden con crisis (2008, Brexit, COVID, guerras comerciales). Proxy macro de risk-off."
    verificado: true
    evidencia: "FRED GEPUCURRENT -> 354 obs, first 1997-01-01:77.62 last 2026-06-01:278.23 (upd 2026-07-08)."
    url: "https://fred.stlouisfed.org/series/GEPUCURRENT"

  # ============ CÍCLICOS / VIVIENDA / PEDIDOS (contexto) ============
  - nombre_interno: HOUST
    descripcion: "Housing Starts (viviendas iniciadas), tasa anual ajustada estacionalmente."
    fuente: fred
    id: "HOUST"
    auth: FRED_API_KEY
    inicio_verificado: "1959-01-01"
    granularidad: mensual
    pista: ambas
    rol: enricher
    relevancia_regimen: "Sector más sensible a tipos; gira antes en el ciclo. Cae con fuerza antes de recesiones (líder clásico)."
    verificado: true
    evidencia: "FRED HOUST -> 810 obs, first 1959-01-01:1657 last 2026-06-01:1427.0 (upd 2026-07-17)."
    url: "https://fred.stlouisfed.org/series/HOUST"

  - nombre_interno: PERMIT
    descripcion: "Building Permits (permisos de construcción), tasa anual ajustada estacionalmente. Componente del índice líder."
    fuente: fred
    id: "PERMIT"
    auth: FRED_API_KEY
    inicio_verificado: "1960-01-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Aún más adelantado que HOUST; parte del Conference Board LEI. Señal temprana de giro del ciclo."
    verificado: true
    evidencia: "FRED PERMIT -> 798 obs, first 1960-01-01:1092 last 2026-06-01:1367.0 (upd 2026-07-17)."
    url: "https://fred.stlouisfed.org/series/PERMIT"

  - nombre_interno: RSAFS
    descripcion: "Advance Retail Sales: Retail and Food Services, ajustado estacionalmente."
    fuente: fred
    id: "RSAFS"
    auth: FRED_API_KEY
    inicio_verificado: "1992-01-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Gasto del consumidor (70% del PIB); su contracción confirma debilidad de demanda en recesión."
    verificado: true
    evidencia: "FRED RSAFS -> 414 obs, first 1992-01-01:159177 last 2026-06-01:768553 (upd 2026-07-16)."
    url: "https://fred.stlouisfed.org/series/RSAFS"

  - nombre_interno: NEWORDER
    descripcion: "Manufacturers' New Orders: Nondefense Capital Goods excl. Aircraft (core capex orders), ajustado estacionalmente."
    fuente: fred
    id: "NEWORDER"
    auth: FRED_API_KEY
    inicio_verificado: "1992-02-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Proxy de inversión empresarial futura; core capex cae antes de las recesiones (líder de inversión)."
    verificado: true
    evidencia: "FRED NEWORDER -> 412 obs, first 1992-02-01:33857 last 2026-05-01:83951 (upd 2026-07-02)."
    url: "https://fred.stlouisfed.org/series/NEWORDER"

  - nombre_interno: TOTALSA
    descripcion: "Total Vehicle Sales (ventas totales de vehículos ligeros), tasa anual."
    fuente: fred
    id: "TOTALSA"
    auth: FRED_API_KEY
    inicio_verificado: "1976-01-01"
    granularidad: mensual
    pista: B
    rol: enricher
    relevancia_regimen: "Consumo discrecional sensible a crédito/tipos; se hunde en recesión. Bien cíclico clásico."
    verificado: true
    evidencia: "FRED TOTALSA -> 606 obs, first 1976-01-01:12.814 last 2026-06-01:16.949 (upd 2026-07-03)."
    url: "https://fred.stlouisfed.org/series/TOTALSA"

  # ============ PIB (contexto trimestral) ============
  - nombre_interno: GDPC1
    descripcion: "Producto Interior Bruto real (encadenado 2017), trimestral, ajustado estacionalmente."
    fuente: fred
    id: "GDPC1"
    auth: FRED_API_KEY
    inicio_verificado: "1947-01-01"
    granularidad: trimestral
    pista: A
    rol: enricher
    relevancia_regimen: "Medida definitiva de recesión (dos trimestres de caída = definición popular). Baja frecuencia, contexto de fondo."
    verificado: true
    evidencia: "FRED GDPC1 -> 317 obs trimestrales, first 1947-01-01:2182.681 last 2026-01-01:24180.419 (upd 2026-06-25)."
    url: "https://fred.stlouisfed.org/series/GDPC1"

  - nombre_interno: GDP_GROWTH_QOQ
    descripcion: "Crecimiento del PIB real, % trimestral anualizado (headline del titular de prensa)."
    fuente: fred
    id: "A191RL1Q225SBEA"
    auth: FRED_API_KEY
    inicio_verificado: "1947-04-01"
    granularidad: trimestral
    pista: A
    rol: enricher
    relevancia_regimen: "Tasa de crecimiento del ciclo; negativa en recesiones. Ancla trimestral del régimen macro."
    verificado: true
    evidencia: "FRED A191RL1Q225SBEA -> 316 obs, first 1947-04-01:-1.0 last 2026-01-01:2.1 (upd 2026-06-25)."
    url: "https://fred.stlouisfed.org/series/A191RL1Q225SBEA"

  # ============ VALIDACIÓN (ground-truth de ciclo, ya hecho) ============
  - nombre_interno: USREC
    descripcion: "NBER-based Recession Indicator (1 = recesión, 0 = expansión), fechas oficiales del ciclo económico de EE.UU."
    fuente: fred
    id: "USREC"
    auth: FRED_API_KEY
    inicio_verificado: "1854-12-01"
    granularidad: mensual
    pista: validacion
    rol: validation
    relevancia_regimen: "GROUND-TRUTH PRINCIPAL del ciclo económico, +170 años. Fechas oficiales NBER de pico-valle. Referencia laxa para etiquetar regímenes de recesión (con el caveat de que NBER data con retraso)."
    verificado: true
    evidencia: "FRED USREC -> 2059 obs mensuales, first 1854-12-01:1 last 2026-06-01:0 (upd 2026-07-01). Cubre todas las recesiones desde 1854."
    url: "https://fred.stlouisfed.org/series/USREC"

  - nombre_interno: RECPROUSM156N
    descripcion: "Smoothed U.S. Recession Probabilities (Chauvet-Piger). Probabilidad de recesión estimada por modelo de cambio de régimen (Markov-switching)."
    fuente: fred
    id: "RECPROUSM156N"
    auth: FRED_API_KEY
    inicio_verificado: "1967-06-01"
    granularidad: mensual
    pista: validacion
    rol: validation
    relevancia_regimen: "Ground-truth probabilístico: un modelo de regímenes ya publicado sobre 4 series coincidentes. Referencia directa para comparar la salida del detector (es literalmente un régimen-switching benchmark)."
    verificado: true
    evidencia: "FRED RECPROUSM156N -> 708 obs, first 1967-06-01:0.92 last 2026-05-01:0.54 (upd 2026-07-01)."
    url: "https://fred.stlouisfed.org/series/RECPROUSM156N"

  - nombre_interno: JHDUSRGDPBR
    descripcion: "GDP-based Recession Indicator (Hamilton). Índice/probabilidad de recesión derivada del PIB por el modelo de Hamilton."
    fuente: fred
    id: "JHDUSRGDPBR"
    auth: FRED_API_KEY
    inicio_verificado: "1967-10-01"
    granularidad: trimestral
    pista: validacion
    rol: validation
    relevancia_regimen: "Segundo benchmark de recesión basado en modelo (PIB); ground-truth alternativo a NBER/Chauvet-Piger. Trimestral."
    verificado: true
    evidencia: "FRED JHDUSRGDPBR -> 233 obs trimestrales, first 1967-10-01:0 last 2025-10-01:0.0 (upd 2026-04-30)."
    url: "https://fred.stlouisfed.org/series/JHDUSRGDPBR"

  - nombre_interno: NFCI
    descripcion: "Chicago Fed National Financial Conditions Index (headline). Índice compuesto de condiciones financieras. SEMANAL."
    fuente: fred
    id: "NFCI"
    auth: FRED_API_KEY
    inicio_verificado: "1971-01-08"
    granularidad: semanal
    pista: validacion
    rol: validation
    relevancia_regimen: ">0 = condiciones más restrictivas que la media = estrés. Ground-truth semanal de régimen financiero. NOTA: solapa con NFCICREDIT del agente de crédito (este es el índice headline completo)."
    verificado: true
    evidencia: "FRED NFCI -> 2897 obs semanales, first 1971-01-08:0.598 last 2026-07-10:-0.538 (upd 2026-07-15)."
    url: "https://fred.stlouisfed.org/series/NFCI"

  # ============ FALLBACKS / NO DISPONIBLES (probados) ============
  - nombre_interno: PMI_ISM
    descripcion: "ISM Manufacturing PMI (y subíndices new orders/production/employment) — el PMI 'de verdad'. NO disponible gratis."
    fuente: fred
    id: "NAPM / ISM (retirados)"
    auth: FRED_API_KEY
    inicio_verificado: null
    granularidad: mensual
    pista: B
    rol: fallback
    relevancia_regimen: "Sería el mejor indicador manufacturero adelantado (>50 = expansión). Usar los Fed regionales (Philly/Empire/Dallas) como proxy libre."
    verificado: false
    evidencia: "IDs NAPM, NAPMPI, NAPMNOI, NAPMEI, ISM, ISMMAN -> HTTP 400. Búsqueda FRED 'PMI'/'ISM Manufacturing PMI'/'Purchasing Managers Index' -> cero resultados relevantes. ISM retiró sus series de FRED por licencia; S&P Global PMI es propietario."
    url: "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/"

  - nombre_interno: LEADING_INDEX_CB
    descripcion: "Conference Board Leading Economic Index (LEI). Indicador líder compuesto clásico."
    fuente: fred
    id: "USSLIND"
    auth: FRED_API_KEY
    inicio_verificado: "1982-01-01"
    granularidad: mensual
    pista: B
    rol: fallback
    relevancia_regimen: "Índice líder muy usado, PERO DESCONTINUADO en FRED (Conference Board lo retiró). Sin continuación viva gratis."
    verificado: true
    evidencia: "FRED USSLIND -> 458 obs, first 1982-01-01:-0.89 last 2020-02-01:1.72 (upd 2020-04-14). Congelada desde feb-2020: NO viva."
    url: "https://fred.stlouisfed.org/series/USSLIND"

  - nombre_interno: OECD_CLI_US
    descripcion: "OECD Composite Leading Indicator para EE.UU. (amplitud ajustada, normalizado)."
    fuente: fred
    id: "USALOLITONOSTSAM"
    auth: FRED_API_KEY
    inicio_verificado: "1955-01-01"
    granularidad: mensual
    pista: B
    rol: fallback
    relevancia_regimen: "Indicador líder OECD con historia larga, PERO rancio: última obs 2024-01. Retraso de +2 años -> no viable como serie viva; solo histórico."
    verificado: true
    evidencia: "FRED USALOLITONOSTSAM -> 829 obs, first 1955-01-01:101.30 last 2024-01-01:99.85 (metadata upd 2025-11-17 pero datos parados en 2024-01)."
    url: "https://fred.stlouisfed.org/series/USALOLITONOSTSAM"
```

---

## Recomendación de priorización para el pipeline

1. **Espina macro profunda (imprescindible, cubre todas las crisis, Pista A):** `INDPRO` (1919),
   `UNRATE`/`PAYEMS` (1948/1939), `CPIAUCSL`/`CPILFESL` (1947/1957), `CFNAI`/`CFNAIMA3` (1967). Todo
   FRED, vivo. Son el contexto de ciclo real que subyace a los regímenes de mercado.

2. **Señales de recesión de una cifra (features directas de régimen):**
   - **`SAHMREALTIME` ≥ 0.50** (flag de recesión en tiempo real, sin look-ahead — la correcta).
   - **`CFNAIMA3` < −0.70** (umbral oficial Chicago Fed).
   - **`RECPROUSM156N`** como probabilidad de recesión ya modelada (también sirve de validación).

3. **Alta frecuencia dentro del macro (puentes semanales, Pista B):** `ICSA`/`IC4WSA`/`CCSA` (claims,
   1967+, semanal), `WEI` (nowcast semanal 2008+), `NFCI` (condiciones financieras semanal 1971+).
   Son lo más cercano a "macro casi diario" y anticipan giros del ciclo.

4. **Régimen inflacionario (clave para correlación acción-bono):** `CPIAUCSL`/`CPILFESL` YoY,
   `PCEPILFE` (medida Fed), `MICH`/`T10YIE` (expectativas). Distinguen risk-off deflacionario (2008,
   2020) de risk-off inflacionario (2022).

5. **Pulso manufacturero tipo-PMI (proxy libre):** `PMI_PROXY_PHILLY` (1968+, el de más historia) +
   Empire/Dallas. **El ISM PMI no está gratis** (hallazgo 1); documentar como limitación.

6. **Validación (ground-truth laxo de ciclo):** `USREC` (NBER, 1854), `RECPROUSM156N` (Chauvet-Piger),
   `JHDUSRGDPBR` (Hamilton), `NFCI`. Los tres primeros son benchmarks de recesión; `RECPROUSM156N` es
   literalmente un modelo de cambio de régimen ya publicado — comparación directa para el detector.

7. **Deuda técnica / caveats honestos:**
   - **Revisiones (vintages):** FRED sirve datos revisados; para backtest estricto point-in-time haría
     falta ALFRED. `SAHMREALTIME`, `WEI` y las encuestas de difusión son lo más "as-of". Documentar.
   - **`USSLIND`** (Conference Board LEI) y **`OECD_CLI_US`** están **muertas/rancias** en FRED — no
     usarlas como series vivas.
   - **Frecuencia mixta:** alinear todo a un calendario común (p.ej. mensual, con forward-fill causal
     de las semanales/trimestrales) y respetar el **lag de publicación** (CPI/paro salen ~2-4 semanas
     tras el mes de referencia; el PIB, semanas después del trimestre).

**Fuentes consultadas / probadas:**
[FRED API series & observations (todas verificadas 2026-07-18)](https://fred.stlouisfed.org/docs/api/fred/) ·
[Nota metodológica Sahm real-time vs current](https://fred.stlouisfed.org/series/SAHMREALTIME) ·
[Chicago Fed CFNAI (umbral −0.70)](https://www.chicagofed.org/publications/cfnai/index) ·
[ISM Report on Business (PMI, propietario, no en FRED)](https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/)
