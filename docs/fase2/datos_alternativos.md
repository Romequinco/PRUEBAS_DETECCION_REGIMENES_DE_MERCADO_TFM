# Datos alternativos y fuentes de la Fase 2

Estado de todas las fuentes de esta página: **a investigar**. Nada de lo descrito está descargado ni validado. Los
detalles proceden de una primera consulta de documentación (octubre de 2026) y deben comprobarse antes de usarlos.
Contexto: [README](README.md) · [ARQUITECTURA](ARQUITECTURA.md) (TimeGate, catálogo de tools).

**Regla común de TimeGate:** cada dato se indexa por su **fecha de publicación** (no la del hecho), igual que
`LAG_PUBLICACION` en [ADR-003](../decisions/ADR-003-causalidad-calendario-estado-ranking.md). Una tool solo devuelve
filas publicadas hasta la fecha de decisión.

## 1. Candidatas a datos alternativos

### Pelosi tracker (operaciones bursátiles del Congreso de EE. UU.)

| Aspecto | Detalle |
|---|---|
| Qué es | Operaciones de legisladores declaradas por la *STOCK Act* (informes PTR) |
| Fuente | [disclosures-clerk.house.gov](https://disclosures-clerk.house.gov), [efdsearch.senate.gov](https://efdsearch.senate.gov); agregadores Quiver Quantitative (API de pago), Capitol Trades; ETFs NANC y KRUZ como proxy |
| Profundidad histórica | Quiver desde 2016; los ETFs NANC/KRUZ desde febrero de 2023 |
| Retraso de publicación | Divulgación hasta 45 días tras la operación; importes en **rangos**, no exactos |
| TimeGate | Indexar por fecha de **publicación del PTR**, nunca por la de la operación |
| Uso previsto | Equity Agent (señal sobre tickers concretos); también Risk como contexto |
| Estado | A investigar |

### Divergencia Polymarket (información privilegiada en mercados de predicción)

| Aspecto | Detalle |
|---|---|
| Qué es | Diferencia entre la probabilidad implícita de un mercado de predicción y lo que dice el resto de la información, como indicio de operaciones informadas |
| Fuente | APIs gratuitas Gamma, Data y CLOB ([docs.polymarket.com](https://docs.polymarket.com)); literatura sobre *informed trading* en mercados de predicción (Harvard Law School Forum); herramientas de detección de Bitquery |
| Profundidad histórica | La propuesta menciona Polymarket desde 2021; la liquidez es **muy baja antes de 2024** |
| Retraso | Cotización continua; el problema no es el retraso sino la hora exacta |
| TimeGate | La divergencia exige la **hora exacta de la noticia** (p. ej. GDELT) para no comparar con información futura |
| Uso previsto | Risk Agent (`get_polymarket`), con cautela por volumen |
| Estado | A investigar |

### EWS — Early Warning System de jets privados (base del botón de cataclismo)

| Aspecto | Detalle |
|---|---|
| Qué es | Vigilancia de unos 31.800 jets privados a partir de ADS-B Exchange cada 30 minutos; z-score frente a una línea base semanal; niveles 1–5; cohorte militar desde junio de 2026 |
| Fuente | [ews.kylemcdonald.net](https://ews.kylemcdonald.net); código en [github.com/kylemcdonald/ews](https://github.com/kylemcdonald/ews); feed JSON público sin documentar |
| Profundidad histórica | Desde el 6 de octubre de 2025 |
| Retraso | Del orden de la frecuencia de muestreo (30 min); por comprobar |
| TimeGate | No aplica al backtest largo (histórico insuficiente); en vivo, marca temporal de cada muestra |
| Uso previsto | Botón de cataclismo ([CLIENTE_Y_CARTERAS.md](CLIENTE_Y_CARTERAS.md)); solo vivo/demo |
| Estado | A investigar (feed no documentado: estabilidad incierta) |

### Otras candidatas

| Fuente | Qué es | Retraso / TimeGate | Uso previsto | Estado |
|---|---|---|---|---|
| Form 4 de insiders (SEC EDGAR) | Operaciones de directivos en su propia empresa | Se presenta en 2 días hábiles; indexar por fecha de presentación | Equity | A investigar |
| Geopolitical Risk Index (Caldara-Iacoviello) y EPU | Índices de riesgo geopolítico e incertidumbre de política económica | Revisiones y publicación mensual; comprobar vintages | Macro, Risk | A investigar |
| Tono de GDELT | Sentimiento agregado de noticias | Marca temporal por evento; riesgo de reprocesado | Macro, Risk | A investigar |
| Put/call y SKEW de CBOE | Posicionamiento en opciones | Diario; comprobar disponibilidad histórica | Risk | A investigar |
| Google Trends | Interés de búsqueda | Normalización cambiante entre descargas: riesgo de mirar el futuro | Risk | A investigar |

## 2. Fuentes principales de la Fase 2

Las fuentes de la propuesta y la presentación al tutor, que alimentan los cuatro índices y la base de series.

| Fuente | Contenido | Alimenta | Notas |
|---|---|---|---|
| Fed, BCE, FMI | Actas, comunicados, Beige Book | Índice Macro, `buscar_macro` | Fecha real de publicación (las actas, unas 3 semanas después) |
| SEC EDGAR | 10-K, 10-Q, 8-K, earnings calls | Índice Equity, `buscar_filings` | Disponibles solo tras su fecha de presentación |
| BIS, FSR (Fed/BCE) | Informes de estabilidad y papers de crisis | Índice Risk, `buscar_riesgo` | Análogos históricos |
| GDELT, RSS (Reuters/FT) | Noticias y eventos | Índice News, `buscar_noticias` | Trozos cortos |
| FRED / ALFRED | Series macro y de mercado (curva, IPC, paro, VIX, spreads) | Base de series, `get_serie` | ALFRED da el *vintage* publicado; limitación declarada en la Fase 1 |
| yfinance | Precios de ETF y acciones | `get_precios`, `optimizar_cartera` | Ajuste por dividendos y splits a revisar |
| SEC XBRL | Ventas, márgenes, deuda | `get_fundamentales` | Por fecha de presentación |
| Polymarket | Probabilidad implícita de eventos | `get_polymarket` | Ver arriba |
