# Datos alternativos candidatos

Candidatas a datos alternativos pedidas en la tutoría ([ADR-005](../decisions/ADR-005-reencuadre-tfm-multiagente.md) §2.2).
Los detalles proceden de una primera consulta de documentación (octubre de 2026); las sondas del plan de datos ya
midieron algunas (Polymarket, EWS, informes del Congreso) y sus conclusiones prevalecen sobre el estado «a investigar»
de estas tablas. Su tratamiento (destino, TimeGate,
historia útil) está en el [plan de datos](plan_datos.md) §4 (Tema 4) y su implementación corresponde al
[Tema 4](../../src/ingesta/noticias_alt/README.md). Contexto: [`README.md`](README.md) (reparto de la Fase 2) ·
[`tfm/arquitectura.md`](../tfm/arquitectura.md) (TimeGate, catálogo de tools).

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
| Uso previsto | Botón de cataclismo ([`tfm/cliente_y_carteras.md`](../tfm/cliente_y_carteras.md)); solo vivo/demo |
| Estado | A investigar (feed no documentado: estabilidad incierta) |

### Otras candidatas

| Fuente | Qué es | Retraso / TimeGate | Uso previsto | Estado |
|---|---|---|---|---|
| Form 4 de insiders (SEC EDGAR) | Operaciones de directivos en su propia empresa | Se presenta en 2 días hábiles; indexar por fecha de presentación | Equity | A investigar |
| Geopolitical Risk Index (Caldara-Iacoviello) y EPU | Índices de riesgo geopolítico e incertidumbre de política económica | Revisiones y publicación mensual; comprobar vintages | Macro, Risk | A investigar |
| Tono de GDELT | Sentimiento agregado de noticias | Marca temporal por evento; riesgo de reprocesado | Macro, Risk | A investigar |
| Put/call y SKEW de CBOE | Posicionamiento en opciones | Diario; comprobar disponibilidad histórica | Risk | A investigar |
| Google Trends | Interés de búsqueda | Normalización cambiante entre descargas: riesgo de mirar el futuro | Risk | A investigar |

## 2. Fuentes principales

Las fuentes principales de la Fase 2 (bancos centrales, EDGAR, informes de riesgo, noticias, FRED/ALFRED, precios,
XBRL) ya no se listan aquí: su reparto por tema está en [`README.md`](README.md) §1 y su tratamiento, en el
[plan de datos](plan_datos.md) §4.
