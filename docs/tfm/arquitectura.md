# Arquitectura del sistema

Contexto: [README del TFM](README.md). Decisión de encuadre: [ADR-005](../decisions/ADR-005-reencuadre-tfm-multiagente.md).

## 1. Flujo general

```
FUENTES -> DATOS INDEXADOS -> TOOLS -> AGENTES -> SALIDA
```

| Capa | Contenido |
|---|---|
| **Fuentes** | Macro y política (Fed, FOMC, BCE, FMI) · empresas (SEC 10-K, 10-Q, calls) · riesgo sistémico (BIS, FSR, papers) · noticias y alternativos (GDELT, Polymarket, RSS) · mercado y macro (precios, volatilidad, crédito, tipos) |
| **Datos indexados** | Cuatro índices vectoriales con troceado distinto según el tipo de documento: **Macro** (por párrafo), **Equity** (por sección), **Risk** (trozos largos), **News** (trozos cortos); más una **base de series** con fecha de publicación |
| **Tools** | Caja de herramientas (§4): el RAG es una tool más, junto a datos y cálculos deterministas |
| **Agentes** | Macro, Equity, Risk, Portfolio y CIO, orquestados con LangGraph; deciden qué consultar y pueden encadenar búsquedas |
| **Salida** | Cartera y memo de inversión con registro de cada llamada y su fuente |

El detector de régimen (Fase 1) es la pieza que alimenta la tool `get_regimen`.

## 2. TimeGate dentro de cada tool

Todas las tools reciben la fecha de decisión y **no devuelven nada posterior a ella**. El mismo código sirve para el
backtest (fecha simulada) y para el modo en vivo (hoy). Consecuencias:

- El TimeGate no es un filtro externo sino parte de cada tool: un agente no puede saltárselo.
- Para series de mercado ya está implementado y testeado en la Fase 1: retrasos de publicación por serie
  (`regimenes.features.lags.LAG_PUBLICACION`, [ADR-003](../decisions/ADR-003-causalidad-calendario-estado-ranking.md))
  aplicados a la serie cruda antes de transformar, y validación *walk-forward*.
- Los documentos se indexan por su **fecha real de publicación** (p. ej. las actas del FOMC, unas tres semanas después de la reunión).
- **Limitación declarada:** los lags fijos no equivalen a los *vintages* en tiempo real (ALFRED); las series macro revisadas pueden
  contener información posterior. ALFRED es más exacto pero de coste alto (ADR-003); su uso en `get_serie` es una decisión de la Fase 2.

## 3. Anatomía de un agente

| Pieza | Pregunta | Qué es | Analogía |
|---|---|---|---|
| **Conocimiento** (datos indexados) | ¿Qué puede consultar? | Documentos y series preparados offline: troceados, vectorizados y con fecha de publicación. No hace nada por sí mismo | La biblioteca y el archivo del analista |
| **Tools** | ¿Qué puede hacer? | Funciones que el agente decide llamar: buscar en un índice, leer una serie, optimizar una cartera. Todas reciben la fecha | El terminal y la calculadora |
| **Skills** | ¿Cómo debe hacerlo? | Instrucciones de experto escritas por nosotros, que el agente carga cuando las necesita: pasos, criterios y formato de salida | El oficio y los manuales internos |

Conocimiento offline no es lo mismo que skills. Las tres piezas se pueden versionar y evaluar por separado.
Ejemplo de skill: «leer un comunicado FOMC» (comparar con el anterior, detectar cambios de lenguaje, clasificar hawkish/dovish y citar).

## 4. Catálogo de tools

Cada tool aplica el TimeGate con `fecha`. Las marcadas **D** son deterministas (el LLM no calcula pesos).

| Tool | Fuente / qué devuelve | Macro | Equity | Risk | Portfolio | CIO |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `buscar_macro(q, fecha)` | Fed, FOMC, BCE, FMI, Beige Book | x | | | | |
| `buscar_filings(tk, q, fecha)` | SEC EDGAR: 10-K, 10-Q, 8-K, earnings calls | | x | | | |
| `buscar_riesgo(q, fecha)` | BIS Quarterly, FSR Fed/BCE, papers de crisis | | | x | | |
| `buscar_noticias(q, fecha)` | GDELT, RSS Reuters/FT | x | x | | | |
| `get_serie(id, fecha)` | FRED/ALFRED: VIX, spreads, curva, IPC, paro... | x | | x | | |
| `get_precios(tk, fecha)` | yfinance (ETF y acciones) | | x | | x | |
| `get_fundamentales(tk, fecha)` | SEC XBRL: ventas, márgenes, deuda | | x | | | |
| `get_polymarket(evento, fecha)` | Probabilidad implícita de eventos de cola | | | x | | |
| `get_regimen(fecha)` | Detector de régimen (Fase 1): estado, probabilidad, días | x | | x | x | |
| `composite_score(bucket, fecha)` **D** | Ranking del bucket: fundamental 40 · momentum 35 · calidad 25 | | x | | | |
| `riesgo_cartera(pesos, fecha)` **D** | Vol ex-ante, VaR, estrés por escenarios | | | x | x | |
| `optimizar_cartera(motor, fecha)` **D** | Risk parity, mínima varianza, máxima diversificación | | | | x | |
| `comprobar_limites(pesos)` **D** | Límites institucionales y alertas | | | | x | x |
| `ajustar_pesos(Δ ≤ 5 pp)` | Desviación acotada y registrada | | | | | x |
| `verificar_citas(memo)` | Cada afirmación respaldada por una fuente recuperada | | | | | x |

Los parámetros del `composite_score` (40/35/25) vienen de la propuesta y su universo está **pendiente de análisis**
(ver [LINEAS_ABIERTAS.md](LINEAS_ABIERTAS.md)). Los datos alternativos candidatos a nuevas tools están en
[DATOS_ALTERNATIVOS.md](DATOS_ALTERNATIVOS.md).

## 5. Contrato provisional de `get_regimen(fecha)`

Salida de la Fase 1; se congela en el notebook `19_decision_final` y se valida en `20_pseudolive`.

| Campo | Contenido |
|---|---|
| `estado` | Uno de los estados fijados por el **detector final** (futura ADR-006). Hipótesis de trabajo: calma / alerta / crisis |
| `probabilidad` | Confianza del detector en ese estado |
| `dias_en_estado` | Días consecutivos en el estado actual |

- Correspondencia de trabajo con la máquina de fusión (`regimenes.fusion`): calma ≈ *normal*, alerta ≈ *vigilancia*, crisis ≈ *confirmado*.
- Solo usa información publicada hasta `fecha` (TimeGate).
- Los estados definitivos y la definición exacta de la probabilidad **no están decididos**.

## 6. Los cinco agentes

Los parámetros numéricos de este apartado (motores, vol objetivo, límites, ±5 pp, composite 40/35/25) proceden de la
presentación al tutor: son **diseño propuesto, sujeto a revisión**. Los ejemplos del 16-mar-2020 de la presentación son
**ilustrativos** (maqueta, no resultados).

### Macro Agent — el economista

*¿Qué están haciendo los bancos centrales y en qué punto del ciclo estamos?*

- **Tools:** `buscar_macro`, `get_serie`, `buscar_noticias`, `get_regimen`.
- **Skills:** leer un comunicado FOMC/BCE (cambios de lenguaje, hawkish/dovish), leer la curva de tipos y el ciclo, citar siempre la frase fuente.
- **Condicionado por el régimen:** calma → ciclo, inflación y sesgo de política; alerta → señales de giro (curva, crédito, empleo); crisis → respuesta de emergencia y liquidez.

### Equity Agent — el analista de empresas

*Dentro de lo que el régimen favorece, ¿qué empresas elegir y cuáles evitar?*

- **Tools:** `composite_score`, `buscar_filings`, `get_fundamentales`, `get_precios`, `buscar_noticias`.
- **Skills:** comparar «Risk Factors» entre dos 10-K consecutivos, detectar red flags contables (accruals, auditor, reexpresiones), escribir una tesis de 3 líneas con citas.
- **Condicionado por el régimen:** calma → más peso a momentum y crecimiento; alerta → calidad (balance y estabilidad); crisis → solo defensivas, liquidez y poca deuda.

### Risk Agent — el gestor de riesgos

*¿Qué puede salir mal, cuánto podría costar y a qué episodio se parece esto?*

- **Tools:** `get_regimen`, `get_serie` (VIX, spread HY-IG, correlación acciones-bonos), `buscar_riesgo`, `get_polymarket`, `riesgo_cartera`.
- **Skills:** buscar análogos históricos, test de estrés por escenarios, leer probabilidades de Polymarket con cautela (volumen).
- **Condicionado por el régimen:** calma → vigilar riesgos de cola infravalorados; alerta → buscar análogos y estresar la cartera; crisis → proponer de-risking y su intensidad.

### Portfolio Agent — el constructor de carteras

*Con el régimen y los tres informes, ¿qué pesos concretos ponemos?*

- **Tools:** `get_regimen`, `optimizar_cartera` (covarianza Ledoit-Wolf, ventana de 252 días en el diseño propuesto), `riesgo_cartera`, `comprobar_limites`, `get_precios`.
- **Skills:** traducir los informes a restricciones (excluir, limitar), elegir el motor por regla sin discrecionalidad, explicar el turnover y los costes.
- **Condicionado por el régimen (diseño propuesto):**

| Régimen | Motor | Vol objetivo |
|---|---|:---:|
| Calma | Risk parity | 15 % |
| Alerta | Máxima diversificación | 10 % |
| Crisis | Mínima varianza, más liquidez | 7 % |

- **Límites (diseño propuesto):** 5 % por acción, 15 % por ETF, 30 % por sector, 5 % mínimo de liquidez. En el producto final los límites los fija el perfil del cliente ([CLIENTE_Y_CARTERAS.md](CLIENTE_Y_CARTERAS.md)).

### CIO Agent — el comité de inversión

*¿Cuál es la decisión final y cómo se explica a un comité?*

- **Tools:** estado del grafo (informes estructurados de los otros cuatro), `ajustar_pesos` (Δ ≤ 5 pp, desviación registrada con su motivo), `comprobar_limites`, `verificar_citas`.
- **Skills:** redactar un memo institucional (plantilla fija), contrastar agentes que discrepan, no afirmar nada sin cita.
- **Condicionado por el régimen:** calma → memo breve que prioriza el coste; alerta → contrasta opiniones discrepantes; crisis → memo detallado con plan de reentrada.
- **Salvaguarda:** ningún ajuste puede saltarse un límite (comprobación determinista).
