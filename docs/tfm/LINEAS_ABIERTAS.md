# Líneas abiertas

Contexto: [README](README.md) · [ADR-005](../decisions/ADR-005-reencuadre-tfm-multiagente.md). Nada de lo que sigue está decidido.

## 1. System One decision agents (línea de investigación)

Modelos de decisión rápidos y tipados que, ante un estado y unas preguntas con formato fijo, devuelven respuestas con
probabilidad en una sola pasada (sin cadena de razonamiento larga). Aparecen en septiembre–octubre de 2026.

| Modelo | Origen | Notas | Enlaces |
|---|---|---|---|
| **Jev** | TypeSafe AI | API propietaria; integración con LangGraph (`langchain-typesafe`) | [typesafe.ai](https://typesafe.ai), [langchain.com/blog/building-a-harness-with-jev](https://www.langchain.com/blog/building-a-harness-with-jev) |
| **Clef / Clef-flash** | Cloudflare | Apache 2.0, Workers AI, versiones de 27B y 9B | [blog.cloudflare.com/clef-decision-models](https://blog.cloudflare.com/clef-decision-models/) |
| **Laya** | Convai Innovations | Encoder ModernBERT-large (421M), Apache 2.0, ejecución local | [huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) |

**Encaje posible:** un nodo «Sistema 1» rápido en el grafo de agentes para decisiones simples (¿ha cambiado el régimen?,
¿conviene rebalancear?, guardarraíl de idoneidad frente al perfil del cliente) que **escala al LLM («Sistema 2») cuando
duda**. Reduciría coste y latencia y daría salidas tipadas y auditables.

**Caveats:**

- Son modelos de 2026: pueden tener **conocimiento posterior a las fechas del backtest**, lo que viola el TimeGate si se
  usan sobre episodios históricos. Habría que acotarlos a entradas estructuradas sin identificadores temporales o
  limitarlos a uso en vivo.
- Dependencia de una API propietaria (Jev) frente a opciones abiertas y locales (Clef, Laya).
- Línea de investigación, no compromiso de implementación.

## 2. Integración opcional de Briefly

Existe una práctica previa, *Briefly*, en [`multimodal-market-briefer`](https://github.com/Romequinco/multimodal-market-briefer),
que cubre IA multimodal. **Si da tiempo, se integra al final** del trabajo. Hoy no forma parte de este repositorio.

## 3. Elementos de la propuesta pendientes de análisis

La presentación al tutor no los menciona. **No están descartados ni confirmados**; se analizarán en profundidad antes de la
Fase 2 (ADR-005 §2.3).

| Elemento de la propuesta | Relación con el diseño actual |
|---|---|
| Universo en dos capas: 42 ETFs («Capa 1») y ~1.300 acciones («Capa 2») | Condiciona `composite_score` y el catálogo de activos; no confundir con la «Capa 1» de este repositorio (primera vuelta de detectores) |
| *Composite score* sobre ese universo | El diseño propuesto lo usa con ponderaciones 40/35/25 (fundamental, momentum, calidad) |
| Infraestructura AWS | Sin decisión de despliegue |
| Modo operacional | La presentación declara el modo en vivo como extra, tras un backtest robusto |
| Reglas de rebalanceo | Relacionadas con la memoria de cartera y el turnover |
| Ventanas de backtest 2000–2024 | Frente a la historia de 1962–2026 de la Fase 1; hay que reconciliar |

## 4. Dónde vivirá el código de las fases 2–5

**Abierto.** Opciones: nuevos paquetes en este repositorio (junto a `regimenes`) o un repositorio aparte que consuma
`get_regimen`. Se decidirá en una ADR propia. Criterios a considerar: aislamiento de dependencias (LLM, vectores), caché y
reproducibilidad de la Fase 1, y despliegue.
