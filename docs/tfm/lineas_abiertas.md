# Líneas abiertas

Contexto: [README](README.md) · [ADR-005](../decisions/ADR-005-reencuadre-tfm-multiagente.md). Salvo el §4 (ya decidido), nada de lo que sigue está decidido.

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

La presentación al tutor no los menciona. **No están descartados ni confirmados** (ADR-005 §2.3). La Fase 2 ya está en
marcha, así que se analizan a medida que cada tema los toca: el plan de datos acota la ventana del backtest y el Tema 1
ya ha definido un universo de activos propio (ver [`plan_datos.md`](../fase2/plan_datos.md)).

| Elemento de la propuesta | Relación con el diseño actual |
|---|---|
| Universo en dos capas: 42 ETFs («Capa 1») y ~1.300 acciones («Capa 2») | Condiciona `composite_score` y el catálogo de activos; no confundir con la «Capa 1» de este repositorio (primera vuelta de detectores) |
| *Composite score* sobre ese universo | El diseño propuesto lo usa con ponderaciones 40/35/25 (fundamental, momentum, calidad) |
| Infraestructura AWS | Sin decisión de despliegue |
| Modo operacional | La presentación declara el modo en vivo como extra, tras un backtest robusto |
| Reglas de rebalanceo | Relacionadas con la memoria de cartera y el turnover |
| Ventanas de backtest 2000–2024 | Frente a la historia de 1962–2026 de la Fase 1; hay que reconciliar |

## 4. Dónde vive el código de las fases 2–5

**Decidido.** La Fase 2 en [ADR-007](../decisions/ADR-007-fase2-ingesta-y-docs.md) y las fases 3–5 en
[ADR-012](../decisions/ADR-012-estructura-repo.md) §2.1. Todo en este repositorio, con un solo `pyproject.toml` y un
paquete de primer nivel por capa:

| Paquete | Fase | Estado |
|---|---|---|
| `src/ingesta` | 2 · un subpaquete por tema ([`docs/fase2`](../fase2/README.md)) | en marcha |
| `src/herramientas` | 2 · contrato de tools tipado entre los datos y los agentes | creado ([README](../../src/herramientas/README.md)) |
| `src/memoria` | 2 · RAG con TimeGate | se crea al empezar |
| `src/agentes` | 3–4 · grafo LangGraph | se crea al empezar |
| `src/cartera` | 4–5 · motores, límites y backtest (numérico puro) | se crea al empezar |

Las dependencias entre paquetes van en un solo sentido (`regimenes`, `ingesta` → `memoria` → `herramientas` →
`agentes`) y las fases 3–5 importan solo de `herramientas`. Cada carpeta se crea en el primer PR que la necesite, nunca
vacía. Las librerías nuevas se instalan siempre con `-c constraints.txt`, para no alterar la huella de la Fase 1.
