# TFM — Multi-Agent RAG System for Regime-Aware Macro-Equity Intelligence

Visión del Trabajo Fin de Máster (MIAX): qué sistema se construye, por fases, y en qué estado está cada una. Es la
**única** copia de la visión: el resto de documentos enlaza aquí en lugar de repetirla. El mapa de toda la
documentación es el índice [`docs/README.md`](../README.md). Decisiones de encuadre: [ADR-005](../decisions/ADR-005-reencuadre-tfm-multiagente.md)
(el régimen es la Fase 1) y [ADR-007](../decisions/ADR-007-fase2-ingesta-y-docs.md) (dónde vive la Fase 2 y cómo se
ordena `docs/`).

## 1. La idea: leer, situar, decidir

Un analista humano no puede leer a tiempo cada acta de la Fed, cada 10-K y cada noticia. Un equipo de agentes LLM sí, y
si además sabe en qué régimen está el mercado puede aplicar el criterio de riesgo adecuado.

| Paso | Qué hace |
|---|---|
| **1 · Leer** | Información heterogénea (bancos centrales, 10-K/10-Q, informes de riesgo, noticias, mercados de predicción, series macro), cada fuente con su RAG especializado. |
| **2 · Situar** | Un detector cuantitativo dice en qué estado está el mercado (hipótesis de trabajo: calma, alerta o crisis; Fase 1). Condiciona todo lo demás. |
| **3 · Decidir** | Cartera con pesos por activo, construida con criterios de riesgo y explicada en un memo de inversión con citas a las fuentes. |

**No predecimos precios:** detectamos el estado del mercado, lo contextualizamos con evidencia y gestionamos el riesgo.
Predecir retornos a corto plazo es muy frágil fuera de muestra; detectar regímenes tiene base empírica sólida
(Hamilton, 1989) y el valor está en un proceso consistente y auditable.

## 2. El producto: una cartera explicada

El usuario recibe una **cartera** (pesos por activo) y un **investment memo** (comité semanal) con citas a las fuentes.
Cuatro rasgos lo distinguen:

1. **Cada afirmación cita su fuente.** El memo se audita frase a frase; se mide con *faithfulness* (RAGAS).
2. **El régimen cambia las reglas.** Cada estado tiene su motor de cartera y sus límites de riesgo.
3. **El LLM no tiene la última palabra en bruto.** El motor cuantitativo propone y el CIO Agent solo ajusta dentro de bandas registradas.
4. **Nada del futuro.** En el backtest cada agente solo ve lo que estaba publicado en esa fecha (TimeGate).

Arquitectura, tools y agentes: [arquitectura.md](arquitectura.md). Cliente, perfil y carteras: [cliente_y_carteras.md](cliente_y_carteras.md).

## 3. Plan de trabajo: cinco fases y su estado

Prioridad declarada: un backtest robusto y honesto antes que el modo operativo en vivo, que queda como extra.
Se construye de abajo arriba: primero las piezas que no dependen de un LLM, después los agentes, uno a uno y de principio a fin.

| Fase | Qué | Estado | Siguiente paso |
|:---:|---|---|---|
| 1 | **Régimen de mercado:** base de datos causal, comparativa de detectores, señal final como tool `get_regimen(fecha)` | **En curso.** Hechos: datos, preprocesado, benchmark de 12 detectores, fusiones y sintéticos (notebooks `00`–`17`). Pendientes: `18` (aumento), `19` (decisión final) y `20` (pseudolive), hoy esqueletos | Elegir la señal final (notebook 19, futura ADR-006) y publicarla como `get_regimen(fecha)` |
| 2 | **Ingesta, RAG y tools:** corpus con fecha de publicación, TimeGate dentro de cada tool, evaluación RAGAS | **En marcha** en `src/ingesta`, repartida en cuatro temas con una rama cada uno. **Tema 1 (mercados y series) completado** en la rama `fase2/t1-mercados-series`: tools `get_serie` y `get_precios` con TimeGate auditado | Temas 2–4: el Tema 3 (empresas) ya está en marcha en la rama `fase2/t3-empresas` (sonda EDGAR); los temas 2 (bancos centrales y riesgo) y 4 (noticias y alternativos) aún no tienen rama publicada (estado a 2026-10-14). Después, troceado, embeddings y primeros índices con evaluación de la recuperación |
| 3 | **Agentes Macro, Equity y Risk** en LangGraph, salidas estructuradas y trazables | Pendiente | Macro Agent completo con sus skills, sobre episodios históricos concretos |
| 4 | **Cartera y memo:** Portfolio Agent y CIO Agent, motores basados en riesgo, límites y alertas | Pendiente | Añadir Equity, Risk, Portfolio y CIO |
| 5 | **Evaluación:** backtest con TimeGate frente a *baselines* (buy & hold, 60/40, momentum), calidad de los memos | Pendiente | Backtest tras escalar los agentes |

Secuencia propuesta al tutor: (1) cerrar la Fase 1, (2) catálogo de tools, (3) primeros índices, (4) primer agente
(Macro), (5) escalar al resto y luego al backtest. Las fases 1 y 2 avanzan en paralelo.

## 4. Dónde está cada fase en el repositorio

| Fase | Código | Documentación |
|:---:|---|---|
| 1 | Paquete [`src/regimenes`](../../src/regimenes) y notebooks `00`–`20`. Sub-fases internas históricas: 1–4 (re-base de datos, ADR-001), D detectores, E fusión, S sintéticos, F decisión final y validación pseudolive; todas se leen como **sub-fases de la Fase 1** | [`docs/fase1/`](../fase1/README.md) (pipeline, mapa del código, teoría y detectores por familia) |
| 2 | Paquete [`src/ingesta`](../../src/ingesta), un subpaquete por tema, cada uno en su rama ([ADR-007](../decisions/ADR-007-fase2-ingesta-y-docs.md)) | [`docs/fase2/`](../fase2/README.md) (reparto y reglas) · [`plan_datos.md`](../fase2/plan_datos.md) (plan de datos vigente) |
| 3–5 | Por decidir en una ADR propia (ver [lineas_abiertas.md](lineas_abiertas.md) §4) | Este directorio |

Salida de la Fase 1: la tool `get_regimen(fecha)`. El notebook `19_decision_final` congela la regla y define el
contrato; `20_pseudolive` lo valida sobre datos no usados. Contrato provisional en [arquitectura.md](arquitectura.md) §5.

## 5. Índice de este directorio

| Fichero | Contenido |
|---|---|
| [arquitectura.md](arquitectura.md) | Fuentes, índices, tools, TimeGate, anatomía de un agente, catálogo de tools, contrato de `get_regimen` y los cinco agentes |
| [cliente_y_carteras.md](cliente_y_carteras.md) | Cliente minorista, formulario de perfil, dos carteras, memoria de cartera, botón de cataclismo; preguntas abiertas |
| [lineas_abiertas.md](lineas_abiertas.md) | System One decision agents, Briefly, elementos de la propuesta pendientes de análisis, ubicación del código de las fases 3–5 |
| [`fuentes/`](fuentes/) | Documentos de origen: [`propuesta_v2.pdf`](fuentes/propuesta_v2.pdf) (propuesta original, parcialmente superada por ADR-005) y [`reunion_tutor.pdf`](fuentes/reunion_tutor.pdf) (presentación al tutor, octubre de 2026) |
| `internos/` | Solo local, no versionado: presentaciones, plan de datos en LaTeX y material de trabajo del estado del arte (ver [`docs/README.md`](../README.md#solo-local)) |

Los datos alternativos candidatos (Pelosi tracker, Polymarket, EWS…) están en [`docs/fase2/datos_alternativos.md`](../fase2/datos_alternativos.md).

## 6. Relación con la propuesta original

La propuesta ([`propuesta_v2.pdf`](fuentes/propuesta_v2.pdf)) ya era un sistema multi-agente RAG; la documentación
del repositorio la había reducido a «detección». El reencuadre ([ADR-005](../decisions/ADR-005-reencuadre-tfm-multiagente.md)) la recupera.

| Se mantiene | Cambia | Por analizar (§2.3 de ADR-005) |
|---|---|---|
| Título y objetivo: agentes Macro, Equity, Risk, Portfolio y CIO con LangGraph, RAG, memo con citas, TimeGate | El **régimen es la Fase 1** y se entrega como tool `get_regimen(fecha)` con tres estados provisionales (calma / alerta / crisis), fijados por el detector final | Universo en dos capas (42 ETFs y ~1.300 acciones) y *composite score* sobre él, infraestructura AWS, modo operacional, reglas de rebalanceo, ventanas de backtest 2000–2024 |
| Evaluación con TimeGate frente a baselines | El HMM t-Student de 4 estados de la propuesta no es el detector por defecto: la comparativa de la Fase 1 decide | No se descartan ni se confirman; el plan de datos ([`plan_datos.md`](../fase2/plan_datos.md)) ya acota la ventana del backtest y el universo |
| | Requisitos nuevos de la tutoría: cliente y perfil, dos carteras, memoria, datos alternativos, cataclismo | |

Nota de terminología: «Capa 1» en este repositorio es la primera vuelta de 12 detectores v1 ([`historia/capa1`](../historia/capa1/README.md)),
no la «Capa 1» de la propuesta (42 ETFs). Y «Fase 2» es siempre la fase del TFM (ingesta y RAG), no una etapa interna
de la Capa 1 (ver [GLOSARIO](../GLOSARIO.md#capa-1-del-repositorio-no-es-la-capa-1-de-la-propuesta)).
