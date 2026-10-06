# ADR-005 — El TFM es un sistema multi-agente; el régimen es su Fase 1

- **Estado:** Aceptada · 2026-10-06 (tras la reunión con el tutor; decisión del equipo).
- **Fuentes:** [`docs/context/TFM_Reunion_Tutor.pdf`](../context/TFM_Reunion_Tutor.pdf) (presentación al
  tutor, octubre de 2026) y [`docs/context/TFM_Proposal_v2.pdf`](../context/TFM_Proposal_v2.pdf)
  (propuesta original, parcialmente superada por esta ADR).
- **Corrige:** [ADR-004](ADR-004-unificacion.md) §2 (*"todo el TFM vive en `regimenes`"*): `regimenes` es
  el código de la Fase 1. ADR-001 a ADR-004 siguen vigentes dentro de la Fase 1.
- **Renumera:** la decisión final del detector de régimen (notebook `19_decision_final`), que hasta ahora
  se llamaba «futura ADR-005», pasa a ser la **futura ADR-006**.

---

## 1. Contexto

La propuesta original ya se titulaba *Multi-Agent RAG System for Regime-Aware Macro-Equity Intelligence
and Dynamic Portfolio Allocation*. Sin embargo, la documentación del repositorio había acabado
presentando el TFM como «un sistema de detección de regímenes», porque todo el trabajo hecho hasta
ahora (datos, 12 detectores, benchmark, fusión, sintéticos) pertenece a esa pieza. En la reunión con el
tutor se fijó el alcance final y se añadieron requisitos nuevos.

## 2. Decisión

**El TFM es *Multi-Agent RAG System for Regime-Aware Macro-Equity Intelligence*:** un equipo de agentes
LLM (Macro, Equity, Risk, Portfolio y CIO, orquestados con LangGraph) que lee información financiera
heterogénea mediante RAG y, sabiendo en qué régimen está el mercado, construye carteras y redacta un
*investment memo* con citas a las fuentes. No predice precios: detecta el estado del mercado, lo
contextualiza con evidencia y gestiona el riesgo. Visión completa en [`docs/tfm/`](../tfm/README.md).

Se trabaja en **cinco fases**:

| Fase | Qué | Estado |
|:---:|---|---|
| 1 | Régimen de mercado: base de datos causal, comparativa de detectores, señal final como tool `get_regimen(fecha)` | 🟡 en curso (este repositorio, paquete `regimenes`) |
| 2 | Ingesta, RAG y tools: corpus con fecha de publicación, TimeGate dentro de cada tool, evaluación RAGAS | 🔜 |
| 3 | Agentes Macro, Equity y Risk en LangGraph, salidas estructuradas y trazables | 🔜 |
| 4 | Cartera y memo: Portfolio Agent y CIO Agent, motores basados en riesgo, límites y alertas | 🔜 |
| 5 | Evaluación: backtest con TimeGate frente a *baselines* (buy & hold, 60/40, momentum), calidad de los memos | 🔜 |

Las etapas internas de la Fase 1 conservan sus nombres históricos (fases 1–4 del re-base de datos de
ADR-001, y D detectores, E fusión, S sintéticos, F decisión final + validación) y pasan a leerse como
**sub-fases de la Fase 1 del TFM**.

### 2.1 Salida de la Fase 1: la tool `get_regimen(fecha)`

La Fase 1 se cierra cuando la señal final de régimen queda publicada como tool para los agentes:
`get_regimen(fecha)` → estado, probabilidad y días en el estado, usando solo información publicada hasta
`fecha`. El **conjunto de estados lo fija el detector final** (futura ADR-006); la hipótesis de trabajo es
**calma / alerta / crisis**, que se corresponde con los estados normal / vigilancia / confirmado de la
máquina de fusión (`regimenes.fusion`). El notebook `19_decision_final` congela la regla y define el
contrato; `20_pseudolive` valida la tool sobre datos no usados.

### 2.2 Requisitos añadidos en la tutoría

Todos quedan **abiertos en su diseño**; se documentan en [`docs/tfm/`](../tfm/README.md):

1. **Cliente minorista** (opción de gestor si da tiempo). Primero un **formulario de perfil** (idea:
   cuestionario de idoneidad tipo MiFID II) cuyas respuestas son los límites del CIO Agent.
2. **Dos carteras** por decisión: la que respeta los límites del cliente y la que recomiendan los
   agentes «en bruto». Qué restricciones conserva la segunda está por decidir.
3. **Memoria de cartera:** cada cartera conoce la anterior (estado persistente que entra a los agentes);
   el formato está por decidir.
4. **Datos alternativos** a investigar: operaciones bursátiles del Congreso («Pelosi tracker»),
   divergencia de información privilegiada en Polymarket y otras candidatas.
5. **Botón de cataclismo** (modo broma): alerta automática a partir de
   [ews.kylemcdonald.net](https://ews.kylemcdonald.net), que vigila vuelos de jets privados; si «los
   millonarios huyen», se desinvierte todo. Su histórico empieza en octubre de 2025: sirve en vivo o
   como demostración, no en el backtest.
6. **System One decision agents** como línea de investigación: modelos de decisión rápidos y tipados
   (Jev de TypeSafe AI, Clef de Cloudflare, Laya de Convai) para decisiones simples antes de escalar al
   LLM.
7. **Extra si da tiempo:** integrar al final la práctica *Briefly*
   ([`multimodal-market-briefer`](https://github.com/Romequinco/multimodal-market-briefer)) para cubrir
   la IA multimodal. Hoy no forma parte del repositorio.

### 2.3 Qué queda por analizar de la propuesta original

La presentación al tutor no menciona varios elementos de la propuesta: el universo en dos capas (42 ETFs
y ~1.300 acciones), el *composite score* sobre todo ese universo, la infraestructura AWS, el modo
operacional, las reglas de rebalanceo y las ventanas de backtest 2000–2024. **No se descartan ni se
confirman**: se analizarán en profundidad antes de la Fase 2.

### 2.4 Terminología

«**Capa 1**» en este repositorio es la primera vuelta exploratoria de los 12 detectores v1
([`historia/capa1/`](../historia/capa1/README.md), tag `capa1-final`). **No** es la «Capa 1» de la
propuesta (42 ETFs) ni guarda relación con su «Capa 2» (acciones). Se mantiene el nombre por coherencia
con el tag y la historia; el GLOSARIO lo aclara.

## 3. Consecuencias

- README, índice de `docs/`, GLOSARIO y descripciones del paquete presentan el TFM completo y sitúan
  este repositorio como su Fase 1.
- Dónde vivirá el código de las fases 2–5 (paquetes nuevos en este repositorio o un repositorio aparte)
  queda **abierto**; se decidirá en una ADR propia.
- El objetivo de `19_decision_final` y `20_pseudolive` deja de ser «el sistema final del TFM» y pasa a
  ser la señal de régimen que consumirán los agentes.
- El notebook `18_sinteticos_aumento` se mantiene.
