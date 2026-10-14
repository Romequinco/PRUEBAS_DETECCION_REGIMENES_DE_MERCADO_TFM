# Cliente y carteras

Requisitos añadidos en la tutoría ([ADR-005](../decisions/ADR-005-reencuadre-tfm-multiagente.md) §2.2). **Todos están
abiertos en su diseño**; este documento fija la intención y deja las preguntas sin responder en la sección final.
Contexto del sistema: [README](README.md) · [ARQUITECTURA](ARQUITECTURA.md).

## 1. Cliente minorista

El usuario objetivo es un **cliente minorista**. La opción de **gestor** (profesional que gestiona varios clientes) es un
extra si da tiempo. El primer paso de cualquier uso es conocer al cliente mediante un formulario de perfil.

## 2. Formulario de perfil

- **Muy abierto:** no hay preguntas ni escalas decididas.
- Idea de partida: un cuestionario de **idoneidad tipo MiFID II** (conocimientos y experiencia, situación financiera,
  objetivos y horizonte, tolerancia a pérdidas).
- Sus respuestas **definen los límites del CIO Agent**: qué riesgo puede asumir la cartera y dentro de qué bandas puede
  ajustar. Sustituirían a los límites institucionales fijos del diseño propuesto (5 % acción, 15 % ETF, 30 % sector,
  5 % liquidez, vol objetivo por régimen), que quedarían como valores por defecto.

## 3. Dos carteras por decisión

| Cartera | Qué es | Restricciones |
|---|---|---|
| **Con límites** | La que respeta el perfil y los límites del cliente | Todas las del perfil |
| **En bruto** | La que recomiendan los agentes sin el filtro del cliente | **Por decidir** qué conserva (¿límites de concentración? ¿liquidez mínima? ¿ninguna?) |

Interés: mostrar cuánto «cuesta» o «protege» el perfil del cliente frente a la recomendación pura del sistema. Ambas se
explican en el memo. Los criterios para comparar su resultado están abiertos.

## 4. Memoria de la cartera anterior

Cada cartera conoce la anterior: un **estado persistente** entra como información a los agentes, de modo que la decisión
tiene en cuenta lo que ya se tiene (turnover, costes, plan de reentrada tras una reducción de riesgo) y no parte de cero.
El **formato está por decidir** (pesos y fecha, memo anterior, registro de ajustes, o combinación). Debe respetar el
TimeGate: en el backtest, la memoria solo contiene decisiones ya tomadas.

## 5. Botón de cataclismo

- **Qué es:** una alerta **automática** (no un botón manual) basada en [EWS](https://ews.kylemcdonald.net) de Kyle McDonald,
  que vigila vuelos de jets privados. Modo broma: si «los millonarios huyen en jet», se **desinvierte todo**.
- **Alcance:** solo **en vivo o demostración**. El histórico de EWS empieza el 6 de octubre de 2025, por lo que no cabe en
  el backtest (que usa ventanas largas) ni participa en la evaluación.
- Detalles de la fuente: [DATOS_ALTERNATIVOS.md](DATOS_ALTERNATIVOS.md).
- Por decidir: umbral de nivel de alerta que dispara la acción, si pasa por el CIO Agent o lo anula, y cómo se registra
  en el memo.

## 6. Preguntas abiertas

1. ¿Qué preguntas y escalas tiene el formulario de perfil y cómo se traducen a límites numéricos del CIO?
2. ¿Qué restricciones conserva la cartera «en bruto»?
3. ¿Cómo se comparan las dos carteras (métricas, frecuencia de divergencia)?
4. ¿Qué formato tiene la memoria de la cartera anterior y qué contiene?
5. ¿Cómo se evalúa el efecto de la memoria sin romper el TimeGate ni contaminar el backtest?
6. ¿El perfil puede cambiar en el tiempo? ¿Cómo se versiona?
7. ¿Se incluye el rol de gestor (varios clientes) o queda fuera del alcance?
8. Cataclismo: ¿umbral, interacción con el CIO y con los límites del cliente, y cómo se presenta en la demo?
9. ¿Los límites institucionales de la presentación al tutor se mantienen como valores por defecto?
