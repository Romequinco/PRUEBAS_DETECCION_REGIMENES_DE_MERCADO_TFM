# `herramientas` · contrato de tools para los agentes

Puente entre los datos (Fase 1 [`regimenes`](../regimenes/README.md) y Fase 2 [`ingesta`](../ingesta))
y los agentes de las fases 3–5. Los agentes importan **solo** de este paquete. Así el detector de régimen o
una fuente pueden cambiar sin romperlos. La decisión está en
[ADR-012](../../docs/decisions/ADR-012-estructura-repo.md) y el catálogo previsto en
[`docs/tfm/arquitectura.md`](../../docs/tfm/arquitectura.md).

| Módulo | Qué hace |
|---|---|
| `contrato.py` | Modelos pydantic comunes: `Cita`, `EntradaTool` (lo que elige el LLM) y `SalidaTool` (respuesta con citas y avisos) |
| `registro.py` | `registrar` (decorador), `catalogo(agente)`, `json_schema(nombre)` (lo que ve el LLM) e `invocar(nombre, args, fecha=...)` |
| `regimen.py` | `get_regimen`: envoltorio tipado de `regimenes.regimen.get_regimen` |

## Reglas

1. **La fecha la pone el sistema, nunca el LLM.** `registrar` rechaza un modelo de entrada con campo
   `fecha`, e `invocar` la recibe aparte. Así ninguna tool puede mirar el futuro (TimeGate).
2. **Toda salida lleva citas.** Cada `SalidaTool` dice de qué fichero o serie sale, con su huella. Esas
   citas son las que usa el memo final.
3. **Solo se envuelve.** La lógica vive en su paquete (`regimenes`, `ingesta/<tema>/tools.py`). Aquí solo se
   tipa y se registra.

```python
from datetime import date
from herramientas import catalogo, invocar, json_schema

json_schema("get_regimen")                                    # esquema para el LLM
invocar("get_regimen", {}, fecha=date(2020, 3, 16)).estado     # 'crisis'
```

Pendiente: envolver `get_serie` y `get_precios` de [`ingesta.mercados_series.tools`](../ingesta/mercados_series/tools.py)
y, cuando exista la memoria RAG, las tools de búsqueda.
