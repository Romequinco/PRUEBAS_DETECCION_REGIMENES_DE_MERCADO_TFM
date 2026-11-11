# `ingesta` · datos de los agentes (Fase 2)

Un subpaquete por tema, más lo común. Las reglas, el contrato y el reparto están en
[`docs/fase2/README.md`](../../docs/fase2/README.md), y la instalación de cada tema (sus extras de `pyproject.toml`), en el
[README raíz](../../README.md#instalación).

| Subpaquete | Qué cubre | Tools previstas |
|---|---|---|
| [`comun/`](comun) | esquema `Documento`/`FichaFuente`, cliente HTTP, rutas por tema y fichas | — |
| [`mercados_series/`](mercados_series/README.md) | precios, curvas, macro y vintages | `get_precios`, `get_serie` |
| [`bancos_riesgo/`](bancos_riesgo/README.md) | bancos centrales, organismos, estabilidad financiera, literatura | `buscar_macro`, `buscar_riesgo` |
| [`empresas/`](empresas/README.md) | filings, fundamentales y sentimiento | `buscar_filings`, `get_fundamentales` |
| [`noticias_alt/`](noticias_alt/README.md) | noticias y datos alternativos | `buscar_noticias`, `get_polymarket` |

Los notebooks de cada tema están en `notebooks/fase2/<tema>/`, los tests en `tests/ingesta/<tema>/` y las salidas
pequeñas en `results/ingesta/<tema>/`. Los agentes no importan este paquete directamente: lo hacen a través de
[`herramientas`](../herramientas/README.md). Este paquete no toca `regimenes` (Fase 1) ni su huella de caché.
