# Orden de implementación

## Fase 0: diseño y cuenta

Completado localmente: estructura de carpetas, documentación, inventario de la
imagen y propuesta de almacenamiento. Pendientes: cuenta Cloudflare, activación
R2 y decisión de presupuesto. No se ha conectado Cloudflare ni creado un recurso
remoto. El repositorio GitHub existente es
`feraltauste1/Bancos-centrales-riesgo-sistem-tico-y-literatura`.

Antes de implementar, concretar rango histórico del piloto, fuentes prioritarias
y frecuencia de actualización. Propuesta inicial: Fed y BCE, una familia por
institución y un lote pequeño de documentos recientes e históricos.

## Fase 1: almacenamiento y catálogo del piloto

Crear entorno dev, credenciales acotadas, contrato de documento y catálogo D1.
Implementar una prueba de subir/descargar/verificar hash y registrar metadatos.
Comprobar repetición segura, error parcial entre R2 y D1 y recuperación.

## Fase 2: recolección verificable

Primeros recolectores Fed/BCE, HTML/PDF, fechas efectivas, deduplicación,
extracción y manifiestos. Aceptar solo lotes con procedencia y conteos comprobados.
Implementar importación manual para fuentes que la requieran.

## Fase 3: automatización y cobertura

Utilizar el remoto GitHub existente y revisar su visibilidad; habilitar tests y jobs
manuales/programados. Añadir BoE, BoJ y BIS. Hacer el backfill por fuente/año,
con checkpoints y seguimiento de volumen, coste y calidad. El proceso se
ejecutará en la nube; GitHub y R2 cumplirán funciones distintas.

Los cambios del pipeline pasan tests mediante push. El refresco de información
se ejecuta por calendario o acción manual; no depende de hacer commits con datos.
Los workflows no se han creado ni activado todavía.

## Fase 4: archivo y RAG

Conservar todo lo permitido en el archivo; seleccionar documentos clave,
fragmentar con páginas y padre y evaluar un índice vectorial. Implementar
consultas por fuente y fecha con TimeGate y citas. Añadir API/Worker cuando
se necesite consultar desde otra aplicación.

## Fase 5: literatura y derivados

Ampliar estabilidad financiera, organismos y bancos secundarios. Recibir
inventarios de papers/datasets/CIO que la imagen no identifica individualmente.
Agregar versiones de papers, huellas de crisis, tono, diferencias entre
comunicados y evaluación de literatura tardía.

## Criterios para pasar a producción

Originales recuperables y verificados por hash; catálogo coherente; procedencia
y licencia conocidas; fechas suficientes para el modo histórico; reejecuciones
sin duplicados; recuperación de fallos; consultas con citas; costes medidos y
restauración probada. El alcance y los umbrales se fijarán durante el piloto.
