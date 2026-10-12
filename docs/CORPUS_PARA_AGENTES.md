# Corpus de conocimiento para agentes

## Objetivo acordado

Reunir mucha información útil de las 14 familias de la captura y convertirla
en una biblioteca que los agentes puedan consultar para fundamentar decisiones.
Se priorizan amplitud temática, fuentes gratuitas, calidad de extracción,
procedencia y recuperación de evidencias. La ingesta puede hacerse por lotes;
no hay un requisito de inmediatez ni de reconstrucción temporal extrema.

Esta decisión actualiza el diseño anterior, que daba mayor protagonismo al
control histórico y a comunicaciones monetarias. Las fechas y versiones se
conservan; el filtro estricto de disponibilidad se activa solo cuando se solicita
una evaluación del pasado. No bloquea la incorporación al corpus general.

## Qué entra primero

| Grupo | Familias de la captura | Función en la biblioteca |
|---|---|---|
| Literatura y análisis de riesgos | BIS Quarterly Review y WP, papers de crisis, informes de estabilidad y GFSR | Mecanismos, resultados empíricos, riesgos y metodologías |
| Comunicaciones y contexto macro | Fed, BCE, BoE/BoJ, discursos BIS, otros bancos, GEP/WEO/OCDE | Argumentos institucionales y contexto de decisiones económicas |
| Carteras y regulación | Literatura cartera/CIO, MiFID, ESMA, CNMV | Métodos de construcción de carteras y documentos regulatorios identificados por edición |
| Complementos y derivados | Datasets abiertos, huellas de crisis y estudio de literatura tardía | Anotaciones, datos de apoyo y evaluaciones; separados del texto original |

Los tres primeros grupos se incorporan desde el piloto mediante muestras
pequeñas y luego cargas amplias. Las prioridades determinan el orden de trabajo,
no excluyen familias. No se pretende replicar las cantidades de la captura
sin disponer de sus listas exactas de documentos.

## Adquisición por lotes

- Empezar por descargas oficiales masivas y catálogos de acceso abierto;
  continuar con páginas individuales y carga manual cuando haga falta.
- Organizar cargas por fuente, serie, edición o año. Retomar desde checkpoints;
  no redescargar archivos idénticos.
- Propuesta inicial de refresco: semanal para comunicaciones e informes activos;
  mensual para descubrir literatura y cambios de catálogos. La frecuencia se
  ajustará al piloto; los documentos no cambian por fijar un cron más frecuente.
- Hacer la carga retrospectiva por etapas según volumen y presupuesto. El ritmo
  de actualización no debe impedir recuperar literatura antigua relevante.
- Ante un documento inaccesible, guardar la referencia como pendiente y seguir
  con el resto. Dar preferencia a una copia abierta autorizada de la misma versión.

## Qué hace que un documento sea utilizable

Para entrar en el corpus consultable general necesita original conservado,
procedencia identificada, estado de acceso y términos registrados, versión o
hash, texto suficientemente completo y vínculo al catálogo. La fecha de descarga
se conoce siempre; las fechas de publicación desconocidas se marcan como tales.

Se registra por separado si el texto procede de HTML, PDF u OCR, si hay páginas
ilegibles, tablas omitidas o errores de extracción. Una bibliografía sin texto
disponible permanece en el catálogo, pero no cuenta como texto listo para consulta.
Un informe de mala extracción se conserva para reprocesar; no se entrega al agente
como evidencia íntegra. La calidad de la extracción y la solidez científica del
contenido son evaluaciones diferentes.

Deduplicar bytes, republicaciones y versiones relacionadas evita que cien copias
del mismo paper aparenten cien evidencias independientes. Preservar traducciones
y revisiones como relaciones explícitas. En literatura, indicar si se trata de
preprint, working paper, manuscrito aceptado o artículo publicado cuando se sepa.

## Consulta de los agentes

El almacenamiento en R2 conserva los documentos. Además necesitamos una capa
de búsqueda y un contrato de consulta; subir PDFs por sí solo no permite a los
agentes encontrar los pasajes útiles.

Operaciones previstas, todavía sin código:

| Operación | Resultado esperado |
|---|---|
| Buscar por pregunta y filtros | Pasajes relevantes por tema, institución, país, idioma, tipo o edición |
| Abrir un documento | Texto completo o sección, con originales y metadatos |
| Ampliar el contexto de un pasaje | Sección, páginas vecinas y documento padre |
| Consultar fuentes de una afirmación | Identificadores, versiones, URLs y localizadores verificables |
| Pedir una consulta histórica | Las mismas operaciones con corte temporal explícito y evidencia suficiente |

Combinar progresivamente búsqueda literal y semántica: términos técnicos,
autores e identificadores deben encontrarse aunque no tengan embeddings. La
elección del motor de búsqueda se tomará con el piloto y se documentará antes
de implementar; ni R2 ni el catálogo D1 sustituyen ese diseño.

Entregar al agente fragmentos con título, fuente, versión, fecha conocida,
página/sección, URL e identificador. Puede pedir más contexto o el texto completo.
Los resúmenes facilitan navegación, pero el documento original respalda las citas.
Una respuesta debería poder expresar falta de evidencia, discrepancias entre
fuentes y el carácter retrospectivo o contextual de un trabajo.

La consulta general puede usar literatura antigua y documentos sin fecha exacta
si cumplen los requisitos de calidad y acceso, mostrando esa incertidumbre.
No se usa por defecto un corte temporal. La fecha sigue siendo necesaria para
valorar actualidad: un informe viejo aporta contexto, y una guía sustituida debe
identificarse como edición histórica. No se fija una caducidad uniforme a la literatura.

## Cómo comprobar que la biblioteca sirve

Medir documentos únicos con texto utilizable por familia, institución, tema,
idioma y periodo; además de bytes y volumen total. Registrar duplicados,
referencias sin texto, errores, cobertura pendiente y coste de procesamiento.

Preparar preguntas de referencia repartidas entre bancos centrales, crisis,
estabilidad y carteras. Revisar si la búsqueda recupera documentos pertinentes,
si el pasaje respalda la afirmación y si la cita permite abrir el original.
Incluir preguntas sin respuesta en el corpus y otras con fuentes discrepantes.
El piloto fijará umbrales con datos reales; el número de PDFs descargados no
demuestra por sí solo que los agentes puedan usarlos bien.

Indexar semánticamente por lotes y ampliar la cobertura según utilidad medida
y presupuesto. Todo texto validado debe ser localizable por catálogo y búsqueda
textual cuando se habilite la consulta; los documentos sin embeddings también
pueden recuperarse y abrirse. Un corpus más grande amplía evidencias disponibles,
pero no garantiza mejores decisiones sin evaluar recuperación y uso de las citas.

## Alcance actual

Se modifica documentación y notebook. No se han escrito recolectores, creado
cuentas, descargado corpus, desplegado recursos ni conectado agentes todavía.
Los contratos, frecuencias e índices descritos son el diseño para la implementación.
