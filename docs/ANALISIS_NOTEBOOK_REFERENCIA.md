# Lectura y adaptación de la sonda de Daniel

Referencia: [t3_01_sonda_empresas.ipynb, rama fase2/t3-empresas](https://github.com/Romequinco/PRUEBAS_DETECCION_REGIMENES_DE_MERCADO_TFM/blob/fase2/t3-empresas/notebooks/fase2/t3_01_sonda_empresas.ipynb).
Leídas las **29 celdas**, tanto Markdown como código, el 14 de octubre de 2026.
La lectura del notebook no verifica por sí sola sus módulos externos ni sus
afirmaciones empíricas; no se han ejecutado las consultas de empresas.

## Estructura del Markdown

Abre con título, fase, responsable, objetivo de una muestra pequeña, universo,
fuentes y requisitos de ejecución. Los nueve apartados guían la lectura:
configuración; catálogo y TimeGate; cobertura y retrasos; descarga y secciones;
fundamentales point-in-time; sentimiento y cambio de riesgos; volumen; fichas y
tablas; resumen para la reunión. Los subapartados distinguen desfase de horas,
extracción por épocas y ejemplo de documento. Antes del código se explica qué
mide, cómo leerlo y qué limitación tiene la muestra.

## Estructura del código

Una configuración común limita la sonda y activa caché. La adquisición,
normalización, verificación de horas, extracción, XBRL, sentimiento y
exportación están en módulos importados. Las celdas del notebook llaman pasos
cortos y muestran tablas o gráficos; el resumen reutiliza los resultados,
sin escribir cifras a mano. Las tablas mantienen errores y fuentes sin datos.
El universo incluye empresas desaparecidas para hacer visible la cobertura.

La cobertura de filings utiliza un catálogo amplio, mientras la extracción
y el sentimiento utilizan un subconjunto. Su histograma agrupa retrasos por
familia de informe y el gráfico de `risk_delta` usa fecha de publicación.
Los XBRL retienen republicaciones y permiten consultar dos cortes distintos.
La extrapolación de volumen aclara cuándo procede de un subconjunto.

## Adaptación a fuentes textuales

Se aplica la misma organización, conservando los 14 apartados originales:
inventario de familias; adquisición limitada; cobertura temporal; gráficos;
texto/fila de ejemplo; calidad de extracción; errores; volumen; exportación
y resumen calculado. Cada institución y familia sigue siendo identificable
en sus propias filas. Configuración, HTTP, adaptadores, extracción, fechas,
auditoría y gráficos están en `src/macro_riesgo/sonda/`.

Se distinguen publicación, evento y aplicación. Para actas se busca el par
reunión/publicación; para normativa, anuncio/aplicación cuando lo declara el
texto. Para papers o informes sin aplicación se muestra la ausencia: no se
inventa ese campo ni se interpreta automáticamente la edición como publicación.
Las fechas parciales no producen lags ni horas ficticias.

No se traslada la regla del cierre de Nueva York a comunicaciones mundiales.
El notebook de Daniel describe una política de uso diario de filings; esa
política no equivale a disponibilidad pública intradía y requeriría revisar
sesiones y festivos para otros mercados. Aquí el corte tiene zona explícita
y una versión con solo día se admite conservadoramente desde el siguiente
día local, cuando se acredita la versión y su zona.

No se presentan el vocabulario de un discurso ni una etiqueta HF como tono
validado. El cambio de conjuntos de palabras es descriptivo; un diccionario
de sentimiento exigiría idioma, licencia y versión. Se conservan fragmentos
con vínculo al original, sin construir un RAG ni entrenar un clasificador.

## Correcciones necesarias en el notebook previo

- Sustituir la gran celda de funciones auxiliares por módulos reutilizables.
- Medir varios documentos cuando el listado lo permite, sin confundir
  primera/última muestra con inicio/fin del histórico.
- Mostrar gráficos en todos los apartados, incluyendo casos sin fechas
  y acceso fallido; ausencia de medición no significa retraso cero.
- Fijar arXiv a v1 para evitar fechar el PDF revisado con `published` de v1.
- No pasar la fecha de la última ficha GFSR a un PDF de edición fija diferente.
- No usar periodo observado, reunión, fecha del nombre del archivo, registro
  Crossref ni depósito como sustitutos automáticos de publicación.
- Conservar originales y hashes, pero distinguir hash de acreditación histórica:
  el hash de hoy no prueba cuándo estuvo público el texto.
- Sincronizar notebook, Markdown y HTML; incluir a **Fernando Dapena Tauste**
  como responsable.

El modo ampliado es otra sonda limitada, no un recolector masivo. Los CSV
permiten revisar las evidencias por fuente; la acreditación histórica de
portales revisables y los vintages de tablas siguen como pendientes explícitos.
