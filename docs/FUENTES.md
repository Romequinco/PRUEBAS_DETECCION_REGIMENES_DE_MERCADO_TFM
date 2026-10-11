# Inventario inicial de la imagen

Fuente del inventario: imagen proporcionada por el usuario. Los recuentos y
estados de acceso pertenecen a esa referencia; no han sido contrastados como
cobertura actual ni representan archivos descargados por este proyecto.
Cada familia necesita URLs de listado, formatos, licencia, reglas de fecha y
una prueba de acceso antes de habilitar su recolector.

| Familia | Contenido previsto | Referencia de la imagen | Tratamiento inicial |
|---|---|---|---|
| Fed | Comunicados, actas, ruedas de prensa, Beige Book, transcripciones y discursos | 245 comunicados, 260 actas, 95 ruedas de prensa; unos 3 M tokens; OK | Piloto oficial, HTML/PDF y cronología de publicaciones |
| BCE | Decisiones, declaraciones, Q&A, accounts y discursos | 212 Q&A, 96 accounts, 2.067 discursos; 1,7 M palabras; OK | Recolector por tipo y extracción Q&A |
| BoE y BoJ | Actas MPC, MPR, discursos y comunicados | 290 MPC; BoJ 261 comunicados y 232 actas; OK | Adaptadores separados y calendario de publicación real |
| Discursos BIS | Discursos de bancos centrales y Annual Report | 13.234 discursos; archivo anual de unos 129 MB; OK | Descarga oficial, manifiestos y procedencia del banco emisor |
| Otros bancos centrales | BCB, BoC, SNB, Riksbank, Norges, RBNZ, RBI, Banxico, BdE, RBA y PBoC | Acceso MIXTO; PBoC marcado bloqueado | Adaptador por institución; API/RSS/listados y carga manual si procede |
| Organismos | Banco Mundial GEP, FMI WEO y OCDE | MIXTO; unas 90 WEO; bloqueos indicados | Revisar acceso y descargar ediciones originales con fechas |
| Datasets abiertos | Corpus de bancos centrales, FOMC y Beige Book en Hugging Face | 24 BC y 15.000 documentos; CC BY indicada, sin fecha de publicación | Identificar repos exactos, verificar licencia y excluir del histórico si falta fecha fiable |
| Estabilidad financiera | FSR Fed/BCE/BoE, ESRB, OFR y FSB | 16/48/58 FSR y 929 otros documentos; 13 M palabras; OK | Informes por edición y entidad, tablas y vínculo al documento padre |
| BIS QR y Working Papers | Quarterly Review y WP | 119 QR, 1.377 WP; 34 M palabras; OK | Catálogo por serie, versión y fecha de disponibilidad |
| FMI GFSR | Global Financial Stability Report | BLOQUEADA según la imagen | Priorizar descarga autorizada/manual y dejar trazabilidad |
| Papers de crisis | Literatura de 37 crisis, 1929–2024; NBER, SSRN, WP y arXiv | 386 papers, 359 abiertos, 265 validados, 13.662 páginas; 5,2 M palabras | Identificadores, DOI cuando exista y primera versión pública |
| Huellas de crisis | Familias de crisis y base ESRB | 18 crisis de la Fase 1; 7 familias; 82 crisis y 29 países | Dataset derivado separado, con metodología y versiones |
| Literatura cartera/CIO | Construcción de carteras e idoneidad; MiFID, ESMA, CNMV y otros | 34 referencias de cartera y 20 de idoneidad; unos 30 MB; OK | Catálogo temático y registro de edición/documento |
| La literatura llega tarde | Estudio de cuándo estaban disponibles los papers de crisis | 18 de 77 papers GFC visibles el 15-09-2008; +1,6 años indicados | Resultado derivado a reproducir; no es una fuente de descarga independiente |

## Entradas oficiales verificadas para iniciar el piloto

- [Calendario FOMC](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm):
  punto de partida para contrastar fechas de reuniones y materiales relacionados.
- [Material histórico FOMC](https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm):
  transcripciones y materiales por año.
- [Accounts BCE](https://www.ecb.europa.eu/press/accounts/html/index.en.html):
  listado oficial.
- [Actas BoJ por año](https://www.boj.or.jp/en/mopo/mpmsche_minu/minu_all/index.htm):
  listado oficial.
- [Discursos BIS](https://www.bis.org/speeches/central-bank): listado oficial.

Estos enlaces son entradas de catálogo, no una garantía de acceso automatizado
a todos los documentos. La aceptación del piloto exigirá comprobar una muestra
real y respetar las condiciones de la fuente.

## Datos pendientes para registrar cada fuente

Identificador estable, institución, categoría, URL oficial, URLs de listado,
método permitido de acceso, formatos, idioma, cobertura deseada, calendario,
límite de peticiones, política de reintentos, licencia, disponibilidad histórica,
selector/parser y política de selección para RAG.

Los IDs/URLs de los datasets Hugging Face y la lista concreta de papers, crisis
y referencias CIO no se pueden reconstruir únicamente a partir de los recuentos
de la imagen. Se reservarán como pendientes hasta disponer del inventario real.
