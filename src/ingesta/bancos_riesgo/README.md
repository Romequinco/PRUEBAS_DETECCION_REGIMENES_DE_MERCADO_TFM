# Tema 2 · Bancos centrales, riesgo sistémico y literatura

**Rama:** `fase2/t2-bancos-riesgo` · **Agentes:** Macro y Risk (Portfolio y CIO, como conocimiento) · **Tools:**
`buscar_macro(q, fecha)`, `buscar_riesgo(q, fecha)` · **Destino:** RAG base + cajón de cada agente. Reglas comunes:
[`docs/fase2/README.md`](../../../docs/fase2/README.md).

## Alcance

**Documentos de texto** institucionales y académicos:
- política monetaria (bancos centrales de todo el mundo);
- organismos internacionales;
- estabilidad financiera;
- papers sobre crisis (estado del arte por crisis y periodo);
- literatura de construcción de carteras e idoneidad (MiFID II).

Recuerda que cada agente tiene **dos RAG**:
- **base:** fija; los documentos más relevantes y el conocimiento general;
- **cajón:** se llena con lo que traen sus tools y persiste.

## Fuentes a investigar

- [ ] Fed: comunicados y actas FOMC, ruedas de prensa, Beige Book, transcripciones, discursos, Monetary Policy Report
- [ ] BCE: decisiones, declaraciones, *accounts*, discursos, Boletín Económico
- [ ] BoE, BoJ y otros bancos centrales (BoC, SNB, Riksbank, Norges, RBA, RBNZ, PBoC, RBI, Banxico, BCB, BdE…)
- [ ] Discursos de bancos centrales recopilados por el BIS
- [ ] Organismos: FMI (WEO, GFSR, Article IV), Banco Mundial, OCDE, BIS (Annual Report)
- [ ] Estabilidad financiera: FSR de Fed, BCE y BoE; ESRB, OFR, FSB; BIS Quarterly Review y Working Papers
- [ ] Papers de crisis (1929 → hoy) **en versiones abiertas y legales**, con su fecha de primera publicación
- [ ] «Huellas» de crisis (indicadores antes y durante cada crisis)
- [ ] Literatura de cartera (risk parity, mínima varianza…) e idoneidad (ESMA, CNMV)
- [ ] Datasets abiertos ya construidos (Hugging Face, Kaggle): licencias y si traen fecha de publicación

> **Acceso legal.** Las sondas del plan ya descartaron algunas de estas fuentes: el **PBoC** (robots.txt con
> `Disallow: /`) y el **GFSR del FMI** (el WAF del FMI bloquea a los clientes que no son navegador) no se descargan
> de forma automática. El PBoC se cubre con los discursos del BIS y datasets abiertos con fecha estimada; el GFSR,
> solo si se aprueba la descarga manual (Q5). Lista completa y motivos:
> [plan de datos §3.4](../../../docs/fase2/plan_datos.md#34-acceso-legal-fuentes-excluidas); puntos abiertos del
> tema: [anexo B](../../../docs/fase2/plan_datos.md#anexo-b-puntos-abiertos-por-tema).

## Entregables

1. Un extractor por fuente con su ficha y sus tests; documentos como `Documento` en jsonl con su `fecha_publicacion`
   real (las actas salen semanas después de la reunión).
2. Catálogo bibliográfico (metadatos + URL abierta, sin texto) en `results/ingesta/bancos_riesgo/`.
3. Propuesta de qué va a la base y qué al cajón, y de cómo trocear cada tipo de documento.

## Preguntas abiertas

- ¿Qué hacer con PDFs que bloquean la descarga automática (descarga manual)?
- ¿Qué modelo para el tono hawkish/dovish (licencia)?

## Sonda de fuentes implementada

**Responsable: Fernando Dapena Tauste.** El
[notebook de fuentes](../../../notebooks/fase2/bancos_riesgo/01_fuentes_prioridad_gratuita.ipynb)
conserva sus 14 apartados independientes y un inventario de 94 familias de documentos,
filas de datos y metadatos. Mide cobertura de la muestra, fechas, retrasos, extracción,
volumen y errores; sus resultados no representan el histórico completo.

El código reutilizable está en `sonda/`, el inventario en `config/sonda_fuentes.json`
y las pruebas sin red en `tests/ingesta/bancos_riesgo/`. El modo rápido limita cada
familia descubierta a tres muestras y la extracción a 40 páginas por PDF.
`HF_TOKEN` y `UNPAYWALL_EMAIL` se leen del entorno cuando hacen falta. Los errores
de acceso quedan registrados sin detener las demás fuentes. El TimeGate histórico
exige acreditar la versión textual, además de la fecha de publicación.

Desde la raíz del repositorio, con el entorno Python del TFM:

```powershell
python -m pip install -e ".[dev]"
python -m pip install -r requirements/ingesta-bancos_riesgo.txt
python -m jupyterlab notebooks/fase2/bancos_riesgo
```

Selecciona el kernel de ese entorno. Las dependencias comunes pertenecen al paquete
del repo; la sonda usa pandas `>=2.1,<3`, compatible con la Fase 1. El proyecto de origen
fijaba pandas 3; ese pin se adapta al integrar la sonda. No se requiere actualizar el
entorno numérico de la Fase 1.

Los originales, versiones, JSONL y CSV se guardan bajo
`<INGESTA_CORPUS_DIR>/bancos_riesgo/{cache,local,exports}/sonda/`. La ruta se resuelve
con `ingesta.comun.rutas`; por defecto usa `./corpus`, ignorado por Git. No se
escribe en `data/`, reservado a la Fase 1. `directorio_corpus` en `ConfigSonda`
permite aislar los archivos durante las pruebas. Las muestras pequeñas autorizadas
para tests pertenecen a `tests/ingesta/bancos_riesgo/fixtures/`.

Para ejecutar el cuaderno completo o regenerar sus versiones de lectura:

```powershell
python -m ingesta.bancos_riesgo.jobs.ejecutar_notebook_sonda
python -m ingesta.bancos_riesgo.jobs.exportar_notebook
python -m pytest tests/ingesta/bancos_riesgo -q
```

El ejecutor acepta `--kernel nombre` y por defecto usa `python3`. La exportación
no consulta la red y escribe HTML y Markdown en
`<INGESTA_CORPUS_DIR>/bancos_riesgo/exports/notebooks/`; la carpeta del notebook
contiene únicamente el `.ipynb`.

## Diseño y próximos pasos

Hay una sonda ejecutable, pero la recolección masiva, los contratos de producción,
las migraciones SQL y los servicios Cloudflare todavía están pendientes. El diseño
propone Python por lotes, R2 para objetos, D1 para el catálogo y GitHub Actions como
ejecutor inicial. Worker e índice vectorial quedan sujetos a validar el piloto.
Las carpetas vacías del proyecto original se han retirado; sus planes se conservan
en la documentación:

- [Objetivo del corpus y consulta por agentes](../../../docs/fase2/bancos_riesgo/corpus_para_agentes.md).
- [Arquitectura, módulos, contratos y validación](../../../docs/fase2/bancos_riesgo/arquitectura.md).
- [Inventario de fuentes](../../../docs/fase2/bancos_riesgo/fuentes.md).
- [Modelo de datos y TimeGate](../../../docs/fase2/bancos_riesgo/modelo_y_timegate.md).
- [Preparación de Cloudflare](../../../docs/fase2/bancos_riesgo/cloudflare.md).
- [Orden de implementación](../../../docs/fase2/bancos_riesgo/roadmap.md).
- [Análisis del notebook de referencia](../../../docs/fase2/bancos_riesgo/analisis_notebook_referencia.md).
