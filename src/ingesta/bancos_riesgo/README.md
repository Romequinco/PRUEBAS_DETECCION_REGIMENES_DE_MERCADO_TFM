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

## Entregables

1. Un extractor por fuente con su ficha y sus tests; documentos como `Documento` en jsonl con su `fecha_publicacion`
   real (las actas salen semanas después de la reunión).
2. Catálogo bibliográfico (metadatos + URL abierta, sin texto) en `results/ingesta/bancos_riesgo/`.
3. Propuesta de qué va a la base y qué al cajón, y de cómo trocear cada tipo de documento.

## Preguntas abiertas

- ¿Qué hacer con PDFs que bloquean la descarga automática (descarga manual)?
- ¿Qué modelo para el tono hawkish/dovish (licencia)?
