# Tema 4 · Noticias y datos alternativos

**Rama:** `fase2/t4-noticias-alt` · **Agentes:** Macro, Equity, Risk (CIO: botón de cataclismo) · **Tools:**
`buscar_noticias(q, fecha)`, `get_polymarket(evento, fecha)` y las que se definan para alternativos · **Destino:**
noticias al cajón, alternativos a la base de series. Reglas comunes: [`docs/fase2/README.md`](../../../docs/fase2/README.md).

## Alcance

Por un lado, **noticias** con su fecha y hora de publicación, históricas y en vivo, de todo el mundo. Por otro, los
**datos alternativos** pedidos en la tutoría. **Todos se investigan y todos entran**
(ver [`docs/tfm/DATOS_ALTERNATIVOS.md`](../../../docs/tfm/DATOS_ALTERNATIVOS.md)).

## Fuentes a investigar

- [ ] Noticias históricas **con texto**: Common Crawl CC-NEWS, datasets de Hugging Face, NYT y Guardian (clave gratuita)…
- [ ] GDELT (eventos, GKG con tono, DOC API)
- [ ] Wikipedia (Current Events, visitas por artículo)
- [ ] Noticias en vivo: RSS de medios, agregadores (respetando robots.txt)
- [ ] Operaciones del Congreso de EE. UU. («Pelosi tracker»): fecha del informe, no de la operación
- [ ] Insiders (Form 4)
- [ ] Mercados de predicción: Polymarket, Kalshi, PredictIt (liquidez por año)
- [ ] Índices: riesgo geopolítico (GPR), incertidumbre (EPU), SKEW, put/call, short volume
- [ ] EWS de jets privados (botón de cataclismo; solo en vivo)
- [ ] Redes y búsquedas (Reddit, X, Google Trends…) y sus restricciones
- [ ] Práctica previa *Briefly*: qué extractores se pueden repensar para nuestro caso

## Entregables

1. Un extractor por fuente con su ficha y sus tests; noticias como `Documento` (fecha de publicación, medio, idioma,
   entidades) y alternativos como series en tabla larga.
2. Mapa de cobertura por épocas: qué texto completo hay en cada periodo.
3. Reglas de limpieza: duplicados y sindicación, idioma y medios.

## Preguntas abiertas

- ¿Qué cobertura de noticias con texto es suficiente para el backtest?
- ¿Licencias de los datasets de terceros?
- ¿Qué umbral del EWS dispara el cataclismo?
