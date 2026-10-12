# Tema 1 · Mercados y series numéricas

**Rama:** `fase2/t1-mercados-series` · **Agentes:** Portfolio, Equity, Macro, Risk · **Tools:** `get_precios(tk, fecha)`, `get_serie(id, fecha)`
· **Destino:** base de series (no vectorial). Reglas comunes: [`docs/fase2/README.md`](../../../docs/fase2/README.md).

## Alcance

Todo dato **numérico** que consultan los agentes:
- precios del universo de activos;
- curvas de tipos y divisas;
- series macro, con su valor tal como se conocía en cada fecha (vintages);
- tipo libre de riesgo, carteras de referencia (baselines) y costes de transacción.

El tutor pidió **la mayor cantidad de activos básicos** a nivel mundial. Los fondos de inversión (y los fondos de fondos)
hay que evaluarlos y justificar si entran o no.

## Fuentes a investigar

- [ ] Precios: yfinance (acciones por mercado, ETF, índices, FX, futuros, cripto) y alternativas (EODHD, Tiingo, Stooq…)
- [ ] Constituyentes históricos de índices (S&P 500, STOXX 600…) y sesgo de supervivencia
- [ ] Curvas soberanas oficiales (US Treasury, BCE, Bundesbank, BoE, MoF Japón…)
- [ ] Divisas y tipos del euro (BCE)
- [ ] Cripto (fuentes públicas de velas)
- [ ] Fondos de inversión (NAV, CNMV, ESMA) — viabilidad
- [x] Reutilizar las series de la Fase 1 (`data/raw`, `configs/catalog.yaml`, `regimenes.features.lags`) como **niveles**
      legibles: hecho en el bloque 2 (`fuentes.fase1`, ficha `fase1_raw`); las ETF ajustadas quedan para `get_precios`
- [ ] FRED / ALFRED (en curso): sustitutos y componentes que no están en `data/raw` hechos en el bloque 2 (`fuentes.fred`, fichas
      `fred_alfred` e `ice_bofa_oas`); vintages exactos de las revisables pendientes (bloque 3)
- [x] Académicas: Ken French, Shiller, Goyal-Welch y OFR FSI en la base; JST R6 como tabla de validación
      (`fuentes.academicos`, fichas `ken_french`, `shiller`, `goyal_welch`, `ofr_fsi`, `jst_macrohistory`)
- [ ] Macro internacional: OCDE, BCE, Eurostat, FMI, Banco Mundial, BIS, DBnomics, bancos centrales
- [ ] Tipo libre de riesgo, baselines (buy & hold, 60/40, momentum) y costes de transacción

## Entregables

1. Un extractor por fuente con su ficha y sus tests (ver «cuándo está hecha una fuente» en `docs/fase2`).
2. **Hecho (bloque 1)**: contrato de `get_precios` y `get_serie`: qué devuelven, con qué fecha de publicación, cómo se
   ajusta el precio a fecha (ver «Contrato (bloque 1)»).
3. **Hecho (bloque 2)**: base de series con fase1 + FRED + académicas (ver «Base de series (bloque 2)»).
4. Limpieza y EDA de las series (huecos, rupturas, series retiradas): primer inventario en el notebook 02.

## Contrato (bloque 1)

**Hora de decisión:** 16:30 America/New_York (con cambio de hora) de cada día natural. Un cierre de
mercado es visible a su hora local de cierre + 30 min ([`config/mercados.yaml`](config/mercados.yaml));
una publicación, a su hora declarada ([`config/publicaciones.yaml`](config/publicaciones.yaml)).

**Regla del TimeGate:** `fecha_publicacion` = primera fecha de decisión cuyo momento (16:30 NY) es
posterior o igual al instante en que el dato es público (`disponible_utc`, se guarda para auditar). Las
tools solo devuelven filas con `fecha_publicacion <= fecha`. La base solo crece: lo que se sabía en una
fecha no cambia al añadir datos publicados después (truncado). En el almacén, ni una serie ni una vela ya
guardadas se pueden cambiar (una corrección tardía heredaría la fecha antigua y se vería hacia atrás),
salvo con `permitir_perdidas` / `permitir_correcciones` para errores propios. El truncado alcanza también
a los mensajes: sin nada publicado hasta `fecha`, las tools responden igual que si el id no existiera
(sin revelar si existirá ni cuándo). La `fecha` es un día natural: `date`, `Timestamp` a medianoche o
texto ISO `AAAA-MM-DD` (se rechazan `02/01/2024`, `today`, enteros y horas distintas de 00:00).

**Series** (tabla larga; una fila = un valor tal como se publicó; clave `(serie, fecha_referencia, vintage)`):

| columna | tipo | significado |
|---|---|---|
| `serie` | str | id canónico con prefijo de fuente (`fred:DGS10`, `ecb:EXR.D.USD.EUR`); alias opcionales |
| `fecha_referencia` | fecha | periodo: día 1 del mes o trimestre, o la sesión |
| `fecha_publicacion` | fecha | primera fecha de decisión en que es visible |
| `disponible_utc` | instante UTC | cuándo se hizo público (nulo si solo se conoce el día) |
| `vintage` | str | `AAAA-MM-DD` (ALFRED), `AAAA-MM` (edición) o `unico` |
| `valor` | float | nulo solo con `nota` (hueco permanente) |
| `unidad`, `divisa`, `fuente` | str | divisa ISO o nula si no es monetaria |
| `metodo_publicacion` | str | `vintage_exacto`, `calendario`, `lag_fijo`, `cierre_mercado`, `derivada`, `fichero` |
| `nota` | str | incidencias (hueco, retraso) |

Consulta as-of: por cada `fecha_referencia`, la fila con mayor `fecha_publicacion <= fecha` (empate:
mayor `vintage`). Nada se publica antes de su periodo (`fecha_publicacion >= fecha_referencia`; el contrato
no admite previsiones). Una serie derivada lleva la mayor `fecha_publicacion` de sus componentes
(`contrato.fecha_publicacion_derivada`; los extractores la usarán). Regla para los extractores: una serie
revisable (`revisable: true` en `config/publicaciones.yaml`) se carga con sus vintages (ALFRED), nunca con el
histórico de hoy y vintage `unico`, que pondría el valor revisado con la fecha de la primera publicación.
Única excepción (bloque 2): las revisables de `data/raw` se cargan ya con `lag_fijo` o `fichero`, vintage `unico`
y la nota obligatoria `catalogo.NOTA_REVISABLE` (lo comprueba `catalogo.validar_contra_catalogo`); el bloque 3
las sustituye por vintages exactos.

**Precios** (as-traded, sin ajustar; clave `(ticker, sesion)`):

| columna | tipo | significado |
|---|---|---|
| `ticker`, `sesion`, `mercado` | str, fecha, str | `mercado` = clave de `config/mercados.yaml` |
| `open`, `high`, `low`, `close`, `volume` | float | OHLCV crudo |
| `dividendo` | float | por acción, en divisa de cotización, en la fecha ex (0 sin evento) |
| `split` | float | ratio nuevo/antiguo (1.0 sin evento) |
| `divisa` | str | de cotización; subunidades `GBp`, `ZAc`, `ILA` |
| `fecha_publicacion`, `disponible_utc` | fecha, instante UTC | cierre local + 30 min |
| `fuente`, `descargado` | str, instante UTC | velas incompletas descartadas |

Nunca se guardan columnas con «adj». El ajuste (splits y dividendos, método multiplicativo de Yahoo)
se reconstruye a la fecha de decisión con solo los eventos visibles en ella. Yahoo no da precios
as-traded ni con `auto_adjust=False`: OHLC y dividendos vienen divididos por los splits posteriores conocidos
al descargar (y el volumen multiplicado). El extractor los pasa a as-traded con `timegate.des_ajustar_yahoo`
(ida y vuelta probada contra `sinteticos.vista_yahoo_hoy`, con contrasplit, split 3:2 y split + dividendo el
mismo día), descarta las velas incompletas (`contrato.descartar_velas_incompletas`) y valida: la vela debe
ser visible después del cierre de su mercado + 30 min. La divisa original se guarda; la conversión a USD se
hace al leer con el FX de referencia del BCE visible en la fecha (en series mensuales o trimestrales, el del
fin del periodo). Cobertura del BCE: desde 1999-01-04 y unas 30 divisas (sin TWD); fuera de ella, `NaN` y
un aviso (pendiente: FX secundario de la Fed, H.10).

Calendario `us_federal`: festivos federales más los cierres extraordinarios de `publicaciones.yaml`
(lutos nacionales, Nochebuenas por orden ejecutiva, huracanes); conservador. Las fichas de fuente que cita
`publicaciones.yaml` (`fred_alfred`, `ecb_exr`) se escribirán con los extractores.

Módulos: [`contrato`](contrato.py) (columnas, validadores), [`visibilidad`](visibilidad.py) (cierres y
publicaciones), [`timegate`](timegate.py) (`as_of_series`, `ajustar_a_fecha`), [`divisas`](divisas.py),
[`almacen`](almacen.py) (parquet), [`tools`](tools.py) (`get_serie`, `get_precios`) y
[`sinteticos`](sinteticos.py) (datos sin red). Demostración con gráficos y prueba de truncado en directo:
[`notebooks/fase2/mercados_series/01_contrato_timegate.ipynb`](../../../notebooks/fase2/mercados_series/01_contrato_timegate.ipynb).

## Catálogo de series (bloque 2)

[`config/catalogo_series.yaml`](config/catalogo_series.yaml) es la única fuente de verdad de las series del
tema: una entrada por serie (id canónico con prefijo de fuente, alias de la Fase 1, unidad y divisa exactas,
método de publicación, licencia, estado, duplicados y sucesores). [`catalogo`](catalogo.py) la valida y
`visibilidad.cargar_publicaciones` fusiona su parte de publicación con
[`config/publicaciones.yaml`](config/publicaciones.yaml) con las mismas reglas (un id o alias en los dos
ficheros es un error). Los retrasos de `lag_fijo` y `fichero` salen de [`config/lags.yaml`](config/lags.yaml):
copia congelada de `LAG_PUBLICACION` (ADR-003) convertida a «desde el fin del periodo» más las series no
diarias que la Fase 1 no cubría. Estados: `activa`, `retirada`, `capada` (OAS de ICE BofA: FRED y ALFRED solo
conservan desde 2023-10-10), `excluida_ajustada` (ETF de yfinance con `auto_adjust=True`: a `get_precios`),
`pendiente_alfred` y `duplicada`. Las series revisables cargadas con el último vintage llevan una nota
obligatoria y `get_serie` avisa de la posible fuga de revisiones: las de FRED, `catalogo.NOTA_REVISABLE`
(«ALFRED en el bloque 3»); las académicas y la OFR, que no están en ALFRED, `catalogo.NOTA_REVISABLE_EDICION`
(archivo propio de ediciones desde el 2026-10-12). Los tipos de cambio (DEX*) llevan `divisa: null`: son
cotizaciones, no importes convertibles. `publicado_desde` marca los índices construidos hacia atrás (NFCI y
subíndices, STLFSI4, regla de Sahm): sus filas anteriores llevan `catalogo.NOTA_RECONSTRUIDA` y `get_serie`
avisa.

Horas de publicación medidas ([`sonda_horas`](sonda_horas.py), solo fechas en
`results/ingesta/mercados_series/tablas/sondas_horas_bloque2.csv`): H.15 publica el dato de D el hábil siguiente
a las 16:15 NY; H.10, el lunes siguiente (martes si es festivo) a las 16:15; Moody's (DBAA, DAAA), el hábil
siguiente hacia las 11:30; los diferenciales que FRED calcula con la curva del Tesoro (T10Y2Y, T10YIE...), el
mismo día a las 17:03 (decisión de D+1), y los que dependen de un dato de D+1 (BAA10Y, T10YFF, CPFF), en D+1 a
las 17:03 (decisión de D+2). Metadatos de FRED: `results/ingesta/mercados_series/tablas/metadatos_fred_bloque2.csv`.

Términos de FRED: los datos se guardan solo en local, sin redistribuir. *This product uses the FRED® API but
is not endorsed or certified by the Federal Reserve Bank of St. Louis.* ICE, S&P, Moody's y Cboe conservan el
copyright de sus series (campo `licencia` del catálogo).

## Base de series (bloque 2)

**Cómo se construye.** `python -m ingesta.mercados_series construir-base [--sin-red] [--reemplazar] [--purgar]
[--sin-escribir]` llama a
[`base.construir_base`](base.py): carga los tres cargadores, une, valida y escribe
`<corpus>/mercados_series/series/base_series.parquet` (D:/tfm_corpus en esta máquina; fuera del repositorio)
con `almacen.escribir_series`, y deja la cobertura por serie (solo metadatos) en
`results/ingesta/mercados_series/tablas/cobertura_series.csv`. `--sin-red` usa solo los crudos ya archivados
y `data/raw` (alrededor de un minuto); sin él, FRED baja lo que no tenga hoy en caché y los académicos la
edición del día. Base que solo crece: las revisiones de valores de las revisables con vintage `unico`
(ediciones académicas nuevas) se admiten solas (`almacen.escribir_series(revisables=...)`: fechas, método y
nota quedan fijos); `--reemplazar` deja cambiar cualquier fila (correcciones propias); una serie guardada que
ya no se construye es un error salvo `--purgar`; `--sin-escribir` no escribe nada. Antes de derivar se aplican
las excepciones de publicación medidas (abajo) y después se anotan en la `nota`, sin tocar valores, los índices
reconstruidos y las anomalías de la fuente (picos aislados que se revierten, cierres en fin de semana, rachas
planas: `tablas/anomalias_series.csv`). `python -m ingesta.mercados_series resumen [--csv]` resume la base
guardada. Recorrido con
gráficos (inventario, cobertura, retrasos, retiradas, el mismo día visto por `get_serie`):
[`notebooks/fase2/mercados_series/02_base_fase1.ipynb`](../../../notebooks/fase2/mercados_series/02_base_fase1.ipynb).

**Fuentes y estado** (una serie, una fuente; si dos cargadores traen el mismo id gana `academicos` >
`fred` > `fase1`):

| cargador | qué aporta | estado |
|---|---|---|
| [`fuentes.fase1`](fuentes/fase1.py) | `data/raw` de la Fase 1 (FRED, yfinance, datahub) | hecho; ficha `fase1_raw` |
| [`fuentes.fred`](fuentes/fred.py) | sustitutos y componentes de FRED/ALFRED: DBAA, DAAA, DEXUSEU, DTWEXBGS, DTWEXAFEGS, TEDRATE, DCPF3M, CPFF, 8 OAS de ICE BofA (archivo propio desde 2026-10-12) y WEI con sus vintages | hecho; fichas `fred_alfred`, `ice_bofa_oas` |
| [`fuentes.academicos`](fuentes/academicos.py) | Ken French, Shiller (CAPE, tipo largo, P, D, E, CPI), Goyal-Welch, OFR FSI; usan la copia de `data/raw` como edición antigua | hecho; fichas `ken_french`, `shiller`, `goyal_welch`, `ofr_fsi`, `jst_macrohistory` |
| derivadas propias | `deriv:A-B` recalculadas con sus componentes de la base (DBAA−DAAA, DED3−DTB3, SOFR−EFFR); la vol. realizada del S&P 500 la fecha `fase1` | hecho |
| bloque 3 | vintages exactos de ALFRED para las revisables | pendiente |
| bloque 4 | OCDE (CLI) y macro internacional | pendiente |
| bloque 6 | precios as-traded (`get_precios`), incluidas las ETF de la Fase 1 | pendiente |

**Reglas de fecha por método** (`metodo_publicacion`):

- `cierre_mercado`: cierre del mercado + 30 min. NYSE/Nasdaq (índices S&P, Dow, Russell) → la decisión del
  mismo día; Cboe (VIX y familia, 16:15), CME y el DXY de ICE → la del día siguiente.
- `calendario`: hora del publicador de `publicaciones.yaml`: H.15 el hábil siguiente a las 16:15; H.10 el
  lunes siguiente; Moody's D+1 a las 12:00; ICE en FRED D+1 a las 09:15; SOFR 08:00 y EFFR 09:00; diferenciales
  calculados por FRED a las 17:05 (D+1 o D+2).
- `lag_fijo`: fin del periodo + `lag` de `config/lags.yaml` (macro mensual y semanal de FRED, OFR FSI);
  `disponible_utc` nulo. Con `habiles` la fecha pasa al primer hábil (medias mensuales de H.15, Chauvet-Piger)
  y con `n_habiles` nunca antes del n-ésimo hábil (NFCI y subíndices: tercer hábil tras el viernes).
- `fichero`: salida del fichero académico: Ken French fin de mes + 45 días, Goyal-Welch 1 de agosto del año
  siguiente, Shiller día 1 de M+2 (D y E, día 1 del cuarto mes tras el FIN DEL TRIMESTRE: sus dos primeros
  meses son interpolaciones hacia el dato trimestral), y nunca antes de una edición archivada que aún no
  tenía el periodo.
- `derivada`: la máxima `fecha_publicacion` de sus componentes (`contrato.fecha_publicacion_derivada`).
- `vintage_exacto`: la fecha de cada vintage de ALFRED (WEI; sus semanas de 2008-2020 solo son visibles desde
  la primera vintage, 2020-04-16).

**Excepciones medidas.** Las reglas fallan cuando la publicación se retrasa (cierres del Gobierno de 1995-96,
2013, 2018-19 y 2025-26; Moody's fuera de FRED de octubre de 2016 a marzo de 2017; medias de H.15 de 1997-2016
publicadas con la edición semanal; series de FRED calculadas con LIBOR, con una semana de retraso).
`python -m ingesta.mercados_series excepciones-alfred` compara la regla de cada serie de FRED `lag_fijo` y de
los publicadores de `fuentes.fred.PUBLICADORES_ALFRED` con la primera vintage de ALFRED (`output_type=4`) de
cada observación posterior a la primera vintage de la serie, y escribe `config/excepciones_publicacion.csv`
(solo fechas) y `tablas/publicacion_alfred_medida.csv`. La base usa `fecha_publicacion = max(regla,
ALFRED)` (`catalogo.aplicar_excepciones`, antes de las derivadas). Hay que regenerarla cuando cambian las
reglas o llegan periodos nuevos (reanudable: la caché del día no repite peticiones). Antes de la primera
vintage de cada serie no hay medición.

**Exclusiones.** Las 24 ETF de yfinance de la Fase 1 (SPY, RSP, XL*, IWM, IWD, IWF, GLD, HYG, IEF, LQD, TLT,
SPHB, SPLV, MTUM) se bajaron con `auto_adjust=True`: sus niveles se reajustaron a posteriori y no entran
(`excluida_ajustada`); `get_serie` lanza `KeyError` y remite a `get_precios`. La barra intradía del día de
descarga de yfinance se descarta. `SHILLER_LONGRATE` de `data/raw` está corrupto (octubres leídos como
enero): la serie sale de `ie_data.xls`. JST R6 no entra en la base (tabla de validación `tabla_jst`).

**`get_serie`.** Resuelve los alias de la Fase 1 desde el catálogo (`tools.alias_por_defecto`: catálogo más
`publicaciones.yaml`); un id sin alias se resuelve sin distinguir mayúsculas si es único (`gw:M.TBL`). Avisa,
como mucho una vez por tipo: huecos (agrupados con su recuento), fechas por regla (`lag_fijo`: la regla no es
la fecha real; corregida con ALFRED donde se midió), fechas retrasadas a ALFRED, índices reconstruidos y
revisables cargadas con el último vintage o edición.

**Limitaciones.**

- Revisiones: las revisables de FRED (CPI, PAYEMS, UNRATE, INDPRO, PIB, NFCI...) llevan los valores de hoy con
  vintage `unico`: posible fuga de revisiones hasta que el bloque 3 cargue sus vintages exactos de ALFRED
  (reescritura con `permitir_perdidas=True`). French, Shiller, Goyal-Welch y la OFR no están en ALFRED: su
  única historia de ediciones es el archivo propio de crudos, desde el 2026-10-12 (el momentum diario de
  French cambió en el 56 % de los días entre las ediciones de julio y septiembre de 2026).
- Fechas antes de la primera vintage de ALFRED de cada serie: solo la regla (puede adelantar la publicación
  real en periodos anómalos). H.15 diario histórico y Ken French (+45 días) sin medición histórica.
- OAS de ICE BofA: solo desde 2023-10-10 (ventana móvil sin confirmar); para no perder historia basta una
  descarga dentro de cada ventana de tres años; la diaria solo hace falta para tener el dato al día
  (`python -m ingesta.mercados_series construir-base`, programable).
- `data/raw` acaba el 2026-07-17 en las diarias (descarga del 2026-07-20); los sustitutos de FRED y las
  académicas llegan a octubre de 2026.
- Retrasos por regla sin calendario oficial (Goyal-Welch, NBER, Hamilton, OFR...): `verificado: false` en
  `lags.yaml`; los de FRED, contrastados con ALFRED donde hay vintages.
- `data/raw`: el instante de descarga está declarado en `config/descargas_fase1.yaml` (si el mtime cambia por
  una copia, se avisa y manda lo declarado).

**Licencias.** Los datos se quedan en local (D:/tfm_corpus). Al repositorio solo van código, fichas,
tablas de metadatos sin valores y fixtures sintéticas. FRED: solo local, sin redistribuir. ICE, S&P, Moody's,
Cboe, FTSE Russell, Nasdaq, Bloomberg y Wilshire conservan el copyright de sus series; Ken French y Shiller,
uso académico con cita; JST, CC BY-NC-SA 4.0 (campo `licencia` del catálogo).
*This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.*

## Decididas

- Divisa base: USD (se guarda la original y se convierte al leer).
- Hora de decisión: 16:30 America/New_York de cada día natural (tras el cierre de Nueva York).

## Preguntas abiertas

- ¿Fuente de precios para la versión final (gratuita o de pago)?
- ¿Cómo se corrige o se declara el sesgo de supervivencia?
