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
- [x] FRED / ALFRED: FRED como fuente viva de todas sus series del catálogo (`fuentes.fred`, fichas `fred_alfred` e
      `ice_bofa_oas`) y vintages exactos de ALFRED de las 161 revisables (bloque 3: `fuentes.alfred`, ficha `alfred_vintages`)
- [x] Académicas: Ken French, Shiller, Goyal-Welch y OFR FSI en la base; JST R6 como tabla de validación
      (`fuentes.academicos`, fichas `ken_french`, `shiller`, `goyal_welch`, `ofr_fsi`, `jst_macrohistory`)
- [ ] Macro internacional: OCDE, BCE, Eurostat, FMI, Banco Mundial, BIS, DBnomics, bancos centrales
- [ ] Tipo libre de riesgo, baselines (buy & hold, 60/40, momentum) y costes de transacción

## Entregables

1. Un extractor por fuente con su ficha y sus tests (ver «cuándo está hecha una fuente» en `docs/fase2`).
2. **Hecho (bloque 1)**: contrato de `get_precios` y `get_serie`: qué devuelven, con qué fecha de publicación, cómo se
   ajusta el precio a fecha (ver «Contrato (bloque 1)»).
3. **Hecho (bloque 2)**: base de series con fase1 + FRED + académicas (ver «Base de series (bloque 2)»).
4. **Hecho (bloque 3)**: vintages exactos de ALFRED, FRED como fuente viva y ediciones académicas con vintage propio
   (ver «Vintages (bloque 3)»).
5. Limpieza y EDA de las series (huecos, rupturas, series retiradas): inventario en el notebook 02 y revisiones en el 03.

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
Desde el bloque 3 las revisables de FRED van con sus vintages exactos de ALFRED y las académicas (French, Shiller,
Goyal-Welch) con un vintage por edición (`AAAA-MM`); solo quedan con vintage `unico` y la nota obligatoria
(`catalogo.NOTAS_REVISABLE`, lo comprueba `catalogo.validar_contra_catalogo`) la OFR FSI y JST, que no tienen
historia de ediciones.

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

**Cómo se construye.** `python -m ingesta.mercados_series construir-base [--sin-red] [--desde-cero] [--reemplazar]
[--purgar] [--sin-escribir]` llama a
[`base.construir_base`](base.py): carga los cuatro cargadores (bloque 3: `alfred`, `fred` vivo, `academicos` y
`fase1`), une, valida y escribe
`<corpus>/mercados_series/series/base_series.parquet` (D:/tfm_corpus en esta máquina; fuera del repositorio)
con `almacen.escribir_series`, y deja la cobertura por serie (solo metadatos) en
`results/ingesta/mercados_series/tablas/cobertura_series.csv`. `--sin-red` usa solo los crudos ya archivados
y `data/raw` (unos 7 minutos con las 161 series de ALFRED); sin él, FRED baja lo que no tenga hoy en caché, ALFRED
las series sin descarga completa y los académicos la edición del día. `--desde-cero` copia antes la base guardada a
`series/respaldo_bloque2.parquet` (si ya existe, se conserva) y la sustituye entera: es lo que hace falta cuando
cambian las reglas o las fuentes. Base que solo crece: las revisiones de valores de las revisables con vintage `unico`
(ediciones académicas nuevas) se admiten solas (`almacen.escribir_series(revisables=...)`: fechas, método y
nota quedan fijos); `--reemplazar` deja cambiar cualquier fila (correcciones propias); una serie guardada que
ya no se construye es un error salvo `--purgar`; `--sin-escribir` no escribe nada. Antes de derivar se aplican
las excepciones de publicación medidas (abajo) y después se anotan en la `nota`, sin tocar valores, los índices
reconstruidos y las anomalías de la fuente (picos aislados que se revierten, cierres en fin de semana, rachas
planas: `tablas/anomalias_series.csv`). `python -m ingesta.mercados_series resumen [--csv]` resume la base
guardada. Recorrido con
gráficos (inventario, cobertura, retrasos, retiradas, el mismo día visto por `get_serie`):
[`notebooks/fase2/mercados_series/02_base_fase1.ipynb`](../../../notebooks/fase2/mercados_series/02_base_fase1.ipynb).

**Fuentes y estado** (una serie, una fuente; si dos cargadores traen el mismo id gana `alfred` > `academicos` >
`fred` > `fase1`, el orden de `catalogo.PRIORIDAD_ORIGENES`; las derivadas se recalculan después):

| cargador | qué aporta | estado |
|---|---|---|
| [`fuentes.alfred`](fuentes/alfred.py) | vintages exactos de ALFRED de las 161 series `vintage_exacto` (empleo, precios, PIB, crédito, encuestas, índices de estrés y de recesión...) | hecho (bloque 3); ficha `alfred_vintages` |
| [`fuentes.fred`](fuentes/fred.py) | FRED como fuente viva: las 80 series de FRED que no van por ALFRED (H.15, H.10, Moody's, NY Fed, Cboe en FRED, OAS de ICE BofA...), con su regla y las excepciones medidas; descarga incremental | hecho (bloque 3); fichas `fred_alfred`, `ice_bofa_oas` |
| [`fuentes.academicos`](fuentes/academicos.py) | Ken French, Shiller (CAPE, tipo largo, P, D, E, CPI) y Goyal-Welch con vintage por edición (`AAAA-MM`); OFR FSI con la última edición | hecho; fichas `ken_french`, `shiller`, `goyal_welch`, `ofr_fsi`, `jst_macrohistory` |
| [`fuentes.fase1`](fuentes/fase1.py) | `data/raw` de la Fase 1: solo lo que no está en FRED (índices de yfinance, datahub) y respaldo de FRED si no trae filas | hecho; ficha `fase1_raw` |
| derivadas propias | `deriv:A-B` recalculadas con sus componentes de la base (DBAA−DAAA, DED3−DTB3, SOFR−EFFR y, desde el bloque 3, BAA10Y = DBAA−DGS10, T10YFF = DGS10−DFF, T5YFF = DGS5−DFF, CPFF = DCPF3M−DFF); la vol. realizada del S&P 500 la fecha `fase1` | hecho |
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
- `vintage_exacto`: la fecha de cada vintage de ALFRED a la hora de su release (ver «Vintages (bloque 3)»).

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

- Revisiones: resueltas con ALFRED para las 161 revisables de FRED (bloque 3). French, Shiller y Goyal-Welch no
  están en ALFRED: su historia de ediciones empieza en la copia de `data/raw` (julio de 2026) y el archivo propio;
  los valores de la primera edición llevan `catalogo.NOTA_PRIMERA_EDICION` (posible fuga de revisiones anteriores).
  La OFR sigue con la última edición y la nota de revisable.
- Fechas antes de la primera vintage de ALFRED de cada serie: valor del primer vintage y fecha de la regla previa
  (puede adelantar la publicación real en periodos anómalos; aviso «anterior a ALFRED»). H.15 diario histórico y
  Ken French (+45 días) sin medición histórica.
- OAS de ICE BofA: solo desde 2023-10-10 (ventana móvil sin confirmar); para no perder historia basta una
  descarga dentro de cada ventana de tres años; la diaria solo hace falta para tener el dato al día
  (`python -m ingesta.mercados_series construir-base`, programable).
- `data/raw` acaba el 2026-07-17 en las diarias (descarga del 2026-07-20); desde el bloque 3 solo sirve los
  índices de yfinance y datahub: todo lo de FRED sale de la fuente viva o de ALFRED y llega a octubre de 2026.
- Retrasos por regla sin calendario oficial (Goyal-Welch, NBER, Hamilton, OFR...): `verificado: false` en
  `lags.yaml`; los de FRED, contrastados con ALFRED donde hay vintages.
- `data/raw`: el instante de descarga está declarado en `config/descargas_fase1.yaml` (si el mtime cambia por
  una copia, se avisa y manda lo declarado).

**Licencias.** Los datos se quedan en local (D:/tfm_corpus). Al repositorio solo van código, fichas,
tablas de metadatos sin valores y fixtures sintéticas. FRED: solo local, sin redistribuir. ICE, S&P, Moody's,
Cboe, FTSE Russell, Nasdaq, Bloomberg y Wilshire conservan el copyright de sus series; Ken French y Shiller,
uso académico con cita; JST, CC BY-NC-SA 4.0 (campo `licencia` del catálogo).
*This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.*

## Vintages (bloque 3)

**Qué hay.** Las 161 series de `tablas/alfred_lista_propuesta.csv` (aprobada) se cargan con TODAS sus vintages de
ALFRED ([`fuentes.alfred`](fuentes/alfred.py)): `series/observations` con `output_type=1` y tiempo real completo,
paginado (cada fila de ALFRED es un periodo de validez: solo valores nuevos o revisados). En la base: 7,19 millones
de filas de 161 series, unas 135 000 vintages; de ellas, 74 621 filas anteriores a ALFRED, 19 101 `'.'` (huecos
de la fuente, con nota) y 116 observaciones retiradas. Crudo: 764 MB de JSON en `<corpus>/crudo/alfred/`.

**Método `vintage_exacto`.** `vintage` = `realtime_start` (`AAAA-MM-DD`); `disponible_utc` = hora del release de su
`publicador` ese día (BLS, BEA y Census 08:30; G.17 09:15; JOLTS y Michigan 10:00; H.8 16:15 + 30 min y H.4.1
16:30 + 5 min, que pasan a la decisión del día siguiente; publicadores sin hora conocida, 23:59, conservador);
`fecha_publicacion` = primera decisión de las 16:30 NY a partir de ahí. Casos especiales:

- **Anterior a ALFRED** (vintage de arranque: el primer vintage trae toda la historia de golpe): esas observaciones
  llevan el valor del primer vintage, `lag_fijo` con la regla previa del catálogo (`regla_previa_alfred`), vintage =
  la fecha de la regla y la nota «anterior a ALFRED: valor del primer vintage (AAAA-MM-DD)». Sin regla previa
  (WEI, GDPNow, SEP), solo se ven desde el primer vintage.
- **Relleno de historia**: periodos antiguos que ALFRED añadió más tarde (USREC 1854-1920 el 2011-11-01, PPI o Z.1
  antiguos) se ven desde el vintage que los añadió (conservador); las medidas de retraso los excluyen.
- **Índices re-estimados enteros** (NFCI y subíndices, STLFSI4, CFSI, KCFSI, RECPROUSM156N, CLI OCDE, BBK, USPHCI):
  mismo método, cada vintage cambia toda la historia; nota y aviso «re-estimado entero».
- **Unidad por vintage** (dólares encadenados, años base: GDPC1, PCEC96, INDPRO...): `unidad` es la del vintage
  vigente; los niveles de vintages distintos no son comparables (nota y aviso).
- **Proyecciones del SEP**: visibles desde su fecha de referencia si siguen vigentes (el contrato no admite previsiones).
- **Derivadas de FRED** (CFNAIMA3, A191RL1Q225SBEA, SAHMCURRENT): coherentes con su origen dentro de cada vintage
  (`alfred.coherencia_derivada`; tolerancias medidas: 0,0101, 0,1 y 0,0051).

**FRED como fuente viva.** Todas las series de FRED del catálogo salen de la API: las 161 revisables por ALFRED y
las 80 restantes por la fuente viva (`fuentes.fred.tabla_fred_vivo`, descarga completa la primera vez e incremental
después, con las excepciones de publicación medidas). BAA10Y, T10YFF, T5YFF y CPFF son ahora derivadas propias
(D+1, máxima fecha de sus componentes); la serie de FRED queda como `duplicada` (reproducida salvo una diferencia de
0,02 el 1991-01-29: `tablas/derivadas_fred_medida.csv`). Los alias de la Fase 1 (`BAA10Y`, `T10YFF`, `T5YFF`) apuntan
a la derivada. `data/raw` coincide con la fuente viva en todas las fechas comunes de las 64 series que comparten (0
valores distintos, 0 fechas solo en `data/raw`; FRED añade 2721 fechas recientes: `tablas/fred_vivo_contraste_fase1.csv`).

**Ediciones académicas.** French, Shiller y Goyal-Welch llevan un vintage por edición (`AAAA-MM`): la copia de
`data/raw` es la edición previa y cada descarga archivada (`crudo/<fuente>/ediciones/`) una nueva; los valores
revisados entran como filas nuevas (`tablas/academicos_ediciones.csv`: French revisó el 54 % de los valores
comunes del momentum diario; Shiller y Goyal-Welch, ninguno).

**Cómo se descarga y se construye** (tandas de menos de 10 minutos, reanudables, 1 petición/s; cada tanda deja una
fila en `<corpus>/mercados_series/logs/descargas.csv`):

```
python -m ingesta.mercados_series descargar-alfred          # repetir mientras salga con código 2
python -m ingesta.mercados_series descargar-fred-vivo       # ídem
python -m ingesta.mercados_series construir-base --sin-red [--desde-cero]
python -m ingesta.mercados_series informe-alfred            # tablas de revisiones y de la regla previa
python -m ingesta.mercados_series.fuentes.alfred --ficha    # ficha alfred_vintages
python -m ingesta.mercados_series.fuentes.fred --vivo --sin-red --fichas
```

Medido el 2026-10-13: ALFRED, 213 peticiones en dos tandas (424 s y 44 s), 0 errores; FRED vivo, 77 peticiones en
83 s, 0 errores; construcción desde cero 442 s; base de 8 636 447 filas y 322 series en 78 MB (row groups de 65 536
filas ordenados por serie). `get_serie` lee solo la serie pedida y lo publicado hasta la fecha, y resuelve la última
fila por periodo en Arrow (`almacen.leer_ultimas`): 0,04-0,4 s por consulta, también en los índices de un millón de
filas.

**`get_serie`** no avisa en las filas `vintage_exacto`; sí en las anteriores a ALFRED, los índices re-estimados,
la unidad por vintage, las ediciones académicas (con las de la primera edición archivada) y los huecos `'.'` (IPC
de octubre de 2025). Ejemplos verificados en la base real (tests `@datos` de `test_t1_base.py`): PAYEMS de noviembre
de 2008 el 2008-12-10 es la primera estimación del 2008-12-05, distinta de la revisada; UNRATE de abril de 2020 se
ve el 2020-05-08 y no el 07; el empleo de septiembre de 2013 se ve el 2013-10-22 (cierre del gobierno).

**Cuánto filtraba el bloque 2** (`tablas/alfred_vs_lag.csv`, `tablas/alfred_revisiones_resumen.csv`, sin niveles):
la regla fija adelantaba la fecha del 2,8 % de los 48 161 periodos medidos (128 de 156 series, hasta 144 días) y
todos los valores eran los revisados de hoy (el 99 % de los periodos se revisa en la serie mediana). Recorrido con
gráficos: [`notebooks/fase2/mercados_series/03_alfred_vintages.ipynb`](../../../notebooks/fase2/mercados_series/03_alfred_vintages.ipynb).

## Deudas pendientes

- Archivo periódico de las OAS de ICE (ventana móvil de 3 años en FRED): pendiente de programar.
- package-data de los YAML de config en `pyproject.toml`: pendiente (PR común cuando haga falta empaquetar; hoy
  funciona con `pip install -e`).
- Descarga periódica de las ediciones académicas (French mensual, Shiller, Goyal-Welch anual) para medir sus
  revisiones futuras: hoy se archiva a mano (`fuentes.academicos --descargar` y `--archivar`).

## Decididas

- Divisa base: USD (se guarda la original y se convierte al leer).
- Hora de decisión: 16:30 America/New_York de cada día natural (tras el cierre de Nueva York).

## Preguntas abiertas

- ¿Fuente de precios para la versión final (gratuita o de pago)?
- ¿Cómo se corrige o se declara el sesgo de supervivencia?
