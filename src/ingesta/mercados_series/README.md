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

- [x] Precios: yfinance as-traded (núcleo de 92 tickers; nivel ampliado de 1.842 tickers en descarga, con posproceso)
      (`proveedores.yahoo`, ficha `yahoo`); Tiingo como alternativa opcional (sin probar: exige cuenta)
- [x] Constituyentes históricos (S&P 500 point-in-time; composición actual de 10 índices más) y sesgo de
      supervivencia declarado y medido (`fuentes.constituyentes`, `supervivencia`; WRDS pendiente)
- [x] Curvas soberanas oficiales: US Treasury, BCE, BoE (ficheros), MoF Japón, BoC y Bundesbank (`fuentes.curvas`,
      `fuentes.bancos_centrales`)
- [x] Divisas y tipos del euro (BCE: EXR, €STR, EONIA, tipos oficiales; `fuentes.bce`)
- [x] Cripto: Coinbase (USD) principal y Binance Vision de control (`fuentes.cripto`)
- [x] Fondos de inversión: fuera de la optimización (solo VBMFX como proxy de AGG antes de 2003-09)
- [x] Reutilizar las series de la Fase 1 (`data/raw`, `configs/catalog.yaml`, `regimenes.features.lags`) como **niveles**
      legibles: hecho en el bloque 2 (`fuentes.fase1`, ficha `fase1_raw`); las ETF ajustadas quedan para `get_precios`
- [x] FRED / ALFRED: FRED como fuente viva de todas sus series del catálogo (`fuentes.fred`, fichas `fred_alfred` e
      `ice_bofa_oas`) y vintages exactos de ALFRED de las 161 revisables (bloque 3: `fuentes.alfred`, ficha `alfred_vintages`)
- [x] Académicas: Ken French, Shiller, Goyal-Welch y OFR FSI en la base; JST R6 como tabla de validación
      (`fuentes.academicos`, fichas `ken_french`, `shiller`, `goyal_welch`, `ofr_fsi`, `jst_macrohistory`)
- [x] Macro internacional: OCDE (ediciones), BCE, Eurostat, FMI (WEO, parcial), Banco Mundial, BIS, Philly RTDSM y
      bancos centrales (bloque 4); DBnomics vetado por `robots.txt`
- [x] Tipo libre de riesgo, baselines (buy & hold, 60/40, 1/N, RSP, momentum sectorial) y costes de transacción
      (bloque 8: `cartera`)

## Entregables

1. Un extractor por fuente con su ficha y sus tests (ver «cuándo está hecha una fuente» en `docs/fase2`).
2. **Hecho (bloque 1)**: contrato de `get_precios` y `get_serie`: qué devuelven, con qué fecha de publicación, cómo se
   ajusta el precio a fecha (ver «Contrato (bloque 1)»).
3. **Hecho (bloque 2)**: base de series con fase1 + FRED + académicas (ver «Base de series (bloque 2)»).
4. **Hecho (bloque 3)**: vintages exactos de ALFRED, FRED como fuente viva y ediciones académicas con vintage propio
   (ver «Vintages (bloque 3)»).
5. Limpieza y EDA de las series (huecos, rupturas, series retiradas): inventario en el notebook 02 y revisiones en el 03.
6. **Hecho (bloques 4–8)**: macro internacional (notebook 04), universo (05), precios as-traded (06), supervivencia
   (07), libre de riesgo, baselines y costes (08). Ver las secciones de cada bloque.
7. **Hecho (bloque 9, cierre)**: calidad global, tabla de fichas y las dos tools en 2008-09-15 y 2020-03-16
   (notebook 09) y truncado real de `get_serie` y `get_precios` (`test_t1_cierre.py`, `@pytest.mark.datos`).

## Cómo usarlo

Datos solo en `<corpus>/mercados_series/` (por defecto `D:/tfm_corpus/mercados_series`); con `pip install -e .`:

```bash
# 1. Descargas (reanudables; cada fuente con robots.txt y ritmo por host)
python -m ingesta.mercados_series descargar-alfred            # vintages de ALFRED (tandas de 8 min)
python -m ingesta.mercados_series descargar-fred-vivo
python -m ingesta.mercados_series descargar-bce     # ídem: eurostat, bis, imf, worldbank, oecd, philly, bancos, curvas, cripto
python -m ingesta.mercados_series.fuentes.academicos --descargar   # French, Shiller, Goyal-Welch, OFR, JST
python -m ingesta.mercados_series descargar-precios --nivel nucleo         # 92 tickers del universo
python -m ingesta.mercados_series descargar-precios --nivel ampliado --fichero logs/tickers_ampliado.csv
python -m ingesta.mercados_series.proveedores.yahoo posproceso              # cobertura y calidad del ampliado
# 2. Base de series (une todas las fuentes; --desde-cero si cambian reglas de fechas, con respaldo)
python -m ingesta.mercados_series construir-base --sin-red [--desde-cero --respaldo respaldo_X.parquet]
python -m ingesta.mercados_series resumen
```

```python
from ingesta.mercados_series.tools import get_serie, get_precios

r = get_serie("DGS10", "2020-03-16")                 # alias o id canónico ('fred:DGS10')
r.datos      # fecha_referencia, valor, fecha_publicacion, vintage, metodo_publicacion, nota
r.avisos     # huecos, fechas por regla, re-estimadas...
p = get_precios(["SPY", "SAP.DE"], "2020-03-16", ajuste="total", divisa="USD")
p.datos      # sesiones visibles a las 16:30 NY, ajuste reconstruido con lo conocido en la fecha y FX del BCE visible
```

Las dos tools devuelven solo lo visible a las 16:30 America/New_York del día pedido: nada con `fecha_publicacion`
posterior, ni sesiones sin cerrar (cierre local + 30 min), ni splits o dividendos futuros en el ajuste.

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
| bloque 4 | macro internacional: BCE, Eurostat, BIS, FMI, Banco Mundial, OCDE, Philly RTDSM y bancos centrales (`base.CARGADORES_OFICIALES`) | hecho; ver «Macro internacional (bloque 4)» |
| bloque 6 | curvas soberanas oficiales (`fuentes.curvas`) en la base; precios as-traded en el almacén de precios (`get_precios`), incluidas las ETF de la Fase 1 | hecho; ver «Precios (bloque 6)» |

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
- **Índices re-estimados enteros** (NFCI y subíndices, STLFSI4, CFSI, KCFSI, RECPROUSM156N, BBK, USPHCI): mismo
  método, cada vintage cambia toda la historia; nota y aviso «re-estimado entero». El CLI de la OCDE no lo es (medido:
  cada edición revisa de mediana el 44-48 % de los periodos comunes): nota propia, sin la marca.
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

## Macro internacional (bloque 4)

Países del alcance: EE. UU., zona euro (composición cambiante `U2`/`EA`; la zona euro pasa a 21 en 2026), G7 (DE, FR,
IT, UK, JP, CA) y China; el resto del mundo, anual (Banco Mundial). Prefijos: `ecb:`, `eurostat:`, `bis:`, `imf:`,
`wb:`, `oecd:`, `philly:`, `boc:`, `boj:`, `bbk:`, `rba:`. Recorrido con gráficos:
[`notebooks/fase2/mercados_series/04_macro_internacional.ipynb`](../../../notebooks/fase2/mercados_series/04_macro_internacional.ipynb).

- **BCE** ([`fuentes.bce`](fuentes/bce.py)): 44 divisas de referencia (vigentes y retiradas; TWD no está), €STR (hábil
  TARGET siguiente a las 08:00 de Berlín), EONIA como predecesor (1999-2021), tipos oficiales DFR y MRO en su fecha de
  efecto, HICP (dataflow con proveedor 4D0, ECOICOP v2), paro y PIB. `includeHistory` solo devuelve la versión vigente:
  las revisables quedan casi todas con la regla previa y nota (posible fuga de revisiones).
- **Eurostat** ([`fuentes.eurostat`](fuentes/eurostat.py)): HICP en `prc_hicp_minr` (ECOICOP v2) y `prc_hicp_manr`
  (v1, retirada en 2025-12; ruptura medida: en EA, 24 de 348 meses distintos, máximo 0,1 puntos), paro, ESI y PIB a
  las 11:00 de Luxemburgo. `une_rt_m` y ESI no traen `EA` de composición cambiante (solo EA20/EA21). Serie
  empalmada `eurostat:HICP_EMP.<geo>` (v1 en los periodos publicados con v1, v2 después; nota y tramo por fila) y
  alias por defecto de la zona euro cambiante: `HICP_EA`, `PIB_EA` (`namq_10_gdp` `EA`) y `PARO_EA` (BCE `LFSI` `U2`).
  Paro y PIB con **tramos por época** en `lags.yaml` (`tramos`, con muestras de notas de prensa y test de no fuga).
- **BIS** ([`fuentes.bis`](fuentes/bis.py)): tipos oficiales, brecha de crédito, servicio de la deuda y tipos de cambio
  efectivos; lag fijo conservador. **Banco Mundial** ([`fuentes.worldbank`](fuentes/worldbank.py)): PIB, IPC y paro de
  217 países, anual, fin de t+1. **FMI WEO** ([`fuentes.imf`](fuentes/imf.py)): solo la edición 2026-10. Las series
  `imf:WEO.<país>.<concepto>` solo traen años anteriores al de la edición; la estimación del año en curso y las
  proyecciones a 1-5 años van a `...PROY.H<h>` (fecha de referencia = año de la edición, nota «proyección, no dato»).
  Las ediciones 2008-2025 están en DBnomics, que prohíbe el acceso por `robots.txt`.
- **OCDE** ([`fuentes.oecd`](fuentes/oecd.py)): `DF_STES_REVISIONS` da ediciones reales (`vintage` = `AAAA-MM`, fecha
  conservadora a fin de mes): CLI desde 2001, IPC y paro desde 1999. La OCDE ya no publica el CLI de la zona euro
  (EA19/EA20: 404 en `DF_STES_REVISIONS` y `DF_CLI`); `oecd:LI_AA.G4E` son los cuatro grandes europeos (DE, FR, IT,
  UK), con nota. Sin paro de China.
- **Philly RTDSM** ([`fuentes.philly_rtdsm`](fuentes/philly_rtdsm.py)): solo los vintages anteriores al primero de
  ALFRED (ROUTPUT 1965-1991, PNB real); PCPI, EMPLOY y RUC no aportan filas. Empalme por vintage con `fred:GDPC1`:
  `deriv:PIB_REAL_USA_VINTAGES` (alias `PIB_REAL_USA`; `base.empalmar_pib_real_usa`, tramo PNB/PIB en la nota).
- **Bancos centrales** ([`fuentes.bancos_centrales`](fuentes/bancos_centrales.py)): BoC (Valet, desde 2001), BoJ (desde
  1998), Bundesbank (Svensson a 10 años, desde 1997) y RBA (desde 2011). BoE (IADB) vetado por `robots.txt`; sus curvas
  salen de los ficheros publicados (bloque 6).

## Universo (bloque 5)

[`config/universo.yaml`](config/universo.yaml) y [`universo`](universo.py): 202 instrumentos (137 ETF, 8 ETP, 22
índices, 6 de volatilidad, 8 divisas, 18 futuros, 2 cripto y VBMFX), por capa (116 tácticos, 54 de contexto, 27
estratégicos y 5 baselines); 52 invertibles por un minorista de la UE con su UCITS equivalente mapeado y empalme con el
proxy de EE. UU. (mediana de 4,5 años antes de cada UCITS). ISIN comprobados con el dígito de control; fechas de
lanzamiento de justETF; divisas medidas en Yahoo (`GBp`, `GBP`, `CHF`, `USX`). Constituyentes
([`fuentes.constituyentes`](fuentes/constituyentes.py)): S&P 500 point-in-time 1996-2026 (fecha efectiva del cambio) y
composición actual de 10 índices (Wikipedia; sin Nikkei 225 ni Ibovespa). El universo ampliado (~1.700 tickers) se
define con los constituyentes. Futuros: solo contexto. Fondos: fuera de la optimización. Recorrido:
[`05_universo.ipynb`](../../../notebooks/fase2/mercados_series/05_universo.ipynb).

## Precios (bloque 6)

**Contrato de precios.** [`proveedores.yahoo`](proveedores/yahoo.py) baja con `auto_adjust=False`, `actions=True` y
`period='max'` (verificado contra `firstTradeDate`), a 0,5 tickers/s como máximo. Yahoo devuelve OHLC, volumen y
dividendos ajustados por los splits posteriores: el adaptador los deshace (`timegate.des_ajustar_yahoo`), descarta la
vela incompleta del día y nunca guarda `Adj Close`. Cada ticker va a `<corpus>/mercados_series/precios/yahoo/<ticker>.parquet`
con `fecha_publicacion` = cierre local + 30 min (Cboe, CME, FX de Yahoo y NAV de fondos, en D+1). `get_precios`
reconstruye el ajuste a fecha (`ajuste='ninguno' | 'splits' | 'total'`) solo con los eventos visibles y convierte a USD
con el FX del BCE visible (subunidades `GBp`, `ZAc`, `ILA`, `USX` normalizadas). Comprobado contra el `Adj Close` de
Yahoo de hoy: error relativo mediano < 1e-4 desde 2000. Calidad: [`calidad`](calidad.py) (saltos sin split, volumen
cero, cierres repetidos, dividendos altos).

- **Núcleo**: 92 de 92 tickers (ETF de la propuesta, AGG, RSP, VBMFX, IBIT, ETHA, 16 índices, 4 divisas, 5 futuros y
  20 acciones grandes): 657.499 sesiones, unos 6 minutos (`python -m ingesta.mercados_series.proveedores.yahoo nucleo`,
  reanudable con registro en `logs/yahoo_nucleo.jsonl`).
- **Nivel ampliado** (1.842 tickers: S&P 500 histórico y constituyentes actuales de 10 índices): descarga reanudable
  en curso (`... proveedores.yahoo ampliado --fichero <csv con columna ticker>`). `... proveedores.yahoo posproceso`
  lee solo lo ya guardado y escribe la cobertura por índice y mercado, los fallidos (vacío, deslistado) y las
  incidencias de `calidad` en `tablas/yahoo_ampliado_cobertura.csv` y en la ficha `yahoo` («descarga en curso: N de
  M» mientras no se haya intentado toda la lista); se repite al terminar.
- **Cripto** ([`fuentes.cripto`](fuentes/cripto.py)): Coinbase BTC-USD y ETH-USD (vela UTC, visible en D+1) como fuente
  principal y Binance Vision (BTCUSDT, ETHUSDT) como control (`tablas/cripto_comparacion_binance_coinbase.csv`).
- **Curvas** ([`fuentes.curvas`](fuentes/curvas.py)): US Treasury (par, desde 1990), BCE (Svensson, desde 2004), BoE
  (gilt nominal, desde 1979), MoF Japón (JGB, desde 1974) y BoC (desde 2001), en la base de series con prefijos `ust:`,
  `ecb:`, `boe:`, `mof:` y `boc:`. Hora de publicación de cada una en su ficha; la del Tesoro (18:00 NY) sin verificar.

Recorrido: [`06_precios_cobertura.ipynb`](../../../notebooks/fase2/mercados_series/06_precios_cobertura.ipynb).

## Supervivencia (bloque 7)

[`supervivencia`](supervivencia.py): el sesgo se **declara y se mide**. Sobre una muestra estratificada de tramos del
S&P 500 (62 salidos y 60 actuales), Yahoo cubre completos 59 de 60 actuales y solo 14 de 62 salidos (40 sin datos); la
fracción media de sesiones cubiertas de los salidos es 0,23 y es nula para las salidas anteriores a 2006
(`tablas/supervivencia_cobertura.csv`, ficha `supervivencia`). Corrección completa: CRSP vía WRDS (pendiente de la
licencia de MIAX); alternativa: Tiingo (exige cuenta propia; sin probar). Comandos: `python -m
ingesta.mercados_series.supervivencia descargar` (reanudable) y `... medir`. Es una medición: no añade series,
publicadores ni mercados, y la rentabilidad diaria de la muestra vive en `<corpus>/mercados_series/supervivencia/` (no
entra en la base ni en `get_serie`/`get_precios`). Recorrido:
[`07_supervivencia.ipynb`](../../../notebooks/fase2/mercados_series/07_supervivencia.ipynb).

## Cartera: libre de riesgo, baselines y costes (bloque 8)

[`cartera`](cartera/): todo punto en el tiempo (cada valor usa solo lo visible en su fecha).

- **Libre de riesgo** ([`cartera.libre_riesgo`](cartera/libre_riesgo.py)): DTB3 para los agentes; French RF para
  evaluar ex post; en EUR, EONIA hasta 2019 y €STR después (`fred:DTB3`, `ff:F3D.RF`, `ecb:EONIA`, `ecb:ESTR`; sin
  series, publicadores ni mercados propios; Euribor 3M no está catalogado).
- **Baselines** ([`cartera.baselines`](cartera/baselines.py)): buy & hold SPY, 60/40 SPY/AGG (VBMFX antes de 2003-09),
  1/N sobre los ETF invertibles vivos en cada fecha, RSP, momentum sectorial 12-1 top 3 (SPDR) y 60/40 europeo (anexo:
  CSPX.L/EUNA.DE desde 2017-11). Se ejecuta al cierre de la fecha de control y la señal solo ve `sesion < fecha`
  (`baselines.vista_senal`; test `test_t1_a_baselines_senal`).
- **Costes** ([`cartera.costes`](cartera/costes.py)): DEGIRO con 10.000 € (IBKR como sensibilidad); coste = comisión +
  0,25 % de FX + spread base por clase × multiplicador de crisis (CHL normalizado, entre 1 y 10)
  (`tablas/costes_modelo.csv`). Comisiones de la tarifa pública sin verificar.

Informe: `python -m ingesta.mercados_series.cartera.informe`; ficha `cartera_baselines_costes`. Los precios
se leen con `almacen.leer_precios` (validados por contrato: los 92 tickers del núcleo, ARCX y NAV_US incluidos). Recorrido:
[`08_libre_riesgo_baselines_costes.ipynb`](../../../notebooks/fase2/mercados_series/08_libre_riesgo_baselines_costes.ipynb).

## Fuentes y estado (cierre, bloque 9)

Estado de cada ficha de `results/ingesta/mercados_series/fichas/` (2026-10-13): **ok** si se ha descargado y validado
entera, **parcial** si falta una parte documentada en la ficha y **bloqueada** si no se puede descargar.

| ficha | estado | cobertura medida | nota |
|---|---|---|---|
| `fred_alfred`, `alfred_vintages` | ok | 1854 – 2026-10 | vintages exactos de las 161 revisables; 80 series vivas |
| `ice_bofa_oas` | parcial | 2023-10-10 – 2026-10 | ventana móvil de 3 años en FRED |
| `fase1_raw` | ok | 1854 – 2026-08 | solo lo que no está en FRED |
| `ken_french`, `shiller`, `goyal_welch`, `ofr_fsi`, `jst_macrohistory` | ok | 1870 – 2026-08 | ediciones archivadas desde 2026; JST solo validación |
| `bce`, `eurostat`, `bis`, `worldbank` | ok | 1945 – 2026-10 | sin vintages salvo la versión vigente |
| `oecd` | parcial | 1949 – 2026-09 | sin CLI de la zona euro ni paro de China |
| `imf_weo` | parcial | 1980 – 2026 | solo la edición 2026-10 (DBnomics vetado por `robots.txt`) |
| `philly_rtdsm` | ok | 1939 – 2026-08 | solo aporta ROUTPUT anterior a ALFRED |
| `boc`, `boj`, `bundesbank`, `rba` | ok | 1997 – 2026-10 | RBA desde 2011; BoC desde 2001 |
| `boe` (IADB) | bloqueada | – | `robots.txt` prohíbe la ruta CSV |
| `ust_curva`, `bce_curvas`, `boe_curvas`, `mof_jgb`, `boc_curva` | ok | 1974 – 2026-10 | curvas oficiales |
| `yahoo` | ok | 1927 – 2026-10 | núcleo de 92 tickers; ampliado en descarga (731 de 1.842 intentados, 599 con datos) |
| `coinbase`, `binance_vision` | ok | 2015 – 2026-10 | Binance de control |
| `universo` | ok | – | 202 instrumentos |
| `sp500_constituyentes` | ok | 1996 – 2026-08 | fecha efectiva del cambio |
| `indices_constituyentes` | parcial | foto de 2026-10-13 | sin Nikkei 225 ni Ibovespa |
| `supervivencia` | ok | 1996 – 2026-08 | medición; corrección WRDS pendiente |
| `cartera_baselines_costes` | ok | 1993 – 2026-10 | comisiones sin verificar; 1/N con lista estática; 60/40 UCITS desde 2017-11 |

Calidad global, cobertura por fuente, método y época y las dos tools en 2008-09-15 y 2020-03-16 (solo metadatos):
[`09_cierre_calidad.ipynb`](../../../notebooks/fase2/mercados_series/09_cierre_calidad.ipynb). Truncado real
(5 series, 5 tickers, 3 fechas): `tests/ingesta/mercados_series/test_t1_cierre.py`.

## Deudas pendientes

Solo lo que queda (2026-10-13, tras la auditoría de cierre). Las de *timegate* llevan severidad: **media** = puede
adelantar una fecha de decisión en una época concreta; **baja** = solo afecta a `disponible_utc` o a una regla ya
conservadora.

Auditoría de cierre (sin fallos): 300 `get_serie` al azar (22 prefijos de fuente, 1990-2026, una de cada cinco en
USD) y 100 `get_precios` (núcleo y ampliado ya descargado) sin `fecha_publicacion` posterior a la fecha ni
`disponible_utc` posterior a las 16:30 NY; truncado real en 15 series × 4 fechas y 10 tickers × 4 fechas × 2 ajustes;
coherencia `fecha_publicacion` ≥ primera decisión de `disponible_utc` en las 9,86 M filas con hora; splits futuros
(AAPL 2020-08-28, NVDA 2024-06-07) sin efecto en el ajuste; cripto visible en D+1; HICP empalmado, WEO (dato y
proyecciones) y DST; baselines con precios truncados idénticos. Corregido: la JGB del viernes anterior a un festivo
japonés (2026-10-09) se veía en el festivo (15 filas; test de regresión) y el resumen de `construir-base` fallaba con
el empalme del PIB de EE. UU. después de escribir la base (test de regresión).

Precios y universo:
- **Nivel ampliado de precios**: descarga en curso (último posproceso: 731 de 1.842 intentados, 599 con datos; los
  fallidos son tickers deslistados del S&P 500). Al terminar: `proveedores.yahoo posproceso`, reejecutar el notebook
  06 y borrar `crudo/yahoo/mercados_fusionados.yaml` (lo escribe aún el proceso en curso, con el código antiguo).
- **Supervivencia**: WRDS/CRSP pendiente de la licencia de MIAX (Tiingo, alternativa sin probar). Al terminar el
  ampliado, contar los tramos salidos con parquet y, si mejora la muestra (hoy 122 tramos), leer esos parquet
  as-traded en `supervivencia.py` en lugar de pedir otra vez a yfinance.
- S&P 500: 28 tramos con el ticker posterior de quiebra (sufijo Q/QQ); el mapeo exige CIK/PERMNO (WRDS).
- VSTOXX (`^V2TX`) no está en Yahoo; constituyentes de Nikkei 225 e Ibovespa sin fuente. ISIN sin verificar: NGAS.L
  y CPER, BITO, VIXY y VBMFX (señal, no se invierten). AIGA.L: Yahoo solo da una vela (otra línea o inactivo).
- Baselines: el 60/40 europeo arranca en 2017-11 (EUNA.DE); alargarlo exige un agregado UCITS más antiguo o el
  empalme con BNDX.

Horas y calendarios (se miden hoy y se aplican hacia atrás; solo adelantan una fecha si en el pasado se publicaba
después de las 16:30 NY o en otro día):
- Verificadas con ediciones archivadas de 2-3 años: BLS (Employment Situation 2010 y 2017, IPC 2017: 8:30), G.17
  (1998: 9:15) y H.8 (2001 y 2008: 16:15). Ver las notas de `publicaciones.yaml`.
- **H.15 diario antes de 2005-06-28 (media)**: el archivo de la Fed solo guarda la edición semanal; si la web no
  publicaba el dato diario, DGS*, DTB*, DFF y demás H.15 diarios se ven hasta una semana antes (son cotizaciones de
  mercado públicas el mismo día, pero no la publicación oficial). Contrastar con un archivo de la página
  `h15/update` o fechar con la edición semanal antes de esa fecha.
- **Ken French antes de las ediciones archivadas (media)**: la regla «+45 días» supone ediciones mensuales desde 1926;
  la biblioteca de datos es de finales de los noventa y se actualizaba con menos frecuencia. Ídem, reglas de
  Goyal-Welch, NBER y OFR sin verificar.
- H.4.1 (baja): el archivo de 1999 dice «the exact time of each release will be announced»; con el margen de 5 min
  ya es decisión del día siguiente. Tesoro 18:00 (baja): ya es día siguiente; sin fuente de la hora histórica.
- BEA, Census y Fed regionales (Chicago, Philadelphia, Kansas City, Dallas, Atlanta, Cleveland) (baja): hora de hoy
  (mañana o mediodía) sin archivo; la fecha la da ALFRED y solo `disponible_utc` puede adelantar unas horas (p. ej.
  la encuesta de Philadelphia salía antes al mediodía).
- **Horizontes del SEP**: solo se ven desde su fecha de referencia; falta decidir si van a una tabla de proyecciones.
- EONIA se fecha con el publicador de €STR (hábil siguiente a las 08:00): conservador.

Macro internacional:
- FMI WEO: ediciones 2008-2025 (DBnomics en `Disallow: /`, comprobado el 2026-10-13: pedir permiso o localizar el
  archivo del FMI) y WEO de agregados de la zona euro; fechas de edición contrastadas solo las de 2026.
- BCE, Eurostat, BIS y Banco Mundial sin historia de revisiones (valores de hoy con regla y nota de revisable).
- Historia larga de RBA (antes de 2011) y BoC (antes de 2001); BoE IADB vetado por `robots.txt`.
- Eurostat sin verificar (conservadoras): paro antes de 2012 (fin de M+2), PIB antes de 2003 (t+71) y PIB por país
  (`EUROSTAT_PIB_PAISES`; no aprovecha los t+30 nacionales). `CREACION_GEO` de EA20 y EA21 sin el registro de cambios
  de Eurostat. OCDE: no hay CLI de la zona euro (G4E es un sustituto con nota). Sin filas en la base:
  `oecd:LI_AA.EA20`, `oecd:UNEMP.CHN`, `philly:EMPLOY`, `philly:PCPI` y `philly:RUC`.

Mantenimiento y pruebas:
- **Archivo periódico de las OAS de ICE** (ventana móvil de 3 años en FRED) y de las ediciones académicas: a mano.
- **package-data** de los YAML de config en `pyproject.toml` (PR común; hoy basta `pip install -e`).
- Comisiones de DEGIRO e IBKR sin verificar; 1/N con la lista estática del núcleo.
- Sin prueba `live` propia: `bce_curvas` (histórico entero, lento), `boe_curvas`, `jst_macrohistory`,
  `indices_constituyentes`; OCDE y Philadelphia solo comprueban `robots.txt` y la página (`test_t1_c_live_fuentes.py`).

## Decididas

- Divisa base: USD (se guarda la original y se convierte al leer).
- Hora de decisión: 16:30 America/New_York de cada día natural (tras el cierre de Nueva York).
- Supervivencia: se declara y se mide (bloque 7); la corrección con WRDS/CRSP depende de la licencia de MIAX.
- Cripto: Coinbase (USD) principal y Binance Vision de control. Futuros, solo contexto. Fondos, fuera de la optimización.
- Yahoo: el mercado declarado en `universo.yaml` manda sobre `exchangeName`; el respaldo por huso solo avisa para los
  tickers fuera del universo.

## Preguntas abiertas

- ¿Fuente de precios para la versión final (Yahoo, Tiingo o WRDS/CRSP si llega la licencia)?
- ¿Proyecciones (SEP, WEO) en una tabla aparte o fuera del tema?
