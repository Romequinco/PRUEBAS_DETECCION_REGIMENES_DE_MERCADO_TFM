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
- [ ] Reutilizar las 166 series de la Fase 1 (`data/raw`, `configs/catalog.yaml`, `regimenes.features.lags`) como **niveles** legibles
- [ ] FRED / ALFRED (vintages en tiempo real) y calendario de publicaciones
- [ ] Macro internacional: OCDE, BCE, Eurostat, FMI, Banco Mundial, BIS, DBnomics, bancos centrales
- [ ] Tipo libre de riesgo, baselines (buy & hold, 60/40, momentum) y costes de transacción

## Entregables

1. Un extractor por fuente con su ficha y sus tests (ver «cuándo está hecha una fuente» en `docs/fase2`).
2. **Hecho (bloque 1)**: contrato de `get_precios` y `get_serie`: qué devuelven, con qué fecha de publicación, cómo se
   ajusta el precio a fecha (ver «Contrato (bloque 1)»).
3. Limpieza y EDA de las series (huecos, rupturas, series retiradas).

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
histórico de hoy y vintage `unico`, que pondría el valor revisado con la fecha de la primera publicación
(pendiente: comprobarlo al escribir cuando haya extractores).

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

## Decididas

- Divisa base: USD (se guarda la original y se convierte al leer).
- Hora de decisión: 16:30 America/New_York de cada día natural (tras el cierre de Nueva York).

## Preguntas abiertas

- ¿Fuente de precios para la versión final (gratuita o de pago)?
- ¿Cómo se corrige o se declara el sesgo de supervivencia?
