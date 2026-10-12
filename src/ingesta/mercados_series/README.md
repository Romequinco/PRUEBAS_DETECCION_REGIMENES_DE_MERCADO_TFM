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
2. Contrato de `get_precios` y `get_serie`: qué devuelven, con qué fecha de publicación, cómo se ajusta el precio a fecha.
3. Limpieza y EDA de las series (huecos, rupturas, series retiradas).

## Preguntas abiertas

- ¿Divisa base del cliente (EUR / USD)? ¿Hora de decisión (tras el cierre o antes de la apertura)?
- ¿Fuente de precios para la versión final (gratuita o de pago)?
- ¿Cómo se corrige o se declara el sesgo de supervivencia?
