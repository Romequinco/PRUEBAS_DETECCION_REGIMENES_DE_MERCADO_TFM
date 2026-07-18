# 01 — Datos y EDA (FASE 1)

> Memoria de la fase de datos. Qué se descargó, decisiones de tratamiento (sin
> imputar), ventana común, hallazgos del EDA y los `DRAWDOWN_TROUGHS` calculados.
> Notebook ejecutado: `notebooks/00_eda.ipynb`. Figuras: `results/eda_*.png`.

## 1. Qué se descargó (set ampliado) y procedencia real

Descarga sin API key. **FRED (`fredgraph.csv`) resultó inaccesible en este
entorno** (timeouts/connection reset sistemáticos; yfinance funcionó sin
problemas). Se aplicaron fallbacks documentados — nada inventado ni omitido en
silencio (`data/raw/provenance.json`):

| Serie interna | Fuente final | Inicio efectivo | Nota |
|---|---|---|---|
| SP500 | yfinance `^GSPC` | 1985-01-02 | índice precio |
| VIX | yfinance `^VIX` | 1990-01-02 | nivel (miedo equity) |
| MOVE | yfinance `^MOVE` | 2002-11-12 | **sí disponible** en yfinance; no hizo falta alternativa |
| TLT | yfinance `TLT` | 2002-07-30 | Treasuries largos |
| IEF | yfinance `IEF` | 2002-07-30 | Treasuries medios |
| HYG | yfinance `HYG` | 2007-04-11 | high yield (**gobierna la ventana común**) |
| GOLD | yfinance `GLD` | 2004-11-18 | oro |
| DXY | yfinance `DX-Y.NYB` | 1985-01-02 | DXY ICE clásico (alternativa permitida a FRED DTWEXBGS) |
| YIELD_10Y_3M | **proxy** `^TNX − ^IRX` (yfinance) | 1985-01-02 | FRED `T10Y3M` no respondió; proxy real = yield 10Y − T-bill 3M |
| ~~HY_OAS~~ | **OMITIDA** | — | FRED `BAMLH0A0HYM2` devolvió respuesta **truncada** (solo 2023+); crédito cubierto por HYG y spread HYG−IEF |

### Decisiones de fallback (justificación)
- **`^MOVE`**: contra lo previsto, yfinance SÍ lo sirve (desde 2002-11). Se usa
  directamente.
- **DXY**: se usa `DX-Y.NYB` (yfinance) como primaria porque FRED es inaccesible
  aquí y es la alternativa estándar contemplada en la consigna.
- **Curva 10Y-3M**: se intenta FRED `T10Y3M`; al fallar, se construye el proxy
  real `^TNX − ^IRX` (ambos desde 1985). Difiere marginalmente de la serie
  oficial (constant-maturity 3M vs discount yield del T-bill 13s) pero la sigue
  muy de cerca. El loader registra qué fuente usó.
- **HY OAS**: era opcional/ampliable. La única respuesta de FRED llegó truncada
  (n=785, solo 2023+) y habría corrompido la ventana común. Se excluye y se añade
  un **guardia anti-truncamiento** en `data_loader._fred_csv` (rechaza respuestas
  con < 500 filas).

> Si en otro entorno hay acceso a FRED, `data_loader` usará automáticamente
> `T10Y3M` y puede reincorporarse `BAMLH0A0HYM2` (y opcionalmente DTWEXBGS).

## 2. Política de tratamiento: SIN imputar
- Cada serie arranca en su fecha real; los huecos previos quedan NaN.
- NaN dentro del rango de cada serie (festivos desalineados entre Yahoo/series de
  tasas) **no se rellenan**. Los detectores trabajan sobre la ventana común con
  `dropna(how='any')`.
- No se usa ningún estadístico de muestra completa (ver causalidad, §5).

## 3. Ventana común
- **Intersección de las 9 series: 2007-04-11 → 2026-06-12 (4 737 obs sin NaN)**,
  gobernada por el inicio de HYG. **Consistente con la tarea previa** (que también
  arrancaba en 2007-04-11 por HYG).
- Tras construir features causales (z-scores con `min_periods=60`), la ventana
  efectiva de features es **2007-07-06 → 2026-06-12 (4 665 obs, 15 features)**.
- `data/raw/coverage_report.csv` tiene el detalle por serie.

### Tensión de cobertura para walk-forward (a tener en cuenta en FASE 3)
La GFC 2008 está muy cerca del inicio de datos (2007-04). Un walk-forward con
ventana de entrenamiento de varios años dejaría 2008–2011 **dentro del train**, no
evaluable out-of-sample. Mitigaciones posibles (a decidir en FASE 3): detectores
que usen un subconjunto de features con histórico más largo (SP500+VIX desde 1990)
para evaluar 2008 OOS, o ventanas de entrenamiento más cortas. Declarado, no
resuelto aquí.

## 4. Hallazgos del EDA

### Fat tails (leptocurtosis) — confirma la crítica al supuesto gaussiano
Kurtosis de exceso (Fisher; 0 = normal) de retornos diarios; Jarque-Bera rechaza
normalidad en todas con p≈0:

| Serie | Skew | Kurtosis exceso |
|---|---:|---:|
| SP500 | −1.16 | **25.6** |
| HYG | +0.31 | **39.6** |
| GOLD | −0.46 | 6.8 |
| TLT | −0.02 | 3.4 |
| IEF | +0.05 | 2.5 |
| DXY | −0.10 | 2.3 |

→ Colas muy gordas en equity y crédito. Justifica explorar emisiones **t-Student**
o mixturas frente al HMM gaussiano (figura `results/eda_fat_tails.png`).

### Correlaciones (incondicionales) con el S&P 500
- TLT −0.31, IEF −0.30 → bonos largos/medios diversifican (de media).
- HYG **+0.67** → el high yield co-mueve con equity (activo de riesgo, buen
  sensor de crédito).
- GOLD +0.07, DXY −0.03 → oro y dólar casi incorrelados con equity (refugio).
- La correlación rolling S&P 500/TLT **cambia de signo** a lo largo del tiempo
  (Gulko 2002) → feature `corr_spx_bond` (figura `results/eda_rolling_corr.png`).

### Vol anualizada (muestra completa): SP500 18.2%, GOLD 18.2%, TLT 14.3%,
HYG 10.9%, DXY 8.1%, IEF 6.8%.

## 5. Features causales y verificación de no look-ahead
15 features en `src/features.py`, todas causales por construcción
(`expanding`/`rolling`, nunca estadísticos de muestra completa):
`SP500_ret_z, SP500_vol_z, VIX_level_z, VIX_change_z, MOVE_level_z, TLT_ret_z,
IEF_ret_z, HYG_ret_z, credit_spread_z, yield_slope_z, DXY_change_z, GOLD_ret_z,
corr_spx_bond, SP500_drawdown, SP500_momentum`.

**Test de causalidad** (`features.assert_causal`): se truncan los datos en
2015-01-01 y se recomputan; los valores hasta esa fecha deben coincidir con los
de la muestra completa. Resultado: **`max_abs_diff = 0.0` en las 15 features**
(el futuro no altera el pasado). Salida en `data/processed/features.parquet`.

## 6. DRAWDOWN_TROUGHS calculados (desde el S&P 500 real)
Mínimo del drawdown (precio / máx_expanding − 1) por episodio. Cableados en
`evaluation.DRAWDOWN_TROUGHS`:

| Episodio | Trough | Max drawdown | En ventana común |
|---|---|---:|---|
| GFC_2008 | 2009-03-09 | −56.8% | sí |
| EuroDebt_2011 | 2011-10-03 | −29.8% | sí |
| COVID_2020 | 2020-03-23 | −33.9% | sí |
| Inflation_2022 | 2022-10-12 | −25.4% | sí |
| (DotCom_2002) | 2002-10-09 | −49.1% | **no** (pre-2007; solo detectores SP500+VIX) |

## 7. Verificación de ventanas de crisis / falsos positivos
Todas las ventanas de `evaluation.py` caen dentro de la ventana común
(≥ 2007-04-11): crisis 2008/2011/2020/2022 y trampas 2013/2018. Confirmado en el
notebook (figura `results/eda_sp500_drawdown.png`).

## 8. Artefactos generados
- `data/raw/raw_panel.parquet`, `yfinance_raw.parquet`, `coverage_report.csv`,
  `provenance.json` (todos gitignored salvo el .md de memoria).
- `data/processed/features.parquet` (15 features causales).
- `notebooks/00_eda.ipynb` ejecutado (0 errores, 4 figuras inline).
- `results/eda_fat_tails.png`, `eda_corr.png`, `eda_sp500_drawdown.png`,
  `eda_rolling_corr.png`.
- `notebooks/_build_eda.py` (reconstruye y reejecuta el notebook).
