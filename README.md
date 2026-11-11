# regime-aware-macro-equity — TFM (MIAX)

*Multi-Agent RAG System for Regime-Aware Macro-Equity Intelligence.* Un **sistema multi-agente con RAG** (agentes
Macro, Equity, Risk, Portfolio y CIO en LangGraph) que lee información financiera con fecha de publicación, sabe en
qué **régimen** está el mercado y con ello construye carteras explicadas en un *investment memo* con citas. Visión,
fases y estado: [`docs/tfm/README.md`](docs/tfm/README.md); decisión de alcance:
[ADR-005](docs/decisions/ADR-005-reencuadre-tfm-multiagente.md); estructura del repositorio:
[ADR-012](docs/decisions/ADR-012-estructura-repo.md).

**Si acabas de llegar:** el índice de la documentación y el orden de lectura según quién lee están en
[`docs/README.md`](docs/README.md#orden-de-lectura).

## Estado de las fases

Detalle y siguiente paso de cada una en [`docs/tfm/README.md`](docs/tfm/README.md#3-plan-de-trabajo-cinco-fases-y-su-estado).

1. **Régimen de mercado** (`src/regimenes`) — en curso: datos, benchmark de 12 detectores y fusiones hechos; faltan la decisión final (notebook 21) y el pseudolive (22).
2. **Ingesta, RAG y tools** (`src/ingesta`, `src/herramientas`) — en marcha: sondas de los cuatro temas en `main` y contrato de tools creado.
3. **Agentes Macro, Equity y Risk** (`src/agentes`) — pendiente.
4. **Cartera y memo** (`src/cartera`, Portfolio y CIO) — pendiente.
5. **Evaluación** (backtest con TimeGate frente a *baselines*) — pendiente.

## Fase 1 en breve

La Fase 1 no busca un detector concreto, sino un **marco de evaluación causal y comparable** (walk-forward, lags
de publicación, mismas métricas para todos) sobre dos pistas de datos congeladas. Su salida es la tool
**`get_regimen(fecha)`**: estado {calma, crisis}, probabilidad y días en el estado, con solo información publicada
hasta `fecha`. Detector provisional ([`configs/regimen.yaml`](configs/regimen.yaml)): **D06 (GJR-GARCH-t) en la
pista A y D05 (Markov-Switching) en la B**, los primeros del MCS de cada pista; añade un contexto para los agentes
(consenso de la fusión online y acuerdo del resto de detectores). Los agentes la consumen a través de
[`src/herramientas`](src/herramientas/README.md), nunca de `regimenes` directamente.

- Pipeline de notebooks, orden de ejecución, huella de caché y rutas: [`docs/fase1/README.md`](docs/fase1/README.md).
- Mapa de módulos del paquete: [`src/regimenes/README.md`](src/regimenes/README.md).
- Ranking y resultados: [`results/benchmark/ranking_v2.csv`](results/benchmark/ranking_v2.csv), notebook
  [`14_comparativa`](notebooks/14_comparativa.ipynb) y los ADR-008 a ADR-011.
- La **línea de datos sintéticos** (generadores, validación, laboratorio y aumento; notebooks 17–20) está archivada
  en la etiqueta git `sinteticos-final`: el aumento no se adopta y `get_regimen` no la usa
  ([ADR-012](docs/decisions/ADR-012-estructura-repo.md) §2.4).
- «Capa 1» es aquí la primera vuelta de 12 detectores v1 (etiqueta `capa1-final`), no la capa de 42 ETFs de la
  propuesta ([GLOSARIO](docs/GLOSARIO.md#capa-1-del-repositorio-no-es-la-capa-1-de-la-propuesta)).

## Instalación

Esta es la fuente única de instalación del repositorio. Todas las dependencias están en
[`pyproject.toml`](pyproject.toml) (distribución `regime-aware-macro-equity`; paquetes `regimenes`, `ingesta` y
`herramientas`), con un extra por uso, y el candado de versiones en [`constraints.txt`](constraints.txt).

```bash
cp .env.example .env                                          # FRED_API_KEY, SEC_EDGAR_USER_AGENT, INGESTA_CORPUS_DIR
python -m pip install -e ".[todo]" -c constraints.txt         # todo: deep, jump, changepoint, dev e ingesta
```

Variantes, siempre con `-c constraints.txt`:

| Para qué | Extras |
|---|---|
| Desarrollo sin torch ni jumpmodels (D12 y D09 se saltan) | `".[dev]"` |
| Fase 2 completa | `".[dev,ingesta]"` |
| Un solo tema de la Fase 2 | `".[dev,ingesta-mercados]"` · `".[dev,ingesta-bancos]"` · `".[dev,ingesta-noticias]"` (el Tema 3, empresas, no necesita extra) |

**Por qué el candado.** La caché del benchmark guarda una huella que incluye las versiones de numpy, pandas, scipy,
scikit-learn, hmmlearn, arch, torch y jumpmodels. Si `pip` actualiza cualquiera de ellas, toda la caché queda
invalidada y hay que re-ejecutar el benchmark (horas). `constraints.txt` fija esas versiones; cambiar una es una
decisión explícita. `requirements.txt` es solo un alias de una línea de la orden anterior.

### Si ya tenías el repo clonado (tras el renombrado)

El repositorio se renombró en GitHub a `regime-aware-macro-equity` y la distribución dejó de llamarse `regimenes`.
Hay que reinstalar una vez:

```bash
git remote set-url origin https://github.com/Romequinco/regime-aware-macro-equity.git
git fetch
git switch main && git pull
python -m pip uninstall -y regimenes                          # quita la distribución antigua
python -m pip install -e ".[todo]" -c constraints.txt         # o los extras de tu tema
# si existe, borra la carpeta src/regimenes.egg-info (metadatos de la distribución antigua)
python -c "import regimenes, ingesta, herramientas"
python -m pytest -q -m "not datos and not resultados"
```

La carpeta local puede seguir llamándose como antes: el nombre del directorio no afecta a nada.

## Mapa del repo

```
regime-aware-macro-equity/
├── README.md · pyproject.toml · constraints.txt · requirements.txt (alias) · Makefile · LICENSE
├── configs/        Fase 1 (benchmark_spec, bloques, catalog, regimen); lo nuevo irá en configs/<paquete>/
├── data/           fuera de git (raw, processed)
├── src/
│   ├── regimenes/      Fase 1 · régimen → get_regimen (congelado por la huella de caché)
│   ├── ingesta/        Fase 2 · un subpaquete por tema
│   ├── herramientas/   contrato de tools tipado para los agentes
│   ├── memoria/        Fase 2 · RAG con TimeGate (se crea al empezar)
│   ├── agentes/        Fases 3–4 · grafo LangGraph (se crea al empezar)
│   └── cartera/        Fases 4–5 · motores, límites, backtest (se crea al empezar)
├── notebooks/      00–16 y 21–22 de la Fase 1 · fase2/<tema>/NN_*.ipynb
├── results/        salidas pequeñas versionadas; binarios regenerables ignorados
├── tests/          espejo de src/
└── docs/           índice, ADR, estado del arte, una carpeta por fase
```

Las fases 3–5 importan solo de `herramientas`, nunca de `regimenes.*` ni de `ingesta.*`
([ADR-012](docs/decisions/ADR-012-estructura-repo.md) §2.1).

## Cómo reproducir

Objetivos del [`Makefile`](Makefile) (GNU Make; en Windows, bajo Git Bash). `make help` los lista.

| Objetivo | Qué hace | Equivalente directo |
|---|---|---|
| `make install` | entorno completo con el candado | `python -m pip install -e ".[todo]" -c constraints.txt` |
| `make datos` | verifica `data/raw/` sin red (`DESCARGAR=1` descarga lo que falte) | `python -m regimenes.datos --offline` |
| `make features` | regenera `data/processed/` | ejecuta `notebooks/03_preprocesado.ipynb` |
| `make benchmark JOBS=4` | benchmark de 12 detectores × 2 pistas con caché verificada (`DETECTOR="D04 D08"` para un subconjunto) | `python -m regimenes.benchmark --track A B --jobs 4` |
| `make notebooks` | ejecuta los notebooks 00–16 en orden | `python -m jupyter nbconvert --to notebook --execute --inplace ...` |
| `make test` · `make test-rapido` | tests completos · sin datos ni resultados locales | `python -m pytest -q` · `-m "not datos and not resultados"` |
| `make lint` | errores reales con ruff | `python -m ruff check src tests` |
| `make limpiar` | borra cachés de Python, pytest y ruff (nunca datos ni resultados) | — |

Tras un benchmark nuevo, el orden de los notebooks no es el numérico: ver
[`docs/fase1/README.md`](docs/fase1/README.md#orden-de-ejecución).

## Dónde seguir

- [`docs/README.md`](docs/README.md) — índice de la documentación y orden de lectura.
- [`src/regimenes/README.md`](src/regimenes/README.md) — mapa del código de la Fase 1, con lo que entra en cada huella.
- [`src/herramientas/README.md`](src/herramientas/README.md) — contrato de tools para los agentes.
- [`docs/fase2/README.md`](docs/fase2/README.md) — reparto de la Fase 2 en cuatro temas y contrato común.
- [`docs/GLOSARIO.md`](docs/GLOSARIO.md) — definiciones de los términos del proyecto.
