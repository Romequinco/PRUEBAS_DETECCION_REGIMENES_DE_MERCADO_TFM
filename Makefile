# Makefile del TFM (paquete ``regimenes``).
#
# Compatible con GNU Make en Linux/macOS y en Windows bajo Git Bash
# (instalar make, p. ej. ``winget install ezwinports.make`` o ``choco install make``).
# Todas las recetas llaman a ``$(PY) -m ...``: no dependen de rm/find/sed.
#
# Uso rapido:
#   make help                       lista los objetivos
#   make install                    entorno de desarrollo completo
#   make datos                      verifica data/raw sin red (DESCARGAR=1 descarga lo que falte)
#   make features                   regenera data/processed ejecutando notebooks/03_preprocesado
#   make benchmark JOBS=4           benchmark completo pistas A y B (cache verificada)
#   make notebooks                  ejecuta los notebooks 00-14 en orden
#   make test / make test-rapido    tests completos / sin datos ni resultados locales
#   make limpiar                    borra caches de Python/pytest/ruff (nunca datos ni resultados)
#
# Variables sobreescribibles: PY, JOBS, DETECTOR, DESCARGAR, NB_TIMEOUT, PYTEST_ARGS.
# En Linux, si ``python`` no existe: ``make test PY=python3``.

PY          ?= python
JOBS        ?= 1
DETECTOR    ?=
DESCARGAR   ?= 0
NB_TIMEOUT  ?= 3600
PYTEST_ARGS ?=

NB_DIR      := notebooks
# 00..14: descarga, EDA, preprocesado, protocolo, familias F1-F7, comparativa y fusiones.
# 15..16 (generadores y validacion de sinteticos) estan implementados pero NO entran aqui: necesitan
# data/sinteticos (~1 GB, gitignored), que genera el 15 con EJECUTAR=True; una vez generado, se
# ejecutan a mano con EJECUTAR=False. 17..20 (laboratorio, aumento, decision final, pseudo-live)
# son esqueletos.
NOTEBOOKS   := $(sort $(wildcard $(NB_DIR)/0[0-9]_*.ipynb $(NB_DIR)/1[0-4]_*.ipynb))
NB_FEATURES := $(NB_DIR)/03_preprocesado.ipynb
NBCONVERT   := $(PY) -m jupyter nbconvert --to notebook --execute --inplace \
               --ExecutePreprocessor.timeout=$(NB_TIMEOUT)

ifeq ($(DESCARGAR),1)
DATOS_FLAGS :=
else
DATOS_FLAGS := --offline
endif

ifneq ($(strip $(DETECTOR)),)
DETECTOR_FLAGS := --detector $(DETECTOR)
else
DETECTOR_FLAGS :=
endif

.DEFAULT_GOAL := help
.PHONY: help install datos features benchmark notebooks test test-rapido lint limpiar

help: ## Muestra esta ayuda
	@$(PY) -c "import re; [print(f'  {m[0]:<13} {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open('Makefile', encoding='utf-8').read(), re.M)]"

install: ## Instala el paquete en editable con extras [deep,jump,dev] (torch, jumpmodels, pytest, jupyter, ruff)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[deep,jump,dev]"

datos: ## Capa de datos: --offline por defecto (verifica data/raw); DESCARGAR=1 descarga lo que falte
	$(PY) -m regimenes.datos $(DATOS_FLAGS)

features: ## Regenera data/processed (paneles pista A/B) ejecutando notebooks/03_preprocesado
	$(NBCONVERT) $(NB_FEATURES)

benchmark: ## Benchmark walk-forward pistas A y B con JOBS procesos (DETECTOR="D04 D08" para un subconjunto)
	$(PY) -m regimenes.benchmark --track A B --jobs $(JOBS) $(DETECTOR_FLAGS)

notebooks: ## Ejecuta en orden los notebooks 00-14 con nbconvert (in-place, NB_TIMEOUT s por celda)
	@$(PY) -c "import sys; nbs=sys.argv[1:]; print(f'{len(nbs)} notebooks:', *nbs, sep='\n  ')" $(NOTEBOOKS)
	@for nb in $(NOTEBOOKS); do \
		echo ">> $$nb"; \
		$(NBCONVERT) "$$nb" || exit 1; \
	done

test: ## Todos los tests (los que necesitan data/ o paneles se saltan con motivo si faltan)
	$(PY) -m pytest -q $(PYTEST_ARGS)

test-rapido: ## Tests sin datos locales ni resultados de benchmark (marcadores 'datos' y 'resultados' fuera)
	$(PY) -m pytest -q -m "not datos and not resultados" $(PYTEST_ARGS)

lint: ## Lint minimo con ruff (errores reales; no reformatea)
	$(PY) -m ruff check src tests

limpiar: ## Borra __pycache__, .pytest_cache, .ruff_cache, .ipynb_checkpoints y build/ (NO toca data/ ni results/)
	$(PY) -c "import pathlib, shutil; r = pathlib.Path('.'); \
	dirs = [*r.rglob('__pycache__'), *r.glob('.pytest_cache'), *r.glob('.ruff_cache'), *r.rglob('.ipynb_checkpoints'), *r.glob('build')]; \
	[shutil.rmtree(d, ignore_errors=True) for d in dirs]; \
	print(f'eliminados {len(dirs)} directorios de cache')"
