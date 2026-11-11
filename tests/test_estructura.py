"""Guardianes de la estructura del repositorio (ADR-012).

Corren en la CI sin datos. Fallan si alguien:

1. Mueve o renombra un fichero que entra en una huella de caché. La huella usa la ruta relativa como clave,
   así que mover uno obliga a re-ejecutar el benchmark, el pool o las ablaciones (decenas de horas). Si el cambio
   es deliberado (detector nuevo, re-ejecución planificada), actualiza la lista de abajo en el mismo PR.
2. Deja ficheros sueltos en la raíz o recrea ``requirements/`` (las dependencias van en ``pyproject.toml``).
3. Pone un notebook fuera de ``notebooks/fase<N>/`` o sin el patrón ``NN_nombre.ipynb``.
4. Crea un paquete en ``src/`` sin README ni ``__init__.py``.
"""
from __future__ import annotations

import re
import subprocess

import pytest

from regimenes import rutas

# --------------------------------------------------------------------------- #
# 1. Ficheros con huella: rutas fijas
# --------------------------------------------------------------------------- #
HUELLA_BENCHMARK = [
    "src/regimenes/detectores/__init__.py",
    "src/regimenes/detectores/base.py",
    "src/regimenes/detectores/f1_reglas/__init__.py",
    "src/regimenes/detectores/f1_reglas/rule_composite_riskoff.py",
    "src/regimenes/detectores/f1_reglas/rule_vix_threshold.py",
    "src/regimenes/detectores/f1_reglas/turbulence_mahalanobis.py",
    "src/regimenes/detectores/f2_clustering/__init__.py",
    "src/regimenes/detectores/f2_clustering/clustering_gmm.py",
    "src/regimenes/detectores/f2_clustering/jump_model.py",
    "src/regimenes/detectores/f3_hmm/__init__.py",
    "src/regimenes/detectores/f3_hmm/_hmm_t_utils.py",
    "src/regimenes/detectores/f3_hmm/_hmm_utils.py",
    "src/regimenes/detectores/f3_hmm/hmm_gaussian_2s.py",
    "src/regimenes/detectores/f3_hmm/hmm_tstudent.py",
    "src/regimenes/detectores/f4_switching/__init__.py",
    "src/regimenes/detectores/f4_switching/markov_switching_var.py",
    "src/regimenes/detectores/f5_garch/__init__.py",
    "src/regimenes/detectores/f5_garch/garch_t_vol.py",
    "src/regimenes/detectores/f5_garch/msgarch_regime.py",
    "src/regimenes/detectores/f6_changepoint/__init__.py",
    "src/regimenes/detectores/f6_changepoint/changepoint_online.py",
    "src/regimenes/detectores/f7_deep/__init__.py",
    "src/regimenes/detectores/f7_deep/deep_ae_regime.py",
    "src/regimenes/detectores/registry.py",
    "src/regimenes/evaluacion/metricas.py",
    "src/regimenes/evaluacion/walk_forward.py",
]
HUELLA_POOL = [
    "configs/bloques.yaml",
    "src/regimenes/benchmark/pool.py",
    "src/regimenes/detectores/pool/__init__.py",
    "src/regimenes/detectores/pool/absorption_ratio.py",
    "src/regimenes/detectores/pool/disponibilidad.py",
    "src/regimenes/detectores/pool/especificaciones.py",
    "src/regimenes/detectores/pool/pool_crudo.py",
    "src/regimenes/detectores/pool/shu.py",
    "src/regimenes/detectores/pool/shu_supervisados.py",
    "src/regimenes/detectores/pool/sparse_jump.py",
    "src/regimenes/detectores/pool/supervisados.py",
    "src/regimenes/features/rentabilidades.py",
]
HUELLA_ABLACIONES = [
    "src/regimenes/benchmark/ablaciones.py",
    "src/regimenes/features/bloques.py",
    "configs/bloques.yaml",
]
FIJOS = [  # también fijos: la huella los cita por ruta o por nombre de módulo (hash AST)
    "configs/benchmark_spec.yaml",
    "src/regimenes/benchmark/ejecucion.py",
    "src/regimenes/benchmark/cache.py",
    "src/regimenes/evaluacion/metricas.py",
    "src/regimenes/rutas.py",
]

AVISO = ("mover, renombrar, añadir o quitar ficheros con huella invalida la caché (ver la docstring de este "
         "test y ADR-012 §3); si es deliberado, actualiza la lista en el mismo PR")


def test_ficheros_con_huella_siguen_en_su_sitio() -> None:
    faltan = [r for r in HUELLA_BENCHMARK + HUELLA_POOL + HUELLA_ABLACIONES + FIJOS if not (rutas.ROOT / r).is_file()]
    assert not faltan, f"{AVISO}. Faltan: {faltan}"


def test_la_huella_del_benchmark_cubre_exactamente_estos_ficheros() -> None:
    from regimenes.benchmark import cache

    actual = sorted(cache._relative_hashes(cache._cache_code_paths()).keys())
    assert actual == sorted(HUELLA_BENCHMARK), AVISO


def test_la_huella_del_pool_cubre_exactamente_estos_ficheros() -> None:
    from regimenes.benchmark import pool

    assert sorted(pool.huella_codigo().keys()) == sorted(HUELLA_POOL), AVISO


# --------------------------------------------------------------------------- #
# 2–4. Estructura
# --------------------------------------------------------------------------- #
RAIZ_PERMITIDA = {
    ".env.example", ".gitattributes", ".github", ".gitignore", "LICENSE", "Makefile", "README.md",
    "configs", "constraints.txt", "data", "docs", "notebooks", "pyproject.toml", "requirements.txt",
    "results", "src", "tests",
}
NOTEBOOK = re.compile(r"^notebooks/fase\d/(?:[a-z_]+/)?\d\d_[A-Za-z0-9_]+\.ipynb$")


def _versionados() -> list[str]:
    try:
        salida = subprocess.run(["git", "ls-files"], cwd=rutas.ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("sin git o fuera de un repositorio")
    return salida.stdout.splitlines()


def test_raiz_sin_ficheros_sueltos() -> None:
    raiz = {f.split("/")[0] for f in _versionados()}
    sobran = sorted(raiz - RAIZ_PERMITIDA)
    assert not sobran, f"ficheros o carpetas nuevos en la raíz: {sobran}; ¿van en docs/, configs/ o src/?"


def test_dependencias_solo_en_pyproject() -> None:
    assert not any(f.startswith("requirements/") for f in _versionados()), "las dependencias van en pyproject.toml (extras)"
    efectivas = [l for l in (rutas.ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.lstrip().startswith("#")]
    assert efectivas == ["-c constraints.txt", "-e .[todo]"], "requirements.txt es solo un alias de pyproject.toml"


def test_notebooks_por_fase_y_numerados() -> None:
    malos = [f for f in _versionados() if f.endswith(".ipynb") and f.startswith("notebooks/") and not NOTEBOOK.match(f)]
    assert not malos, f"notebooks fuera de notebooks/fase<N>/[<tema>/]NN_nombre.ipynb: {malos}"


def test_paquetes_de_src_documentados() -> None:
    versionados = set(_versionados())
    paquetes = sorted({f.split("/")[1] for f in versionados if f.startswith("src/") and f.count("/") >= 2})
    for paquete in paquetes:
        assert f"src/{paquete}/__init__.py" in versionados, f"src/{paquete} sin __init__.py"
        assert f"src/{paquete}/README.md" in versionados, f"src/{paquete} sin README.md"
