"""pool_crudo.py — Variantes no supervisadas sobre el pool sin comprimir (ADR-010 §2–§3).

Menús:

- ``P`` · pool completo: las columnas de ``configs/bloques.yaml`` de la pista (35 en A, 77 en
  B), en el orden del fichero, ya transformadas por el preprocesado (z-scores causales).
- ``R`` · 22 rentabilidades log diarias multiactivo, solo en B
  (:mod:`regimenes.features.rentabilidades`).

Variantes (ids del ejecutor ``regimenes.benchmark.pool``):

| Id | Detector | Pistas |
|---|---|---|
| ``D10-P`` | D10 oficial (``f1_reglas.turbulence_mahalanobis``, sin tocar) sobre P | A, B |
| ``D10-R`` | D10 oficial sobre R (Kritzman y Li, 2010) | B |
| ``D14-R`` | Absorption ratio (:mod:`.absorption_ratio`) sobre R | B |
| ``D18-P`` | Sparse jump model (:mod:`.sparse_jump`) sobre P | A, B |

D10-P, D10-R y D18-P van dentro de :class:`~regimenes.detectores.pool.disponibilidad.ConDisponibilidad`
(regla de 756 observaciones del §2.1). D10 conserva sus parámetros de v2 (``q_in = 0,90``,
``q_out = 0,70``, ``min_dwell = 5``, ``cov_min_periods = 252``) y su forma de tratar los NaN al
predecir (``predecir_con_nan=True``): solo cambian las columnas. D14 no se ajusta y su propia
especificación ya fija qué filas usa (las 500 últimas con las 22 series), así que no lleva el
envoltorio. Todas: paso de reajuste 21 (como D09 y D10), ventana expansiva y semilla 42.
"""

from __future__ import annotations

from regimenes.detectores.pool.especificaciones import SpecPool
from regimenes.detectores.registry import SEED
from regimenes.features.bloques import cargar_bloques
from regimenes.features.rentabilidades import columnas_r

D10_CLASE = "regimenes.detectores.f1_reglas.turbulence_mahalanobis.TurbulenceMahalanobis"
D10_PARAMS = {"q_in": 0.90, "q_out": 0.70, "min_dwell": 5, "cov_min_periods": 252}
D14_PARAMS = {"ventana": 500, "semivida": 250.0, "n_autovectores": 4, "corto": 15, "largo": 252,
              "umbral": 1.0}
D18_JUMP_PENALTY = 50.0
D18_MAX_FEATS = {"A": 9, "B": 14}            # tamaño de N (CORE_A / CORE_B)
MIN_OBS = 756


def columnas_p(track: str) -> tuple[str, ...]:
    """Columnas del menú P de la pista, en el orden de ``configs/bloques.yaml``."""
    return tuple(c for b in cargar_bloques(track).values() for c in b["columnas"])


def bloque_de_columna(track: str) -> dict[str, str]:
    """``{columna del menú P: bloque de bloques.yaml}`` (para el informe de D18)."""
    return {c: nombre for nombre, b in cargar_bloques(track).items() for c in b["columnas"]}


def _d10(track: str, menu: str, cols: tuple[str, ...]) -> SpecPool:
    return SpecPool(
        detector_id=f"D10-{menu}", label=f"Turbulencia Mahalanobis (menú {menu})",
        family="Reglas multivariantes", module="disponibilidad", class_name="ConDisponibilidad",
        features=cols,
        params={"base": D10_CLASE, "params": dict(D10_PARAMS), "min_obs": MIN_OBS,
                "predecir_con_nan": True},
        variant=f"menú {menu}: {len(cols)} columnas; D10 de v2 con disponibilidad (ADR-010 §2.1)",
        menu=menu,
    )


def especificaciones_pool_crudo(track: str) -> list[SpecPool]:
    """D10-P y D18-P en las dos pistas; D10-R y D14-R además en B."""
    track = track.upper()
    if track not in {"A", "B"}:
        raise ValueError("track debe ser 'A' o 'B'.")
    p = columnas_p(track)
    specs = [_d10(track, "P", p)]
    if track == "B":
        r = columnas_r()
        specs.append(_d10(track, "R", r))
        specs.append(SpecPool(
            detector_id="D14-R", label="Absorption ratio (menú R)", family="Reglas multivariantes",
            module="absorption_ratio", class_name="AbsorptionRatio", features=r,
            params={**D14_PARAMS, "random_state": SEED},
            variant="menú R: 22 rentabilidades; AR n=4, ventana 500, semivida 250, ΔAR >= 1",
            menu="R",
        ))
    specs.append(SpecPool(
        detector_id="D18-P", label="Sparse jump model (menú P)", family="Jump model",
        module="disponibilidad", class_name="ConDisponibilidad", features=p,
        params={"base": "sparse_jump.SparseJumpPool",
                "params": {"n_states": 2, "jump_penalty": D18_JUMP_PENALTY,
                           "max_feats": D18_MAX_FEATS[track], "random_state": SEED},
                "min_obs": MIN_OBS},
        variant=f"menú P: {len(p)} columnas; max_feats={D18_MAX_FEATS[track]} (κ²), λ=50",
        menu="P",
    ))
    return sorted(specs, key=lambda s: s.detector_id)
