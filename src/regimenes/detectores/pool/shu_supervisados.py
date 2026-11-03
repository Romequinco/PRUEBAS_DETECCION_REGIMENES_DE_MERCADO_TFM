"""Especificaciones de D15, D16 y D17 (ADR-010 §3).

- ``D15-U``: jump model de Shu, Yu y Mulvey sobre ``SP500_ret`` (solo aporta las fechas; las features
  salen de la historia cruda, §3.1). Walk-forward expansivo con paso 21; la ventana móvil de 3.000
  días es interna al detector.
- ``D16-P`` / ``D17-P``: logit L1 y random forest sobre el pool completo, envueltos en
  ``disponibilidad.ConDisponibilidad`` (≥ 756 observaciones por columna, §2.1).
- ``D16-N`` / ``D17-N``: los mismos sobre el núcleo N (``registry.CORE_A`` / ``CORE_B``).
- ``D16-P-excl`` / ``D17-P-excl``: sensibilidad del §3.2 que saca del ajuste las crisis sin terminar
  (sin peso en el veredicto).
"""

from __future__ import annotations

from regimenes.detectores.pool.especificaciones import SpecPool
from regimenes.detectores.registry import CORE_A, CORE_B, SEED
from regimenes.features.bloques import cargar_bloques

MIN_OBS_P: int = 756

PARAMS_D16: dict = {"C": 1.0}
PARAMS_D17: dict = {"n_estimators": 500, "max_depth": 5, "min_samples_leaf": 50}


def columnas_pool(track: str) -> tuple[str, ...]:
    """Columnas del menú P: todas las de ``configs/bloques.yaml`` de la pista, en su orden."""
    return tuple(c for b in cargar_bloques(track).values() for c in b["columnas"])


def _supervisado(track: str, ident: str, etiqueta: str, clase: str, hiper: dict, menu: str,
                 excluir: bool = False) -> SpecPool:
    params_base = {**hiper, "pista": track, "excluir_sin_terminar": excluir, "random_state": SEED}
    sufijo = "-excl" if excluir else ""
    variante = ("sensibilidad §3.2: crisis sin terminar fuera del ajuste" if excluir
                else "principal §3.2: crisis sin terminar cuentan como calma")
    if menu == "P":
        cols = columnas_pool(track)
        return SpecPool(
            f"{ident}-P{sufijo}", etiqueta, "Supervisados", "disponibilidad", "ConDisponibilidad",
            cols, {"base": f"supervisados.{clase}", "params": params_base, "min_obs": MIN_OBS_P},
            variant=f"menú P: {len(cols)} columnas; {variante}", menu="P")
    cols = CORE_A if track == "A" else CORE_B
    return SpecPool(f"{ident}-N{sufijo}", etiqueta, "Supervisados", "supervisados", clase, cols,
                    params_base, variant=f"menú N: {len(cols)} columnas; {variante}", menu="N")


def especificaciones_shu_supervisados(track: str) -> list[SpecPool]:
    """D15-U, D16-P, D16-N, D17-P, D17-N y las sensibilidades D16-P-excl y D17-P-excl."""
    track = track.upper()
    if track not in {"A", "B"}:
        raise ValueError("track debe ser 'A' o 'B'.")
    d15 = SpecPool(
        "D15-U", "Jump model de Shu, Yu y Mulvey (2024)", "Jump model", "shu", "ShuJumpModel",
        ("SP500_ret",), {"random_state": SEED}, slow=True,
        variant="menú U: features del paper desde la historia cruda (SP500 + DTB3); "
                "λ elegida en cada reajuste por Sharpe online", menu="U")
    return [
        d15,
        _supervisado(track, "D16", "Logit L1", "LogitL1", PARAMS_D16, "P"),
        _supervisado(track, "D16", "Logit L1", "LogitL1", PARAMS_D16, "N"),
        _supervisado(track, "D17", "Random forest", "BosqueAleatorio", PARAMS_D17, "P"),
        _supervisado(track, "D17", "Random forest", "BosqueAleatorio", PARAMS_D17, "N"),
        _supervisado(track, "D16", "Logit L1", "LogitL1", PARAMS_D16, "P", excluir=True),
        _supervisado(track, "D17", "Random forest", "BosqueAleatorio", PARAMS_D17, "P", excluir=True),
    ]
