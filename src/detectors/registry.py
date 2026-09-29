"""Configuración reproducible de los 12 detectores de la Capa 1 sobre datos v2."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = ROOT / "capa1_exploracion"


# Semilla explícita de los detectores estocásticos (D03 GMM, D04/D08 HMM, D09 jump
# model, D12 AE+GMM). Coincide con el defecto de sus constructores congelados
# (random_state=42), así que no cambia ningún resultado; se declara aquí para que
# el ajuste de cada fold y el ajuste final (AIC/BIC/loglik) no dependan de ese
# defecto y para que la semilla forme parte de la huella de caché (asdict(spec)).
# D05 (search_reps=0), D11 (arranques fijos) y los detectores deterministas no
# consumen aleatoriedad.
SEED = 42

CORE_A = (
    "SP500_ret_z",
    "SP500_vol_z",
    "SP500_momentum",
    "SP500_drawdown",
    "FF_MKT_z",
    "DGS10_change_z",
    "credit_BaaAaa_mensual_z",
    "term_spread_hist_z",
    "INDPRO_yoy_z",
)

CORE_B = CORE_A + (
    "VIX_level_z",
    "MOVE_level_z",
    "credit_BAA10Y_z",
    "DFII10_change_z",
    "corr_spx_bond",
)


@dataclass(frozen=True)
class DetectorSpec:
    """Una configuración del benchmark; no contiene estado ajustado."""

    detector_id: str
    label: str
    family: str
    module: str
    class_name: str
    features: tuple[str, ...]
    params: dict[str, Any]
    step: int = 21
    expanding: bool = True
    slow: bool = False
    variant: str = ""

    def factory(self):
        """Crea una instancia nueva, como exige ``evaluation.walk_forward``."""
        legacy = str(LEGACY_ROOT)
        if legacy not in sys.path:
            sys.path.insert(0, legacy)
        cls = getattr(import_module(f"detectors.{self.module}"), self.class_name)
        return cls(**self.params)

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.detector_id,
            "detector": self.label,
            "familia": self.family,
            "n_features": len(self.features),
            "features": ", ".join(self.features),
            "step": self.step,
            "ventana_train": "expanding" if self.expanding else "rolling",
            "lento": self.slow,
            "variante_v2": self.variant,
        }


def detector_specs(track: str) -> list[DetectorSpec]:
    """Devuelve los 12 detectores configurados para la pista ``A`` o ``B``.

    Las configuraciones son hipótesis previas al benchmark. No se eligen columnas
    mirando sus métricas posteriores. En A, donde VIX no pertenece al panel
    congelado, D1 usa la volatilidad realizada como equivalente histórico y queda
    marcado explícitamente como variante; no se oculta bajo el nombre original.
    """
    track = track.upper()
    if track not in {"A", "B"}:
        raise ValueError("track debe ser 'A' o 'B'.")

    core = CORE_A if track == "A" else CORE_B
    d1_feature = "SP500_vol_z" if track == "A" else "VIX_level_z"
    d1_variant = (
        "A: umbral de volatilidad realizada; VIX no existe en la pista histórica"
        if track == "A" else "B: regla VIX original"
    )
    d2_signs = (
        {
            "SP500_vol_z": +1,
            "credit_BaaAaa_mensual_z": +1,
            "term_spread_hist_z": -1,
            "SP500_drawdown": -1,
        }
        if track == "A"
        else {
            "VIX_level_z": +1,
            "credit_BAA10Y_z": +1,
            "slope_10y2y_z": -1,
            "SP500_drawdown": -1,
        }
    )
    d10_features = (
        ("SP500_ret", "SP500_vol_z", "DGS10_change_z", "FF_MKT_z")
        if track == "A"
        else ("SP500_ret", "VIX_change_z", "DXY_change_z", "slope_10y2y_z")
    )

    return [
        DetectorSpec("D01", "Regla de volatilidad/VIX", "Reglas", "rule_vix_threshold", "RuleVixThreshold", (d1_feature,), {"feature": d1_feature}, variant=d1_variant),
        DetectorSpec("D02", "Regla compuesta risk-off", "Reglas", "rule_composite_riskoff", "RuleCompositeRiskoff", tuple(d2_signs), {"signs": d2_signs}),
        DetectorSpec("D03", "GMM (K=3)", "Clustering", "clustering_gmm", "ClusteringGMM", core, {"n_states": 3, "random_state": SEED}),
        DetectorSpec("D04", "HMM gaussiano (K=2)", "HMM", "hmm_gaussian_2s", "HMMGaussian2S", core, {"n_states": 2, "n_init": 5, "features": list(core), "random_state": SEED}),
        DetectorSpec("D05", "Markov-Switching varianza", "Markov-Switching", "markov_switching_var", "MarkovSwitchingVar", ("SP500_ret",), {"feature": "SP500_ret", "n_states": 2}, step=63, slow=True),
        DetectorSpec("D06", "GJR-GARCH-t", "GARCH", "garch_t_vol", "GarchTVol", ("SP500_ret",), {"feature": "SP500_ret"}),
        DetectorSpec("D07", "CUSUM online robusto", "Change-point", "changepoint_online", "ChangepointOnline", ("SP500_ret",), {"feature": "SP500_ret", "cost": "robust"}),
        DetectorSpec("D08", "HMM t-Student (K=4)", "HMM", "hmm_tstudent", "HMMTStudent", core, {"n_states": 4, "n_init": 3, "t_n_iter": 25, "features": list(core), "random_state": SEED}, step=126, slow=True),
        DetectorSpec("D09", "Statistical Jump Model", "Jump model", "jump_model", "JumpModel", core, {"n_states": 2, "jump_penalty": 50.0, "random_state": SEED}),
        DetectorSpec("D10", "Turbulencia Mahalanobis", "Reglas multivariantes", "turbulence_mahalanobis", "TurbulenceMahalanobis", d10_features, {"features": list(d10_features)}),
        DetectorSpec("D11", "MS-GARCH", "GARCH", "msgarch_regime", "MSGarchRegime", ("SP500_ret",), {"feature": "SP500_ret", "n_init": 1, "maxiter": 100}, step=126, expanding=False, slow=True),
        DetectorSpec("D12", "Autoencoder + GMM", "Deep learning", "deep_ae_regime", "DeepAERegime", core, {"n_states": 2, "latent_dim": 2, "hidden": 8, "epochs": 40, "gmm_n_init": 3, "random_state": SEED}, slow=True),
    ]


def specs_table(track: str) -> pd.DataFrame:
    return pd.DataFrame([spec.to_row() for spec in detector_specs(track)])
