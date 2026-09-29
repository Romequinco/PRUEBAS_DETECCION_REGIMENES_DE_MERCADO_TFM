"""Familia 1, reglas: rule_vix_threshold (D01), rule_composite_riskoff (D02), turbulence_mahalanobis (D10).

Implementaciones movidas con git mv desde la Capa 1 congelada (tag capa1-final).
"""

# Submodulos publicos; no se importan aqui (carga perezosa via DetectorSpec.factory()).
__all__ = ["rule_composite_riskoff", "rule_vix_threshold", "turbulence_mahalanobis"]
