"""
Core neural-network components for MAGI.
"""

from .model import (
    CVAE_CatEnergy_CatUV, 
    CVAE_CatEnergy_CatUV_TaskAdaptive,
    CVAE_CatEnergy_ContGeom_TaskAdaptive,
    CVAE_CatEnergy_ContPhi_TaskAdaptive
)

__all__ = [
    "CVAE_CatEnergy_CatUV",
    "CVAE_CatEnergy_CatUV_TaskAdaptive",
    "CVAE_CatEnergy_ContGeom_TaskAdaptive",
    "CVAE_CatEnergy_ContPhi_TaskAdaptive"
]