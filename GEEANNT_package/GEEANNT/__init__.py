"""
GEEANNT
Geant4 Efficiency Enhancing Artificial Neural Network Toolkit
"""

__version__ = "0.6.0"

# ==========================================================
# Environment / configuration helpers
# ==========================================================
from .config import initialize_environment, print_tf_info

# Backward-compatible aliases
set_seed = initialize_environment
configure_tensorflow = initialize_environment

# ==========================================================
# Core public API
# ==========================================================
from .core import CVAE_CatEnergy_CatUV
from .training import (
    build_default_callbacks,
    compile_model,
    fit_model,
    train_single_run,
)
from .utils import plot_history

__all__ = [
    "__version__",
    "initialize_environment",
    "print_tf_info",
    "set_seed",
    "configure_tensorflow",
    "CVAE_CatEnergy_CatUV",
    "build_default_callbacks",
    "compile_model",
    "fit_model",
    "train_single_run",
    "plot_history",
]