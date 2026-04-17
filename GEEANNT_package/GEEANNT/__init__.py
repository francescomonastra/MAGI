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

# ==========================================================
# Data API
# ==========================================================
from .data import (
    load_detector_table,
    report_basic_table_checks,
    build_physical_features,
    print_physical_summary,
    build_energy_bins,
    build_feature_dataframe,
    report_feature_dataframe,
    filter_particle_types_and_discretize_uv,
    report_discretized_features,
    split_feature_data,
    report_split_summary,
    scale_continuous_features,
    report_scaled_features,
    to_one_hot,
    build_conditioning_and_weights,
    report_conditioning,
    build_tf_datasets,
    report_tf_datasets,
    report_energy_binning_diagnostics,
)

# ==========================================================
# Plotting / utilities
# ==========================================================
from .utils import (
    plot_history,
    plot_dist,
    plot_dist_by_class,
    plot_correlation_matrix,
    plot_covariance_matrix,
    plot_pairwise_sample,
    plot_pairgrid_physics,
    print_model_structure,
)

# ==========================================================
# Geneation API and validation API
# ==========================================================
from .generation import (
    sample_types,
    one_hot_from_idx,
    energy_from_idx,
    generate_latent_outputs,
    renorm_cos_sin,
    u_from_s,
    reconstruct_generated_features,
    reconstruct_generated_physics,
    reconstruct_real_test_physics,
)
from .validation import (
    compute_wasserstein_scores,
    report_generated_constraints,
    compare_hist_with_residuals,
    report_final_ranges,
    report_norm_checks,
    build_real_generated_featureframes,
)

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
    "load_detector_table",
    "report_basic_table_checks",
    "build_physical_features",
    "print_physical_summary",
    "plot_history",
    "plot_dist",
    "plot_dist_by_class",
    "plot_correlation_matrix",
    "plot_covariance_matrix",
    "plot_pairwise_sample",
    "plot_pairgrid_physics",
    "build_energy_bins",
    "build_feature_dataframe",
    "report_feature_dataframe",
    "filter_particle_types_and_discretize_uv",
    "report_discretized_features",
    "split_feature_data",
    "report_split_summary",
    "scale_continuous_features",
    "report_scaled_features",
    "to_one_hot",
    "build_conditioning_and_weights",
    "report_conditioning",
    "build_tf_datasets",
    "report_tf_datasets",
    "report_energy_binning_diagnostics",
    "sample_types",
    "one_hot_from_idx",
    "energy_from_idx",
    "generate_latent_outputs",
    "renorm_cos_sin",
    "u_from_s",
    "reconstruct_generated_features",
    "reconstruct_generated_physics",
    "reconstruct_real_test_physics",
    "compute_wasserstein_scores",
    "report_generated_constraints",
    "compare_hist_with_residuals",
    "report_final_ranges",
    "report_norm_checks",
    "build_real_generated_featureframes",
]