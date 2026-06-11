"""
GEEANNT
Geant4 Efficiency Enhancing Artificial Neural Network Toolkit
"""

__version__ = "0.7.0"

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
from .core import (
    CVAE_CatEnergy_CatUV,
    CVAE_CatEnergy_CatUV_TaskAdaptive,
    CVAE_CatEnergy_ContGeom_TaskAdaptive
)
from .training import (
    build_default_callbacks,
    compile_model,
    fit_model,
    train_single_run,
    TaskAdaptiveLossScheduler,
    TaskAdaptiveTrainingMonitor,
    ValidationEnergyDistributionMonitor,
    save_training_checkpoint,
    save_final_trained_model,
    extract_callback_metadata,
    load_json,
    load_metadata,
    load_task_adaptive_model_for_generation,
)

# ==========================================================
# Data API
# ==========================================================
from .data import (
    save_detector_table,
    load_detector_table,
    report_basic_table_checks,
    build_physical_features,
    print_physical_summary,
    build_energy_bins,
    build_feature_dataframe,
    report_feature_dataframe,
    fit_quantile_geometry_transforms,

    # v0.6
    filter_particle_types_and_discretize_uv,
    report_discretized_features,

    # v0.7
    filter_particle_types_continuous_geometry,
    report_continuous_geometry_features,

    # shared
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
    set_plot_theme,
    plot_history,
    plot_dist,
    plot_dist_by_class,
    plot_correlation_matrix,
    plot_covariance_matrix,
    plot_pairwise_sample,
    plot_pairgrid_physics,
    print_model_structure,
    print_trainable_status,
    print_model_tree_with_params,
    print_duplicate_trainable_variables,
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
    generated_physics_to_detector_dataframe,
    generate_detector_table_to_file,
    generate_detector_input_file,
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
    "CVAE_CatEnergy_CatUV_TaskAdaptive",
    "CVAE_CatEnergy_ContGeom_TaskAdaptive",
    "build_default_callbacks",
    "compile_model",
    "fit_model",
    "train_single_run",
    "TaskAdaptiveLossScheduler",
    "TaskAdaptiveTrainingMonitor",
    "ValidationEnergyDistributionMonitor",
    "save_training_checkpoint",
    "save_final_trained_model",
    "extract_callback_metadata",
    "load_json",
    "load_metadata",
    "load_task_adaptive_model_for_generation",
    "load_detector_table",
    "save_detector_table",
    "report_basic_table_checks",
    "build_physical_features",
    "print_physical_summary",
    "print_model_structure",
    "print_trainable_status",
    "print_model_tree_with_params",
    "set_plot_theme",
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
    "fit_quantile_geometry_transforms",

    # v0.6
    "filter_particle_types_and_discretize_uv",
    "report_discretized_features",

    # v0.7
    "filter_particle_types_continuous_geometry",
    "report_continuous_geometry_features",

    # shared
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
    "fit_quantile_geometry_transforms",
    "sample_types",
    "one_hot_from_idx",
    "energy_from_idx",
    "generate_latent_outputs",
    "renorm_cos_sin",
    "u_from_s",
    "reconstruct_generated_features",
    "reconstruct_generated_physics",
    "reconstruct_real_test_physics",
    "generated_physics_to_detector_dataframe",
    "generate_detector_table_to_file",
    "generate_detector_input_file", 
    "compute_wasserstein_scores",
    "report_generated_constraints",
    "compare_hist_with_residuals",
    "report_final_ranges",
    "report_norm_checks",
    "build_real_generated_featureframes",
]