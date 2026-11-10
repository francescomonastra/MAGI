from .io import (
    load_detector_table,
    select_first_entry,
    report_basic_table_checks,
    save_detector_table,
    save_normalization_summary,
    load_normalization_summary,
    save_candidate_energy_lines,
    load_candidate_energy_lines,
    load_primaries_table,
    load_run_summaries,
    DEFAULT_COLUMNS_DET_PRIMTRUTH,
)

from .primary_truth import (
    build_multiplicity_table,
    sample_multiplicity,
    resample_primary_energy,
    sample_group_types,
    PRIMARY_JOIN_KEY_RULE,
)

from .preprocessing import (
    compute_primary_fraction,
    build_physical_features,
    print_physical_summary,
    build_energy_bins,
    DEFAULT_CANDIDATE_ENERGY_LINES,
    detect_energy_lines,
    refine_peak_energy,
    print_detected_energy_lines,
    build_gate_targets,
    add_uncollided_gate_target,
    build_zone_probs_table,
    primary_energy_knots_from_transformer,
    line_logsigma_from_resolution,
    measure_line_centroid,
    confirm_unresolved_candidate_lines,
    fit_cdf_warp_knots,
    transform_quantile_values,
    build_feature_dataframe,
    report_feature_dataframe,
    fit_quantile_geometry_transforms,
)

from .dataset import (
    # v0.6
    filter_particle_types_and_discretize_uv,
    report_discretized_features,

    # v0.7
    filter_particle_types_continuous_geometry,
    report_continuous_geometry_features,

    # HENABICS primary truth
    build_primary_conditioning,

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

__all__ = [
    # io
    "load_detector_table",
    "select_first_entry",
    "save_detector_table",
    "report_basic_table_checks",
    "save_normalization_summary",
    "load_normalization_summary",
    "save_candidate_energy_lines",
    "load_candidate_energy_lines",
    "load_primaries_table",
    "load_run_summaries",
    "DEFAULT_COLUMNS_DET_PRIMTRUTH",
    "build_multiplicity_table",
    "sample_multiplicity",
    "resample_primary_energy",
    "sample_group_types",
    "PRIMARY_JOIN_KEY_RULE",
    "build_primary_conditioning",

    # preprocessing
    "compute_primary_fraction",
    "build_physical_features",
    "print_physical_summary",
    "build_energy_bins",
    "DEFAULT_CANDIDATE_ENERGY_LINES",
    "detect_energy_lines",
    "refine_peak_energy",
    "print_detected_energy_lines",
    "build_gate_targets",
    "add_uncollided_gate_target",
    "build_zone_probs_table",
    "primary_energy_knots_from_transformer",
    "line_logsigma_from_resolution",
    "measure_line_centroid",
    "confirm_unresolved_candidate_lines",
    "fit_cdf_warp_knots",
    "transform_quantile_values",
    "build_feature_dataframe",
    "report_feature_dataframe",
    "fit_quantile_geometry_transforms",

    # dataset v0.6
    "filter_particle_types_and_discretize_uv",
    "report_discretized_features",

    # dataset v0.7
    "filter_particle_types_continuous_geometry",
    "report_continuous_geometry_features",

    # shared dataset tools
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
]