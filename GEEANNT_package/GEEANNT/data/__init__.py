from .io import (
    load_detector_table,
    report_basic_table_checks,
    save_detector_table,
)

from .preprocessing import (
    build_physical_features,
    print_physical_summary,
    build_energy_bins,
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
    "save_detector_table",
    "report_basic_table_checks",

    # preprocessing
    "build_physical_features",
    "print_physical_summary",
    "build_energy_bins",
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