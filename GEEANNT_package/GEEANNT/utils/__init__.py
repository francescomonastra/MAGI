"""
Utility functions for GEEANNT.
"""

from .plotting import (
    plot_history,
    plot_dist,
    plot_dist_by_class,
    plot_correlation_matrix,
    plot_covariance_matrix,
    plot_pairwise_sample,
    plot_pairgrid_physics,
)

from .model_inspection import (
    print_model_structure,
    print_trainable_status,
    print_model_tree_with_params,
    print_duplicate_trainable_variables

)
__all__ = [
    "plot_history",
    "plot_dist",
    "plot_dist_by_class",
    "plot_correlation_matrix",
    "plot_covariance_matrix",
    "plot_pairwise_sample",
    "plot_pairgrid_physics",
    "print_model_structure",
    "print_trainable_status",
    "print_model_tree_with_params",
    "print_duplicate_trainable_variables",
]