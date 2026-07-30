"""
Utility functions for MAGI.
"""

from .plotting import (
    plot_history,
    plot_dist,
    plot_dist_by_class,
    plot_correlation_matrix,
    plot_covariance_matrix,
    plot_pairwise_sample,
    plot_pairgrid_physics,
    set_plot_theme,
    _save_and_show,
)

from .model_inspection import (
    print_model_structure,
    print_trainable_status,
    print_model_tree_with_params,
    print_duplicate_trainable_variables

)

from .circuit_viz import (
    render_routing_circuit_html,
    save_routing_circuit,
)

from .full_circuit import (
    compute_full_circuit_trace,
    render_full_circuit_html,
    save_full_circuit,
)

__all__ = [
    "set_plot_theme",
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
    "_save_and_show",
    "render_routing_circuit_html",
    "save_routing_circuit",
    "compute_full_circuit_trace",
    "render_full_circuit_html",
    "save_full_circuit",
]