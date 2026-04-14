"""
High-level user-facing API for GEEANNT.

This module provides a simpler interface for notebook users.
"""

from .config import initialize_environment, print_tf_info
from .core import CVAE_CatEnergy_CatUV
from .training import (
    build_default_callbacks,
    compile_model,
    fit_model,
    train_single_run,
)
from .utils import plot_history


def setup(seed=42, cpu_only=True, quiet=True, show_info=False):
    """
    Initialize the GEEANNT runtime environment.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    cpu_only : bool
        If True, force TensorFlow to run without GPU.
    quiet : bool
        If True, reduce TensorFlow verbosity.
    show_info : bool
        If True, print TensorFlow runtime information.
    """
    initialize_environment(seed=seed, cpu_only=cpu_only, quiet=quiet)
    if show_info:
        print_tf_info()


def build_model(**kwargs):
    """
    Convenience wrapper around the main CVAE model constructor.
    """
    return CVAE_CatEnergy_CatUV(**kwargs)


def train_model(
    model,
    train_ds,
    val_ds,
    learning_rate=2e-4,
    epochs=60,
    callbacks=None,
    verbose=1,
):
    """
    Compile and train a model in a single call.
    """
    return train_single_run(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        learning_rate=learning_rate,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )


def plot_training(history, keys=None, show_available=True):
    """
    Wrapper around the standard training-history plotter.
    """
    plot_history(history, keys=keys, show_available=show_available)