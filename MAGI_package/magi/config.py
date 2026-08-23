"""
Configuration helpers for MAGI.
"""

import os
import random
import numpy as np


def initialize_environment(seed=42, cpu_only=True, quiet=True):
    """
    Configure the runtime environment.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    cpu_only : bool
        If True, disable GPU visibility for TensorFlow.
    quiet : bool
        If True, reduce TensorFlow log verbosity.

    Returns
    -------
    None

    Notes
    -----
    Call this before anything else in a notebook or script. `cpu_only` in
    particular only takes effect if it runs before TensorFlow initializes
    its devices - setting it after the first TF op has no effect. In scripts
    that import magi at module level, set CUDA_VISIBLE_DEVICES=-1 in the
    environment instead.
    """
    if quiet:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import tensorflow as tf

    if cpu_only:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            # This may fail if TF runtime is already initialized.
            pass

    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def print_tf_info():
    """
    Print TensorFlow / Keras version and visible devices.

    Use it right after initialize_environment to confirm the run is on the
    device you intended - the CPU-only setting is easy to lose.

    Returns
    -------
    None
    """
    import tensorflow as tf
    from tensorflow import keras

    print("TensorFlow version:", tf.__version__)
    print("Keras version     :", keras.__version__)

    print("\nPhysical devices:", tf.config.list_physical_devices())
    print("GPU devices     :", tf.config.list_physical_devices("GPU"))
    print("Logical devices :", tf.config.list_logical_devices())