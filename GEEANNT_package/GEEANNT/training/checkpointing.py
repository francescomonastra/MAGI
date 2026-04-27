"""
Checkpointing utilities for GEEANNT.

This module provides robust save utilities for subclassed Keras models,
including task-adaptive training metadata, callback configurations, callback
states, task weights, preprocessing metadata, and training history.
"""

import os
import json
import time
import numpy as np


# ==========================================================
# JSON utilities
# ==========================================================

def _to_json_safe(obj):
    """
    Convert common non-JSON-serializable objects into JSON-safe objects.
    """
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    # TensorFlow tensors / variables
    try:
        import tensorflow as tf
        if isinstance(obj, (tf.Tensor, tf.Variable)):
            arr = obj.numpy()
            if np.ndim(arr) == 0:
                return arr.item()
            return arr.tolist()
    except Exception:
        pass

    # Keras / TensorFlow configs are often already JSON-safe
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


# ==========================================================
# Keras subclassed model save helper
# ==========================================================

def _ensure_model_saveable(model):
    """
    Make a subclassed Keras model saveable.

    Keras subclassed models can have all variables created after training,
    but still report model.built == False if the outer model was never called
    through call(). In that case, save_weights() may raise an error.

    If weights already exist, we safely mark the model as built.
    """
    if getattr(model, "built", False):
        return

    try:
        if len(model.weights) > 0:
            model.built = True
            return
    except Exception:
        pass

    raise ValueError(
        "Model is not built and has no weights. "
        "Train the model or run a forward pass before saving."
    )


# ==========================================================
# Callback metadata extraction
# ==========================================================

def extract_callback_metadata(callbacks):
    """
    Extract reproducible metadata from known GEEANNT / Keras callbacks.

    This stores both:
      - callback configuration
      - adaptive scheduler state, when available

    The goal is to make the run scientifically reproducible.
    """
    metadata = {}

    if callbacks is None:
        return metadata

    for i, cb in enumerate(callbacks):
        name = cb.__class__.__name__
        key = f"{i:02d}_{name}"

        if name == "ValidationEnergyDistributionMonitor":
            metadata[key] = {
                "class": name,
                "n_samples": getattr(cb, "n_samples", None),
                "energy_mode": getattr(cb, "energy_mode", None),
                "min_real_count": getattr(cb, "min_real_count", None),
                "seed": getattr(cb, "seed", None),
                "every_n_epochs": getattr(cb, "every_n_epochs", None),
                "verbose": getattr(cb, "verbose", None),
                "n_val_events": int(len(getattr(cb, "E_val_raw", []))),
                "n_energy_bins": int(len(getattr(cb, "energy_bins", [])) - 1),
            }

        elif name == "TaskAdaptiveLossScheduler":
            metadata[key] = {
                "class": name,
                "task_configs": _to_json_safe(getattr(cb, "task_configs", None)),
                "state": _to_json_safe(getattr(cb, "state", None)),
                "verbose": getattr(cb, "verbose", None),
            }

        elif name == "TaskAdaptiveTrainingMonitor":
            metadata[key] = {
                "class": name,
                "metrics_to_show": _to_json_safe(getattr(cb, "metrics_to_show", None)),
                "show_task_weights": getattr(cb, "show_task_weights", None),
                "show_scheduler_state": getattr(cb, "show_scheduler_state", None),
                "every_n_epochs": getattr(cb, "every_n_epochs", None),
            }

        elif name == "EarlyStopping":
            metadata[key] = {
                "class": name,
                "monitor": getattr(cb, "monitor", None),
                "patience": getattr(cb, "patience", None),
                "min_delta": float(getattr(cb, "min_delta", 0.0)),
                "restore_best_weights": getattr(cb, "restore_best_weights", None),
                "baseline": _to_json_safe(getattr(cb, "baseline", None)),
                "start_from_epoch": getattr(cb, "start_from_epoch", None),
            }

        elif name == "ReduceLROnPlateau":
            metadata[key] = {
                "class": name,
                "monitor": getattr(cb, "monitor", None),
                "factor": getattr(cb, "factor", None),
                "patience": getattr(cb, "patience", None),
                "min_delta": float(getattr(cb, "min_delta", 0.0)),
                "cooldown": getattr(cb, "cooldown", None),
                "min_lr": float(getattr(cb, "min_lr", 0.0)),
            }

        else:
            metadata[key] = {
                "class": name,
                "warning": "Callback type not explicitly supported by metadata extractor.",
            }

    return metadata


# ==========================================================
# Model metadata construction
# ==========================================================

def build_model_metadata(
    *,
    model,
    model_config,
    preprocessing_metadata,
    training_metadata=None,
    evaluation_metadata=None,
    callbacks=None,
    notes=None,
):
    """
    Collect all information needed to reproduce or reuse a trained model.
    """
    training_metadata = dict(training_metadata or {})

    if callbacks is not None:
        training_metadata["callbacks"] = extract_callback_metadata(callbacks)

    # Current task weights
    if hasattr(model, "task_weights"):
        training_metadata["final_task_weights"] = _to_json_safe(model.task_weights)

    # Optimizer configuration
    if hasattr(model, "optimizer") and model.optimizer is not None:
        try:
            training_metadata["optimizer_config"] = _to_json_safe(
                model.optimizer.get_config()
            )
        except Exception:
            training_metadata["optimizer_config"] = "Could not serialize optimizer config."

    metadata = {
        "created_at_unix": time.time(),
        "model_class": model.__class__.__name__,
        "model_built": bool(getattr(model, "built", False)),
        "model_config": _to_json_safe(model_config or {}),
        "preprocessing_metadata": _to_json_safe(preprocessing_metadata or {}),
        "training_metadata": _to_json_safe(training_metadata),
        "evaluation_metadata": _to_json_safe(evaluation_metadata or {}),
    }

    if notes is not None:
        metadata["notes"] = str(notes)

    return metadata


# ==========================================================
# History save helper
# ==========================================================

def _save_history(history, history_path):
    """
    Save Keras History object or plain history dictionary.
    """
    if history is None:
        return None

    h = history.history if hasattr(history, "history") else history

    with open(history_path, "w") as f:
        json.dump(_to_json_safe(h), f, indent=2)

    return history_path


# ==========================================================
# Task weights save helper
# ==========================================================

def _save_task_weights(model, task_weights_path):
    """
    Save task weights if the model has a task_weights attribute.
    """
    if not hasattr(model, "task_weights"):
        return None

    with open(task_weights_path, "w") as f:
        json.dump(_to_json_safe(model.task_weights), f, indent=2)

    return task_weights_path


# ==========================================================
# Lightweight checkpoint
# ==========================================================

def save_training_checkpoint(
    *,
    model,
    checkpoint_dir,
    checkpoint_name="checkpoint",
    epoch=None,
    history=None,
    model_config=None,
    preprocessing_metadata=None,
    training_metadata=None,
    evaluation_metadata=None,
    callbacks=None,
    notes=None,
):
    """
    Save a lightweight training checkpoint.

    Saves:
      - model weights
      - task weights
      - history
      - metadata
      - callback configuration/state

    This is useful during development or after a run.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    suffix = f"_epoch_{epoch:03d}" if epoch is not None else ""
    base = os.path.join(checkpoint_dir, f"{checkpoint_name}{suffix}")

    weights_path = base + ".weights.h5"
    task_weights_path = base + "_task_weights.json"
    history_path = base + "_history.json"
    metadata_path = base + "_metadata.json"

    _ensure_model_saveable(model)
    model.save_weights(weights_path)

    saved_task_weights_path = _save_task_weights(model, task_weights_path)
    saved_history_path = _save_history(history, history_path)

    metadata = build_model_metadata(
        model=model,
        model_config=model_config or {},
        preprocessing_metadata=preprocessing_metadata or {},
        training_metadata=training_metadata or {},
        evaluation_metadata=evaluation_metadata or {},
        callbacks=callbacks,
        notes=notes,
    )

    with open(metadata_path, "w") as f:
        json.dump(_to_json_safe(metadata), f, indent=2)

    return {
        "weights_path": weights_path,
        "task_weights_path": saved_task_weights_path,
        "history_path": saved_history_path,
        "metadata_path": metadata_path,
    }


# ==========================================================
# Final trained model save
# ==========================================================

def save_final_trained_model(
    *,
    model,
    save_dir,
    model_name="geeannt_model",
    history=None,
    model_config=None,
    preprocessing_metadata=None,
    training_metadata=None,
    evaluation_metadata=None,
    callbacks=None,
    notes=None,
):
    """
    Save a complete final trained model package.

    Saves:
      - final model weights
      - model config
      - full metadata
      - training history
      - task weights
      - human-readable summary

    This should be used for a model considered good enough to archive.
    """
    os.makedirs(save_dir, exist_ok=True)

    weights_path = os.path.join(save_dir, f"{model_name}.weights.h5")
    config_path = os.path.join(save_dir, f"{model_name}_config.json")
    metadata_path = os.path.join(save_dir, f"{model_name}_metadata.json")
    history_path = os.path.join(save_dir, f"{model_name}_history.json")
    task_weights_path = os.path.join(save_dir, f"{model_name}_task_weights.json")
    summary_path = os.path.join(save_dir, f"{model_name}_summary.txt")

    _ensure_model_saveable(model)
    model.save_weights(weights_path)

    with open(config_path, "w") as f:
        json.dump(_to_json_safe(model_config or {}), f, indent=2)

    saved_history_path = _save_history(history, history_path)
    saved_task_weights_path = _save_task_weights(model, task_weights_path)

    metadata = build_model_metadata(
        model=model,
        model_config=model_config or {},
        preprocessing_metadata=preprocessing_metadata or {},
        training_metadata=training_metadata or {},
        evaluation_metadata=evaluation_metadata or {},
        callbacks=callbacks,
        notes=notes,
    )

    with open(metadata_path, "w") as f:
        json.dump(_to_json_safe(metadata), f, indent=2)

    with open(summary_path, "w") as f:
        f.write("GEEANNT trained model summary\n")
        f.write("=============================\n\n")
        f.write(f"Model name: {model_name}\n")
        f.write(f"Model class: {model.__class__.__name__}\n")
        f.write(f"Created at unix: {metadata['created_at_unix']}\n\n")
        f.write(f"Weights: {weights_path}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Metadata: {metadata_path}\n")

        if saved_history_path is not None:
            f.write(f"History: {saved_history_path}\n")

        if saved_task_weights_path is not None:
            f.write(f"Task weights: {saved_task_weights_path}\n")

        if callbacks is not None:
            f.write("Callbacks: saved in metadata\n")

        if notes is not None:
            f.write("\nNotes:\n")
            f.write(str(notes))
            f.write("\n")

    print(f"Saved final trained model to: {save_dir}")

    return {
        "weights_path": weights_path,
        "config_path": config_path,
        "metadata_path": metadata_path,
        "history_path": saved_history_path,
        "task_weights_path": saved_task_weights_path,
        "summary_path": summary_path,
    }


# ==========================================================
# Metadata loader
# ==========================================================

def load_metadata(metadata_path):
    """
    Load a saved metadata JSON file.
    """
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_json(path):
    """
    Load a generic JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)