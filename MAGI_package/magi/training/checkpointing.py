"""
Checkpointing utilities for MAGI.

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
    Extract reproducible metadata from known MAGI / Keras callbacks.

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
    normalization_metadata=None,
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
        "normalization": _to_json_safe(normalization_metadata or {}),
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
    normalization_metadata=None,
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
        normalization_metadata=normalization_metadata,
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
    model_name="magi_model",
    history=None,
    model_config=None,
    preprocessing_metadata=None,
    training_metadata=None,
    evaluation_metadata=None,
    callbacks=None,
    notes=None,
    normalization_metadata=None,
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
        normalization_metadata=normalization_metadata,
    )

    with open(metadata_path, "w") as f:
        json.dump(_to_json_safe(metadata), f, indent=2)

    with open(summary_path, "w") as f:
        f.write("MAGI trained model summary\n")
        f.write("===========================\n\n")
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
    

def load_task_adaptive_model_for_generation(
    *,
    save_dir,
    model_name,
    model_config,
    energy_bins,
    u_v_bins=None,
    n_types=None,
    type_weights=None,
    radius=100.0,
    compile_model_fn=None,
    learning_rate=None,
    verbose=1,
):
    """
    Load a saved task-adaptive MAGI model for generation.

    Supports:
        - CVAE_CatEnergy_CatUV_TaskAdaptive       v0.6
        - CVAE_CatEnergy_ContGeom_TaskAdaptive   v0.7
        - CVAE_CatEnergy_ContPhi_TaskAdaptive    v0.7.2
        - CVAE_MixEnergy_ContPhi_TaskAdaptive    v0.8 (flow continuum +
          fixed-line mixture; reconstructed from model.to_generation_config()).
    """
    import os
    import json
    import numpy as np
    import tensorflow as tf

    from magi.core import (
        CVAE_CatEnergy_CatUV_TaskAdaptive,
        CVAE_CatEnergy_ContGeom_TaskAdaptive,
        CVAE_CatEnergy_ContPhi_TaskAdaptive,
        CVAE_MixEnergy_ContPhi_TaskAdaptive,
    )

    weights_path = os.path.join(save_dir, f"{model_name}.weights.h5")
    task_weights_path = os.path.join(save_dir, f"{model_name}_task_weights.json")

    n_energy_bins = len(energy_bins) - 1

    model_class = model_config.get(
        "model_class",
        model_config.get("class_name", "CVAE_CatEnergy_CatUV_TaskAdaptive"),
    )

    if model_class == "CVAE_CatEnergy_CatUV_TaskAdaptive":
        if u_v_bins is None:
            raise ValueError(
                "u_v_bins is required to load CVAE_CatEnergy_CatUV_TaskAdaptive."
            )

        n_uv_bins = len(u_v_bins) - 1

        model = CVAE_CatEnergy_CatUV_TaskAdaptive(
            n_types=int(n_types),
            n_energy_bins=int(n_energy_bins),
            n_uv_bins=int(n_uv_bins),
            uv_bin_edges=np.asarray(u_v_bins, dtype=np.float32),

            latent_dim=model_config["latent_dim"],
            hidden=tuple(model_config["hidden"]),
            beta=model_config["beta"],
            type_weights=type_weights,

            min_log_sigma=model_config.get("min_log_sigma", -6.0),
            max_log_sigma=model_config.get("max_log_sigma", 1.5),

            lambda_sigma=model_config.get("lambda_sigma", 2e-3),
            sigma_target=model_config.get("sigma_target", -2.5),
            lambda_phi=model_config.get("lambda_phi", 2e-2),
            lambda_phi_r=model_config.get("lambda_phi_r", 2e-2),

            w_energy=model_config["w_energy"],
            w_sr=model_config["w_sr"],
            w_uv=model_config["w_uv"],
            w_phi_r=model_config["w_phi_r"],
            w_phi_v=model_config["w_phi_v"],
            w_ur=model_config.get("w_ur", 0.01),
            w_xy=model_config.get("w_xy", 0.0),
            w_vxy=model_config.get("w_vxy", 0.05),

            phi_v_ang_weight=model_config.get("phi_v_ang_weight", 1.0),
            phi_v_mse_weight=model_config.get("phi_v_mse_weight", 0.6),

            sphere_R=model_config.get("sphere_R", radius),
            sample_uv_uniform_inside_bin=model_config.get(
                "sample_uv_uniform_inside_bin",
                True,
            ),
        )

        dummy_y_cont = tf.zeros((1, 5), dtype=tf.float32)
        dummy_E = tf.zeros((1,), dtype=tf.int32)
        dummy_uv = tf.zeros((1,), dtype=tf.int32)
        dummy_cond = tf.zeros((1, int(n_types)), dtype=tf.float32)

        dummy_E_onehot = tf.one_hot(
            dummy_E,
            depth=int(n_energy_bins),
            dtype=tf.float32,
        )
        dummy_uv_onehot = tf.one_hot(
            dummy_uv,
            depth=int(n_uv_bins),
            dtype=tf.float32,
        )

        dummy_encoder_input = tf.concat(
            [dummy_y_cont, dummy_E_onehot, dummy_uv_onehot, dummy_cond],
            axis=1,
        )

        _ = model.encoder(dummy_encoder_input, training=False)

        dummy_z = tf.zeros((1, model.latent_dim), dtype=tf.float32)
        _ = model.decode(dummy_z, dummy_cond)

    elif model_class == "CVAE_CatEnergy_ContGeom_TaskAdaptive":
        y_cont_dim = int(model_config.get("y_cont_dim", 6))

        model = CVAE_CatEnergy_ContGeom_TaskAdaptive(
            n_types=int(n_types),
            n_energy_bins=int(n_energy_bins),
            y_cont_dim=y_cont_dim,

            latent_dim=model_config["latent_dim"],
            hidden=tuple(model_config["hidden"]),
            beta=model_config["beta"],
            type_weights=type_weights,

            min_log_sigma=model_config.get("min_log_sigma", -6.0),
            max_log_sigma=model_config.get("max_log_sigma", 1.5),

            lambda_sigma=model_config.get("lambda_sigma", 2e-3),
            sigma_target=model_config.get("sigma_target", -2.5),
            lambda_phi=model_config.get("lambda_phi", 2e-2),
            lambda_phi_r=model_config.get("lambda_phi_r", 2e-2),

            w_energy=model_config.get("w_energy", 1.0),
            w_ur=model_config.get("w_ur", 1.0),
            w_uv=model_config.get("w_uv", 1.0),
            w_phi_r=model_config.get("w_phi_r", 1.0),
            w_phi_v=model_config.get("w_phi_v", 1.0),

            energy_sampling_temperature=model_config.get(
                "energy_sampling_temperature",
                1.0,
            ),

            phi_v_ang_weight=model_config.get("phi_v_ang_weight", 1.2),
            phi_v_mse_weight=model_config.get("phi_v_mse_weight", 0.5),

            stem_width=model_config.get("stem_width", 64),
            deep_decoder_hidden=tuple(
                model_config.get("deep_decoder_hidden", (128, 128, 64))
            ),
            energy_branch_hidden=tuple(
                model_config.get("energy_branch_hidden", (48, 48))
            ),
        )

        dummy_y_cont = tf.zeros((1, y_cont_dim), dtype=tf.float32)
        dummy_E = tf.zeros((1,), dtype=tf.int32)
        dummy_cond = tf.zeros((1, int(n_types)), dtype=tf.float32)

        dummy_E_onehot = tf.one_hot(
            dummy_E,
            depth=int(n_energy_bins),
            dtype=tf.float32,
        )

        dummy_encoder_input = tf.concat(
            [dummy_y_cont, dummy_E_onehot, dummy_cond],
            axis=1,
        )

        _ = model.encoder(dummy_encoder_input, training=False)

        dummy_z = tf.zeros((1, model.latent_dim), dtype=tf.float32)
        _ = model.decode(dummy_z, dummy_cond)

    elif model_class == "CVAE_CatEnergy_ContPhi_TaskAdaptive":
        y_cont_dim = int(model_config.get("y_cont_dim", 4))

        model = CVAE_CatEnergy_ContPhi_TaskAdaptive(
            n_types=int(n_types),
            n_energy_bins=int(n_energy_bins),
            y_cont_dim=y_cont_dim,

            latent_dim=model_config["latent_dim"],
            hidden=tuple(model_config["hidden"]),
            beta=model_config["beta"],
            type_weights=type_weights,

            min_log_sigma=model_config.get("min_log_sigma", -6.0),
            max_log_sigma=model_config.get("max_log_sigma", 1.5),

            lambda_sigma=model_config.get("lambda_sigma", 2e-3),
            sigma_target=model_config.get("sigma_target", -2.5),

            w_energy=model_config.get("w_energy", 1.0),
            w_ur=model_config.get("w_ur", 1.0),
            w_uv=model_config.get("w_uv", 1.0),
            w_phi_r=model_config.get("w_phi_r", 1.0),
            w_phi_v=model_config.get("w_phi_v", 1.0),

            energy_sampling_temperature=model_config.get(
                "energy_sampling_temperature",
                1.0,
            ),

            stem_width=model_config.get("stem_width", 64),
            deep_decoder_hidden=tuple(
                model_config.get("deep_decoder_hidden", (128, 128, 64))
            ),
            energy_branch_hidden=tuple(
                model_config.get("energy_branch_hidden", (48, 48))
            ),
        )

        dummy_y_cont = tf.zeros((1, y_cont_dim), dtype=tf.float32)
        dummy_E = tf.zeros((1,), dtype=tf.int32)
        dummy_cond = tf.zeros((1, int(n_types)), dtype=tf.float32)

        dummy_E_onehot = tf.one_hot(
            dummy_E,
            depth=int(n_energy_bins),
            dtype=tf.float32,
        )

        dummy_encoder_input = tf.concat(
            [dummy_y_cont, dummy_E_onehot, dummy_cond],
            axis=1,
        )

        _ = model.encoder(dummy_encoder_input, training=False)

        dummy_z = tf.zeros((1, model.latent_dim), dtype=tf.float32)
        _ = model.decode(dummy_z, dummy_cond)

    elif model_class == "CVAE_MixEnergy_ContPhi_TaskAdaptive":
        # v0.8 mixture energy head. Reconstruct from the config produced by
        # model.to_generation_config() (persist that as model_config at save
        # time). This class has no categorical energy head, so its encoder
        # input is [y_cont, cond] (no one-hot energy) and y_cont carries the
        # gate-target columns: y_cont_dim = 4 geometry + 1 energy_y +
        # (n_lines + 1) gate targets.
        line_positions_y = np.asarray(
            model_config["line_positions_y"], dtype=np.float32
        )
        n_lines = int(line_positions_y.shape[0])
        y_cont_dim = 4 + 1 + (n_lines + 1)

        model = CVAE_MixEnergy_ContPhi_TaskAdaptive(
            n_types=int(n_types),
            line_positions_y=line_positions_y,
            latent_dim=model_config["latent_dim"],
            hidden=tuple(model_config["hidden"]),
            beta=model_config["beta"],
            type_weights=type_weights,

            n_continuum_components=model_config.get("n_continuum_components", 1),
            min_log_sigma=model_config.get("min_log_sigma", -6.0),
            max_log_sigma=model_config.get("max_log_sigma", 1.5),
            sigma_target=model_config.get("sigma_target", -2.0),
            lambda_sigma=model_config.get("lambda_sigma", 1e-3),

            continuum_mode=model_config.get("continuum_mode", "gaussian"),
            energy_flow_condition=model_config.get("energy_flow_condition", "z_cond"),
            gate_focal_gamma=model_config.get("gate_focal_gamma", 0.0),
            gate_class_weights=model_config.get("gate_class_weights"),
            continuum_flow_bins=model_config.get("continuum_flow_bins", 8),
            continuum_flow_transforms=model_config.get("continuum_flow_transforms", 2),
            continuum_flow_interval=model_config.get("continuum_flow_interval", 5.0),
            continuum_flow_conditioner_hidden=tuple(
                model_config.get("continuum_flow_conditioner_hidden", (64,))
            ),
            continuum_flow_y_mean=model_config.get("continuum_flow_y_mean", 0.0),
            continuum_flow_y_scale=model_config.get("continuum_flow_y_scale", 1.0),
            continuum_flow_warp=model_config.get("continuum_flow_warp", "affine"),
            continuum_flow_warp_y_knots=model_config.get("continuum_flow_warp_y_knots"),
            continuum_flow_warp_z_knots=model_config.get("continuum_flow_warp_z_knots"),

            prior=model_config.get("prior", "gaussian"),
            prior_n_layers=model_config.get("prior_n_layers", 6),
            prior_hidden=tuple(model_config.get("prior_hidden", (64, 64))),
            prior_log_scale_clamp=model_config.get("prior_log_scale_clamp", 3.0),

            line_logsigma_trainable=model_config.get("line_logsigma_trainable", True),
            energy_sampling_temperature=model_config.get(
                "energy_sampling_temperature", 1.0
            ),
            stem_width=model_config.get("stem_width", 64),
            deep_decoder_hidden=tuple(
                model_config.get("deep_decoder_hidden", (128, 128, 64))
            ),
            energy_branch_hidden=tuple(
                model_config.get("energy_branch_hidden", (48, 48))
            ),
            energy_cont_head_hidden=tuple(
                model_config.get("energy_cont_head_hidden", (64, 32))
            ),
        )

        dummy_y_cont = tf.zeros((1, y_cont_dim), dtype=tf.float32)
        dummy_cond = tf.zeros((1, int(n_types)), dtype=tf.float32)

        # This head's encoder takes [y_cont, cond] (no one-hot energy column).
        dummy_encoder_input = tf.concat([dummy_y_cont, dummy_cond], axis=1)
        _ = model.encoder(dummy_encoder_input, training=False)

        dummy_z = tf.zeros((1, model.latent_dim), dtype=tf.float32)
        _ = model.decode(dummy_z, dummy_cond)

    else:
        raise ValueError(f"Unsupported model_class: {model_class}")

    model.built = True
    model.load_weights(weights_path)

    if os.path.exists(task_weights_path) and hasattr(model, "task_weights"):
        with open(task_weights_path, "r") as f:
            saved_task_weights = json.load(f)

        for k, v in saved_task_weights.items():
            if k in model.task_weights:
                model.set_task_weight(k, float(v))
            else:
                model.task_weights[k] = float(v)

    if compile_model_fn is not None:
        lr = 2e-4 if learning_rate is None else learning_rate
        compile_model_fn(model, learning_rate=lr)

    if verbose:
        print("Loaded task-adaptive model.")
        print("Model class:", model_class)
        print("Weights:", weights_path)

        if os.path.exists(task_weights_path):
            print("Task weights:", task_weights_path)

        if hasattr(model, "task_weights"):
            print("Current task weights:", model.task_weights)

    return model

