"""Checkpoint save/load config-match guard (v0.8.1 Phase 4). A tiny
CVAE_MixEnergy_ContPhi_TaskAdaptive round-trips through
save_final_trained_model -> load_task_adaptive_model_for_generation, and a
config missing a required architecture key must raise instead of silently
rebuilding a different architecture (the old model_config.get(key, <default>)
behavior - see the CAVEAT this replaced in
training/checkpointing.load_task_adaptive_model_for_generation).
"""
import json

import numpy as np
import pytest
import tensorflow as tf

import magi
from magi.training.checkpointing import _V08_MIXTURE_REQUIRED_KEYS

N_TYPES = 2
LINE_POSITIONS_Y = np.array([np.log10(0.511)], dtype=np.float32)


def _build_tiny_model():
    model = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
        n_types=N_TYPES, line_positions_y=LINE_POSITIONS_Y, latent_dim=4,
        hidden=(16, 16), beta=0.2, continuum_mode="flow",
        continuum_flow_bins=8, continuum_flow_transforms=2,
        continuum_flow_warp="affine",
        energy_flow_condition="z_cond", prior="coupling",
        prior_n_layers=2, prior_hidden=(8, 8),
        line_logsigma_init=np.array([-9.0], dtype=np.float32),
        line_logsigma_trainable=False,
    )
    n_lines = LINE_POSITIONS_Y.shape[0]
    y_cont_dim = 4 + 1 + (n_lines + 1)
    dummy_y_cont = tf.zeros((2, y_cont_dim), dtype=tf.float32)
    dummy_cond = tf.zeros((2, N_TYPES), dtype=tf.float32)
    _ = model.encoder(tf.concat([dummy_y_cont, dummy_cond], axis=1), training=False)
    _ = model.decode(tf.zeros((2, model.latent_dim), dtype=tf.float32), dummy_cond)
    return model


def test_to_generation_config_has_every_required_key_and_a_version_stamp():
    model = _build_tiny_model()
    config = model.to_generation_config()
    assert config.get("config_version") == 2
    missing = [k for k in _V08_MIXTURE_REQUIRED_KEYS if k not in config]
    assert not missing


def test_save_then_load_round_trips(tmp_path):
    model = _build_tiny_model()
    save_dir = str(tmp_path / "ckpt")
    magi.save_final_trained_model(
        model=model, save_dir=save_dir, model_name="mix_tiny",
        model_config=model.to_generation_config(),
    )
    with open(f"{save_dir}/mix_tiny_config.json") as f:
        saved_config = json.load(f)

    reloaded = magi.load_task_adaptive_model_for_generation(
        save_dir=save_dir, model_name="mix_tiny", model_config=saved_config,
        n_types=N_TYPES, radius=100.0, verbose=0,
    )
    assert reloaded.continuum_mode == model.continuum_mode
    assert reloaded.prior_mode == model.prior_mode
    np.testing.assert_allclose(
        reloaded.line_positions_y.numpy(), model.line_positions_y.numpy()
    )


@pytest.mark.parametrize("missing_key", [
    "continuum_mode", "prior", "continuum_flow_warp", "energy_flow_condition",
])
def test_missing_required_key_raises_instead_of_silently_defaulting(tmp_path, missing_key):
    model = _build_tiny_model()
    save_dir = str(tmp_path / "ckpt")
    magi.save_final_trained_model(
        model=model, save_dir=save_dir, model_name="mix_tiny",
        model_config=model.to_generation_config(),
    )
    with open(f"{save_dir}/mix_tiny_config.json") as f:
        config = json.load(f)
    del config[missing_key]

    with pytest.raises(ValueError, match=missing_key):
        magi.load_task_adaptive_model_for_generation(
            save_dir=save_dir, model_name="mix_tiny", model_config=config,
            n_types=N_TYPES, radius=100.0, verbose=0,
        )
