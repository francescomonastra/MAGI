"""HENABICS primary truth: the conditioning vector widens beyond the
crossing-type one-hot to [onehot(crossing type), onehot(primary type),
qt(primary energy)] (docs/MAGI_HENABICS_integration_requirements.md).

Covers what is easy to get wrong: every sub-network that reads cond gets the
new width, the per-type lookups (type_weights, zone_probs) still read only the
first n_types columns, the width survives a save/load round trip, and a
checkpoint saved before cond_dim existed rebuilds the original architecture.
"""
import json
import os

import numpy as np
import pytest
import tensorflow as tf

import magi

N_TYPES = 3
N_PRIMARY_TYPES = 2
COND_DIM = N_TYPES + N_PRIMARY_TYPES + 1  # + qt(log10 E_prim)
LINE_POSITIONS_Y = np.array([np.log10(0.1), np.log10(0.5)], dtype=np.float32)
N_ZONES = LINE_POSITIONS_Y.shape[0] + 1
Y_CONT_DIM = 4 + 1 + N_ZONES

OLD_CKPT = os.path.join(
    os.path.dirname(__file__), "..", "..", "trained_models", "v0_8_2_CR_clipnorm5"
)


def _zone_probs():
    return np.array([
        [0.9, 0.08, 0.02],
        [0.5, 0.30, 0.20],
        [0.0, 0.00, 1.00],
    ], dtype=np.float32)


def _build_model(**kwargs):
    model = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
        gate_focal_gamma=0.0,
        n_types=N_TYPES, line_positions_y=LINE_POSITIONS_Y, latent_dim=4,
        hidden=(16, 16), beta=0.2, continuum_mode="flow",
        continuum_flow_bins=8, continuum_flow_transforms=2,
        continuum_flow_warp="affine",
        energy_flow_condition="z_cond", prior="coupling",
        prior_n_layers=2, prior_hidden=(8, 8),
        line_logsigma_init=np.array([-9.0, -9.0], dtype=np.float32),
        line_logsigma_trainable=False,
        **kwargs,
    )
    dummy_cond = tf.zeros((2, model.cond_dim), dtype=tf.float32)
    _ = model.encoder(
        tf.concat([tf.zeros((2, Y_CONT_DIM), dtype=tf.float32), dummy_cond], axis=1),
        training=False,
    )
    _ = model.decode(tf.zeros((2, model.latent_dim), dtype=tf.float32), dummy_cond)
    return model


def _wide_cond(type_idx, rng):
    n = len(type_idx)
    prim_idx = rng.integers(0, N_PRIMARY_TYPES, size=n)
    qt_e = rng.normal(size=(n, 1)).astype(np.float32)
    return tf.constant(np.concatenate([
        np.eye(N_TYPES, dtype=np.float32)[type_idx],
        np.eye(N_PRIMARY_TYPES, dtype=np.float32)[prim_idx],
        qt_e,
    ], axis=1))


def test_default_cond_dim_is_n_types():
    model = _build_model()
    assert model.cond_dim == N_TYPES
    assert model.prior.cond_dim == N_TYPES
    assert model.to_generation_config()["cond_dim"] == N_TYPES


def test_cond_dim_below_n_types_raises():
    with pytest.raises(ValueError, match="cond_dim"):
        _build_model(cond_dim=N_TYPES - 1)


def test_wide_cond_reaches_every_sub_network():
    model = _build_model(
        cond_dim=COND_DIM, prior_zone_conditioning=True, zone_probs=_zone_probs()
    )
    assert model.encoder.input_shape[-1] == Y_CONT_DIM + COND_DIM
    assert model.prior.cond_dim == COND_DIM + N_ZONES


def test_train_step_and_generate_with_wide_cond():
    """type_weights and zone_probs are indexed by crossing type; with the
    extra columns appended a plain argmax(cond) could land on a primary-type
    or energy column and gather out of range."""
    model = _build_model(
        cond_dim=COND_DIM, prior_zone_conditioning=True, zone_probs=_zone_probs(),
        type_weights=np.array([1.0, 2.0, 0.5], dtype=np.float32),
    )
    rng = np.random.default_rng(0)
    n = 64
    type_idx = rng.integers(0, N_TYPES, size=n)
    cond = _wide_cond(type_idx, rng)
    # Large energy column: argmax over the full row would pick it.
    cond = tf.concat([cond[:, :-1], tf.fill((n, 1), 50.0)], axis=1)
    y_cont = tf.constant(rng.normal(size=(n, Y_CONT_DIM)).astype(np.float32))

    magi.compile_model(model, learning_rate=1e-3)
    logs = model.train_step(((y_cont, tf.zeros((n,), tf.int32), cond),
                             tf.zeros((n,), tf.float32)))
    assert np.isfinite(float(logs["loss"]))

    out = model.generate(cond, n)
    assert out["y_cont"].shape == (n, 4)
    assert np.all(np.isfinite(out["energy_y"].numpy()))


def test_wide_cond_round_trips_through_checkpoint(tmp_path):
    model = _build_model(cond_dim=COND_DIM)
    save_dir = str(tmp_path / "ckpt")
    magi.save_final_trained_model(
        model=model, save_dir=save_dir, model_name="mix_prim",
        model_config=model.to_generation_config(),
    )
    with open(f"{save_dir}/mix_prim_config.json") as f:
        config = json.load(f)
    assert config["cond_dim"] == COND_DIM

    reloaded = magi.load_task_adaptive_model_for_generation(
        save_dir=save_dir, model_name="mix_prim", model_config=config,
        n_types=N_TYPES, radius=100.0, verbose=0,
    )
    assert reloaded.cond_dim == COND_DIM
    for a, b in zip(model.encoder.get_weights(), reloaded.encoder.get_weights()):
        np.testing.assert_array_equal(a, b)


def test_generate_with_no_lines():
    """A small training set can leave no line above the count cut; generation
    must then sample the continuum only instead of gathering from an empty
    line axis."""
    model = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
        n_types=N_TYPES, line_positions_y=np.zeros((0,), dtype=np.float32),
        latent_dim=4, hidden=(16, 16), continuum_mode="flow",
        continuum_flow_bins=8, continuum_flow_transforms=2,
        prior="coupling", prior_n_layers=2, prior_hidden=(8, 8),
        cond_dim=COND_DIM,
    )
    cond = _wide_cond(np.zeros(16, dtype=np.int64), np.random.default_rng(1))
    out = model.generate(cond, 16)
    assert np.all(out["energy_component_idx"].numpy() == 0)
    assert np.all(np.isfinite(out["energy_y"].numpy()))


@pytest.mark.skipif(not os.path.isdir(OLD_CKPT), reason="v0_8_2_CR_clipnorm5 not present")
def test_pre_cond_dim_checkpoint_still_loads():
    with open(os.path.join(OLD_CKPT, "mix_CR_config.json")) as f:
        config = json.load(f)
    assert "cond_dim" not in config
    model = magi.load_task_adaptive_model_for_generation(
        save_dir=OLD_CKPT, model_name="mix_CR", model_config=config,
        n_types=config["n_types"], radius=100.0, verbose=0,
    )
    assert model.cond_dim == config["n_types"]
    cond = tf.one_hot(np.zeros(16, dtype=np.int32), depth=model.n_types)
    out = model.generate(cond, 16)
    assert np.all(np.isfinite(out["energy_y"].numpy()))
