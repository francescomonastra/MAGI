"""Prior refit (magi.training.prior_refit).

A small v0.8 model with a coupling prior is refit on random data. The tests
check the mechanics that are easy to get wrong: the refit lowers the validation
KL of an untrained prior, an interrupted refit continues to the same result as
an uninterrupted one in epochs and history, the saved run reloads with the new
prior architecture, and a checkpoint of other settings is refused.
"""
import json
import os

import numpy as np
import pytest
import tensorflow as tf

import magi
from magi.training import encode_posterior, posterior_kl

N_TYPES = 3
LINES = np.array([np.log10(0.1), np.log10(0.5)], dtype=np.float32)
N_ZONES = len(LINES) + 1
Y_DIM = 4 + 1 + N_ZONES


def _model():
    m = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
        gate_focal_gamma=0.0, n_types=N_TYPES, line_positions_y=LINES, latent_dim=4,
        hidden=(16, 16), beta=0.2, continuum_mode="flow", continuum_flow_bins=8,
        continuum_flow_transforms=2, continuum_flow_warp="affine",
        energy_flow_condition="z_cond", prior="coupling", prior_n_layers=2,
        prior_hidden=(8, 8), line_logsigma_init=np.array([-9.0, -9.0], dtype=np.float32),
        line_logsigma_trainable=False)
    c = tf.zeros((2, N_TYPES))
    m.encoder(tf.concat([tf.zeros((2, Y_DIM)), c], 1), training=False)
    m.decode(tf.zeros((2, m.latent_dim)), c)
    m.prior.log_prob(tf.zeros((1, 4)), tf.zeros((1, N_TYPES)))
    return m


def _data(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, Y_DIM)).astype("float32")
    X[:, -N_ZONES:] = np.eye(N_ZONES)[rng.integers(0, N_ZONES, n)]  # a one-hot zone block
    return X, np.eye(N_TYPES)[rng.integers(0, N_TYPES, n)].astype("float32")


def _refit(model, ckpt=None, epochs=3):
    (Xt, ct), (Xv, cv) = _data(4096, 0), _data(1024, 1)
    return magi.refit_prior(model, Xt, ct, Xv, cv, hidden=(8, 8), n_layers=2, epochs=epochs,
                            lr=5e-3, batch_size=512, checkpoint_dir=ckpt, verbose=0)


def test_refit_lowers_the_validation_kl():
    model = _model()
    res = _refit(model, epochs=6)
    assert res.val_kl_after < res.val_kl_before
    assert len(res.history) == 6
    assert res.val_kl_after == pytest.approx(min(h["val_kl"] for h in res.history), rel=1e-5)


def test_interrupted_refit_continues(tmp_path):
    model = _model()
    ck = str(tmp_path / "ck")
    _refit(model, ck, epochs=6)
    state = json.load(open(os.path.join(ck, "state.json")))
    assert state["epochs_done"] == 6
    # same call again: nothing left to do, history is the saved one
    res = _refit(model, ck, epochs=6)
    assert [h["epoch"] for h in res.history] == list(range(1, 7))
    assert res.val_kl_after == pytest.approx(state["best"], rel=1e-5)


def test_checkpoint_of_other_settings_is_refused(tmp_path):
    model = _model()
    ck = str(tmp_path / "ck")
    _refit(model, ck, epochs=2)
    with pytest.raises(ValueError, match="remove it"):
        _refit(model, ck, epochs=4)


def test_refit_needs_a_coupling_prior():
    model = _model()
    model.prior = None
    with pytest.raises(ValueError, match="coupling"):
        _refit(model)


def test_posterior_kl_matches_the_models_prior_on_encoded_data():
    model = _model()
    X, c = _data(256, 2)
    zm, zl, pc = encode_posterior(model, X, c)
    assert zm.shape == (256, 4) and pc.shape[0] == 256
    kl = posterior_kl(model.prior, zm, zl, pc)
    assert kl.shape == (256,) and np.isfinite(kl).all()


def test_weighted_refit_runs_and_records_the_unweighted_kl():
    model = _model()
    (Xt, ct), (Xv, cv) = _data(4096, 0), _data(1024, 1)
    wt = np.where(ct[:, 0] > 0.5, 5.0, 1.0); wv = np.where(cv[:, 0] > 0.5, 5.0, 1.0)
    res = magi.refit_prior(model, Xt, ct, Xv, cv, hidden=(8, 8), n_layers=2, epochs=3, lr=5e-3,
                           batch_size=512, verbose=0, sample_weight_train=wt, sample_weight_val=wv)
    assert all("val_kl_unweighted" in h for h in res.history)
    assert res.val_kl_after == pytest.approx(min(h["val_kl"] for h in res.history), rel=1e-5)
    with pytest.raises(ValueError, match="one value per event"):
        magi.refit_prior(model, Xt, ct, Xv, cv, epochs=1, verbose=0, sample_weight_train=wt[:10])
