"""Resumable training (magi.training.resume).

A small dense model on an unshuffled dataset, so that training is
deterministic and a resumed run can be compared with an uninterrupted one
weight by weight. The callbacks are given short patiences so that their state
(EarlyStopping counter, ReduceLROnPlateau learning rate) is actually exercised.
"""
import os
import shutil

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

import magi
from magi.training import find_checkpoint, load_resume_state


def _data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(512, 6)).astype("float32")
    y = (x[:, :1] * 0.7 - x[:, 1:2] ** 2 + 0.1 * rng.normal(size=(512, 1))).astype("float32")
    ds = lambda a, b: tf.data.Dataset.from_tensor_slices((a, b)).batch(64)  # no shuffle
    return ds(x[:384], y[:384]), ds(x[384:], y[384:])


def _model():
    keras.utils.set_random_seed(3)
    m = keras.Sequential([keras.layers.Input((6,)), keras.layers.Dense(16, activation="tanh"),
                          keras.layers.Dense(1)])
    magi.compile_model(m, learning_rate=5e-3, clipnorm=5.0)
    m.compile(optimizer=m.optimizer, loss="mse")  # plain Keras model: give it a loss
    return m


def _callbacks(patience=50):
    return [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                          restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1,
                                              min_lr=1e-6, verbose=0)]


def _fit(model, ckpt, epochs, resume=False, **kw):
    train, val = _data()
    return magi.fit_model(model, train, val, epochs=epochs, callbacks=_callbacks(**kw.pop("cb", {})),
                          verbose=0, checkpoint_dir=ckpt, resume=resume, **kw)


def test_resumed_run_equals_uninterrupted_run(tmp_path):
    full = _model()
    h_full = _fit(full, str(tmp_path / "full"), epochs=10)

    part = _model()
    _fit(part, str(tmp_path / "part"), epochs=4)
    assert load_resume_state(str(tmp_path / "part"))["epochs_completed"] == 4

    resumed = _model()  # a fresh process: new model, new optimizer, new callbacks
    h_res = _fit(resumed, str(tmp_path / "part"), epochs=10, resume=True)

    assert h_res.epoch == list(range(10))
    for k in ("loss", "val_loss"):
        np.testing.assert_allclose(h_res.history[k], h_full.history[k], rtol=1e-5, atol=1e-7)
    for a, b in zip(resumed.get_weights(), full.get_weights()):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    assert float(resumed.optimizer.learning_rate) == pytest.approx(float(full.optimizer.learning_rate))
    assert int(resumed.optimizer.iterations) == int(full.optimizer.iterations)


def test_resume_without_checkpoint_starts_from_scratch(tmp_path):
    h = _fit(_model(), str(tmp_path / "new"), epochs=3, resume=True)
    assert len(h.history["loss"]) == 3


def test_resume_needs_a_directory():
    train, val = _data()
    with pytest.raises(ValueError, match="checkpoint_dir"):
        magi.fit_model(_model(), train, val, epochs=1, verbose=0, resume=True)


def test_finished_run_is_not_retrained(tmp_path):
    d = str(tmp_path / "done")
    m1 = _model()
    _fit(m1, d, epochs=3)
    m2 = _model()
    h = _fit(m2, d, epochs=3, resume=True)
    assert len(h.history["loss"]) == 3
    for a, b in zip(m2.get_weights(), m1.get_weights()):  # same weights come back
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)


def test_early_stopped_run_can_be_carried_on(tmp_path):
    d = str(tmp_path / "es")
    h = _fit(_model(), d, epochs=200, cb={"patience": 1})
    n_stop = len(h.history["loss"])
    saved = load_resume_state(d)["callbacks"]["early_stopping"]
    assert n_stop < 200 and load_resume_state(d)["early_stopped"]
    assert saved["stopped_epoch"] > 0 and saved["wait"] >= 1
    shutil.copytree(d, str(tmp_path / "es_reset"))

    # epochs == epochs done: nothing is trained, the callback state just comes back.
    train, val = _data()
    cbs = _callbacks(1)
    magi.fit_model(_model(), train, val, epochs=n_stop, callbacks=cbs, verbose=0,
                   checkpoint_dir=d, resume=True)
    es = cbs[0]
    assert (es.wait, es.stopped_epoch) == (saved["wait"], saved["stopped_epoch"])
    assert es.best == pytest.approx(saved["best"]) and es.best_weights is not None

    # With the reset the counter is cleared, the best value kept, and the run carries on.
    cbs = _callbacks(100)
    h3 = magi.fit_model(_model(), train, val, epochs=n_stop + 20, callbacks=cbs, verbose=0,
                        checkpoint_dir=str(tmp_path / "es_reset"), resume=True,
                        reset_early_stopping=True)
    assert len(h3.history["loss"]) == n_stop + 20
    assert min(h3.history["val_loss"]) <= saved["best"] + 1e-9


def test_interrupted_write_keeps_the_previous_checkpoint(tmp_path):
    d = str(tmp_path / "swap")
    _fit(_model(), d, epochs=3)
    cur = os.path.join(d, "ckpt")
    # Simulate a kill between the two renames of a swap: only ckpt_old is left.
    os.replace(cur, os.path.join(d, "ckpt_old"))
    os.makedirs(os.path.join(d, "ckpt_new"))  # half-written slot, no state.json
    assert find_checkpoint(d).endswith("ckpt_old")
    assert load_resume_state(d)["epochs_completed"] == 3


def test_wrong_optimizer_is_rejected(tmp_path):
    d = str(tmp_path / "opt")
    _fit(_model(), d, epochs=2)
    m = _model()
    m.compile(optimizer=keras.optimizers.SGD(1e-2, momentum=0.9), loss="mse")
    train, val = _data()
    with pytest.raises(ValueError, match="optimizer"):
        magi.fit_model(m, train, val, epochs=4, callbacks=_callbacks(), verbose=0,
                       checkpoint_dir=d, resume=True)
