"""
Resumable training for MAGI.

A long fit can stop before it is finished: the process is killed, the machine
sleeps, a Colab session ends, or EarlyStopping fires a few epochs too soon.
`ResumeCheckpoint` writes, after every epoch, everything that a continuation
needs, and `fit_model(..., checkpoint_dir=..., resume=True)` picks the run up
from it:

    model weights            the current ones, plus the best ones held by
                             EarlyStopping(restore_best_weights=True)
    optimizer state          Adam moments and step count, and the learning rate
    callback state           EarlyStopping (best, wait), ReduceLROnPlateau
                             (best, wait, cooldown), TaskAdaptiveLossScheduler
                             (per-task plateau counters)
    task weights             model.task_weights, which the scheduler decays
    history                  every logged metric of the epochs already done

The continuation is exact for all of these. The order in which the shuffled
training data are drawn is not restored, so a resumed run is statistically
equivalent to an uninterrupted one but not bit-identical to it (with an
unshuffled dataset it is bit-identical, which is what the unit test checks).

The checkpoint is written to a temporary directory and swapped in, so a run
killed in the middle of a write leaves the previous epoch's checkpoint intact.

Typical use:

    history = magi.fit_model(model, train_ds, val_ds, epochs=40,
                             callbacks=magi.build_default_callbacks(),
                             checkpoint_dir="trained_models/my_run/resume",
                             resume=True)

Running the same line again after an interruption continues from the last
completed epoch; `epochs` is the total, as in Keras. To recover a run that
early stopping ended too soon, raise `epochs` (and, if wanted, `patience` on
the EarlyStopping instance) and pass `reset_early_stopping=True`, which clears
the patience counter and keeps the best value found so far.
"""

import json
import os
import shutil

import numpy as np
from tensorflow import keras

STATE_FILE = "state.json"
ARRAYS_FILE = "arrays.npz"
FORMAT_VERSION = 1


def _to_py(x):
    """JSON-safe Python scalar or list from numpy / tensor values."""
    if isinstance(x, dict):
        return {str(k): _to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_py(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return float(keras.ops.convert_to_numpy(x))


def _slot_paths(directory):
    return (os.path.join(directory, "ckpt"),
            os.path.join(directory, "ckpt_new"),
            os.path.join(directory, "ckpt_old"))


def find_checkpoint(directory):
    """
    Find the newest complete resume checkpoint in a directory.

    Falls back to the previous slot if the process died between the two
    renames of a swap, so an interrupted write never hides the last good
    epoch.

    Parameters
    ----------
    directory : str
        The `checkpoint_dir` given to fit_model.

    Returns
    -------
    str or None
        Path of the checkpoint slot, or None if the directory holds none.
    """
    cur, _, old = _slot_paths(directory)
    for path in (cur, old):
        if os.path.exists(os.path.join(path, STATE_FILE)):
            return path
    return None


def load_resume_state(directory):
    """
    Read the JSON state of the newest resume checkpoint.

    Useful to see how far an interrupted run got without loading a model:
    the state holds `epochs_completed`, `early_stopped`, the learning rate,
    the task weights, the callback state and the metric history.

    Parameters
    ----------
    directory : str
        The `checkpoint_dir` given to fit_model.

    Returns
    -------
    dict or None
        The saved state, or None if the directory holds no checkpoint.
    """
    path = find_checkpoint(directory)
    if path is None:
        return None
    with open(os.path.join(path, STATE_FILE)) as f:
        return json.load(f)


def _find(callbacks, cls):
    for cb in callbacks:
        if isinstance(cb, cls):
            return cb
    return None


class ResumeCheckpoint(keras.callbacks.Callback):
    """Write a resumable checkpoint after every epoch and, if asked, restore
    one at the start of `fit`.

    Put it last in the callback list (fit_model does this), so that the other
    callbacks have already updated their state when it saves and have already
    reset it when it restores. `callbacks` is the list of the *other*
    callbacks, whose state is saved.
    """

    def __init__(self, directory, callbacks=(), resume=False,
                 reset_early_stopping=False, verbose=1):
        super().__init__()
        self.directory = directory
        self.others = [cb for cb in callbacks if cb is not self]
        self.resume = bool(resume)
        self.reset_early_stopping = bool(reset_early_stopping)
        self.verbose = verbose
        self.history = {}
        self.initial_epoch = 0
        self._restored = False
        if self.resume:
            state = load_resume_state(directory)
            if state is not None:
                self.initial_epoch = int(state["epochs_completed"])
                self.history = {k: list(v) for k, v in state["history"].items()}

    # -- what is saved -----------------------------------------------------

    def _callback_state(self):
        from .adaptive_callbacks import TaskAdaptiveLossScheduler

        out, arrays = {}, {}
        es = _find(self.others, keras.callbacks.EarlyStopping)
        if es is not None:
            out["early_stopping"] = {
                "best": _to_py(es.best), "wait": int(es.wait),
                "stopped_epoch": int(es.stopped_epoch),
                "best_epoch": int(getattr(es, "best_epoch", 0)),
                "has_best_weights": es.best_weights is not None,
            }
            if es.best_weights is not None:
                for i, w in enumerate(es.best_weights):
                    arrays[f"best_weights_{i}"] = np.asarray(w)
        lr = _find(self.others, keras.callbacks.ReduceLROnPlateau)
        if lr is not None:
            out["reduce_lr"] = {"best": _to_py(lr.best), "wait": int(lr.wait),
                                "cooldown_counter": int(lr.cooldown_counter)}
        sched = _find(self.others, TaskAdaptiveLossScheduler)
        if sched is not None:
            out["task_scheduler"] = _to_py(sched.state)
        return out, arrays

    def _save(self, epochs_completed):
        cur, new, old = _slot_paths(self.directory)
        os.makedirs(self.directory, exist_ok=True)
        shutil.rmtree(new, ignore_errors=True)
        os.makedirs(new)

        cb_state, arrays = self._callback_state()
        for i, w in enumerate(self.model.get_weights()):
            arrays[f"weights_{i}"] = np.asarray(w)
        opt = self.model.optimizer
        for i, v in enumerate(opt.variables):
            arrays[f"opt_{i}"] = keras.ops.convert_to_numpy(v)
        np.savez(os.path.join(new, ARRAYS_FILE), **arrays)

        es = cb_state.get("early_stopping", {})
        state = {
            "format_version": FORMAT_VERSION,
            "epochs_completed": int(epochs_completed),
            "early_stopped": bool(es.get("stopped_epoch", 0) > 0),
            "history": _to_py(self.history),
            "learning_rate": _to_py(opt.learning_rate),
            "n_weights": len(self.model.get_weights()),
            "n_opt_variables": len(opt.variables),
            "task_weights": (
                {k: _to_py(v) for k, v in self.model.task_weights.items()}
                if hasattr(self.model, "task_weights") else {}),
            "callbacks": cb_state,
        }
        with open(os.path.join(new, STATE_FILE), "w") as f:
            json.dump(state, f)

        shutil.rmtree(old, ignore_errors=True)
        if os.path.exists(cur):
            os.replace(cur, old)
        os.replace(new, cur)
        shutil.rmtree(old, ignore_errors=True)

    # -- restore -----------------------------------------------------------

    def _restore(self):
        from .adaptive_callbacks import TaskAdaptiveLossScheduler

        path = find_checkpoint(self.directory)
        with open(os.path.join(path, STATE_FILE)) as f:
            state = json.load(f)
        arrays = np.load(os.path.join(path, ARRAYS_FILE))

        weights = self.model.get_weights()
        if len(weights) != state["n_weights"]:
            raise ValueError(
                f"{path} holds {state['n_weights']} weight arrays but the model has "
                f"{len(weights)}: it was saved from a different architecture.")
        self.model.set_weights([arrays[f"weights_{i}"] for i in range(len(weights))])

        opt = self.model.optimizer
        if not opt.built:
            # Keras creates the Adam moments at the first update, after
            # on_train_begin; create them now so their saved values can go in.
            opt.build(self.model.trainable_variables)
        if len(opt.variables) != state["n_opt_variables"]:
            raise ValueError(
                f"{path} holds {state['n_opt_variables']} optimizer variables but the "
                f"optimizer has {len(opt.variables)}: compile with the same optimizer "
                "and clipnorm as the interrupted run.")
        for i, v in enumerate(opt.variables):
            v.assign(arrays[f"opt_{i}"])
        opt.learning_rate.assign(state["learning_rate"])

        if hasattr(self.model, "task_weights"):
            for k, v in state["task_weights"].items():
                if k in self.model.task_weights:
                    self.model.set_task_weight(k, float(v))

        cbs = state["callbacks"]
        es = _find(self.others, keras.callbacks.EarlyStopping)
        if es is not None and "early_stopping" in cbs:
            s = cbs["early_stopping"]
            es.best, es.best_epoch = s["best"], s["best_epoch"]
            es.wait, es.stopped_epoch = s["wait"], s["stopped_epoch"]
            if s["has_best_weights"]:
                es.best_weights = [arrays[f"best_weights_{i}"] for i in range(len(weights))]
            if self.reset_early_stopping:
                es.wait, es.stopped_epoch = 0, 0
        lr = _find(self.others, keras.callbacks.ReduceLROnPlateau)
        if lr is not None and "reduce_lr" in cbs:
            s = cbs["reduce_lr"]
            lr.best, lr.wait, lr.cooldown_counter = s["best"], s["wait"], s["cooldown_counter"]
        sched = _find(self.others, TaskAdaptiveLossScheduler)
        if sched is not None and "task_scheduler" in cbs:
            sched.state = cbs["task_scheduler"]
        self._restored = True
        if self.verbose:
            print(f"[MAGI] resumed from {path}: {state['epochs_completed']} epochs done, "
                  f"lr {state['learning_rate']:.3g}"
                  + (", early-stopping counter reset" if self.reset_early_stopping else ""),
                  flush=True)

    # -- Keras hooks -------------------------------------------------------

    def on_train_begin(self, logs=None):
        if self.resume and self.initial_epoch > 0 and not self._restored:
            self._restore()

    def on_epoch_end(self, epoch, logs=None):
        for k, v in (logs or {}).items():
            self.history.setdefault(k, []).append(_to_py(v))
        self._save(epoch + 1)

    def on_train_end(self, logs=None):
        # EarlyStopping puts the best weights back here, after the last
        # on_epoch_end: save again so that the checkpoint of a finished run
        # holds the weights the run ended with.
        n = len(next(iter(self.history.values()), []))
        if n:
            self._save(n)
