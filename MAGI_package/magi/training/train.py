"""
Training utilities for MAGI.
Compatible with:
  - v0.6 discrete-u_v models
  - v0.7 continuous-geometry cos/sin phi models
  - v0.7.2 continuous-geometry quantile-phi models
  - v0.8 mixture-energy models

The helpers here are head-agnostic: each model computes its own loss inside
train_step, so compiling and fitting looks the same for every version.
"""

from tensorflow import keras


def build_default_callbacks(
    monitor="val_loss",
    early_patience=8,
    lr_patience=6,
    factor=0.5,
    min_lr=1e-5,
    verbose=1,
):
    """
    Build the default callback list used during training.

    Compatible with both:
      - v0.6 discrete-u_v models
      - v0.7 continuous-geometry models

    Returns EarlyStopping (restoring the best weights) and
    ReduceLROnPlateau. The task-adaptive callbacks in
    magi.training.adaptive_callbacks are NOT included - add them yourself
    when training a TaskAdaptive head.

    Parameters
    ----------
    monitor : str
        Metric the callbacks watch. "val_loss" by default.

    early_patience : int
        Epochs without improvement before EarlyStopping fires. Should stay
        larger than `lr_patience`, so a learning-rate drop gets a chance to
        help before the run is abandoned.

    lr_patience : int
        Epochs without improvement before the learning rate is reduced.

    factor : float
        Multiplier applied to the learning rate on each reduction.

    min_lr : float
        Floor on the learning rate.

    verbose : int
        Keras verbosity for both callbacks.

    Returns
    -------
    list[keras.callbacks.Callback]
        Ready to pass as `callbacks` to fit_model or train_single_run.
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=early_patience,
            restore_best_weights=True,
            verbose=verbose,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=lr_patience,
            min_lr=min_lr,
            verbose=verbose,
        ),
    ]


def compile_model(
    model,
    learning_rate=2e-4,
    optimizer="adam",
    clipnorm=None,
):
    """
    Compile the model.

    Parameters
    ----------
    model : keras.Model
        MAGI model instance.

    learning_rate : float
        Optimizer learning rate.

    optimizer : str
        Supported:
          - "adam"
          - "adamw"

    clipnorm : float or None
        Optional gradient clipping norm. Worth setting for the v0.8 mixture
        head, whose flow and line terms can produce occasional large
        gradients early in training.

    Returns
    -------
    keras.Model
        The same `model`, compiled in place and returned for chaining.
    """
    optimizer = str(optimizer).lower()

    if optimizer == "adam":
        opt = keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=clipnorm,
        )

    elif optimizer == "adamw":
        opt = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            clipnorm=clipnorm,
        )

    else:
        raise ValueError("optimizer must be one of: 'adam', 'adamw'")

    model.compile(optimizer=opt)

    return model


def fit_model(
    model,
    train_ds,
    val_ds,
    epochs=60,
    callbacks=None,
    verbose=1,
):
    """
    Fit the model and return the Keras history object.

    The model must already be compiled - use train_single_run if you want
    compile and fit in one call.

    Parameters
    ----------
    model : keras.Model
        A compiled MAGI model.

    train_ds, val_ds : tf.data.Dataset
        Batched datasets from build_tf_datasets, i.e.
        `dataset_tf_pack["train_ds"]` and `["val_ds"]`.

    epochs : int
        Maximum epochs. EarlyStopping usually stops sooner.

    callbacks : list or None
        Callbacks to use. None means Keras defaults only - pass
        build_default_callbacks() explicitly to get early stopping and
        learning-rate reduction.

    verbose : int
        Keras verbosity. Use 2 for one line per epoch in a log file.

    Returns
    -------
    keras.callbacks.History
        Pass `history.history` to save_final_trained_model and plot_history.
    """
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )

    return history


def train_single_run(
    model,
    train_ds,
    val_ds,
    learning_rate=2e-4,
    epochs=60,
    callbacks=None,
    verbose=1,
    optimizer="adam",
    clipnorm=None,
):
    """
    Convenience wrapper for compile + fit.

    Parameters
    ----------
    model : keras.Model
        An uncompiled MAGI model.

    train_ds, val_ds : tf.data.Dataset
        Batched datasets from build_tf_datasets.

    learning_rate : float
        Optimizer learning rate, passed to compile_model.

    epochs : int
        Maximum epochs.

    callbacks : list or None
        Callbacks to use; None means Keras defaults only.

    verbose : int
        Keras verbosity.

    optimizer : str
        "adam" or "adamw", passed to compile_model.

    clipnorm : float or None
        Gradient clipping norm, passed to compile_model.

    Returns
    -------
    keras.callbacks.History
        The fit history, as from fit_model.
    """
    compile_model(
        model,
        learning_rate=learning_rate,
        optimizer=optimizer,
        clipnorm=clipnorm,
    )

    if callbacks is None:
        callbacks = build_default_callbacks()

    history = fit_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )

    return history