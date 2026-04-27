"""
Training utilities for GEEANNT.
"""

from tensorflow import keras
import tensorflow as tf
import numpy as np


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


def compile_model(model, learning_rate=2e-4):
    """
    Compile the model with the default Adam optimizer.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate)
    )
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
):
    """
    Convenience wrapper for compile + fit.
    """
    compile_model(model, learning_rate=learning_rate)

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
