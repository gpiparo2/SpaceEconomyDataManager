#!/usr/bin/env python3
"""
vineyard_optuna.py
------------------
This module integrates Optuna for hyperparameter optimization of the vineyard classification model.
It defines an objective function that builds, trains, and evaluates the model using a subset of the data.
New hyperparameters allow optimization of the number of filters for both Sentinel-2 and Sentinel-1 branches.
"""

import optuna
import tensorflow as tf
from .vineyard_model import build_vineyard_classification_model

def objective(trial, input_shapes, num_classes, train_ds, val_ds, epochs=10):
    """
    Objective function for Optuna optimization of the vineyard classification model.

    Parameters:
        trial (optuna.trial.Trial): Optuna trial object.
        input_shapes (dict): Dictionary of input shapes for the sensors.
        num_classes (int): Number of classes for classification.
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        epochs (int): Number of epochs to train.

    Returns:
        float: The validation loss after training.
    """
    # Hyperparameters for common layers
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    l2_reg = trial.suggest_float("l2_reg", 1e-4, 1e-3, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    
    # New hyperparameters for each branch filters (for Sentinel-2 and Sentinel-1)
    s2_filters1 = trial.suggest_int("s2_filters1", 16, 64, step=16)
    s2_filters2 = trial.suggest_int("s2_filters2", 32, 128, step=32)
    s1_filters1 = trial.suggest_int("s1_filters1", 16, 64, step=16)
    s1_filters2 = trial.suggest_int("s1_filters2", 32, 128, step=32)

    model = build_vineyard_classification_model(
        input_shapes,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        s2_filters1=s2_filters1, s2_filters2=s2_filters2,
        s1_filters1=s1_filters1, s1_filters2=s1_filters2
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if num_classes == 1:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)
    val_loss, _ = model.evaluate(val_ds, verbose=0)
    return val_loss
