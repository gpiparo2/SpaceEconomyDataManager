#!/usr/bin/env python3
"""
burned_area_optuna.py
---------------------
This module integrates Optuna for hyperparameter optimization of the burned area segmentation model.
It defines an objective function that builds, trains, and evaluates the model using a subset of the data.
"""

import optuna
import tensorflow as tf
from .burned_area_model import build_burned_area_segmentation_model

def objective(trial, input_shapes, train_ds, val_ds, epochs=10):
    """
    Objective function for Optuna optimization of the burned area segmentation model.

    Parameters:
        trial (optuna.trial.Trial): Optuna trial object.
        input_shapes (dict): Dictionary of input shapes for the sensors.
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        epochs (int): Number of epochs to train.

    Returns:
        float: The validation loss after training.
    """

    print(input_shapes)
    # Hyperparameters for common layers
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    l2_reg = trial.suggest_float("l2_reg", 1e-4, 1e-3, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    
    # New hyperparameters for each branch filters
    s2_filters1 = trial.suggest_int("s2_filters1", 2, 8, step=2)
    s2_filters2 = trial.suggest_int("s2_filters2",2, 8, step=2)
    s1_filters1 = trial.suggest_int("s1_filters1", 2, 8, step=2)
    s1_filters2 = trial.suggest_int("s1_filters2", 2, 8, step=2)
    s3olci_filters1 = trial.suggest_int("s3olci_filters1", 2, 8, step=2)
    s3olci_filters2 = trial.suggest_int("s3olci_filters2",2, 8, step=2)
    s3slstr_filters1 = trial.suggest_int("s3slstr_filters1", 2, 8, step=2)
    s3slstr_filters2 = trial.suggest_int("s3slstr_filters2",2, 8, step=2)
    dem_filters1 = trial.suggest_int("dem_filters1",2, 8, step=2)
    dem_filters2 = trial.suggest_int("dem_filters2", 2, 8, step=2)

    # Build model with suggested hyperparameters
    model = build_burned_area_segmentation_model(
        input_shapes,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        s2_filters1=s2_filters1, s2_filters2=s2_filters2,
        s1_filters1=s1_filters1, s1_filters2=s1_filters2,
        s3olci_filters1=s3olci_filters1, s3olci_filters2=s3olci_filters2,
        s3slstr_filters1=s3slstr_filters1, s3slstr_filters2=s3slstr_filters2,
        dem_filters1=dem_filters1, dem_filters2=dem_filters2
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)
    val_loss, _ = model.evaluate(val_ds, verbose=0)
    return val_loss
