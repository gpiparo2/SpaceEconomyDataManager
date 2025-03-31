#!/usr/bin/env python3
"""
vineyard_model.py
-----------------
This module defines the machine learning model for vineyard classification.
It builds a multi-sensor classification model that processes inputs from various satellites
and outputs a classification score (binary or multiclass).
Now, additional hyperparameters allow choosing the number of filters for each branch.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers

def build_vineyard_classification_model(input_shapes, num_classes=1, dropout_rate=0.2, l2_reg=5e-4,
                                        s2_filters1=32, s2_filters2=64,
                                        s1_filters1=32, s1_filters2=64):
    """
    Builds a classification model with multi-sensor inputs for vineyard analysis.

    Parameters:
      input_shapes (dict): Dictionary with sensor names as keys and input shapes as values.
                           For example, {"Sentinel-2": (3, 256, 256, 17), "Sentinel-1": (3, 256, 256, 2)}.
      num_classes (int): Number of classes for classification (default is 1 for binary classification).
      dropout_rate (float): Dropout rate.
      l2_reg (float): L2 regularization factor.
      s2_filters1 (int): Number of filters for the first Conv2D layer in Sentinel-2 branch.
      s2_filters2 (int): Number of filters for the second Conv2D layer in Sentinel-2 branch.
      s1_filters1 (int): Number of filters for the first Conv2D layer in Sentinel-1 branch.
      s1_filters2 (int): Number of filters for the second Conv2D layer in Sentinel-1 branch.

    Returns:
      tf.keras.Model: Compiled classification model.
    """
    inputs = {}
    branch_outputs = {}

    # Sentinel-2 branch
    inp_s2 = Input(shape=input_shapes["Sentinel-2"], name="Sentinel-2")
    inputs["Sentinel-2"] = inp_s2
    # Process temporal dimension with TimeDistributed layers
    x = layers.TimeDistributed(layers.Conv2D(filters=s2_filters1, kernel_size=(3, 3), activation='relu',
                                             padding='same', kernel_regularizer=regularizers.l2(l2_reg)))(inp_s2)
    x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(x)
    x = layers.TimeDistributed(layers.Dropout(dropout_rate))(x)
    x = layers.TimeDistributed(layers.Conv2D(filters=s2_filters2, kernel_size=(3, 3), activation='relu',
                                             padding='same', kernel_regularizer=regularizers.l2(l2_reg)))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.TimeDistributed(layers.Dense(128, activation='relu'))(x)
    # Aggregate temporal dimension
    x = layers.GlobalAveragePooling1D()(x)
    branch_outputs["Sentinel-2"] = x

    # Sentinel-1 branch
    inp_s1 = Input(shape=input_shapes["Sentinel-1"], name="Sentinel-1")
    inputs["Sentinel-1"] = inp_s1
    y = layers.TimeDistributed(layers.Conv2D(filters=s1_filters1, kernel_size=(3, 3), activation='relu',
                                             padding='same', kernel_regularizer=regularizers.l2(l2_reg)))(inp_s1)
    y = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(y)
    y = layers.TimeDistributed(layers.Dropout(dropout_rate))(y)
    y = layers.TimeDistributed(layers.Conv2D(filters=s1_filters2, kernel_size=(3, 3), activation='relu',
                                             padding='same', kernel_regularizer=regularizers.l2(l2_reg)))(y)
    y = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(y)
    y = layers.TimeDistributed(layers.Flatten())(y)
    y = layers.TimeDistributed(layers.Dense(128, activation='relu'))(y)
    y = layers.GlobalAveragePooling1D()(y)
    branch_outputs["Sentinel-1"] = y

    # Concatenate branches
    concatenated = layers.concatenate(list(branch_outputs.values()), axis=-1)
    z = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(concatenated)
    z = layers.Dropout(dropout_rate)(z)
    
    if num_classes == 1:
        output = layers.Dense(1, activation='sigmoid', name="classification")(z)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        output = layers.Dense(num_classes, activation='softmax', name="classification")(z)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']

    model = models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    return model