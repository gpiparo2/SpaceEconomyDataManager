#!/usr/bin/env python3
"""
burned_area_model.py
--------------------
This module defines the machine learning model for burned area segmentation.
It builds a multi-sensor segmentation model that processes inputs from various satellites
and outputs a binary segmentation map.
Additional hyperparameters allow optimization of the number of filters for each branch.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, Input
#Important to enable eager execution
tf.data.experimental.enable_debug_mode()
tf.config.run_functions_eagerly(True)

def build_burned_area_segmentation_model(input_shapes, dropout_rate=0.2, l2_reg=5e-4,
                                           s2_filters1=32, s2_filters2=64,
                                           s1_filters1=32, s1_filters2=64,
                                           s3olci_filters1=32, s3olci_filters2=64,
                                           s3slstr_filters1=32, s3slstr_filters2=64,
                                           dem_filters1=32, dem_filters2=64):
    """
    Builds a segmentation model with multi-sensor inputs.

    Each sensor branch is processed with ConvLSTM2D (or Conv2D for DEM) and pooling layers
    such that the final feature map of each branch has a spatial resolution of 128x128.
    All branches are then concatenated and upsampled to match the original resolution.

    Parameters:
      input_shapes (dict): Dictionary with sensor names as keys and input shapes as values.
      dropout_rate (float): Dropout rate.
      l2_reg (float): L2 regularization factor.
      s2_filters1 (int): Number of filters for first ConvLSTM2D layer in Sentinel-2 branch.
      s2_filters2 (int): Number of filters for second ConvLSTM2D layer in Sentinel-2 branch.
      s1_filters1 (int): Number of filters for first ConvLSTM2D layer in Sentinel-1 branch.
      s1_filters2 (int): Number of filters for second ConvLSTM2D layer in Sentinel-1 branch.
      s3olci_filters1 (int): Number of filters for first ConvLSTM2D layer in Sentinel-3-OLCI branch.
      s3olci_filters2 (int): Number of filters for second ConvLSTM2D layer in Sentinel-3-OLCI branch.
      s3slstr_filters1 (int): Number of filters for first ConvLSTM2D layer in Sentinel-3-SLSTR-Thermal branch.
      s3slstr_filters2 (int): Number of filters for second ConvLSTM2D layer in Sentinel-3-SLSTR-Thermal branch.
      dem_filters1 (int): Number of filters for first Conv2D layer in DEM branch.
      dem_filters2 (int): Number of filters for second Conv2D layer in DEM branch.

    Returns:
      tf.keras.Model: Compiled segmentation model.
    """
    inputs = {}
    branch_outputs = {}
    # Sentinel-2 branch
    inp_s2 = Input(shape=input_shapes["Sentinel-2"], name="Sentinel-2")
    inputs["Sentinel-2"] = inp_s2
    x = layers.ConvLSTM2D(filters=s2_filters1, kernel_size=(3, 3), padding='same', activation='relu',
                          return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg))(inp_s2)
    x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(x)
    x = layers.TimeDistributed(layers.Dropout(dropout_rate))(x)
    x = layers.ConvLSTM2D(filters=s2_filters2, kernel_size=(3, 3), padding='same', activation='relu',
                          return_sequences=False, kernel_regularizer=regularizers.l2(l2_reg))(x)
    #x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    #x = layers.Dropout(dropout_rate)(x)
    #x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    #x = layers.Dropout(dropout_rate)(x)
    branch_outputs["Sentinel-2"] = x  # (128,128, s2_filters2)

    # Sentinel-1 branch
    inp_s1 = Input(shape=input_shapes["Sentinel-1"], name="Sentinel-1")
    inputs["Sentinel-1"] = inp_s1
    y = layers.ConvLSTM2D(filters=s1_filters1, kernel_size=(3, 3), padding='same', activation='relu',
                          return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg))(inp_s1)
    y = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(y)
    y = layers.TimeDistributed(layers.Dropout(dropout_rate))(y)
    y = layers.ConvLSTM2D(filters=s1_filters2, kernel_size=(3, 3), padding='same', activation='relu',
                          return_sequences=False, kernel_regularizer=regularizers.l2(l2_reg))(y)
    #y = layers.MaxPooling2D(pool_size=(2, 2))(y)
    #y = layers.Dropout(dropout_rate)(y)
    #y = layers.MaxPooling2D(pool_size=(2, 2))(y)
    #y = layers.Dropout(dropout_rate)(y)
    branch_outputs["Sentinel-1"] = y  # (128,128, s1_filters2)

    # Sentinel-3-OLCI branch
    inp_s3olci = Input(shape=input_shapes["Sentinel-3-OLCI"], name="Sentinel-3-OLCI")
    inputs["Sentinel-3-OLCI"] = inp_s3olci
    z = layers.ConvLSTM2D(filters=s3olci_filters1, kernel_size=(3, 3), padding='same', activation='relu',
                          return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg))(inp_s3olci)
    z = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(z)
    z = layers.TimeDistributed(layers.Dropout(dropout_rate))(z)
    z = layers.ConvLSTM2D(filters=s3olci_filters2, kernel_size=(3, 3), padding='same', activation='relu',
                          return_sequences=False, kernel_regularizer=regularizers.l2(l2_reg))(z)
    branch_outputs["Sentinel-3-OLCI"] = z  # (128,128, s3olci_filters2)

    # Sentinel-3-SLSTR-Thermal branch
    inp_s3slstr = Input(shape=input_shapes["Sentinel-3-SLSTR-Thermal"], name="Sentinel-3-SLSTR-Thermal")
    inputs["Sentinel-3-SLSTR-Thermal"] = inp_s3slstr
    w = layers.ConvLSTM2D(filters=s3slstr_filters1, kernel_size=(3, 3), padding='same', activation='relu',
                          return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg))(inp_s3slstr)
    w = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(w)
    w = layers.TimeDistributed(layers.Dropout(dropout_rate))(w)
    w = layers.ConvLSTM2D(filters=s3slstr_filters2, kernel_size=(3, 3), padding='same', activation='relu',
                          return_sequences=False, kernel_regularizer=regularizers.l2(l2_reg))(w)
    branch_outputs["Sentinel-3-SLSTR-Thermal"] = w  # (128,128, s3slstr_filters2)

    # DEM branch (using Conv2D)
    inp_dem = Input(shape=input_shapes["DEM"], name="DEM")
    inputs["DEM"] = inp_dem
    v = layers.Conv2D(filters=dem_filters1, kernel_size=(3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(inp_dem)
    v = layers.MaxPooling2D(pool_size=(2, 2))(v)
    v = layers.Dropout(dropout_rate)(v)
    v = layers.Conv2D(filters=dem_filters2, kernel_size=(3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(v)
    #v = layers.MaxPooling2D(pool_size=(2, 2))(v)
    #v = layers.Dropout(dropout_rate)(v)
    #v = layers.MaxPooling2D(pool_size=(2, 2))(v)
    #v = layers.Dropout(dropout_rate)(v)
    branch_outputs["DEM"] = v  # (128,128, dem_filters2)

    # Concatenate all branches
    concatenated = layers.concatenate(list(branch_outputs.values()), axis=-1)  # e.g., (128,128, total_filters)
    # Decoder: Upsample to original resolution (assume Sentinel-2 original resolution 1024x1024)
   # x_dec = layers.UpSampling2D(size=(4, 4))(concatenated)
    x_dec = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg))(concatenated)
    x_dec = layers.Dropout(dropout_rate)(x_dec)
    x_dec = layers.UpSampling2D(size=(2, 2))(x_dec)
    x_dec = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg))(x_dec)
    x_dec = layers.Dropout(dropout_rate)(x_dec)
    output = layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', name="segmentation")(x_dec)

    model = models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
