#!/usr/bin/env python3
"""
model_selection.py
------------------
This module provides functions for model evaluation and selection.
It includes functions to evaluate a model on a given dataset and compare multiple models.
"""

import numpy as np
import tensorflow as tf

def evaluate_model(model, dataset):
    """
    Evaluates the model on the provided dataset.

    Parameters:
        model (tf.keras.Model): The model to evaluate.
        dataset (tf.data.Dataset): The dataset on which to evaluate the model.

    Returns:
        dict: A dictionary containing loss and metrics values.
    """
    results = model.evaluate(dataset, verbose=0)
    metrics = model.metrics_names
    return dict(zip(metrics, results))

def compare_models(models, dataset, metric='accuracy'):
    """
    Compares multiple models on a given dataset based on a specified metric.

    Parameters:
        models (list of tf.keras.Model): List of models to compare.
        dataset (tf.data.Dataset): The dataset on which to evaluate the models.
        metric (str): The metric to use for comparison (default: 'accuracy').

    Returns:
        list of tuple: List of (model, metric_value) tuples sorted by metric value in descending order.
    """
    model_metrics = []
    for model in models:
        eval_results = evaluate_model(model, dataset)
        model_metrics.append((model, eval_results.get(metric, None)))
    # Sort models by metric value in descending order
    model_metrics.sort(key=lambda x: x[1] if x[1] is not None else -np.inf, reverse=True)
    return model_metrics
