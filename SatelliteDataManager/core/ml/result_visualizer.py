#!/usr/bin/env python3
"""
result_visualizer.py
--------------------
This module provides functions to visualize training results and evaluation metrics.
Functions include plotting training history, ROC curves, and threshold optimization.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def plot_training_history(history):
    """
    Plots all metrics contained in the history object.
    For each metric (e.g., loss, accuracy, precision, recall, etc.) a separate plot is generated
    with training and validation values (if available).
    
    Parameters:
        history (tf.keras.callbacks.History): History object returned by model.fit().
    
    Returns:
        None.
    """
    metrics = [m for m in history.history.keys() if not m.startswith("val_")]
    num_metrics = len(metrics)
    plt.figure(figsize=(6 * num_metrics, 5))
    
    for idx, metric in enumerate(metrics):
        plt.subplot(1, num_metrics, idx + 1)
        train_vals = history.history[metric]
        val_metric = "val_" + metric
        plt.plot(train_vals, 'bo-', label=f'Train {metric}')
        if val_metric in history.history:
            plt.plot(history.history[val_metric], 'ro-', label=f'Val {metric}')
        plt.title(metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_pred):
    """
    Plots the ROC curve given true labels and predicted scores.

    Parameters:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted scores or probabilities.

    Returns:
        None.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()

def plot_threshold_optimization(y_true, y_pred, metric_function, thresholds=np.linspace(0, 1, 50)):
    """
    Plots a given metric (e.g., F1 score) computed over a range of thresholds.
    This helps in selecting the optimal threshold on the model output.
    
    Parameters:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted probabilities.
        metric_function (callable): Function that computes the metric given y_true and binary predictions.
        thresholds (array-like): Array of threshold values to evaluate.
        
    Returns:
        None.
    """
    metrics = []
    for t in thresholds:
        binary_preds = (y_pred >= t).astype(int)
        metrics.append(metric_function(y_true, binary_preds))
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, metrics, 'b-', label='Metric')
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Threshold Optimization")
    plt.legend()
    plt.show()
