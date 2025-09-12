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
from ..data_visualizer import DataVisualizer
import tensorflow as tf
import json
import math
from skimage.measure import find_contours


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

def visualize_tfrecord_with_predictions(
    tfrecord_path: str,
    prediction_mask: np.ndarray | None = None,
    crop: bool = False,
    crop_factor: int = 1,
    rgb_indices: tuple[int, int, int] = (3, 2, 1),
    cmap_pred: str = "Reds",
    save_path: str | None = None,
    show_input_data: bool = True,
):
    """
    Visualise TFRecord contents plus a full-size Sentinel-2 RGB overlay that shows:
      • Ground-truth contours (blue)
      • Model prediction mask (red, semi-transparent)

    Parameters
    ----------
    tfrecord_path : str
        Path to the TFRecord file.
    prediction_mask : np.ndarray | None
        Either a full-size 2-D mask (H × W) or a stack of cropped patches
        shaped (k², H_patch, W_patch) if `crop=True`.
    crop : bool, default=False
        Set to True when the TFRecord was created with cropped patches.
    crop_factor : int, default=1
        The k value used to split the original image into k×k patches.
    rgb_indices : tuple[int, int, int], default=(3, 2, 1)
        Band indices for the Sentinel-2 RGB composite (R, G, B).
    cmap_pred : str, default="Reds"
        Matplotlib colormap for the prediction overlay.
    save_path : str | None, default=None
        If provided, saves the final figure to this path (PNG).
    show_diagnostics : bool, default=True
        If True, runs the built-in diagnostic plots via DataVisualizer
        before rendering the RGB overlay. Set to False to skip them.
    """
    # ------------------------------------------------------------------ #
    # (0) Optional diagnostic plots
    # ------------------------------------------------------------------ #
    if show_input_data:
        vis = DataVisualizer()
        vis.inspect_and_visualize_custom_tfrecord(
            tfrecord_path=tfrecord_path,
            crop=crop,
            crop_factor=crop_factor,
        )

    # ------------------------------------------------------------------ #
    # (1) Helper to reassemble a k×k grid of patches
    # ------------------------------------------------------------------ #
    
    def _reassemble_patch_grid(
            patches: np.ndarray,
            k: int,
            gap: int = 5
    ) -> np.ndarray:
        n_patches, patch_H, patch_W = patches.shape
        if n_patches != crop_factor * crop_factor:
            raise ValueError(f"Number of patches ({n_patches}) does not equal crop_factor^2 ({crop_factor**2}).")
        rows = []
        for r in range(crop_factor):
            row_patches = patches[r * crop_factor:(r + 1) * crop_factor]
            padded = [np.pad(p, ((0, 0), (0, gap)), mode='constant', constant_values=0) for p in row_patches]
            row = np.hstack(padded)
            rows.append(row)
        grid = np.vstack([np.pad(r_img, ((0, gap), (0, 0)), mode='constant', constant_values=0) for r_img in rows])
        return grid
    
    # ------------------------------------------------------------------ #
    # (2) Read all Sentinel-2 and label features from the TFRecord
    # ------------------------------------------------------------------ #
    raw_ds = tf.data.TFRecordDataset(tfrecord_path)
    img_feats, label_feats = [], []
    for rec in raw_ds:
        ex = tf.train.Example.FromString(rec.numpy()).features.feature
        if "Sentinel-2_image" in ex:
            img_feats.append(ex)
        if "activation_label" in ex:
            label_feats.append(ex)

    if not img_feats:
        print("No Sentinel-2 data found – aborting overlay.")
        return

    # ------------------------------------------------------------------ #
    # (3) Build an array (n_patches, H, W, C) → RGB mosaic
    # ------------------------------------------------------------------ #
    img_patches = []
    for ex in img_feats:
        buf = ex["Sentinel-2_image"].bytes_list.value[0]
        H = ex["Sentinel-2_height"].int64_list.value[0]
        W = ex["Sentinel-2_width"].int64_list.value[0]
        C = ex["Sentinel-2_channels"].int64_list.value[0]
        n_steps = ex.get(
            "Sentinel-2_n_steps", tf.train.Int64List(value=[1])
        ).int64_list.value[0]

        arr = np.frombuffer(buf, dtype=np.float32)
        arr = arr.reshape((n_steps, H, W, C))[-1] if n_steps > 1 else arr.reshape((H, W, C))
        img_patches.append(arr)
    img_patches = np.stack(img_patches, axis=0)                        # (k², Hₚ, Wₚ, C)

    k = crop_factor if crop else 1
    bands = [
        _reassemble_patch_grid(img_patches[..., b], k)                # no gap
        for b in range(img_patches.shape[-1])
    ]
    r_idx, g_idx, b_idx = rgb_indices
    rgb = np.stack([bands[r_idx], bands[g_idx], bands[b_idx]], axis=-1)

    rgb = (rgb - rgb.min()) / (0.5*rgb.ptp())
    rgb[rgb>1] = 1
    
    # ------------------------------------------------------------------ #
    # (4) Reassemble the ground-truth label (if present)
    # ------------------------------------------------------------------ #
    full_label = None
    if label_feats:
        lab_patches = []
        for ex in label_feats:
            buf = ex["activation_label"].bytes_list.value[0]
            LH = ex["activation_label_height"].int64_list.value[0]
            LW = ex["activation_label_width"].int64_list.value[0]
            lab_patches.append(np.frombuffer(buf, dtype=np.uint8).reshape((LH, LW)))
        lab_patches = np.stack(lab_patches, axis=0)
        full_label = _reassemble_patch_grid(lab_patches, k)

    # ------------------------------------------------------------------ #
    # (5) Reassemble the prediction mask (if passed as patches)
    # ------------------------------------------------------------------ #
    if prediction_mask is not None and crop and prediction_mask.ndim == 3:
        prediction_mask = _reassemble_patch_grid(prediction_mask, k)

    # ------------------------------------------------------------------ #
    # (6) Extract GT contours
    # ------------------------------------------------------------------ #
    contours = []
    if full_label is not None:
        try:
            contours = find_contours(full_label, 0.5)
        except Exception as e:
            print(f"Contour extraction failed: {e}")

    # ------------------------------------------------------------------ #
    # (7) Final plot
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title("Sentinel-2 RGB (last time-step)\nGround truth (blue) • Prediction (red)")
    
    for c in contours:                                               # c: (rows, cols)
        ax.plot(c[:, 1], c[:, 0], color="blue", linewidth=1)

    if prediction_mask is not None:
        if prediction_mask.shape != rgb.shape[:2]:
            raise ValueError(
                f"`prediction_mask` shape {prediction_mask.shape} does not match RGB {rgb.shape[:2]}"
            )
        alpha = np.clip(prediction_mask.astype(float), 0.0, 1.0) * 0.5
        ax.imshow(prediction_mask, cmap=cmap_pred, alpha=alpha)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
