#!/usr/bin/env python3
"""
data_split.py
-------------
This module provides functions to perform train/test splitting and k-fold cross-validation
on datasets. Additional functions support stratified splitting based on labels.
Labels can be a 2D binary map, a single binary value, or a multiclass scalar.
"""

import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split, KFold, StratifiedKFold

def process_label(label):
    """
    Process a single label for stratification.
    If the label is a 2D array (or an array with ndim>=2), computes its mean and thresholds it at 0.5.
    Otherwise returns the label as is.
    
    Parameters:
        label: The label value (can be scalar or numpy array).
    
    Returns:
        Processed label (scalar) for stratification.
    """
    if isinstance(label, np.ndarray):
        if label.ndim >= 2:
            return int(np.mean(label) > 0.5)
        elif label.ndim == 1 and label.size == 1:
            return label.item()
        else:
            return label
    else:
        return label

def train_test_split(data, test_size=0.2, shuffle=True, random_state=None):
    """
    Splits the data into training and testing sets without stratification.
    
    Parameters:
        data (array-like): The data to split.
        test_size (float): Proportion of the dataset to include in the test split.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_state (int, optional): Random state for reproducibility.
    
    Returns:
        tuple: (train_data, test_data)
    """
    return sk_train_test_split(data, test_size=test_size, shuffle=shuffle, random_state=random_state)

def stratified_train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=None):
    """
    Splits the data into training and testing sets in a stratified manner based on labels.
    
    Parameters:
        data (array-like): The data to split.
        labels (array-like): Corresponding labels for stratification.
        test_size (float): Proportion of the dataset to include in the test split.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_state (int, optional): Random state for reproducibility.
    
    Returns:
        tuple: (train_data, test_data, train_labels, test_labels)
    """
    processed_labels = [process_label(lbl) for lbl in labels]
    return sk_train_test_split(data, labels, test_size=test_size, shuffle=shuffle, random_state=random_state, stratify=processed_labels)

def kfold_split(data, k=5, shuffle=True, random_state=None):
    """
    Generates indices for k-fold cross-validation without stratification.
    
    Parameters:
        data (array-like): The data to split.
        k (int): Number of folds.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_state (int, optional): Random state for reproducibility.
    
    Returns:
        generator: A generator yielding (train_indices, test_indices) for each fold.
    """
    kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    return kf.split(np.array(data))

def stratified_kfold_split(data, labels, k=5, shuffle=True, random_state=None):
    """
    Generates indices for k-fold cross-validation in a stratified manner based on labels.
    
    Parameters:
        data (array-like): The data to split.
        labels (array-like): Corresponding labels for stratification.
        k (int): Number of folds.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_state (int, optional): Random state for reproducibility.
    
    Returns:
        generator: A generator yielding (train_indices, test_indices) for each fold.
    """
    processed_labels = [process_label(lbl) for lbl in labels]
    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    return skf.split(np.array(data), np.array(processed_labels))
