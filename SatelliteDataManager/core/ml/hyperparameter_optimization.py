#!/usr/bin/env python3
"""
hyperparameter_optimization.py
------------------------------
This module provides helper functions to integrate with Optuna for hyperparameter optimization.
It includes a function to run an Optuna study given an objective function.
"""

import optuna

def run_optuna_study(objective, n_trials=50, timeout=None, study_name="optimization_study", direction="minimize", storage=None):
    """
    Runs an Optuna study using the provided objective function.

    Parameters:
        objective (callable): The objective function to optimize.
        n_trials (int): Number of trials to run (default: 50).
        timeout (int, optional): Stop study after the given number of seconds.
        study_name (str): Name of the study.
        direction (str): "minimize" or "maximize" (default: "minimize").
        storage (str, optional): Database URL for study storage (if needed).

    Returns:
        optuna.study.Study: The completed Optuna study.
    """
    study = optuna.create_study(study_name=study_name, direction=direction, storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study
