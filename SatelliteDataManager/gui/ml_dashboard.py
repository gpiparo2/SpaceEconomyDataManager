#!/usr/bin/env python3
"""
ml_dashboard.py
---------------
This module defines an interactive ML Dashboard that integrates the new ML pipeline functionalities into the GUI.
It provides sections for:
  1. Data Splitting & Preprocessing
  2. Hyperparameter Optimization (Optuna)
  3. K-Fold Cross Validation
  4. Final Evaluation

Users can select the analysis type (Burned Area or Vineyard) via a dropdown.
Each section is implemented using ipywidgets, and the dashboard integrates the ML functions from the core and analysis modules.
"""

import os
import json
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

# Import common GUI functions (for styling)
from SatelliteDataManager.gui.common_gui import inject_css

# Import ML modules and functions
from SatelliteDataManager.core.sdm import SDM
from SatelliteDataManager.core.ml.data_split import train_test_split, stratified_train_test_split, kfold_split, stratified_kfold_split
from SatelliteDataManager.core.ml.hyperparameter_optimization import run_optuna_study
from SatelliteDataManager.core.ml.result_visualizer import plot_training_history, plot_roc_curve

# Import analysis-specific modules:
# For Burned Area:
from SatelliteDataManager.analyses.burned_area.burned_area_model import build_burned_area_segmentation_model
from SatelliteDataManager.analyses.burned_area.burned_area_optuna import objective as burned_area_objective
# For Vineyard:
from SatelliteDataManager.analyses.vineyard.vineyard_model import build_vineyard_classification_model
from SatelliteDataManager.analyses.vineyard.vineyard_optuna import objective as vineyard_objective

# Default configurations (per analysis type)
DEFAULT_TFRECORD_FOLDER_BURNED = "./test/tfrecords"
DEFAULT_TFRECORD_FOLDER_VINEYARD = "./test/tfrecords_vineyard"
DEFAULT_BATCH_SIZE = 16
DEFAULT_INPUT_SHAPES_BURNED = {
    "Sentinel-2": (3, 256, 256, 17),
    "Sentinel-1": (3, 256, 256, 2),
    "Sentinel-3-OLCI": (12, 256, 256, 21),
    "Sentinel-3-SLSTR-Thermal": (12, 256, 256, 5),
    "DEM": (256, 256, 1)
}
DEFAULT_INPUT_SHAPES_VINEYARD = {
    "Sentinel-2": (3, 256, 256, 17),
    "Sentinel-1": (3, 256, 256, 2)
}

def simulate_labels(file_list):
    """Simulate binary labels for stratification (for demo purposes)."""
    return np.random.randint(0, 2, size=len(file_list))

class MLDashboard:
    def __init__(self):
        inject_css()
        self.out_log = widgets.Output()
        
        # Dropdown per selezionare il tipo di analisi
        self.analysis_dropdown = widgets.Dropdown(
            options=["Burned Area", "Vineyard"],
            description="Analysis Type:",
            value="Burned Area"
        )
        
        # Input per il percorso dei TFRecord
        self.tfrecord_folder_input = widgets.Text(
            value=DEFAULT_TFRECORD_FOLDER_BURNED,
            description="TFRecord Folder:"
        )
        
        # --------------------------
        # Section 1: Data Splitting & Preprocessing
        # --------------------------
        self.stratify_checkbox = widgets.Checkbox(
            value=True,
            description="Stratified Split"
        )
        self.test_size_slider = widgets.FloatSlider(
            value=0.25,
            min=0.1,
            max=0.5,
            step=0.05,
            description="Test Size:"
        )
        self.split_button = widgets.Button(
            description="Run Split",
            button_style="info"
        )
        self.split_output = widgets.Output()
        self.split_button.on_click(self.run_split)
        
        self.split_section = widgets.VBox([
            widgets.HTML("<h3>Data Splitting & Preprocessing</h3>"),
            self.stratify_checkbox,
            self.test_size_slider,
            self.split_button,
            self.split_output
        ])
        
        # --------------------------
        # Section 2: Hyperparameter Optimization
        # --------------------------
        self.opt_trials_slider = widgets.IntSlider(
            value=5,
            min=1,
            max=20,
            step=1,
            description="Optuna Trials:"
        )
        self.opt_epochs_slider = widgets.IntSlider(
            value=5,
            min=1,
            max=20,
            step=1,
            description="Epochs/Trial:"
        )
        self.opt_button = widgets.Button(
            description="Run Optimization",
            button_style="warning"
        )
        self.opt_output = widgets.Output()
        self.opt_button.on_click(self.run_optimization)
        
        self.opt_section = widgets.VBox([
            widgets.HTML("<h3>Hyperparameter Optimization</h3>"),
            self.opt_trials_slider,
            self.opt_epochs_slider,
            self.opt_button,
            self.opt_output
        ])
        
        # --------------------------
        # Section 3: K-Fold Cross Validation
        # --------------------------
        self.kfold_slider = widgets.IntSlider(
            value=5,
            min=2,
            max=10,
            step=1,
            description="Number of Folds:"
        )
        self.kfold_button = widgets.Button(
            description="Run K-Fold CV",
            button_style="primary"
        )
        self.kfold_output = widgets.Output()
        self.kfold_button.on_click(self.run_kfold)
        
        self.kfold_section = widgets.VBox([
            widgets.HTML("<h3>K-Fold Cross Validation</h3>"),
            self.kfold_slider,
            self.kfold_button,
            self.kfold_output
        ])
        
        # --------------------------
        # Section 4: Final Evaluation
        # --------------------------
        self.eval_button = widgets.Button(
            description="Evaluate Final Model",
            button_style="success"
        )
        self.eval_output = widgets.Output()
        self.eval_button.on_click(self.run_final_evaluation)
        
        self.eval_section = widgets.VBox([
            widgets.HTML("<h3>Final Evaluation</h3>"),
            self.eval_button,
            self.eval_output
        ])
        
        # Organize sections in a Tab widget
        self.tab = widgets.Tab(children=[
            self.split_section,
            self.opt_section,
            self.kfold_section,
            self.eval_section
        ])
        self.tab.set_title(0, "Data Split")
        self.tab.set_title(1, "Optimization")
        self.tab.set_title(2, "K-Fold CV")
        self.tab.set_title(3, "Final Eval")
        
        # Overall layout: analysis dropdown, TFRecord folder input, and the tabs
        self.dashboard = widgets.VBox([
            self.analysis_dropdown,
            self.tfrecord_folder_input,
            self.tab,
            self.out_log
        ])
        
        # Internal variables per pipeline step
        self.train_files = None
        self.test_files = None
        self.best_params = None
        self.best_model = None
        self.sdm = None  # Will be initialized later
        
    def display(self):
        display(self.dashboard)
    
    def get_input_shapes(self):
        """Return input shapes based on the selected analysis type."""
        if self.analysis_dropdown.value == "Burned Area":
            return DEFAULT_INPUT_SHAPES_BURNED
        else:
            return DEFAULT_INPUT_SHAPES_VINEYARD
    
    def get_opt_objective(self):
        """Return the appropriate Optuna objective function."""
        if self.analysis_dropdown.value == "Burned Area":
            return burned_area_objective
        else:
            return vineyard_objective
    
    def get_build_model_fn(self):
        """Return the function to build the model based on analysis type."""
        if self.analysis_dropdown.value == "Burned Area":
            return build_burned_area_segmentation_model
        else:
            return build_vineyard_classification_model
    
    def init_sdm(self):
        """Initialize the SDM instance with current settings."""
        tfrecord_folder = self.tfrecord_folder_input.value
        self.sdm = SDM(
            config=SHConfig("peppe"),
            data_folder="./test/raw",
            manipulated_folder="./test/man",
            tfrecord_folder=tfrecord_folder
        )
    
    def run_split(self, b):
        with self.split_output:
            clear_output()
            tfrecord_folder = self.tfrecord_folder_input.value
            files = sorted([os.path.join(tfrecord_folder, f) for f in os.listdir(tfrecord_folder) if f.endswith(".tfrecord")])
            print("Found", len(files), "TFRecord files.")
            labels = simulate_labels(files)
            if self.stratify_checkbox.value:
                from SatelliteDataManager.core.ml.data_split import stratified_train_test_split
                train_files, test_files, _, _ = stratified_train_test_split(np.array(files), labels, test_size=self.test_size_slider.value, random_state=42)
                print(f"Stratified split: {len(train_files)} train files, {len(test_files)} test files")
            else:
                from SatelliteDataManager.core.ml.data_split import train_test_split
                train_files, test_files = train_test_split(np.array(files), test_size=self.test_size_slider.value, random_state=42)
                print(f"Random split: {len(train_files)} train files, {len(test_files)} test files")
            self.train_files = train_files
            self.test_files = test_files
    
    def run_optimization(self, b):
        with self.opt_output:
            clear_output()
            print("Starting hyperparameter optimization...")
            self.init_sdm()
            # Create datasets from train and test files
            train_ds = self.sdm.dataset_preparer.get_dataset(
                tfrecord_files=self.train_files.tolist(),
                batch_size=DEFAULT_BATCH_SIZE,
                augment=True,
                crop=False,
                min_label_percentage=0.
            )
            test_ds = self.sdm.dataset_preparer.get_dataset(
                tfrecord_files=self.test_files.tolist(),
                batch_size=DEFAULT_BATCH_SIZE,
                augment=False,
                crop=False,
                min_label_percentage=0.
            )
            input_shapes = self.get_input_shapes()
            objective_fn = self.get_opt_objective()
            trials = self.opt_trials_slider.value
            epochs = self.opt_epochs_slider.value
            study = run_optuna_study(
                objective=lambda trial: objective_fn(trial, input_shapes, num_classes=1, train_ds=train_ds, val_ds=test_ds, epochs=epochs),
                n_trials=trials,
                study_name=f"{self.analysis_dropdown.value.lower()}_optimization",
                direction="minimize"
            )
            self.best_params = study.best_params
            print("Optimization complete. Best hyperparameters:")
            print(self.best_params)
            # Build best model using the optimized parameters
            build_model_fn = self.get_build_model_fn()
            # For Burned Area, pass additional parameters; for Vineyard, only use Sentinel-2 and Sentinel-1.
            self.best_model = build_model_fn(
                input_shapes,
                num_classes=1,
                dropout_rate=self.best_params["dropout_rate"],
                l2_reg=self.best_params["l2_reg"],
                s2_filters1=self.best_params.get("s2_filters1", 32),
                s2_filters2=self.best_params.get("s2_filters2", 64),
                s1_filters1=self.best_params.get("s1_filters1", 32),
                s1_filters2=self.best_params.get("s1_filters2", 64),
                s3olci_filters1=self.best_params.get("s3olci_filters1", 32),
                s3olci_filters2=self.best_params.get("s3olci_filters2", 64),
                s3slstr_filters1=self.best_params.get("s3slstr_filters1", 32),
                s3slstr_filters2=self.best_params.get("s3slstr_filters2", 64),
                dem_filters1=self.best_params.get("dem_filters1", 32),
                dem_filters2=self.best_params.get("dem_filters2", 64)
            )
            print("Best model built. Summary:")
            self.best_model.summary()
    
    def run_kfold(self, b):
        with self.kfold_output:
            clear_output()
            print("Starting k-fold cross validation...")
            stratify = self.stratify_checkbox.value
            k = self.kfold_slider.value
            files = self.train_files
            labels = simulate_labels(files)
            if stratify:
                from SatelliteDataManager.core.ml.data_split import stratified_kfold_split
                fold_splits = list(stratified_kfold_split(files, labels, k=k, random_state=42))
            else:
                from SatelliteDataManager.core.ml.data_split import kfold_split
                fold_splits = list(kfold_split(files, k=k, random_state=42))
            fold_metrics = []
            for fold, (train_idx, val_idx) in enumerate(fold_splits):
                print(f"\nFold {fold+1}/{k}")
                fold_train_files = files[train_idx]
                fold_val_files = files[val_idx]
                fold_train_ds = self.sdm.dataset_preparer.get_dataset(
                    tfrecord_files=fold_train_files.tolist(),
                    batch_size=DEFAULT_BATCH_SIZE,
                    augment=True,
                    crop=False,
                    min_label_percentage=0.
                )
                fold_val_ds = self.sdm.dataset_preparer.get_dataset(
                    tfrecord_files=fold_val_files.tolist(),
                    batch_size=DEFAULT_BATCH_SIZE,
                    augment=False,
                    crop=False,
                    min_label_percentage=0.
                )
                build_model_fn = self.get_build_model_fn()
                fold_model = build_model_fn(
                    self.get_input_shapes(),
                    num_classes=1,
                    dropout_rate=self.best_params["dropout_rate"],
                    l2_reg=self.best_params["l2_reg"],
                    s2_filters1=self.best_params.get("s2_filters1", 32),
                    s2_filters2=self.best_params.get("s2_filters2", 64),
                    s1_filters1=self.best_params.get("s1_filters1", 32),
                    s1_filters2=self.best_params.get("s1_filters2", 64),
                    s3olci_filters1=self.best_params.get("s3olci_filters1", 32),
                    s3olci_filters2=self.best_params.get("s3olci_filters2", 64),
                    s3slstr_filters1=self.best_params.get("s3slstr_filters1", 32),
                    s3slstr_filters2=self.best_params.get("s3slstr_filters2", 64),
                    dem_filters1=self.best_params.get("dem_filters1", 32),
                    dem_filters2=self.best_params.get("dem_filters2", 64)
                )
                history_fold = fold_model.fit(fold_train_ds, validation_data=fold_val_ds, epochs=5, verbose=0)
                metrics_fold = fold_model.evaluate(fold_val_ds, verbose=0)
                fold_metrics.append(dict(zip(fold_model.metrics_names, metrics_fold)))
                print(f"Fold {fold+1} metrics:", fold_metrics[-1])
            avg_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics]) for metric in fold_metrics[0].keys()}
            print("\nAverage k-fold metrics:", avg_metrics)
    
    def run_final_evaluation(self, b):
        with self.eval_output:
            clear_output()
            print("Evaluating final model on train and test sets...")
            train_ds = self.sdm.dataset_preparer.get_dataset(
                tfrecord_files=self.train_files.tolist(),
                batch_size=DEFAULT_BATCH_SIZE,
                augment=False,
                crop=False,
                min_label_percentage=0.
            )
            test_ds = self.sdm.dataset_preparer.get_dataset(
                tfrecord_files=self.test_files.tolist(),
                batch_size=DEFAULT_BATCH_SIZE,
                augment=False,
                crop=False,
                min_label_percentage=0.
            )
            train_eval = self.best_model.evaluate(train_ds, verbose=0)
            test_eval = self.best_model.evaluate(test_ds, verbose=0)
            print("Train set metrics:", dict(zip(self.best_model.metrics_names, train_eval)))
            print("Test set metrics:", dict(zip(self.best_model.metrics_names, test_eval)))
            print("\nPlotting training history:")
            plot_training_history(self.opt_output.history if hasattr(self.opt_output, "history") else {})
            all_labels = []
            all_preds = []
            for batch_inputs, batch_labels in test_ds:
                preds = self.best_model.predict(batch_inputs)
                all_labels.append(batch_labels.numpy().flatten())
                all_preds.append(preds.flatten())
            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)
            print("\nPlotting ROC curve:")
            plot_roc_curve(all_labels, all_preds)

