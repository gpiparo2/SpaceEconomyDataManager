# SpaceEconomyDataManager

**SpaceEconomyDataManager** is a modular Python library for acquiring, processing, and analyzing satellite imagery. Built around the Sentinel Hub APIs, it supports automated workflows for various Earth observation use cases, including:

- 🔥 **Burned Area Segmentation**
- 🍇 **Vineyard Classification**
- 💧 **Irrigation Monitoring**

It also features interactive GUIs to facilitate dataset configuration and exploration, with a clear separation between core functionalities and domain-specific analyses.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Library Structure](#library-structure)
- [Core Modules](#core-modules)
- [Analysis Modules](#analysis-modules)
  - [Burned Area Segmentation](#burned-area-segmentation)
  - [Vineyard Classification](#vineyard-classification)
  - [Semaforo Irrigation](#semaforo-irrigation)
- [Interactive Dashboards](#interactive-dashboards)
- [Usage Examples](#usage-examples)
- [Documentation](#documentation)
- [License](#license)
- [Contact](#contact)

---

## 🛰️ Overview

SpaceEconomyDataManager simplifies Earth observation workflows by integrating:
- Sentinel Hub imagery access
- Automated preprocessing pipelines
- Flexible dataset builders
- Machine learning-ready outputs in TFRecord format
- Interactive GUI dashboards for visual configuration

---

## ⚙️ Installation

> **Requirements**: Python 3.6+

Install all required packages:
```bash
pip install requirements.txt
```

---

### 🛡️ Sentinel Hub Authentication

To access satellite data, you need valid Sentinel Hub credentials.

1. Register at [Copernicus Sentinel Hub](https://dataspace.copernicus.eu/copernicus-data-space-ecosystem-dashboard)
2. Create a new OAuth **client** in your account dashboard
3. Note down your **Client ID** and **Client Secret**

Store these credentials in your environment or provide them when prompted by the application.  
You can also authenticate in notebooks or GUIs using the provided widgets.

---

## 🧱 Library Structure

```
SpaceEconomyDataManager/
├── SatelliteDataManager/
│   ├── core/
│   │   ├── data_download.py                # Download Sentinel data using evalscripts
│   │   ├── data_manipulator.py             # Organize and standardize downloaded images
│   │   ├── dataset_preparation.py          # Normalize, mask, and serialize to TFRecords
│   │   ├── data_visualizer.py              # Visualization utilities (RGB, NDVI, etc.)
│   │   ├── sdm.py                          # Unified wrapper to manage the full pipeline
│   │   └── ml/
│   │       ├── data_split.py               # Train/test splitting and stratified sampling
│   │       ├── model_selection.py          # Model selection and training logic
│   │       ├── hyperparameter_optimization.py  # Randomized search over hyperparams
│   │       └── result_visualizer.py        # Visualization of metrics and predictions
│   ├── gui/
│   │   ├── common_gui.py                   # Shared GUI elements and widgets
│   │   └── ml_dashboard.py                 # Dashboard for ML training and evaluation
│   └── analyses/                           #Location of external analysis modules
│
├── config/                                  # JSON files and evalscripts for indices
├── last_session/                            # Stores session metadata
├── docs/                                    # Documentation and usage guide
├── test_download_code.ipynb                 # Code-driven usage example
├── test_download_gui.ipynb                  # GUI-based usage example
└── README.md   
```

---

## 🔩 Core Modules

Each core module is reusable across multiple analysis domains. The core system is designed to handle end-to-end satellite data workflows — from download, organization, transformation, visualization, to machine learning preparation.

### 📁 `core/` — General Data Handling

| File | Description |
|------|-------------|
| `data_download.py` | Handles Sentinel Hub authentication and image download. Includes loading of evalscripts, multi-temporal acquisitions, and data export to GeoTIFF format. |
| `data_manipulator.py` | Post-download reorganization of TIFFs, renaming files by satellite and date, compressing or formatting them into NumPy arrays or CSV for pipeline integration. |
| `dataset_preparation.py` | Builds machine learning datasets. Includes normalization (quantile or min-max), masking with vector geometries, image patching, TFRecord generation, and augmentation. |
| `data_visualizer.py` | Provides utility functions for plotting satellite bands, RGB composites, NDVI, and inspecting the structure of TFRecord datasets. |
| `sdm.py` | Central manager class that integrates all core components into a single accessible API. Used to orchestrate full workflows via code or GUI. |

---

### 📁 `ml/` — Machine Learning Toolkit

| File | Description |
|------|-------------|
| `data_split.py` | Provides stratified or random splitting of datasets into train/validation/test subsets. Supports user-defined grouping or time-based partitioning. |
| `model_selection.py` | Defines various model training routines and evaluation strategies. Supports classifiers, regressors, and segmentation models. |
| `hyperparameter_optimization.py` | Implements random search and tuning routines for model hyperparameters using cross-validation. |
| `result_visualizer.py` | Visualizes training curves, confusion matrices, ROC curves, and prediction vs ground truth overlays. Helps evaluate model performance. |

---

### 📁 `gui/` — Interactive Tools

| File | Description |
|------|-------------|
| `common_gui.py` | Shared widgets and helper functions used across GUIs, such as authentication forms, dropdowns, loggers, and layout builders. |
| `ml_dashboard.py` | GUI application for managing model training, hyperparameter tuning, and live visualization of results. Supports easy switching of dataset, model, and parameters. |

---

---

## 🔗 Linked Analysis Modules (Submodules)

| Module Name | Description | Repository |
|-------------|-------------|------------|
| 🔥 `SEDM_Wildfire` | Burned Area Segmentation using Sentinel imagery | [Link to repo](https://github.com/gpiparo2/SEDM_Wildfire) |
| 🍇 `SEDM_Vineyard` | Vineyard classification based on polygon data (private) | [Link to repo](https://github.com/gpiparo2/SEDM_Vineyard) |
| 💧 `SEDM_Irrigation` | Evapotranspiration-based irrigation monitoring (private) | [Link to repo](https://github.com/gpiparo2/SEDM_Irrigation) |

> ⚠️ These submodules may be private. Request access from the repository maintainer if needed.

---

## 🖥️ Interactive Dashboards

Found in `gui/` and `analyses/**/dashboard.py` files, these ipywidgets-based GUIs allow:

- Credential authentication
- Dataset parameter configuration
- File selection and preprocessing
- Live TFRecord preview and visualization

> No code required — just run the notebooks and interact visually.

---

## 🧪 Usage Examples

- [`test_download_code.ipynb`](test_download_code.ipynb): Code-driven pipeline usage  
- [`test_download_gui.ipynb`](test_download_gui.ipynb): Dashboard-based exploration

---

## 📄 Documentation

Inline docstrings are provided throughout the code.  
For a full module-by-module guide, see [docs/usage.md](docs/usage.md)

---

## 📬 Contact

Developed by **Giuseppe Piparo**  
📧 [giuseppe.piparo@ct.infn.it](mailto:giuseppe.piparo@ct.infn.it)

---

## 🛰️ External References

- [Sentinel Hub](https://www.sentinel-hub.com/)
- [Copernicus Open Access Hub](https://scihub.copernicus.eu/dhus/#/home)
- [Copernicus Emergency Management Service (CEMS)](https://emergency.copernicus.eu/)

```
