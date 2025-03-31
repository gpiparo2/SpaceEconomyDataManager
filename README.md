# 🚀 SpaceEconomyDataManager

**SpaceEconomyDataManager** is a modular Python library for acquiring, processing, and analyzing satellite imagery. Built around the Sentinel Hub APIs, it supports automated workflows for various Earth observation use cases, including:

- 🔥 **Burned Area Segmentation**
- 🍇 **Vineyard Classification**
- 💧 **Irrigation Monitoring (Semaforo Irrigation)**

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
│
│   ├── analyses/
│   │   ├── burned_area/
│   │   │   ├── burned_area_dataset_builder.py   # TFRecord builder for fire segmentation
│   │   │   └── burned_area_dashboard.py         # GUI for configuring burned area workflows
│   │   ├── vineyard/
│   │   │   ├── vineyard_dataset_builder.py      # TFRecord builder for vineyard classification
│   │   │   └── vineyard_dashboard.py            # GUI for vineyard analysis
│   │   └── semaforo_irrigation/
│   │       ├── evapotranspiration_analysis_functions.py  # ET metrics and irrigation logic
│   │       ├── SensorDataManager.py                     # Sensor integration module
│   │       └── evapotranspiration_dashboard.py          # GUI for irrigation analysis
│
│   └── gui/
│       ├── common_gui.py                    # Shared GUI elements and widgets
│       └── ml_dashboard.py                  # Dashboard for ML training and evaluation
│
├── config/                                   # JSON files and evalscripts for indices
├── last_session/                             # Stores session metadata
├── docs/                                     # Documentation
├── test_download_code.ipynb                  # Code-driven example
├── test_download_gui.ipynb                   # GUI-based example
└── README.md
```

---

## 🔩 Core Modules

Each core module is reusable across multiple analysis domains:

| Module              | Description |
|---------------------|-------------|
| `data_download.py`  | Authenticates and downloads imagery from Sentinel Hub using evalscripts |
| `data_manipulator.py` | Organizes, renames, and prepares files for dataset generation |
| `dataset_preparation.py` | Handles normalization, masking, cropping, and TFRecord serialization |
| `data_visualizer.py` | Provides utilities for image and NDVI visualization |
| `ml/` | Contains model selection, hyperparameter tuning, and results visualization |
| `sdm.py` | Wrapper class exposing the full core pipeline |

---

## 🧪 Analysis Modules

### 🔥 Burned Area Segmentation

Located in `analyses/burned_area/`

This module automates the construction of a TFRecord dataset to train models for fire damage detection.

**Key Features:**
- Reads CEMS activation files (fire GeoJSONs)
- Applies satellite-specific temporal sampling
- Masks fire areas and builds binary labels
- Allows normalization, cropping, and patch generation

**Input:** Fire activations from [Copernicus EMS](https://emergency.copernicus.eu/)

---

### 🍇 Vineyard Classification

Located in `analyses/vineyard/`

Automates the process of generating a labeled dataset from vineyard polygons.

**Key Features:**
- Loads vineyard GeoJSONs and uses `"Classe"` field for label extraction
- Supports multi-class or binary labeling
- Retrieves imagery around fixed reference dates
- Generates structured TFRecord examples

---

### 💧 Semaforo Irrigation

Located in `analyses/semaforo_irrigation/`

Focuses on evapotranspiration analysis for precision agriculture.

**Key Features:**
- Computes evapotranspiration metrics using Sentinel imagery
- Includes functions for calculating water stress and irrigation need
- Enables dashboard-based visualization of ET maps

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
(NOT READY)

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
