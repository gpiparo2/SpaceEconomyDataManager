Below is the extended README.md content formatted in Markdown. You can copy and paste the following code into a Markdown cell:


# SpaceEconomyDataManager

SpaceEconomyDataManager is a comprehensive Python library designed for downloading, processing, and analyzing satellite imagery. It integrates with Sentinel Hub services to retrieve data and supports two specialized analyses: **Burned Area Segmentation** (fire analysis) and **Vineyard Classification**. The library is organized into modular components so that core functions can be reused or extracted for custom workflows.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Library Structure](#library-structure)
- [Usage](#usage)
  - [Getting Started with Sentinel Hub](#getting-started-with-sentinel-hub)
  - [Core Modules](#core-modules)
    - [DataDownload](#datadownload)
    - [DataManipulator](#datamanipulator)
    - [DatasetPreparation](#datasetpreparation)
    - [DataVisualizer](#datavisualizer)
    - [SDM (Core Wrapper)](#sdm-core-wrapper)
  - [Analysis Modules](#analysis-modules)
    - [Burned Area Segmentation Module](#burned-area-segmentation-module)
    - [Vineyard Classification Module](#vineyard-classification-module)
  - [Obtaining Activations from CEMS](#obtaining-activations-from-cems)
- [Interactive GUIs](#interactive-guis)
- [Examples and Testing](#examples-and-testing)
- [Documentation](#documentation)
- [License](#license)
- [Contact](#contact)

---

## Overview

SpaceEconomyDataManager is designed to simplify and automate the workflow of acquiring and processing satellite data. Its functionality is divided into two main parts:

### Core Modules

These modules handle common tasks such as:

- **DataDownload:** Authenticates with Sentinel Hub, sets up evalscripts, and downloads imagery.
- **DataManipulator:** Organizes and renames files, preparing datasets for analysis.
- **DatasetPreparation:** Preprocesses images (e.g., normalization, augmentation, cropping) and serializes data into TFRecord format.
- **DataVisualizer:** Visualizes satellite images and inspects TFRecord files.

### Analysis Modules

These modules are dedicated to specific use cases:

- **Burned Area Segmentation:** Focuses on fire segmentation analysis using activation information from emergency services.
- **Vineyard Classification:** Targets vineyard area analysis, including spectral evaluation and label extraction.

Additionally, the library offers interactive dashboards (GUI) for non-programmatic configuration and execution of workflows.

---

## Installation

To update
```

Ensure you have Python 3.6 or later and the following dependencies installed:

- numpy
- rasterio
- shapely
- geopandas
- tensorflow
- sentinelhub
- ipywidgets

---

## Library Structure

The library is organized as follows:

```
SpaceEconomyDataManager/
├── setup.py
├── README.md
├── test_download_code.ipynb
├── test_download_gui.ipynb
├── config/
│   ├── indices_config.json   
│   └── evalscripts.py    
├── docs/
│   └── usage.md         # Detailed usage guide for all functions
└── SatelliteDataManager/
    ├── __init__.py
    ├── core/            # Core functionalities
    │   ├── __init__.py
    │   ├── data_download.py      # Handles API authentication, evalscript setup, and image downloads
    │   ├── data_manipulator.py   # Organizes and renames downloaded files
    │   ├── dataset_preparation.py  # Prepares TFRecord datasets (normalization, augmentation, cropping, etc.)
    │   ├── data_visualizer.py    # Visualization tools for images and TFRecord inspection
    │   └── sdm.py                # Master class initializing all core modules
    │
    ├── analyses/         # Analysis-specific modules
    │   ├── __init__.py
    │   ├── burned_area/         # Fire segmentation analysis modules
    │   │   ├── __init__.py
    │   │   ├── custom_dataset_builder.py  # Custom TFRecord builder for burned area segmentation
    │   │   └── burned_area_dashboard.py     # Interactive dashboard for burned area analysis
    │   └── vineyard/           # Vineyard classification analysis modules
    │       ├── __init__.py
    │       ├── custom_dataset_builder.py  # Custom TFRecord builder for vineyard classification
    │       └── vineyard_dashboard.py        # Interactive dashboard for vineyard analysis
    │
    └── gui/               # Shared GUI components
        ├── __init__.py
        └── common_gui.py  # Common functions and widgets for dashboards
```

---

## Usage

### Getting Started with Sentinel Hub

Before using the library, ensure you have valid Sentinel Hub credentials.  
To obtain an account and download data, visit the following pages:

- **Copernicus Open Access Hub:** [https://scihub.copernicus.eu/dhus/#/home](https://scihub.copernicus.eu/dhus/#/home)
- **Sentinel Hub:** [https://www.sentinel-hub.com/](https://www.sentinel-hub.com/)

Detailed instructions for account creation and data download are provided on these websites.

### Core Modules

#### DataDownload

- **Functionality:**  
  Provides API authentication, evalscript setup, and downloading of satellite images.
- **Key Methods:**  
  - `authenticate_api()`: Authenticates using client credentials.
  - `set_evalscripts()`: Retrieves evalscripts for each satellite.
  - `download_images()`: Downloads images over specified time intervals.

Refer to the inline docstrings for detailed method descriptions.

#### DataManipulator

- **Functionality:**  
  Organizes and renames downloaded TIFF files into a standardized structure.
- **Key Methods:**  
  - `manipulate_data()`: Renames and copies files into satellite-specific folders.
  - `prepare_spectral_bands_dataset_list()`: Creates dataset lists for further processing.
  - `store_data_as_csv()` and `store_data_compressed()`: Exports processed data.

#### DatasetPreparation

- **Functionality:**  
  Prepares TFRecord datasets from organized satellite data.
- **Key Methods:**  
  - `load_image()`: Loads a TIFF image and retrieves its metadata.
  - `apply_geojson_mask()`: Masks an image based on a GeoJSON polygon.
  - `compute_global_quantiles()` and `global_normalize_image()`: Support normalization.
  - `crop_image_to_patches()`: Splits images into patches.
  - `parse_dataset()`: Deserializes TFRecord examples.

#### DataVisualizer

- **Functionality:**  
  Provides visualization tools for satellite images and TFRecord datasets.
- **Key Methods:**  
  - `display_image()` and `display_all_bands()`: Visualize images and individual bands.
  - `calculate_ndvi()` and `visualize_ndvi()`: Compute and display NDVI.
  - `inspect_and_visualize_custom_tfrecord()`: Inspects and visualizes TFRecords, including DEM data and labels.
  
#### SDM (Core Wrapper)

- **Functionality:**  
  Acts as a wrapper to initialize and provide direct access to all core modules.
- **Usage:**  
  Once instantiated, you can access the core modules via attributes:
  - `sdm.data_downloader`
  - `sdm.data_manipulator`
  - `sdm.data_visualizer`
  - `sdm.dataset_preparer`

### Analysis Modules

#### Burned Area Segmentation Module

- **Purpose:**  
  Builds custom TFRecord datasets for burned area segmentation (fire analysis).
- **Workflow:**  
  - Reads activation-specific sensor data.
  - Applies a temporal sampling strategy:
    - Sentinel-2 & Sentinel-1: Three time steps (first, median, last).
    - Sentinel-3-OLCI & Sentinel-3-SLSTR-Thermal: One image per day.
    - DEM: The central image.
  - Computes a binary fire mask from a fire GeoJSON file.
  - Optionally performs normalization, masking, and cropping.
  - Serializes data into TFRecord examples.
  
### Obtaining Activations from CEMS

For fire segmentation analysis, you need activation data from the Copernicus Emergency Management Service (CEMS).  
Visit the [CEMS website](https://emergency.copernicus.eu/) to download activation information (usually provided as a JSON file) that details fire events, geographic extents, and timestamps. This activation file is then used by the burned area segmentation module to guide the download and processing of satellite imagery.


#### Vineyard Classification Module

- **Purpose:**  
  Builds custom TFRecord datasets for vineyard classification.
- **Workflow:**  
  - Reads vineyard GeoJSON files to extract the polygon and the `"Classe"` property.
  - Downloads satellite images for the vineyard area over a defined temporal window (e.g., days around a fixed reference date).
  - Optionally applies normalization, masking, and cropping.
  - Extracts the vineyard label (either multiclass or binarized based on a threshold).
  - Serializes sensor data, acquisition dates, and the label into TFRecord examples.
  

---

## Interactive GUIs

The library provides two interactive dashboards:

- **Burned Area Dashboard:**  
  Located in `SatelliteDataManager/analyses/burned_area/burned_area_dashboard.py`, this dashboard allows you to authenticate with Sentinel Hub, configure dataset parameters (temporal windows, sensor selection, normalization, masking, cropping, etc.), build the dataset, and visualize generated TFRecord files.

- **Vineyard Dashboard:**  
  Located in `SatelliteDataManager/analyses/vineyard/vineyard_dashboard.py`, this dashboard offers a similar interface for configuring and visualizing vineyard datasets.

Both dashboards use ipywidgets to provide a user-friendly, interactive experience.

---

## Examples and Testing

Two example notebooks are provided:

- **test_download_code.ipynb:** Demonstrates how to invoke the library's functions directly using code.
- **test_download_gui.ipynb:** Provides interactive dashboards for building and visualizing datasets without manually writing code.

---

## Documentation

For detailed descriptions of every function and class (including parameters, return values, and usage examples), refer to the inline docstrings in each module. Additionally, please consult the [docs/usage.md](docs/usage.md) file for a comprehensive usage guide organized by module.

---

## License



---

## Contact

For questions or support, please contact [giuseppe.piparo@ct.infn.it](mailto:giuseppe.piparo@ct.infn.it).

---

## References

- **Sentinel Hub Documentation:** [https://docs.sentinel-hub.com/](https://docs.sentinel-hub.com/)
- **Copernicus Open Access Hub:** [https://scihub.copernicus.eu/dhus/#/home](https://scihub.copernicus.eu/dhus/#/home)
- **Copernicus Emergency Management Service (CEMS):** [https://emergency.copernicus.eu/](https://emergency.copernicus.eu/)

---

```