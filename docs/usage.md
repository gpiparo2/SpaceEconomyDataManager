### docs/usage.md

```markdown
# Usage Guide for SpaceEconomyDataManager

This document provides a comprehensive overview of all functions and classes in the SpaceEconomyDataManager library. The guide is organized by module, offering detailed information (derived from the docstrings) on core functionalities, burned area segmentation analysis, and vineyard classification analysis.

> **Note:** This guide is modular—you may extract or remove sections as needed for your documentation purposes.

## Table of Contents

- [Core Modules](#core-modules)
  - [DataDownload](#datadownload)
  - [DataManipulator](#datamanipulator)
  - [DatasetPreparation](#datasetpreparation)
  - [DataVisualizer](#datavisualizer)
  - [SDM](#sdm)
- [Analyses Modules](#analyses-modules)
  - [Burned Area Segmentation](#burned-area-segmentation)
  - [Vineyard Classification](#vineyard-classification)
- [Interactive GUI](#interactive-gui)
- [References](#references)

## Core Modules

### DataDownload

**Module:** `SatelliteDataManager/core/data_download.py`

- **Purpose:**  
  Handles the downloading of satellite imagery from Sentinel Hub.  
- **Key Features:**  
  - API authentication via client credentials.  
  - Setup of evalscripts (JavaScript code) for different satellites.  
  - Input validation and GeoJSON parsing to extract bounding boxes.  
  - Downloading images over specified time intervals for multiple satellites.

Refer to the docstrings in the code for detailed information on methods such as `authenticate_api()`, `set_evalscripts()`, `download_images()`, and `download_interval()`.

### DataManipulator

**Module:** `SatelliteDataManager/core/data_manipulator.py`

- **Purpose:**  
  Organizes and renames downloaded TIFF files.
- **Key Features:**  
  - Renames files using a standardized naming convention (area_satellite_fromDate_toDate.tiff).  
  - Organizes files into folders by satellite type.  
  - Prepares datasets for further processing by assembling spectral bands and associated masks.  
  - Exports data to CSV or compressed NPZ formats.

Detailed instructions for methods like `manipulate_data()`, `prepare_spectral_bands_dataset_list()`, and `store_data_as_csv()` are available in the docstrings.

### DatasetPreparation

**Module:** `SatelliteDataManager/core/dataset_preparation.py`

- **Purpose:**  
  Prepares TFRecord datasets for machine learning by processing organized satellite data.
- **Key Features:**  
  - Loading TIFF images and extracting metadata.  
  - Applying GeoJSON masks to images.  
  - Computing global quantiles for normalization.  
  - Cropping images into non-overlapping patches.  
  - Data augmentation methods for generating rotations and reflections.  
  - Serialization of sensor data and labels into TFRecord examples.

Consult the docstrings for functions such as `load_image()`, `apply_geojson_mask()`, `compute_global_quantiles()`, `global_normalize_image()`, `crop_image_to_patches()`, and `parse_dataset()`.

### DataVisualizer

**Module:** `SatelliteDataManager/core/data_visualizer.py`

- **Purpose:**  
  Provides utility functions to visualize satellite images and inspect TFRecord datasets.
- **Key Features:**  
  - Display single images or multi-band grids (with RGB composites).  
  - Apply polygon masks and calculate vegetation indices (NDVI).  
  - Inspect and visualize the contents of TFRecord files, including DEM data.
- **Special Note:**  
  The function `inspect_and_visualize_custom_tfrecord()` now supports visualizing DEM data and prints the label if it is a scalar.

### SDM

**Module:** `SatelliteDataManager/core/sdm.py`

- **Purpose:**  
  Acts as a wrapper to initialize and provide direct access to all core modules.
- **Key Features:**  
  - Instantiates and holds instances of DataDownload, DataManipulator, DataVisualizer, and DatasetPreparation.
  - Enables users to call functions directly via attributes like `sdm.data_downloader`, `sdm.data_manipulator`, etc.

## Analyses Modules

### Burned Area Segmentation

**Module:** `SatelliteDataManager/analyses/burned_area/custom_dataset_builder.py`

- **Purpose:**  
  Builds a custom TFRecord dataset for burned area (fire) segmentation.
- **Key Features:**  
  - Reads activation-specific sensor image files from organized folders.  
  - Applies a temporal sampling strategy:
    - Sentinel-2 & Sentinel-1: 3 time steps (first, median, last).
    - Sentinel-3-OLCI & Sentinel-3-SLSTR-Thermal: one image per day.
    - DEM: the central image.
  - Computes a binary label mask from a fire GeoJSON file.
  - Offers optional preprocessing such as normalization, masking, and cropping (with patch extraction).
  - Serializes all sensor data and metadata into TFRecord examples.
- **Usage Tip:**  
  For detailed instructions on obtaining activation data (e.g., from the Copernicus Emergency Management Service), refer to the [CEMS website](https://emergency.copernicus.eu/).

### Vineyard Classification

**Module:** `SatelliteDataManager/analyses/vineyard/custom_dataset_builder.py`

- **Purpose:**  
  Builds a custom TFRecord dataset for vineyard classification.
- **Key Features:**  
  - Reads vineyard information from GeoJSON files containing the polygon and the `"Classe"` property.
  - Downloads satellite imagery for the vineyard area using a specified temporal window (e.g., days before/after a fixed reference date).
  - Optionally applies normalization, masking, and cropping.
  - Extracts the vineyard label from the `"Classe"` property; the label can be maintained as multiclass or binarized based on a threshold.
  - Serializes sensor data, acquisition dates, and the label into TFRecord examples.
- **Usage Tip:**  
  Adjust the `label_threshold` parameter to control whether the classification should be binary or multiclass.

## Interactive GUI

The library also provides interactive dashboards for both burned area segmentation and vineyard classification:
- **Burned Area Dashboard:**  
  Located in `SatelliteDataManager/analyses/burned_area/burned_area_dashboard.py`. This dashboard enables users to configure download parameters, authenticate with Sentinel Hub, and trigger dataset building and visualization.
- **Vineyard Dashboard:**  
  Located in `SatelliteDataManager/analyses/vineyard/vineyard_dashboard.py`. This dashboard offers similar interactivity for vineyard dataset configuration and visualization.

## References

- **Sentinel Hub:** [Sentinel Hub Website](https://www.sentinel-hub.com/)
- **Copernicus Open Access Hub:** [Copernicus Open Access Hub](https://scihub.copernicus.eu/dhus/#/home)
- **Copernicus Emergency Management Service (CEMS):** [CEMS Website](https://emergency.copernicus.eu/)
- **Additional Documentation:** Refer to the docstrings in each module for more detailed technical information.

---

For further details on function parameters, return types, and internal logic, please refer to the inline documentation provided in the source code.