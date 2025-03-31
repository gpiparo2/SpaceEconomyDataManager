#!/usr/bin/env python3
"""
burned_area_dataset_builder.py
-------------------------
This module defines the BurnedAreaSegmentationDatasetBuilder class, which builds a custom
TFRecord dataset for burned area segmentation. For each activation, the class:
  - Reads sensor image files from activation-specific subfolders.
  - Applies a temporal sampling strategy:
      • For Sentinel-2 and Sentinel-1: selects 3 time steps (first, median, and last).
      • For Sentinel-3-OLCI and Sentinel-3-SLSTR-Thermal: selects one image per day.
      • For DEM: selects the central image.
  - Computes a binary label mask from a fire GeoJSON file.
  - Serializes all sensor data and metadata into TFRecord examples.
  
Additional features:
  - If cropping is enabled (crop_factor > 1), the code crops synchronously all sensors and the label.
    Instead of writing a single tf.train.Example per activation, the data are split into patches,
    so that each patch is stored as an individual example; this maintains a consistent input shape.
  - The get_dataset() method is modified so that in cropped mode no further unbatching is needed.
  
All functions include detailed inline comments and comprehensive docstrings.
"""

import os
import glob
import json
import numpy as np
import tensorflow as tf
import math
import gc
from datetime import datetime, timedelta
from sentinelhub import SHConfig
from rasterio.features import geometry_mask
import logging

# Suppress rasterio warnings.
logging.getLogger('rasterio').setLevel(logging.ERROR)

# Import core modules using relative imports.
from ...core.dataset_preparation import DatasetPreparation
from ...core.sdm import SDM


class BurnedAreaSegmentationDatasetBuilder:
    """
    Builds a custom TFRecord dataset for burned area segmentation.

    For each activation, the builder:
      - Reads sensor image files from organized activation subfolders.
      - Applies a temporal sampling strategy:
          • Sentinel-2 and Sentinel-1: 3 time steps (first, median, last).
          • Sentinel-3-OLCI and Sentinel-3-SLSTR-Thermal: one image per day.
          • DEM: the central image.
      - Computes a binary label mask from a fire GeoJSON.
      - Serializes all sensor data and metadata into TFRecord examples.

    Additional configuration parameters allow customization of:
      1) The temporal window (number of days before and after the fire date).
      2) The list of sensors (and optionally, their band selection parameters) to include.
      3) The download parameters (evalscript_params) for each sensor.
      4) The base folder for fire information.
    """

    def __init__(self,
                 config: SHConfig,
                 activation_info_path: str,
                 base_data_folder: str,
                 base_manipulated_folder: str,
                 tfrecord_folder: str,
                 sampleType: str = "FLOAT32",
                 download: bool = False):
        """
        Initializes the BurnedAreaSegmentationDatasetBuilder with the provided configuration and paths.

        Parameters:
          config (SHConfig): Sentinel Hub configuration.
          activation_info_path (str): Path to the JSON file containing activation information.
          base_data_folder (str): Folder containing raw downloaded data.
          base_manipulated_folder (str): Folder containing organized (manipulated) data.
          tfrecord_folder (str): Folder where TFRecord files will be saved.
          sampleType (str): Data type for the output samples (default "FLOAT32").
          download (bool): If True, triggers data download for each activation.
        """
        self.config = config
        self.activation_info_path = activation_info_path
        self.base_data_folder = base_data_folder
        self.base_manipulated_folder = base_manipulated_folder
        self.tfrecord_folder = tfrecord_folder
        os.makedirs(self.tfrecord_folder, exist_ok=True)
        self.sampleType = sampleType
        self.download = download

        # Load activation information from the provided JSON file.
        with open(self.activation_info_path) as f:
            self.activation_info = json.load(f)
        self.already_downloaded = []
        self.stored_list = []
        self.skipped_list = []
        self.current_activation = None

    def _get_files_for_satellite(self, satellite: str, activation: str) -> list:
        """
        Retrieves all organized TIFF files for a given satellite and activation.
        
        Parameters:
          satellite (str): Name of the satellite.
          activation (str): Activation identifier.
        
        Returns:
          list: List of file paths matching the activation and satellite pattern.
        """
        activation_folder = os.path.join(self.base_manipulated_folder, activation, satellite)
        if not os.path.exists(activation_folder):
            return []
        pattern = os.path.join(activation_folder, f"{activation}_{satellite}_*.tiff")
        return glob.glob(pattern)

    def _load_dates_from_files(self, file_list: list, dp: DatasetPreparation) -> list:
        """
        Extracts acquisition start dates from a list of file paths using the DatasetPreparation instance.
        
        Parameters:
          file_list (list): List of file paths.
          dp (DatasetPreparation): An instance of DatasetPreparation (for date parsing).
        
        Returns:
          list: Sorted list of datetime objects extracted from the filenames.
        """
        dates = []
        for file in file_list:
            try:
                from_date, _ = dp.parse_dates_from_filename(file)
                dates.append(from_date)
            except Exception as e:
                print(f"Error parsing date from {file}: {e}")
        return sorted(dates)

    def compute_activation_label(self, fire_geojson_path: str, reference_shape: tuple, transform) -> np.ndarray:
        """
        Computes a binary label mask for an activation based on a fire GeoJSON file.

        The resulting mask has pixels with value 1 indicating burned areas.

        Parameters:
          fire_geojson_path (str): Path to the fire GeoJSON file.
          reference_shape (tuple): The (height, width) of the reference image.
          transform: Affine transformation corresponding to the reference image.
        
        Returns:
          np.ndarray: A binary mask (dtype uint8) with shape (height, width).
        """
        try:
            with open(fire_geojson_path) as f:
                fire_geojson = json.load(f)
            mask_arr = geometry_mask(
                [feature["geometry"] for feature in fire_geojson['features']],
                invert=True,
                transform=transform,
                out_shape=reference_shape
            )
            return mask_arr.astype(np.uint8)
        except Exception as e:
            print(f"Error computing activation label from {fire_geojson_path}: {e}")
            return np.zeros(reference_shape, dtype=np.uint8)

    def build_dataset(self, output_filename: str, fire_geojson_path: str, apply_normalization: bool = False, 
                      apply_mask: bool = False, mask_geojson_path: str = None, crop: bool = False, 
                      crop_factor: int = 1):
        """
        Builds a TFRecord dataset for the current activation by aggregating sensor data using
        a defined temporal sampling strategy and applying preprocessing steps.
        
        If cropping is enabled (crop is True and crop_factor > 1), images and labels are split into patches,
        and each patch is written as an individual TFRecord example.
        
        Parameters:
          output_filename (str): The name for the resulting TFRecord file.
          fire_geojson_path (str): Path to the fire GeoJSON file used to compute the label mask.
          apply_normalization (bool): Whether to normalize the images.
          apply_mask (bool): Whether to apply an additional mask to the images.
          mask_geojson_path (str): Path to the GeoJSON file to use for masking.
          crop (bool): If True, applies cropping to images.
          crop_factor (int): Number of patches per side (if cropping is enabled).
        
        Returns:
          None.
        """
        # Create an instance of DatasetPreparation for the current activation.
        dp = DatasetPreparation(
            data_folder=os.path.join(self.base_manipulated_folder, self.current_activation),
            tfrecord_folder=self.tfrecord_folder
        )
        tfrecord_path = os.path.join(self.tfrecord_folder, output_filename)
        print(f"Writing TFRecord(s) to: {tfrecord_path}")
        dataset_features = {}

        # Process time-series sensors: Sentinel-2 and Sentinel-1.
        for sensor in ["Sentinel-2", "Sentinel-1"]:
            files = self._get_files_for_satellite(sensor, self.current_activation)
            dates = self._load_dates_from_files(files, dp)
            if not dates:
                print(f"No files found for {sensor} in activation {self.current_activation}.")
                continue
            selected_dates = dp.select_temporal_steps(dates, n_steps=3)
            sensor_data = []
            if apply_normalization:
                global_q_low, global_q_up = dp.get_global_quantiles_for_sensor(sensor)
            for sel_date in selected_dates:
                closest_file = min(files, key=lambda f: abs(dp.parse_dates_from_filename(f)[0] - sel_date))
                image, meta = dp.load_image(closest_file)
                if apply_normalization:
                    image = dp.global_normalize_image(image, global_q_low, global_q_up)
                if apply_mask and mask_geojson_path:
                    transform = meta.get("transform", None)
                    image = dp.apply_geojson_mask(image, mask_geojson_path, transform)
                if crop and crop_factor > 1:
                    image = dp.crop_image_to_patches(image, crop_factor)
                sensor_data.append({
                    "date": dp.parse_dates_from_filename(closest_file)[0].strftime("%Y-%m-%d"),
                    "image": image,
                    "metadata": meta
                })
            if crop and crop_factor > 1:
                sensor_array = np.stack([entry["image"] for entry in sensor_data], axis=0)
                dates_list = [entry["date"] for entry in sensor_data]
                dataset_features[sensor] = {"dates": dates_list, "image": sensor_array}
            else:
                sensor_array = np.stack([entry["image"] for entry in sensor_data], axis=0)
                dates_list = [entry["date"] for entry in sensor_data]
                dataset_features[sensor] = {"dates": dates_list, "image": sensor_array}

        # Process daily sensors: Sentinel-3-OLCI and Sentinel-3-SLSTR-Thermal.
        for sensor in ["Sentinel-3-OLCI", "Sentinel-3-SLSTR-Thermal"]:
            files = self._get_files_for_satellite(sensor, self.current_activation)
            dates = self._load_dates_from_files(files, dp)
            if not dates:
                print(f"No files found for {sensor} in activation {self.current_activation}.")
                continue
            daily_dates = dp.select_daily_images(dates)
            sensor_data = []
            if apply_normalization:
                global_q_low, global_q_up = dp.get_global_quantiles_for_sensor(sensor)
            for day in daily_dates:
                closest_file = min(files, key=lambda f: abs(dp.parse_dates_from_filename(f)[0] - day))
                image, meta = dp.load_image(closest_file)
                if apply_normalization:
                    image = dp.global_normalize_image(image, global_q_low, global_q_up)
                if apply_mask and mask_geojson_path:
                    transform = meta.get("transform", None)
                    image = dp.apply_geojson_mask(image, mask_geojson_path, transform)
                if crop and crop_factor > 1:
                    image = dp.crop_image_to_patches(image, crop_factor)
                sensor_data.append({
                    "date": dp.parse_dates_from_filename(closest_file)[0].strftime("%Y-%m-%d"),
                    "image": image,
                    "metadata": meta
                })
            if crop and crop_factor > 1:
                sensor_array = np.stack([entry["image"] for entry in sensor_data], axis=0)
                dates_list = [entry["date"] for entry in sensor_data]
                dataset_features[sensor] = {"dates": dates_list, "image": sensor_array}
            else:
                sensor_array = np.stack([entry["image"] for entry in sensor_data], axis=0)
                dates_list = [entry["date"] for entry in sensor_data]
                dataset_features[sensor] = {"dates": dates_list, "image": sensor_array}

        # Process DEM: select the central image.
        dem_files = self._get_files_for_satellite("DEM", self.current_activation)
        dem_data = None
        dem_dates = self._load_dates_from_files(dem_files, dp)
        if dem_dates:
            central_date = dp.select_central_date(dem_dates)
            dem_file = min(dem_files, key=lambda f: abs(dp.parse_dates_from_filename(f)[0] - central_date))
            image, meta = dp.load_image(dem_file)
            if apply_normalization:
                global_q_low, global_q_up = dp.get_global_quantiles_for_sensor("DEM")
                image = dp.global_normalize_image(image, global_q_low, global_q_up)
            if apply_mask and mask_geojson_path:
                transform = meta.get("transform", None)
                image = dp.apply_geojson_mask(image, mask_geojson_path, transform)
            if crop and crop_factor > 1:
                image = dp.crop_image_to_patches(image, crop_factor)
            dem_data = {"date": dp.parse_dates_from_filename(dem_file)[0].strftime("%Y-%m-%d"),
                        "image": image,
                        "metadata": meta}
        dataset_features["DEM"] = dem_data

        # Compute the activation label using a reference Sentinel-2 image.
        ref_files = self._get_files_for_satellite("Sentinel-2", self.current_activation)
        if ref_files:
            ref_image, ref_meta = dp.load_image(ref_files[0])
            ref_shape = (ref_image.shape[0], ref_image.shape[1])
            ref_transform = ref_meta.get("transform")
            activation_label = self.compute_activation_label(fire_geojson_path, ref_shape, ref_transform)
        else:
            activation_label = np.zeros((512, 512), dtype=np.uint8)
        if crop and crop_factor > 1:
            activation_label = dp.crop_image_to_patches(np.expand_dims(activation_label, axis=-1), crop_factor)

        # --- Serialize and write TFRecord examples ---
        if not (crop and crop_factor > 1):
            feature_dict = {}
            for sensor in ["Sentinel-2", "Sentinel-1", "Sentinel-3-OLCI", "Sentinel-3-SLSTR-Thermal"]:
                if sensor in dataset_features:
                    sensor_array = dataset_features[sensor]["image"]
                    image_bytes = sensor_array.tobytes()
                    feature_dict[f"{sensor}_dates"] = dp._bytes_feature(json.dumps(dataset_features[sensor]["dates"]).encode('utf-8'))
                    feature_dict[f"{sensor}_image"] = dp._bytes_feature(image_bytes)
                    feature_dict[f"{sensor}_height"] = dp._int64_feature(sensor_array.shape[1])
                    feature_dict[f"{sensor}_width"] = dp._int64_feature(sensor_array.shape[2])
                    feature_dict[f"{sensor}_channels"] = dp._int64_feature(sensor_array.shape[3])
                    feature_dict[f"{sensor}_n_steps"] = dp._int64_feature(sensor_array.shape[0])
            if dem_data is not None:
                feature_dict["DEM_image"] = dp._bytes_feature(dem_data["image"].tobytes())
                feature_dict["DEM_height"] = dp._int64_feature(dem_data["image"].shape[0])
                feature_dict["DEM_width"] = dp._int64_feature(dem_data["image"].shape[1])
                feature_dict["DEM_channels"] = dp._int64_feature(dem_data["image"].shape[2])
            feature_dict["activation_label"] = dp._bytes_feature(activation_label.tobytes())
            feature_dict["activation_label_height"] = dp._int64_feature(activation_label.shape[0])
            feature_dict["activation_label_width"] = dp._int64_feature(activation_label.shape[1])
            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            writer = tf.io.TFRecordWriter(tfrecord_path)
            writer.write(example.SerializeToString())
            writer.close()
            print(f"Custom burned area segmentation TFRecord created at: {tfrecord_path}")
        else:
            n_patches = dataset_features["Sentinel-2"]["image"].shape[1]
            writer = tf.io.TFRecordWriter(tfrecord_path)
            for i in range(n_patches):
                feature_dict = {}
                for sensor in ["Sentinel-2", "Sentinel-1", "Sentinel-3-OLCI", "Sentinel-3-SLSTR-Thermal"]:
                    if sensor in dataset_features:
                        sensor_array = dataset_features[sensor]["image"]
                        patch_image = sensor_array[:, i, :, :, :]
                        image_bytes = patch_image.tobytes()
                        feature_dict[f"{sensor}_dates"] = dp._bytes_feature(json.dumps(dataset_features[sensor]["dates"]).encode('utf-8'))
                        feature_dict[f"{sensor}_image"] = dp._bytes_feature(image_bytes)
                        feature_dict[f"{sensor}_height"] = dp._int64_feature(patch_image.shape[1])
                        feature_dict[f"{sensor}_width"] = dp._int64_feature(patch_image.shape[2])
                        feature_dict[f"{sensor}_channels"] = dp._int64_feature(patch_image.shape[3])
                        feature_dict[f"{sensor}_n_steps"] = dp._int64_feature(patch_image.shape[0])
                if dem_data is not None:
                    patch_dem = dem_data["image"][i, :, :, :]
                    feature_dict["DEM_image"] = dp._bytes_feature(patch_dem.tobytes())
                    feature_dict["DEM_height"] = dp._int64_feature(patch_dem.shape[0])
                    feature_dict["DEM_width"] = dp._int64_feature(patch_dem.shape[1])
                    feature_dict["DEM_channels"] = dp._int64_feature(patch_dem.shape[2])
                patch_label = activation_label[i, :, :, :]
                feature_dict["activation_label"] = dp._bytes_feature(patch_label.tobytes())
                feature_dict["activation_label_height"] = dp._int64_feature(patch_label.shape[0])
                feature_dict["activation_label_width"] = dp._int64_feature(patch_label.shape[1])
                example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                writer.write(example.SerializeToString())
            writer.close()
            print(f"Custom cropped TFRecord created at: {tfrecord_path}")
        # End of build_dataset

    def build_dataset_for_all_activations(self,
                                          max_activations: int = None,
                                          use_skip_list: bool = False,
                                          start_activation: str = None,
                                          download_params: dict = None,
                                          apply_normalization: bool = False,
                                          apply_mask: bool = False,
                                          crop: bool = False,
                                          crop_factor: int = 1,
                                          time_window: int = 30,
                                          sensors: list = None,
                                          sensor_bands: dict = None,
                                          evalscript_params: dict = None,
                                          fire_info_base_folder: str = "../") -> None:
        """
        Processes all activations listed in the activation info file and builds a TFRecord dataset for each.
        
        For each activation, the builder:
          - Reads the corresponding fire area and event GeoJSON files.
          - Downloads sensor data using a temporal window defined around the fire date.
          - Applies the specified preprocessing steps (normalization, masking, cropping).
          - Organizes and serializes the sensor data and the activation label into TFRecord examples.
        
        Additional parameters allow customization of:
          - The temporal window (number of days to subtract/add from the fire date).
          - The list of sensors to include.
          - Per-sensor band selection parameters.
          - Per-sensor download (evalscript) parameters.
          - The base folder containing fire information.
        
        Parameters:
          max_activations (int, optional): Maximum number of activations to process.
          use_skip_list (bool, optional): If True, uses a predefined list of activations to skip.
          start_activation (str, optional): Activation ID from which to begin processing.
          download_params (dict, optional): Dictionary of per-sensor download parameters.
          apply_normalization (bool): Whether to apply normalization.
          apply_mask (bool): Whether to apply masking.
          crop (bool): Whether to apply cropping.
          crop_factor (int): Crop factor (number of patches per side) if cropping is enabled.
          time_window (int): Number of days to subtract/add from the fire date.
          sensors (list): List of sensor names to process.
          sensor_bands (dict): Optional per-sensor band selection parameters.
          evalscript_params (dict): Optional per-sensor download parameters.
          fire_info_base_folder (str): Base folder where fire information is stored.
        
        Returns:
          None.
        """
        skip_list = ["EMSR390-AOI01"]
        activation_list = []
        # Build a list of activation IDs from the activation info JSON.
        for key, value in self.activation_info.items():
            for subkey in value.keys():
                activation_list.append(f"{key}-{subkey}")
        activation_list = sorted(activation_list)
        if start_activation:
            activation_list = [act for act in activation_list if act >= start_activation]
        
        if sensors is None:
            sensors = ["Sentinel-2", "Sentinel-1", "Sentinel-3-OLCI", "Sentinel-3-SLSTR-Thermal", "DEM"]
        
        processed = 0
        for activation in activation_list:
            try:
                key, subkey = activation.split('-', 1)
                subvalue = self.activation_info[key][subkey]
            except Exception as e:
                print(f"Error parsing activation ID {activation}: {e}")
                continue
            if use_skip_list and activation in skip_list:
                print(f"Activation {activation} is in the skip list... skipping")
                continue
            if activation in self.already_downloaded:
                print(f"Activation {activation} already processed... skipping")
                continue

            activation_raw_folder = os.path.join(self.base_data_folder, activation, "raw")
            activation_manipulated_folder = os.path.join(self.base_manipulated_folder, activation)
            os.makedirs(activation_raw_folder, exist_ok=True)
            os.makedirs(activation_manipulated_folder, exist_ok=True)
            self.current_activation = activation
            print(f"Processing activation: {activation}")

            # Determine paths for the area and fire GeoJSON files based on the activation information.
            if subvalue['report'] != "none":
                geojson_path = os.path.join(fire_info_base_folder, "FireData", key,
                                            f"{key}_{subkey}_GRA_PRODUCT_{subvalue['report']}_VECTORS_{subvalue['version']}_vector",
                                            f"{key}_{subkey}_GRA_PRODUCT_areaOfInterestA_{subvalue['report']}_{subvalue['version']}.json")
                fire_geojson_path = os.path.join(fire_info_base_folder, "FireData", key,
                                                f"{key}_{subkey}_GRA_PRODUCT_{subvalue['report']}_VECTORS_{subvalue['version']}_vector",
                                                f"{key}_{subkey}_GRA_PRODUCT_observedEventA_{subvalue['report']}_{subvalue['version']}.json")
            else:
                geojson_path = os.path.join(fire_info_base_folder, "FireData", key,
                                            f"{key}_{subkey}_GRA_PRODUCT_{subvalue['version']}",
                                            f"{key}_{subkey}_GRA_PRODUCT_areaOfInterestA_{subvalue['version']}.json")
                fire_geojson_path = os.path.join(fire_info_base_folder, "FireData", key,
                                                f"{key}_{subkey}_GRA_PRODUCT_{subvalue['version']}",
                                                f"{key}_{subkey}_GRA_PRODUCT_observedEventA_{subvalue['version']}.json")
            if not os.path.exists(geojson_path):
                print(f"Skipping activation {activation}: Area GeoJSON not found.")
                self.skipped_list.append(activation)
                continue
            if not os.path.exists(fire_geojson_path):
                print(f"Skipping activation {activation}: Fire GeoJSON not found.")
                self.skipped_list.append(activation)
                continue

            # Import SDM class from the core module.
            sdm_instance = SDM(
                config=self.config,
                data_folder=activation_raw_folder,
                manipulated_folder=activation_manipulated_folder,
                tfrecord_folder=self.tfrecord_folder
            )
            try:
                fire_date = datetime.strptime(subvalue['date'], '%Y-%m-%d')
            except Exception as e:
                print(f"Error parsing fire date for activation {activation}: {e}")
                self.skipped_list.append(activation)
                continue

            sat_date_from = fire_date - timedelta(days=time_window)
            sat_date_to = fire_date + timedelta(days=time_window)

            if download_params:
                interval_days_dict = { s: download_params.get("interval_days", {}).get(s, 20) for s in sensors }
                size_dict = { s: download_params.get("size", {}).get(s, (128, 128)) for s in sensors }
                mosaicking_order_dict = { s: download_params.get("mosaicking_order", {}).get(s, "mostRecent") for s in sensors }
                resolutions = download_params.get("resolutions", {
                    "Sentinel-2": 10,
                    "Sentinel-1": 10,
                    "Sentinel-3-OLCI": 300,
                    "Sentinel-3-SLSTR-Thermal": 1000,
                    "DEM": 500
                })
            else:
                interval_days_dict = { s: 20 for s in sensors }
                size_dict = { s: (128, 128) for s in sensors }
                mosaicking_order_dict = { s: "mostRecent" for s in sensors }
                resolutions = {"Sentinel-2": 10, "Sentinel-1": 10,
                               "Sentinel-3-OLCI": 300, "Sentinel-3-SLSTR-Thermal": 1000,
                               "DEM": 500}

            date_from_dict = { s: sat_date_from for s in sensors }
            date_to_dict = { s: sat_date_to for s in sensors }

            print("Interval days dict:", interval_days_dict)
            print("Size dict:", size_dict)
            print("Mosaicking order dict:", mosaicking_order_dict)
            print("Date from dict:", date_from_dict)
            print("Date to dict:", date_to_dict)

            with open(geojson_path) as geo_f:
                location_geojson = json.load(geo_f)

            if evalscript_params is not None:
                sdm_instance.data_downloader.evalscript_params = evalscript_params
            if sensor_bands is not None:
                for sensor in sensor_bands:
                    if sensor in sdm_instance.data_downloader.evalscript_params:
                        sdm_instance.data_downloader.evalscript_params[sensor].update(sensor_bands[sensor])
                    else:
                        sdm_instance.data_downloader.evalscript_params[sensor] = sensor_bands[sensor]

            # Download sensor images for the activation.
            sdm_instance.data_downloader.download_images(
                date_from=date_from_dict,
                date_to=date_to_dict,
                interval_days=interval_days_dict,
                location_geojson=location_geojson,
                satellites=sensors,
                data_collection_names=None,
                size=size_dict,
                mosaicking_order=mosaicking_order_dict
            )
            print(f"Data download completed for activation {activation}.")

            # Organize downloaded data into satellite-specific folders.
            sdm_instance.data_manipulator.manipulate_data(area_name=activation, clean_source=False)
            output_filename = f"{activation}_custom.tfrecord"
            try:
                self.build_dataset(output_filename, fire_geojson_path, apply_normalization, apply_mask, geojson_path, crop, crop_factor)
                self.stored_list.append(activation)
                self.already_downloaded.append(activation)
            except Exception as e:
                print(f"Error processing activation {activation}: {e}")
                self.skipped_list.append(activation)
            processed += 1
            self.current_activation = None
            del sdm_instance
            gc.collect()
            if max_activations is not None and processed >= max_activations:
                print("Reached maximum activations limit.")
                return
        print("\nStored activations:")
        for act in self.stored_list:
            print("\t", act)
        print("\nSkipped activations:")
        for act in self.skipped_list:
            print("\t", act)
