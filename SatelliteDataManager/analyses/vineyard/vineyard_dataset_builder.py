#!/usr/bin/env python3
"""
vineyard_dataset_builder.py
---------------------------------
This module defines the VineyardDatasetBuilder class, which builds a custom TFRecord dataset
for vineyard classification. For each vineyard (represented by a GeoJSON file), the builder:
  - Reads the vineyard info file (containing the polygon and the "Classe" property).
  - Downloads satellite images for the vineyard area using a specified temporal window:
      • Sentinel-2 and Sentinel-1: 3 time steps (first, median, last)
      • Sentinel-3-OLCI and Sentinel-3-SLSTR-Thermal: one image per day
      • DEM: the central image
  - Optionally applies normalization, masking, cropping, and other pre-processing.
  - Extracts a label from the "Classe" property; the label is kept as is (multiclass) or binarized
    based on a provided threshold.
  - Serializes all sensor data, dates, and the label into TFRecord examples.
  
Helper functions (e.g., parse_dates_from_filename, select_temporal_steps, select_daily_images,
select_central_date, load_image, global_normalize_image, apply_geojson_mask, crop_image_to_patches,
and TFRecord feature creators) are assumed to be available in the DatasetPreparation class.
The SDM class is assumed to manage data downloading and manipulation.
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


class VineyardDatasetBuilder:
    """
    Builds a custom TFRecord dataset for vineyard classification.
    
    For each vineyard (defined by a GeoJSON info file), the builder:
      - Reads the vineyard info file to extract the polygon and the "Classe" property.
      - Downloads satellite images for the vineyard area over a specified temporal window.
      - Optionally applies normalization, masking, and cropping.
      - Extracts a label from the "Classe" property; the label is kept as is (multiclass)
        or binarized based on a threshold.
      - Serializes sensor data and the label into TFRecord examples.
    """

    def __init__(self,
                 config: SHConfig,
                 vineyard_info_folder: str,
                 base_data_folder: str,
                 base_manipulated_folder: str,
                 tfrecord_folder: str,
                 sampleType: str = "FLOAT32",
                 download: bool = False,
                 binarize_label: bool = True,
                 label_threshold: float = 2.0):
        """
        Initializes the VineyardDatasetBuilder.
        
        Parameters:
          config (SHConfig): Sentinel Hub configuration.
          vineyard_info_folder (str): Folder containing the vineyard GeoJSON info files.
          base_data_folder (str): Folder to store raw downloaded data.
          base_manipulated_folder (str): Folder to store organized (manipulated) data.
          tfrecord_folder (str): Folder to save TFRecord files.
          sampleType (str): Data type for the output samples (default: "FLOAT32").
          download (bool): If True, downloads data for each vineyard.
          binarize_label (bool): If True, converts the "Classe" value to binary using label_threshold.
          label_threshold (float): Threshold to binarize the "Classe" property.
        """
        self.config = config
        self.vineyard_info_folder = vineyard_info_folder
        self.base_data_folder = base_data_folder
        self.base_manipulated_folder = base_manipulated_folder
        self.tfrecord_folder = tfrecord_folder
        os.makedirs(self.tfrecord_folder, exist_ok=True)
        self.sampleType = sampleType
        self.download = download
        self.binarize_label = binarize_label
        self.label_threshold = label_threshold

        # Load all vineyard GeoJSON info files.
        self.vineyard_files = sorted([os.path.join(vineyard_info_folder, f)
                                      for f in os.listdir(vineyard_info_folder)
                                      if f.endswith(".json")])
        self.processed_vineyards = []
        self.skipped_vineyards = []
        self.current_vineyard = None

    def _get_files_for_satellite(self, satellite: str, vineyard_id: str) -> list:
        """
        Retrieves all manipulated TIFF files for a given satellite and vineyard.
        
        Parameters:
          satellite (str): Name of the satellite.
          vineyard_id (str): Identifier for the vineyard.
        
        Returns:
          list: List of file paths matching the pattern in the vineyard folder.
        """
        vineyard_folder = os.path.join(self.base_manipulated_folder, vineyard_id, satellite)
        if not os.path.exists(vineyard_folder):
            return []
        pattern = os.path.join(vineyard_folder, f"{vineyard_id}_{satellite}_*.tiff")
        return glob.glob(pattern)

    def _load_dates_from_files(self, file_list: list, dp: DatasetPreparation) -> list:
        """
        Extracts acquisition dates from file names using a DatasetPreparation instance.
        
        Parameters:
          file_list (list): List of file paths.
          dp (DatasetPreparation): Instance of DatasetPreparation used for parsing dates.
        
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

    def compute_vineyard_label(self, geojson: dict) -> int:
        """
        Extracts the label from the vineyard GeoJSON.
        
        The label is derived from the "Classe" property of the first feature.
        If binarize_label is True, the label is set to 1 if Classe >= label_threshold; otherwise, 0.
        
        Parameters:
          geojson (dict): Vineyard GeoJSON.
        
        Returns:
          int: The extracted (or binarized) label.
        """
        try:
            classe_value = geojson["features"][0]["properties"]["Classe"]
        except Exception as e:
            print(f"Error extracting label from GeoJSON: {e}")
            return 0
        if self.binarize_label:
            return 1 if classe_value >= self.label_threshold else 0
        else:
            return classe_value

    def build_dataset(self, output_filename: str, vineyard_geojson_path: str, 
                      apply_normalization: bool = False, 
                      apply_mask: bool = False, 
                      mask_geojson_path: str = None, 
                      crop: bool = False, 
                      crop_factor: int = 1):
        """
        Builds a TFRecord dataset for the current vineyard by aggregating sensor data using a defined
        temporal sampling strategy and applying optional preprocessing.
        
        If cropping is enabled (crop is True and crop_factor > 1), images and labels are split into
        non-overlapping patches and each patch is stored as an individual TFRecord example.
        
        Parameters:
          output_filename (str): Name of the resulting TFRecord file.
          vineyard_geojson_path (str): Path to the vineyard GeoJSON info file.
          apply_normalization (bool): Whether to apply normalization.
          apply_mask (bool): Whether to apply a mask.
          mask_geojson_path (str): Path to the GeoJSON file to be used as a mask.
          crop (bool): Whether to apply cropping.
          crop_factor (int): Number of patches per side if cropping is enabled.
        
        Returns:
          None.
        """
        # Create an instance of DatasetPreparation for the current vineyard.
        dp = DatasetPreparation(
            data_folder=os.path.join(self.base_manipulated_folder, self.current_vineyard),
            tfrecord_folder=self.tfrecord_folder
        )
        tfrecord_path = os.path.join(self.tfrecord_folder, output_filename)
        print(f"Writing TFRecord(s) to: {tfrecord_path}")
        dataset_features = {}

        # Process time-series sensors: Sentinel-2 and Sentinel-1.
        for sensor in ["Sentinel-2", "Sentinel-1"]:
            files = self._get_files_for_satellite(sensor, self.current_vineyard)
            dates = self._load_dates_from_files(files, dp)
            if not dates:
                print(f"No files found for {sensor} in vineyard {self.current_vineyard}.")
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
            sensor_array = np.stack([entry["image"] for entry in sensor_data], axis=0)
            dates_list = [entry["date"] for entry in sensor_data]
            dataset_features[sensor] = {"dates": dates_list, "image": sensor_array}

        # Process daily sensors: Sentinel-3-OLCI and Sentinel-3-SLSTR-Thermal.
        for sensor in ["Sentinel-3-OLCI", "Sentinel-3-SLSTR-Thermal"]:
            files = self._get_files_for_satellite(sensor, self.current_vineyard)
            dates = self._load_dates_from_files(files, dp)
            if not dates:
                print(f"No files found for {sensor} in vineyard {self.current_vineyard}.")
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
            sensor_array = np.stack([entry["image"] for entry in sensor_data], axis=0)
            dates_list = [entry["date"] for entry in sensor_data]
            dataset_features[sensor] = {"dates": dates_list, "image": sensor_array}

        # Process DEM: select the central image.
        dem_files = self._get_files_for_satellite("DEM", self.current_vineyard)
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

        # Extract vineyard label from the vineyard GeoJSON.
        try:
            with open(vineyard_geojson_path, "r") as f:
                vineyard_geojson = json.load(f)
            label = self.compute_vineyard_label(vineyard_geojson)
        except Exception as e:
            print(f"Error loading vineyard GeoJSON {vineyard_geojson_path}: {e}")
            label = 0

        # --- Write TFRecord examples ---
        if not (crop and crop_factor > 1):
            feature_dict = {}
            # Serialize sensor data.
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
            # Serialize vineyard label (as an integer).
            feature_dict["vineyard_label"] = dp._int64_feature(label)
            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            writer = tf.io.TFRecordWriter(tfrecord_path)
            writer.write(example.SerializeToString())
            writer.close()
            print(f"Custom vineyard classification TFRecord created at: {tfrecord_path}")
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
                feature_dict["vineyard_label"] = dp._int64_feature(label)
                example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                writer.write(example.SerializeToString())
            writer.close()
            print(f"Custom cropped TFRecord created at: {tfrecord_path}")
        # End of build_dataset

    def build_dataset_for_all_vineyards(self,
                                        max_vineyards: int = None,
                                        download_params: dict = None,
                                        apply_normalization: bool = False,
                                        apply_mask: bool = False,
                                        crop: bool = False,
                                        crop_factor: int = 1,
                                        time_window: int = 30,
                                        sensors: list = None,
                                        sensor_bands: dict = None,
                                        evalscript_params: dict = None) -> None:
        """
        Processes all vineyard GeoJSON files and builds a TFRecord dataset for each.
        
        For each vineyard:
          - Loads the GeoJSON file to extract the polygon and the "Classe" property.
          - Downloads satellite images for the vineyard area over a temporal window defined as
            time_window days before and after a reference date (here fixed as "2024-06-19").
          - Applies per-sensor download parameters, sensor selection, and evalscript parameters.
          - Optionally applies normalization, masking, and cropping.
          - Organizes the downloaded data via DataManipulator.
          - Serializes sensor data and the label into a TFRecord.
        
        Parameters:
          max_vineyards (int, optional): Maximum number of vineyards to process.
          download_params (dict, optional): Dictionary of per-sensor download parameters.
          apply_normalization (bool): Whether to apply normalization.
          apply_mask (bool): Whether to apply masking.
          crop (bool): Whether to apply cropping.
          crop_factor (int): Crop factor (number of patches per side) if cropping is enabled.
          time_window (int): Number of days to subtract/add from the reference date.
          sensors (list): List of sensor names to process; defaults to all available sensors.
          sensor_bands (dict): Optional dictionary with per-sensor band selection parameters.
          evalscript_params (dict): Optional dictionary with per-sensor download parameters.
        
        Returns:
          None.
        """
        if sensors is None:
            sensors = ["Sentinel-2", "Sentinel-1", "Sentinel-3-OLCI", "Sentinel-3-SLSTR-Thermal", "DEM"]
        
        processed = 0
        for vineyard_file in self.vineyard_files:
            if max_vineyards is not None and processed >= max_vineyards:
                break
            try:
                with open(vineyard_file, "r") as f:
                    geojson = json.load(f)
            except Exception as e:
                print(f"Error loading vineyard file {vineyard_file}: {e}")
                self.skipped_vineyards.append(vineyard_file)
                continue

            # Extract label from the GeoJSON.
            label = self.compute_vineyard_label(geojson)
            
            # Determine a reference date (fixed for now; can be modified to extract from the file).
            reference_date = datetime.strptime("2024-06-19", '%Y-%m-%d')
            
            # Define temporal window.
            date_from = reference_date - timedelta(days=time_window)
            date_to = reference_date + timedelta(days=time_window)
            
            # Identify vineyard by filename (without extension).
            vineyard_id = os.path.splitext(os.path.basename(vineyard_file))[0]
            vineyard_raw_folder = os.path.join(self.base_data_folder, vineyard_id)
            vineyard_man_folder = os.path.join(self.base_manipulated_folder, vineyard_id)
            os.makedirs(vineyard_raw_folder, exist_ok=True)
            os.makedirs(vineyard_man_folder, exist_ok=True)
            self.current_vineyard = vineyard_id
            print(f"Processing vineyard: {vineyard_id}")

            # Configure download parameters.
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

            date_from_dict = { s: date_from for s in sensors }
            date_to_dict = { s: date_to for s in sensors }

            print("Interval days dict:", interval_days_dict)
            print("Size dict:", size_dict)
            print("Mosaicking order dict:", mosaicking_order_dict)
            print("Date from dict:", date_from_dict)
            print("Date to dict:", date_to_dict)

            # Create an SDM instance for this vineyard.
            sdm_instance = SDM(
                config=self.config,
                data_folder=vineyard_raw_folder,
                manipulated_folder=vineyard_man_folder,
                tfrecord_folder=self.tfrecord_folder
            )

            # Integrate evalscript_params and sensor_bands if provided.
            if evalscript_params is not None:
                sdm_instance.data_downloader.evalscript_params = evalscript_params
            if sensor_bands is not None:
                for sensor in sensor_bands:
                    if sensor in sdm_instance.data_downloader.evalscript_params:
                        sdm_instance.data_downloader.evalscript_params[sensor].update(sensor_bands[sensor])
                    else:
                        sdm_instance.data_downloader.evalscript_params[sensor] = sensor_bands[sensor]

            # Download images for the vineyard using the vineyard polygon.
            sdm_instance.data_downloader.download_images(
                date_from=date_from_dict,
                date_to=date_to_dict,
                interval_days=interval_days_dict,
                location_geojson=geojson,
                satellites=sensors,
                data_collection_names=None,
                size=size_dict,
                mosaicking_order=mosaicking_order_dict
            )
            print(f"Data download completed for vineyard {vineyard_id}.")

            # Organize the downloaded data.
            sdm_instance.data_manipulator.manipulate_data(area_name=vineyard_id, clean_source=False)
            
            # Build TFRecord for this vineyard.
            tfrecord_filename = f"{vineyard_id}_vineyard.tfrecord"
            try:
                self.build_dataset(tfrecord_filename, vineyard_file,
                                   apply_normalization=apply_normalization,
                                   apply_mask=apply_mask,
                                   mask_geojson_path=vineyard_file,
                                   crop=crop,
                                   crop_factor=crop_factor)
                print(f"TFRecord created for vineyard {vineyard_id} with label {label}.")
                self.processed_vineyards.append((vineyard_id, label))
            except Exception as e:
                print(f"Error processing vineyard {vineyard_id}: {e}")
                self.skipped_vineyards.append(vineyard_id)
            processed += 1
            self.current_vineyard = None
            del sdm_instance
            gc.collect()
        print("\nProcessed vineyards:")
        for vid, lab in self.processed_vineyards:
            print(f"\t {vid} -> label: {lab}")
        print("\nSkipped vineyards:")
        for vid in self.skipped_vineyards:
            print(f"\t {vid}")
