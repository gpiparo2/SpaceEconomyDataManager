#!/usr/bin/env python3
"""
et_dataset_builder.py
---------------------------------
This module defines the ETDatasetBuilder class, which builds a custom TFRecord dataset
for evapotranspiration estimation. For each ET chunk (defined in the Excel file):
  - Reads the ET Excel file to obtain the start date, end date, and TotalET value.
  - Downloads satellite images for the specified period:
      • Sentinel-2 and Sentinel-1: 1 time step (e.g., the median image)
      • Sentinel-3-OLCI and Sentinel-3-SLSTR-Thermal: one image per day
      • DEM: the central image
  - Optionally applies normalization, masking, cropping, and other pre-processing.
  - Serializes all sensor data, dates, the period information, and the ET value into TFRecord examples.
  
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
import pandas as pd

# Suppress rasterio warnings.
logging.getLogger('rasterio').setLevel(logging.ERROR)

# Import core modules using relative imports.
from ....core.dataset_preparation import DatasetPreparation
from ....core.sdm import SDM


class ETDatasetBuilder:
    """
    Builds a custom TFRecord dataset for evapotranspiration estimation.
    
    For each ET chunk (defined in the Excel file):
      - Reads the start date, end date, and TotalET value.
      - Downloads satellite images for the specified period.
      - Optionally applies normalization, masking, and cropping.
      - Serializes sensor data, period information, and the ET value into a TFRecord example.
    """
    def __init__(self,
                 config: SHConfig,
                 et_excel_file: str,
                 base_data_folder: str,
                 base_manipulated_folder: str,
                 tfrecord_folder: str,
                 sampleType: str = "FLOAT32",
                 download: bool = True):
        """
        Initializes the ETDatasetBuilder.
        
        Parameters:
          config (SHConfig): Sentinel Hub configuration.
          et_excel_file (str): Path to the ET Excel file (with columns StartDate, EndDate, TotalET).
          base_data_folder (str): Folder to store raw satellite data.
          base_manipulated_folder (str): Folder to store manipulated data.
          tfrecord_folder (str): Folder to save TFRecord files.
          sampleType (str): Data type for the output samples (default: "FLOAT32").
          download (bool): If True, downloads the satellite data.
        """
        self.config = config
        self.et_excel_file = et_excel_file
        self.base_data_folder = base_data_folder
        self.base_manipulated_folder = base_manipulated_folder
        self.tfrecord_folder = tfrecord_folder
        os.makedirs(self.tfrecord_folder, exist_ok=True)
        self.sampleType = sampleType
        self.download = download

        # Load ET data from the Excel file (assumes columns: StartDate, EndDate, TotalET)
        self.et_data = pd.read_excel(self.et_excel_file)
        # Convert date columns to datetime (date only)
        self.et_data['StartDate'] = pd.to_datetime(self.et_data['StartDate']).dt.date
        self.et_data['EndDate'] = pd.to_datetime(self.et_data['EndDate']).dt.date

        self.processed_chunks = []
        self.skipped_chunks = []
        self.current_chunk = None

    def _get_files_for_satellite(self, satellite: str, chunk_id: str) -> list:
        """
        Retrieves all manipulated TIFF files for a given sensor and chunk.
        
        Parameters:
          satellite (str): Sensor name.
          chunk_id (str): Identifier for the chunk (e.g., et_20250310_20250316).
          
        Returns:
          list: List of file paths matching the pattern in the chunk folder.
        """
        chunk_folder = os.path.join(self.base_manipulated_folder, chunk_id, satellite)
        if not os.path.exists(chunk_folder):
            return []
        pattern = os.path.join(chunk_folder, f"{chunk_id}_{satellite}_*.tiff")
        return glob.glob(pattern)

    def _load_dates_from_files(self, file_list: list, dp: DatasetPreparation) -> list:
        """
        Extracts acquisition dates from filenames using a DatasetPreparation instance.
        
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
                print(f"Error extracting date from {file}: {e}")
        return sorted(dates)

    def build_dataset(self,
                    chunk_id: str,
                    start_date,
                    end_date,
                    et_label: float,
                    apply_normalization: bool = False,
                    apply_mask: bool = False,
                    mask_geojson_path: str = None,
                    crop: bool = False,
                    crop_factor: int = 1):
        """
        Builds a TFRecord dataset for an ET chunk by aggregating satellite data
        using a defined temporal sampling strategy and applying optional preprocessing.
        
        For the sensors:
        - Sentinel-2 and Sentinel-1: three images are selected (e.g., first, median and last).
        - Sentinel-3-OLCI and Sentinel-3-SLSTR-Thermal: daily images are selected (e.g., 7 images for one week).
        - (Optional) DEM: the central image is selected.
        
        Parameters:
        chunk_id (str): Identifier for the ET chunk.
        start_date: Start date of the period.
        end_date: End date of the period.
        et_label (float): ET value to associate (label).
        apply_normalization (bool): If True, applies normalization.
        apply_mask (bool): If True, applies a mask.
        mask_geojson_path (str): Path to the GeoJSON file for masking (if needed).
        crop (bool): If True, applies cropping into patches.
        crop_factor (int): Number of patches per side if cropping is enabled.
        """
        dp = DatasetPreparation(
            data_folder=os.path.join(self.base_manipulated_folder, chunk_id),
            tfrecord_folder=self.tfrecord_folder
        )
        tfrecord_path = os.path.join(self.tfrecord_folder, f"{chunk_id}_et.tfrecord")
        print(f"Writing TFRecord to: {tfrecord_path}")
        dataset_features = {}

        # Process time-series sensors: Sentinel-2 and Sentinel-1 (three images selected).
        for sensor in ["Sentinel-2", "Sentinel-1"]:
            files = self._get_files_for_satellite(sensor, chunk_id)
            dates = self._load_dates_from_files(files, dp)
            if not dates:
                print(f"No files found for {sensor} in chunk {chunk_id}.")
                continue
            # Use 3 temporal steps (instead of 1) as per the vineyard builder.
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
            files = self._get_files_for_satellite(sensor, chunk_id)
            dates = self._load_dates_from_files(files, dp)
            if not dates:
                print(f"No files found for {sensor} in chunk {chunk_id}.")
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
        dem_files = self._get_files_for_satellite("DEM", chunk_id)
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
                dem_image = dem_data["image"]
                feature_dict["DEM_image"] = dp._bytes_feature(dem_image.tobytes())
                feature_dict["DEM_height"] = dp._int64_feature(dem_image.shape[0])
                feature_dict["DEM_width"] = dp._int64_feature(dem_image.shape[1])
                feature_dict["DEM_channels"] = dp._int64_feature(dem_image.shape[2])
            feature_dict["ET_label"] = dp._float_feature([et_label])
            period_info = {"StartDate": str(start_date), "EndDate": str(end_date)}
            feature_dict["period_info"] = dp._bytes_feature(json.dumps(period_info).encode('utf-8'))
            
            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            writer = tf.io.TFRecordWriter(tfrecord_path)
            writer.write(example.SerializeToString())
            writer.close()
            print(f"ET TFRecord created: {tfrecord_path}")
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
                feature_dict["ET_label"] = dp._float_feature([et_label])
                period_info = {"StartDate": str(start_date), "EndDate": str(end_date)}
                feature_dict["period_info"] = dp._bytes_feature(json.dumps(period_info).encode('utf-8'))
                
                example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                writer.write(example.SerializeToString())
            writer.close()
            print(f"Custom cropped TFRecord created at: {tfrecord_path}")


    def build_dataset_for_all_chunks(self,
                                    download_params: dict = None,
                                    apply_normalization: bool = False,
                                    apply_mask: bool = False,
                                    crop: bool = False,
                                    crop_factor: int = 1,
                                    sensors: list = None,
                                    roi_geojson_path: str = None,
                                    sensor_bands: dict = None,
                                    evalscript_params: dict = None):
        """
        Processes all ET chunks from the Excel file and builds a TFRecord dataset for each.
        
        For each row in the Excel file:
        - Determines the time period and the ET value.
        - Configures the raw and manipulated folders for the chunk.
        - Optionally downloads the satellite data using SDM and organizes it.
        - Builds the TFRecord associating the satellite image set with the ET value.
        
        Additional parameters:
        sensors (list): List of sensor names to process; if None, a default list is used.
        roi_geojson_path (str): Path to a JSON file containing the region coordinates.
                                If provided, its content is used as the download area.
        download_params (dict): If provided, this dictionary is used to override the default download parameters.
        sensor_bands (dict): Optional dictionary with per-sensor band selection parameters.
        evalscript_params (dict): Optional dictionary with per-sensor download parameters.
        """
        if sensors is None:
            sensors = ["Sentinel-2", "Sentinel-1", "Sentinel-3-OLCI", "Sentinel-3-SLSTR-Thermal", "DEM"]

        # Load ROI GeoJSON if provided.
        if roi_geojson_path and os.path.exists(roi_geojson_path):
            with open(roi_geojson_path, "r") as f:
                roi_geojson = json.load(f)
        else:
            roi_geojson = None

        processed = 0
        for idx, row in self.et_data.iterrows():
            row_start = row['StartDate']
            row_end = row['EndDate']
            et_value = row['TotalET']
            chunk_id = f"et_{row_start.strftime('%Y%m%d')}_{row_end.strftime('%Y%m%d')}"
            print(f"\nProcessing chunk {chunk_id} with ET: {et_value}")
            
            chunk_raw_folder = os.path.join(self.base_data_folder, chunk_id)
            chunk_man_folder = os.path.join(self.base_manipulated_folder, chunk_id)
            os.makedirs(chunk_raw_folder, exist_ok=True)
            os.makedirs(chunk_man_folder, exist_ok=True)
            self.current_chunk = chunk_id

            # Calcola i parametri di download per il chunk in base alle date.
            if download_params is None:
                curr_interval = {}
                for s in sensors:
                    if s in ["Sentinel-2", "Sentinel-1"]:
                        curr_interval[s] = (row_end - row_start).days
                    elif s in ["Sentinel-3-OLCI", "Sentinel-3-SLSTR-Thermal"]:
                        curr_interval[s] = 1
                    else:
                        curr_interval[s] = (row_end - row_start).days
                curr_size = { s: (128, 128) for s in sensors }
                curr_mosaicking = { s: "mostRecent" for s in sensors }
            else:
                curr_interval = { s: download_params.get("interval_days", {}).get(s, (row_end - row_start).days)
                                for s in sensors }
                curr_size = { s: download_params.get("size", {}).get(s, (128, 128)) for s in sensors }
                curr_mosaicking = { s: download_params.get("mosaicking_order", {}).get(s, "mostRecent")
                                    for s in sensors }

            # Imposta i dizionari delle date come oggetti datetime.
            date_from_dict = { s: datetime.combine(row_start, datetime.min.time()) for s in sensors }
            date_to_dict = { s: datetime.combine(row_end, datetime.min.time()) for s in sensors }

            print("Download parameters:")
            print("  Interval days:", curr_interval)
            print("  Size:", curr_size)
            print("  Mosaicking order:", curr_mosaicking)
            print("  Date from:", date_from_dict)
            print("  Date to:", date_to_dict)

            sdm_instance = SDM(
                config=self.config,
                data_folder=chunk_raw_folder,
                manipulated_folder=chunk_man_folder,
                tfrecord_folder=self.tfrecord_folder
            )
            # Integra evalscript_params e sensor_bands se forniti.
            if evalscript_params is not None:
                sdm_instance.data_downloader.evalscript_params = evalscript_params
            if sensor_bands is not None:
                for sensor in sensor_bands:
                    if sensor in sdm_instance.data_downloader.evalscript_params:
                        sdm_instance.data_downloader.evalscript_params[sensor].update(sensor_bands[sensor])
                    else:
                        sdm_instance.data_downloader.evalscript_params[sensor] = sensor_bands[sensor]

            sdm_instance.data_downloader.download_images(
                date_from=date_from_dict,
                date_to=date_to_dict,
                interval_days=curr_interval,
                location_geojson=roi_geojson,
                satellites=sensors,
                data_collection_names=None,
                size=curr_size,
                mosaicking_order=curr_mosaicking
            )
            print(f"Download completed for chunk {chunk_id}.")

            sdm_instance.data_manipulator.manipulate_data(area_name=chunk_id, clean_source=False)
            
            try:
                self.build_dataset(chunk_id, row_start, row_end, et_value,
                                apply_normalization=apply_normalization,
                                apply_mask=apply_mask,
                                mask_geojson_path=None,
                                crop=crop,
                                crop_factor=crop_factor)
                print(f"TFRecord created for chunk {chunk_id} with ET: {et_value}")
                self.processed_chunks.append((chunk_id, et_value))
            except Exception as e:
                print(f"Error processing chunk {chunk_id}: {e}")
                self.skipped_chunks.append(chunk_id)
            processed += 1
            self.current_chunk = None
            del sdm_instance
            gc.collect()

        print("\nProcessed chunks:")
        for cid, et_val in self.processed_chunks:
            print(f"\t {cid} -> ET: {et_val}")
        print("\nSkipped chunks:")
        for cid in self.skipped_chunks:
            print(f"\t {cid}")


