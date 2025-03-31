#!/usr/bin/env python3
"""
dataset_preparation.py
----------------------
This module defines the DatasetPreparation class, which processes organized satellite data
and creates TFRecord datasets for machine learning applications. It supports a range of tasks including:
  - Selecting satellites, spectral bands, and temporal steps.
  - (Optionally) Computing vegetation indices.
  - Data augmentation, normalization, and image cropping.
  - Label assignment (currently a placeholder).

Additional features:
  1. Ability to apply a GeoJSON mask to images based on polygon coordinates.
  2. A normalization function that scales each band using the 99th percentile to mitigate outliers.
  3. A data augmentation method that generates all unique rotations and reflections of an image.

TFRecord examples include both the image data and associated metadata.
All functions are documented with detailed inline comments and professional docstrings.
"""

import os
import glob
import json
import numpy as np
import tensorflow as tf
import rasterio
from datetime import datetime
import math
from affine import Affine
import geopandas as gpd
from typing import Tuple, List, Optional

class DatasetPreparation:
    """
    A class for preparing TFRecord datasets from organized satellite imagery.
    
    This class provides generic methods for:
      - Loading TIFF images and extracting metadata.
      - Applying GeoJSON masks to images.
      - Computing global quantiles across datasets for normalization.
      - Cropping images into non-overlapping patches.
      - Serializing sensor data and labels into TFRecord examples.
      
    It also offers several utility (static) methods for date parsing, data augmentation, and TFRecord feature creation.
    """
    
    def __init__(self, data_folder: str = "../data_off", tfrecord_folder: str = "../tfrecords"):
        """
        Initializes the DatasetPreparation object with the specified data and output folders.
        
        Parameters:
          data_folder (str): Directory containing the organized (manipulated) data.
          tfrecord_folder (str): Directory where TFRecord files will be saved.
        """
        self.data_folder = data_folder
        self.tfrecord_folder = tfrecord_folder
        os.makedirs(self.tfrecord_folder, exist_ok=True)
        # Cache for storing computed global quantiles per sensor to avoid redundant calculations.
        self.global_quantiles: dict = {}

    def load_image(self, filepath: str) -> Tuple[np.ndarray, dict]:
        """
        Loads a TIFF image and retrieves its metadata.
        
        Parameters:
          filepath (str): Path to the TIFF file.
        
        Returns:
          tuple: A tuple containing:
                 - image: A NumPy array of shape (height, width, nbands) representing the image.
                 - meta: A dictionary containing the metadata from the raster file.
        """
        with rasterio.open(filepath) as src:
            image = src.read()
            # Convert image from (bands, H, W) to (H, W, bands) for easier processing.
            image = np.moveaxis(image, 0, -1)
            meta = src.meta
        return image, meta

    # Import Affine from the affine package (for use in the apply_geojson_mask method)
    from affine import Affine

    def apply_geojson_mask(self, image: np.ndarray, geojson_path: str, transform: Optional[Affine] = None, target_crs: Optional[str] = None) -> np.ndarray:
        """
        Applies a polygon mask to an image based on a GeoJSON file.
        Pixels outside the specified polygon(s) are set to zero.
        
        Parameters:
            image (np.ndarray): Input image array with shape (height, width, channels).
            geojson_path (str): Path to the GeoJSON file containing polygon geometries.
            transform (Affine, optional): Affine transform mapping image pixel coordinates to the coordinate system of the mask.
                                          If None, an identity transform is assumed.
            target_crs (str, optional): The coordinate reference system (e.g., "EPSG:4326") for the image.
                                        If provided and differing from the GeoJSON's CRS, the geometries will be reprojected.
        
        Returns:
            np.ndarray: The masked image where pixels outside the polygon are set to zero.
        """
        import geopandas as gpd
        from rasterio.features import geometry_mask
        from affine import Affine

        # Load the GeoJSON data as a GeoDataFrame.
        gdf = gpd.read_file(geojson_path)
        
        # Reproject geometries if a target CRS is specified and differs from the GeoJSON's CRS.
        if target_crs is not None:
            if gdf.crs is None:
                gdf.set_crs(target_crs, inplace=True)
            elif gdf.crs.to_string() != target_crs:
                gdf = gdf.to_crs(target_crs)
        
        # Get image dimensions.
        height, width = image.shape[:2]
        
        # If no affine transform is provided, use the identity transform (assuming pixel coordinates).
        if transform is None:
            transform = Affine.identity()
        
        # Create a boolean mask where pixels inside the polygon are True.
        mask = geometry_mask(
            [geom for geom in gdf.geometry],
            transform=transform,
            invert=True,
            out_shape=(height, width)
        )
        
        # Apply the mask to each channel of the image.
        if image.ndim == 3:
            masked_image = image * mask[..., None]
        else:
            masked_image = image * mask
        return masked_image

    def compute_global_quantiles(self, images: List[np.ndarray], quantile_low: float = 1, quantile_up: float = 99) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the lower and upper quantiles for each band across a list of images.
        
        This is useful for dataset-wide normalization, scaling each band based on the distribution
        of pixel values.
        
        Parameters:
          images (list[np.ndarray]): List of images, each with shape (H, W, bands).
          quantile_low (float): Lower quantile percentage (default is 1).
          quantile_up (float): Upper quantile percentage (default is 99).
        
        Returns:
          Tuple[np.ndarray, np.ndarray]: Two 1D arrays (one for each band) containing the lower and upper quantiles.
        """
        # Flatten each image to a 2D array of pixels and concatenate all pixels across images.
        all_pixels = np.concatenate([img.reshape(-1, img.shape[2]) for img in images], axis=0)
        # Replace NaN and infinite values with zeros.
        all_pixels = np.nan_to_num(all_pixels, nan=0.0, posinf=0.0, neginf=0.0)
        q_low = np.percentile(all_pixels, quantile_low, axis=0)
        q_up = np.percentile(all_pixels, quantile_up, axis=0)
        return q_low, q_up

    def _get_all_files_for_sensor(self, sensor: str) -> List[str]:
        """
        Retrieves a list of all TIFF file paths for a specified sensor within the data folder.
        
        Parameters:
          sensor (str): Sensor name (e.g., "Sentinel-2").
        
        Returns:
          list: List of file paths matching the sensor.
        """
        sensor_folder = os.path.join(self.data_folder, sensor)
        return glob.glob(os.path.join(sensor_folder, "*.tiff"))
    
    def get_global_quantiles_for_sensor(self, sensor: str, quantile_low: float = 1, quantile_up: float = 99) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes and caches the global quantiles for a specified sensor using all available images.
        
        Parameters:
          sensor (str): Sensor name (e.g., "Sentinel-2").
          quantile_low (float): Lower quantile percentage (default is 1).
          quantile_up (float): Upper quantile percentage (default is 99).
        
        Returns:
          Tuple[np.ndarray, np.ndarray]: Lower and upper quantile arrays for the sensor.
        """
        # Return cached quantiles if available.
        if sensor in self.global_quantiles:
            return self.global_quantiles[sensor]
        file_list = self._get_all_files_for_sensor(sensor)
        images = []
        for f in file_list:
            img, _ = self.load_image(f)
            images.append(img)
        q_low, q_up = self.compute_global_quantiles(images, quantile_low, quantile_up)
        self.global_quantiles[sensor] = (q_low, q_up)
        return q_low, q_up

    def global_normalize_image(self, image: np.ndarray, q_low: np.ndarray, q_up: np.ndarray) -> np.ndarray:
        """
        Normalizes an image using the global quantiles computed across the dataset.
        
        The normalization is applied per band in a linear fashion:
          - Pixel values ≤ q_low are mapped to 0.
          - Pixel values ≥ q_up are mapped to 1.
          - Values in between are scaled linearly.
        
        Parameters:
          image (np.ndarray): Input image with shape (H, W, bands).
          q_low (np.ndarray): Lower quantile values per band.
          q_up (np.ndarray): Upper quantile values per band.
        
        Returns:
          np.ndarray: The normalized image with pixel values in the range [0, 1].
        """
        normalized = np.empty_like(image, dtype=np.float32)
        # Process each band separately.
        for b in range(image.shape[2]):
            band = image[:, :, b].astype(np.float32)
            if q_up[b] == q_low[b]:
                normalized[:, :, b] = band
            else:
                normalized[:, :, b] = np.clip((band - q_low[b]) / (q_up[b] - q_low[b]), 0, 1)
        return normalized
    
    def crop_image_to_patches(self, image: np.ndarray, crop_factor: int) -> np.ndarray:
        """
        Splits an image into a grid of non-overlapping patches.
        
        For instance, if crop_factor is 2, the image is divided into 4 patches (2 per side).
        
        Parameters:
          image (np.ndarray): Input image of shape (height, width, channels).
          crop_factor (int): Number of patches per side.
        
        Returns:
          np.ndarray: An array of patches with shape 
                      (num_patches, patch_height, patch_width, channels),
                      where num_patches = crop_factor * crop_factor.
        """
        # Extract original image dimensions.
        H, W = image.shape[:2]
        C = image.shape[2] if image.ndim == 3 else 1

        # Compute the dimensions for each patch.
        patch_H = H // crop_factor
        patch_W = W // crop_factor

        # Ensure the image dimensions are exactly divisible by crop_factor.
        new_H = patch_H * crop_factor
        new_W = patch_W * crop_factor
        image_cropped = image[:new_H, :new_W, ...]

        # If the image is grayscale (2D), add a channel dimension.
        if image_cropped.ndim == 2:
            image_cropped = image_cropped[..., np.newaxis]

        # Reshape and transpose to extract patches.
        patches = image_cropped.reshape(crop_factor, patch_H, crop_factor, patch_W, C)
        patches = np.transpose(patches, (0, 2, 1, 3, 4))
        patches = patches.reshape(crop_factor * crop_factor, patch_H, patch_W, C)
        return patches

    # --- Generic Utility Methods (static) ---

    @staticmethod
    def parse_dates_from_filename(filename: str) -> Tuple[datetime, datetime]:
        """
        Parses a filename formatted as: areaName_satellite_fromDate_toDate.tiff,
        extracting the acquisition start and end dates.
        
        Parameters:
          filename (str): The filename to parse.
        
        Returns:
          tuple: A tuple (from_date, to_date) as datetime objects.
        """
        import os
        basename = os.path.splitext(os.path.basename(filename))[0]
        parts = basename.split('_')
        from_date = datetime.strptime(parts[-2], "%Y-%m-%d")
        to_date = datetime.strptime(parts[-1], "%Y-%m-%d")
        return from_date, to_date

    @staticmethod
    def select_temporal_steps(dates: list, n_steps: int = 3) -> list:
        """
        Selects a subset of dates from a sorted list to represent the temporal sampling.
        For example, with n_steps=3, it returns the first, median, and last dates.
        
        Parameters:
          dates (list): A sorted list of datetime objects.
          n_steps (int): Number of time steps to select.
        
        Returns:
          list: A list of selected datetime objects.
        """
        if len(dates) <= n_steps:
            return dates
        if n_steps == 3:
            return [dates[0], dates[len(dates)//2], dates[-1]]
        indices = np.linspace(0, len(dates)-1, n_steps, dtype=int)
        return [dates[i] for i in indices]

    @staticmethod
    def select_daily_images(dates: list) -> list:
        """
        From a list of datetime objects, selects one image per unique day.
        
        Parameters:
          dates (list): A list of datetime objects.
        
        Returns:
          list: A list of datetime objects, one per unique day.
        """
        unique_days = sorted({d.date() for d in dates})
        return [datetime.combine(day, datetime.min.time()) for day in unique_days]

    @staticmethod
    def select_central_date(dates: list) -> datetime:
        """
        Selects the central date from a list of datetime objects.
        
        Parameters:
          dates (list): A list of datetime objects.
        
        Returns:
          datetime: The central date.
        
        Raises:
          ValueError: If the input list is empty.
        """
        if not dates:
            raise ValueError("No dates provided")
        dates_sorted = sorted(dates)
        return dates_sorted[len(dates_sorted)//2]

    @staticmethod
    def _bytes_feature(value: bytes) -> tf.train.Feature:
        """
        Creates a TFRecord bytes_list feature from a byte string.
        
        Parameters:
          value (bytes): The byte string.
        
        Returns:
          tf.train.Feature: A feature for TFRecord serialization.
        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value: int) -> tf.train.Feature:
        """
        Creates a TFRecord int64_list feature from an integer.
        
        Parameters:
          value (int): The integer value.
        
        Returns:
          tf.train.Feature: A feature for TFRecord serialization.
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    @staticmethod
    def _float_feature(value: list) -> tf.train.Feature:
        """
        Creates a TFRecord float_list feature from a list of floats.
        
        Parameters:
          value (list of float): The list of float values.
        
        Returns:
          tf.train.Feature: A feature for TFRecord serialization.
        """
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def simple_augment_tensor(image: tf.Tensor) -> tf.Tensor:
        """
        Applies random rotation (0°, 90°, 180°, or 270°) and random horizontal and vertical flips
        to an image tensor for data augmentation purposes.
        
        Parameters:
          image (tf.Tensor): Input image tensor.
        
        Returns:
          tf.Tensor: Augmented image tensor.
        """
        # If the image is batched (4D), apply augmentation to each individual image.
        if image.get_shape().ndims == 4:
            return tf.map_fn(lambda img: DatasetPreparation.simple_augment_tensor(img), image)
        # Choose a random rotation angle.
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        return image

    @staticmethod
    def random_augment(inputs: dict, label: tf.Tensor) -> tuple:
        """
        Applies random augmentation to each sensor image and the corresponding label.
        
        Parameters:
          inputs (dict): Dictionary mapping sensor names to their image tensors.
          label (tf.Tensor): Tensor representing the label image.
        
        Returns:
          tuple: A tuple (augmented_inputs, augmented_label) with augmented data.
        """
        augmented_inputs = {}
        for sensor, tensor in inputs.items():
            augmented = DatasetPreparation.simple_augment_tensor(tensor)
            augmented.set_shape(tensor.shape)
            augmented_inputs[sensor] = augmented
        augmented_label = DatasetPreparation.simple_augment_tensor(label)
        augmented_label.set_shape(label.shape)
        return augmented_inputs, augmented_label

    @staticmethod
    def augment_example(inputs: dict, label: tf.Tensor) -> list:
        """
        Given an example consisting of sensor inputs and a label, returns a list containing both
        the original and augmented examples.
        
        Parameters:
          inputs (dict): Dictionary of sensor image tensors.
          label (tf.Tensor): Label tensor.
        
        Returns:
          list: A list containing two tuples: the original (inputs, label) and the augmented (aug_inputs, aug_label).
        """
        aug_inputs, aug_label = DatasetPreparation.random_augment(inputs, label)
        return [(inputs, label), (aug_inputs, aug_label)]

    @staticmethod
    def _build_feature_description(sensors: list = None, label_key: str = "activation_label") -> dict:
        """
        Constructs a dictionary that describes the features contained in a TFRecord example.
        The description includes information for sensor images and the label.
        
        Parameters:
          sensors (list): List of sensor names to include. Defaults to all available sensors.
          label_key (str): Key for the label. Expected values: "activation_label", "vineyard_label", or "ET_label".
        
        Returns:
          dict: A mapping from feature keys to tf.io.FixedLenFeature objects.
        """
        default_sensors = ["Sentinel-2", "Sentinel-1", "Sentinel-3-OLCI", "Sentinel-3-SLSTR-Thermal", "DEM"]
        if sensors is None:
            sensors = default_sensors
        
        # Build label feature description based on label_key.
        feature_description = {}
        if label_key == "activation_label":
            feature_description["activation_label"] = tf.io.FixedLenFeature([], tf.string)
            feature_description["activation_label_height"] = tf.io.FixedLenFeature([], tf.int64)
            feature_description["activation_label_width"] = tf.io.FixedLenFeature([], tf.int64)
        elif label_key == "vineyard_label":
            feature_description["vineyard_label"] = tf.io.FixedLenFeature([], tf.int64)
        elif label_key == "ET_label":
            feature_description["ET_label"] = tf.io.FixedLenFeature([], tf.float32)
            feature_description["period_info"] = tf.io.FixedLenFeature([], tf.string)
        else:
            raise ValueError(f"Unknown label_key: {label_key}")

        # Add sensor image features.
        for sensor in sensors:
            if sensor == "DEM":
                feature_description[f"{sensor}_image"] = tf.io.FixedLenFeature([], tf.string)
                feature_description[f"{sensor}_height"] = tf.io.FixedLenFeature([], tf.int64)
                feature_description[f"{sensor}_width"] = tf.io.FixedLenFeature([], tf.int64)
                feature_description[f"{sensor}_channels"] = tf.io.FixedLenFeature([], tf.int64)
            else:
                feature_description[f"{sensor}_image"] = tf.io.FixedLenFeature([], tf.string)
                feature_description[f"{sensor}_height"] = tf.io.FixedLenFeature([], tf.int64)
                feature_description[f"{sensor}_width"] = tf.io.FixedLenFeature([], tf.int64)
                feature_description[f"{sensor}_channels"] = tf.io.FixedLenFeature([], tf.int64)
                feature_description[f"{sensor}_n_steps"] = tf.io.FixedLenFeature([], tf.int64)
        return feature_description


    @staticmethod
    def _decode_sensor(parsed_features: dict, sensor_key: str) -> tf.Tensor:
        """
        Decodes a sensor image from a parsed TFRecord example.
        
        For time-series data (if n_steps > 1), the image is reshaped accordingly and a check is performed
        to ensure that consecutive temporal steps are not identical (i.e. images must differ).
        
        Parameters:
          parsed_features (dict): Dictionary of features parsed from a TFRecord example.
          sensor_key (str): The key representing the sensor (e.g., "Sentinel-2").
        
        Returns:
          tf.Tensor: The decoded image tensor.
        
        Raises:
          tf.errors.InvalidArgumentError: If consecutive temporal steps are too similar.
        """
        image_raw = parsed_features[f"{sensor_key}_image"]
        height = tf.cast(parsed_features[f"{sensor_key}_height"], tf.int32)
        width = tf.cast(parsed_features[f"{sensor_key}_width"], tf.int32)
        channels = tf.cast(parsed_features[f"{sensor_key}_channels"], tf.int32)
        if sensor_key != "DEM":
            n_steps = tf.cast(parsed_features[f"{sensor_key}_n_steps"], tf.int32)
        else:
            n_steps = 1
        image = tf.io.decode_raw(image_raw, tf.float32)
        if n_steps > 1:
            image = tf.reshape(image, [n_steps, height, width, channels])
            # Check that consecutive temporal steps differ.
            def check_diff(i, acc):
                diff = tf.reduce_mean(tf.abs(image[i + 1] - image[i]))
                # Raise error if the difference is too small (threshold can be adjusted)
                tf.debugging.assert_greater(diff, 1e-3,
                    message=f"Consecutive temporal steps for sensor {sensor_key} are too similar.")
                return i + 1, acc
            i = tf.constant(0)
            cond = lambda i, _: tf.less(i, n_steps - 1)
            #_, _ = tf.while_loop(cond, check_diff, [i, tf.constant(0.0)])
        else:
            image = tf.reshape(image, [height, width, channels])
        # Replace NaN and Inf values with zeros.
        image = tf.where(tf.math.is_nan(image), tf.zeros_like(image), image)
        image = tf.where(tf.math.is_inf(image), tf.zeros_like(image), image)
        return image


    @staticmethod
    def _filter_nonzero(inputs: dict, label: tf.Tensor, 
                        min_image_nonzero_percentage: float, 
                        min_label_nonzero_percentage: float, 
                        label_key: str) -> tf.Tensor:
        """
        Filters an example based on the percentage of non-zero pixels in sensor images and label.
        
        Parameters:
          inputs (dict): Dictionary of sensor image tensors.
          label (tf.Tensor): The decoded label tensor.
          min_image_nonzero_percentage (float): Minimum percentage (0.0-1.0) of non-zero pixels required in sensor images.
          min_label_nonzero_percentage (float): Minimum percentage (0.0-1.0) of non-zero pixels required in the label (only used if label_key is "activation_label").
          label_key (str): The key for the label.
        
        Returns:
          tf.Tensor: A boolean scalar tensor. True if the example passes the thresholds, False otherwise.
        """
        sensor_pass = True
        for sensor in inputs:
            # Compute the ratio of non-zero pixels in the sensor image.
            nonzero_ratio = tf.reduce_mean(tf.cast(tf.not_equal(inputs[sensor], 0), tf.float32))
            sensor_pass = tf.logical_and(sensor_pass, tf.greater_equal(nonzero_ratio, min_image_nonzero_percentage))
        
        if label_key == "activation_label":
            label_nonzero = tf.reduce_mean(tf.cast(tf.not_equal(label, 0), tf.float32))
            label_pass = tf.greater_equal(label_nonzero, min_label_nonzero_percentage)
        else:
            label_pass = True

        return tf.logical_and(sensor_pass, label_pass)


    @staticmethod
    def parse_dataset(example_proto: tf.Tensor, 
                      augment: bool = False, 
                      crop: bool = False,
                      crop_factor: int = 1, 
                      sensors: list = None,
                      label_key: str = "activation_label") -> tf.data.Dataset:
        """
        Parses a single serialized TFRecord example into sensor inputs and a label.
        
        The function decodes the features, reshapes sensor data appropriately (accounting for time steps),
        decodes the label based on the provided label_key, and optionally applies random augmentation.
        Additionally, it will later be used for filtering based on the percentage of non-zero pixels.
        
        Parameters:
          example_proto (tf.Tensor): Serialized TFRecord example.
          augment (bool): Whether to apply random data augmentation.
          crop (bool): Indicates if the example contains cropped images.
          crop_factor (int): The crop factor used during cropping.
          min_image_nonzero_percentage (float): Minimum non-zero pixel ratio for sensor images.
          min_label_nonzero_percentage (float): Minimum non-zero pixel ratio for the label (used if label_key is "activation_label").
          sensors (list): List of sensor names to parse; if None, defaults to all sensors.
          label_key (str): Key for the label. Expected values: "activation_label", "vineyard_label", or "ET_label".
        
        Returns:
          tf.data.Dataset: A dataset containing a single tuple (inputs, label).
        """

        inputs = {}
        default_sensors = ["Sentinel-2", "Sentinel-1", "Sentinel-3-OLCI", "Sentinel-3-SLSTR-Thermal", "DEM"]
        if sensors is None:
            sensors = default_sensors


        feature_description = DatasetPreparation._build_feature_description(sensors, label_key)
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        
        
        for sensor in sensors:
            inputs[sensor] = DatasetPreparation._decode_sensor(parsed_features, sensor)
        
        # Decode the label based on label_key.
        if label_key == "activation_label":
            label_height = tf.cast(parsed_features["activation_label_height"], tf.int32)
            label_width = tf.cast(parsed_features["activation_label_width"], tf.int32)
            label = tf.io.decode_raw(parsed_features["activation_label"], tf.uint8)
            label = tf.reshape(label, [label_height, label_width, 1])
            label = tf.cast(label, tf.uint8)
        elif label_key == "vineyard_label":
            label = tf.cast(parsed_features["vineyard_label"], tf.int32)
        elif label_key == "ET_label":
            label = tf.cast(parsed_features["ET_label"], tf.float32)
            # Optionally, you could also decode period_info if needed.
        else:
            raise ValueError(f"Unknown label_key: {label_key}")

        if augment:
            inputs, label = DatasetPreparation.random_augment(inputs, label)
        
        # Return as a single-example dataset.
        return tf.data.Dataset.from_tensors((inputs, label))


    @staticmethod
    def get_dataset(tfrecord_files: list, 
                    batch_size: int = 4, 
                    shuffle_buffer: int = None,
                    augment: bool = False, 
                    crop: bool = False, 
                    crop_factor: int = 1,
                    min_image_nonzero_percentage: float = 0.0,
                    min_label_nonzero_percentage: float = 0.0,
                    sensors: list = None,
                    label_key: str = "activation_label") -> tf.data.Dataset:
        """
        Constructs a tf.data.Dataset from a list of TFRecord files with options for batching, shuffling, augmentation,
        and filtering examples based on a minimum percentage of non-zero pixels in both sensor images and label.
        
        Parameters:
          tfrecord_files (list): List of file paths to TFRecord files.
          batch_size (int): Number of examples per batch.
          shuffle_buffer (int): Buffer size used for shuffling the dataset.
          augment (bool): Whether to apply random data augmentation.
          crop (bool): Indicates if the TFRecord contains cropped images.
          crop_factor (int): Crop factor used when cropping was applied.
          min_image_nonzero_percentage (float): Minimum non-zero pixel ratio for sensor images (0.0-1.0).
          min_label_nonzero_percentage (float): Minimum non-zero pixel ratio for the label (only used if label_key is "activation_label").
          sensors (list): List of sensor names to include; if None, defaults to all sensors.
          label_key (str): Key for the label. Expected values: "activation_label", "vineyard_label", or "ET_label".
        
        Returns:
          tf.data.Dataset: A dataset yielding tuples of (inputs, label) ready for training.
        """

        dataset = tf.data.TFRecordDataset(tfrecord_files)
        dataset = dataset.flat_map(lambda x: DatasetPreparation.parse_dataset(
            x, augment, crop, crop_factor, sensors, label_key))
        
        # Apply filtering based on non-zero pixel criteria.
        def filter_fn(inputs, label):
            return DatasetPreparation._filter_nonzero(inputs, label, 
                                                      min_image_nonzero_percentage, 
                                                      min_label_nonzero_percentage, 
                                                      label_key)
        
        dataset = dataset.filter(filter_fn)
        if shuffle_buffer:
          dataset = dataset.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        else:
          dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    @staticmethod
    def print_dataset_characteristics(dataset: tf.data.Dataset, sensors: list = None) -> None:
        """
        Inspects one batch from the dataset and prints the shape and data type of each sensor input,
        as well as the activation label.
        
        Parameters:
          dataset (tf.data.Dataset): The dataset to inspect.
          sensors (list): Optional list of sensor names to display; if None, all sensors are printed.
        
        Returns:
          None.
        """
        for batch_inputs, batch_labels in dataset.take(1):
            print("Batch size:", batch_labels.shape[0])
            if sensors is None:
                sensors = list(batch_inputs.keys())
            for sensor in sensors:
                if sensor in batch_inputs:
                    tensor = batch_inputs[sensor]
                    print(f"Sensor '{sensor}': shape = {tensor.shape}, dtype = {tensor.dtype}")
            print(f"Activation label: shape = {batch_labels.shape}, dtype = {batch_labels.dtype}")
