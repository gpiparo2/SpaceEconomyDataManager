#!/usr/bin/env python3
"""
data_manipulator.py
-------------------
This module provides the DataManipulator class which is responsible for organizing,
renaming and preparing downloaded satellite imagery for further analysis.

Main functionalities include:
  - Renaming TIFF files based on area name, satellite type, and acquisition dates.
  - Organizing files into separate folders by satellite.
  - Preparing dataset lists with spectral bands and associated metadata.
  - Exporting spectral data (and corresponding masks) as CSV or compressed NPZ files.

This module is part of the core functionalities.
"""

import os
import json
import shutil
from datetime import datetime
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
import numpy as np
import geopandas as gpd


class DataManipulator:
    """
    Class responsible for post-processing downloaded satellite images.
    
    It provides methods to:
      - Rename and reorganize TIFF files into a standardized folder structure.
      - Map raw data collection names to human-friendly satellite names.
      - Prepare datasets by loading images, extracting bands and masks,
        and generating lists of spectral band data with associated metadata.
      - Save processed data into CSV or NPZ compressed formats.
    """

    def __init__(self, source_folder: str = "../data", destination_folder: str = "../data_off"):
        """
        Initializes the DataManipulator with specified source and destination folders.
        
        Parameters:
          source_folder (str): Path to the folder containing raw downloaded data.
          destination_folder (str): Path to the folder where organized data will be stored.
        """
        self.source_folder = source_folder
        self.destination_folder = destination_folder

    def map_data_collection_to_satellite(self, collection_name: str) -> str:
        """
        Converts a raw data collection name (as returned by Sentinel Hub) into a human-friendly satellite name.
        
        Parameters:
          collection_name (str): Raw data collection name.
        
        Returns:
          str: Human-friendly satellite name.
        """
        cn = collection_name.lower()
        print(f"Mapping data collection: {cn}")
        if "sentinel-2" in cn:
            return "Sentinel-2"
        elif "sentinel-1" in cn:
            return "Sentinel-1"
        elif "sentinel-3-olci" in cn:
            return "Sentinel-3-OLCI"
        elif "sentinel-3-slstr" in cn or "s3-slstr" in cn:
            return "Sentinel-3-SLSTR"
        elif "dem" in cn:
            return "DEM"
        else:
            return "UnknownSatellite"

    def manipulate_data(self, area_name: str, clean_source: bool = False):
        """
        Organizes downloaded TIFF files by renaming them according to a standardized convention
        and copying them into satellite-specific subfolders.
        
        The new filename follows the pattern:
            areaName_satelliteName_fromDate_toDate.tiff
        
        Parameters:
          area_name (str): Name of the area (should not contain underscores).
          clean_source (bool, optional): If True, deletes files/folders from the source folder after processing.
        
        Returns:
          None.
        
        Note:
          This method currently assumes that only non-fused (single) data entries are present.
        """
        print(f"**STARTING DATA ORGANIZATION FROM {self.source_folder}**")
        # Iterate through all subdirectories in the source folder.
        for subdir in sorted(os.listdir(self.source_folder)):
            subdir_path = os.path.join(self.source_folder, subdir)
            if not os.path.isdir(subdir_path):
                continue

            # Locate the required JSON and TIFF files.
            request_json_path = os.path.join(subdir_path, 'request.json')
            # Prefer 'response.tiff' over 'fused_data.tif' (fusion no longer supported)
            tiff_path = os.path.join(subdir_path, 'response.tiff')
            if not os.path.exists(tiff_path) or not os.path.exists(request_json_path):
                continue

            # Load the request payload to extract metadata.
            with open(request_json_path, 'r') as fjson:
                request_data = json.load(fjson)
            data_entries = request_data['request']['payload']['input']['data']
            # Assume a single (non-fused) data entry.
            data_collection_name = data_entries[0]['type']
            if data_collection_name.lower() == "sentinel-3-slstr":
                evalscript = request_data['request']['payload'].get('evalscript', "").lower()
                # Distinguish between optical and thermal SLSTR based on evalscript content.
                if "reflectance" in evalscript:
                    satellite_name = "Sentinel-3-SLSTR-Optical"
                elif "brightness temperature" in evalscript or "brightness" in evalscript:
                    satellite_name = "Sentinel-3-SLSTR-Thermal"
                else:
                    satellite_name = "Sentinel-3-SLSTR"
            else:
                satellite_name = self.map_data_collection_to_satellite(data_collection_name)
            # Extract acquisition dates from the request payload.
            from_time = data_entries[0]['dataFilter']['timeRange']['from']
            to_time = data_entries[0]['dataFilter']['timeRange']['to']
            from_time_fmt = datetime.strptime(from_time, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
            to_time_fmt = datetime.strptime(to_time, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
            # Construct the new filename.
            new_filename = f"{area_name}_{satellite_name}_{from_time_fmt}_{to_time_fmt}.tiff"
            destination_dir = os.path.join(self.destination_folder, satellite_name)
            os.makedirs(destination_dir, exist_ok=True)
            # Copy the TIFF file to the destination directory with the new filename.
            shutil.copy(tiff_path, os.path.join(destination_dir, new_filename))
            print(f"Copied data from {subdir} to {destination_dir}")

        # Optionally clean up the source folder.
        if clean_source:
            for subdir, dirs, files in os.walk(self.source_folder):
                for file in files:
                    os.remove(os.path.join(subdir, file))
                for dir in dirs:
                    shutil.rmtree(os.path.join(subdir, dir))
            print(f"Cleaned source data directory {self.source_folder}")

    def generate_binary_mask(self, geojson_path: str, raster_shape: tuple, transform) -> np.ndarray:
        """
        Generates a binary mask for a raster image based on the polygons defined in a GeoJSON file.
        
        Parameters:
          geojson_path (str): Path to the GeoJSON file containing polygon geometries.
          raster_shape (tuple): Tuple specifying the raster dimensions (height, width).
          transform: Affine transformation associated with the raster image.
        
        Returns:
          np.ndarray: A binary mask (dtype uint8) where pixels inside the polygon(s) are set to 1.
        """
        with open(geojson_path) as f:
            geojson = json.load(f)
        mask_arr = geometry_mask(
            [feature["geometry"] for feature in geojson['features']],
            invert=True,
            transform=transform,
            out_shape=raster_shape
        )
        return mask_arr.astype(np.uint8)

    def get_satellite_band_names(self, satellite_name: str, count: int) -> list:
        """
        Returns a list of descriptive band names for a given satellite.
        
        The function provides default naming conventions for various satellites.
        If the expected number of bands differs from the default list, the list is extended or truncated accordingly.
        
        Parameters:
          satellite_name (str): Name of the satellite.
          count (int): Expected number of bands.
        
        Returns:
          list: List of band names.
        """
        if satellite_name == "Sentinel-2":
            band_names = ["Aerosol", "Blue", "Green", "Red", "Red Edge 1", "Red Edge 2",
                          "Red Edge 3", "NIR 1", "NIR 2", "Water Vapour", "SWIR1", "SWIR2",
                          "AOT", "SCL", "SNW", "CLD", "dataMask"]
        elif satellite_name == "Sentinel-1":
            band_names = [f"Band_{i+1}" for i in range(count)]
        elif satellite_name == "Sentinel-3-OLCI":
            band_names = [f"B{i:02d}" for i in range(1, count+1)]
        elif satellite_name == "Sentinel-3-SLSTR":
            band_names = [f"Band_{i+1}" for i in range(count)]
        elif satellite_name == "DEM":
            band_names = ["DEM"]
        else:
            band_names = [f"Band_{i+1}" for i in range(count)]
        if len(band_names) < count:
            band_names += [f"Band_{i+1}" for i in range(len(band_names), count)]
        elif len(band_names) > count:
            band_names = band_names[:count]
        return band_names

    def prepare_spectral_bands_dataset_list(self, area_name: str, date_from: str = None, date_to: str = None,
                                            crop_mask_path: str = None, crop: bool = False, 
                                            binary_mask_path: str = None, binary_mask: bool = False) -> dict:
        """
        Prepares a dictionary of dataset lists for each satellite based on the organized TIFF files.
        
        For each satellite folder, the method collects TIFF files whose acquisition dates fall within
        the specified period (if provided) and reads the image data along with associated metadata.
        Each entry in the returned dictionary is a tuple containing:
            (from_date, to_date, bands_array, band_names, fire_mask, crop_mask)
        
        Parameters:
          area_name (str): Name of the area.
          date_from (str, optional): Lower bound date (format: 'YYYY-MM-DD').
          date_to (str, optional): Upper bound date (format: 'YYYY-MM-DD').
          crop_mask_path (str, optional): Path to a GeoJSON file used for cropping the images.
          crop (bool, optional): Whether to apply cropping to the images.
          binary_mask_path (str, optional): Path to a GeoJSON file used to generate a binary mask.
          binary_mask (bool, optional): Whether to generate a binary mask.
        
        Returns:
          dict: Dictionary with satellite names as keys and lists of tuples as values.
        """
        date_from_obj = datetime.strptime(date_from, '%Y-%m-%d') if date_from else None
        date_to_obj = datetime.strptime(date_to, '%Y-%m-%d') if date_to else None

        possible_dirs = os.listdir(self.destination_folder)
        satellites = [d for d in possible_dirs if os.path.isdir(os.path.join(self.destination_folder, d)) and d not in ["vegetation_indices"]]
        result = {}
        for sat in satellites:
            sat_dir = os.path.join(self.destination_folder, sat)
            if not os.path.exists(sat_dir):
                continue
            sat_data_list = []
            for file in sorted(os.listdir(sat_dir)):
                if file.endswith('.tiff'):
                    file_without_ext = os.path.splitext(file)[0]
                    parts = file_without_ext.split('_')
                    from_str = parts[-2]
                    to_str = parts[-1]
                    file_date_from_obj = datetime.strptime(from_str, '%Y-%m-%d')
                    file_date_to_obj = datetime.strptime(to_str, '%Y-%m-%d')
                    # Filter files based on provided date range.
                    if (date_from_obj is None or file_date_from_obj >= date_from_obj) and \
                       (date_to_obj is None or file_date_to_obj <= date_to_obj):
                        file_path = os.path.join(sat_dir, file)
                        with rasterio.open(file_path) as src:
                            # Apply cropping if requested.
                            if crop and crop_mask_path:
                                crop_mask_geometry = gpd.read_file(crop_mask_path)
                                bands, _ = mask(src, crop_mask_geometry.geometry, crop=True)
                            else:
                                bands = src.read()
                            # Reorder bands to have shape (height, width, nbands).
                            bands_reshaped = np.moveaxis(bands, 0, -1)
                            # Initialize empty masks.
                            generated_fire_mask = np.zeros((bands_reshaped.shape[0], bands_reshaped.shape[1]), dtype=np.uint8)
                            generated_crop_mask = np.zeros((bands_reshaped.shape[0], bands_reshaped.shape[1]), dtype=np.uint8)
                            if binary_mask and binary_mask_path:
                                generated_fire_mask = self.generate_binary_mask(binary_mask_path, src.shape, src.transform)
                            if crop and crop_mask_path:
                                generated_crop_mask = self.generate_binary_mask(crop_mask_path, src.shape, src.transform)
                            band_names = self.get_satellite_band_names(sat, bands_reshaped.shape[2])
                            sat_data_list.append((file_date_from_obj, file_date_to_obj, bands_reshaped, band_names, generated_fire_mask, generated_crop_mask))
            result[sat] = sat_data_list
        return result

    def extract_spectral_data(self, data_list: list) -> np.ndarray:
        """
        Extracts and stacks the spectral band arrays from a list of image tuples.
        
        Parameters:
          data_list (list): List of tuples containing image data.
        
        Returns:
          np.ndarray: A 4D array with shape (time, height, width, nbands).
        
        Raises:
          ValueError: If image dimensions are inconsistent across the dataset.
        """
        if len(data_list) == 0:
            return np.array([])
        array_list = [el[2] for el in data_list]
        x, y, nbands = array_list[0].shape
        for arr in array_list:
            if arr.shape != (x, y, nbands):
                raise ValueError("Array dimensions do not match.")
        return np.stack(array_list, axis=0)

    def extract_masks(self, data_list: list) -> np.ndarray:
        """
        Extracts and stacks binary masks from a list of image tuples.
        
        Parameters:
          data_list (list): List of tuples where the fifth element is the binary mask.
        
        Returns:
          np.ndarray: A 3D array with shape (time, height, width) containing the masks.
        """
        if len(data_list) == 0:
            return np.array([])
        mask_list = [el[4] for el in data_list]
        return np.stack(mask_list, axis=0)

    def store_data_as_csv(self, area_name: str, binary_mask_path: str, crop_mask_path: str, date_from: str = None, date_to: str = None):
        """
        Exports the spectral band data and corresponding masks into CSV files.
        Each satellite's dataset is stored in a separate CSV file.

        The CSV file begins with a header line listing the band names and additional columns for fire and crop masks,
        followed by a line containing size information, and finally the flattened spectral data with appended masks.

        Parameters:
          area_name (str): Name of the area.
          binary_mask_path (str): Path to the GeoJSON file for generating the binary mask.
          crop_mask_path (str): Path to the GeoJSON file for generating the crop mask.
          date_from (str, optional): Lower bound date (format: 'YYYY-MM-DD').
          date_to (str, optional): Upper bound date (format: 'YYYY-MM-DD').

        Returns:
          None.
        """
        spectral_data = self.prepare_spectral_bands_dataset_list(
            area_name=area_name,
            date_from=date_from,
            date_to=date_to,
            crop=False,
            binary_mask=True,
            binary_mask_path=binary_mask_path,
            crop_mask_path=crop_mask_path
        )
        for sat, data_list in spectral_data.items():
            for (df, dt, bands_reshaped, band_names, fire_mask, crop_mask) in data_list:
                from_time = df.strftime("%Y-%m-%d")
                to_time = dt.strftime("%Y-%m-%d")
                width = bands_reshaped.shape[0]
                height = bands_reshaped.shape[1]
                pixels = width * height
                spectral_array_flat = bands_reshaped.reshape((pixels, bands_reshaped.shape[2]))
                fire_mask_flat = fire_mask.reshape((pixels, 1))
                crop_mask_flat = crop_mask.reshape((pixels, 1))
                header_string = '\t'.join(band_names) + '\tFireBinaryMask\tCropBinaryMask\n'
                size_string = f"{width},{height},{spectral_array_flat.shape[1] + 2}\n"
                final_array = np.concatenate((spectral_array_flat, fire_mask_flat, crop_mask_flat), axis=1)
                csv_filename = f"{area_name}_{sat}_{from_time}_{to_time}.csv"
                csv_filepath = os.path.join(self.destination_folder, csv_filename)
                with open(csv_filepath, 'w') as file:
                    file.write(header_string)
                    file.write(size_string)
                with open(csv_filepath, 'a') as file:
                    np.savetxt(file, final_array, delimiter=',')
                print(f"Data for {sat} stored as CSV in {csv_filepath}")

    def store_data_compressed(self, area_name: str, binary_mask_path: str, crop_mask_path: str,
                              date_from: str = None, date_to: str = None, clean_manipulated: bool = False):
        """
        Exports the spectral band data and associated masks into compressed NPZ files.
        Each satellite's dataset is stored separately.

        Parameters:
          area_name (str): Name of the area.
          binary_mask_path (str): Path to the binary mask GeoJSON file.
          crop_mask_path (str): Path to the crop mask GeoJSON file.
          date_from (str, optional): Lower bound date (format: 'YYYY-MM-DD').
          date_to (str, optional): Upper bound date (format: 'YYYY-MM-DD').
          clean_manipulated (bool, optional): If True, deletes the source directories after processing.

        Returns:
          None.
        """
        spectral_data = self.prepare_spectral_bands_dataset_list(
            area_name=area_name,
            date_from=date_from,
            date_to=date_to,
            crop=False,
            binary_mask=True,
            binary_mask_path=binary_mask_path,
            crop_mask_path=crop_mask_path
        )
        for sat, data_list in spectral_data.items():
            for (df, dt, bands_reshaped, band_names, fire_mask, crop_mask) in data_list:
                from_time = df.strftime("%Y-%m-%d")
                to_time = dt.strftime("%Y-%m-%d")
                fire_mask_exp = np.expand_dims(fire_mask, axis=-1)
                crop_mask_exp = np.expand_dims(crop_mask, axis=-1)
                dataset_array = np.concatenate((bands_reshaped, fire_mask_exp, crop_mask_exp), axis=2)
                header_list = band_names + ['FireBinaryMask', 'CropBinaryMask']
                header_array = np.array(header_list)
                filename = f"{area_name}_{sat}_{from_time}_{to_time}"
                filepath = os.path.join(self.destination_folder, filename)
                np.savez_compressed(filepath, header=header_array, data=dataset_array)
                print(f"Data for {sat} stored in {filepath}.npz")
        if clean_manipulated:
            possible_dirs = os.listdir(self.destination_folder)
            for d in possible_dirs:
                if d in ["multispectral", "Sentinel-2", "Sentinel-1", "Sentinel-3-OLCI", "Sentinel-3-SLSTR", "DEM"]:
                    dir_path = os.path.join(self.destination_folder, d)
                    if os.path.exists(dir_path):
                        shutil.rmtree(dir_path)
