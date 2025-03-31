#!/usr/bin/env python3
"""
data_download.py
----------------
This module provides the DataDownload class for handling the download of satellite images via
Sentinel Hub services. It supports multiple satellites including Sentinel-1, Sentinel-2, Sentinel-3 (OLCI and SLSTR) and DEM.

Key functionalities include:
  - API authentication using client credentials.
  - Setting up evalscripts (JavaScript code) for each satellite.
  - Validating input parameters and parsing GeoJSON files to obtain bounding boxes.
  - Downloading images over specified time intervals, using per-satellite configuration
    for parameters such as date ranges, interval days, image size, and mosaicking order.

Note:
  This module is part of the core functionalities and is intended to be used via the SDM master class.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Any

from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, bbox_to_dimensions, BBox, MimeType, CRS
from shapely.geometry import shape
import rasterio
from rasterio.transform import from_bounds
import numpy as np

# Import evalscripts from the config package (located at the top level of the project)
from config import evalscripts

# Configure logging for detailed information output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDownload:
    """
    Class responsible for downloading satellite imagery using Sentinel Hub services.

    The class provides methods for:
      - Authenticating with Sentinel Hub.
      - Configuring and retrieving evalscripts for each satellite.
      - Validating download parameters.
      - Parsing GeoJSON to extract the bounding box.
      - Executing download requests over specified time intervals.
    """

    def __init__(
        self,
        config: Optional[SHConfig] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        config_name: str = 'sentinelhub_config',
        data_folder: str = "../data",
        evalscript_names: Optional[Dict[str, str]] = None,
        units: str = "REFLECTANCE",
        sampleType: str = "FLOAT32",
        evalscript_params: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initializes a new DataDownload instance with the given configuration and download parameters.

        Parameters:
          config (SHConfig, optional): Pre-existing Sentinel Hub configuration object.
          client_id (str, optional): Client ID for Sentinel Hub API.
          client_secret (str, optional): Client secret for Sentinel Hub API.
          config_name (str): Identifier used when saving the configuration.
          data_folder (str): Directory where downloaded images will be stored.
          evalscript_names (dict, optional): Mapping of satellite names to corresponding evalscript function names.
          units (str): Default units for spectral data (default: "REFLECTANCE").
          sampleType (str): Default sample type for the downloaded data (default: "FLOAT32").
          evalscript_params (dict, optional): Additional parameters to pass to each evalscript.
        """
        # Use provided configuration if available; otherwise, instantiate a new one and perform authentication.
        if config and isinstance(config, SHConfig):
            self.config = config
        else:
            self.config = SHConfig()
            self.authenticate_api(client_id, client_secret, config_name)

        # Ensure the data folder exists; create it if missing.
        self.data_folder = data_folder
        os.makedirs(data_folder, exist_ok=True)

        # Warn if API credentials are not set.
        if not self.config.sh_client_id or not self.config.sh_client_secret:
            logger.warning("Warning: 'sh_client_id' and 'sh_client_secret' are not set.")

        # Initialize internal dictionaries to store evalscripts and associated band information.
        self.evalscripts: Dict[str, str] = {}
        self.evalscript_band_info: Dict[str, List[str]] = {}
        self.evalscript_params = evalscript_params if evalscript_params else {}

    def authenticate_api(self, client_id: str, client_secret: str, config_name: str):
        """
        Authenticates with Sentinel Hub using the provided client credentials and saves the configuration.

        Parameters:
          client_id (str): Sentinel Hub client ID.
          client_secret (str): Sentinel Hub client secret.
          config_name (str): Identifier used for saving the configuration.

        Side Effects:
          Updates self.config with the provided credentials and saves the configuration.
        """
        self.config.sh_client_id = client_id
        self.config.sh_client_secret = client_secret
        # Set Sentinel Hub endpoints for token and base URL
        self.config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        self.config.sh_base_url = "https://sh.dataspace.copernicus.eu"
        # Save configuration for later reuse
        self.config.save(config_name)

    def set_evalscripts(
        self,
        evalscript_names: Optional[Dict[str, str]] = None,
        units: str = "REFLECTANCE",
        sampleType: str = "FLOAT32",
        evalscript_params: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Configures evalscripts for each satellite by retrieving the appropriate function from the evalscripts module.
        Satellite-specific parameters from evalscript_params are merged with default values.

        Parameters:
          evalscript_names (dict, optional): Mapping from satellite name to evalscript function name.
          units (str): Default units for the spectral bands.
          sampleType (str): Default sample type for the output data.
          evalscript_params (dict, optional): Additional per-satellite parameters for the evalscript functions.

        Raises:
          ValueError: If the evalscript function for a satellite cannot be found.
        """
        # Define default evalscript function names if none provided.
        if evalscript_names is None:
            evalscript_names = {
                'Sentinel-2': 'evalscript_S2L2A',
                'Sentinel-1': 'evalscript_S1GRD',
                'Sentinel-3-OLCI': 'evalscript_S3_OLCI_L1B',
                'Sentinel-3-SLSTR-Optical': 'evalscript_S3_SLSTR_L1B_optical',
                'Sentinel-3-SLSTR-Thermal': 'evalscript_S3_SLSTR_L1B_thermal',
                'DEM': 'evalscript_DEM'
            }
        if evalscript_params is None:
            evalscript_params = {}

        # Reset internal dictionaries
        self.evalscripts = {}
        self.evalscript_band_info = {}

        # Loop over satellites to set up their evalscripts.
        for satellite, func_name in evalscript_names.items():
            evalscript_func = getattr(evalscripts, func_name, None)
            if callable(evalscript_func):
                # Copy satellite-specific parameters, if provided
                params = evalscript_params.get(satellite, {}).copy()
                # Satellite-specific parameter defaults and evalscript invocation:
                if satellite == 'Sentinel-1':
                    params.setdefault('polarizations', ['VV', 'VH'])
                    params.setdefault('backCoeff', 'GAMMA0_ELLIPSOID')
                    params.setdefault('orthorectify', False)
                    params.setdefault('demInstance', 'COPERNICUS')
                    params.setdefault('sampleType', sampleType)
                    evalscript, bands = evalscript_func(**params, return_band_list=True)
                elif satellite == 'Sentinel-2':
                    # Set default units and sampleType if not provided in params.
                    params.setdefault('units', units)
                    params.setdefault('sampleType', sampleType)
                    evalscript, bands = evalscript_func(**params, return_band_list=True)
                elif satellite in ['Sentinel-3-OLCI', 'Sentinel-3-SLSTR-Optical', 'Sentinel-3-SLSTR-Thermal']:
                    params.setdefault('sampleType', sampleType)
                    evalscript, bands = evalscript_func(**params, return_band_list=True)
                elif satellite == 'DEM':
                    params.setdefault('sampleType', sampleType)
                    params.setdefault('demInstance', 'COPERNICUS_30')
                    evalscript, bands = evalscript_func(**params, return_band_list=True)
                else:
                    evalscript, bands = evalscript_func(**params, return_band_list=True)

                # Store the evalscript and its band information
                self.evalscripts[satellite] = evalscript
                self.evalscript_band_info[satellite] = bands
            else:
                raise ValueError(f"No evalscript found with name: {func_name} for satellite: {satellite}")

    def download_images(
        self,
        date_from: Dict[str, datetime],
        date_to: Dict[str, datetime],
        interval_days: Dict[str, int],
        location_geojson: Dict[str, Any],
        satellites: Optional[List[str]] = None,
        data_collection_names: Optional[Dict[str, str]] = None,
        resolutions: Optional[Dict[str, int]] = None,
        size: Optional[Dict[str, Tuple[int, int]]] = None,
        mosaicking_order: Dict[str, str] = None
    ):
        """
        Downloads satellite images for the specified satellites and time intervals.

        Parameters:
          date_from (dict): Dictionary mapping satellite names to the start datetime.
          date_to (dict): Dictionary mapping satellite names to the end datetime.
          interval_days (dict): Dictionary mapping satellite names to the interval (in days) between images.
          location_geojson (dict): GeoJSON object defining the area of interest.
          satellites (list, optional): List of satellite names (default is ['Sentinel-2']).
          data_collection_names (dict, optional): Mapping from satellite names to specific data collection identifiers.
          resolutions (dict, optional): Mapping from satellite names to desired resolution.
          size (dict, optional): Mapping from satellite names to the image size (width, height) in pixels.
          mosaicking_order (dict): Dictionary mapping satellite names to the desired mosaicking order.

        Returns:
          None. Downloaded images are saved in the configured data folder.

        Raises:
          ValueError: If no evalscript is set for a given satellite.
        """
        # Validate input parameters; raises exception if parameters are invalid.
        self.validate_inputs(date_from, date_to, interval_days, resolutions, size, mosaicking_order)
        # Parse the GeoJSON to obtain the bounding box (BBox) in WGS84.
        bbox = self.parse_geojson(location_geojson)
        # Use Sentinel-2 as default if no satellites are specified.
        if satellites is None:
            satellites = ['Sentinel-2']
        # Retrieve DataCollection objects for the given satellites.
        data_collections = self.get_data_collections(satellites, data_collection_names)
        # Configure evalscripts with any additional parameters.
        self.set_evalscripts(evalscript_params=self.evalscript_params)
        for satellite in satellites:
            if satellite not in self.evalscripts:
                raise ValueError(f"No evalscript set for satellite: {satellite}")
        logger.info("**STARTING DATA DOWNLOAD**")

        # Loop over satellites and initiate download for each.
        for satellite in satellites:
            self.download_satellite_data(
                date_from=date_from,
                date_to=date_to,
                interval_days=interval_days,
                bbox=bbox,
                size=size,
                mosaicking_order=mosaicking_order,
                satellite=satellite,
                data_collection=data_collections[satellite],
                resolution=None  # Optionally, a resolution override can be used here.
            )

    def get_data_collections(self, satellites: List[str], data_collection_names: Optional[Dict[str, str]]) -> Dict[str, DataCollection]:
        """
        Maps satellite names to their corresponding DataCollection objects based on Sentinel Hub collections.

        Parameters:
          satellites (list): List of satellite names.
          data_collection_names (dict, optional): Dictionary mapping satellite names to data collection identifiers.

        Returns:
          dict: A dictionary mapping each satellite name to its DataCollection object.

        Raises:
          ValueError: If a required data collection name is missing or invalid.
        """
        if data_collection_names is None:
            data_collection_names = {
                'Sentinel-2': 'SENTINEL2_L2A',
                'Sentinel-1': 'SENTINEL1_IW',
                'Sentinel-3-OLCI': 'SENTINEL3_OLCI',
                'Sentinel-3-SLSTR-Optical': 'SENTINEL3_SLSTR',
                'Sentinel-3-SLSTR-Thermal': 'SENTINEL3_SLSTR',
                'DEM': 'DEM'
            }
        data_collections = {}
        for satellite in satellites:
            collection_name = data_collection_names.get(satellite)
            if not collection_name:
                raise ValueError(f"No data collection name provided for satellite: {satellite}")
            if hasattr(DataCollection, collection_name):
                # Retrieve the DataCollection and define it using the configuration's base URL.
                data_collection = getattr(DataCollection, collection_name)
                data_collection = data_collection.define_from(collection_name.lower(), service_url=self.config.sh_base_url)
                data_collections[satellite] = data_collection
            else:
                raise ValueError(f"Invalid data collection name '{collection_name}' for satellite '{satellite}'.")
        return data_collections

    def validate_inputs(
        self,
        date_from: Dict[str, datetime],
        date_to: Dict[str, datetime],
        interval_days: Dict[str, int],
        resolutions: Dict[str, int],
        size: Dict[str, Tuple[int, int]],
        mosaicking_order: Dict[str, str]
    ):
        """
        Validates the input parameters for the download_images method.

        Ensures that all parameters are provided as dictionaries with keys corresponding to satellite names.

        Parameters:
          date_from (dict): Mapping from satellite name to start datetime.
          date_to (dict): Mapping from satellite name to end datetime.
          interval_days (dict): Mapping from satellite name to interval in days (must be positive integers).
          resolutions (dict): Mapping from satellite name to resolution (integer).
          size (dict): Mapping from satellite name to a tuple (width, height) of integers.
          mosaicking_order (dict): Mapping from satellite name to a string indicating mosaicking order.

        Raises:
          TypeError or ValueError if any parameter does not conform to the expected format.
        """
        # Check that each start date is a datetime object.
        for sat, start_date in date_from.items():
            if not isinstance(start_date, datetime):
                raise TypeError(f"date_from for satellite {sat} must be a datetime object.")
        # Check that each end date is a datetime object and later than the start date.
        for sat, end_date in date_to.items():
            if not isinstance(end_date, datetime):
                raise TypeError(f"date_to for satellite {sat} must be a datetime object.")
            if date_from[sat] >= end_date:
                raise ValueError(f"For satellite {sat}, date_from must be earlier than date_to.")
        # Ensure interval_days are positive integers.
        for sat, days in interval_days.items():
            if not isinstance(days, int) or days <= 0:
                raise ValueError(f"interval_days for satellite {sat} must be a positive integer.")
        # Validate that size is provided as a tuple of two integers.
        for sat, sz in size.items():
            if not (isinstance(sz, tuple) and len(sz) == 2 and all(isinstance(x, int) for x in sz)):
                raise TypeError(f"size for satellite {sat} must be a tuple of two integers.")
        # Validate that mosaicking_order is a string.
        for sat, order in mosaicking_order.items():
            if not isinstance(order, str):
                raise TypeError(f"mosaicking_order for satellite {sat} must be a string.")

    def parse_geojson(self, location_geojson: Dict[str, Any]) -> BBox:
        """
        Parses the provided GeoJSON object and computes a bounding box (BBox) in WGS84 coordinate reference system.

        Parameters:
          location_geojson (dict): A GeoJSON object representing the area of interest.

        Returns:
          BBox: A bounding box object with coordinates in WGS84.

        Raises:
          ValueError: If the GeoJSON type is unsupported or invalid.
        """
        try:
            # Handle FeatureCollection by aggregating all feature geometries.
            if location_geojson.get('type') == 'FeatureCollection':
                features = location_geojson.get('features', [])
                if not features:
                    raise ValueError("FeatureCollection contains no features.")
                geometries = [shape(feature.get('geometry')) for feature in features if feature.get('geometry')]
                if not geometries:
                    raise ValueError("No geometries found in FeatureCollection.")
                minx, miny, maxx, maxy = geometries[0].bounds
                for geom in geometries[1:]:
                    gxmin, gymin, gxmax, gymax = geom.bounds
                    minx = min(minx, gxmin)
                    miny = min(miny, gymin)
                    maxx = max(maxx, gxmax)
                    maxy = max(maxy, gymax)
                bbox = BBox(bbox=[minx, miny, maxx, maxy], crs=CRS.WGS84)
                return bbox
            # Handle single Feature.
            elif location_geojson.get('type') == 'Feature':
                geometry_json = location_geojson.get('geometry')
                if not geometry_json:
                    raise ValueError("Feature has no geometry.")
                shapely_geom = shape(geometry_json)
                minx, miny, maxx, maxy = shapely_geom.bounds
                bbox = BBox(bbox=[minx, miny, maxx, maxy], crs=CRS.WGS84)
                return bbox
            # Handle Polygon or MultiPolygon directly.
            elif location_geojson.get('type') in ['Polygon', 'MultiPolygon']:
                shapely_geom = shape(location_geojson)
                minx, miny, maxx, maxy = shapely_geom.bounds
                bbox = BBox(bbox=[minx, miny, maxx, maxy], crs=CRS.WGS84)
                return bbox
            else:
                raise ValueError(f"Unsupported GeoJSON type: {location_geojson.get('type')}")
        except Exception as e:
            raise ValueError(f"Invalid GeoJSON format: {e}")

    def generate_date_intervals(
        self,
        date_from: datetime,
        date_to: datetime,
        interval_days: int
    ) -> List[Tuple[datetime, datetime]]:
        """
        Generates a list of time intervals between two dates, partitioning the period into segments of a given number of days.

        Parameters:
          date_from (datetime): The starting date.
          date_to (datetime): The ending date.
          interval_days (int): Number of days per interval.

        Returns:
          List[Tuple[datetime, datetime]]: A list where each element is a tuple (start_date, end_date) for the interval.
        """
        intervals = []
        current_date = date_from
        while current_date < date_to:
            next_date = current_date + timedelta(days=interval_days)
            end_date = min(next_date - timedelta(days=1), date_to)
            intervals.append((current_date, end_date))
            current_date = next_date
        return intervals

    def download_satellite_data(
        self,
        date_from: Dict[str, datetime],
        date_to: Dict[str, datetime],
        interval_days: Dict[str, int],
        bbox: BBox,
        size: Dict[str, Tuple[int, int]],
        mosaicking_order: Dict[str, str],
        satellite: str,
        data_collection: DataCollection,
        resolution: Optional[int] = None
    ):
        """
        Downloads data for a specific satellite by splitting the requested time period into intervals and
        issuing Sentinel Hub requests for each interval.

        Parameters:
          date_from (dict): Dictionary mapping satellite names to the start datetime.
          date_to (dict): Dictionary mapping satellite names to the end datetime.
          interval_days (dict): Dictionary mapping satellite names to the interval (in days) between acquisitions.
          bbox (BBox): Bounding box for the area of interest.
          size (dict): Dictionary mapping satellite names to desired image size (width, height).
          mosaicking_order (dict): Dictionary mapping satellite names to the mosaicking order.
          satellite (str): The satellite for which data are being downloaded.
          data_collection (DataCollection): The DataCollection object representing the satellite's data.
          resolution (int, optional): If provided, overrides the size using the specified resolution.

        Returns:
          None. Downloaded data are saved directly in the configured data folder.
        """
        # Retrieve satellite-specific download parameters.
        sat_date_from = date_from[satellite]
        sat_date_to = date_to[satellite]
        sat_interval = interval_days[satellite]
        sat_size = size[satellite]
        sat_mosaicking_order = mosaicking_order[satellite]

        # Create intervals between the start and end dates.
        date_intervals = self.generate_date_intervals(sat_date_from, sat_date_to, sat_interval)
        total_intervals = len(date_intervals)

        # Loop over each interval and attempt to download the corresponding imagery.
        for idx, (date_in, date_fin) in enumerate(date_intervals, start=1):
            logger.info(f"Downloading {satellite} images for interval: {date_in.date()} to {date_fin.date()} ({idx}/{total_intervals})")
            try:
                self.download_interval(
                    date_in, date_fin, bbox, sat_size, sat_mosaicking_order,
                    satellite, data_collection, resolution
                )
                logger.info("Data downloaded and saved.")
            except Exception as e:
                logger.error(f"Failed to download {satellite} data for interval {date_in.date()} to {date_fin.date()}: {e}")

    def download_interval(
        self,
        date_in: datetime,
        date_fin: datetime,
        bbox: BBox,
        size: Tuple[int, int],
        mosaicking_order: str,
        satellite: str,
        data_collection: DataCollection,
        resolution: Optional[int] = None
    ):
        """
        Constructs and executes a Sentinel Hub request for a given time interval and downloads the imagery.

        Parameters:
          date_in (datetime): Start date of the interval.
          date_fin (datetime): End date of the interval.
          bbox (BBox): Bounding box for the area of interest.
          size (Tuple[int, int]): Desired image dimensions (width, height) in pixels.
          mosaicking_order (str): Mosaicking order to apply for image selection.
          satellite (str): Satellite name.
          data_collection (DataCollection): Data collection object for the satellite.
          resolution (int, optional): If specified, overrides size based on resolution.

        Returns:
          None. The downloaded image is saved in self.data_folder.
        """
        # Retrieve the list of bands requested by the evalscript for this satellite.
        requested_bands = self.evalscript_band_info.get(satellite, [])
        available_bands = data_collection.bands
        available_band_names = [band.name for band in available_bands]

        # Include metabands if present.
        if hasattr(data_collection, 'metabands'):
            available_metabands = [band.name for band in data_collection.metabands]
            available_band_names += available_metabands

        # Verify that all requested bands are available (skip check for DEM).
        unavailable_bands = [band for band in requested_bands if band not in available_band_names]
        if unavailable_bands and satellite != 'DEM':
            raise ValueError(
                f"The following bands are not available in the data collection '{data_collection}': {unavailable_bands}. "
                f"Available bands are: {available_band_names}"
            )

        # If a resolution is provided, compute the image size based on the bounding box and resolution.
        if resolution:
            size = bbox_to_dimensions(bbox, resolution=resolution)

        # Create the Sentinel Hub request using the evalscript and other parameters.
        request = SentinelHubRequest(
            evalscript=self.evalscripts[satellite],
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=data_collection,
                    time_interval=(date_in, date_fin),
                    mosaicking_order=mosaicking_order
                )
            ],
            responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
            bbox=bbox,
            size=size,
            config=self.config,
            data_folder=self.data_folder
        )
        # Execute the request; images are saved automatically.
        request.get_data(save_data=True)
