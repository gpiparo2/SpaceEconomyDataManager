#!/usr/bin/env python3
"""
sdm.py
------
This module defines the SDM class, a wrapper that initializes and provides access to all the core modules:
  - DataDownload: For downloading satellite images via Sentinel Hub.
  - DataManipulator: For organizing and renaming downloaded files.
  - DataVisualizer: For visualizing satellite imagery and inspecting TFRecord datasets.
  - DatasetPreparation: For creating and processing TFRecord datasets for machine learning.

Users can directly call, for example, sdm.data_downloader.download_images(...) or 
sdm.data_manipulator.manipulate_data(...), without needing additional wrapper methods.
"""

import os
from typing import Optional
from datetime import datetime
from sentinelhub import SHConfig

# Import core modules using relative imports.
from .data_download import DataDownload
from .data_manipulator import DataManipulator
from .data_visualizer import DataVisualizer
from .dataset_preparation import DatasetPreparation


class SDM:
    """
    A wrapper class that aggregates and initializes the core modules for satellite data management.

    This class does not implement additional wrapper methods. Instead, it instantiates the following modules:
      - data_downloader: For downloading satellite images.
      - data_manipulator: For organizing, renaming, and preparing the downloaded data.
      - data_visualizer: For visualizing images and inspecting TFRecord datasets.
      - dataset_preparer: For constructing TFRecord datasets for machine learning.

    Users can access these modules directly via the SDM instance.
    """

    def __init__(self,
                 config: Optional[SHConfig] = None,
                 client_id: Optional[str] = None,
                 client_secret: Optional[str] = None,
                 config_name: str = 'sentinelhub_config',
                 data_folder: str = "../data",
                 manipulated_folder: str = "../data_off",
                 tfrecord_folder: str = "../tfrecords"):
        """
        Initializes the SDM class and its core module instances.

        Parameters:
          config (SHConfig, optional): An existing Sentinel Hub configuration. If not provided, a new configuration
                                       is created using client_id and client_secret.
          client_id (str, optional): Sentinel Hub client ID (used if config is not provided).
          client_secret (str, optional): Sentinel Hub client secret (used if config is not provided).
          config_name (str): Name under which to save the Sentinel Hub configuration.
          data_folder (str): Directory containing raw downloaded satellite data.
          manipulated_folder (str): Directory where organized (post-processed) data will be stored.
          tfrecord_folder (str): Directory where TFRecord files will be saved.

        Side Effects:
          Initializes and stores instances of DataDownload, DataManipulator, DataVisualizer, and DatasetPreparation.
        """
        # Use the provided configuration if available; otherwise, create a new one and set credentials.
        if config and isinstance(config, SHConfig):
            self.config = config
        else:
            self.config = SHConfig()
            self.config.sh_client_id = client_id
            self.config.sh_client_secret = client_secret
            self.config.save(config_name)
        
        # Initialize core module instances with the proper folder paths.
        self.data_downloader = DataDownload(config=self.config, data_folder=data_folder)
        self.data_manipulator = DataManipulator(source_folder=data_folder, destination_folder=manipulated_folder)
        self.data_visualizer = DataVisualizer()
        self.dataset_preparer = DatasetPreparation(data_folder=manipulated_folder, tfrecord_folder=tfrecord_folder)
