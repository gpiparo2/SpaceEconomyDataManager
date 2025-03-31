#!/usr/bin/env python3
"""
data_visualizer.py
------------------
This module provides the DataVisualizer class that contains utility functions for visualizing
satellite imagery and inspecting TFRecord datasets produced by the custom dataset builder.

Key functionalities include:
  - Displaying a single image or each band in a grid.
  - Applying a polygon mask to images.
  - Calculating and displaying vegetation indices (e.g., NDVI).
  - Inspecting and visualizing the contents of custom TFRecord files, including DEM data.
  
Additional modifications:
  - In inspect_and_visualize_custom_tfrecord, if the activation label is a scalar value,
    the value is printed at the beginning; if it is a 2D map, it is displayed first.
  
All functions are documented with detailed inline comments and comprehensive docstrings.
"""

import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import tensorflow as tf

class DataVisualizer:
    """
    Provides a set of utility functions to visualize satellite imagery and inspect TFRecord datasets.
    """
    def __init__(self):
        """
        Initializes the DataVisualizer.
        """
        pass

    def display_image(self, image_array: np.ndarray, title: str = "Satellite Image", cmap: str = "gray", save_path: str = None):
        """
        Displays a single image. For multi-band images, if at least three bands are present, an RGB composite is shown.
        
        Parameters:
          image_array (np.ndarray): Image array (height, width, nbands).
          title (str): Title of the plot.
          cmap (str): Colormap to use for single-band images.
          save_path (str, optional): File path to save the figure.
        
        Returns:
          None.
        """
        plt.figure(figsize=(8, 6))
        if image_array.ndim == 2:
            plt.imshow(image_array, cmap=cmap)
        elif image_array.ndim == 3:
            if image_array.shape[2] >= 3:
                # Normalize the first three bands for an RGB composite.
                img_norm = (image_array[:, :, :3] - np.min(image_array[:, :, :3])) / (np.ptp(image_array[:, :, :3]) + 1e-8)
                plt.imshow(img_norm)
            else:
                plt.imshow(image_array[:, :, 0], cmap=cmap)
        plt.title(title)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    def display_all_bands(self, image_array: np.ndarray, band_names: list = None, save_path: str = None):
        """
        Displays each band of a multi-band image in a grid layout.
        
        Parameters:
          image_array (np.ndarray): Image array with shape (height, width, nbands).
          band_names (list, optional): List of band names to label the subplots.
          save_path (str, optional): File path to save the resulting plot.
        
        Returns:
          None.
        """
        if image_array.ndim != 3:
            raise ValueError("Image array must be 3-dimensional (height, width, nbands)")
        nbands = image_array.shape[2]
        cols = math.ceil(math.sqrt(nbands))
        rows = math.ceil(nbands / cols)
        plt.figure(figsize=(4 * cols, 4 * rows))
        for i in range(nbands):
            plt.subplot(rows, cols, i + 1)
            band = image_array[:, :, i]
            title = band_names[i] if band_names and i < len(band_names) else f"Band {i+1}"
            plt.imshow(band, cmap='viridis')
            plt.title(f"{title}\nMin: {band.min():.2f} | Max: {band.max():.2f}")
            plt.axis('off')
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    def apply_polygon_mask(self, raster_path: str, geojson_path: str, crop: bool = False) -> np.ndarray:
        """
        Applies a polygon mask to a raster image, setting pixels outside the polygon to zero.
        
        Parameters:
          raster_path (str): Path to the TIFF file.
          geojson_path (str): Path to the GeoJSON file with the polygon geometry.
          crop (bool): If True, crops the image to the polygon's bounding box.
        
        Returns:
          np.ndarray: The masked image array.
        """
        with rasterio.open(raster_path) as src:
            geojson = gpd.read_file(geojson_path)
            out_image, out_transform = mask(src, geojson.geometry, crop=crop)
            image_array = np.moveaxis(out_image, 0, -1)
        return image_array

    def calculate_ndvi(self, image_array: np.ndarray, red_band_index: int = 3, nir_band_index: int = 7) -> np.ndarray:
        """
        Calculates the NDVI (Normalized Difference Vegetation Index) for a Sentinel-2 image.
        
        Parameters:
          image_array (np.ndarray): Input image array.
          red_band_index (int): Index of the red band.
          nir_band_index (int): Index of the near-infrared band.
        
        Returns:
          np.ndarray: The computed NDVI array.
        """
        red = image_array[:, :, red_band_index].astype(float)
        nir = image_array[:, :, nir_band_index].astype(float)
        denominator = (nir + red)
        denominator[denominator == 0] = 1e-5
        return (nir - red) / denominator

    def visualize_ndvi(self, image_array: np.ndarray, red_band_index: int = 3, nir_band_index: int = 7, save_path: str = None):
        """
        Computes and displays the NDVI for a Sentinel-2 image.
        
        Parameters:
          image_array (np.ndarray): Input Sentinel-2 image array.
          red_band_index (int): Index of the red band.
          nir_band_index (int): Index of the near-infrared band.
          save_path (str, optional): File path to save the NDVI plot.
        
        Returns:
          None.
        """
        ndvi = self.calculate_ndvi(image_array, red_band_index, nir_band_index)
        plt.figure(figsize=(8, 6))
        plt.imshow(ndvi, cmap='RdYlGn')
        plt.colorbar(label='NDVI', fraction=0.046, pad=0.04)
        plt.title("NDVI")
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    def inspect_tfrecord(self, tfrecord_path: str):
        """
        Inspects a TFRecord file by parsing and printing metadata for each example.
        
        Parameters:
          tfrecord_path (str): Path to the TFRecord file.
        
        Returns:
          None.
        """
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'channels': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'acquisition_date': tf.io.FixedLenFeature([], tf.string),
            'satellite': tf.io.FixedLenFeature([], tf.string, default_value="Unknown")
        }
        def _parse_function(proto):
            return tf.io.parse_single_example(proto, feature_description)
        parsed_dataset = dataset.map(_parse_function)
        for idx, record in enumerate(parsed_dataset):
            height = int(record['height'].numpy())
            width = int(record['width'].numpy())
            channels = int(record['channels'].numpy())
            acq_date = record['acquisition_date'].numpy().decode('utf-8')
            satellite = record['satellite'].numpy().decode('utf-8')
            print(f"Example {idx}: Satellite: {satellite}, Acquisition Date: {acq_date}, Shape: {height} x {width} x {channels}")

    def inspect_and_visualize_custom_tfrecord(self, tfrecord_path: str, crop: bool = False, crop_factor: int = 1) -> None:
        """
        Inspects and visualizes the content of a custom TFRecord file.

        For each sensor, the function decodes and reshapes the image data using stored metadata.
        - In non-cropped mode, a single tf.train.Example contains the full image (or time-series);
        the function displays each band and an RGB composite if available.
        - In cropped mode, each example corresponds to a patch.
        The function groups patches (per sensor and time step) and reassembles the full image.
        
        Additional functionality:
        - Label Handling:
            * If the activation label is a 2D map, it is reassembled and displayed separately.
            Its contour (computed at level 0.5 via skimage.find_contours) is overlaid on each sensor’s canvas in bright red.
            * If the label is a scalar value (vineyard label), that value is appended to each sensor's title.
        - For each sensor’s canvas, the title includes:
            * Sensor name and timestep number.
            * Acquisition date range (start and end dates).
            * Pixel dimensions (width × height) shown on the axes.
            * Normalization status.
            * (For vineyard datasets, additional label threshold and class info may be added.)
        
        Parameters:
        tfrecord_path (str): Path to the TFRecord file.
        crop (bool): Indicates whether the TFRecord contains cropped images.
        crop_factor (int): Crop factor used during TFRecord creation.
        
        Returns:
        None.
        """
        import matplotlib.pyplot as plt
        import math
        import numpy as np
        import tensorflow as tf
        import json

        # Helper function: reassemble patches for one band.
        def reassemble_patch_grid_band(patches: np.ndarray, crop_factor: int, gap: int = 5) -> np.ndarray:
            """
            Reassembles a grid of patches for a single band.

            Parameters:
            patches (np.ndarray): Array of patches with shape (n_patches, patch_H, patch_W).
            crop_factor (int): Number of patches per side (n_patches should equal crop_factor^2).
            gap (int): Gap (in pixels) to insert between patches.

            Returns:
            np.ndarray: Reassembled image grid.

            Raises:
            ValueError: If the number of patches does not equal crop_factor^2.
            """
            n_patches, patch_H, patch_W = patches.shape
            if n_patches != crop_factor * crop_factor:
                raise ValueError(f"Number of patches ({n_patches}) does not equal crop_factor^2 ({crop_factor**2}).")
            rows = []
            for r in range(crop_factor):
                row_patches = patches[r * crop_factor:(r + 1) * crop_factor]
                padded = [np.pad(p, ((0, 0), (0, gap)), mode='constant', constant_values=0) for p in row_patches]
                row = np.hstack(padded)
                rows.append(row)
            grid = np.vstack([np.pad(r_img, ((0, gap), (0, 0)), mode='constant', constant_values=0) for r_img in rows])
            return grid

        # --- Load all records from the TFRecord ---
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        all_records = [rec for rec in raw_dataset]
        if not all_records:
            print("No records found in TFRecord.")
            return

        # Initialize groups.
        sensor_groups = {}   # sensor name -> list of feature dicts
        dem_group = []       # List to hold DEM features.
        label_examples = []  # List for label features (tuple: (label_type, feats))
        for rec in all_records:
            example = tf.train.Example.FromString(rec.numpy())
            feats = example.features.feature
            # For labels, prefer activation_label (2D map) over vineyard_label (scalar).
            if "activation_label" in feats:
                label_examples.append(("activation", feats))
            elif "vineyard_label" in feats:
                label_examples.append(("vineyard", feats))
            elif "ET_label" in feats:
                label_examples.append(("ET_label", feats)) 
            # Group sensor features; group DEM separately.
            for key in feats:
                if key.endswith("_image"):
                    if key.startswith("DEM"):
                        dem_group.append(feats)
                    else:
                        sensor = key.rsplit("_", 1)[0]
                        sensor_groups.setdefault(sensor, []).append(feats)

        gap = 5

        # --- Process Label ---
        label_is_scalar = False
        label_value = None
        label_img = None
        label_contours = None
        if label_examples:
            # Determine label type from the first label example.
            label_type, _ = label_examples[0]
            if label_type == "activation":
                print("\nReassembling Activation Label:")
                label_list = []
                for l_type, feats in label_examples:
                    label_bytes = feats["activation_label"].bytes_list.value[0]
                    label_height = int(feats["activation_label_height"].int64_list.value[0])
                    label_width = int(feats["activation_label_width"].int64_list.value[0])
                    # Reshape as a 2D array.
                    lab = np.frombuffer(label_bytes, dtype=np.uint8).reshape((label_height, label_width))
                    label_list.append(lab)
                try:
                    labels_all = np.stack(label_list, axis=0)
                    if labels_all.size == 1:
                        label_is_scalar = True
                        label_value = labels_all.item()
                    else:
                        label_img = reassemble_patch_grid_band(labels_all, crop_factor, gap)
                except Exception as e:
                    print(f"Error reassembling activation label patches: {e}")
                    label_img = np.concatenate(label_list, axis=1)
                # Compute contours using skimage for efficiency.
                if label_img is not None and not label_is_scalar:
                    try:
                        from skimage.measure import find_contours
                        label_contours = find_contours(label_img, 0.5)
                    except Exception as e:
                        print(f"Error computing label contours: {e}")
            elif label_type == "vineyard":
                # Process vineyard label as a scalar value.
                label_is_scalar = True
                label_value = label_examples[0][1]["vineyard_label"].int64_list.value[0]
            else:
                label_is_scalar = True
                label_value = label_examples[0][1]["ET_label"].float_list.value[0]
        else:
            print("No label found in TFRecord.")

        # If a 2D label map exists, display it separately.
        if label_img is not None and not label_is_scalar:
            plt.figure(figsize=(6, 6))
            im = plt.imshow(label_img, cmap="viridis")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title("Activation Label")
            plt.xlabel("x (pixel)")
            plt.ylabel("y (pixel)")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        # --- Sensor-specific Band Names Dictionary ---
        band_names_dict = {
            "Sentinel-2": ["Aerosol", "Blue", "Green", "Red", "Red Edge 1", "Red Edge 2", "Red Edge 3", "NIR 1", "NIR 2", "Water Vapour", "SWIR1", "SWIR2", "AOT", "SCL", "SNW", "CLD", "dataMask"],
            "Sentinel-1": ["VV", "VH", "HH", "HV"],
            "Sentinel-3-OLCI": [f"OLCI Band B{i:02d}" for i in range(1, 22)],
            "Sentinel-3-SLSTR-Thermal": ["SLSTR-Thermal Band S7", "SLSTR-Thermal Band S8", "SLSTR-Thermal Band S9", "SLSTR-Thermal Band F1", "SLSTR-Thermal Band F2"],
        }

        # --- Process and visualize sensor groups (excluding DEM) ---
        for sensor, feats_list in sensor_groups.items():
            print(f"\nReassembling sensor: {sensor}")
            # Retrieve stored acquisition dates.
            try:
                dates_json = feats_list[0].get(f"{sensor}_dates", None)
                if dates_json:
                    dates_list = json.loads(dates_json.bytes_list.value[0].decode('utf-8'))
                    date_start = dates_list[0] if dates_list else "N/A"
                    date_end = dates_list[-1] if dates_list else "N/A"
                else:
                    date_start, date_end = "N/A", "N/A"
            except Exception:
                date_start, date_end = "N/A", "N/A"

            # Retrieve correct band names.
            try:
                first_feats = feats_list[0]
                channels = int(first_feats[f"{sensor}_channels"].int64_list.value[0])
                if sensor in band_names_dict:
                    full_band_names = band_names_dict[sensor]
                    if channels <= len(full_band_names):
                        band_names = full_band_names[:channels]
                    else:
                        band_names = full_band_names + [f"Band {i+1}" for i in range(len(full_band_names), channels)]
                else:
                    band_names = [f"Band {i+1}" for i in range(channels)]
            except Exception:
                band_names = [f"Band {i+1}" for i in range(1)]

            # Determine number of time steps.
            try:
                n_steps = int(first_feats.get(f"{sensor}_n_steps", tf.train.Int64List(value=[1])).int64_list.value[0])
            except Exception:
                n_steps = 1
            n_patches = len(feats_list)
            print(f"{sensor} (cropped): n_steps = {n_steps}, total patches = {n_patches}")

            patch_list = []
            for feats in feats_list:
                image_bytes = feats[f"{sensor}_image"].bytes_list.value[0]
                height = int(feats[f"{sensor}_height"].int64_list.value[0])
                width = int(feats[f"{sensor}_width"].int64_list.value[0])
                channels = int(feats[f"{sensor}_channels"].int64_list.value[0])
                img = np.frombuffer(image_bytes, dtype=np.float32)
                if n_steps > 1:
                    img = img.reshape((n_steps, height, width, channels))
                else:
                    img = img.reshape((height, width, channels))
                    img = np.expand_dims(img, axis=0)
                patch_list.append(img)
            patches_all = np.stack(patch_list, axis=0)

            # For each time step, reassemble and display patches.
            for t in range(n_steps):
                patches_t = patches_all[:, t, :, :, :]
                channels_count = patches_t.shape[-1]
                band_images = []
                for b in range(channels_count):
                    patches_band = patches_t[..., b]
                    try:
                        grid_img = reassemble_patch_grid_band(patches_band, crop_factor, gap)
                    except Exception as e:
                        print(f"Error reassembling {sensor} band {b+1} at time step {t+1}: {e}")
                        continue
                    band_images.append(grid_img)
                # Compute default RGB composite if available.
                rgb_composite = None
                rgb_map = {
                    "Sentinel-2": (3, 2, 1),
                    "Sentinel-3-OLCI": (7, 5, 3)
                }
                if sensor in rgb_map and channels_count >= max(rgb_map[sensor]) + 1:
                    r_idx, g_idx, b_idx = rgb_map[sensor]
                    try:
                        grid_r = reassemble_patch_grid_band(patches_t[..., r_idx], crop_factor, gap)
                        grid_g = reassemble_patch_grid_band(patches_t[..., g_idx], crop_factor, gap)
                        grid_b = reassemble_patch_grid_band(patches_t[..., b_idx], crop_factor, gap)
                        rgb_composite = np.stack([grid_r, grid_g, grid_b], axis=-1)
                        rgb_composite = (rgb_composite - np.min(rgb_composite)) / (np.ptp(rgb_composite) + 1e-8)
                    except Exception as e:
                        print(f"Error assembling RGB composite for {sensor} at time step {t+1}: {e}")

                # For Sentinel-1, compute a custom composite: [VV, 2*VH, VV/VH/100.0].
                custom_composite = None
                if sensor == "Sentinel-1" and channels_count >= 2:
                    try:
                        VV = patches_t[..., 0]
                        VH = patches_t[..., 1]
                        with np.errstate(divide='ignore', invalid='ignore'):
                            custom_R = VV
                            custom_G = 2 * VH
                            custom_B = np.where(VH == 0, 0, VV / VH / 100.0)
                        grid_R = reassemble_patch_grid_band(custom_R, crop_factor, gap)
                        grid_G = reassemble_patch_grid_band(custom_G, crop_factor, gap)
                        grid_B = reassemble_patch_grid_band(custom_B, crop_factor, gap)
                        custom_composite = np.stack([grid_R, grid_G, grid_B], axis=-1)
                        #custom_composite = (custom_composite - np.min(custom_composite)) / (np.ptp(custom_composite) + 1e-8)
                    except Exception as e:
                        print(f"Error assembling custom RGB composite for Sentinel-1 at time step {t+1}: {e}")

                patch_H, patch_W = patches_t.shape[1], patches_t.shape[2]
                # Prepare title string.
                title = f"{sensor} - Timestep {t+1}\nDate Range: {date_start} to {date_end}"
                if label_is_scalar:
                    title += f"\nLabel: {label_value}"
                # Set up canvas: number of plots.
                n_plots = len(band_images)
                if rgb_composite is not None:
                    n_plots += 1
                if sensor == "Sentinel-1" and custom_composite is not None:
                    n_plots += 1
                cols = math.ceil(math.sqrt(n_plots))
                rows_subplot = math.ceil(n_plots / cols)
                fig, axs = plt.subplots(rows_subplot, cols, figsize=(4 * cols, 4 * rows_subplot))
                axs = np.array(axs).flatten()
                plot_idx = 0
                # Plot each band image.
                for i, img in enumerate(band_images):
                    im = axs[plot_idx].imshow(img, cmap='viridis')
                    plt.colorbar(im, ax=axs[plot_idx], fraction=0.046, pad=0.04)
                    axs[plot_idx].set_title(f"{band_names[i]}", fontsize=9)
                    axs[plot_idx].set_xlabel("x (pixel)", fontsize=8)
                    axs[plot_idx].set_ylabel("y (pixel)", fontsize=8)
                    axs[plot_idx].axis('on')
                    # Overlay label contour if available.
                    if (label_contours is not None) and (not label_is_scalar) and (label_img is not None):
                        scale_y = img.shape[0] / label_img.shape[0]
                        scale_x = img.shape[1] / label_img.shape[1]
                        for contour in label_contours:
                            contour = np.atleast_2d(contour)
                            if contour.ndim != 2 or contour.shape[1] < 2:
                                continue
                            scaled_contour = np.copy(contour)
                            scaled_contour[:, 0] *= scale_y
                            scaled_contour[:, 1] *= scale_x
                            try:
                                axs[plot_idx].plot(scaled_contour[:, 1], scaled_contour[:, 0], color="red", linewidth=1)
                            except Exception as e:
                                print(f"Contour plotting error on {sensor} band {i+1} at timestep {t+1}: {e}")
                    plot_idx += 1

                # Plot the RGB composite if available.
                if rgb_composite is not None:
                    im = axs[plot_idx].imshow(rgb_composite)
                    axs[plot_idx].set_title("RGB Composite", fontsize=9)
                    axs[plot_idx].set_xlabel("x (pixel)", fontsize=8)
                    axs[plot_idx].set_ylabel("y (pixel)", fontsize=8)
                    axs[plot_idx].axis('on')
                    if (label_contours is not None) and (not label_is_scalar) and (label_img is not None):
                        scale_y = rgb_composite.shape[0] / label_img.shape[0]
                        scale_x = rgb_composite.shape[1] / label_img.shape[1]
                        for contour in label_contours:
                            contour = np.atleast_2d(contour)
                            if contour.ndim != 2 or contour.shape[1] < 2:
                                continue
                            scaled_contour = np.copy(contour)
                            scaled_contour[:, 0] *= scale_y
                            scaled_contour[:, 1] *= scale_x
                            try:
                                axs[plot_idx].plot(scaled_contour[:, 1], scaled_contour[:, 0], color="red", linewidth=1)
                            except Exception as e:
                                print(f"Contour plotting error on RGB composite for {sensor} at timestep {t+1}: {e}")
                    plot_idx += 1

                # For Sentinel-1, plot the custom composite.
                if sensor == "Sentinel-1" and custom_composite is not None:
                    im = axs[plot_idx].imshow(custom_composite)
                    axs[plot_idx].set_title("Custom Sentinel-1 Composite", fontsize=9)
                    axs[plot_idx].set_xlabel("x (pixel)", fontsize=8)
                    axs[plot_idx].set_ylabel("y (pixel)", fontsize=8)
                    axs[plot_idx].axis('on')
                    if (label_contours is not None) and (not label_is_scalar) and (label_img is not None):
                        scale_y = custom_composite.shape[0] / label_img.shape[0]
                        scale_x = custom_composite.shape[1] / label_img.shape[1]
                        for contour in label_contours:
                            contour = np.atleast_2d(contour)
                            if contour.ndim != 2 or contour.shape[1] < 2:
                                continue
                            scaled_contour = np.copy(contour)
                            scaled_contour[:, 0] *= scale_y
                            scaled_contour[:, 1] *= scale_x
                            try:
                                axs[plot_idx].plot(scaled_contour[:, 1], scaled_contour[:, 0], color="red", linewidth=1)
                            except Exception as e:
                                print(f"Contour plotting error on custom Sentinel-1 composite at timestep {t+1}: {e}")
                    plot_idx += 1

                # Turn off any extra axes.
                for j in range(plot_idx, len(axs)):
                    axs[j].axis('off')
                plt.suptitle(title)
                plt.tight_layout()
                plt.show()

        # --- Process and visualize DEM data if available ---
        if dem_group:
            print("\nReassembling DEM data:")
            n_steps = 1  # Typically, DEM is a single image.
            n_patches = len(dem_group)
            patch_list = []
            for feats in dem_group:
                image_bytes = feats["DEM_image"].bytes_list.value[0]
                height = int(feats["DEM_height"].int64_list.value[0])
                width = int(feats["DEM_width"].int64_list.value[0])
                channels = int(feats["DEM_channels"].int64_list.value[0])
                img = np.frombuffer(image_bytes, dtype=np.float32).reshape((height, width, channels))
                img = np.expand_dims(img, axis=0)
                patch_list.append(img)
            patches_all = np.stack(patch_list, axis=0)
            try:
                dem_image = reassemble_patch_grid_band(patches_all[:, 0, :, :, 0], crop_factor, gap)
                plt.figure(figsize=(6, 6))
                im = plt.imshow(dem_image, cmap='terrain')
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.title("DEM")
                plt.xlabel("x (pixel)")
                plt.ylabel("y (pixel)")
                # Overlay DEM label contour if available.
                if (label_contours is not None) and (not label_is_scalar) and (label_img is not None):
                    scale_y = dem_image.shape[0] / label_img.shape[0]
                    scale_x = dem_image.shape[1] / label_img.shape[1]
                    for contour in label_contours:
                        contour = np.atleast_2d(contour)
                        if contour.ndim != 2 or contour.shape[1] < 2:
                            continue
                        scaled_contour = np.copy(contour)
                        scaled_contour[:, 0] *= scale_y
                        scaled_contour[:, 1] *= scale_x
                        try:
                            plt.gca().plot(scaled_contour[:, 1], scaled_contour[:, 0], color="red", linewidth=1)
                        except Exception as e:
                            print(f"Contour plotting error on DEM: {e}")
                plt.axis('on')
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error reassembling DEM patches: {e}")

        if not label_examples:
            print("No activation label found in TFRecord.")





    def visualize_batch(self, dataset: tf.data.Dataset, sensors_to_show: list = None,
                            max_examples: int = 4, crop: bool = False, crop_factor: int = 1) -> None:
            """
            Visualizes a batch of examples from the dataset.
            
            In non-cropped mode:
            - For each example, if the sensor data is a full image or time series (shape: [T, H, W, channels] or [H, W, channels]),
                the function displays (for example, the first time step for time series).
            
            In cropped mode:
            - Each example corresponds to a single patch.
            - For time series sensors, the image is expected to have shape (T, patch_H, patch_W, channels),
                and the function displays the first time step for visualization.
            
            Parameters:
            dataset (tf.data.Dataset): The dataset yielding (inputs, label) tuples.
            sensors_to_show (list, optional): List of sensor names to display (e.g., ["Sentinel-1"]). If None, all sensors are shown.
            max_examples (int): Maximum number of batch examples to visualize.
            crop (bool): Indicates if the TFRecord was built with cropping.
            crop_factor (int): Crop factor used during TFRecord creation.
            
            Returns:
            None.
            """
            import matplotlib.pyplot as plt
            import math
            import numpy as np

            # Define RGB mapping.
            rgb_map = {
                "Sentinel-2": (3, 2, 1),
                "Sentinel-3-OLCI": (7, 5, 3)
            }
            
            def process_sensor_image(image: np.ndarray, crop: bool) -> np.ndarray:
                """
                Processes the sensor image for visualization.
                
                In both cropped and non-cropped mode:
                - If the image is time-series (shape: (T, H, W, channels)), returns the first time step.
                - Otherwise, returns the image as is.
                """
                if image.ndim == 4:
                    return image[0]
                else:
                    return image

            for batch_inputs, batch_labels in dataset.take(max_examples):
                batch_size = batch_labels.shape[0]
                print(f"Visualizing a batch of {batch_size} examples.")
                for i in range(batch_size):
                    # Convert tensors to numpy arrays.
                    inputs_example = {s: batch_inputs[s][i].numpy() for s in batch_inputs}
                    label_example = batch_labels[i].numpy()
                    if sensors_to_show is None:
                        sensors_to_show = list(inputs_example.keys())
                    num_sensors = len(sensors_to_show)
                    fig, axs = plt.subplots(1, num_sensors + 1, figsize=(4 * (num_sensors + 1), 4))
                    for j, sensor in enumerate(sensors_to_show):
                        image = inputs_example[sensor]
                        proc_image = process_sensor_image(image, crop)
                        channels = proc_image.shape[-1]
                        # Custom handling for Sentinel-1.
                        if sensor == "Sentinel-1" and channels == 2:
                            VV = proc_image[..., 0]
                            VH = proc_image[..., 1]
                            comp = np.stack([VV, 2*VH, VV/(VH+1e-8)/100.0], axis=-1)
                            comp = (comp - np.min(comp)) / (np.ptp(comp) + 1e-8)
                            axs[j].imshow(comp)
                        # Custom handling for Sentinel-3-SLSTR-Thermal.
                        elif sensor == "Sentinel-3-SLSTR-Thermal" and channels >= 3:
                            comp = proc_image[..., :3]
                            comp = (comp - np.min(comp)) / (np.ptp(comp) + 1e-8)
                            axs[j].imshow(comp)
                        elif sensor in rgb_map and channels >= max(rgb_map[sensor]) + 1:
                            indices = rgb_map[sensor]
                            comp = proc_image[..., list(indices)]
                            comp = (comp - np.min(comp)) / (np.ptp(comp) + 1e-8)
                            axs[j].imshow(comp)
                        else:
                            axs[j].imshow(proc_image)
                        axs[j].set_title(f"{sensor}")
                        axs[j].axis('off')
                    if label_example.ndim == 3:
                        axs[-1].imshow(label_example[:, :, 0], cmap='viridis')
                    else:
                        axs[-1].imshow(label_example, cmap='viridis')
                    axs[-1].set_title("Label")
                    axs[-1].axis('off')
                    plt.tight_layout()
                    plt.show()