#!/usr/bin/env python3
"""
vineyard_dashboard.py
---------------------
This module defines the interactive dashboard for building and visualizing vineyard datasets.
It uses ipywidgets and functions from the common GUI module to create sensor-specific options
and dashboard controls. Additionally, it provides a Sentinel Hub authentication section,
a "Load Last Session" button, and a backup session panel for custom backup file paths.
It also includes a refresh button to clear outputs and update file listings.
All fields are empty by default.

The dashboard enables users to configure parameters for vineyard dataset building and to visualize the resulting TFRecord files.
"""

import os
import ipywidgets as widgets
from IPython.display import display, HTML
from sentinelhub import SHConfig

# Import common GUI functions from the shared module.
from ...gui.common_gui import (
    inject_css,
    build_sensor_parameters_container,
    create_auth_section,
    create_load_session_section,
    create_backup_section,
    create_refresh_button,
    extract_sensor_params,
    load_sensor_params,  
    save_last_session,
    SHConfigManager  # Now using the singleton manager

)
# Import core and analysis modules using relative imports.
from ...analyses.vineyard.vineyard_dataset_builder import VineyardDatasetBuilder
from ...core.sdm import SDM

def build_vineyard_dashboard():
    """
    Builds and returns the dashboard widget for vineyard dataset building.
    
    Returns:
      widgets.VBox: The complete vineyard dashboard.
    """
    inject_css()
    
    # --- Vineyard-specific Dataset Parameters ---
    vineyard_info_base_folder_w = widgets.Text(
        value="",
        description="Vineyard Info Base Folder:",
        layout=widgets.Layout(width='90%')
    )
    vineyard_info_folder_w = widgets.Text(
        value="",
        description="Vineyard Info Folder:",
        layout=widgets.Layout(width='90%')
    )
    base_data_folder_w = widgets.Text(
        value="",
        description="Base Raw Folder:",
        layout=widgets.Layout(width='90%')
    )
    base_manipulated_folder_w = widgets.Text(
        value="",
        description="Manipulated Folder:",
        layout=widgets.Layout(width='90%')
    )
    tfrecord_folder_w = widgets.Text(
        value="",
        description="TFRecord Folder:",
        layout=widgets.Layout(width='90%')
    )
    
    download_w = widgets.Checkbox(value=False, description="Download Data")
    max_vineyards_w = widgets.IntText(
        value=0, 
        description="Max Vineyards:", 
        layout=widgets.Layout(width='90%')
    )
    time_window_w = widgets.IntSlider(
        value=0, min=0, max=60, 
        description="Time Window (days):", 
        layout=widgets.Layout(width='90%')
    )
    apply_normalization_w = widgets.Checkbox(value=False, description="Apply Normalization")
    apply_mask_w = widgets.Checkbox(value=False, description="Apply Mask")
    crop_w = widgets.Checkbox(value=False, description="Crop")
    crop_factor_w = widgets.IntSlider(
        value=1, min=1, max=10, 
        description="Crop Factor:", 
        layout=widgets.Layout(width='90%')
    )
    sensors_w = widgets.SelectMultiple(
        options=["Sentinel-2", "Sentinel-1", "Sentinel-3-OLCI", "Sentinel-3-SLSTR-Thermal", "DEM"],
        value=(),
        description="Sensors:",
        layout=widgets.Layout(width='90%', height='80px')
    )
    sensor_params_container = widgets.VBox([], layout=widgets.Layout(width='100%'))
    def on_sensors_change(change):
        if change['name'] == 'value':
            sensor_params_container.children = [build_sensor_parameters_container(list(change['new']))]
    sensors_w.observe(on_sensors_change, names='value')
    sensor_params_container.children = [build_sensor_parameters_container(list(sensors_w.value))]
    
    binarize_label_w = widgets.Checkbox(value=False, description="Binarize Label")
    label_threshold_w = widgets.FloatText(
        value=0.0, 
        description="Label Threshold:", 
        layout=widgets.Layout(width='90%')
    )
    
    # --- TFRecord Visualization Widgets (must be defined before refresh button) ---
    tfrecord_dropdown_vineyard = widgets.Dropdown(
        options=[], 
        description="TFRecord File:", 
        layout=widgets.Layout(width='90%')
    )
    output_vineyard = widgets.Output(layout=widgets.Layout(width='90%'))
    output_visualize_vineyard = widgets.Output(layout=widgets.Layout(width='90%'))
    
    # --- Define a function to update TFRecord file listings ---
    def update_tfrecord_options_v():
        if os.path.exists(tfrecord_folder_w.value):
            files = [f for f in os.listdir(tfrecord_folder_w.value) if f.endswith(".tfrecord")]
            tfrecord_dropdown_vineyard.options = files
        else:
            tfrecord_dropdown_vineyard.options = []
    
    # Create a refresh button that clears output areas and updates directory listings.
    refresh_button = create_refresh_button(
        refresh_callbacks=[update_tfrecord_options_v],
        output_widgets=[output_vineyard, output_visualize_vineyard]
    )
    
    header = widgets.HBox([
        widgets.HTML("<h2>Dataset Builder - Vineyard</h2>"),
        refresh_button
    ], layout=widgets.Layout(justify_content='space-between', width='90%'))
    
    # --- Sentinel Hub Authentication Section (common) ---
    auth_dict = create_auth_section()
    auth_section = auth_dict["auth_section"]
    config_profile_w = auth_dict["config_profile_w"]
    client_id_w = auth_dict["client_id_w"]
    client_secret_w = auth_dict["client_secret_w"]
    
    # --- Load Last Session Section (common) ---
    def update_widgets_from_session_vineyard(session_data):
        config_profile_w.value = session_data.get("config_profile", "")
        client_id_w.value = session_data.get("client_id", "")
        client_secret_w.value = session_data.get("client_secret", "")
        vineyard_info_base_folder_w.value = session_data.get("vineyard_info_base_folder", "")
        vineyard_info_folder_w.value = session_data.get("vineyard_info_folder", "")
        base_data_folder_w.value = session_data.get("base_data_folder", "")
        base_manipulated_folder_w.value = session_data.get("base_manipulated_folder", "")
        tfrecord_folder_w.value = session_data.get("tfrecord_folder", "")
        download_w.value = session_data.get("download", False)
        time_window_w.value = session_data.get("time_window", 0)
        max_vineyards_w.value = session_data.get("max_vineyards", 0)
        apply_normalization_w.value = session_data.get("apply_normalization", False)
        apply_mask_w.value = session_data.get("apply_mask", False)
        crop_w.value = session_data.get("crop", False)
        crop_factor_w.value = session_data.get("crop_factor", 1)
        sensors = session_data.get("sensors", [])
        sensors_w.value = tuple(sensors)
        sensor_params_container.children = [build_sensor_parameters_container(list(sensors))]
        load_sensor_params(sensor_params_container, session_data.get("sensor_params", {}))
        binarize_label_w.value = session_data.get("binarize_label", False)
        label_threshold_w.value = session_data.get("label_threshold", 0.0)
    
    load_session_button, session_output = create_load_session_section(update_widgets_from_session_vineyard)
    
    # --- Backup Session Section (common) ---
    def collect_session_data_vineyard():
        return {
            "config_profile": config_profile_w.value,
            "client_id": client_id_w.value,
            "client_secret": client_secret_w.value,
            "vineyard_info_base_folder": vineyard_info_base_folder_w.value,
            "vineyard_info_folder": vineyard_info_folder_w.value,
            "base_data_folder": base_data_folder_w.value,
            "base_manipulated_folder": base_manipulated_folder_w.value,
            "tfrecord_folder": tfrecord_folder_w.value,
            "download": download_w.value,
            "time_window": time_window_w.value,
            "max_vineyards": max_vineyards_w.value,
            "apply_normalization": apply_normalization_w.value,
            "apply_mask": apply_mask_w.value,
            "crop": crop_w.value,
            "crop_factor": crop_factor_w.value,
            "sensors": list(sensors_w.value),
            "sensor_params": extract_sensor_params(sensor_params_container),
            "binarize_label": binarize_label_w.value,
            "label_threshold": label_threshold_w.value
        }
    
    backup_section = create_backup_section(collect_session_data_vineyard, update_widgets_from_session_vineyard)
    
    # --- Build Dataset Button ---
    build_button = widgets.Button(
        description="Build Vineyard Dataset", 
        button_style='success', 
        layout=widgets.Layout(width='90%')
    )
    
    def on_build_clicked(b):
        with output_vineyard:
            output_vineyard.clear_output()
            print("Building Vineyard Dataset...")
            try:
                config = SHConfigManager.instance().get_config() or SHConfig()
                sensor_params = {}
                evalscript_params = {}
                download_params = {"interval_days": {}, "size": {}, "mosaicking_order": {}}

                param_mapping = {
                    "Select Sentinel-2 bands": "bands",
                    "Select Sentinel-3 OLCI bands": "bands",
                    "Select Sentinel-3 SLSTR Thermal bands": "bands",
                    "Resolution": "resolution",
                    "Select sample type": "sampleType",
                    "Select units": "units",
                    "Select polarizations": "polarizations",
                    "Select backscatter coefficient": "backCoeff",
                    "Orthorectify data?": "orthorectify",
                    "Select DEM instance": "demInstance"
                }

                evalscript_allowed_params = {
                    "Sentinel-1": ["polarizations", "backCoeff", "orthorectify", "demInstance", "sampleType"],
                    "Sentinel-2": ["bands", "units", "sampleType"],
                    "Sentinel-3-OLCI": ["bands", "sampleType"],
                    "Sentinel-3-SLSTR-Thermal": ["bands", "sampleType"],
                    "DEM": ["demInstance", "sampleType"]
                }

                for sensor_container in sensor_params_container.children[0].children:
                    sensor_label = sensor_container.children[0].value.replace("<b>", "").replace("</b>", "").replace(" Options:", "").strip()
                    controls_vbox = sensor_container.children[1]

                    interval_days = 20  # default
                    size = (128, 128)  # default
                    mosaicking_order = "mostRecent"  # default

                    sensor_params[sensor_label] = {}
                    evalscript_params[sensor_label] = {}

                    for control in controls_vbox.children:
                        if isinstance(control, widgets.VBox):
                            header_text = control.children[0].value.replace("<b>", "").replace(":</b>", "").strip()
                            size_hbox = control.children[1]
                            width_val = size_hbox.children[0].value
                            height_val = size_hbox.children[1].value
                            size = (width_val, height_val)
                        else:
                            val = list(control.value) if isinstance(control.value, tuple) else control.value
                            desc = control.description.replace(f" for {sensor_label}", "").strip()

                            if desc == "Interval Days":
                                interval_days = val
                                download_params["interval_days"][sensor_label] = interval_days
                            elif desc == "Mosaicking Order":
                                mosaicking_order = val
                                download_params["mosaicking_order"][sensor_label] = mosaicking_order
                            else:
                                key = param_mapping.get(desc, desc)
                                if key in evalscript_allowed_params.get(sensor_label, []):
                                    evalscript_params[sensor_label][key] = val
                                    sensor_params[sensor_label][key] = val

                    download_params["size"][sensor_label] = size

                from pprint import pprint

                #pprint(download_params)
                #pprint(sensor_params)
                #pprint(evalscript_params)

                builder = VineyardDatasetBuilder(
                    config=config,
                    vineyard_info_folder=vineyard_info_folder_w.value,
                    base_data_folder=base_data_folder_w.value,
                    base_manipulated_folder=base_manipulated_folder_w.value,
                    tfrecord_folder=tfrecord_folder_w.value,
                    sampleType="FLOAT32",
                    download=download_w.value,
                    binarize_label=binarize_label_w.value,
                    label_threshold=label_threshold_w.value
                )

                builder.build_dataset_for_all_vineyards(
                    max_vineyards=max_vineyards_w.value,
                    download_params=download_params,
                    apply_normalization=apply_normalization_w.value,
                    apply_mask=apply_mask_w.value,
                    crop=crop_w.value,
                    crop_factor=crop_factor_w.value,
                    time_window=time_window_w.value,
                    sensors=list(sensors_w.value),
                    sensor_bands=sensor_params,
                    evalscript_params=evalscript_params
                )

                print("Dataset built successfully.")
                session_data = collect_session_data_vineyard()
                save_last_session(session_data)
                print("Session saved.")

            except Exception as e:
                print("Error building dataset:", e)

    build_button.on_click(on_build_clicked)

    
    # --- Visualization Section ---
    visualize_button_vineyard = widgets.Button(
        description="Visualize TFRecord", 
        button_style='info', 
        layout=widgets.Layout(width='90%')
    )
    
    def on_visualize_clicked_v(b):
        with output_visualize_vineyard:
            output_visualize_vineyard.clear_output()
            tfrecord_file = tfrecord_dropdown_vineyard.value
            if tfrecord_file:
                full_path = os.path.join(tfrecord_folder_w.value, tfrecord_file)
                print(f"Visualizing {full_path} ...")
                config = SHConfigManager.instance().get_config() or SHConfig()
                sdm = SDM(
                    config=config,
                    data_folder=base_data_folder_w.value,
                    manipulated_folder=base_manipulated_folder_w.value,
                    tfrecord_folder=tfrecord_folder_w.value
                )
                sdm.data_visualizer.inspect_and_visualize_custom_tfrecord(
                    full_path, 
                    crop=crop_w.value, 
                    crop_factor=crop_factor_w.value
                )
            else:
                print("No TFRecord file selected.")
    
    visualize_button_vineyard.on_click(on_visualize_clicked_v)
    tfrecord_folder_w.observe(lambda change: update_tfrecord_options_v(), names='value')
    update_tfrecord_options_v()
    
    # --- Assemble the Dashboard ---
    dashboard_components = [
        header,
        auth_section,
        load_session_button,
        session_output,
        backup_section,
        vineyard_info_base_folder_w,
        vineyard_info_folder_w,
        base_data_folder_w,
        base_manipulated_folder_w,
        tfrecord_folder_w,
        download_w,
        max_vineyards_w,
        time_window_w,
        apply_normalization_w,
        apply_mask_w,
        crop_w,
        crop_factor_w,
        sensors_w,
        sensor_params_container,
        binarize_label_w,
        label_threshold_w,
        build_button,
        output_vineyard,
        widgets.HTML("<h3 style='text-align: center;'>Visualize TFRecord</h3>"),
        tfrecord_dropdown_vineyard,
        visualize_button_vineyard,
        output_visualize_vineyard
    ]
    
    vineyard_dashboard = widgets.VBox(dashboard_components, layout=widgets.Layout(margin='20px auto', width='90%'))
    vineyard_dashboard.add_class("vineyard-dashboard")
    return vineyard_dashboard
