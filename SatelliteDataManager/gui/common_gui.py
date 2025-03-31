#!/usr/bin/env python3
"""
common_gui.py

This module defines common functions and configurations for the interactive dashboards.
It includes:
  - CSS injection for dashboard styling.
  - A SENSOR_OPTIONS dictionary that specifies available options for sensor configuration,
    including both evalscript parameters and download parameters.
  - Functions to create sensor-specific widgets and assemble a container for sensor parameters.
  - Functions to save and load session parameters with customizable backup file paths.
  - Common GUI components: authentication section, session load section, backup session section,
    and a refresh button to clear outputs and update directory listings.
  - A singleton class SHConfigManager to manage the Sentinel Hub configuration.

These utilities facilitate the creation of consistent, customizable GUI components in the dashboards.
"""

import os
import json
import ipywidgets as widgets
from IPython.display import display, HTML
from sentinelhub import SHConfig  # For Sentinel Hub configuration

class SHConfigManager:
    """
    Singleton class to manage the Sentinel Hub configuration.
    """
    _instance = None

    def __init__(self):
        self.config = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = SHConfigManager()
        return cls._instance

    def set_config(self, config):
        self.config = config

    def get_config(self):
        return self.config

# Define the default file path where the last session parameters are stored.
SESSION_FILE = "./last_session/last_session.json"


def inject_css(): 
    """ 
    Injects custom CSS styles for the interactive dashboards.

    Three dashboard styles are defined:
    - .burned-area-dashboard: Uses a fire-inspired gradient (bright red to orange).
    - .vineyard-dashboard: Uses an emerald green gradient.
    - .sensor-dashboard: Uses a blue gradient.
    Additionally, the .widget-label class ensures a consistent minimum width for labels.
    A .dashboard-header class is also provided to allow positioning a refresh button at the top right.

    Returns:
    None.
    """
    display(HTML("""
    <style>
        .burned-area-dashboard {
            background: linear-gradient(45deg, #FF4500, #FF8C00) !important;
            padding: 20px;
            border-radius: 10px;
        }
        .vineyard-dashboard {
            background: linear-gradient(45deg, #50C878, #66FF66) !important;
            padding: 20px;
            border-radius: 10px;
        }
        .evapotranspiration-dashboard {
            background: linear-gradient(45deg, #007BFF, #00BFFF) !important;
            padding: 20px;
            border-radius: 10px;
        }
        .widget-label {
            min-width: 150px !important;
        }
        .dashboard-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
    </style>
    """))

# SENSOR_OPTIONS defines available options for sensor-specific configuration.
SENSOR_OPTIONS = {
    "Sentinel-2": {
        "bands": {
            "description": "Select Sentinel-2 bands",
            "options": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "AOT", "SCL", "SNW", "CLD", "dataMask"],
            "default": []
        },
        "evalscript": {
            "units": {
                "description": "Select units for Sentinel-2",
                "options": ["", "REFLECTANCE", "DN"],
                "default": ""
            },
            "sampleType": {
                "description": "Select sample type for Sentinel-2",
                "options": ["", "AUTO", "INT8", "UINT8", "INT16", "UINT16", "FLOAT32"],
                "default": ""
            }
        },
        "download": {
            "interval_days": {
                "description": "Interval Days for Sentinel-2",
                "default": 12
            },
            "size": {
                "description": "Image Size for Sentinel-2",
                "default": {"width": 256, "height": 256}
            },
            "mosaicking_order": {
                "description": "Mosaicking Order for Sentinel-2",
                "options": ["mostRecent", "leastRecent", "leastCC"],
                "default": "leastCC"
            },
            "resolutions": {
                "description": "Resolution for Sentinel-2",
                "default": None
            }
        }
    },
    "Sentinel-1": {
        "bands": {
            "description": "Select polarizations for Sentinel-1",
            "options": ["VV", "VH", "HH", "HV"],
            "default": []
        },
        "evalscript": {
            "backCoeff": {
                "description": "Select backscatter coefficient for Sentinel-1",
                "options": ["", "SIGMA0_ELLIPSOID", "GAMMA0_ELLIPSOID", "GAMMA0_TERRAIN", "BETA0"],
                "default": ""
            },
            "orthorectify": {
                "description": "Orthorectify data?",
                "options": ["", "False", "True"],
                "default": ""
            },
            "demInstance": {
                "description": "Select DEM instance for Sentinel-1",
                "options": ["", "COPERNICUS", "COPERNICUS_30", "COPERNICUS_90", "MAPZEN"],
                "default": ""
            },
            "sampleType": {
                "description": "Select sample type for Sentinel-1",
                "options": ["", "AUTO", "INT8", "UINT8", "INT16", "UINT16", "FLOAT32"],
                "default": ""
            }
        },
        "download": {
            "interval_days": {
                "description": "Interval Days for Sentinel-1",
                "default": 12
            },
            "size": {
                "description": "Image Size for Sentinel-1",
                "default": {"width": 256, "height": 256}
            },
            "mosaicking_order": {
                "description": "Mosaicking Order for Sentinel-1",
                "options": ["mostRecent", "leastRecent"],
                "default": "mostRecent"
            },
            "resolutions": {
                "description": "Resolution for Sentinel-1",
                "default": None
            }
        }
    },
    "Sentinel-3-OLCI": {
        "bands": {
            "description": "Select Sentinel-3 OLCI bands",
            "options": [f"B{str(i).zfill(2)}" for i in range(1, 22)],
            "default": []
        },
        "evalscript": {
            "sampleType": {
                "description": "Select sample type for Sentinel-3-OLCI",
                "options": ["", "AUTO", "INT8", "UINT8", "INT16", "UINT16", "FLOAT32"],
                "default": ""
            }
        },
        "download": {
            "interval_days": {
                "description": "Interval Days for Sentinel-3-OLCI",
                "default": 3
            },
            "size": {
                "description": "Image Size for Sentinel-3-OLCI",
                "default": {"width": 256, "height": 256}
            },
            "mosaicking_order": {
                "description": "Mosaicking Order for Sentinel-3-OLCI",
                "options": ["mostRecent", "leastRecent"],
                "default": "mostRecent"
            },
            "resolutions": {
                "description": "Resolution for Sentinel-3-OLCI",
                "default": None
            }
        }
    },
    "Sentinel-3-SLSTR-Thermal": {
        "bands": {
            "description": "Select Sentinel-3-SLSTR-Thermal bands",
            "options": ["S7", "S8", "S9", "F1", "F2"],
            "default": []
        },
        "evalscript": {
            "sampleType": {
                "description": "Select sample type for Sentinel-3-SLSTR-Thermal",
                "options": ["", "AUTO", "INT8", "UINT8", "INT16", "UINT16", "FLOAT32"],
                "default": ""
            }
        },
        "download": {
            "interval_days": {
                "description": "Interval Days for Sentinel-3-SLSTR-Thermal",
                "default": 3
            },
            "size": {
                "description": "Image Size for Sentinel-3-SLSTR-Thermal",
                "default": {"width": 256, "height": 256}
            },
            "mosaicking_order": {
                "description": "Mosaicking Order for Sentinel-3-SLSTR-Thermal",
                "options": ["mostRecent", "leastRecent"],
                "default": "mostRecent"
            },
            "resolutions": {
                "description": "Resolution for Sentinel-3-SLSTR-Thermal",
                "default": None
            }
        }
    },
    "Sentinel-3-SLSTR-Optical": {
        "bands": {
            "description": "Select Sentinel-3-SLSTR-Optical bands",
            "options": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "default": []
        },
        "evalscript": {
            "sampleType": {
                "description": "Select sample type for Sentinel-3-SLSTR-Optical",
                "options": ["", "AUTO", "INT8", "UINT8", "INT16", "UINT16", "FLOAT32"],
                "default": ""
            }
        },
        "download": {
            "interval_days": {
                "description": "Interval Days for Sentinel-3-SLSTR-Optical",
                "default": 12
            },
            "size": {
                "description": "Image Size for Sentinel-3-SLSTR-Optical",
                "default": {"width": 256, "height": 256}
            },
            "mosaicking_order": {
                "description": "Mosaicking Order for Sentinel-3-SLSTR-Optical",
                "options": ["mostRecent", "leastRecent"],
                "default": "mostRecent"
            },
            "resolutions": {
                "description": "Resolution for Sentinel-3-SLSTR-Optical",
                "default": None
            }
        }
    },
    "DEM": {
        "bands": None,
        "evalscript": {
            "sampleType": {
                "description": "Select sample type for DEM",
                "options": ["", "AUTO", "INT8", "UINT8", "INT16", "UINT16", "FLOAT32"],
                "default": ""
            },
            "demInstance": {
                "description": "Select DEM instance for DEM",
                "options": ["", "COPERNICUS", "COPERNICUS_30", "COPERNICUS_90", "MAPZEN"],
                "default": ""
            }
        },
        "download": {
            "interval_days": {
                "description": "Interval Days for DEM",
                "default": 36
            },
            "size": {
                "description": "Image Size for DEM",
                "default": {"width": 256, "height": 256}
            },
            "mosaicking_order": {
                "description": "Mosaicking Order for DEM",
                "options": ["mostRecent", "leastRecent"],
                "default": "mostRecent"
            },
            "resolutions": {
                "description": "Resolution for DEM",
                "default": None
            }
        }
    }
}

def create_sensor_widgets(sensor_name):
    """
    Creates and returns a widget container (VBox) with sensor-specific controls for parameter selection.

    For the given sensor, this function creates:
      - A multi-select widget for band selection (if applicable).
      - Dropdown menus for each evalscript parameter.
      - Numeric input widgets for download parameters:
          * For "interval_days" and "resolutions": an IntText widget.
          * For "size": two IntText widgets (one for width and one for height).
          * For other parameters (like mosaicking_order): a Dropdown widget.
      
    Parameters:
      sensor_name (str): Name of the sensor (e.g., "Sentinel-2").

    Returns:
      widgets.VBox: A vertical container with the sensor-specific widgets.
    """
    sensor_config = SENSOR_OPTIONS.get(sensor_name)
    sensor_widgets = []
    
    # Band selection widget.
    if sensor_config and sensor_config.get("bands"):
        band_conf = sensor_config["bands"]
        band_widget = widgets.SelectMultiple(
            options=band_conf["options"],
            value=tuple(band_conf["default"]),
            description=band_conf["description"],
            layout=widgets.Layout(width='90%')
        )
        sensor_widgets.append(band_widget)
    
    # Evalscript parameter widgets.
    if sensor_config and sensor_config.get("evalscript"):
        for param, conf in sensor_config["evalscript"].items():
            dropdown = widgets.Dropdown(
                options=conf["options"],
                value=conf["default"],
                description=conf["description"],
                layout=widgets.Layout(width='90%')
            )
            sensor_widgets.append(dropdown)
    
    # Download parameter widgets.
    if sensor_config and sensor_config.get("download"):
        for param, conf in sensor_config["download"].items():
            if param in ["interval_days", "resolutions"]:
                int_widget = widgets.IntText(
                    value=conf["default"] if conf["default"] is not None else 0,
                    description=conf["description"],
                    layout=widgets.Layout(width='90%')
                )
                sensor_widgets.append(int_widget)
            elif param == "size":
                default_width = conf["default"].get("width", 256)
                default_height = conf["default"].get("height", 256)
                width_widget = widgets.IntText(
                    value=default_width,
                    description="Width:",
                    layout=widgets.Layout(width='45%')
                )
                height_widget = widgets.IntText(
                    value=default_height,
                    description="Height:",
                    layout=widgets.Layout(width='45%')
                )
                size_container = widgets.HBox([width_widget, height_widget])
                size_box = widgets.VBox([widgets.HTML("<b>" + conf["description"] + ":</b>"), size_container])
                sensor_widgets.append(size_box)
            else:
                dropdown = widgets.Dropdown(
                    options=conf["options"],
                    value=conf["default"],
                    description=conf["description"],
                    layout=widgets.Layout(width='90%')
                )
                sensor_widgets.append(dropdown)
    
    sensor_box = widgets.VBox(sensor_widgets, layout=widgets.Layout(border='solid 1px gray', padding='10px', width='95%'))
    sensor_box_label = widgets.HTML(f"<b>{sensor_name} Options:</b>")
    return widgets.VBox([sensor_box_label, sensor_box])

def build_sensor_parameters_container(selected_sensors):
    """
    Constructs a container that aggregates sensor-specific parameter widgets for all selected sensors.

    Parameters:
      selected_sensors (list): List of sensor names selected by the user.

    Returns:
      widgets.VBox: A vertical container (VBox) containing the sensor-specific widgets.
    """
    sensor_widgets_list = []
    for sensor in selected_sensors:
        sensor_widgets_list.append(create_sensor_widgets(sensor))
    return widgets.VBox(sensor_widgets_list, layout=widgets.Layout(width='100%'))

def save_last_session(session_data, file_path=None):
    """
    Saves the provided session parameters to a JSON file for later retrieval.

    The session parameters are stored in the file specified by file_path.
    If no file_path is provided, the default SESSION_FILE is used.

    Parameters:
      session_data (dict): Dictionary containing session parameters.
      file_path (str): Optional file path for saving the session.

    Returns:
      None.
    """
    if file_path is None:
        file_path = SESSION_FILE
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(session_data, f)
    except Exception as e:
        print("Error saving session:", e)

def load_last_session(file_path=None):
    """
    Loads session parameters from a JSON file if it exists.

    Parameters:
      file_path (str): Optional file path to load the session from.
                       If not provided, the default SESSION_FILE is used.

    Returns:
      dict: A dictionary of session parameters if available, otherwise an empty dictionary.
    """
    if file_path is None:
        file_path = SESSION_FILE
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            return data
        except Exception as e:
            print("Error loading session:", e)
    return {}

def create_refresh_button(refresh_callbacks=None, output_widgets=None):
    """
    Creates a refresh button that clears specified output widgets and calls provided refresh callbacks.

    When clicked, the button clears the content of the output widgets (e.g., terminal and plot outputs)
    and invokes the refresh callbacks (e.g., to update directory listings).

    Parameters:
      refresh_callbacks (list): A list of callback functions to be invoked on refresh.
      output_widgets (list): A list of ipywidgets.Output widgets to be cleared.

    Returns:
      widgets.Button: The refresh button widget.
    """
    refresh_button = widgets.Button(description="Refresh", button_style="info", layout=widgets.Layout(width='auto'))
    
    def on_refresh_clicked(b):
        if output_widgets:
            for out in output_widgets:
                out.clear_output()
        if refresh_callbacks:
            for cb in refresh_callbacks:
                cb()
        print("Refresh completed.")
    
    refresh_button.on_click(on_refresh_clicked)
    return refresh_button

def create_auth_section():
    """
    Creates and returns the Sentinel Hub authentication section common to dashboards.

    This section includes:
      - New Profile checkbox.
      - Text fields for Profile Name, Client ID, and Client Secret.
      - An Authenticate button and output area.
      - Internal logic to handle authentication and update the SHConfigManager singleton.

    Returns:
      dict: A dictionary containing the authentication section widget and its key components.
    """
    new_profile_checkbox = widgets.Checkbox(
        value=False,
        description="New Profile",
        layout=widgets.Layout(width='90%')
    )
    config_profile_w = widgets.Text(
        value="",
        description="Profile Name:",
        layout=widgets.Layout(width='90%')
    )
    client_id_w = widgets.Text(
        value="",
        description="Client ID:",
        layout=widgets.Layout(width='90%')
    )
    client_secret_w = widgets.Text(
        value="",
        description="Client Secret:",
        layout=widgets.Layout(width='90%')
    )
    client_id_w.layout.display = "none"
    client_secret_w.layout.display = "none"
    
    def toggle_profile_fields(change):
        if change['new']:
            client_id_w.layout.display = "block"
            client_secret_w.layout.display = "block"
        else:
            client_id_w.layout.display = "none"
            client_secret_w.layout.display = "none"
    
    new_profile_checkbox.observe(toggle_profile_fields, names='value')
    
    auth_button = widgets.Button(
        description="Authenticate", 
        button_style='primary', 
        layout=widgets.Layout(width='90%')
    )
    auth_output = widgets.Output(layout=widgets.Layout(width='90%'))
    
    def on_authenticate_clicked(b):
        profile_name = config_profile_w.value.strip()
        if not profile_name:
            with auth_output:
                auth_output.clear_output()
                print("Please enter a profile name.")
            return
        if new_profile_checkbox.value:
            client_id = client_id_w.value.strip()
            client_secret = client_secret_w.value.strip()
            if not client_id or not client_secret:
                with auth_output:
                    auth_output.clear_output()
                    print("Please fill in both Client ID and Client Secret for a new profile.")
                return
            config = SHConfig()
            config.sh_client_id = client_id
            config.sh_client_secret = client_secret
            config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
            config.sh_base_url = "https://sh.dataspace.copernicus.eu"
            config.save(profile_name)
        else:
            config = SHConfig(profile_name)
        SHConfigManager.instance().set_config(config)
        auth_output.clear_output()
        with auth_output:
            print("Authentication successful. Profile:", profile_name)
    
    auth_button.on_click(on_authenticate_clicked)
    
    auth_section = widgets.VBox([
        widgets.HTML("<h3>Sentinel Hub Authentication</h3>"),
        new_profile_checkbox,
        config_profile_w,
        client_id_w,
        client_secret_w,
        auth_button,
        auth_output
    ], layout=widgets.Layout(width='90%'))
    return {
        "auth_section": auth_section,
        "config_profile_w": config_profile_w,
        "client_id_w": client_id_w,
        "client_secret_w": client_secret_w,
        "auth_output": auth_output,
    }

def create_load_session_section(update_widgets_from_session):
    """
    Creates a section with a 'Load Last Session' button to load session parameters.

    When the button is clicked, it loads the session data from the default backup file and
    calls the provided update_widgets_from_session function to update the dashboard widgets.

    Parameters:
      update_widgets_from_session (function): A function that takes session_data dict as input and updates widgets.

    Returns:
      tuple: (load_session_button, session_output) widgets.
    """
    load_session_button = widgets.Button(
        description="Load Last Session", 
        button_style='warning', 
        layout=widgets.Layout(width='90%')
    )
    session_output = widgets.Output(layout=widgets.Layout(width='90%'))
    
    def on_load_session_clicked(b):
        session_data = load_last_session()
        if not session_data:
            with session_output:
                session_output.clear_output()
                print("No session data found.")
            return
        update_widgets_from_session(session_data)
        with session_output:
            session_output.clear_output()
            print("Session loaded.")
    
    load_session_button.on_click(on_load_session_clicked)
    return load_session_button, session_output

def create_backup_section(collect_session_data, update_widgets_from_session):
    """
    Creates a backup session configuration section with save and load functionality.

    This section includes:
      - A text input for specifying the backup file path.
      - Buttons to save the current session to backup and load a session from backup.
      - An output area for backup session messages.

    Parameters:
      collect_session_data (function): A function that returns the current session data as a dict.
      update_widgets_from_session (function): A function that updates dashboard widgets from session data.

    Returns:
      widgets.VBox: The backup session configuration section.
    """
    backup_file_w = widgets.Text(
        value="",
        description="Backup File Path:",
        layout=widgets.Layout(width='90%')
    )
    save_backup_button = widgets.Button(
        description="Save Session to Backup", 
        button_style='info', 
        layout=widgets.Layout(width='45%')
    )
    load_backup_button = widgets.Button(
        description="Load Session from Backup", 
        button_style='info', 
        layout=widgets.Layout(width='45%')
    )
    backup_output = widgets.Output(layout=widgets.Layout(width='90%'))
    
    def on_save_backup_clicked(b):
        session_data = collect_session_data()
        backup_path = backup_file_w.value.strip()
        if not backup_path:
            with backup_output:
                backup_output.clear_output()
                print("Please enter a backup file path.")
            return
        save_last_session(session_data, file_path=backup_path)
        with backup_output:
            backup_output.clear_output()
            print("Session saved to backup file:", backup_path)
    
    def on_load_backup_clicked(b):
        backup_path = backup_file_w.value.strip()
        if not backup_path:
            with backup_output:
                backup_output.clear_output()
                print("Please enter a backup file path.")
            return
        session_data = load_last_session(file_path=backup_path)
        if not session_data:
            with backup_output:
                backup_output.clear_output()
                print("No session data found in backup file.")
            return
        update_widgets_from_session(session_data)
        with backup_output:
            backup_output.clear_output()
            print("Session loaded from backup file:", backup_path)
    
    save_backup_button.on_click(on_save_backup_clicked)
    load_backup_button.on_click(on_load_backup_clicked)
    
    backup_section = widgets.VBox([
        widgets.HTML("<h3>Backup Session Configuration</h3>"),
        backup_file_w,
        widgets.HBox([save_backup_button, load_backup_button]),
        backup_output
    ], layout=widgets.Layout(width='90%'))
    
    return backup_section


def extract_sensor_params(sensor_params_container):
    """
    Extracts sensor parameter values from the sensor_params_container widget structure.

    Assumes that sensor_params_container.children[0] is a VBox containing one widget per sensor.
    Each sensor widget is expected to have:
      - A header (HTML) with the sensor name.
      - A container (VBox) containing control widgets for the parameters.
        * If a widget is a VBox (used for the "size" parameter), the two IntText values (width and height)
          are extracted.
        * Otherwise, the widget's value is taken (or a list of values if the widget is a SelectMultiple).

    Returns:
      dict: A dictionary with sensor names as keys and the extracted parameters as values.
    """
    sensor_params = {}
    if sensor_params_container.children:
        # Assume that sensor_params_container.children[0] contains all sensor widgets.
        for sensor_container in sensor_params_container.children[0].children:
            # Extract the sensor name by removing HTML tags.
            sensor_label = sensor_container.children[0].value.replace("<b>", "").replace("</b>", "")
            controls_vbox = sensor_container.children[1]
            sensor_params[sensor_label] = {}
            for control in controls_vbox.children:
                if isinstance(control, widgets.VBox):
                    header_text = control.children[0].value.replace("<b>", "").replace(":</b>", "").strip()
                    size_hbox = control.children[1]
                    width_val = size_hbox.children[0].value
                    height_val = size_hbox.children[1].value
                    sensor_params[sensor_label][header_text] = {"width": width_val, "height": height_val}
                else:
                    sensor_params[sensor_label][control.description] = (
                        list(control.value) if isinstance(control.value, tuple) else control.value
                    )
    return sensor_params

def load_sensor_params(sensor_params_container, saved_sensor_params):
    """
    Updates the sensor parameter widgets in sensor_params_container with saved values.
    
    Expects saved_sensor_params to be a dictionary with sensor names as keys and a dictionary
    of parameter values as values. For each sensor widget, the function looks up the sensor name,
    then for each control widget:
      - If the control is a VBox (used for "size"), it updates the width and height IntText widgets.
      - Otherwise, it updates the widget's value using its description as the key.
    
    Parameters:
      sensor_params_container (widgets.VBox): The container holding sensor parameter widgets.
      saved_sensor_params (dict): The dictionary of saved sensor parameters.
    """
    if sensor_params_container.children:
        # sensor_params_container.children[0] holds all sensor widgets.
        for sensor_container in sensor_params_container.children[0].children:
            # Extract sensor name (removing HTML tags)
            sensor_label = sensor_container.children[0].value.replace("<b>", "").replace("</b>", "")
            if sensor_label in saved_sensor_params:
                saved_params = saved_sensor_params[sensor_label]
                controls_vbox = sensor_container.children[1]
                for control in controls_vbox.children:
                    if isinstance(control, widgets.VBox):
                        header_text = control.children[0].value.replace("<b>", "").replace(":</b>", "").strip()
                        if header_text in saved_params and isinstance(saved_params[header_text], dict):
                            size_hbox = control.children[1]
                            width_val = saved_params[header_text].get("width")
                            height_val = saved_params[header_text].get("height")
                            if width_val is not None:
                                size_hbox.children[0].value = width_val
                            if height_val is not None:
                                size_hbox.children[1].value = height_val
                    else:
                        desc = control.description
                        if desc in saved_params:
                            control.value = saved_params[desc]



