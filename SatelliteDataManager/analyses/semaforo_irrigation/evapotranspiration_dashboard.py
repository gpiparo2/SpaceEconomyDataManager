#!/usr/bin/env python3 
""" 
evapotranspiration_analysis_dashboard.py
------------------------------------------
This module builds an interactive dashboard for iterative evapotranspiration analysis.
The dashboard allows the user to:
    - Load sensor data from a specified folder.
    - Select an overall date range and a subperiod length (in days) for analysis.
    - Plot individual sensor series (filtered by the overall date range).
    - Compare selected sensor series (up to 6) on a single plot with multiple y-axes.
    - Generate scatter plots for selected variables versus another variable with Pearson correlation.
    - Perform iterative ET₀ analysis over subperiods, save results to an Excel file,
      and plot the total evapotranspiration for each subperiod.
"""

import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from IPython.display import display, clear_output 
import ipywidgets as widgets 
from ...gui.common_gui import inject_css
from .SensorDataManager import SensorDataManager 
from .evapotranspiration_analysis_functions import (
    compute_et0, compute_water_volume, convert_hectares_to_square_meters, 
    analyze_evapotranspiration_iterative
)

def build_evapotranspiration_dashboard(): 
    """ 
    Builds and displays the Evapotranspiration Analysis Dashboard.
    The dashboard provides:
      - Input fields to specify the sensor data folder, overall date range, and subperiod length.
      - Buttons to plot individual sensor series (filtered by overall date range), comparative plots, and scatter plots.
      - A section to run iterative ET₀ analysis over subperiods, save results to an Excel file,
        and plot the total evapotranspiration for each subperiod.
    
    Returns:
      widgets.VBox: The complete evapotranspiration dashboard.
    """
    inject_css()

    # Input widget for sensor data folder
    sensor_folder_w = widgets.Text(
        value="sensor_data",
        description="Sensor Data Folder:",
        layout=widgets.Layout(width='60%')
    )
    # Overall date range pickers
    overall_start_w = widgets.DatePicker(
        description="Overall Start Date:",
        disabled=False
    )
    overall_end_w = widgets.DatePicker(
        description="Overall End Date:",
        disabled=False
    )
    # Subperiod length (in days)
    days_w = widgets.FloatText(
        value=7.0,
        description="Days per period:",
        layout=widgets.Layout(width='40%')
    )
    
    # Buttons arranged as in the previous version
    plot_individual_btn = widgets.Button(
        description="Plot Individual Series",
        button_style="info"
    )
    compare_series_w = widgets.SelectMultiple(
        options=['Temperature', 'SolarRadiation', 'WindSpeed', 'AirHumidity', 'N', 'P', 'K', 'Conductivity', 'Precipitation'],
        description="Compare Series:",
        layout=widgets.Layout(width='60%')
    )
    plot_comparative_btn = widgets.Button(
        description="Plot Comparative Series",
        button_style="info"
    )
    scatter_x_w = widgets.Dropdown(
        options=['Temperature', 'SolarRadiation', 'WindSpeed', 'AirHumidity', 'N', 'P', 'K', 'Conductivity', 'Precipitation'],
        description="Scatter X:"
    )
    scatter_y_w = widgets.SelectMultiple(
        options=['Temperature', 'SolarRadiation', 'WindSpeed', 'AirHumidity', 'N', 'P', 'K', 'Conductivity', 'Precipitation'],
        description="Scatter Y:",
        layout=widgets.Layout(width='60%')
    )
    plot_scatter_btn = widgets.Button(
        description="Plot Scatter(s)",
        button_style="info"
    )

    # Inputs for ET₀ analysis
    crop_coef_w = widgets.FloatText(
        value=0.4,
        description="Crop Coefficient:",
        layout=widgets.Layout(width='40%')
    )
    area_ha_w = widgets.FloatText(
        value=1.0,
        description="Area (ha):",
        layout=widgets.Layout(width='40%')
    )

    # Button to run iterative ET₀ analysis
    run_iterative_btn = widgets.Button(
        description="Run Iterative ET₀ Analysis",
        button_style="success"
    )

    # Output area
    output_area = widgets.Output()

    # Helper function to filter a series by overall start and end dates
    def filter_series_by_date(series, start, end):
        if pd.api.types.is_datetime64_any_dtype(series.index):
            return series[(series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))]
        return series

    # Callback functions
    def on_plot_individual_clicked(b):
        with output_area:
            clear_output()
            folder = sensor_folder_w.value
            sdm = SensorDataManager(folder)
            data_dict = {}
            data_dict.update(sdm.load_meteorological_data())
            data_dict.update(sdm.load_fertilizer_data())
            # Apply overall date filtering if dates are provided
            overall_start = overall_start_w.value
            overall_end = overall_end_w.value
            if overall_start and overall_end:
                for key in data_dict:
                    data_dict[key] = filter_series_by_date(data_dict[key], overall_start, overall_end)
            print("Individual Time Series Plots:")
            sdm.plot_individual_series(data_dict)

    def on_plot_comparative_clicked(b):
        with output_area:
            clear_output()
            folder = sensor_folder_w.value
            sdm = SensorDataManager(folder)
            data_dict = {}
            data_dict.update(sdm.load_meteorological_data())
            data_dict.update(sdm.load_fertilizer_data())
            # Apply overall date filtering
            overall_start = overall_start_w.value
            overall_end = overall_end_w.value
            if overall_start and overall_end:
                for key in data_dict:
                    data_dict[key] = filter_series_by_date(data_dict[key], overall_start, overall_end)
            selected = list(compare_series_w.value)
            if not selected:
                print("Please select at least one series for comparison.")
                return
            print("Comparative Plot:")
            sdm.plot_comparative_series(data_dict, selected)

    def on_plot_scatter_clicked(b):
        with output_area:
            clear_output()
            folder = sensor_folder_w.value
            sdm = SensorDataManager(folder)
            data_dict = {}
            data_dict.update(sdm.load_meteorological_data())
            data_dict.update(sdm.load_fertilizer_data())
            # Apply overall date filtering
            overall_start = overall_start_w.value
            overall_end = overall_end_w.value
            if overall_start and overall_end:
                for key in data_dict:
                    data_dict[key] = filter_series_by_date(data_dict[key], overall_start, overall_end)
            x_var = scatter_x_w.value
            y_vars = list(scatter_y_w.value)
            if not x_var or not y_vars:
                print("Please select variables for scatter plot.")
                return
            print("Scatter Plot(s):")
            x_series = data_dict.get(x_var)
            if x_series is None or x_series.empty:
                print(f"No data available for {x_var}")
                return
            for y_var in y_vars:
                y_series = data_dict.get(y_var)
                if y_series is None or y_series.empty:
                    print(f"No data available for {y_var}")
                    continue
                sdm.plot_scatter_with_correlation(x_series, y_series, x_label=x_var, y_label=y_var)

    def on_run_iterative_clicked(b):
        with output_area:
            clear_output()
            folder = sensor_folder_w.value
            temp_path = os.path.join(folder, "../Climate/air_temperature.xlsx")
            rad_path = os.path.join(folder, "../Climate/radiation.xlsx")
            wind_path = os.path.join(folder, "../Climate/wind.xlsx")
            humidity_path = os.path.join(folder, "../Climate/air_humidity.xlsx")
            precip_path = os.path.join(folder, "../Climate/precipitation.xlsx")
            soil_path = os.path.join(folder, "./Humidity.xlsx")
            overall_start = overall_start_w.value
            overall_end = overall_end_w.value
            if overall_start is None or overall_end is None:
                print("Please select valid overall start and end dates.")
                return
            try:
                period_days = float(days_w.value)
            except Exception as e:
                print("Invalid value for days per period.")
                return
            results_df = analyze_evapotranspiration_iterative(
                temp_path, rad_path, wind_path, humidity_path,
                precip_path, soil_path, crop_coef_w.value, area_ha_w.value,
                period_days, overall_start, overall_end
            )
            print("\nIterative Evapotranspiration Results:")
            display(results_df)

    # Bind callbacks
    plot_individual_btn.on_click(on_plot_individual_clicked)
    plot_comparative_btn.on_click(on_plot_comparative_clicked)
    plot_scatter_btn.on_click(on_plot_scatter_clicked)
    run_iterative_btn.on_click(on_run_iterative_clicked)

    # Assemble the dashboard layout (buttons arranged as before)
    dashboard = widgets.VBox([
        sensor_folder_w,
        widgets.HBox([overall_start_w, overall_end_w]),
        plot_individual_btn,
        compare_series_w,
        plot_comparative_btn,
        widgets.HBox([scatter_x_w, scatter_y_w]),
        plot_scatter_btn,
        widgets.HBox([crop_coef_w, area_ha_w, days_w]),
        run_iterative_btn,
        output_area
    ])

    dashboard.add_class("evapotranspiration-dashboard")
    return dashboard

