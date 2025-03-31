#!/usr/bin/env python3
"""
SensorDataManager.py
---------------------
This module defines the SensorDataManager class which is responsible for managing and processing ground sensor data.
It provides functions to load data from Excel files, synchronize the series, compute meteorological parameters (atmospheric pressure, delta, gamma, net radiation),
calculate reference evapotranspiration (ET₀) using the Penman-Monteith equation, and compute irrigation water volume requirements.
Additionally, visualization functions are provided for:
  - Plotting individual time series.
  - Creating comparative plots (with multiple y-axes) for selected series.
  - Plotting scatter plots with the Pearson correlation coefficient.
"""

import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import matplotlib.dates as mdates

# Dictionary mapping sensor names to units
UNIT_MAP = {
    'Temperature': '°C',
    'SolarRadiation': 'MJ/m²/day',
    'WindSpeed': 'm/s',
    'AirHumidity': '%',
    'N': 'mg/kg',
    'P': 'mg/kg',
    'K': 'mg/kg',
    'Conductivity': 'mS/cm',
    'Precipitation': 'mm',
    'SoilMoisture': '%'
}

class SensorDataManager:
    """
    A class to manage ground sensor data and perform related computations.
    
    Attributes:
        base_folder (str): The directory where sensor data files are stored.
    """

    def __init__(self, base_folder):
        """
        Initializes the SensorDataManager with a base folder for sensor data files.
        
        Parameters:
            base_folder (str): Path to the folder containing sensor data files.
        """
        self.base_folder = base_folder

    def load_excel_data(self, filename, usecols=None, skiprows=0):
        """
        Loads data from an Excel file.
        
        Parameters:
            filename (str): The name of the Excel file.
            usecols (list or int, optional): Columns to load.
            skiprows (int, optional): Number of rows to skip at the beginning.
            
        Returns:
            pandas.DataFrame: The loaded DataFrame.
        """
        file_path = os.path.join(self.base_folder, filename)
        return pd.read_excel(file_path, usecols=usecols, skiprows=skiprows)

    def load_fertilizer_data(self):
        """
        Loads and processes fertilizer-related sensor data from Excel files.
        
        Returns:
            dict: A dictionary with keys 'N', 'P', 'K', 'Conductivity', 'Precipitation'
                and values as pandas Series with datetime indices.
        """
        files = {
            'N': 'N.xlsx',
            'P': 'P.xlsx',
            'K': 'K.xlsx',
            'Conductivity': 'Conductivity.xlsx',
            'Precipitation': '../Climate/precipitation.xlsx'
        }
        
        data = {}
        for key, fname in files.items():
            df = self.load_excel_data(fname, usecols=[0, 1])
            df.columns = ['Date', key]
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date']).set_index('Date')
            data[key] = df[key].dropna()
        return data

    def load_meteorological_data(self):
        """
        Loads and processes meteorological sensor data from Excel files.
        Each file is expected to have two columns: the first is the measurement date and the second is the value.
        
        Returns:
            dict: A dictionary with keys 'Temperature', 'SolarRadiation', 'WindSpeed', 'AirHumidity'
                  and values as pandas Series with datetime indices.
        """
        # Temperature data
        t_df = self.load_excel_data('../Climate/air_temperature.xlsx')
        t_df.columns = ['Date', 'Temperature']
        t_df['Date'] = pd.to_datetime(t_df['Date'], dayfirst=True, errors='coerce')
        t_df = t_df.dropna(subset=['Date']).set_index('Date')
        temperature = t_df['Temperature'].dropna()

        # Solar Radiation data
        rad_df = self.load_excel_data('../Climate/radiation.xlsx')
        rad_df.columns = ['Date', 'SolarRadiation']
        rad_df['Date'] = pd.to_datetime(rad_df['Date'], dayfirst=True, errors='coerce')
        rad_df = rad_df.dropna(subset=['Date']).set_index('Date')
        solar_radiation = rad_df['SolarRadiation'].dropna()

        # Wind Speed data
        vento_df = self.load_excel_data('../Climate/wind.xlsx')
        vento_df.columns = ['Date', 'WindSpeed']
        vento_df['Date'] = pd.to_datetime(vento_df['Date'], dayfirst=True, errors='coerce')
        vento_df = vento_df.dropna(subset=['Date']).set_index('Date')
        wind_speed = vento_df['WindSpeed'].dropna()

        # Air Humidity data
        u_df = self.load_excel_data('../Climate/air_humidity.xlsx')
        u_df.columns = ['Date', 'AirHumidity']
        u_df['Date'] = pd.to_datetime(u_df['Date'], dayfirst=True, errors='coerce')
        u_df = u_df.dropna(subset=['Date']).set_index('Date')
        air_humidity = u_df['AirHumidity'].dropna()

        # Synchronize indices
        common_index = temperature.index.intersection(solar_radiation.index)\
                        .intersection(wind_speed.index).intersection(air_humidity.index)
        temperature = temperature.loc[common_index]
        solar_radiation = solar_radiation.loc[common_index]
        wind_speed = wind_speed.loc[common_index]
        air_humidity = air_humidity.loc[common_index]

        return {
            'Temperature': temperature,
            'SolarRadiation': solar_radiation,
            'WindSpeed': wind_speed,
            'AirHumidity': air_humidity
        }

    def load_soil_moisture_data(self, filename='humidity_consociato_P.xlsx'):
        """
        Loads soil moisture data from an Excel file.
        The file is expected to have two columns: Date and SoilMoisture.
        
        Parameters:
            filename (str): The name of the Excel file containing soil moisture data.
        
        Returns:
            pandas.DataFrame: A DataFrame with a datetime index and a column 'SoilMoisture'.
        """
        df = self.load_excel_data(filename)
        df.columns = ['Date', 'SoilMoisture']
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date']).set_index('Date')
        return df.dropna()

    # ----------------- Visualization Functions -----------------

    def plot_individual_series(self, data_dict):
        """
        Plots individual time series for each sensor parameter, including units in axis labels.
        
        Parameters:
            data_dict (dict): A dictionary where keys are parameter names and values are pandas Series.
        """
        for key, series in data_dict.items():
            if series.empty:
                continue
            plt.figure(figsize=(10, 4))
            plt.plot(series.index, series, label=key)
            # Format x-axis as dates if applicable
            if pd.api.types.is_datetime64_any_dtype(series.index):
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.gcf().autofmt_xdate()
            unit = UNIT_MAP.get(key, "")
            plt.xlabel("Time")
            plt.ylabel(f"{key} ({unit})" if unit else key)
            plt.title(f"{key} Time Series")
            plt.legend()
            plt.show()

    def plot_comparative_series(self, data_dict, selected_series):
        """
        Plots selected time series on a single figure using multiple y-axes.
        Up to 6 series can be compared. Each series is plotted on its own y-axis.
        Units are included in the axis labels and time is formatted as dates.
        
        Parameters:
            data_dict (dict): A dictionary where keys are parameter names and values are pandas Series.
            selected_series (list): List of keys to plot from data_dict.
        """
        if not selected_series:
            print("No series selected for comparison.")
            return

        fig, host = plt.subplots(figsize=(12, 6))
        fig.subplots_adjust(right=0.75)
        
        axes = [host]
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # Create additional axes for each extra series
        for i in range(1, len(selected_series)):
            par = host.twinx()
            par.spines["right"].set_position(("axes", 1 + 0.1 * i))
            axes.append(par)

        for ax, key, color in zip(axes, selected_series, colors):
            series = data_dict.get(key)
            if series is None or series.empty:
                continue
            # Normalize series (z-score) for comparison
            norm_series = (series - series.mean()) / series.std()
            ax.plot(series.index, norm_series, label=key, color=color)
            unit = UNIT_MAP.get(key, "")
            ax.set_ylabel(f"{key} ({unit})" if unit else key, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # Format x-axis as dates if applicable
            if pd.api.types.is_datetime64_any_dtype(series.index):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.gcf().autofmt_xdate()
                        # Format x-axis as dates if applicable

        host.set_xlabel("Time")
        host.set_title("Comparative Plot of Selected Series")
        lines, labels = [], []
        for ax in axes:
            line, label = ax.get_legend_handles_labels()
            lines.extend(line)
            labels.extend(label)
        host.legend(lines, labels, loc='upper left')
        plt.show()

    def plot_scatter_with_correlation(self, x_series, y_series, x_label="X", y_label="Y"):
        """
        Plots a scatter plot of two series and computes the Pearson correlation coefficient.
        Axis labels include units if available, and if the x values are datetime, they are formatted as dates.
        
        Parameters:
            x_series (pandas.Series): The x-axis data.
            y_series (pandas.Series): The y-axis data.
            x_label (str): Label for x-axis.
            y_label (str): Label for y-axis.
        """
        # Align the series on common indices
        common_index = x_series.index.intersection(y_series.index)
        if len(common_index) == 0:
            print("No overlapping data for scatter plot.")
            return
        x_vals = x_series.loc[common_index]
        y_vals = y_series.loc[common_index]
        corr_coef, _ = pearsonr(x_vals, y_vals)
        plt.figure(figsize=(6, 4))
        plt.scatter(x_vals, y_vals)
        # If x values are datetime, format the x-axis accordingly
        if pd.api.types.is_datetime64_any_dtype(x_vals):
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()
        x_unit = UNIT_MAP.get(x_label, "")
        y_unit = UNIT_MAP.get(y_label, "")
        plt.xlabel(f"{x_label} ({x_unit})" if x_unit else x_label)
        plt.ylabel(f"{y_label} ({y_unit})" if y_unit else y_label)
        plt.title(f"Scatter Plot: {x_label} vs {y_label}\nPearson r = {corr_coef:.2f}")
        plt.show()

    def plot_multiple_scatter(self, x_series, y_series_list, x_label="X"):
        """
        Creates scatter plots for one x_series against each series in y_series_list.
        Axis labels include units and if the x values are datetime, they are formatted as dates.
        
        Parameters:
            x_series (pandas.Series): The x-axis data.
            y_series_list (list of tuples): Each tuple contains (series, label) for y-axis.
            x_label (str): Label for x-axis.
        """
        n = len(y_series_list)
        fig, axs = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axs = [axs]
        for ax, (y_series, y_label) in zip(axs, y_series_list):
            common_index = x_series.index.intersection(y_series.index)
            if len(common_index) == 0:
                ax.set_title(f"No overlap for {y_label}")
                continue
            x_vals = x_series.loc[common_index]
            y_vals = y_series.loc[common_index]
            corr_coef, _ = pearsonr(x_vals, y_vals)
            ax.scatter(x_vals, y_vals)
            # Format x-axis as dates if x values are datetime
            if pd.api.types.is_datetime64_any_dtype(x_vals):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.gcf().autofmt_xdate()
            x_unit = UNIT_MAP.get(x_label, "")
            y_unit = UNIT_MAP.get(y_label, "")
            ax.set_xlabel(f"{x_label} ({x_unit})" if x_unit else x_label)
            ax.set_ylabel(f"{y_label} ({y_unit})" if y_unit else y_label)
            ax.set_title(f"{x_label} vs {y_label}\nPearson r = {corr_coef:.2f}")
        plt.tight_layout()
        plt.show()

    # ----------------- End of Visualization Functions -----------------

    def compute_atmospheric_pressure(self, altitude):
        """
        Computes the atmospheric pressure at a given altitude.
        
        Parameters:
            altitude (float): Altitude in meters.
            
        Returns:
            float: Atmospheric pressure in kPa.
        """
        return 101.3 * ((293 - 0.0065 * altitude) / 293) ** 5.26

    def delta(self, temperature):
        """
        Computes the slope of the saturation vapor pressure curve (delta) at a given temperature.
        
        Parameters:
            temperature (float): Temperature in °C.
            
        Returns:
            float: Delta value in kPa/°C.
        """
        return (4098 * (0.6108 * math.exp((17.27 * temperature) / (temperature + 237.3)))) / ((temperature + 237.3) ** 2)

    def gamma(self, pressure):
        """
        Computes the psychrometric constant (gamma) given atmospheric pressure.
        
        Parameters:
            pressure (float): Atmospheric pressure in kPa.
            
        Returns:
            float: Gamma value in kPa/°C.
        """
        return 0.665e-3 * pressure

    def compute_net_radiation(self, rs, rns, rnl):
        """
        Computes the net radiation.
        
        Parameters:
            rs (float): Incoming solar radiation (MJ/m²/day).
            rns (float): Reflected solar radiation (MJ/m²/day).
            rnl (float): Net longwave radiation (MJ/m²/day).
            
        Returns:
            float: Net radiation (MJ/m²/day).
        """
        return rs - rns - rnl

    def compute_et0(self, temperature, wind_speed, air_humidity, solar_radiation):
        """
        Computes the reference evapotranspiration (ET₀) using the Penman-Monteith equation.
        
        Parameters:
            temperature (float): Mean air temperature (°C).
            wind_speed (float): Mean wind speed (m/s).
            air_humidity (float): Mean relative humidity (%).
            solar_radiation (float): Mean solar radiation (MJ/m²/day).
            
        Returns:
            float: Reference evapotranspiration (mm/day).
        """
        p = self.compute_atmospheric_pressure(altitude=0)  # assuming sea level
        delta_val = self.delta(temperature)
        gamma_val = self.gamma(p)
        rns = 0.1 * solar_radiation
        rnl = 0.1 * solar_radiation
        net_rad = self.compute_net_radiation(solar_radiation, rns, rnl)
        et0 = (0.408 * delta_val * net_rad + gamma_val * (900 / (temperature + 273)) * wind_speed * (1 - air_humidity / 100)) \
              / (delta_val + gamma_val * (1 + 0.34 * wind_speed))
        return et0

    def compute_irrigation_volume(self, min_humidity, max_humidity, area, depth):
        """
        Computes the volume of water needed for irrigation based on soil moisture differences.
        
        Parameters:
            min_humidity (float): Minimum soil moisture (%).
            max_humidity (float): Maximum soil moisture (%).
            area (float): Irrigated area (m²).
            depth (float): Sensor depth (m).
            
        Returns:
            float: Volume of water (m³).
        """
        delta_humidity = (max_humidity - min_humidity) / 100
        return delta_humidity * area * depth

    def convert_hectares_to_square_meters(self, hectares):
        """
        Converts an area from hectares to square meters.
        
        Parameters:
            hectares (float): Area in hectares.
            
        Returns:
            float: Area in square meters.
        """
        return hectares * 10000
