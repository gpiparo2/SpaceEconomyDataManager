#!/usr/bin/env python3
"""
evapotranspiration_analysis_functions.py
------------------------------------------
This module provides functions for analyzing evapotranspiration based on ground sensor data.
It includes functions for computing atmospheric pressure, delta, gamma, net radiation, reference
evapotranspiration (ET₀), and irrigation water volume. Additional helper functions for unit conversion are also provided.
Furthermore, the module includes an iterative analysis function that splits the overall period into
subperiods of a given length and computes complete evapotranspiration metrics for each subperiod.
"""

import math
import pandas as pd
from datetime import timedelta
import os
import matplotlib.pyplot as plt

def compute_pressure(altitude):
    """
    Computes atmospheric pressure at a given altitude.
    
    Parameters:
        altitude (float): Altitude in meters.
        
    Returns:
        float: Atmospheric pressure in kPa.
    """
    return 101.3 * ((293 - 0.0065 * altitude) / 293) ** 5.26

def compute_delta(temperature):
    """
    Computes the slope of the saturation vapor pressure curve (delta) at a given temperature.
    
    Parameters:
        temperature (float): Temperature in °C.
        
    Returns:
        float: Delta value in kPa/°C.
    """
    return (4098 * (0.6108 * math.exp((17.27 * temperature) / (temperature + 237.3)))) / ((temperature + 237.3) ** 2)

def compute_gamma(pressure):
    """
    Computes the psychrometric constant (gamma) given atmospheric pressure.
    
    Parameters:
        pressure (float): Atmospheric pressure in kPa.
        
    Returns:
        float: Gamma value in kPa/°C.
    """
    return 0.665e-3 * pressure

def compute_net_radiation(rs, rns, rnl):
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

def convert_Wm2_to_MJm2_day(rs_w_m2):
    """
    Converts solar radiation from W/m² to MJ/m²/day.
    
    Parameters:
        rs_w_m2 (float): Solar radiation in W/m².
        
    Returns:
        float: Solar radiation in MJ/m²/day.
    """
    return rs_w_m2 * 0.0864

def compute_et0(mean_temperature, mean_wind_speed, mean_air_humidity, solar_radiation_MJ):
    """
    Computes the reference evapotranspiration (ET₀) using the Penman-Monteith equation.
    
    Parameters:
        mean_temperature (float): Mean air temperature (°C).
        mean_wind_speed (float): Mean wind speed (m/s).
        mean_air_humidity (float): Mean relative humidity (%).
        solar_radiation_MJ (float): Mean solar radiation (MJ/m²/day).
        
    Returns:
        float: Reference evapotranspiration (mm/day).
    """
    pressure = compute_pressure(0)  # assuming sea level
    delta_val = compute_delta(mean_temperature)
    gamma_val = compute_gamma(pressure)
    rns = 0.1 * solar_radiation_MJ
    rnl = 0.1 * solar_radiation_MJ
    net_rad = compute_net_radiation(solar_radiation_MJ, rns, rnl)
    et0 = (0.408 * delta_val * net_rad + gamma_val * (900 / (mean_temperature + 273)) *
           mean_wind_speed * (1 - mean_air_humidity / 100)) / (delta_val + gamma_val * (1 + 0.34 * mean_wind_speed))
    return et0

def compute_water_volume(min_moisture, max_moisture, irrigated_area, depth):
    """
    Computes the volume of water required for irrigation based on soil moisture differences.
    
    Parameters:
        min_moisture (float): Minimum soil moisture (%).
        max_moisture (float): Maximum soil moisture (%).
        irrigated_area (float): Irrigated area (m²).
        depth (float): Sensor depth (m).
        
    Returns:
        float: Volume of water (m³).
    """
    delta_moisture = (max_moisture - min_moisture) / 100
    return delta_moisture * irrigated_area * depth

def convert_hectares_to_square_meters(hectares):
    """
    Converts an area from hectares to square meters.
    
    Parameters:
        hectares (float): Area in hectares.
        
    Returns:
        float: Area in square meters.
    """
    return hectares * 10000

def analyze_evapotranspiration_iterative(temp_file, rad_file, wind_file, humidity_file, 
                                         precip_file, soil_file, crop_coef, hectares, days, 
                                         overall_start_date, overall_end_date):
    """
    Performs iterative evapotranspiration analysis over multiple subperiods.
    The overall period from overall_start_date to overall_end_date is divided into intervals of length 'days'.
    For each interval, the function:
      - Loads and filters the meteorological and soil data,
      - Computes mean temperature, wind speed, air humidity, and solar radiation,
      - Calculates ET₀ and then adjusts it using the crop coefficient to obtain ETc,
      - Computes additional metrics (Pressure, Delta, Gamma, Net Radiation),
      - Processes soil moisture and precipitation data to calculate irrigation water requirements,
      - Computes the total evapotranspiration for the subperiod as ETc * number_of_days.
    The results for each subperiod (including all computed metrics) are saved to an Excel file and a plot is generated.
    
    Parameters:
        temp_file (str): Path to temperature data Excel file (with Date and Temperature columns).
        rad_file (str): Path to solar radiation data Excel file (with Date and SolarRadiation columns).
        wind_file (str): Path to wind speed data Excel file (with Date and WindSpeed columns).
        humidity_file (str): Path to air humidity data Excel file (with Date and AirHumidity columns).
        precip_file (str): Path to precipitation data Excel file (with Date and Precipitation columns).
        soil_file (str): Path to soil moisture data Excel file (with Date and SoilMoisture columns).
        crop_coef (float): Crop coefficient.
        hectares (float): Irrigated area in hectares.
        days (int or float): Length of each subperiod (in days).
        overall_start_date (datetime.date): Overall start date.
        overall_end_date (datetime.date): Overall end date.
        
    Returns:
        DataFrame: A DataFrame with complete computed metrics for each subperiod.
    """
    overall_start = pd.Timestamp(overall_start_date)
    overall_end = pd.Timestamp(overall_end_date)
    
    results = []
    sensor_depth = 0.20  # fixed sensor depth in meters
    current_start = overall_start
    while current_start <= overall_end:
        current_end = min(current_start + pd.Timedelta(days=days - 1), overall_end)
        period_days = (current_end - current_start).days + 1
        
        # Load and filter temperature data using only date (ignoring time)
        temp_df = pd.read_excel(temp_file, usecols=[0,1])
        temp_df.columns = ['Date', 'Temperature']
        temp_df['Date'] = pd.to_datetime(temp_df['Date'], dayfirst=True, errors='coerce')
        temp_df = temp_df.dropna(subset=['Date']).set_index('Date')
        temp_period = temp_df.loc[(temp_df.index.date >= current_start.date()) & (temp_df.index.date <= current_end.date())]
        
        rad_df = pd.read_excel(rad_file, usecols=[0,1])
        rad_df.columns = ['Date', 'SolarRadiation']
        rad_df['Date'] = pd.to_datetime(rad_df['Date'], dayfirst=True, errors='coerce')
        rad_df = rad_df.dropna(subset=['Date']).set_index('Date')
        rad_period = rad_df.loc[(rad_df.index.date >= current_start.date()) & (rad_df.index.date <= current_end.date())]
        
        wind_df = pd.read_excel(wind_file, usecols=[0,1])
        wind_df.columns = ['Date', 'WindSpeed']
        wind_df['Date'] = pd.to_datetime(wind_df['Date'], dayfirst=True, errors='coerce')
        wind_df = wind_df.dropna(subset=['Date']).set_index('Date')
        wind_period = wind_df.loc[(wind_df.index.date >= current_start.date()) & (wind_df.index.date <= current_end.date())]
        
        humidity_df = pd.read_excel(humidity_file, usecols=[0,1])
        humidity_df.columns = ['Date', 'AirHumidity']
        humidity_df['Date'] = pd.to_datetime(humidity_df['Date'], dayfirst=True, errors='coerce')
        humidity_df = humidity_df.dropna(subset=['Date']).set_index('Date')
        humidity_period = humidity_df.loc[(humidity_df.index.date >= current_start.date()) & (humidity_df.index.date <= current_end.date())]
        
        # Skip if any essential meteorological series is empty
        if temp_period.empty or rad_period.empty or wind_period.empty or humidity_period.empty:
            print(f"No meteorological data available for period {current_start.date()} to {current_end.date()}, skipping.")
            current_start = current_end + pd.Timedelta(days=1)
            continue
        
        # Compute mean values for meteorological data
        mean_temp = temp_period['Temperature'].mean()
        mean_rad = convert_Wm2_to_MJm2_day(rad_period['SolarRadiation'].mean())
        mean_wind = wind_period['WindSpeed'].mean()
        mean_humidity = humidity_period['AirHumidity'].mean()
        
        # Compute additional atmospheric parameters
        pressure = compute_pressure(0)
        delta_val = compute_delta(mean_temp)
        gamma_val = compute_gamma(pressure)
        rns = 0.1 * mean_rad
        rnl = 0.1 * mean_rad
        net_rad = compute_net_radiation(mean_rad, rns, rnl)
        
        # Compute ET₀ and then crop-adjusted ET (ETc)
        et0 = compute_et0(mean_temp, mean_wind, mean_humidity, mean_rad)
        etc = et0 * crop_coef
        total_et = etc * period_days
        
        # Process soil moisture data for the subperiod
        soil_df = pd.read_excel(soil_file, usecols=[0,1])
        soil_df.columns = ['Date', 'SoilMoisture']
        soil_df['Date'] = pd.to_datetime(soil_df['Date'], dayfirst=True, errors='coerce')
        soil_df = soil_df.dropna(subset=['Date']).set_index('Date')
        soil_period = soil_df.loc[(soil_df.index.date >= current_start.date()) & (soil_df.index.date <= current_end.date())]
        if soil_period.empty:
            print(f"No soil moisture data available for period {current_start.date()} to {current_end.date()}, skipping soil analysis.")
            min_soil = None
            max_soil = None
            water_volume = None
        else:
            soil_hourly = soil_period.resample('h').mean()
            min_soil = soil_hourly['SoilMoisture'].min()
            max_soil = soil_hourly['SoilMoisture'].max()
            water_volume = compute_water_volume(min_soil, max_soil, convert_hectares_to_square_meters(hectares), sensor_depth)
            # Apply smoothing and plot soil moisture for the current chunk
            window_size = 12
            soil_smoothed = soil_period['SoilMoisture'].rolling(window=window_size, center=True).mean()
            plt.figure(figsize=(10, 6))
            plt.plot(soil_period.index, soil_period['SoilMoisture'], label='Original Soil Moisture', alpha=0.7)
            plt.plot(soil_period.index, soil_smoothed, label=f'Smoothed Soil Moisture (window={window_size})', color='red', linewidth=2)
            plt.xlabel("Time")
            plt.ylabel("Soil Moisture (%)")
            plt.title(f"Soil Moisture from {current_start.date()} to {current_end.date()}")
            plt.legend()
            plt.show()
        
        # Process precipitation data for the subperiod
        precip_df = pd.read_excel(precip_file, usecols=[0,1], skiprows=[0])
        precip_df.columns = ['Date', 'Precipitation']
        precip_df['Date'] = pd.to_datetime(precip_df['Date'], dayfirst=True, errors='coerce')
        precip_df = precip_df.dropna(subset=['Date']).set_index('Date')
        precip_period = precip_df.loc[(precip_df.index.date >= current_start.date()) & (precip_df.index.date <= current_end.date())]
        if precip_period.empty:
            precipitation_sum_mm = 0
        else:
            precipitation_sum_mm = precip_period['Precipitation'].sum() * 0.7  # correction factor
        irrigated_area = convert_hectares_to_square_meters(hectares)
        precipitation_sum_m3 = (precipitation_sum_mm / 1000) * irrigated_area
        
        # Compute additional irrigation metrics
        ettot = etc * irrigated_area * 0.001
        et_precip_m3 = etc * precipitation_sum_m3
        if water_volume is not None and water_volume != 0:
            ratio = et_precip_m3 / (water_volume * period_days)
        else:
            ratio = None
        if ratio is not None:
            etmet = ratio * irrigated_area * 0.001 * 0.5
        else:
            etmet = None
        if etmet is not None:
            etirr = ettot - etmet
            evaporated_volume = etirr * period_days
            total_irrigation_water = (water_volume * period_days + evaporated_volume) - precipitation_sum_m3
        else:
            etirr = None
            evaporated_volume = None
            total_irrigation_water = None
        
        # Print organized output for the current subperiod
        print(f"Period: {current_start.date()} to {current_end.date()}")
        print(f"  Mean Temperature      : {mean_temp:.2f} °C")
        print(f"  Mean Wind Speed       : {mean_wind:.2f} m/s")
        print(f"  Mean Air Humidity     : {mean_humidity:.2f} %")
        print(f"  Mean Solar Radiation  : {mean_rad:.2f} MJ/m²/day")
        print(f"  Atmospheric Pressure  : {pressure:.2f} kPa")
        print(f"  Delta                 : {delta_val:.2f} kPa/°C")
        print(f"  Gamma                 : {gamma_val:.2f} kPa/°C")
        print(f"  Net Radiation         : {net_rad:.2f} MJ/m²/day")
        print(f"  Reference ET (ET₀)    : {et0:.2f} mm/day")
        print(f"  Crop-Adjusted ET (ETc): {etc:.2f} mm/day")
        print(f"  Total ET (over {period_days} days): {total_et:.2f} mm")
        if water_volume is not None:
            print(f"  Soil Moisture - Min   : {min_soil}")
            print(f"  Soil Moisture - Max   : {max_soil}")
            print(f"  Irrigation Water Volume: {water_volume:.2f} m³")
        print(f"  Precipitation Sum     : {precipitation_sum_m3:.2f} m³")
        if etmet is not None:
            print(f"  ETmet                 : {etmet:.2f}")
        if etirr is not None:
            print(f"  ETirr                 : {etirr:.2f}")
        if total_irrigation_water is not None:
            print(f"  Total Irrigation Water: {total_irrigation_water:.2f} m³")
        print("")
        
        results.append({
            'StartDate': current_start.date(),
            'EndDate': current_end.date(),
            'PeriodDays': period_days,
            'MeanTemperature': mean_temp,
            'MeanWindSpeed': mean_wind,
            'MeanAirHumidity': mean_humidity,
            'MeanSolarRadiation': mean_rad,
            'Pressure': pressure,
            'Delta': delta_val,
            'Gamma': gamma_val,
            'NetRadiation': net_rad,
            'ET0': et0,
            'CropAdjustedET': etc,
            'TotalET': total_et,
            'MinSoil': min_soil,
            'MaxSoil': max_soil,
            'WaterVolume': water_volume,
            'PrecipitationSum_m3': precipitation_sum_m3,
            'ETmet': etmet,
            'ETirr': etirr,
            'TotalIrrigationWater': total_irrigation_water
        })
        
        current_start = current_end + pd.Timedelta(days=1)
    
    results_df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(soil_file), "iterative_evapotranspiration.xlsx")
    results_df.to_excel(output_path, index=False)
    print(f"\nIterative evapotranspiration results saved to {output_path}")
    
    # Plot Total ET vs. Start Date
    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(results_df['StartDate']), results_df['TotalET'], marker='o', linestyle='-')
    plt.xlabel("Start Date")
    plt.ylabel("Total Evapotranspiration (mm)")
    plt.title("Iterative Evapotranspiration Analysis")
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.show()
    
    # Plot Total Irrigation Water vs. Start Date if available
    if results_df['TotalIrrigationWater'].notnull().all():
        plt.figure(figsize=(10, 5))
        plt.plot(pd.to_datetime(results_df['StartDate']), results_df['TotalIrrigationWater'], marker='o', linestyle='-')
        plt.xlabel("Start Date")
        plt.ylabel("Total Irrigation Water (m³)")
        plt.title("Iterative Irrigation Water Analysis")
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        plt.show()
    
    return results_df
