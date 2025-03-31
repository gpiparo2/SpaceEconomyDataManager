# evalscripts.py

def evalscript_S1GRD(polarizations=None, backCoeff='SIGMA0_ELLIPSOID', orthorectify=False, demInstance='COPERNICUS', sampleType='FLOAT32', return_band_list=False):
    """
    Evalscript for Sentinel-1 GRD data that includes selected polarizations and optional additional parameters. 

    Parameters:
        polarizations (list): List of polarizations to include (e.g., ['VV', 'VH']). If None, include all available.
        backCoeff (str): Backscatter coefficient to use ('SIGMA0_ELLIPSOID', 'GAMMA0_ELLIPSOID', or 'GAMMA0_TERRAIN').
        orthorectify (bool): Whether to orthorectify the data using a Digital Elevation Model (DEM).
        demInstance (str): DEM instance to use ('COPERNICUS', 'MAPZEN', etc.) if orthorectify is True.
        sampleType (str): Sample type for the output data ('FLOAT32', 'UINT16', etc.).
        return_band_list (bool): If True, returns a tuple (evalscript, band_list).

    Returns:
        str or tuple: The evalscript string to be used in Sentinel Hub requests. If return_band_list is True, also returns the list of bands used.
    """
    # Default polarizations if none are specified
    if polarizations is None:
        polarizations = ['VV', 'VH', 'HH', 'HV']

    # Available bands and their default units
    available_bands = ['VV', 'VH', 'HH', 'HV']
    # special_bands = ['localIncidenceAngle', 'scatteringArea', 'elevationAngle', 'azimuthAngle']
    special_bands = []
    all_bands = [band for band in available_bands if band in polarizations] + special_bands

    # Build the evalscript
    evalscript = f"""
//VERSION=3
/*{{
processing: {{
    "backCoeff": "{backCoeff}",
    "orthorectify": {str(orthorectify).lower()},
    "demInstance": "{demInstance}"
}}
}}*/
function setup() {{
    return {{
        input: [{{
            bands: [{', '.join(f'"{band}"' for band in all_bands)}],
            metadata: ["orthorectification"]
        }}],
        output: {{
            id: "default",
            bands: {len(all_bands)},
            sampleType: "{sampleType}"
        }}
    }};
}}

function evaluatePixel(sample) {{
    return [{', '.join(f'sample.{band}' for band in all_bands)}];
}}
"""

    if return_band_list:
        return evalscript, all_bands
    else:
        return evalscript

    
def evalscript_S2L2A(units="REFLECTANCE", sampleType="AUTO", bands: list = None, return_band_list=False):
    """
    Evalscript for Sentinel-2 Level 2A data that includes spectral bands and special bands.
    Allows overriding the default spectral band list via the 'bands' parameter.
    
    Parameters:
        units (str): Units for spectral bands (e.g., 'REFLECTANCE' or 'DN').
        sampleType (str): Sample type for spectral bands (e.g., 'AUTO', 'UINT8', 'UINT16', 'FLOAT32').
        bands (list): Optional list of spectral bands to use; if not provided, defaults are used.
        return_band_list (bool): If True, returns a tuple (evalscript, band_list).
    
    Returns:
        str or tuple: The evalscript string for Sentinel-2 requests, and optionally the list of bands used.
    """
    default_spectral_bands = [
        "B01", "B02", "B03", "B04", "B05", "B06",
        "B07", "B08", "B8A", "B09", "B11", "B12"
    ]
    special_bands = ["AOT", "SCL", "SNW", "CLD", "dataMask"]
    if bands is None:
        all_bands = default_spectral_bands + special_bands
    else:
        all_bands = bands
 

    units_dict = {
        "AOT": "OPTICAL_DEPTH",
        "SCL": "DN",
        "SNW": "PERCENT",
        "CLD": "PERCENT",
        "dataMask": "DN"
    }
    units_list = []
    for band in all_bands:
        unit = units_dict.get(band, units)
        units_list.append(f'"{unit}"')
    units_js = "[\n    " + ",\n    ".join(units_list) + "\n]"
    evalscript = f"""
//VERSION=3
function setup() {{
    return {{
        input: [{{
            bands: [{", ".join(f'"{band}"' for band in all_bands)}],
            units: {units_js}
        }}],
        output: {{
            id: "default",
            bands: {len(all_bands)},
            sampleType: "{sampleType}"
        }}
    }};
}}

function evaluatePixel(sample) {{
    return [{", ".join(f'sample.{band}' for band in all_bands)}];
}}
"""
    if return_band_list:
        return evalscript, all_bands
    else:
        return evalscript
    
    
def evalscript_S3_OLCI_L1B(sampleType="FLOAT32", bands: list = None, return_band_list=False):
    """
    Evalscript for Sentinel-3 OLCI L1B data that includes spectral bands.
    Allows overriding the default band list via the 'bands' parameter.
    
    Parameters:
        sampleType (str): Sample type for the output data (e.g., 'FLOAT32').
        bands (list): Optional list of bands to use; if not provided, defaults to OLCI bands.
        return_band_list (bool): If True, returns a tuple (evalscript, band_list).
    
    Returns:
        str or tuple: The evalscript string for Sentinel-3 OLCI requests, and optionally the list of bands used.
    """
    if bands is None:
        spectral_bands = [f"B{str(i).zfill(2)}" for i in range(1, 22)]
    else:
        spectral_bands = bands
    evalscript = f"""
//VERSION=3
function setup() {{
    return {{
        input: [{{
            bands: [{", ".join(f'"{band}"' for band in spectral_bands)}],
            units: "REFLECTANCE"
        }}],
        output: {{
            id: "default",
            bands: {len(spectral_bands)},
            sampleType: "{sampleType}"
        }}
    }};
}}

function evaluatePixel(sample) {{
    return [{", ".join(f'sample.{band}' for band in spectral_bands)}];
}}
"""
    if return_band_list:
        return evalscript, spectral_bands
    else:
        return evalscript

def evalscript_S3_SLSTR_L1B_optical(sampleType="FLOAT32", bands: list = None, return_band_list=False):
    """
    Evalscript for Sentinel-3 SLSTR optical data.
    Allows overriding the default band list (default S1 to S6) via the 'bands' parameter.
    
    Parameters:
        sampleType (str): Sample type for the output data.
        bands (list): Optional list of optical bands to use; if not provided, defaults to S1 to S6.
        return_band_list (bool): If True, returns a tuple (evalscript, band_list).
    
    Returns:
        str or tuple: The evalscript string for Sentinel-3 SLSTR optical requests, and optionally the list of bands used.
    """
    if bands is None:
        optical_bands = [f"S{str(i)}" for i in range(1, 7)]
    else:
        optical_bands = bands
    units_list = ['"REFLECTANCE"' for _ in optical_bands]
    units_js = "[\n    " + ",\n    ".join(units_list) + "\n]"
    evalscript = f"""
//VERSION=3
function setup() {{
    return {{
        input: [{{
            bands: [{", ".join(f'"{band}"' for band in optical_bands)}],
            units: {units_js}
        }}],
        output: {{
            id: "default",
            bands: {len(optical_bands)},
            sampleType: "{sampleType}"
        }}
    }};
}}

function evaluatePixel(sample) {{
    return [{", ".join(f'sample.{band}' for band in optical_bands)}];
}}
"""
    if return_band_list:
        return evalscript, optical_bands
    else:
        return evalscript

def evalscript_S3_SLSTR_L1B_thermal(sampleType="FLOAT32", bands: list = None, return_band_list=False):
    """
    Evalscript for Sentinel-3 SLSTR thermal data.
    Allows overriding the default band list (default S7-S9, F1, F2) via the 'bands' parameter.
    
    Parameters:
        sampleType (str): Sample type for the output data.
        bands (list): Optional list of thermal bands to use; if not provided, defaults to S7, S8, S9, F1, F2.
        return_band_list (bool): If True, returns a tuple (evalscript, band_list).
    
    Returns:
        str or tuple: The evalscript string for Sentinel-3 SLSTR thermal requests, and optionally the list of bands used.
    """
    if bands is None:
        thermal_bands = [f"S{str(i)}" for i in range(7, 10)] + ["F1", "F2"]
    else:
        thermal_bands = bands
    units_list = ['"BRIGHTNESS_TEMPERATURE"' for _ in thermal_bands]
    units_js = "[\n    " + ",\n    ".join(units_list) + "\n]"
    evalscript = f"""
//VERSION=3
function setup() {{
    return {{
        input: [{{
            bands: [{", ".join(f'"{band}"' for band in thermal_bands)}],
            units: {units_js}
        }}],
        output: {{
            id: "default",
            bands: {len(thermal_bands)},
            sampleType: "{sampleType}"
        }}
    }};
}}

function evaluatePixel(sample) {{
    return [{", ".join(f'sample.{band}' for band in thermal_bands)}];
}}
"""
    if return_band_list:
        return evalscript, thermal_bands
    else:
        return evalscript



    
def evalscript_DEM(sampleType="FLOAT32", demInstance="COPERNICUS_30", return_band_list=False):
    """
    Evalscript for Digital Elevation Model (DEM) data.

    Parameters:
        sampleType (str): Sample type for the output data ('FLOAT32', 'UINT16', etc.).
        demInstance (str): DEM instance to use ('COPERNICUS', 'COPERNICUS_30', 'COPERNICUS_90').
        return_band_list (bool): If True, returns a tuple (evalscript, band_list).

    Returns:
        str or tuple: The evalscript string to be used in Sentinel Hub requests.
    """
    evalscript = f"""
//VERSION=3
/*{{
"demInstance": "{demInstance}"
}}*/
function setup() {{
    return {{
        input: [{{
            bands: ["DEM"],
            units: "meters"
        }}],
        output: {{
            id: "default",
            bands: 1,
            sampleType: "{sampleType}"
        }}
    }};
}}

function evaluatePixel(sample) {{
    return [sample.DEM];
}}
"""
    if return_band_list:
        return evalscript, ['DEM']
    else:
        return evalscript