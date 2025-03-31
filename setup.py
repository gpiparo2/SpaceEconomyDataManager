#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="SpaceEconomyDataManager",
    version="0.1.0",
    author="Giuseppe Piparo",
    author_email="giuseppe.piparo@ct.infn.it",
    description="A library for downloading, processing, and analyzing satellite data from Copernicus constellation.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="",  # update with your URL if applicable
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=[
        "numpy",
        "rasterio",
        "shapely",
        "geopandas",
        "tensorflow",
        "sentinelhub",
        "ipywidgets"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6'
)