import os
import sys
import glob

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
from netCDF4 import Dataset
import shapely.geometry
import cartopy.io.shapereader as shpreader
import csv
import scipy.interpolate

# print hello world
print("Hello World!")

# print the current working directory
print("Current working directory: ", os.getcwd())

dir = "/storage/silver/clearheads/Data/ERA5_data/native_grid/T2m_U100m_V100m_MSLP"

# list all files in the directory
files = glob.glob(dir + "/*")

print("Files in the directory: ", files)
