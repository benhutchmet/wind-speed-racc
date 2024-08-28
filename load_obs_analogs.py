#!/usr/bin/env python

"""
load_obs_analogs.py
===================

Script which, given a series of dates, loads the corresponding hourly data for
these days and produces a new .nc file containing the hourly data for the 
selected days.

Then passes the hourly data through Hannah B's CLEARHEADS-era power system
models to produce onshore and offshore wind capacity factors for different 
countries and regions across Europe.

Usage:
------

$ python load_obs_analogs.py

"""

# Import local libraries
import os
import sys
import glob

# Import third-party libraries
import iris
import cftime
import pandas as pd

def main():
    # Set up the hard coded paths
    analogs_path = "/home/users/pn832950/100m_wind/csv_files/s1960_lead1-30_month11_mse_df_model_matched_analogs_1960-1965.csv"

    # Load the analogs
    analogs = pd.read_csv(analogs_path)

    # Print the head of the dataframe
    print(analogs.head())


if __name__ == "__main__":
    main()