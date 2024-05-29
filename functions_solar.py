"""
Functions which use the met data for t2m and rsds to convert into solar power.

Based on Hannah's S2S4E code.
"""

import os
import sys
import glob
import argparse
import time

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# Import the functions from the load_wind_functions.py file
import load_wind_functions as lwf

# Import the dictionaries
import dictionaries as dicts


# Write a function to process the solar power output
def solar_PV_model(
    ds_temp: xr.Dataset,
    ds_rsds: xr.Dataset,
    country: str,
    rsds_varname: str = "ssrd",
    temp_varname: str = "t2m",
    T_ref: float = 25.0,
    eff_reff: float = 0.9,
    beta_ref: float = 0.0042,
    G_ref: float = 1000.0,
) -> pd.DataFrame:
    """
    Function to process the solar power output using the temperature and rsds data.

    Parameters
    ----------

    ds_temp: xr.Dataset
        The dataset containing the temperature data.

    ds_rsds: xr.Dataset
        The dataset containing the rsds data.

    country: str
        The country to process the data for.

    rsds_varname: str
        The variable name for rsds in the dataset.

    temp_varname: str
        The variable name for temperature in the dataset.

    T_ref: float
        The reference temperature for the solar power model.

    eff_reff: float
        The reference efficiency for the solar power model.

    beta_ref: float
        The reference coefficient for the solar power model.

    G_ref: float
        The reference solar irradiance for the solar power model.

    Returns
    -------

    df: pd.DataFrame
        The DataFrame containing the solar power output.

    """

    # apply the country mask to the temp data
    temp_country = lwf.apply_country_mask(
        ds=ds_temp,
        country=country,
        pop_weights=0,
    )

    # apply the country mask to the rsds data
    rsds_country = lwf.apply_country_mask(
        ds=ds_rsds,
        country=country,
        pop_weights=0,
    )

    # Apply the conversions to the data
    # Convert the temperature data from Kelvin to Celsius
    temp_country_data = temp_country.values - 273.15

    # Convert the rsds data from J/m^2 to W/m^2
    rsds_country_data = rsds_country.values / 3600

    

    return df


def main():
    print("Processing solar power output using t2m and rsds data.")

    # Hard code the years for testing
    first_year = 1950
    last_year = 1950
    last_month = 1
    first_month = 1

    # Set up the paths to the RSDS data
    S2S4E_dir_rsds = "/storage/silver/S2S4E/energymet/ERA5/RSDS/native_grid_hourly/"
    CLEARHEADS_dir_rsds = "/storage/silver/clearheads/Data/ERA5_data/native_grid/RSDS/"

    # Start a timer
    start_time = time.time()

    # Initialize an empty dataframe
    all_data = pd.DataFrame()

    # Load the data for t2m
    t2m = lwf.load_obs_data(
        last_year=last_year,
        last_month=last_month,
        first_year=first_year,
        first_month=first_month,
        parallel=False,  # will take a while to run
        bias_correct_wind=False,
        preprocess=lwf.preprocess_temp,
    )

    # Load the data for rsds
    rsds = lwf.load_obs_data(
        last_year=last_year,
        last_month=last_month,
        first_year=first_year,
        first_month=first_month,
        parallel=False,  # will take a while to run
        bias_correct_wind=False,
        preprocess=lwf.preprocess_rsds,
    )

    # Loop over countries
    for country in tqdm(dicts.country_list_nuts0, desc="Looping over countries"):
        # Print the country which we are processing
        print(f"Processing country: {country}")

        # if country is in ["Macedonia"] skip
        if country in ["Macedonia"]:
            print(f"Skipping {country}")

        #

    # Load the data for t2m

    # load the data for rsds

    # pass through solar power function


if __name__ == "__main__":
    main()
