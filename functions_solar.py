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
    UK_solar_farm_dist_file: str = None,
    EU_solar_farm_dist_file: str = None,
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

    UK_solar_farm_dist_file: str
        The file containing the distribution of solar farms in the UK.

    EU_solar_farm_dist_file: str
        The file containing the distribution of solar farms in the EU.

    Returns
    -------

    df: pd.DataFrame
        The DataFrame containing the solar power output.

    """

    if UK_solar_farm_dist_file is not None:
        # not implemented yet error
        print("UK_solar_farm_dist_file not implemented yet.")

    if EU_solar_farm_dist_file is not None:
        # not implemented yet error
        print("EU_solar_farm_dist_file not implemented yet.")

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

    # Load the country mask
    country_mask = lwf.load_country_mask(
        ds=ds_temp,
        country=country,
        pop_weights=0,
    )

    # Apply the conversions to the data
    # Convert the temperature data from Kelvin to Celsius
    temp_country_data = temp_country[temp_varname].values - 273.15

    # # convert the temperature data from Kelvin to Celsius
    # temp_country_data = temp_country_data - 273.15

    # Convert the rsds data from J/m^2 to W/m^2
    rsds_country_data = rsds_country[rsds_varname].values / 3600

    # Calculate the relative efficiency of the panel
    rel_eff_panel = eff_reff * (1 - beta_ref * (temp_country_data - T_ref))

    # Calculate the capacity factor of panel
    cap_fac_panel = np.nan_to_num(rel_eff_panel * (rsds_country_data / G_ref))

    # Calculate the spatial mean solar cf
    spatial_mean_solar_cf = np.zeros([len(cap_fac_panel)])

    # Loop over the time steps
    # Any weighted averaging here?
    for i in range(0, len(cap_fac_panel)):
        # Calculate the mean of the capacity factor
        spatial_mean_solar_cf[i] = np.average(cap_fac_panel[i, :, :], weights=country_mask)

    # Create a DataFrame to store the data
    df = pd.DataFrame(
        {
            "time": temp_country.time.values,
            f"solar_cf_{country}": spatial_mean_solar_cf,
        }
    )
    
    return df


def main():
    print("Processing solar power output using t2m and rsds data.")

    # Hard code the years for testing
    first_year = 1960
    last_year = 1960
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
        S2S4E_dir=S2S4E_dir_rsds,
        CLEARHEADS_dir=CLEARHEADS_dir_rsds,
        parallel=False,  # will take a while to run
        bias_correct_wind=False,
        preprocess=lwf.preprocess_rsds,
    )

    # Loop over countries
    for country in tqdm(dicts.country_list_nuts0[:5], desc="Looping over countries"):
        # Print the country which we are processing
        print(f"Processing country: {country}")

        # if country is in ["Macedonia"] skip
        if country in ["Macedonia"]:
            print(f"Skipping {country}")

        # pass the data to the solar_PV_model function
        df = solar_PV_model(
            ds_temp=t2m,
            ds_rsds=rsds,
            country=country,
        )

        # # print the head of the df
        # print(f"Head of the df for {country}:")
        # print(df.head())

        # Concat the data to the all_data DataFrame
        all_data = pd.concat([all_data, df], axis=1)

    # Print the head of the all_data DataFrame
    print(all_data.head())

    # Set up the output dir
    output_dir = "/storage/silver/clearheads/Ben/saved_ERA5_data"

    # set up the fname
    fname = f"ERA5_solar_cfs_daily_{first_year}_{first_month}_{last_year}_{last_month}_all_countries.csv"

    # Save the data to a csv file
    # if the output directory does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # if the path does not exist, create it
    if not os.path.exists(os.path.join(output_dir, fname)):
        all_data.to_csv(os.path.join(output_dir, fname))
    else:
        print(f"File {fname} already exists.")

if __name__ == "__main__":
    main()
