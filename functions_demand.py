"""
Functions adapted from Hannah Bloomfield's code for S2S4E for European demand model. 

First testing with daily reanalysis data for the UK.

Before moving on to see whether decadal predictions can be used for this.
"""

import glob
import os
import sys
import re

import numpy as np
import cartopy.io.shapereader as shpreader
from netCDF4 import Dataset
import shapely.geometry as sgeom
import pandas as pd
import xarray as xr
from tqdm import tqdm

# Import the user defined dictionaries
import dictionaries as dicts

# Define a function to calculate the spatial mean of a masked dataset
# and convert to a dataframe
def calc_spatial_mean(
    ds: xr.Dataset,
    country: str,
    variable: str,
    convert_kelv_to_cel: bool = True,
) -> pd.DataFrame:
    """
    Calculate the spatial mean of a masked dataset and convert to a DataFrame.

    Parameters
    ----------

    ds: xr.Dataset
        The dataset to calculate the spatial mean from.

    country: str
        The country to calculate the spatial mean for.

    variable: str
        The variable to calculate the spatial mean for.

    convert_kelv_to_cel: bool
        Whether to convert the data from Kelvin to Celsius.

    Returns
    -------

    df: pd.DataFrame
        The DataFrame of the spatial mean.

    """

    # Extract the data for the country
    ds = ds[variable]

    # Convert to a numpy array
    data = ds.values

    # Take the mean over the lat and lon dimensions
    data_mean = np.nanmean(data, axis=(1, 2))

    # Set up the time index
    time_index = ds.time.values

    # Set up the DataFrame
    df = pd.DataFrame(data_mean, index=time_index, columns=[f"{country}_{variable}"])

    # If convert_kelv_to_cel is True
    if convert_kelv_to_cel:
        # Convert the data from Kelvin to Celsius
        df[f"{country}_{variable}"] = df[f"{country}_{variable}"] - 273.15

    return df

#  Calculate the heating degree days and cooling degree days
def calc_hdd_cdd(
    df: pd.DataFrame,
    hdd_base: float = 15.5,
    cdd_base: float = 22.0,
    temp_suffix: str = "t2m",
    hdd_suffix: str = "hdd",
    cdd_suffix: str = "cdd",
) -> pd.DataFrame:
    """
    Calculate the heating degree days and cooling degree days.

    Parameters
    ----------

    df: pd.DataFrame
        The CLEARHEADS data.

    hdd_base: float
        The base temperature for the heating degree days.

    cdd_base: float
        The base temperature for the cooling degree days.

    temp_suffix: str
        The suffix for the temperature.

    hdd_suffix: str
        The suffix for the heating degree days.

    cdd_suffix: str
        The suffix for the cooling degree days.

    Returns
    -------

    df: pd.DataFrame
        The CLEARHEADS data with the heating degree days and cooling degree days.

    """

    # if the data is not already in daily format, resample to daily
    if df.index.freq != "D":
        print("Resampling to daily")

        # Resample the data
        df = df.resample("D").mean()

    # if the first column does not contain the temperature suffix
    if temp_suffix not in df.columns[0]:
        # add the temperature suffix to the columns
        df.columns = [f"{col}_{temp_suffix}" for col in df.columns]

    # Loop over the columns
    for col in df.columns:
        # strip t2m from the column name
        col_raw = col.replace(f"_{temp_suffix}", "")

        # set up the column names
        hdd_col = f"{col_raw}_{hdd_suffix}"
        cdd_col = f"{col_raw}_{cdd_suffix}"

        # Calculate the heating degree days
        df[hdd_col] = df[col].apply(lambda x: max(0, hdd_base - x))

        # Calculate the cooling degree days
        df[cdd_col] = df[col].apply(lambda x: max(0, x - cdd_base))

    return df

# Write a function which calculates the weather dependent demand
# Based on the heating and cooling degree days and the demand coefficients
def calc_national_wd_demand(
    df: pd.DataFrame,
    fpath_reg_coefs: str = "/home/users/pn832950/UREAD_energy_models_demo_scripts/ERA5_Regression_coeffs_demand_model.csv",
    demand_year: float = 2017.0,
    country_names: dict = dicts.countries_nuts_id,
    hdd_name: str = "HDD",
    cdd_name: str = "CDD",
) -> pd.DataFrame:
    """
    Calculate the national weather dependent demand.

    Parameters
    ----------

    df: pd.DataFrame
        The CLEARHEADS data.

    fpath_reg_coefs: str
        The file path for the regression coefficients.

    demand_years: float
        The year for the time coefficient.

    country_names: dict
        The dictionary of country names. Matched up the full country names
        with the NUTS IDs.

    hdd_name: str
        The name of the heating degree days.

    cdd_name: str
        The name of the cooling degree days.

    Returns
    -------

    df: pd.DataFrame
        The CLEARHEADS data with the national weather dependent demand.

    """

    # Loop over the columns in the DataFrame
    for col in df.columns:
        # Loop over the country names
        for country_name, country_id in country_names.items():
            # print(f"Calculating demand for {country_name}")
            # print(f"Country ID: {country_id}")
            # if the country id is in the column name
            if country_id in col:
                # Split the column name by _
                col_split = col.split("_")

                # Set up the new column name
                new_col = f"{country_name}_{col_split[1]}"

                # Update the column name
                df = df.rename(columns={col: new_col})

    # Load int the regression coefficients data
    reg_coeffs = pd.read_csv(fpath_reg_coefs)

    # Set the index to the first column
    reg_coeffs.set_index("Unnamed: 0", inplace=True)

    # Loop over the columns in the DataFrame
    for reg_col in reg_coeffs.columns:
        if reg_col != "Unnamed: 0":
            # Split the column name by _regression
            # e.g. Austria
            country = reg_col.split("_regression")[0]

            # if df contains f{country}_hdd and f{country}_cdd
            if f"{country}_hdd" in df.columns and f"{country}_cdd" in df.columns:
                # Extract the time coefficient for col
                time_coeff = reg_coeffs.loc["time", reg_col]

                # Extract the hdd coefficient for col
                hdd_coeff = reg_coeffs.at[hdd_name, reg_col]

                # Extract the cdd coefficient for col
                cdd_coeff = reg_coeffs.at[cdd_name, reg_col]

                # print the coefficients
                # print(f"Time coefficient: {time_coeff}")
                # print(f"HDD coefficient: {hdd_coeff}")
                # print(f"CDD coefficient: {cdd_coeff}")

                # Calculate the demand
                df[f"{country}_demand"] = (
                    (time_coeff * demand_year)
                    + (hdd_coeff * df[f"{country}_hdd"])
                    + (cdd_coeff * df[f"{country}_cdd"])
                )

    return df


# Define a function to save the dataframe
def save_df(
    df: pd.DataFrame,
    fname: str,
    fdir: str = "/gws/nopw/j04/canari/users/benhutch/met_to_energy_dfs",
    ftype: str = "csv",
) -> None:
    """
    Save the DataFrame.

    Parameters
    ----------

    df: pd.DataFrame
        The DataFrame to save.

    fname: str
        The filename to save the DataFrame.

    fdir: str
        The directory to save the DataFrame.

    ftype: str
        The file type to save the DataFrame.

    Returns
    -------

    None

    """

    # If the file type is csv
    if ftype == "csv":
        # Save the DataFrame as a CSV
        df.to_csv(f"{fdir}/{fname}.csv")
    else:
        raise NotImplementedError(f"File type {ftype} not implemented.")

    return None

# define a main function for testing
def main():
    # set up the args
    model_variable = "tas"
    model = "HadGEM3-GC31-MM"
    init_years = np.arange(1960, 1965 + 1)
    experiment = "dcppA-hindcast"
    frequency = "day"

    # print testing complete
    print("Testing complete")

    return None


# Run the main function
if __name__ == "__main__":
    main()
