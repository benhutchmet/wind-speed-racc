"""
Processing the ERA5 t2m data into demand data.
"""

# Imports
import os
import sys
import glob
import argparse
import time

import pandas as pd
from tqdm import tqdm

# Import the functions from the load_wind_functions.py file
import load_wind_functions as lwf

# Import the demand functions
import functions_demand as fd

# Import the dictionaries
import dictionaries as dicts

# define a main function
def main():
    """
    Main function for processing the hourly temperature data into daily temperature data
    for a given country and given year.
    
    Parameters:
    -----------
    
        - None
    
    Returns:
    --------
    
        - None
    """
    
    # Hard code the years for testing
    first_year = 1950
    last_year = 1950
    last_month = 1
    first_month = 1
    
    # start a timer
    start_time = time.time()

    # Initialize an empty dataframe
    all_data = pd.DataFrame()

    # Loop over countries
    for country in tqdm(dicts.country_list_nuts0, desc="Looping over countries"):
        # Print the country which we are processing
        print(f"Processing country: {country}")

        # if country is in ["Macedonia"] skip
        if country in ["Macedonia"]:
            print(f"Skipping {country}")
            continue


        # Load the data
        temp_data = lwf.load_obs_data(
            last_year=last_year,
            last_month=last_month,
            first_year=first_year,
            first_month=first_month,
            parallel=False, # will take a while to run
            bias_correct_wind=False,
            preprocess=lwf.preprocess_temp
        )


        # subset to time series for country as dataframe
        ds = lwf.apply_country_mask(
            ds=temp_data,
            country=country,
        )

        # if country contains spaces, replace these with _
        country = country.replace(" ", "_")

        # Calculate the mean for the country
        df = fd.calc_spatial_mean(
            ds=ds,
            country=country,
            variable="t2m",
        )


        # Calculate the heating degree days and cooling degree days
        df = fd.calc_hdd_cdd(
            df=df,
            temp_suffix="t2m",
        )


        # Calculate the weather dependent demand
        df = fd.calc_national_wd_demand(
            df=df,
        )

        # Append the data for this country to the main DataFrame
        all_data = pd.concat([all_data, df], axis=1)

    # print the head of the temp data
    print(all_data.head())

    # Set up a directory to save in
    output_dir = "/storage/silver/clearheads/Ben/saved_ERA5_data"

    # set up the fname
    fname = f"ERA5_wd_demand_daily_{first_year}_{last_year}_all_countries.csv"

    # set up the path
    path = os.path.join(output_dir, fname)

    # if the path doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # if the path doesn;t already exist, save the data
    if not os.path.exists(path):
        all_data.to_csv(path)

    end_time = time.time()

    # print the time taken
    print(f"Time taken: {end_time - start_time} seconds.")

    # print that we are exiting the function
    print("Exiting the function.")
    sys.exit(0)

# Call the main function
if __name__ == "__main__":
    main()