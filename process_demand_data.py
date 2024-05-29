"""
Processing the ERA5 t2m data into demand data.
"""

# Imports
import os
import sys
import glob
import argparse
import time

# Import the functions from the load_wind_functions.py file
import load_wind_functions as lwf

# Import the demand functions
import functions_demand as fd

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

    # print the temperature data
    print(temp_data)

    # subset to time series for country as dataframe
    ds = lwf.apply_country_mask(
        ds=temp_data,
        country="United Kingdom",
    )

    # print the subset data
    print(f"Subset data: {ds}")

    # Calculate the mean for the country
    df = fd.calc_spatial_mean(
        ds=ds,
        country="United_Kingdom",
        variable="t2m",
    )

    # print the head of the dataframe
    print(f"Head of the dataframe: {df.head()}")

    # Calculate the heating degree days and cooling degree days
    df = fd.calc_hdd_cdd(
        df=df,
        temp_suffix="t2m",
    )

    # print the head of the dataframe
    print(f"Head of the dataframe: {df.head()}")

    # Calculate the weather dependent demand
    df = fd.calc_national_wd_demand(
        df=df,
    )

    # print the head of the dataframe
    print(f"Head of the dataframe: {df.head()}")

    # Set up a directory to save in
    output_dir = "/storage/silver/clearheads/Ben/saved_ERA5_data"

    # set up the fname
    fname = f"ERA5_t2m_daily_{first_year}_{last_year}.nc"

    # set up the path
    path = os.path.join(output_dir, fname)

    # # if the path doesn't exist, create it
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # # if the path doesn;t already exist, save the data
    # if not os.path.exists(path):
    #     temp_data.to_netcdf(path)

    end_time = time.time()

    # print the time taken
    print(f"Time taken: {end_time - start_time} seconds.")

    # print that we are exiting the function
    print("Exiting the function.")
    sys.exit(0)

# Call the main function
if __name__ == "__main__":
    main()