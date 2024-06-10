"""
Processing the ERA5 data for transfer to JASMIN for bias adjustment.
"""

# Imports
import os
import sys
import glob
import argparse
import time

# Import the functions from the load_wind_functions.py file
import load_wind_functions as lwf

# define a main function
def main():
    """
    Main function for processing the hourly wind data into daily wind data
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
    last_year = 2020
    
    # start a timer
    start_time = time.time()

    # Load the data
    temp_data = lwf.load_obs_data(
        last_year=last_year,
        first_year=first_year,
        parallel=False, # will take a while to run
        bias_correct=False,
        preprocess=lwf.preprocess_temp
    )

    # print the temperature data
    print(temp_data)

    # Set up a directory to save in
    output_dir = "/storage/silver/clearheads/Ben/saved_ERA5_data"

    # set up the fname
    fname = f"ERA5_wind_daily_{first_year}_{last_year}.nc"

    # set up the path
    path = os.path.join(output_dir, fname)

    # if the path doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # if the path doesn;t already exist, save the data
    if not os.path.exists(path):
        temp_data.to_netcdf(path)

        end_time = time.time()

    # print the time taken
    print(f"Time taken: {end_time - start_time} seconds.")

    # print that we are exiting the function
    print("Exiting the function.")
    sys.exit(0)

# Call the main function
if __name__ == "__main__":
    main()