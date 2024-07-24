"""
process_ERA5_data.py
====================

Processing the ERA5 data for transfer to JASMIN for bias adjustment.

Usage:
------

    python process_ERA5_data.py <start_year> <end_year> <variable_name>

Example:
--------

    python process_ERA5_data.py 1950 1960 temp

Arguments:
----------

    - start_year: int
        The start year for the data.
    
    - end_year: int
        The end year for the data.
    
    - variable_name: str
        The name of the variable to process.

Returns:
--------

    - None

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
    first_year = sys.argv[1]
    last_year = sys.argv[2]
    variable_name = sys.argv[3]
    
    # if first year is not an integer, raise an error
    if type(first_year) != int:
        # convert the first year to an integer
        first_year = int(first_year)

    # if last year is not an integer, raise an error
    if type(last_year) != int:
        # convert the last year to an integer
        last_year = int(last_year)

    # print the args
    print(f"First year: {first_year}")
    print(f"Last year: {last_year}")
    print(f"Variable name: {variable_name}")

    # print the type of the first year
    print(f"Type of first year: {type(first_year)}")
    print(f"Type of last year: {type(last_year)}")
    print(f"Type of variable name: {type(variable_name)}")

    # start a timer
    start_time = time.time()

    # depending on the variable name, load the data
    if variable_name == "wind":
        print("Loading the wind data.")
        # Load the data
        temp_data = lwf.load_obs_data(
            last_year=last_year,
            first_year=first_year,
            parallel=False, # will take a while to run
            bias_correct_wind=True, # bias correct for wind speed
            preprocess=lwf.preprocess,
        )
    elif variable_name == "rsds":
        print("Loading the rsds data.")
        # Load the data
        temp_data = lwf.load_obs_data(
            last_year=last_year,
            first_year=first_year,
            S2S4E_dir="/storage/silver/S2S4E/energymet/ERA5/RSDS/native_grid_hourly/",
            CLEARHEADS_dir="/storage/silver/clearheads/Data/ERA5_data/native_grid/RSDS/",
            parallel=False, # will take a while to run
            bias_correct_wind=False, # don't bias correct for wind speed
            preprocess=lwf.preprocess_rsds,
        )
    else:
        raise ValueError("Variable name not recognised.")

    # print the temperature data
    print(temp_data)

    # Set up a directory to save in
    output_dir = "/storage/silver/clearheads/Ben/saved_ERA5_data"

    # set up the fname
    fname = f"ERA5_{variable_name}_daily_{first_year}_{last_year}.nc"

    # set up the path
    path = os.path.join(output_dir, fname)

    # if the path doesn't exist, create it
    if not os.path.exists(output_dir):
        print(f"Creating the directory: {output_dir}")
        os.makedirs(output_dir)

    # if the path doesn't already exist, save the data
    if not os.path.exists(path):
        print(f"Saving the data to: {path}")
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