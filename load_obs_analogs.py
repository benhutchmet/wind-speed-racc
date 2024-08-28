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

# Specific imports
from tqdm import tqdm
from iris.util import equalise_attributes

# Import specific functions
from functions_for_creating_NUTS_data import find_data_path, load_appropriate_mask, load_appropriate_data

def main():
    # Set up the hard coded paths
    analogs_path = "/home/users/pn832950/100m_wind/csv_files/s1960_lead1-30_month11_mse_df_model_matched_analogs_1960-1965.csv"
    varname = "speed100m"
    u100_name = "u100"
    v100_name = "v100"
    start_date = "1960-11-01 11:00:00"

    # Set up the parameters
    country = "United Kingdom"
    nuts_level = 0
    pop_weights = 0
    sp_weights = 0
    # wp_weights = 0 # for no location weighting
    wp_weights = 1 # for location weighting
    wp_sim = 0 # current distribution from thewindpower.net
    ons_ofs = "ons" # verify onshore first (offshore worst)
    field_str = "wp" # wind power
    cc_flag = 0 # no climate change

    # Load the analogs
    analogs = pd.read_csv(analogs_path)

    # remove the "Unnamed: 0" column
    analogs = analogs.drop(columns=["Unnamed: 0"])

    # subset to the first member
    analogs = analogs[analogs["member"] == 1]

    # extract the time values as a list
    time_values = analogs["time"].values

    # # Print the time values
    # print(time_values)

    # convert these to a list of datetime objects using cftime
    time_values = [cftime.num2date(i, "hours since 1900-01-01", "gregorian") for i in time_values]

    # # Print the time values
    # print(time_values)

    # create a list of time values starting from the start date
    time_values_fix = pd.date_range(start=start_date, periods=len(time_values), freq="D")

    # # Print the time values
    # print(time_values_fix)

    # convert this from datetime to cftime
    time_values_fix = [cftime.DatetimeGregorian(i.year, i.month, i.day, i.hour) for i in time_values_fix]

    # Define the reference date
    ref_date = cftime.DatetimeGregorian(1900, 1, 1)

    # Convert the cftime.DatetimeGregorian objects to "time since" values
    # Convert the cftime.DatetimeGregorian objects to "hours since" values
    time_values_numeric = [(i - ref_date).total_seconds() / (60 * 60) for i in time_values_fix]

    # # print the time values
    # print(time_values_numeric)

    # # Print the time values
    # print(time_values_fix)

    # sys.exit()

    # create an empty cube list
    cube_list_u100 = []
    cube_list_v100 = []

    # loop through the time values
    for time, time_fix in zip(time_values, time_values_numeric):
        # Extract the year, month and day
        year = time.year
        month = time.month
        day = time.day

        # # print the year month and day
        # print(f"{year}-{month}-{day}")

        # form the path to the file
        file_loc, file_name = find_data_path(
            year=year,
            month=month,
            field_str=varname,
            daily_flag=False,
        )

        # # print the file location and file name
        # print(f"file_loc: {file_loc}")
        # print(f"file_name: {file_name}")
        
        # form the fpath
        fpath = os.path.join(file_loc, file_name)

        # assert that the file exists
        assert os.path.exists(fpath), f"File does not exist: {fpath}"

        # load the cube
        cubes_u100 = iris.load(fpath, u100_name)
        cubes_v100 = iris.load(fpath, v100_name)

        # extract the data for the specific year, month and day
        # set up the constraint for yyyy-mm-dd
        constraint = iris.Constraint(time=lambda cell: cell.point == time)

        # extract the cube
        cube_u100 = cubes_u100.extract(constraint)[0]
        cube_v100 = cubes_v100.extract(constraint)[0]

        # reset the time coordinates as time_fix
        cube_u100.coord("time").points = [time_fix]
        cube_v100.coord("time").points = [time_fix]

        # append the cube to the cube list
        cube_list_u100.append(cube_u100)
        cube_list_v100.append(cube_v100)

    # # # print the cube list
    # print(cube_list_u100)
        
    #     # Flatten the list of lists into a single list of cubes
    # flat_list = [item for sublist in cube_list_u100 for item in sublist]

    # Create a CubeList from the list of cubes
    cube_list_u100 = iris.cube.CubeList(cube_list_u100)
    cube_list_v100 = iris.cube.CubeList(cube_list_v100)

    # # Print the cube list
    # print(f"cube_list: {cube_list}")
    # print(f"type cube_list: {type(cube_list)}")

    # equalise the attributes
    removed_attrs = equalise_attributes(cube_list_u100)
    removed_attrs = equalise_attributes(cube_list_v100)

    # print(f"cube_list: {cube_list}")
    # print(f"type cube_list: {type(cube_list)}")

    # Merge the cubes
    merged_cube_u100 = cube_list_u100.merge_cube()
    merged_cube_v100 = cube_list_v100.merge_cube()

    # Print the merged cube
    # print(merged_cube_u100)

    # # # # Print the merged cube
    # print(merged_cube_v100)

    # Load the appropriate mask for the UK
    country_mask = load_appropriate_mask(
        COUNTRY=country,
        NUTS_lev=nuts_level,
        pop_weights=pop_weights,
        sp_weights=sp_weights,
        wp_weights=wp_weights,
        WP_sim=wp_sim,
        ons_ofs=ons_ofs,
        NUTS9_country_choice=country,
    )

    # Load the clmate data and convert to a time series of area agg data

if __name__ == "__main__":
    main()