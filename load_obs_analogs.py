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
import time

# Import third-party libraries
import iris
import cftime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Specific imports
from tqdm import tqdm
from iris.util import equalise_attributes

# Import specific functions
from functions_for_creating_NUTS_data import (
    find_data_path,
    load_appropriate_mask,
    load_appropriate_data,
)


def main():
    # Set up the hard coded paths
    analogs_path = "/home/users/pn832950/100m_wind/csv_files/s1960_lead1-30_month11_mse_df_model_matched_analogs_1960-1965.csv"
    varname = "speed100m"
    u100_name = "u100"
    v100_name = "v100"
    start_date = "1960-11-01 00:00:00"

    # Set up the parameters
    country = "United Kingdom"
    nuts_level = 0
    pop_weights = 0
    sp_weights = 0
    # wp_weights = 0 # for no location weighting
    wp_weights = 1  # for location weighting
    wp_sim = 0  # current distribution from thewindpower.net
    ons_ofs = "ons"  # verify onshore first (offshore worst)
    field_str = "wp"  # wind power
    cc_flag = 0  # no climate change

    # if wp_weights == 0 then raise value error
    if wp_weights == 0:
        print("wp_weights is zero, code only works for wp_weights = 1")
        raise ValueError("wp_weights must be 1")

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
    time_values = [
        cftime.num2date(i, "hours since 1900-01-01", "gregorian") for i in time_values
    ]

    # # Print the time values
    # print(time_values)

    # sys.exit()

    # create a list of time values starting from the start date
    time_values_fix = pd.date_range(
        start=start_date, periods=len(time_values) * 24, freq="h"
    )

    # Print the time values
    print(time_values_fix)
    print(np.shape(time_values_fix))

    # convert this from datetime to cftime
    time_values_fix = [
        cftime.DatetimeGregorian(i.year, i.month, i.day, i.hour)
        for i in time_values_fix
    ]

    # Define the reference date
    ref_date = cftime.DatetimeGregorian(1900, 1, 1)

    # Convert the cftime.DatetimeGregorian objects to "time since" values
    # Convert the cftime.DatetimeGregorian objects to "hours since" values
    time_values_numeric = [
        (i - ref_date).total_seconds() / (60 * 60) for i in time_values_fix
    ]

    # # print the time values
    # print(time_values_numeric)

    # # Print the time values
    # print(time_values_fix)

    # # print the shape of the time values
    # print(np.shape(time_values_fix))
    # print(np.shape(time_values_numeric))

    # sys.exit()

    # create an empty cube list
    cube_list_u100 = []
    cube_list_v100 = []

    # loop through the time values
    for i, time_this in enumerate(time_values):
        # Extract the year, month and day
        year = time_this.year
        month = time_this.month
        day = time_this.day

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

        # # Create a cftime.DatetimeGregorian object with the specific year, month, and day
        # specific_time = cftime.DatetimeGregorian(year, month, day)

        # # print the specific time
        # print(f"specific_time: {specific_time}")

        # Set up the constraint for the specific year, month, and day
        constraint = iris.Constraint(
            time=lambda cell: cell.point.year == year
            and cell.point.month == month
            and cell.point.day == day
        )

        # extract the cube
        cube_u100 = cubes_u100.extract(constraint)[0]
        cube_v100 = cubes_v100.extract(constraint)[0]

        # # print the cube
        # print(cube_u100)

        # sys.exit()

        # # print time values numeric
        # print(np.shape(time_values_numeric))

        # # print the time fix
        # print(time_values_numeric[i * 24 : (i + 1) * 24])
        # print(time_values_fix[i * 24 : (i + 1) * 24])

        # reset the time coordinates as time_fix
        cube_u100.coord("time").points = time_values_numeric[i * 24 : (i + 1) * 24]
        cube_v100.coord("time").points = time_values_numeric[i * 24 : (i + 1) * 24]

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

    # print the cube list
    print(f"cube_list_u100: {cube_list_u100}")


    # # Print the cube list
    # print(f"cube_list: {cube_list}")
    # print(f"type cube_list: {type(cube_list)}")

    # equalise the attributes
    removed_attrs = equalise_attributes(cube_list_u100)
    removed_attrs = equalise_attributes(cube_list_v100)

    print(f"cube_list: {cube_list_u100}")
    print(f"type cube_list: {type(cube_list_u100)}")

    # sys.exit()

    # Merge the cubes
    # Concatenate the cubes
    concatenated_cube_u100 = cube_list_u100.concatenate_cube()
    concatenated_cube_v100 = cube_list_v100.concatenate_cube()

    # print the concatenated cube
    print(f"concatenated_cube_u100: {concatenated_cube_u100}")
    print(f"concatenated_cube_v100: {concatenated_cube_v100}")

    # sys.exit()

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

    # Extract the lats and lons as an array of zeros
    lats = np.zeros(np.shape(country_mask)[0])
    lons = np.zeros(np.shape(country_mask)[1])

    # Create the wind power data
    pc_p = []
    pc_w = []
    # firstly load the power curve
    if ons_ofs == "ons":
        with open("/home/users/pn832950/100m_wind/power_curve/power_onshore.csv") as f:
            for line in f:
                columns = line.split()
                # print columns[0]
                pc_p.append(float(columns[1][0:8]))  # get power curve output (CF)
                pc_w.append(float(columns[0][0:8]))  # get power curve output (CF)
    elif ons_ofs == "ofs":
        with open("/home/users/pn832950/100m_wind/power_curve/power_offshore.csv") as f:
            for line in f:
                columns = line.split()
                # print columns[0]
                pc_p.append(float(columns[1][0:8]))  # get power curve output (CF)
                pc_w.append(float(columns[0][0:8]))  # get power curve output (CF)

    # interpolate appropriately
    power_curve_w = np.array(pc_w)
    power_curve_p = np.array(pc_p)
    pc_winds = np.linspace(0, 50, 501)  # make it finer resolution
    pc_power = np.interp(pc_winds, power_curve_w, power_curve_p)

    # Set up the mask matrix reshape - load all the data for the correct
    # region and then weight at the end
    MASK_MATRIX_RESHAPE = np.zeros_like(country_mask) + 1

    # load the country weather data and apply the mask
    data_u100 = concatenated_cube_u100.data
    data_v100 = concatenated_cube_v100.data

    # process the si100
    si100 = np.sqrt(data_u100**2 + data_v100**2)

    # Apply bias correction as ERA5 winds are low
    bc_fpath = "/home/users/pn832950/UREAD_energy_models_demo_scripts/ERA5_speed100m_mean_factor_v16_hourly.npy"

    # load the correction factors
    correction_factors = np.load(bc_fpath)
    data_corrected = np.zeros(np.shape(data_u100))
    for i in range(0, np.shape(data_u100)[0]):
        data_corrected[i, :, :] = si100[i, :, :] + correction_factors

    if ons_ofs == "ons":
        data = data_corrected * (71.0 / 100.0) ** (
            1.0 / 7.0
        )  # average heights from UK windpower.net 2021 onshore wind farms
    elif ons_ofs == "ofs":
        data = data_corrected * (92.0 / 100.0) ** (
            1.0 / 7.0
        )  # average heights from UK windpower.net 2021 offshore wind farms
    else:
        raise ValueError("ons_ofs must be ons or ofs")

    # set up the matrix to be nan where there are zeros
    MASK_MATRIX_RESHAPE[MASK_MATRIX_RESHAPE == 0.] = np.nan

    # apply the mask
    country_masked_data = np.zeros(np.shape(data))

    # loop through the data
    for i in range(0, np.shape(data)[0]):
        country_masked_data[i, :, :] = data[i, :, :] * MASK_MATRIX_RESHAPE

    # get the number of hours
    maxqhr = np.shape(country_masked_data)[0]

    # if the wp_weights are zero
    if wp_weights == 0:
        print("wp_weights is zero")
        # set the total MW as the country mask
        total_MW = np.zeros_like(country_mask) + 1
    # if the wp_weights are greater than zero
    elif wp_weights == 1:
        print("wp_weights is one")
        # set the total MW as the country mask
        total_MW = country_mask
    else:
        raise ValueError("wp_weights must be 0 or 1")
    
    # create an array to fill with capacity factors
    cf = np.zeros_like(country_masked_data)
    for qhr in range(0, maxqhr):
        # Extract the wind speed for the hour
        VAR_QHR = country_masked_data[qhr, :, :]

        # apply the masks
        Reshaped_speed = np.reshape(
            VAR_QHR, [np.shape(VAR_QHR)[0] * np.shape(VAR_QHR)[1]]
        )

        test2 = np.digitize(
            Reshaped_speed, pc_winds, right=False
        )  # indexing starts from 1 so needs -1: 0 in the next bit to start from the lowest bin.
        test2[test2 == len(pc_winds)] = (
            500  # make sure the bins don't go off the end (power is zero by then anyway)
        )
        p_hh_temp = 0.5 * (pc_power[test2 - 1] + pc_power[test2])
        capacity_factor_of_pannel = np.reshape(
            p_hh_temp, [np.shape(VAR_QHR)[0], np.shape(VAR_QHR)[1]]
        )
        capacity_factor_of_pannel[capacity_factor_of_pannel <= 0.006] = 0.0
        cf[qhr, :, :] = capacity_factor_of_pannel * total_MW

    # set zeros to NaNs where the mask is zero
    cf[cf == 0.0] = np.nan


    # if no weights are used anywhere:
    if np.sum(pop_weights + sp_weights + wp_weights) == 0:
        print("No weights used")
        country_aggregate = np.nanmean(np.nanmean(cf, axis=2), axis=1)
        # if weights are usef somewhere
    elif np.sum(pop_weights + sp_weights + wp_weights) > 0:
        print("One of: pop_weights, sp_weights, wp_weights is greater than zero")
        country_aggregate = np.nansum(np.nansum(cf, axis=2), axis=1) / np.nansum(
            country_mask
        )
    else:
        print("Problem with weights")

    # make sure any zeros are set to NaN
    country_aggregate[country_aggregate == 0.0] = np.nan

    # print the country aggregate
    print(country_aggregate)

    # set up a DataFrame
    df = pd.DataFrame(country_aggregate, index=time_values_fix, columns=["UK_wp"])

    # print the DataFrame
    print(df)

    # plot the DataFrame
    df.plot()

    # include a title
    plt.title("UK Wind Power Capacity Factor for member 1, init 1960-11-01")

    # set up the time
    save_time = time.strftime("%Y-%m-%d_%H:%M:%S")

    # save the plot to a file
    plt.savefig(f"/home/users/pn832950/100m_wind/plots/UK_wp_{save_time}_wp_weights_{wp_weights}_country_{country}_ons_ofs_{ons_ofs}_field_str_{field_str}_cc_flag_{cc_flag}_wp_sim_{wp_sim}.png")

if __name__ == "__main__":
    main()
