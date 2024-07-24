"""
Favour for Simon.

Loading Hannah's hourly temperature data into a UK mean temperature between
1990 and 2020.
"""

# import the required libraries
import os
import sys
import glob
import argparse
import time

import numpy as np
import pandas as pd

from load_wind_functions import load_obs_data, preprocess_temp, apply_country_mask

from functions_demand import calc_spatial_mean


# define the main function
def main():
    # sety up the start time
    start_time = time.time()

    # Parse the arguments
    parser = argparse.ArgumentParser(description="Process the temperature data.")

    # Add the arguments
    parser.add_argument(
        "--start_year",
        type=int,
        help="The start year for the temperature data.",
        required=True,
    )

    parser.add_argument(
        "--end_year",
        type=int,
        help="The end year for the temperature data.",
        required=True,
    )

    parser.add_argument(
        "--first_month",
        type=int,
        help="The first month for the temperature data.",
        required=True,
    )

    parser.add_argument(
        "--last_month",
        type=int,
        help="The last month for the temperature data.",
        required=True,
    )

    # parse the arguments
    args = parser.parse_args()

    # set the start year
    start_year = args.start_year
    end_year = args.end_year
    first_month = args.first_month
    last_month = args.last_month

    # print the args
    print(f"Start year: {start_year}")
    print(f"End year: {end_year}")
    print(f"First month: {first_month}")
    print(f"Last month: {last_month}")

    # Load the data
    ds = load_obs_data(
        last_year=end_year,
        last_month=last_month,
        first_year=start_year,
        first_month=first_month,
        parallel=False,
        bias_correct_wind=False,
        preprocess=preprocess_temp,
        daily_mean=False,
    )

    # apply the country mask
    ds = apply_country_mask(
        ds=ds,
        country="United Kingdom",
    )

    # calculate the spatial mean
    df = calc_spatial_mean(
        ds=ds,
        country="United Kingdom",
        variable="t2m",
        convert_kelv_to_cel=True,
    )

    # print ds
    print(df)

    # set up the output dirctory
    output_dir = "/storage/silver/clearheads/Ben/saved_ERA5_data"

    # set up the fname
    fname = f"ERA5_UK_mean_temp_{start_year}-{first_month}_{end_year}-{last_month}.csv"

    # set up the path
    path = os.path.join(output_dir, fname)

    # iuf the output directory does not exist, create it
    if not os.path.exists(output_dir):
        print(f"Creating the directory: {output_dir}")
        os.makedirs(output_dir)

    # if the path does not exist, save the data
    if not os.path.exists(path):
        print(f"Saving the data to: {path}")
        df.to_csv(path)

    # print the end time
    end_time = time.time()

    # print the time taken
    print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
