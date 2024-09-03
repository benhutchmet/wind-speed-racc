#!/usr/bin/env python

"""
verify_obs_analogs.py
===================

Verifying the observed analogs method for getting reasonable wind power
capacity factors from DePreSys. First match each day of the decadal prediction
to a day in the observed data based on the circulation pattern over the 
North Atlantic/European region. Then extract all of these dates for a full 
season (e.g. NDJFM) and calculate the capacity factor for each day.

Here we are verifying that the results produced by this are broadly in line
with that from CLEARHEADS. CLEARHEADS uses the same ERA5 reanalysis data and 
the same models to calculate wind and solar capacity factors.

We are not necessarily expecting the results to be identical, but we are hoping
that in the ensemble mean, they broadly agree.

The first step is loading the analogs data into a dataframe. And loading the
observed data into a second dataframe. Dates might be tricky here, as DePreSys
works on a 360 day calendar.

Usage:
------

$ python verify_obs_analogs.py

"""

# Import local libraries
import os
import sys
import time
import argparse

# Import third-party libraries
import iris
import cftime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the main function
def main():
    # Start the timer
    start_time = time.time()

    # Set up the hard-coded variables
    dfs_dir = "/storage/silver/clearheads/Ben/csv_files/wind_power_analogs"
    ons_ofs = "ons"
    wp_weights = "1"
    country = "United Kingdom"


    # Set up the list of members 1-10
    members = np.arange(1, 11)

    # set uyp the testpath
    testpath = os.path.join(dfs_dir, f"UK_wp_1962_month_11-3_member_4_wp_weights_1_country_United Kingdom_ons_ofs_ons_field_str_wp_cc_flag_0_wp_sim_0.csv")

    # assert that the file exists
    assert os.path.exists(testpath), f"File does not exist: {testpath}"

    # Load the data
    df = pd.read_csv(testpath)

    # remove the "Unnamed: 0" column
    df = df.drop(columns=["Unnamed: 0"])

    # Print the head
    print(df.head())

    # print the tail
    print(df.tail())

    # print the shape of the dataframe
    print(df.shape)


    sys.exit()

    # loop over the members
    for m in members:
        # Set up the fname
        fpath = os.path.join(dfs_dir, f"UK_wp_1960-11-01 00:00:00_month_11-3_member_{m}_wp_weights_{wp_weights}_country_{country}_ons_ofs_{ons_ofs}_field_str_wp_cc_flag_0_wp_sim_0.csv")

        # assert that the file exists
        assert os.path.exists(fpath), f"File does not exist: {fpath}"

        # Load the data
        df = pd.read_csv(fpath)

        # remove the "Unnamed: 0" column
        df = df.drop(columns=["Unnamed: 0"])

        # Print the head
        print(df.head())

        # print the tail
        print(df.tail())

        # print the shape of the dataframe
        print(df.shape)

    # End the timer
    end_time = time.time()

    # Print the time taken
    print('Time taken: ', end_time - start_time)

    return None

# Run the main function
if __name__ == '__main__':
    main()
