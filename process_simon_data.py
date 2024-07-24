"""
Process the .csv data for simon
"""

import os
import sys
import glob

import pandas as pd
import numpy as np
from tqdm import tqdm

def main():
    # Set up the forlder in which the data are stored
    saved_dir = "/storage/silver/clearheads/Ben/saved_ERA5_data"

    # set uip the test file
    test_file = "ERA5_UK_mean_temp_1990-1_1990-12.csv"

    # # read the file
    # df = pd.read_csv(os.path.join(saved_dir, test_file))

    # # print the head
    # print(df.tail())

    # # convert "Unnamed: 0" to datetime
    # df["Unnamed: 0"] = pd.to_datetime(df["Unnamed: 0"])

    # # set the index
    # df.set_index("Unnamed: 0", inplace=True)

    # # remove the name of the index
    # df.index.name = None

    # # print the head
    # print(df.tail())

    # set up an empty DataFrame
    full_df = pd.DataFrame()

    # loop over the years
    for year in tqdm(np.arange(1990, 2020 + 1)):
        # Set up the fname
        fname = f"ERA5_UK_mean_temp_{year}-1_{year}-12.csv"

        # read the file
        df = pd.read_csv(os.path.join(saved_dir, fname))
        
        # Convert "Unnamed: 0" to datetime
        df["Unnamed: 0"] = pd.to_datetime(df["Unnamed: 0"])

        # Set the index
        df.set_index("Unnamed: 0", inplace=True)

        # Remove the name of the index
        df.index.name = None

        # Append the data to the full DataFrame
        full_df = pd.concat([full_df, df], axis=0)

    # print the head
    print(full_df.head())

    # print the tail
    print(full_df.tail())

    # set up the output directory
    output_dir = "/storage/silver/clearheads/Ben/saved_ERA5_data"

    # Set up the fname
    fname = "ERA5_UK_mean_temp_hourly_1990-2020.csv"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(os.path.join(output_dir, fname)):
        print(f"Saving the data to: {os.path.join(output_dir, fname)}")
        full_df.to_csv(os.path.join(output_dir, fname))

if __name__ == "__main__":
    main()