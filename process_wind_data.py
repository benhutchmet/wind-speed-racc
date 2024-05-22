"""
process_wind_data.py
====================

Script for processing the hourly wind data into daily wind power output
for a given country and given year.

Usage:
------

    python process_wind_data.py <country> <year>

Example:
--------

    python process_wind_data.py United_Kingdom 2015

Inputs:
-------

    - country: str
        Name of the country for which the data is to be processed.
        The country name should be in the format of the country name
        as it appears in the data files.

    - year: int
        Year for which the data is to be processed.

Outputs:
--------

    - A csv file containing the daily wind power output for the given
      country and year. Saved to /storage/silver/clearheads/Ben/saved_ERA5_data
      directory.
    
"""

# Imports
import os
import sys
import glob
import argparse

# Import the functions from the load_wind_functions.py file
import load_wind_functions as lwf

# Define the main function
def main():
    """
    Main function for processing the hourly wind data into daily wind power output
    for a given country and given year.

    Parameters:
    -----------

        - None

    Returns:
    --------

        - None
    """

    # Set up the hard-coded variables
    first_month = 1
    last_month = 12
    ons_ofs = "ons" # For now we don't have correct power curves
    output_dir = "/storage/silver/clearheads/Ben/csv_files/wind_power/"

    # Parse the CLIs
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument("country", help="Name of the country for which the data is to be processed.", type=str)
    parser.add_argument("year", help="Year for which the data is to be processed.", type=int)

    # Print the year and country we are processing
    args = parser.parse_args()

    # Get the country and year
    print(f"Processing data for {args.country} in {args.year}.")

    # Load the wind data for the given year
    ds = lwf.load_wind_data(
        last_year=args.year,
        last_month=last_month,
        first_year=args.year,
        first_month=first_month,
    )

    # Apply the country mask
    ds = lwf.apply_country_mask(
        ds=ds,
        country=args.country,
    )

    # Create the wind power data
    ds_power = lwf.create_wind_power_data(
        ds=ds,
        ons_ofs=ons_ofs,
    )

    # if the country name contains spaces,
    # replace them with underscores
    if " " in args.country:
        args.country = args.country.replace(" ", "_")
        print(f"Country name has been replaced with {args.country}.")

    # Form the dataframe
    df = lwf.form_wind_power_dataframe(
        cfs=ds_power,
        ds=ds,
        country_name=args.country,
    )

    # Save the dataframe to a csv file
    lwf.save_wind_power_data(
        cfs_df=df,
        output_dir=output_dir,
        country_name=args.country,
        first_year=args.year,
        first_month=first_month,
        last_year=args.year,
        last_month=last_month,
        ons_ofs=ons_ofs,
    )

    # Print that the data has been saved
    print(f"Data has been saved to {output_dir}.")
    print("Process complete.")

    return None

# Call the main function
if __name__ == "__main__":
    main()