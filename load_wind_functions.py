"""
Functions for loading in the wind data from CLEARHEADS and S2S4E.

Thanks to Hannah for downloading all of the data!
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import xarray as xr
import iris
import shapely.geometry as sgeom
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import regionmask
from tqdm import tqdm

# Specific imports
from ncdata.iris_xarray import cubes_to_xarray, cubes_from_xarray


# Define a function to load the 100m wind speed data from the CLEARHEADS and S2S4E directories
# S2S4E - ERA5_1hr_2020_12_DET.nc
# CLEARHEADS - ERA5_1hr_1978_12_DET.nc
def load_wind_data(
    last_year: int,
    last_month: int = 12,
    first_year: int = 1950,
    first_month: int = 1,
    S2S4E_dir: str = "/storage/silver/S2S4E/energymet/ERA5/native_grid_hourly/",
    CLEARHEADS_dir: str = "/storage/silver/clearheads/Data/ERA5_data/native_grid/T2m_U100m_V100m_MSLP/",
    engine: str = "netcdf4",
    parallel: bool = True,
    bias_correct: bool = True,
):
    """
    Load the 100m wind speed data from the CLEARHEADS and S2S4E directories.

    Parameters
    ----------

    last_year : int
        The last year of the data to be loaded.

    last_month : int
        The last month of the data to be loaded.

    first_year : int
        The first year of the data to be loaded.

    first_month : int
        The first month of the data to be loaded.

    S2S4E_dir : str
        The directory containing the S2S4E data.

    CLEARHEADS_dir : str
        The directory containing the CLEARHEADS data.

    engine : str
        The engine to use for loading the data.

    parallel : bool
        Whether to use parallel loading.

    bias_correct : bool
        Whether to use Hannah's bias correction for onshore and offshore wind speeds.

    Returns
    -------
    ds : xarray.Dataset
        The 100m and 10m wind speed data.
    """
    # Create an empty list to store the data
    ERA5_files = []

    # create a list of the files to load based on the years and months provided
    for year in range(first_year, last_year + 1):
        if year == last_year:
            for month in range(1, last_month + 1):
                if month < 10:
                    month = f"0{month}"
                else:
                    month = f"{month}"
                # Choose the directory based on the year
                directory = CLEARHEADS_dir if year < 1979 else S2S4E_dir
                # glob the files in the chosen directory
                for file in glob.glob(directory + f"ERA5_1hr_{year}_{month}_DET.nc"):
                    ERA5_files.append(file)
        else:
            for month in range(1, 13):
                if month < 10:
                    month = f"0{month}"
                else:
                    month = f"{month}"
                # Choose the directory based on the year
                directory = CLEARHEADS_dir if year < 1979 else S2S4E_dir
                # glob the files in the chosen directory
                for file in glob.glob(directory + f"ERA5_1hr_{year}_{month}_DET.nc"):
                    ERA5_files.append(file)

    # Print the length of the list
    print("Number of files: ", len(ERA5_files))

    # Load the data
    ds = xr.open_mfdataset(
        ERA5_files,
        combine="by_coords",
        preprocess=lambda ds: preprocess(
            ds,
            bias_correct=bias_correct,
        ),
        engine=engine,
        parallel=parallel,
        coords="minimal",
        data_vars="minimal",
        compat="override",
    ).squeeze()

    # chunk the data
    ds = ds.chunk({"time": "auto", "latitude": "auto", "longitude": "auto"})

    # Take a daily mean
    ds = ds.resample(time="D").mean()

    return ds


# Define the function for preprocessing the data
def preprocess(
    ds: xr.Dataset,
    u100_name="u100",
    v100_name="v100",
    si100_name="si100",
    u10_name="u10",
    v10_name="v10",
    si10_name="si10",
    t2m_name="t2m",
    msl_name="msl",
    bias_correct: bool = False,
    bias_correct_file: str = "/home/users/pn832950/UREAD_energy_models_demo_scripts/ERA5_turbine_array_total_BC_v16_hourly.nc",
) -> xr.Dataset:
    """
    Preprocess the data.

    Parameters
    ----------

    ds : xarray.Dataset
        The dataset to be preprocessed.

    u100_name : str
        The name of the zonal wind component.

    v100_name : str
        The name of the meridional wind component.

    si100_name : str
        The name of the wind speed at 100m to be output.

    u10_name : str
        The name of the zonal wind component at 10m.

    v10_name : str
        The name of the meridional wind component at 10m.

    si10_name : str
        The name of the wind speed at 10m to be output.

    t2m_name : str
        The name of the 2m temperature.

    msl_name : str
        The name of the mean sea level pressure.

    bias_correct : bool
        Whether or not to apply Hannah's bias correction.

    bias_correct_file : str
        The file containing the bias correction data.

    Returns
    -------

    ds : xarray.Dataset

    """

    # Calculate the wind speed at 100m
    ds[si100_name] = np.sqrt(ds[u100_name] ** 2 + ds[v100_name] ** 2)

    # If bias correction is required
    if bias_correct:
        # Load the bias correction data
        bc = xr.open_dataset(bias_correct_file)

        # Convert the DataArrays to numpy arrays
        si100_name_np = ds[si100_name].values
        bc_totals_np = bc["totals"].values

        # create a new numpy array to store the result
        si100_bc_np = np.zeros(np.shape(si100_name_np))

        # Perform the addition
        for i in tqdm(
            range(np.shape(si100_name_np)[0]), desc="Applying bias correction"
        ):
            si100_bc_np[i, :, :] = si100_name_np[i, :, :] + bc_totals_np

        # Convert the result back to an xarray DataArray
        si100_bc = xr.DataArray(
            data=si100_bc_np,
            dims=ds[si100_name].dims,
            coords=ds[si100_name].coords,
        )

        # Add the new DataArray to the dataset
        ds = ds.assign(si100_bc=si100_bc)

        # Set up the variables
        # Power law exponent from UK windpower.net 2021 onshore wind farm heights
        ds["si100_ons"] = ds["si100_bc"] * (71.0 / 100.0) ** (1.0 / 7.0)

        # Same but for offshore
        # Average height of offshore wind farms in the UK 2021 from UK windpower.net
        ds["si100_ofs"] = ds["si100_bc"] * (92.0 / 100.0) ** (1.0 / 7.0)

        # Drop si100_name  in favour of si100_ons and si100_ofs
        ds = ds.drop_vars([si100_name])

    # Calculate the wind speed at 10m
    ds[si10_name] = np.sqrt(ds[u10_name] ** 2 + ds[v10_name] ** 2)

    # Drop the other variables
    ds = ds.drop_vars([u100_name, v100_name, u10_name, v10_name, t2m_name, msl_name])

    return ds


# Define a function which applies a mask to the data
def apply_country_mask(
    ds: xr.Dataset,
    country: str,
    pop_weights: int = 0,
) -> xr.Dataset:
    """
    Apply a mask to the data for a specific country.

    Parameters
    ----------

    ds : xarray.Dataset
        The dataset to be masked.

    country : str
        The country to be masked.

    pop_weights : int
        The population weights to be applied.

    Returns
    -------

    ds : xarray.Dataset
        The masked dataset.

    """

    # Identify an appropriate shapefile for the country
    countries_shp = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )

    # Find the country
    country_shp = None
    for country_shp in shpreader.Reader(countries_shp).records():
        if country_shp.attributes["NAME_LONG"] == country:
            print("Found the country!")

            # Load using geopandas
            country_shp_gp = gpd.read_file(countries_shp)

            # Filter the dataframe to only include the row for the UK
            country_shp_gp = country_shp_gp[country_shp_gp["NAME_LONG"] == country]

    # Ensure that the 'numbers' column exists in the geodataframe
    if "numbers" not in country_shp_gp.columns:
        country_shp_gp["numbers"] = np.array([1])

    # Create the mask using the regionmask and geopandas
    country_mask_poly = regionmask.from_geopandas(
        country_shp_gp,
        names="NAME_LONG",
        abbrevs="ABBREV",
        numbers="numbers",
    )

    # Print the mask
    print(f"Country mask: {country_mask_poly}")

    # Select the first timestep of the data
    country_mask = country_mask_poly.mask(
        ds.isel(time=0), lon_name="longitude", lat_name="latitude"
    )

    if country == "United Kingdom":
        print("Masking out Northern Ireland.")
        # If the country is the UK then mask out Northern Ireland
        country_mask = country_mask.where(
            ~(
                (country_mask.latitude < 55.3)
                & (country_mask.latitude > 54.0)
                & (country_mask.longitude < -5.0)
            ),
            other=np.nan,
        )

    # print the country mask
    print(f"Country mask: {country_mask}")

    # Extract the lat and lon values
    mask_lats = country_mask.latitude.values
    mask_lons = country_mask.longitude.values

    ID_REGION = 1  # only 1 region in this instance

    # Select mask for specific region
    sel_country_mask = country_mask.where(country_mask == ID_REGION).values

    # Select the smallest box containing the entire mask
    # the coordinate points where the mask is not NaN
    # id_lon = mask_lons[np.where(~np.all(np.isnan(sel_country_mask), axis=0))]
    # id_lat = mask_lats[np.where(~np.all(np.isnan(sel_country_mask), axis=1))]

    # # print the first and last values of the lat and lon
    # print(f"Latitude: {id_lat[0]} to {id_lat[-1]}")
    # print(f"Longitude: {id_lon[0]} to {id_lon[-1]}")

    # # print the shape of ds
    # print("ds shape:", ds.dims)

    # # print ds
    # print(ds)

    # Select the data within the mask
    out_ds = ds.compute().where(country_mask == ID_REGION)

    # print the data
    print(out_ds)

    return out_ds


# define a function which saves the data
def save_wind_data(
    ds: xr.Dataset,
    output_dir: str,
    fname: str,
) -> None:
    """
    Save the wind data to a netCDF file.

    Parameters
    ----------

    ds : xarray.Dataset
        The dataset to be saved.

    output_dir : str
        The directory to save the file in.

    fname : str
        The name of the file to be saved.

    Returns
    -------

    None
    """

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the data
    ds.to_netcdf(os.path.join(output_dir, fname))

    return None


# Define a function to create the wind power data
# FIXME: Find the correct onshore and offshore power curves here
# FIXME: Find the correct installed capacity data here
def create_wind_power_data(
    ds: xr.Dataset,
    country: str = "United_Kingdom",
    ons_ofs: str = "ons",
    bc_si100_name: str = "si100_bc",
    min_cf: float = 0.0006,
    onshore_curve_file: str = "/home/users/pn832950/100m_wind/power_curve/powercurve.csv",
    offshore_curve_file: str = "/home/users/pn832950/100m_wind/power_curve/powercurve.csv",
    installed_capacities_dir: str = "/storage/silver/S2S4E/zd907959/MERRA2_wind_model/python_version/",
) -> xr.Dataset:
    """
    Loads in datasets containing the 100m wind speed data from ERA5 (in the first
    version) and converts this into an array of wind power capacity factor.

    Parameters
    ----------

    ds : xarray.Dataset
        The dataset containing the 100m wind speed data.

    ons_ofs : str
        The type of wind farm to be considered (either onshore or offshore).

    bc_si100_name : str
        The name of the bias corrected 100m wind speed data.
        Default is "si100_bc".

    min_cf : float
        The minimum capacity factor.
        Default is 0.0006.

    onshore_curve_file : str
        The file containing the onshore power curve data.
        Default is "/home/users/pn832950/100m_wind/power_curve/powercurve.csv".
        Random power curve from S2S4E.

    offshore_curve_file : str
        The file containing the offshore power curve data.
        Default is "/home/users/pn832950/100m_wind/power_curve/powercurve.csv".
        Random power curve from S2S4E.

    installed_capacities_idr : str
        The file containing the installed capacities data.
        Default is "/storage/silver/S2S4E/zd907959/MERRA2_wind_model/python_version/".
        Random installed capacities data from S2S4E.

    Returns
    -------

    ds : xarray.Dataset
        The dataset containing the wind power capacity factor data.

    """

    # TODO: Get the correct installed capacities
    # Form the filepath
    installed_capacities_file = os.path.join(
        installed_capacities_dir, f"{country}windfarm_dist.nc"
    )

    # glob the file
    installed_capacities_files = glob.glob(installed_capacities_file)

    # assert that the file exists
    assert len(installed_capacities_files) == 1, "Installed capacities file not found."

    # Load the installed capacities data
    installed_capacities = xr.open_dataset(installed_capacities_files[0])

    ic_lat = installed_capacities["lat"].values
    ic_lon = installed_capacities["lon"].values

    ds_lat = ds["latitude"].values
    ds_lon = ds["longitude"].values

    # if the lats and lons are not the same, interpolate the installed capacities
    if not np.array_equal(ic_lat, ds_lat) or not np.array_equal(ic_lon, ds_lon):
        print("Lats and lons are not the same.")
        print("Interpolating installed capacities to the same grid as the wind speed data.")
        print("First lat and lon of installed capacities:", ic_lat[0], ic_lon[0])
        print("First lat and lon of wind speed data:", ds_lat[0], ds_lon[0])
        print("Last lat and lon of installed capacities:", ic_lat[-1], ic_lon[-1])
        print("Last lat and lon of wind speed data:", ds_lat[-1], ds_lon[-1])
        print("Regridding the installed capacities data.")

        # convert ds from xarray object to iris object
        ds_cube = cubes_from_xarray(ds)

        # convert installed capacities from xarray object to iris object
        ic_cube = cubes_from_xarray(installed_capacities)

        # extract the bc_si100_name cube
        bc_si100_cube = ds_cube.extract(bc_si100_name)[0]

        # extract the totals cube
        ic_cube = ic_cube.extract("totals")[0]

        # if the coords
        if ic_cube.coords != bc_si100_cube.coords:
            # rename lat and lon to latitude and longitude
            ic_cube.coord("lat").rename("latitude")
            ic_cube.coord("lon").rename("longitude")

        # Ensure the units of the coordinates match
        ic_cube.coord('latitude').units = bc_si100_cube.coord('latitude').units
        ic_cube.coord('longitude').units = bc_si100_cube.coord('longitude').units

        # Ensure the attributes of the coordinates match
        ic_cube.coord('latitude').attributes = bc_si100_cube.coord('latitude').attributes
        ic_cube.coord('longitude').attributes = bc_si100_cube.coord('longitude').attributes

        # print the types of these cubes
        print(f"bc_si100_cube: {bc_si100_cube}")
        print(f"ic_cube: {ic_cube}")
        print(f"bc_si100_cube type: {type(bc_si100_cube)}")
        print(f"ic_cube type: {type(ic_cube)}")

        # regrid the installed capacities to the same grid as the wind speed data
        ic_cube_regrid = ic_cube.regrid(bc_si100_cube, iris.analysis.Linear())
        
        # print the regridded cube
        print(f"Regridded cube: {ic_cube_regrid}")

    # Extract the values
    # Flip to get the correct order of lat lon
    total_MW = np.flip(installed_capacities["totals"].values) / 1000

    # print the shape of the total MW
    print("Total MW shape:", total_MW.shape)

    # Depending on the type of wind farm, load in the appropriate power curve
    if ons_ofs == "ons":
        print("Loading in the onshore power curve.")

        # Load in the onshore power curve
        power_curve = pd.read_csv(onshore_curve_file, header=None)

    elif ons_ofs == "ofs":
        print("Loading in the offshore power curve.")

        # Load in the offshore power curve
        power_curve = pd.read_csv(offshore_curve_file, header=None)
    else:
        print("Invalid wind farm type. Please choose either onshore or offshore.")
        sys.exit()

    # print the head of the dataframe
    print(power_curve.head())

    # Add column names to the power curve
    power_curve.columns = ["Wind speed (m/s)", "Power (W)"]

    # Print the power curve head
    print(power_curve.head())

    # Generate an array for wind speeds
    pc_winds = np.linspace(0, 50, 501)

    # Using np.interp, find the power output for each wind speed
    pc_power = np.interp(
        pc_winds, power_curve["Wind speed (m/s)"], power_curve["Power (W)"]
    )

    # Add these to a new dataframe
    pc_df = pd.DataFrame({"Wind speed (m/s)": pc_winds, "Power (W)": pc_power})

    print("Power curve dataframe:")
    print("----------------------")

    # Print the shape of the dataframe
    print(pc_df.shape)

    # print the head of the dataframe
    print(pc_df.head())

    # print the power curve dataframe
    print(pc_df.tail())

    # Extract the wind speed data from the dataset
    wind_speed = ds[bc_si100_name].values

    # Extract the values of the wind speed
    wind_speed_vals = ds[f"si100_{ons_ofs}"].values

    # Create an empty array to store the power data
    cfs = np.zeros(np.shape(wind_speed))

    # print the shape of the wind speed values
    print("Wind speed values shape:", np.shape(wind_speed_vals))

    # print the shape of the cfs arr
    print("CFs shape:", np.shape(cfs))

    # Extract total MW as the array values of the installed capacities regrid
    total_MW = ic_cube_regrid.data

    # print the shape of the total MW
    print("Total MW shape:", total_MW.shape)

    # print pc_df again
    print("Power curve dataframe head:", pc_df.head())

    # Loop over the time axis
    for i in tqdm(range(0, np.shape(wind_speed)[0]), desc="Creating wind power data"):
        # Extract the wind speed data for the current timestep
        wind_speed_vals_i = wind_speed_vals[i, :, :]

        # print the shape of the wind speed values
        print("Wind speed values shape:", np.shape(wind_speed_vals_i))

        # Set any NaN values to zero
        wind_speed_vals_i[np.isnan(wind_speed_vals_i)] = 0.0

        # reshape into a 1D array
        reshaped_wind_speed_vals = np.reshape(
            wind_speed_vals_i,
            [np.shape(wind_speed_vals_i)[0] * np.shape(wind_speed_vals_i)[1]],
        )

        # print the reshaped wind speed values
        print("Reshaped wind speed values:", reshaped_wind_speed_vals)

        # print the shape of the reshaped wind speed values
        print("Reshaped wind speed values shape:", np.shape(reshaped_wind_speed_vals))

        # print pc_df["Wind speed (m/s)"]
        print("Power curve wind speed values:", pc_df["Wind speed (m/s)"])

        # Categorise each wind speed value into a power output
        cfs_i = np.digitize(
            reshaped_wind_speed_vals, pc_df["Wind speed (m/s)"], right=False
        )

        # print the power value for the first index in cfs_i
        print("Power value for first index in cfs_i:", pc_df["Power (W)"][cfs_i[0]])

        # Print the shape of the cfs_i array
        print("CFs_i shape:", np.shape(cfs_i))

        print("CFs_i:", cfs_i)

        # Make sure the bins don't go out of range
        cfs_i[cfs_i == len(pc_df)] = len(pc_df) - 1

        # print the shape of the cfs_i array
        print("CFs_i shape:", np.shape(cfs_i))

        # print the cfs_i array
        print("CFs_i:", cfs_i)

        # convert pc_df["Power (W)"] to a numpy array of values
        pc_power_vals = pc_df["Power (W)"].values

        # print the things wew are subseting in p_bins
        print("pc_power_vals[cfs_i]:", pc_power_vals[cfs_i])
        print("pc_power_vals[cfs_i - 1]:", pc_power_vals[cfs_i - 1])

        # Calculate the average power output for each bin
        p_bins = 0.5 * (pc_power_vals[cfs_i] + pc_power_vals[cfs_i - 1])

        # print the shape of the power bins
        print("Power bins shape:", np.shape(p_bins))

        # Reshape the power output array
        cfs_i = np.reshape(
            p_bins, [np.shape(wind_speed_vals_i)[0], np.shape(wind_speed_vals_i)[1]]
        )

        # Set any values below the minimum capacity factor to the minimum capacity factor
        cfs_i[cfs_i < min_cf] = 0.0

        # Multiply by the installed capacity in MW
        cfs[i, :, :] = cfs_i * total_MW

    return cfs


# define the main function
def main():
    # Set up the parameters
    # Just load in a single month of data in this test case
    last_year = 1950
    last_month = 1

    # load the wind data
    ds = load_wind_data(
        last_year=last_year,
        last_month=last_month,
    )

    # print the data
    print(ds)

    # Apply the mask
    ds = apply_country_mask(
        ds=ds,
        country="United Kingdom",
    )

    # Create the wind power data
    cfs = create_wind_power_data(
        ds=ds,
        ons_ofs="ons",
    )

    # print the shape of the cfs
    print("CFs shape:", np.shape(cfs))

    # extract the first time step
    cfs_i = cfs[0, :, :]

    # Convert the array to a pandas dataframe
    cfs_df = pd.DataFrame(cfs_i)

    # Save the dataframe to a csv file
    cfs_df.to_csv(f"/home/users/pn832950/100m_wind/csv_files/UK_wind_power_data_{last_year}_{last_month}.csv")

    # print the data
    print("-------------------")
    print(ds)
    print("-------------------")

    # # print that we are exiting the function
    print("Exiting the function.")
    sys.exit()

    # # # Set the output directory
    # output_dir = "/storage/silver/clearheads/Ben/saved_ERA5_data"

    # # # Set the filename
    # fname = f"ERA5_100m_10m_wind_speed_daily_1950_01-{last_year}_{last_month}.nc"

    # # # Save the data
    # save_wind_data(
    #     ds=ds,
    #     output_dir=output_dir,
    #     fname=fname,
    # )


if __name__ == "__main__":
    main()
