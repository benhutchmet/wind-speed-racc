import cdsapi
import os
import argparse


def download_ERA5_to_RACC(
    year: int,
    month: int,
) -> None:
    """
    Download ERA5 data for a given year and month to be used in RACC.
    
    Parameters
    ----------
    
    year: int
        The year to download data for.
    
    month: int
        The month to download data for.

    Returns
    -------

    None

    """

    m = str(month).zfill(2) # make sure it is 01, 02 etc

    if m in ['04','06','09','11']:
        days = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']
    elif m in ['01','03','05','07','08','10','12']:
        days = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
    else:
        if year in [1940,1944,1948,1952,1956,1960,1964,1968,1972,1976,1980,1984,1988,1992,1996,2000,2004,2008,2012,2016,2020,2024]:
            days = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29']
        else:
            days = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28']

    #print(str(YEAR))
    #print(str(MONTH))
    y = str(year)

    client = cdsapi.Client()
    
    dataset = 'reanalysis-era5-single-levels'
    request = {
        'product_type': ['reanalysis'],
        'variable': ['100m_u_component_of_wind', '100m_v_component_of_wind', '2m_temperature', 'mean_sea_level_pressure', 'surface_net_solar_radiation'],
        'year': [y],
        'month': [m],
        'day': days,
        'area': [72,-40,34,35], # N/W/S/E
        'time': ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00'],
        'data_format': 'netcdf',
    }
    
    target = '/storage/silver/clearheads/Ben/saved_ERA5_data/download_cds/ERA5_EU_1hr_uv100m_t2m_msl_ssrd_' + str(y) + '_' + str(m) + '.nc'
    
    client.retrieve(dataset, request, target)
    
    return None

# define the main function
def main():
    
    # set up the argument parser
    parser = argparse.ArgumentParser(description='Download ERA5 data for RACC')
    start_year = parser.add_argument('start_year', type=int, help='The year to download data for')
    end_year = parser.add_argument('end_year', type=int, help='The year to download data for')
    start_month = parser.add_argument('start_month', type=int, help='The month to start downloading data for')
    end_month = parser.add_argument('end_month', type=int, help='The month to end downloading data for')

    # parse the arguments
    args = parser.parse_args()

    for year in range(args.start_year, args.end_year + 1):
        for month in range(args.start_month, args.end_month + 1):
            # download the ERA5 data
            download_ERA5_to_RACC(
                year=year,
                month=month,
            )

    return None

# run the main function
if __name__ == '__main__':
    main()