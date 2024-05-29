### Notes ###

#### Files which could do with updating ####

* Temp/demand - regression coefficients for the different countries (*ERA5_Regression_coeffs_demand_model.csv*)
    * Will these need updating? Will demand/temp patterns have changed since S2S4E work?

* 100m wind speed - bias correction file for the wind speeds from ERA5, corrected to the global wind atlas (?, *UREAD_energy_models_demo_scripts/ERA5_speed100m_mean_factor_v16_hourly.npy*).
    * Will this have changes since CLEARHEADS?

#### Files which are missing ####

* Power curves for wind turbines:
    * Onshore and offshore
    * Currently using *powercurve.csv* from random RACC directory (either S2S4E or CLEARHEADS).
    * I would like:
    * Onshore - */home/users/zd907959/code_folders/S2S4E/MERRA2_models/MERRA2_wind_model/python_version/power_onshore.csv*
    * Offshore - */home/users/zd907959/code_folders/S2S4E/MERRA2_models/MERRA2_wind_model/python_version/power_offshore.csv*
    * Or others if more up to date?

* Installed wind farm distributions:
    * Currently using those in dir: */storage/silver/S2S4E/zd907959/MERRA2_wind_model/python_version/{country}windfarm_dist.nc*
    * I imagine these have likely changed since 2018 when they were first updated.
    * I would like:
    * 2021 wind farm distributions: */home/users/zd907959/code_folders/CLEARHEADS/wind_power_model/data/{COUNTRY}windfarm_dist_{ons_ofs}_2021.nc*
    * Future wind farm distributions: */home/users/zd907959/code_folders/CLEARHEADS/wind_power_model/data/{COUNTRY}windfarm_dist_{ons_ofs}_future.nc*
    * Or any more up-to-date distributions (CCC 2024/2035?) or how could I scrape this data myself from windpower.net?