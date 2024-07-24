#!/bin/bash

#these are all the default values anyway 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1  
#SBATCH --threads-per-core=1 
#SBATCH --job-name=bens_temp_job
#SBATCH --output=/home/users/pn832950/100m_wind/output/bens_temp_job_%j_%a.out
#SBATCH --error=/home/users/pn832950/100m_wind/error/bens_temp_job_%j_%a.err

# time limit and memory allocation 
#SBATCH --time=04:00:00 # 4 hours
#SBATCH --mem=30G # 30GB

# 'short' is the default partition, with max time of 24h
# longer jobs are submitted to the partition 'long'
#SBATCH --partition=short

# load the required modules
module load anaconda

# set up a USAGE message
USAGE="Usage: sbatch submit_ERA5_temp_process.bash <first_year> <last_year> <start_month> <end_month>"

# if the number of arguments is not zero
if [ "$#" -ne 4 ]; then
    echo $USAGE
    exit 1
fi

# extract the args
START_YEAR=$1
END_YEAR=$2
FIRST_MONTH=$3
LAST_MONTH=$4

# Set up the process_sctipt
PROCESS_SCRIPT="/home/users/pn832950/100m_wind/process_temp_data.py"

# run the script
# run the script and append the output to a file
# Run the script
python ${PROCESS_SCRIPT} --start_year ${START_YEAR} --end_year ${END_YEAR} --first_month ${FIRST_MONTH} --last_month ${LAST_MONTH}