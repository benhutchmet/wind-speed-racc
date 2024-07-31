#!/bin/bash

#these are all the default values anyway 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1  
#SBATCH --threads-per-core=1 
#SBATCH --job-name=bens_wind_job
#SBATCH --output=/home/users/pn832950/100m_wind/output/bens_ERA5_job_%j_%a.out
#SBATCH --error=/home/users/pn832950/100m_wind/error/bens_ERA5_job_%j_%a.err

# time limit and memory allocation 
#SBATCH --time=04:00:00 # 4 hours
#SBATCH --mem=200G # 200GB

# 'short' is the default partition, with max time of 24h
# longer jobs are submitted to the partition 'long'
#SBATCH --partition=short

# load the required modules
module load anaconda

# set up a USAGE message
USAGE="Usage: sbatch submit_ERA5_process.bash <first_year> <last_year> <variable_name>"

# if the number of arguments is not equal to 3, print the USAGE message and exit
if [ "$#" -ne 3 ]; then
    echo $USAGE
    exit 1
fi

# Extract the args
first_year=$1
last_year=$2
variable_name=$3

# Echo the args
echo "First year: $first_year"
echo "Last year: $last_year"
echo "Variable name: $variable_name"

# Set up the process_sctipt
PROCESS_SCRIPT="/home/users/pn832950/100m_wind/process_ERA5_data.py"

# run the script
# run the script and append the output to a file
python ${PROCESS_SCRIPT} ${first_year} ${last_year} ${variable_name}