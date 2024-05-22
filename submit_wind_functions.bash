#!/bin/bash

#these are all the default values anyway 
#SBATCH --ntasks=1 
#SBATCH --array=1-10
#SBATCH --cpus-per-task=1  
#SBATCH --threads-per-core=1 
#SBATCH --job-name=bens_wind_job
#SBATCH --output=/home/users/pn832950/100m_wind/output/bens_wind_job_%j_%a.out
#SBATCH --error=/home/users/pn832950/100m_wind/error/bens_wind_job_%j_%a.err

# time limit and memory allocation 
#SBATCH --time=00:30:00 # 30 minutes
#SBATCH --mem=2G # 2GB

# 'short' is the default partition, with max time of 24h
# longer jobs are submitted to the partition 'long'
#SBATCH --partition=scavenger # For array jobs, this will be the default partition

# load the required modules
module load anaconda

# # activate my conda environment
# source ~/.conda/envs/bens-racc-env/bin/activate

# Define a function which takes the ${SLURM_ARRAY_TASK_ID} as an argument
# e.g. 1, then translates this into the first year index e.g. 1950
# Define this function in bash
function translate_task_id_to_year() {
    # Define the base year
    local base_year=1950

    # Subtract 1 from the task ID because array indices start at 1
    local task_id=$((SLURM_ARRAY_TASK_ID - 1))

    # Add the task ID to the base year to get the year
    local year=$((base_year + task_id))

    echo $year
}

# Call the function and store the result in a variable
year=$(translate_task_id_to_year)

# run the script
# run the script and append the output to a file
python /home/users/pn832950/100m_wind/process_wind_data.py "United Kingdom" ${year}