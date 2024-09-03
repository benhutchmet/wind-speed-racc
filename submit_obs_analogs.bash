#!/bin/bash

#these are all the default values anyway 
#SBATCH --ntasks=1 
#SBATCH --array=1-5
#SBATCH --cpus-per-task=1  
#SBATCH --threads-per-core=1 
#SBATCH --job-name=bens_wind_analogs_job
#SBATCH --output=/home/users/pn832950/100m_wind/output/bens_wind_analogs_job_%j_%a.out
#SBATCH --error=/home/users/pn832950/100m_wind/error/bens_wind_analogs_job_%j_%a.err

# time limit and memory allocation 
#SBATCH --time=00:60:00 # 60 minutes
#SBATCH --mem=15G # 15GB

# 'short' is the default partition, with max time of 24h
# longer jobs are submitted to the partition 'long'
#SBATCH --partition=scavenger # For array jobs, this will be the default partition

# start a timer
SECONDS=0

USG_MSG="Usage: sbatch submit_obs_analogs.bash <ons_ofs>"

# set up the usage message
if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo $USG_MSG
    exit 1
fi

# Set up the list of members 1-10
MEMBERS=$(seq 1 10)

# hard code the WP_WEIGHTS
WP_WEIGHTS="1"

# load the required modules
module load anaconda

# # activate my conda environment
# source ~/.conda/envs/bens-racc-env/bin/activate

# Define a function which takes the ${SLURM_ARRAY_TASK_ID} as an argument
# e.g. 1, then translates this into the first year index e.g. 1950
# Define this function in bash
function translate_task_id_to_year() {
    # Define the base year
    local base_year=1960

    # Subtract 1 from the task ID because array indices start at 1
    local task_id=$((SLURM_ARRAY_TASK_ID - 1))

    # Add the task ID to the base year to get the year
    local year=$((base_year + task_id))

    echo $year
}

# Call the function and store the result in a variable
year=$(translate_task_id_to_year)

# set up the process script
PROCESS_SCRIPT="/home/users/pn832950/100m_wind/load_obs_analogs.py"

# Loop over the members
for member in ${MEMBERS}; do
    # cho the args
    echo "init_year: ${year}"
    echo "member: ${member}"
    echo "ons_ofs: ${1}"
    echo "WP_WEIGHTS: ${WP_WEIGHTS}"
    
    # run the script
    # run the script and append the output to a file
    python ${PROCESS_SCRIPT} --init_year ${year} --member ${member} --ons_ofs ${1} --wp_weights ${WP_WEIGHTS}
done

# echo how long the script took
minutes=$((SECONDS / 60))
seconds=$((SECONDS % 60))
echo "The script took $minutes minutes and $seconds seconds to run"
echo "Process complete"
# end
exit 0