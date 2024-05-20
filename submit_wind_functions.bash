#!/bin/bash

#these are all the default values anyway 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1  
#SBATCH --threads-per-core=1 

# the job name and output file
#SBATCH --job-name=bens_wind_job #(default is the script name)
# Set where the output goes
#SBATCH --output=/home/users/pn832950/100m_wind/output/bens_wind_job_%j.out #(default is the job name with %j appended)
# Set the error file
#SBATCH --error=/home/users/pn832950/100m_wind/error/bens_wind_job_%j.err #(default is the job name with %j appended)

# time limit and memory allocation 
#SBATCH --time=00:20:00 # 20 minutes
#SBATCH --mem=2G # 2GB

# 'short' is the default partition, with max time of 24h
# longer jobs are submitted to the partition 'long'
#SBATCH --partition=short

# load the required modules
module load anaconda

# # activate my conda environment
# source ~/.conda/envs/bens-racc-env/bin/activate

# run the script
# run the script and append the output to a file
python /home/users/pn832950/100m_wind/load_wind_functions.py >> /home/users/pn832950/100m_wind/python_output_$(date +%s).out