#!/bin/bash
# Load the Flight Environment
source "${flight_ROOT:-/opt/flight}"/etc/profile.d/conda.sh

# Function to create experiment directory
function create_experiment_dir() {
    local experiment_name=$1
    local base_dir=$2
    local results_dir="${base_dir}/${experiment_name}-outputs/${SLURM_JOB_ID}"

    echo "Your results will be stored in: ${results_dir}"
    mkdir -p "${results_dir}"
    echo "${results_dir}"
}

# Activate the conda environment hello
function activate_env() {
    conda activate $1
}
