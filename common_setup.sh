#!/bin/bash

# Ensure Conda is initialized properly
# Replace "/opt/anaconda3/etc/profile.d/conda.sh" with the actual path to your conda.sh
source /opt/gridware/depots/761a7df9/el7/pkg/apps/anaconda3/2023.03/bin/etc/profile.d/conda.sh


# Function to create an experiment directory
function create_experiment_dir() {
    local experiment_name=$1
    local base_dir=$2
    local results_dir="${base_dir}/${experiment_name}-outputs/${SLURM_JOB_ID}"

    echo "Your results will be stored in: ${results_dir}"
    mkdir -p "${results_dir}"
    return "${results_dir}"
}

# Function to activate the conda environment
function activate_env() {
    local env_name=$1
    echo "Activating Conda environment: ${env_name}"
    conda activate "${env_name}"
}

# Example usage within a SLURM script or other batch job
# activate_env "hello"
# results_dir=$(create_experiment_dir "MyExperiment" "/path/to/base/dir")
