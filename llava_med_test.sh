#!/bin/bash -l

################# Part-1 Slurm directives ####################
#SBATCH -D /users/jjls2000/sharedscratch/Dissertation  # Set working directory
#SBATCH --export=ALL
#SBATCH -o llava-med-test-%j.out  # Standard output file
#SBATCH -e llava-med-test-%j.err  # Standard error file
#SBATCH -J slake-test  # Job name
#SBATCH --time=02:00:00  # Job run time
#SBATCH --mem=32G  # Memory required
#SBATCH --gres=gpu:1  # GPU resource allocation
#SBATCH -p gpu  # Partition

# Source the flight environment setup script for managing Conda environments
source "${flight_ROOT:-/opt/flight}"/etc/setup.sh

# Load necessary modules (Adjust these if different modules are needed)
flight env activate gridware  # Activate the flight gridware environment
module load mpi/openmpi  # Load MPI module for parallel processing

# Function to activate conda environment
function activate_env() {
    local env_name=$1
    echo "Activating conda environment: $env_name"
    conda activate "$env_name"
}

# Function to create an experiment directory
function create_experiment_dir() {
    local experiment_name=$1
    local base_dir=$2
    local results_dir="${base_dir}/${experiment_name}-outputs/${SLURM_JOB_ID}"
    echo "Your results will be stored in: $results_dir"
    mkdir -p "$results_dir"
    echo "$results_dir"
}

# Activate the required Conda environment
activate_env "llavamed"

# Define experiment name based on current date and time
EXPERIMENT_NAME="slake_test_$(date +%Y%m%d_%H%M%S)"

# Create a directory to store the results of the experiment
RESULTS_DIR=$(create_experiment_dir "$EXPERIMENT_NAME" "/users/jjls2000/sharedscratch/Dissertation")

# Execute the Python script using the model and data files
python /users/jjls2000/sharedscratch/LLaVA-Med/llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --model-path /users/jjls2000/sharedscratch/Dissertation/checkpoints/slake \
    --question-file /users/jjls2000/LLaVA-Med/data/eval/llava_med_eval_qa50_qa.jsonl \
    --image-folder /users/jjls2000/sharedscratch/Dissertation/data/images \
    --answers-file "${RESULTS_DIR}/answer-file-${SLURM_JOB_ID}.jsonl" \
    --temperature 0.0

# Handle Git operations for version control and collaboration
cd /users/jjls2000/sharedscratch/Dissertation
git add "${RESULTS_DIR}/answer-file-${SLURM_JOB_ID}.jsonl"
git commit -m "Add output for job ${SLURM_JOB_ID}"
git push origin main
