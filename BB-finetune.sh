#!/bin/bash -l
#SBATCH -D /users/jjls2000/sharedscratch/Dissertation  # Set working directory
#SBATCH --export=ALL
#SBATCH -o llava-med-test-%j.out  # Standard output file
#SBATCH -e llava-med-test-%j.err  # Standard error file
#SBATCH -J slake-test  # Job name
#SBATCH --time=02:00:00  # Job run time
#SBATCH --mem=32G  # Memory required
#SBATCH --gres=gpu:1  # GPU resource allocation
#SBATCH -p gpu  # Partition

# Activate the conda environment
source /opt/gridware/depots/761a7df9/el7/pkg/apps/anaconda3/2023.03/bin/etc/profile.d/conda.sh
conda activate llavamed

# Define the experiment name and results directory
EXPERIMENT_NAME="slake_test_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="/users/jjls2000/sharedscratch/Dissertation/results/${EXPERIMENT_NAME}"
mkdir -p "${RESULTS_DIR}"  # Ensure the directory exists

# Fine-tune the model
python /users/jjls2000/sharedscratch/LLaVA-Med/fine_tune_script.py \
    --model-name "microsoft/llava-med-v1.5-mistral-7b" \
    --data-path "/users/jjls2000/sharedscratch/Dissertation/data/augmented_dataset.json" \
    --output-dir "${RESULTS_DIR}" \
    --learning-rate 2e-5 \
    --epochs 3 \
    --batch-size 16
