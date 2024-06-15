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

# Activate the conda environment
source /opt/gridware/depots/761a7df9/el7/pkg/apps/anaconda3/2023.03/bin/etc/profile.d/conda.sh
conda activate llavamed

# Define the experiment name and results directory
EXPERIMENT_NAME="slake_test_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="/users/jjls2000/sharedscratch/Dissertation/results/${EXPERIMENT_NAME}"
mkdir -p "${RESULTS_DIR}"  # Ensure the directory exists

# Print out environment variables and paths for debugging
echo "Results Directory: ${RESULTS_DIR}"
echo "Experiment Name: ${EXPERIMENT_NAME}"

# Execute the Python script using the model and data files
python /users/jjls2000/sharedscratch/LLaVA-Med/llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --model-path "microsoft/llava-med-v1.5-mistral-7b" \  # Updated to use Hugging Face model ID
    --question-file /users/jjls2000/LLaVA-Med/data/eval/llava_med_eval_qa50_qa.jsonl \
    --image-folder /users/jjls2000/sharedscratch/Dissertation/data/images \
    --answers-file "${RESULTS_DIR}/answer-file-${SLURM_JOB_ID}.jsonl" \
    --temperature 0.0

# Check the existence of the output file before committing to Git
if [ -f "${RESULTS_DIR}/answer-file-${SLURM_JOB_ID}.jsonl" ]; then
    cd /users/jjls2000/sharedscratch/Dissertation
    git add "${RESULTS_DIR}/answer-file-${SLURM_JOB_ID}.jsonl"
    git commit -m "Add output for job ${SLURM_JOB_ID}"
    git push origin main
else
    echo "Output file not found: ${RESULTS_DIR}/answer-file-${SLURM_JOB_ID}.jsonl"
fi
