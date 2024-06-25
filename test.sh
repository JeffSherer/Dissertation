#!/bin/bash -l

# SLURM Directives
#SBATCH -D /users/jjls2000/sharedscratch/Dissertation
#SBATCH --export=ALL
#SBATCH -o llava-med-test-%j.out
#SBATCH -e llava-med-test-%j.err
#SBATCH -J slake-test
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p gpu

# Load necessary modules
module load anaconda3/2023.03 2>/dev/null || source /opt/gridware/depots/761a7df9/el7/pkg/apps/anaconda3/2023.03/bin/etc/profile.d/conda.sh
conda activate llavamed

# Prepare environment
EXPERIMENT_NAME="slake_test_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="/users/jjls2000/sharedscratch/Dissertation/results/${EXPERIMENT_NAME}"
mkdir -p "${RESULTS_DIR}" || { echo "Failed to create directory"; exit 1; }

# Execute
python /users/jjls2000/sharedscratch/LLaVA-Med/llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --model-path "microsoft/llava-med-v1.5-mistral-7b" \
    --question-file /users/jjls2000/sharedscratch/LLaVA-Med/data/eval/llava_med_eval_qa50_qa.jsonl \
    --image-folder "/users/jjls2000/sharedscratch/Dissertation/data/images" \
    --answers-file "${RESULTS_DIR}/answer-file-${SLURM_JOB_ID}.jsonl" \
    --temperature 0.0 > "${RESULTS_DIR}/model_run-${SLURM_JOB_ID}.log" 2>&1

# Check and commit to Git
if [ -f "${RESULTS_DIR}/answer-file-${SLURM_JOB_ID}.jsonl" ]; then
    cd /users/jjls2000/sharedscratch/Dissertation
    git add "${RESULTS_DIR}/answer-file-${SLURM_JOB_ID}.jsonl"
    git commit -m "Add output for job ${SLURM_JOB_ID}"
    git push origin main
else
    echo "Output file not found: ${RESULTS_DIR}/answer-file-${SLURM_JOB_ID}.jsonl"
    exit 1
fi
