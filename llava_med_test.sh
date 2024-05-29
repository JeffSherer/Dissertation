#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D /users/jjls2000/sharedscratch/Dissertation
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
#SBATCH -o llava-med-test-%j.out
#SBATCH -e llava-med-test-%j.err
## Job name
#SBATCH -J slake-test
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=02:00:00
## Memory limit (in megabytes)
#SBATCH --mem=32G
## GPU requirements
#SBATCH --gres=gpu:1
## Specify partition
#SBATCH -p gpu

################# Part-2 Shell script ####################
# Source the conda initialization script
source /opt/gridware/depots/761a7df9/el9/pkg/apps/anaconda3/2023.03/etc/profile.d/conda.sh

# Activate the conda environment
conda activate llavamed

# Define the experiment name
EXPERIMENT_NAME="slake_test_$(date +%Y%m%d_%H%M%S)"

# Create new experiment directory and organize initial results
python3 ~/sharedscratch/Dissertation/manage_experiments.py $EXPERIMENT_NAME

# Verify the script path
if [ ! -f /users/jjls2000/LLaVA-Med/llava/eval/model_vqa.py ]; then
  echo "Script model_vqa.py not found!"
  exit 1
fi

# Run the test script with SLaKE checkpoint
python /users/jjls2000/LLaVA-Med/llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --model-path /users/jjls2000/sharedscratch/Dissertation/checkpoints/slake \
    --question-file /users/jjls2000/LLaVA-Med/data/eval/llava_med_eval_qa50_qa.jsonl \
    --image-folder /users/jjls2000/sharedscratch/Dissertation/data/images \
    --answers-file /users/jjls2000/sharedscratch/Dissertation/results/${EXPERIMENT_NAME}/answer-file-${SLURM_JOB_ID}.jsonl \
    --temperature 0.0

# Move results after experiment completes
python3 ~/sharedscratch/Dissertation/manage_experiments.py $EXPERIMENT_NAME
