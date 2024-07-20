#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working directory
#SBATCH -D /users/jjls2000/sharedscratch/Dissertation
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
#SBATCH -o /users/jjls2000/sharedscratch/Dissertation/job-%j.output
#SBATCH -e /users/jjls2000/sharedscratch/Dissertation/job-%j.error
## Job name
#SBATCH -J evaluation-job
## Run time: "hours:minutes:seconds"
#SBATCH --time=02:00:00
## Memory limit (in gigabytes)
#SBATCH --mem=32G
## GPU requirements
#SBATCH --gres=gpu:1  # Requesting 1 GPU
## Specify partition
#SBATCH --partition=gpu

################# Part-2 Shell script ####################
# Ensure the OpenAI API Key is set in your environment before running this script
if [ -z "$OPENAI_API_KEY" ]; then
  echo "ERROR: The environment variable OPENAI_API_KEY is not set."
  exit 1
fi

# Correct path to conda.sh
CONDA_PATH="/opt/gridware/depots/761a7df9/el7/pkg/apps/anaconda3/2023.03/bin/etc/profile.d/conda.sh"

if [ ! -f "$CONDA_PATH" ]; then
  echo "ERROR: conda.sh not found at $CONDA_PATH"
  exit 1
fi

# Activate the conda environment
source $CONDA_PATH
conda activate llava-med

# Run the Python evaluation script
python /users/jjls2000/sharedscratch/Dissertation/llava/eval/eval_multimodal_chat_gpt_score.py \
  --answers-file /users/jjls2000/sharedscratch/Dissertation/results/llava_med_eval_answers.jsonl \
  --question-file /users/jjls2000/sharedscratch/Dissertation/data/eval/llava_med_eval_qa50_qa.jsonl \
  --scores-file /users/jjls2000/sharedscratch/Dissertation/results/eval_scores.jsonl

