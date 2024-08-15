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
#SBATCH --time=10:00:00
## Memory limit (in gigabytes)
#SBATCH --mem=16G
## GPU requirements
#SBATCH --gres=gpu:1  # Requesting 1 GPU
## Specify partition
#SBATCH --partition=gpu

################# Part-2 Shell commands ####################

# Load necessary modules
module load anaconda

# Activate your conda environment
source activate llava-med-eval

# Run your script
PYTHONPATH=. python llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --model-path /mnt/scratch/users/jjls2000/Dissertation/checkpoints/llava-med-v1.5-mistral-7b-BBL_NoMod \
    --question-file data/eval/llava_med_eval_qa50_qa.jsonl \
    --image-folder data/figures \
    --answers-file /mnt/scratch/users/jjls2000/Dissertation/results/Experiment_2_Eval_Files/BBL/answer-file.jsonl \
    --temperature 0.0

