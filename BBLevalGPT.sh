#!/bin/bash
#SBATCH --job-name=model_vqa
#SBATCH --output=results/model_vqa_output_%j.txt
#SBATCH --error=results/model_vqa_error_%j.txt
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Load the necessary modules or activate your conda environment
module load anaconda
source activate llava-med-eval

# Change to the directory where your script is located
cd /users/jjls2000/sharedscratch/Dissertation

# Run the command
PYTHONPATH=. python llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --model-path /users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-med-v1.5-mistral-7b-BBL-3 \
    --question-file data/eval/llava_med_eval_qa50_qa.jsonl \
    --image-folder data/figures \
    --answers-file /users/jjls2000/sharedscratch/Dissertation/results/answers_bbl-3.jsonl \
    --temperature 0.0
