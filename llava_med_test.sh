#!/bin/bash
#SBATCH --job-name=llava-med-test
#SBATCH --output=llava-med-test.out
#SBATCH --error=llava-med-test.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00

# Load necessary modules
flight env activate gridware
module load apps/anaconda3/2023.03
module load apps/nvidia-cuda/12.3.2

# Activate the conda environment
conda activate llavamed

# Change to the LLaVA-Med directory
cd ~/LLaVA-Med

# Run the test script
python llava/eval/model_vqa.py --conv-mode mistral_instruct \
    --model-path microsoft/llava-med-v1.5-mistral-7b \
    --question-file data/eval/llava_med_eval_qa50_qa.jsonl \
    --image-folder data/images \
    --answers-file ~/Dissertation/results/answer-file.jsonl \
    --temperature 0.0

# Change to the Dissertation directory and process results
cd ~/Dissertation
python scripts/process_results.py --input results/answer-file.jsonl --output results/processed_results.jsonl
