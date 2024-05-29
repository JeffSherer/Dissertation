#!/bin/bash
#SBATCH --job-name=llava-med-test
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

echo "Starting job on $(date)"
echo "Running on node $(hostname)"

# Load necessary modules and activate environment
flight env activate gridware
module load apps/nvidia-cuda/11.2.2
source ~/.bashrc
conda activate llavamed

echo "Environment setup complete"

# Run the test script
python /users/jjls2000/LLaVA-Med/llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --model-path microsoft/llava-med-v1.5-mistral-7b \
    --question-file /users/jjls2000/Dissertation/data/eval/llava_med_eval_qa50_qa.jsonl \
    --image-folder /users/jjls2000/Dissertation/data/images \
    --answers-file /users/jjls2000/Dissertation/results/answer-file.jsonl \
    --temperature 0.0

echo "Job finished on $(date)"
