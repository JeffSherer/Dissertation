#!/bin/bash
#SBATCH --job-name=llava-med-test
#SBATCH --output=llava-med-test.out
#SBATCH --error=llava-med-test.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Load necessary modules and activate environment
flight env activate gridware
module load apps/nvidia-cuda/11.2.2
conda activate llavamed



# Run the test script
python /users/jjls2000/LLaVA-Med/llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --model-path microsoft/llava-med-v1.5-mistral-7b \
    --question-file /users/jjls2000/Dissertation/data/eval/llava_med_eval_qa50_qa.jsonl \
    --image-folder /users/jjls2000/Dissertation/data/images \
    --answers-file /users/jjls2000/Dissertation/results/answer-file.jsonl \
    --temperature 0.0


