#!/bin/bash
#SBATCH --job-name=slakeAnswer_BBF3   # Job name
#SBATCH --output=slakeAnswer_BBF3.out # Output file
#SBATCH --error=slakeAnswer_BBF3.err  # Error file
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --gres=gpu:1                  # Number of GPUs (use the right amount for your job)
#SBATCH --mem=32G                     # Memory per node
#SBATCH --time=02:00:00               # Time limit hrs:min:sec
#SBATCH --partition=gpu               # Partition name

# Load your environment, replace 'llava_test_env' with the correct conda environment if different
module load cuda/11.7  # Ensure the appropriate CUDA module is loaded
source activate llava_test_env

# Navigate to the directory where your script is located
cd /users/jjls2000/sharedscratch/Dissertation

# Execute the Python script with the desired arguments
python /users/jjls2000/sharedscratch/Dissertation/slakeAnswer.py \
    --model-path /users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-med-v1.5-mistral-7b-BBF-3 \
    --model-base /users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-med-v1.5-mistral-7b \
    --image-folder /users/jjls2000/sharedscratch/Dissertation/data/imgs-1 \
    --question-file /users/jjls2000/sharedscratch/Dissertation/Slake1.0/test_questions_fixed.jsonl \
    --answers-file /users/jjls2000/sharedscratch/Dissertation/results/final/answers-BBF--Slake3.jsonl \
    --conv-mode mistral_instruct \
    --temperature 0.0
