#!/bin/bash
#SBATCH --job-name=llava-med-test
#SBATCH --output=llava-med-test.out
#SBATCH --error=llava-med-test.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00

# Activate the gridware environment and load necessary modules
flight env activate gridware
module load apps/nvidia-cuda/12.3.2
module load apps/anaconda3/2023.03

# Activate your conda environment
source activate llavamed

# Run the LLaVA-Med model worker
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path microsoft/llava-med-v1.5-mistral-7b --multi-modal

# (Optional) Run a test message if needed
# python -m llava.serve.test_message --model-name llava-med-v1.5-mistral-7b --controller http://localhost:10000

# (Optional) Launch the gradio web server if needed
# python -m llava.serve.gradio_web_server --controller http://localhost:10000
