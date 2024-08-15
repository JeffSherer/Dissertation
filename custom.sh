#!/bin/bash
#SBATCH --job-name=llava_test
#SBATCH --output=cuda_test_%j.out
#SBATCH --error=cuda_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu

# Load necessary modules
module load cuda/11.7  # Adjust the CUDA version as needed

# Run the test script
python cuda_test.py
