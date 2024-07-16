#!/bin/bash
## Working dir
#SBATCH -D /users/jjls2000/sharedscratch/Dissertation
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
#SBATCH -o /users/jjls2000/sharedscratch/Dissertation/llava-med-test-%j.out
## Error File
#SBATCH -e /users/jjls2000/sharedscratch/Dissertation/llava-med-test-%j.err
## Job name
#SBATCH -J llava-job
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=24:10:00
## Memory limit (in megabytes)
#SBATCH --mem=32G
## GPU requirements
#SBATCH --gres=gpu:1
## Specify partition
#SBATCH -p gpu

module load libs/nvidia-cuda/11.8.0/bin  # Load CUDA 11.8 module

# Activate your environment if needed
source activate llava  # or 'conda activate llava'

# Install Bits and Bytes
pip install bitsandbytes-cuda118  # Adjust the command based on the needed CUDA version

echo "Installation of Bits and Bytes completed."
