#!/bin/bash
#SBATCH --job-name=install-bitsandbytes
#SBATCH --gres=gpu:1              # Request GPU generic resources; adjust if a different GPU count or type is available
#SBATCH --mem=8G                  # Memory requirement; increase or decrease based on job needs and availability
#SBATCH --time=0-00:30            # Time limit hrs:min:sec
#SBATCH --output=install_bnb_%j.log  # Standard output and error log
#SBATCH --partition=gpu           # Make sure to specify the correct partition that has GPU resources

module load libs/nvidia-cuda/11.8.0/bin  # Load CUDA 11.8 module

# Activate your environment if needed
source activate llava  # or 'conda activate llava'

# Install Bits and Bytes
pip install bitsandbytes-cuda118  # Adjust the command based on the needed CUDA version

echo "Installation of Bits and Bytes completed."
