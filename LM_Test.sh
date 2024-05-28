#!/bin/bash
#SBATCH --job-name=llava-med-test
#SBATCH --partition=gpu       # Specify the GPU partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH --mem=16G             # Request 16 GB of memory
#SBATCH --time=01:00:00       # Set the time limit to 1 hour
#SBATCH --output=llava_med_test_%j.out

# Load the Anaconda module and activate the environment
module load anaconda/2021
conda activate llavamed

# Change to the LLaVA-Med directory
cd /path/to/your/working/directory/LLaVA-Med

# Run the test script to check data access
python test_data_access.py
