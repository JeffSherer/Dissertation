#!/bin/bash
#SBATCH --job-name=llava-med-test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2  # Adjust based on how many GPUs you want to test
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=llava_med_test_%j.out

# Load the Anaconda module and activate the environment
module load anaconda/2021
conda activate llavamed

# Run the LLaVA-Med model evaluation script
python ./llava/eval/model_vqa.py  # Adjust this script path if necessary
