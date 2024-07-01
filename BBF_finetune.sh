#!/bin/bash -l

################# Part-1 Slurm directives ####################
#SBATCH -D /users/jjls2000/sharedscratch/Dissertation  # Set working directory
#SBATCH --export=ALL
#SBATCH -o llava-med-test-%j.out  # Standard output file
#SBATCH -e llava-med-test-%j.err  # Standard error file
#SBATCH -J slake-test  # Job name
#SBATCH --time=02:00:00  # Job run time
#SBATCH --mem=32G  # Memory required
#SBATCH --gres=gpu:1  # GPU resource allocation
#SBATCH -p gpu  # Partition

################# Part-2 Environment Setup ####################

# Load necessary modules
module load apps/anaconda3/2023.03
module load libs/nvidia-cuda/11.8.0

# Activate Conda environment
source activate llavamed  # Make sure this points to the correct path where your conda environments are managed

################# Part-3 Define Experiment and Directories ####################

# Define paths for the dataset and results directory
DATA_PATH="/users/jjls2000/sharedscratch/Dissertation/Slake1.0/augmented"
RESULTS_DIR="/users/jjls2000/sharedscratch/Dissertation/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"  # Ensure the directory exists

# Define dataset files
BBF_TRAIN_JSON="${DATA_PATH}/BBF_train.json"
BBL_TRAIN_JSON="${DATA_PATH}/BBL_train.json"

################# Part-4 Execute Fine-Tuning Script ####################

# Execute the fine-tuning using the BBF dataset
deepspeed /users/jjls2000/sharedscratch/Dissertation/train/train_mem.py \  # Corrected path for the training script
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path /users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-llavammed-7b \
    --version "llava_med_v1.5" \
    --data_path "${BBF_TRAIN_JSON}" \
    --image_folder /users/jjls2000/sharedscratch/Dissertation/data/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter /users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-llavammed-7b-pretrain/mm_projector.bin \
    --output_dir "${RESULTS_DIR}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --logging_steps 100 \
    --gradient_checkpointing True \
    --save_total_limit 1 \
    --report_to none  # Change as per your tracking system, e.g., wandb

echo "Training completed for BBF dataset."

# Optionally run a second job for the BBL dataset by changing --data_path to "${BBL_TRAIN_JSON}"

################# Part-5 Optional Post-Processing ####################

# Check for output and log results, perhaps using Git or another method to manage results
if [ -f "${RESULTS_DIR}/output_model.bin" ]; then
    echo "Model successfully trained and saved."
else
    echo "Training failed or output model not saved."
fi
