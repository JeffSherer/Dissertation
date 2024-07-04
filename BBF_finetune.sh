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

# Activate Conda environment
source activate llavamed  # Ensure this points to the correct path where your conda environments are managed

# Set CUDA environment variable
export CUDA_HOME=$CUDA_PATH  # Assuming the module load sets CUDA_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Set Triton cache directory to a non-NFS path
export TRITON_CACHE_DIR=/users/jjls2000/local_cache
mkdir -p $TRITON_CACHE_DIR

################# Part-3 Define Experiment and Directories ####################

# Define paths for the dataset and results directory
DATA_PATH="/users/jjls2000/sharedscratch/Dissertation/Slake1.0/augmented"
RESULTS_DIR="/users/jjls2000/sharedscratch/Dissertation/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"  # Ensure the directory exists

# Define dataset file
BBF_TRAIN_JSON="${DATA_PATH}/BBF_train.json"

# Ensure the Python script can find the module
export PYTHONPATH="/users/jjls2000/sharedscratch/Dissertation:${PYTHONPATH}"

# Model and prompt versions
PROMPT_VERSION="llava_med_v1.5"
MODEL_VERSION="llava-llavammed-7b"

################# Part-4 Execute Fine-Tuning Script ####################

# Execute the fine-tuning using the BBF dataset
deepspeed /users/jjls2000/sharedscratch/Dissertation/llava/train/train_mem.py \
    --deepspeed /users/jjls2000/sharedscratch/Dissertation/scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path "microsoft/llava-med-v1.5-mistral-7b" \
    --version $PROMPT_VERSION \
    --data_path "${BBF_TRAIN_JSON}" \
    --image_folder /users/jjls2000/sharedscratch/Dissertation/data/imgs-1 \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter /users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-$MODEL_VERSION-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "${RESULTS_DIR}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to none  # Change as per your tracking system, e.g., wandb

echo "Training completed for BBF dataset."

################# Part-5 Optional Post-Processing ####################

# Check for output and log results, perhaps using Git or another method to manage results
if [ -f "${RESULTS_DIR}/output_model.bin" ]; then
    echo "Model successfully trained and saved."
else
    echo "Training failed or output model not saved."
fi
