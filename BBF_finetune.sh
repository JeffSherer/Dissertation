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

# Set CUDA environment variables
export CUDA_HOME=/opt/gridware/depots/761a7df9/el9/pkg/libs/nvidia-cuda/11.8.0
export PATH=$CUDA_HOME/bin:$CUDA_HOME/bin/bin:$PATH

# Set Triton cache directory to a non-NFS path
export TRITON_CACHE_DIR=/users/jjls2000/local_cache
mkdir -p $TRITON_CACHE_DIR

# Activate Conda environment using full path
source /opt/gridware/depots/761a7df9/el9/pkg/apps/anaconda3/2023.03/bin/activate llavamed

# Ensure the Python script can find the module
export PYTHONPATH="/users/jjls2000/sharedscratch/Dissertation:${PYTHONPATH}"

# Verify environment setup
echo "CUDA_HOME is set to: $CUDA_HOME"
echo "PATH is set to: $PATH"
which nvcc
nvcc -V
echo "PYTHONPATH is set to: $PYTHONPATH"

################# Part-3 Define Experiment and Directories ####################

# Define paths for the dataset and results directory
DATA_PATH="/users/jjls2000/sharedscratch/Dissertation/Slake1.0/augmented"
RESULTS_DIR="/users/jjls2000/sharedscratch/Dissertation/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"  # Ensure the directory exists

# Define dataset file
BBF_TRAIN_JSON="${DATA_PATH}/BBF_train.json"

################# Part-4 Execute Fine-Tuning Script ####################

# Execute the fine-tuning using the BBF dataset
deepspeed /users/jjls2000/sharedscratch/Dissertation/llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed /users/jjls2000/sharedscratch/Dissertation/scripts/zero2.json \
    --model_name_or_path microsoft/llava-med-v1.5-mistral-7b \
    --version v1 \
    --data_path "${BBF_TRAIN_JSON}" \
    --image_folder /users/jjls2000/sharedscratch/Dissertation/data/imgs-1 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${RESULTS_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True

echo "Training completed for BBF dataset."

################# Part-5 Optional Post-Processing ####################

# Check for output and log results, perhaps using Git or another method to manage results
if [ -f "${RESULTS_DIR}/output_model.bin" ]; then
    echo "Model successfully trained and saved."
else
    echo "Training failed or output model not saved."
fi
