#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D /users/jjls2000
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
#SBATCH -o llava-med-test-%j.out
#SBATCH -e llava-med-test-%j.err
## Job name
#SBATCH -J gpu-job
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=01:00:00
## Memory limit (in megabytes)
#SBATCH --mem=32G
## GPU requirements
#SBATCH --gres=gpu:1
## Specify partition
#SBATCH -p gpu

################# Part-2 Shell script ####################
# Activate Flight Environment
source "${flight_ROOT:-/opt/flight}"/etc/setup.sh

# Activate Package Ecosystem
flight env activate gridware

# Load necessary modules
module load mpi/openmpi
module load cuda/11.8

# Create results directory
RESULTS_DIR="$(pwd)/${SLURM_JOB_NAME}-outputs/${SLURM_JOB_ID}"
echo "Your results will be stored in: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

# Debugging commands to verify setup
echo "CUDA_HOME is set to: $CUDA_HOME"
nvcc --version
nvidia-smi

# Run your training script
/users/jjls2000/.conda/envs/llavamed_new/bin/deepspeed /users/jjls2000/sharedscratch/Dissertation/llava/train/train_mem.py \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --deepspeed /users/jjls2000/sharedscratch/Dissertation/scripts/zero3.json \
    --model_name_or_path /users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-v1.5-7b \
    --version llava_v1.5 \
    --data_path "/users/jjls2000/sharedscratch/Dissertation/Slake1.0/augmented/BBF_train.json" \
    --image_folder "/users/jjls2000/sharedscratch/Dissertation/data/imgs-1" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter "/users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-v1.5-7b/mm_projector_extracted/mm_projector/data.pkl" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "/users/jjls2000/sharedscratch/Dissertation/results/$(date +%Y%m%d_%H%M%S)" \
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
    --lazy_preprocess True \
    --report_to wandb

echo "Training completed for BBF dataset."

# Optional post-processing
RESULTS_DIR="/users/jjls2000/sharedscratch/Dissertation/results/$(date +%Y%m%d_%H%M%S)"
if [ -f "${RESULTS_DIR}/output_model.bin" ]; then
    echo "Model successfully trained and saved."
else
    echo "Training failed or output model not saved."
fi
