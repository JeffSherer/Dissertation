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

# Load necessary modules and activate environment
source /etc/profile.d/modules.sh
module load apps/anaconda3/2023.03/bin
module load libs/nvidia-cuda/11.8.0/bin
export CUDA_HOME=/opt/gridware/depots/761a7df9/el9/pkg/libs/nvidia-cuda/11.8.0

# Activate Conda environment
source activate llavamed

################# Part-3 Define Experiment and Directories ####################

# Define experiment name and results directory
EXPERIMENT_NAME="slake_test_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="/users/jjls2000/sharedscratch/Dissertation/results/${EXPERIMENT_NAME}"
mkdir -p "${RESULTS_DIR}"  # Ensure the directory exists

# Print out environment variables and paths for debugging
echo "Results Directory: ${RESULTS_DIR}"
echo "Experiment Name: ${EXPERIMENT_NAME}"

################# Part-4 Execute Fine-Tuning Script ####################

# Execute the fine-tuning script using DeepSpeed
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path /users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-llavammed-7b \
    --version "llava_med_v1.5" \
    --data_path /users/jjls2000/sharedscratch/Dissertation/Slake1.0/augmented/BBF_train.json \
    --image_folder /users/jjls2000/sharedscratch/Dissertation/data/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter /users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-llavammed-7b-pretrain/mm_projector.bin \
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
    --save_steps 5000 \
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
    --report_to wandb

################# Part-5 Post-Processing ####################

# Check the existence of the output file before committing to Git
if [ -f "${RESULTS_DIR}/answer-file-${SLURM_JOB_ID}.jsonl" ]; then
    cd /users/jjls2000/sharedscratch/Dissertation
    git add "${RESULTS_DIR}/answer-file-${SLURM_JOB_ID}.jsonl"
    git commit -m "Add output for job ${SLURM_JOB_ID}"
    git push origin main
else
    echo "Output file not found: ${RESULTS_DIR}/answer-file-${SLURM_JOB_ID}.jsonl"
fi

# Deactivate Conda environment after job completion (optional)
if command -v flight &> /dev/null; then
    flight env deactivate
fi
