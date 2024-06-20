#!/bin/bash -l

################# Part-1 Slurm directives ####################
#SBATCH -D /users/jjls2000/sharedscratch/Dissertation  # Set working directory
#SBATCH --export=ALL
#SBATCH -o llava-med-finetune-%j.out  # Standard output file
#SBATCH -e llava-med-finetune-%j.err  # Standard error file
#SBATCH -J llava-med-finetune  # Job name
#SBATCH --time=48:00:00  # Job run time
#SBATCH --mem=32G  # Memory required
#SBATCH --gres=gpu:1  # GPU resource allocation
#SBATCH -p gpu  # Partition

# Activate the conda environment
source /opt/gridware/depots/761a7df9/el7/pkg/apps/anaconda3/2023.03/bin/etc/profile.d/conda.sh
conda activate llavamed

# Define the experiment name and results directory
EXPERIMENT_NAME="llava_med_finetune_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="/users/jjls2000/sharedscratch/Dissertation/results/${EXPERIMENT_NAME}"
mkdir -p "${RESULTS_DIR}"  # Ensure the directory exists

# Print out environment variables and paths for debugging
echo "Results Directory: ${RESULTS_DIR}"
echo "Experiment Name: ${EXPERIMENT_NAME}"

# Execute the fine-tuning script
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

# Check the existence of the output file before committing to Git
if [ -f "${RESULTS_DIR}/answer-file-${SLURM_JOB_ID}.jsonl" ]; then
    cd /users/jjls2000/sharedscratch/Dissertation
    git add "${RESULTS_DIR}/answer-file-${SLURM_JOB_ID}.jsonl"
    git commit -m "Add output for job ${SLURM_JOB_ID}"
    git push origin main
else
    echo "Output file not found: ${RESULTS_DIR}/answer-file-${SLURM_JOB_ID}.jsonl"
fi
