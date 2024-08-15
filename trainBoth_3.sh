#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D /users/jjls2000/sharedscratch/Dissertation
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
#SBATCH -o /users/jjls2000/sharedscratch/Dissertation/llava-med-train-%j.out
#SBATCH -e /users/jjls2000/sharedscratch/Dissertation/llava-med-train-%j.err
## Job name
#SBATCH -J llava-job
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=48:00:00
## Memory limit (in megabytes)
#SBATCH --mem=128G
## GPU requirements
#SBATCH --gres=gpu:2
## Specify partition
#SBATCH -p gpu

################# Part-2 Shell script ####################

# Load modules or set up the environment here
# Example for conda setup

# Make sure conda is initialized
source /users/jjls2000/.bashrc

# Activate the conda environment
conda activate llava-med

# Add DeepSpeed to the PATH (if it isn't already)
export PATH=/users/jjls2000/.conda/envs/llava-med/bin:$PATH

# Train BBF model
deepspeed /users/jjls2000/sharedscratch/Dissertation/llava/train/train_mem.py \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --deepspeed /users/jjls2000/sharedscratch/Dissertation/scripts/zero3.json \
    --model_name_or_path /users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-med-v1.5-mistral-7b \
    --version v1 \
    --data_path /users/jjls2000/sharedscratch/Dissertation/Slake1.0/augmented_filtered/BBF_train_medium.json \
    --image_folder /users/jjls2000/sharedscratch/Dissertation/data/imgs-1 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-med-v1.5-mistral-7b-BBF-3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

# Train BBL model
deepspeed /users/jjls2000/sharedscratch/Dissertation/llava/train/train_mem.py \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --deepspeed /users/jjls2000/sharedscratch/Dissertation/scripts/zero3.json \
    --model_name_or_path /users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-med-v1.5-mistral-7b \
    --version v1 \
    --data_path /users/jjls2000/sharedscratch/Dissertation/Slake1.0/augmented_filtered/BBL_train_medium.json \
    --image_folder /users/jjls2000/sharedscratch/Dissertation/data/imgs-1 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-med-v1.5-mistral-7b-BBL-3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
