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

################# Part-3 Execute Fine-Tuning Script ####################

# Use the absolute path to the deepspeed in your Conda environment
/users/jjls2000/.conda/envs/llavamed_new/bin/deepspeed /users/jjls2000/sharedscratch/Dissertation/llava/train/train_mem.py \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --deepspeed /users/jjls2000/sharedscratch/Dissertation/scripts/zero3.json \
    --model_name_or_path "microsoft/llava-med-v1.5-mistral-7b" \
    --version llava_med_v1.5 \
    --data_path "/users/jjls2000/sharedscratch/Dissertation/Slake1.0/augmented/BBF_train.json" \
    --image_folder "/users/jjls2000/sharedscratch/Dissertation/data/imgs-1" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter "/users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-llava-llavammed-7b-pretrain/mm_projector.bin" \
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

################# Part-4 Optional Post-Processing ####################

# Check for output and log results, perhaps using Git or another method to manage results
RESULTS_DIR="/users/jjls2000/sharedscratch/Dissertation/results/$(date +%Y%m%d_%H%M%S)"
if [ -f "${RESULTS_DIR}/output_model.bin" ]; then
    echo "Model successfully trained and saved."
else
    echo "Training failed or output model not saved."
fi
