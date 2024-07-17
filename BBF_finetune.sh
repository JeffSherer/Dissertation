#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D /users/jjls2000/sharedscratch/Dissertation
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
#SBATCH -o /users/jjls2000/sharedscratch/Dissertation/llava-med-test-%j.out
#SBATCH -e /users/jjls2000/sharedscratch/Dissertation/llava-med-test-%j.err
## Job name
#SBATCH -J llava-job
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=24:10:00
## Memory limit (in megabytes)
#SBATCH --mem=32G
## GPU requirements
#SBATCH --gres=gpu:1
## Specify partition
#SBATCH -p gpu

################# Part-2 Shell script ####################
# Activate Conda environment
source /users/jjls2000/.bashrc
conda activate llava

# Correct CUDA paths
CUDA_PATH=/opt/gridware/depots/761a7df9/el7/pkg/libs/nvidia-cuda/11.8.0
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/users/jjls2000/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Set PYTHONPATH to include llava directory
export PYTHONPATH=/users/jjls2000/sharedscratch/Dissertation:$PYTHONPATH

# Verify nvcc exists
if [ -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "nvcc found at $CUDA_HOME/bin/nvcc"
else
    echo "nvcc not found in $CUDA_HOME/bin"
    exit 1
fi

# Check GPU availability and details
nvidia-smi

# Verify DeepSpeed config file
if [ -f "/users/jjls2000/sharedscratch/Dissertation/scripts/zero3.json" ]; then
    echo "DeepSpeed config found"
else
    echo "DeepSpeed config not found"
    exit 1
fi

# Verify model checkpoint
if [ -f "/users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-med-v1.5-mistral-7b/pytorch_model.bin" ]; then
    echo "Model checkpoint found"
else
    echo "Model checkpoint not found"
    exit 1
fi

# Verify data path
if [ -f "/users/jjls2000/sharedscratch/Dissertation/Slake1.0/augmented/BBF_train.json" ]; then
    echo "Data path found"
else
    echo "Data path not found"
    exit 1
fi

# Verify image folder
if [ -d "/users/jjls2000/sharedscratch/Dissertation/data/imgs-1" ]; then
    echo "Image folder found"
else
    echo "Image folder not found"
    exit 1
fi

# Run the training script with deepspeed
deepspeed /users/jjls2000/sharedscratch/Dissertation/llava/train/train_mem.py \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --deepspeed /users/jjls2000/sharedscratch/Dissertation/scripts/zero3.json \
    --model_name_or_path /users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-med-v1.5-mistral-7b \
    --version llava_v1.5 \
    --data_path /users/jjls2000/sharedscratch/Dissertation/Slake1.0/augmented/BBF_train.json \
    --image_folder /users/jjls2000/sharedscratch/Dissertation/data/imgs-1 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter '/users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-v1.5-7b/mm_projector_extracted/mm_projector/data.pkl' \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /users/jjls2000/sharedscratch/Dissertation/results/20240709_214950 \
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
