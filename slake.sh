#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D /users/jjls2000/sharedscratch/Dissertation
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
#SBATCH -o /users/jjls2000/sharedscratch/Dissertation/llava-med-test-%j.out
## Error File
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

# Ensure CUDA paths are correct
export CUDA_HOME=/opt/gridware/depots/761a7df9/el9/pkg/libs/nvidia-cuda/11.8.0/bin
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/users/jjls2000/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Verify nvcc exists
if [ -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "nvcc found at $CUDA_HOME/bin/nvcc"
else
    echo "nvcc not found in $CUDA_HOME/bin"
    exit 1
fi

# Set PYTHONPATH to include llava directory
export PYTHONPATH=/users/jjls2000/sharedscratch/Dissertation:$PYTHONPATH

# Check GPU availability and details
nvidia-smi

# Additional diagnostic commands
echo "Running on node(s): $SLURM_JOB_NODELIST"
echo "Using GPU device(s): $CUDA_VISIBLE_DEVICES"

# Run the training script with torchrun
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    /users/jjls2000/sharedscratch/Dissertation/llava/train/train_mem.py \
    --model_name_or_path /users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-med-7b-slake-delta \
    --data_path /users/jjls2000/sharedscratch/Dissertation/Slake1.0/augmented/BBF_train.json \
    --image_folder /users/jjls2000/sharedscratch/Dissertation/data/imgs-1 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir /users/jjls2000/sharedscratch/Dissertation/results/20240709_214950 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
