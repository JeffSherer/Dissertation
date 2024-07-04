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

# Load the required modules or set up environment variables
export CUDA_HOME=/opt/gridware/depots/761a7df9/el9/pkg/libs/nvidia-cuda/11.8.0
export PATH=$CUDA_HOME/bin:$PATH

# Set Triton cache directory to a non-NFS path
export TRITON_CACHE_DIR=/users/jjls2000/triton_cache
mkdir -p $TRITON_CACHE_DIR

# Ensure the Python script can find the module
export PYTHONPATH="/users/jjls2000/sharedscratch/Dissertation:${PYTHONPATH}"

# Activate the Conda environment
source /opt/gridware/depots/761a7df9/el9/pkg/apps/anaconda3/2023.03/bin/activate llavamed_new

# Print environment setup for debugging
echo "Using Python from: $(which python)"
python -c "import transformers; print('Transformers version:', transformers.__version__)"
python -c "from transformers import LlamaConfig; print('LlamaConfig imported successfully')"

################# Part-3 Execute Fine-Tuning Script ####################

# Use the absolute path to the Python interpreter in your Conda environment
/users/jjls2000/.conda/envs/llavamed_new/bin/deepspeed /users/jjls2000/sharedscratch/Dissertation/llava/train/train_mem.py \
    --lora_enable True \
    --deepspeed /users/jjls2000/sharedscratch/Dissertation/scripts/zero2.json \
    --model_name_or_path "microsoft/llava-med-v1.5-mistral-7b" \
    --version llava_med_v1.5 \
    --data_path "/users/jjls2000/sharedscratch/Dissertation/Slake1.0/augmented/BBF_train.json" \
    --image_folder "/users/jjls2000/sharedscratch/Dissertation/data/imgs-1" \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter "/users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-llava-llavammed-7b-pretrain/mm_projector.bin" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "/users/jjls2000/sharedscratch/Dissertation/results/$(date +%Y%m%d_%H%M%S)" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-
