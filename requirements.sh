#!/bin/bash
#SBATCH --job-name=llava-setup
#SBATCH --output=llava-setup.out
#SBATCH --error=llava-setup.err
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# Load necessary modules (if any)
module load anaconda/2023.03

# Create and activate conda environment
conda create -n llava python=3.10 -y
source activate llava

# Install required packages
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install gradio==4.16.0 gradio-client==0.8.1 timm==0.6.13 transformers==4.37.2
pip install accelerate==0.21.0 bitsandbytes==0.39.0 einops==0.6.1 peft==0.2.0 joblib==1.3.1 scipy==1.10.1 threadpoolctl==2.2.0
pip install huggingface_hub==0.23.4 requests==2.32.3 charset-normalizer==3.3.2

# Verify installation (optional, for debugging purposes)
python -c "import torch; print('torch:', torch.__version__); import torchvision; print('torchvision:', torchvision.__version__); import torchaudio; print('torchaudio:', torchaudio.__version__); import gradio; print('gradio:', gradio.__version__); import gradio_client; print('gradio_client:', gradio_client.__version__); import timm; print('timm:', timm.__version__); import transformers; print('transformers:', transformers.__version__)"

# Run your training script
python /users/jjls2000/sharedscratch/Dissertation/llava/train/train_mem.py
