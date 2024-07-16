#!/bin/bash
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
#SBATCH --time=01:00:00
## Memory limit (in megabytes)
#SBATCH --mem=32G
## GPU requirements
#SBATCH --gres=gpu:1
## Specify partition
#SBATCH -p gpu

# Load CUDA module
source /opt/flight/etc/setup.sh
module load libs/nvidia-cuda/11.8.0/bin

# Activate conda environment
source activate llava

# Install CUDA toolkit via conda
conda install -c conda-forge cudatoolkit=11.8 -y

# Install bitsandbytes
pip install bitsandbytes-cuda118

# Verify bitsandbytes installation
wget https://gist.githubusercontent.com/TimDettmers/1f5188c6ee6ed69d211b7fe4e381e713/raw/4d17c3d09ccdb57e9ab7eca0171f2ace6e4d2858/check_bnb_install.py
python check_bnb_install.py

echo "Bits and Bytes installation and verification completed."
