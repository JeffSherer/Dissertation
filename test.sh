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
flight env activate gridware
module load libs/nvidia-cuda/11.8.0/bin

# Correct path for conda.sh
source /users/jjls2000/anaconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate llava

# Install CUDA toolkit via conda
conda install -c conda-forge cudatoolkit=11.8 -y

# Install bitsandbytes for the correct CUDA version
pip install bitsandbytes

# Verify bitsandbytes installation
echo "Verifying bitsandbytes installation"
python -c "import bitsandbytes as bnb; print(bnb.__version__); adam = bnb.optim.Adam([bnb.nn.Parameter([1.0, 2.0, 3.0])]); print('bitsandbytes is successfully installed and functioning')"

echo "Bits and Bytes installation and verification completed."
