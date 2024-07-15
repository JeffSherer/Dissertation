#!/bin/bash

# Activate the Conda environment
source /opt/flight/etc/setup.sh
flight env activate gridware
conda activate llavamed_new

# Check Python version
echo "Python version:"
python --version

# Check transformers version
echo "Transformers version:"
python -c "import transformers; print(transformers.__version__)"

# Check tokenizers version
echo "Tokenizers version:"
python -c "import tokenizers; print(tokenizers.__version__)"

# Check if LlamaConfig can be imported
echo "Checking LlamaConfig import:"
python -c "from transformers import LlamaConfig; print('LlamaConfig Import Successful')"

# Check CUDA environment variables
echo "CUDA_HOME:"
echo $CUDA_HOME

echo "CUDA version:"
nvcc --version

# List paths in the environment
echo "Python path:"
python -c "import sys; print(sys.path)"

# List all installed packages in the Conda environment
echo "List of installed packages in Conda environment:"
conda list

# Check if llava module is in the Python path
echo "Checking if llava module is in Python path:"
python -c "import llava; print('llava module found')"

# List directories to check for llava path issues
echo "Listing llava directories:"
ls -l /users/jjls2000/sharedscratch/Dissertation/llava
ls -l /users/jjls2000/sharedscratch/Dissertation/llava/model
ls -l /mnt/scratch/users/jjls2000/Dissertation/llava

echo "All checks completed."
