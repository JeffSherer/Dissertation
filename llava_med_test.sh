#!/bin/bash -l

################# Part-1 Slurm directives ####################
#SBATCH -D /users/jjls2000/sharedscratch/Dissertation
#SBATCH --export=ALL
#SBATCH -o llava-med-test-%j.out
#SBATCH -e llava-med-test-%j.err
#SBATCH -J slake-test
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p gpu

# Source the common setup script
source /users/jjls2000/sharedscratch/Dissertation/common_setup.sh
#!/bin/bash -l

################# Part-1 Slurm directives ####################
#SBATCH -D /users/jjls2000/sharedscratch/Dissertation
#SBATCH --export=ALL
#SBATCH -o llava-med-test-%j.out
#SBATCH -e llava-med-test-%j.err
#SBATCH -J slake-test
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p gpu


# Define the experiment name
EXPERIMENT_NAME="slake_test_$(date +%Y%m%d_%H%M%S)"

# Ensure the results directory exists
RESULTS_DIR="/users/jjls2000/sharedscratch/Dissertation/results/${EXPERIMENT_NAME}"
mkdir -p "${RESULTS_DIR}"

# Run the test script with SLaKE checkpoint
python /users/jjls2000/sharedscratch/LLaVA-Med/llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --model-path /users/jjls2000/sharedscratch/Dissertation/checkpoints/slake \
    --question-file /users/jjls2000/LLaVA-Med/data/eval/llava_med_eval_qa50_qa.jsonl \
    --image-folder /users/jjls2000/sharedscratch/Dissertation/data/images \
    --answers-file ${RESULTS_DIR}/answer-file-${SLURM_JOB_ID}.jsonl \
    --temperature 0.0

# Git operations (Ensure SSH keys are set for non-interactive authentication)
cd /users/jjls2000/sharedscratch/Dissertation
git add ${RESULTS_DIR}/answer-file-${SLURM_JOB_ID}.jsonl
git commit -m "Add output for job ${SLURM_JOB_ID}"
git push origin main

# Activate the conda environment
activate_env llavamed

# Define the experiment name
EXPERIMENT_NAME="slake_test_$(date +%Y%m%d_%H%M%S)"

# Create new experiment directory and organize initial results
RESULTS_DIR=$(create_experiment_dir $EXPERIMENT_NAME "/users/jjls2000/sharedscratch/Dissertation")

# Run the test script with SLaKE checkpoint

python /users/jjls2000/sharedscratch/LLaVA-Med/llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --model-path /users/jjls2000/sharedscratch/Dissertation/checkpoints/slake \
    --question-file /users/jjls2000/LLaVA-Med/data/eval/llava_med_eval_qa50_qa.jsonl \
    --image-folder /users/jjls2000/sharedscratch/Dissertation/data/images \
    --answers-file ${RESULTS_DIR}/answer-file-${SLURM_JOB_ID}.jsonl \
    --temperature 0.0


    # Add output files to Git repository
git add /users/jjls2000/sharedscratch/Dissertation/results/${EXPERIMENT_NAME}/answer-file-${SLURM_JOB_ID}.jsonl

# Commit the changes with a message containing job ID
git commit -m "Add output for job ${SLURM_JOB_ID}"

# Push changes to GitHub
git push origin main
