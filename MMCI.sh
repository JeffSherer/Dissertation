#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working directory
#SBATCH -D /users/jjls2000/sharedscratch/Dissertation
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
#SBATCH -o job-%j.output
#SBATCH -e job-%j.error
## Job name
#SBATCH -J eval-multimodal-chat-gpt
## Run time: "hours:minutes:seconds"
#SBATCH --time=01:00:00
## Memory limit (in megabytes)
#SBATCH --mem=4096
## Specify partition
#SBATCH -p nodes

################# Part-2 Shell script ####################
#===========================
#  Application launch commands
#---------------------------
echo "Starting evaluation job"

python eval_multimodal_chat_gpt_score.py \
    --answers-file /users/jjls2000/sharedscratch/Dissertation/results/llava_med_eval_answers.jsonl \
    --question-file /users/jjls2000/sharedscratch/Dissertation/data/eval/llava_med_eval_qa50_qa.jsonl \
    --scores-file /users/jjls2000/sharedscratch/Dissertation/results/eval_scores.jsonl

echo "Evaluation job completed"
