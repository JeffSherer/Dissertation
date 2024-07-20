#!/bin/bash

#SBATCH -D /users/jjls2000/sharedscratch/Dissertation
#SBATCH --export=ALL
#SBATCH -o /users/jjls2000/sharedscratch/Dissertation/job-%j.output
#SBATCH -e /users/jjls2000/sharedscratch/Dissertation/job-%j.error
#SBATCH -J evaluation-job
#SBATCH --time=02:00:00
#SBATCH --mem=2048
#SBATCH --partition=gpu

source /opt/gridware/depots/761a7df9/el9/pkg/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate llava-med

# Set your OpenAI API key here
export OPENAI_API_KEY="your-ope*******-key"

python /users/jjls2000/sharedscratch/Dissertation/llava/eval/eval_multimodal_chat_gpt_score.py --answers-file /users/jjls2000/sharedscratch/Dissertation/results/llava_med_eval_answers.jsonl --question-file /users/jjls2000/sharedscratch/Dissertation/data/eval/llava_med_eval_qa50_qa.jsonl --scores-file /users/jjls2000/sharedscratch/Dissertation/results/eval_scores.jsonl
