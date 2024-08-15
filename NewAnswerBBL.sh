#!/bin/bash

## SBATCH directives begin here

## Working directory
#SBATCH -D /users/jjls2000/sharedscratch/Dissertation

## Environment variables
#SBATCH --export=ALL

## Output and Error Files
#SBATCH -o /users/jjls2000/sharedscratch/Dissertation/job-%j.output
#SBATCH -e /users/jjls2000/sharedscratch/Dissertation/job-%j.error

## Job name
#SBATCH -J evaluation-job

## Run time: "hours:minutes:seconds"
#SBATCH --time=10:00:00

## Memory limit (in gigabytes)
#SBATCH --mem=32G

## GPU requirements
#SBATCH --gres=gpu:1  # Requesting 1 GPU

## Specify partition
#SBATCH --partition=gpu

## Load any necessary modules (e.g., Python, CUDA, etc.)
# module load python/3.8
# module load cuda/11.1

## Activate your conda environment if needed
source activate llava_test_env

## Run your Python script
python NewSlakeAnswerGeneration.py \
    --model-path /users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-med-v1.5-mistral-7b-BBL-3-fixed \
    --image-folder /users/jjls2000/sharedscratch/Dissertation/data/imgs-1 \
    --question-file /users/jjls2000/sharedscratch/Dissertation/Slake1.0/test_questions_fixed.jsonl \
    --answers-file /users/jjls2000/sharedscratch/Dissertation/results/final/slake/new_slake_answers-BBL.jsonl
