import os
import shutil
import sys
from datetime import datetime

def create_experiment_directory(base_path, experiment_name):
    # Create the new experiment directory structure
    experiment_path = os.path.join(base_path, experiment_name)
    logs_path = os.path.join(experiment_path, "logs")
    os.makedirs(logs_path, exist_ok=True)
    print(f"Created directories for {experiment_name}")

    return experiment_path, logs_path

def move_files_to_experiment(experiment_path, logs_path, result_files, log_files):
    # Move result files
    for file in result_files:
        if os.path.exists(file):
            shutil.move(file, experiment_path)
            print(f"Moved {file} to {experiment_path}")
        else:
            print(f"Warning: {file} not found and was not moved.")

    # Move log files
    for file in log_files:
        if os.path.exists(file):
            shutil.move(file, logs_path)
            print(f"Moved {file} to {logs_path}")
        else:
            print(f"Warning: {file} not found and was not moved.")

def main():
    # Base path for experiments
    base_path = os.path.join(os.getcwd(), "results")
    
    # Get experiment name from command line argument
    if len(sys.argv) < 2:
        print("Usage: python manage_experiments.py <experiment_name>")
        sys.exit(1)
    
    experiment_name = sys.argv[1]
    
    # Define result and log files to move
    result_files = ["answer-file.jsonl"]
    log_files = ["llava-med-test.out", "llava-med-test.err"]
    
    # Create new experiment directory
    experiment_path, logs_path = create_experiment_directory(base_path, experiment_name)
    
    # Move result and log files to the new experiment directory
    move_files_to_experiment(experiment_path, logs_path, result_files, log_files)
    print(f"Experiment {experiment_name} organized successfully.")

if __name__ == "__main__":
    main()
