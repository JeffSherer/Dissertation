import json
import os

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def combine_files(file_paths):
    combined_data = []
    for file_path in file_paths:
        data = read_json(file_path)
        combined_data.extend(data)
    return combined_data

# Define paths
base_dir = '/users/jjls2000/sharedscratch/Dissertation/Slake1.0/augmented'
output_dir = '/users/jjls2000/sharedscratch/Dissertation/Slake1.0/augmented'

# BBL dataset files
BBL_train_file = os.path.join(base_dir, 'BBL_train.json')
BBL_validate_file = os.path.join(base_dir, 'BBL_validate.json')
BBL_test_file = os.path.join(base_dir, 'BBL_test.json')

# BBF dataset files
BBF_train_file = os.path.join(base_dir, 'BBF_train.json')
BBF_validate_file = os.path.join(base_dir, 'BBF_validate.json')
BBF_test_file = os.path.join(base_dir, 'BBF_test.json')

# Combine train and validate for medium datasets
BBL_train_medium = combine_files([BBL_train_file, BBL_validate_file])
BBF_train_medium = combine_files([BBF_train_file, BBF_validate_file])

# Combine train, validate, and test for large datasets
BBL_train_large = combine_files([BBL_train_file, BBL_validate_file, BBL_test_file])
BBF_train_large = combine_files([BBF_train_file, BBF_validate_file, BBF_test_file])

# Write combined data to new files
write_json(BBL_train_medium, os.path.join(output_dir, 'BBL_train_medium.json'))
write_json(BBF_train_medium, os.path.join(output_dir, 'BBF_train_medium.json'))
write_json(BBL_train_large, os.path.join(output_dir, 'BBL_train_large.json'))
write_json(BBF_train_large, os.path.join(output_dir, 'BBF_train_large.json'))

print("Datasets have been successfully combined and saved.")
