import json
import os
import re

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def contains_chinese(text):
    # Check if the text contains Chinese characters
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def filter_samples(data):
    filtered_data = []
    for item in data:
        keep = True
        for conversation in item.get('conversations', []):
            if contains_chinese(conversation['value']):
                keep = False
                break
        if keep:
            filtered_data.append(item)
    return filtered_data

# Define paths
base_dir = '/users/jjls2000/sharedscratch/Dissertation/Slake1.0/augmented'
output_dir = '/users/jjls2000/sharedscratch/Dissertation/Slake1.0/augmented_filtered'

os.makedirs(output_dir, exist_ok=True)

# Dataset files
files = [
    'BBL_train.json', 'BBF_train.json', 'BBL_train_medium.json', 'BBF_train_medium.json',
    'BBL_train_large.json', 'BBF_train_large.json', 'BBL_validate.json', 'BBF_validate.json',
    'BBL_test.json', 'BBF_test.json'
]

for file_name in files:
    input_file_path = os.path.join(base_dir, file_name)
    output_file_path = os.path.join(output_dir, file_name)
    
    # Read the data
    data = read_json(input_file_path)
    
    # Filter the data
    filtered_data = filter_samples(data)
    
    # Write the filtered data
    write_json(filtered_data, output_file_path)

print("Filtered datasets have been successfully created and saved.")
