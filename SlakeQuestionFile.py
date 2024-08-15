import json
import os
import shutil

# Paths
input_json_path = "/mnt/scratch/users/jjls2000/Dissertation/Slake1.0/test.json"
question_file_path = "/mnt/scratch/users/jjls2000/Dissertation/Slake1.0/test_questions.json"
image_base_path = "/mnt/scratch/users/jjls2000/Dissertation/data/imgs-1/"
test_images_path = "/mnt/scratch/users/jjls2000/Dissertation/data/test_images/"

# Ensure the test images directory exists
os.makedirs(test_images_path, exist_ok=True)

# Clear out old test images
for root, dirs, files in os.walk(test_images_path):
    for file in files:
        os.remove(os.path.join(root, file))

# Load the JSON data
with open(input_json_path, 'r') as f:
    data = json.load(f)

# Filter out non-English questions and create question data
question_data = []
for item in data:
    if item['q_lang'] == 'en':
        image_path = os.path.join(image_base_path, item['img_name'])

        # Ensure the subdirectory exists in test_images_path
        subdirectory = os.path.join(test_images_path, os.path.dirname(item['img_name']))
        os.makedirs(subdirectory, exist_ok=True)

        # Copy the original image to the test_images directory
        shutil.copy(image_path, os.path.join(test_images_path, item['img_name']))

        # Create a new entry for the question
        new_item = {
            "question_id": item['qid'],
            "image": item['img_name'],
            "pair_id": item['qid'],
            "text": item['question'],
            "answer_type": item['answer_type'],  # Include answer_type for evaluation later
            "domain": {
                "ct_scan": item['modality'].lower() == 'ct',
                "mri": item['modality'].lower() == 'mri',
                "chest_xray": item['modality'].lower() == 'x-ray'
            },
            "type": "conversation"
        }
        question_data.append(new_item)

# Save the question file
with open(question_file_path, 'w') as f:
    json.dump(question_data, f, indent=4)

print(f"Question file has been saved to {question_file_path}")
