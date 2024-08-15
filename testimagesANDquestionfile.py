import json
import os

# Define the paths
image_folder = '/users/jjls2000/sharedscratch/Dissertation/data/imgs-1'
question_file = '/users/jjls2000/sharedscratch/Dissertation/Slake1.0/test_questions_fixed.jsonl'

# Load the main question file
with open(question_file, 'r') as f:
    main_questions = [json.loads(line) for line in f]

# Create a dictionary to map question IDs to their texts from the main question file
main_question_dict = {q['question_id']: q['text'] for q in main_questions}

# Initialize counters and mismatch list
total_checks = 0
mismatches = []

# Iterate through each image directory
for root, dirs, files in os.walk(image_folder):
    print(f"Checking directory: {root}")  # Debugging output

    if 'question.json' in files:
        print(f"Found question.json in {root}")  # Debugging output
        question_json_path = os.path.join(root, 'question.json')

        # Load the question.json file
        with open(question_json_path, 'r') as f:
            img_questions = json.load(f)

        # Debugging output: Print the questions in the question.json
        print(f"Questions in {root}:")
        for q in img_questions:
            print(f"  - Question ID: {q.get('qid')}, Text: {q.get('question')}")

        # Check if each question matches the corresponding question in the main file
        for img_question in img_questions:
            question_id = img_question.get('qid')
            question_text = img_question.get('question')

            # Compare with the main question file
            if question_id in main_question_dict:
                total_checks += 1
                if main_question_dict[question_id] != question_text:
                    mismatches.append({
                        'question_id': question_id,
                        'expected_question': main_question_dict[question_id],
                        'found_question': question_text,
                        'image_directory': root
                    })
            else:
                print(f"Question ID {question_id} not found in main question file.")

# Report results
print(f"Total checks performed: {total_checks}")
if mismatches:
    print(f"Found {len(mismatches)} mismatches:")
    for mismatch in mismatches:
        print(f"Question ID: {mismatch['question_id']}")
        print(f"Expected: {mismatch['expected_question']}")
        print(f"Found: {mismatch['found_question']}")
        print(f"Image Directory: {mismatch['image_directory']}")
        print("-" * 50)
else:
    print("All questions match with the corresponding questions in the main file.")
