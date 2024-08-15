import json

def load_jsonl_file(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def create_gold_standard(question_file, reference_file, output_file):
    # Load the question file (JSONL) and the reference test file (JSON)
    questions = load_jsonl_file(question_file)
    reference_data = load_json_file(reference_file)

    # Create a lookup dictionary for reference data based on question_id
    reference_lookup = {entry['qid']: entry for entry in reference_data}

    gold_standard = []

    for question in questions:
        question_id = question['question_id']

        if question_id in reference_lookup:
            ref_entry = reference_lookup[question_id]

            # Extract the necessary fields
            gold_entry = {
                "question_id": question_id,
                "question": ref_entry['question'],
                "answer": ref_entry['answer'],
                "answer_type": ref_entry['answer_type'],
                "bbox": ref_entry.get('bbox', [])  # Include bounding box if available
            }

            gold_standard.append(gold_entry)
        else:
            print(f"Warning: Question ID {question_id} not found in the reference file.")

    # Save the new gold standard to the output file
    with open(output_file, 'w') as outfile:
        json.dump(gold_standard, outfile, indent=4)

    print(f"Gold standard test file created at {output_file}")

def load_json_file(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    question_file = '/users/jjls2000/sharedscratch/Dissertation/Slake1.0/test_questions_fixed.jsonl'
    reference_file = '/users/jjls2000/sharedscratch/Dissertation/Slake1.0/test.json'
    output_file = '/users/jjls2000/sharedscratch/Dissertation/results/final/slake/new_gold_standard.json'

    create_gold_standard(question_file, reference_file, output_file)
