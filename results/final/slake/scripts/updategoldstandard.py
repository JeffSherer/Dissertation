import json

def align_gold_standard_structure(gold_standard_file, output_file):
    # Load the gold standard data
    with open(gold_standard_file, 'r') as f:
        gold_standard_data = json.load(f)

    aligned_data = []
    for entry in gold_standard_data:
        aligned_entry = {
            "question_id": entry["qid"],  # Align question_id with qid
            "prompt": entry["question"],  # Use the question as the prompt
            "text": entry["answer"],  # Use the answer as the text
            "answer_id": "",  # Leave answer_id empty as it's not needed for evaluation
            "model_id": "",  # Leave model_id empty as it's not needed for evaluation
            "metadata": {}  # Include an empty metadata dictionary
        }
        aligned_data.append(aligned_entry)

    # Save the aligned data to the output file
    with open(output_file, 'w') as f:
        for entry in aligned_data:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    gold_standard_file = '/users/jjls2000/sharedscratch/Dissertation/Slake1.0/new_filtered_gold_standard.json'
    output_file = '/users/jjls2000/sharedscratch/Dissertation/Slake1.0/aligned_gold_standard.jsonl'

    align_gold_standard_structure(gold_standard_file, output_file)

    print(f"Aligned gold standard file saved to {output_file}")
