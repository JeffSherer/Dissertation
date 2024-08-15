import json

def load_jsonl_file(jsonl_file):
    with open(jsonl_file, 'r') as f:
        return [json.loads(line) for line in f]

def load_json_file(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def compare_answers(accuracy_file, gold_standard_file):
    # Load the data
    accuracy_data = load_jsonl_file(accuracy_file)
    gold_standard_data = load_json_file(gold_standard_file)

    # Create a lookup dictionary for the gold standard answers
    gold_standard_dict = {entry['qid']: entry for entry in gold_standard_data}

    discrepancies = []

    for entry in accuracy_data:
        qid = entry['question_id']
        if qid in gold_standard_dict:
            gold_standard_answer = gold_standard_dict[qid]['answer'].strip().lower()
            model_answer = entry['text'].strip().lower()

            if gold_standard_answer != model_answer:
                discrepancies.append({
                    "question_id": qid,
                    "gold_standard_answer": gold_standard_answer,
                    "model_answer": model_answer
                })

    return discrepancies

if __name__ == "__main__":
    accuracy_file = '/users/jjls2000/sharedscratch/Dissertation/results/final/slake/slakeanswerBBF-AccuracyRecall_100.jsonl'
    gold_standard_file = '/users/jjls2000/sharedscratch/Dissertation/Slake1.0/new_filtered_gold_standard.json'
    
    discrepancies = compare_answers(accuracy_file, gold_standard_file)

    if discrepancies:
        print(f"Found {len(discrepancies)} discrepancies between accuracy file and gold standard:")
        print("Example discrepancy:")
        print(json.dumps(discrepancies[0], indent=4))
    else:
        print("No discrepancies found between the accuracy file and the gold standard file.")
