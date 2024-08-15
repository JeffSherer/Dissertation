import json
import random

# Paths to the files
answer_files = {
    "BBL": '/users/jjls2000/sharedscratch/Dissertation/results/final/slake/slakeanswerBBL-BBOX_100.jsonl'
}
gold_standard_file = '/users/jjls2000/sharedscratch/Dissertation/results/final/slake/new_gold_standard.json'

# Function to load the answer data
def load_answer_data(answer_file):
    with open(answer_file, 'r') as f:
        return [json.loads(line) for line in f]

# Function to load the gold standard data
def load_gold_standard(gold_file):
    with open(gold_file, 'r') as f:
        return json.load(f)

# Function to print sample answers
def print_sample_answers(answer_data, gold_data, num_samples=5):
    # Find the key used for question IDs in the gold standard data
    sample_gold_entry = gold_data[0]
    qid_key = 'qid' if 'qid' in sample_gold_entry else 'question_id'
    
    gold_dict = {item[qid_key]: item for item in gold_data}
    sampled_answers = random.sample(answer_data, num_samples)
    
    for answer in sampled_answers:
        qid = answer['question_id']
        gold_answer = gold_dict.get(qid, {})
        
        print(f"Question ID: {qid}")
        print(f"Question: {gold_answer.get('question', 'N/A')}")
        print(f"Model Answer: {answer['text']}")
        print(f"Gold Standard Answer: {gold_answer.get('answer', 'N/A')}")
        print("-" * 50)

# Main script execution
if __name__ == "__main__":
    try:
        # Load the gold standard data
        gold_data = load_gold_standard(gold_standard_file)

        for model, answer_file in answer_files.items():
            print(f"Sampling answers from {model} model...\n")

            # Load the answer data
            answer_data = load_answer_data(answer_file)

            # Print sample answers
            print_sample_answers(answer_data, gold_data)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
