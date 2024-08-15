import json

# Paths to the files
answer_files = {
    "BBF": '/users/jjls2000/sharedscratch/Dissertation/results/final/slake/slakeanswerBBF-AccuracyRecall.jsonl',
    "BBL": '/users/jjls2000/sharedscratch/Dissertation/results/final/slake/slakeanswerBBL-AccuracyRecall.jsonl'
}
gold_standard_file = '/users/jjls2000/sharedscratch/Dissertation/results/final/slake/new_gold_standard.json'

# Function to load JSONL answer data
def load_jsonl_data(filepath):
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

# Function to load the gold standard data
def load_gold_standard(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# Function to compare answers with the gold standard
def compare_answers(gold_standard_data, answer_data):
    discrepancies = []
    matches = 0
    total = 0

    gold_dict = {item['question_id']: item for item in gold_standard_data}

    for answer in answer_data:
        qid = answer['question_id']
        if qid in gold_dict:
            total += 1
            expected_answer = gold_dict[qid]['answer'].strip().lower()
            predicted_answer = answer['text'].strip().lower()

            if expected_answer == predicted_answer:
                matches += 1
            else:
                discrepancies.append({
                    "question_id": qid,
                    "question": gold_dict[qid]['question'],
                    "expected_answer": expected_answer,
                    "predicted_answer": predicted_answer
                })

    return matches, total, discrepancies

# Main script execution
if __name__ == "__main__":
    try:
        # Load the gold standard data
        gold_standard_data = load_gold_standard(gold_standard_file)

        for model, answer_file in answer_files.items():
            print(f"Comparing {model} model with gold standard...")

            # Load the answer data
            answer_data = load_jsonl_data(answer_file)

            # Compare answers
            matches, total, discrepancies = compare_answers(gold_standard_data, answer_data)

            # Print results
            print(f"\n{model} Model - Matches: {matches}/{total}, Accuracy: {(matches/total) * 100:.2f}%")
            print(f"Number of discrepancies: {len(discrepancies)}\n")

            # Print a few discrepancies for review
            if discrepancies:
                print("Sample discrepancies:")
                for discrepancy in discrepancies[:5]:  # Print first 5 discrepancies
                    print(f"Question ID: {discrepancy['question_id']}")
                    print(f"Question: {discrepancy['question']}")
                    print(f"Expected Answer: {discrepancy['expected_answer']}")
                    print(f"Predicted Answer: {discrepancy['predicted_answer']}")
                    print("-" * 40)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
