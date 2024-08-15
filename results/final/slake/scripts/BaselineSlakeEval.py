import json
import argparse
from transformers import pipeline

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Function to load the answer data (assuming JSONL format)
def load_answer_data(answer_file):
    with open(answer_file, 'r') as f:
        return [json.loads(line) for line in f]

# Function to load the gold standard data (assuming JSONL format)
def load_gold_standard_data(gold_standard_file):
    with open(gold_standard_file, 'r') as f:
        return [json.loads(line) for line in f]

# Function to calculate sentiment similarity
def calculate_sentiment_similarity(text1, text2):
    sentiment1 = sentiment_pipeline(text1)[0]['label']
    sentiment2 = sentiment_pipeline(text2)[0]['label']
    return sentiment1 == sentiment2

# Calculate accuracy and recall with sentiment analysis
def evaluate_accuracy_recall(answer_data, gold_standard_data):
    correct_answers = 0
    total_questions = 0
    closed_correct = 0
    closed_total = 0
    open_recall = 0
    open_total = 0

    gold_standard_dict = {item['question_id']: item for item in gold_standard_data}

    for answer in answer_data:
        qid = answer['question_id']
        if qid in gold_standard_dict:
            total_questions += 1

            print(f"Processing QID: {qid}")

            gold_standard_entry = gold_standard_dict[qid]
            gold_standard_answer = gold_standard_entry.get('answer', '').strip().lower()
            answer_text = answer.get('text', '').strip().lower()
            answer_type = gold_standard_entry.get('answer_type', '')

            # Debugging: Output the relevant data being compared
            print(f"Gold Standard Answer: {gold_standard_answer}")
            print(f"Answer Text: {answer_text}")
            print(f"Answer Type: {answer_type}")

            if not gold_standard_answer:
                print(f"Missing gold standard answer for QID {qid}")
                continue  # Skip this entry if the gold standard answer is missing

            if answer_type == "CLOSED":
                closed_total += 1
                if gold_standard_answer == answer_text or calculate_sentiment_similarity(gold_standard_answer, answer_text):
                    closed_correct += 1
                    correct_answers += 1

            elif answer_type == "OPEN":
                open_total += 1
                if gold_standard_answer in answer_text or calculate_sentiment_similarity(gold_standard_answer, answer_text):
                    open_recall += 1
                    correct_answers += 1

            # Also add to correct_answers for open-ended questions that match
            if gold_standard_answer == answer_text or calculate_sentiment_similarity(gold_standard_answer, answer_text):
                correct_answers += 1

    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    closed_accuracy = closed_correct / closed_total if closed_total > 0 else 0
    open_recall_rate = open_recall / open_total if open_total > 0 else 0

    return accuracy, closed_accuracy, open_recall_rate

# Main script execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate accuracy and recall based on the gold standard.")
    parser.add_argument('--answer-file', type=str, required=True, help='Path to the answer file')
    parser.add_argument('--gold-standard-file', type=str, required=True, help='Path to the gold standard file')

    args = parser.parse_args()

    try:
        # Load the gold standard data
        gold_standard_data = load_gold_standard_data(args.gold_standard_file)

        print(f"Evaluating model using answer file: {args.answer_file}")

        # Load the answer data
        answer_data = load_answer_data(args.answer_file)

        # Evaluate accuracy and recall
        accuracy, closed_accuracy, open_recall_rate = evaluate_accuracy_recall(answer_data, gold_standard_data)

        print(f"Overall Accuracy: {accuracy:.2%}")
        print(f"Closed-Ended Questions Accuracy: {closed_accuracy:.2%}")
        print(f"Open-Ended Questions Recall: {open_recall_rate:.2%}")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

