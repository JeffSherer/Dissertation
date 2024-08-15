import argparse
from collections import defaultdict
import pandas as pd
import json
import util  # Ensure util.py is available in the same directory or in the PYTHONPATH

# Function to get the domain of the question
def get_domain(x):
    for domain in ['chest_xray', 'mri', 'histology', 'gross', 'ct_scan']:
        if domain in x:  # Check if the domain is present
            return domain
    return 'unknown'

# Main function to process the scores file
def main(args):
    scores_data = util.load_file_jsonl(args.scores_file)
    predictions = []
    
    # Collect entries and check for missing keys
    for i, x in enumerate(scores_data):
        try:
            question_id = x.get('question_id', f"default_id_{i}")
            question_type = x.get('type', 'unknown')
            domain = get_domain(x)
            gpt_eval = x['gpt_eval'].split('\n')[0].split(' ')
            predictions.append((question_id, question_type, domain, gpt_eval))
        except KeyError as e:
            print(f"Missing key {e} in entry {i}: {x}")
            continue  # Skip entries with missing keys
    
    if not predictions:
        print("No valid predictions found. Exiting.")
        return

    score_type_dict = defaultdict(lambda: defaultdict(list))
    for q_id, q_type, domain, (a1_score, a2_score) in predictions:
        score_type_dict[q_type][1].append(a1_score)
        score_type_dict[q_type][2].append(a2_score)
        score_type_dict['overall'][1].append(a1_score)
        score_type_dict['overall'][2].append(a2_score)
        if domain:  # Ensure domain is not None
            score_type_dict[domain][1].append(a1_score)
            score_type_dict[domain][2].append(a2_score)

    result = defaultdict(dict)

    for q_type, score_dict in score_type_dict.items():
        result[q_type]['gpt4_score'] = util.get_avg(score_dict[1])
        result[q_type]['pred_score'] = util.get_avg(score_dict[2])
        result[q_type]['pred_relative_score'] = util.get_avg([float(s2)/float(s1) for s1, s2 in zip(score_dict[1], score_dict[2])]) * 100
        result[q_type]['data_size'] = len(score_dict[1])

    df = pd.DataFrame.from_dict(result, orient='index').filter(['gpt4_score', 'pred_score', 'pred_relative_score', 'data_size'])
    print(df)

# Entry point of the script
if __name__ == '__main__':
    parser = argparse.ArgumentParser("GPT-4 Multimodal Chat Eval Postprocessing", add_help=True)
    parser.add_argument("--scores-file", default="", metavar="FILE", help="input path to gpt-4 score file")
    args = parser.parse_args()
    main(args)
