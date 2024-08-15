import argparse
from collections import defaultdict
import pandas as pd
import util

def main(args):
    scores_data = util.load_file_jsonl(args.scores_file)

    # Extract relevant information from the scores_data
    predictions = [
        (x['question'], x['gpt_eval'].split('\n')[0].split(' '))
        for x in scores_data
    ]

    score_type_dict = defaultdict(lambda: defaultdict(list))

    # Populate the score_type_dict with the GPT evaluation scores
    for question, (a1_score, a2_score) in predictions:
        try:
            # Only add scores that can be converted to float
            score_type_dict['overall'][1].append(float(a1_score))
            score_type_dict['overall'][2].append(float(a2_score))
        except ValueError:
            print(f"Skipping non-numeric scores: {a1_score}, {a2_score} for question: {question}")

    result = defaultdict(dict)

    # Calculate averages and relative scores
    for q_type, score_dict in score_type_dict.items():
        if score_dict[1] and score_dict[2]:  # Ensure lists are not empty
            result[q_type]['gpt4_score'] = util.get_avg(score_dict[1])
            result[q_type]['pred_score'] = util.get_avg(score_dict[2])
            # Calculate relative score, but avoid division by zero
            relative_scores = [
                float(s2) / float(s1) * 100 if s1 != 0 else 0
                for s1, s2 in zip(score_dict[1], score_dict[2])
            ]
            result[q_type]['pred_relative_score'] = util.get_avg(relative_scores)
            result[q_type]['data_size'] = len(score_dict[1])

    # Create a DataFrame from the result dictionary
    df = pd.DataFrame.from_dict(result).filter(['overall'])
    print(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("GPT-4 Multimodal Chat Eval Postprocessing", add_help=True)
    parser.add_argument("--scores-file", default="", metavar="FILE", help="input path to gpt-4 score file")
    args = parser.parse_args()
    main(args)
