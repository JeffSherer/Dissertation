import json

def fix_and_filter_bbf_scores(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            item = json.loads(line)

            # Ensure that gpt_eval contains exactly two scores followed by a single comment
            item['gpt_eval'] = fix_gpt_eval(item.get('gpt_eval', ''))

            json.dump(item, outfile)
            outfile.write('\n')

    print(f"Filtered and fixed content saved to {output_file}")

def fix_gpt_eval(gpt_eval):
    # Split the gpt_eval by lines
    lines = gpt_eval.split('\n')
    
    if len(lines) > 0:
        # Extract the scores (first line) and ensure there are exactly two scores
        scores = lines[0].split()[:2]  # Only keep the first two scores
        if len(scores) < 2:
            scores.append('0')  # Add a default score if only one score is present

        # Reconstruct gpt_eval with exactly two scores
        fixed_gpt_eval = " ".join(scores)

        # Include only the first relevant comment if available
        if len(lines) > 1:
            fixed_gpt_eval += f"\n{lines[1]}"

        return fixed_gpt_eval
    return gpt_eval

# File paths
bbf_file_path = '/users/jjls2000/sharedscratch/Dissertation/results/final/llava_med/eval_scores_BBF-3-LM.jsonl'
output_file_path = '/users/jjls2000/sharedscratch/Dissertation/results/final/llava_med/fixed_eval_scores_BBF-3-LM.jsonl'

fix_and_filter_bbf_scores(bbf_file_path, output_file_path)
