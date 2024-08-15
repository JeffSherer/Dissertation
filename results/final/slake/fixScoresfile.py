import json

def fix_scores(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            item = json.loads(line)
            
            # Fix null fields by replacing them with default values or removing them
            item['question_id'] = item.get('question_id') or "N/A"
            item['image'] = item.get('image') or "N/A"
            item['pair_id'] = item.get('pair_id') or "N/A"
            item['text'] = item.get('text') or "N/A"
            item['domain'] = item.get('domain') or {d: False for d in ['chest_xray', 'mri', 'histology', 'gross', 'ct_scan']}
            item['type'] = item.get('type') or "N/A"
            
            # Ensure gpt_eval exists and is in the correct format
            gpt_eval = item.get('gpt_eval')
            if gpt_eval:
                try:
                    # Split the first line into scores and ignore any extra splits
                    scores = gpt_eval.split('\n')[0].split()[:2]
                    # Ensure exactly two scores, default to 0 if less
                    if len(scores) < 2:
                        scores = ["1", "0"]  # Ensure s1 is not zero
                    if int(scores[0]) == 0:  # Prevent s1 from being zero
                        scores[0] = "1"
                    a1_score, a2_score = map(int, scores)
                    # Reconstruct gpt_eval to ensure it's in the correct format
                    item['gpt_eval'] = f"{a1_score} {a2_score}\n" + "\n".join(gpt_eval.split("\n")[1:])
                except (ValueError, IndexError):
                    # If parsing fails, set gpt_eval to a default value
                    item['gpt_eval'] = "1 0\nInvalid gpt_eval format"
            else:
                item['gpt_eval'] = "1 0\nMissing gpt_eval"

            # Write the fixed item back to the output file
            json.dump(item, outfile)
            outfile.write('\n')

    print(f"File transformed and saved as {output_file}")

if __name__ == "__main__":
    # Define input and output files
    input_file = '/users/jjls2000/sharedscratch/Dissertation/results/final/llava_med/eval_scores_BBL-3-LM.jsonl'
    output_file = '/users/jjls2000/sharedscratch/Dissertation/results/final/llava_med/fixed_eval_scores_BBL-3-LM.jsonl'
    
    # Run the fix function
    fix_scores(input_file, output_file)
