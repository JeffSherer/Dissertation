import json

def compare_jsonl_files(file1, file2, type_of_comparison):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = [json.loads(line) for line in f1]
        lines2 = [json.loads(line) for line in f2]

        if len(lines1) != len(lines2):
            print(f"Number of entries differ between {type_of_comparison}: {len(lines1)} vs {len(lines2)}")
        
        for i, (item1, item2) in enumerate(zip(lines1, lines2), 1):
            if item1.keys() != item2.keys():
                print(f"Line {i}: Structure mismatch in {type_of_comparison}")
            else:
                if type_of_comparison == "answers":
                    ans1_1 = item1.get('ans1')
                    ans1_2 = item2.get('ans1')
                    ans2_1 = item1.get('ans2')
                    ans2_2 = item2.get('ans2')
                    
                    if ans1_1 != ans1_2 or ans2_1 != ans2_2:
                        print(f"Line {i}: Answers mismatch in {type_of_comparison}")
                elif type_of_comparison == "scores":
                    score1_1 = item1.get('gpt_eval', '').split('\n')[0]
                    score1_2 = item2.get('gpt_eval', '').split('\n')[0]
                    
                    if score1_1 != score1_2:
                        print(f"Line {i}: Scores mismatch in {type_of_comparison}")

# Define file paths
bbl_answers_file = '/users/jjls2000/sharedscratch/Dissertation/results/final/llava_med/answer_BBL-3-fixed-LM.jsonl'
bbf_no_bbox_answers_file = '/users/jjls2000/sharedscratch/Dissertation/results/final/llava_med/answers_bbf-3_no_bbox_LM.jsonl'
bbl_scores_file = '/users/jjls2000/sharedscratch/Dissertation/results/final/llava_med/fixed_eval_scores_BBL-3-LM.jsonl'
bbf_scores_file = '/users/jjls2000/sharedscratch/Dissertation/results/final/llava_med/eval_scores_BBF-3-LM.jsonl'

# Compare the answer files
print("Comparing Answer Files:")
compare_jsonl_files(bbl_answers_file, bbf_no_bbox_answers_file, "answers")

# Compare the score files
print("\nComparing Score Files:")
compare_jsonl_files(bbl_scores_file, bbf_scores_file, "scores")
