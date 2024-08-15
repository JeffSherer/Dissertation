import json
import os

# Paths to the files
question_file = '/users/jjls2000/sharedscratch/Dissertation/Slake1.0/test_questions_fixed.jsonl'
gold_standard_file = '/users/jjls2000/sharedscratch/Dissertation/Slake1.0/test.json'
updated_question_file = '/users/jjls2000/sharedscratch/Dissertation/Slake1.0/test_questions_fixed_100.jsonl'

# Load the current question file
with open(question_file, 'r') as f:
    questions = [json.loads(line) for line in f]

# Load the gold standard file
with open(gold_standard_file, 'r') as f:
    gold_standard_data = json.load(f)

gold_standard_dict = {item['qid']: item for item in gold_standard_data}

# Update the questions with the data from the gold standard
updated_questions = []
for question in questions:
    qid = question['question_id']
    if qid in gold_standard_dict:
        updated_question = question.copy()
        updated_question.update({
            'img_id': gold_standard_dict[qid]['img_id'],
            'img_name': gold_standard_dict[qid]['img_name'],
            'question': gold_standard_dict[qid]['question'],
            'answer': gold_standard_dict[qid]['answer'],
            'q_lang': gold_standard_dict[qid]['q_lang'],
            'location': gold_standard_dict[qid]['location'],
            'modality': gold_standard_dict[qid]['modality'],
            'answer_type': gold_standard_dict[qid]['answer_type'],
            'base_type': gold_standard_dict[qid]['base_type'],
            'content_type': gold_standard_dict[qid]['content_type'],
            'triple': gold_standard_dict[qid]['triple'],
        })
        updated_questions.append(updated_question)

# Write the updated questions to the new file
with open(updated_question_file, 'w') as f:
    for updated_question in updated_questions:
        f.write(json.dumps(updated_question) + '\n')

print(f"Updated question file saved as {updated_question_file}")
