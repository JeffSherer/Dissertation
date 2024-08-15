import os
import json
import argparse
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path

import llm
import util

INSTRUCT_PROMPT = """We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with caption describing the same image.
Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""
ROLE = 'Assistant'

# Generate instruction for GPT-4 to score the two answers.
def conv_to_str(fig_label, fig_caption, fig_context, question, ans1, ans2):
    return (f'[Context]\n'
            f'Figure Caption:\n{fig_label}: {fig_caption}\n\n'
            f'Figure Context:\n\t- {fig_context}\n\n'
            f'[Question]\n{question}\n\n'
            f'[{ROLE} 1]\n{ans1}\n\n[End of {ROLE} 1]\n\n'
            f'[{ROLE} 2]\n{ans2}\n\n[End of {ROLE} 2]\n\n'
            f'[System]\n{INSTRUCT_PROMPT}\n\n')

def compare_messages_gen(fig_label, fig_caption, fig_context, question, ans1, ans2):
    messages = [
        {"role": "system", "content": """'You are a helpful and precise assistant for checking the quality of the answer."""},
    ]
    messages.append({"role": "user", "content": conv_to_str(fig_label, fig_caption, fig_context, question, ans1, ans2)})
    return messages

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def infer(samples):
    model_inst = llm.GPT("gpt-4")

    BATCH_SIZE = 1
    results = []
    
    print('Starting Multimodal Chat GPT Scoring Eval')

    for sample in tqdm(samples):
        input_msg = compare_messages_gen(sample['fig_label'], sample['fig_caption'], sample['in_text_mention'], sample['question'], sample['ans1'], sample['ans2'])
        batch = [input_msg]
        try:
            inference_results = [x.strip() for chunk_messages in chunk([x for x in batch if x], BATCH_SIZE) for x in model_inst.infer(chunk_messages)]
            for item, inference_result in zip([sample], inference_results):
                item['gpt_eval'] = inference_result
            results.append(sample)
        except llm.OpenAIError as e:
            print(f"An error occurred during inference: {e}")
            continue

    print(f"Result Size: {len(results)}")
    return results

def main(args):
    answer_data = util.load_file_jsonl(args.answers_file)
    question_data = util.load_file_jsonl(args.question_file)
    
    samples = []
    for question, answer in zip(question_data, answer_data):
        sample = {}
        sample['question'] = question['text']
        sample['ans1'] = answer['text']  # Predicted answer from the answer file
        sample['ans2'] = question['answer']  # Expected answer from the question file
        sample['fig_label'] = answer.get('fig_label', '')  # Assuming you may have this data
        sample['fig_caption'] = answer.get('fig_caption', '')  # Assuming you may have this data
        sample['in_text_mention'] = answer.get('metadata', {}).get('in_text_mention', '')  # Assuming metadata contains relevant context
        samples.append(sample)
    
    results = infer(samples)

    # Create parent directory of output score files if it doesn't exist
    os.makedirs(Path(args.scores_file).parent, exist_ok=True)

    with open(args.scores_file, 'w') as f:
       for row in results:
          f.write(json.dumps(row)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("GPT-4 Multimodal Chat Scoring", add_help=True)
    parser.add_argument("--answers-file", default="", metavar="FILE", help="path to model answer file")
    parser.add_argument("--question-file", default="/users/jjls2000/sharedscratch/Dissertation/Slake1.0/test_questions_fixed_100.jsonl", metavar="FILE", help="path to multichat questions file")
    parser.add_argument("--scores-file", default="", metavar="FILE", help="path to save gpt-4 score file")
    args = parser.parse_args()
    main(args)
