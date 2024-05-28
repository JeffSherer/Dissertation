# test_llava_med.py
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "microsoft/llava-med-v1.5-mistral-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load test data
test_data_path = "data/eval/llava_med_eval_qa50_qa.jsonl"
with open(test_data_path, 'r') as file:
    test_data = [json.loads(line) for line in file]

# Prepare output file
output_file_path = "llava_med_test_results.jsonl"
with open(output_file_path, 'w') as outfile:

    # Run inference on each test example
    for example in test_data:
        inputs = tokenizer(example['question'], return_tensors="pt").to(device)
        outputs = model.generate(inputs['input_ids'])
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Write the result to the output file
        result = {
            'question': example['question'],
            'expected_answer': example['expected_answer'],
            'model_response': response
        }
        outfile.write(json.dumps(result) + "\n")

print(f"Inference completed. Results saved to {output_file_path}")
