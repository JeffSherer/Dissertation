import json

def extract_answer(text):
    # Find the "Answer:" part in the text
    start = text.find("Answer:")
    if start == -1:
        return ""
    return text[start:].strip()

# Paths to input and output files
input_file = 'answers_bbf-3.jsonl'
output_file = 'answers_bbf-3_no_bbox.jsonl'

# Open input and output files
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # Parse each line as JSON
        data = json.loads(line)
        # Extract only the "Answer:" part from the text field
        data['text'] = extract_answer(data['text'])
        # Remove the bbox field if it exists
        if 'bbox' in data:
            del data['bbox']
        # Write the modified data to the output file
        outfile.write(json.dumps(data) + '\n')

print(f"Processed file saved as {output_file}")
