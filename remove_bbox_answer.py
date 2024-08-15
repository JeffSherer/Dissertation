import json
import re

def remove_bbox_fields(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            try:
                entry = json.loads(line)
                
                # Remove 'bbox' field if it exists
                if 'bbox' in entry:
                    del entry['bbox']
                
                # Remove bounding box coordinates from 'text' field if they exist
                if 'text' in entry:
                    entry['text'] = remove_bbox_coordinates(entry['text'])
                
                # Write the modified entry to the output file
                outfile.write(json.dumps(entry) + '\n')
            except json.JSONDecodeError:
                print(f"JSONDecodeError in line: {line}")

def remove_bbox_coordinates(text):
    # Regular expression pattern to match bounding box coordinates
    bbox_pattern = r"Bounding box: \[[^\]]*\];?"
    # Substitute the pattern with an empty string
    cleaned_text = re.sub(bbox_pattern, '', text)
    return cleaned_text

# Paths for the input and output files
input_files = [
    '/users/jjls2000/sharedscratch/Dissertation/evaluation/answers_bbf-3.jsonl',
    '/users/jjls2000/sharedscratch/Dissertation/evaluation/answers_bbl-3.jsonl'
]
output_files = [
    '/users/jjls2000/sharedscratch/Dissertation/evaluation/answers_bbf-3_no_bbox.jsonl',
    '/users/jjls2000/sharedscratch/Dissertation/evaluation/answers_bbl-3_no_bbox.jsonl'
]

# Process both files
for input_file, output_file in zip(input_files, output_files):
    remove_bbox_fields(input_file, output_file)
