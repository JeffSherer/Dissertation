import json

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')

def main():
    # Paths to your files (Update these paths according to your file structure)
    test_file_path = '/users/jjls2000/sharedscratch/Dissertation/Slake1.0/test.json'
    gold_file_path = '/users/jjls2000/sharedscratch/Dissertation/Slake1.0/augmented_filtered/BBL_train_large.json'
    answer_file_path = '/users/jjls2000/sharedscratch/Dissertation/results/final/slake/slakeanswerBBF-IoU_100.jsonl'
    output_gold_standard_path = '/users/jjls2000/sharedscratch/Dissertation/results/final/slake/new_iou_gold_standard.jsonl'

    # Load the data
    print("Loading test.json...")
    test_data = load_json(test_file_path)
    print("Loading BBL_train_large.json...")
    gold_data = load_json(gold_file_path)
    print("Loading answer file...")
    answer_data = load_jsonl(answer_file_path)

    # Create a mapping from qid to img_name
    print("\nCreating mapping from qid to img_name...")
    qid_to_img_name = {entry['qid']: entry['img_name'] for entry in test_data}

    # Create a new gold standard list
    gold_standard = []
    missing_entries = []

    for answer in answer_data:
        question_id = answer['question_id']
        img_name = qid_to_img_name.get(question_id)
        
        if img_name:
            # Find the corresponding entry in the gold data
            gold_entry = next((entry for entry in gold_data if entry['image'] == img_name), None)
            
            if gold_entry:
                # Extract bounding boxes from the conversation field
                bounding_boxes = []
                for convo in gold_entry['conversations']:
                    if 'Bounding box' in convo['value']:
                        bbox_str = convo['value'].split('Bounding box: ')[-1].split(';')[0].strip('[]')
                        bbox_values = list(map(float, bbox_str.split(',')))
                        bounding_boxes.append(bbox_values)
                
                if bounding_boxes:
                    gold_standard.append({
                        'question_id': question_id,
                        'bbox': bounding_boxes
                    })
                else:
                    print(f"No bounding boxes found for question_id: {question_id} in image: {img_name}")
                    missing_entries.append({'question_id': question_id, 'img_name': img_name})
            else:
                print(f"No gold standard entry found for image: {img_name}")
                missing_entries.append({'question_id': question_id, 'img_name': img_name})
        else:
            print(f"No mapping found for question_id: {question_id}")
            missing_entries.append({'question_id': question_id, 'img_name': None})

    # Save the new gold standard to a file
    save_jsonl(gold_standard, output_gold_standard_path)
    print(f"Gold standard file saved to {output_gold_standard_path}")

    # Log missing entries
    if missing_entries:
        print(f"\nLogging missing entries. Total missing: {len(missing_entries)}")
        with open('/users/jjls2000/sharedscratch/Dissertation/results/final/slake/missing_entries_log.json', 'w') as log_file:
            json.dump(missing_entries, log_file, indent=4)
        print("Missing entries have been logged.")

if __name__ == "__main__":
    main()
