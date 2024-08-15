import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def check_missing_with_indirect_mapping(missing_question_ids, test_file_path, gold_file_path):
    # Load the test.json file
    print("Loading test.json...")
    test_data = load_json(test_file_path)

    # Create a mapping from qid to img_name
    qid_to_img_name = {entry['qid']: entry['img_name'] for entry in test_data}

    # Load the gold standard file as a regular JSON file
    print("Loading gold standard file...")
    gold_data = load_json(gold_file_path)  # Assume this is a regular JSON, not JSONL

    # Create a set of img_names from the gold standard
    img_names_in_gold = {entry['image'] for entry in gold_data}

    # Check which question_ids map to img_names not in the gold standard
    missing_img_names = []
    for question_id in missing_question_ids:
        img_name = qid_to_img_name.get(question_id)
        if img_name and img_name not in img_names_in_gold:
            missing_img_names.append((question_id, img_name))

    if missing_img_names:
        print(f"The following question_ids map to img_names not in the gold standard ({len(missing_img_names)}):")
        for qid, img in missing_img_names:
            print(f"question_id: {qid} -> img_name: {img}")
    else:
        print("All missing question_ids map to img_names present in the gold standard.")

def main():
    # Missing question_ids from the previous step
    missing_question_ids = [
        12162, 12164, 12165, 12166, 12167, 12169, 12222, 12224, 12225, 12226,
        12228, 12229, 12230, 12231, 12232, 12233, 12234, 12765, 12766, 12767,
        12768, 12769, 12946, 12948, 12949, 12950, 12954, 12956, 12957, 12958,
        12962, 12963, 12964, 12965, 12966, 12967, 12970, 12972, 12973, 12974,
        12975, 12977, 12978, 12979, 12982, 12983, 12984, 12985, 12986, 12990,
        12991, 12992, 12993, 12994
    ]

    # Paths to your files
    test_file_path = '/users/jjls2000/sharedscratch/Dissertation/Slake1.0/test.json'
    gold_file_path = '/users/jjls2000/sharedscratch/Dissertation/Slake1.0/augmented_filtered/BBL_train_large.json'

    # Check the missing question_ids using the indirect mapping
    check_missing_with_indirect_mapping(missing_question_ids, test_file_path, gold_file_path)

if __name__ == "__main__":
    main()
