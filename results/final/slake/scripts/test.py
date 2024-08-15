import json

def load_jsonl_file(jsonl_file, num_entries=3):
    with open(jsonl_file, 'r') as f:
        for i in range(num_entries):
            line = f.readline()
            if not line:
                break
            data = json.loads(line)
            print(f"Entry {i+1}: {json.dumps(data, indent=4)}")

if __name__ == "__main__":
    original_file = '/users/jjls2000/sharedscratch/Dissertation/results/final/slake/slakeanswerBBF-BBOX_100.jsonl'
    load_jsonl_file(original_file)
