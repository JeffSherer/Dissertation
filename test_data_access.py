import os

def test_read(file_path):
    print(f"Reading from {file_path}")
    try:
        with open(file_path, 'r') as file:
            print(file.readline())
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

if __name__ == "__main__":
    data_dir = '/path/to/your/working/directory/yourdatarepo/data'  # Update this path
    try:
        files = os.listdir(data_dir)
        print(f"Files in {data_dir}: {files}")
    except Exception as e:
        print(f"Error accessing directory {data_dir}: {e}")
    
    test_file = os.path.join(data_dir, 'some_data_file.ext')  # Replace with an actual file name
    test_read(test_file)
