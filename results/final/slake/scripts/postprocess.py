import re
import json
import argparse

def remove_bbox_sentences(text):
    """
    Removes sentences from the text that contain mentions of bounding boxes.
    """
    bbox_sentence_pattern = r'[^.]*\bbounding box(?:\s+coordinates)?[^.]*\.\s*|\([^)]+\)\s*|\n\s*'
    text_without_bbox = re.sub(bbox_sentence_pattern, '', text, flags=re.IGNORECASE).strip()
    return text_without_bbox

def extract_bbox_from_text(text):
    """
    Extracts bounding box coordinates from the text.
    Assumes coordinates are given in the format (x1, y1, x2, y2).
    """
    bbox_pattern = r'\(\d+,\s*\d+,\s*\d+,\s*\d+\)'
    bboxes = re.findall(bbox_pattern, text)
    parsed_bboxes = []
    
    for bbox in bboxes:
        coords = list(map(int, re.findall(r'\d+', bbox)))
        parsed_bboxes.append(coords)
    
    return parsed_bboxes

def process_files(input_file, accuracy_recall_file, iou_file):
    """
    Processes the input file to generate an accuracy/recall file and an IoU file.
    - The accuracy/recall file has bounding box information removed.
    - The IoU file contains only bounding box information.
    """
    with open(input_file, 'r') as infile, \
         open(accuracy_recall_file, 'w') as acc_file, \
         open(iou_file, 'w') as iou_outfile:
        
        for line in infile:
            data = json.loads(line)
            text_content = data['text']
            
            # Extract bounding boxes from text
            bboxes = extract_bbox_from_text(text_content)
            data['bbox'] = bboxes
            
            # Remove bounding box-related sentences from text for accuracy/recall file
            text_without_bbox = remove_bbox_sentences(text_content)
            
            # Prepare data for accuracy/recall file
            acc_data = data.copy()
            acc_data['text'] = text_without_bbox
            if 'bbox' in acc_data:
                del acc_data['bbox']  # Remove bbox field
            acc_file.write(json.dumps(acc_data) + "\n")
            
            # Prepare data for IoU file
            if bboxes:
                iou_data = {
                    "question_id": data.get('question_id'),
                    "bbox": bboxes
                }
                iou_outfile.write(json.dumps(iou_data) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input JSONL file to generate accuracy/recall and IoU files.")
    parser.add_argument('--input-file', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--accuracy-recall-file', type=str, required=True, help='Path to output Accuracy/Recall file')
    parser.add_argument('--iou-file', type=str, required=True, help='Path to output IoU file')
    args = parser.parse_args()

    process_files(args.input_file, args.accuracy_recall_file, args.iou_file)
