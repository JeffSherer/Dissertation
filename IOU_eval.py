import json
import torch

# Configure paths for loading data
PREDICTIONS_PATH = '/users/jjls2000/sharedscratch/Dissertation/results/llava_med_eval_answers.jsonl'
GROUND_TRUTH_PATH = '/users/jjls2000/sharedscratch/Dissertation/data/eval/llava_med_eval_qa50_qa.jsonl'

class Boxes:
    def __init__(self, boxes):
        self.boxes = boxes

    def area(self):
        x1, y1, x2, y2 = self.boxes.T
        return (x2 - x1) * (y2 - y1)

def matched_pairwise_iou(boxes1, boxes2):
    x1, y1, x2, y2 = boxes1.boxes.T
    x1g, y1g, x2g, y2g = boxes2.boxes.T

    xA = torch.max(x1, x1g)
    yA = torch.max(y1, y1g)
    xB = torch.min(x2, x2g)
    yB = torch.min(y2, y2g)

    interArea = (xB - xA).clamp(0) * (yB - yA).clamp(0)
    boxAArea = boxes1.area()
    boxBArea = boxes2.area()

    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

class RefCOCOAccuracy:
    def __init__(self, threshold=0.5):
        self.correct = 0
        self.total = 0
        self.threshold = threshold

    def update(self, outputs, batch):
        indices = torch.where(batch['visual_token_ids'] == outputs[:, None])
        if indices[0].shape[0] > 0:
            predicted_bbox_coords = batch['object_coordinates'][indices]
            predicted_bbox = Boxes(predicted_bbox_coords)
            target_bbox = Boxes(torch.stack(batch['raw_target'])[indices[0]])
            ious = matched_pairwise_iou(predicted_bbox, target_bbox)
            self.correct += torch.sum(ious >= self.threshold).item()
        self.total += outputs.shape[0]

    def compute(self):
        return self.correct / self.total if self.total > 0 else 0

def load_predictions(predictions_path):
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    # Example format adjustment, modify as needed
    return torch.tensor(predictions['boxes'])

def load_ground_truth(ground_truth_path):
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    # Example format adjustment, modify as needed
    return {
        'visual_token_ids': torch.tensor(ground_truth['visual_token_ids']),
        'object_coordinates': torch.tensor(ground_truth['object_coordinates']),
        'raw_target': torch.tensor(ground_truth['raw_target'])
    }

def evaluate():
    predicted_boxes = load_predictions(PREDICTIONS_PATH)
    ground_truth_data = load_ground_truth(GROUND_TRUTH_PATH)

    batch = {
        'visual_token_ids': ground_truth_data['visual_token_ids'],
        'object_coordinates': ground_truth_data['object_coordinates'],
        'raw_target': ground_truth_data['raw_target']
    }

    accuracy_metric = RefCOCOAccuracy(threshold=0.5)
    accuracy_metric.update(predicted_boxes, batch)
    accuracy = accuracy_metric.compute()
    print(f'IoU-based Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    evaluate()
