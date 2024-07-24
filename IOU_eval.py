import json
import torch
import math
from typing import Union

# Box-related classes and functions
class Boxes:
    def __init__(self, tensor: torch.Tensor) -> None:
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32, device=device)
        if tensor.dim() != 2 and tensor.size(-1) != 4:
            raise AssertionError(f"Tensor shape is incorrect. Current shape is {tensor.size()}")
        self.tensor = tensor

    def area(self) -> torch.Tensor:
        box = self.tensor
        width_difference = box[:, 2] - box[:, 0]
        height_difference = box[:, 3] - box[:, 1]
        area = width_difference * height_difference
        return area

    def __getitem__(self, index: Union[int, slice, torch.BoolTensor]) -> "Boxes":
        if isinstance(index, int):
            return Boxes(self.tensor[index].view(1, -1))
        box = self.tensor[index]
        if box.dim() != 2:
            raise AssertionError(f"Indexing on Boxes with {index} failed to return a matrix!")
        return Boxes(box)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def clone(self) -> "Boxes":
        return Boxes(self.tensor.clone())

    def to(self, device: torch.device) -> "Boxes":
        return Boxes(self.tensor.to(device=device))


def matched_pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    if len(boxes1) != len(boxes2):
        raise AssertionError(f"boxlists should have the same number of entries, got {len(boxes1)} and {len(boxes2)}")
    area1 = boxes1.area()
    area2 = boxes2.area()
    box1, box2 = boxes1.tensor, boxes2.tensor
    lt = torch.max(box1[:, :2], box2[:, :2])
    rb = torch.min(box1[:, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    iou = inter / (area1 + area2 - inter)
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
    return torch.tensor(predictions['boxes'])

def load_ground_truth(ground_truth_path):
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    return {
        'visual_token_ids': torch.tensor(ground_truth['visual_token_ids']),
        'object_coordinates': torch.tensor(ground_truth['object_coordinates']),
        'raw_target': torch.tensor(ground_truth['raw_target'])
    }

def evaluate(predictions_path, ground_truth_path):
    predicted_boxes = load_predictions(predictions_path)
    ground_truth_data = load_ground_truth(ground_truth_path)
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
    predictions_path = '/users/jjls2000/sharedscratch/Dissertation/results/llava_med_eval_answers.jsonl'
    ground_truth_path = '/users/jjls2000/sharedscratch/Dissertation/data/eval/llava_med_eval_qa50_qa.jsonl'
    evaluate(predictions_path, ground_truth_path)
