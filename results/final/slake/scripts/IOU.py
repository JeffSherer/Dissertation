import json
import torch
from torchmetrics import Metric
from typing import Any, List, Dict

class IoUMetric(Metric):
    def __init__(self, dist_sync_on_step: bool = True) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("iou_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        ious = self.compute_iou(preds, target)
        self.iou_sum += torch.sum(ious)
        self.total += ious.numel()

    def compute(self) -> torch.Tensor:
        return self.iou_sum / self.total if self.total > 0 else torch.tensor(0.0)

    def compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        inter = self.intersection(boxes1, boxes2)
        area1 = self.area(boxes1)
        area2 = self.area(boxes2)
        union = area1 + area2 - inter
        ious = inter / union
        return ious

    def intersection(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        return wh[:, 0] * wh[:, 1]

    def area(self, boxes: torch.Tensor) -> torch.Tensor:
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def load_jsonl_file(jsonl_file: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    with open(jsonl_file, 'r') as f:
        return [json.loads(line) for line in f]

def evaluate_iou(answer_file: str, gold_standard_file: str) -> None:
    """Evaluate IoU between predicted and gold standard bounding boxes."""
    # Load data
    answer_data = load_jsonl_file(answer_file)
    gold_standard_data = load_jsonl_file(gold_standard_file)

    # Create a dictionary for quick lookup of gold standard bounding boxes
    gold_standard_dict = {item['question_id']: item['bbox'] for item in gold_standard_data}

    # Initialize IoU metric
    iou_metric = IoUMetric()

    unmatched_ids = []

    for answer in answer_data:
        qid = answer['question_id']
        if qid in gold_standard_dict:
            gold_standard_bbox = gold_standard_dict[qid]
            predicted_bbox = answer.get('bbox', [])

            if gold_standard_bbox and predicted_bbox:
                # Convert lists to torch tensors
                gold_tensor = torch.tensor(gold_standard_bbox, dtype=torch.float32)
                pred_tensor = torch.tensor(predicted_bbox, dtype=torch.float32)

                # Debugging outputs
                print(f"Question ID: {qid}")
                print(f"Gold BBox: {gold_tensor}")
                print(f"Predicted BBox: {pred_tensor}")

                # Update IoU metric
                iou_metric.update(pred_tensor, gold_tensor)
            else:
                print(f"No bounding boxes found for question_id: {qid}")
        else:
            unmatched_ids.append(qid)

    # Compute final IoU
    final_iou = iou_metric.compute()
    print(f"Final IoU: {final_iou:.4f}")
    if unmatched_ids:
        print(f"Unmatched question_ids: {unmatched_ids}")

def test_iou():
    """Test IoU metric with known bounding boxes."""
    gold_bbox = [[50, 50, 150, 150]]  # Perfect overlap
    pred_bbox = [[50, 50, 150, 150]]
    gold_tensor = torch.tensor(gold_bbox, dtype=torch.float32)
    pred_tensor = torch.tensor(pred_bbox, dtype=torch.float32)

    metric = IoUMetric()
    metric.update(pred_tensor, gold_tensor)
    print(f"Expected IoU: 1.0, Calculated IoU: {metric.compute().item()}")

    # Partial overlap
    gold_bbox = [[50, 50, 150, 150]]
    pred_bbox = [[100, 100, 200, 200]]
    gold_tensor = torch.tensor(gold_bbox, dtype=torch.float32)
    pred_tensor = torch.tensor(pred_bbox, dtype=torch.float32)

    metric = IoUMetric()
    metric.update(pred_tensor, gold_tensor)
    print(f"Expected IoU: 0.1429, Calculated IoU: {metric.compute().item()}")

    # No overlap
    gold_bbox = [[50, 50, 150, 150]]
    pred_bbox = [[200, 200, 300, 300]]
    gold_tensor = torch.tensor(gold_bbox, dtype=torch.float32)
    pred_tensor = torch.tensor(pred_bbox, dtype=torch.float32)

    metric = IoUMetric()
    metric.update(pred_tensor, gold_tensor)
    print(f"Expected IoU: 0.0, Calculated IoU: {metric.compute().item()}")

if __name__ == "__main__":
    # Uncomment to run tests
    # test_iou()

    # Replace with your actual paths
    answer_file = "/users/jjls2000/sharedscratch/Dissertation/results/final/slake/output_bbox_only-BBL.jsonl"
    gold_standard_file = "/users/jjls2000/sharedscratch/Dissertation/results/final/slake/new_iou_gold_standard.jsonl"
    
    evaluate_iou(answer_file, gold_standard_file)
