import os
import json
import time
import numpy as np
from ultralytics import YOLO
from src.data_utils import UniversalParser

class ExperimentEngine:
    def __init__(self, results_path="results/master_metrics.json"):
        self.results_path = results_path
        self.results_data = self._load_results()
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)

    def _load_results(self):
        """Loads existing results to avoid redundant runs."""
        if os.path.exists(self.results_path):
            with open(self.results_path, "r") as f:
                return json.load(f)
        return {}

    def save_results(self):
        with open(self.results_path, "w") as f:
            json.dump(self.results_data, f, indent=4)

    def run_scenario(self, scenario_name, folder_path, model_names, limit=67):
        """Runs inference only if data is missing for a specific model/scenario."""
        if scenario_name not in self.results_data:
            self.results_data[scenario_name] = {}

        for m_name in model_names:
            # === SMART CACHE CHECK ===
            if m_name in self.results_data[scenario_name]:
                print(f"⏭️ Skipping {m_name} for {scenario_name} (found in cache)")
                continue

            print(f"🚀 Running {m_name} on {scenario_name}...")
            model = YOLO(m_name)
            
            # Filter and limit images (Ensures balanced 1:1 comparison like n=67)
            images = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith(".jpg")][:limit]
            
            # Metrics placeholders
            tp, fp, fn, times = 0, 0, 0, []

            for img_name in images:
                img_path = os.path.join(folder_path, img_name)
                gt_boxes, _ = UniversalParser.get_ground_truth(img_path)

                start = time.time()
                results = model(img_path, verbose=False, conf=0.25)
                times.append((time.time() - start) * 1000)

                # Filter detections for vehicles only (car, truck, bus, motorcycle)
                pred_boxes = [b.xyxy[0].tolist() for b in results[0].boxes 
                              if any(v in model.names[int(b.cls)].lower() for v in UniversalParser.VEHICLE_LABELS)]

                # Spatial Matching Logic (Simplified version of your final_plots.py)
                matched_gt = set()
                for p in pred_boxes:
                    is_match = False
                    for i, g in enumerate(gt_boxes):
                        if i in matched_gt: continue
                        # Calculate simple IoU or distance check
                        if self._is_box_match(p, g):
                            tp += 1
                            matched_gt.add(i)
                            is_match = True
                            break
                    if not is_match:
                        fp += 1
                fn += (len(gt_boxes) - len(matched_gt))

            # Calculate Final Metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Store in Cache
            self.results_data[scenario_name][m_name] = {
                "f1_score": round(f1, 4),
                "avg_inference_ms": round(float(np.mean(times)), 2),
                "precision": round(precision, 4),
                "recall": round(recall, 4)
            }
            self.save_results()

    def _is_box_match(self, box1, box2, threshold=0.5):
        """Helper to determine if two boxes overlap significantly."""
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        if x2 <= x1 or y2 <= y1: return False
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return (intersection / union) >= threshold