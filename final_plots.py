import os
import json
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# -------------------------------
# CONFIGURATION
# -------------------------------
OUTPUT_DIR = "final_plots_alekya"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOW_FOLDER = "TrafficCAM/raw/Complexity/Low_Complexity/Low_All"
HIGH_FOLDER = "TrafficCAM/raw/Complexity/High_Complexity/High_All"
DAWN_FOLDER = "DAWNDataset/High_Complexity"

MODELS_V8 = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
MODELS_V26 = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt"]

GW_BLUE = "#033C5A"
GW_GOLD = "#FFC72C"

# Unified classes across YOLO and TrafficCAM
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle", "motorbike", "vehicle", "auto", "lcv", "lmv"]
IOU_THRESHOLD = 0.5 # standard for F1

# -------------------------------
# SPATIAL LOGIC
# -------------------------------
def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def get_spatial_gt(img_path, dataset_type):
    """Extracts bounding box coordinates [x_min, y_min, x_max, y_max]"""
    boxes = []
    if dataset_type == "trafficcam":
        json_path = img_path.replace(".jpg", ".json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                for shape in data.get("shapes", []):
                    if shape["label"].lower() in VEHICLE_CLASSES:
                        pts = np.array(shape["points"])
                        boxes.append([pts[:,0].min(), pts[:,1].min(), pts[:,0].max(), pts[:,1].max()])
    else: # DAWN TXT uses normalized [class, x_center, y_center, w, h]
        img = cv2.imread(img_path)
        if img is None: return [] # Safety check
        h, w, _ = img.shape
        txt_path = img_path.replace(".jpg", ".txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    # DAWN is [class, x_center, y_center, width, height]
                    cx, cy, bw, bh = map(float, parts[1:])
                    # Convert to absolute pixel coordinates for IoU matching
                    x_min = (cx - bw/2) * w
                    y_min = (cy - bh/2) * h
                    x_max = (cx + bw/2) * w
                    y_max = (cy + bh/2) * h
                    boxes.append([x_min, y_min, x_max, y_max])
    return boxes

def evaluate_scenario(model_list, folder_path, dataset_type):
    avg_f1s, avg_times = [], []
    images = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith(".jpg")]

    for model_path in model_list:
        model = YOLO(model_path)
        tp_total, fp_total, fn_total, img_times = 0, 0, 0, []

        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            gt_boxes = get_spatial_gt(img_path, dataset_type)
            
            start = time.time()
            results = model(img_path, verbose=False, conf=0.25)
            img_times.append((time.time() - start) * 1000)

            # Filter vehicle-only predictions
            pred_boxes = [b.xyxy[0].tolist() for b in results[0].boxes 
                          if any(v in model.names[int(b.cls)].lower() for v in VEHICLE_CLASSES)]

            # One-to-one Spatial Matching
            matched_gt = set()
            tp, fp = 0, 0
            for p in pred_boxes:
                best_iou, best_idx = 0, -1
                for i, g in enumerate(gt_boxes):
                    if i in matched_gt: continue
                    iou = compute_iou(p, g)
                    if iou > best_iou:
                        best_iou, best_idx = iou, i
                
                if best_iou >= IOU_THRESHOLD:
                    tp += 1
                    matched_gt.add(best_idx)
                else:
                    fp += 1
            
            tp_total += tp
            fp_total += fp
            fn_total += (len(gt_boxes) - len(matched_gt))

        # Dataset-level F1 calculation
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_f1s.append(f1)
        avg_times.append(np.mean(img_times))
        
    return avg_f1s, avg_times

def plot_save(v8_data, v26_data, title, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(v26_data[1], v26_data[0], 'o-', color=GW_BLUE, label="YOLOv26 (Target)", lw=2)
    plt.plot(v8_data[1], v8_data[0], 'o-', color=GW_GOLD, label="YOLOv8 (Baseline)", lw=2)
    plt.xlabel("Average Inference Time (ms)"); plt.ylabel("F1 Score (IoU=0.5)")
    plt.title(title, fontweight="bold"); plt.legend(); plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.png"), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.pdf"))
    plt.close()

if __name__ == "__main__":
    scenarios = [
        (LOW_FOLDER, "trafficcam", "Low Complexity (TrafficCAM Clear)", "low_complexity"),
        (HIGH_FOLDER, "trafficcam", "High Traffic (TrafficCAM Busy)", "high_traffic"),
        (DAWN_FOLDER, "dawn", "Bad Weather (DAWN)", "bad_weather")
    ]
    for folder, d_type, title, fname in scenarios:
        v8_res = evaluate_scenario(MODELS_V8, folder, d_type)
        v26_res = evaluate_scenario(MODELS_V26, folder, d_type)
        plot_save(v8_res, v26_res, title, fname)
    print(f"\nAll 3 graphs saved to /{OUTPUT_DIR}")