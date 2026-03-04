import os
import json
import glob

# ===============================
# CONFIG
# ===============================

IOU_THRESHOLD = 0.5

VEHICLE_CLASSES = [
    "car", "truck", "bus", "motorcycle",
    "motorbike", "vehicle", "auto", "lcv", "lmv"
]

# YOLO COCO class mapping (for v8 / v26 pretrained models)
COCO_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    # remaining not needed for vehicle-only filtering
}

MODELS = [
    "26n_low", "26s_low", "26m_low", "26l_low",
    "26n_high", "26s_high", "26m_high", "26l_high",
    "8n_low", "8s_low", "8m_low", "8l_low",
    "8n_high", "8s_high", "8m_high", "8l_high",
]

LOW_GT_PATH = "TrafficCAM/raw/Complexity/Low_Complexity/Low_All"
HIGH_GT_PATH = "TrafficCAM/raw/Complexity/High_Complexity/High_All"

# ===============================
# IOU FUNCTION
# ===============================

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# ===============================
# LOAD GROUND TRUTH (Vehicle-Only)
# ===============================

def load_ground_truth(gt_folder):
    gt_boxes = {}

    json_files = glob.glob(os.path.join(gt_folder, "*.json"))

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        width = data["imageWidth"]
        height = data["imageHeight"]

        boxes = []
        for shape in data["shapes"]:

            label = shape["label"].lower()
            if label not in VEHICLE_CLASSES:
                continue  # Vehicle-only filtering

            points = shape["points"]

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            x_min = min(xs)
            y_min = min(ys)
            x_max = max(xs)
            y_max = max(ys)

            boxes.append([x_min, y_min, x_max, y_max])

        filename = os.path.basename(jf).replace(".json", ".jpg")
        gt_boxes[filename] = (boxes, width, height)

    return gt_boxes


# ===============================
# LOAD PREDICTIONS (Vehicle-Only)
# ===============================

def load_predictions(pred_folder, gt_info):
    predictions = {}

    label_files = glob.glob(os.path.join(pred_folder, "labels", "*.txt"))

    for lf in label_files:
        filename = os.path.basename(lf).replace(".txt", ".jpg")

        if filename not in gt_info:
            continue

        width = gt_info[filename][1]
        height = gt_info[filename][2]

        boxes = []

        with open(lf) as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])

            # Only evaluate vehicle classes
            if class_id not in COCO_CLASS_NAMES:
                continue

            class_name = COCO_CLASS_NAMES[class_id].lower()

            if class_name not in VEHICLE_CLASSES:
                continue

            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            x_min = (xc - w/2) * width
            y_min = (yc - h/2) * height
            x_max = (xc + w/2) * width
            y_max = (yc + h/2) * height

            boxes.append([x_min, y_min, x_max, y_max])

        predictions[filename] = boxes

    return predictions


# ===============================
# EVALUATE
# ===============================

def evaluate(gt_boxes, pred_boxes):
    TP = 0
    FP = 0
    FN = 0

    for filename in gt_boxes:
        gt_list = gt_boxes[filename][0]
        pred_list = pred_boxes.get(filename, [])

        matched_gt = set()

        for p in pred_list:
            best_iou = 0
            best_gt_idx = -1

            for i, g in enumerate(gt_list):
                if i in matched_gt:
                    continue

                iou = compute_iou(p, g)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= IOU_THRESHOLD:
                TP += 1
                matched_gt.add(best_gt_idx)
            else:
                FP += 1

        FN += len(gt_list) - len(matched_gt)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
    }


# ===============================
# MAIN LOOP
# ===============================

for model in MODELS:

    print("\n=======================")
    print(model)
    print("=======================")

    if "low" in model:
        gt_folder = LOW_GT_PATH
    else:
        gt_folder = HIGH_GT_PATH

    gt_boxes = load_ground_truth(gt_folder)

    nested_path = os.path.join("runs/detect/runs/detect/predictions", model)
    normal_path = os.path.join("runs/detect/predictions", model)

    if os.path.exists(nested_path):
        pred_folder = nested_path
    else:
        pred_folder = normal_path

    pred_boxes = load_predictions(pred_folder, gt_boxes)

    results = evaluate(gt_boxes, pred_boxes)

    print(results)

