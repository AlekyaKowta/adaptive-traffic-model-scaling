import os
import xml.etree.ElementTree as ET
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO

# -------------------------------
# CONFIGURATION
# -------------------------------
LOW_FOLDER = "DAWNDataset/Low_Complexity"
HIGH_FOLDER = "DAWNDataset/High_Complexity"
LIMIT = 67  # Ensuring balanced 1:1 comparison

MODELS = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt"]
VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike", "vehicle"]

# -------------------------------
# GROUND TRUTH PARSERS
# -------------------------------

def get_txt_gt(txt_path):
    """Counts lines in a YOLO .txt file for DAWN."""
    if not os.path.exists(txt_path): return 0
    with open(txt_path, "r") as f:
        return sum(1 for line in f if line.strip())

def get_xml_gt(xml_path, filename):
    """Counts <target> tags for a frame in UA-DETRAC XML."""
    if not os.path.exists(xml_path): return 0
    try:
        # Extract digits for frame number (e.g., img00063.jpg -> 63)
        frame_num = str(int(''.join(filter(str.isdigit, filename))))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        frame = root.find(f".//frame[@num='{frame_num}']")
        if frame is not None:
            target_list = frame.find('target_list')
            return len(target_list.findall('target')) if target_list is not None else 0
    except Exception as e:
        print(f"Error parsing XML for {filename}: {e}")
    return 0

# -------------------------------
# EVALUATION ENGINE 
# -------------------------------

def evaluate_folder(model, folder_path, is_xml=False):
    total_accuracy = 0
    total_time = 0
    image_count = 0
    
    # Filter and limit images to ensure balanced data
    images = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith(".jpg")][:LIMIT]

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        
        # 1. Get Ground Truth
        if is_xml:
            xml_files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]
            xml_path = os.path.join(folder_path, xml_files[0]) if xml_files else ""
            ground_truth = get_xml_gt(xml_path, img_name)
        else:
            txt_path = img_path.replace(".jpg", ".txt")
            ground_truth = get_txt_gt(txt_path)

        # 2. Model Prediction + Timing
        start_time = time.time()
        results = model(img_path, verbose=False, conf=0.5)
        inference_time = (time.time() - start_time) * 1000
        total_time += inference_time

        predicted = sum(1 for cls in results[0].boxes.cls if model.names[int(cls)] in VEHICLE_CLASSES)

        # 3. Count-based accuracy ratio
        if max(predicted, ground_truth) > 0:
            accuracy = min(predicted, ground_truth) / max(predicted, ground_truth)
        else:
            accuracy = 1.0  # both zero

        total_accuracy += accuracy
        image_count += 1

    avg_accuracy = total_accuracy / image_count if image_count > 0 else 0
    avg_time = total_time / image_count if image_count > 0 else 0

    return avg_accuracy, avg_time

# -------------------------------
# MAIN EXECUTION
# -------------------------------

print(f"\n=========== FINAL EVALUATION ({LIMIT} vs {LIMIT}) ===========\n")

low_results = []
high_results = []

for model_name in MODELS:
    print(f"Evaluating model: {model_name}")
    model = YOLO(model_name)

    # UA-DETRAC is XML (is_xml=True), DAWN is TXT (is_xml=False)
    low_acc, low_time = evaluate_folder(model, LOW_FOLDER, is_xml=True)
    high_acc, high_time = evaluate_folder(model, HIGH_FOLDER, is_xml=False)

    low_results.append(low_acc)
    high_results.append(high_acc)

    print(f"  Low Complexity  -> Accuracy: {low_acc:.4f} | Avg Time: {low_time:.2f} ms")
    print(f"  High Complexity -> Accuracy: {high_acc:.4f} | Avg Time: {high_time:.2f} ms")
    print("-" * 54)

# -------------------------------
# RELATIVE IMPROVEMENT ANALYSIS
# -------------------------------

print("\n=========== RELATIVE IMPROVEMENT ANALYSIS ===========")

low_gain = low_results[-1] - low_results[0]
high_gain = high_results[-1] - high_results[0]

low_relative = (low_gain / low_results[0]) * 100 if low_results[0] > 0 else 0
high_relative = (high_gain / high_results[0]) * 100 if high_results[0] > 0 else 0

print(f"Low Complexity Absolute Gain:  {low_gain:.4f}")
print(f"High Complexity Absolute Gain: {high_gain:.4f}")
print("-" * 54)
print(f"Low Complexity Relative Gain:  {low_relative:.2f}%")
print(f"High Complexity Relative Gain: {high_relative:.2f}%")
print("====================================================\n")