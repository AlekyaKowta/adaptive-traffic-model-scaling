import os
import json
import time
from ultralytics import YOLO

# -------------------------------
# CONFIGURATION
# -------------------------------

LOW_FOLDER = "TrafficCAM/raw/Complexity/Low_Complexity/Low_All"
HIGH_FOLDER = "TrafficCAM/raw/Complexity/High_Complexity/High_All"

MODELS = [
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt"
]

# YOLO vehicle classes (COCO)
YOLO_VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

# Ground truth vehicle labels (TrafficCAM)
GT_VEHICLE_LABELS = ["Auto", "LCV", "LMV", "MotorBike", "Truck"]


# -------------------------------
# HELPER FUNCTION
# -------------------------------

def evaluate_folder(model, folder_path):
    total_accuracy = 0
    image_count = 0
    total_time = 0

    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):

            image_path = os.path.join(folder_path, file)
            json_path = image_path.replace(".jpg", ".json")

            if not os.path.exists(json_path):
                continue

            # -------------------------------
            # Ground Truth Vehicle Count
            # -------------------------------

            with open(json_path, "r") as f:
                data = json.load(f)

                ground_truth = 0
                for shape in data["shapes"]:
                    if shape["label"] in GT_VEHICLE_LABELS:
                        ground_truth += 1

            # -------------------------------
            # YOLO Prediction
            # -------------------------------

            start_time = time.time()
            results = model(image_path, verbose=False)
            inference_time = (time.time() - start_time) * 1000
            total_time += inference_time

            predicted = 0
            for cls in results[0].boxes.cls:
                class_name = model.names[int(cls)]
                if class_name in YOLO_VEHICLE_CLASSES:
                    predicted += 1

            # -------------------------------
            # Count-based Accuracy
            # -------------------------------

            if max(predicted, ground_truth) > 0:
                accuracy = min(predicted, ground_truth) / max(predicted, ground_truth)
            else:
                accuracy = 1.0  # both zero

            total_accuracy += accuracy
            image_count += 1

    avg_accuracy = total_accuracy / image_count
    avg_time = total_time / image_count

    return avg_accuracy, avg_time


# -------------------------------
# MAIN EVALUATION
# -------------------------------

print("\n=========== FINAL VEHICLE-ONLY EVALUATION ===========\n")

low_results = []
high_results = []

for model_name in MODELS:
    print(f"Evaluating model: {model_name}")
    model = YOLO(model_name)

    low_acc, low_time = evaluate_folder(model, LOW_FOLDER)
    high_acc, high_time = evaluate_folder(model, HIGH_FOLDER)

    low_results.append(low_acc)
    high_results.append(high_acc)

    print(f"  Low Complexity  -> Accuracy: {low_acc:.4f} | Avg Inference Time: {low_time:.2f} ms")
    print(f"  High Complexity -> Accuracy: {high_acc:.4f} | Avg Inference Time: {high_time:.2f} ms")
    print("------------------------------------------------------")


# -------------------------------
# RELATIVE IMPROVEMENT ANALYSIS
# -------------------------------

print("\n=========== RELATIVE IMPROVEMENT ANALYSIS ===========")

low_gain = low_results[-1] - low_results[0]
high_gain = high_results[-1] - high_results[0]

print(f"Low Complexity Absolute Gain:  {low_gain:.4f}")
print(f"High Complexity Absolute Gain: {high_gain:.4f}")
print("------------------------------------------------------")

low_relative = (low_gain / low_results[0]) * 100
high_relative = (high_gain / high_results[0]) * 100

print(f"Low Complexity Relative Gain:  {low_relative:.2f}%")
print(f"High Complexity Relative Gain: {high_relative:.2f}%")
print("======================================================\n")

