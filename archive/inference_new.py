import os
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import f1_score

# -------------------------------
# CONFIGURATION
# -------------------------------
# Using your "Alekya Dataset" (DAWN) for Bad Weather
BAD_WEATHER_FOLDER = "DAWNDataset/High_Complexity"
LIMIT = 67

# Models for Comparison
MODELS_V8 = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
MODELS_V26 = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt"]

# GW Official Theme Colors
GW_BLUE = "#033C5A"
GW_GOLD = "#FFC72C"

# Consistent Vehicle Classes for Grant Consistency
VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike", "vehicle"]

def get_txt_labels(txt_path):
    if not os.path.exists(txt_path): return []
    with open(txt_path, "r") as f:
        return [line.strip().split()[0] for line in f if line.strip()]

def evaluate_family(model_list, folder_path):
    avg_f1s = []
    avg_times = []
    
    images = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith(".jpg")][:LIMIT]

    for model_path in model_list:
        print(f"  Evaluating {model_path}...")
        model = YOLO(model_path)
        img_f1s = []
        img_times = []

        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            gt_path = img_path.replace(".jpg", ".txt")
            
            # 1. Ground Truth
            gt_labels = get_txt_labels(gt_path)
            gt_binary = [1] * len(gt_labels) 

            # 2. Inference + Timing
            start = time.time()
            results = model(img_path, verbose=False, conf=0.5)
            img_times.append((time.time() - start) * 1000)

            # 3. Predictions (Fixed: ID to String mapping)
            pred_binary = []
            for box in results[0].boxes:
                label = model.names[int(box.cls)].lower()
                if any(v in label for v in VEHICLE_CLASSES):
                    pred_binary.append(1)

            # 4. F1 Calculation with Zero Division fix
            max_len = max(len(gt_binary), len(pred_binary))
            if max_len == 0:
                img_f1s.append(1.0)
            else:
                y_true = gt_binary + [0] * (max_len - len(gt_binary))
                y_pred = pred_binary + [0] * (max_len - len(pred_binary))
                img_f1s.append(f1_score(y_true, y_pred, zero_division=1))

        avg_f1s.append(sum(img_f1s) / len(img_f1s))
        avg_times.append(sum(img_times) / len(img_times))
        
    return avg_f1s, avg_times

if __name__ == "__main__":
    print(f"\n=========== BAD WEATHER EVALUATION: v8 vs v26 (n={LIMIT}) ===========")
    
    v8_f1, v8_t = evaluate_family(MODELS_V8, BAD_WEATHER_FOLDER)
    v26_f1, v26_t = evaluate_family(MODELS_V26, BAD_WEATHER_FOLDER)

    # Plotting with GW formatting
    plt.figure(figsize=(8, 6))
    plt.plot(v26_t, v26_f1, 'o-', color=GW_BLUE, label="YOLOv26 (Grant Target)", linewidth=2)
    plt.plot(v8_t, v8_f1, 'o-', color=GW_GOLD, label="YOLOv8 (Baseline)", linewidth=2)

    plt.xlabel("Average Inference Time (ms)")
    plt.ylabel("Vehicle-Only F1 Score")
    plt.title("Performance Trade-off in Bad Weather")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # Required export formats
    plt.savefig("bad_weather_tradeoff.png")
    plt.savefig("bad_weather_tradeoff.pdf")
    print("\nResults saved as bad_weather_tradeoff.png and .pdf")
    plt.show()