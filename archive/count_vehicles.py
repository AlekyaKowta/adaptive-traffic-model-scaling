import os
import json

def compute_average(folder_path):
    counts = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(os.path.join(folder_path, file)) as f:
                data = json.load(f)
                counts.append(len(data["shapes"]))
    
    if len(counts) == 0:
        return 0
    
    return sum(counts) / len(counts)


base_path = "TrafficCAM/raw/Complexity"

low_path = os.path.join(base_path, "Low_Complexity")
high_path = os.path.join(base_path, "High_Complexity")

print("\n--- LOW COMPLEXITY FOLDERS ---")
for folder in os.listdir(low_path):
    folder_full_path = os.path.join(low_path, folder)
    if os.path.isdir(folder_full_path):
        avg = compute_average(folder_full_path)
        print(f"{folder} -> Average vehicles: {avg:.2f}")

print("\n--- HIGH COMPLEXITY FOLDERS ---")
for folder in os.listdir(high_path):
    folder_full_path = os.path.join(high_path, folder)
    if os.path.isdir(folder_full_path):
        avg = compute_average(folder_full_path)
        print(f"{folder} -> Average vehicles: {avg:.2f}")

