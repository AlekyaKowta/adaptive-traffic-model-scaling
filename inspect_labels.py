import os
import json

FOLDER = "TrafficCAM/raw/Complexity/Low_Complexity/Low_All"

unique_labels = set()

for file in os.listdir(FOLDER):
    if file.endswith(".json"):
        path = os.path.join(FOLDER, file)

        with open(path, "r") as f:
            data = json.load(f)

            for shape in data["shapes"]:
                label = shape["label"]
                unique_labels.add(label)

print("Unique labels found in ground truth:")
print(sorted(unique_labels))

