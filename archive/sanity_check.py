import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

TEST_FOLDER = "TrafficCAM/raw/Complexity/Low_Complexity/Low_All"

all_detected_classes = set()

for file in os.listdir(TEST_FOLDER):
    if file.endswith(".jpg"):
        image_path = os.path.join(TEST_FOLDER, file)
        results = model(image_path, verbose=False)

        for cls in results[0].boxes.cls:
            class_name = model.names[int(cls)]
            all_detected_classes.add(class_name)

print("\nClasses detected by YOLO:")
print(sorted(all_detected_classes))

