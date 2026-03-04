import os
import json

class UniversalParser:
    """Consolidates TrafficCAM JSON and DAWN TXT parsing logic."""
    
    # Combined labels from your evaluate_folder.py and final_plots.py
    VEHICLE_LABELS = ["car", "truck", "bus", "motorcycle", "auto", "lcv", "lmv", "vehicle"]

    @staticmethod
    def get_ground_truth(img_path):
        """Detects format and returns a list of bounding boxes [x1, y1, x2, y2]."""
        # 1. Try TrafficCAM JSON
        json_path = img_path.replace(".jpg", ".json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                boxes = []
                for shape in data.get("shapes", []):
                    if shape["label"].lower() in UniversalParser.VEHICLE_LABELS:
                        import numpy as np
                        pts = np.array(shape["points"])
                        boxes.append([pts[:,0].min(), pts[:,1].min(), pts[:,0].max(), pts[:,1].max()])
                return boxes, "TrafficCAM"

        # 2. Try DAWN TXT (YOLO format)
        txt_path = img_path.replace(".jpg", ".txt")
        if os.path.exists(txt_path):
            # Note: Requires image dimensions to de-normalize coordinates
            import cv2
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            boxes = []
            with open(txt_path, "r") as f:
                for line in f:
                    # Logic from your final_plots.py
                    parts = line.strip().split()
                    cx, cy, bw, bh = map(float, parts[1:])
                    x_min, y_min = (cx - bw/2) * w, (cy - bh/2) * h
                    x_max, y_max = (cx + bw/2) * w, (cy + bh/2) * h
                    boxes.append([x_min, y_min, x_max, y_max])
            return boxes, "DAWN"

        return [], "Unknown"