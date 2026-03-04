import os
from src.engine import ExperimentEngine

# --- CONFIGURATION & PATHS ---
SCENARIOS = {
    "Traffic_Clear": "data/TrafficCAM/raw/Complexity/Low_Complexity/Low_All",
    "Traffic_Rush":  "data/TrafficCAM/raw/Complexity/High_Complexity/High_All",
    "Weather_Simple": "data/DAWNDataset/Low_Complexity",
    "Weather_Extreme": "data/DAWNDataset/High_Complexity"
}

# The model families we are comparing
MODELS_V8 = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
MODELS_V26 = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt"]

# Combine them into one list for the loop
ALL_MODELS = MODELS_V8 + MODELS_V26

def main():
    engine = ExperimentEngine(results_path="results/master_metrics.json")

    print("🚀 Starting Unified Comparison: YOLOv8 vs YOLOv26")
    print(f"Tracking {len(SCENARIOS)} scenarios across {len(ALL_MODELS)} models.")
    print("-------------------------------------------------------")

    for name, path in SCENARIOS.items():
        if not os.listdir(path): # Quick check if folder is empty
            print(f"⚠️ Warning: No data in {path}. Skipping.")
            continue

        # Run the engine for ALL models in this scenario
        engine.run_scenario(
            scenario_name=name,
            folder_path=path,
            model_names=ALL_MODELS, # Now running both v8 and v26
            limit=67 
        )

    print("\n✅ Comparison Complete. Data cached in results/master_metrics.json.")
    print("Run 'python src/plotter.py' to visualize the generational gap.")

if __name__ == "__main__":
    main()