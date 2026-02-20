# Adaptive Traffic Model Scaling
Model scaling analysis for adaptive traffic monitoring using YOLOv8 and a curated TrafficCAM subset.

## Overview

This project evaluates the trade-off between model size, accuracy, and inference time for traffic vehicle counting under different scene complexities.

We compare four YOLOv8 models:
- YOLOv8n (Nano)
- YOLOv8s (Small)
- YOLOv8m (Medium)
- YOLOv8l (Large)

Evaluation is performed on:
- Low Complexity traffic scenes (120 images)
- High Complexity traffic scenes (120 images)

## Evaluation Metric

Accuracy is computed using count-based matching:

accuracy = min(predicted, ground_truth) / max(predicted, ground_truth)

Average inference time per image (ms) is also recorded.

## Repository Structure

TrafficCAM/
├── raw/
│   └── Complexity/
│       ├── Low_Complexity/
│       │   └── Low_All/
│       └── High_Complexity/
│           └── High_All/

DAWNDataset/
└── Complexity/
    ├── Low_Complexity/
    └── High_Complexity/



Files:
- evaluate_folder.py        → Full evaluation script
- plot_results.py           → Accuracy comparison plot
- plot_gain.py              → Relative gain over Nano
- plot_tradeoff.py          → Accuracy vs Inference time tradeoff
- count_vehicles.py         → Basic vehicle counting utility
- inference_new.py          → Inferences on DAWN Dataset
- new_exp_plot_gain.py      → Relative gain over Nano
- new_exp_plot_results.py   → Accuracy comparison plot
- new_exp_plot_tradeoff.py  → Accuracy vs Inference time tradeoff

## How to Run

Install dependencies:

pip install ultralytics matplotlib

Run evaluation:

python evaluate_folder.py

Generate plots:

python plot_results.py
python plot_gain.py
python plot_tradeoff.py

python new_exp_plot_results.py
python new_exp_plot_gain.py
python new_exp_plot_tradeoff.py

# Final Plots

## Setup & Intallation
1. Create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate 

2. Install dependencies:
pip install -r alekya_requirements/requirements.txt

## Running the Evaluation
The final_plots.py script executes the full evaluation pipeline, including spatial IoU matching and dataset-level F1 calculation.  

To generate 3 benchmark graphs:
python3 final_plots.py

## Methodology Alignment
Our pipeline is synchronized to a unified research standard to ensure consistency across the lab:
1. Spatial Matching: Uses Intersection over Union ($IoU \ge 0.5$) for True Positive validation.
2. Filtering: Restricted to Vehicle-Only classes (Car, Truck, Bus, Auto, LCV, LMV, etc.).
3. Inference Settings: Confidence threshold set to 0.25.Coordinate Systems: Automatically de-normalizes DAWN TXT coordinates to absolute pixels for spatial accuracy.

## Result Data

| Model | Low Complexity | High Traffic | Bad Weather |
| :--- | :---: | :---: | :---: |
| **yolov8n** | 0.7093 | 0.4020 | 0.6298 |
| **yolov8l** | 0.7836 | 0.5384 | 0.7417 |
| **yolo26n** | 0.6756 | 0.3648 | 0.5567 |
| **yolo26l** | 0.7710 | 0.5342 | 0.6811 |

## Plots
All three plots can be found in final_plots_all_3 folder.

## Summary

Results demonstrate that model scaling improves accuracy more significantly in low complexity scenes than in high complexity scenes, while inference time increases substantially with model size.

This highlights the trade-off between computational cost and accuracy for adaptive traffic monitoring systems.

