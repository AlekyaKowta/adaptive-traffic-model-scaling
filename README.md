
# Adaptive Traffic Model Scaling: Dynamic Resource Tuning

**Graduate Research Project | The George Washington University**

## 📌 Project Overview

This repository provides experimental validation for a dynamic traffic management system that tunes computational resources based on physical world conditions. We demonstrate that while lightweight models (Nano) are sufficient for clear, low-traffic scenes, complex environments (heavy traffic or extreme weather) necessitate larger model scales to maintain safety-critical accuracy.

## 🏗 Modular Architecture Refactor

The project has been restructured from fragmented scripts into a systematic, modular layout to improve maintainability and research reproducibility:

* **`src/data_utils.py` (Universal Parser)**: Consolidates handling for multiple ground truth formats. It automatically detects and parses **TrafficCAM (JSON)** and **DAWN (TXT)** labels.
* **`src/engine.py` (Smart Execution)**: Features a persistent results cache (`results/master_metrics.json`). It checks existing data before execution, skipping redundant YOLO inferences to save hours of processing time.
* **`src/metrics.py` (Standardized Evaluation)**: Centralizes logic for **F1 Score** and **Spatial Matching (IoU $\ge$ 0.5)**. This ensures a fair generational comparison between YOLOv8 and YOLOv26.
* **`src/plotter.py` (Iterative Visualization)**: Decouples data generation from plotting. This allows for near-instant updates to figures without re-running ML models.

## 🚀 Workflow

### 1. Installation

```bash
pip install -r alekya-requirements.txt

```

### 2. Running the Experiment

To execute the full evaluation across all complexity scenarios (TrafficCAM and DAWN):

```bash
python main.py

```

*The engine will automatically populate the JSON cache. If interrupted, it will resume from the last saved state.*

### 3. Generating Figures

To generate publication-ready comparison graphs:

```bash
python src/plotter.py

```

## 📊 Summary of Results

The experimental data confirms that "Accuracy Gain" from scaling is significantly more pronounced in complex scenes:

| Model Family | Avg. Inference (ms) | F1 Score (Simple) | F1 Score (Complex) |
| --- | --- | --- | --- |
| **YOLOv8 (Baseline)** | 20 - 150ms | Baseline Performance | Significant Drop |
| **YOLOv26 (Target)** | 20 - 115ms | High | Robust Recovery |

### Key Takeaways

* **Low Complexity (Clear Traffic)**: Smaller models perform within a narrow margin of Large models, justifying a "Resource-Saving" mode.
* **High Complexity (Extreme Weather)**: The scaling benefit is non-linear; larger models provide a substantial F1-score boost required for reliable detection in low-visibility environments.
