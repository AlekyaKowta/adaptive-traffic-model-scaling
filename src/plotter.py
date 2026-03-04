import json
import matplotlib.pyplot as plt
import os
import numpy as np

class ResearchPlotter:
    def __init__(self, results_path="results/master_metrics.json", output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(results_path, "r") as f:
            self.data = json.load(f)

        # Official GW Theme Colors
        self.GW_BLUE = "#033C5A"
        self.GW_GOLD = "#FFC72C"

    def plot_generational_comparison(self, scenario, title, filename):
        """Plots YOLOv8 vs YOLOv26 for a single scenario (e.g., Weather_Extreme)."""
        plt.figure(figsize=(10, 6))
        
        if scenario not in self.data:
            print(f"⚠️ Missing data for {scenario}")
            return

        scenario_data = self.data[scenario]
        
        # Split models into families
        v8_models = [m for m in scenario_data.keys() if 'yolov8' in m]
        v26_models = [m for m in scenario_data.keys() if 'yolo26' in m]

        # Sort by inference time (Nano -> Large)
        v8_models.sort(key=lambda m: scenario_data[m]["avg_inference_ms"])
        v26_models.sort(key=lambda m: scenario_data[m]["avg_inference_ms"])

        # Plot v8 (Dashed Baseline)
        v8_times = [scenario_data[m]["avg_inference_ms"] for m in v8_models]
        v8_f1s = [scenario_data[m]["f1_score"] for m in v8_models]
        plt.plot(v8_times, v8_f1s, 'o--', color='gray', label="YOLOv8 (Baseline)", alpha=0.6)

        # Plot v26 (Solid Target)
        v26_times = [scenario_data[m]["avg_inference_ms"] for m in v26_models]
        v26_f1s = [scenario_data[m]["f1_score"] for m in v26_models]
        plt.plot(v26_times, v26_f1s, 'o-', color=self.GW_BLUE, label="YOLOv26 (Target)", linewidth=2.5)

        # Annotate model scales
        for m in v26_models:
            label = m.replace('yolo26', '').replace('.pt', '').upper()
            plt.annotate(label, (scenario_data[m]["avg_inference_ms"], scenario_data[m]["f1_score"]), 
                         textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', color=self.GW_BLUE)

        plt.xlabel("Average Inference Time (ms)")
        plt.ylabel("Vehicle-Only F1 Score")
        plt.title(f"{title}\nGenerational Comparison", fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(os.path.join(self.output_dir, f"{filename}.png"), dpi=300)
        plt.close()

    def plot_scaling_gain(self, scenarios):
        """Bar chart showing F1 gain from Nano to Large for different complexities."""
        plt.figure(figsize=(8, 6))
        
        labels = []
        gains = []

        for scenario in scenarios:
            if scenario not in self.data: continue
            
            # Use YOLOv26 as the primary scaling metric
            models = [m for m in self.data[scenario].keys() if 'yolo26' in m]
            models.sort(key=lambda m: self.data[scenario][m]["avg_inference_ms"])
            
            if len(models) >= 2:
                nano_f1 = self.data[scenario][models[0]]["f1_score"]
                large_f1 = self.data[scenario][models[-1]]["f1_score"]
                gain = ((large_f1 - nano_f1) / nano_f1) * 100 if nano_f1 > 0 else 0
                
                labels.append(scenario.replace('_', ' '))
                gains.append(gain)

        bars = plt.bar(labels, gains, color=[self.GW_BLUE, self.GW_GOLD, 'salmon', 'green'][:len(labels)])
        
        plt.ylabel("F1 Score Gain (%) [Nano → Large]")
        plt.title("Scaling Benefit: Simple vs Complex Scenes", fontweight='bold')
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}%", ha='center', fontweight='bold')

        plt.savefig(os.path.join(self.output_dir, "scaling_gain_comparison.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    plotter = ResearchPlotter()
    
    # 1. Plot individual generational gaps
    plotter.plot_generational_comparison("Traffic_Clear", "Low Complexity: Clear Traffic", "v8_vs_v26_clear")
    plotter.plot_generational_comparison("Weather_Extreme", "High Complexity: Extreme Weather (DAWN)", "v8_vs_v26_weather")
    
    # 2. Plot Scaling Gain (The key 'Grant' figure showing complex scenes benefit more from Large models)
    plotter.plot_scaling_gain(["Traffic_Clear", "Weather_Extreme"])
    
    print("✅ All research graphs saved to /outputs")