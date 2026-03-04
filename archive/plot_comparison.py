import matplotlib.pyplot as plt

# ---------------------------------------------------------
# DATA INPUT (Using your actual terminal results)
# ---------------------------------------------------------
models = ["Nano", "Small", "Medium", "Large"]

# YOLOv8 Results
v8_low_acc = [0.5522, 0.6466, 0.6949, 0.7192]
v8_low_time = [24.13, 41.81, 84.76, 154.57]
v8_high_acc = [0.3854, 0.4849, 0.5352, 0.5402]
v8_high_time = [23.50, 39.68, 83.70, 136.36]

# YOLO 26 Results 
v26_low_acc = [0.5498, 0.6734, 0.7498, 0.7739]
v26_low_time = [24.32, 43.93, 85.11, 106.98]
v26_high_acc = [0.2716, 0.4454, 0.5208, 0.5243]
v26_high_time = [23.64, 40.13, 79.66, 100.03]

# ---------------------------------------------------------
# PLOT 1: GENERATIONAL TRADE-OFF (ACCURACY vs SPEED)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

# Plot YOLOv8
plt.plot(v8_low_time, v8_low_acc, 'o--', color='lightblue', label='v8 - Low Comp')
plt.plot(v8_high_time, v8_high_acc, 'o--', color='salmon', label='v8 - High Comp')

# Plot YOLO 26
plt.plot(v26_low_time, v26_low_acc, 'o-', color='blue', label='v26 - Low Comp', linewidth=2)
plt.plot(v26_high_time, v26_high_acc, 'o-', color='red', label='v26 - High Comp', linewidth=2)

# Labels for each point
for i, txt in enumerate(models):
    plt.annotate(txt, (v26_low_time[i], v26_low_acc[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel("Inference Time (ms)")
plt.ylabel("Accuracy Ratio")
plt.title("Trade-off Comparison: YOLOv8 vs YOLO 26")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig("comparison_tradeoff.png")
plt.show()

# ---------------------------------------------------------
# PLOT 2: RELATIVE GAIN COMPARISON (NANO TO LARGE)
# ---------------------------------------------------------
# Calculate Gains
v8_high_gain = (v8_high_acc[-1] - v8_high_acc[0]) / v8_high_acc[0] * 100
v26_high_gain = (v26_high_acc[-1] - v26_high_acc[0]) / v26_high_acc[0] * 100

labels = ['YOLOv8', 'YOLO 26']
gains = [v8_high_gain, v26_high_gain]

plt.figure(figsize=(6, 5))
bars = plt.bar(labels, gains, color=['gray', 'green'])
plt.ylabel("Accuracy Gain (%)")
plt.title("Scaling Benefit in High Complexity Scenes\n(Nano to Large)")

# Add percentage labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

plt.savefig("scaling_gain_bar.png")
plt.show()