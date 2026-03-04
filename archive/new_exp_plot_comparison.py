import matplotlib.pyplot as plt

# ---------------------------------------------------------
# DATA INPUT: NEW EXPERIMENT (n=67 Balanced Sample)
# ---------------------------------------------------------
models = ["Nano", "Small", "Medium", "Large"]

# YOLOv8 Results (Based on your v8 baseline runs)
v8_low_acc = [0.7314, 0.6045, 0.6866, 0.6582] # UA-DETRAC
v8_low_time = [22.65, 36.35, 71.04, 125.25]
v8_high_acc = [0.4150, 0.5348, 0.6308, 0.6134] # DAWN
v8_high_time = [21.11, 35.23, 66.42, 116.10]

# YOLO 26 Results (The "Hero" data)
v26_low_acc = [0.8718, 0.7306, 0.7805, 0.7477] # UA-DETRAC
v26_low_time = [20.10, 36.23, 76.62, 97.16]
v26_high_acc = [0.4152, 0.5297, 0.5581, 0.6073] # DAWN
v26_high_time = [21.26, 41.04, 84.65, 101.37]

# ---------------------------------------------------------
# PLOT 1: GENERATIONAL TRADE-OFF (ACCURACY vs SPEED)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

# Plot YOLOv8 (Dashed lines)
plt.plot(v8_low_time, v8_low_acc, 'o--', color='lightblue', label='v8 - Simple (UA-DETRAC)')
plt.plot(v8_high_time, v8_high_acc, 'o--', color='salmon', label='v8 - Complex (DAWN)')

# Plot YOLO 26 (Solid lines)
plt.plot(v26_low_time, v26_low_acc, 'o-', color='blue', label='v26 - Simple (UA-DETRAC)', linewidth=2)
plt.plot(v26_high_time, v26_high_acc, 'o-', color='red', label='v26 - Complex (DAWN)', linewidth=2)

# Annotate points for YOLO 26
for i, txt in enumerate(models):
    plt.annotate(txt, (v26_high_time[i], v26_high_acc[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.xlabel("Average Inference Time (ms)")
plt.ylabel("Accuracy Ratio")
plt.title("Generational Trade-off: v8 vs v26 (n=67 Balanced)")
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig("new_exp_tradeoff.png")
plt.show()

# ---------------------------------------------------------
# PLOT 2: RELATIVE GAIN COMPARISON (NANO TO LARGE)
# ---------------------------------------------------------
# High Complexity Scaling Benefit
v8_high_gain = (v8_high_acc[-1] - v8_high_acc[0]) / v8_high_acc[0] * 100
v26_high_gain = (v26_high_acc[-1] - v26_high_acc[0]) / v26_high_acc[0] * 100

labels = ['YOLOv8', 'YOLO 26']
gains = [v8_high_gain, v26_high_gain]

plt.figure(figsize=(7, 5))
bars = plt.bar(labels, gains, color=['gray', 'green'], alpha=0.8)
plt.ylabel("Accuracy Gain (%)")
plt.title("Scaling Benefit in Complex Scenes (DAWN)\n(Nano to Large Model)")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', fontweight='bold')

plt.ylim(0, max(gains) + 10)
plt.savefig("new_exp_scaling_gain.png")
plt.show()