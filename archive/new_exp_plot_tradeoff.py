import matplotlib.pyplot as plt

# -------------------------
# NEW EXPERIMENT RESULTS (n=67)
# -------------------------
models = ["Nano", "Small", "Medium", "Large"]

GW_BLUE = "#033C5A"
GW_GOLD = "#FFC72C"

# Low complexity: UA-DETRAC
low_accuracy = [0.8718, 0.7306, 0.7805, 0.7477]
low_time = [20.10, 36.23, 76.62, 97.16]

# High complexity: DAWN
high_accuracy = [0.4152, 0.5297, 0.5581, 0.6073]
high_time = [21.26, 41.04, 84.65, 101.37]

# -------------------------
# Plot Tradeoff
# -------------------------
plt.figure(figsize=(10, 7))

# Plotting the lines
plt.plot(low_time, low_accuracy, marker='o', linestyle='-', color=GW_BLUE, label="Low Complexity (UA-DETRAC)")
plt.plot(high_time, high_accuracy, marker='o', linestyle='-', color=GW_GOLD, label="High Complexity (DAWN)")

# Annotate each point with the model name
for i, model_name in enumerate(models):
    plt.annotate(model_name, (low_time[i], low_accuracy[i]), textcoords="offset points", xytext=(0,10), ha='center', color='blue')
    plt.annotate(model_name, (high_time[i], high_accuracy[i]), textcoords="offset points", xytext=(0,10), ha='center', color='red')

plt.xlabel("Average Inference Time (ms)")
plt.ylabel("Vehicle-Only F1 Score") # Updated metric
plt.title("Accuracy vs Inference Time (YOLO 26 Scaling Tradeoff)")
plt.ylim(0, 1.0)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig("plot_tradeoff_new_dataset.png")
plt.savefig("plot_tradeoff_new_dataset.pdf")
plt.show()