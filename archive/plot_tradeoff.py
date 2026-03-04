import matplotlib.pyplot as plt

# -------------------------
# YOUR FINAL RESULTS
# -------------------------

models = ["Nano", "Small", "Medium", "Large"]

# Low complexity
low_accuracy = [0.5522, 0.6466, 0.6949, 0.7192]
low_time = [22.65, 36.35, 71.04, 125.25]

# High complexity
high_accuracy = [0.3854, 0.4849, 0.5352, 0.5402]
high_time = [21.11, 35.23, 66.42, 116.10]

# -------------------------
# Plot
# -------------------------

plt.figure(figsize=(8,6))

plt.plot(low_time, low_accuracy, marker='o', label="Low Complexity")
plt.plot(high_time, high_accuracy, marker='o', label="High Complexity")

for i in range(len(models)):
    plt.text(low_time[i], low_accuracy[i], models[i])
    plt.text(high_time[i], high_accuracy[i], models[i])

plt.xlabel("Inference Time (ms)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Inference Time (Model Scaling Tradeoff)")
plt.legend()
plt.grid(True)

plt.show()

