import matplotlib.pyplot as plt


GW_BLUE = "#033C5A"
GW_GOLD = "#FFC72C"

# Model sizes
models = ["Nano", "Small", "Medium", "Large"]

# RESULTS FROM YOUR NEW DATASET EVALUATION (n=67)
# Low Complexity: UA-DETRAC (Clear/Sunny)
# High Complexity: DAWN (Dust/Tornado)
low_accuracy = [0.8718, 0.7306, 0.7805, 0.7477]
high_accuracy = [0.4152, 0.5297, 0.5581, 0.6073]

plt.figure(figsize=(10, 6))

# Plotting both lines for direct comparison
plt.plot(models, low_accuracy, marker='o', label="Low Complexity (Simple)", color=GW_BLUE, linewidth=2)
plt.plot(models, high_accuracy, marker='o', label="High Complexity (Complex)", color=GW_GOLD, linewidth=2)

# Graph styling to match professional standards
plt.xlabel("Model Size")
plt.ylabel("Accuracy Ratio")
plt.title("YOLO 26: Model Size vs Accuracy (New Dataset Comparison)")
plt.ylim(0, 1.0) # Accuracy is a ratio from 0 to 1
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Save the final image for the grant report
plt.savefig("plot_results_new_dataset.png")
plt.savefig("plot_results_new_dataset.pdf")
plt.show()