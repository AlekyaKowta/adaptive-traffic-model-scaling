import matplotlib.pyplot as plt

# Data from your latest terminal output
models = ["Nano", "Small", "Medium", "Large"]
low_acc =  [0.8718, 0.7306, 0.7805, 0.7477]
high_acc = [0.4152, 0.5297, 0.5581, 0.6073]

# Compute Absolute Gain relative to Nano
low_gain = [acc - low_acc[0] for acc in low_acc]
high_gain = [acc - high_acc[0] for acc in high_acc]

plt.figure(figsize=(8, 6))
plt.plot(models, low_gain, marker='o', linewidth=2, label="Low Complexity (UA-DETRAC)")
plt.plot(models, high_gain, marker='o', linewidth=2, label="High Complexity (DAWN)", color='red')

# Add a reference line at 0
plt.axhline(0, color='black', linestyle='--', alpha=0.3)

plt.title("Accuracy Gain vs Model Size (Relative to Nano)\n[New Balanced Dataset, n=67]")
plt.xlabel("Model Size")
plt.ylabel("Accuracy Gain (Difference from Nano)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.savefig("new_data_gain_plot.png")
plt.show()