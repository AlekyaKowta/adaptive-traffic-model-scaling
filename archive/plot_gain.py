import matplotlib.pyplot as plt

# Your FINAL evaluation numbers
models = ["Nano", "Small", "Medium", "Large"]

low_acc =  [0.5522, 0.6466, 0.6949, 0.7192]
high_acc = [0.3854, 0.4849, 0.5352, 0.5402]

# Compute gain relative to Nano
low_gain = [acc - low_acc[0] for acc in low_acc]
high_gain = [acc - high_acc[0] for acc in high_acc]

plt.figure()

plt.plot(models, low_gain, marker='o', label="Low Complexity")
plt.plot(models, high_gain, marker='o', label="High Complexity")

plt.title("Accuracy Gain vs Model Size (Relative to Nano)")
plt.xlabel("Model Size")
plt.ylabel("Accuracy Gain Over Nano")
plt.legend()

plt.show()

