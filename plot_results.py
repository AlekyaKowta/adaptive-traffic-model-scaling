import matplotlib.pyplot as plt

# Model sizes
models = ["Nano", "Small", "Medium", "Large"]

# Vehicle-only results
low_acc =  [0.6613, 0.6437, 0.6884, 0.6930]
high_acc = [0.4461, 0.5627, 0.6209, 0.6253]

# -------------------------
# Low Complexity Plot
# -------------------------

plt.figure()
plt.plot(models, low_acc, marker='o')
plt.xlabel("Model Size")
plt.ylabel("Accuracy")
plt.title("Model Size vs Accuracy (Low Complexity)")
plt.ylim(0, 1)
plt.grid(True)
plt.show()

# -------------------------
# High Complexity Plot
# -------------------------

plt.figure()
plt.plot(models, high_acc, marker='o')
plt.xlabel("Model Size")
plt.ylabel("Accuracy")
plt.title("Model Size vs Accuracy (High Complexity)")
plt.ylim(0, 1)
plt.grid(True)
plt.show()

