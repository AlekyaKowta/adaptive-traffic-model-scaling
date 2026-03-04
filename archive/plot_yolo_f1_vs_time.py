import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "figure_yolo_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# DATA (From your evaluation)
# --------------------------

# LOW COMPLEXITY
yolo26_time_low = [16.2, 33.3, 67.0, 85.7]
yolo26_f1_low   = [0.7113, 0.7742, 0.7560, 0.7376]

yolo8_time_low  = [16.2, 32.5, 67.6, 121.6]
yolo8_f1_low    = [0.7492, 0.7488, 0.7560, 0.7453]

# HIGH COMPLEXITY
yolo26_time_high = [15.9, 30.3, 63.8, 80.0]
yolo26_f1_high   = [0.3940, 0.4931, 0.5735, 0.5758]

yolo8_time_high  = [16.6, 30.6, 64.5, 112.0]
yolo8_f1_high    = [0.4271, 0.5142, 0.5623, 0.5768]

plt.rcParams.update({
    "axes.edgecolor": "#000000",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})

def plot_graph(x1, y1, x2, y2, title, filename, y_min, y_max):
    fig, ax = plt.subplots(figsize=(6, 5))

    # YOLOv26
    ax.plot(
        x1, y1,
        "-o",
        color="#033C5A",
        linewidth=2,
        markersize=6,
        label="YOLOv26"
    )

    # YOLOv8
    ax.plot(
        x2, y2,
        "-o",
        color="#FFC72C",
        linewidth=2,
        markersize=6,
        label="YOLOv8"
    )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Inference Time (ms)")
    ax.set_ylabel("F1 Score (IoU = 0.5)")

    # Tightened Y-axis
    ax.set_ylim(y_min, y_max)

    # Subtle grid
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    ax.legend(loc="best", fontsize=9)

    ax.set_facecolor("white")
    ax.spines[:].set_color("black")

    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, filename + ".png"),
                dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, filename + ".pdf"),
                bbox_inches="tight")

    plt.close()
    print(f"Saved: {filename}.png and .pdf")


# Generate Final Graphs

plot_graph(
    yolo26_time_low, yolo26_f1_low,
    yolo8_time_low, yolo8_f1_low,
    "Inference Time vs F1 Score (Low Complexity)",
    "f1_vs_time_low",
    0.70, 0.78
)

plot_graph(
    yolo26_time_high, yolo26_f1_high,
    yolo8_time_high, yolo8_f1_high,
    "Inference Time vs F1 Score (High Complexity)",
    "f1_vs_time_high",
    0.38, 0.60
)

