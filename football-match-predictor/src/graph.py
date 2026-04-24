import json
import matplotlib.pyplot as plt

# Use your saved results file
path = "football-match-predictor/models/form_window_validation.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# sort by window
data = sorted(data, key=lambda x: x["window"])

windows = [d["window"] for d in data]
rf_test = [d["rf_test_accuracy"] * 100 for d in data]
lr_test = [d["lr_test_accuracy"] * 100 for d in data]
rf_train = [d["rf_train_accuracy"] * 100 for d in data]

# best RF point
best_idx = max(range(len(rf_test)), key=lambda i: rf_test[i])

plt.figure(figsize=(10, 6))
plt.plot(windows, rf_test, marker="o", linewidth=2, label="Random Forest (Test)")
plt.plot(windows, lr_test, marker="s", linewidth=2, label="Logistic Regression (Test)")
plt.plot(windows, rf_train, linestyle="--", alpha=0.6, label="Random Forest (Train)")

plt.scatter([windows[best_idx]], [rf_test[best_idx]], color="red", zorder=5)
plt.annotate(
    f"Best RF: w={windows[best_idx]}, {rf_test[best_idx]:.2f}%",
    (windows[best_idx], rf_test[best_idx]),
    textcoords="offset points",
    xytext=(10, 8),
)

plt.title("Form Window vs Accuracy")
plt.xlabel("Form window (last N matches)")
plt.ylabel("Accuracy (%)")
plt.xticks(windows)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()