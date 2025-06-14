import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

# === Load data ===
file_path = "avg_trash_score_by_tract.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File not found: {file_path}")

df = pd.read_csv(file_path)
df = df.dropna(subset=["avg_score", "trash_score_slope", "fire_distance_slope", "avg_fire_distance_km"])

# === Derived metric
df["log_fire_risk"] = -np.log1p(df["avg_fire_distance_km"])

# === Regression function
def run_model(y, x):
    X = sm.add_constant(df[x])
    model = sm.OLS(df[y], X).fit()
    print(f"\n📈 Regression: {y} ~ {x}")
    print(model.summary())

# === Binned plot function
def plot_binned(x, y, title, filename, color):
    print(f"\n📊 Plotting {y} by {x} deciles...")
    df[f"{x}_bin"] = pd.qcut(df[x], 10, duplicates="drop")
    binned = df.groupby(f"{x}_bin")[y].mean()
    plt.figure(figsize=(10, 5))
    binned.plot(kind="bar", color=color)
    plt.ylabel(y.replace("_", " ").title())
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"📈 Saved: {filename}")

# === Correlations
print("\n🔁 Forward correlations:")
print(f"trash_score_slope → fire_distance_slope: {df['trash_score_slope'].corr(-df['fire_distance_slope']):.3f}")
print(f"avg_score → fire_distance_slope: {df['avg_score'].corr(-df['fire_distance_slope']):.3f}")
print(f"trash_score_slope → log_fire_risk: {df['trash_score_slope'].corr(df['log_fire_risk']):.3f}")
print(f"avg_score → log_fire_risk: {df['avg_score'].corr(df['log_fire_risk']):.3f}")

print("\n🔁 Reverse correlations:")
print(f"fire_distance_slope → trash_score_slope: {df['fire_distance_slope'].corr(df['trash_score_slope']):.3f}")
print(f"log_fire_risk → trash_score_slope: {df['log_fire_risk'].corr(df['trash_score_slope']):.3f}")
print(f"fire_distance_slope → avg_score: {df['fire_distance_slope'].corr(df['avg_score']):.3f}")
print(f"log_fire_risk → avg_score: {df['log_fire_risk'].corr(df['avg_score']):.3f}")

# === Forward regressions
run_model("fire_distance_slope", "trash_score_slope")
run_model("log_fire_risk", "trash_score_slope")
run_model("fire_distance_slope", "avg_score")
run_model("log_fire_risk", "avg_score")

# === Reverse regressions
run_model("trash_score_slope", "fire_distance_slope")
run_model("trash_score_slope", "log_fire_risk")
run_model("avg_score", "fire_distance_slope")
run_model("avg_score", "log_fire_risk")

# === Plots (forward direction only)
plot_binned("trash_score_slope", "fire_distance_slope",
            "🔥 Fire Distance Slope vs. Trash Slope (Binned)",
            "fire_distance_slope_vs_trash_slope_bins.png", "firebrick")

plot_binned("trash_score_slope", "log_fire_risk",
            "🔥 Log Fire Risk vs. Trash Slope (Binned)",
            "log_fire_risk_vs_trash_slope_bins.png", "darkorange")

plot_binned("avg_score", "fire_distance_slope",
            "🔥 Fire Distance Slope vs. Avg Trash Score (Binned)",
            "fire_distance_slope_vs_avg_trash_bins.png", "steelblue")

plot_binned("avg_score", "log_fire_risk",
            "🔥 Log Fire Risk vs. Avg Trash Score (Binned)",
            "log_fire_risk_vs_avg_trash_bins.png", "orange")
