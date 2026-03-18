
import numpy as np
import matplotlib.pyplot as plt
import os

# Use RE-SAC root as base dir (one level up from plots/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration: Path -> Label, Color, Linestyle, MainFile, StdFile(Optional)
experiments = {
    os.path.join(BASE_DIR, "results/baseline_sac"): ("Baseline (SAC)", "black", "-", "q_values.npy", "q_stds_episode.npy"),
    os.path.join(BASE_DIR, "results/ensemble_5/logs"): ("Ensemble 5", "#ff7f0e", "-", "q_values_episode.npy", "q_stds_episode.npy"),
    os.path.join(BASE_DIR, "results/ensemble_10/logs"): ("Ensemble 10", "#2ca02c", "-", "q_values_episode.npy", "q_stds_episode.npy"),
    os.path.join(BASE_DIR, "results/ensemble_20/logs"): ("Ensemble 20", "#d62728", "-", "q_values_episode.npy", "q_stds_episode.npy"),
    os.path.join(BASE_DIR, "results/ensemble_40/logs"): ("Ensemble 40", "#9467bd", "-", "q_values_episode.npy", "q_stds_episode.npy"),
    os.path.join(BASE_DIR, "results/aleatoric_only/logs"): ("Aleatoric Only", "#8c564b", "--", "q_values_episode.npy", "q_stds_episode.npy"),
    # DSAC-v1
    os.path.join(BASE_DIR, "results/dsac_v1/logs"): ("DSAC-v1", "#00ced1", "-.", "q_values_episode.npy", "q_stds_episode.npy"),
    # BAC
    os.path.join(BASE_DIR, "results/bac/logs"): ("BAC", "#9400D3", ":", "q_values_episode.npy", "q_stds_episode.npy"),
}

OUTPUT_FILE = os.path.join(BASE_DIR, "results/all_q_values_comparison.png")

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return np.array(smoothed_points)

plt.figure(figsize=(18, 10))

# Iterate and Plot
for log_dir, (label, color, linestyle, filename, std_filename) in experiments.items():
    file_path = os.path.join(log_dir, filename)

    if not os.path.exists(file_path):
        print(f"Warning: Missing {file_path}, skipping.")
        continue

    try:
        q_data = np.load(file_path)
        q_std = None

        # Handle Baseline which might be (N, 2)
        if q_data.ndim == 2:
            q_std = np.std(q_data, axis=1)
            q_data = np.mean(q_data, axis=1)
        elif std_filename:
            std_path = os.path.join(log_dir, std_filename)
            if os.path.exists(std_path):
                q_std = np.load(std_path)

        x = np.arange(len(q_data))
        smoothed_mean = smooth_curve(q_data, factor=0.95)
        plt.plot(x, smoothed_mean, label=label, color=color, linestyle=linestyle, linewidth=2.5 if "DSAC" in label else 2.0)

        # Plot Uncertainty Band (Smoothed)
        if q_std is not None and len(q_std) == len(x):
            smoothed_std = smooth_curve(q_std, factor=0.95)
            upper = smoothed_mean + 2 * smoothed_std
            lower = smoothed_mean - 2 * smoothed_std
            plt.fill_between(x, lower, upper, color=color, alpha=0.15)

        print(f"Loaded {label}: {len(q_data)} episodes, Last 100 Mean: {np.mean(q_data[-100:])}")

    except Exception as e:
        print(f"Error loading {label}: {e}")

plt.title("Comparative Analysis: Q-Value Estimation with Uncertainty (Mean ± 2σ)", fontsize=20, fontweight='bold')
plt.xlabel("Episode", fontsize=16)
plt.ylabel("Q-Value", fontsize=16)
plt.legend(loc='lower right', fontsize=14, ncol=2)
plt.tick_params(labelsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300)
print(f"Q-Value comparison plot saved to {OUTPUT_FILE}")
