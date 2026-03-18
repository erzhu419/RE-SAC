
import numpy as np
import matplotlib.pyplot as plt
import os

# Use RE-SAC root as base dir (one level up from plots/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration: Path -> Label, Color, Linestyle, Type (npy or npz)
experiments = {
    os.path.join(BASE_DIR, "results/baseline_sac"): ("Baseline (SAC)", "black", "-", "npy"),
    os.path.join(BASE_DIR, "results/ensemble_2/logs"): ("Ensemble 2", "#1f77b4", "-", "npy"),
    os.path.join(BASE_DIR, "results/ensemble_5/logs"): ("Ensemble 5", "#ff7f0e", "-", "npy"),
    os.path.join(BASE_DIR, "results/ensemble_10/logs"): ("Ensemble 10", "#2ca02c", "-", "npy"),
    os.path.join(BASE_DIR, "results/ensemble_20/logs"): ("Ensemble 20", "#d62728", "-", "npy"),
    os.path.join(BASE_DIR, "results/ensemble_40/logs"): ("Ensemble 40", "#9467bd", "-", "npy"),
    os.path.join(BASE_DIR, "results/aleatoric_only/logs"): ("Aleatoric Only", "#8c564b", "--", "npy"),
    os.path.join(BASE_DIR, "results/epistemic_only/logs"): ("Epistemic Only", "#e377c2", "--", "npy"),
    # DSAC-v1
    os.path.join(BASE_DIR, "results/dsac_v1/full_rewards_debug.npy"): ("DSAC-v1", "#00ced1", "-.", "npy_direct"),
    # BAC
    os.path.join(BASE_DIR, "results/bac/logs"): ("BAC", "#9400D3", ":", "npy"),
}

OUTPUT_FILE = os.path.join(BASE_DIR, "results/all_experiments_comparison.png")

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
for path, (label, color, linestyle, file_type) in experiments.items():
    try:
        if file_type == "npy":
            reward_path = os.path.join(path, "rewards.npy")
            if not os.path.exists(reward_path):
                print(f"Warning: {reward_path} not found.")
                continue
            rewards = np.load(reward_path)
        elif file_type == "npy_direct":
            if not os.path.exists(path):
                print(f"Warning: {path} not found.")
                continue
            rewards = np.load(path)
        elif file_type == "npz":
            if not os.path.exists(path):
                print(f"Warning: {path} not found.")
                continue
            data = np.load(path)
            rewards = data['rewards']
        else:
            continue

        x = np.arange(len(rewards))
        smoothed = smooth_curve(rewards, factor=0.95)
        plt.plot(x, smoothed, label=label, color=color, linestyle=linestyle, linewidth=2.5 if "DSAC" in label else 2.0)
        print(f"Loaded {label}: {len(rewards)} episodes, Last 100 Mean: {np.mean(rewards[-100:])}")

    except Exception as e:
        print(f"Error loading {path} ({label}): {e}")

plt.title("Comparative Analysis: Ensemble vs SAC vs DSAC Versions", fontsize=20, fontweight='bold')
plt.xlabel("Episode", fontsize=16)
plt.ylabel("Cumulative Reward (Smoothed)", fontsize=16)
plt.legend(loc='lower right', fontsize=14, ncol=2)
plt.tick_params(labelsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300)
print(f"Comparison plot saved to {OUTPUT_FILE}")
