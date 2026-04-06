"""Plot Oracle Q-value comparison figure.

Two-panel figure matching the bus transit style:
  Top: Mean Q-Value Predictions vs Ground Truth (by rareness bin)
  Bottom: Oracle Q-Error (MAE) & Data Density (by rareness bin)

Usage:
    cd RE-SAC
    python -m jax_experiments.q_analysis.plot_q_comparison --env Hopper-v2
"""
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

DATA_DIR = "jax_experiments/q_analysis/results"
ENVS = ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]

# ── Style ────────────────────────────────────────────────────────────────────
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12
LINE_WIDTH = 2.5
MARKER_SIZE = 6
FIG_SIZE = (16, 14)
DPI = 300


def plot_env(env_name):
    """Generate the two-panel comparison figure for one environment."""
    stats_path = os.path.join(DATA_DIR, f"bin_stats_{env_name}.pkl")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"No bin stats found: {stats_path}\n"
                                "Run analyze_q_accuracy.py first.")

    with open(stats_path, 'rb') as f:
        data = pickle.load(f)

    meta = data['meta']
    stats = data['stats']

    bin_centers = meta['bin_centers']
    global_r_clip = meta['global_r_clip']
    algos = meta['algos']
    labels = meta['labels']
    colors = meta['colors']
    markers = meta['markers']
    num_bins = meta['num_bins']

    bar_width_ratio = 0.8 / len(algos)
    offsets = np.linspace(
        -bar_width_ratio * (len(algos) - 1) / 2,
         bar_width_ratio * (len(algos) - 1) / 2,
        len(algos)
    )
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.2

    mpl.rcParams.update({'font.size': TICK_FONTSIZE})

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=FIG_SIZE, sharex=True,
        gridspec_kw={'height_ratios': [1, 1]}
    )
    ax2 = ax1.twinx()

    for i, algo in enumerate(algos):
        if algo not in stats:
            continue
        s = stats[algo]

        # Per-algo ground truth (dashed, same color)
        q_real = s['q_real']
        valid_gt = ~np.isnan(q_real)
        if valid_gt.any():
            ax0.plot(bin_centers[valid_gt], q_real[valid_gt],
                     color=colors[algo], linestyle='--', linewidth=1.5,
                     alpha=0.6)

        # Q-value prediction curve (solid)
        q_pred = s['q_pred']
        valid_q = ~np.isnan(q_pred)
        if valid_q.any():
            ax0.plot(bin_centers[valid_q], q_pred[valid_q],
                     color=colors[algo], linewidth=LINE_WIDTH,
                     marker=markers[algo], markersize=MARKER_SIZE,
                     label=labels[algo])

        # MAE curve
        mae = s['mae']
        valid_m = ~np.isnan(mae)
        if valid_m.any():
            ax1.plot(bin_centers[valid_m], mae[valid_m],
                     color=colors[algo], linewidth=LINE_WIDTH,
                     marker=markers[algo], markersize=MARKER_SIZE,
                     label=f'{labels[algo]} MAE', zorder=5)

        # Density bars (percentage)
        density = s['density']
        total = s['total_count']
        density_pct = (density / total) * 100 if total > 0 else density
        nonzero = density > 0
        if nonzero.any():
            ax2.bar(bin_centers[nonzero] + offsets[i] * bin_width,
                    density_pct[nonzero],
                    width=bar_width_ratio * bin_width,
                    alpha=0.15, color=colors[algo],
                    label=f'{labels[algo]} Density (%)')

    # Add dummy legend entry for dashed lines (ground truth)
    ax0.plot([], [], 'k--', linewidth=1.5, alpha=0.6, label='Ground Truth (MC Return)')

    # ── Formatting ───────────────────────────────────────────────────────────
    env_short = env_name.replace('-v2', '')
    ax0.set_title(f'Q-Value Predictions vs Ground Truth by Rareness ({env_short})',
                  fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax0.set_ylabel('Q-Value', fontsize=LABEL_FONTSIZE)
    ax0.legend(loc='best', fontsize=LEGEND_FONTSIZE)
    ax0.tick_params(labelsize=TICK_FONTSIZE, labelbottom=True)
    ax0.grid(alpha=0.3)

    ax1.set_title(f'Oracle Q-Error (MAE) & Data Density ({env_short})',
                  fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax1.set_xlabel(f'Mahalanobis Rareness (Clipped at {global_r_clip})',
                   fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('Mean Absolute Error of Best Head', fontsize=LABEL_FONTSIZE)
    ax1.axhline(0, color='red', linestyle='-', alpha=0.3)
    ax1.tick_params(labelsize=TICK_FONTSIZE)
    ax1.grid(alpha=0.3)

    ax2.set_ylabel('Data Density (%)', color='gray', fontsize=LABEL_FONTSIZE)
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=TICK_FONTSIZE)

    lines1, lbs1 = ax1.get_legend_handles_labels()
    lines2, lbs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbs1 + lbs2,
               loc='upper left', fontsize=LEGEND_FONTSIZE)

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(DATA_DIR, f"oracle_q_comparison_{env_name}.png")
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    print(f"Saved figure to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot Q-value comparison")
    parser.add_argument("--env", type=str, default="Hopper-v2", choices=ENVS)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    envs = ENVS if args.all else [args.env]
    for env_name in envs:
        try:
            plot_env(env_name)
        except FileNotFoundError as e:
            print(f"⚠️ {e}")

    print("\n✅ Plotting complete!")


if __name__ == "__main__":
    main()
