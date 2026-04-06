#!/usr/bin/env python3
"""Plot training curves for RE-SAC experiments.

Usage:
    python jax_experiments/plot_curves.py                          # default
    python jax_experiments/plot_curves.py --log_dir path/to/logs   # custom
    python jax_experiments/plot_curves.py --envs Hopper-v2         # single env
    python jax_experiments/plot_curves.py --out results/fig.png    # custom output
"""
import re, os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


# ── Defaults ────────────────────────────────────────────────────────────────
ALL_ENVS = ["Hopper-v2", "Walker2d-v2", "HalfCheetah-v2", "Ant-v2"]
COLOR = "#4C9BE8"
LABEL = "RE-SAC"

# Theme
BG   = "#0F1117"
CARD = "#1A1D27"
GRID = "#252837"


# ── Helpers ─────────────────────────────────────────────────────────────────
def parse_log(path):
    """Parse a training log file → dict of arrays."""
    iters, rewards, eval_iters, eval_rewards = [], [], [], []
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            if not line.startswith("Iter"):
                continue
            m_iter = re.search(r"Iter\s+(\d+)", line)
            m_rew  = re.search(r"Reward:\s+([\d.]+)", line)
            m_eval = re.search(r"Eval:\s+([\d.]+)", line)
            if m_iter and m_rew:
                iters.append(int(m_iter.group(1)))
                rewards.append(float(m_rew.group(1)))
            if m_iter and m_eval:
                eval_iters.append(int(m_iter.group(1)))
                eval_rewards.append(float(m_eval.group(1)))
    if not iters:
        return None
    return {
        "iters": np.array(iters),
        "rewards": np.array(rewards),
        "eval_iters": np.array(eval_iters),
        "eval_rewards": np.array(eval_rewards),
    }


def smooth(x, w=20):
    """Moving average with automatic window sizing."""
    if len(x) <= w:
        return x, np.arange(len(x))
    kern = np.ones(w) / w
    return np.convolve(x, kern, 'valid'), np.arange(w - 1, len(x))


def find_log(log_dir, env, seed=8):
    """Find the most-recently-modified log matching this env."""
    import glob
    env_short = env.split('-')[0]  # 'Hopper-v2' -> 'Hopper'
    patterns = [
        os.path.join(log_dir, f"resac_{env_short}*.log"),
        os.path.join(log_dir, f"resac*{env_short}*.log"),
        os.path.join(log_dir, f"resac_{env}_{seed}*.log"),
    ]
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat))
    if matches:
        return max(set(matches), key=os.path.getmtime)
    return os.path.join(log_dir, f"resac_{env}_{seed}.log")  # fallback


# ── Main ────────────────────────────────────────────────────────────────────
def plot(log_dir, envs, out_path, seed=8, max_iters=2000, title_extra=""):
    n_envs = len(envs)
    fig = plt.figure(figsize=(5.5 * n_envs, 10), facecolor=BG)

    # Title
    fig.text(0.5, 0.975, f"{LABEL}  —  Training Curves{title_extra}",
             ha='center', va='top', fontsize=16, fontweight='bold', color='white')
    fig.text(0.5, 0.944, f"Brax spring · Seed {seed} · Target {max_iters} iters",
             ha='center', va='top', fontsize=10, color='#777')

    gs = gridspec.GridSpec(2, n_envs, figure=fig,
                           top=0.92, bottom=0.09,
                           left=max(0.07, 0.04 + 0.03 / n_envs),
                           right=0.97,
                           hspace=0.50, wspace=0.28)

    ROW_YLABEL = ["Training Reward (smoothed)", "Eval Reward"]

    # Load data
    all_data = {}
    for env in envs:
        path = find_log(log_dir, env, seed)
        all_data[env] = parse_log(path)

    for col, env in enumerate(envs):
        for row in range(2):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor(CARD)
            ax.tick_params(colors='#999', labelsize=8, length=3)
            for sp in ax.spines.values():
                sp.set_color('#2E3145')
                sp.set_linewidth(0.7)
            ax.grid(True, color=GRID, linewidth=0.6, linestyle='--', alpha=0.8)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

            plotted = False
            d = all_data[env]
            if d is not None:
                if row == 0 and len(d['iters']) > 5:
                    sw = min(30, max(5, len(d['rewards']) // 8))
                    sm, idx = smooth(d['rewards'], sw)
                    xs = d['iters'][idx]
                    ax.plot(xs, sm, color=COLOR, lw=2.0, label=LABEL, alpha=0.95, zorder=3)
                    ax.fill_between(xs, sm * 0.88, sm * 1.12,
                                    color=COLOR, alpha=0.08, zorder=2)
                    plotted = True

                elif row == 1 and len(d['eval_iters']) >= 2:
                    ax.plot(d['eval_iters'], d['eval_rewards'],
                            color=COLOR, lw=2.2, marker='o', ms=4,
                            label=LABEL, alpha=0.95, zorder=3,
                            markeredgecolor='white', markeredgewidth=0.5)
                    plotted = True

            # Labels
            if row == 0:
                env_short = env.replace('-v2', '').replace('-v4', '').replace('-v5', '')
                ax.set_title(env_short, color='white', fontsize=12,
                             fontweight='bold', pad=7)
                # Progress badge
                if d is not None and len(d['iters']) > 0:
                    n_iters = d['iters'][-1]
                    pct = n_iters / max_iters * 100
                    ax.text(0.98, 0.97, f"{n_iters}/{max_iters}  ({pct:.0f}%)",
                            ha='right', va='top', transform=ax.transAxes,
                            fontsize=7.5, color='#8AF', style='italic')

            if col == 0:
                ax.set_ylabel(ROW_YLABEL[row], color='#BBB', fontsize=9)
            if row == 1:
                ax.set_xlabel("Iteration", color='#888', fontsize=8.5)

            if plotted:
                ax.legend(fontsize=8.5, loc='upper left',
                          framealpha=0.3, facecolor='#111',
                          edgecolor='#333', labelcolor='white',
                          handlelength=1.8)
            else:
                ax.text(0.5, 0.5, "No data yet", ha='center', va='center',
                        transform=ax.transAxes, color='#444', fontsize=11,
                        style='italic')

    # Footer stats
    lines = []
    for env in envs:
        d = all_data[env]
        if d and len(d['iters']) > 0:
            lines.append(
                f"RE-SAC·{env.split('-')[0]}: "
                f"{d['iters'][-1]} iters, R={d['rewards'][-1]:.0f}"
            )
    if lines:
        mid = (len(lines) + 1) // 2
        fig.text(0.5, 0.028, "   |   ".join(lines[:mid]),
                 ha='center', fontsize=7.5, color='#555', family='monospace')
        if len(lines) > mid:
            fig.text(0.5, 0.012, "   |   ".join(lines[mid:]),
                     ha='center', fontsize=7.5, color='#555', family='monospace')

    plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor=BG)
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RE-SAC training curves")
    parser.add_argument("--log_dir", default="jax_experiments/logs",
                        help="Directory containing .log files")
    parser.add_argument("--envs", nargs="+", default=None,
                        help="Environments to plot (default: auto-detect)")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--out", default="jax_experiments/results/training_curves.png",
                        help="Output image path")
    parser.add_argument("--title", default="", help="Extra text for title")
    args = parser.parse_args()

    # Auto-detect envs from log files if not specified
    if args.envs is None:
        found = set()
        if os.path.isdir(args.log_dir):
            for f in os.listdir(args.log_dir):
                for env in ALL_ENVS:
                    if env in f:
                        found.add(env)
        args.envs = sorted(found, key=lambda e: ALL_ENVS.index(e)) or ["Hopper-v2"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot(args.log_dir, args.envs, args.out,
         seed=args.seed, max_iters=args.max_iters, title_extra=args.title)
