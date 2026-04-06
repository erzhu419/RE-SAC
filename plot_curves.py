#!/usr/bin/env python3
"""Plot training curves for JAX experiments (SAC / TD3 / DSAC / RE-SAC).

Usage (run from RE-SAC root):
    python plot_curves.py                          # auto-detect all algos + envs
    python plot_curves.py --algos sac td3 resac    # specific algos
    python plot_curves.py --envs Hopper-v2         # specific envs
    python plot_curves.py --metric eval            # eval_reward only
    python plot_curves.py --out my_fig.png
"""
import re, os, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ── Config ───────────────────────────────────────────────────────────────────
LOG_DIR   = "jax_experiments/logs"
SAVE_ROOT = "jax_experiments/results"
SEED      = 8
MAX_ITERS = 2000

ALL_ENVS  = ["Hopper-v2", "Walker2d-v2", "HalfCheetah-v2", "Ant-v2"]
ALL_ALGOS = ["sac", "td3", "dsac", "resac"]

ALGO_STYLE = {
    "resac": dict(color="#4C9BE8", label="RE-SAC",  lw=2.5, zorder=5),
    "sac":   dict(color="#F2994A", label="SAC",     lw=1.8, zorder=4),
    "td3":   dict(color="#6FCF97", label="TD3",     lw=1.8, zorder=3),
    "dsac":  dict(color="#BB86FC", label="DSAC",    lw=1.8, zorder=2),
}

# Dark theme
BG   = "#0F1117"
CARD = "#1A1D27"
GRID = "#252837"


# ── Helpers ──────────────────────────────────────────────────────────────────
def parse_log(path):
    """Parse a training log → dict of arrays. Returns None if empty."""
    iters, rewards, eval_iters, eval_rewards = [], [], [], []
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            if not line.startswith("Iter"):
                continue
            m_iter = re.search(r"Iter\s+(\d+)", line)
            m_rew  = re.search(r"Reward:\s*([-\d.]+)", line)
            m_eval = re.search(r"Eval:\s*([-\d.]+)", line)
            if m_iter and m_rew:
                iters.append(int(m_iter.group(1)))
                rewards.append(float(m_rew.group(1)))
            if m_iter and m_eval:
                eval_iters.append(int(m_iter.group(1)))
                eval_rewards.append(float(m_eval.group(1)))
    if not iters:
        return None
    return {
        "iters":        np.array(iters),
        "rewards":      np.array(rewards),
        "eval_iters":   np.array(eval_iters),
        "eval_rewards": np.array(eval_rewards),
    }


def find_log(log_dir, algo, env, seed=8):
    """Find log file for given (algo, env, seed)."""
    # Exact match first
    exact = os.path.join(log_dir, f"{algo}_{env}_{seed}.log")
    if os.path.exists(exact):
        return exact
    # Glob fallback (handles resac_Hopper-v2_8.log etc.)
    env_short = env.split('-')[0]
    patterns = [
        os.path.join(log_dir, f"{algo}_{env}_{seed}*.log"),
        os.path.join(log_dir, f"{algo}_{env_short}*.log"),
        os.path.join(log_dir, f"{algo}*{env_short}*.log"),
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return max(hits, key=os.path.getmtime)
    # Also check results/<algo>_<env>_<seed>/logs/training.log
    results_log = os.path.join(
        SAVE_ROOT, f"{algo}_{env}_{seed}", "logs", "training.log")
    if os.path.exists(results_log):
        return results_log
    return None


def smooth(x, w=20):
    if len(x) <= w:
        return x, np.arange(len(x))
    kern = np.ones(w) / w
    return np.convolve(x, kern, 'valid'), np.arange(w - 1, len(x))


def estimate_eta(d, max_iters):
    """Return ETA string from last log line timing info."""
    if d is None or len(d['iters']) == 0:
        return "N/A"
    last_iter = int(d['iters'][-1])
    remaining = max_iters - last_iter - 1
    if remaining <= 0:
        return "Done"
    return f"{last_iter+1}/{max_iters} ({100*(last_iter+1)/max_iters:.0f}%)"


def ax_style(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors='#999', labelsize=8, length=3)
    for sp in ax.spines.values():
        sp.set_color('#2E3145')
        sp.set_linewidth(0.7)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle='--', alpha=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))


# ── Main ─────────────────────────────────────────────────────────────────────
def plot(log_dir, algos, envs, out_path, seed=8, max_iters=2000,
         metric="both", smooth_w=None):
    """
    metric: 'train' | 'eval' | 'both'
    """
    n_rows = 2 if metric == "both" else 1
    n_cols = len(envs)
    fig = plt.figure(figsize=(5.5 * n_cols, 5 * n_rows + 1.2), facecolor=BG)

    algo_labels = " / ".join(ALGO_STYLE[a]['label'] for a in algos if a in ALGO_STYLE)
    fig.text(0.5, 0.99, f"JAX Experiments — {algo_labels}",
             ha='center', va='top', fontsize=15, fontweight='bold', color='white')
    fig.text(0.5, 0.965,
             f"Brax spring  ·  Seed {seed}  ·  Target {max_iters} iters  ·  "
             f"{', '.join(e.split('-')[0] for e in envs)}",
             ha='center', va='top', fontsize=9.5, color='#777')

    top = 0.94 if n_rows == 1 else 0.93
    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        top=top, bottom=0.10,
        left=max(0.07, 0.04 + 0.03 / n_cols),
        right=0.97,
        hspace=0.50, wspace=0.28)

    row_ylabels = {
        "train": "Training Reward (smoothed)",
        "eval":  "Eval Reward",
    }
    rows = []
    if metric in ("train", "both"):
        rows.append("train")
    if metric in ("eval", "both"):
        rows.append("eval")

    # Load all data upfront
    data = {}  # (algo, env) -> parsed dict or None
    for algo in algos:
        for env in envs:
            path = find_log(log_dir, algo, env, seed)
            data[(algo, env)] = parse_log(path) if path else None

    for col, env in enumerate(envs):
        for row_idx, row_type in enumerate(rows):
            ax = fig.add_subplot(gs[row_idx, col])
            ax_style(ax)

            any_plotted = False
            progress_texts = []

            for algo in algos:
                style = ALGO_STYLE.get(algo, {})
                d = data[(algo, env)]
                if d is None:
                    continue

                clr  = style.get('color', '#888')
                lbl  = style.get('label', algo.upper())
                lw   = style.get('lw', 1.8)
                zord = style.get('zorder', 3)

                if row_type == "train" and len(d['iters']) > 5:
                    w = smooth_w or min(40, max(5, len(d['rewards']) // 8))
                    sm, idx = smooth(d['rewards'], w)
                    xs = d['iters'][idx]
                    ax.plot(xs, sm, color=clr, lw=lw, label=lbl,
                            alpha=0.92, zorder=zord)
                    ax.fill_between(xs, sm * 0.90, sm * 1.10,
                                    color=clr, alpha=0.06, zorder=zord - 1)
                    any_plotted = True
                    progress_texts.append(
                        f"{lbl}: {estimate_eta(d, max_iters)}")

                elif row_type == "eval" and len(d.get('eval_iters', [])) >= 2:
                    ax.plot(d['eval_iters'], d['eval_rewards'],
                            color=clr, lw=lw, marker='o', ms=3.5,
                            label=lbl, alpha=0.92, zorder=zord,
                            markeredgecolor='white', markeredgewidth=0.4)
                    any_plotted = True

            # Titles / labels
            if row_idx == 0:
                env_short = env.replace('-v2', '').replace('-v4', '')
                ax.set_title(env_short, color='white', fontsize=12,
                             fontweight='bold', pad=7)
                # Progress badge (top-right)
                if progress_texts:
                    badge = "  |  ".join(progress_texts)
                    ax.text(0.99, 0.98, badge,
                            ha='right', va='top',
                            transform=ax.transAxes,
                            fontsize=6.5, color='#8AF', style='italic',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      facecolor='#0F1117', alpha=0.6,
                                      edgecolor='none'))

            if col == 0:
                ax.set_ylabel(row_ylabels[row_type], color='#BBB', fontsize=9)
            if row_idx == len(rows) - 1:
                ax.set_xlabel("Iteration", color='#888', fontsize=8.5)

            if any_plotted:
                ax.legend(fontsize=8.5, loc='upper left',
                          framealpha=0.35, facecolor='#111',
                          edgecolor='#333', labelcolor='white',
                          handlelength=1.8, ncol=1)
            else:
                ax.text(0.5, 0.5, "No data yet",
                        ha='center', va='center',
                        transform=ax.transAxes,
                        color='#444', fontsize=11, style='italic')

    # Footer: per-algo summary
    summary_lines = []
    for algo in algos:
        lbl = ALGO_STYLE.get(algo, {}).get('label', algo.upper())
        parts = []
        for env in envs:
            d = data[(algo, env)]
            if d and len(d['iters']) > 0:
                parts.append(f"{env.split('-')[0]}:{d['iters'][-1]:.0f}it "
                             f"R={d['rewards'][-1]:.0f}")
        if parts:
            summary_lines.append(f"{lbl}: " + "  ".join(parts))

    for li, line in enumerate(summary_lines[:3]):
        fig.text(0.5, 0.045 - li * 0.016, line,
                 ha='center', fontsize=7, color='#555', family='monospace')

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor=BG)
    print(f"Saved → {out_path}")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot JAX experiment training curves")
    parser.add_argument("--log_dir",   default=LOG_DIR)
    parser.add_argument("--algos",     nargs="+", default=None,
                        help="Algorithms to plot (default: auto-detect)")
    parser.add_argument("--envs",      nargs="+", default=None,
                        help="Environments to plot (default: auto-detect)")
    parser.add_argument("--seed",      type=int, default=SEED)
    parser.add_argument("--max_iters", type=int, default=MAX_ITERS)
    parser.add_argument("--metric",    default="both",
                        choices=["train", "eval", "both"])
    parser.add_argument("--smooth",    type=int, default=None,
                        help="Smoothing window (default: auto)")
    parser.add_argument("--out",       default="jax_experiments/results/training_curves.png")
    args = parser.parse_args()

    # Auto-detect algos
    if args.algos is None:
        found_algos = set()
        if os.path.isdir(args.log_dir):
            for f in os.listdir(args.log_dir):
                for a in ALL_ALGOS:
                    if f.startswith(a + "_"):
                        found_algos.add(a)
        args.algos = [a for a in ALL_ALGOS if a in found_algos] or ["resac"]

    # Auto-detect envs
    if args.envs is None:
        found_envs = set()
        if os.path.isdir(args.log_dir):
            for f in os.listdir(args.log_dir):
                for e in ALL_ENVS:
                    if e in f:
                        found_envs.add(e)
        args.envs = sorted(found_envs, key=lambda e: ALL_ENVS.index(e)) or ["Hopper-v2"]

    print(f"Plotting algos={args.algos}  envs={args.envs}")
    plot(args.log_dir, args.algos, args.envs, args.out,
         seed=args.seed, max_iters=args.max_iters,
         metric=args.metric, smooth_w=args.smooth)
