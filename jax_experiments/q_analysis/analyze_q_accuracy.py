"""Analyze Q-value accuracy: Mahalanobis rareness + Oracle error.

Loads collected Q-data, computes:
1. Mahalanobis rareness (distance from typical states)
2. Variance-matching alignment between predicted and real Q
3. Oracle best-head error (pick critic head closest to G_t)
4. Bins by rareness, computes per-bin MAE

Usage:
    cd RE-SAC
    conda run -n jax-rl python -m jax_experiments.q_analysis.analyze_q_accuracy \
        --env Hopper-v2
"""
import os
import pickle
import argparse
import numpy as np
from collections import defaultdict

DATA_DIR = "jax_experiments/q_analysis/results"
ALGOS = ["sac", "resac", "resac_v2", "dsac", "td3",
         "resac_v5", "resac_v5b", "resac_v6b"]
ENVS = ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]

# Per-env best new version (for focused comparison)
BEST_NEW = {
    "Hopper-v2": "resac_v5",
    "Walker2d-v2": "resac_v5b",
    "HalfCheetah-v2": "resac_v5",
    "Ant-v2": "resac_v6b",
}

LABELS = {
    'resac': 'RE-SAC v1',
    'resac_v2': 'RE-SAC v2',
    'resac_v5': 'RE-SAC v5',
    'resac_v5b': 'RE-SAC v5b',
    'resac_v6b': 'RE-SAC v6b',
    'sac': 'Vanilla SAC',
    'dsac': 'DSAC',
    'td3': 'TD3',
}

COLORS = {
    'resac': '#4488FF',
    'resac_v2': '#9944FF',
    'resac_v5': '#FF4444',
    'resac_v5b': '#FF8800',
    'resac_v6b': '#44BB44',
    'sac': 'orange',
    'dsac': '#d62728',
    'td3': '#2ca02c',
}

MARKERS = {
    'resac': 'o',
    'resac_v2': 'P',
    'resac_v5': '*',
    'resac_v5b': 'h',
    'resac_v6b': 'v',
    'sac': 's',
    'dsac': 'D',
    'td3': '^',
}


def compute_mahalanobis_rareness(obs_array):
    """Compute Mahalanobis distance of each observation from the global distribution.

    Args:
        obs_array: (N, obs_dim) array of observations

    Returns:
        (N,) array of Mahalanobis distances
    """
    mean = obs_array.mean(axis=0)
    cov = np.cov(obs_array, rowvar=False)
    # Regularize for numerical stability
    cov += np.eye(cov.shape[0]) * 1e-6

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        print("  Warning: covariance matrix singular, using pseudo-inverse")
        cov_inv = np.linalg.pinv(cov)

    diff = obs_array - mean
    # Vectorized: sqrt(diff @ cov_inv @ diff.T) for each row
    distances = np.sqrt(np.sum(np.dot(diff, cov_inv) * diff, axis=1))
    return distances


def variance_matching_alignment(q_pred, q_real):
    """Align predictions to real Q via variance matching.

    Rescale: aligned = (q_pred - mean_pred) / std_pred * std_real + mean_real

    Returns:
        aligned_pred: (N,) aligned predictions
        params: (pred_mean, pred_std, real_mean, real_std)
    """
    pred_mean = q_pred.mean()
    pred_std = q_pred.std() + 1e-8
    real_mean = q_real.mean()
    real_std = q_real.std() + 1e-8

    aligned = (q_pred - pred_mean) / pred_std * real_std + real_mean
    return aligned, (pred_mean, pred_std, real_mean, real_std)


def oracle_best_head_error(q_heads_list, q_real_array, align_params):
    """Compute oracle error: for each sample, pick the head closest to G_t.

    Args:
        q_heads_list: list of (N_heads,) arrays, one per sample
        q_real_array: (N,) real Q values
        align_params: (pred_mean, pred_std, real_mean, real_std)

    Returns:
        oracle_errors: (N,) signed error of the best head
        oracle_q_pred: (N,) the aligned Q prediction of the best head
    """
    pred_mean, pred_std, real_mean, real_std = align_params

    oracle_errors = []
    oracle_q_pred = []
    for q_heads, q_real in zip(q_heads_list, q_real_array):
        # Align all heads
        aligned_heads = (np.array(q_heads) - pred_mean) / pred_std * real_std + real_mean
        diffs = aligned_heads - q_real
        best_idx = np.argmin(np.abs(diffs))
        oracle_errors.append(diffs[best_idx])
        oracle_q_pred.append(aligned_heads[best_idx])

    return np.array(oracle_errors), np.array(oracle_q_pred)


def analyze_single_algo(data_path, global_r_clip=6.0, num_bins=30):
    """Analyze one algo's collected data.

    Returns dict with binned statistics.
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    algo = data['algo']
    steps = data['steps']
    n_steps = len(steps)
    print(f"  {algo.upper()}: {n_steps} steps loaded")

    # Extract arrays
    obs_array = np.array([s['obs'] for s in steps])
    q_pure_list = [s['q_pure'] for s in steps]
    q_real_array = np.array([s['q_real'] for s in steps])

    # Mean predicted Q (across heads)
    q_pred_mean = np.array([qp.mean() for qp in q_pure_list])

    # 1. Mahalanobis rareness
    print(f"  Computing Mahalanobis rareness...")
    rareness = compute_mahalanobis_rareness(obs_array)

    # 2. Variance matching alignment
    print(f"  Variance matching alignment...")
    aligned_pred, align_params = variance_matching_alignment(q_pred_mean, q_real_array)

    # 3. Oracle best-head error
    print(f"  Computing oracle best-head error...")
    oracle_errors, oracle_q_pred = oracle_best_head_error(
        q_pure_list, q_real_array, align_params)

    overall_mae = np.abs(oracle_errors).mean()
    print(f"  Overall Oracle MAE: {overall_mae:.2f}")

    # 4. Bin by rareness
    rareness_clipped = np.clip(rareness, 0, global_r_clip)
    bin_edges = np.linspace(0, global_r_clip, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(rareness_clipped, bin_edges[1:])  # 0-indexed bin
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Per-bin statistics
    mae_per_bin = np.full(num_bins, np.nan)
    q_pred_per_bin = np.full(num_bins, np.nan)
    q_real_per_bin = np.full(num_bins, np.nan)
    density_per_bin = np.zeros(num_bins, dtype=int)

    for b in range(num_bins):
        mask = bin_indices == b
        count = mask.sum()
        density_per_bin[b] = count
        if count > 0:
            mae_per_bin[b] = np.abs(oracle_errors[mask]).mean()
            q_pred_per_bin[b] = oracle_q_pred[mask].mean()
            q_real_per_bin[b] = q_real_array[mask].mean()

    return {
        'algo': algo,
        'mae': mae_per_bin,
        'q_pred': q_pred_per_bin,
        'q_real': q_real_per_bin,
        'density': density_per_bin,
        'total_count': n_steps,
        'overall_mae': overall_mae,
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
    }


def auto_detect_r_clip(env_name, percentile=99):
    """Auto-detect a good rareness clipping value from the data."""
    all_rareness = []
    for algo in ALGOS:
        data_path = os.path.join(DATA_DIR, f"q_data_{algo}_{env_name}.pkl")
        if not os.path.exists(data_path):
            continue
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        obs = np.array([s['obs'] for s in data['steps']])
        rareness = compute_mahalanobis_rareness(obs)
        all_rareness.append(rareness)
    if not all_rareness:
        return 6.0
    combined = np.concatenate(all_rareness)
    clip_val = np.percentile(combined, percentile)
    # Round up to nearest 0.5
    clip_val = np.ceil(clip_val * 2) / 2
    return max(clip_val, 3.0)  # minimum 3.0


def analyze_env(env_name, global_r_clip=None, num_bins=30):
    """Run analysis for all algos on one environment."""
    print(f"\n{'='*60}")
    print(f"  Analyzing: {env_name}")
    print(f"{'='*60}")

    # Auto-detect rareness clip if not specified
    if global_r_clip is None:
        global_r_clip = auto_detect_r_clip(env_name)
        print(f"  Auto-detected r_clip = {global_r_clip}")

    all_stats = {}

    for algo in ALGOS:
        data_path = os.path.join(DATA_DIR, f"q_data_{algo}_{env_name}.pkl")
        if not os.path.exists(data_path):
            print(f"  ⚠️ Skipping {algo}: {data_path} not found")
            continue

        stats = analyze_single_algo(data_path, global_r_clip, num_bins)
        all_stats[algo] = stats

    if not all_stats:
        print(f"  No data found for {env_name}!")
        return None

    # Save combined stats
    combined = {
        'meta': {
            'env_name': env_name,
            'global_r_clip': global_r_clip,
            'num_bins': num_bins,
            'bin_edges': all_stats[list(all_stats.keys())[0]]['bin_edges'],
            'bin_centers': all_stats[list(all_stats.keys())[0]]['bin_centers'],
            'algos': list(all_stats.keys()),
            'labels': LABELS,
            'colors': COLORS,
            'markers': MARKERS,
        },
        'stats': all_stats,
    }

    out_path = os.path.join(DATA_DIR, f"bin_stats_{env_name}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(combined, f)
    print(f"\n  Saved bin stats to {out_path}")

    # Print summary table
    print(f"\n  {'Algo':<12} {'Oracle MAE':>12} {'Steps':>8}")
    print(f"  {'-'*36}")
    for algo, s in all_stats.items():
        print(f"  {LABELS[algo]:<12} {s['overall_mae']:>12.2f} {s['total_count']:>8}")

    return combined


def main():
    parser = argparse.ArgumentParser(description="Analyze Q-value accuracy")
    parser.add_argument("--env", type=str, default="Hopper-v2", choices=ENVS)
    parser.add_argument("--all", action="store_true", help="Analyze all 4 envs")
    parser.add_argument("--r_clip", type=float, default=None,
                        help="Rareness clipping value (auto-detected if not set)")
    parser.add_argument("--n_bins", type=int, default=30)
    args = parser.parse_args()

    envs = ENVS if args.all else [args.env]

    for env_name in envs:
        analyze_env(env_name, args.r_clip, args.n_bins)

    print(f"\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
