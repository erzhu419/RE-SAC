"""Metrics logging: per-iteration recording and periodic numpy save."""
import os
import time
import numpy as np
from collections import defaultdict


class Logger:
    """Accumulates per-iteration metrics and saves them as .npy files."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.data = defaultdict(list)
        self.start_time = time.time()

    def log(self, key: str, value: float):
        self.data[key].append(value)

    def log_dict(self, d: dict):
        for k, v in d.items():
            self.data[k].append(v)

    def save(self):
        for key, values in self.data.items():
            np.save(os.path.join(self.log_dir, f"{key}.npy"), np.array(values))

    def last(self, key: str, default=0.0):
        vals = self.data.get(key, [])
        return vals[-1] if vals else default

    def mean_last_n(self, key: str, n: int = 10, default=0.0):
        vals = self.data.get(key, [])
        if len(vals) == 0:
            return default
        return float(np.mean(vals[-n:]))

    def elapsed(self):
        return time.time() - self.start_time

    def print_status(self, iteration: int, extra: str = ""):
        elapsed = self.elapsed()
        reward = self.mean_last_n("eval_reward", 5)
        q_std = self.last("q_std_mean")
        print(
            f"Iter {iteration:4d} | "
            f"Reward: {reward:8.1f} | "
            f"Q-std: {q_std:6.2f} | "
            f"Time: {elapsed:6.0f}s"
            + (f" | {extra}" if extra else "")
        )
