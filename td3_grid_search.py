"""
1-D grid search for TD3 on Pendulum-v1.

For each of the 3 hyperparameters (actor_lr, tau, policy_delay), 3 candidate
values are swept while the other two are held at the Optuna best. One of the
3 candidates is always the Optuna best itself.

Training: 500 episodes x 3 seeds per candidate.

Outputs  (results/Pendulum-v1/TD3/grid_search/):
  Data:
    grid_{param_name}.npz  — shape (n_values, n_seeds, n_episodes)
  Figures (figures/):
    TD3_grid_{param_name}_Pendulum_v1.png
    TD3_grid_{param_name}_Pendulum_v1_smoothed.png
    TD3_baseline_Pendulum_v1.png
    TD3_baseline_Pendulum_v1_smoothed.png
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

import experiment_config as cfg
from algorithms.td3 import train_td3
from utils import moving_average


# ============================================
# Config
# ============================================

JSON_PATH    = "td3_best_params_Pendulum-v1.json"
BASELINE_NPZ = "results/Pendulum-v1/TD3/all_returns.npz"
OUTPUT_DIR   = "results/Pendulum-v1/TD3/grid_search"
FIGURE_DIR   = os.path.join(OUTPUT_DIR, "figures")

ENV_NAME      = "Pendulum-v1"
SEEDS         = [0, 1, 2]
NUM_EPISODES  = 500
SMOOTH_WINDOW = 9


# ============================================
# Load Optuna best params
# ============================================

with open(JSON_PATH) as f:
    best = json.load(f)

B = best["best_params"]
best_actor_lr    = B["actor_lr"]
best_tau         = B["tau"]
best_policy_delay = int(B["policy_delay"])

print(f"Optuna best: actor_lr={best_actor_lr:.4e}, tau={best_tau:.4e}, policy_delay={best_policy_delay}")


# ============================================
# Grid definitions
# One of the 3 values is always the Optuna best.
# ============================================

GRIDS = {
    "actor_lr":    [5e-4,  best_actor_lr, 5e-3],
    "tau":         [1e-3,  best_tau,      1e-2],
    "policy_delay":[2,     best_policy_delay, 8],
}

# Fixed values for params not being swept (always the Optuna best)
FIXED = {
    "actor_lr":    best_actor_lr,
    "tau":         best_tau,
    "policy_delay": best_policy_delay,
}


# ============================================
# Base training config (non-tuned params)
# ============================================

BASE = {
    "critic_lr":         cfg.TD3_CRITIC_LR,
    "num_episodes":      NUM_EPISODES,
    "hidden_dim":        cfg.HIDDEN_DIM,
    "gamma":             cfg.GAMMA,
    "exploration_noise": cfg.TD3_EXPLORATION_NOISE,
    "policy_noise":      cfg.TD3_POLICY_NOISE,
    "noise_clip":        cfg.TD3_NOISE_CLIP,
    "buffer_size":       cfg.TD3_BUFFER_SIZE,
    "minimal_size":      cfg.TD3_MINIMAL_SIZE,
    "batch_size":        cfg.TD3_BATCH_SIZE,
    "env_name":          ENV_NAME,
    "show_progress":     True,
}


# ============================================
# Statistics helpers
# ============================================

def mean_and_ci(curves):
    """(n_seeds, n_episodes) -> mean, 95% CI half-width."""
    n    = curves.shape[0]
    mean = curves.mean(axis=0)
    std  = curves.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    t    = stats.t.ppf(0.975, df=max(n - 1, 1))
    return mean, t * std / np.sqrt(n)


def smooth_per_seed(curves):
    return np.array([moving_average(curves[i], SMOOTH_WINDOW) for i in range(curves.shape[0])])


# ============================================
# Plotting helper
# ============================================

def _save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fmt_value(v):
    """Compact label for a hyperparameter value."""
    if isinstance(v, float):
        return f"{v:.2e}"
    return str(v)


def plot_param_grid(param_name, values, all_returns_list, best_value, smooth):
    """
    One figure: 3 curves (one per value), mean ± 95% CI.
    all_returns_list: list of (n_seeds, n_episodes) arrays, one per value.
    """
    safe_env = ENV_NAME.replace("-", "_")
    suffix   = "_smoothed" if smooth else ""
    title    = f"TD3 on {ENV_NAME} — {param_name}"

    fig, ax = plt.subplots(figsize=(10, 6))

    for v, curves in zip(values, all_returns_list):
        data = smooth_per_seed(curves) if smooth else curves
        mean, ci = mean_and_ci(data)
        x   = np.arange(1, len(mean) + 1)
        lbl = fmt_value(v)
        if v == best_value:
            lbl += " (Optuna best)"
        ax.plot(x, mean, label=lbl)
        ax.fill_between(x, mean - ci, mean + ci, alpha=0.2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fname = f"TD3_grid_{param_name}_{safe_env}{suffix}.png"
    _save_fig(fig, os.path.join(FIGURE_DIR, fname))


def plot_baseline(baseline_all, smooth):
    safe_env = ENV_NAME.replace("-", "_")
    suffix   = "_smoothed" if smooth else ""
    title    = f"TD3 Baseline on {ENV_NAME}"

    data = smooth_per_seed(baseline_all) if smooth else baseline_all
    mean, ci = mean_and_ci(data)
    x = np.arange(1, len(mean) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, mean, label="Baseline (mean)")
    ax.fill_between(x, mean - ci, mean + ci, alpha=0.2, label="95% CI")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fname = f"TD3_baseline_{safe_env}{suffix}.png"
    _save_fig(fig, os.path.join(FIGURE_DIR, fname))


# ============================================
# Main
# ============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Baseline ----
print("\nLoading baseline...")
baseline_data = np.load(BASELINE_NPZ)
n_avail       = baseline_data["baseline_all_returns"].shape[1]
n_base        = min(500, n_avail)
baseline_all  = baseline_data["baseline_all_returns"][:, :n_base]
print(f"  Baseline shape after crop: {baseline_all.shape}")

for smooth in [False, True]:
    plot_baseline(baseline_all, smooth)

# ---- Grid search ----
for param_name, values in GRIDS.items():

    print(f"\n{'='*55}")
    print(f"  Sweeping: {param_name}  ->  {values}")
    print(f"{'='*55}")

    all_returns_list = []

    for v in values:
        config = {
            **BASE,
            **FIXED,           # fix all tuned params to Optuna best
            param_name: v,     # override the one being swept
        }

        seed_returns = []
        for seed in SEEDS:
            print(f"  {param_name}={fmt_value(v)}  seed={seed}")
            returns = train_td3(**config, seed=seed)
            seed_returns.append(returns)

        all_returns_list.append(np.array(seed_returns))   # (n_seeds, n_episodes)

    # Stack: (n_values, n_seeds, n_episodes)
    stacked = np.array(all_returns_list)

    # Save
    npz_path = os.path.join(OUTPUT_DIR, f"grid_{param_name}.npz")
    np.savez(npz_path, all_returns=stacked, values=np.array(values, dtype=object))
    print(f"  Saved {npz_path}")

    # Figures
    best_value = FIXED[param_name]   # Optuna best for this param
    for smooth in [False, True]:
        plot_param_grid(param_name, values, all_returns_list, best_value, smooth)

print("\nDone.")
