"""
Train TD3 on Pendulum-v1 using best Optuna params, then plot tuned vs baseline.

Outputs (saved to results/Pendulum-v1/TD3/figures/):
  TD3_tuned_vs_baseline_Pendulum_v1.png
  TD3_tuned_vs_baseline_Pendulum_v1_smoothed.png
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import experiment_config as cfg
from algorithms.td3 import train_td3
from utils import moving_average


# ============================================
# Config
# ============================================

JSON_PATH     = "results/Best_Parameters/td3_best_params_Pendulum-v1.json"
BASELINE_NPZ  = "results/Pendulum-v1/TD3/all_returns.npz"
OUTPUT_DIR    = "figures"
SEEDS         = [0, 1, 2]
SMOOTH_WINDOW = 9


# ============================================
# Load best params and baseline
# ============================================

with open(JSON_PATH) as f:
    best = json.load(f)

best_params = best["best_params"]
env_name    = best["env_name"]

baseline_data = np.load(BASELINE_NPZ)
baseline_all  = baseline_data["baseline_all_returns"]   # (n_seeds, n_episodes)
num_episodes  = baseline_all.shape[1]

print(f"Baseline: {baseline_all.shape[0]} seeds, {num_episodes} episodes")
print(f"Best params: {best_params}")


# ============================================
# Train with best params
# ============================================

train_config = {
    "actor_lr":          best_params["actor_lr"],
    "critic_lr":         cfg.TD3_CRITIC_LR,
    "num_episodes":      num_episodes,
    "hidden_dim":        cfg.HIDDEN_DIM,
    "gamma":             cfg.GAMMA,
    "tau":               best_params["tau"],
    "exploration_noise": cfg.TD3_EXPLORATION_NOISE,
    "policy_noise":      cfg.TD3_POLICY_NOISE,
    "noise_clip":        cfg.TD3_NOISE_CLIP,
    "policy_delay":      best_params["policy_delay"],
    "buffer_size":       cfg.TD3_BUFFER_SIZE,
    "minimal_size":      cfg.TD3_MINIMAL_SIZE,
    "batch_size":        cfg.TD3_BATCH_SIZE,
    "env_name":          env_name,
    "show_progress":     True,
}

print(f"\nTraining TD3 (tuned) on {env_name} ...")

tuned_returns = []
for seed in SEEDS:
    returns = train_td3(**train_config, seed=seed)
    final = float(np.mean(returns[-20:]))
    print(f"  seed {seed} | final mean (last 20 ep): {final:.2f}")
    tuned_returns.append(returns)

tuned_all = np.array(tuned_returns)   # (n_seeds, n_episodes)


# ============================================
# Statistics helpers
# ============================================

def mean_and_ci(curves):
    """curves: (n_seeds, n_episodes) → mean and 95% CI per episode."""
    n = curves.shape[0]
    mean = curves.mean(axis=0)
    std  = curves.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    t    = stats.t.ppf(0.975, df=max(n - 1, 1))
    return mean, t * std / np.sqrt(n)


def smooth_curves(curves, window):
    return np.array([moving_average(curves[i], window) for i in range(curves.shape[0])])


# ============================================
# Plotting helper
# ============================================

def save_figure(b_mean, b_ci, t_mean, t_ci, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))

    xb = np.arange(len(b_mean))
    xt = np.arange(len(t_mean))

    ax.plot(xb, b_mean, label="Baseline")
    ax.fill_between(xb, b_mean - b_ci, b_mean + b_ci, alpha=0.2)

    ax.plot(xt, t_mean, label="Tuned (best params)")
    ax.fill_between(xt, t_mean - t_ci, t_mean + t_ci, alpha=0.2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {path}")


# ============================================
# Raw
# ============================================

b_mean, b_ci = mean_and_ci(baseline_all)
t_mean, t_ci = mean_and_ci(tuned_all)

save_figure(
    b_mean, b_ci, t_mean, t_ci,
    title=f"TD3 on {env_name}",
    filename="TD3_tuned_vs_baseline_Pendulum_v1.png",
)

# ============================================
# Smoothed
# ============================================

b_mean_sm, b_ci_sm = mean_and_ci(smooth_curves(baseline_all, SMOOTH_WINDOW))
t_mean_sm, t_ci_sm = mean_and_ci(smooth_curves(tuned_all,    SMOOTH_WINDOW))

save_figure(
    b_mean_sm, b_ci_sm, t_mean_sm, t_ci_sm,
    title=f"TD3 on {env_name} — Smoothed (window={SMOOTH_WINDOW})",
    filename="TD3_tuned_vs_baseline_Pendulum_v1_smoothed.png",
)
