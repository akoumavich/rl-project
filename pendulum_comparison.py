"""
Train and compare PPO, SAC, TD3 on Pendulum-v1.

For each algorithm:
  1. Baseline:      3 seeds, 200 episodes
  2. Optuna tuning: 10 trials x 3 seeds
  3. Tuned run:     3 seeds, 200 episodes with best params

Saves per algorithm  (results/Pendulum-v1/<ALG>/):
  baseline_all_returns.npz
  baseline_episode_returns.csv
  optuna_tuned_all_returns.npz
  optuna_tuned_episode_returns.csv
  optuna_trials.csv
  <alg>_best_params_Pendulum-v1.json

Figures per algorithm  (results/Pendulum-v1/<ALG>/figures/):
  <ALG>_baseline_vs_tuned_Pendulum_v1_episode.png
  <ALG>_baseline_vs_tuned_Pendulum_v1_episode_smoothed.png
  <ALG>_baseline_vs_tuned_Pendulum_v1_timestep.png
  <ALG>_baseline_vs_tuned_Pendulum_v1_timestep_smoothed.png
  <ALG>_param_scan_Pendulum_v1.png

Combined figures  (results/Pendulum-v1/figures/):
  comparison_Pendulum_v1_episode.png
  comparison_Pendulum_v1_episode_smoothed.png
  comparison_Pendulum_v1_timestep.png
  comparison_Pendulum_v1_timestep_smoothed.png
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from scipy import stats

import experiment_config as cfg
from compare import run_multi_seed
from utils import moving_average


# ============================================
# Constants
# ============================================

ENV_NAME          = "Pendulum-v1"
SEEDS             = [0, 1, 2]
NUM_EPISODES      = 200
N_TRIALS          = 10
SMOOTH_WINDOW     = 9
STEPS_PER_EPISODE = 200   # Pendulum-v1 always truncates at 200 steps

RESULTS_ROOT  = f"results/{ENV_NAME}"
COMBINED_DIR  = f"{RESULTS_ROOT}/figures"


# ============================================
# Base configs  (one per algorithm)
# ============================================

_common = {
    "env_name":     ENV_NAME,
    "num_episodes": NUM_EPISODES,
    "hidden_dim":   cfg.HIDDEN_DIM,
    "gamma":        cfg.GAMMA,
}

BASE_CONFIGS = {
    "PPO": {
        **_common,
        "algorithm":  "PPO",
        "actor_lr":   cfg.PPO_CONTINUOUS_ACTOR_LR,
        "critic_lr":  cfg.PPO_CONTINUOUS_CRITIC_LR,
        "lmbda":      cfg.PPO_CONTINUOUS_LMBDA,
        "epochs":     cfg.PPO_CONTINUOUS_EPOCHS,
        "eps":        cfg.PPO_CONTINUOUS_EPS,
    },
    "SAC": {
        **_common,
        "algorithm":    "SAC",
        "actor_lr":     cfg.SAC_ACTOR_LR,
        "critic_lr":    cfg.SAC_CRITIC_LR,
        "alpha_lr":     cfg.SAC_ALPHA_LR,
        "tau":          cfg.SAC_TAU,
        "buffer_size":  cfg.SAC_BUFFER_SIZE,
        "minimal_size": cfg.SAC_MINIMAL_SIZE,
        "batch_size":   cfg.SAC_BATCH_SIZE,
    },
    "TD3": {
        **_common,
        "algorithm":         "TD3",
        "actor_lr":          cfg.TD3_ACTOR_LR,
        "critic_lr":         cfg.TD3_CRITIC_LR,
        "tau":               cfg.TD3_TAU,
        "exploration_noise": cfg.TD3_EXPLORATION_NOISE,
        "policy_noise":      cfg.TD3_POLICY_NOISE,
        "noise_clip":        cfg.TD3_NOISE_CLIP,
        "policy_delay":      cfg.TD3_POLICY_DELAY,
        "buffer_size":       cfg.TD3_BUFFER_SIZE,
        "minimal_size":      cfg.TD3_MINIMAL_SIZE,
        "batch_size":        cfg.TD3_BATCH_SIZE,
    },
}

PARAM_SPACES = {alg: cfg.PARAM_SPACES[alg] for alg in ["PPO", "SAC", "TD3"]}


# ============================================
# Statistics
# ============================================

def mean_and_ci(curves):
    """(n_seeds, n_points) -> (mean, 95% CI half-width) per point."""
    n    = curves.shape[0]
    mean = curves.mean(axis=0)
    std  = curves.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    t    = stats.t.ppf(0.975, df=max(n - 1, 1))
    return mean, t * std / np.sqrt(n)


def smooth_per_seed(curves, window):
    return np.array([moving_average(curves[i], window) for i in range(curves.shape[0])])


# ============================================
# Save helpers
# ============================================

def save_returns(all_returns, seeds, label, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    npz_path = os.path.join(output_dir, f"{label}_all_returns.npz")
    np.savez(
        npz_path,
        **{
            f"{label}_all_returns":  all_returns,
            f"{label}_mean_returns": all_returns.mean(axis=0),
            f"{label}_std_returns":  all_returns.std(axis=0, ddof=1),
        }
    )
    print(f"  Saved {npz_path}")

    rows = [
        {"seed": seed, "episode": ep, "return": float(all_returns[s, ep])}
        for s, seed in enumerate(seeds)
        for ep in range(all_returns.shape[1])
    ]
    csv_path = os.path.join(output_dir, f"{label}_episode_returns.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")


def _save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================
# Optuna tuning
# ============================================

def tune(base_config, param_space, n_trials, seeds, alg_name):
    """TPE search with per-trial verbose prints."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        trial_config = base_config.copy()
        for name, (kind, low, high) in param_space.items():
            if kind == "log_float":
                trial_config[name] = trial.suggest_float(name, low, high, log=True)
            elif kind == "float":
                trial_config[name] = trial.suggest_float(name, low, high)
            elif kind == "int":
                trial_config[name] = trial.suggest_int(name, low, high)
            elif kind == "log_int":
                trial_config[name] = trial.suggest_int(name, low, high, log=True)
        return run_multi_seed(trial_config, seeds=seeds, show_progress=False)["final_mean"]

    def on_trial_end(study, trial):
        rounded = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in trial.params.items()
        }
        print(
            f"  [{alg_name}] Trial {trial.number:2d} | "
            f"value: {trial.value:9.2f} | "
            f"best: {study.best_value:9.2f} | "
            f"params: {rounded}"
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0)
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[on_trial_end],
        show_progress_bar=False,
    )
    return study


# ============================================
# Per-algorithm figures
# ============================================

def plot_vs(b_all, t_all, alg_name, output_dir):
    """
    4 figures: baseline vs tuned, episode and timestep axes, raw and smoothed.
    """
    safe_env = ENV_NAME.replace("-", "_")
    safe_alg = alg_name.replace(" ", "_")
    n_ep     = b_all.shape[1]
    ep_x     = np.arange(1, n_ep + 1)
    ts_x     = ep_x * STEPS_PER_EPISODE

    for smooth in [False, True]:
        b = smooth_per_seed(b_all, SMOOTH_WINDOW) if smooth else b_all
        t = smooth_per_seed(t_all, SMOOTH_WINDOW) if smooth else t_all
        b_mean, b_ci = mean_and_ci(b)
        t_mean, t_ci = mean_and_ci(t)

        suffix       = "_smoothed" if smooth else ""
        title_suffix = f" — Smoothed (window={SMOOTH_WINDOW})" if smooth else ""

        for full_x, xlabel, xkind in [
            (ep_x, "Episode",  "episode"),
            (ts_x, "Timestep", "timestep"),
        ]:
            x = full_x[:len(b_mean)]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, b_mean, label="Baseline")
            ax.fill_between(x, b_mean - b_ci, b_mean + b_ci, alpha=0.2)
            ax.plot(x, t_mean, label="Tuned (best params)")
            ax.fill_between(x, t_mean - t_ci, t_mean + t_ci, alpha=0.2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Return")
            ax.set_title(f"{alg_name} on {ENV_NAME}{title_suffix}")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fname = f"{safe_alg}_baseline_vs_tuned_{safe_env}_{xkind}{suffix}.png"
            _save_fig(fig, os.path.join(output_dir, fname))


def plot_param_scan(study, param_space, alg_name, output_dir):
    """One figure with one subplot per tuned parameter."""
    safe_env    = ENV_NAME.replace("-", "_")
    safe_alg    = alg_name.replace(" ", "_")
    trials_df   = study.trials_dataframe()
    param_names = [p for p in param_space if f"params_{p}" in trials_df.columns]

    if not param_names:
        return

    ncols = min(len(param_names), 3)
    nrows = (len(param_names) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat  = axes.reshape(-1)

    for ax, pname in zip(axes_flat, param_names):
        kind = param_space[pname][0]
        x    = trials_df[f"params_{pname}"]
        y    = trials_df["value"]
        ax.scatter(x, y, s=30, alpha=0.5)
        ax.scatter(
            [study.best_params[pname]], [study.best_value],
            color="red", s=90, zorder=5, label="Best"
        )
        if "log" in kind:
            ax.set_xscale("log")
        ax.set_xlabel(pname)
        ax.set_ylabel("Final Mean Return")
        ax.set_title(pname)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes_flat[len(param_names):]:
        ax.set_visible(False)

    fig.suptitle(f"{alg_name} Parameter Scan on {ENV_NAME}", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, os.path.join(output_dir, f"{safe_alg}_param_scan_{safe_env}.png"))


# ============================================
# Combined figures (all 3 tuned algorithms)
# ============================================

def plot_combined(tuned_dict, output_dir):
    """4 figures: per-episode and per-timestep, raw and smoothed."""
    safe_env = ENV_NAME.replace("-", "_")
    n_ep     = next(iter(tuned_dict.values())).shape[1]
    ep_x     = np.arange(1, n_ep + 1)
    ts_x     = ep_x * STEPS_PER_EPISODE

    for smooth in [False, True]:
        suffix       = "_smoothed" if smooth else ""
        title_suffix = f" — Smoothed (window={SMOOTH_WINDOW})" if smooth else ""

        for full_x, xlabel, xkind in [
            (ep_x, "Episode",  "episode"),
            (ts_x, "Timestep", "timestep"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 6))

            for alg_name, all_returns in tuned_dict.items():
                curves = smooth_per_seed(all_returns, SMOOTH_WINDOW) if smooth else all_returns
                mean, ci = mean_and_ci(curves)
                x = full_x[:len(mean)]
                ax.plot(x, mean, label=alg_name)
                ax.fill_between(x, mean - ci, mean + ci, alpha=0.15)

            ax.set_xlabel(xlabel)
            ax.set_ylabel("Return")
            ax.set_title(f"PPO vs SAC vs TD3 on {ENV_NAME}{title_suffix}")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            _save_fig(fig, os.path.join(output_dir, f"comparison_{safe_env}_{xkind}{suffix}.png"))


# ============================================
# Main loop
# ============================================

tuned_results = {}

for alg_name in ["PPO", "SAC", "TD3"]:

    print(f"\n{'=' * 60}")
    print(f"  {alg_name} on {ENV_NAME}")
    print(f"{'=' * 60}")

    base_config = BASE_CONFIGS[alg_name]
    param_space = PARAM_SPACES[alg_name]
    alg_dir     = f"{RESULTS_ROOT}/{alg_name}"
    figure_dir  = f"{alg_dir}/figures"

    # ---- 1. Baseline ----
    print(f"\n[{alg_name}] Baseline ({len(SEEDS)} seeds, {NUM_EPISODES} episodes)")
    baseline_result = run_multi_seed(base_config, seeds=SEEDS, show_progress=True)
    baseline_all    = baseline_result["all_returns"]
    print(f"[{alg_name}] Baseline final mean return: {baseline_result['final_mean']:.2f}")
    save_returns(baseline_all, SEEDS, "baseline", alg_dir)

    # ---- 2. Optuna tuning ----
    print(f"\n[{alg_name}] Optuna tuning ({N_TRIALS} trials x {len(SEEDS)} seeds)")
    study = tune(base_config, param_space, N_TRIALS, SEEDS, alg_name)
    print(f"\n[{alg_name}] Best params: {study.best_params}")
    print(f"[{alg_name}] Best value:  {study.best_value:.2f}")

    os.makedirs(alg_dir, exist_ok=True)

    trials_path = os.path.join(alg_dir, "optuna_trials.csv")
    study.trials_dataframe().to_csv(trials_path, index=False)
    print(f"  Saved {trials_path}")

    best_json_path = os.path.join(alg_dir, f"{alg_name.lower()}_best_params_{ENV_NAME}.json")
    with open(best_json_path, "w") as f:
        json.dump(
            {
                "algorithm":   alg_name,
                "env_name":    ENV_NAME,
                "best_value":  study.best_value,
                "best_params": study.best_params,
            },
            f,
            indent=2,
        )
    print(f"  Saved {best_json_path}")

    # ---- 3. Train with best params ----
    tuned_config = {**base_config, **study.best_params}
    print(f"\n[{alg_name}] Tuned run ({len(SEEDS)} seeds, {NUM_EPISODES} episodes)")
    tuned_result = run_multi_seed(tuned_config, seeds=SEEDS, show_progress=True)
    tuned_all    = tuned_result["all_returns"]
    print(f"[{alg_name}] Tuned final mean return: {tuned_result['final_mean']:.2f}")
    save_returns(tuned_all, SEEDS, "optuna_tuned", alg_dir)
    tuned_results[alg_name] = tuned_all

    # ---- 4. Per-algorithm figures ----
    print(f"\n[{alg_name}] Saving figures...")
    plot_vs(baseline_all, tuned_all, alg_name, figure_dir)
    plot_param_scan(study, param_space, alg_name, figure_dir)


# ---- 5. Combined figures ----
print(f"\n{'=' * 60}")
print("  Combined comparison")
print(f"{'=' * 60}")
plot_combined(tuned_results, COMBINED_DIR)

print("\nDone.")
