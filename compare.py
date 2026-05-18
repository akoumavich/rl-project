# ============================================
# Experiment comparison, visualization, and saving
# ============================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym

from algorithms.dqn import train_dqn
from algorithms.ppo_cartpole import train_ppo as train_ppo_discrete
from algorithms.ppo_pendulum import train_ppo as train_ppo_continuous
from algorithms.sac import train_sac
from algorithms.td3 import train_td3
from utils import moving_average


def train_ppo_auto(
    actor_lr,
    critic_lr,
    num_episodes,
    hidden_dim,
    gamma,
    lmbda,
    epochs,
    eps,
    seed=0,
    env_name="CartPole-v1",
    show_progress=False
):
    """
    Automatically choose PPO implementation based on the action space.

    Discrete action spaces use algorithms/ppo_cartpole.py.
    Continuous action spaces use algorithms/ppo_pendulum.py.
    """
    env = gym.make(env_name)
    action_space = env.action_space
    env.close()

    if isinstance(action_space, gym.spaces.Discrete):
        print(f"[PPO Auto] {env_name} detected as discrete action space.")

        return train_ppo_discrete(
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            num_episodes=num_episodes,
            hidden_dim=hidden_dim,
            gamma=gamma,
            lmbda=lmbda,
            epochs=epochs,
            eps=eps,
            seed=seed,
            env_name=env_name,
            show_progress=show_progress
        )

    elif isinstance(action_space, gym.spaces.Box):
        print(f"[PPO Auto] {env_name} detected as continuous action space.")

        return train_ppo_continuous(
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            num_episodes=num_episodes,
            hidden_dim=hidden_dim,
            gamma=gamma,
            lmbda=lmbda,
            epochs=epochs,
            eps=eps,
            seed=seed,
            env_name=env_name,
            show_progress=show_progress
        )

    else:
        raise ValueError(
            f"PPO only supports Discrete and Box action spaces, "
            f"but got {action_space}."
        )


def run_multi_seed(config, seeds=[0, 1, 2], show_progress=False):
    """
    Run the same experiment across multiple random seeds.
    """
    all_returns = []

    algorithm_name = config.get("algorithm", "DQN")

    for seed in seeds:
        if algorithm_name == "DQN":
            returns = train_dqn(
                lr=config["lr"],
                num_episodes=config["num_episodes"],
                hidden_dim=config["hidden_dim"],
                gamma=config["gamma"],
                epsilon_start=config["epsilon_start"],
                epsilon_end=config["epsilon_end"],
                epsilon_decay_steps=config["epsilon_decay_steps"],
                target_update=config["target_update"],
                buffer_size=config["buffer_size"],
                minimal_size=config["minimal_size"],
                batch_size=config["batch_size"],
                seed=seed,
                env_name=config["env_name"],
                show_progress=show_progress
            )

        elif algorithm_name == "PPO":
            returns = train_ppo_auto(
                actor_lr=config["actor_lr"],
                critic_lr=config["critic_lr"],
                num_episodes=config["num_episodes"],
                hidden_dim=config["hidden_dim"],
                gamma=config["gamma"],
                lmbda=config["lmbda"],
                epochs=config["epochs"],
                eps=config["eps"],
                seed=seed,
                env_name=config["env_name"],
                show_progress=show_progress
            )

        elif algorithm_name == "SAC":
            returns = train_sac(
                actor_lr=config["actor_lr"],
                critic_lr=config["critic_lr"],
                alpha_lr=config["alpha_lr"],
                num_episodes=config["num_episodes"],
                hidden_dim=config["hidden_dim"],
                gamma=config["gamma"],
                tau=config["tau"],
                buffer_size=config["buffer_size"],
                minimal_size=config["minimal_size"],
                batch_size=config["batch_size"],
                seed=seed,
                env_name=config["env_name"],
                show_progress=show_progress
            )

        elif algorithm_name == "TD3":
            returns = train_td3(
                actor_lr=config["actor_lr"],
                critic_lr=config["critic_lr"],
                num_episodes=config["num_episodes"],
                hidden_dim=config["hidden_dim"],
                gamma=config["gamma"],
                tau=config["tau"],
                exploration_noise=config["exploration_noise"],
                policy_noise=config["policy_noise"],
                noise_clip=config["noise_clip"],
                policy_delay=config["policy_delay"],
                buffer_size=config["buffer_size"],
                minimal_size=config["minimal_size"],
                batch_size=config["batch_size"],
                seed=seed,
                env_name=config["env_name"],
                show_progress=show_progress
            )

        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        all_returns.append(returns)

    all_returns = np.array(all_returns)

    mean_returns = all_returns.mean(axis=0)
    std_returns = all_returns.std(axis=0)
    final_mean = mean_returns[-20:].mean()

    return {
        "algorithm": algorithm_name,
        "all_returns": all_returns,
        "mean_returns": mean_returns,
        "std_returns": std_returns,
        "final_mean": final_mean
    }


def tune_one_param(
    base_config,
    param_name,
    param_values,
    seeds=[0, 1, 2],
    show_progress=False
):
    """
    Tune one hyperparameter while keeping other hyperparameters fixed.
    """
    results = {}

    print(f"\n===== Tuning {param_name} =====")

    for value in param_values:
        config = base_config.copy()
        config[param_name] = value

        print(f"\nRunning {param_name} = {value}")

        result = run_multi_seed(
            config,
            seeds=seeds,
            show_progress=show_progress
        )

        results[value] = result

        print(
            f"Final mean return "
            f"(last 20 episodes average): {result['final_mean']:.2f}"
        )

    return results


def plot_baseline_result(
    baseline_result,
    env_name="CartPole-v1",
    smooth_window=9,
    output_dir="results/figures",
    save=True,
    show=True
):
    """
    Plot and save the baseline learning curve.
    """
    os.makedirs(output_dir, exist_ok=True)

    algorithm_name = baseline_result.get("algorithm", "DQN")
    safe_algorithm_name = algorithm_name.replace(" ", "_")
    safe_env_name = env_name.replace("-", "_")

    plt.figure(figsize=(10, 6))

    baseline_smooth = moving_average(
        baseline_result["mean_returns"],
        smooth_window
    )

    plt.plot(baseline_smooth, label=f"{algorithm_name} baseline")

    plt.xlabel("Episode")
    plt.ylabel("Average Return over Seeds")
    plt.title(f"{algorithm_name} Baseline on {env_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        figure_path = os.path.join(
            output_dir,
            f"{safe_algorithm_name}_baseline_{safe_env_name}.png"
        )
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {figure_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_tuning_results(
    results,
    param_name,
    smooth_window=9,
    output_dir="results/figures",
    save=True,
    show=True
):
    """
    Plot and save smoothed learning curves for one hyperparameter tuning experiment.
    """
    os.makedirs(output_dir, exist_ok=True)

    algorithm_name = next(iter(results.values())).get("algorithm", "DQN")
    safe_algorithm_name = algorithm_name.replace(" ", "_")

    plt.figure(figsize=(10, 6))

    for value, result in results.items():
        mean_returns = result["mean_returns"]

        if len(mean_returns) >= smooth_window:
            smooth_returns = moving_average(mean_returns, smooth_window)
            x = np.arange(len(smooth_returns))
            plt.plot(x, smooth_returns, label=f"{param_name}={value}")
        else:
            plt.plot(mean_returns, label=f"{param_name}={value}")

    plt.xlabel("Episode")
    plt.ylabel("Average Return over Seeds")
    plt.title(f"{algorithm_name} Hyperparameter Tuning: {param_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        figure_path = os.path.join(
            output_dir,
            f"{safe_algorithm_name}_tuning_{param_name}.png"
        )
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {figure_path}")

    if show:
        plt.show()
    else:
        plt.close()


def print_tuning_summary(results, param_name):
    """
    Print a ranked summary of one hyperparameter tuning experiment.
    """
    print(f"\n===== Summary for {param_name} =====")

    summary = []

    for value, result in results.items():
        summary.append((value, result["final_mean"]))

    summary = sorted(summary, key=lambda x: x[1], reverse=True)

    for value, score in summary:
        print(f"{param_name} = {value:<8} | final mean return = {score:.2f}")

    best_value, best_score = summary[0]

    print(f"\nBest {param_name}: {best_value}, score = {best_score:.2f}")


def collect_summary_rows(results, experiment_name, param_name):
    """
    Convert tuning results into rows for a summary CSV table.
    """
    rows = []

    for value, result in results.items():
        rows.append({
            "experiment": experiment_name,
            "param_name": param_name,
            "param_value": value,
            "final_mean_return": result["final_mean"],
            "num_seeds": result["all_returns"].shape[0],
            "num_episodes": result["all_returns"].shape[1]
        })

    return rows


def save_experiment_results(
    baseline_result,
    tuning_results_dict,
    output_dir="results"
):
    """
    Save all experiment data.

    Outputs:
    1. results/summary.csv
       A readable summary table.

    2. results/all_returns.npz
       Full return curves for later plotting or analysis.
    """
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []

    summary_rows.append({
        "experiment": "baseline",
        "param_name": "baseline",
        "param_value": "baseline",
        "final_mean_return": baseline_result["final_mean"],
        "num_seeds": baseline_result["all_returns"].shape[0],
        "num_episodes": baseline_result["all_returns"].shape[1]
    })

    arrays_to_save = {
        "baseline_all_returns": baseline_result["all_returns"],
        "baseline_mean_returns": baseline_result["mean_returns"],
        "baseline_std_returns": baseline_result["std_returns"]
    }

    for experiment_name, item in tuning_results_dict.items():
        param_name = item["param_name"]
        results = item["results"]

        summary_rows.extend(
            collect_summary_rows(
                results,
                experiment_name,
                param_name
            )
        )

        for value, result in results.items():
            key_prefix = f"{experiment_name}_{param_name}_{value}"

            arrays_to_save[f"{key_prefix}_all_returns"] = result["all_returns"]
            arrays_to_save[f"{key_prefix}_mean_returns"] = result["mean_returns"]
            arrays_to_save[f"{key_prefix}_std_returns"] = result["std_returns"]

    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(output_dir, "summary.csv")
    curves_path = os.path.join(output_dir, "all_returns.npz")

    summary_df.to_csv(summary_path, index=False)
    np.savez(curves_path, **arrays_to_save)

    print(f"\nSaved summary table to: {summary_path}")
    print(f"Saved full return curves to: {curves_path}")


def save_episode_results_to_csv(
    baseline_result,
    tuning_results_dict,
    seeds=[0, 1, 2],
    output_dir="results"
):
    """
    Save per-episode returns for baseline and hyperparameter tuning experiments.

    The saved CSV can be directly used for plotting later.
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = []

    # =========================
    # 1. Save baseline results
    # =========================
    baseline_all_returns = baseline_result["all_returns"]
    algorithm_name = baseline_result.get("algorithm", "DQN")

    for seed_idx, seed in enumerate(seeds):
        for episode, episode_return in enumerate(baseline_all_returns[seed_idx]):
            rows.append({
                "experiment": "baseline",
                "algorithm": algorithm_name,
                "param_name": "baseline",
                "param_value": "baseline",
                "seed": seed,
                "episode": episode,
                "return": episode_return
            })

    # =========================
    # 2. Save tuning results
    # =========================
    for experiment_name, item in tuning_results_dict.items():
        param_name = item["param_name"]
        results = item["results"]

        for param_value, result in results.items():
            all_returns = result["all_returns"]
            algorithm_name = result.get("algorithm", "DQN")

            for seed_idx, seed in enumerate(seeds):
                for episode, episode_return in enumerate(all_returns[seed_idx]):
                    rows.append({
                        "experiment": experiment_name,
                        "algorithm": algorithm_name,
                        "param_name": param_name,
                        "param_value": param_value,
                        "seed": seed,
                        "episode": episode,
                        "return": episode_return
                    })

    episode_df = pd.DataFrame(rows)

    output_path = os.path.join(output_dir, "episode_returns.csv")
    episode_df.to_csv(output_path, index=False)

    print(f"Saved per-episode results to: {output_path}")


def plot_algorithm_comparison(
    results_dict,
    env_name="CartPole-v1",
    smooth_window=9,
    output_dir="results/figures",
    save=True,
    show=True
):
    """
    Plot learning curves for different algorithms.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for algorithm_name, result in results_dict.items():
        mean_returns = result["mean_returns"]

        if len(mean_returns) >= smooth_window:
            smooth_returns = moving_average(mean_returns, smooth_window)
            x = np.arange(len(smooth_returns))
            plt.plot(x, smooth_returns, label=algorithm_name)
        else:
            plt.plot(mean_returns, label=algorithm_name)

    plt.xlabel("Episode")
    plt.ylabel("Average Return over Seeds")
    plt.title(f"Algorithm Comparison on {env_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        safe_env_name = env_name.replace("-", "_")
        figure_path = os.path.join(
            output_dir,
            f"algorithm_comparison_{safe_env_name}.png"
        )
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {figure_path}")

    if show:
        plt.show()
    else:
        plt.close()