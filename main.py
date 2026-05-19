# ============================================
# Main script for running selected RL algorithms
# ============================================

import argparse

from compare import (
    run_multi_seed,
    plot_baseline_result,
    save_experiment_results,
    save_episode_results_to_csv,
    tune_with_optuna,
    plot_optuna_results,
    save_optuna_results,
)

import gymnasium as gym
import experiment_config as cfg


# ============================================
# CLI arguments (override experiment_config.py)
# ============================================

_parser = argparse.ArgumentParser(description="Run RL experiments.")
_parser.add_argument(
    "--alg",
    choices=["DQN", "PPO", "SAC", "TD3"],
    default=None,
    help="Algorithm to run. Overrides ALGORITHMS_TO_RUN in experiment_config.py."
)
_parser.add_argument(
    "--env",
    choices=[
        "CartPole-v1",
        "MountainCar-v0",
        "Acrobot-v1",
        "Pendulum-v1",
        "MountainCarContinuous-v0",
    ],
    default=None,
    help="Environment to run. Overrides ENV_NAME in experiment_config.py."
)
_args = _parser.parse_args()

algorithms_to_run = [_args.alg] if _args.alg else cfg.ALGORITHMS_TO_RUN
env_name = _args.env if _args.env else cfg.ENV_NAME


# ============================================
# Common configuration
# ============================================

common_config = {
    "num_episodes": cfg.NUM_EPISODES,
    "hidden_dim": cfg.HIDDEN_DIM,
    "gamma": cfg.GAMMA,
    "env_name": env_name
}


# ============================================
# Environment-specific PPO configuration
# ============================================

def get_action_space_type(env_name):
    """
    Automatically detect whether the environment uses discrete or continuous actions.
    """
    env = gym.make(env_name)
    action_space = env.action_space
    env.close()

    if isinstance(action_space, gym.spaces.Discrete):
        return "discrete"

    elif isinstance(action_space, gym.spaces.Box):
        return "continuous"

    else:
        raise ValueError(
            f"Unsupported action space for {env_name}: {action_space}"
        )


ACTION_SPACE_TYPE = get_action_space_type(env_name)


# ============================================
# Environment-specific PPO configuration
# ============================================

if ACTION_SPACE_TYPE == "discrete":

    ppo_config = {
        **common_config,
        "algorithm": "PPO",
        "actor_lr": cfg.PPO_DISCRETE_ACTOR_LR,
        "critic_lr": cfg.PPO_DISCRETE_CRITIC_LR,
        "lmbda": cfg.PPO_DISCRETE_LMBDA,
        "epochs": cfg.PPO_DISCRETE_EPOCHS,
        "eps": cfg.PPO_DISCRETE_EPS
    }

    ppo_actor_lr_values = cfg.PPO_DISCRETE_ACTOR_LR_VALUES
    ppo_eps_values = cfg.PPO_DISCRETE_EPS_VALUES
    ppo_epochs_values = cfg.PPO_DISCRETE_EPOCHS_VALUES

elif ACTION_SPACE_TYPE == "continuous":

    ppo_config = {
        **common_config,
        "algorithm": "PPO",
        "actor_lr": cfg.PPO_CONTINUOUS_ACTOR_LR,
        "critic_lr": cfg.PPO_CONTINUOUS_CRITIC_LR,
        "lmbda": cfg.PPO_CONTINUOUS_LMBDA,
        "epochs": cfg.PPO_CONTINUOUS_EPOCHS,
        "eps": cfg.PPO_CONTINUOUS_EPS
    }

    ppo_actor_lr_values = cfg.PPO_CONTINUOUS_ACTOR_LR_VALUES
    ppo_eps_values = cfg.PPO_CONTINUOUS_EPS_VALUES
    ppo_epochs_values = cfg.PPO_CONTINUOUS_EPOCHS_VALUES


# ============================================
# Algorithm-specific configurations
# ============================================

dqn_config = {
    **common_config,
    "algorithm": "DQN",
    "lr": cfg.DQN_LR,
    "epsilon_start": cfg.DQN_EPSILON_START,
    "epsilon_end": cfg.DQN_EPSILON_END,
    "epsilon_decay_steps": cfg.DQN_EPSILON_DECAY_STEPS,
    "target_update": cfg.DQN_TARGET_UPDATE,
    "buffer_size": cfg.DQN_BUFFER_SIZE,
    "minimal_size": cfg.DQN_MINIMAL_SIZE,
    "batch_size": cfg.DQN_BATCH_SIZE
}


sac_config = {
    **common_config,
    "algorithm": "SAC",
    "actor_lr": cfg.SAC_ACTOR_LR,
    "critic_lr": cfg.SAC_CRITIC_LR,
    "alpha_lr": cfg.SAC_ALPHA_LR,
    "tau": cfg.SAC_TAU,
    "buffer_size": cfg.SAC_BUFFER_SIZE,
    "minimal_size": cfg.SAC_MINIMAL_SIZE,
    "batch_size": cfg.SAC_BATCH_SIZE
}


td3_config = {
    **common_config,
    "algorithm": "TD3",
    "actor_lr": cfg.TD3_ACTOR_LR,
    "critic_lr": cfg.TD3_CRITIC_LR,
    "tau": cfg.TD3_TAU,
    "exploration_noise": cfg.TD3_EXPLORATION_NOISE,
    "policy_noise": cfg.TD3_POLICY_NOISE,
    "noise_clip": cfg.TD3_NOISE_CLIP,
    "policy_delay": cfg.TD3_POLICY_DELAY,
    "buffer_size": cfg.TD3_BUFFER_SIZE,
    "minimal_size": cfg.TD3_MINIMAL_SIZE,
    "batch_size": cfg.TD3_BATCH_SIZE,
}


ALGORITHM_CONFIGS = {
    "DQN": dqn_config,
    "PPO": ppo_config,
    "SAC": sac_config,
    "TD3": td3_config,
}


# ============================================
# Compatibility check
# ============================================

def should_skip_algorithm(algorithm_name, action_space_type):
    """
    Skip algorithms that are incompatible with the selected environment.
    """
    if action_space_type == "continuous" and algorithm_name == "DQN":
        return True

    if action_space_type == "discrete" and algorithm_name == "TD3":
        return True

    return False


# ============================================
# Main experiment loop
# ============================================

for algorithm_name in algorithms_to_run:

    print("\n" + "=" * 60)
    print(f"Running algorithm: {algorithm_name}")
    print(f"Environment: {env_name}")
    print(f"Action space type: {ACTION_SPACE_TYPE}")
    print("=" * 60)

    if algorithm_name not in ALGORITHM_CONFIGS:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. "
            f"Available options are: {list(ALGORITHM_CONFIGS.keys())}"
        )

    if should_skip_algorithm(algorithm_name, ACTION_SPACE_TYPE):
        print(
            f"Skipping {algorithm_name} on {env_name}: "
            f"{algorithm_name} is not compatible with {ACTION_SPACE_TYPE} action spaces."
        )
        continue

    # Each algorithm has its own config
    config = ALGORITHM_CONFIGS[algorithm_name].copy()

    # Each algorithm and environment has its own result folder
    algorithm_output_dir = f"results/{config['env_name']}/{algorithm_name}"
    algorithm_figure_dir = f"{algorithm_output_dir}/figures"

    # ============================================
    # 1. Baseline experiment
    # ============================================

    baseline_result = run_multi_seed(
        config,
        seeds=cfg.SEEDS,
        show_progress=True
    )

    print(
        f"\n{algorithm_name} baseline final mean return:",
        baseline_result["final_mean"]
    )

    plot_baseline_result(
        baseline_result,
        env_name=config["env_name"],
        output_dir=algorithm_figure_dir,
        save=True,
        show=False
    )

    # ============================================
    # 2. Optuna hyperparameter search
    # ============================================

    if cfg.RUN_TUNING:

        param_space = cfg.PARAM_SPACES.get(algorithm_name, {})

        print(f"\n===== Optuna tuning: {algorithm_name} on {env_name} =====")
        print(f"Trials: {cfg.N_TRIALS} | Params: {list(param_space.keys())}")

        study = tune_with_optuna(
            config,
            param_space,
            n_trials=cfg.N_TRIALS,
            seeds=cfg.SEEDS,
        )

        print(f"\nBest params:            {study.best_params}")
        print(f"Best final mean return: {study.best_value:.2f}")

        plot_optuna_results(
            study,
            param_space,
            algorithm_name=algorithm_name,
            env_name=env_name,
            output_dir=algorithm_figure_dir,
            save=True,
            show=False,
        )

        save_optuna_results(study, algorithm_output_dir, algorithm_name, env_name)

    else:
        print("\nSkipping hyperparameter tuning because RUN_TUNING = False.")

    # ============================================
    # 3. Save baseline experiment data
    # ============================================

    save_experiment_results(
        baseline_result=baseline_result,
        tuning_results_dict={},
        output_dir=algorithm_output_dir
    )

    save_episode_results_to_csv(
        baseline_result=baseline_result,
        tuning_results_dict={},
        seeds=cfg.SEEDS,
        output_dir=algorithm_output_dir
    )

    print(f"\nFinished running {algorithm_name}.")
    print(f"Results saved to: {algorithm_output_dir}")


print("\nAll selected experiments finished.")