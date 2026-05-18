# ============================================
# Main script for running selected RL algorithms
# ============================================

from compare import (
    run_multi_seed,
    tune_one_param,
    plot_baseline_result,
    plot_tuning_results,
    print_tuning_summary,
    save_experiment_results,
    save_episode_results_to_csv,
)

import gymnasium as gym
import experiment_config as cfg


# ============================================
# Common configuration
# ============================================

common_config = {
    "num_episodes": cfg.NUM_EPISODES,
    "hidden_dim": cfg.HIDDEN_DIM,
    "gamma": cfg.GAMMA,
    "env_name": cfg.ENV_NAME
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


ACTION_SPACE_TYPE = get_action_space_type(cfg.ENV_NAME)


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

for algorithm_name in cfg.ALGORITHMS_TO_RUN:

    print("\n" + "=" * 60)
    print(f"Running algorithm: {algorithm_name}")
    print(f"Environment: {cfg.ENV_NAME}")
    print(f"Action space type: {ACTION_SPACE_TYPE}")
    print("=" * 60)

    if algorithm_name not in ALGORITHM_CONFIGS:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. "
            f"Available options are: {list(ALGORITHM_CONFIGS.keys())}"
        )

    if should_skip_algorithm(algorithm_name, ACTION_SPACE_TYPE):
        print(
            f"Skipping {algorithm_name} on {cfg.ENV_NAME}: "
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
    # 2. Hyperparameter experiments
    # ============================================

    tuning_results_dict = {}

    if cfg.RUN_TUNING:

        # ============================================
        # 2.1 DQN tuning
        # ============================================

        if algorithm_name == "DQN":

            # Learning rate
            lr_results = tune_one_param(
                config,
                "lr",
                cfg.DQN_LR_VALUES,
                seeds=cfg.SEEDS,
                show_progress=True
            )

            plot_tuning_results(
                lr_results,
                "lr",
                output_dir=algorithm_figure_dir,
                save=True,
                show=False
            )

            print_tuning_summary(lr_results, "lr")

            tuning_results_dict["learning_rate_tuning"] = {
                "param_name": "lr",
                "results": lr_results
            }

            # Epsilon decay steps
            epsilon_decay_results = tune_one_param(
                config,
                "epsilon_decay_steps",
                cfg.DQN_EPSILON_DECAY_STEPS_VALUES,
                seeds=cfg.SEEDS,
                show_progress=True
            )

            plot_tuning_results(
                epsilon_decay_results,
                "epsilon_decay_steps",
                output_dir=algorithm_figure_dir,
                save=True,
                show=False
            )

            print_tuning_summary(
                epsilon_decay_results,
                "epsilon_decay_steps"
            )

            tuning_results_dict["epsilon_decay_steps_tuning"] = {
                "param_name": "epsilon_decay_steps",
                "results": epsilon_decay_results
            }

            # Target update frequency
            target_update_results = tune_one_param(
                config,
                "target_update",
                cfg.DQN_TARGET_UPDATE_VALUES,
                seeds=cfg.SEEDS,
                show_progress=True
            )

            plot_tuning_results(
                target_update_results,
                "target_update",
                output_dir=algorithm_figure_dir,
                save=True,
                show=False
            )

            print_tuning_summary(target_update_results, "target_update")

            tuning_results_dict["target_update_tuning"] = {
                "param_name": "target_update",
                "results": target_update_results
            }

            # Batch size
            batch_size_results = tune_one_param(
                config,
                "batch_size",
                cfg.DQN_BATCH_SIZE_VALUES,
                seeds=cfg.SEEDS,
                show_progress=True
            )

            plot_tuning_results(
                batch_size_results,
                "batch_size",
                output_dir=algorithm_figure_dir,
                save=True,
                show=False
            )

            print_tuning_summary(batch_size_results, "batch_size")

            tuning_results_dict["batch_size_tuning"] = {
                "param_name": "batch_size",
                "results": batch_size_results
            }

        # ============================================
        # 2.2 PPO tuning
        # ============================================

        elif algorithm_name == "PPO":

            # Actor learning rate
            actor_lr_results = tune_one_param(
                config,
                "actor_lr",
                ppo_actor_lr_values,
                seeds=cfg.SEEDS,
                show_progress=True
            )

            plot_tuning_results(
                actor_lr_results,
                "actor_lr",
                output_dir=algorithm_figure_dir,
                save=True,
                show=False
            )

            print_tuning_summary(actor_lr_results, "actor_lr")

            tuning_results_dict["actor_lr_tuning"] = {
                "param_name": "actor_lr",
                "results": actor_lr_results
            }

            # PPO clipping epsilon
            eps_results = tune_one_param(
                config,
                "eps",
                ppo_eps_values,
                seeds=cfg.SEEDS,
                show_progress=True
            )

            plot_tuning_results(
                eps_results,
                "eps",
                output_dir=algorithm_figure_dir,
                save=True,
                show=False
            )

            print_tuning_summary(eps_results, "eps")

            tuning_results_dict["eps_tuning"] = {
                "param_name": "eps",
                "results": eps_results
            }

            # PPO update epochs
            epochs_results = tune_one_param(
                config,
                "epochs",
                ppo_epochs_values,
                seeds=cfg.SEEDS,
                show_progress=True
            )

            plot_tuning_results(
                epochs_results,
                "epochs",
                output_dir=algorithm_figure_dir,
                save=True,
                show=False
            )

            print_tuning_summary(epochs_results, "epochs")

            tuning_results_dict["epochs_tuning"] = {
                "param_name": "epochs",
                "results": epochs_results
            }

        # ============================================
        # 2.3 SAC tuning
        # ============================================

        elif algorithm_name == "SAC":

            # Actor learning rate
            actor_lr_results = tune_one_param(
                config,
                "actor_lr",
                cfg.SAC_ACTOR_LR_VALUES,
                seeds=cfg.SEEDS,
                show_progress=True
            )

            plot_tuning_results(
                actor_lr_results,
                "actor_lr",
                output_dir=algorithm_figure_dir,
                save=True,
                show=False
            )

            print_tuning_summary(actor_lr_results, "actor_lr")

            tuning_results_dict["actor_lr_tuning"] = {
                "param_name": "actor_lr",
                "results": actor_lr_results
            }

            # Alpha learning rate
            alpha_lr_results = tune_one_param(
                config,
                "alpha_lr",
                cfg.SAC_ALPHA_LR_VALUES,
                seeds=cfg.SEEDS,
                show_progress=True
            )

            plot_tuning_results(
                alpha_lr_results,
                "alpha_lr",
                output_dir=algorithm_figure_dir,
                save=True,
                show=False
            )

            print_tuning_summary(alpha_lr_results, "alpha_lr")

            tuning_results_dict["alpha_lr_tuning"] = {
                "param_name": "alpha_lr",
                "results": alpha_lr_results
            }

            # Soft update coefficient
            tau_results = tune_one_param(
                config,
                "tau",
                cfg.SAC_TAU_VALUES,
                seeds=cfg.SEEDS,
                show_progress=True
            )

            plot_tuning_results(
                tau_results,
                "tau",
                output_dir=algorithm_figure_dir,
                save=True,
                show=False
            )

            print_tuning_summary(tau_results, "tau")

            tuning_results_dict["tau_tuning"] = {
                "param_name": "tau",
                "results": tau_results
            }

        # ============================================
        # 2.4 TD3 tuning
        # ============================================

        elif algorithm_name == "TD3":

            # Actor learning rate
            actor_lr_results = tune_one_param(
                config,
                "actor_lr",
                cfg.TD3_ACTOR_LR_VALUES,
                seeds=cfg.SEEDS,
                show_progress=True
            )

            plot_tuning_results(
                actor_lr_results,
                "actor_lr",
                output_dir=algorithm_figure_dir,
                save=True,
                show=False
            )

            print_tuning_summary(actor_lr_results, "actor_lr")

            tuning_results_dict["actor_lr_tuning"] = {
                "param_name": "actor_lr",
                "results": actor_lr_results
            }

            # Soft update coefficient
            tau_results = tune_one_param(
                config,
                "tau",
                cfg.TD3_TAU_VALUES,
                seeds=cfg.SEEDS,
                show_progress=True
            )

            plot_tuning_results(
                tau_results,
                "tau",
                output_dir=algorithm_figure_dir,
                save=True,
                show=False
            )

            print_tuning_summary(tau_results, "tau")

            tuning_results_dict["tau_tuning"] = {
                "param_name": "tau",
                "results": tau_results
            }

            # Policy update delay
            policy_delay_results = tune_one_param(
                config,
                "policy_delay",
                cfg.TD3_POLICY_DELAY_VALUES,
                seeds=cfg.SEEDS,
                show_progress=True
            )

            plot_tuning_results(
                policy_delay_results,
                "policy_delay",
                output_dir=algorithm_figure_dir,
                save=True,
                show=False
            )

            print_tuning_summary(policy_delay_results, "policy_delay")

            tuning_results_dict["policy_delay_tuning"] = {
                "param_name": "policy_delay",
                "results": policy_delay_results
            }

    else:
        print("\nSkipping hyperparameter tuning because RUN_TUNING = False.")

    # ============================================
    # 3. Save all experiment data
    # ============================================

    save_experiment_results(
        baseline_result=baseline_result,
        tuning_results_dict=tuning_results_dict,
        output_dir=algorithm_output_dir
    )

    save_episode_results_to_csv(
        baseline_result=baseline_result,
        tuning_results_dict=tuning_results_dict,
        seeds=cfg.SEEDS,
        output_dir=algorithm_output_dir
    )

    print(f"\nFinished running {algorithm_name}.")
    print(f"Results saved to: {algorithm_output_dir}")


print("\nAll selected experiments finished.")