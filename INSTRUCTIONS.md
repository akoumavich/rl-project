# Implementation Instructions: TD3 + Full Environment Support

## Goal

Add TD3 (Twin Delayed Deep Deterministic Policy Gradient) and enable all four algorithms
(DQN, PPO, SAC, TD3) to run, tune, and produce figures on all five environments, following
the existing codebase structure exactly. No new abstractions. No new frameworks. Every
new piece of code should read as if it was written alongside the existing files.

---

## Algorithm–Environment Compatibility

| Algorithm | CartPole-v1 | MountainCar-v0 | Acrobot-v1 | MountainCarContinuous-v0 | Pendulum-v1 |
|-----------|:-----------:|:--------------:|:----------:|:------------------------:|:-----------:|
| DQN       | ✓           | ✓              | ✓          | ✗                        | ✗           |
| PPO       | ✓           | ✓              | ✓          | ✓                        | ✓           |
| SAC       | ✓           | ✓              | ✓          | ✓                        | ✓           |
| TD3       | ✗           | ✗              | ✗          | ✓                        | ✓           |

`should_skip_algorithm` in `main.py` already handles DQN on continuous envs. It must also
skip TD3 on discrete envs (`action_space_type == "discrete"`). SAC and PPO never skip.

---

## File 1 — CREATE `algorithms/td3.py`

Model this file on `algorithms/sac.py`. Same header comment style, same section dividers.

### 1.1 Networks

```python
class PolicyNet(nn.Module):
    """
    Deterministic actor for continuous action spaces.
    Output is tanh-bounded and scaled by action_bound.
    """
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound): ...
    def forward(self, x): ...  # returns action (no distribution)


class QValueNet(nn.Module):
    """
    Critic: takes (state, action) concatenated, outputs scalar Q value.
    Two hidden layers, same as SACContinuous QValueNet.
    """
    def __init__(self, state_dim, hidden_dim, action_dim): ...
    def forward(self, state, action): ...  # cat([state, action], dim=1)
```

### 1.2 Agent class `TD3`

```python
class TD3:
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        action_bound,
        actor_lr,
        critic_lr,
        tau,
        gamma,
        exploration_noise,   # std of Gaussian noise added to actions during data collection
        policy_noise,        # std of noise added to target actions for smoothing
        noise_clip,          # clip range for target policy noise: [-noise_clip, noise_clip]
        policy_delay,        # actor (and target) updated every policy_delay critic steps
        device
    ):
        # actor + target_actor  (PolicyNet)
        # critic_1 + target_critic_1  (QValueNet)
        # critic_2 + target_critic_2  (QValueNet)
        # Initialize targets with same weights as originals
        # actor_optimizer (Adam, actor_lr)
        # critic_1_optimizer (Adam, critic_lr)
        # critic_2_optimizer (Adam, critic_lr)
        # self.total_it = 0  (step counter for delayed updates)
```

**`take_action(self, state)`**
- Deterministic forward pass through actor
- Add Gaussian noise `N(0, exploration_noise)` clipped to `[-action_bound, action_bound]`
- Returns numpy array

**`soft_update(self, net, target_net)`**
- Same polyak averaging as `SACContinuous.soft_update`

**`update(self, transition_dict)`**
```
self.total_it += 1

# Unpack states, actions, rewards, next_states, dones from transition_dict

# --- Critic update (every step) ---
with torch.no_grad():
    # Target policy smoothing: add clipped noise to next actions
    noise = (torch.randn_like(next_actions) * policy_noise).clamp(-noise_clip, noise_clip)
    next_actions = (target_actor(next_states) + noise).clamp(-action_bound, action_bound)

    # Clipped double-Q target
    target_Q = min(target_critic_1(next_states, next_actions),
                   target_critic_2(next_states, next_actions))
    td_target = rewards + gamma * target_Q * (1 - dones)

# MSE loss for both critics, update both optimizers

# --- Delayed actor update (every policy_delay steps) ---
if self.total_it % self.policy_delay == 0:
    actor_loss = -mean(critic_1(states, actor(states)))
    update actor_optimizer

    soft_update actor → target_actor
    soft_update critic_1 → target_critic_1
    soft_update critic_2 → target_critic_2
```

### 1.3 Training function

```python
def train_td3(
    actor_lr,
    critic_lr,
    num_episodes,
    hidden_dim,
    gamma,
    tau,
    exploration_noise,
    policy_noise,
    noise_clip,
    policy_delay,
    buffer_size,
    minimal_size,
    batch_size,
    seed=0,
    env_name="Pendulum-v1",
    show_progress=False
):
```

Follow `train_sac` exactly:
- `set_seed(seed)`, make `env`, `ReplayBuffer`, instantiate `TD3`
- Only supports `gym.spaces.Box`. Raise `ValueError` if action space is not Box.
- `action_bound = float(env.action_space.high[0])` (assumes symmetric bounds)
- Episode loop: collect transitions, add to buffer, update when `buffer.size() > minimal_size`
- Return `return_list`
- No reward shaping (TD3 with standard hyperparameters handles Pendulum and
  MountainCarContinuous without it)

---

## File 2 — MODIFY `algorithms/ppo_pendulum.py`

### 2.1 Add `get_default_episode_length` (copy the pattern from `ppo_cartpole.py`)

```python
def get_default_episode_length(env_name):
    if env_name == "Pendulum-v1":
        return 200
    if env_name == "MountainCarContinuous-v0":
        return 999
    return 200
```

### 2.2 Add `shape_reward` for continuous environments

```python
def shape_reward(env_name, reward):
    if env_name == "Pendulum-v1":
        return (reward + 8.0) / 8.0
    return reward
```

### 2.3 In `train_ppo`, replace the two hardcoded lines:

Replace:
```python
shaped_reward = (reward + 8.0) / 8.0
...
len_episode = 200
```

With:
```python
shaped_reward = shape_reward(env_name, reward)
...
len_episode = get_default_episode_length(env_name)
```

---

## File 3 — MODIFY `experiment_config.py`

### 3.1 Update the top-level comment blocks (no functional change, just accuracy)

```python
# Available algorithms: "DQN", "PPO", "SAC", "TD3"
ALGORITHMS_TO_RUN = ["DQN"]

# Discrete:   CartPole-v1, MountainCar-v0, Acrobot-v1
# Continuous: Pendulum-v1, MountainCarContinuous-v0
ENV_NAME = "CartPole-v1"
```

### 3.2 Add TD3 hyperparameters (after the SAC section, same style)

```python
# TD3 settings (continuous action environments only)
TD3_ACTOR_LR = 3e-4
TD3_CRITIC_LR = 3e-4
TD3_TAU = 0.005
TD3_EXPLORATION_NOISE = 0.1
TD3_POLICY_NOISE = 0.2
TD3_NOISE_CLIP = 0.5
TD3_POLICY_DELAY = 2
TD3_BUFFER_SIZE = 10000
TD3_MINIMAL_SIZE = 1000
TD3_BATCH_SIZE = 256

# TD3 tuning values
TD3_ACTOR_LR_VALUES = [1e-4, 3e-4, 1e-3]
TD3_TAU_VALUES = [0.001, 0.005, 0.01]
TD3_POLICY_DELAY_VALUES = [1, 2, 4]
```

---

## File 4 — MODIFY `compare.py`

### 4.1 Add import at the top

```python
from algorithms.td3 import train_td3
```

### 4.2 Add TD3 branch in `run_multi_seed`

After the `elif algorithm_name == "SAC":` block, add:

```python
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
```

---

## File 5 — MODIFY `main.py`

### 5.1 Add `td3_config` (after `sac_config`, same style)

```python
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
```

### 5.2 Add to `ALGORITHM_CONFIGS`

```python
ALGORITHM_CONFIGS = {
    "DQN": dqn_config,
    "PPO": ppo_config,
    "SAC": sac_config,
    "TD3": td3_config,
}
```

### 5.3 Update `should_skip_algorithm`

Add one new condition:

```python
if action_space_type == "discrete" and algorithm_name == "TD3":
    return True
```

### 5.4 Add TD3 tuning section

Inside the `if cfg.RUN_TUNING:` block, after the `elif algorithm_name == "SAC":` block:

```python
elif algorithm_name == "TD3":

    # Actor learning rate
    actor_lr_results = tune_one_param(
        config,
        "actor_lr",
        cfg.TD3_ACTOR_LR_VALUES,
        seeds=cfg.SEEDS,
        show_progress=True
    )
    plot_tuning_results(actor_lr_results, "actor_lr", output_dir=algorithm_figure_dir, save=True, show=False)
    print_tuning_summary(actor_lr_results, "actor_lr")
    tuning_results_dict["actor_lr_tuning"] = {"param_name": "actor_lr", "results": actor_lr_results}

    # Soft update coefficient
    tau_results = tune_one_param(
        config,
        "tau",
        cfg.TD3_TAU_VALUES,
        seeds=cfg.SEEDS,
        show_progress=True
    )
    plot_tuning_results(tau_results, "tau", output_dir=algorithm_figure_dir, save=True, show=False)
    print_tuning_summary(tau_results, "tau")
    tuning_results_dict["tau_tuning"] = {"param_name": "tau", "results": tau_results}

    # Policy update delay
    policy_delay_results = tune_one_param(
        config,
        "policy_delay",
        cfg.TD3_POLICY_DELAY_VALUES,
        seeds=cfg.SEEDS,
        show_progress=True
    )
    plot_tuning_results(policy_delay_results, "policy_delay", output_dir=algorithm_figure_dir, save=True, show=False)
    print_tuning_summary(policy_delay_results, "policy_delay")
    tuning_results_dict["policy_delay_tuning"] = {"param_name": "policy_delay", "results": policy_delay_results}
```

---

## Verification Steps

After implementing, run each of the following in sequence (edit `experiment_config.py`
before each run). Each should complete without error and write figures to the
expected `results/<ENV>/<ALGO>/figures/` directory.

```
# 1. DQN on CartPole
ALGORITHMS_TO_RUN = ["DQN"], ENV_NAME = "CartPole-v1", RUN_TUNING = False

# 2. DQN on MountainCar
ALGORITHMS_TO_RUN = ["DQN"], ENV_NAME = "MountainCar-v0", RUN_TUNING = False

# 3. DQN on Acrobot
ALGORITHMS_TO_RUN = ["DQN"], ENV_NAME = "Acrobot-v1", RUN_TUNING = False

# 4. PPO on Pendulum (tests ppo_pendulum.py get_default_episode_length + shape_reward)
ALGORITHMS_TO_RUN = ["PPO"], ENV_NAME = "Pendulum-v1", RUN_TUNING = False

# 5. PPO on MountainCarContinuous (tests the new len_episode=999 branch)
ALGORITHMS_TO_RUN = ["PPO"], ENV_NAME = "MountainCarContinuous-v0", RUN_TUNING = False

# 6. SAC on Pendulum
ALGORITHMS_TO_RUN = ["SAC"], ENV_NAME = "Pendulum-v1", RUN_TUNING = False

# 7. TD3 on Pendulum (smoke test the new algorithm)
ALGORITHMS_TO_RUN = ["TD3"], ENV_NAME = "Pendulum-v1", RUN_TUNING = False

# 8. TD3 on MountainCarContinuous
ALGORITHMS_TO_RUN = ["TD3"], ENV_NAME = "MountainCarContinuous-v0", RUN_TUNING = False

# 9. Verify skip logic: TD3 should be silently skipped on a discrete env
ALGORITHMS_TO_RUN = ["TD3"], ENV_NAME = "CartPole-v1", RUN_TUNING = False

# 10. Full tuning run for TD3
ALGORITHMS_TO_RUN = ["TD3"], ENV_NAME = "Pendulum-v1", RUN_TUNING = True
```

---

## Constraints (from CLAUDE.md)

- Touch only what is listed above. Do not clean up, reformat, or refactor adjacent code.
- `td3.py` should use the same import order and section-divider style as `sac.py`.
- Do not add GAE, entropy bonuses, or any feature not present in the existing SAC/DQN implementations.
- `train_td3` returns only `return_list`, same as every other training function.
- No new files beyond `algorithms/td3.py` and this instructions file.
