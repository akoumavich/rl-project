# ==============================================================================
# PPO for continuous control environments
# ==============================================================================

import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gymnasium as gym
from tqdm import tqdm

from utils import set_seed


# ============================================
# 1. Actor network
# ============================================

class ActorNet(nn.Module):
    """
    Actor network for continuous action spaces.

    It outputs the mean and standard deviation of a Gaussian policy.
    The mean is bounded by tanh and then scaled to the action range.
    """

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.mu_out = nn.Linear(hidden_dim, action_dim)
        self.sigma_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        mu = self.action_bound * torch.tanh(self.mu_out(x))
        sigma = F.softplus(self.sigma_out(x)) + 1e-5

        return mu, sigma


# ============================================
# 2. Critic network
# ============================================

class CriticNet(nn.Module):
    """
    Critic network.

    It estimates the state-value function V(s).
    """

    def __init__(self, state_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# ============================================
# 3. PPO agent
# ============================================

class PPO:
    """
    PPO agent for continuous control.

    actor_model is the current policy.
    actor_old_model stores the old policy before PPO updates.
    """

    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        action_bound,
        actor_lr,
        critic_lr,
        gamma,
        eps,
        actor_update_steps,
        critic_update_steps,
        device
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.gamma = gamma
        self.eps = eps
        self.actor_update_steps = actor_update_steps
        self.critic_update_steps = critic_update_steps
        self.device = device

        self.actor_model = ActorNet(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            action_bound=action_bound
        ).to(device)

        self.actor_old_model = ActorNet(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            action_bound=action_bound
        ).to(device)

        self.critic_model = CriticNet(
            state_dim=state_dim,
            hidden_dim=hidden_dim
        ).to(device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor_model.parameters(),
            lr=actor_lr
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic_model.parameters(),
            lr=critic_lr
        )

    def take_action(self, state):
        """
        Sample action from the current Gaussian policy.
        """

        state = torch.tensor(
            state,
            dtype=torch.float32
        ).to(self.device)

        mu, sigma = self.actor_model(state)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()

        action = action.detach().cpu().numpy()

        action = np.clip(
            action,
            -self.action_bound,
            self.action_bound
        )

        return action

    def discount_reward(self, rewards, next_state):
        """
        Compute bootstrapped discounted returns.

        rewards are shaped rewards used only for training the critic.
        """

        next_state = torch.tensor(
            next_state,
            dtype=torch.float32
        ).to(self.device)

        target = self.critic_model(next_state).detach()

        target_list = []

        for reward in rewards[::-1]:
            target = reward + self.gamma * target
            target_list.append(target)

        target_list.reverse()

        target_list = torch.cat(target_list).detach()

        return target_list

    def calculate_advantage(self, states, targets):
        """
        Advantage = target return minus critic value.
        """

        states = torch.tensor(
            states,
            dtype=torch.float32
        ).to(self.device)

        values = self.critic_model(states).reshape(-1)

        advantage = targets - values

        return advantage.detach()

    def actor_learn(self, states, actions, advantage):
        """
        Update actor using PPO clipped surrogate objective.
        """

        states = torch.tensor(
            states,
            dtype=torch.float32
        ).to(self.device)

        actions = torch.tensor(
            actions,
            dtype=torch.float32
        ).reshape(-1, self.action_dim).to(self.device)

        advantage = advantage.reshape(-1, 1)

        mu, sigma = self.actor_model(states)
        pi = torch.distributions.Normal(mu, sigma)

        old_mu, old_sigma = self.actor_old_model(states)
        old_pi = torch.distributions.Normal(
            old_mu.detach(),
            old_sigma.detach()
        )

        log_prob = pi.log_prob(actions).sum(dim=1, keepdim=True)
        old_log_prob = old_pi.log_prob(actions).sum(dim=1, keepdim=True)

        ratio = torch.exp(log_prob - old_log_prob)

        surr1 = ratio * advantage

        surr2 = torch.clamp(
            ratio,
            1.0 - self.eps,
            1.0 + self.eps
        ) * advantage

        actor_loss = -torch.mean(torch.min(surr1, surr2))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def critic_learn(self, states, targets):
        """
        Update critic by fitting bootstrapped discounted returns.
        """

        states = torch.tensor(
            states,
            dtype=torch.float32
        ).to(self.device)

        targets = targets.to(self.device)

        values = self.critic_model(states).reshape(-1)

        critic_loss = F.mse_loss(values, targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def update(self, states, actions, targets):
        """
        Run PPO update.

        First copy current actor into old actor.
        Then update actor and critic multiple times.
        """

        self.actor_old_model.load_state_dict(
            self.actor_model.state_dict()
        )

        advantage = self.calculate_advantage(states, targets)

        last_actor_loss = 0.0
        last_critic_loss = 0.0

        for _ in range(self.actor_update_steps):
            last_actor_loss = self.actor_learn(
                states,
                actions,
                advantage
            )

        for _ in range(self.critic_update_steps):
            last_critic_loss = self.critic_learn(
                states,
                targets
            )

        return {
            "actor_loss": last_actor_loss,
            "critic_loss": last_critic_loss
        }


# ============================================
# 4. Helper functions
# ============================================

def get_default_episode_length(env_name):
    """
    Default rollout length for N-step batch update.
    """
    if env_name == "Pendulum-v1":
        return 200

    if env_name == "MountainCarContinuous-v0":
        return 999

    return 200


def shape_reward(env_name, reward):
    """
    Reward shaping used only for PPO training.

    Original rewards are still used for evaluation.
    """
    if env_name == "Pendulum-v1":
        return (reward + 8.0) / 8.0

    return reward


# ============================================
# 5. Training function for compare.py
# ============================================

def train_ppo(
    actor_lr,
    critic_lr,
    num_episodes,
    hidden_dim,
    gamma,
    lmbda,
    epochs,
    eps,
    seed=0,
    env_name="Pendulum-v1",
    show_progress=False,
    save_model=True,
    model_dir="results/models",
    model_name="ppo_continuous"
):
    """
    Train PPO on a continuous Gymnasium environment.

    The function keeps the same interface as compare.py.
    lmbda is not used in this simplified PPO version.
    epochs is used as both actor and critic update steps.
    """

    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    env = gym.make(env_name)

    env.reset(seed=seed)
    env.action_space.seed(seed)

    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError(
            f"This PPO version is for continuous action spaces only, "
            f"but got {env.action_space}."
        )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    action_low = env.action_space.low
    action_high = env.action_space.high

    if not np.allclose(action_low, -action_high):
        raise ValueError(
            "This simplified PPO version assumes symmetric action bounds."
        )

    action_bound = float(action_high[0])

    print(
        f"[PPO] env={env_name} | "
        f"action_type=continuous | "
        f"action_dim={action_dim} | "
        f"action_range=[{action_low}, {action_high}]"
    )

    agent = PPO(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        action_bound=action_bound,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        eps=eps,
        actor_update_steps=epochs,
        critic_update_steps=epochs,
        device=device
    )

    batch_size = 32
    len_episode = get_default_episode_length(env_name)

    return_list = []

    episode_iter = range(num_episodes)

    if show_progress:
        episode_iter = tqdm(
            episode_iter,
            desc=f"PPO {env_name} seed={seed}",
            leave=False
        )

    for episode in episode_iter:

        state, _ = env.reset()

        episode_return = 0.0

        states = []
        actions = []
        shaped_rewards = []

        last_update_info = {
            "actor_loss": 0.0,
            "critic_loss": 0.0
        }

        for t in range(len_episode):

            action = agent.take_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            episode_return += reward

            states.append(state)
            actions.append(action)

            shaped_reward = shape_reward(env_name, reward)
            shaped_rewards.append(shaped_reward)

            state = next_state

            if (t + 1) % batch_size == 0 or t == len_episode - 1 or done:

                states_array = np.array(states)
                actions_array = np.array(actions)
                rewards_array = np.array(shaped_rewards)

                targets = agent.discount_reward(
                    rewards_array,
                    next_state
                )

                last_update_info = agent.update(
                    states_array,
                    actions_array,
                    targets
                )

                states = []
                actions = []
                shaped_rewards = []

            if done:
                break

        return_list.append(episode_return)

        if show_progress:
            recent_returns = return_list[-10:]
            recent_mean_return = np.mean(recent_returns)

            episode_iter.set_postfix({
                "return": f"{episode_return:.1f}",
                "avg10": f"{recent_mean_return:.1f}",
                "actor_loss": f"{last_update_info['actor_loss']:.3f}",
                "critic_loss": f"{last_update_info['critic_loss']:.3f}"
            })

    # ============================================
    # Save trained model for visualization
    # ============================================

    if save_model:
        os.makedirs(model_dir, exist_ok=True)

        actor_path = os.path.join(
            model_dir,
            f"{model_name}_{env_name}_seed{seed}_actor.pth"
        )

        checkpoint_path = os.path.join(
            model_dir,
            f"{model_name}_{env_name}_seed{seed}_checkpoint.pth"
        )

        torch.save(
            agent.actor_model.state_dict(),
            actor_path
        )

        checkpoint = {
            "actor_state_dict": agent.actor_model.state_dict(),
            "actor_old_state_dict": agent.actor_old_model.state_dict(),
            "critic_state_dict": agent.critic_model.state_dict(),
            "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
            "return_list": return_list,
            "config": {
                "env_name": env_name,
                "actor_lr": actor_lr,
                "critic_lr": critic_lr,
                "num_episodes": num_episodes,
                "hidden_dim": hidden_dim,
                "gamma": gamma,
                "lmbda": lmbda,
                "epochs": epochs,
                "eps": eps,
                "seed": seed,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "action_bound": action_bound,
                "batch_size": batch_size,
                "len_episode": len_episode
            }
        }

        torch.save(
            checkpoint,
            checkpoint_path
        )

        print(f"[PPO] Actor saved to: {actor_path}")
        print(f"[PPO] Checkpoint saved to: {checkpoint_path}")

    env.close()

    return return_list