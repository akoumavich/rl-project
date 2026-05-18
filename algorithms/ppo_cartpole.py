# ==============================================================================
# PPO for discrete control environments
# Designed to match the continuous PPO implementation structure as closely as possible
# ==============================================================================

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
    Actor network for discrete action spaces.

    It outputs a probability distribution over discrete actions.
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.action_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        probs = F.softmax(self.action_out(x), dim=-1)
        return probs


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
    PPO agent for discrete control.

    This version follows the same structure as the continuous PPO version:
    actor_model is the current policy.
    actor_old_model stores the old policy before PPO updates.
    """
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
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

        self.gamma = gamma
        self.eps = eps
        self.actor_update_steps = actor_update_steps
        self.critic_update_steps = critic_update_steps
        self.device = device

        self.actor_model = ActorNet(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim
        ).to(device)

        self.actor_old_model = ActorNet(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim
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
        Sample action from the current categorical policy.
        """
        state = torch.tensor(
            state,
            dtype=torch.float32
        ).to(self.device)

        probs = self.actor_model(state)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()

        return action.item()

    def discount_reward(self, rewards, next_state, done=False):
        """
        Compute bootstrapped discounted returns.

        rewards can be original rewards or shaped rewards.
        If done is True, the final bootstrap value is set to zero.
        Otherwise, the critic value of next_state is used for bootstrapping.
        """
        next_state = torch.tensor(
            next_state,
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            if done:
                target = torch.tensor(
                    [0.0],
                    dtype=torch.float32
                ).to(self.device)
            else:
                target = self.critic_model(next_state).reshape(-1)

        target_list = []

        for reward in rewards[::-1]:
            reward = torch.tensor(
                [reward],
                dtype=torch.float32
            ).to(self.device)

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
            dtype=torch.long
        ).reshape(-1).to(self.device)

        advantage = advantage.reshape(-1, 1)

        probs = self.actor_model(states)
        pi = torch.distributions.Categorical(probs)

        old_probs = self.actor_old_model(states)
        old_pi = torch.distributions.Categorical(old_probs.detach())

        log_prob = pi.log_prob(actions).reshape(-1, 1)
        old_log_prob = old_pi.log_prob(actions).reshape(-1, 1)

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
# 4. Reward shaping
# ============================================

def shape_reward(env_name, reward):
    """
    Reward shaping used only for PPO training.

    Original rewards are still used for evaluation.
    The shaping is intentionally simple and can be adjusted for each environment.
    """
    if env_name == "MountainCar-v0":
        return reward + 1.0

    if env_name == "Acrobot-v1":
        return reward + 1.0

    if env_name == "CartPole-v1":
        return reward

    return reward


def get_default_episode_length(env_name):
    """
    Default rollout length for N-step batch update.
    """
    if env_name == "CartPole-v1":
        return 500

    if env_name == "MountainCar-v0":
        return 200

    if env_name == "Acrobot-v1":
        return 500

    return 500


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
    env_name="CartPole-v1",
    show_progress=False
):
    """
    Train PPO on a discrete Gymnasium environment.

    The function keeps the same interface as your compare.py.
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

    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError(
            f"This PPO version is for discrete action spaces only, "
            f"but got {env.action_space}."
        )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(
        f"[PPO] env={env_name} | "
        f"action_type=discrete | "
        f"action_dim={action_dim}"
    )

    agent = PPO(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
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
                    next_state,
                    done=terminated
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

    env.close()

    return return_list
