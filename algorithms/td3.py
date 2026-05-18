# ============================================
# TD3 algorithm for continuous action environments
# ============================================

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import gymnasium as gym
from tqdm import tqdm

from utils import ReplayBuffer, set_seed


# ============================================
# Networks
# ============================================

class PolicyNet(nn.Module):
    """
    Deterministic actor for continuous action spaces.
    Output is tanh-bounded and scaled by action_bound.
    """
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


class QValueNet(nn.Module):
    """
    Critic: takes (state, action) concatenated, outputs scalar Q value.
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


# ============================================
# TD3 Agent
# ============================================

class TD3:
    """
    TD3 agent for continuous action environments.

    Key features over DDPG:
    1. Twin critics to reduce overestimation bias.
    2. Delayed actor and target updates (every policy_delay critic steps).
    3. Target policy smoothing (noise added to target actions).
    """
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
        exploration_noise,
        policy_noise,
        noise_clip,
        policy_delay,
        device
    ):
        self.action_bound = action_bound
        self.tau = tau
        self.gamma = gamma
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.device = device
        self.total_it = 0

        self.actor = PolicyNet(
            state_dim, hidden_dim, action_dim, action_bound
        ).to(device)

        self.target_actor = PolicyNet(
            state_dim, hidden_dim, action_dim, action_bound
        ).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)

        self.target_critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )

        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(), lr=critic_lr
        )

        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(), lr=critic_lr
        )

    def take_action(self, state):
        state = torch.tensor(
            np.array([state]), dtype=torch.float
        ).to(self.device)

        action = self.actor(state).detach().cpu().numpy()[0]

        noise = np.random.normal(0, self.exploration_noise, size=action.shape)

        return np.clip(action + noise, -self.action_bound, self.action_bound)

    def soft_update(self, net, target_net):
        for target_param, param in zip(
            target_net.parameters(), net.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def update(self, transition_dict):
        self.total_it += 1

        states = torch.tensor(
            transition_dict["states"], dtype=torch.float
        ).to(self.device)

        actions = torch.tensor(
            transition_dict["actions"], dtype=torch.float
        ).to(self.device)

        rewards = torch.tensor(
            transition_dict["rewards"], dtype=torch.float
        ).view(-1, 1).to(self.device)

        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)

        dones = torch.tensor(
            transition_dict["dones"], dtype=torch.float
        ).view(-1, 1).to(self.device)

        with torch.no_grad():
            noise = (
                torch.randn_like(actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_actions = (
                self.target_actor(next_states) + noise
            ).clamp(-self.action_bound, self.action_bound)

            target_q1 = self.target_critic_1(next_states, next_actions)
            target_q2 = self.target_critic_2(next_states, next_actions)
            td_target = (
                rewards
                + self.gamma * torch.min(target_q1, target_q2) * (1 - dones)
            )

        critic_1_loss = F.mse_loss(self.critic_1(states, actions), td_target)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        critic_2_loss = F.mse_loss(self.critic_2(states, actions), td_target)
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic_1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic_1, self.target_critic_1)
            self.soft_update(self.critic_2, self.target_critic_2)


# ============================================
# Training function for compare.py
# ============================================

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
    """
    Train TD3 on a continuous Gymnasium environment.

    Returns:
        return_list: list of episode returns
    """
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError(
            f"TD3 only supports continuous action spaces, "
            f"but got {env.action_space}."
        )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0])

    print(
        f"[TD3] env={env_name} | "
        f"action_dim={action_dim} | "
        f"action_range=[{env.action_space.low}, {env.action_space.high}]"
    )

    agent = TD3(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        action_bound=action_bound,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        tau=tau,
        gamma=gamma,
        exploration_noise=exploration_noise,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        policy_delay=policy_delay,
        device=device
    )

    replay_buffer = ReplayBuffer(buffer_size)

    return_list = []

    episode_iter = range(num_episodes)

    if show_progress:
        episode_iter = tqdm(
            episode_iter,
            desc=f"TD3 seed={seed}",
            leave=False
        )

    for _ in episode_iter:
        state, _ = env.reset()
        done = False
        episode_return = 0.0

        while not done:
            action = agent.take_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_return += reward

            if replay_buffer.size() > minimal_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(
                    batch_size
                )

                transition_dict = {
                    "states": states,
                    "actions": actions,
                    "rewards": rewards,
                    "next_states": next_states,
                    "dones": dones,
                }

                agent.update(transition_dict)

        return_list.append(episode_return)

    env.close()

    return return_list
