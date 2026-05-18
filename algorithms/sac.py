# ============================================
# SAC algorithm for discrete and continuous action environments
# ============================================

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal
import gymnasium as gym
from tqdm import tqdm

from utils import ReplayBuffer, set_seed


# ============================================
# Continuous SAC Networks
# ============================================

class PolicyNetContinuous(nn.Module):
    """
    Policy network for continuous action spaces.

    It outputs a tanh-squashed Gaussian action.
    """
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))

        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x)) + 1e-6

        dist = Normal(mu, std)

        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action)

        # Tanh correction
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        action = action * self.action_bound

        return action, log_prob


class QValueNetContinuous(nn.Module):
    """
    Q network for continuous action spaces.

    It estimates Q(s, a).
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
# Continuous SAC Agent
# ============================================

class SACContinuous:
    """
    SAC for continuous action environments.
    """
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        action_bound,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy,
        tau,
        gamma,
        device,
        reward_scale=False
    ):
        self.actor = PolicyNetContinuous(
            state_dim,
            hidden_dim,
            action_dim,
            action_bound
        ).to(device)

        self.critic_1 = QValueNetContinuous(
            state_dim,
            hidden_dim,
            action_dim
        ).to(device)

        self.critic_2 = QValueNetContinuous(
            state_dim,
            hidden_dim,
            action_dim
        ).to(device)

        self.target_critic_1 = QValueNetContinuous(
            state_dim,
            hidden_dim,
            action_dim
        ).to(device)

        self.target_critic_2 = QValueNetContinuous(
            state_dim,
            hidden_dim,
            action_dim
        ).to(device)

        self.target_critic_1.load_state_dict(
            self.critic_1.state_dict()
        )
        self.target_critic_2.load_state_dict(
            self.critic_2.state_dict()
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr
        )

        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(),
            lr=critic_lr
        )

        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(),
            lr=critic_lr
        )

        self.log_alpha = torch.tensor(
            np.log(0.01),
            dtype=torch.float,
            device=device,
            requires_grad=True
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=alpha_lr
        )

        self.target_entropy = target_entropy
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.reward_scale = reward_scale

    def take_action(self, state):
        state = torch.tensor(
            np.array([state]),
            dtype=torch.float
        ).to(self.device)

        action, _ = self.actor(state)

        return action.detach().cpu().numpy()[0]

    def calc_target(self, rewards, next_states, dones):
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob

        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)

        next_value = (
            torch.min(q1_value, q2_value)
            + self.log_alpha.exp() * entropy
        )

        td_target = rewards + self.gamma * next_value * (1 - dones)

        return td_target

    def soft_update(self, net, target_net):
        for target_param, param in zip(
            target_net.parameters(),
            net.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau)
                + param.data * self.tau
            )

    def update(self, transition_dict):
        states = torch.tensor(
            transition_dict["states"],
            dtype=torch.float
        ).to(self.device)

        actions = torch.tensor(
            transition_dict["actions"],
            dtype=torch.float
        ).to(self.device)

        rewards = torch.tensor(
            transition_dict["rewards"],
            dtype=torch.float
        ).view(-1, 1).to(self.device)

        next_states = torch.tensor(
            transition_dict["next_states"],
            dtype=torch.float
        ).to(self.device)

        dones = torch.tensor(
            transition_dict["dones"],
            dtype=torch.float
        ).view(-1, 1).to(self.device)

        if self.reward_scale:
            rewards = (rewards + 8.0) / 8.0

        td_target = self.calc_target(
            rewards,
            next_states,
            dones
        )

        critic_1_loss = torch.mean(
            F.mse_loss(
                self.critic_1(states, actions),
                td_target.detach()
            )
        )

        critic_2_loss = torch.mean(
            F.mse_loss(
                self.critic_2(states, actions),
                td_target.detach()
            )
        )

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        new_actions, log_prob = self.actor(states)
        entropy = -log_prob

        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)

        actor_loss = torch.mean(
            -self.log_alpha.exp() * entropy
            - torch.min(q1_value, q2_value)
        )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean(
            (
                entropy.detach()
                - self.target_entropy
            ) * self.log_alpha.exp()
        )

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


# ============================================
# Discrete SAC Networks
# ============================================

class PolicyNetDiscrete(nn.Module):
    """
    Policy network for discrete action spaces.

    It outputs action probabilities.
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class QValueNetDiscrete(nn.Module):
    """
    Q network for discrete action spaces.

    It outputs Q values for all actions.
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ============================================
# Discrete SAC Agent
# ============================================

class SACDiscrete:
    """
    SAC for discrete action environments.
    """
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy,
        tau,
        gamma,
        device
    ):
        self.actor = PolicyNetDiscrete(
            state_dim,
            hidden_dim,
            action_dim
        ).to(device)

        self.critic_1 = QValueNetDiscrete(
            state_dim,
            hidden_dim,
            action_dim
        ).to(device)

        self.critic_2 = QValueNetDiscrete(
            state_dim,
            hidden_dim,
            action_dim
        ).to(device)

        self.target_critic_1 = QValueNetDiscrete(
            state_dim,
            hidden_dim,
            action_dim
        ).to(device)

        self.target_critic_2 = QValueNetDiscrete(
            state_dim,
            hidden_dim,
            action_dim
        ).to(device)

        self.target_critic_1.load_state_dict(
            self.critic_1.state_dict()
        )
        self.target_critic_2.load_state_dict(
            self.critic_2.state_dict()
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr
        )

        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(),
            lr=critic_lr
        )

        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(),
            lr=critic_lr
        )

        self.log_alpha = torch.tensor(
            np.log(0.01),
            dtype=torch.float,
            device=device,
            requires_grad=True
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=alpha_lr
        )

        self.target_entropy = target_entropy
        self.tau = tau
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor(
            np.array([state]),
            dtype=torch.float
        ).to(self.device)

        probs = self.actor(state)
        action_dist = Categorical(probs)
        action = action_dist.sample()

        return action.item()

    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)

        entropy = -torch.sum(
            next_probs * next_log_probs,
            dim=1,
            keepdim=True
        )

        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)

        min_qvalue = torch.sum(
            next_probs * torch.min(q1_value, q2_value),
            dim=1,
            keepdim=True
        )

        next_value = (
            min_qvalue
            + self.log_alpha.exp() * entropy
        )

        td_target = rewards + self.gamma * next_value * (1 - dones)

        return td_target

    def soft_update(self, net, target_net):
        for target_param, param in zip(
            target_net.parameters(),
            net.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau)
                + param.data * self.tau
            )

    def update(self, transition_dict):
        states = torch.tensor(
            transition_dict["states"],
            dtype=torch.float
        ).to(self.device)

        actions = torch.tensor(
            transition_dict["actions"],
            dtype=torch.long
        ).view(-1, 1).to(self.device)

        rewards = torch.tensor(
            transition_dict["rewards"],
            dtype=torch.float
        ).view(-1, 1).to(self.device)

        next_states = torch.tensor(
            transition_dict["next_states"],
            dtype=torch.float
        ).to(self.device)

        dones = torch.tensor(
            transition_dict["dones"],
            dtype=torch.float
        ).view(-1, 1).to(self.device)

        td_target = self.calc_target(
            rewards,
            next_states,
            dones
        )

        critic_1_q_values = self.critic_1(states).gather(
            1,
            actions
        )

        critic_2_q_values = self.critic_2(states).gather(
            1,
            actions
        )

        critic_1_loss = torch.mean(
            F.mse_loss(
                critic_1_q_values,
                td_target.detach()
            )
        )

        critic_2_loss = torch.mean(
            F.mse_loss(
                critic_2_q_values,
                td_target.detach()
            )
        )

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)

        entropy = -torch.sum(
            probs * log_probs,
            dim=1,
            keepdim=True
        )

        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)

        min_qvalue = torch.sum(
            probs * torch.min(q1_value, q2_value),
            dim=1,
            keepdim=True
        )

        actor_loss = torch.mean(
            -self.log_alpha.exp() * entropy
            - min_qvalue
        )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean(
            (
                entropy.detach()
                - self.target_entropy
            ) * self.log_alpha.exp()
        )

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


# ============================================
# Training function for compare.py
# ============================================

def train_sac(
    actor_lr,
    critic_lr,
    alpha_lr,
    num_episodes,
    hidden_dim,
    gamma,
    tau,
    buffer_size,
    minimal_size,
    batch_size,
    seed=0,
    env_name="CartPole-v1",
    show_progress=False
):
    """
    Train SAC on a Gymnasium environment.

    It automatically chooses:
    1. SACDiscrete for discrete action environments.
    2. SACContinuous for continuous action environments.

    Returns:
        return_list: list of episode returns
    """

    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    env = gym.make(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    state_dim = env.observation_space.shape[0]

    replay_buffer = ReplayBuffer(buffer_size)

    is_discrete = isinstance(
        env.action_space,
        gym.spaces.Discrete
    )

    if is_discrete:
        action_dim = env.action_space.n
        target_entropy = -1.0

        agent = SACDiscrete(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            target_entropy=target_entropy,
            tau=tau,
            gamma=gamma,
            device=device
        )

    else:
        action_dim = env.action_space.shape[0]
        action_bound = float(env.action_space.high[0])
        target_entropy = -float(action_dim)

        reward_scale = env_name.startswith("Pendulum")

        agent = SACContinuous(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            action_bound=action_bound,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            target_entropy=target_entropy,
            tau=tau,
            gamma=gamma,
            device=device,
            reward_scale=reward_scale
        )

    return_list = []

    episode_iter = range(num_episodes)

    if show_progress:
        episode_iter = tqdm(
            episode_iter,
            desc=f"SAC seed={seed}",
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

            replay_buffer.add(
                state,
                action,
                reward,
                next_state,
                done
            )

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
                    "dones": dones
                }

                agent.update(transition_dict)

        return_list.append(episode_return)

    env.close()

    return return_list