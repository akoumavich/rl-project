# ============================================
# Original SAC from Haarnoja et al. 2018
# "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
# Reinforcement Learning with a Stochastic Actor"
#
# This file follows Algorithm 1 in the PDF:
#   - policy pi_phi(a | s)
#   - state value V_psi(s)
#   - target state value V_bar_psi(s)
#   - two soft Q-functions Q_theta_1(s, a), Q_theta_2(s, a)
#   - no automatic temperature tuning
#   - no discrete-action SAC extension
# ============================================

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import gymnasium as gym
from tqdm import tqdm

from utils import ReplayBuffer, set_seed


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class PolicyNet(nn.Module):
    """
    Stochastic Gaussian policy with tanh squashing.

    This is the reparameterized policy a_t = f_phi(epsilon_t; s_t)
    used by Equation 11 and Equation 12 in the paper.
    """
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)

        self.action_bound = action_bound

    def forward(self, state, deterministic=False):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mu, std)

        if deterministic:
            raw_action = mu
        else:
            raw_action = dist.rsample()

        squashed_action = torch.tanh(raw_action)
        action = squashed_action * self.action_bound

        log_prob = dist.log_prob(raw_action)
        log_prob = log_prob - torch.log(1 - squashed_action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob


class ValueNet(nn.Module):
    """
    State value function V_psi(s), trained with Equation 5.
    """
    def __init__(self, state_dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class QValueNet(nn.Module):
    """
    Soft Q-function Q_theta(s, a), trained with Equation 7.
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


class SAC:
    """
    Continuous-action SAC matching Algorithm 1 in the PDF.

    The paper omits the temperature alpha in the practical losses by
    subsuming it into reward scaling. Therefore this implementation uses
    reward_scale instead of automatic alpha tuning.
    """
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        action_bound,
        actor_lr,
        critic_lr,
        value_lr,
        tau,
        gamma,
        device,
        reward_scale=1.0
    ):
        self.actor = PolicyNet(
            state_dim,
            hidden_dim,
            action_dim,
            action_bound
        ).to(device)

        self.value = ValueNet(
            state_dim,
            hidden_dim
        ).to(device)

        self.target_value = ValueNet(
            state_dim,
            hidden_dim
        ).to(device)

        self.critic_1 = QValueNet(
            state_dim,
            hidden_dim,
            action_dim
        ).to(device)

        self.critic_2 = QValueNet(
            state_dim,
            hidden_dim,
            action_dim
        ).to(device)

        self.target_value.load_state_dict(self.value.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(),
            lr=value_lr
        )
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(),
            lr=critic_lr
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(),
            lr=critic_lr
        )

        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.reward_scale = reward_scale

    def take_action(self, state, deterministic=False):
        state = torch.tensor(
            np.array([state]),
            dtype=torch.float
        ).to(self.device)

        action, _ = self.actor(
            state,
            deterministic=deterministic
        )

        return action.detach().cpu().numpy()[0]

    def soft_update_target_value(self):
        for target_param, param in zip(
            self.target_value.parameters(),
            self.value.parameters()
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

        rewards = rewards * self.reward_scale

        # Equation 5: J_V = 1/2 * (V(s) - E_a[Q(s,a) - log pi(a|s)])^2
        new_actions, log_probs = self.actor(states)
        q1_new_actions = self.critic_1(states, new_actions)
        q2_new_actions = self.critic_2(states, new_actions)
        min_q_new_actions = torch.min(q1_new_actions, q2_new_actions)

        value_target = min_q_new_actions - log_probs
        value_loss = F.mse_loss(
            self.value(states),
            value_target.detach()
        )

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Equation 7 and Equation 8:
        # Q_hat(s,a) = r(s,a) + gamma * V_bar(s')
        with torch.no_grad():
            q_target = (
                rewards
                + self.gamma
                * self.target_value(next_states)
                * (1 - dones)
            )

        critic_1_loss = F.mse_loss(
            self.critic_1(states, actions),
            q_target
        )
        critic_2_loss = F.mse_loss(
            self.critic_2(states, actions),
            q_target
        )

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Equation 12: J_pi = E[log pi(f_phi(e; s) | s) - Q(s, f_phi(e; s))]
        new_actions, log_probs = self.actor(states)
        q1_new_actions = self.critic_1(states, new_actions)
        q2_new_actions = self.critic_2(states, new_actions)
        min_q_new_actions = torch.min(q1_new_actions, q2_new_actions)

        actor_loss = torch.mean(log_probs - min_q_new_actions)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Algorithm 1: target value moving average.
        self.soft_update_target_value()


def train_sac(
    actor_lr,
    critic_lr,
    value_lr,
    num_episodes,
    hidden_dim,
    gamma,
    tau,
    buffer_size,
    minimal_size,
    batch_size,
    seed=0,
    env_name="Pendulum-v1",
    reward_scale=1.0,
    show_progress=False
):
    """
    Train original SAC on a continuous-action Gymnasium environment.

    This intentionally rejects discrete action spaces because the PDF
    derives and evaluates SAC for continuous control.
    """
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    env = gym.make(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    if not isinstance(env.action_space, gym.spaces.Box):
        env.close()
        raise ValueError(
            "Original PDF SAC expects a continuous Box action space."
        )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = torch.tensor(
        env.action_space.high,
        dtype=torch.float,
        device=device
    )

    replay_buffer = ReplayBuffer(buffer_size)

    agent = SAC(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        action_bound=action_bound,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        value_lr=value_lr,
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
            desc=f"Original SAC seed={seed}",
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
                states, actions, rewards, next_states, dones = (
                    replay_buffer.sample(batch_size)
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
