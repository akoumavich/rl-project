# ============================================
# DQN implementation for discrete-action environments
# ============================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from utils import ReplayBuffer, set_seed


class Qnet(nn.Module):
    """
    A simple Q-network with one hidden layer.

    Input:
        state

    Output:
        Q-values for all possible actions
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        Forward pass of the Q-network.
        """
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN:
    """
    Deep Q-Network agent.

    Main components:
    1. Online Q-network
    2. Target Q-network
    3. Epsilon-greedy exploration
    4. Experience replay
    """

    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        learning_rate,
        gamma,
        epsilon,
        target_update,
        device
    ):
        self.action_dim = action_dim

        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)

        self.optimizer = torch.optim.RMSprop(
            self.q_net.parameters(),
            lr=learning_rate
        )

        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

        # Initialize the target network with the same parameters
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def take_action(self, state):
        """
        Select an action using epsilon-greedy exploration.
        """
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(
                [state],
                dtype=torch.float32
            ).to(self.device)

            action = self.q_net(state).argmax().item()

        return action

    def update(self, transition_dict):
        """
        Update the Q-network using one mini-batch of transitions.
        """
        states = torch.tensor(
            transition_dict["states"],
            dtype=torch.float32
        ).to(self.device)

        actions = torch.tensor(
            transition_dict["actions"],
            dtype=torch.long
        ).view(-1, 1).to(self.device)

        rewards = torch.tensor(
            transition_dict["rewards"],
            dtype=torch.float32
        ).view(-1, 1).to(self.device)

        next_states = torch.tensor(
            transition_dict["next_states"],
            dtype=torch.float32
        ).to(self.device)

        dones = torch.tensor(
            transition_dict["dones"],
            dtype=torch.float32
        ).view(-1, 1).to(self.device)

        # Current Q-value Q(s, a)
        q_values = self.q_net(states).gather(1, actions)

        # Target Q-value:
        # r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update the target network
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1

class ConvolutionalQnet(torch.nn.Module):
    ''' Convolutional Qnet '''
    def __init__(self, action_dim, in_channels=4):
        super(ConvolutionalQnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512)
        self.head = torch.nn.Linear(512, action_dim)

    def forward(self, x):
        x = x / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x))
        return self.head(x)


def train_dqn(
    lr=2e-3,
    num_episodes=500,
    hidden_dim=128,
    gamma=0.98,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay_steps=10000,
    target_update=10,
    buffer_size=10000,
    minimal_size=500,
    batch_size=64,
    seed=0,
    env_name="CartPole-v1",
    show_progress=False
):
    """
    Train one DQN agent on one environment with one random seed.

    This function returns the episode return list, which can later be used
    for plotting learning curves and comparing different hyperparameters.

    Epsilon is linearly annealed from epsilon_start to epsilon_end
    according to environment interaction steps.
    """
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

    env = gym.make(env_name)
    env.action_space.seed(seed)

    replay_buffer = ReplayBuffer(buffer_size)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQN(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        learning_rate=lr,
        gamma=gamma,
        epsilon=epsilon_start,
        target_update=target_update,
        device=device
    )

    return_list = []
    total_steps = 0

    outer_loop = range(10)

    if show_progress:
        outer_loop = tqdm(outer_loop, desc=f"Seed {seed}")

    for _ in outer_loop:
        for _ in range(int(num_episodes / 10)):
            episode_return = 0
            state, info = env.reset()

            done = False

            while not done:
                agent.epsilon = max(
                    epsilon_end,
                    epsilon_start - total_steps / epsilon_decay_steps * (epsilon_start - epsilon_end)
                )

                action = agent.take_action(state)

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                total_steps += 1

                replay_buffer.add(state, action, reward, next_state, done)

                state = next_state
                episode_return += reward

                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)

                    transition_dict = {
                        "states": b_s,
                        "actions": b_a,
                        "next_states": b_ns,
                        "rewards": b_r,
                        "dones": b_d
                    }

                    agent.update(transition_dict)

            return_list.append(episode_return)

    env.close()

    return return_list