# ============================================
# Utility functions and replay buffer
# ============================================

import random
import collections
import numpy as np
import torch


class ReplayBuffer:
    """
    Experience replay buffer for off-policy reinforcement learning.

    It stores transitions of the form:
    state, action, reward, next_state, done.

    DQN uses this buffer to sample mini-batches of past experience,
    which improves sample efficiency and reduces correlation between updates.
    """

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Add one transition to the replay buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly sample a mini-batch of transitions.
        """
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)

        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def size(self):
        """
        Return the current number of transitions stored in the buffer.
        """
        return len(self.buffer)


def set_seed(seed):
    """
    Set random seeds for reproducibility.

    This controls randomness from Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(data, window_size=9):
    """
    Compute a simple moving average for smoothing learning curves.
    """
    data = np.array(data, dtype=np.float32)

    if len(data) < window_size:
        return data

    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode="valid")

def compute_advantage(gamma, lmbda, td_delta):
    """
    Compute advantage using Generalized Advantage Estimation.
    """
    if isinstance(td_delta, torch.Tensor):
        td_delta = td_delta.detach().cpu().numpy()

    td_delta = td_delta.reshape(-1)

    advantage_list = []
    advantage = 0.0

    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)

    advantage_list.reverse()

    return torch.tensor(
        advantage_list,
        dtype=torch.float
    ).view(-1, 1)