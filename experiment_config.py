# ============================================
# Experiment configuration
# ============================================

# Algorithms to run
# Available options:
# "DQN", "PPO", "SAC", "TD3"
ALGORITHMS_TO_RUN = ["TD3"]


# Environment
# Discrete action environments:   CartPole-v1, MountainCar-v0, Acrobot-v1
# Continuous action environments: Pendulum-v1, MountainCarContinuous-v0
ENV_NAME = "Pendulum-v1"


# Basic experiment settings
NUM_EPISODES = 1000
SEEDS = [0, 1, 2]
HIDDEN_DIM = 256
GAMMA = 0.9926973161132563


# DQN settings
DQN_LR = 0.0015601373686407556

# Epsilon-greedy exploration schedule
DQN_EPSILON_START = 1.0
DQN_EPSILON_END = 0.1017015525126623
DQN_EPSILON_DECAY_STEPS = 3000

DQN_TARGET_UPDATE = 20
DQN_BUFFER_SIZE = 2000
DQN_MINIMAL_SIZE = 500
DQN_BATCH_SIZE = 128

# PPO settings for discrete action environments
PPO_DISCRETE_ACTOR_LR = 0.0006675568795816485
PPO_DISCRETE_CRITIC_LR = 0.0006539403121809076
PPO_DISCRETE_LMBDA = 0.908225386881102
PPO_DISCRETE_EPOCHS = 6
PPO_DISCRETE_EPS = 0.2578627908175063

# PPO settings for continuous action environments
PPO_CONTINUOUS_ACTOR_LR = 0.00016732292370154916
PPO_CONTINUOUS_CRITIC_LR = 0.0009953591055501898
PPO_CONTINUOUS_LMBDA = 0.95
PPO_CONTINUOUS_EPOCHS = 7
PPO_CONTINUOUS_EPS = 0.20771168925712913

# SAC settings
SAC_ACTOR_LR = 3e-4
SAC_CRITIC_LR = 3e-3
SAC_ALPHA_LR = 3e-4
SAC_TAU = 0.005
SAC_BUFFER_SIZE = 10000
SAC_MINIMAL_SIZE = 500
SAC_BATCH_SIZE = 64


# Tuning switch
# True: run baseline + hyperparameter tuning
# False: only run baseline
RUN_TUNING = True


# DQN / DoubleDQN tuning values
DQN_LR_VALUES = [1e-4, 2.5e-4, 5e-4, 1e-3]
DQN_EPSILON_DECAY_STEPS_VALUES = [5000, 10000, 20000]
DQN_TARGET_UPDATE_VALUES = [5, 10, 20, 50]
DQN_BATCH_SIZE_VALUES = [32, 64, 128]

# PPO tuning values for discrete environments
PPO_DISCRETE_ACTOR_LR_VALUES = [1e-4, 5e-4, 1e-3]
PPO_DISCRETE_EPS_VALUES = [0.1, 0.2, 0.3]
PPO_DISCRETE_EPOCHS_VALUES = [5, 10, 15]


# PPO tuning values for continuous environments
PPO_CONTINUOUS_ACTOR_LR_VALUES = [1e-5, 1e-4, 5e-4]
PPO_CONTINUOUS_EPS_VALUES = [0.1, 0.2, 0.3]
PPO_CONTINUOUS_EPOCHS_VALUES = [5, 10, 15]


# SAC tuning values
SAC_ACTOR_LR_VALUES = [1e-4, 3e-4, 1e-3]
SAC_ALPHA_LR_VALUES = [1e-4, 3e-4, 1e-3]
SAC_TAU_VALUES = [0.001, 0.005, 0.01]


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