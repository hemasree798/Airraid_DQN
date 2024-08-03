import gym
import torch
from agent import DQNAgent
from trainer import Trainer
from gym.wrappers import FrameStack

# Hyperparameters
learning_rate = 0.00025
gamma = 0.99
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99
batch_size = 32
memory_size = 100000
target_update_frequency = 10
num_episodes = 100

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the environment
env = gym.make("ALE/AirRaid-v5")

# Apply frame stacking
num_frames = 4
env = FrameStack(env, num_frames)

# Define input shape based on frame stacking
input_shape = (num_frames, 80, 80)  # Adjust as per your preprocessing
num_actions = env.action_space.n

# Create the DQN agent
agent = DQNAgent(
    input_shape,
    num_actions,
    lr=learning_rate,
    gamma=gamma,
    epsilon=epsilon_start,
    epsilon_min=epsilon_min,
    epsilon_decay=epsilon_decay,
    batch_size=batch_size,
    memory_size=memory_size,
    target_update=target_update_frequency,
    device=device
)

# Create the trainer
trainer = Trainer(env, agent)

# Train the agent
trainer.train(num_episodes=num_episodes)

# Evaluate the agent
#trainer.evaluate(num_eval_pisodes=100)
