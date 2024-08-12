import gymnasium as gym
import torch
from agent import DQNAgent
from trainer import Trainer
#from gymnasium.wrappers import FrameStack

# Hyperparameters
learning_rate = 0.0001
gamma = 0.99
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory_size = 10000
target_update_frequency = 10000
num_episodes = 1000
frame_stack_size = 4  # Number of frames to stack

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the Air Raid environment
env = gym.make("ALE/AirRaid-v5")
input_shape = (frame_stack_size, 80, 80)  # Adjusted to (C, H, W) after preprocessing
num_actions = env.action_space.n
#env.metadata['render_fps'] = 60

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
trainer = Trainer(env, agent, frame_stack_size=frame_stack_size)


# Train the agent
trainer.train(num_episodes=num_episodes)

#model_path = r'C:\Users\hemas\Documents\Applied_AI_and_ML_Courses\Reinforcement_Learning\Airraid_DQN\sachin_code\models\dqn_airraid_550.pth'

#agent.policy_net.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
#agent.policy_net.eval()

# Evaluate the agent
#trainer.evaluate(num_eval_episodes=3)