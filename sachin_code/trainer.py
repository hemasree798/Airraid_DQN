import torch
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
from agent import DQNAgent
from utils import preprocess_state
import numpy as np
from datetime import datetime

# Ensure we use an interactive backend
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on your environment

class Trainer:
    def __init__(self, env, agent, log_dir='logs', model_dir='models'):
        self.env = env
        self.agent = agent
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(log_dir, timestamp)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.model_dir = model_dir

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Initialize the plot
        self.fig, self.ax = plt.subplots()
        self.img_displayed = None
        plt.ion()  # Enable interactive mode

    def show_state(self, state, episode, step,reward):
        #print(f"State shape before squeeze: {state.shape}")
        state = state.squeeze(0)  # Remove batch dimension
        #print(f"State shape after squeeze: {state.shape}")
        if len(state.shape) == 3:
            state = state.permute(1, 2, 0).cpu().numpy()
            if self.img_displayed is None:
                self.img_displayed = self.ax.imshow(state)
            else:
                self.img_displayed.set_data(state)
            self.ax.set_title(f"Episode: {episode}, Step: {step}, Total Reward: {reward}")
            plt.pause(0.001)
            self.fig.canvas.draw_idle()
        else:
            print("State has incorrect number of dimensions for visualization")

    def train(self, num_episodes):
        total_time = 0
        for episode in range(num_episodes):
            state = self.env.reset()
            try:
                state = preprocess_state(state, self.agent.device)
                #print(f"Initial state shape: {state.shape}")
            except Exception as e:
                print(f"Error preprocessing state at episode {episode}: {e}")
                continue
            total_reward = 0

            for time_step in range(10000):  # Limit steps per episode
                total_time += 1
                action = self.agent.select_action(state)
                next_state, reward, done, info,_ = self.env.step(action)
                #print(reward,action)
                try:
                    next_state = preprocess_state(next_state, self.agent.device)
                    #print(f"Next state shape: {next_state.shape}")
                except Exception as e:
                    print(f"Error preprocessing next state at episode {episode}, time step {time_step}: {e}")
                    break
                total_reward += reward

                self.agent.store_transition(state, action, reward, next_state, done)
                state = next_state
                if total_time%100 == 0:
                    self.agent.update_policy(self.writer)
                if total_time % 1000 == 0:
                    self.agent.update_epsilon()
                if done:
                    break

                # Visualize the state at each step
                self.show_state(state, episode, time_step,total_reward)

            self.writer.add_scalar('Training/TotalReward', total_reward, episode)
            self.writer.add_scalar('Training/Epsilon', self.agent.epsilon, episode)
            self.writer.add_scalar('Training/Number of steps', time_step, episode)

            # Save the model every 50 episodes
            if episode % 50 == 0:
                torch.save(self.agent.policy_net.state_dict(), f'{self.model_dir}/dqn_airraid_{episode}.pth')

            print(f"Episode {episode + 1}/{num_episodes}: Total Reward: {total_reward} Total Time: {total_time}")
        self.writer.close()
        plt.close(self.fig)

    def evaluate(self, num_eval_episodes):
        total_rewards = []

        for episode in range(num_eval_episodes):
            state = self.env.reset()
            try:
                state = preprocess_state(state, self.agent.device)
                #print(f"Initial state shape (evaluation): {state.shape}")
            except Exception as e:
                print(f"Error preprocessing state at eval episode {episode}: {e}")
                continue
            done = False
            total_reward = 0

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, info,_ = self.env.step(action)
                try:
                    next_state = preprocess_state(next_state, self.agent.device)
                    #print(f"Next state shape (evaluation): {next_state.shape}")
                except Exception as e:
                    print(f"Error preprocessing next state at eval episode {episode}: {e}")
                    break
                state = next_state
                total_reward += reward

            total_rewards.append(total_reward)

        average_reward = np.mean(total_rewards)
        self.writer.add_scalar('Evaluation/AverageReward', average_reward, 0)
        print(f"Average Reward over {num_eval_episodes} episodes: {average_reward}")

        self.writer.close()

        # Close the plot after evaluation
        plt.close(self.fig)
