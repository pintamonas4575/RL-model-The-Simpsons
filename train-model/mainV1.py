import sys
import random
import math
import time
import matplotlib.pyplot as plt
import io
import cv2
import psutil
import os
import gc
import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QBuffer, QRect, QTimer
from PyQt5.QtGui import QPixmap, QImage, QBrush, QPalette
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QWidget, QPushButton, QVBoxLayout, QFrame

from environment import Scratch_Game_Environment

class RL_Agent1():

    def __init__(self):
        self.global_reward = 0
        self.alpha = 0.1  # Learning rate
        self.gamma = 1.0  # Discount factor
        self.epsilon = 0.25  # Exploration rate

    def add_env(self, game_env: Scratch_Game_Environment):
        self.game_env = game_env
        self.num_states = self.game_env.total_squares # total states (cells); ej:585
        self.q_table = np.zeros(shape=(self.num_states, self.num_states)) # each action goes to every state
        self.visited_frames = set()
        
    def choose_action(self, current_state: int) -> int:
        if random.random() < self.epsilon:  # Exploration
            return random.randint(0, self.num_states - 1)
        else:  # Exploitation
            best_action = np.argmax(self.q_table[current_state, :])  # best action based on Q-table
            return self.radial_scan(current_state, best_action)

    def radial_scan(self, current_state: int, best_action: int) -> int:
        """
        Perform a radial scan around the best action to find a potentially better action.
        """
        radius = 2  # Define the radius of the scan
        best_reward = self.q_table[current_state, best_action]
        best_action_in_radius = best_action

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor_state = best_action + dx + dy * int(math.sqrt(self.num_states))
                if 0 <= neighbor_state < self.num_states:
                    if self.q_table[current_state, neighbor_state] > best_reward:
                        best_reward = self.q_table[current_state, neighbor_state]
                        best_action_in_radius = neighbor_state
        return best_action_in_radius

    def get_reward(self, next_frame: QFrame):
        if next_frame in self.visited_frames:
            reward = -50  # Penalty for revisiting
        else:
            self.visited_frames.add(next_frame)
            response = self.game_env.remove_square(next_frame)
            if response: # red frame
                reward = 100
            else: # blue frame
                reward = -10

        self.global_reward += reward
        return reward

    def update_q_table(self, current_state: int, action: int, reward: int, next_state: int):
        """with training loop, action==next_state

        update q-table value based on Bellman´s equation
        """
        self.q_table[current_state, action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state, :]) - self.q_table[current_state, action])

    def finish_game(self) -> None:
        QTimer.singleShot(0, self.game_env.close_button.click)

    def reset_agent(self):
        self.visited_frames.clear()
        self.global_reward = 0

"""---------------------------------------------------------"""
"""---------------------------------------------------------"""
agent = RL_Agent1()
EPISODES = 1000
trace = 100
max_reward = -9999999
min_actions_done = 9999999
min_area_scratched = 999

my_env = Scratch_Game_Environment(frame_size=20, scratching_area=(110,98,770,300), num_emojis=3)
agent.add_env(my_env)

rewards = []
max_rewards = []
actions_done = []
min_actions_done_list = []
areas_scratched = []
min_areas_scratched = []

start = time.time()
for i in range(EPISODES):

    agent.game_env.reset_env()
    agent.reset_agent()

    done = False
    num_actions_done = 0

    current_state: int = random.randint(0, agent.num_states - 1)  # Start at a random state
    while not done:
        num_actions_done+=1

        # Choose an action (next state)
        action: int = agent.choose_action(current_state)
        next_state = action  # Since the actions are cells, the destiny is the action
        next_frame = agent.game_env.squares[next_state]

        # Calculate reward
        reward: int = agent.get_reward(next_frame)

        # Update Q-table
        agent.update_q_table(current_state, action, reward, next_state)

        # Update current state
        current_state = next_state
        done = all(not s for s in agent.game_env.emoji_frame_track.values())

    max_reward = agent.global_reward if agent.global_reward > max_reward else max_reward
    min_actions_done = num_actions_done if num_actions_done < min_actions_done else min_actions_done

    final_percentage_scratched = (agent.game_env.scratched_count / len(agent.game_env.squares)) * 100
    min_area_scratched = final_percentage_scratched if final_percentage_scratched < min_area_scratched else min_area_scratched
    if i % trace == 0 or i == EPISODES-1:
        print(f"-----------------EPOCH {i+1}---------------------")
        print("Actions done:", num_actions_done)
        print("reward:", agent.global_reward)
        print("max reward:", max_reward)
        print("min actions done:", min_actions_done)

        print(f"Final scratched area: {final_percentage_scratched:.2f}%")

        agent.game_env.app.processEvents()
        agent.game_env.get_window_image_and_save(True, f"episodes/episode_{i+1}.png")
    
    # ---------------data for graphics----------------
    rewards.append(agent.global_reward)
    max_rewards.append(max_reward)
    actions_done.append(num_actions_done)
    min_actions_done_list.append(min_actions_done)
    areas_scratched.append(final_percentage_scratched)
    min_areas_scratched.append(min_area_scratched)
    # ---------------data for graphics----------------

    agent.finish_game() # "app.quit" and "del app"
    gc.collect()  # Explicitly run garbage collection to free resources
    agent.game_env.app.exec() # run app

minutos, segundos = divmod(time.time()-start, 60)
print(f"****Tiempo total: {int(minutos)} minutos y {segundos:.2f} segundos****")

"""**********************************************************"""

plt.figure(figsize=(18, 12))

plt.subplot(3, 1, 1)
plt.plot(rewards, label='Rewards')
plt.plot(max_rewards, label='Max Rewards')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Rewards vs Episodes')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(actions_done, label='Actions Done')
plt.plot(min_actions_done_list, label='Min Actions Done')
plt.xlabel('Episodes')
plt.ylabel('Actions Done')
plt.title('Actions Done vs Episodes')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(areas_scratched, label='Area Scratched')
plt.plot(min_areas_scratched, label='Min Area Scratched')
plt.xlabel('Episodes')
plt.ylabel('Area Scratched')
plt.title('Area Scratched vs Episodes')
plt.legend()

plt.tight_layout()
plt.savefig("episodes/PLOT.png")
plt.close()