import random
import math
import time
import gc
import sys
import os
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFrame
# ------------------------------
# Get the absolute path of the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the project root directory (go up one level from "V0_free_actions")
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Add the project root to sys.path to allow imports
if project_root not in sys.path:
    sys.path.append(project_root)
# ------------------------------
from environment import Scratch_Game_Environment # V0_free_actions/environment.py
from utils.functionalities import plot_results

class RL_Agent_01():

    def __init__(self, game_env: Scratch_Game_Environment):
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.98  # Discount factor
        self.epsilon = 0.25  # Exploration rate

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
            reward = -10  # Penalty for revisiting
        else:
            self.visited_frames.add(next_frame)
            response = self.game_env.remove_square(next_frame)
            if response: # red frame
                reward = 100
            else: # blue frame
                reward = -1
        return reward

    def update_q_table(self, current_state: int, action: int, reward: int, next_state: int):
        """
        with training loop, action==next_state
        update q-table value based on Bellman´s equation
        """
        self.q_table[current_state, action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state, :]) - self.q_table[current_state, action])

    def finish_game(self) -> None:
        QTimer.singleShot(0, self.game_env.close_button.click)

    def reset_agent(self):
        self.visited_frames.clear()

"""---------------------------------------------------------"""
"""---------------------------------------------------------"""
my_env = Scratch_Game_Environment(frame_size=20, scratching_area=(110,98,770,300), num_emojis=3)
agent = RL_Agent_01(game_env=my_env)

EPISODES = 1000
trace = 100
rewards = []
max_rewards = []
actions_done = []
min_actions_done = []
areas_scratched = []
min_areas_scratched = []
max_reward = -9999999
min_actions = 9999999
min_area_scratched = 999

start = time.time()
for i in range(EPISODES):

    agent.game_env.reset_env()
    agent.reset_agent()

    done = False
    episode_actions = 0
    episode_reward = 0

    current_state: int = random.randint(0, agent.num_states - 1)  # Start at a random state
    while not done:
        episode_actions+=1

        # Choose an action (next state)
        action: int = agent.choose_action(current_state)
        next_state = action  # Since the actions are cells, the destiny is the action
        next_frame = agent.game_env.squares[next_state]

        # Calculate reward
        reward = agent.get_reward(next_frame)
        episode_reward += reward

        # Update Q-table
        agent.update_q_table(current_state, action, reward, next_state)

        # Update current state
        current_state = next_state
        done = all(not s for s in agent.game_env.emoji_frame_track.values())

    episode_percentage = (agent.game_env.scratched_count / agent.game_env.total_squares) * 100

    max_reward = episode_reward if episode_reward > max_reward else max_reward
    min_actions = episode_actions if episode_actions < min_actions else min_actions
    min_area_scratched = episode_percentage if episode_percentage < min_area_scratched else min_area_scratched

    if i % trace == 0 or i == EPISODES-1:
        print(f"-----------------EPISODE {i+1}---------------------")
        print(f"Actions done: {episode_actions}")
        print(f"Reward: {episode_reward}")
        print(f"Final scratched area: {episode_percentage:.2f}%")
        print(f"Max reward: {max_reward}")
        print(f"Min actions done: {min_actions}")
        print(f"Min scratched area: {min_area_scratched:.2f}%")

        agent.game_env.app.processEvents()
        agent.game_env.get_window_image_and_save(True, f"episodes/episode_{i+1}.png")
    
    # ---------------data for graphics----------------
    rewards.append(episode_reward)
    actions_done.append(episode_actions)
    areas_scratched.append(episode_percentage)
    max_rewards.append(max_reward)
    min_actions_done.append(min_actions)
    min_areas_scratched.append(min_area_scratched)
    # ---------------data for graphics----------------

    agent.finish_game() # "app.quit" and "del app"
    gc.collect()  # Explicitly run garbage collection to free resources
    agent.game_env.app.exec() # run app

minutes, seconds = divmod(time.time()-start, 60)
print(f"****Total trining time: {int(minutes)} minutes and {seconds:.2f} seconds****")

"""**********************************************************"""
"""**********************************************************"""

plot_results(rewards, actions_done, areas_scratched,
             max_rewards, min_actions_done, min_areas_scratched,
             f"V0_1_Qtable_{EPISODES}.png", (int(minutes), seconds))
