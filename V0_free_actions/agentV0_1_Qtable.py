import random
import math
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFrame

from environment import Scratch_Game_Environment

class RL_Agent_01():
    """Reinforcement Learning Agent with q-table where each action goes to every state. "Free" movement"""

    def __init__(self, game_env: Scratch_Game_Environment):
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.98  # Discount factor
        self.epsilon = 0.15  # Exploration rate

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