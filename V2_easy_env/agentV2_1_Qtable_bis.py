import random
import math
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFrame

from environmentV2 import Scratch_Game_Environment2 #V2_easy_env/environment.py

class RL_Agent_21():
    """Reinforcement Learning Agent with q-table where each action goes to every state. "Free" movement"""

    def __init__(self, game_env: Scratch_Game_Environment2):
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.98  # Discount factor
        self.epsilon = 0.15  # Exploration rate

        self.game_env = game_env
        self.num_states = self.game_env.total_squares # total states (cells); ej:585
        self.num_actions = self.num_states 
        self.q_table = np.zeros(shape=(self.num_actions, self.num_actions)) # each action goes to every state, ej: 585x585
        np.fill_diagonal(self.q_table, -np.inf) # fill diagonal with -inf to avoid self-loop

    def update_q_table(self, current_state: int, action: int, reward: int, next_state: int) -> None:
        """Update q-table value based on Bellman´s equation."""

        self.q_table[current_state, action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state, :]) - self.q_table[current_state, action])

    def finish_game(self) -> None:
        QTimer.singleShot(0, self.game_env.close_button.click)