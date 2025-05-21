import random
import numpy as np
from PyQt5.QtCore import QTimer
from environmentV5 import Scratch_Game_Environment5

class RL_Agent_51():
    """Reinforcement Learning Agent."""

    def __init__(self, game_env: Scratch_Game_Environment5):
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor

        self.game_env = game_env
        self.num_states = self.game_env.total_squares # total cells; ej:585
        self.num_actions = self.num_states
        self.q_table = np.zeros(shape=(self.num_actions, self.num_actions)) # ej: 585x585
        np.fill_diagonal(self.q_table, -np.inf) # fill diagonal with -inf to avoid self-loop

    def choose_action(self, current_action: int, state: list[int], epsilon: float) -> int:
        """Choose the next action based on epsilon-greedy policy."""

        possible_actions = [i for i, val in enumerate(state) if val == -1]
        if random.random() < epsilon:
            action_index = random.choice(possible_actions)
        else:
            q_values = self.q_table[current_action, possible_actions]
            action_index = possible_actions[np.argmax(q_values)]

        return action_index

    def update_q_table(self, current_action: int, action: int, reward: int, next_state: list[int]) -> None:
        """Update q-table value based on Bellman´s equation."""

        self.q_table[current_action, action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state, :]) - self.q_table[current_action, action])

    def finish_game(self) -> None:
        QTimer.singleShot(0, self.game_env.close_button.click)