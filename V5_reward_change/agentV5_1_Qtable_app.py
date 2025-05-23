import random
import numpy as np
from environmentV5 import Scratch_Game_Environment5

class RL_Agent_51_Streamlit():
    """Reinforcement Learning Agent. Q-Table is NxN, where N is the number of squares in the game."""

    def __init__(self, num_actions: int) -> None:
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor

        self.q_table = np.zeros(shape=(num_actions, num_actions))
        np.fill_diagonal(self.q_table, -np.inf) # avoid self-loop

    def choose_action(self, current_action: int, current_state: list[int], epsilon: float) -> int:
        """Choose the next action based on epsilon-greedy policy. See the possible actions in the current state."""

        possible_actions = [i for i, val in enumerate(current_state) if val == -1]
        print("Number_of_possible_actions", len(possible_actions))
        # breakpoint()
        if random.random() < epsilon:
            action_index = random.choice(possible_actions)
        else:
            q_values = self.q_table[current_action, possible_actions]
            action_index = possible_actions[np.argmax(q_values)]
        return action_index

    def update_q_table(self, current_action: int, action: int, reward: int, next_state: list[int]) -> None:
        """Update q-table value based on Bellman´s equation."""

        self.q_table[current_action, action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state, :]) - self.q_table[current_action, action])