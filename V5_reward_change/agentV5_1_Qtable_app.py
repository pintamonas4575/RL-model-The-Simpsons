import random
import numpy as np

class RL_Agent_51_Streamlit():
    """Reinforcement Learning Agent. Q-Table is NxN, where N is the number of squares in the game."""

    def __init__(self, alpha: float, gamma: float, epsilon: float, num_actions: int) -> None:
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate

        self.q_table = np.zeros(shape=(num_actions, num_actions))
        np.fill_diagonal(self.q_table, -np.inf) # avoid self-loop

    def choose_action(self, current_action: int, current_state: list[int]) -> int:
        """Choose the next action based on epsilon-greedy policy. See the possible actions in the current state."""

        possible_actions = [i for i, val in enumerate(current_state) if val == -1]
        if random.random() < self.epsilon:
            action_index = random.choice(possible_actions)
        else:
            q_values = self.q_table[current_action, possible_actions]
            action_index = possible_actions[np.argmax(q_values)]
        return action_index

    def update_q_table(self, current_cell: int, action: int, reward: int, next_state: list[int]) -> None:
        """Update q-table value based on Bellman´s equation."""

        self.q_table[current_cell, action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state, :]) - self.q_table[current_cell, action])