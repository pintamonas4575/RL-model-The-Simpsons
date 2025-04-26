import random
import time
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFrame

from V4_easier_agent_moves.environmentV4 import Scratch_Game_Environment4

class Agent():

    def __init__(self):
        self.global_reward = 0
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.3  # Exploration rate

    def add_env(self, game_env: Scratch_Game_Environment4):
        self.game_env = game_env
        self.num_states = self.game_env.total_squares # total states (cells); ej:585
        self.q_table = np.zeros(shape=(self.num_states, self.num_states)) # each action goes to every state
        self.visited_frames = set()

    # choose an action (destiny cell)
    def choose_action(self, current_state: int) -> int:
        if random.random() < self.epsilon:  # Exploration
            return random.randint(0, self.num_states - 1)
        else:  # Exploitation
            return np.argmax(self.q_table[current_state, :])  # best action based on Q-table

    def get_reward(self, next_frame: QFrame):
        if next_frame in self.visited_frames:
            reward = -100  # Penalty for revisiting
        else:
            self.visited_frames.add(next_frame)
            response = self.game_env.remove_square(next_frame)
            if response: # red frame
                reward = 5000
            else: # blue frame
                reward = -15

        self.global_reward += reward
        return reward

    def update_q_table(self, current_state: int, action: int, reward: int, next_state: int):
        """
        with training loop, action==next_state
        update q-table value based on Bellman's equation
        """
        self.q_table[current_state, action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state, :]) - self.q_table[current_state, action])

    def finish_game(self) -> None:
        QTimer.singleShot(0, self.game_env.close_button.click)

    def reset_agent(self):
        self.visited_frames.clear()
        self.global_reward = 0

"""---------------------------------------------------------"""
"""---------------------------------------------------------"""
agent = Agent()
max_reward = -9999999

my_env = Scratch_Game_Environment4(frame_size=20, scratching_area=(110,98,770,300))
agent.add_env(my_env)

agent.game_env.env_reset()
agent.reset_agent()
agent.game_env.window.show()

done = False
num_actions_done = 0

current_state: int = random.randint(0, agent.num_states - 1)  # Start at a random state
while not done:
    num_actions_done += 1

    # Choose an action (next state)
    action: int = agent.choose_action(current_state)
    next_state = action  # Since the actions are cells, the destiny is the action
    next_frame = agent.game_env.frames[next_state]

    # Calculate reward
    reward: int = agent.get_reward(next_frame)

    # Update Q-table
    agent.update_q_table(current_state, action, reward, next_state)

    # Update current state
    current_state = next_state
    done = all(not s for s in agent.game_env.emoji_frame_track.values())

    agent.game_env.app.processEvents()
    time.sleep(0.02)  # Add a delay to see the changes in the window

final_percentage_scratched = (agent.game_env.scratched_count / len(agent.game_env.frames)) * 100
print(f"Final scratched area: {final_percentage_scratched:.2f}%")


agent.finish_game()  # "app.quit" and "del app"
agent.game_env.app.exec()  # run app

# ----------------------------------
