import random
import time
import gc
import numpy as np
from PyQt5.QtCore import QTimer

from environmentV4 import Scratch_Game_Environment4
from utils.functionalities import plot_results

"""**********************************************************"""

class RL_Agent_41():
    """Reinforcement Learning Agent."""

    def __init__(self, game_env: Scratch_Game_Environment4):
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor

        self.game_env = game_env
        self.num_states = self.game_env.total_squares # total states (cells); ej:585
        self.num_actions = 4
        self.q_table = np.zeros(shape=(self.num_states, self.num_actions)) # every cell; up, down, left, right; ej: 585x4

    def choose_action(self, state: np.ndarray[int], epsilon: float) -> int:
        """Choose the next action based on epsilon-greedy policy."""

        if random.random() < epsilon:
            action_index = random.choice(state)
        else:
            # tengo que coger cada posible acción y ver cual es la que tiene el valor más alto en la tabla Q
            q_values = agent.q_table[current_action, :]
            action_index = state[np.argmax(q_values)]
        return action_index

    def update_q_table(self, current_state: int, action: int, reward: int, next_state: int) -> None:
        """Update q-table value based on Bellman´s equation."""

        # self.q_table[current_state, action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state, :]) - self.q_table[current_state, action])
        self.q_table[action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state, :]) - self.q_table[action])

    def finish_game(self) -> None:
        QTimer.singleShot(0, self.game_env.close_button.click)

"""**********************************************************"""
my_env = Scratch_Game_Environment4(frame_size=40, scratching_area=(110,98,770,300))
agent = RL_Agent_41(game_env=my_env)

rewards, max_rewards = [], []
actions_done, min_actions_done = [], []
areas_scratched, min_areas_scratched = [], []
max_reward = -99999
min_actions = 99999
min_area_scratched = 999
path_to_save = f"V4_version/V4_1_Qtable_{my_env.total_squares}_1"

# epsilon = 0.9 
epsilon = 0.5
# epsilon = 0.0  
agent.game_env.window.show()
agent.game_env.env_reset()

done = False
episode_actions = 0
episode_reward = 0

# Update epsilon at the beginning of the episode using exponential decay.
epsilon *= np.exp(-0.001 * 1)

start = time.time()
while not done:
    if episode_actions == 0:
        current_action: int = random.randint(0, agent.num_states - 1)  # Start at a random cell
        current_state, _, _ = my_env.env_step(current_action)
    action_index = agent.choose_action(current_state, epsilon)
    next_state, reward, done = my_env.env_step(action_index)
    agent.update_q_table(current_action, action_index, reward, next_state)

    episode_actions += 1
    episode_reward += reward
    current_state = next_state

    agent.game_env.app.processEvents()
    time.sleep(0.02)  # Add a delay to see the changes in the window

episode_percentage = (agent.game_env.scratched_count / agent.game_env.total_squares) * 100

max_reward = episode_reward if episode_reward > max_reward else max_reward
min_actions = episode_actions if episode_actions < min_actions else min_actions
min_area_scratched = episode_percentage if episode_percentage < min_area_scratched else min_area_scratched

print(f"-----------------EPISODE 1---------------------")
print(f"Actions done: {episode_actions}")
print(f"Reward: {episode_reward}")
print(f"Final scratched area: {episode_percentage:.2f}%")
print(f"Min actions done: {min_actions}")
print(f"Max reward: {max_reward}")
print(f"Min scratched area: {min_area_scratched:.2f}%")

agent.game_env.app.processEvents()
agent.game_env.get_window_image_and_save(True, f"episodes/V4_1_episode_1.png")

# ---------------data for graphics----------------
rewards.append(episode_reward)
actions_done.append(episode_actions)
areas_scratched.append(episode_percentage)
max_rewards.append(max_reward)
min_actions_done.append(min_actions)
min_areas_scratched.append(min_area_scratched)
# ---------------data for graphics----------------

agent.finish_game() # "app.quit" and "del app"
# gc.collect()  # Explicitly run garbage collection to free resources
agent.game_env.app.exec() # run app

minutes, seconds = divmod(time.time()-start, 60)
print(f"****Total training time: {int(minutes)} minutes and {seconds:.2f} seconds****")
