import gc
import time
import numpy as np
import random

from environmentV5_app import Scratch_Game_Environment5_Streamlit
from utils.functionalities import plot_results

class RL_Agent_51():
    """Reinforcement Learning Agent."""

    def __init__(self, game_env: Scratch_Game_Environment5_Streamlit):
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

"""**********************************************************"""
my_env = Scratch_Game_Environment5_Streamlit(frame_size=50, scratching_area=(110,98,770,300))
agent = RL_Agent_51(game_env=my_env)

EPISODES = 500
trace = 100
rewards, max_rewards = [], []
actions_done, min_actions_done = [], []
areas_scratched, min_areas_scratched = [], []
max_reward, min_actions, min_area_scratched = -99999, 99999, 999
path_to_save = f"V5_version/V5_1_Qtable_{agent.game_env.total_squares}_{EPISODES}"

epsilon = 0.9
# epsilon = 0.5
# epsilon = 0.0

start = time.time()
for i in range(EPISODES):

    agent.game_env.env_reset()

    done = False
    episode_actions = 0
    episode_reward = 0

    current_state = agent.game_env.frames_mask.copy()
    current_action = agent.game_env.total_squares // 2

    epsilon *= np.exp(-0.001 * i) # exponential decay

    while not done:
        episode_actions += 1

        action_index = agent.choose_action(current_action, current_state, epsilon)
        next_state, reward, done = agent.game_env.env_step(action_index)
        agent.update_q_table(current_action, action_index, reward, next_state)

        episode_reward += reward
        current_state = next_state.copy()

    episode_percentage = (agent.game_env.scratched_count / agent.game_env.total_squares) * 100

    max_reward = episode_reward if episode_reward > max_reward else max_reward
    min_actions = episode_actions if episode_actions < min_actions else min_actions
    min_area_scratched = episode_percentage if episode_percentage < min_area_scratched else min_area_scratched

    if i % trace == 0 or i == EPISODES-1:
        print(f"-----------------EPISODE {i}---------------------")
        print(f"Actions done: {episode_actions}")
        print(f"Reward: {episode_reward}")
        print(f"Final scratched area: {episode_percentage:.2f}%")

    # ---------------data for graphics----------------
    rewards.append(episode_reward)
    actions_done.append(episode_actions)
    areas_scratched.append(episode_percentage)
    max_rewards.append(max_reward)
    min_actions_done.append(min_actions)
    min_areas_scratched.append(min_area_scratched)
    # ---------------data for graphics----------------

    gc.collect()  # Explicitly run garbage collection to free resources

print("*" * 50)
print(f"Max reward: {max_reward}")
print(f"Min actions done: {min_actions}")
print(f"Min scratched area: {min_area_scratched:.2f}%")

minutes, seconds = divmod(time.time()-start, 60)
print(f"***** Total training time: {int(minutes)} minutes and {seconds:.2f} seconds *****")

"""**********************************************************"""

# always saves in "results" folder
plot_results(rewards, actions_done, areas_scratched,
             max_rewards, min_actions_done, min_areas_scratched,
             f"{path_to_save}.png", time_taken=(int(minutes), seconds))