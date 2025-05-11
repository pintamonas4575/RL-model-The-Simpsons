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
        self.gamma = 0.95  # Discount factor

        self.game_env = game_env
        self.num_states = self.game_env.total_squares # total states (cells); ej:585
        self.num_actions = 4
        self.q_table = np.zeros(shape=(self.num_states, self.num_actions)) # every cell; up, down, left, right; ej: 585x4

    def choose_action_up_down_left_right(self, cell_index: int, possible_actions: np.ndarray, epsilon: float) -> int:
        """Choose the next action between up, down, left, right based on epsilon-greedy policy."""
        # me pasan los índices de las celdas vecinas (arriba, abajo, izquierda, derecha)
        # ej: np.array([-1, 9, 3, -1]); tengo que coger las que no sean -1
        # y mirar en la q-table los valores de las acciones que corresponden a las celdas válidas
        possible_valid_actions = np.where(possible_actions != -1)[0] # índices de acciones válidas; ej: [1,3], sólo puedo ir abajo y derecha

        if random.random() < epsilon:
            action_index = random.choice(possible_valid_actions)
        else:
            q_values = self.q_table[cell_index, possible_valid_actions]
            action_index = np.argmax(q_values)  # choose the action with the highest q-value among the valid neighbors
        return action_index

    def update_q_table(self, cell_index: int, move_action_index: int, reward: int, next_cell_index: int) -> None:
        """Update q-table value based on Bellman´s equation."""
        # Bellman equation: Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))

        next_cell_neighbors, _, _ = self.game_env.env_step(next_cell_index, only_neighbors=True)  # 4 neighbors of the next cell
        next_cell_valid_actions = np.where(next_cell_neighbors != -1)[0] # valid neighbors of the next cell

        self.q_table[cell_index, move_action_index] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_cell_index, next_cell_valid_actions]) - self.q_table[cell_index, move_action_index])

    def finish_game(self) -> None:
        QTimer.singleShot(0, self.game_env.close_button.click)

"""**********************************************************"""
my_env = Scratch_Game_Environment4(frame_size=40, scratching_area=(110,98,770,300))
agent = RL_Agent_41(game_env=my_env)

EPISODES = 100
trace = 10
rewards, max_rewards = [], []
actions_done, min_actions_done = [], []
areas_scratched, min_areas_scratched = [], []
max_reward = -99999
min_actions = 99999
min_area_scratched = 999
path_to_save = f"V4_version/V4_1_Qtable_{agent.game_env.total_squares}_{EPISODES}"

epsilon = 0.9
# epsilon = 0.5
# epsilon = 0.0

start = time.time()
for i in range(EPISODES):

    agent.game_env.env_reset()

    done = False
    episode_actions = 0
    episode_reward = 0

    # Update epsilon at the beginning of the episode using exponential decay.
    epsilon *= np.exp(-0.001 * i)

    # indice_celda_actual = random.randint(0, agent.num_states - 1) # first cell of the episode
    indice_celda_actual = 0 # always start on the top left corner
    # indice_celda_actual = agent.num_states // 2 # start in the middle of the grid

    while not done:
        fila_actual = agent.q_table[indice_celda_actual, :]
        indices_celdas_vecinas, reward, done = agent.game_env.env_step(cell_index=indice_celda_actual)
        move_action_index = agent.choose_action_up_down_left_right(cell_index=indice_celda_actual, possible_actions=fila_actual, epsilon=epsilon) # devuelve [0,1,2,3]
        indice_proxima_celda = indices_celdas_vecinas[move_action_index]  # el índice de la celda a la que me muevo (0-N-1)
        fila_siguiente = agent.q_table[indice_proxima_celda, :]
        agent.update_q_table(cell_index=indice_celda_actual, move_action_index=move_action_index, reward=reward, next_cell_index=indice_proxima_celda)

        episode_actions += 1
        episode_reward += reward
        indice_celda_actual = indice_proxima_celda
        fila_actual = fila_siguiente

        # agent.game_env.app.processEvents()
        # time.sleep(0.02)  # delay to see the changes in the window

    episode_percentage = (agent.game_env.scratched_count / agent.game_env.total_squares) * 100

    max_reward = episode_reward if episode_reward > max_reward else max_reward
    min_actions = episode_actions if episode_actions < min_actions else min_actions
    min_area_scratched = episode_percentage if episode_percentage < min_area_scratched else min_area_scratched

    if i % trace == 0 or i == EPISODES-1:
        print(f"-----------------EPISODE {i}---------------------")
        print(f"Actions done: {episode_actions}")
        print(f"Reward: {episode_reward}")
        print(f"Final scratched area: {episode_percentage:.2f}%")
        # print(f"Frames removed: {agent.game_env.scratched_count}")

        agent.game_env.app.processEvents()
        agent.game_env.get_window_image_and_save(True, f"episodes/V4_1_episode_{i}.png")
    
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
    # agent.game_env.app.exec() # run app

print("*" * 50)
print(f"Total game frames: {agent.game_env.total_squares}")
print(f"Max reward: {max_reward}")
print(f"Min actions done: {min_actions}")
print(f"Min scratched area: {min_area_scratched:.2f}%")

minutes, seconds = divmod(time.time()-start, 60)
print(f"***** Total training time: {int(minutes)} minutes and {seconds:.2f} seconds *****")

"""**********************************************************"""

np.savetxt(f"results/{path_to_save}.txt", agent.q_table) # Save Q-table

# always saves in "results" folder
plot_results(rewards, actions_done, areas_scratched,
             max_rewards, min_actions_done, min_areas_scratched,
             f"{path_to_save}.png", time_taken=(int(minutes), seconds))
