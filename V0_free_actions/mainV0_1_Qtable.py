import random
import time
import gc
import numpy as np

from environment import Scratch_Game_Environment # V0_free_actions/environment.py
from utils.functionalities import plot_results
from agentV0_1_Qtable import RL_Agent_01

"""**********************************************************"""
my_env = Scratch_Game_Environment(frame_size=20, scratching_area=(110,98,770,300), num_emojis=3)
agent = RL_Agent_01(game_env=my_env)

EPISODES = 1000
trace = 250
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

np.savetxt(f"results/V0_version/V0_1_Qtable_{EPISODES}.txt", agent.q_table) # Save Q-table

# always saves in "results" folder
plot_results(rewards, actions_done, areas_scratched,
             max_rewards, min_actions_done, min_areas_scratched,
             f"V0_version/V0_1_Qtable_{EPISODES}.png", time_taken=(int(minutes), seconds))
