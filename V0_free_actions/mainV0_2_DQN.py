import random
import time
import gc
import torch

from environment import Scratch_Game_Environment # V0_free_actions/environment.py
from utils.functionalities import plot_results
from agentV0_2_DQN import RL_Agent_02
from agentV0_2_DQN_change import RL_Agent_02_change

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

"""**********************************************************"""
my_env = Scratch_Game_Environment(frame_size=20, scratching_area=(110,98,770,300), num_emojis=3)
layer_dims = [my_env.total_squares,512,512,1024,512,512,my_env.total_squares]
agent = RL_Agent_02(game_env=my_env, nn_arquitecture=layer_dims)
# agent = RL_Agent_02_change(game_env=my_env, nn_arquitecture=layer_dims)

agent.policy_dqn.print_num_parameters()
agent.target_dqn.print_num_parameters()

EPISODES = 10
trace = 2
rewards = []
max_rewards = []
actions_done = []
min_actions_done = []
areas_scratched = []
min_areas_scratched = []
max_reward = -9999999
min_actions = 9999999
min_area_scratched = 100 # % of the area

start = time.time()
for i in range(EPISODES):
    agent.game_env.reset_env()
    agent.reset_agent()

    done = False
    episode_actions = 0
    episode_reward = 0

    current_state = random.randint(0, agent.num_states - 1)  # Start at a random state
    while not done:
        episode_actions += 1

        action = agent.choose_action(current_state)
        next_state = action  # Since the actions are cells, the destiny is the action
        next_frame = agent.game_env.squares[next_state]

        reward = agent.get_reward(next_frame)
        episode_reward += reward
        agent.remember(current_state, action, reward, next_state, done)

        current_state = next_state
        agent.replay(agent.batch_size)
        done = all(not s for s in agent.game_env.emoji_frame_track.values())

        if episode_actions % 1000 == 0:
            agent.update_target_network()

    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

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
print(f"****Total trining time: {int(minutes)} minutes y {seconds:.2f} seconds****")

"""**********************************************************"""

# save models
torch.save(agent.policy_dqn.state_dict(), f"results/models/V0_2_policy_DQN_{EPISODES}.pt")
torch.save(agent.target_dqn.state_dict(), f"results/models/V0_2_target_DQN_{EPISODES}.pt")

# always saves in "results" folder
plot_results(rewards, actions_done, areas_scratched,
             max_rewards, min_actions_done, min_areas_scratched,
             f"V0_version/V0_2_DQN_{EPISODES}.png", time_taken=(int(minutes), seconds))
