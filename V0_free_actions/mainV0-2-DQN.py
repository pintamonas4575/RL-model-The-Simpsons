import random
import time
import matplotlib.pyplot as plt
import gc
import torch
from torch import nn
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFrame
from collections import deque

from environment import Scratch_Game_Environment # V0_free_actions/environment.py
from utils.functionalities import plot_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Custom_DQN(nn.Module):
    """Neural Network"""
    def __init__(self, layer_dims: list):
        super(Custom_DQN, self).__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:  # No ReLU on the output layer
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def print_num_parameters(self) -> None:
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {total_params}")

class ReplayMemory:
    """Memory replay for batch training"""
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.memory)
    
    def add(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.memory, batch_size)

class RL_Agent_02():
    """Reinforcement Learning Agent"""
    def __init__(self, game_env: Scratch_Game_Environment, nn_arquitecture: list): 
        self.game_env = game_env
        self.visited_frames = set()
        self.num_states = self.game_env.total_squares

        self.epsilon = 1.0
        self.epsilon_decay = 0.996
        self.epsilon_min = 0.08
        self.gamma = 0.99
        self.batch_size = 16
        self.learning_rate = 0.0001
        self.memory = ReplayMemory(capacity=100000)
        self.policy_dqn = Custom_DQN(nn_arquitecture).to(device)
        self.target_dqn = Custom_DQN(nn_arquitecture).to(device)
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.add(state, action, reward, next_state, done)
    
    def choose_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.num_states - 1)  # Exploración
        one_hot_state = torch.zeros(self.num_states, dtype=torch.float32).to(device) # One-hot encoding
        one_hot_state[state] = 1.0
        q_values = self.policy_dqn(one_hot_state)
        return torch.argmax(q_values).item()

    def get_reward(self, frame: QFrame) -> int:
        if frame in self.visited_frames:
            reward = -50  # Penalty for revisiting
        else:
            self.visited_frames.add(frame)
            response = self.game_env.remove_square(frame)
            if response: # red frame
                reward = 100
            else: # blue frame
                reward = -10
        return reward
    
    def replay(self, batch_size: int) -> None:
        if len(self.memory) < batch_size: # Not enough samples to train
            return
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        one_hot_states = torch.zeros((batch_size, self.num_states), dtype=torch.float32).to(device)
        for idx, state in enumerate(states):
            state = int(state.item())
            one_hot_states[idx, state] = 1.0

        one_hot_next_states = torch.zeros((batch_size, self.num_states), dtype=torch.float32).to(device)
        for idx, state in enumerate(next_states):
            state = int(state.item())
            one_hot_next_states[idx, state] = 1.0

        q_values = self.policy_dqn.forward(one_hot_states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_dqn.forward(one_hot_next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn.forward(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

    def finish_game(self) -> None:
        QTimer.singleShot(0, self.game_env.close_button.click)
    
    def reset_agent(self):
        self.visited_frames.clear()

"""---------------------------------------------------------"""
"""---------------------------------------------------------"""
my_env = Scratch_Game_Environment(frame_size=20, scratching_area=(110,98,770,300), num_emojis=3)
layer_dims = [my_env.total_squares,512,256,512,my_env.total_squares]
agent = RL_Agent_02(game_env=my_env, nn_arquitecture=layer_dims)

# agent.policy_dqn.print_num_parameters()
# agent.target_dqn.print_num_parameters()

EPISODES = 30
trace = 2
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
    num_actions_done = 0
    episode_reward = 0

    current_state = random.randint(0, agent.num_states - 1)  # Start at a random state
    while not done:
        num_actions_done += 1
        # if num_actions_done % 500 == 0:
        #     print(f"Actions done: {num_actions_done}", end='\r')

        action = agent.choose_action(current_state)
        next_state = action  # Since the actions are cells, the destiny is the action
        next_frame = agent.game_env.squares[next_state]

        reward = agent.get_reward(next_frame)
        episode_reward += reward
        agent.remember(current_state, action, reward, next_state, done)

        current_state = next_state
        done = all(not s for s in agent.game_env.emoji_frame_track.values())

        agent.replay(agent.batch_size)

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    agent.update_target_network()

    max_reward = episode_reward if episode_reward > max_reward else max_reward
    min_actions = num_actions_done if num_actions_done < min_actions else min_actions

    final_percentage_scratched = (agent.game_env.scratched_count / agent.game_env.total_squares) * 100
    min_area_scratched = final_percentage_scratched if final_percentage_scratched < min_area_scratched else min_area_scratched

    if i % trace == 0 or i == EPISODES-1:
        print(f"-----------------EPOCH {i + 1}---------------------")
        print(f"Actions done: {num_actions_done}")
        print(f"Reward: {episode_reward}")
        print(f"Final scratched area: {final_percentage_scratched:.2f}%")
        print(f"Max reward: {max_reward}")
        print(f"Min actions done: {min_actions}")
        print(f"Min scratched area: {min_area_scratched:.2f}%")

        agent.game_env.app.processEvents()
        agent.game_env.get_window_image_and_save(True, f"episodes/episode_{i + 1}.png")

    # -------------Save data for plotting----------------
    rewards.append(episode_reward)
    max_rewards.append(max_reward)
    actions_done.append(num_actions_done)
    min_actions_done.append(min_actions)
    areas_scratched.append(final_percentage_scratched)
    min_areas_scratched.append(min_area_scratched)
    # -------------Save data for plotting----------------

    agent.finish_game()
    gc.collect()
    agent.game_env.app.exec()

minutes, seconds = divmod(time.time()-start, 60)
print(f"****Total trining time: {int(minutes)} minutes y {seconds:.2f} seconds****")

torch.save(agent.policy_dqn.state_dict(), f"results/models/policy_dqn_{EPISODES}.pt")
torch.save(agent.target_dqn.state_dict(), f"results//models/target_dqn_{EPISODES}.pt")

"""**********************************************************"""
"""**********************************************************"""

plot_results(rewards, actions_done, areas_scratched,
             max_rewards, min_actions_done, min_areas_scratched,
             EPISODES, f"results/V0_2_DQN_{EPISODES}.png")