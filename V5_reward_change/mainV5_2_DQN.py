import gc
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from PyQt5.QtCore import QTimer

from environmentV5 import Scratch_Game_Environment5
from utils.functionalities import plot_results

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print("device:", device)

class Custom_DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(Custom_DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def print_num_parameters(self) -> None:
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {total_params}")

    def print_gpu_memory_usage(self) -> None:
        if device == "cuda":
            allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
            reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 2)
            print(f"Allocated GPU memory: {allocated_memory:.2f} MB")
            print(f"Reserved GPU memory: {reserved_memory:.2f} MB")
        else:
            print("CUDA not available or not being used")

class RL_Agent_52():
    """Reinforcement Learning Agent using DQN."""

    def __init__(self, game_env: Scratch_Game_Environment5):
        self.alpha = 0.001  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 128
        self.memory = deque(maxlen=100000)

        self.game_env = game_env
        self.num_states = self.game_env.total_squares # total states (cells); ej:585

        # Initialize the DQN model
        self.target_dqn = Custom_DQN(self.num_states, self.num_states).to(device)
        self.policy_dqn = Custom_DQN(self.num_states, self.num_states).to(device)
        self.optimizer = optim.Adam(self.policy_dqn.parameters(), lr=self.alpha)
        self.loss_fn = nn.SmoothL1Loss()

    def remember(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """Store the experience in memory."""
        if len(self.memory) >= self.memory.maxlen:
            random_sample = random.sample(self.memory, 1)
            self.memory.remove(random_sample[0])
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, current_action: int, state: list[int]) -> int:
        """Choose the next action based on epsilon-greedy policy."""

        possible_actions = [i for i, val in enumerate(state) if val == -1]
        if random.random() < self.epsilon:
            action_index = random.choice(possible_actions)
        else:
            state_tensor = torch.eye(self.num_states, dtype=torch.float32)[current_action].to(device).cpu().detach()
            q_values = self.policy_dqn(state_tensor)
            action_index = torch.argmax(q_values).item()

        return action_index
        
    def replay(self) -> None:
        """Train the DQN model using experience replay."""
        if len(self.memory) < self.batch_size:
            return

        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])

        # states = torch.tensor(states, dtype=torch.float32).to(device)
        # actions = torch.tensor(actions).to(device)
        # rewards = torch.tensor(rewards).to(device)
        # next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        # dones = torch.tensor(dones).to(device)

        # Convert integer states to one-hot encoded vectors to match the input dimensions of the network
        states = torch.eye(self.num_states, dtype=torch.float32)[list(states)].to(device)
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards).to(device)
        next_states = torch.eye(self.num_states, dtype=torch.float32)[list(next_states)].to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        q_values = self.policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_dqn(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self) -> None:
        """Update the target network with the weights of the policy network."""
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

    def finish_game(self) -> None:
        """Finish the game and close the environment."""
        QTimer.singleShot(0, self.game_env.close_button.click)

"""**********************************************************"""
my_env = Scratch_Game_Environment5(frame_size=50, scratching_area=(110,98,770,300))
agent = RL_Agent_52(game_env=my_env)

EPISODES = 1
trace = 1
rewards, max_rewards = [], []
actions_done, min_actions_done = [], []
areas_scratched, min_areas_scratched = [], []
max_reward, min_actions, min_area_scratched = -99999, 99999, 999
path_to_save = f"V5_version/V5_2_DQN_{agent.game_env.total_squares}_{EPISODES}"
step_counter = 0

start = time.time()
for i in range(EPISODES):
    agent.game_env.env_reset()

    done = False
    episode_actions = 0
    episode_reward = 0

    # Update epsilon at the beginning of the episode using exponential decay.
    # agent.epsilon *= np.exp(-0.001 * i)
    agent.epsilon = 0.4

    current_state = agent.game_env.frames_mask
    # indice_celda_actual = 0 # always start on the top left corner
    # indice_celda_actual = random.randint(0, agent.num_states - 1) # first cell of the episode
    indice_celda_actual = agent.num_states // 2 # start in the middle of the grid

    while not done:
        episode_actions += 1
        step_counter += 1

        print("indice_celda_actual:", indice_celda_actual, step_counter)

        action_index = agent.choose_action(indice_celda_actual, current_state)
        next_state, reward, done = agent.game_env.env_step(action_index=indice_celda_actual)
        indice_proxima_celda = next_state[action_index]
        agent.remember(indice_celda_actual, action_index, reward, indice_celda_actual, done)
        agent.replay()

        indice_celda_actual = indice_proxima_celda
        episode_reward += reward

        if step_counter % 1000 == 0:
            agent.update_target_network()
            print(f"Actions done: {episode_actions}")

    episode_percentage = (agent.game_env.scratched_count / agent.game_env.total_squares) * 100

    max_reward = episode_reward if episode_reward > max_reward else max_reward
    min_actions = episode_actions if episode_actions < min_actions else min_actions
    min_area_scratched = episode_percentage if episode_percentage < min_area_scratched else min_area_scratched

    if i % trace == 0 or i == EPISODES-1:
        print(f"-----------------EPISODE {i+1}---------------------")
        print(f"Episode reward: {episode_reward}")
        print(f"Episode actions: {episode_actions}")
        print(f"Episode percentage: {episode_percentage:.2f}%")
        agent.game_env.app.processEvents()
        agent.game_env.get_window_image_and_save(True, f"episodes/V5_2_episode_{i}.png")
    
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

minutes, seconds = divmod(time.time()-start, 60)
print(f"***** Total training time: {int(minutes)} minutes and {seconds:.2f} seconds *****")

"""**********************************************************"""

torch.save(agent.policy_dqn.state_dict(), f"results/V5_version/policy_{agent.game_env.total_squares}_{EPISODES}.pth")
torch.save(agent.target_dqn.state_dict(), f"results/V5_version/target_{agent.game_env.total_squares}_{EPISODES}.pth")

# always saves in "results" folder
plot_results(rewards, actions_done, areas_scratched,
             max_rewards, min_actions_done, min_areas_scratched,
             f"{path_to_save}.png", time_taken=(int(minutes), seconds))

