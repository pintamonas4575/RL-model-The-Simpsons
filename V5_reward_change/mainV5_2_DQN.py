import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
from PIL import Image

from environmentV5_app import Scratch_Game_Environment5_Streamlit
from utils.functionalities import plot_results

device = "cpu"

class Custom_DQN(nn.Module):
    """Custom Deep Q-Network (DQN) model."""
    def __init__(self, input_dim: int, output_dim: int):
        super(Custom_DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
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

    def __init__(self, num_actions: int, agent_parameters: dict[str, int | float]) -> None:
        self.alpha = agent_parameters["alpha"]  # Learning rate
        self.gamma = agent_parameters["gamma"]  # Discount factor
        self.epsilon = agent_parameters["epsilon"]  # Exploration rate
        self.epsilon_decay = agent_parameters["epsilon_decay"]
        self.epsilon_min = agent_parameters["epsilon_min"]
        self.batch_size = agent_parameters["batch_size"]
        self.memory = deque(maxlen=15000)

        self.num_actions = num_actions
        self.target_dqn = Custom_DQN(self.num_actions, self.num_actions).to(device)
        self.policy_dqn = Custom_DQN(self.num_actions, self.num_actions).to(device)
        self.optimizer = optim.Adam(self.policy_dqn.parameters(), lr=self.alpha)
        self.loss_fn = nn.SmoothL1Loss()

    def remember(self, current_state: list[int], action: int, next_state: list[int], reward: int, done: bool) -> None:
        """Store the experience in memory."""
        # NOTE: current_state and next_state are in raw mode
        if len(self.memory) >= self.memory.maxlen:
            n_samples_to_remove = self.batch_size
            random_samples = random.sample(self.memory, n_samples_to_remove)
            for sample in random_samples:
                self.memory.remove(sample)
        self.memory.append((current_state, action, next_state, reward, done))

    def choose_action(self, current_state: list[int]) -> int:
        """Choose the next action based on epsilon-greedy policy."""
        possible_actions = [i for i, val in enumerate(current_state) if val == -1]
        if random.random() < self.epsilon:
            action_index = random.choice(possible_actions)
        else:
            current_state = torch.FloatTensor(current_state).unsqueeze(0).to(device) # [batch, num_actions]
            with torch.no_grad():
                q_values: torch.Tensor = self.policy_dqn(current_state)
            q_values_np = q_values[0].cpu().numpy()
            masked_q_values = np.full_like(q_values_np, -np.inf)
            masked_q_values[possible_actions] = q_values_np[possible_actions]
            action_index = np.argmax(masked_q_values)

        return action_index
        
    def replay(self) -> None:
        """Train the DQN model using replay experiences."""
        if len(self.memory) < self.batch_size:
            return

        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        current_states, actions, next_states, rewards, dones = zip(*[self.memory[i] for i in batch])

        current_states = torch.tensor(current_states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        q_values = self.policy_dqn(current_states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_dqn(next_states).max(1)[0]
        target_q_values: torch.Tensor = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self) -> None:
        """Update the target network with the weights of the policy network."""
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

"""**********************************************************"""
agent_parameters = {
    "alpha": 0.001,  # Learning rate
    "gamma": 0.80,  # Discount factor
    "epsilon": 0.9,  # Exploration rate
    "epsilon_decay": 0.995,
    "epsilon_min": 0.05,
    "batch_size": 64
}

my_env = Scratch_Game_Environment5_Streamlit(frame_size=50, scratching_area=(110,98,770,300))
agent = RL_Agent_52(num_actions=my_env.total_squares, agent_parameters=agent_parameters)

EPISODES = 500
trace = 50
epsilon_history = []
rewards, max_rewards = [], []
actions_done, min_actions_done = [], []
areas_scratched, min_areas_scratched = [], []
max_reward, min_actions, min_area_scratched = -99999, 99999, 999
path_to_save = f"V5_version/V5_2_DQN_{agent.num_actions}_{EPISODES}"
step_counter = 0

start = time.time()
for i in range(EPISODES):
    my_env.env_reset()

    done = False
    episode_actions = 0
    episode_reward = 0

    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
    epsilon_history.append(agent.epsilon)

    # current_state = my_env.frames_mask
    current_state = my_env.frames_mask.copy()

    while not done:
        episode_actions += 1
        step_counter += 1

        action_index = agent.choose_action(current_state)
        next_state, reward, done = my_env.env_step(action_index)
        agent.remember(current_state, action_index, next_state, reward, done)
        agent.replay()

        episode_reward += reward
        current_state = next_state.copy()
        # current_state = next_state

        if step_counter % (agent.num_actions//2) == 0:
            agent.update_target_network()

    episode_percentage = (my_env.scratched_count / my_env.total_squares) * 100

    max_reward = episode_reward if episode_reward > max_reward else max_reward
    min_actions = episode_actions if episode_actions < min_actions else min_actions
    min_area_scratched = episode_percentage if episode_percentage < min_area_scratched else min_area_scratched

    if i % trace == 0 or i == EPISODES-1:
        print(f"-----------------EPISODE {i+1}---------------------")
        print(f"Episode reward: {episode_reward}")
        print(f"Episode actions done: {episode_actions}")
        print(f"Episode percentage: {episode_percentage:.2f}%")
        image: Image.Image = my_env.get_window_image()
        image.save(f"episodes/V5_2_episode_{i}.png")
    
    # ---------------data for graphics----------------
    rewards.append(episode_reward)
    actions_done.append(episode_actions)
    areas_scratched.append(episode_percentage)
    max_rewards.append(max_reward)
    min_actions_done.append(min_actions)
    min_areas_scratched.append(min_area_scratched)
    # ---------------data for graphics----------------

minutes, seconds = divmod(time.time()-start, 60)
print(f"***** Total training time: {int(minutes)} minutes and {seconds:.2f} seconds *****")

"""**********************************************************"""

torch.save(agent.policy_dqn.state_dict(), f"results/V5_version/V5_2_policy_{agent.num_actions}_{EPISODES}.pth")
torch.save(agent.target_dqn.state_dict(), f"results/V5_version/V5_2_target_{agent.num_actions}_{EPISODES}.pth")

# always saves in "results" folder
plot_results(rewards, actions_done, areas_scratched,
             max_rewards, min_actions_done, min_areas_scratched,
             f"{path_to_save}.png", time_taken=(int(minutes), seconds))

from utils.functionalities import plot_epsilon_history
plot_epsilon_history(epsilon_history, f"results/{path_to_save}_epsilon.png")
