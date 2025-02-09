import random
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFrame
from collections import deque

from environment import Scratch_Game_Environment # V0_free_actions/environment.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Custom_DQN(nn.Module):
    """Neural Network"""
    def __init__(self, layer_dims: list):
        super(Custom_DQN, self).__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.extend([
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def print_num_parameters(self) -> None:
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {total_params}")

    def print_gpu_memory_usage(self) -> None:
        if torch.cuda.is_available():
            allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
            reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 2)
            print(f"Allocated GPU memory: {allocated_memory:.2f} MB")
            print(f"Reserved GPU memory: {reserved_memory:.2f} MB")
        else:
            print("CUDA is not available.")

class ReplayMemory:
    """Memory replay for batch training"""
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
        self.indices = np.arange(capacity)  # For potential prioritization

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

        self.epsilon = 0.8
        self.epsilon_decay = 0.996
        self.epsilon_min = 0.08
        self.gamma = 0.99
        self.batch_size = 64
        self.learning_rate = 0.0001
        self.memory = ReplayMemory(capacity=100000)
        self.policy_dqn = Custom_DQN(nn_arquitecture).to(device)
        self.target_dqn = Custom_DQN(nn_arquitecture).to(device)
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate)
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.SmoothL1Loss()

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.add(state, action, reward, next_state, done)
    
    def choose_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.num_states - 1)  # Exploración
        one_hot_state = torch.zeros(self.num_states, dtype=torch.float32).to(device) # One-hot encoding
        one_hot_state[state] = 1.0
        q_values = self.policy_dqn(one_hot_state.unsqueeze(0)) # to convert to (1, num_states)
        return torch.argmax(q_values).item()

    def get_reward(self, frame: QFrame) -> int:
        """Execute the action and return the reward"""
        if frame in self.visited_frames:
            reward = -10  # Penalty for revisiting
        else:
            self.visited_frames.add(frame)
            response = self.game_env.remove_square(frame)
            if response: # red frame
                reward = 100
            else: # blue frame
                reward = -1
        return reward
    
    def replay(self, batch_size: int) -> None:
        if len(self.memory) < batch_size: # Not enough samples to batch-train
            return
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.LongTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.LongTensor(next_states).to(device)
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
        # q_values = self.policy_dqn.forward(one_hot_states).gather(1, actions)
        next_q_values = self.target_dqn.forward(one_hot_next_states).max(1)[0]
        # expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        expected_q_values = rewards + self.gamma * next_q_values * (1 - self.epsilon)

        # expected_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values.unsqueeze(1) * (1 - dones.unsqueeze(1))

        loss = self.loss_fn.forward(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_dqn.parameters():
        #     param.grad.data.clamp_(-1, 1) # to avoid exploding gradients
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

    def finish_game(self) -> None:
        QTimer.singleShot(0, self.game_env.close_button.click)
    
    def reset_agent(self):
        self.visited_frames.clear()