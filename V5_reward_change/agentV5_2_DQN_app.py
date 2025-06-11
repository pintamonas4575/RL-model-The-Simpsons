import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque

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
    
    def get_num_parameters(self) -> int:
        total = sum(p.numel() for p in self.parameters())
        return f"{total:,}".replace(",", ".")
class RL_Agent_52():
    """Reinforcement Learning Agent using DQN."""

    def __init__(self, num_actions: int, agent_parameters: dict[str, int | float]) -> None:
        self.alpha = agent_parameters["alpha"]  # Learning rate
        self.gamma = agent_parameters["gamma"]  # Discount factor
        self.epsilon = agent_parameters["epsilon"]  # Exploration rate
        self.batch_size = agent_parameters["batch_size"]
        self.memory = deque(maxlen=agent_parameters["memory_size"])

        self.num_actions = num_actions
        self.target_dqn = Custom_DQN(self.num_actions, self.num_actions).to(device)
        self.policy_dqn = Custom_DQN(self.num_actions, self.num_actions).to(device)
        self.optimizer = optim.Adam(self.policy_dqn.parameters(), lr=self.alpha)
        self.loss_fn = nn.SmoothL1Loss()

    def remember(self, current_state: list[int], action: int, next_state: list[int], reward: int, done: bool) -> None:
        """Store the experience in memory. If memory is full, remove random samples."""
        # NOTE: current_state and next_state are in raw mode
        if len(self.memory) >= self.memory.maxlen:
            random_samples = random.sample(self.memory, k=self.batch_size)
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