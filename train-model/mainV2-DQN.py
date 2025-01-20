import random
import time
import matplotlib.pyplot as plt
import gc
import torch
from torch import nn
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFrame
from collections import deque

from environment import Scratch_Game_Environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Custom_DQN(nn.Module):
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

    def print_num_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {total_params}")

class RL_Agent2():

    def __init__(self, game_env: Scratch_Game_Environment, nn_arquitecture: list): 
        self.game_env = game_env
        self.global_reward = 0
        self.visited_frames = set()
        self.num_states = self.game_env.total_squares

        self.policy_dqn = Custom_DQN(nn_arquitecture).to(device)
        self.target_dqn = Custom_DQN(nn_arquitecture).to(device)
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()
        self.batch_size = 4
        self.gamma = 0.99
        self.epsilon = 0.15
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=10000)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    # def choose_action(self, state):
    #     if random.random() < self.epsilon:
    #         return random.randint(0, self.num_states - 1) # num states = possible actions
    #     state = torch.FloatTensor(state).to(device)
    #     # with torch.no_grad():
    #         # q_values = self.policy_dqn(state)
    #     q_values = self.policy_dqn(state)
    #     return torch.argmax(q_values).item()
    
    def choose_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.num_states - 1)  # Exploración
        # One-hot encoding
        one_hot_state = torch.zeros(self.num_states, dtype=torch.float32).to(device)
        one_hot_state[state] = 1.0
        q_values = self.policy_dqn(one_hot_state)
        return torch.argmax(q_values).item()

    def get_reward(self, frame: QFrame):
        if frame in self.visited_frames:
            reward = -50  # Penalty for revisiting
        else:
            self.visited_frames.add(frame)
            response = self.game_env.remove_square(frame)
            if response: # red frame
                reward = 100
            else: # blue frame
                reward = -10

        self.global_reward += reward
        return reward
    
    def replay(self, batch_size: int):
        if len(self.memory) < batch_size:
            print("Replay passed")
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        one_hot_states = torch.zeros((batch_size, self.num_states), dtype=torch.float32).to(device)
        # print("one_hot_states 1:", one_hot_states.shape) # type: torch.Tensor

        for idx, state in enumerate(states):
            state = int(state.item())
            one_hot_states[idx, state] = 1.0

        # print("one_hot_states 2:", one_hot_states.shape) # type: torch.Tensor

        # q_values = self.policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_values = self.policy_dqn(one_hot_states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_dqn(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

    def finish_game(self) -> None:
        QTimer.singleShot(0, self.game_env.close_button.click)
    
    def reset_agent(self):
        self.global_reward = 0
        self.visited_frames.clear()

"""---------------------------------------------------------"""
"""---------------------------------------------------------"""
my_env = Scratch_Game_Environment(frame_size=20, scratching_area=(110,98,770,300), num_emojis=3)
layer_dims = [my_env.total_squares,256,256,my_env.total_squares]
agent = RL_Agent2(game_env=my_env, nn_arquitecture=layer_dims)

# agent.policy_dqn.print_num_parameters()
# agent.target_dqn.print_num_parameters()

EPISODES = 1000
trace = 100
rewards = []
max_rewards = []
actions_done = []
min_actions_done_list = []
areas_scratched = []
min_areas_scratched = []
max_reward = -9999999
min_actions_done = 9999999
min_area_scratched = 999

start = time.time()
for i in range(EPISODES):
    agent.game_env.reset_env()
    agent.reset_agent()

    done = False
    num_actions_done = 0

    current_state = random.randint(0, agent.num_states - 1)  # Start at a random state
    while not done:
        num_actions_done += 1

        action = agent.choose_action(current_state)
        next_state = action  # Since the actions are cells, the destiny is the action
        next_frame = agent.game_env.squares[next_state]

        reward = agent.get_reward(next_frame)
        agent.remember(current_state, action, reward, next_state, done)

        current_state = next_state
        done = all(not s for s in agent.game_env.emoji_frame_track.values())

        agent.replay(agent.batch_size)

    agent.update_target_network()

    max_reward = agent.global_reward if agent.global_reward > max_reward else max_reward
    min_actions_done = num_actions_done if num_actions_done < min_actions_done else min_actions_done

    final_percentage_scratched = (agent.game_env.scratched_count / agent.game_env.total_squares) * 100
    min_area_scratched = final_percentage_scratched if final_percentage_scratched < min_area_scratched else min_area_scratched

    if i % trace == 0 or i == EPISODES-1:
        print(f"-----------------EPOCH {i + 1}---------------------")
        print("Actions done:", num_actions_done)
        print("reward:", agent.global_reward)
        print("max reward:", max_reward)
        print("min actions done:", min_actions_done)
        print(f"Final scratched area: {final_percentage_scratched:.2f}%")

        agent.game_env.app.processEvents()
        agent.game_env.get_window_image_and_save(True, f"episodes/episode_{i + 1}.png")

    rewards.append(agent.global_reward)
    max_rewards.append(max_reward)
    actions_done.append(num_actions_done)
    min_actions_done_list.append(min_actions_done)
    areas_scratched.append(final_percentage_scratched)
    min_areas_scratched.append(min_area_scratched)

    agent.finish_game()
    gc.collect()
    agent.game_env.app.exec()

minutos, segundos = divmod(time.time() - start, 60)
print(f"****Tiempo total: {int(minutos)} minutos y {segundos:.2f} segundos****")

torch.save(agent.policy_dqn.state_dict(), f"results/policy_dqn-{EPISODES}.pt")
torch.save(agent.target_dqn.state_dict(), f"results/target_dqn-{EPISODES}.pt")

plt.figure(figsize=(18, 12))
plt.subplot(3, 1, 1)
plt.plot(rewards, label='Rewards')
plt.plot(max_rewards, label='Max Rewards')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Rewards vs Episodes')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(actions_done, label='Actions Done')
plt.plot(min_actions_done_list, label='Min Actions Done')
plt.xlabel('Episodes')
plt.ylabel('Actions Done')
plt.title('Actions Done vs Episodes')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(areas_scratched, label='Area Scratched')
plt.plot(min_areas_scratched, label='Min Area Scratched')
plt.xlabel('Episodes')
plt.ylabel('Area Scratched')
plt.title('Area Scratched vs Episodes')
plt.legend()

plt.tight_layout()
plt.savefig(f"episodes/plot2-DQN-{EPISODES}.png")
plt.close()
