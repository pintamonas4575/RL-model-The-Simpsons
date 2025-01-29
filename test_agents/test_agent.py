import torch
from V0_free_actions.environment import Scratch_Game_Environment

# Load the networks from .pt files
policy_network = torch.load('results/models/policy_dqn_30.pt')
target_network = torch.load('results/models/target_dqn_30.pt')

env = Scratch_Game_Environment() 

# Test the agent in the environment
obs = env.reset()
done = False
while not done:
    action, _states = policy_network.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

env.close()