import sys
import random
import io
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QBuffer, QRect, QTimer
from PyQt5.QtGui import QPixmap, QImage, QBrush, QPalette
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QWidget, QPushButton, QVBoxLayout, QFrame

from environment import Scratch_Game_Environment

# # RL Training Loop
# def train_rl_agent():
#     env = MyEnvironment(app, window, window.label, window.button)
#     num_episodes = 10

#     for episode in range(num_episodes):
#         print(f"Episode {episode + 1}")
#         state = env.reset()
#         done = False

#         while not done:
#             action = random.choice([0, 1])  # Randomly choose an action (0: do nothing, 1: click)
#             next_state, reward, done = env.act(action)
#             print(f"Action: {action}, State: {next_state}, Reward: {reward}, Done: {done}")

my_env = Scratch_Game_Environment(frame_size=20, initial_push=False, show_results=True)


