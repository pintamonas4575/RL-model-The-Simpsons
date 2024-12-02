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


my_game = Scratch_Game_Environment(frame_size=20, initial_push=True)
QTimer.singleShot(1000, my_game.close_button.click)
my_game.show_results()