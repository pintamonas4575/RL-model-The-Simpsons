import sys
import random
import io
import os
import cv2
import math
import numpy as np
from PIL import Image

import numpy as np

class ScratchGameEnvironmentHeadless:
    def __init__(self, frame_size: int, scratching_area: tuple[int,int,int,int]):
        self.FRAME_SIZE = frame_size
        self.rect_x, self.rect_y = scratching_area[0], scratching_area[1]
        self.rect_width, self.rect_height = scratching_area[2], scratching_area[3]

        self.rows = self.rect_height // self.FRAME_SIZE
        self.cols = self.rect_width // self.FRAME_SIZE
        self.total_squares = self.rows * self.cols

        self.frames_mask = [-1] * self.total_squares  # -1: not scratched, 0: empty, 1: reward
        self.scratched_count = 0
        self.good_frames = set(np.random.choice(range(self.total_squares), size=self.total_squares // 5, replace=False))  # 20% good

    def env_step(self, action_index: int) -> tuple[list[int], int, bool]:
        if self.frames_mask[action_index] != -1:
            return self.frames_mask, 0, False  # no-op if already scratched

        self.scratched_count += 1
        reward = 3 if action_index in self.good_frames else -2
        self.frames_mask[action_index] = 1 if reward > 0 else 0
        done = all(self.frames_mask[i] != -1 for i in self.good_frames)
        return self.frames_mask.copy(), reward, done

    def env_reset(self):
        self.__init__(self.FRAME_SIZE, (self.rect_x, self.rect_y, self.rect_width, self.rect_height))

    def get_visual_grid(self) -> list[str]:
        # Useful for Streamlit: return color per cell
        colors = []
        for i, state in enumerate(self.frames_mask):
            if state == -1:
                colors.append("blue")
            elif state == 0:
                colors.append("black")
            elif state == 1:
                colors.append("red")
        return colors
