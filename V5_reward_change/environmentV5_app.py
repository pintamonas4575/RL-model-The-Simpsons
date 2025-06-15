import cv2
import math
import os
import random
import numpy as np
from PIL import Image
from typing import Any

class Scratch_Game_Environment5_Streamlit:
    def __init__(self, frame_size: int, scratching_area: tuple[int, int, int, int], random_emojis: bool = False):
        self.scratched_count = 0
        self.FRAME_SIZE = frame_size
        self.random_emojis = random_emojis
        self.rect_x, self.rect_y = scratching_area[0], scratching_area[1]
        self.rect_width, self.rect_height = scratching_area[2], scratching_area[3]
        self.number_of_rows = math.ceil(self.rect_height / self.FRAME_SIZE)
        self.number_of_columns = math.ceil(self.rect_width / self.FRAME_SIZE)
        self.total_squares = self.number_of_rows * self.number_of_columns
        self.frames_mask = [-1] * self.total_squares  # -1 no scratched, 0 bad, 1 good

        self.emoji_images = self.get_emoji_images()
        self.game_image = Image.new("RGBA", (self.rect_width, self.rect_height), (255, 255, 255, 255))

        self.good_frames_idx = set()
        self.squares_images: list[dict[str, Any]] = []

        self._setup_environment_and_contours()
    
    def get_emoji_images(self) -> list[Image.Image]:
        """Return a list of emoji images, depending if the user want them random or not."""
        if not self.random_emojis:
            try:
                emoji_images = [Image.open("../emojis/axe.png") for _ in range(3)]
            except FileNotFoundError:
                emoji_images = [Image.open("emojis/axe.png") for _ in range(3)]
        else:
            try:
                emoji_images = [Image.open(f"../emojis/{emoji_name}") for emoji_name in random.choices(os.listdir("../emojis"), k=3)]
            except FileNotFoundError:
                emoji_images = [Image.open(f"emojis/{emoji_name}") for emoji_name in random.choices(os.listdir("emojis"), k=3)]
        return emoji_images

    def _setup_environment_and_contours(self):
        """Set up the environment by identifying contours and placing the frames."""
        try:
            self.background_image = Image.open("../utils/space.jpg").resize((self.rect_width, self.rect_height)) # local deployment
        except FileNotFoundError:
            self.background_image = Image.open("utils/space.jpg").resize((self.rect_width, self.rect_height)) # local tests and cloud deployment

        """Place the emoji images"""
        aux_img = Image.new("RGBA", (self.rect_width, self.rect_height), (255,255,255,255))
        emoji_width, emoji_height = self.emoji_images[0].size
        gap = 40
        total_width = 3 * emoji_width + 2 * gap
        start_x = (self.rect_width - total_width) // 2
        y = (self.rect_height - emoji_height) // 2
        self.emoji_bboxes = []
        for i, emoji_img in enumerate(self.emoji_images):
            x = start_x + i * (emoji_width + gap)
            self.emoji_bboxes.append((x, y, x + emoji_width, y + emoji_height))
            aux_img.paste(emoji_img, (x, y), emoji_img.convert("RGBA"))

        for (x1, y1, x2, y2), emoji_img in zip(self.emoji_bboxes, self.emoji_images):
            self.background_image.paste(emoji_img, (x1, y1), emoji_img.convert("RGBA"))

        """Detect contours"""
        gray = cv2.cvtColor(np.asarray(aux_img), cv2.COLOR_RGBA2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        background = np.ones_like(np.asarray(aux_img)) * 255
        numpy_contours = cv2.drawContours(background, contours, -1, (0, 0, 0), thickness=cv2.FILLED)
        contour_mask = Image.fromarray(numpy_contours).convert("L")
        contour_mask = np.array(contour_mask)
        aux_img.close()

        """Place the frames"""
        idx = 0
        for row in range(self.number_of_rows):
            for col in range(self.number_of_columns):
                y0 = row * self.FRAME_SIZE
                x0 = col * self.FRAME_SIZE
                frame_area = contour_mask[y0:y0+self.FRAME_SIZE, x0:x0+self.FRAME_SIZE]
                if np.sum(frame_area == 0) > 10:
                    self.good_frames_idx.add(idx)
                    color = (255, 0, 0, 255)  # red if it's good
                else:
                    color = (0, 0, 255, 255)  # blue if it's bad
                square = Image.new("RGBA", (self.FRAME_SIZE, self.FRAME_SIZE), color)
                self.squares_images.append({"img": square, "coords": (x0, y0), "visible": True})
                self.game_image.paste(square, (x0, y0), square.convert("RGBA")) # the image I will show
                idx += 1

    def get_window_image(self) -> Image:
        """Return the current game image status for displaying purposes."""
        return self.game_image
    
    def scratch_frame(self, idx: int) -> tuple[int, bool]:
        """Replace the frame area in game_image with the corresponding area from background_image"""
        x0, y0 = self.squares_images[idx]["coords"]
        area = (x0, y0, x0 + self.FRAME_SIZE, y0 + self.FRAME_SIZE)
        frame_patch = self.background_image.crop(area)
        self.game_image.paste(frame_patch, (x0, y0))
        
        if idx in self.good_frames_idx:
            self.frames_mask[idx] = 1
            self.good_frames_idx.remove(idx)
        else:
            self.frames_mask[idx] = 0

        self.scratched_count += 1
        game_done = not self.good_frames_idx # true if "good_frames_idx" it's empty
        reward_for_0s = -2 * self.frames_mask.count(0)
        reward_for_1s = 3 * self.frames_mask.count(1)
        total_reward = reward_for_0s + reward_for_1s
        return total_reward, game_done
    
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    """Functions for agents to interact with the environment"""
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    def env_step(self, action_index: int) -> tuple[list[int], int, bool]:
        """Removed the frame selected and return the next state, reward, and game status."""
        reward, game_done = self.scratch_frame(action_index) # self.frames_mask is updated here
        next_state = self.frames_mask
        return next_state, reward, game_done
    
    def env_reset(self):
        """Reset the environment to its initial state for a new episode."""
        self.__init__(self.FRAME_SIZE, (self.rect_x, self.rect_y, self.rect_width, self.rect_height), self.random_emojis)