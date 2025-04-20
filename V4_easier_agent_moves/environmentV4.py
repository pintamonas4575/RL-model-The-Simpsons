import sys
import random
import io
import os
import cv2
import math
import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QBuffer, QRect, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QWidget, QPushButton, QVBoxLayout, QFrame

class Scratch_Game_Environment4():
    """
    Scratch Game environment used to play the Scratch&Win game. No rewards, only discover the symbols.
    
    This version aims to be the simplest one, where there are always the same emojis.
    """
    
    def __init__(self, frame_size: int, scratching_area: tuple[int,int,int,int]) -> None:
        self.emoji_labels: list[QLabel] = []
        self.emoji_bboxes: list[tuple] = []
        self.scratched_count = 0  
        self.FRAME_SIZE = frame_size
        
        self.emoji_frame_track = {i: set() for i in range(3)} # adding the good frames to later check
        self.rect_x, self.rect_y = scratching_area[0], scratching_area[1]
        self.rect_width, self.rect_height = scratching_area[2], scratching_area[3]

        self.number_of_rows = math.ceil(self.rect_height / self.FRAME_SIZE)
        self.number_of_columns = math.ceil(self.rect_width / self.FRAME_SIZE)
        self.total_squares = self.number_of_rows * self.number_of_columns 

        self.frames = np.array([], dtype=object) # list of QFrames
        # --------------------------------------------
        self.app = QApplication([])
        self.window = QMainWindow()
        self.window.setWindowTitle("RL Model Scratch & Win")
        self.window.setGeometry(500, 250, 1000, 500) # 1000x500
        self.display_emojis()
        contour_mask = self.get_contours_mask()
        self.set_emoji_bboxes()
        self.add_scratching_frames(np.asarray(contour_mask))

        self.close_button = QPushButton("Finish Game", self.window)
        self.close_button.setGeometry(450, 450, 100, 40)  
        self.close_button.clicked.connect(self.calculate_scratched_percentage)

        background_pixmap = QPixmap("utils/space.jpg")  
        background_pixmap = background_pixmap.scaled(1000, 500, Qt.KeepAspectRatioByExpanding) # resize
        self.background_label = QLabel(self.window)
        self.background_label.setPixmap(background_pixmap)
        self.background_label.setGeometry(0, 0, 1000, 500)
        self.background_label.lower()
    
    def get_window_image_and_save(self, save: bool, file_name: str) -> Image.Image:
        """Get window state and optionally save as an image."""
        
        screenshot = self.window.grab()
        q_image = screenshot.toImage()
        if save:
            q_image.save(file_name)

        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        q_image.save(buffer, "PNG")
        pil_image = Image.open(io.BytesIO(buffer.data()))

        return pil_image.copy()

    def display_emojis(self) -> None:
        """Place 3 emojis alligned in the center of the window"""

        # symbols = os.listdir("emojis")
        # emoji = random.choice(symbols)
        # emojis = [emoji]*3
        emojis = ["axe.png", "axe.png", "axe.png"]

        vertical_layout = QVBoxLayout()
        vertical_layout.setAlignment(Qt.AlignCenter)  

        h_layout = QHBoxLayout()
        h_layout.setAlignment(Qt.AlignCenter)
        h_layout.setSpacing(40)  # space between emojis

        for emoji in emojis:
            pixmap = QPixmap(f"emojis/{emoji}") 
            label = QLabel(self.window)  # QLabel to display the image
            label.setPixmap(pixmap)  # image to the label
            h_layout.addWidget(label)  # label to the layout
            self.emoji_labels.append(label)

        # Add horizontal layout into the vertical layout
        vertical_layout.addLayout(h_layout)

        # Add vertical layout into the window
        central_widget = QWidget(self.window)
        central_widget.setLayout(vertical_layout)
        self.window.setCentralWidget(central_widget)

    def get_contours_mask(self) -> Image.Image:
        """Gets initial image and creates the symbol's contours"""

        pil_image = self.get_window_image_and_save(False, "1-dinamic-figures.png")

        gray = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        kernel = np.ones((3, 3), np.uint8)  # Adjust kernel size as needed
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        background = np.ones_like(pil_image) * 255  # White background

        numpy_contours = cv2.drawContours(background, contours, -1, (0, 0, 0), thickness=cv2.FILLED)  # black fill, BGR mode
        contour_mask = Image.fromarray(numpy_contours).convert("L") # to gray scale

        return contour_mask
    
    def set_emoji_bboxes(self) -> None:
        """Set the bboxes for the displayed emojis. For later checking each frame removed"""

        for label in self.emoji_labels:
            x = label.pos().x()
            y = label.pos().y()
            width = label.width()
            height = label.height()
            self.emoji_bboxes.append((x, y, x + width, y + height))

    def add_scratching_frames(self, contour_mask: np.ndarray) -> None:
        """Add all the scratchable areas and update a dictionary of frames with black mask pixels."""

        def create_event_handler(frame):
            def handler(event):  # "event" is mandatory
                self.remove_square(frame)
            return handler
        
        # create the frames along the rows from left to right and top to bottom
        for y in range(self.rect_y, self.rect_y + self.rect_height, self.FRAME_SIZE):
            for x in range(self.rect_x, self.rect_x + self.rect_width, self.FRAME_SIZE):
                frame = QFrame(self.window)
                frame.setGeometry(QRect(x, y, self.FRAME_SIZE, self.FRAME_SIZE))
                frame.setStyleSheet("background-color: blue")

                for i, (ex1, ey1, ex2, ey2) in enumerate(self.emoji_bboxes):
                    # Calculate the intersection area between the frame and emoji bounding box
                    ix1, iy1 = max(x, ex1), max(y, ey1)
                    ix2, iy2 = min(x + self.FRAME_SIZE, ex2), min(y + self.FRAME_SIZE, ey2)
                    mask_area = contour_mask[iy1:iy2, ix1:ix2]
                    if np.sum(mask_area == 0) > 10:  # threshold for black pixels in the mask
                        self.emoji_frame_track[i].add(frame)
                        frame.setStyleSheet("background-color: red")

                frame.mousePressEvent = create_event_handler(frame)
                self.frames = np.append(self.frames, frame)
        
    def remove_square(self, frame: QFrame) -> tuple[bool, bool]:
        """
        Function to remove a square, update the count, and check for symbol overlap and emoji completion.

        Returns
            **good_frame**: True if the removed square was an emoji frame, False otherwise.
            **game_done**: True if all the good frames have been removed, False otherwise.   
        """
        frame.hide() # once you hide it, you can't "click" it again
        self.scratched_count += 1 # the sum won't add if you click hidden frames
        good_frame = False
        game_done = False

        for i in range(len(self.emoji_frame_track)):
            if frame in self.emoji_frame_track[i]:
                good_frame = True
                self.emoji_frame_track[i].remove(frame)
                game_done = all(not s for s in self.emoji_frame_track.values()) # if all sets are empty

        return good_frame, game_done

    def calculate_scratched_percentage(self) -> None:
        """Function to calculate and display scratched percentages"""

        # final_percentage_scratched = (self.scratched_count / self.total_squares) * 100
        # print(f"ENV: Final scratched area: {final_percentage_scratched:.2f}%")
        # print(f"ENV: Total frames scratched: {self.scratched_count}")
        # self.app.quit()
        self.app.exit()

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    """Functions for agents to interact with the environment"""
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    def env_step(self, action_index: int) -> tuple[np.ndarray, int, bool]:
        response, game_done = self.remove_square(self.frames[action_index])
        if response: # red frame
            reward = 15
        else: # blue frame
            reward = -1

        cell_row = action_index // self.number_of_columns
        cell_col = action_index % self.number_of_columns
        above_idx = (cell_row - 1) * self.number_of_columns + cell_col
        below_idx = (cell_row + 1) * self.number_of_columns + cell_col
        left_idx = cell_row * self.number_of_columns + (cell_col - 1)
        right_idx = cell_row * self.number_of_columns + (cell_col + 1)

        neighbor_indices = [above_idx, below_idx, left_idx, right_idx]

        for i, idx in enumerate(neighbor_indices.copy()):
            # Check if index is within valid range
            if idx < 0 or idx >= self.frames.size:
                neighbor_indices.remove(idx)
                continue

            # For horizontal neighbors, ensure they remain in the same row as 'index_to_select'
            neighbor_row = idx // self.number_of_columns
            if abs(idx - action_index) == 1 and neighbor_row != cell_row:
                neighbor_indices.remove(idx)

        next_state = np.array(neighbor_indices)
        return next_state, reward, game_done

    def env_reset(self):
        """Function to clean and reset the environment in place, ready for another play"""
        del self.app
        del self.window
        self.__init__(frame_size=self.FRAME_SIZE, scratching_area=(self.rect_x, self.rect_y, self.rect_width, self.rect_height))

"""----------------------------------------------------------------------------"""

# my_env = Scratch_Game_Environment4(frame_size=40, scratching_area=(110,98,770,300))
# print(f"total number of cells: {my_env.total_squares}")
# good_cells = sum([len(my_env.emoji_frame_track[i]) for i in range(3)])
# print(f"total number of good cells: {good_cells}")
# print(f"percentage of good cells: {good_cells/my_env.total_squares*100:.2f}%")
# next_state, reward, game_done = my_env.env_step(10)
# my_env.window.show()
# my_env.app.exec()
