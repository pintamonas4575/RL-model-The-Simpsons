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

class Scratch_Game_Environment5():
    """
    Scratch Game environment used to play the Scratch&Win game. No rewards, only discover the symbols.
    
    This version aims to be the simplest one, where there are always the same emojis.
    """
    
    def __init__(self, frame_size: int, scratching_area: tuple[int,int,int,int]) -> None:
        self.emoji_labels: list[QLabel] = []
        self.emoji_bboxes: list[tuple] = []
        self.scratched_count = 0  
        self.FRAME_SIZE = frame_size
        
        self.rect_x, self.rect_y = scratching_area[0], scratching_area[1]
        self.rect_width, self.rect_height = scratching_area[2], scratching_area[3]

        self.number_of_rows = math.ceil(self.rect_height / self.FRAME_SIZE)
        self.number_of_columns = math.ceil(self.rect_width / self.FRAME_SIZE)
        self.total_squares = self.number_of_rows * self.number_of_columns 

        self.frames: list[QFrame] = [] # original frame list
        self.good_frames: list[QFrame] = [] # frame list with emojis
        self.frames_mask = [-1] * self.total_squares # -1 not scratched, 0 bad frame, 1 good frame
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

        vertical_layout.addLayout(h_layout)
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
                self.remove_frame(frame)
            return handler
        
        for y in range(self.rect_y, self.rect_y + self.rect_height, self.FRAME_SIZE):
            for x in range(self.rect_x, self.rect_x + self.rect_width, self.FRAME_SIZE):
                frame = QFrame(self.window)
                frame.setGeometry(QRect(x, y, self.FRAME_SIZE, self.FRAME_SIZE))
                frame.setStyleSheet("background-color: blue")

                for (ex1, ey1, ex2, ey2) in self.emoji_bboxes:
                    ix1, iy1 = max(x, ex1), max(y, ey1)
                    ix2, iy2 = min(x + self.FRAME_SIZE, ex2), min(y + self.FRAME_SIZE, ey2)
                    mask_area = contour_mask[iy1:iy2, ix1:ix2]
                    if np.sum(mask_area == 0) > 10:  # threshold for black pixels in the mask
                        self.good_frames.append(frame)
                        frame.setStyleSheet("background-color: red")
                        break

                frame.mousePressEvent = create_event_handler(frame)
                self.frames.append(frame)

    def remove_frame(self, frame: QFrame) -> tuple[int, bool]:
        """Returns reward and if the game status."""
        frame.hide()
        self.scratched_count += 1
        game_done = False

        frame_index = self.frames.index(frame)
        if frame in self.good_frames:
            self.good_frames.remove(frame)
            self.frames_mask[frame_index] = 1
            game_done = not self.good_frames # true if good_frames is empty
        else:
            self.frames_mask[frame_index] = 0

        numero_de_0s = self.frames_mask.count(0)
        numero_de_1s = self.frames_mask.count(1)
        recompensa_por_0s = -1 * numero_de_0s
        recompensa_por_1s = 10 * numero_de_1s
        recompensa_total = recompensa_por_0s + recompensa_por_1s

        print(f"recompensa por el frame {frame_index}: {recompensa_total}")

        return recompensa_total, game_done

    def calculate_scratched_percentage(self) -> None:
        """Calculate the percentage of scratched areas and print it."""
        scratched_percentage = (self.scratched_count / self.total_squares) * 100
        print(f"frames scratched: {self.scratched_count}")
        print(f"Scratched percentage: {scratched_percentage:.2f}%")
        print(self.frames_mask)
        self.app.exit()

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    """Functions for agents to interact with the environment"""
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    def env_step(self, action_index: int) -> tuple[list[int], int, bool]:
        """Returns **next_state**: Mask of the frames states; **reward**: Cumulative reward; **game_done**."""
        action_frame = self.frames[action_index]
        reward, game_done = self.remove_frame(action_frame) # self.frames_mask is updated here
        next_state = self.frames_mask
        return next_state, reward, game_done

    def env_reset(self):
        """Clean and reset the environment inplace, ready for another play"""
        del self.app
        del self.window
        self.__init__(frame_size=self.FRAME_SIZE, scratching_area=(self.rect_x, self.rect_y, self.rect_width, self.rect_height))

"""----------------------------------------------------------------------------"""

my_env = Scratch_Game_Environment5(frame_size=50, scratching_area=(110,98,770,300))

next_state, reward, game_done = my_env.env_step(0)

# next_state, reward, game_done = my_env.env_step(344)
# print(f"total number of cells: {my_env.total_squares}")
# good_cells = sum([len(my_env.emoji_frame_track[i]) for i in range(3)])
# print(f"total number of good cells: {good_cells}")
# print(f"percentage of good cells: {good_cells/my_env.total_squares*100:.2f}%")

my_env.window.show()
my_env.app.exec()