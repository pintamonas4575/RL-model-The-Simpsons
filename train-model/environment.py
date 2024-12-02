import sys
import random
import io
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QBuffer, QRect, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QWidget, QPushButton, QVBoxLayout, QFrame

class Scratch_Game_Environment():

    emoji_bboxes = [
            (218, 168, 218+164, 168+164),  # Emoji 1 (x1, y1, x2, y2)
            (418, 168, 418+164, 168+164),  # Emoji 2
            (618, 168, 618+164, 168+164),  # Emoji 3
    ]
    
    def __init__(self, frame_size: int, initial_push: bool):
        self.images = []
        self.emoji_labels = []
        self.scratched_count = 0  
        self.total_squares = 0 
        self.FRAME_SIZE = frame_size
        self.squares: list[QFrame] = []
        self.emoji_scratch_track = {i: set() for i in range(len(self.emoji_bboxes))}
        # --------------------------------------------
        self.app = QApplication([])
        self.window = QMainWindow()
        self.window.setWindowTitle("RL Model Scratch & Win")
        self.window.setGeometry(500, 250, 1000, 500) # 1000x500

        self.display_emojis()
        contour_mask = self.get_contours_mask()
        self.add_scratching_frames(np.asarray(contour_mask))

        if initial_push:
            random_percentage = random.randint(50, 85)
            squares_to_remove = int(self.total_squares * (random_percentage / 100))
            random.shuffle(self.squares)
            for i in range(squares_to_remove): # funciona
                square = self.squares[i]  
                frame_pos = square.pos() 
                x, y = frame_pos.x(), frame_pos.y()
                self.remove_square(square, (x, y))

        self.close_button = QPushButton("Finish Game", self.window)
        self.close_button.setGeometry(450, 450, 100, 40)  
        self.close_button.clicked.connect(self.calculate_scratched_percentage)
    
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

        symbols = ["hammer.png", "axe.png", "bomb.png", "doughnut.png", "star.png"]
        emojis = random.choices(symbols, k=3)

        vertical_layout = QVBoxLayout()
        vertical_layout.setAlignment(Qt.AlignmentFlag(0))  # vertical 'Qt.AlignCenter'

        h_layout = QHBoxLayout()
        h_layout.setAlignment(Qt.AlignmentFlag(4))  # horizontal 'Qt.AlignCenter'
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

    def add_scratching_frames(self, contour_mask: np.ndarray) -> None:
        """Add all the scratchable areas and update a dictionary of frames with black mask pixels."""

        square_size = self.FRAME_SIZE
        start_x = 110 
        start_y = 98
        rect_width = 770
        rect_height = 300

        for x in range(start_x, start_x + rect_width, square_size):
            for y in range(start_y, start_y + rect_height, square_size):
                frame_bbox = (x, y, x + square_size, y + square_size)

                frame = QFrame(self.window)
                frame.setGeometry(QRect(x, y, square_size, square_size))
                frame.setStyleSheet("background-color: blue")

                for i, (ex1, ey1, ex2, ey2) in enumerate(self.emoji_bboxes):
                    # Calculate the intersection area between the frame and emoji bounding box
                    ix1, iy1 = max(x, ex1), max(y, ey1)
                    ix2, iy2 = min(x + square_size, ex2), min(y + square_size, ey2)

                    mask_area = contour_mask[iy1:iy2, ix1:ix2]
                    if np.any(mask_area == 0):  # Check for black pixels in the mask
                        self.emoji_scratch_track[i].add(frame_bbox)
                        frame.setStyleSheet("background-color: red")

                frame.mousePressEvent = lambda event, s=frame, coords=(x, y): self.remove_square(s, coords)
                self.squares.append(frame)
        self.total_squares = len(self.squares)

    def remove_square(self, frame: QFrame, frame_coords: tuple[int, int]) -> None:
        """Function to remove a square, update the count, and check for symbol overlap and emoji completion."""

        frame.hide() 
        self.scratched_count += 1 
        x, y = frame_coords
        frame_bbox = (x, y, x + self.FRAME_SIZE, y + self.FRAME_SIZE)

        for i in range(len(self.emoji_scratch_track)):
            if frame_bbox in self.emoji_scratch_track[i]:
                self.emoji_scratch_track[i].remove(frame_bbox)
                if len(self.emoji_scratch_track[i]) == 0: # all emoji parts have been removed
                    print("Congratulations!", f"You revealed Emoji {i+1}!")
                    if all(not s for s in self.emoji_scratch_track.values()): # if all sets are empty
                        print("¡¡¡ All emojis revealed !!!")
                        break

    def calculate_scratched_percentage(self) -> None:
        """Function to calculate and display scratched percentages"""

        final_percentage_scratched = (self.scratched_count / len(self.squares)) * 100
        print(f"Final scratched area: {final_percentage_scratched:.2f}%")
        self.window.close()

    def show_results(self) -> None:
        """Display the game situation whenever needed"""

        # QTimer.singleShot(500, self.close_button.click)
        self.window.show()
        sys.exit(self.app.exec_())

"""----------------------------------------------------------------------------"""

my_game = Scratch_Game_Environment(frame_size=20, initial_push=False)
my_game.show_results()
