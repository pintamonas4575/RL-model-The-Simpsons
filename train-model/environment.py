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
    """Scratch Game environment used to play the Scratch&Win game. No rewards, only discover the symbols"""
    
    def __init__(self, frame_size: int, initial_push: bool, num_emojis: int):
        self.emoji_labels: list[QLabel] = []
        self.emoji_bboxes = []
        self.scratched_count = 0  
        self.total_squares = 0 
        self.FRAME_SIZE = frame_size
        self.squares: list[QFrame] = [] # all the squares to later call and remove them
        self.emoji_frame_track = {i: set() for i in range(num_emojis)} # adding the good frames to later check

        # --------------------------------------------
        self.app = QApplication([])
        self.window = QMainWindow()
        self.window.setWindowTitle("RL Model Scratch & Win")
        self.window.setGeometry(500, 250, 1000, 500) # 1000x500

        self.display_emojis()
        contour_mask = self.get_contours_mask()
        self.set_emoji_bboxes()
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

        background_pixmap = QPixmap("utils/space.jpg")  
        background_pixmap = background_pixmap.scaled(1000, 500, Qt.KeepAspectRatioByExpanding) # resize
        background_label = QLabel(self.window)
        background_label.setPixmap(background_pixmap)
        background_label.setGeometry(0, 0, 1000, 500)
        background_label.lower()

        # if show_results:
        #     self.window.show()
        # else:
        #     QTimer.singleShot(500, self.close_button.click)
        # sys.exit(self.app.exec())
    
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
        emojis = random.choices(symbols, k=len(self.emoji_frame_track))

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

        square_size = self.FRAME_SIZE
        start_x = 110 
        start_y = 98
        rect_width = 770
        rect_height = 300

        for x in range(start_x, start_x + rect_width, square_size):
            for y in range(start_y, start_y + rect_height, square_size):

                frame = QFrame(self.window)
                frame.setGeometry(QRect(x, y, square_size, square_size))
                frame.setStyleSheet("background-color: blue")

                for i, (ex1, ey1, ex2, ey2) in enumerate(self.emoji_bboxes):
                    # Calculate the intersection area between the frame and emoji bounding box
                    ix1, iy1 = max(x, ex1), max(y, ey1)
                    ix2, iy2 = min(x + square_size, ex2), min(y + square_size, ey2)

                    mask_area = contour_mask[iy1:iy2, ix1:ix2]
                    if np.sum(mask_area == 0) > 10:  # threshold for black pixels in the mask
                        self.emoji_frame_track[i].add(frame)
                        frame.setStyleSheet("background-color: red")

                frame.mousePressEvent = lambda event, s=frame: self.remove_square(s)
                self.squares.append(frame)
        self.total_squares = len(self.squares)

    def remove_square(self, frame: QFrame) -> bool:
        """Function to remove a square, update the count, and check for symbol overlap and emoji completion."""

        frame.hide() 
        self.scratched_count += 1 
        good_frame = False

        for i in range(len(self.emoji_frame_track)):
            if frame in self.emoji_frame_track[i]:
                self.emoji_frame_track[i].remove(frame)
                # if len(self.emoji_frame_track[i]) == 0: # all emoji parts have been removed
                    # print("Congratulations!", f"You revealed Emoji {i+1}!")
                if all(not s for s in self.emoji_frame_track.values()): # if all sets are empty
                    print("¡¡¡ All emojis revealed !!!")
                    break
                good_frame = True

        return good_frame

    def calculate_scratched_percentage(self) -> None:
        """Function to calculate and display scratched percentages"""

        final_percentage_scratched = (self.scratched_count / len(self.squares)) * 100
        print(f"Final scratched area: {final_percentage_scratched:.2f}%")
        self.window.close()
        self.app.closeAllWindows()

"""----------------------------------------------------------------------------"""

# my_game = Scratch_Game_Environment(frame_size=20, initial_push=False, show_results=True, num_emojis=3)
