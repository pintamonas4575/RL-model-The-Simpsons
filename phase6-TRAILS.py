import sys
import random
import io
import cv2
import tempfile
import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QBuffer, QRect
from PyQt5.QtGui import QPixmap, QImage, QBrush, QPalette
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QWidget, QPushButton, QVBoxLayout, QFrame

# --------------------------------------------------
def get_window_image_and_save(window: QMainWindow, save: bool, file_name: str) -> Image.Image:
    """Get window state and optionally save as an image."""
    
    screenshot = window.grab()
    q_image = screenshot.toImage()
    if save:
        q_image.save(file_name)

    buffer = QBuffer()
    buffer.open(QBuffer.ReadWrite)
    q_image.save(buffer, "PNG")
    pil_image = Image.open(io.BytesIO(buffer.data()))

    return pil_image.copy()
# --------------------------------------------------
def display_emojis(window: QMainWindow) -> None:
    """Place 3 emojis alligned in the center of the window"""

    symbols = ["hammer.png", "axe.png", "bomb.png", "doughnut.png", "star.png"]
    emojis = random.choices(symbols, k=3)

    vertical_layout = QVBoxLayout()
    vertical_layout.setAlignment(Qt.AlignCenter)  # Center items vertically

    h_layout = QHBoxLayout()
    h_layout.setAlignment(Qt.AlignCenter)  # Center items horizontally
    h_layout.setSpacing(40)  # space between emojis

    for emoji in emojis:
        pixmap = QPixmap(f"emojis/{emoji}") 
        label = QLabel(window)  # QLabel to display the image
        label.setPixmap(pixmap)  # image to the label
        h_layout.addWidget(label)  # label to the layout

    # Add horizontal layout into the vertical layout
    vertical_layout.addLayout(h_layout)

    # Create a central widget to set the layout in the window
    central_widget = QWidget(window)
    central_widget.setLayout(vertical_layout)
    window.setCentralWidget(central_widget)
# --------------------------------------------------
def get_contours_mask() -> Image.Image:
    """Gets initial image and creates the symbol's contours"""

    pil_image = get_window_image_and_save(window, True, "1-dinamic-figures.png")

    gray = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    background = np.ones_like(pil_image) * 255  # White background

    numpy_contours = cv2.drawContours(background, contours, -1, (0, 0, 0), thickness=cv2.FILLED)  # black fill, BGR mode
    cv2.imwrite("3-coloured-contours.png", numpy_contours)

    contour_mask = Image.fromarray(numpy_contours).convert("L") # to gray scale

    return contour_mask
# --------------------------------------------------
def add_scratching_frames(window: QWidget, contour_mask: np.ndarray) -> tuple[list[QFrame],dict[int, set]]:
    """Add all the scratchable areas and update a dictionary of frames with black mask pixels."""
    
    emoji_bboxes = [
        (218, 168, 218+164, 168+164),  # Emoji 1 (x1, y1, x2, y2)
        (418, 168, 418+164, 168+164),  # Emoji 2
        (618, 168, 618+164, 168+164),  # Emoji 3
    ]   
    squares: list[QFrame] = []
    emoji_scratch_track = {i: set() for i in range(len(emoji_bboxes))}

    square_size = FRAME_SIZE
    start_x = 130 
    start_y = 98
    rect_width = 750
    rect_height = 300

    for x in range(start_x, start_x + rect_width, square_size):
        for y in range(start_y, start_y + rect_height, square_size):
            frame_bbox = (x, y, x + square_size, y + square_size)

            frame = QFrame(window)
            frame.setGeometry(QRect(x, y, square_size, square_size))
            frame.setStyleSheet("background-color: gray")

            for i, (ex1, ey1, ex2, ey2) in enumerate(emoji_bboxes):
                # Calculate the intersection area between the frame and emoji bounding box
                ix1, iy1 = max(x, ex1), max(y, ey1)
                ix2, iy2 = min(x + square_size, ex2), min(y + square_size, ey2)

                mask_area = contour_mask[iy1:iy2, ix1:ix2]
                if np.any(mask_area == 0):  # Check for black pixels in the mask
                    emoji_scratch_track[i].add(frame_bbox)
                    frame.setStyleSheet("background-color: red")

            frame.mousePressEvent = lambda event, s=frame, coords=(x, y): remove_square(s, coords, square_size, emoji_scratch_track)
            squares.append(frame)

    return squares, emoji_scratch_track 
# --------------------------------------------------
def remove_square(frame: QFrame, frame_coords: tuple[int, int], square_size: int, emoji_scratch_track: dict[int,set]) -> None:
    """Function to remove a square, update the count, and check for symbol overlap and emoji completion."""

    global scratched_count
    frame.hide() 
    scratched_count += 1 

    x, y = frame_coords
    frame_bbox = (x, y, x + square_size, y + square_size)

    for i in range(len(emoji_scratch_track)):
        if frame_bbox in emoji_scratch_track[i]:
            emoji_scratch_track[i].remove(frame_bbox)
            if len(emoji_scratch_track[i]) == 0: # all emoji parts have been removed
                print("Congratulations!", f"You revealed Emoji {i+1}!")
                if all(not s for s in emoji_scratch_track.values()): # if all sets are empty
                    print("¡¡¡ All emojis revealed !!!")
                    break
# --------------------------------------------------
def calculate_scratched_percentage() -> None:
    """Function to calculate and display scratched percentages"""

    global scratched_count
    final_percentage_scratched = (scratched_count / total_squares) * 100
    print(f"Initial scratched area: {random_percentage:.2f}%")
    print(f"Final scratched area: {final_percentage_scratched:.2f}%")
    window.close()
# --------------------------------------------------
# Maintain images and emoji labels "alive"
images = []
emoji_labels = []
scratched_count = 0  # Counter for scratched squares
total_squares = 0    # Total number of squares
FRAME_SIZE = 20
# --------------------------------------------------
app = QApplication([])
window = QMainWindow()
window.setWindowTitle("RL Model Scratch & Win")
window.setGeometry(500, 250, 1000, 500)

# 1: Display emojis
display_emojis(window)

# 2: Get contours mask
contour_mask = get_contours_mask() # PIL image

# 3: Add scratching layer
# 4: Check removed frames and warn if symbol found
squares, emoji_scratch_track = add_scratching_frames(window, np.asarray(contour_mask))  # Example dimensions

# for automatic initial scratch
total_squares = len(squares)
random_percentage = random.randint(80, 90)
squares_to_remove = int(total_squares * (random_percentage / 100))
random.shuffle(squares)
# for i in range(squares_to_remove): # funciona
#     square = squares[i]  
#     frame_pos = square.pos() 
#     x, y = frame_pos.x(), frame_pos.y()
#     remove_square(square, (x, y), FRAME_SIZE, emoji_scratch_track)

# background image
background_pixmap = QPixmap("utils/bg.png")  
background_pixmap = background_pixmap.scaled(1000, 500, Qt.KeepAspectRatioByExpanding) # resize
background_label = QLabel(window)
background_label.setPixmap(background_pixmap)
background_label.setGeometry(0, 0, 1000, 500)
background_label.lower()

close_button = QPushButton("Finish Game", window)
close_button.setGeometry(450, 450, 100, 40)  # Button at the bottom center
close_button.clicked.connect(calculate_scratched_percentage)

window.show()
sys.exit(app.exec_())
