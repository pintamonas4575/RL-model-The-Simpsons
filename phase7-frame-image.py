from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel
from PIL import Image
from PyQt5.QtCore import QRect
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget

def add_scratching_frames_debug(window: QWidget, image_path: str) -> list[QLabel]:
    """Add scratchable frames with correctly sliced image parts."""
    
    square_size = FRAME_SIZE
    start_x = 130
    start_y = 98
    rect_width = 750
    rect_height = 300

    # Load and resize the image using PIL
    full_image = Image.open(image_path).convert("RGBA")
    full_image = full_image.resize((rect_width, rect_height), Image.Resampling.LANCZOS)

    labels = []

    # Loop through the grid to create frames
    for y in range(0, rect_height, square_size):
        for x in range(0, rect_width, square_size):
            # Crop the corresponding part of the image
            cropped_image = full_image.crop((x, y, x + square_size, y + square_size))

            # Convert the cropped PIL image to a QPixmap
            cropped_image_np = np.array(cropped_image)
            h, w, c = cropped_image_np.shape
            q_image = QImage(cropped_image_np.data, w, h, QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(q_image)

            # Create QLabel and set its pixmap
            label = QLabel(window)
            label.setGeometry(QRect(start_x + x, start_y + y, square_size, square_size))
            label.setStyleSheet("border: 2px solid red;")
            label.setPixmap(pixmap)
            label.setScaledContents(True)  # Ensure the image fits the label
            label.show()

            # Add a mousePressEvent to make it scratchable
            label.mousePressEvent = lambda event, lbl=label: lbl.hide()  # Hide the frame on click

            labels.append(label)

    return labels

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
FRAME_SIZE = 50  # Define the size of each frame (change as needed)
IMAGE_PATH = "utils/bg.png"  # Path to your image file (ensure this is correct)

class ScratchableWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scratchable Frames")
        self.setGeometry(100, 100, 1000, 600)  # Set the size of the main window
        
        # Create a central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        # Add the scratchable frames
        self.frames = add_scratching_frames_debug(central_widget, IMAGE_PATH)
        
        # Show all frames
        for frame in self.frames:
            frame.show()

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScratchableWindow()
    window.show()
    sys.exit(app.exec_())
