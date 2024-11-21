import tkinter as tk
import random
from tkinter import ttk, messagebox
from PIL import ImageGrab, Image

import cv2
import numpy as np
import matplotlib.pyplot as plt

symbols = ["hammer.png", "axe.png", "bomb.png", "doughnut.png", "star.png"]
emojis = random.choices(symbols, k=3)

# Maintain images and emoji labels "alive"
images = []
emoji_labels = []
scratched_count = 0  # Counter for scratched squares
total_squares = 0    # Total number of squares
# -------------------------------------------------------
def get_contours_mask(window: tk.Tk) -> Image.Image:
    """Gets initial image and creates the symbol's contours"""
    window.overrideredirect(True)
    window.update()
    x = window.winfo_rootx()
    y = window.winfo_rooty()
    width = window.winfo_width()
    height = window.winfo_height()

    pil_image = ImageGrab.grab(bbox=(x, y, x + width, y + height))
    pil_image.save("dinamic-contours.png")
    window.overrideredirect(False) # this causes the initial flash

    gray = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    background = np.ones_like(pil_image) * 255  # White background

    numpy_contours = cv2.drawContours(background, contours, -1, (0, 0, 0), thickness=cv2.FILLED)  # black fill, BGR mode
    cv2.imwrite("dinamic-outer-contours.png", numpy_contours)

    contour_mask = Image.fromarray(numpy_contours).convert("L") # to gray scale

    return contour_mask
# -------------------------------------------------------
def calculate_scratched_percentage() -> None:
    """Function to calculate and display scratched percentages"""
    global scratched_count
    final_percentage_scratched = (scratched_count / total_squares) * 100
    print(f"Initial scratched area: {random_percentage:.2f}%")
    print(f"Final scratched area: {final_percentage_scratched:.2f}%")
    window.destroy()
# -------------------------------------------------------
def remove_square(rect: tk.Frame, frame_coords: tuple, contour_mask: Image.Image, rect_size: int) -> None:
    """
    Function to remove a square, update the count, and check for symbol overlap and emoji completion.

    :param rect: The tkinter Frame representing the square.
    :param frame_coords: Tuple (x, y) for the top-left corner of the square.
    :param contour_mask: Binary mask of the emoji contour (PIL Image).
    :param rect_size: Size of the frame (width and height).
    """
    global scratched_count
    rect.destroy()  # Remove the frame
    scratched_count += 1  # Increment scratched count

    x, y = frame_coords
    cropped_region = contour_mask.crop((x, y, x + rect_size, y + rect_size))
    # plt.imshow(cropped_region)
    # plt.show()

    pixel_data = cropped_region.getdata()
    # print(set(pixel_data))

    # Check if the frame overlaps a black part of the symbol
    if 0 in pixel_data:  # Black pixels represent parts of the emoji
        print("Scratched a symbol part! (is_revealed)")
        for idx, bbox in enumerate(emoji_bboxes):
            bx1, by1, bx2, by2 = bbox
            # Check if the frame is inside the current emoji's bounding box
            if bx1 <= x < bx2 and by1 <= y < by2:
                emoji_scratch_track[idx].add((x, y))
                
                # Check if the whole emoji is revealed
                if is_emoji_revealed(contour_mask, bbox, rect_size, emoji_scratch_track[idx]):
                    print(f"Emoji {idx + 1} fully revealed!")
                    print("Congratulations!", f"You revealed Emoji {idx + 1}!")
                    break

def is_emoji_revealed(contour_mask: Image.Image, bbox: tuple, rect_size: int, scratched_coords: set) -> bool:
    """
    Check if the entire emoji in the bounding box is revealed.

    :param contour_mask: Binary mask of the emoji contour (PIL Image).
    :param bbox: Tuple (x1, y1, x2, y2) for the emoji bounding box.
    :param rect_size: Size of the frames.
    :param scratched_coords: Set of all (x, y) coordinates scratched for this emoji.
    :return: True if the entire emoji is revealed, False otherwise.
    """
    x1, y1, x2, y2 = bbox
    for x in range(x1, x2, rect_size):
        for y in range(y1, y2, rect_size):
            cropped_region = contour_mask.crop((x, y, x + rect_size, y + rect_size))
            pixel_data = cropped_region.getdata()
            if 0 in pixel_data and (x, y) not in scratched_coords:
                return False  # There are still black parts not scratched
    return True  # All black parts are scratched
# -------------------------------------------------------
# -------------------------------------------------------

window = tk.Tk()
window.title("RL Model Scratch & Win")
window.geometry('1000x500')

# Display emojis
for i, emoji in enumerate(emojis):
    emoji_image = tk.PhotoImage(name=emoji, file=f"emojis/{emoji}")
    images.append(emoji_image)  # Save reference
    label = ttk.Label(window, image=emoji_image)
    label.place(relx=0.3 + i * 0.2, rely=0.5, anchor="center")
    emoji_labels.append(label)

contour_mask = get_contours_mask(window) # PIL image


emoji_bboxes = [
    (218, 168, 218+164, 168+164),  # Emoji 1 (x1, y1, x2, y2)
    (418, 168, 418+164, 168+164),  # Emoji 2
    (618, 168, 618+164, 168+164),  # Emoji 3
]

canvas = tk.Canvas(window, width=1000, height=500, highlightthickness=0, bg="white")
canvas.place(x=0, y=0)

# Draw the bounding boxes on the canvas
for bbox in emoji_bboxes:
    x1, y1, x2, y2 = bbox
    canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)

for widget in window.winfo_children():
    if widget != canvas:
        widget.lift()









# Initialize tracking for each emoji's scratched areas
emoji_scratch_track = {i: set() for i in range(len(emoji_bboxes))}

squares: list[tk.Frame] = []
rect_size = 20
start_x = 148
start_y = 98
rect_width = 50 + 164 * 3 + 36 * 2 + 70
rect_height = 50 + 164 + 70

for x in range(start_x, start_x + rect_width, rect_size):
    for y in range(start_y, start_y + rect_height, rect_size):
        rect = tk.Frame(window, width=rect_size, height=rect_size, bg="gray")
        rect.place(x=x, y=y)
        # rect.bind("<Button-1>",lambda event, r=rect, coords=(x, y): on_remove_square(r, coords, contour_mask, rect_size))
        rect.bind("<Button-1>",lambda event, r=rect, coords=(x, y): remove_square(r, coords, contour_mask, rect_size))
        squares.append(rect)

total_squares = len(squares)
random_percentage = random.randint(80, 90)
squares_to_remove = int(total_squares * (random_percentage / 100))
random.shuffle(squares)

for i in range(squares_to_remove):
    squares[i].destroy()
    scratched_count += 1




# "Finish Game" button
finish_button = ttk.Button(window, text="Finish Game", command=calculate_scratched_percentage)
finish_button.place(relx=0.5, rely=0.9, anchor="center")

# finish_button.invoke()
window.mainloop()
