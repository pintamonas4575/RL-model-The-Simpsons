import tkinter as tk
import random
from tkinter import ttk
from PIL import ImageGrab, Image, ImageTk
import cv2
import numpy as np

# -------------------------------------------------------
def get_window_image_and_save(save: bool, file_name: str) -> Image.Image | None:
    """Get window state and possibility to save as Image"""

    window.overrideredirect(True)
    window.update()
    x = window.winfo_rootx()
    y = window.winfo_rooty()
    width = window.winfo_width()
    height = window.winfo_height()

    pil_image = ImageGrab.grab(bbox=(x, y, x + width, y + height))
    if save:
        pil_image.save(file_name)
    window.overrideredirect(False)

    return pil_image if save else None
# -------------------------------------------------------
def display_emojis_BACK() -> None:
    """Place 3 emojis alligned in the center of the window"""

    symbols = ["hammer.png", "axe.png", "bomb.png", "doughnut.png", "star.png"]
    emojis = random.choices(symbols, k=3)
    for i, emoji in enumerate(emojis):
        emoji_image = tk.PhotoImage(name=emoji, file=f"emojis/{emoji}")
        images.append(emoji_image)  # Save reference
        label = ttk.Label(window, image=emoji_image)
        label.place(relx=0.3 + i * 0.2, rely=0.5, anchor="center")
        emoji_labels.append(label)

def display_emojis() -> None:
    """Display emojis with transparent backgrounds on the tkinter window."""
    symbols = ["hammer.png", "axe.png", "bomb.png", "doughnut.png", "star.png"]
    emojis = random.choices(symbols, k=3)

    for i, emoji in enumerate(emojis):
        emoji_image = Image.open(f"emojis/{emoji}").convert("RGBA")
        emoji_photo = ImageTk.PhotoImage(emoji_image)

        images.append(emoji_photo)

        canvas = tk.Canvas(window, width=emoji_image.width, height=emoji_image.height, highlightthickness=0, bg="white")
        canvas.place(relx=0.3 + i * 0.2, rely=0.5, anchor="center")
        canvas.create_image(0, 0, image=emoji_photo, anchor="nw")

        emoji_labels.append(canvas)
# -------------------------------------------------------
def get_contours_mask() -> Image.Image:
    """Gets initial image and creates the symbol's contours"""

    pil_image = get_window_image_and_save(True, "1-dinamic-figures.png")

    gray = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    background = np.ones_like(pil_image) * 255  # White background

    numpy_contours = cv2.drawContours(background, contours, -1, (0, 0, 0), thickness=cv2.FILLED)  # black fill, BGR mode
    cv2.imwrite("2-dinamic-contours.png", numpy_contours)

    contour_mask = Image.fromarray(numpy_contours).convert("L") # to gray scale

    return contour_mask
# ------------------------------------------------------
def add_scratching_frames(window: tk.Tk, contour_mask: np.ndarray) -> dict[int, set]:
    rect_size = FRAME_SIZE
    start_x = 148
    start_y = 98
    rect_width = 50 + 164 * 3 + 36 * 2 + 70
    rect_height = 50 + 164 + 70

    # tracking dictionary
    emoji_scratch_track = {i: set() for i in range(len(emoji_bboxes))}

    for x in range(start_x, start_x + rect_width, rect_size):
        for y in range(start_y, start_y + rect_height, rect_size):
            frame_bbox = (x, y, x + rect_size, y + rect_size)

            rect = tk.Frame(window, width=rect_size, height=rect_size, bg="gray")
            rect.place(x=x, y=y)

            # Check emoji contour masks
            for i, (ex1, ey1, ex2, ey2) in enumerate(emoji_bboxes):
                # Calculate the intersection area between the frame and emoji bounding box
                ix1, iy1 = max(x, ex1), max(y, ey1)
                ix2, iy2 = min(x + rect_size, ex2), min(y + rect_size, ey2)

                mask_area = contour_mask[iy1:iy2, ix1:ix2]
                if np.any(mask_area == 0):  # Check for black pixels in the mask
                    emoji_scratch_track[i].add(frame_bbox)
                    rect.configure(bg="red")

            rect.bind("<Button-1>",lambda event, r=rect, coords=(x, y): remove_square(r, coords, rect_size, emoji_scratch_track))
            squares.append(rect)
    
    get_window_image_and_save(True, "3-learnable-contours.png")

    return emoji_scratch_track
# -------------------------------------------------------
def remove_square(rect: tk.Frame, frame_coords: tuple[int, int], rect_size: int, emoji_scratch_track: dict[int,set]) -> None:
    """Function to remove a square, update the count, and check for symbol overlap and emoji completion."""

    global scratched_count
    rect.destroy()  
    scratched_count += 1 

    x, y = frame_coords
    frame_bbox = (x, y, x + rect_size, y + rect_size)

    for idx, emoji_bbox in enumerate(emoji_bboxes):
        if frame_bbox in emoji_scratch_track[idx]:
            emoji_scratch_track[idx].remove(frame_bbox)
            if len(emoji_scratch_track[idx]) == 0: # all emoji parts have been removed
                print("Congratulations!", f"You revealed Emoji {idx + 1}!")
                if all(not s for s in emoji_scratch_track.values()): # if all sets are empty
                    print("¡¡¡ All emojis revealed !!!")
                break
# -------------------------------------------------------
def calculate_scratched_percentage() -> None:
    """Function to calculate and display scratched percentages"""

    global scratched_count
    final_percentage_scratched = (scratched_count / total_squares) * 100
    print(f"Initial scratched area: {random_percentage:.2f}%")
    print(f"Final scratched area: {final_percentage_scratched:.2f}%")
    window.destroy()
# ------------------------------------------------------
# ------------------------------------------------------
# Maintain images and emoji labels "alive"
images = []
emoji_labels = []
scratched_count = 0  # Counter for scratched squares
total_squares = 0    # Total number of squares
FRAME_SIZE = 20

emoji_bboxes = [
    (218, 168, 218+164, 168+164),  # Emoji 1 (x1, y1, x2, y2)
    (418, 168, 418+164, 168+164),  # Emoji 2
    (618, 168, 618+164, 168+164),  # Emoji 3
]
# ------------------------------------------------------

window = tk.Tk()
window.config(bg="black")
window.title("RL Model Scratch & Win")
window.geometry('1000x500')

# 1: Display emojis
# canvas = Canvas(window, width=500, height=250)
# canvas.pack(fill="both", expand=True)
# display_emojis(canvas)
# display_emojis_BACK()
display_emojis()

# 2: Get contours mask
contour_mask = get_contours_mask() # PIL image

# 3: Add scratching layer
# 4: check removed frames and warn if symbol found
squares: list[tk.Frame] = []
emoji_scratch_track = add_scratching_frames(window, np.asarray(contour_mask))

# image: Image.Image = Image.open("utils/darth-vader1.jpg")  # Replace with your image path
# image = image.resize((1000, 500), Image.Resampling.LANCZOS)  # Resize to 1000x500
# background_image = ImageTk.PhotoImage(image)
# background_label = tk.Label(window, image=background_image)
# background_label.place(x=0, y=0, relwidth=1, relheight=1)
# background_label.lower()





# for automatic initial scratch
total_squares = len(squares)
random_percentage = random.randint(80, 90)
squares_to_remove = int(total_squares * (random_percentage / 100))
random.shuffle(squares)
for i in range(squares_to_remove): # funciona
    rect = squares[i]  
    coords = rect.place_info()  # Get coordinates of the square
    x = int(coords['x'])  # x coordinate
    y = int(coords['y'])  # y coordinate
    remove_square(rect, (x, y), FRAME_SIZE, emoji_scratch_track) 

# "Finish Game" button
finish_button = ttk.Button(window, text="Finish Game", command=calculate_scratched_percentage)
finish_button.place(relx=0.5, rely=0.9, anchor="center")

# finish_button.invoke()
window.mainloop()
