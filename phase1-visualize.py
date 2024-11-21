import random
import tkinter as tk
from tkinter import ttk
from PIL import ImageGrab

def catch_and_save() -> None:
    try:
        x = window.winfo_rootx()
        y = window.winfo_rooty()
        width = window.winfo_width()
        height = window.winfo_height()

        imagen = ImageGrab.grab(bbox=(x, y, x + width, y + height))
        imagen.save("1-window-image.png") 
        print("Screenshot saved")
        window.quit()
    except Exception as e:
        print("Error saving window:", e)
# ------------------------------------------------------------
def get_emoji_positions(emoji_labels: list[tk.Label]) -> None:
    for i, label in enumerate(emoji_labels):
        x = label.winfo_x()
        y = label.winfo_y()
        width = label.winfo_width()
        height = label.winfo_height()
        print(f"Emoji {i+1}: {x=}, {y=}, {width=}, {height=}")
# ------------------------------------------------------------

# symbols = ["🔨", "🪓", "💣", "🍩", "⭐"]
symbols = ["hammer.png", "axe.png", "bomb.png", "doughnut.png", "star.png"]
emojis = random.choices(symbols, k=3)

try:
    window = tk.Tk()
    window.title("RL model Scratch & Win")
    window.geometry('1000x500')

    # Disable window decorations (for straight bottom corners and not rounded in screenshots)
    window.overrideredirect(True)

    # mantain images "alive"
    images = []

    for i, emoji in enumerate(emojis):
        emoji_image = tk.PhotoImage(name=emoji, file=f"emojis/{emoji}")
        emoji_image = emoji_image.subsample(1, 1)
        images.append(emoji_image)  # save reference
        label = ttk.Label(window, image=emoji_image)
        label.place(relx=0.3 + i * 0.2, rely=0.5, anchor="center")

    window.after(300, catch_and_save)  # 300ms before saving
    # window.after(100, get_emoji_positions(emoji_labels))
    window.mainloop()

except Exception as e:
    print(e)