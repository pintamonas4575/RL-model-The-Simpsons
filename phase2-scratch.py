import tkinter as tk
import random
from tkinter import ttk

symbols = ["hammer.png", "axe.png", "bomb.png", "doughnut.png", "star.png"]
emojis = random.choices(symbols, k=3)

window = tk.Tk()
window.title("RL model Scratch & Win")
window.geometry('1000x500')

for i, emoji in enumerate(emojis):
    emoji_image = tk.PhotoImage(name=emoji, file=f"emojis/{emoji}")
    emoji_image = emoji_image.subsample(1, 1)
    label = ttk.Label(window, image=emoji_image)
    label.place(relx=0.3 + i * 0.2, rely=0.5, anchor="center")

# "scratching" surface config, static values
rect_size = 20  # size of each scratching square
start_x = 148   # left margin adjust; first emoji, x=218 (-70=148, beacause of Pitagoras)
start_y = 98   # upper margin adjust; emojis,      y=168 (-70=98, beacause of Pitagoras)
rect_width = 50+164*3+36*2+70  # total scratching surface width
rect_height = 50+164+70 # total scratching surface height

# Create grid with scratching squares in a rectangular area over the emojis
for x in range(start_x, start_x + rect_width, rect_size):
    for y in range(start_y, start_y + rect_height, rect_size):
        # gray frame for each square
        rect = tk.Frame(window, width=rect_size, height=rect_size, bg="gray")
        rect.place(x=x, y=y)
        # asign an event for destroying after clicking on it
        rect.bind("<Button-1>", lambda e, r=rect: r.destroy())

window.mainloop()