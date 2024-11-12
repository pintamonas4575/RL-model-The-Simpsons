import tkinter as tk
import random
from tkinter import ttk

# symbols = ["🔨", "🪓", "💣", "🍩", "⭐"]
symbols = ["hammer.png", "axe.png", "bomb.png", "doughnut.png", "star.png"]
emojis = random.choices(symbols, k=3)

def get_emoji_positions():
    for i, label in enumerate(emoji_labels):  # emoji_labels es la lista de etiquetas con los emojis
        x = label.winfo_x()
        y = label.winfo_y()
        width = label.winfo_width()
        height = label.winfo_height()
        print(f"Emoji {i+1}: {x=}, {y=}, {width=}, {height=}")

try:
    window = tk.Tk()
    # window.config(bg='black')
    window.title("Modelo RL Rasca y gana")
    window.geometry('1000x500')

    # maintain images "alive"
    images = []
    emoji_labels = []

    for i, emoji in enumerate(emojis):
        emoji_image = tk.PhotoImage(name=emoji, file=f"emojis/{emoji}")
        emoji_image = emoji_image.subsample(1, 1)
        images.append(emoji_image)  # Guardar referencia de la imagen
        label = ttk.Label(window, image=emoji_image)
        label.place(relx=0.3 + i * 0.2, rely=0.5, anchor="center")
        emoji_labels.append(label)

    # "scraping" surface config, static values
    rect_size = 20  # size of each scraping square
    start_x = 148   # left margin adjust; first emoji, x=218 (-70, beacause of Pitagoras)
    start_y = 98   # upper margin adjust; emojis,      y=168 (-70, beacause of Pitagoras)
    rect_width = 50+164*3+36*2+70  # total sraping surface width
    rect_height = 50+164+70 # total sraping surface height

    # Crear la cuadrícula de "cuadrados de rascado" en un área rectangular sobre los emojis
    for x in range(start_x, start_x + rect_width, rect_size):
        for y in range(start_y, start_y + rect_height, rect_size):
            # Crear un frame gris para cada cuadrado
            rect = tk.Frame(window, width=rect_size, height=rect_size, bg="gray")
            rect.place(x=x, y=y)

            # Asignar el evento para eliminar el cuadrado al hacer clic
            rect.bind("<Button-1>", lambda e, r=rect: r.destroy())

    window.after(100, get_emoji_positions)
    window.mainloop()

except Exception as e:
    print(e)
    window.destroy()