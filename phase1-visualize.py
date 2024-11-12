import tkinter as tk
import random
from tkinter import ttk
from PIL import ImageGrab

def capturar_y_guardar():
    try:
        # Obtener las coordenadas de la ventana
        x = window.winfo_rootx()
        y = window.winfo_rooty()
        width = window.winfo_width()
        height = window.winfo_height()

        imagen = ImageGrab.grab(bbox=(x, y, x + width, y + height))
        imagen.save("1-window-image.png") 
        print("Captura guardada")
        window.quit()
    except Exception as e:
        print("Error al capturar la ventana:", e)
# ------------------------------------------------------------

# symbols = ["🔨", "🪓", "💣", "🍩", "⭐"]
symbols = ["hammer.png", "axe.png", "bomb.png", "doughnut.png", "star.png"]
emojis = random.choices(symbols, k=3)

try:
    window = tk.Tk()
    # window.config(bg='black')
    window.title("Modelo RL Rasca y gana")
    window.geometry('1000x500')

    # mantener "vivas" las imágenes
    images = []

    for i, emoji in enumerate(emojis):
        emoji_image = tk.PhotoImage(name=emoji, file=f"emojis/{emoji}")
        emoji_image = emoji_image.subsample(1, 1)
        images.append(emoji_image)  # Guardar referencia de la imagen
        label = ttk.Label(window, image=emoji_image)
        label.place(relx=0.3 + i * 0.2, rely=0.5, anchor="center")

    window.after(500, capturar_y_guardar)  # Esperar 100ms antes de capturar
    window.mainloop()

except Exception as e:
    print(e)