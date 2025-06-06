import tkinter as tk
from tkinter import ttk
import random

symbols = ["hammer.png", "axe.png", "bomb.png", "doughnut.png", "star.png"]
emojis = random.choices(symbols, k=3)

# Maintain images and emoji labels "alive"
images = []
emoji_labels = []
scratched_count = 0  # Counter for scratched squares
total_squares = 0    # Total number of squares

# Function to calculate and display scratched percentages
def calculate_scratched_percentage():
    global scratched_count
    final_percentage_scratched = (scratched_count / total_squares) * 100
    print(f"Initial scratched area: {random_percentage:.2f}%")
    print(f"Final scratched area: {final_percentage_scratched:.2f}%")
    window.destroy()

# Function to remove a square and update the count
def remove_square(rect: tk.Frame) -> None:
    global scratched_count
    rect.destroy()
    scratched_count += 1
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

# "Scratching" surface configuration
rect_size = 20
start_x = 148
start_y = 98
rect_width = 50 + 164 * 3 + 36 * 2 + 70
rect_height = 50 + 164 + 70

# list to keep track of all square frames
squares: list[tk.Frame] = []

# Grid of squares
for x in range(start_x, start_x + rect_width, rect_size):
    for y in range(start_y, start_y + rect_height, rect_size):
        rect = tk.Frame(window, width=rect_size, height=rect_size, bg="gray")
        rect.place(x=x, y=y)
        rect.bind("<Button-1>", lambda _, r=rect: remove_square(r))
        squares.append(rect)

total_squares = len(squares)
random_percentage = random.randint(10, 70)
squares_to_remove = int(total_squares * (random_percentage / 100))
random.shuffle(squares)  # Shuffle to randomize selection

for i in range(squares_to_remove):
    squares[i].destroy()
    scratched_count += 1

# "Finish Game" button
finish_button = ttk.Button(window, text="Finish Game", command=calculate_scratched_percentage)
finish_button.place(relx=0.5, rely=0.9, anchor="center")

# window.after(2000, finish_button.invoke) # invoke button´s command

window.mainloop()
