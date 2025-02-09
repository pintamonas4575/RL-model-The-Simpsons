# 🤖 TFM-RL-model-The-Simpsons 🤖
Implementation of a Reinforcement Learning (RL) model to learn to scratch the less possible surface on the scratch game of the Badulaque of the app "The Simpsons Springfield".

🙋‍♂️ This project, and as a consequence, its files, belongs to the **master´s final project** (thesis) of **Alejandro Mendoza** in **Machine Learning and Big Data master degree** of the Polytechnic University of Madrid (UPM).

# 🧭 Workflow
1. Generate random tickets in every execution
2. Generate tickets with scrathing surface above
3. Automated scratching
4. Obtain percentage of scratched surface 
5. Animate/Warn when a symbol is fully discovered
6. (Esthetic look of window)
7. RL model for scratching
   1. Test to remove all the frames with the agent to see if the game works
   2. (random scratching strategy)
   3. Reward scratching stategy

# 🏆 Model´s objective
Obtain the less scrathced surface as possible, obviously showing the three symbols. There aren´t any rewards for same symbols.

**NOTE:** Symbols change in every ticket generation.

# 📘 Notebooks "auxiliar" and "contours"
*"auxiliar.ipynb"* is used for different tests.

*"contours.ipynb"* is used for showing the creation of the pixel mask and obtaining the valid contours.

# 📜 Scripts "phaseX.py"
Incremental versions of the game until having the complete functionality. You can execute "phase6-final-game.py" to play yourself. 

# 📂 Folder "checkpoint-photos"
Different phases that a ticket has during the game.

# 📂 Folder "emojis"
Emoji images used for generating the tickets. Downloaded from the Emojipedia.

# 📂 Folder "utils"
Some useful files like the window background of the tickets or the script to plot the training process.

# 📂 Folder "V0"
Agents trained with full liberty configuration. This means, after removing a cell, the agent can go to any cell (including repetition).

# 📂 Folder "V1"
Agents trained with radial strategies towards the game. This means, after removing a cell, the agent can go to any cell around the removed one.

# 📂 Folder "results"
Results of all agents of all versions of the game.

# 🛠️ Used resorces
Github emojis: [ikatyang](https://github.com/ikatyang/emoji-cheat-sheet)

Emojipedia (downloaded last iOS emoji versions): [Emojipedia](https://emojipedia.org/)

Tkinter 8.6 (later unsed): [Poor Python documentation](https://docs.python.org/3.11/library/tkinter.html)

PyQT5 (later, the one used): [Official web](https://www.riverbankcomputing.com/static/Docs/PyQt5/)

PyQT5 better documentation: [Qt.io](https://doc.qt.io/qt-5/classes.html) 

ChatGPT: [ChatGPT](https://chatgpt.com/)

Stackoverflow: [Searching questions](https://stackoverflow.com/)

# ⚖️ License 
Spanish governement surely scratches the money out of our pockets.

# 👤 Contact
Any doubt or suggestion contact with the author:

Alejandro Mendoza: alejandro.embi@gmail.com 
