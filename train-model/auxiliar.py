import sys
import random
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QTimer

class MyEnvironment:
    def __init__(self, app, window, label, button):
        self.app = app
        self.window = window
        self.label = label
        self.button = button
        self.state = 0  # Example: 0 means "not clicked", 1 means "clicked"
        self.reward = 0
        self.done = False

    def observe(self):
        """Return the current state as an observation."""
        return self.state

    def act(self, action):
        """Perform an action and update the environment."""
        if action == 0:
            # Do nothing
            self.label.setText("Agent did nothing.")
        elif action == 1:
            # Press the button
            self.label.setText("Agent pressed the button!")
            self.button.click()
            self.state = 1
            self.reward = 10  # Assign a reward
            self.done = True  # End the episode

        return self.observe(), self.reward, self.done

    def reset(self):
        """Reset the environment for a new episode."""
        self.state = 0
        self.reward = 0
        self.done = False
        self.label.setText("Environment reset!")
        return self.observe()

# PyQt Window
class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.label = QLabel("Environment initialized.")
        self.button = QPushButton("Click Me!")
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)

# RL Training Loop
def train_rl_agent():
    env = MyEnvironment(app, window, window.label, window.button)
    num_episodes = 10

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}")
        state = env.reset()
        done = False

        while not done:
            action = random.choice([0, 1])  # Randomly choose an action (0: do nothing, 1: click)
            next_state, reward, done = env.act(action)
            print(f"Action: {action}, State: {next_state}, Reward: {reward}, Done: {done}")

# Main PyQt App
app = QApplication(sys.argv)
window = MyWindow()
window.show()

# Run RL training after the app starts
QTimer.singleShot(1000, train_rl_agent)

sys.exit(app.exec_())
