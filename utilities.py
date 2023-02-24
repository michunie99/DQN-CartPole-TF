import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use("fivethirtyeight")

class LearningAnimation():

    def __init__(self):
        self.rewards = []
        # self.fig = plt.figure()

    def update_plot(self):

        plt.cla()
        plt.plot(self.rewards, '*-')
        plt.tight_layout()
        plt.xlabel("Epochs")
        plt.ylabel("Reward")
        plt.show(block = False)
        plt.pause(0.01)


    def update_on_epoch(self, reward):
        self.rewards.append(reward)

    def start_animation(self):
        plt.figure(1)
        plt.show(block = False)


    def save_plot(self, name):
        plt.savefig(name)