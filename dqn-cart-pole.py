import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from collections import deque
import random
import os
from statistics import mean
from multiprocessing import Process

from DQN import Agent
from utilities import LearningAnimation


if __name__ == "__main__":

    # Set up (rework it to argsparse ???)
    LR = 1e-3
    UPDATE_INT = 100
    GAMMA = 0.99
    EPS = 1
    EPS_DEC = 1e-3
    EPS_END = 0.01
    BATCH_SIZE = 128*2
    NUM_EPISODES = 300 
    SAVE_INTERVALS = 50
    CLIP_NORM = 1
    CHECKPOINT_DIR = f"./checkpoints_updata{UPDATE_INT}_episode_{NUM_EPISODES}_batchsize_{BATCH_SIZE}"

    # Create the directory for data
    if not os.path.isdir(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)

    env = gym.make('CartPole-v1')
    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = Agent(      lr=LR,
                        gamma=GAMMA,
                        n_inputs=n_inputs,
                        n_actions=n_actions,
                        eps=EPS,
                        update_int=UPDATE_INT,
                        batch_size=BATCH_SIZE,
                        eps_dec=EPS_DEC,
                        eps_end=EPS_END,
                        max_len=100_000,
                        clip_norm=CLIP_NORM)

    animation = LearningAnimation()
    animation.start_animation()

    # Training loop
    reward_history = []
    epsilon_history = []
    loss_history = []
    best_reward = 0
    
    for i_episode in range(NUM_EPISODES):
        done = False
        observation, _ = env.reset()
        episode_reward = 0
        loss_episode = []
        # Run episode
        while not done:
            action = agent.select_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    
            # Update episode reward
            episode_reward += reward
    
            # Store transition
            agent.store_transition(observation, action, reward, next_observation, done)
            
            observation = next_observation
    
            # Perform optimimisation
            loss = agent.optimize()

            if loss != None:
                loss_episode.append(loss)
    
        # Save best model
        if episode_reward >= best_reward:
            agent.save_model(os.path.join(CHECKPOINT_DIR, "best_model.ckpt"))
            best_reward = episode_reward
        
        # Save model every few intervals
        if i_episode % SAVE_INTERVALS == 0:
            agent.save_model(os.path.join(CHECKPOINT_DIR, f"ch_episode_{i_episode:03}.ckpt"))
        
        # Save the episode reward to history
        reward_history.append(episode_reward)
        epsilon_history.append(agent.get_epsilon())
        loss_history.extend(loss_episode)
        
        # Add data to the animation buffer to draw
        animation.update_on_epoch(episode_reward)
        animation.update_plot()
        # Calculate mean loss
        try:
            mean_loss = mean(loss_episode)
        except:
            mean_loss = 0

        print(f"Episode_{i_episode}, mean reward: {episode_reward}, mean loss: {mean_loss}")

    
    animation.save_plot(os.path.join(CHECKPOINT_DIR, "reward.jpg"))
    # Save history
    agent.save_model(os.path.join(CHECKPOINT_DIR, "final_model.ckpt"))
    np.savetxt(os.path.join(CHECKPOINT_DIR, "loss.csv"), np.array(loss_history), delimiter=',')
    np.savetxt(os.path.join(CHECKPOINT_DIR, "reward.csv"), np.array(reward_history), delimiter=',')
    np.savetxt(os.path.join(CHECKPOINT_DIR, "epsilon.csv"), np.array(epsilon_history), delimiter=',')