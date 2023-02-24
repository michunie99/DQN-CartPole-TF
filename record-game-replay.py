import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from DQN import Agent

def record_run(env, agent, path):
    video = VideoRecorder(env, path)

    state, _ = env.reset()
    # Evaluate agent
    done = False
    while not done:
        env.render()
        video.capture_frame()
        action = agent.select_action(state)
        new_state, reward, terminated, truncated, _  = env.step(action)
        done = terminated or truncated
        state = new_state

    video.close()

if __name__ == "__main__":

    EXPERIMENT_DIR = "./checkpoints_updata100_episode_300_batchsize_128"
    MAX_TIME_STEPS = 1000

    # Initialize agent
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = Agent(      lr=0,
                        gamma=0,
                        n_inputs=n_inputs,
                        n_actions=n_actions,
                        eps=0,
                        update_int=0,
                        batch_size=0,
                        eps_dec=0,
                        eps_end=0,
                        max_len=1,
                        clip_norm=0)

    
    for model_name in os.listdir(EXPERIMENT_DIR):
        if model_name.find('.ckpt'):
            model_path = os.path.join(EXPERIMENT_DIR, model_name)
            agent.load_model(model_path)

            recording_name = os.path.join(EXPERIMENT_DIR, f"{model_name.split('.')[0]}.mp4")
            record_run(env, agent, recording_name)

    env.close()
    