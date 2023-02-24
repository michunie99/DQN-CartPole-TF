import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


from DQN import Agent


def evaluate_agent(env, max_steps, n_eval_episodes, agent):
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        state, _ = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            new_state, reward, terminated, truncated, _  = env.step(action)
            done = terminated or truncated
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

if __name__ == "__main__":

    EXPERIMENT_DIR = "checkpoints_updata100_episode_300_batchsize_64"
    MODEL_NAME = "best_model.ckpt"

    # Initialize agent
    env = gym.make('CartPole-v1')
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

    model_path = os.path.join(EXPERIMENT_DIR, MODEL_NAME)

    agent.load_model(model_path)

    # Evaluate agent
    mean_reward, std_reward = evaluate_agent(env, 1000, 10, agent)

    print(f"Mean reward: {mean_reward}, std: {std_reward}")