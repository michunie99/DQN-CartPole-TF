# DQN-CartPole-TF

This repository contains scripts for implementation of DQN created from scratch using Tensorflow 2. Short description of the files:
* DQN.py - RL agent and learning algorithm
* dqn-cart-pole.py - learn agent
* evaluate-cart-pole.py - evaluate agent over many episodes
* record-game-replay.py - run agent and save the replay

The inplemented DQN is a dual DQN. The agent has two neural networks for predicting Q-function. One lags behind and is used for calculating Q values in learning.The video bellow represents the agent learning with progressing epochs.

https://user-images.githubusercontent.com/81962102/221127479-86a62383-ceca-4b51-ae50-9fc8b9bbec3d.mp4

## Learning results

I have conducted some experimets editing the hyper parameters and observing agent performance:

**Update - how often the second DQN is updated**

* Update - 10, epochs - 300
![checkpoints_update_10_epochs_300](https://user-images.githubusercontent.com/81962102/221135516-1e6552bd-82bf-4ad0-9445-bed074d68368.png)

* Update - 100, epochs - 300
![checkpoints_update_100_epochs_300](https://user-images.githubusercontent.com/81962102/221135672-6fb4f1c6-8acf-4e43-a835-5a87598e2875.png)

* Update - 1000, epochs - 600
![checkpoints_updata1000_episode_600_batchsize_128](https://user-images.githubusercontent.com/81962102/221135953-aa16c786-b121-4848-8681-f32a0b079fee.png)


The best performance was achived for the update of 10. The values of 100 and 1000 were to big and the newtork didn't learn corret behavior to stabilize the pole.
