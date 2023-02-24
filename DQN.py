import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)
        return states, actions, rewards, next_states, done

    def __len__(self):
        return len(self.buffer)

def build_qnetwork(n_inputs, n_actions, lr, clip_norm=None):
    " Build a simple Q-Network "
    arch = []
    # Add input layer
    arch.append(keras.layers.Input(shape=n_inputs))

    # Q-Network: Dense(120) -> Dense(84) -> Output(env.action_space.n)
    arch.append(keras.layers.Dense(128, activation='relu' ))
    arch.append(keras.layers.Dense(84, activation='relu' ))
    arch.append(keras.layers.Dense(n_actions))

    model = keras.models.Sequential(arch)
    model.compile(  optimizer=keras.optimizers.Adam(learning_rate=lr, global_clipnorm=clip_norm),
                    loss=tf.keras.losses.MeanSquaredError())

    return model

class Agent():

    def __init__(self, lr, gamma, n_inputs, n_actions, eps, update_int,
                batch_size, eps_dec=1e-3, eps_end=0.01, max_len=10000, clip_norm=None):
        
        # All hyperparameters
        self.steps = 0
        self.update_int = update_int
        self.gamma = gamma
        self.eps = eps
        self.eps_dec = eps_dec
        self.eps_end = eps_end
        self.batch_size = batch_size
        # Work for discerete actions
        self.action_space = [i for i in range(n_actions)]

        # Replay buffer
        self.replay = ReplayBuffer(max_len)

        # DQN networks
        self.q_target = build_qnetwork(n_inputs, n_actions, lr, clip_norm)
        self.q_policy = build_qnetwork(n_inputs, n_actions, lr, clip_norm)

        # Copy weights
        self.q_target.set_weights(self.q_policy.get_weights())


    def get_epsilon(self):
        return self.eps

    def store_transition(self, state, action, reward, next_state, done):
        self.replay.put(state, action, reward, next_state, done)

    def select_action(self, observation): 
        if random.random() > self.eps:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            action = np.argmax(self.q_policy(state, training=False))
        else:
            action = random.choice(self.action_space)
        return action

    def optimize(self):

        if len(self.replay) < self.batch_size:
            return

        # Step 1 - sample data form buffer
        state, action, reward, next_state, dones = self.replay.sample(self.batch_size)

        state = tf.convert_to_tensor(state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        q_eval = self.q_policy(state, training=False)
        q_next = self.q_target(next_state, training=False)

        # Step 2 - calculate the target Q values
        q_target = np.copy(q_eval)

        # q_target = q_eval
        batch_idx = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_idx, action] = reward + \
                                self.gamma * np.max(q_next, axis=1) * (1 - dones)

        q_target = tf.convert_to_tensor(q_target, dtype=tf.float32)
        
        history = self.q_policy.train_on_batch(     state, 
                                                    q_target,
                                                    return_dict=True)


        if self.steps % self.update_int == 0:
            self.q_target.set_weights(self.q_policy.get_weights()) 

        # Update steps counter and epsilon
        self.steps += 1

        if self.eps > self.eps_end:
            self.eps = max(self.eps_end, self.eps - self.eps_dec)

        return history['loss']

    def save_model(self, name):
        self.q_policy.save(name)

    def load_model(self, name):
        self.q_policy = load_model(name)
        self.q_target = tf.keras.models.clone_model(self.q_policy)
