import random
import pickle
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=2e4):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def push(self, state, action_with_param, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = [state, action_with_param, reward, next_state, done]
        self.position = int((self.position + 1) % self.capacity)

    def __len__(self):
        return len(self.buffer)

    def backup(self, filename):
        data = {
            'position': self.position,
            'capacity': self.capacity,
            'buffer': self.buffer
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def restore(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.position = data['position']
            self.capacity = data['capacity']
            self.buffer = data['buffer']