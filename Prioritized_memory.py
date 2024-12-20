import numpy as np
import random
import pickle
from SumTree import SumTree

class PrioritizedReplayBuffer:
    def __init__(self, capacity=20000, alpha=0.6, beta=0.4, epsilon=0.0001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.tree = SumTree(capacity)
        self.position = 0
        self.beta_increment_per_sampling = 0.001

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def push(self, state, action_with_param, reward, next_state, done):
        data = (state, action_with_param, reward, next_state, done)
        # priority = np.max(self.tree.tree[-self.tree.capacity:])  # Maximum priority for a new sample
        # priority = np.max(self.tree.tree[-int(self.tree.capacity):])
        priority = 1
        self.tree.add(priority, data)
        # print(self.tree.data)

    def __len__(self):
        return self.tree.n_entries

    def backup(self, filename):
        data = {
            'position': self.tree.write,
            'n_entries': self.tree.n_entries,
            'capacity': self.capacity,
            'tree': self.tree.tree.tolist(),
            'data': self.tree.data.tolist()
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def restore(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.tree.write = data['position']
            self.tree.n_entries = data['n_entries']
            self.capacity = data['capacity']
            self.tree.tree = np.array(data['tree'])
            self.tree.data = np.array(data['data'])