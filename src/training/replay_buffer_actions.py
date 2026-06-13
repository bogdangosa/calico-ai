import random
import numpy as np
from collections import deque


class ReplayBufferActions:

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, flat_features, action, reward, next_state, next_flat_features, next_mask, done):
        self.buffer.append((state, flat_features, action, reward, next_state, next_flat_features, next_mask, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, flat_features, action, reward, next_state, next_flat_features, next_mask, done = zip(*batch)

        return (
            np.array(state, dtype=np.float32),
            np.array(flat_features, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(next_flat_features, dtype=np.float32),
            np.array(next_mask, dtype=np.bool_),
            np.array(done, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)
