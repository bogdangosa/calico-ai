import random
import numpy as np
from collections import deque

class ReplayBuffer:
    """
    A simple replay buffer for storing and sampling experience transitions.
    """
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, reward, next_state, done):
        """
        Add a transition to the buffer.
        """
        self.buffer.append((state, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Sample a random batch of transitions from the buffer.
        """
        batch = random.sample(self.buffer, batch_size)
        state, reward, next_state, done = zip(*batch)
        
        return (
            np.concatenate(state),
            np.array(reward, dtype=np.float32),
            np.concatenate(next_state),
            np.array(done, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)
